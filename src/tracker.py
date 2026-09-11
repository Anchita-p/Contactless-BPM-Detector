import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))

from config import BUFFER_DURATION_S

# Exact ROI landmark indices per ARCHITECTURE.md spec
FOREHEAD_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_CHEEK_IDX = [118, 119, 100, 126, 209, 49, 129]
RIGHT_CHEEK_IDX = [347, 348, 329, 355, 429, 279, 358]

# Schematic wireframe styling for the ROI visualizer
WIREFRAME_COLOR = (255, 255, 0)  # cyan, BGR (B=255, G=255, R=0)
WIREFRAME_ALPHA = 0.4
DOT_COLOR = (255, 255, 255)      # white
DOT_RADIUS = 1

STATUS_LOCKED_COLOR = (60, 200, 60)     # green, BGR
STATUS_SEARCHING_COLOR = (50, 50, 230)  # red, BGR

STATUS_CALIBRATING_COLOR = (0, 215, 255)  # amber, BGR

PROCESS_WIDTH = 480  # default analysis width (offline evaluation keeps this)
# Live capture analyses at this width instead. The validated 30 FPS and SNR
# figures were measured with a 320-wide capture, i.e. with inference running on
# a 320-wide frame; keeping that for the live path means a bigger, nicer
# picture in the dashboard costs no accuracy and no frame rate.
LIVE_ANALYSIS_WIDTH = 320
# Requested from the camera hardware. Larger than PROCESS_WIDTH on purpose: the
# stream shown in the dashboard is worth the pixels, while inference and the
# spatial average still run on the PROCESS_WIDTH copy, so this costs nothing.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

STATE_CALIBRATING = "CALIBRATING"
STATE_MEASURING = "MEASURING"

# Flush triggers. A face lost for longer than this many consecutive frames
# means the buffer now straddles a discontinuity, so the samples on either
# side of the gap cannot belong to the same pulse window.
MAX_CONSECUTIVE_MISSES = 15
# A per-frame centroid jump beyond this fraction of the face's bounding-box
# diagonal is not head motion, it's the mesh re-locking onto a different
# position. Normalizing by face size keeps the threshold valid whether the
# subject sits close to the camera or far from it.
MAX_CENTROID_SHIFT_RATIO = 0.15
# Likewise a sudden change in apparent face size, which catches the mesh
# snapping to a different scale without the centroid moving much.
MAX_SCALE_CHANGE_RATIO = 1.5

# Rolling buffer capacity: BUFFER_DURATION_S of frames, with headroom for
# cameras running faster than the 30 FPS the pipeline resamples to. This is a
# hard ceiling, not a hint -- above it the deque evicts faster than
# BUFFER_DURATION_S accumulates and the tracker never leaves CALIBRATING.
MAX_EXPECTED_FPS = 60.0
BUFFER_MAXLEN = int(BUFFER_DURATION_S * MAX_EXPECTED_FPS)


class FaceROITracker:
    def __init__(self, camera_index=0, max_num_faces=1, on_flush=None,
                 analysis_width=PROCESS_WIDTH):
        self.camera_index = camera_index
        # Frames wider than this are downscaled before inference and before the
        # spatial average. The frame returned to the caller keeps its own size.
        self.analysis_width = analysis_width
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=False,  # base mesh only; forehead/cheek indices don't need iris refinement
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Rolling buffers: [R, G, B] spatial averages and their parallel
        # capture timestamps in seconds.
        self.rgb_signal = deque(maxlen=BUFFER_MAXLEN)
        self.timestamps = deque(maxlen=BUFFER_MAXLEN)
        self._prev_time = None
        self.fps = 0.0

        # Called with no arguments whenever the buffer is auto-flushed, so
        # downstream state (e.g. the BPM smoother) can reset alongside it.
        self.on_flush = on_flush
        self.state = STATE_CALIBRATING
        self.consecutive_misses = 0
        self._prev_centroid = None
        self._prev_scale = None

    def flush(self):
        """Discard the rolling buffer and return to CALIBRATING."""
        self.rgb_signal.clear()
        self.timestamps.clear()
        self.consecutive_misses = 0
        self._prev_centroid = None
        self._prev_scale = None
        self.state = STATE_CALIBRATING
        if self.on_flush is not None:
            self.on_flush()

    def _buffered_span_s(self):
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def open_capture(self):
        """Open the webcam, requesting a low capture resolution at the
        hardware level so MediaPipe never has to process more pixels than
        it needs to track a face."""
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        return cap

    @staticmethod
    def landmarks_to_points(landmarks, indices, width, height):
        # MediaPipe landmarks are normalized to [0, 1]; scale to pixel coordinates.
        return [
            (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
            for idx in indices
        ]

    @staticmethod
    def roi_mask(points, frame_shape):
        # Build a binary mask covering the convex hull of the given ROI points.
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    @staticmethod
    def spatial_average(frame_rgb, mask):
        # Mean R, G, B over every pixel inside the mask (the rPPG "spatial
        # pooling" step). cv2.mean is the same arithmetic as boolean-indexing
        # then averaging, but roughly ten times faster because it never
        # materializes the selected pixels as a new array.
        if not np.any(mask):
            return None
        return np.array(cv2.mean(frame_rgb, mask=mask)[:3], dtype=np.float64)

    @staticmethod
    def _draw_roi_wireframe(frame, forehead_pts, left_cheek_pts, right_cheek_pts):
        """Draw a minimalist schematic: tiny white dots on each landmark plus
        thin, highly translucent cyan wireframe outlines around each ROI.
        The underlying video is left untouched other than these thin marks."""
        roi_point_lists = [forehead_pts, left_cheek_pts, right_cheek_pts]
        hulls = [cv2.convexHull(np.array(pts, dtype=np.int32)) for pts in roi_point_lists]

        # Scale the marks with the frame, or they turn into invisible specks on
        # a high-resolution capture and a heavy scribble on a small one.
        scale = max(frame.shape[1] / 480.0, 1.0)
        line_px = max(int(round(scale)), 1)
        dot_px = max(int(round(DOT_RADIUS * scale)), 1)

        # Blend only the face's bounding box, not the whole frame. Copying and
        # alpha-blending a full HD frame every tick costs more than the entire
        # rPPG pipeline does.
        all_pts = np.vstack(hulls).reshape(-1, 2)
        x0, y0 = np.maximum(all_pts.min(axis=0) - line_px * 2, 0)
        x1 = min(all_pts[:, 0].max() + line_px * 2 + 1, frame.shape[1])
        y1 = min(all_pts[:, 1].max() + line_px * 2 + 1, frame.shape[0])

        region = frame[y0:y1, x0:x1]
        if region.size:
            overlay = region.copy()
            shifted = [hull - np.array([[x0, y0]], dtype=np.int32) for hull in hulls]
            for hull in shifted:
                cv2.polylines(overlay, [hull], True, WIREFRAME_COLOR, line_px, cv2.LINE_AA)
            cv2.addWeighted(overlay, WIREFRAME_ALPHA, region, 1 - WIREFRAME_ALPHA, 0, dst=region)

        for pts in roi_point_lists:
            for x, y in pts:
                cv2.circle(frame, (x, y), dot_px, DOT_COLOR, -1, cv2.LINE_AA)

    @staticmethod
    def _roi_geometry(points):
        """Centroid and bounding-box diagonal (the face-scale reference) of
        the combined ROI landmark set."""
        pts = np.asarray(points, dtype=np.float64)
        centroid = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        return centroid, float(np.hypot(extent[0], extent[1]))

    def _is_violent_shift(self, centroid, scale):
        if self._prev_centroid is None or scale <= 0 or self._prev_scale <= 0:
            return False
        displacement = float(np.linalg.norm(centroid - self._prev_centroid))
        if displacement / scale > MAX_CENTROID_SHIFT_RATIO:
            return True
        scale_ratio = scale / self._prev_scale
        return scale_ratio > MAX_SCALE_CHANGE_RATIO or scale_ratio < 1 / MAX_SCALE_CHANGE_RATIO

    def process_frame(self, frame, timestamp=None):
        """Detect face landmarks, paint ROI overlays, and return
        (annotated_frame, avg_rgb, face_found).

        `timestamp` stamps the sample appended to the rolling buffer; it
        defaults to wall-clock time.monotonic() for live capture, and the
        offline evaluator passes a synthetic video-derived time instead.
        """
        ts = time.monotonic() if timestamp is None else timestamp

        # Analysis runs on a downscaled copy; the frame handed back to the
        # caller keeps its original resolution. Decoupling the two means the
        # camera can be captured at a size worth displaying without changing
        # what the pipeline measures or what inference costs.
        h0, w0 = frame.shape[:2]
        if w0 > self.analysis_width:
            scale = self.analysis_width / w0
            small = cv2.resize(frame, (self.analysis_width, int(h0 * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame

        h, w = small.shape[:2]
        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # MediaPipe only reads this array; skip the copy-on-write overhead
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self.consecutive_misses += 1
            # The geometry reference deliberately survives a short dropout, so
            # a face reacquired somewhere else still gets compared against
            # where it was before the gap. Only flush() clears it. Ordinary
            # motion across a long dropout can therefore trip the shift guard,
            # which is the intended trade: a spurious flush costs a
            # recalibration, a missed one reports a spliced, fabricated BPM.
            #
            # Auto-flush trigger 1: the mesh has been lost long enough that the
            # buffer would splice unrelated stretches of signal together.
            if self.consecutive_misses > MAX_CONSECUTIVE_MISSES and self.rgb_signal:
                self.flush()
            self._update_state()
            return frame, None, False

        self.consecutive_misses = 0
        landmarks = results.multi_face_landmarks[0].landmark

        # Step 1 (ARCHITECTURE.md): locate the three ROI polygons for this frame.
        forehead_pts = self.landmarks_to_points(landmarks, FOREHEAD_IDX, w, h)
        left_cheek_pts = self.landmarks_to_points(landmarks, LEFT_CHEEK_IDX, w, h)
        right_cheek_pts = self.landmarks_to_points(landmarks, RIGHT_CHEEK_IDX, w, h)

        # Auto-flush trigger 2: the ROI jumped or resized far beyond what head
        # motion can produce in one frame, so the samples already buffered came
        # from different skin than the ones about to be.
        centroid, scale = self._roi_geometry(forehead_pts + left_cheek_pts + right_cheek_pts)
        if self._is_violent_shift(centroid, scale) and self.rgb_signal:
            self.flush()
        self._prev_centroid = centroid
        self._prev_scale = scale

        # Step 2: pool the three ROIs into one combined skin mask and average
        # R, G, B across it -- this is the raw per-frame rPPG sample.
        forehead_mask = self.roi_mask(forehead_pts, small.shape)
        left_mask = self.roi_mask(left_cheek_pts, small.shape)
        right_mask = self.roi_mask(right_cheek_pts, small.shape)
        combined_mask = forehead_mask | left_mask | right_mask

        avg_rgb = self.spatial_average(frame_rgb, combined_mask)
        if avg_rgb is not None:
            self.rgb_signal.append(avg_rgb)
            self.timestamps.append(ts)

        self._update_state()
        # Landmarks are normalized, so the overlay is re-derived at the frame's
        # own resolution rather than upscaled from the analysis copy.
        self._draw_roi_wireframe(
            frame,
            self.landmarks_to_points(landmarks, FOREHEAD_IDX, w0, h0),
            self.landmarks_to_points(landmarks, LEFT_CHEEK_IDX, w0, h0),
            self.landmarks_to_points(landmarks, RIGHT_CHEEK_IDX, w0, h0),
        )

        return frame, avg_rgb, True

    def _update_state(self):
        # MEASURING once the buffer holds a full analysis window plus the
        # normalization lead-in; anything less is still filling up.
        self.state = (STATE_MEASURING if self._buffered_span_s() >= BUFFER_DURATION_S
                      else STATE_CALIBRATING)

    def _update_fps(self):
        # Exponential moving average of instantaneous FPS, smoothed so the
        # on-screen counter doesn't jitter frame to frame.
        now = time.time()
        if self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                instantaneous = 1.0 / dt
                self.fps = instantaneous if self.fps == 0 else (0.9 * self.fps + 0.1 * instantaneous)
        self._prev_time = now
        return self.fps

    def draw_hud(self, frame, face_found):
        """Draw the top-left status + FPS HUD overlay."""
        fps = self._update_fps()
        if not face_found:
            status_text = "Status: Searching..."
            status_color = STATUS_SEARCHING_COLOR
        elif self.state == STATE_CALIBRATING:
            status_text = "Status: Calibrating..."
            status_color = STATUS_CALIBRATING_COLOR
        else:
            status_text = "Status: Face Locked"
            status_color = STATUS_LOCKED_COLOR

        cv2.putText(frame, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def get_signal(self):
        """Return the accumulated N x 3 array of spatial-average R, G, B values."""
        return np.array(self.rgb_signal, dtype=np.float64)

    def get_timestamps(self):
        """Return the accumulated N-length array of time.monotonic() capture
        timestamps, parallel to get_signal()."""
        return np.array(self.timestamps, dtype=np.float64)

    def run(self):
        cap = self.open_capture()
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame, avg_rgb, face_found = self.process_frame(frame)
                self.draw_hud(frame, face_found)

                cv2.imshow("rPPG ROI Tracker - press 'q' to quit", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()


if __name__ == "__main__":
    tracker = FaceROITracker()
    tracker.run()
