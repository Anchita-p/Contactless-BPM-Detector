import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tracker import (
    FOREHEAD_IDX,
    LEFT_CHEEK_IDX,
    MAX_CONSECUTIVE_MISSES,
    PROCESS_WIDTH,
    RIGHT_CHEEK_IDX,
    STATE_CALIBRATING,
    FaceROITracker,
)


class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeMesh:
    """Stand-in for MediaPipe FaceMesh whose detection can be toggled, so both
    the miss and hit branches run without loading the real model.

    `center` and `half` place a square face in normalized coordinates; moving
    them between frames simulates the ROI shifting.
    """

    NUM_LANDMARKS = 478  # must cover the highest ROI index in use (454)

    def __init__(self):
        self.detect = False
        self.center = (0.5, 0.5)
        self.half = 0.2

    def process(self, frame_rgb):
        result = type("Result", (), {})()
        if not self.detect:
            result.multi_face_landmarks = None
            return result
        cx, cy = self.center
        # Alternate the four corners of a square so the landmark set has real
        # extent, giving _roi_geometry a meaningful centroid and scale.
        corners = [(cx - self.half, cy - self.half), (cx + self.half, cy - self.half),
                   (cx + self.half, cy + self.half), (cx - self.half, cy + self.half)]
        face = type("Face", (), {})()
        face.landmark = [_FakeLandmark(*corners[i % 4]) for i in range(self.NUM_LANDMARKS)]
        result.multi_face_landmarks = [face]
        return result


def _bare_tracker(**overrides):
    """A tracker with the safeguard state initialized but no MediaPipe model."""
    tracker = FaceROITracker.__new__(FaceROITracker)
    tracker.rgb_signal = deque(maxlen=100)
    tracker.timestamps = deque(maxlen=100)
    tracker.face_mesh = _FakeMesh()
    tracker.analysis_width = PROCESS_WIDTH
    tracker.on_flush = None
    tracker.state = STATE_CALIBRATING
    tracker.consecutive_misses = 0
    tracker._prev_centroid = None
    tracker._prev_scale = None
    for key, value in overrides.items():
        setattr(tracker, key, value)
    return tracker


def test_roi_index_lists_match_architecture_md_spec():
    assert len(FOREHEAD_IDX) == 36
    assert len(LEFT_CHEEK_IDX) == 7
    assert len(RIGHT_CHEEK_IDX) == 7
    # indices must be unique within each ROI
    assert len(set(FOREHEAD_IDX)) == len(FOREHEAD_IDX)
    assert len(set(LEFT_CHEEK_IDX)) == len(LEFT_CHEEK_IDX)
    assert len(set(RIGHT_CHEEK_IDX)) == len(RIGHT_CHEEK_IDX)


def test_landmarks_to_points_scales_to_pixel_coordinates():
    class FakeLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = {0: FakeLandmark(0.5, 0.5), 1: FakeLandmark(0.0, 1.0)}
    points = FaceROITracker.landmarks_to_points(landmarks, [0, 1], width=100, height=200)
    assert points == [(50, 100), (0, 200)]


def test_roi_mask_fills_polygon_region():
    points = [(10, 10), (90, 10), (90, 90), (10, 90)]
    mask = FaceROITracker.roi_mask(points, frame_shape=(100, 100, 3))
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 255  # center of square is filled
    assert mask[0, 0] == 0      # corner outside polygon is empty


def test_spatial_average_returns_mean_rgb_within_mask():
    frame_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    frame_rgb[:, :5] = [10, 20, 30]
    frame_rgb[:, 5:] = [100, 100, 100]

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :5] = 255  # only select the left half

    avg = FaceROITracker.spatial_average(frame_rgb, mask)
    np.testing.assert_allclose(avg, [10, 20, 30])


def test_spatial_average_returns_none_for_empty_mask():
    frame_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert FaceROITracker.spatial_average(frame_rgb, mask) is None


def _fill_buffer(tracker, count=20):
    for i in range(count):
        tracker.rgb_signal.append(np.array([100.0, 110.0, 120.0]))
        tracker.timestamps.append(i / 30.0)


def test_face_loss_does_not_flush_at_the_miss_limit():
    tracker = _bare_tracker()
    _fill_buffer(tracker)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for _ in range(MAX_CONSECUTIVE_MISSES):
        tracker.process_frame(frame)

    assert tracker.consecutive_misses == MAX_CONSECUTIVE_MISSES
    assert len(tracker.rgb_signal) == 20


def test_face_loss_flushes_one_frame_past_the_limit():
    flushed = []
    tracker = _bare_tracker(on_flush=lambda: flushed.append(True))
    _fill_buffer(tracker)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for _ in range(MAX_CONSECUTIVE_MISSES + 1):
        tracker.process_frame(frame)

    assert len(tracker.rgb_signal) == 0
    assert len(tracker.timestamps) == 0
    assert tracker.state == STATE_CALIBRATING
    assert flushed == [True]


def test_intermittent_misses_never_reach_the_flush_limit():
    """A face lost and refound repeatedly must not accumulate toward a flush:
    only an unbroken run of misses counts."""
    flushed = []
    tracker = _bare_tracker(on_flush=lambda: flushed.append(True))
    _fill_buffer(tracker)
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    for _ in range(6):
        tracker.face_mesh.detect = False
        for _ in range(MAX_CONSECUTIVE_MISSES - 1):
            tracker.process_frame(frame)
        tracker.face_mesh.detect = True
        tracker.process_frame(frame)
        assert tracker.consecutive_misses == 0

    assert flushed == []


def _square_roi(cx, cy, half):
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half)]


def test_ordinary_head_motion_is_not_a_violent_shift():
    tracker = _bare_tracker()
    centroid, scale = FaceROITracker._roi_geometry(_square_roi(100, 100, 50))
    tracker._prev_centroid, tracker._prev_scale = centroid, scale

    # ~3% of face width per frame -- a brisk but ordinary head turn at 30 FPS.
    moved, moved_scale = FaceROITracker._roi_geometry(_square_roi(103, 102, 50))
    assert not tracker._is_violent_shift(moved, moved_scale)


def test_large_centroid_jump_is_a_violent_shift():
    tracker = _bare_tracker()
    centroid, scale = FaceROITracker._roi_geometry(_square_roi(100, 100, 50))
    tracker._prev_centroid, tracker._prev_scale = centroid, scale

    jumped, jumped_scale = FaceROITracker._roi_geometry(_square_roi(160, 100, 50))
    assert tracker._is_violent_shift(jumped, jumped_scale)


def test_sudden_scale_change_is_a_violent_shift():
    tracker = _bare_tracker()
    centroid, scale = FaceROITracker._roi_geometry(_square_roi(100, 100, 50))
    tracker._prev_centroid, tracker._prev_scale = centroid, scale

    resized, resized_scale = FaceROITracker._roi_geometry(_square_roi(100, 100, 100))
    assert tracker._is_violent_shift(resized, resized_scale)


def test_shift_threshold_is_scale_invariant():
    """The same relative motion must read the same whether the subject is
    close to the camera or far from it."""
    def moved_by(half, fraction):
        tracker = _bare_tracker()
        centroid, scale = FaceROITracker._roi_geometry(_square_roi(500, 500, half))
        tracker._prev_centroid, tracker._prev_scale = centroid, scale
        offset = int(2 * half * fraction)
        new_centroid, new_scale = FaceROITracker._roi_geometry(
            _square_roi(500 + offset, 500, half))
        return tracker._is_violent_shift(new_centroid, new_scale)

    assert moved_by(50, 0.03) == moved_by(200, 0.03) is False
    assert moved_by(50, 0.4) == moved_by(200, 0.4) is True


def test_no_shift_reference_before_the_first_detected_frame():
    tracker = _bare_tracker()
    centroid, scale = FaceROITracker._roi_geometry(_square_roi(100, 100, 50))
    assert not tracker._is_violent_shift(centroid, scale)


def test_shift_guard_survives_a_short_dropout():
    """MediaPipe drops frames during exactly the fast motion that relocates
    the ROI. The pre-dropout geometry must persist across a sub-limit gap, or
    the buffer silently splices two different face positions together."""
    flushed = []
    tracker = _bare_tracker(on_flush=lambda: flushed.append(True))
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    tracker.face_mesh.detect = True
    tracker.process_frame(frame)
    _fill_buffer(tracker)

    # Lost for fewer frames than the flush limit -- no face-loss flush.
    tracker.face_mesh.detect = False
    for _ in range(MAX_CONSECUTIVE_MISSES - 1):
        tracker.process_frame(frame)
    assert flushed == []

    # Reacquired far from where it was: the shift guard must still catch it.
    tracker.face_mesh.detect = True
    tracker.face_mesh.center = (0.9, 0.5)
    tracker.process_frame(frame)

    assert flushed == [True]
    assert tracker.state == STATE_CALIBRATING


def test_steady_face_never_flushes():
    flushed = []
    tracker = _bare_tracker(on_flush=lambda: flushed.append(True))
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    tracker.face_mesh.detect = True
    for i in range(60):
        # Slow drift: a couple of percent of face width per frame.
        tracker.face_mesh.center = (0.5 + 0.001 * i, 0.5)
        tracker.process_frame(frame, timestamp=i / 30.0)

    assert flushed == []
    assert len(tracker.rgb_signal) == 60


def test_get_signal_returns_empty_array_before_any_frames():
    tracker = FaceROITracker.__new__(FaceROITracker)
    tracker.rgb_signal = []
    signal = FaceROITracker.get_signal(tracker)
    assert signal.shape == (0,)


def test_get_timestamps_returns_array_parallel_to_signal():
    tracker = FaceROITracker.__new__(FaceROITracker)
    tracker.timestamps = [1.0, 1.1, 1.2]
    timestamps = FaceROITracker.get_timestamps(tracker)
    assert timestamps.shape == (3,)
    np.testing.assert_allclose(timestamps, [1.0, 1.1, 1.2])
