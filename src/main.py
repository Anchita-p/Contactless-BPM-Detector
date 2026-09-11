"""rPPG BPM detector entry point.

Default: live webcam loop (tracker -> resample to fixed 30 Hz -> POS ->
Butterworth bandpass -> BPM), displaying real-time FPS and smoothed BPM on
the video feed. Press 'q' to quit.

--video PATH: instead, run the same pipeline offline against a pre-recorded
video file (see evaluate.py), printing a progress bar and the resulting
BPM-over-time readings.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from tracker import FaceROITracker, LIVE_ANALYSIS_WIDTH
from rppg_pos import ALGORITHM_KEYS
from evaluate import (
    HRV_KEY,
    ResultSmoother,
    compute_algorithm_signals,
    evaluate_video,
    print_readings,
)

PRINT_INTERVAL_S = 1.0
# Running all three algorithms costs ~23 ms, which would eat most of a 33 ms
# frame budget if done every frame. The 10 s analysis window barely moves
# between frames, so recomputing at 5 Hz is indistinguishable on screen and
# leaves the capture loop its full frame rate.
ANALYSIS_INTERVAL_S = 0.2
BPM_COLOR = (0, 215, 255)  # amber, BGR
PRIMARY_COLOR = (255, 255, 255)


def run_live():
    smoothers = ResultSmoother()
    # An auto-flush clears the tracker's buffer; every algorithm's smoother has
    # to drop its readings with it, or the HUD keeps showing pre-flush averages.
    tracker = FaceROITracker(camera_index=0, on_flush=smoothers.reset,
                             analysis_width=LIVE_ANALYSIS_WIDTH)
    cap = tracker.open_capture()
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {tracker.camera_index}")

    last_print_time = time.time()
    last_analysis_time = 0.0
    smoothed = smoothers.update(None)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
            frame, _, face_found = tracker.process_frame(frame)
            tracker.draw_hud(frame, face_found)  # Status + FPS, top-left

            # Recompute every algorithm from the tracker's accumulated signal;
            # returns None until a full window of data exists.
            if time.monotonic() - last_analysis_time >= ANALYSIS_INTERVAL_S:
                last_analysis_time = time.monotonic()
                results = compute_algorithm_signals(tracker.get_signal(),
                                                    tracker.get_timestamps())
                smoothed = smoothers.update(results)

            # POS is the primary readout; CHROM and the green baseline sit
            # underneath it for live comparison.
            primary = smoothed[ALGORITHM_KEYS[0]]
            primary_text = f"BPM: {primary:.0f}" if primary is not None else "BPM: --"
            cv2.putText(frame, primary_text, (12, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, BPM_COLOR, 2, cv2.LINE_AA)

            for row, key in enumerate(ALGORITHM_KEYS):
                value = smoothed[key]
                text = f"{key}: {value:.0f}" if value is not None else f"{key}: --"
                cv2.putText(frame, text, (12, 106 + row * 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, PRIMARY_COLOR, 1, cv2.LINE_AA)

            hrv = smoothed[HRV_KEY]
            hrv_text = (f"stress: {hrv['tier']} ({hrv['rmssd_ms']:.0f}ms)"
                        if hrv["rmssd_ms"] is not None else "stress: --")
            cv2.putText(frame, hrv_text, (12, 106 + len(ALGORITHM_KEYS) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRIMARY_COLOR, 1, cv2.LINE_AA)

            cv2.imshow("rPPG BPM Detector - press 'q' to quit", frame)

            now = time.time()
            if now - last_print_time >= PRINT_INTERVAL_S:
                last_print_time = now
                summary = "  ".join(
                    f"{key}={smoothed[key]:.0f}" if smoothed[key] is not None else f"{key}=--"
                    for key in ALGORITHM_KEYS
                )
                print(f"FPS: {tracker.fps:.1f}  {summary}  {hrv_text}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.face_mesh.close()


def run_offline(video_path):
    print_readings(evaluate_video(video_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Contactless rPPG BPM detector")
    parser.add_argument("--video", type=str, default=None,
                         help="Path to a pre-recorded video file to evaluate "
                              "offline; if omitted, runs the live webcam.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.video:
        run_offline(args.video)
    else:
        run_live()
