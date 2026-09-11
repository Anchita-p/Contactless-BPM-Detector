"""Record a webcam clip for offline testing, then score it.

    python record_test_clip.py                  # 30 s, then analyse it
    python record_test_clip.py --duration 45
    python record_test_clip.py --output data/bright_light.mp4
    python record_test_clip.py --no-analyse     # just record

A preview window shows you framing and a countdown while it records. Face
detection is deliberately NOT run during capture: MediaPipe would halve the
frame rate, and the recording's frame rate is the one thing that must not be
compromised. Detection happens afterwards, against the saved file.

Frames are buffered in memory and written out only at the end, with the video's
FPS metadata set to the *measured* capture rate rather than a guessed one.
evaluate.py derives its timestamps from the file's own reported FPS
(frame_index / video_fps), so a wrong value there would silently distort every
downstream BPM estimate.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2

from tracker import FaceROITracker

DEFAULT_DURATION_S = 30
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "test_clip.mp4")
WARMUP_FRAMES = 10


def _preserve_existing(path):
    """Never silently destroy a previous recording."""
    if not os.path.exists(path):
        return
    base, ext = os.path.splitext(path)
    backup = f"{base}.previous{ext}"
    os.replace(path, backup)
    print(f"Existing clip moved to {os.path.basename(backup)}")


def record_clip(duration_s=DEFAULT_DURATION_S, output_path=DEFAULT_OUTPUT):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tracker = FaceROITracker()
    cap = tracker.open_capture()
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Grant camera permission and retry.")

    # The first frames can arrive at the pre-cap.set() resolution before the
    # camera settles on the requested size; drop them so every kept frame
    # has a consistent shape.
    for _ in range(WARMUP_FRAMES):
        cap.read()

    print(f"\nRecording {duration_s}s. Sit still, face the light, look at the camera.")
    print("Press q to abort.\n")

    frames = []
    expected_shape = None
    start = time.time()

    try:
        while True:
            remaining = duration_s - (time.time() - start)
            if remaining <= 0:
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            if expected_shape is None:
                expected_shape = frame.shape
            if frame.shape != expected_shape:
                continue  # drop anomalous frames rather than corrupt the writer
            frames.append(frame)

            # Preview is a copy, so the countdown never gets baked into the
            # frames that are actually saved and analysed.
            preview = frame.copy()
            cv2.putText(preview, f"{remaining:4.1f}s", (14, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(preview, f"{remaining:4.1f}s", (14, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Recording - press q to abort", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Aborted.")
                return None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.face_mesh.close()

    if not frames:
        raise RuntimeError("No frames were captured.")

    elapsed = time.time() - start
    actual_fps = len(frames) / elapsed if elapsed > 0 else 0.0
    print(f"Captured {len(frames)} frames in {elapsed:.1f}s ({actual_fps:.2f} fps)")
    if actual_fps < 10:
        print("  WARNING: below 10 fps leaves little margin over the 2.5 Hz pulse band.")

    _preserve_existing(output_path)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             actual_fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    print(f"Saved {width}x{height} to {output_path}")
    return output_path


def analyse_clip(path):
    """Report whether the clip clears the confidence gate, and by how much."""
    import numpy as np
    from bpm_calc import SNR_THRESHOLD, snr_to_db
    from evaluate import STEP_S, compute_algorithm_signals
    from rppg_pos import ALGORITHM_KEYS

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = FaceROITracker()

    snrs = {k: [] for k in ALGORITHM_KEYS}
    bpms = {k: [] for k in ALGORITHM_KEYS}
    faces = frames = 0
    index, last_emit = 0, -STEP_S

    print("\nAnalysing...")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            timestamp = index / fps
            _, _, found = tracker.process_frame(frame, timestamp=timestamp)
            faces += bool(found)

            if len(tracker.rgb_signal) and timestamp - last_emit >= STEP_S:
                last_emit = timestamp
                results = compute_algorithm_signals(tracker.get_signal(),
                                                    tracker.get_timestamps())
                if results:
                    for key in ALGORITHM_KEYS:
                        if results[key]["snr_db"] is not None:
                            snrs[key].append(results[key]["snr_db"])
                        if results[key]["bpm"] is not None:
                            bpms[key].append(results[key]["bpm"])
            index += 1
    finally:
        cap.release()
        tracker.face_mesh.close()

    gate = snr_to_db(SNR_THRESHOLD)
    print(f"\nFace detected in {faces}/{frames} frames ({100 * faces / max(frames, 1):.0f}%)")
    if faces < frames * 0.9:
        print("  WARNING: the face was lost often. Re-record with better framing or light.")

    if not any(snrs.values()):
        print("Not enough usable video to analyse -- the clip needs to exceed 13 s "
              "of continuous face tracking.")
        return

    print(f"Confidence gate: {gate:.2f} dB\n")
    print(f"{'Algorithm':11s}{'windows':>9}{'best':>8}{'median':>9}{'passed':>9}{'BPM':>10}")
    print("-" * 56)
    for key in ALGORITHM_KEYS:
        values = np.array(snrs[key])
        if not values.size:
            continue
        passed = int((values >= gate).sum())
        bpm = f"{np.median(bpms[key]):.0f}" if bpms[key] else "--"
        print(f"{key:11s}{values.size:>9}{values.max():>8.1f}"
              f"{np.median(values):>9.1f}{passed:>9}{bpm:>10}")

    best = max(max(v) for v in snrs.values() if v)
    if best >= gate:
        print(f"\nCLEARED the gate -- best window {best:.1f} dB. "
              "This clip is usable as a real test fixture.")
    else:
        print(f"\nDid NOT clear -- best window {best:.1f} dB, {gate - best:.1f} dB short. "
              "Try brighter, more even light and less movement.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                        help=f"seconds to record (default {DEFAULT_DURATION_S})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="where to save the clip")
    parser.add_argument("--no-analyse", action="store_true",
                        help="record only, skip the signal-quality report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved = record_clip(duration_s=args.duration, output_path=args.output)
    if saved and not args.no_analyse:
        analyse_clip(saved)
