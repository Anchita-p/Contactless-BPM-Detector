"""Offline evaluator: run the tracker -> POS -> bandpass -> resample -> FFT
pipeline against a pre-recorded video file instead of the live webcam.

There's no live wall-clock constraint here, so every frame is processed
regardless of MediaPipe's per-frame latency, and each frame is stamped with
a synthetic timestamp derived from the video's own reported FPS
(frame_index / video_fps) rather than wall-clock time. That guarantees a
perfectly uniform, gap-free capture rate, isolating the algorithm's
accuracy from any live-camera performance limitations.

This module also holds the shared pipeline core (`compute_bpm`) that the
live loop in main.py reuses, so there is one canonical implementation of
"raw RGB+timestamps -> resample -> POS -> bandpass -> FFT" for both paths.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tracker import FaceROITracker
from rppg_pos import ALGORITHM_KEYS, ALGORITHMS
from hrv import HRVSmoother, compute_hrv, no_pulse_result
from filters import bandpass_filter, resample_to_uniform_rate, temporal_mean_normalize
from bpm_calc import (
    STATUS_NO_PULSE,
    STATUS_OK,
    MultiAlgorithmSmoother,
    estimate_bpm_from_signal,
    signal_snr_db,
)
from config import BUFFER_DURATION_S, NORM_WINDOW_S, TARGET_FS, WINDOW_DURATION_S

TARGET_WINDOW_SAMPLES = int(TARGET_FS * WINDOW_DURATION_S)  # 300
NORM_WINDOW_SAMPLES = int(TARGET_FS * NORM_WINDOW_S)        # 90
# Analysis window plus the lead-in every one of its samples needs behind it
# for a full trailing mean.
REQUIRED_SAMPLES = TARGET_WINDOW_SAMPLES + NORM_WINDOW_SAMPLES - 1
MIN_RAW_SAMPLES_FOR_INTERP = 4  # cubic interpolation needs at least degree+1 points
MIN_STABLE_VARIANCE = 1e-6

HRV_KEY = "hrv"
HRV_SOURCE_KEY = "pos"  # HRV is derived from the POS waveform

STEP_S = 1.0  # emit one BPM reading per this many synthetic seconds of video


def is_pulse_signal_stable(filtered_signal, min_variance=MIN_STABLE_VARIANCE):
    # Cheap pre-filter before the FFT: reject empty/NaN/near-flat signals
    # (e.g. no face detected) before spending a peak search on them.
    if filtered_signal.size == 0 or not np.all(np.isfinite(filtered_signal)):
        return False
    return float(np.var(filtered_signal)) > min_variance


def compute_algorithm_signals(raw_rgb, raw_timestamps):
    """Run every algorithm in ALGORITHMS over one shared, identically
    preprocessed window.

    The window slice, uniform resample and temporal-mean normalization happen
    once and fan out to all three, so the only difference between them is the
    projection itself.

    Returns:
        None when there isn't yet a full window of data -- the whole pipeline
        is still calibrating. Otherwise a dict of algorithm key ->
        {"label", "signal", "bpm", "status", "snr_db"}, where `bpm` is None for an
        algorithm whose own FFT peak failed the SNR confidence gate. The two
        cases are kept distinct so the UI can tell "calibrating" from
        "this algorithm found no pulse".

        The dict carries one extra, non-algorithm entry under HRV_KEY: the
        HRV/Stress Index result derived from the POS waveform (see hrv.py).
        Iterate ALGORITHM_KEYS, not the dict, to visit only the algorithms.
    """
    if raw_rgb.shape[0] < MIN_RAW_SAMPLES_FOR_INTERP:
        return None

    # Take the most recent analysis window plus its normalization lead-in.
    cutoff = raw_timestamps[-1] - BUFFER_DURATION_S
    in_window = raw_timestamps >= cutoff
    window_rgb = raw_rgb[in_window]
    window_ts = raw_timestamps[in_window]

    if window_ts.shape[0] < MIN_RAW_SAMPLES_FOR_INTERP:
        return None

    # Resample to a fixed rate so the FFT resolution doesn't depend on the
    # camera's actual (jittery) frame rate, then pin to an exact-length window.
    _, uniform_rgb = resample_to_uniform_rate(window_rgb, window_ts, TARGET_FS)
    if uniform_rgb.shape[0] < REQUIRED_SAMPLES:
        return None
    uniform_rgb = uniform_rgb[-REQUIRED_SAMPLES:]

    # Strip slow lighting/auto-exposure drift, then discard the lead-in so
    # only samples with a full trailing baseline reach the algorithms.
    normalized = temporal_mean_normalize(uniform_rgb, NORM_WINDOW_SAMPLES)
    window = normalized[-TARGET_WINDOW_SAMPLES:]

    results = {}
    for spec in ALGORITHMS:
        pulse = spec.extract(window, TARGET_FS)
        # CHROM bandpasses internally as part of its alpha step; filtering it
        # again would double the filter order and break the comparison.
        filtered = pulse if spec.self_filtered else bandpass_filter(pulse, fs=TARGET_FS)

        if is_pulse_signal_stable(filtered):
            bpm, status = estimate_bpm_from_signal(filtered, fs=TARGET_FS)
        else:
            bpm, status = 0.0, STATUS_NO_PULSE

        results[spec.key] = {
            "label": spec.label,
            "signal": filtered,
            "bpm": bpm if status == STATUS_OK else None,
            "status": status,
            "snr_db": signal_snr_db(filtered, TARGET_FS),
        }

    # HRV rides on the POS waveform, but only once POS itself has cleared its
    # SNR gate -- running peak detection on noise would invent beats and
    # produce a confident-looking stress tier from nothing.
    source = results[HRV_SOURCE_KEY]
    results[HRV_KEY] = (
        compute_hrv(source["signal"], TARGET_FS, expected_bpm=source["bpm"])
        if source["bpm"] is not None else no_pulse_result()
    )
    return results


class ResultSmoother:
    """Every piece of carried-over state the pipeline has, resettable as a unit.

    A tracker flush invalidates all of it at once, so `reset` is the single
    callback to hand to FaceROITracker(on_flush=...) -- wiring the BPM and HRV
    smoothers up separately is how one of them gets forgotten and keeps
    displaying pre-flush values.
    """

    def __init__(self):
        self._bpm = MultiAlgorithmSmoother(ALGORITHM_KEYS)
        self._hrv = HRVSmoother()

    def reset(self):
        self._bpm.reset()
        self._hrv.reset()

    def update(self, results):
        """Fold one round of pipeline output in. `results` may be None while
        the pipeline is still calibrating.

        Returns: dict of algorithm key -> smoothed BPM, plus HRV_KEY -> the
        smoothed HRV result.
        """
        reading = self._bpm.update(results)
        raw_hrv = results[HRV_KEY] if results else no_pulse_result()
        reading[HRV_KEY] = self._hrv.update(raw_hrv)
        return reading


def evaluate_video(video_path):
    """Process every frame of `video_path` through the full rPPG pipeline,
    using synthetic (frame_index / video_fps) timestamps instead of
    wall-clock time so the capture rate is perfectly uniform with zero
    dropped frames.

    Returns a list of (synthetic_timestamp_s, reading) tuples, one per STEP_S
    seconds of video. Each `reading` maps every algorithm key to its smoothed
    BPM, plus HRV_KEY to the smoothed HRV result and "snr_db" to the primary
    algorithm's confidence for that window. Smoothed BPMs are None until a full
    BUFFER_DURATION_S-second window has accumulated, and return to None
    whenever the tracker auto-flushes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    smoothers = ResultSmoother()
    tracker = FaceROITracker(on_flush=smoothers.reset)

    readings = []

    frame_index = 0
    last_emit_ts = -STEP_S
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Synthetic, video-derived timestamp -- not wall-clock time -- so
            # every frame lands on a perfectly uniform grid with zero gaps.
            synthetic_ts = frame_index / video_fps
            tracker.process_frame(frame, timestamp=synthetic_ts)

            # Emit one BPM reading per STEP_S seconds of (synthetic) video time.
            if len(tracker.rgb_signal) and synthetic_ts - last_emit_ts >= STEP_S:
                last_emit_ts = synthetic_ts
                results = compute_algorithm_signals(tracker.get_signal(),
                                                    tracker.get_timestamps())
                reading = smoothers.update(results)
                # Carry the primary algorithm's confidence alongside the
                # reading so dataset evaluation can report a mean SNR without
                # re-running the pipeline. Extra key, same tuple shape.
                reading["snr_db"] = results[HRV_SOURCE_KEY]["snr_db"] if results else None
                readings.append((synthetic_ts, reading))

            frame_index += 1
            if total_frames > 0:
                pct = 100 * frame_index / total_frames
                print(f"\rProcessing: {pct:5.1f}% ({frame_index}/{total_frames})",
                      end="", flush=True)
    finally:
        cap.release()
        tracker.face_mesh.close()

    print()  # newline after the progress bar
    return readings


def print_readings(readings):
    """Render evaluate_video output as a per-algorithm table with HRV."""
    print("BPM readings over time:")
    header = "  ".join(f"{key.upper():>8}" for key in ALGORITHM_KEYS)
    print(f"      time  {header}  {'RMSSD':>9}  {'STRESS':>8}")
    for ts, reading in readings:
        cells = "  ".join(
            f"{reading[key]:>8.1f}" if reading[key] is not None else f"{'--':>8}"
            for key in ALGORITHM_KEYS
        )
        hrv = reading[HRV_KEY]
        rmssd = f"{hrv['rmssd_ms']:>7.1f}ms" if hrv["rmssd_ms"] is not None else f"{'--':>9}"
        print(f"  t={ts:6.2f}s  {cells}  {rmssd}  {hrv['tier'] or '--':>8}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <path_to_video>")
        sys.exit(1)

    print_readings(evaluate_video(sys.argv[1]))
