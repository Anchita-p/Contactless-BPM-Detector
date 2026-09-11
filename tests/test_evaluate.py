import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import (
    REQUIRED_SAMPLES,
    TARGET_FS,
    TARGET_WINDOW_SAMPLES,
    HRV_KEY,
    compute_algorithm_signals,
    evaluate_video,
)
from rppg_pos import ALGORITHM_KEYS


def _synthetic_uniform_rgb(freq_hz=1.2, fs=TARGET_FS, n=REQUIRED_SAMPLES, seed=0):
    """A pulsatile RGB window with a channel-dependent *phase*, not just a
    channel-dependent amplitude.

    Amplitude scaling alone makes every channel a multiple of one sinusoid, so
    any linear combination of R, G and B recovers it and the three algorithms
    agree no matter what their projection matrices are -- a sign typo would
    still pass. Per-channel phase (plus a slow specular term the projections
    are supposed to cancel) makes agreement a real claim.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    baseline = np.array([160.0, 120.0, 100.0])
    amplitude = np.array([2.0, 3.0, 1.0])
    phase = np.array([0.0, 0.35, 0.8])  # radians of channel-dependent lag

    pulse = np.sin(2 * np.pi * freq_hz * t[:, None] + phase[None, :])
    specular = 1.5 * np.sin(2 * np.pi * 0.15 * t)[:, None]  # slow shared glare

    rgb = baseline[None, :] + amplitude[None, :] * pulse + specular
    rgb += rng.normal(scale=0.05, size=rgb.shape)
    return rgb, t


def test_returns_none_before_full_window():
    rgb, t = _synthetic_uniform_rgb(n=10)
    assert compute_algorithm_signals(rgb, t) is None


def test_returns_none_without_the_normalization_lead_in():
    """A full 10 s analysis window is not enough on its own; every sample in
    it needs a full trailing mean behind it before a reading is emitted."""
    rgb, t = _synthetic_uniform_rgb(n=TARGET_WINDOW_SAMPLES)
    assert compute_algorithm_signals(rgb, t) is None


def test_every_algorithm_reports_once_the_window_is_full():
    rgb, t = _synthetic_uniform_rgb(freq_hz=1.2)  # 72 BPM
    results = compute_algorithm_signals(rgb, t)

    assert set(results) == set(ALGORITHM_KEYS) | {HRV_KEY}
    for key in ALGORITHM_KEYS:
        entry = results[key]
        assert entry["signal"].shape == (TARGET_WINDOW_SAMPLES,)
        assert entry["label"]
        assert entry["bpm"] is not None, f"{key} found no pulse in a clean synthetic window"
        assert 40 <= entry["bpm"] <= 160  # sane physiological range


def test_all_algorithms_agree_on_a_pulse_with_per_channel_phase():
    """Three different projections of a window whose channels are out of phase
    and carry a shared specular term must still land in the same FFT bin."""
    rgb, t = _synthetic_uniform_rgb(freq_hz=1.2)
    results = compute_algorithm_signals(rgb, t)

    estimates = [results[key]["bpm"] for key in ALGORITHM_KEYS]
    assert max(estimates) - min(estimates) < 6  # one bin at 30 Hz / 300 samples


def test_signals_are_distinct_across_algorithms_in_the_pipeline():
    """The registry fan-out must not collapse two algorithms onto the same
    math once the shared preprocessing is applied. Correlation, not
    np.allclose: two signals differing only by a scale factor are the same
    estimator and must still fail this."""
    rgb, t = _synthetic_uniform_rgb(freq_hz=1.2)
    results = compute_algorithm_signals(rgb, t)

    for a, b in [("pos", "chrom"), ("pos", "green"), ("chrom", "green")]:
        sa, sb = results[a]["signal"], results[b]["signal"]
        correlation = np.corrcoef(sa - sa.mean(), sb - sb.mean())[0, 1]
        assert abs(correlation) < 0.999, f"{a} and {b} are effectively identical"


def test_flat_input_reports_no_pulse_per_algorithm_not_a_missing_window():
    """A dead signal must come back as a populated dict with bpm None, not as
    None -- the UI has to tell 'no pulse' apart from 'still calibrating'."""
    n = REQUIRED_SAMPLES
    rgb = np.tile([160.0, 120.0, 100.0], (n, 1))
    t = np.arange(n) / TARGET_FS

    results = compute_algorithm_signals(rgb, t)
    assert results is not None
    for key in ALGORITHM_KEYS:
        assert results[key]["bpm"] is None
        assert results[key]["status"] != "OK"


def test_evaluate_video_raises_for_missing_file():
    with pytest.raises(RuntimeError):
        evaluate_video("/nonexistent/path/to/video.mp4")


def test_hrv_is_reported_alongside_the_algorithms():
    rgb, t = _synthetic_uniform_rgb(freq_hz=1.2)
    hrv = compute_algorithm_signals(rgb, t)[HRV_KEY]

    assert set(hrv) == {"rmssd_ms", "tier", "n_beats", "status"}
    # A clean synthetic pulse is perfectly regular, so RMSSD should be tiny
    # and the tier should land in the high-stress bucket.
    assert hrv["rmssd_ms"] is not None
    assert hrv["n_beats"] > 0


def test_hrv_reports_no_pulse_when_pos_fails_its_snr_gate():
    """Peak detection on noise would invent beats and produce a
    confident-looking stress tier from nothing."""
    n = REQUIRED_SAMPLES
    rgb = np.tile([160.0, 120.0, 100.0], (n, 1))
    t = np.arange(n) / TARGET_FS

    results = compute_algorithm_signals(rgb, t)
    assert results["pos"]["bpm"] is None
    assert results[HRV_KEY]["rmssd_ms"] is None
    assert results[HRV_KEY]["tier"] is None
    assert results[HRV_KEY]["n_beats"] == 0


def test_result_smoother_reset_clears_bpm_and_hrv_together():
    """ResultSmoother.reset is the single on_flush target in both entry points.
    If it forgets either half, that half keeps showing pre-flush values -- the
    exact bug the unified object exists to prevent."""
    from evaluate import ResultSmoother
    from rppg_pos import ALGORITHM_KEYS as keys

    smoother = ResultSmoother()
    populated = {key: {"bpm": 70.0} for key in keys}
    populated[HRV_KEY] = {"rmssd_ms": 40.0, "tier": None, "n_beats": 11, "status": "OK"}

    smoother.update(populated)
    warm = smoother.update(populated)
    assert all(warm[key] == 70.0 for key in keys)
    assert warm[HRV_KEY]["rmssd_ms"] == pytest.approx(40.0)

    smoother.reset()
    after = smoother.update(None)

    assert all(after[key] is None for key in keys), "BPM smoothers survived the flush"
    assert after[HRV_KEY]["rmssd_ms"] is None, "HRV smoother survived the flush"
    assert after[HRV_KEY]["tier"] is None
