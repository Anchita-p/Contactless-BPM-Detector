import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bpm_calc import (
    BPMSmoother,
    MultiAlgorithmSmoother,
    STATUS_NO_PULSE,
    STATUS_OK,
    estimate_bpm_from_signal,
)


def test_multi_smoother_reset_clears_every_algorithm():
    """A tracker flush invalidates all three histories at once; if only one
    clears, the others keep displaying a pre-flush average."""
    smoothers = MultiAlgorithmSmoother(["pos", "chrom", "green"])
    smoothers.update({key: {"bpm": 70.0} for key in ["pos", "chrom", "green"]})
    assert all(v == 70.0 for v in smoothers.update(
        {key: {"bpm": 70.0} for key in ["pos", "chrom", "green"]}).values())

    smoothers.reset()
    assert smoothers.update(None) == {"pos": None, "chrom": None, "green": None}


def test_multi_smoother_averages_each_algorithm_independently():
    smoothers = MultiAlgorithmSmoother(["pos", "green"])
    smoothers.update({"pos": {"bpm": 60.0}, "green": {"bpm": 100.0}})
    out = smoothers.update({"pos": {"bpm": 80.0}, "green": {"bpm": 120.0}})

    assert out["pos"] == 70.0
    assert out["green"] == 110.0


def test_multi_smoother_holds_last_average_when_one_algorithm_drops_out():
    """A single algorithm failing its SNR gate must not wipe its history --
    only a flush does that."""
    smoothers = MultiAlgorithmSmoother(["pos", "green"])
    smoothers.update({"pos": {"bpm": 70.0}, "green": {"bpm": 90.0}})
    out = smoothers.update({"pos": {"bpm": None}, "green": {"bpm": 90.0}})

    assert out["pos"] == 70.0
    assert out["green"] == 90.0


def test_estimate_bpm_from_signal_recovers_known_frequency():
    fs = 30
    n = 90
    t = np.arange(n) / fs
    freq_hz = 1.2  # 72 BPM, inside the 0.7-2.5 Hz passband
    signal = np.sin(2 * np.pi * freq_hz * t)

    bpm, status = estimate_bpm_from_signal(signal, fs=fs)

    assert status == STATUS_OK
    assert abs(bpm - freq_hz * 60.0) < (fs / n) * 60.0  # within one FFT bin, in BPM


def test_estimate_bpm_from_signal_returns_no_pulse_for_too_short_signal():
    assert estimate_bpm_from_signal(np.array([1.0]), fs=30) == (0.0, STATUS_NO_PULSE)
    assert estimate_bpm_from_signal(np.array([]), fs=30) == (0.0, STATUS_NO_PULSE)


def test_estimate_bpm_from_signal_returns_no_pulse_when_no_in_band_bins():
    # fs=1 Hz: rfftfreq max is 0.5 Hz, entirely below the 0.7-2.5 Hz passband.
    signal = np.sin(2 * np.pi * 0.1 * np.arange(20))
    assert estimate_bpm_from_signal(signal, fs=1) == (0.0, STATUS_NO_PULSE)


def test_estimate_bpm_from_signal_flags_flat_signal_as_no_pulse():
    # A static image / no face -> an all-zero (or near-zero) pulse signal,
    # with no real spectral peak to be confident about.
    flat_signal = np.zeros(90)
    bpm, status = estimate_bpm_from_signal(flat_signal, fs=30)
    assert status == STATUS_NO_PULSE
    assert bpm == 0.0


def test_estimate_bpm_from_signal_flags_pure_noise_as_no_pulse():
    # White noise has no concentrated spectral peak, so it should fail the
    # SNR/prominence gate even though it technically has in-band power.
    # Uses a production-realistic window (300 samples @ 30 Hz, i.e.
    # evaluate.py's TARGET_WINDOW_SAMPLES) since a short window gives the
    # periodogram too few in-band bins for the ratio to be statistically
    # reliable -- argmax-of-noise order statistics get noisier as bin count
    # shrinks.
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=1.0, size=300)
    bpm, status = estimate_bpm_from_signal(noise, fs=30)
    assert status == STATUS_NO_PULSE
    assert bpm == 0.0


def test_estimate_bpm_from_signal_accepts_strong_peak_over_noise_floor():
    fs = 30
    n = 90
    t = np.arange(n) / fs
    freq_hz = 1.2  # 72 BPM
    rng = np.random.default_rng(1)
    # A clear sinusoid with only a little noise should confidently pass the
    # SNR gate and recover the correct frequency.
    signal = np.sin(2 * np.pi * freq_hz * t) + rng.normal(scale=0.02, size=n)

    bpm, status = estimate_bpm_from_signal(signal, fs=fs)

    assert status == STATUS_OK
    assert abs(bpm - freq_hz * 60.0) < (fs / n) * 60.0


def test_bpm_smoother_averages_recent_readings():
    smoother = BPMSmoother(buffer_size=3)
    assert smoother.update(60.0) == 60.0
    assert smoother.update(70.0) == 65.0
    assert smoother.update(80.0) == 70.0
    # buffer_size=3, so the oldest reading (60.0) rolls off
    assert smoother.update(90.0) == 80.0


def test_bpm_smoother_ignores_none_readings():
    smoother = BPMSmoother(buffer_size=5)
    assert smoother.update(None) is None
    smoother.update(72.0)
    assert smoother.update(None) == 72.0


def test_peak_interpolation_beats_fft_bin_quantisation():
    """A 300-sample window at 30 Hz has 6 BPM-wide bins. Reporting the bin
    centre makes a steady pulse appear to jump in 6 BPM steps, which reads as
    the measurement being unstable. Sub-bin interpolation has to recover rates
    that fall between bin centres."""
    from filters import bandpass_filter

    fs, n = 30.0, 300
    bin_width_bpm = (fs / n) * 60.0
    assert bin_width_bpm == pytest.approx(6.0)

    worst = 0.0
    for true_bpm in (72.0, 72.75, 73.5, 74.25, 75.0, 75.75, 76.5, 77.25):
        t = np.arange(n) / fs
        signal = bandpass_filter(np.sin(2 * np.pi * (true_bpm / 60.0) * t), fs=fs)
        recovered, status = estimate_bpm_from_signal(signal, fs=fs)
        assert status == STATUS_OK
        worst = max(worst, abs(recovered - true_bpm))

    # Without interpolation the worst case is half a bin, 3 BPM.
    assert worst < 0.5, f"worst error {worst:.2f} BPM suggests interpolation is not applied"


def test_peak_interpolation_never_moves_more_than_half_a_bin():
    """A refined peak must stay inside its own bin; anything further means the
    parabola was fitted to something that is not a peak."""
    from filters import bandpass_filter

    fs, n = 30.0, 300
    rng = np.random.default_rng(4)
    for _ in range(15):
        true_bpm = rng.uniform(45.0, 145.0)
        t = np.arange(n) / fs
        signal = bandpass_filter(np.sin(2 * np.pi * (true_bpm / 60.0) * t), fs=fs)
        recovered, status = estimate_bpm_from_signal(signal, fs=fs)
        if status == STATUS_OK:
            assert abs(recovered - true_bpm) < 3.0
