import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from filters import bandpass_filter, resample_to_uniform_rate, temporal_mean_normalize
from bpm_calc import STATUS_OK, estimate_bpm_from_signal
from config import NORM_WINDOW_S, TARGET_FS

NORM_WINDOW = int(TARGET_FS * NORM_WINDOW_S)


def _pulse_trace(bpm, n_samples, fs=TARGET_FS, amplitude=0.02, base=(120.0, 100.0, 90.0)):
    """Synthetic RGB trace carrying a clean sinusoidal pulse at `bpm`."""
    t = np.arange(n_samples) / fs
    wave = amplitude * np.sin(2 * np.pi * (bpm / 60.0) * t)
    return np.column_stack([c * (1 + wave) for c in base])


@pytest.mark.parametrize("bpm", [42, 70, 150])
def test_temporal_mean_normalize_preserves_the_pulse_frequency(bpm):
    """The trailing-mean divide is a high-pass; its corner must sit below the
    0.7 Hz band edge so it doesn't shift the recovered BPM."""
    n = 389  # 300-sample analysis window plus the NORM_WINDOW lead-in
    rgb = _pulse_trace(bpm, n)
    normalized = temporal_mean_normalize(rgb, NORM_WINDOW)[-300:]

    filtered = bandpass_filter(normalized[:, 1], fs=TARGET_FS)
    recovered, status = estimate_bpm_from_signal(filtered, fs=TARGET_FS)
    assert status == STATUS_OK
    assert abs(recovered - bpm) < 3  # within one FFT bin (30/300 Hz = 6 BPM wide)


@pytest.mark.parametrize("bpm", [42, 70, 150])
def test_temporal_mean_normalize_largely_preserves_amplitude(bpm):
    n = 389
    rgb = _pulse_trace(bpm, n)
    normalized = temporal_mean_normalize(rgb, NORM_WINDOW)[-300:]

    # Relative modulation depth of the green channel, before and after.
    raw_depth = np.std(rgb[-300:, 1]) / np.mean(rgb[-300:, 1])
    norm_depth = np.std(normalized[:, 1])
    assert norm_depth > 0.5 * raw_depth


def test_temporal_mean_normalize_removes_an_auto_exposure_step():
    n = 389
    rgb = _pulse_trace(70, n)
    rgb[n // 2:] *= 1.3  # camera auto-exposure jumps mid-window

    normalized = temporal_mean_normalize(rgb, NORM_WINDOW)

    # After the step has scrolled out of the trailing window, the normalized
    # signal is back on the same baseline it had before it.
    settled = normalized[n // 2 + NORM_WINDOW:, 1]
    before = normalized[NORM_WINDOW:n // 2, 1]
    assert abs(settled.mean() - before.mean()) < 0.01
    assert abs(settled.mean() - 1.0) < 0.01


def test_temporal_mean_normalize_uses_a_trailing_not_centered_mean():
    rgb = np.tile(np.arange(1, 7, dtype=np.float64)[:, None], (1, 3))
    normalized = temporal_mean_normalize(rgb, window=3)
    # Sample 4 (value 4) over the mean of samples 2,3,4 -> 4 / 3.0
    np.testing.assert_allclose(normalized[3], 4.0 / 3.0)
    # The head uses an expanding mean, so sample 0 normalizes to exactly 1.
    np.testing.assert_allclose(normalized[0], 1.0)


def test_temporal_mean_normalize_rejects_bad_shapes():
    with pytest.raises(ValueError):
        temporal_mean_normalize(np.zeros((10, 2)), window=3)
    with pytest.raises(ValueError):
        temporal_mean_normalize(np.ones((10, 3)), window=0)


def test_bandpass_filter_preserves_shape_and_has_no_nans():
    fs = 30
    t = np.arange(90) / fs
    signal = np.sin(2 * np.pi * 1.2 * t)  # 72 BPM, inside the passband

    filtered = bandpass_filter(signal, fs=fs)

    assert filtered.shape == signal.shape
    assert not np.isnan(filtered).any()
    assert not np.isinf(filtered).any()


def test_bandpass_filter_attenuates_out_of_band_frequencies():
    fs = 30
    n = 150
    t = np.arange(n) / fs

    passband_signal = np.sin(2 * np.pi * 1.2 * t)   # 72 BPM, inside 0.7-2.5 Hz
    low_freq_drift = np.sin(2 * np.pi * 0.1 * t)     # slow lighting drift, below passband
    high_freq_noise = np.sin(2 * np.pi * 8.0 * t)    # sensor noise, above passband

    combined = passband_signal + low_freq_drift + high_freq_noise
    filtered = bandpass_filter(combined, fs=fs)
    filtered_passband_only = bandpass_filter(passband_signal, fs=fs)

    # The filtered combined signal should closely match filtering the
    # passband component alone, since drift/noise should be attenuated.
    correlation = np.corrcoef(filtered, filtered_passband_only)[0, 1]
    assert correlation > 0.9


def test_resample_to_uniform_rate_produces_fixed_grid_regardless_of_jitter():
    # Irregular (jittery) capture timestamps over ~5 seconds, sparse enough
    # that raw fps varies frame to frame like a real webcam under load.
    rng = np.random.default_rng(0)
    raw_timestamps = np.sort(rng.uniform(0, 5, size=40))
    raw_timestamps[0], raw_timestamps[-1] = 0.0, 5.0
    raw_rgb = np.column_stack([
        160 + 2 * np.sin(2 * np.pi * 1.2 * raw_timestamps),
        120 + 3 * np.sin(2 * np.pi * 1.2 * raw_timestamps),
        100 + 1 * np.sin(2 * np.pi * 1.2 * raw_timestamps),
    ])

    target_fs = 30.0
    uniform_ts, uniform_rgb = resample_to_uniform_rate(raw_rgb, raw_timestamps, target_fs)

    assert uniform_rgb.shape == (uniform_ts.shape[0], 3)
    # Evenly spaced at exactly target_fs, regardless of the irregular input spacing.
    spacings = np.diff(uniform_ts)
    np.testing.assert_allclose(spacings, 1.0 / target_fs, rtol=1e-6)
    assert not np.isnan(uniform_rgb).any()


def test_resample_to_uniform_rate_supports_linear_kind():
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    rgb = np.column_stack([timestamps, timestamps * 2, timestamps * 3])

    _, uniform_rgb = resample_to_uniform_rate(rgb, timestamps, target_fs=10.0, kind="linear")

    assert not np.isnan(uniform_rgb).any()
    assert uniform_rgb.shape[1] == 3
