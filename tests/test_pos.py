import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rppg_pos import (
    ALGORITHM_KEYS,
    ALGORITHMS,
    extract_pulse_chrom,
    extract_pulse_green,
    extract_pulse_pos,
)

FS = 30


def _simulate_rgb_window(freq_hz, fs=FS, n_frames=90, seed=0):
    """Simulate a plausible rPPG RGB window: a skin-tone DC baseline plus a
    small pulsatile modulation at the given frequency (channel-dependent
    amplitude, as in real blood-volume-pulse color modulation)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fs
    pulse = np.sin(2 * np.pi * freq_hz * t)

    baseline = np.array([160.0, 120.0, 100.0])  # typical skin RGB baseline
    amplitude = np.array([2.0, 3.0, 1.0])        # G channel carries strongest AC component

    rgb = baseline[None, :] + amplitude[None, :] * pulse[:, None]
    rgb += rng.normal(scale=0.05, size=rgb.shape)  # small sensor noise
    return rgb


def _dominant_freq(signal, fs=FS):
    freqs = np.fft.rfftfreq(signal.size, d=1 / fs)
    magnitudes = np.abs(np.fft.rfft(signal - signal.mean()))
    return freqs[np.argmax(magnitudes)]


def _extract_all(rgb, fs=FS):
    return {spec.key: spec.extract(rgb, fs) for spec in ALGORITHMS}


def test_registry_exposes_the_three_compared_algorithms():
    assert ALGORITHM_KEYS == ("pos", "chrom", "green")
    # Only CHROM bandpasses internally, as its alpha step requires.
    assert [spec.self_filtered for spec in ALGORITHMS] == [False, True, False]


def test_pos_uses_the_wang_projection_not_chroms():
    """Regression canary. POS (Wang 2017) is S1 = G-B, S2 = -2R+G+B. An
    earlier version of this module implemented CHROM's projection under the
    POS name; if anyone reintroduces that, POS stops being a distinct row in
    the comparison and this fails."""
    rgb = _simulate_rgb_window(freq_hz=1.2)
    normalized = rgb / rgb.mean(axis=0)
    r, g, b = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    s1 = g - b
    s2 = -2 * r + g + b
    expected = s1 + (np.std(s1) / np.std(s2)) * s2

    np.testing.assert_allclose(extract_pulse_pos(rgb), expected)

    # ...and it is NOT the CHROM projection.
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    chrom_projection = x - (np.std(x) / np.std(y)) * y
    assert not np.allclose(extract_pulse_pos(rgb), chrom_projection)


def test_chrom_computes_alpha_after_bandpassing():
    """CHROM's defining step is that alpha comes from the band-limited X and
    Y, not the raw projections."""
    from filters import bandpass_filter

    rgb = _simulate_rgb_window(freq_hz=1.2)
    normalized = rgb / rgb.mean(axis=0)
    r, g, b = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    xf = bandpass_filter(3 * r - 2 * g, fs=FS)
    yf = bandpass_filter(1.5 * r + g - 1.5 * b, fs=FS)
    expected = xf - (np.std(xf) / np.std(yf)) * yf

    np.testing.assert_allclose(extract_pulse_chrom(rgb, fs=FS), expected)


def test_green_baseline_is_the_normalized_green_channel():
    rgb = _simulate_rgb_window(freq_hz=1.2)
    expected = rgb[:, 1] / rgb[:, 1].mean()
    np.testing.assert_allclose(extract_pulse_green(rgb), expected)


def test_the_three_algorithms_produce_distinct_signals():
    """The comparison is meaningless if two 'algorithms' are the same math.
    Compare correlation, since scale alone would not make them distinct."""
    rgb = _simulate_rgb_window(freq_hz=1.2)
    signals = _extract_all(rgb)

    for a, b in [("pos", "chrom"), ("pos", "green"), ("chrom", "green")]:
        sa, sb = signals[a], signals[b]
        correlation = np.corrcoef(sa - sa.mean(), sb - sb.mean())[0, 1]
        assert abs(correlation) < 0.999, f"{a} and {b} are effectively identical"


@pytest.mark.parametrize("key", ALGORITHM_KEYS)
def test_every_algorithm_recovers_a_known_frequency(key):
    freq_hz = 2.0  # 120 BPM, lands exactly on an FFT bin (k=6 of N=90)
    rgb = _simulate_rgb_window(freq_hz=freq_hz, n_frames=90)
    signal = _extract_all(rgb)[key]

    assert abs(_dominant_freq(signal) - freq_hz) < FS / 90


@pytest.mark.parametrize("key", ALGORITHM_KEYS)
def test_every_algorithm_returns_clean_finite_output(key):
    rgb = _simulate_rgb_window(freq_hz=1.2, n_frames=45)
    signal = _extract_all(rgb)[key]

    assert signal.shape == (45,)
    assert np.all(np.isfinite(signal))


@pytest.mark.parametrize("key", ALGORITHM_KEYS)
def test_every_algorithm_handles_a_constant_signal_without_nan(key):
    constant_rgb = np.tile([150.0, 100.0, 90.0], (60, 1))
    signal = _extract_all(constant_rgb)[key]

    assert signal.shape == (60,)
    assert np.all(np.isfinite(signal))


@pytest.mark.parametrize("key", ALGORITHM_KEYS)
def test_every_algorithm_rejects_wrong_shape(key):
    spec = {s.key: s for s in ALGORITHMS}[key]
    with pytest.raises(ValueError):
        spec.extract(np.zeros((10, 4)), FS)
    with pytest.raises(ValueError):
        spec.extract(np.zeros(10), FS)
