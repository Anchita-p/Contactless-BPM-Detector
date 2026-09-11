"""The three pulse-extraction algorithms compared in the paper.

All three consume the *same* RGB window: already resampled to a uniform rate
and already passed through the pipeline's rolling temporal-mean normalization
(see filters.temporal_mean_normalize), so no algorithm gets a lighting-drift
advantage over the others. Each additionally applies the whole-window mean
normalization its own paper specifies, which is close to a no-op on an
already-normalized window but keeps every function correct when called
standalone.

Naming note: an earlier version of this module implemented the projection
X = 3R-2G, Y = 1.5R+G-1.5B under the name "POS". That is CHROM's projection
(de Haan & Jeanne, 2013), not POS's. It now lives in extract_pulse_chrom
where it belongs, and extract_pulse_pos implements the actual POS projection
from Wang et al. (2017).
"""
from collections import namedtuple

import numpy as np

from filters import bandpass_filter


def _normalize_window(rgb_window):
    """Validate an (N, 3) window and divide each channel by its window mean."""
    rgb = np.asarray(rgb_window, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb_window must have shape (N, 3), got {rgb.shape}")

    mean = rgb.mean(axis=0)
    if np.any(mean == 0):
        raise ValueError("Channel mean is zero; cannot temporally normalize.")
    return rgb / mean


def _ratio_of_deviations(numerator, denominator):
    """The sigma(a)/sigma(b) tuning weight both projections use to cancel the
    specular (non-pulsatile) component. Zero denominator means a flat signal,
    which has no pulse to weight."""
    spread = np.std(denominator)
    return np.std(numerator) / spread if spread != 0 else 0.0


def extract_pulse_pos(rgb_window):
    """POS -- Plane-Orthogonal-to-Skin (Wang et al., 2017).

    Projects onto the plane orthogonal to the skin-tone vector:
    S1 = G - B and S2 = -2R + G + B, combined as h = S1 + alpha * S2.

    Deviation from the paper: POS as published slides a short (~1.6 s) window,
    normalizes and projects within each, then overlap-adds. Here the pipeline
    supplies one normalization for every algorithm so the comparison stays
    controlled, and the projection runs once over the full window.

    Returns:
        1D array of shape (N,): the raw, unfiltered pulse signal.
    """
    normalized = _normalize_window(rgb_window)
    r, g, b = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    s1 = g - b
    s2 = -2 * r + g + b
    return s1 + _ratio_of_deviations(s1, s2) * s2


def extract_pulse_chrom(rgb_window, fs):
    """CHROM -- chrominance-based extraction (de Haan & Jeanne, 2013).

    Projects onto two chrominance axes, X = 3R - 2G and Y = 1.5R + G - 1.5B,
    then combines them as S = Xf - alpha * Yf.

    The bandpass sits *inside* this algorithm by design: CHROM computes alpha
    from the band-limited X and Y, which is the step that distinguishes it
    from a bare projection. The returned signal is therefore already filtered
    -- callers must not filter it again, or it becomes an effectively
    8th-order response and stops being comparable to the other two.

    Returns:
        1D array of shape (N,): the band-limited pulse signal.
    """
    normalized = _normalize_window(rgb_window)
    r, g, b = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    x_filtered = bandpass_filter(3 * r - 2 * g, fs=fs)
    y_filtered = bandpass_filter(1.5 * r + g - 1.5 * b, fs=fs)
    return x_filtered - _ratio_of_deviations(x_filtered, y_filtered) * y_filtered


def extract_pulse_green(rgb_window):
    """Raw green-channel baseline -- the spatial average of G alone.

    No chrominance projection at all. Green carries the strongest
    blood-volume modulation, so this is the control the two projection
    algorithms have to beat.

    Returns:
        1D array of shape (N,): the raw, unfiltered green trace.
    """
    return _normalize_window(rgb_window)[:, 1]


AlgorithmSpec = namedtuple("AlgorithmSpec", "key label extract self_filtered")

# `extract` takes (rgb_window, fs) uniformly so the pipeline can fan out over
# these without special-casing. `self_filtered` marks an algorithm that
# bandpasses internally and must not be filtered again downstream.
ALGORITHMS = (
    AlgorithmSpec("pos", "POS (Wang et al., 2017)",
                  lambda rgb, fs: extract_pulse_pos(rgb), False),
    AlgorithmSpec("chrom", "CHROM (de Haan & Jeanne, 2013)",
                  extract_pulse_chrom, True),
    AlgorithmSpec("green", "Raw Green (baseline)",
                  lambda rgb, fs: extract_pulse_green(rgb), False),
)
ALGORITHM_KEYS = tuple(spec.key for spec in ALGORITHMS)
