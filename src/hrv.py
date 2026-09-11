"""Heart Rate Variability -> Stress Index, from the filtered BVP waveform.

Pipeline: cubic-spline upsample to UPSAMPLE_FS -> peak detection -> inter-beat
intervals -> screening -> RMSSD -> categorical tier.

Two caveats that belong with any number this module produces:

1. The upsample buys *resolution*, not accuracy. The input has already been
   through a 2nd-order Butterworth at 0.7-2.5 Hz, which strips the harmonics
   that give a pulse its sharp systolic peak, so what gets interpolated is
   close to a sinusoid. Sub-frame peak positions are real in the sense that
   they beat 33 ms quantization, but the underlying timing was already smeared
   by the band limit.

2. RMSSD over the pipeline's 10 s window rests on roughly 9 intervals at a
   resting heart rate. Conventional short-term HRV reporting starts at 30 s.
   See tests/test_hrv.py::test_rmssd_spread_across_windows_is_reported for the
   measured window-to-window spread on this pipeline.

3. The same narrow passband acts as a resonator, so a beat that is entirely
   absent gets reconstructed by filter ringing rather than showing up as a
   gap. This pipeline therefore cannot detect dropped beats: under motion or
   low SNR it reports plausible regular intervals instead of flagging that
   detection failed. screen_ibis stays as a defensive net, but this filter
   rarely lets an artifact reach it -- see
   tests/test_hrv.py::test_bandpass_reconstructs_a_missing_beat_so_dropouts_are_invisible.
"""
from collections import deque

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

UPSAMPLE_FS = 250.0

# Physiologically plausible interval bounds, matching the 42-150 BPM band the
# rest of the pipeline already enforces.
MIN_IBI_S = 60.0 / 150.0   # 0.40 s
MAX_IBI_S = 60.0 / 42.0    # ~1.43 s

# A peak must stand this many standard deviations proud of its surroundings.
# Without it, ripple in a band-limited signal registers as extra beats.
MIN_PROMINENCE_STD = 0.3

# The 0.7-2.5 Hz passband admits a pulse's 2nd harmonic for any rate below
# 75 BPM, and that harmonic puts a secondary peak in the middle of every beat.
# Spacing peaks off the absolute 0.4 s band edge therefore double-counts most
# resting heart rates. When the FFT has already told us the rate, require
# peaks to be at least this fraction of the expected interval apart -- loose
# enough for real beat-to-beat variability, tight enough to reject harmonics.
MIN_IBI_FRACTION_OF_EXPECTED = 0.7

# One missed or doubled beat produces an interval ~2x or ~0.5x the true one,
# which inflates RMSSD by an order of magnitude and flips the tier on its own.
# Intervals this far from the window median are treated as detection errors.
MAX_MEDIAN_DEVIATION = 0.2

# RMSSD needs successive differences; below this the estimate is meaningless
# even by this module's already-loose standards.
MIN_USABLE_IBIS = 4

# Tier thresholds, in milliseconds. RMSSD is *inversely* related to stress:
# high variability means a relaxed, parasympathetically-driven heart.
#
# WARNING: these are population-level resting figures derived from ECG, not
# validated against rPPG on this pipeline. They are a starting point to be
# calibrated against ground truth before any published claim rests on them.
HIGH_STRESS_BELOW_MS = 20.0
LOW_STRESS_ABOVE_MS = 50.0

TIER_HIGH = "High"
TIER_MODERATE = "Moderate"
TIER_LOW = "Low"

# Readings arrive from heavily overlapping windows, so averaging them buys much
# less than sqrt(N) -- the real cause of visible tier flicker is a value sitting
# near a threshold and drifting back and forth across it. Hysteresis fixes that
# directly: once in a tier, the value has to clear the boundary by this margin
# before the label changes.
TIER_HYSTERESIS_MS = 3.0
HRV_SMOOTHING_BUFFER = 8

STATUS_OK = "OK"
STATUS_NO_PULSE = "No Pulse / Low Signal"
STATUS_TOO_FEW_BEATS = "Too Few Beats"


def _result(rmssd_ms, tier, n_beats, status):
    return {"rmssd_ms": rmssd_ms, "tier": tier, "n_beats": n_beats, "status": status}


def no_pulse_result():
    """The HRV entry to report when the source waveform never cleared its own
    pulse-confidence gate -- peak detection on noise would invent beats."""
    return _result(None, None, 0, STATUS_NO_PULSE)


def upsample_waveform(signal, fs, target_fs=UPSAMPLE_FS):
    """Cubic-spline upsample a 1D waveform onto a `target_fs` grid.

    Returns:
        (times, values): both 1D, spanning the same duration as the input.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if signal.size < 4:  # cubic interpolation needs at least degree + 1 points
        raise ValueError(f"need at least 4 samples to upsample, got {signal.size}")

    duration = (signal.size - 1) / fs
    source_times = np.arange(signal.size) / fs
    target_times = np.linspace(0.0, duration, int(round(duration * target_fs)) + 1)

    interpolator = interp1d(source_times, signal, kind="cubic", assume_sorted=True)
    return target_times, interpolator(target_times)


def minimum_beat_spacing_s(expected_bpm=None):
    """Smallest interval two detected peaks may be apart.

    Falls back to the absolute band edge when the heart rate isn't known,
    which is permissive enough to admit harmonic double-peaks -- pass
    `expected_bpm` from the FFT estimate whenever it's available.
    """
    if expected_bpm is None or expected_bpm <= 0:
        return MIN_IBI_S
    return max(MIN_IBI_S, MIN_IBI_FRACTION_OF_EXPECTED * 60.0 / expected_bpm)


def detect_beat_times(times, values, expected_bpm=None):
    """Locate systolic peaks and return their times in seconds."""
    spread = np.std(values)
    if spread == 0:
        return np.empty(0)

    sample_period = times[1] - times[0]
    spacing_samples = max(int(round(minimum_beat_spacing_s(expected_bpm) / sample_period)), 1)
    peaks, _ = find_peaks(
        values,
        distance=spacing_samples,
        prominence=MIN_PROMINENCE_STD * spread,
    )
    return times[peaks]


def screen_ibis(ibis):
    """Drop intervals that reflect detection errors rather than real beats.

    Note the intervals surviving this are no longer strictly *successive* where
    something was removed -- standard practice in HRV artifact correction, but
    it does mean RMSSD is computed across an occasional splice.
    """
    ibis = np.asarray(ibis, dtype=np.float64)
    plausible = ibis[(ibis >= MIN_IBI_S) & (ibis <= MAX_IBI_S)]
    if plausible.size == 0:
        return plausible

    median = np.median(plausible)
    return plausible[np.abs(plausible - median) <= MAX_MEDIAN_DEVIATION * median]


def compute_rmssd(ibis_s):
    """Root Mean Square of Successive Differences, in milliseconds."""
    differences = np.diff(np.asarray(ibis_s, dtype=np.float64))
    return float(np.sqrt(np.mean(differences ** 2)) * 1000.0)


def classify_stress(rmssd_ms):
    """Map RMSSD onto a qualitative tier. Lower variability -> higher stress."""
    if rmssd_ms < HIGH_STRESS_BELOW_MS:
        return TIER_HIGH
    if rmssd_ms > LOW_STRESS_ABOVE_MS:
        return TIER_LOW
    return TIER_MODERATE


def _tier_bounds(tier):
    """The RMSSD range a tier owns, as (low, high)."""
    if tier == TIER_HIGH:
        return -np.inf, HIGH_STRESS_BELOW_MS
    if tier == TIER_LOW:
        return LOW_STRESS_ABOVE_MS, np.inf
    return HIGH_STRESS_BELOW_MS, LOW_STRESS_ABOVE_MS


def classify_stress_with_hysteresis(rmssd_ms, current_tier):
    """Tier classification that resists boundary chatter.

    Stays in `current_tier` until the value leaves that tier's range by
    TIER_HYSTERESIS_MS, so a reading hovering on a threshold doesn't alternate
    between two labels. Pass None as `current_tier` for the first reading.
    """
    if current_tier is None:
        return classify_stress(rmssd_ms)

    low, high = _tier_bounds(current_tier)
    if low - TIER_HYSTERESIS_MS <= rmssd_ms <= high + TIER_HYSTERESIS_MS:
        return current_tier
    return classify_stress(rmssd_ms)


class HRVSmoother:
    """Moving average over RMSSD, with a hysteretic tier derived from it.

    The tier is re-derived from the *smoothed* RMSSD rather than being voted on
    across raw tiers -- smoothing the continuous metric and classifying once is
    what keeps the label consistent with the number displayed beside it.
    """

    def __init__(self, buffer_size=HRV_SMOOTHING_BUFFER):
        self._readings = deque(maxlen=buffer_size)
        self._tier = None

    def reset(self):
        self._readings.clear()
        self._tier = None

    def update(self, hrv_result):
        """Fold one window's HRV result in and return the smoothed version.

        The returned dict keeps every key of the input, with `rmssd_ms` and
        `tier` replaced by their smoothed values and the window's own estimate
        preserved as `rmssd_raw_ms` for plotting and for the paper.
        """
        raw = hrv_result["rmssd_ms"]
        if raw is not None:
            self._readings.append(raw)

        if not self._readings:
            self._tier = None
            return {**hrv_result, "rmssd_raw_ms": raw, "rmssd_ms": None, "tier": None}

        smoothed = float(np.mean(self._readings))
        self._tier = classify_stress_with_hysteresis(smoothed, self._tier)
        return {**hrv_result, "rmssd_raw_ms": raw,
                "rmssd_ms": smoothed, "tier": self._tier}


def compute_hrv(filtered_signal, fs, expected_bpm=None):
    """Full HRV pipeline over one filtered BVP window.

    Args:
        filtered_signal: 1D band-limited pulse waveform.
        fs: its sampling rate in Hz.
        expected_bpm: the FFT's rate estimate for this same window. Strongly
            recommended -- without it, peak spacing falls back to the 0.4 s
            band edge and harmonic peaks are counted as extra beats.

    Returns:
        dict with "rmssd_ms" (float or None), "tier" (str or None),
        "n_beats" (int, before interval screening) and "status". `rmssd_ms` is
        None whenever too few usable intervals survive; `n_beats` is exposed so
        the UI can show why a tier is missing rather than just blanking it.
    """
    signal = np.asarray(filtered_signal, dtype=np.float64)
    if signal.size < 4 or not np.all(np.isfinite(signal)):
        return no_pulse_result()

    times, values = upsample_waveform(signal, fs)
    beat_times = detect_beat_times(times, values, expected_bpm)
    usable = screen_ibis(np.diff(beat_times))

    if usable.size < MIN_USABLE_IBIS:
        return _result(None, None, int(beat_times.size), STATUS_TOO_FEW_BEATS)

    rmssd = compute_rmssd(usable)
    return _result(rmssd, classify_stress(rmssd), int(beat_times.size), STATUS_OK)
