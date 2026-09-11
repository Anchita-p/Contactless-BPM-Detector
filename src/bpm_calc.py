from collections import deque

import numpy as np

LOW_HZ = 0.7   # 42 BPM
HIGH_HZ = 2.5  # 150 BPM
DEFAULT_BUFFER_SIZE = 8

STATUS_OK = "OK"
STATUS_NO_PULSE = "No Pulse / Low Signal"

# The dominant peak's power must exceed this multiple of the average power
# of the rest of the in-band spectrum to be trusted as a real pulse rather
# than background noise (a static image, no face, or a dead ROI). Because
# argmax always finds *some* peak even in pure noise, this needs real
# margin: empirically, white noise's peak/surrounding ratio at a 300-sample,
# 30 Hz window has a 99th percentile of ~12, while genuine pulse signals
# (even at 1:1 signal:noise amplitude) score 25+ -- 15.0 sits well above the
# noise floor with room to spare below real signals.
SNR_THRESHOLD = 15.0
# Bins immediately around the peak are considered part of the peak itself,
# not the surrounding noise floor, so a narrow true pulse peak doesn't get
# averaged into its own noise estimate.
PEAK_EXCLUDE_BINS = 1
# Absolute floor below which a "peak" is really just numerical zero (e.g. a
# perfectly flat signal), which would otherwise divide-by-zero into a
# meaningless "infinite" SNR.
MIN_PEAK_POWER = 1e-12
# Display ceiling for the reported SNR. A perfectly clean synthetic signal has
# zero surrounding power and would otherwise report infinite dB.
MAX_SNR_DB = 60.0


def peak_snr(filtered_signal, fs):
    """Locate the dominant in-band spectral peak and score its confidence.

    The single source of truth for the SNR figure: both the BPM confidence
    gate and the value reported to the UI come from here, so what the
    dashboard displays can never disagree with the gate that accepted it.

    Returns:
        (snr, dominant_hz) as a linear power ratio and frequency, or
        (None, None) when there is no usable in-band spectrum at all.
    """
    signal = np.asarray(filtered_signal, dtype=np.float64)
    n = signal.size
    if n < 2:
        return None, None

    # Hanning window reduces spectral leakage before the FFT.
    windowed = signal * np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(windowed)) ** 2  # power spectral density

    # Restrict the search to the physiologically plausible heart-rate band.
    in_band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if not np.any(in_band):
        return None, None

    band_freqs = freqs[in_band]
    band_psd = psd[in_band]
    peak_idx = int(np.argmax(band_psd))
    peak_power = band_psd[peak_idx]

    if peak_power <= MIN_PEAK_POWER:
        return None, None

    # Compare the peak against the average power of the remaining in-band bins,
    # excluding the peak's immediate neighborhood.
    lo = max(peak_idx - PEAK_EXCLUDE_BINS, 0)
    hi = min(peak_idx + PEAK_EXCLUDE_BINS + 1, band_psd.size)
    surrounding_mask = np.ones_like(band_psd, dtype=bool)
    surrounding_mask[lo:hi] = False
    surrounding_power = band_psd[surrounding_mask].mean() if np.any(surrounding_mask) else 0.0

    snr = peak_power / surrounding_power if surrounding_power > 0 else np.inf
    return float(snr), _refine_peak_frequency(band_freqs, band_psd, peak_idx)


def _refine_peak_frequency(band_freqs, band_psd, peak_idx):
    """Interpolate the true peak position between FFT bins.

    A 300-sample window at 30 Hz gives bins 0.1 Hz apart -- **6 BPM** -- so
    reporting the bin centre alone makes the displayed rate jump in 6 BPM
    steps as the peak drifts across a boundary. That reads as the measurement
    being unstable when the underlying pulse is steady.

    Fitting a parabola through the peak and its two neighbours (in log power,
    which is the right shape for a windowed sinusoid) recovers the position
    within the bin. Standard practice in frequency estimation, and it costs
    three array lookups.
    """
    if peak_idx == 0 or peak_idx >= band_psd.size - 1:
        return float(band_freqs[peak_idx])  # no neighbour on one side

    left, centre, right = band_psd[peak_idx - 1:peak_idx + 2]
    if min(left, centre, right) <= MIN_PEAK_POWER:
        return float(band_freqs[peak_idx])

    left, centre, right = np.log(left), np.log(centre), np.log(right)
    denominator = left - 2.0 * centre + right
    if denominator == 0:
        return float(band_freqs[peak_idx])

    offset = 0.5 * (left - right) / denominator
    if not np.isfinite(offset) or abs(offset) > 0.5:
        return float(band_freqs[peak_idx])  # not a well-formed peak

    bin_width = band_freqs[1] - band_freqs[0]
    return float(band_freqs[peak_idx] + offset * bin_width)


def snr_to_db(snr):
    """Linear power ratio -> decibels, clamped for display."""
    if snr is None or snr <= 0:
        return None
    if not np.isfinite(snr):
        return MAX_SNR_DB
    return min(float(10.0 * np.log10(snr)), MAX_SNR_DB)


def signal_snr_db(filtered_signal, fs):
    """Confidence of a filtered waveform's dominant peak, in dB, or None."""
    snr, _ = peak_snr(filtered_signal, fs)
    return snr_to_db(snr)


def estimate_bpm_from_signal(filtered_signal, fs):
    """Estimate BPM from a filtered pulse signal via windowed FFT peak search,
    gated by a peak-prominence / SNR confidence check.

    Args:
        filtered_signal: 1D array-like, band-limited pulse signal.
        fs: sampling rate of `filtered_signal` in Hz.

    Returns:
        (bpm, status): `bpm` is a float — the estimated BPM when a confident
        in-band peak is found, or 0.0 when it isn't. `status` is
        STATUS_OK or STATUS_NO_PULSE.
    """
    snr, dominant_hz = peak_snr(filtered_signal, fs)
    if snr is None or snr < SNR_THRESHOLD:
        return 0.0, STATUS_NO_PULSE
    return dominant_hz * 60.0, STATUS_OK  # Hz -> BPM


class BPMSmoother:
    """Moving-average buffer of recent BPM readings, for stable UI display."""

    def __init__(self, buffer_size=DEFAULT_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self._readings = deque(maxlen=buffer_size)

    def reset(self):
        """Drop every buffered reading. Called when the tracker flushes, so a
        stale average isn't displayed as though it were a live measurement."""
        self._readings.clear()

    def update(self, bpm):
        """Push a new reading (if not None) and return the current smoothed BPM."""
        if bpm is not None:
            self._readings.append(bpm)
        if not self._readings:
            return None
        return float(np.mean(self._readings))


class MultiAlgorithmSmoother:
    """One BPMSmoother per algorithm, reset as a unit.

    A tracker flush invalidates every algorithm's history at once, so they
    have to clear together -- resetting only one would leave the others
    displaying a pre-flush average as though it were a live measurement.
    """

    def __init__(self, keys, buffer_size=DEFAULT_BUFFER_SIZE):
        self._smoothers = {key: BPMSmoother(buffer_size) for key in keys}

    def reset(self):
        for smoother in self._smoothers.values():
            smoother.reset()

    def update(self, results):
        """Push one round of per-algorithm results and return the smoothed
        BPM for each. `results` is the dict from the extraction pipeline, or
        None when there isn't yet enough data for any algorithm to run.

        Returns: dict of algorithm key -> smoothed BPM (or None).
        """
        return {
            key: smoother.update(None if results is None else results[key]["bpm"])
            for key, smoother in self._smoothers.items()
        }
