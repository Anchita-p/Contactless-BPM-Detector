import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

LOW_CUTOFF_HZ = 0.7   # 42 BPM
HIGH_CUTOFF_HZ = 2.5  # 150 BPM
FILTER_ORDER = 2


def bandpass_filter(signal, fs, low_hz=LOW_CUTOFF_HZ, high_hz=HIGH_CUTOFF_HZ, order=FILTER_ORDER):
    """2nd-order Butterworth bandpass filter (0.7-2.5 Hz), zero-phase via filtfilt.

    Args:
        signal: 1D array-like pulse signal.
        fs: sampling rate of `signal` in Hz.

    Returns:
        1D numpy array, same shape as `signal`, band-limited to the
        physiologically plausible heart-rate range.
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = 0.5 * fs
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, signal)  # forward-backward filtering -> zero phase delay


def temporal_mean_normalize(rgb, window):
    """Divide each RGB channel by its own trailing rolling mean.

    Camera auto-exposure and changing room light move all three channels by a
    slow common factor that has nothing to do with blood volume. Dividing by a
    trailing mean shorter than the analysis window removes that drift before
    POS ever sees it. POS's own whole-window normalization stays in place --
    this is an additional, shorter-baseline stage, not a replacement.

    Args:
        rgb: array of shape (N, 3).
        window: trailing mean length in samples. Samples before the first full
            window use an expanding mean, so callers wanting a uniform
            baseline should pass `window - 1` extra lead-in samples and
            discard the head of the result.

    Returns:
        Array of shape (N, 3), each value divided by its channel's trailing mean.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape (N, 3), got {rgb.shape}")
    window = int(window)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = rgb.shape[0]
    prefix = np.vstack([np.zeros((1, 3)), np.cumsum(rgb, axis=0)])
    idx = np.arange(n)
    lo = np.maximum(idx - window + 1, 0)
    trailing_sum = prefix[idx + 1] - prefix[lo]
    means = trailing_sum / (idx - lo + 1)[:, None]

    if np.any(means == 0):
        raise ValueError("Trailing channel mean is zero; cannot normalize.")
    return rgb / means


def resample_to_uniform_rate(rgb, timestamps, target_fs, kind="cubic"):
    """Resample a variable-rate RGB time series onto a uniform time grid.

    Webcam capture rate jitters frame-to-frame, which smears the FFT bin
    width used for BPM estimation. Interpolating onto a fixed-rate grid
    decouples the FFT's sampling rate from the camera's actual (variable)
    frame rate, so the frequency resolution stays constant regardless of
    momentary FPS dips.

    Args:
        rgb: array of shape (N, 3), spatially-averaged RGB per captured frame.
        timestamps: array of shape (N,), monotonic capture time in seconds,
            parallel to `rgb` and strictly increasing.
        target_fs: desired uniform sampling rate in Hz.
        kind: interpolation kind passed to scipy.interpolate.interp1d
            (e.g. "cubic" or "linear").

    Returns:
        (uniform_timestamps, uniform_rgb): uniform_timestamps has shape (M,)
        and uniform_rgb has shape (M, 3), evenly spaced at `target_fs` Hz
        across the original capture span.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    # Build the fixed-rate output grid spanning the same time range as the input.
    duration = timestamps[-1] - timestamps[0]
    num_samples = max(int(round(duration * target_fs)) + 1, 2)
    uniform_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)

    # Fit one interpolator per RGB channel (axis=0 broadcasts over the 3 columns).
    interpolator = interp1d(timestamps, rgb, axis=0, kind=kind, assume_sorted=True)
    uniform_rgb = interpolator(uniform_timestamps)

    return uniform_timestamps, uniform_rgb
