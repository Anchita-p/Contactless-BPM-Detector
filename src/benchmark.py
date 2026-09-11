"""Offline dataset evaluation: ground-truth parsing and error metrics.

Pure functions, no FastAPI and no file-upload handling, so the metric maths can
be tested directly. Ground-truth formats in this space are not standardized
(UBFC-rPPG ships whitespace-separated rows, PURE ships JSON, exports vary), so
the parser here is deliberately tolerant and reports what it decided.
"""
import csv
import io
import math

import numpy as np

from rppg_pos import ALGORITHM_KEYS

# A series whose median falls in this range is heart rate; anything else is a
# raw waveform, a sample index or something we should not silently score
# against. UBFC-rPPG ships the PPG waveform as its FIRST row, so picking by
# position rather than plausibility reads the waveform as BPM.
MIN_PLAUSIBLE_BPM = 25.0
MAX_PLAUSIBLE_BPM = 240.0

TIME_COLUMN_HINTS = ("time", "timestamp", "seconds", "sec", "t")
BPM_COLUMN_HINTS = ("bpm", "hr", "heart_rate", "heartrate", "pulse", "ground_truth", "gt")


def _floats(values):
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(math.nan)
    return out


def _looks_numeric(row):
    if not row:
        return False
    return all(not math.isnan(v) for v in _floats(row))


def _finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _is_time_like(values):
    """Non-decreasing and actually advancing: a clock, not a measurement."""
    arr = _finite(values)
    return arr.size >= 2 and bool(np.all(np.diff(arr) >= 0)) and arr[-1] > arr[0]


def _is_bpm_like(values):
    arr = _finite(values)
    return arr.size >= 1 and MIN_PLAUSIBLE_BPM <= float(np.median(arr)) <= MAX_PLAUSIBLE_BPM


def _pick(header, hints):
    for index, name in enumerate(header):
        cleaned = name.strip().lower().replace(" ", "_")
        if cleaned in hints:
            return index
    for index, name in enumerate(header):
        cleaned = name.strip().lower().replace(" ", "_")
        if any(hint in cleaned for hint in hints):
            return index
    return None


def parse_ground_truth(text, duration_s=None):
    """Read a ground-truth pulse file into (times_s, bpm).

    Accepts, in order of preference:
      * a header naming a time-like and a BPM-like column;
      * two numeric columns, taken as time then BPM;
      * one numeric column of BPM values, spread evenly across `duration_s`.

    Whitespace-separated files (UBFC-rPPG style) are handled as well as commas.

    Returns:
        (times, bpm, note) -- two float arrays plus a human-readable string
        saying how the file was interpreted, so the UI can show its assumption
        rather than silently guessing wrong.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Ground-truth file is empty.")

    delimiter = "," if "," in stripped.splitlines()[0] else None
    if delimiter:
        rows = [r for r in csv.reader(io.StringIO(stripped)) if r]
    else:
        rows = [line.split() for line in stripped.splitlines() if line.split()]

    header = None
    if not _looks_numeric(rows[0]):
        header = rows[0]
        rows = rows[1:]
    if not rows:
        raise ValueError("Ground-truth file has a header but no data rows.")

    columns = list(zip(*[_floats(r) for r in rows if r]))
    if not columns:
        raise ValueError("Could not read any numeric columns from the ground truth.")

    # A UBFC-style file is 3 very long rows rather than 3 columns; transpose it.
    transposed = len(columns) > len(rows) and len(rows) <= 4
    if transposed:
        columns = [_floats(r) for r in rows]

    if header is not None:
        time_idx = _pick(header, TIME_COLUMN_HINTS)
        bpm_idx = _pick(header, BPM_COLUMN_HINTS)
        if bpm_idx is not None:
            bpm = np.asarray(columns[bpm_idx], dtype=float)
            if time_idx is not None and time_idx != bpm_idx:
                times = np.asarray(columns[time_idx], dtype=float)
                return times, bpm, f"columns '{header[time_idx]}' and '{header[bpm_idx]}'"
            times = _even_times(bpm.size, duration_s)
            return times, bpm, f"column '{header[bpm_idx]}', timestamps assumed even"

    # No usable header: choose by plausibility rather than by position. The
    # series that looks like a heart rate is the heart rate, wherever it sits.
    unit = "row" if transposed else "column"
    bpm_idx = next((i for i, c in enumerate(columns) if _is_bpm_like(c)), None)

    if bpm_idx is not None:
        bpm = np.asarray(columns[bpm_idx], dtype=float)
        time_idx = next((i for i, c in enumerate(columns)
                         if i != bpm_idx and _is_time_like(c)), None)
        if time_idx is not None:
            return (np.asarray(columns[time_idx], dtype=float), bpm,
                    f"{unit} {time_idx + 1} read as time, {unit} {bpm_idx + 1} as BPM")
        return (_even_times(bpm.size, duration_s), bpm,
                f"{unit} {bpm_idx + 1} read as BPM, timestamps assumed even")

    raise ValueError(
        "Could not find a heart-rate series in that file: no column or row has "
        f"a median between {MIN_PLAUSIBLE_BPM:.0f} and {MAX_PLAUSIBLE_BPM:.0f} BPM. "
        "If your file stores something other than BPM (a raw PPG waveform, for "
        "instance), convert it to beats per minute first.")


def _even_times(count, duration_s):
    if count < 2:
        return np.zeros(count, dtype=float)
    span = duration_s if duration_s else float(count - 1)
    return np.linspace(0.0, span, count)


def align_ground_truth(pred_times, gt_times, gt_bpm):
    """Interpolate ground truth onto the prediction timestamps.

    Values outside the ground truth's own time span come back as NaN rather
    than being clamped, so extrapolated points cannot silently flatter the
    error metrics.
    """
    pred_times = np.asarray(pred_times, dtype=float)
    gt_times = np.asarray(gt_times, dtype=float)
    gt_bpm = np.asarray(gt_bpm, dtype=float)

    usable = np.isfinite(gt_times) & np.isfinite(gt_bpm)
    gt_times, gt_bpm = gt_times[usable], gt_bpm[usable]
    if gt_times.size == 0:
        return np.full(pred_times.shape, np.nan)

    order = np.argsort(gt_times)
    gt_times, gt_bpm = gt_times[order], gt_bpm[order]

    aligned = np.interp(pred_times, gt_times, gt_bpm)
    outside = (pred_times < gt_times[0]) | (pred_times > gt_times[-1])
    aligned[outside] = np.nan
    return aligned


# Readings this close to the truth are conventionally counted as correct in
# heart-rate validation work; it is roughly the resolution of the method.
ACCURACY_TOLERANCE_BPM = 5.0


def error_metrics(predicted, truth):
    """Score one algorithm's BPM series against the ground truth.

    Each metric answers a different question, which is why several are
    reported rather than one:

      mae      Typical error, in BPM. The headline number.
      rmse     Also BPM, but squares errors first, so it punishes occasional
               large misses. rmse >> mae means rare bad failures rather than
               steady drift.
      mse      RMSE squared, in BPM^2. Reported because papers cite it;
               it carries no information rmse does not, and its units are
               not interpretable.
      mape     Error as a percentage of the true rate, so results are
               comparable across people whose resting rates differ.
      r        Pearson correlation. Answers "does it track changes?"
               independently of whether it reads consistently high or low --
               a method can have poor mae but excellent r if it is simply
               offset.
      bias     Mean signed error (Bland-Altman). Positive means the algorithm
               systematically over-reads. mae hides direction; this shows it.
      loa      Bland-Altman limits of agreement, bias +/- 1.96 sd. The range
               within which 95% of individual readings fall -- the figure a
               clinical comparison actually turns on.
      within_5 Share of readings inside ACCURACY_TOLERANCE_BPM of the truth.

    Metrics are computed only where both series have a value, and `n` is
    reported alongside because a score over five overlapping samples is not
    comparable to one over five hundred.
    """
    predicted = np.asarray([np.nan if v is None else v for v in predicted], dtype=float)
    truth = np.asarray(truth, dtype=float)

    empty = {"mae": None, "rmse": None, "mse": None, "mape": None,
             "r": None, "bias": None, "loa": None, "within_5": None, "n": 0}

    both = np.isfinite(predicted) & np.isfinite(truth)
    if not np.any(both):
        return empty

    pred, true = predicted[both], truth[both]
    error = pred - true
    mse = float(np.mean(error ** 2))

    # Percentage error is undefined where the true rate is zero.
    nonzero = true != 0
    mape = (round(float(np.mean(np.abs(error[nonzero] / true[nonzero])) * 100.0), 2)
            if np.any(nonzero) else None)

    # Correlation needs at least two points and variation in both series.
    r = None
    if pred.size >= 2 and pred.std() > 0 and true.std() > 0:
        r = round(float(np.corrcoef(pred, true)[0, 1]), 3)

    bias = float(np.mean(error))
    spread = float(np.std(error, ddof=1)) if error.size > 1 else 0.0

    return {
        "mae": round(float(np.mean(np.abs(error))), 2),
        "rmse": round(float(np.sqrt(mse)), 2),
        "mse": round(mse, 2),
        "mape": mape,
        "r": r,
        "bias": round(bias, 2),
        "loa": [round(bias - 1.96 * spread, 2), round(bias + 1.96 * spread, 2)],
        "within_5": round(float(np.mean(np.abs(error) <= ACCURACY_TOLERANCE_BPM) * 100.0), 1),
        "n": int(both.sum()),
    }


def summarize_series(values):
    """Mean of a series that may contain None, or None if it is entirely empty."""
    usable = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.mean(usable)), 2) if usable else None


def evaluate_readings(readings, snr_series, ground_truth=None, duration_s=None):
    """Turn evaluate_video output into the dataset-evaluation response.

    Args:
        readings: list of (timestamp_s, reading) from evaluate_video.
        snr_series: list of per-reading POS SNR values, parallel to `readings`.
        ground_truth: raw text of an uploaded ground-truth file, or None.
        duration_s: video duration, used when the ground truth carries no
            timestamps of its own.

    Returns:
        A JSON-serializable dict of timestamps, per-algorithm BPM series,
        per-algorithm error metrics, the aligned ground truth and mean SNR.
    """
    timestamps = [round(float(t), 2) for t, _ in readings]
    series = {key: [reading[key] for _, reading in readings] for key in ALGORITHM_KEYS}

    response = {
        "timestamps": timestamps,
        "algorithms": {
            key: {
                "bpm": series[key],
                "mean_bpm": summarize_series(series[key]),
                "metrics": error_metrics([], []),
            }
            for key in ALGORITHM_KEYS
        },
        "ground_truth": None,
        "ground_truth_note": None,
        "mean_snr_db": summarize_series(snr_series),
        "frames_evaluated": len(readings),
    }

    if not ground_truth:
        return response

    gt_times, gt_bpm, note = parse_ground_truth(ground_truth, duration_s=duration_s)
    aligned = align_ground_truth(timestamps, gt_times, gt_bpm)

    response["ground_truth"] = [None if math.isnan(v) else round(float(v), 2) for v in aligned]
    response["ground_truth_note"] = note
    for key in ALGORITHM_KEYS:
        response["algorithms"][key]["metrics"] = error_metrics(series[key], aligned)
    return response
