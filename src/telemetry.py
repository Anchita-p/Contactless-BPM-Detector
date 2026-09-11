"""Telemetry payload construction for the live dashboard.

Pure functions over the pipeline's result dictionaries — no FastAPI, no camera,
no sockets — so the populated payload can be unit tested without hardware.
"""
import numpy as np

from config import BUFFER_DURATION_S, TARGET_FS
from evaluate import HRV_KEY
from rppg_pos import ALGORITHM_KEYS

PRIMARY_KEY = "pos"  # canonical POS (Wang 2017) is the headline reading

# The dashboard scrolls a 5 s excerpt of the 10 s analysis window.
WAVEFORM_SECONDS = 5.0
WAVEFORM_SAMPLES = int(TARGET_FS * WAVEFORM_SECONDS)

# Spread across the three algorithms, in BPM, below which they are considered
# to corroborate each other. One FFT bin at 30 Hz over 300 samples is 6 BPM, so
# agreement inside that is agreement to the limit of the method's resolution.
AGREEMENT_TOLERANCE_BPM = 6.0

# Tracker states are CALIBRATING / MEASURING plus a separate face_found flag;
# the UI contract asks for these three instead. Mapped here rather than
# renaming the tracker's own states, which ARCHITECTURE.md documents and tests import.
STATE_CALIBRATING = "CALIBRATING"
STATE_TRACKING = "TRACKING"
STATE_FACE_LOST = "FACE_LOST"


def _round(value, digits=1):
    return None if value is None else round(float(value), digits)


def tracker_state(face_found, measuring):
    if not face_found:
        return STATE_FACE_LOST
    return STATE_TRACKING if measuring else STATE_CALIBRATING


def waveform_excerpt(results, key=PRIMARY_KEY, samples=WAVEFORM_SAMPLES):
    """The most recent `samples` of one filtered waveform, normalized to
    roughly [-1, 1] so the frontend can plot it on a fixed axis."""
    if results is None:
        return []

    signal = np.asarray(results[key]["signal"], dtype=np.float64)[-samples:]
    if signal.size == 0:
        return []

    centred = signal - signal.mean()
    peak = np.abs(centred).max()
    scaled = centred if peak == 0 else centred / peak
    return [round(float(v), 4) for v in scaled]


def algorithm_agreement(reading):
    """How closely the three algorithms corroborate each other right now.

    Wide disagreement is the tell-tale of a motion artifact: a real pulse shows
    up in all three projections, a movement artifact usually does not.
    """
    values = [reading[key] for key in ALGORITHM_KEYS if reading[key] is not None]
    if len(values) < 2:
        return {"spread_bpm": None, "consistent": False, "reporting": len(values)}

    spread = max(values) - min(values)
    return {
        "spread_bpm": _round(spread),
        "consistent": bool(spread <= AGREEMENT_TOLERANCE_BPM),
        "reporting": len(values),
    }


def build_payload(reading, results, face_found, fps, buffered_span_s):
    """Assemble one 5 Hz telemetry frame.

    Args:
        reading: smoothed output of ResultSmoother.update.
        results: raw output of compute_algorithm_signals, or None while the
            pipeline is still filling its window.
        face_found: whether the tracker saw a face in the latest frame.
        fps: the tracker's current smoothed capture rate.
        buffered_span_s: seconds of contiguous samples currently buffered.

    Returns:
        A JSON-serializable dict. Every numeric field is either a rounded float
        or None -- None consistently means "not available", never zero.
    """
    hrv = reading[HRV_KEY]
    measuring = results is not None
    progress = min(max(buffered_span_s / BUFFER_DURATION_S, 0.0), 1.0)

    primary = results[PRIMARY_KEY] if measuring else None

    return {
        "bpm": _round(reading[PRIMARY_KEY], 0),
        "stress": {
            "tier": hrv["tier"],
            "rmssd": _round(hrv["rmssd_ms"]),
            "status": hrv["status"],
            "n_beats": hrv["n_beats"],
        },
        "algorithms": {
            key: {
                "bpm": _round(reading[key], 0),
                "label": results[key]["label"] if measuring else None,
                "snr_db": _round(results[key]["snr_db"]) if measuring else None,
                "status": results[key]["status"] if measuring else STATE_CALIBRATING,
            }
            for key in ALGORITHM_KEYS
        },
        "agreement": algorithm_agreement(reading),
        "quality": {
            "snr_db": _round(primary["snr_db"]) if primary else None,
            "fps": _round(fps),
            "state": tracker_state(face_found, measuring),
        },
        "waveform": waveform_excerpt(results),
        "calibration_progress": round(progress, 3),
        "calibration_seconds": round(min(buffered_span_s, BUFFER_DURATION_S), 1),
        "calibration_target_seconds": BUFFER_DURATION_S,
    }
