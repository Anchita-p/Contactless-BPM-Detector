import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import BUFFER_DURATION_S
from evaluate import HRV_KEY, ResultSmoother, compute_algorithm_signals
from rppg_pos import ALGORITHM_KEYS
from telemetry import (
    STATE_CALIBRATING,
    STATE_FACE_LOST,
    STATE_TRACKING,
    WAVEFORM_SAMPLES,
    algorithm_agreement,
    build_payload,
    tracker_state,
    waveform_excerpt,
)
from test_evaluate import _synthetic_uniform_rgb


def _populated():
    """A real pipeline result where all three algorithms report, so the
    populated payload can be exercised without a camera."""
    rgb, t = _synthetic_uniform_rgb(freq_hz=1.2)
    results = compute_algorithm_signals(rgb, t)
    smoother = ResultSmoother()
    for _ in range(3):
        reading = smoother.update(results)
    return reading, results


def test_state_mapping_covers_the_three_ui_states():
    assert tracker_state(face_found=False, measuring=True) == STATE_FACE_LOST
    assert tracker_state(face_found=False, measuring=False) == STATE_FACE_LOST
    assert tracker_state(face_found=True, measuring=False) == STATE_CALIBRATING
    assert tracker_state(face_found=True, measuring=True) == STATE_TRACKING


def test_payload_is_fully_populated_when_the_pipeline_reports():
    reading, results = _populated()
    payload = build_payload(reading, results, face_found=True, fps=30.0,
                            buffered_span_s=BUFFER_DURATION_S)

    assert payload["bpm"] is not None
    assert payload["quality"]["state"] == STATE_TRACKING
    assert payload["quality"]["fps"] == 30.0
    assert payload["quality"]["snr_db"] is not None
    assert payload["calibration_progress"] == 1.0
    assert len(payload["waveform"]) == WAVEFORM_SAMPLES

    for key in ALGORITHM_KEYS:
        entry = payload["algorithms"][key]
        assert entry["bpm"] is not None, f"{key} reported nothing on a clean window"
        assert entry["label"]
        assert entry["snr_db"] is not None


def test_payload_while_calibrating_reports_progress_not_values():
    smoother = ResultSmoother()
    reading = smoother.update(None)
    payload = build_payload(reading, None, face_found=True, fps=29.4,
                            buffered_span_s=BUFFER_DURATION_S / 2)

    assert payload["bpm"] is None
    assert payload["quality"]["state"] == STATE_CALIBRATING
    assert payload["quality"]["snr_db"] is None
    assert payload["waveform"] == []
    assert payload["calibration_progress"] == pytest.approx(0.5)
    assert all(payload["algorithms"][key]["bpm"] is None for key in ALGORITHM_KEYS)


def test_calibration_progress_is_clamped_to_one():
    smoother = ResultSmoother()
    reading = smoother.update(None)
    payload = build_payload(reading, None, face_found=True, fps=30.0,
                            buffered_span_s=BUFFER_DURATION_S * 5)
    assert payload["calibration_progress"] == 1.0
    assert payload["calibration_seconds"] == BUFFER_DURATION_S


def test_payload_is_json_serializable():
    """numpy floats survive arithmetic but not json.dumps -- the payload must
    contain plain Python numbers or the WebSocket send fails at runtime."""
    import json

    reading, results = _populated()
    payload = build_payload(reading, results, face_found=True, fps=30.0,
                            buffered_span_s=BUFFER_DURATION_S)
    encoded = json.dumps(payload)
    assert json.loads(encoded)["bpm"] == payload["bpm"]


def test_waveform_is_normalized_and_trimmed():
    _, results = _populated()
    wave = waveform_excerpt(results)

    assert len(wave) == WAVEFORM_SAMPLES  # 5 s of the 10 s window
    assert max(abs(v) for v in wave) == pytest.approx(1.0, abs=1e-6)


def test_waveform_handles_a_flat_signal_without_dividing_by_zero():
    results = {"pos": {"signal": np.zeros(300)}}
    assert all(v == 0 for v in waveform_excerpt(results))


def test_agreement_flags_a_wide_spread_as_inconsistent():
    close = {"pos": 72.0, "chrom": 74.0, "green": 71.0, HRV_KEY: None}
    assert algorithm_agreement(close)["consistent"] is True

    wide = {"pos": 60.0, "chrom": 95.0, "green": 71.0, HRV_KEY: None}
    result = algorithm_agreement(wide)
    assert result["consistent"] is False
    assert result["spread_bpm"] == 35.0


def test_agreement_needs_two_reporting_algorithms():
    lonely = {"pos": 72.0, "chrom": None, "green": None, HRV_KEY: None}
    result = algorithm_agreement(lonely)
    assert result["spread_bpm"] is None
    assert result["reporting"] == 1


def test_agreement_reports_how_many_algorithms_actually_contributed():
    """Regression: a live run with POS 82, CHROM 78 and Raw Green finding no
    pulse was described in the UI as "all three methods agree". Two agreeing
    is not three, and the payload has to carry the count so the wording can
    reflect it."""
    two_of_three = {"pos": 82.0, "chrom": 78.0, "green": None, HRV_KEY: None}
    result = algorithm_agreement(two_of_three)

    assert result["reporting"] == 2
    assert result["consistent"] is True     # the two that reported do agree...
    assert result["spread_bpm"] == 4.0      # ...but the UI must not claim three

    all_three = {"pos": 82.0, "chrom": 78.0, "green": 80.0, HRV_KEY: None}
    assert algorithm_agreement(all_three)["reporting"] == 3
