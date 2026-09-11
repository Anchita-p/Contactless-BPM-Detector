import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import (
    align_ground_truth,
    error_metrics,
    evaluate_readings,
    parse_ground_truth,
    summarize_series,
)
from evaluate import HRV_KEY
from rppg_pos import ALGORITHM_KEYS


def test_parses_a_csv_with_named_time_and_bpm_columns():
    times, bpm, note = parse_ground_truth("time,bpm\n0.0,70\n1.0,72\n2.0,74\n")
    np.testing.assert_allclose(times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(bpm, [70, 72, 74])
    assert "bpm" in note


def test_parses_alternative_column_names():
    _, bpm, _ = parse_ground_truth("timestamp,heart_rate\n0,60\n1,61\n")
    np.testing.assert_allclose(bpm, [60, 61])


def test_parses_two_bare_numeric_columns_as_time_and_bpm():
    times, bpm, note = parse_ground_truth("0.0,70\n0.5,71\n1.0,72\n")
    np.testing.assert_allclose(times, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(bpm, [70, 71, 72])
    assert "column 2 as BPM" in note


def test_parses_a_single_bpm_column_spread_over_the_video_duration():
    times, bpm, note = parse_ground_truth("70\n72\n74\n76\n", duration_s=30.0)
    np.testing.assert_allclose(bpm, [70, 72, 74, 76])
    assert times[0] == 0.0 and times[-1] == pytest.approx(30.0)
    assert "assumed even" in note


def test_parses_whitespace_separated_ubfc_style_rows():
    """UBFC-rPPG ships three long whitespace-separated rows: the raw PPG
    waveform FIRST, then heart rate, then timestamps.

    Regression: an earlier version picked by position and read the waveform as
    BPM, silently scoring every metric against values near zero. The series has
    to be chosen by plausibility instead.
    """
    text = "0.1 -0.4 0.9 -0.2\n70 71 72 73\n0.0 1.0 2.0 3.0\n"
    times, bpm, note = parse_ground_truth(text)

    np.testing.assert_allclose(bpm, [70, 71, 72, 73])
    np.testing.assert_allclose(times, [0.0, 1.0, 2.0, 3.0])
    assert "row 2 as BPM" in note


def test_bpm_series_is_found_wherever_it_sits():
    """Column order varies between exports; the heart rate is whichever series
    actually looks like a heart rate."""
    bpm_first = "72 73 74\n0.0 1.0 2.0\n"
    times, bpm, _ = parse_ground_truth(bpm_first)
    np.testing.assert_allclose(bpm, [72, 73, 74])
    np.testing.assert_allclose(times, [0.0, 1.0, 2.0])


def test_a_file_with_no_plausible_heart_rate_is_rejected():
    """A raw waveform on its own cannot be scored against, and saying so beats
    reporting confident metrics computed from nonsense."""
    with pytest.raises(ValueError, match="heart-rate series"):
        parse_ground_truth("0.1 -0.4 0.9 -0.2 0.3 -0.8\n")


def test_empty_ground_truth_is_a_clear_error():
    with pytest.raises(ValueError):
        parse_ground_truth("   \n  ")


def test_alignment_marks_points_outside_the_truth_span_as_nan():
    """Extrapolated points must not silently flatter the error metrics."""
    aligned = align_ground_truth([0.0, 5.0, 50.0], [0.0, 10.0], [60.0, 80.0])
    assert aligned[0] == pytest.approx(60.0)
    assert aligned[1] == pytest.approx(70.0)  # interpolated
    assert np.isnan(aligned[2])               # beyond the ground truth


def test_metrics_match_a_hand_computed_value():
    metrics = error_metrics([70.0, 80.0, 90.0], [72.0, 76.0, 90.0])
    # errors: -2, +4, 0
    assert metrics["mae"] == pytest.approx((2 + 4 + 0) / 3, abs=0.01)
    assert metrics["rmse"] == pytest.approx(np.sqrt((4 + 16 + 0) / 3), abs=0.01)
    assert metrics["n"] == 3


def test_metrics_skip_points_where_either_side_is_missing():
    metrics = error_metrics([70.0, None, 90.0], [72.0, 76.0, np.nan])
    assert metrics["n"] == 1
    assert metrics["mae"] == pytest.approx(2.0)


def test_metrics_report_zero_samples_rather_than_crashing():
    metrics = error_metrics([None, None], [np.nan, np.nan])
    assert metrics["n"] == 0
    assert all(metrics[k] is None for k in ("mae", "rmse", "mse", "mape", "r", "bias", "loa", "within_5"))


def test_mse_is_rmse_squared():
    m = error_metrics([70.0, 80.0, 90.0], [72.0, 76.0, 90.0])
    assert m["mse"] == pytest.approx(m["rmse"] ** 2, abs=0.05)


def test_mape_is_scale_free():
    """The same 5 BPM error is a bigger deal at 50 BPM than at 150."""
    slow = error_metrics([55.0, 55.0], [50.0, 50.0])
    fast = error_metrics([155.0, 155.0], [150.0, 150.0])
    assert slow["mae"] == fast["mae"] == pytest.approx(5.0)
    assert slow["mape"] > fast["mape"]
    assert slow["mape"] == pytest.approx(10.0, abs=0.01)


def test_bias_shows_direction_that_mae_hides():
    """Two algorithms with identical MAE, one reading high and one reading
    low, must be distinguishable."""
    high = error_metrics([75.0, 75.0], [70.0, 70.0])
    low = error_metrics([65.0, 65.0], [70.0, 70.0])
    assert high["mae"] == low["mae"]
    assert high["bias"] == pytest.approx(5.0)
    assert low["bias"] == pytest.approx(-5.0)


def test_correlation_rewards_tracking_even_with_an_offset():
    """A method that is consistently 10 BPM high still tracks perfectly."""
    offset = error_metrics([70.0, 80.0, 90.0, 100.0], [60.0, 70.0, 80.0, 90.0])
    assert offset["r"] == pytest.approx(1.0)
    assert offset["mae"] == pytest.approx(10.0)
    assert offset["bias"] == pytest.approx(10.0)


def test_correlation_is_none_when_a_series_never_varies():
    flat = error_metrics([70.0, 70.0, 70.0], [72.0, 74.0, 71.0])
    assert flat["r"] is None


def test_limits_of_agreement_bracket_the_bias():
    m = error_metrics([70.0, 74.0, 78.0, 71.0], [72.0, 72.0, 72.0, 72.0])
    lo, hi = m["loa"]
    assert lo < m["bias"] < hi


def test_within_5_counts_readings_close_enough_to_be_useful():
    m = error_metrics([70.0, 74.0, 90.0, 71.0], [72.0, 72.0, 72.0, 72.0])
    assert m["within_5"] == pytest.approx(75.0)  # three of four inside 5 BPM


def test_summarize_series_ignores_missing_values():
    assert summarize_series([10.0, None, 20.0]) == 15.0
    assert summarize_series([None, None]) is None


def _readings(values):
    return [
        (float(i), {**{key: v for key in ALGORITHM_KEYS}, HRV_KEY: {}})
        for i, v in enumerate(values)
    ]


def test_evaluate_readings_without_ground_truth_returns_predictions_only():
    response = evaluate_readings(_readings([70.0, 72.0, 74.0]), [12.0, 13.0, 11.0])

    assert response["ground_truth"] is None
    assert response["mean_snr_db"] == 12.0
    assert response["frames_evaluated"] == 3
    for key in ALGORITHM_KEYS:
        assert response["algorithms"][key]["bpm"] == [70.0, 72.0, 74.0]
        assert response["algorithms"][key]["metrics"]["n"] == 0


def test_evaluate_readings_scores_every_algorithm_against_ground_truth():
    response = evaluate_readings(
        _readings([70.0, 72.0, 74.0]), [12.0, 12.0, 12.0],
        ground_truth="time,bpm\n0,71\n1,71\n2,71\n")

    assert response["ground_truth"] == [71.0, 71.0, 71.0]
    for key in ALGORITHM_KEYS:
        metrics = response["algorithms"][key]["metrics"]
        assert metrics["n"] == 3
        assert metrics["mae"] == pytest.approx((1 + 1 + 3) / 3, abs=0.01)


def test_evaluate_readings_tolerates_algorithms_that_found_no_pulse():
    readings = [(0.0, {**{k: None for k in ALGORITHM_KEYS}, HRV_KEY: {}})]
    response = evaluate_readings(readings, [None], ground_truth="time,bpm\n0,71\n1,71\n")

    assert response["mean_snr_db"] is None
    for key in ALGORITHM_KEYS:
        assert response["algorithms"][key]["metrics"]["n"] == 0
