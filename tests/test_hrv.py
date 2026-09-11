import os
import sys

from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hrv import (
    HIGH_STRESS_BELOW_MS,
    LOW_STRESS_ABOVE_MS,
    MIN_USABLE_IBIS,
    STATUS_NO_PULSE,
    STATUS_OK,
    STATUS_TOO_FEW_BEATS,
    TIER_HIGH,
    TIER_LOW,
    TIER_HYSTERESIS_MS,
    TIER_MODERATE,
    UPSAMPLE_FS,
    HRVSmoother,
    classify_stress_with_hysteresis,
    classify_stress,
    compute_hrv,
    compute_rmssd,
    detect_beat_times,
    minimum_beat_spacing_s,
    screen_ibis,
    upsample_waveform,
)
from filters import bandpass_filter

FS = 30.0
WINDOW_SAMPLES = 300  # the pipeline's 10 s window


def _bvp_from_ibis(ibis_s, fs=FS, n=WINDOW_SAMPLES):
    """Build a band-limited BVP waveform whose beats sit at the cumulative
    sum of `ibis_s`, then push it through the same bandpass the real pipeline
    applies -- so tests see the smeared waveform HRV actually receives."""
    t = np.arange(n) / fs
    beat_times = np.concatenate([[0.0], np.cumsum(ibis_s)])

    # Sum of narrow bumps at each beat time, then band-limit.
    signal = np.zeros_like(t)
    for beat in beat_times:
        signal += np.exp(-0.5 * ((t - beat) / 0.05) ** 2)
    return bandpass_filter(signal, fs=fs)


def _constant_ibis(interval_s, duration_s=10.0):
    return np.full(int(duration_s / interval_s), interval_s)


def test_upsample_reaches_the_target_rate_and_preserves_duration():
    signal = np.sin(2 * np.pi * 1.2 * np.arange(WINDOW_SAMPLES) / FS)
    times, values = upsample_waveform(signal, FS)

    expected_duration = (WINDOW_SAMPLES - 1) / FS
    assert abs(times[-1] - expected_duration) < 1e-9
    assert times.size == values.size
    # ~250 Hz across the window, up from 30.
    assert abs(1.0 / (times[1] - times[0]) - UPSAMPLE_FS) < 1.0


def test_upsample_gives_sub_frame_peak_resolution():
    """A peak deliberately placed between two 30 Hz samples must be located
    more precisely than the 33 ms frame period."""
    true_peak_s = 5.0 + 0.5 / FS  # exactly half a frame off the sample grid
    t = np.arange(WINDOW_SAMPLES) / FS
    signal = np.exp(-0.5 * ((t - true_peak_s) / 0.15) ** 2)

    times, values = upsample_waveform(signal, FS)
    recovered = times[np.argmax(values)]

    coarse = t[np.argmax(signal)]
    assert abs(recovered - true_peak_s) < abs(coarse - true_peak_s)
    assert abs(recovered - true_peak_s) < 1.0 / FS


def test_upsample_rejects_too_few_samples():
    with pytest.raises(ValueError):
        upsample_waveform(np.zeros(3), FS)
    with pytest.raises(ValueError):
        upsample_waveform(np.zeros((10, 2)), FS)


def test_beat_detection_finds_every_beat_at_a_steady_rate():
    ibis = _constant_ibis(1.0)  # 60 BPM
    signal = _bvp_from_ibis(ibis)
    times, values = upsample_waveform(signal, FS)
    beats = detect_beat_times(times, values, expected_bpm=60.0)

    # 10 s at 60 BPM, allowing for edge beats lost to the filter transient.
    assert 8 <= beats.size <= 11
    intervals = np.diff(beats)
    assert np.all(np.abs(intervals - 1.0) < 0.05)


def test_harmonic_peaks_are_not_counted_as_beats():
    """The 0.7-2.5 Hz passband keeps a pulse's 2nd harmonic for any rate below
    75 BPM, putting a secondary peak inside every beat. Spacing peaks off the
    absolute band edge alone counts those as real beats and halves every IBI."""
    ibis = _constant_ibis(1.0)  # 60 BPM -> 2nd harmonic at 2 Hz, in band
    times, values = upsample_waveform(_bvp_from_ibis(ibis), FS)

    naive = detect_beat_times(times, values)                        # no rate hint
    informed = detect_beat_times(times, values, expected_bpm=60.0)  # rate known

    assert naive.size > 15, "fixture no longer reproduces the harmonic double-peak"
    assert informed.size <= 11
    assert np.median(np.diff(informed)) == pytest.approx(1.0, abs=0.05)


def test_beat_spacing_tracks_the_expected_rate():
    assert minimum_beat_spacing_s(None) == pytest.approx(0.4)
    assert minimum_beat_spacing_s(60.0) == pytest.approx(0.7)   # 0.7 * 1.0 s
    assert minimum_beat_spacing_s(120.0) == pytest.approx(0.4)  # floored at band edge
    assert minimum_beat_spacing_s(0.0) == pytest.approx(0.4)


def test_beat_detection_returns_nothing_for_a_flat_signal():
    assert detect_beat_times(np.arange(100) / UPSAMPLE_FS, np.zeros(100)).size == 0


def test_rmssd_of_perfectly_regular_intervals_is_zero():
    assert compute_rmssd(_constant_ibis(0.8)) == pytest.approx(0.0)


def test_rmssd_matches_a_hand_computed_value():
    ibis = np.array([0.80, 0.82, 0.79, 0.83])  # differences: +20, -30, +40 ms
    expected = np.sqrt((20 ** 2 + 30 ** 2 + 40 ** 2) / 3)
    assert compute_rmssd(ibis) == pytest.approx(expected, rel=1e-9)


def test_screening_drops_a_doubled_interval_from_a_missed_beat():
    """A missed beat merges two intervals into one ~2x outlier, which would
    dominate RMSSD entirely if it survived."""
    ibis = np.array([0.80, 0.81, 1.60, 0.79, 0.80, 0.82])
    screened = screen_ibis(ibis)

    assert 1.60 not in screened
    assert screened.size == 5
    assert compute_rmssd(screened) < 30


def test_screening_drops_physiologically_impossible_intervals():
    ibis = np.array([0.80, 0.20, 0.81, 3.00, 0.79, 0.80])
    screened = screen_ibis(ibis)

    assert screened.min() >= 0.4
    assert screened.max() <= 1.43


@pytest.mark.parametrize("rmssd_ms,expected", [
    (5.0, TIER_HIGH),
    (HIGH_STRESS_BELOW_MS - 0.1, TIER_HIGH),
    (HIGH_STRESS_BELOW_MS, TIER_MODERATE),      # boundary is inclusive-moderate
    (35.0, TIER_MODERATE),
    (LOW_STRESS_ABOVE_MS, TIER_MODERATE),
    (LOW_STRESS_ABOVE_MS + 0.1, TIER_LOW),
    (120.0, TIER_LOW),
])
def test_tier_boundaries(rmssd_ms, expected):
    assert classify_stress(rmssd_ms) == expected


def test_higher_variability_never_maps_to_higher_stress():
    tiers = [classify_stress(v) for v in [5, 15, 25, 45, 60, 100]]
    rank = {TIER_HIGH: 2, TIER_MODERATE: 1, TIER_LOW: 0}
    ranks = [rank[t] for t in tiers]
    assert ranks == sorted(ranks, reverse=True)


def test_compute_hrv_end_to_end_on_a_steady_pulse():
    signal = _bvp_from_ibis(_constant_ibis(1.0))
    result = compute_hrv(signal, FS, expected_bpm=60.0)

    assert result["status"] == STATUS_OK
    assert result["rmssd_ms"] is not None
    assert result["tier"] in (TIER_LOW, TIER_MODERATE, TIER_HIGH)
    assert result["n_beats"] >= MIN_USABLE_IBIS + 1


def test_compute_hrv_reports_too_few_beats_rather_than_raising():
    # A 2 s window cannot contain enough beats for RMSSD.
    signal = _bvp_from_ibis(_constant_ibis(1.0, duration_s=2.0), n=60)
    result = compute_hrv(signal, FS, expected_bpm=60.0)

    assert result["rmssd_ms"] is None
    assert result["tier"] is None
    assert result["status"] == STATUS_TOO_FEW_BEATS


def test_compute_hrv_handles_degenerate_input_without_raising():
    for bad in [np.zeros(2), np.full(WINDOW_SAMPLES, np.nan), np.zeros(WINDOW_SAMPLES)]:
        result = compute_hrv(bad, FS, expected_bpm=60.0)
        assert result["rmssd_ms"] is None
        assert result["status"] in (STATUS_NO_PULSE, STATUS_TOO_FEW_BEATS)


def test_a_dropped_beat_does_not_swing_the_reported_rmssd():
    """Screening is what stands between one detection error and a tier flip."""
    ibis = _constant_ibis(0.85)
    jitter = np.random.default_rng(0).normal(scale=0.015, size=ibis.size)
    clean = compute_rmssd(screen_ibis(ibis + jitter))

    with_drop = (ibis + jitter).copy()
    with_drop[4] += with_drop[5]  # merge two beats into one missed detection
    with_drop = np.delete(with_drop, 5)
    dropped = compute_rmssd(screen_ibis(with_drop))

    assert abs(dropped - clean) < 10  # ms


def _windows_at_jitter(rng, true_jitter_ms, n_windows):
    """Repeated independent 10 s windows drawn from one fixed physiology."""
    true_rmssd, recovered, tiers = [], [], []
    for _ in range(n_windows):
        ibis = 0.85 + rng.normal(scale=true_jitter_ms / 1000.0, size=12)
        true_rmssd.append(compute_rmssd(ibis))
        result = compute_hrv(_bvp_from_ibis(ibis), FS, expected_bpm=60.0 / 0.85)
        if result["rmssd_ms"] is not None:
            recovered.append(result["rmssd_ms"])
            tiers.append(result["tier"])
    return np.array(true_rmssd), np.array(recovered), tiers


def test_rmssd_tracks_true_variability_across_the_tier_range():
    """RMSSD magnitude must be usable, not just self-consistent.

    Peak spacing is floored at MIN_IBI_FRACTION_OF_EXPECTED of the expected
    interval, so intervals far below the mean get merged and RMSSD eventually
    saturates. This pins that the saturation point sits well above the
    20/50 ms tier boundaries -- otherwise the top tier would be an artifact of
    the detector rather than physiology.
    """
    rng = np.random.default_rng(11)
    for jitter_ms in (10, 25, 40):
        true_rmssd, recovered, _ = _windows_at_jitter(rng, jitter_ms, 40)
        ratio = recovered.mean() / true_rmssd.mean()
        assert 0.9 < ratio < 1.1, (
            f"at {jitter_ms}ms jitter the recovered RMSSD is {ratio:.2f}x the "
            "true value; the tier-relevant range is no longer linear"
        )


def test_tier_stability_at_fixed_physiology_is_reported(capsys):
    """The decisive measurement for whether the Stress Index is publishable.

    RMSSD over a 10 s window rests on ~9 intervals, so the tier can vary
    between windows drawn from *identical* physiology. This holds the true
    jitter fixed and reports how often the tier changes anyway. The printed
    distribution is the finding; the assertion only pins the headline claim,
    that the modal tier is still correct at each level. Run with -s to see it.
    """
    rng = np.random.default_rng(7)
    expected_modal_tier = {10: TIER_HIGH, 25: TIER_MODERATE, 60: TIER_LOW}

    for jitter_ms, expected_tier in expected_modal_tier.items():
        _, recovered, tiers = _windows_at_jitter(rng, jitter_ms, 60)
        counts = Counter(tiers)
        modal_tier, modal_count = counts.most_common(1)[0]
        agreement = modal_count / len(tiers)

        with capsys.disabled():
            spread = "  ".join(f"{name}:{n / len(tiers):.0%}"
                               for name, n in sorted(counts.items()))
            print(f"\n  true jitter {jitter_ms:>3}ms -> RMSSD "
                  f"{recovered.mean():5.1f}+/-{recovered.std():4.1f}ms | {spread}")

        assert modal_tier == expected_tier
        assert agreement > 0.75, (
            f"at {jitter_ms}ms jitter the tier is only stable {agreement:.0%} "
            "of the time; the Stress Index is reporting noise"
        )


def test_bandpass_reconstructs_a_missing_beat_so_dropouts_are_invisible():
    """A documented limitation, not a feature.

    The 0.7-2.5 Hz Butterworth is narrow enough to act as a resonator: when a
    beat is entirely absent from the waveform, the filter rings through the
    gap and produces a peak where the beat should have been. Detection then
    reports a perfectly regular interval sequence and RMSSD looks healthy.

    Consequence for the Stress Index: this pipeline cannot notice dropped
    beats. Under motion or low SNR it will report plausible, regular IBIs
    rather than flagging unreliable detection. It also means screen_ibis is a
    defensive net this filter rarely exercises -- it earns its place only if
    the passband is ever widened.
    """
    t = np.arange(WINDOW_SAMPLES) / FS
    beat_times = np.arange(0.0, 10.0, 0.85)
    amplitudes = np.ones(beat_times.size)
    amplitudes[5] = 0.0  # this beat does not happen at all

    signal = np.zeros_like(t)
    for beat, amplitude in zip(beat_times, amplitudes):
        signal += amplitude * np.exp(-0.5 * ((t - beat) / 0.05) ** 2)
    signal = bandpass_filter(signal, fs=FS)

    beats = detect_beat_times(*upsample_waveform(signal, FS), expected_bpm=60.0 / 0.85)
    intervals = np.diff(beats)

    # The absent beat is reconstructed: no ~1.7 s gap appears.
    assert intervals.max() < 1.2, (
        "the filter no longer fills beat dropouts -- screen_ibis is now load "
        "bearing and needs an end-to-end test"
    )


def _hrv(rmssd_ms, n_beats=11, status=STATUS_OK):
    return {"rmssd_ms": rmssd_ms, "tier": None, "n_beats": n_beats, "status": status}


def test_smoother_averages_rmssd_and_keeps_the_raw_value():
    smoother = HRVSmoother()
    smoother.update(_hrv(20.0))
    out = smoother.update(_hrv(40.0))

    assert out["rmssd_ms"] == pytest.approx(30.0)
    assert out["rmssd_raw_ms"] == pytest.approx(40.0)
    assert out["n_beats"] == 11


def test_smoother_returns_none_before_any_usable_reading():
    smoother = HRVSmoother()
    out = smoother.update(_hrv(None, n_beats=0, status=STATUS_TOO_FEW_BEATS))

    assert out["rmssd_ms"] is None
    assert out["tier"] is None
    assert out["status"] == STATUS_TOO_FEW_BEATS


def test_smoother_reset_clears_both_average_and_tier():
    smoother = HRVSmoother()
    for _ in range(4):
        smoother.update(_hrv(10.0))
    assert smoother.update(_hrv(10.0))["tier"] == TIER_HIGH

    smoother.reset()
    assert smoother.update(_hrv(None))["tier"] is None


def test_hysteresis_holds_a_tier_until_the_boundary_is_cleared():
    # Sitting just past a boundary is not enough to change the label...
    assert classify_stress_with_hysteresis(LOW_STRESS_ABOVE_MS + 1.0, TIER_MODERATE) == TIER_MODERATE
    assert classify_stress_with_hysteresis(HIGH_STRESS_BELOW_MS - 1.0, TIER_MODERATE) == TIER_MODERATE
    # ...clearing it by more than the margin is.
    margin = TIER_HYSTERESIS_MS + 1.0
    assert classify_stress_with_hysteresis(LOW_STRESS_ABOVE_MS + margin, TIER_MODERATE) == TIER_LOW
    assert classify_stress_with_hysteresis(HIGH_STRESS_BELOW_MS - margin, TIER_MODERATE) == TIER_HIGH


def test_hysteresis_has_no_effect_on_the_first_reading():
    assert classify_stress_with_hysteresis(10.0, None) == TIER_HIGH
    assert classify_stress_with_hysteresis(35.0, None) == TIER_MODERATE
    assert classify_stress_with_hysteresis(80.0, None) == TIER_LOW


def test_smoothing_suppresses_tier_flicker_at_a_boundary(capsys):
    """The reason smoothing was added. A subject whose true RMSSD sits on the
    Moderate/Low boundary makes the raw tier alternate every few readings;
    smoothing plus hysteresis has to collapse that to near-zero changes."""
    rng = np.random.default_rng(3)
    on_boundary_ms = LOW_STRESS_ABOVE_MS  # worst case: exactly on the threshold

    smoother = HRVSmoother()
    raw_tiers, smoothed_tiers = [], []
    for _ in range(60):
        reading = on_boundary_ms + rng.normal(scale=6.0)
        raw_tiers.append(classify_stress(reading))
        smoothed_tiers.append(smoother.update(_hrv(reading))["tier"])

    def changes(seq):
        return sum(a != b for a, b in zip(seq, seq[1:]))

    raw_changes, smoothed_changes = changes(raw_tiers), changes(smoothed_tiers)
    with capsys.disabled():
        print(f"\n  tier changes over 60 readings on the boundary: "
              f"raw={raw_changes}  smoothed={smoothed_changes}")

    assert raw_changes > 10, "fixture no longer produces the flicker being fixed"
    assert smoothed_changes <= 2
