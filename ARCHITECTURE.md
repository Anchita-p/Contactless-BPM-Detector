# Project: Contactless rPPG BPM Detector

- Role: <role>Act as a Senior Computer Vision and Signal Processing Engineer.</role>
- Tech Stack: Python 3.11+, OpenCV, MediaPipe (Face Mesh), SciPy, NumPy.
- Core Goal: <goal>Extract heart rate (BPM) from facial video using Remote
  Photoplethysmography (rPPG), comparing three published algorithms, and derive
  an HRV-based Stress Index.</goal>

This file is the source of truth for the backend architecture. Every constant
quoted here is the live value in the module named beside it — if you change one,
change it here too.

## Reference Research

- Benchmark: "Remote photoplethysmography (rPPG) in the wild: Remote heart rate
  imaging via online webcams" (Springer Nature / Behavior Research Methods, 2024).
- Dataset: OSF "rPPG in the wild" repository, project `kv8md`.

> **Standing rule: where a spec in this file conflicts with the published
> algorithm, the paper wins.** Earlier revisions specified CHROM's projection
> (`X = 3R-2G`, `Y = 1.5R+G-1.5B`) under the name "POS". Those are CHROM's
> equations. POS uses a different projection matrix entirely. A comparative
> analysis is worthless if two of its rows are the same math, so any change that
> would collapse two named algorithms onto identical equations is a defect and is
> guarded by a regression test.

## Pipeline Overview

```
[Camera / Video File]
        │
        ▼
[MediaPipe Face Mesh]  ──►  face-oval hull + cheek polygons
        │                    (auto-flush safeguards live here)
        ▼
[Spatial Pixel Averaging]  ──►  one mean (R, G, B) per frame
        │
        ▼
[Rolling Buffer]  ──►  13 s deque, wall-clock or synthetic timestamps
        │
        ▼
[Resample to uniform 30 Hz]  ──►  decouples FFT resolution from camera jitter
        │
        ▼
[Rolling Temporal-Mean Normalization]  ──►  3 s trailing mean, shared by all
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
      [POS]        [CHROM]      [Raw Green]      ← one shared (300, 3) window
        │              │              │
        ▼              ▼              ▼
   [Bandpass]    (self-filtered)  [Bandpass]     ← 2nd-order Butterworth
        │              │              │
        └──────────────┴──────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   [FFT + SNR gate]              [HRV module]    ← POS waveform only
        │                             │
        ▼                             ▼
   [BPM smoothing]            [250 Hz spline → peaks → RMSSD → tier]
        │                             │
        └──────────────┬──────────────┘
                       ▼
              [Unified result dict]
```

## Module Specifications

### 1. Face Tracking & ROI Extraction (`src/tracker.py`)

- MediaPipe Face Mesh, `max_num_faces=1`, `refine_landmarks=False`.
- ROI landmark indices:
  - Face oval (named `FOREHEAD_IDX` for historical reasons — it is the whole
    outline, including chin landmark 152): `10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109`
  - Left cheek: `118, 119, 100, 126, 209, 49, 129`
  - Right cheek: `347, 348, 329, 355, 429, 279, 358`
- The three ROIs are pooled into one combined mask; output is the spatial mean
  of R, G and B over it.
- **What is actually sampled: nearly the whole face.** `roi_mask` takes the
  *convex hull*, so the face-oval polygon fills in everything inside it —
  measured at **83% of the face bounding box**, including nose, nostrils, lips,
  chin and eyes. The two cheek polygons lie inside that hull and are therefore
  redundant. Do not describe this as "forehead and cheeks"; that was wrong in
  earlier revisions of this file and in the README.
- **Whether that is optimal is untested.** Spatial averaging suppresses sensor
  noise as ~1/√N but only for pixels that carry pulse; non-pulsatile area
  (sclera, lips, specular highlight on the nose) dilutes the signal instead.
  Measured on `data/test_clip.mp4`, restricting to a true forehead patch plus
  the cheeks used **one third of the pixels for the same SNR** (best window
  10.8 dB vs 10.5 dB) — within noise on a single weak clip, so no change was
  made. Re-run that comparison on a clip that clears the gate before deciding.
- **Visual UI requirement:** draw small dots (`cv2.circle`) on these exact
  landmark points so the operator can confirm only skin is being analyzed.
- **OPEN DECISION — overlay honesty.** The current overlay draws the oval
  outline plus two cheek patches, which reads as "we sample these patches" when
  in fact the whole hull is sampled and the cheek polygons are redundant.
  Three options were discussed and none chosen yet:
  1. Tint the real sampled region (recommended) — unambiguous, cheap, and it
     tracks whatever the mask becomes if the ROI is later restricted.
  2. Tinted region plus a sparse 60-80 point mesh, for the high-tech look.
  3. Full 468-point MediaPipe tessellation — rejected in discussion: it implies
     point-wise sampling, emphasises the eye/lip regions that contribute least,
     costs ~2,600 anti-aliased edges per frame, and obscures the face the
     operator needs to see to check lighting and framing.
- `process_frame(frame, timestamp=None)` — `timestamp` defaults to
  `time.monotonic()` for live capture; the offline evaluator injects a synthetic
  `frame_index / video_fps` instead, so the capture grid is perfectly uniform.

#### Safeguards (both reset state to `CALIBRATING` and fire `on_flush`)

| Trigger | Constant | Value | Rationale |
|---|---|---|---|
| Face lost | `MAX_CONSECUTIVE_MISSES` | 15 | Flushes on the **16th** consecutive miss. Beyond that the buffer would splice two unrelated stretches of signal across a gap. |
| ROI centroid jump | `MAX_CENTROID_SHIFT_RATIO` | 0.15 | Fraction of the ROI bounding-box diagonal in one frame. Normalized by face scale so the threshold holds whether the subject sits near or far. |
| ROI scale change | `MAX_SCALE_CHANGE_RATIO` | 1.5 | Catches the mesh snapping to a different scale without the centroid moving. |

- The geometry reference **deliberately survives a short dropout.** MediaPipe
  drops frames during exactly the fast motion that relocates the ROI, so
  clearing it on every miss would disable the shift guard precisely when it is
  needed. Only `flush()` clears it. The trade is accepted knowingly: a spurious
  flush costs a recalibration, a missed one reports a spliced, fabricated BPM.
- `on_flush` **must** reset every downstream smoother. `evaluate.ResultSmoother`
  is the single object to wire to it: it owns the per-algorithm
  `MultiAlgorithmSmoother` and the `HRVSmoother`, and resets both as a unit.
  Wiring them up separately is how one gets forgotten and keeps displaying
  pre-flush values — a bug this repo has hit more than once, so
  `test_result_smoother_reset_clears_bpm_and_hrv_together` guards it.
- Buffer is a `deque` sized `BUFFER_MAXLEN = 780` (13 s × `MAX_EXPECTED_FPS` 60).
  That FPS figure is a **hard ceiling**, not a hint: above it the deque evicts
  faster than 13 s accumulates and the tracker never leaves `CALIBRATING`.
- States: `CALIBRATING` → `MEASURING` once the buffer spans `BUFFER_DURATION_S`.

### 2. Shared Preprocessing (`src/config.py`, `src/filters.py`)

| Constant | Value | Meaning |
|---|---|---|
| `TARGET_FS` | 30.0 Hz | Fixed resample rate; FFT resolution no longer depends on camera jitter. |
| `WINDOW_DURATION_S` | 10.0 s | Analysis window (`TARGET_WINDOW_SAMPLES` = 300). |
| `NORM_WINDOW_S` | 3.0 s | Trailing-mean length (`NORM_WINDOW_SAMPLES` = 90). |
| `BUFFER_DURATION_S` | 13.0 s | Window + lead-in (`REQUIRED_SAMPLES` = 389). |

- **Temporal mean normalization** (`filters.temporal_mean_normalize`): each
  channel divided by its own 3 s trailing rolling mean. Removes camera
  auto-exposure steps and slow room-light drift before any projection runs.
- 3 s is deliberate. The trailing divide is a **moving-average high-pass**; a
  shorter window (1 s) eats the bottom of the 0.7 Hz band and biases low heart
  rates upward. It must also stay *shorter* than the analysis window, or it
  degenerates into the whole-window normalization POS/CHROM already do
  internally and becomes a no-op.
- **Time to first reading is 13 s, not 10 s.** Every sample in the emitted
  300-sample window needs a full 90-sample trailing mean behind it, so the
  lead-in is required before any BPM is reported. The extra 3 s displays as
  `CALIBRATING`.
- **Butterworth bandpass** (`filters.bandpass_filter`): `FILTER_ORDER = 2`,
  `LOW_CUTOFF_HZ = 0.7` to `HIGH_CUTOFF_HZ = 2.5` (42–150 BPM), zero-phase via
  `filtfilt`. Applied once per algorithm, at the point that algorithm's own
  paper specifies.

### 3. Three-Way Comparative Pipeline (`src/rppg_pos.py`)

All three consume **one shared, identically preprocessed `(300, 3)` window**, so
the only difference between them is the projection. The registry `ALGORITHMS` /
`ALGORITHM_KEYS` drives the fan-out; add an algorithm there and every consumer
picks it up.

| Key | Algorithm | Projection | `self_filtered` |
|---|---|---|---|
| `pos` | **POS** — Wang et al., 2017 | `S1 = G-B`, `S2 = -2R+G+B`, `α = σ(S1)/σ(S2)`, `H = S1 + α·S2` | `False` |
| `chrom` | **CHROM** — de Haan & Jeanne, 2013 | `X = 3R-2G`, `Y = 1.5R+G-1.5B`, bandpass **first**, then `α = σ(Xf)/σ(Yf)`, `S = Xf - α·Yf` | `True` |
| `green` | **Raw Green** (baseline) | The normalized green channel alone — the control the projections must beat. | `False` |

- **CHROM is already filtered on return.** Computing α from the band-limited X
  and Y is its defining step. Filtering it again makes the response effectively
  8th-order and breaks apples-to-apples comparison. The `self_filtered` flag
  enforces this; never bypass it.
- POS deviation from the paper: published POS slides a ~1.6 s window,
  normalizing and projecting within each, then overlap-adds. Here the pipeline
  supplies one normalization for every algorithm so the comparison stays
  controlled, and the projection runs once over the full window.

### 4. BPM Estimation (`src/bpm_calc.py`)

- Hanning window → `rfft` → dominant peak within 0.7–2.5 Hz → Hz × 60.
- **Sub-bin peak interpolation.** A 300-sample window at 30 Hz has bins 0.1 Hz
  apart, i.e. **6 BPM wide**, so reporting the bin centre makes a steady pulse
  step visibly between 72, 78, 84… `_refine_peak_frequency` fits a parabola
  through the peak and its neighbours in log power. Measured worst-case error
  across a bin drops from 3.0 BPM to **0.09 BPM**.
- **SNR confidence gate**, `SNR_THRESHOLD = 15.0`: the peak's power must exceed
  15× the mean power of the rest of the in-band spectrum (excluding
  `PEAK_EXCLUDE_BINS = 1` on each side). `argmax` always finds *some* peak even
  in pure noise, so this margin is what separates a real pulse from a dead ROI.
  Below threshold the algorithm reports `STATUS_NO_PULSE` and `bpm = None`.
- `BPMSmoother`: moving average over the last `DEFAULT_BUFFER_SIZE = 8` readings.
- `MultiAlgorithmSmoother`: one `BPMSmoother` per algorithm, reset as a unit on
  tracker flush.

### 5. HRV / Stress Index (`src/hrv.py`)

Derived from the **filtered POS waveform** (`HRV_SOURCE_KEY = "pos"`), and only
once POS has cleared its own SNR gate — peak detection on noise would invent
beats and produce a confident-looking tier from nothing.

1. **Cubic spline upsample** to `UPSAMPLE_FS = 250 Hz` for sub-frame peak
   resolution (vs. 33 ms frame quantization).
2. **Peak detection** (`scipy.signal.find_peaks`), prominence ≥
   `MIN_PROMINENCE_STD = 0.3` × signal σ, spacing ≥
   `MIN_IBI_FRACTION_OF_EXPECTED = 0.7` × the expected interval from the FFT's
   own rate estimate.
3. **IBI screening**: reject intervals outside `MIN_IBI_S`–`MAX_IBI_S`
   (0.40–1.43 s = the same 42–150 BPM band), then drop any deviating more than
   `MAX_MEDIAN_DEVIATION = 20%` from the window median.
4. **RMSSD** = √(mean(successive ΔIBI²)), in ms. Requires ≥ `MIN_USABLE_IBIS = 4`
   surviving intervals, else `rmssd_ms = None` with `STATUS_TOO_FEW_BEATS`.
5. **Tier** — RMSSD is *inversely* related to stress:

   | RMSSD | Tier |
   |---|---|
   | < `HIGH_STRESS_BELOW_MS` (20 ms) | `High` |
   | 20–50 ms | `Moderate` |
   | > `LOW_STRESS_ABOVE_MS` (50 ms) | `Low` |

#### Known limitations — carry these into any published claim

- **Peak spacing must use the FFT rate estimate.** The 0.7–2.5 Hz passband keeps
  a pulse's 2nd harmonic for any rate below 75 BPM, putting a secondary peak
  inside every beat. Spacing off the absolute 0.4 s band edge alone detected
  **19 beats where there were 10**, halving every IBI.
- **The upsample buys resolution, not accuracy.** The input is already
  band-limited, so its systolic peak was smeared before interpolation.
- **Dropped beats are invisible.** The narrow passband acts as a resonator: a
  beat at *zero* amplitude is still reconstructed by filter ringing and reported
  as a regular interval. Under motion or low SNR this yields plausible IBIs
  rather than flagging that detection failed. `screen_ibis` is consequently a
  defensive net this filter rarely exercises; it becomes load-bearing only if the
  passband or peak spacing changes.
- **Tier thresholds are ECG-derived population figures, not rPPG-validated.**
  Calibrate against ground truth before publishing.
- **First validated live run (second machine, 640x480, ~30 FPS):** a 30.8 s
  clip, 897 frames, 100% face detection, best window **13.9 dB against the
  11.76 dB gate** -- cleared. Live readings 78-82 BPM with POS and CHROM
  agreeing to 3.8 BPM. This is the first end-to-end confirmation that the
  confidence gate is correctly calibrated rather than merely strict: the
  original development machine's clip peaked at 10.7 dB and was rejected.
- **Measured tier stability** at fixed physiology, 60 windows each:

  | True jitter | Recovered RMSSD | Tier distribution |
  |---|---|---|
  | 10 ms | 13.0 ± 3.6 ms | High 95%, Moderate 5% |
  | 25 ms | 31.7 ± 7.9 ms | Moderate 93%, High 5%, Low 2% |
  | 60 ms | 80.1 ± 21.4 ms | Low 93%, Moderate 7% |

  Modal tier is correct at every level, but ~1 window in 15 misclassifies.
  RMSSD *magnitude* is the stronger metric: recovered tracks true within 3% up
  to ~58 ms, saturating only above ~140 ms — well clear of the tier boundaries.
  **Prefer RMSSD as the reported number; treat the tier as display convenience.**
#### Smoothing (`HRVSmoother`)

RMSSD *is* smoothed, by a moving average of `HRV_SMOOTHING_BUFFER = 8` readings.
The tier is then re-derived from the **smoothed** RMSSD, so the label always
agrees with the number displayed beside it — rather than voting across raw tiers.

Averaging alone is not enough. Readings come from heavily overlapping windows
(a 10 s window advancing 0.2 s), so consecutive estimates are strongly
correlated and averaging buys far less than √N. The visible flicker is
**boundary chatter**: a value sitting near a threshold drifting back and forth
across it. `classify_stress_with_hysteresis` fixes that directly — once in a
tier, the value must clear that tier's bounds by `TIER_HYSTERESIS_MS = 3.0`
before the label changes. Measured on a subject sitting exactly on the
Moderate/Low boundary: **30 tier changes over 60 readings → 1.**

The window's own unsmoothed estimate is preserved as `rmssd_raw_ms` for
plotting and for the paper.

### 6. Performance Budget

- Verified 30 FPS on a Ryzen laptop → a **33 ms per-frame budget**.
- Running all three algorithms plus HRV costs **~23 ms per call**, which would
  consume ~69% of that budget if done every frame.
- The live loop therefore recomputes on an interval, not per frame:
  `main.ANALYSIS_INTERVAL_S = 0.2` (**5 Hz**). The 10 s window barely moves
  between frames, so this is visually indistinguishable and returns the capture
  loop its full frame rate.
- The offline evaluator has no wall-clock constraint and emits one reading per
  `STEP_S = 1.0` s of video.
- **Display resolution is decoupled from analysis resolution.** Capture is
  requested at 640×480 so the dashboard preview is worth looking at, while
  `FaceROITracker(analysis_width=...)` controls what inference and the spatial
  average actually run on. The live paths pass `LIVE_ANALYSIS_WIDTH = 320`,
  matching the configuration the 30 FPS and SNR figures were measured under;
  offline evaluation keeps `PROCESS_WIDTH = 480`. `process_frame` returns the
  frame at its original size with the ROI overlay drawn at that scale.
  Measured on the reference machine: 27.9 FPS at 640×480, versus 24 FPS at
  320×240 before the split.
- Two optimizations make that possible and should not be undone: the ROI
  overlay alpha-blends only the face's bounding box rather than the whole
  frame, and `spatial_average` uses `cv2.mean` instead of boolean indexing.
- The web preview is throttled independently of capture (`web_app.STREAM_FPS`
  = 15) and JPEG encoding is **skipped entirely when no client is streaming**
  `/video_feed`, so hiding the preview costs the pipeline nothing.

### 7. Unified Result Dictionary

`evaluate.compute_algorithm_signals(raw_rgb, raw_timestamps)` is the single
entry point the UI consumes.

**Returns `None`** when there is not yet a full 13 s window — the pipeline is
still calibrating. Otherwise:

```python
{
    "pos":   {"label": str, "signal": np.ndarray(300,), "bpm": float | None, "status": str},
    "chrom": {"label": str, "signal": np.ndarray(300,), "bpm": float | None, "status": str},
    "green": {"label": str, "signal": np.ndarray(300,), "bpm": float | None, "status": str},
    "hrv":   {"rmssd_ms": float | None, "tier": str | None, "n_beats": int, "status": str},
}
```

After `ResultSmoother.update`, the `hrv` entry additionally carries
`rmssd_raw_ms` (this window's own estimate, before smoothing) and its
`rmssd_ms` / `tier` are the smoothed, hysteretic values.

- **Two failure modes stay distinct.** A bare `None` return means "still
  calibrating"; a populated dict with a per-key `bpm: None` means "this
  algorithm found no pulse". The UI must not collapse them.
- `signal` is the **filtered** waveform, ready to plot directly.
- `hrv.n_beats` is exposed so the UI can explain *why* a tier is missing rather
  than blanking it.
- `HRV_KEY = "hrv"` is a **non-algorithm** entry in the same dict. Iterate
  `ALGORITHM_KEYS`, never the dict itself, to visit only the algorithms.

### 8. Web Layer (`src/web_app.py`, `src/telemetry.py`, `src/benchmark.py`)

Launch: `uvicorn src.web_app:app --port 8000`, from the repository root.

`web_app.py` owns transport only; all signal processing is imported. Two rules
keep it responsive and must not be reverted:

- **No hardware at import.** `CaptureWorker.__init__` allocates plain attributes
  only; the camera opens and the thread spawns on an explicit `/api/start`.
- **One bad frame must not end the session.** The pipeline can legitimately
  raise on a degenerate window (an all-black frame gives a zero channel mean;
  duplicate timestamps upset the resampler). Before this was handled, any such
  exception killed the worker thread, released the camera and looked to the
  user like the camera switching itself off at random. Failures now skip the
  frame, and only `MAX_CONSECUTIVE_FAILURES = 30` in a row aborts — with the
  reason surfaced through `/api/status` so the UI can show it.
- **`CAMERA_WARMUP_S = 1.5`** of capture is discarded before calibration
  counts, because webcams ramp auto-exposure over the first second or two and
  that ramp otherwise poisons the first analysis window.
- **No heavy imports at import.** scipy, mediapipe and cv2 cost ~40 s to load
  cold on the reference machine, which would delay uvicorn binding its port.
  `_load_backend()` defers them to first use and the lifespan handler warms them
  on a background thread. Measured: import 42.5 s → 14.9 s, port bound in 1 s.

**One camera, one owner.** `cv2.VideoCapture(0)` cannot be opened twice and both
`/video_feed` and `/ws/telemetry` need the same frames, so a single
`CaptureWorker` thread owns the camera, the tracker and the `ResultSmoother`,
and publishes the latest JPEG plus the latest payload behind a lock. The
endpoints only read that shared state, and the 5 Hz analysis gate lives in the
worker — never in the WebSocket handler, where N clients would multiply the cost.

| Route | Purpose |
|---|---|
| `GET /` | Dashboard (Jinja2 template) |
| `GET /video_feed` | MJPEG stream; `409` when capture is not running |
| `WS /ws/telemetry` | Telemetry payload at 5 Hz |
| `POST /api/start`, `/api/stop` | Idempotent capture control |
| `GET /api/status` | Running flag, backend-warm flag, last payload |
| `POST /api/analyze_dataset` | Upload video + optional ground truth |
| `POST /api/analyze_sample` | Same, against the bundled clip |
| `GET /api/sample_clip` | Whether the bundled clip exists |

- `telemetry.build_payload` is pure, so the populated payload is unit-tested
  without a camera. It maps the tracker's `CALIBRATING`/`MEASURING` plus the
  `face_found` flag onto the UI's `CALIBRATING`/`TRACKING`/`FACE_LOST`; the
  tracker's own state names are unchanged.
- `benchmark.error_metrics` returns MAE, RMSE, MSE, MAPE, Pearson r, bias,
  Bland-Altman limits of agreement, and the share within
  `ACCURACY_TOLERANCE_BPM = 5`, plus `n`. Metrics are computed only where both
  series have a value; points outside the ground truth's own span are NaN rather
  than extrapolated, so they cannot flatter the score.
- `benchmark.parse_ground_truth` picks the heart-rate series **by plausibility**
  (median between `MIN_PLAUSIBLE_BPM` 25 and `MAX_PLAUSIBLE_BPM` 240), not by
  position. UBFC-rPPG puts the raw PPG waveform in its first row and the BPM in
  its second; choosing by position read the waveform as heart rate and scored
  every metric against values near zero. A file with no plausible BPM series is
  rejected with an explanation rather than parsed into nonsense.

## Execution Rules

- Think step-by-step. Verify matrix dimensions during every projection.
- Write tests for each module before moving to the next.
- Run `./venv/bin/python -m pytest tests -q` before declaring work complete.
- **Current status: 135 tests passing.**

  | File | Tests |
  |---|---|
  | `tests/test_benchmark.py` | 23 |
  | `tests/test_bpm_calc.py` | 13 |
  | `tests/test_evaluate.py` | 10 |
  | `tests/test_filters.py` | 13 |
  | `tests/test_hrv.py` | 32 |
  | `tests/test_pos.py` | 17 |
  | `tests/test_telemetry.py` | 10 |
  | `tests/test_tracker.py` | 17 |

- Prefer mutation-testing a new safeguard (break it deliberately, confirm a test
  fails) over trusting a green suite. Several tests in this repo exist because
  that check caught a guard that nothing actually exercised.
