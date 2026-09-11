# PulseVision — Contactless Heart Rate & Stress from a Webcam

Measures heart rate and an HRV-derived stress index from facial video, with no
contact and no wearable. Three published rPPG algorithms run side by side on the
same signal so a reading can be cross-checked rather than taken on trust.

Built for research and evaluation. **Not a medical device.**

---

## Setup on a new machine

Requires **Python 3.11 or 3.12**. MediaPipe does not ship wheels for 3.13+, and
building it from source will hang.

### 1. Create and activate a virtual environment

This repo is private. You need to accept the collaborator invite (check your
email or GitHub notifications) before `git clone` below will work.

```bash
git clone https://github.com/Anchita-p/Contactless-BPM-Detector.git rppg_detector
cd rppg_detector

python3.11 -m venv venv          # be explicit; a bare `python3` may be 3.13+
source venv/bin/activate         # Windows: venv\Scripts\activate
python --version                 # confirm 3.11.x or 3.12.x before continuing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt --only-binary=:all:
```

The `--only-binary=:all:` flag is not optional in practice. Without it pip can
fall back to building `mediapipe` or `scipy` from source, which appears to hang
for a very long time with no output.

### 3. Run the test suite

```bash
pytest tests -q
```

Expect **135 passed**. Three tests print measured figures rather than only
asserting — run `pytest tests -q -s` to see them.

### 4. Launch the web app

```bash
uvicorn src.web_app:app --port 8000
```

Then open **http://localhost:8000**.

Run it from the repository root (that is how `src.web_app` resolves); everything
the app itself opens — templates, static files, the sample clip — is located
relative to the source file, so no other path assumptions apply.

On macOS the first **Start** will trigger a camera-permission prompt for your
terminal application. Grant it and press Start again.

### 5. Pushing changes

GitHub no longer accepts your account password for `git push` over HTTPS. Use
a personal access token instead:

1. GitHub → your profile picture → **Settings** → **Developer settings** →
   **Personal access tokens** → **Tokens (classic)** → **Generate new token**.
   Scope: `repo`. Copy it immediately — it's shown once.
2. `git push`, and when prompted, enter your GitHub username, then paste the
   token as the password. macOS Keychain remembers it after that.

If a push fails with `RPC failed; HTTP 400` / `unexpected disconnect while
reading sideband packet`, it's a buffer-size issue, not your credentials. Fix
it once and retry:

```bash
git config --global http.postBuffer 157286400
git config --global http.version HTTP/1.1
git push
```

---

## Using it

### Live Monitor

Press **Start**. You will see yourself with cyan outlines and white dots tracing
the face outline and both cheeks. The region actually analysed is the convex
hull of those points, so it covers most of the face — roughly 83% of the face
bounding box, nose and chin included. If the outline drifts onto hair, glasses
or background, the reading will suffer.

A progress bar runs for about 13 seconds. Nothing is displayed until it
completes: the pipeline needs a full 10-second analysis window plus a 3-second
normalization lead-in before any number is honest.

After that you get heart rate, stress tier with RMSSD, signal quality, all three
algorithms side by side, and a live pulse waveform, updating five times a second.

**On a slower machine**, tick *"Hide the video while measuring"*. That closes the
video stream, which stops the server encoding JPEGs entirely, while the reading
continues. Below 14 FPS the page also disables its own animations automatically.

### Accuracy Check

The live view can tell you a number looks steady, not that it is correct. This
tab answers that by replaying a recorded video against a pulse reading you
already trust.

Upload a video of a face, optionally with a ground-truth heart-rate file, and
press **Run the check**. Or press **Try the sample clip** to run the bundled
video with one click.

Ground truth is a `.csv` or `.txt`; the parser is deliberately tolerant and tells
you how it read your file:

```
time,bpm
0.0,72
1.0,73
2.0,71
```

The heart-rate series is identified by its values, not its position, so column
order does not matter. Two bare numeric columns work, a single column of BPM
works (spread evenly across the video), and UBFC-rPPG's three whitespace-
separated rows — PPG waveform, heart rate, timestamps — work as well. The app
tells you which column or row it used; check that line before trusting the
numbers.

One clip and one ground-truth file at a time. A public dataset is a folder of
many subjects, so you run them one subject at a time.

You get MAE, RMSE, MSE, MAPE, Pearson r, bias, Bland-Altman 95% limits of
agreement, and the share of readings within 5 BPM — per algorithm.

---

## Command-line entry points

```bash
python src/main.py                     # OpenCV window instead of the browser
python src/main.py --video clip.mp4    # offline evaluation, printed as a table
python src/evaluate.py clip.mp4        # same, directly
```

---

## Layout

```
src/
  web_app.py    FastAPI service: MJPEG stream, 5 Hz telemetry socket, upload API
  tracker.py    MediaPipe face mesh, ROI extraction, rolling buffer, auto-flush
  filters.py    Temporal-mean normalization, Butterworth bandpass, resampling
  rppg_pos.py   POS, CHROM and Raw Green extraction, and the algorithm registry
  bpm_calc.py   FFT peak search, SNR confidence gate, BPM smoothing
  hrv.py        Upsampling, beat detection, RMSSD, stress tier with hysteresis
  evaluate.py   Shared pipeline core and the offline video evaluator
  benchmark.py  Ground-truth parsing and the error metrics
  telemetry.py  Telemetry payload construction
  config.py     Shared timing constants
templates/      Dashboard HTML
static/         Dashboard JavaScript
tests/          135 tests
```

`ARCHITECTURE.md` is the architecture specification and the source of truth for every
tunable constant. If you change one in the code, change it there too.

---

## Known limitations

Carry these into any published claim.

- **A reading needs cooperation.** Even lighting, face filling a good part of the
  frame, and stillness. Movement and talking are what break it.
- **The sampled region is the whole face hull, and that may not be optimal.**
  Non-pulsatile area — sclera, lips, the specular highlight on the nose —
  dilutes the average rather than strengthening it. On the bundled clip, a
  forehead-plus-cheeks region reached the same signal quality using a third of
  the pixels. Worth re-testing on a clip that clears the gate.
- **The confidence gate is strict by design.** `SNR_THRESHOLD = 15x` sits above
  the level pure noise reaches, so a weak signal reports no pulse rather than a
  confident fabrication. The bundled sample clip misses it by about 1 dB.
- **Stress tiers are uncalibrated for this method.** The 20 ms / 50 ms thresholds
  come from ECG population data. Modal tier is correct at every level tested, but
  roughly one window in fifteen misclassifies. **Prefer RMSSD as the reported
  number and treat the tier as display convenience.**
- **Dropped beats are invisible.** The narrow passband reconstructs a missing
  beat through filter ringing, so under motion or low signal the HRV module can
  report plausible intervals instead of flagging failure.
- **No algorithm ranking has been validated here.** POS is the default on the
  strength of the published comparisons, not on measurements from this build.
  The Accuracy Check tab exists to settle that, and needs a clip that clears the
  confidence gate with a simultaneously recorded ground truth.
