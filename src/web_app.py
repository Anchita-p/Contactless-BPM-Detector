"""FastAPI service for the PulseVision rPPG dashboard.

    uvicorn src.web_app:app --port 8000

All signal processing is imported from the existing backend; nothing is
reimplemented here. This module owns only transport concerns: one camera, an
MJPEG stream, a 5 Hz telemetry socket, and the dataset-evaluation endpoint.

Two startup rules keep the server responsive:

1. **No hardware at import.** `CaptureWorker.__init__` allocates nothing but
   plain attributes; the camera is opened and the capture thread spawned only
   by an explicit `/api/start`.

2. **No heavy imports at import.** The signal-processing stack is slow to load
   cold -- measured on this machine at roughly 24 s for scipy, 10 s for
   mediapipe and 7 s for cv2 -- which would delay uvicorn binding its port by
   the better part of a minute. `_load_backend()` defers all of it to first
   use, and the lifespan handler warms it on a background thread so the port
   is listening immediately and the stack is usually ready before anyone
   clicks Start.

Camera ownership is the central design point. `cv2.VideoCapture(0)` cannot be
opened twice, and both `/video_feed` and `/ws/telemetry` need the same frames,
so a single `CaptureWorker` thread owns the camera and the tracker, runs the
pipeline, and publishes the latest annotated JPEG and the latest telemetry
payload behind a lock. The endpoints only read that shared state.
"""
import asyncio
import contextlib
import os
import sys
import tempfile
import threading
import time
import types

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "templates"))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

# Matches main.ANALYSIS_INTERVAL_S: the pipeline costs ~23 ms per call, so it
# runs on its own 5 Hz gate inside the worker rather than per frame -- and
# never inside a WebSocket handler, where N clients would multiply the cost.
ANALYSIS_INTERVAL_S = 0.2
TELEMETRY_INTERVAL_S = 0.2
JPEG_QUALITY = 78
TARGET_FRAME_INTERVAL_S = 1.0 / 30.0
# The preview does not need the capture rate. Encoding and shipping 30 JPEGs a
# second costs the server an encode and the browser a decode plus a repaint
# each time, which on a modest machine steals the CPU the pipeline needs.
# 15 Hz looks identical for a talking head and halves both sides of that.
STREAM_FPS = 15
STREAM_INTERVAL_S = 1.0 / STREAM_FPS
FIRST_FRAME_TIMEOUT_S = 8.0
# A single bad frame must not end the session. The pipeline can legitimately
# raise on a degenerate window -- an all-black frame makes a channel mean zero,
# and duplicate timestamps upset the resampler -- so a failure skips that frame
# and only an unbroken run of them is treated as fatal.
MAX_CONSECUTIVE_FAILURES = 30
# Discard the first moments of capture: webcams ramp auto-exposure and white
# balance over the first second or two, and feeding that ramp into the rolling
# buffer poisons the very first analysis window.
CAMERA_WARMUP_S = 1.5
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
# A clip that ships with the repo, so the accuracy check can be tried without
# first finding a dataset video.
SAMPLE_CLIP = os.path.join(PROJECT_ROOT, "data", "test_clip.mp4")

_backend = None
_backend_lock = threading.Lock()


def _load_backend():
    """Import the signal-processing stack, once, on first use.

    Returns a namespace of the backend symbols this module needs. Safe to call
    concurrently and cheap after the first call.
    """
    global _backend
    with _backend_lock:
        if _backend is None:
            import cv2
            from tracker import FaceROITracker, LIVE_ANALYSIS_WIDTH
            from evaluate import HRV_SOURCE_KEY, ResultSmoother, compute_algorithm_signals, evaluate_video
            from benchmark import evaluate_readings
            from telemetry import build_payload

            _backend = types.SimpleNamespace(
                cv2=cv2,
                FaceROITracker=FaceROITracker,
                LIVE_ANALYSIS_WIDTH=LIVE_ANALYSIS_WIDTH,
                ResultSmoother=ResultSmoother,
                compute_algorithm_signals=compute_algorithm_signals,
                evaluate_video=evaluate_video,
                evaluate_readings=evaluate_readings,
                build_payload=build_payload,
                HRV_SOURCE_KEY=HRV_SOURCE_KEY,
            )
    return _backend


class CaptureWorker:
    """Owns the camera, the tracker and the smoother for the lifetime of a run.

    Constructing one touches no hardware -- see the module docstring.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._frame_jpeg = None
        self._payload = None
        self._error = None
        self._frame_ready = threading.Event()
        # Number of open /video_feed responses. With nobody watching there is
        # no reason to spend an encode per frame at all.
        self._viewers = 0

    def add_viewer(self, delta):
        with self._lock:
            self._viewers = max(self._viewers + delta, 0)

    @property
    def viewers(self):
        with self._lock:
            return self._viewers

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Idempotent: starting an already-running worker is a no-op."""
        if self.running:
            return True

        # Load the backend before spawning the thread, so the first-frame
        # timeout below measures the camera rather than a cold scipy import.
        backend = _load_backend()

        with self._lock:
            if self.running:
                return True
            self._stop.clear()
            self._error = None
            self._frame_ready.clear()
            self._thread = threading.Thread(target=self._run, args=(backend,), daemon=True)
            self._thread.start()

        # Surface a camera that refuses to open as an error the UI can show,
        # rather than a stream that silently never yields a frame.
        self._frame_ready.wait(timeout=FIRST_FRAME_TIMEOUT_S)
        return self._error is None

    def stop(self):
        """Idempotent. Always releases the camera and the MediaPipe graph --
        a worker left holding the device makes the next start fail obscurely."""
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop.set()
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            self._frame_jpeg = None
            self._payload = None

    def snapshot(self):
        with self._lock:
            return self._frame_jpeg, self._payload, self._error

    def _publish(self, jpeg, payload):
        with self._lock:
            if jpeg is not None:
                self._frame_jpeg = jpeg
            if payload is not None:
                self._payload = payload
        self._frame_ready.set()

    def _run(self, backend):
        cv2 = backend.cv2
        smoothers = backend.ResultSmoother()
        # Built as a pair: on_flush must reset the very smoother whose output
        # the telemetry payload is read from.
        tracker = backend.FaceROITracker(camera_index=self.camera_index,
                                         on_flush=smoothers.reset,
                                         analysis_width=backend.LIVE_ANALYSIS_WIDTH)
        cap = tracker.open_capture()

        if not cap.isOpened():
            with self._lock:
                self._error = (f"Could not open camera index {self.camera_index}. "
                               "Grant camera permission to your terminal and retry.")
            cap.release()
            tracker.face_mesh.close()
            self._frame_ready.set()
            return

        reading = smoothers.update(None)
        results = None
        last_analysis = 0.0
        last_encode = 0.0

        started_at = time.monotonic()
        warmed_up = False
        failures = 0

        try:
            while not self._stop.is_set():
                frame_started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                try:
                    frame = cv2.flip(frame, 1)  # natural selfie view
                    frame, _, face_found = tracker.process_frame(frame)

                    # Drop everything captured while the camera was still
                    # settling, so calibration starts from usable pixels.
                    if not warmed_up and frame_started - started_at >= CAMERA_WARMUP_S:
                        warmed_up = True
                        tracker.flush()
                    # No cv2.putText HUD here: text baked into the JPEG is
                    # aliased and unstyleable. The dashboard overlays state and
                    # FPS as real DOM elements. draw_hud stays for main.py.
                    tracker._update_fps()

                    now = time.monotonic()
                    if now - last_analysis >= ANALYSIS_INTERVAL_S:
                        last_analysis = now
                        results = backend.compute_algorithm_signals(tracker.get_signal(),
                                                                    tracker.get_timestamps())
                        reading = smoothers.update(results)

                    jpeg = None
                    if self.viewers and now - last_encode >= STREAM_INTERVAL_S:
                        last_encode = now
                        encoded, buffer = cv2.imencode(
                            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                        if encoded:
                            jpeg = buffer.tobytes()

                    payload = backend.build_payload(
                        reading=reading,
                        results=results,
                        face_found=face_found,
                        fps=tracker.fps,
                        buffered_span_s=tracker._buffered_span_s(),
                    )
                    self._publish(jpeg, payload)
                    failures = 0

                except Exception as exc:   # noqa: BLE001 - one bad frame is not fatal
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        with self._lock:
                            self._error = (
                                f"Capture stopped after {failures} consecutive errors "
                                f"({type(exc).__name__}: {exc}). Try restarting it.")
                        self._frame_ready.set()
                        break
                    continue

                # Pace to the capture budget so a fast camera cannot spin the
                # CPU faster than the pipeline was measured for.
                spare = TARGET_FRAME_INTERVAL_S - (time.monotonic() - frame_started)
                if spare > 0:
                    time.sleep(spare)
        finally:
            cap.release()
            tracker.face_mesh.close()


worker = CaptureWorker()


@contextlib.asynccontextmanager
async def lifespan(app):
    # Warm the heavy imports off the critical path: the port is already
    # listening, and the stack is usually ready before anyone clicks Start.
    threading.Thread(target=_load_backend, daemon=True).start()
    yield
    worker.stop()


app = FastAPI(title="PulseVision rPPG Telemetry", lifespan=lifespan)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(request=request, name="index.html")


@app.post("/api/start")
async def start_capture():
    started = await asyncio.to_thread(worker.start)
    _, _, error = worker.snapshot()
    if not started:
        raise HTTPException(status_code=503, detail=error or "Camera failed to start.")
    return {"running": True}


@app.post("/api/stop")
async def stop_capture():
    await asyncio.to_thread(worker.stop)
    return {"running": False}


@app.get("/api/status")
def status():
    _, payload, error = worker.snapshot()
    return {
        "running": worker.running,
        "backend_ready": _backend is not None,
        "error": error,
        "telemetry": payload,
    }


def _mjpeg_frames():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    worker.add_viewer(1)
    last_sent = None
    try:
        while worker.running:
            jpeg, _, _ = worker.snapshot()
            # Only ship genuinely new frames; re-sending the same bytes makes
            # the browser decode and repaint for nothing.
            if jpeg is not None and jpeg is not last_sent:
                last_sent = jpeg
                yield boundary + jpeg + b"\r\n"
            time.sleep(STREAM_INTERVAL_S / 2)
    finally:
        worker.add_viewer(-1)


@app.get("/video_feed")
def video_feed():
    """MJPEG stream of the annotated camera feed.

    The generator exits as soon as the worker stops, so a closed browser tab
    cannot leave it spinning forever.
    """
    if not worker.running:
        raise HTTPException(status_code=409, detail="Capture is not running.")
    return StreamingResponse(_mjpeg_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket):
    """Pushes the worker's latest payload at 5 Hz. Never runs the pipeline
    itself -- it only serializes whatever the worker most recently produced."""
    await websocket.accept()
    try:
        while True:
            _, payload, error = worker.snapshot()
            await websocket.send_json({
                "running": worker.running,
                "error": error,
                "telemetry": payload,
            })
            await asyncio.sleep(TELEMETRY_INTERVAL_S)
    except (WebSocketDisconnect, RuntimeError):
        pass


def _save_upload(upload, allowed_suffixes=None):
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if allowed_suffixes and suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or upload.filename}'. "
                   f"Expected one of {sorted(allowed_suffixes)}.")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4")
    handle.write(upload.file.read())
    handle.close()
    return handle.name


def _run_dataset_evaluation(video_path, ground_truth_text):
    backend = _load_backend()
    readings = backend.evaluate_video(video_path)
    if not readings:
        raise HTTPException(status_code=422,
                            detail="No frames could be read from that video.")
    snr_series = [reading.get("snr_db") for _, reading in readings]
    duration_s = readings[-1][0]
    return backend.evaluate_readings(readings, snr_series,
                                     ground_truth=ground_truth_text,
                                     duration_s=duration_s)


@app.get("/api/sample_clip")
def sample_clip_available():
    """Whether the bundled demo clip exists, so the UI can offer it or not."""
    return {"available": os.path.exists(SAMPLE_CLIP),
            "name": os.path.basename(SAMPLE_CLIP)}


@app.post("/api/analyze_sample")
async def analyze_sample():
    """Run the bundled clip, so the accuracy check is usable with one click."""
    if not os.path.exists(SAMPLE_CLIP):
        raise HTTPException(status_code=404, detail="No sample clip is bundled with this install.")
    return await asyncio.to_thread(_run_dataset_evaluation, SAMPLE_CLIP, None)


@app.post("/api/analyze_dataset")
async def analyze_dataset(video: UploadFile = File(...),
                          ground_truth: UploadFile = File(None)):
    """Run an uploaded clip through the offline pipeline and score it.

    Runs in a worker thread: the evaluation is CPU-bound and minutes long for a
    full dataset clip, and must not block the event loop serving the telemetry
    socket.
    """
    video_path = _save_upload(video, ALLOWED_VIDEO_SUFFIXES)
    truth_text = None
    if ground_truth is not None and ground_truth.filename:
        raw = await ground_truth.read()
        truth_text = raw.decode("utf-8", errors="replace")

    try:
        return await asyncio.to_thread(_run_dataset_evaluation, video_path, truth_text)
    except HTTPException:
        raise
    except ValueError as exc:  # malformed ground truth is a user error, not a crash
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        os.unlink(video_path)
