"""High-FPS MediaPipe Face Landmarker overlay for live camera stream.

Runs a dedicated background thread that processes raw camera frames as fast
as the hardware allows, capped at 30 FPS to bound CPU/GIL usage. Draws the full
478-point MediaPipe FaceLandmarker tessellation, contours, and irises in green
onto each frame and exposes the result via ``get_latest_mesh_frame()``.

Design constraints:
- RTX 3050 (4 GB VRAM) — CPU-only MediaPipe to avoid VRAM pressure from GPU
  inference pipeline (Whisper, LLM, TTS already saturate VRAM budget).
- Zero blocking of asyncio event loop — dedicated daemon thread only.
- Thread-safe double-buffer with lock: writer thread sets latest frame,
  MJPEG stream reads a copy under the same lock.
- Graceful no-op when MediaPipe is unavailable (module-level try/except).
- Uses MediaPipe Tasks API (mediapipe >= 0.10 / 1.x): FaceLandmarker with
  VIDEO running mode for maximum per-frame throughput on a single thread.
  The legacy ``mp.solutions.face_mesh`` API was removed in mediapipe >= 0.10.

Model file:
  models/vision/face_landmarker.task  (float16, ~3 MB)
  Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/
            face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.perception.camera import CameraCapture

logger = get_logger("perception.face_mesh")

# ---------------------------------------------------------------------------
# Optional MediaPipe import — fails gracefully if package not installed or
# if the .task model file is missing.
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp  # type: ignore[import-untyped]
    from mediapipe.tasks.python import BaseOptions as _MpBaseOptions  # type: ignore[import-untyped]
    from mediapipe.tasks.python import vision as _mp_vision  # type: ignore[import-untyped]

    _FaceLandmarker = _mp_vision.FaceLandmarker
    _FaceLandmarkerOptions = _mp_vision.FaceLandmarkerOptions
    _VisionRunningMode = _mp_vision.RunningMode
    _FaceLandmarksConnections = _mp_vision.FaceLandmarksConnections

    _MEDIAPIPE_AVAILABLE = True
    logger.info("mediapipe_tasks_available", version=mp.__version__)
except (ModuleNotFoundError, ImportError, AttributeError) as _mp_err:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "mediapipe_not_available",
        error=str(_mp_err),
        hint="Run: .venv/bin/pip install mediapipe",
    )


# ---------------------------------------------------------------------------
# Drawing constants (Google-green palette)
# ---------------------------------------------------------------------------
_TESSELATION_COLOR = (0, 255, 128)   # BGR bright green
_CONTOUR_COLOR = (0, 230, 60)        # BGR slightly deeper green
_IRIS_COLOR = (0, 200, 255)          # BGR cyan-green for irises
_TESSELATION_THICKNESS = 1
_CONTOUR_THICKNESS = 1
_IRIS_THICKNESS = 1

# Hard ceiling for the overlay worker — even when max_fps is uncapped the
# worker never spins faster than 30 FPS (MediaPipe VIDEO mode plus drawing
# at higher rates only burns CPU/GIL without visual benefit).
_OVERLAY_MAX_FPS = 30.0


# ---------------------------------------------------------------------------
# Connection sets — build once as tuples for fast iteration
# ---------------------------------------------------------------------------
def _build_connections() -> dict[str, tuple[tuple[int, int], ...]]:
    """Build connection index tuples from FaceLandmarksConnections.

    Returns:
        dict[str, tuple[tuple[int, int], ...]]: Named connection sets as
            flat (start, end) index tuples, suitable for cv2.line loops.
    """
    if not _MEDIAPIPE_AVAILABLE:
        return {}
    c = _FaceLandmarksConnections
    return {
        "tesselation": tuple(
            (conn.start, conn.end) for conn in c.FACE_LANDMARKS_TESSELATION
        ),
        "contours": tuple(
            (conn.start, conn.end) for conn in (
                list(c.FACE_LANDMARKS_CONTOURS)
                + list(c.FACE_LANDMARKS_FACE_OVAL)
            )
        ),
        "irises": tuple(
            (conn.start, conn.end) for conn in (
                list(c.FACE_LANDMARKS_LEFT_IRIS)
                + list(c.FACE_LANDMARKS_RIGHT_IRIS)
            )
        ),
    }


_CONNECTIONS: dict[str, tuple[tuple[int, int], ...]] = {}


class FaceMeshOverlay:
    """Dedicated high-FPS MediaPipe Face Landmarker compositing thread.

    Reads raw frames directly from CameraCapture at max hardware speed,
    applies the 478-point face landmark tessellation and contour overlays in
    green, and exposes the composite frame for the MJPEG stream.

    Uses the MediaPipe Tasks API (FaceLandmarker with VIDEO running mode)
    which is the only supported API in mediapipe >= 0.10 / 1.x.

    Attributes:
        camera: CameraCapture instance to read raw frames from.
        enabled: Whether the face mesh thread is active.
        model_path: Path to the face_landmarker.task file.
    """

    _DEFAULT_MODEL_PATH = "models/vision/face_landmarker.task"

    def __init__(
        self,
        camera: "CameraCapture",
        max_fps: float = 0.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 4,
        draw_tesselation: bool = True,
        draw_contours: bool = True,
        draw_irises: bool = True,
        model_path: str = _DEFAULT_MODEL_PATH,
    ) -> None:
        """Initialize the FaceMeshOverlay.

        Args:
            camera: CameraCapture instance to pull raw frames from.
            max_fps: Target FPS cap (0 = hardware default; the worker is
                always capped at 30 FPS via _OVERLAY_MAX_FPS). Note: the loop
                still yields via time.sleep(0) between frames so Python GIL is
                released.
            min_detection_confidence: Face detection confidence threshold [0, 1].
            min_tracking_confidence: Landmark tracking confidence threshold [0, 1].
            max_num_faces: Maximum number of simultaneous faces to track.
            draw_tesselation: If True, draw full tesselation edges.
            draw_contours: If True, draw facial feature contours (lips, brows,
                eye outlines, oval).
            draw_irises: If True, draw iris landmark rings.
            model_path: Path to the face_landmarker.task model bundle.
        """
        self.camera = camera
        self.max_fps = max_fps
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_faces = max_num_faces
        self.draw_tesselation = draw_tesselation
        self.draw_contours = draw_contours
        self.draw_irises = draw_irises
        self.model_path = model_path

        # Model file presence check (required for Tasks API)
        self._model_ok = Path(model_path).exists()
        self.enabled = _MEDIAPIPE_AVAILABLE and self._model_ok

        if _MEDIAPIPE_AVAILABLE and not self._model_ok:
            logger.warning(
                "face_mesh_model_missing",
                path=model_path,
                hint=(
                    "Download: wget -q https://storage.googleapis.com/mediapipe-models"
                    "/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    f" -O {model_path}"
                ),
            )

        self._lock = threading.Lock()
        self._latest_mesh_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

        # Precomputed static edge-index arrays (built once in start()) mapping
        # landmark-index pairs; used for single-call cv2.polylines rendering.
        self._edge_idx: dict[str, np.ndarray] = {}

        # Performance counters
        self._frame_count: int = 0
        self._last_fps_time: float = time.time()
        self._measured_fps: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Return True if MediaPipe is installed, model exists, and worker thread is alive.

        Also verifies the background thread actually stayed alive: a worker that
        crashed on a fatal error flips ``_running`` to False, so this property
        stops lying about availability.
        """
        if not (_MEDIAPIPE_AVAILABLE and self._model_ok and self._running):
            return False
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def last_error(self) -> Optional[str]:
        """Return the last error message seen by the worker thread (or None)."""
        return self._last_error

    def status(self) -> dict[str, object]:
        """Return a structured status snapshot of the overlay subsystem."""
        thread = self._thread
        return {
            "mediapipe_available": _MEDIAPIPE_AVAILABLE,
            "model_ok": self._model_ok,
            "enabled": self.enabled,
            "running": self._running,
            "thread_alive": bool(thread is not None and thread.is_alive()),
            "measured_fps": self._measured_fps,
            "last_error": self._last_error,
        }

    @property
    def measured_fps(self) -> float:
        """Return the measured processing FPS of the face mesh thread."""
        return self._measured_fps

    def start(self) -> None:
        """Start the face mesh background thread.

        No-op if MediaPipe is unavailable, model file is missing, or thread
        is already running.
        """
        if not _MEDIAPIPE_AVAILABLE:
            logger.warning("face_mesh_start_skipped", reason="mediapipe not installed")
            return
        if not self._model_ok:
            # Recheck model path in case it was downloaded after init
            self._model_ok = Path(self.model_path).exists()
            if not self._model_ok:
                logger.warning(
                    "face_mesh_start_skipped",
                    reason="model file missing",
                    path=self.model_path,
                )
                return
            # Model now available — re-evaluate the public `enabled` flag that
            # was frozen at construction time, and build connections.
            self.enabled = _MEDIAPIPE_AVAILABLE and self._model_ok
            logger.info("face_mesh_model_late_detected", path=self.model_path)
            if not _CONNECTIONS:
                _CONNECTIONS.update(_build_connections())

        if self._running:
            return

        # Build connection index tuples on first start
        if not _CONNECTIONS:
            _CONNECTIONS.update(_build_connections())

        # Precompute static edge-index arrays once for vectorized drawing
        self._prepare_edge_arrays()

        self._running = True
        self._thread = threading.Thread(
            target=self._mesh_worker,
            name="FaceMeshWorkerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "face_mesh_thread_started",
            max_fps=min(self.max_fps, _OVERLAY_MAX_FPS) if self.max_fps > 0 else _OVERLAY_MAX_FPS,
        )

    def stop(self) -> None:
        """Stop the face mesh background thread gracefully."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._latest_mesh_frame = None
        logger.info("face_mesh_thread_stopped")

    def get_latest_mesh_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent face-mesh-annotated frame.

        Returns:
            np.ndarray | None: BGR frame with mesh overlay drawn, or None
                if no frame is available yet or camera is paused.
        """
        with self._lock:
            if self._latest_mesh_frame is None:
                return None
            return self._latest_mesh_frame.copy()

    # ------------------------------------------------------------------
    # Internal: background mesh worker
    # ------------------------------------------------------------------

    def _mesh_worker(self) -> None:
        """Background thread: run MediaPipe FaceLandmarker capped at 30 FPS.

        Initializes a single FaceLandmarker context per thread (not
        thread-safe across threads in MediaPipe), reads frames from
        CameraCapture, applies the mesh overlay, and stores the result.

        Uses VIDEO running mode which gives the best throughput for
        per-frame calls without requiring timestamps or async callbacks.
        A monotonic timestamp_ms is synthesised from time.perf_counter();
        strict monotonicity is guaranteed with ``ts = max(prev_ts + 1, now_ms)``
        because the Tasks API rejects non-increasing timestamps.

        The whole per-frame body is wrapped in try/except: recoverable errors
        are logged (rate-limited) and the landmarker context is recreated.
        Repeated consecutive failures mark the worker dead: ``_running`` is
        cleared and the error stored in ``last_error``/``status()`` so
        ``is_available`` stops reporting True.
        """
        # Cap the overlay loop at 30 FPS even when configured uncapped.
        effective_fps = min(self.max_fps, _OVERLAY_MAX_FPS) if self.max_fps > 0 else _OVERLAY_MAX_FPS
        min_frame_time = 1.0 / effective_fps

        max_consecutive_errors = 5
        error_log_interval_s = 5.0
        consecutive_errors = 0
        last_error_logged = 0.0
        prev_timestamp_ms = -1  # strictly monotonic across context recreations

        while self._running:
            face_landmarker = None
            try:
                options = _FaceLandmarkerOptions(
                    base_options=_MpBaseOptions(model_asset_path=str(self.model_path)),
                    running_mode=_VisionRunningMode.VIDEO,
                    num_faces=self.max_num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                face_landmarker = _FaceLandmarker.create_from_options(options)
                logger.info(
                    "face_mesh_context_ready",
                    max_faces=self.max_num_faces,
                    model=self.model_path,
                )
                consecutive_errors = 0  # context healthy again

                t_epoch = time.perf_counter()  # epoch for synthesised timestamps

                while self._running:
                    t0 = time.perf_counter()

                    # Skip processing when camera is paused (privacy mode)
                    if self.camera.is_paused:
                        with self._lock:
                            self._latest_mesh_frame = None
                        time.sleep(0.1)
                        continue

                    raw_frame = self.camera.get_latest_frame()
                    if raw_frame is None:
                        time.sleep(0.005)
                        continue

                    # MediaPipe Tasks API expects RGB
                    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    # Synthesise a STRICTLY monotonically increasing timestamp
                    # in milliseconds (VIDEO mode raises on equal/older ts).
                    timestamp_ms = max(
                        prev_timestamp_ms + 1,
                        int((time.perf_counter() - t_epoch) * 1000),
                    )
                    prev_timestamp_ms = timestamp_ms

                    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                    # Draw overlays on a BGR copy
                    annotated = raw_frame.copy()
                    h, w = annotated.shape[:2]

                    if results.face_landmarks:
                        for face_lm in results.face_landmarks:
                            self._draw_face_mesh(annotated, face_lm, w, h)

                    # Overlay FPS counter (top-right corner, semi-transparent)
                    fps_text = f"FACE MESH  {self._measured_fps:.0f} FPS"
                    (tw, th), _ = cv2.getTextSize(
                        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
                    )
                    fx = w - tw - 8
                    fy = 18
                    cv2.rectangle(
                        annotated, (fx - 4, fy - th - 3), (fx + tw + 4, fy + 3),
                        (4, 18, 14), -1,
                    )
                    cv2.rectangle(
                        annotated, (fx - 4, fy - th - 3), (fx + tw + 4, fy + 3),
                        _TESSELATION_COLOR, 1,
                    )
                    cv2.putText(
                        annotated, fps_text, (fx, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        _TESSELATION_COLOR, 1, cv2.LINE_AA,
                    )

                    with self._lock:
                        self._latest_mesh_frame = annotated

                    # FPS accounting
                    self._frame_count += 1
                    now = time.perf_counter()
                    elapsed = now - self._last_fps_time
                    if elapsed >= 1.0:
                        self._measured_fps = self._frame_count / elapsed
                        self._frame_count = 0
                        self._last_fps_time = now

                    # FPS cap: sleep only the remaining budget
                    dt = time.perf_counter() - t0
                    if dt < min_frame_time:
                        time.sleep(min_frame_time - dt)
                    else:
                        # Yield GIL so other threads can run between frames
                        time.sleep(0)

            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                if not self._running:
                    break  # graceful shutdown while an exception unwound

                consecutive_errors += 1
                now_ts = time.time()
                if now_ts - last_error_logged >= error_log_interval_s:
                    last_error_logged = now_ts
                    logger.warning(
                        "face_mesh_worker_error_recreating_context",
                        error=self._last_error,
                        consecutive_errors=consecutive_errors,
                    )

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        "face_mesh_worker_fatal_stopping",
                        error=self._last_error,
                        consecutive_errors=consecutive_errors,
                    )
                    self._running = False
                    break

                # Back off briefly, then recreate the landmarker context.
                time.sleep(1.0)

            finally:
                if face_landmarker is not None:
                    try:
                        face_landmarker.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Internal: drawing helpers
    # ------------------------------------------------------------------

    def _prepare_edge_arrays(self) -> None:
        """Precompute static landmark-index edge arrays once.

        Converts the ~2556 tessellation, ~160 contour, and iris connection
        tuples into int32 arrays of shape (N, 2) at startup so per-frame
        drawing is a single vectorized index lookup + one cv2.polylines call
        per connection set instead of thousands of cv2.line calls.
        """
        self._edge_idx = {}
        for name, conns in _CONNECTIONS.items():
            if conns:
                self._edge_idx[name] = np.array(conns, dtype=np.int32).reshape(-1, 2)
            else:
                self._edge_idx[name] = np.empty((0, 2), dtype=np.int32)

    def _draw_edge_set(
        self,
        image: np.ndarray,
        pts: np.ndarray,
        key: str,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw one precomputed connection set as a single cv2.polylines call.

        Args:
            image: BGR frame to draw onto (mutated in-place).
            pts: (num_pts, 2) int32 pixel coordinates of the current face.
            key: Connection-set name in ``self._edge_idx``.
            color: BGR line color.
            thickness: Line thickness in pixels.
        """
        idx = self._edge_idx.get(key)
        if idx is None or idx.size == 0:
            return
        num_pts = pts.shape[0]
        valid = idx[(idx[:, 0] < num_pts) & (idx[:, 1] < num_pts)]
        if valid.size == 0:
            return
        # pts[valid] -> (k, 2, 2) int32: k 2-point polylines rendered in ONE call.
        cv2.polylines(
            image, pts[valid], isClosed=False, color=color,
            thickness=thickness, lineType=cv2.LINE_AA,
        )

    def _draw_face_mesh(
        self,
        image: np.ndarray,
        face_landmarks: list,
        width: int,
        height: int,
    ) -> None:
        """Draw MediaPipe face mesh tessellation, contours, and irises.

        Vectorized hot path: landmark pixel coordinates are built with
        np.fromiter (no per-point tuple allocations in Python loops), each
        connection set is rendered with ONE cv2.polylines call over
        precomputed static edge-index pairs, and the 478 landmark dots are
        scattered with a single vectorized fancy-index write. Visual output
        matches the previous per-line/per-circle green mesh rendering.

        Args:
            image: BGR frame to draw onto (mutated in-place).
            face_landmarks: Single face normalized landmark list from
                FaceLandmarkerResult.face_landmarks (list of NormalizedLandmark).
            width: Frame pixel width.
            height: Frame pixel height.
        """
        num_pts = len(face_landmarks)
        if num_pts == 0:
            return

        xs = np.fromiter((lm.x for lm in face_landmarks), dtype=np.float64, count=num_pts)
        ys = np.fromiter((lm.y for lm in face_landmarks), dtype=np.float64, count=num_pts)
        pts = np.empty((num_pts, 2), dtype=np.int32)
        pts[:, 0] = xs * width
        pts[:, 1] = ys * height

        # --- Tessellation: all FACE_LANDMARKS_TESSELATION edges in one call ---
        if self.draw_tesselation:
            self._draw_edge_set(image, pts, "tesselation", _TESSELATION_COLOR, _TESSELATION_THICKNESS)

        # --- Contours: lips, eyes, brows, face oval ---
        if self.draw_contours:
            self._draw_edge_set(image, pts, "contours", _CONTOUR_COLOR, _CONTOUR_THICKNESS)

        # --- Irises: lines around iris landmark rings ---
        if self.draw_irises:
            self._draw_edge_set(image, pts, "irises", _IRIS_COLOR, _IRIS_THICKNESS)

        # --- Landmark dots: single vectorized scatter at each of the 478 points ---
        inside = (
            (pts[:, 0] >= 0) & (pts[:, 0] < width)
            & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        )
        dot_pts = pts[inside]
        if dot_pts.size > 0:
            image[dot_pts[:, 1], dot_pts[:, 0]] = _TESSELATION_COLOR
