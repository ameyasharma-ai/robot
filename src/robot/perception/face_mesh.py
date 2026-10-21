"""High-FPS Google MediaPipe Face Mesh overlay for live camera stream.

Runs a dedicated background thread that processes raw camera frames at the
maximum rate the hardware supports (typically 30–60 FPS). Draws the full
468-point MediaPipe Face Mesh tessellation, contours, and irises in green
onto each frame and exposes the result via ``get_latest_mesh_frame()``.

Design constraints:
- RTX 3050 (4 GB VRAM) — CPU-only MediaPipe to avoid VRAM pressure from GPU
  inference pipeline (Whisper, LLM, TTS already saturate VRAM budget).
- Zero blocking of asyncio event loop — dedicated daemon thread only.
- Thread-safe double-buffer with lock: writer thread sets latest frame,
  MJPEG stream reads a copy under the same lock.
- Graceful no-op when MediaPipe is unavailable (module-level try/except).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.perception.camera import CameraCapture

logger = get_logger("perception.face_mesh")

# ---------------------------------------------------------------------------
# Optional MediaPipe import — fails gracefully if package not installed
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp  # type: ignore[import-untyped]

    _mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
    _mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
    _mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore[attr-defined]
    _MEDIAPIPE_AVAILABLE = True
    logger.info("mediapipe_face_mesh_available", version=mp.__version__)
except ModuleNotFoundError:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "mediapipe_not_installed",
        hint="Run: .venv/bin/pip install mediapipe",
    )


# ---------------------------------------------------------------------------
# Drawing constants (Google-green palette matching perception engine HUD)
# ---------------------------------------------------------------------------
_TESSELATION_COLOR = (0, 255, 128)   # BGR bright green
_CONTOUR_COLOR = (0, 230, 60)        # BGR slightly deeper green
_IRIS_COLOR = (0, 200, 255)          # BGR cyan-green for irises
_TESSELATION_THICKNESS = 1
_CONTOUR_THICKNESS = 1
_IRIS_THICKNESS = 1


class FaceMeshOverlay:
    """Dedicated high-FPS MediaPipe Face Mesh compositing thread.

    Reads raw frames directly from CameraCapture at max hardware speed,
    applies the 468-point face mesh tessellation and contour overlays in
    green, and exposes the composite frame for the MJPEG stream.

    Attributes:
        camera: CameraCapture instance to read raw frames from.
        enabled: Whether the face mesh thread is active.
    """

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
    ) -> None:
        """Initialize the FaceMeshOverlay.

        Args:
            camera: CameraCapture instance to pull raw frames from.
            max_fps: Target FPS cap (0 = uncapped, runs as fast as hardware
                allows). Note: the loop still yields via time.sleep(0) between
                frames so Python GIL is released.
            min_detection_confidence: MediaPipe face detection confidence
                threshold [0, 1].
            min_tracking_confidence: MediaPipe landmark tracking confidence
                threshold [0, 1].
            max_num_faces: Maximum number of simultaneous faces to track.
            draw_tesselation: If True, draw full 468-point mesh tessellation.
            draw_contours: If True, draw facial feature contours (lips, brows,
                eye outlines, oval).
            draw_irises: If True, draw iris landmark rings (requires
                refine_landmarks=True in FaceMesh).
        """
        self.camera = camera
        self.max_fps = max_fps
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_faces = max_num_faces
        self.draw_tesselation = draw_tesselation
        self.draw_contours = draw_contours
        self.draw_irises = draw_irises

        self.enabled = _MEDIAPIPE_AVAILABLE

        self._lock = threading.Lock()
        self._latest_mesh_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Performance counters
        self._frame_count: int = 0
        self._last_fps_time: float = time.time()
        self._measured_fps: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Return True if MediaPipe is installed and face mesh is active."""
        return _MEDIAPIPE_AVAILABLE and self._running

    @property
    def measured_fps(self) -> float:
        """Return the measured processing FPS of the face mesh thread."""
        return self._measured_fps

    def start(self) -> None:
        """Start the face mesh background thread.

        No-op if MediaPipe is unavailable or thread is already running.
        """
        if not _MEDIAPIPE_AVAILABLE:
            logger.warning(
                "face_mesh_start_skipped",
                reason="mediapipe not installed",
            )
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._mesh_worker,
            name="FaceMeshWorkerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("face_mesh_thread_started", max_fps=self.max_fps or "uncapped")

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
        """Background thread: run MediaPipe Face Mesh at max hardware FPS.

        Initializes a single FaceMesh context per thread (not thread-safe
        across threads in MediaPipe), reads frames from CameraCapture,
        applies the 468-point mesh overlay, and stores the result.

        The ``refine_landmarks=True`` flag enables the 5 iris landmark
        points per eye, enabling iris circle drawing.
        """
        min_frame_time = (1.0 / self.max_fps) if self.max_fps > 0 else 0.0

        with _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,   # enables 5 iris landmarks per eye
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as face_mesh:
            logger.info(
                "face_mesh_context_ready",
                max_faces=self.max_num_faces,
                refine_landmarks=True,
            )

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

                # MediaPipe expects RGB, OpenCV gives BGR
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

                # Mark as non-writeable before processing for performance
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True

                # Draw overlays on a BGR copy
                annotated = raw_frame.copy()
                h, w = annotated.shape[:2]

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self._draw_face_mesh(annotated, face_landmarks, w, h)

                # Overlay FPS counter (top-right corner, semi-transparent look)
                fps_text = f"FACE MESH  {self._measured_fps:.0f} FPS"
                (tw, th), _ = cv2.getTextSize(
                    fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
                )
                fx = w - tw - 8
                fy = 18
                cv2.rectangle(
                    annotated, (fx - 4, fy - th - 3), (fx + tw + 4, fy + 3),
                    (4, 18, 14), -1
                )
                cv2.rectangle(
                    annotated, (fx - 4, fy - th - 3), (fx + tw + 4, fy + 3),
                    _TESSELATION_COLOR, 1
                )
                cv2.putText(
                    annotated, fps_text, (fx, fy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    _TESSELATION_COLOR, 1, cv2.LINE_AA
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

                # FPS cap: sleep only the remaining budget (skip if uncapped)
                if min_frame_time > 0:
                    dt = time.perf_counter() - t0
                    if dt < min_frame_time:
                        time.sleep(min_frame_time - dt)
                else:
                    # Yield GIL so other threads can run between frames
                    time.sleep(0)

    # ------------------------------------------------------------------
    # Internal: drawing helpers
    # ------------------------------------------------------------------

    def _draw_face_mesh(
        self,
        image: np.ndarray,
        face_landmarks: "mp.framework.formats.landmark_pb2.NormalizedLandmarkList",
        width: int,
        height: int,
    ) -> None:
        """Draw MediaPipe face mesh tessellation, contours, and irises.

        Uses manual pixel-level drawing for full color/thickness control
        (green palette, 1-px AA lines) instead of MediaPipe's default
        drawing_utils which uses gray/white colors.

        Args:
            image: BGR frame to draw onto (mutated in-place).
            face_landmarks: Single face landmark set from MediaPipe results.
            width: Frame pixel width.
            height: Frame pixel height.
        """
        # Convert landmarks to pixel coordinates array once
        pts = np.array(
            [(int(lm.x * width), int(lm.y * height))
             for lm in face_landmarks.landmark],
            dtype=np.int32,
        )

        # --- Tessellation: draw all FACE_CONNECTIONS edges ---
        if self.draw_tesselation:
            for conn in _mp_face_mesh.FACEMESH_TESSELATION:
                s, e = conn
                if s < len(pts) and e < len(pts):
                    cv2.line(
                        image,
                        tuple(pts[s]),
                        tuple(pts[e]),
                        _TESSELATION_COLOR,
                        _TESSELATION_THICKNESS,
                        cv2.LINE_AA,
                    )

        # --- Contours: lips, eyes, brows, face oval ---
        if self.draw_contours:
            for contour_set in (
                _mp_face_mesh.FACEMESH_CONTOURS,
                _mp_face_mesh.FACEMESH_FACE_OVAL,
            ):
                for conn in contour_set:
                    s, e = conn
                    if s < len(pts) and e < len(pts):
                        cv2.line(
                            image,
                            tuple(pts[s]),
                            tuple(pts[e]),
                            _CONTOUR_COLOR,
                            _CONTOUR_THICKNESS,
                            cv2.LINE_AA,
                        )

        # --- Irises: draw circles around iris landmark rings ---
        if self.draw_irises and hasattr(_mp_face_mesh, "FACEMESH_IRISES"):
            for conn in _mp_face_mesh.FACEMESH_IRISES:
                s, e = conn
                if s < len(pts) and e < len(pts):
                    cv2.line(
                        image,
                        tuple(pts[s]),
                        tuple(pts[e]),
                        _IRIS_COLOR,
                        _IRIS_THICKNESS,
                        cv2.LINE_AA,
                    )

        # --- Landmark dots: draw small circle at each landmark point ---
        for pt in pts:
            cv2.circle(image, tuple(pt), 1, _TESSELATION_COLOR, -1, cv2.LINE_AA)
