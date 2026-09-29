"""Asynchronous camera acquisition using OpenCV.

Features:
- Dedicated background worker thread to prevent blocking asyncio event loop
- Thread-safe frame caching (latest frame always available with zero copy wait)
- Configurable camera index, width, height, and FPS
- Graceful error handling if camera is unavailable or disconnected
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from robot.telemetry import get_logger

logger = get_logger("perception.camera")


class CameraCapture:
    """Non-blocking camera capture wrapper using OpenCV."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0
        self._is_opened: bool = False

    @property
    def is_running(self) -> bool:
        return self._running and self._is_opened

    def start(self) -> bool:
        """Start the background capture thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="CameraCaptureWorker")
        self._thread.start()
        logger.info("camera_worker_started", device_index=self.device_index)
        return True

    def _capture_worker(self) -> None:
        """Continuously reads frames from camera into the cached buffer."""
        import cv2

        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            logger.warning("camera_open_failed", device_index=self.device_index)
            self._is_opened = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Enable hardware auto-focus and auto-exposure if supported
        try:
            # 1 = continuous autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass

        try:
            # 0.75 / 3 = auto exposure depending on backend
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        except Exception:
            pass

        self._is_opened = True
        logger.info("camera_opened_successfully", device_index=self.device_index)

        interval = 1.0 / max(1, self.target_fps)

        try:
            while self._running:
                t0 = time.time()
                ret, frame = cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._last_frame_time = time.time()
                        self._frame_count += 1
                else:
                    time.sleep(0.05)

                dt = time.time() - t0
                if dt < interval:
                    time.sleep(interval - dt)
        finally:
            cap.release()
            self._is_opened = False
            logger.info("camera_released")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest captured frame (BGR format), or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self) -> None:
        """Stop the background capture thread and release hardware resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._latest_frame = None
        logger.info("camera_stopped")
