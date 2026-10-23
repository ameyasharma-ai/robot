"""Asynchronous camera acquisition using OpenCV.

Features:
- Dedicated background worker thread to prevent blocking asyncio event loop
- Thread-safe frame caching (latest frame always available with zero copy wait)
- Configurable camera index, width, height, and FPS
- Graceful error handling if camera is unavailable or disconnected
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Generator, Optional

import numpy as np

from robot.telemetry import get_logger

logger = get_logger("perception.camera")


# fd 2 is process-global, so concurrent silence_c_stderr() calls from
# different threads would corrupt each other's saved descriptors. Serialize
# all silences through this module-level lock (not reentrant — never call
# silence_c_stderr() while already inside another silence block on the same thread).
_SILENCE_C_STDERR_LOCK = threading.Lock()


@contextlib.contextmanager
def silence_c_stderr() -> Generator[None, None, None]:
    """Temporarily redirect C-level stderr (fd 2) to devnull.

    Suppresses noisy C++ backend warning spam from OpenCV and ffmpeg
    when probing nonexistent or loopback video capture devices.

    The global redirection is serialized with a module-level lock and every
    descriptor (saved stderr, devnull) is closed on all paths via finally,
    so no fd leaks remain if setup or restore partially fails.

    Yields:
        None.
    """
    with _SILENCE_C_STDERR_LOCK:
        stderr_fd = 2
        saved_stderr: int = -1
        devnull: int = -1
        redirected = False
        try:
            try:
                saved_stderr = os.dup(stderr_fd)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stderr_fd)
                redirected = True
            except Exception:
                # Setup failed: run un-silenced instead of leaving fds corrupted.
                pass
            finally:
                if devnull != -1:
                    try:
                        os.close(devnull)
                    except OSError:
                        pass
            yield
        finally:
            if redirected:
                try:
                    os.dup2(saved_stderr, stderr_fd)
                finally:
                    os.close(saved_stderr)
            elif saved_stderr != -1:
                # dup succeeded but dup2 failed before redirection.
                try:
                    os.close(saved_stderr)
                except OSError:
                    pass


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
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0
        self._is_opened: bool = False

    @property
    def is_running(self) -> bool:
        return self._running and self._is_opened and not self._paused

    @property
    def is_paused(self) -> bool:
        """Return True if camera capture is temporarily paused."""
        return self._paused

    def pause(self) -> None:
        """Pause capture and release hardware video device for privacy or chat mode."""
        self._paused = True
        with self._lock:
            self._latest_frame = None
        logger.info("camera_paused_hardware_released")

    def resume(self) -> None:
        """Resume camera capture and reconnect to video device."""
        self._paused = False
        logger.info("camera_resumed")

    def start(self) -> bool:
        """Start the background capture thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="CameraCaptureWorker")
        self._thread.start()
        logger.info("camera_worker_started", device_index=self.device_index)
        return True

    def _find_working_camera(self) -> int | None:
        """Probe candidate video device indices to locate an active, working camera.

        Returns:
            int | None: Index of the first responsive video capture device, or None.
        """
        import cv2

        candidates: list[int] = []
        if isinstance(self.device_index, int) and self.device_index >= 0:
            candidates.append(self.device_index)
        for idx in (0, 1, 2, 3, 4):
            if idx not in candidates:
                candidates.append(idx)

        for idx in candidates:
            try:
                with silence_c_stderr():
                    test_cap = cv2.VideoCapture(idx)
                    if test_cap.isOpened():
                        ret, test_frame = test_cap.read()
                        test_cap.release()
                        if ret and test_frame is not None and test_frame.size > 0:
                            logger.info("camera_auto_detected", candidate_index=idx)
                            return idx
            except Exception:
                pass
        return None

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """Capture and return the most recent camera frame or take an on-demand frame.

        Returns:
            np.ndarray | None: BGR numpy image array, or None if camera is unavailable or paused.
        """
        if self._paused:
            return None

        frame = self.get_latest_frame()
        if frame is not None and frame.size > 0:
            return frame.copy()

        # Fallback to direct snapshot if background loop has not yet delivered a frame
        working_idx = self._find_working_camera()
        if working_idx is not None:
            try:
                import cv2
                with silence_c_stderr():
                    temp_cap = cv2.VideoCapture(working_idx)
                    if temp_cap.isOpened():
                        ret, snap = temp_cap.read()
                        temp_cap.release()
                        if ret and snap is not None and snap.size > 0:
                            return snap
            except Exception:
                pass
        return None

    def _capture_worker(self) -> None:
        """Continuously read frames from camera with self-healing auto-reconnect and device probing."""
        self._reconnect_count = 0
        self._reconnect_backoff = 1.0

        while self._running:
            if self._paused:
                self._is_opened = False
                with self._lock:
                    self._latest_frame = None
                time.sleep(0.1)
                continue

            import cv2

            os.environ["OPENCV_LOG_LEVEL"] = "OFF"
            os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
            with silence_c_stderr():
                cap = cv2.VideoCapture(self.device_index)

            # If default index fails, automatically probe for available device
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass

                working_idx = self._find_working_camera()
                if working_idx is not None and working_idx != self.device_index:
                    logger.info("switching_to_working_camera", old_index=self.device_index, new_index=working_idx)
                    self.device_index = working_idx
                    with silence_c_stderr():
                        cap = cv2.VideoCapture(self.device_index)

            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                self._is_opened = False
                self._reconnect_count += 1
                if self._reconnect_count <= 3:
                    logger.warning(
                        "camera_open_failed_retrying",
                        device_index=self.device_index,
                        backoff_s=self._reconnect_backoff,
                    )
                elif self._reconnect_count == 4:
                    logger.info(
                        "camera_device_unavailable_dormant",
                        device_index=self.device_index,
                        note="No video capture device detected; will retry periodically in background every 60s",
                    )
                else:
                    logger.debug("camera_open_dormant_retry", device_index=self.device_index)

                # Exponential backoff sleep (interrupted if stop() is called)
                sleep_start = time.time()
                while self._running and not self._paused and (time.time() - sleep_start) < self._reconnect_backoff:
                    time.sleep(0.1)
                self._reconnect_backoff = min(60.0 if self._reconnect_count >= 4 else 10.0, self._reconnect_backoff * 2.0)
                continue

            # Connected successfully
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Enable hardware auto-focus and auto-exposure if supported
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass

            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            except Exception:
                pass

            self._is_opened = True
            self._reconnect_backoff = 1.0
            logger.info("camera_opened_successfully", device_index=self.device_index, reconnects=self._reconnect_count)

            interval = 1.0 / max(1, self.target_fps)
            consecutive_failures = 0

            try:
                while self._running and not self._paused:
                    t0 = time.time()
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        consecutive_failures = 0
                        with self._lock:
                            self._latest_frame = frame
                            self._last_frame_time = time.time()
                            self._frame_count += 1
                    else:
                        consecutive_failures += 1
                        time.sleep(0.05)
                        if consecutive_failures > 30:
                            logger.warning(
                                "camera_stream_lost_triggering_reconnect",
                                device_index=self.device_index,
                                failures=consecutive_failures,
                            )
                            break

                    dt = time.time() - t0
                    if dt < interval:
                        time.sleep(interval - dt)
            finally:
                cap.release()
                self._is_opened = False
                self._reconnect_count += 1

        logger.info("camera_worker_terminated", device_index=self.device_index)

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
