"""Tests for camera capture resilience and self-healing auto-reconnect."""

import time
import pytest
from unittest.mock import MagicMock, patch

from robot.perception.camera import CameraCapture


def test_camera_init_defaults():
    cam = CameraCapture(device_index=0, width=640, height=480, fps=30)
    assert cam.device_index == 0
    assert cam.target_width == 640
    assert cam.target_height == 480
    assert cam.target_fps == 30
    assert not cam.is_running
    assert cam.get_latest_frame() is None


def test_camera_reconnection_loop_emulation():
    cam = CameraCapture(device_index=99, width=320, height=240, fps=15)
    
    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2.VideoCapture.return_value = mock_cap

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        cam.start()
        time.sleep(0.2)
        assert cam._reconnect_backoff > 1.0 or not cam.is_running
        cam.stop()
        assert not cam.is_running
