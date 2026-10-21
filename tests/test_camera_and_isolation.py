"""Tests for Camera Snapshot, Multimodal Vision Prompt Injection, Persona Isolation & LM Studio Settings.

Verifies:
1. Camera auto-detection and snapshot frame acquisition.
2. Multimodal vision formatting in context engine (image_data_url support).
3. Memory and user profile isolation (zero profile leakage for custom personas).
4. LM Studio / llama-server hardware settings persistence, CLI generation, and reset.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from robot.core.context import ContextEngine, format_user_core_profile
from robot.llm.settings import LlamaServerSettings, SETTINGS_FILE
from robot.perception.camera import CameraCapture


def test_camera_snapshot_fallback_when_unopened():
    """Verify capture_snapshot gracefully returns None if no camera is available."""
    cam = CameraCapture(device_index=99)
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap_cls.return_value = mock_instance

        frame = cam.capture_snapshot()
        assert frame is None


def test_camera_snapshot_success():
    """Verify capture_snapshot returns ndarray when frame is read successfully."""
    cam = CameraCapture(device_index=1)
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, fake_frame)
        mock_cap_cls.return_value = mock_instance

        frame = cam.capture_snapshot()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert mock_instance.release.call_count >= 1


def test_context_engine_multimodal_vision_image_url():
    """Verify ContextEngine injects image_url content block when image_data_url is provided."""
    engine = ContextEngine()
    test_image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
    user_query = "What do you see?"

    messages = engine.build_messages(
        system_prompt="You are Jarvis.",
        user_text=user_query,
        image_data_url=test_image_url,
    )

    # Last message should be user turn with multipart content list
    last_msg = messages[-1]
    assert last_msg["role"] == "user"
    assert isinstance(last_msg["content"], list)
    assert len(last_msg["content"]) == 2
    assert last_msg["content"][0]["type"] == "text"
    assert user_query in last_msg["content"][0]["text"]
    assert last_msg["content"][1]["type"] == "image_url"
    assert last_msg["content"][1]["image_url"]["url"] == test_image_url


def test_context_engine_text_only_when_no_image():
    """Verify ContextEngine formats standard string content when no image_data_url is given."""
    engine = ContextEngine()
    messages = engine.build_messages(
        system_prompt="You are Jarvis.",
        user_text="Hello Jarvis",
    )

    last_msg = messages[-1]
    assert last_msg["role"] == "user"
    assert isinstance(last_msg["content"], str)
    assert "Hello Jarvis" in last_msg["content"]


def test_user_profile_isolation_zero_leakage():
    """Verify format_user_core_profile returns empty string when user profile is empty or None."""
    # When empty string or empty dict (from custom/isolated personas), MUST NOT inject any profile sheet
    assert format_user_core_profile("") == ""
    assert format_user_core_profile({}) == ""

    # When actual facts exist, format properly
    valid_profile = {
        "user_id": "test_user",
        "name": "Alex",
        "facts": ["Likes robotics", "Lives in Zurich"],
    }
    rendered = format_user_core_profile(valid_profile)
    assert "Alex" in rendered
    assert "Likes robotics" in rendered
    assert "USER CORE IDENTITY PROFILE" in rendered


def test_llama_settings_load_save_cli(tmp_path):
    """Verify LM Studio settings serialization, modification, and CLI argument generation."""
    custom_cfg_path = tmp_path / "llama_settings.json"

    # Default settings
    s = LlamaServerSettings.load(custom_cfg_path)
    assert s.n_gpu_layers == 99
    assert s.ctx_size == 16384
    assert s.flash_attn is True

    # Modify and save
    s.n_gpu_layers = 32
    s.ctx_size = 8192
    s.temp = 0.4
    s.save(custom_cfg_path)

    # Reload from disk
    reloaded = LlamaServerSettings.load(custom_cfg_path)
    assert reloaded.n_gpu_layers == 32
    assert reloaded.ctx_size == 8192
    assert reloaded.temp == 0.4

    # Verify CLI args generation
    cli_args = reloaded.build_cli_args(
        bin_path="/usr/bin/llama-server",
        model_path="/models/test.gguf",
        port=8080,
    )
    assert "--n-gpu-layers" in cli_args or "-ngl" in cli_args
    assert "32" in cli_args
    assert "--ctx-size" in cli_args or "-c" in cli_args
    assert "8192" in cli_args
    assert "--flash-attn" in cli_args or "-fa" in cli_args
