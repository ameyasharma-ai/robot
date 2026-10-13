"""Unit tests for Dual-Column Custom API (OpenRouter + AI Credits) architecture."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from robot.api import create_app
from robot.config import update_env_variable


@pytest.fixture
def test_engine() -> MagicMock:
    """Fixture providing a mock RobotEngine configured for API tests."""
    engine = MagicMock()
    engine.config = {
        "llm": {
            "openai_compatible": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-or-test-123",
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "context_window": 131072,
                "enable_prompt_caching": True,
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-or-test-123",
                "model": "meta-llama/llama-3.3-70b-instruct:free",
            },
            "aicredits": {
                "base_url": "https://aicredits.in/v1",
                "api_key": "sk-live-test-456",
                "model": "aion-labs/aion-3.0",
            },
        }
    }
    engine.state.state.mode.value = "idle"
    engine.state.state.active_personality = "luna"
    engine.state.state.turn_count = 1
    engine.state.state.services = {}
    engine.telemetry.snapshot.return_value = {}
    engine.player.is_playing = False
    engine._is_speaking = False
    engine.voice_enabled = True
    engine.is_chat_mode = False
    engine.llm_provider_type = "openai_compatible"
    return engine


def test_llm_status_dual_providers(test_engine: MagicMock) -> None:
    """Verify that /api/llm/status returns both openrouter and aicredits configs independently."""
    app = create_app(test_engine)
    client = TestClient(app)

    response = client.get("/api/llm/status")
    assert response.status_code == 200
    data = response.json()

    assert "openrouter" in data
    assert "aicredits" in data
    assert data["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
    assert data["openrouter"]["api_key"] == "sk-or-test-123"
    assert data["aicredits"]["base_url"] == "https://aicredits.in/v1"
    assert data["aicredits"]["api_key"] == "sk-live-test-456"


def test_dual_api_config_persistence(test_engine: MagicMock) -> None:
    """Verify that updating both providers persists them concurrently without clearing either."""
    app = create_app(test_engine)
    client = TestClient(app)

    with patch("robot.config.update_env_variable") as mock_update_env:
        payload = {
            "active_provider": "aicredits",
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-or-new-key-111",
                "model": "meta-llama/llama-3.3-70b-instruct:free",
            },
            "aicredits": {
                "base_url": "https://aicredits.in/v1",
                "api_key": "sk-live-new-key-222",
                "model": "aion-labs/aion-3.0",
            },
            "context_window": 128000,
            "enable_prompt_caching": True,
        }
        response = client.post("/api/llm/openai/config", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

        # Verify OpenRouter was preserved in config
        or_cfg = test_engine.config["llm"]["openrouter"]
        assert or_cfg["api_key"] == "sk-or-new-key-111"

        # Verify AICredits was preserved in config
        ai_cfg = test_engine.config["llm"]["aicredits"]
        assert ai_cfg["api_key"] == "sk-live-new-key-222"

        # Verify active provider is set to aicredits
        active_cfg = test_engine.config["llm"]["openai_compatible"]
        assert active_cfg["base_url"] == "https://aicredits.in/v1"
        assert active_cfg["api_key"] == "sk-live-new-key-222"
        assert active_cfg["model"] == "aion-labs/aion-3.0"


def test_view_mode_toggle_synchronization(test_engine: MagicMock) -> None:
    """Verify that /api/view-mode synchronizes engine.is_chat_mode."""
    app = create_app(test_engine)
    client = TestClient(app)

    resp_chat = client.post("/api/view-mode", json={"mode": "chat"})
    assert resp_chat.status_code == 200
    assert test_engine.is_chat_mode is True

    resp_dual = client.post("/api/view-mode", json={"mode": "dual"})
    assert resp_dual.status_code == 200
    assert test_engine.is_chat_mode is False
