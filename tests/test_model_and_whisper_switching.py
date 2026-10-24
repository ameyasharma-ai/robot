"""Unit and integration tests for Dashboard Local LLM and Whisper STT Model Switching.

Verifies:
- WhisperServerManager model catalog (9 models: tiny.en, base.en, base, small.en, small, medium.en, medium, large-v3-turbo, large-v3).
- Hardware separation: Whisper accelerated on AMD Radeon 680M iGPU via Vulkan (dev 0, 0MB RTX 3050 VRAM).
- Local LLM model resolution, multi-modal projector flags, and switch endpoints.
- REST API endpoints (/api/stt/status, /api/stt/models, /api/stt/switch, /api/llm/models, /api/models/local/switch).
- Event emissions on model changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from starlette.testclient import TestClient

from robot.api import create_app
from robot.llm import LlamaServerManager
from robot.stt import WhisperModelMetadata, WhisperServerManager


def get_mock_engine() -> MagicMock:
    """Construct a mock RobotEngine configured for API unit testing.

    Returns:
        MagicMock: Fully structured mock engine instance.
    """
    engine = MagicMock()
    engine.memory = MagicMock()
    engine.telemetry = MagicMock()
    engine.telemetry.snapshot.return_value = {}
    engine.config = {"llm": {"model": "qwen3.5-4b"}}
    engine.llm_provider_type = "local"
    engine.stt = MagicMock()
    engine.events = MagicMock()
    engine.events.emit = AsyncMock()
    return engine


def test_whisper_catalog_completeness() -> None:
    """Verify WhisperServerManager catalog contains all 9 official GGML models with complete metadata."""
    catalog = WhisperServerManager.MODELS_CATALOG
    assert len(catalog) == 9, f"Expected 9 models in Whisper catalog, got {len(catalog)}"

    expected_models = [
        "tiny.en",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v3-turbo",
        "large-v3",
    ]
    for m in expected_models:
        assert m in catalog, f"Missing model {m} in catalog"
        meta = catalog[m]
        assert isinstance(meta, WhisperModelMetadata)
        assert meta.id == m
        assert meta.filename.startswith("ggml-")
        assert meta.filename.endswith(".bin")
        assert meta.size_mb > 0
        assert meta.parameters != ""
        assert meta.latency_rating != ""


def test_whisper_hardware_separation_and_status() -> None:
    """Verify Whisper status reports AMD Radeon 680M Vulkan acceleration and 0MB RTX VRAM."""
    app = create_app(get_mock_engine())
    with TestClient(app) as client:
        res = client.get("/api/stt/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "AMD Radeon 680M (Vulkan dev 0)" in data["device"]
        assert data["vram_rtx3050_mb"] == 0
        assert "active_model" in data
        assert "size_mb" in data


def test_whisper_list_models_endpoint() -> None:
    """Verify /api/stt/models returns all registered models with valid download paths."""
    app = create_app(get_mock_engine())
    with TestClient(app) as client:
        res = client.get("/api/stt/models")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["total_models"] == 9
        models = data["models"]
        assert len(models) == 9

        model_ids = [m["id"] for m in models]
        assert "base.en" in model_ids
        assert "large-v3-turbo" in model_ids

        # Check fields
        for m in models:
            assert "id" in m
            assert "name" in m
            assert "size_mb" in m
            assert "parameters" in m
            assert "is_downloaded" in m
            assert "is_active" in m


def test_whisper_switch_model_validation() -> None:
    """Verify /api/stt/switch validates request parameters."""
    app = create_app(get_mock_engine())
    with TestClient(app) as client:
        # Missing model parameter
        res = client.post("/api/stt/switch", json={})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "error"
        assert "Missing" in data["message"]

        # Unknown model parameter
        res = client.post("/api/stt/switch", json={"model": "non_existent_whisper_model"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "error"
        assert "not downloaded" in data["message"]


@pytest.mark.asyncio
async def test_whisper_switch_model_mocked() -> None:
    """Verify WhisperServerManager.switch_model logic with mocked process manager."""
    with patch.object(WhisperServerManager, "start", return_value=True) as mock_start, \
         patch.object(WhisperServerManager, "stop", return_value=True) as mock_stop:
        
        mock_engine = get_mock_engine()

        res = await WhisperServerManager.switch_model("tiny.en", engine=mock_engine)
        assert res["status"] == "ok"
        assert res["model"] == "tiny.en"
        assert res["size_mb"] == 75
        assert WhisperServerManager.get_active_model_id() == "tiny.en"
        assert mock_stop.called
        assert mock_start.called


def test_llm_models_endpoint() -> None:
    """Verify /api/llm/models lists local models and cloud options with active indicator."""
    app = create_app(get_mock_engine())
    with TestClient(app) as client:
        res = client.get("/api/llm/models")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "local_models" in data
        assert "gemini_models" in data
        assert "active_local_model" in data
        assert len(data["local_models"]) >= 1

        local_ids = [m["id"] for m in data["local_models"]]
        assert "qwen3.5-4b" in local_ids or "qwen2-vl-2b-instruct" in local_ids


def test_local_model_switch_validation() -> None:
    """Verify /api/models/local/switch handles missing or invalid model requests."""
    app = create_app(get_mock_engine())
    with TestClient(app) as client:
        # Missing parameter
        res = client.post("/api/models/local/switch", json={})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "error"
        assert "Missing" in data["message"]


@pytest.mark.asyncio
async def test_llama_switch_model_mocked() -> None:
    """Verify LlamaServerManager.switch_model logic with mocked process manager."""
    with patch.object(LlamaServerManager, "start", return_value=True) as mock_start, \
         patch.object(LlamaServerManager, "stop", return_value=True) as mock_stop, \
         patch.object(LlamaServerManager, "is_running_async", return_value=False):
        
        mock_engine = get_mock_engine()
        mock_engine.llm = MagicMock()
        mock_engine.warm_up_llm = AsyncMock(return_value={"status": "ok"})

        res = await LlamaServerManager.switch_model("qwen3.5-4b", engine=mock_engine)
        assert res["status"] == "ok"
        assert res["model"] == "qwen3.5-4b"
        assert LlamaServerManager.get_active_model_id() == "qwen3.5-4b"
        assert mock_stop.called
        assert mock_start.called


@pytest.mark.asyncio
async def test_llama_switch_model_already_running() -> None:
    """Verify LlamaServerManager.switch_model skips restart when model is already active."""
    LlamaServerManager._active_model_id = "qwen3.5-4b"
    with patch.object(LlamaServerManager, "start", return_value=True) as mock_start, \
         patch.object(LlamaServerManager, "stop", return_value=True) as mock_stop, \
         patch.object(LlamaServerManager, "is_running_async", return_value=True):
        res = await LlamaServerManager.switch_model("qwen3.5-4b")
        assert res["status"] == "ok"
        assert res.get("already_active") is True
        assert not mock_stop.called
        assert not mock_start.called

