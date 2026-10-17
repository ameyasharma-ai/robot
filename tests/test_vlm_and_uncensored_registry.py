"""Unit tests for 2B Vision Language Models (VLM) and <4B Uncensored Model Registry.

Tests catalog metadata, tool usage capabilities, vision projector resolution,
GGUF header validation, config integration, and the local model API endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from robot.api import create_app
from robot.config import load_system_config
from robot.llm import (
    ModelCategory,
    ModelMetadata,
    ModelRegistry,
)


def test_registry_contains_at_least_ten_models() -> None:
    """Verify that the model registry contains at least 10 curated models."""
    models = ModelRegistry.list_models()
    assert len(models) >= 10, f"Expected at least 10 models in registry, found {len(models)}"


def test_registry_vlm_2b_models_capabilities() -> None:
    """Verify 2B Vision Language Models have vision enabled and tool usage options."""
    vlm_models = ModelRegistry.list_vlm_2b_models()
    assert len(vlm_models) >= 3, f"Expected at least 3 VLM models, found {len(vlm_models)}"

    for model in vlm_models:
        assert model.is_vision is True
        assert model.mmproj_filename is not None
        assert model.mmproj_filename.endswith(".gguf")
        assert model.filename.endswith(".gguf")

    # Verify default 2B VLM supports tool usage
    default_model = ModelRegistry.get_default_model()
    assert default_model.id == "qwen2-vl-2b-instruct"
    assert default_model.is_vision is True
    assert default_model.supports_tools is True
    assert default_model.mmproj_filename == "mmproj-Qwen2-VL-2B-Instruct-f16.gguf"


def test_registry_uncensored_under_4b_models() -> None:
    """Verify uncensored and abliterated models under 4B parameters."""
    uncensored_models = ModelRegistry.list_uncensored_models()
    assert len(uncensored_models) >= 6, (
        f"Expected at least 6 uncensored models under 4B, found {len(uncensored_models)}"
    )

    for model in uncensored_models:
        assert model.is_uncensored is True
        assert model.filename.endswith(".gguf")
        # Ensure parameter count does not exceed 4B
        assert any(
            p in model.parameters for p in ["0.5B", "1.2B", "1.7B", "2B", "3B", "3.8B", "4B"]
        ), f"Model {model.id} parameter count unexpected: {model.parameters}"


def test_registry_tool_calling_support() -> None:
    """Verify that tool-capable models are correctly identified."""
    tool_models = [m for m in ModelRegistry.list_models() if m.supports_tools]
    assert len(tool_models) >= 5

    # Check key models supporting tools
    coder = ModelRegistry.get("qwen2.5-coder-3b-abliterated")
    assert coder is not None
    assert coder.supports_tools is True
    assert coder.is_uncensored is True

    hermes = ModelRegistry.get("hermes-3-llama-3.2-3b")
    assert hermes is not None
    assert hermes.supports_tools is True


def test_registry_path_resolution(tmp_path: Path) -> None:
    """Verify filesystem path resolution for model weights and mmproj.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    m_path, mm_path = ModelRegistry.resolve_paths("qwen2-vl-2b-instruct", base_dir=tmp_path)
    assert m_path == tmp_path / "models" / "llm" / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
    assert mm_path == tmp_path / "models" / "vision" / "mmproj-Qwen2-VL-2B-Instruct-f16.gguf"

    # Non-vision model should return None for mmproj
    m_path2, mm_path2 = ModelRegistry.resolve_paths("llama-3.2-1b-uncensored", base_dir=tmp_path)
    assert m_path2 == tmp_path / "models" / "llm" / "Llama-3.2-1B-Instruct-Uncensored.Q4_K_M.gguf"
    assert mm_path2 is None


def test_registry_unknown_model_raises_keyerror() -> None:
    """Verify resolving paths for an unregistered model raises KeyError."""
    with pytest.raises(KeyError):
        ModelRegistry.resolve_paths("non-existent-model-xyz")


def test_verify_gguf_header(tmp_path: Path) -> None:
    """Verify GGUF binary magic header detection.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    valid_gguf = tmp_path / "test_valid.gguf"
    valid_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

    invalid_gguf = tmp_path / "test_invalid.gguf"
    invalid_gguf.write_bytes(b"NOT_GGUF_HEADER")

    assert ModelRegistry.verify_gguf_header(valid_gguf) is True
    assert ModelRegistry.verify_gguf_header(invalid_gguf) is False
    assert ModelRegistry.verify_gguf_header(tmp_path / "missing.gguf") is False


def test_config_llm_multimodal_and_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify config loading and environment variable overrides for model paths and vision."""
    from robot.config import _apply_env_overrides

    # Test baseline config contains required LLM fields
    cfg = load_system_config()
    llm = cfg.get("llm", {})
    assert "model" in llm
    assert "model_path" in llm
    assert "multimodal" in llm

    # Test environment variable overrides via _apply_env_overrides
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-coder-3b-abliterated")
    monkeypatch.setenv("LLM_MODEL_PATH", "models/llm/custom.gguf")
    monkeypatch.setenv("LLM_MULTIMODAL", "false")
    monkeypatch.setenv("LLM_MMPROJ_PATH", "")

    test_cfg: dict[str, Any] = {}
    _apply_env_overrides(test_cfg)
    overridden_llm = test_cfg.get("llm", {})
    assert overridden_llm.get("model") == "qwen2.5-coder-3b-abliterated"
    assert overridden_llm.get("model_path") == "models/llm/custom.gguf"
    assert overridden_llm.get("multimodal") is False
    assert overridden_llm.get("mmproj_path") == ""


def test_api_local_models_endpoint() -> None:
    """Verify the /api/models/local endpoint returns catalog and download statuses."""
    mock_engine = MagicMock()
    mock_engine.memory = MagicMock()
    mock_engine.telemetry = MagicMock()
    mock_engine.telemetry.snapshot.return_value = {}
    mock_engine.config = {"llm": {"model": "qwen2-vl-2b-instruct"}}
    app = create_app(mock_engine)
    client = TestClient(app)
    response = client.get("/api/models/local")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert "total_models" in data
    assert "default_model" in data
    assert data["default_model"] == "qwen2-vl-2b-instruct"
    assert data["total_models"] >= 10

    # Ensure model metadata is properly serialized
    first_model = next(m for m in data["models"] if m["id"] == "qwen2-vl-2b-instruct")
    assert first_model["is_vision"] is True
    assert first_model["supports_tools"] is True
    assert "is_downloaded" in first_model
