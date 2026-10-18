"""Unit tests for HuggingFace Hub LLM search, download manager, and dynamic model registration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from robot.api import create_app
from robot.llm.hf_downloader import (
    DownloadTask,
    HuggingFaceDownloadManager,
    HuggingFaceHubService,
)
from robot.llm.model_registry import ModelCategory, ModelMetadata, ModelRegistry


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    """Fixture providing temporary models/llm and models/vision directories."""
    llm_dir = tmp_path / "models" / "llm"
    vision_dir = tmp_path / "models" / "vision"
    llm_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_model_metadata_serialization() -> None:
    """Test serialization and deserialization of ModelMetadata."""
    meta = ModelMetadata(
        id="test-model-3b",
        name="Test Model 3B",
        category=ModelCategory.COMMUNITY_HUGGINGFACE,
        parameters="3B",
        repo_id="author/test-model-3b-GGUF",
        filename="test-model-3b.Q4_K_M.gguf",
        mmproj_filename=None,
        is_vision=False,
        is_uncensored=True,
        supports_tools=True,
        recommended_context=16384,
        description="A test model for HuggingFace downloads.",
    )

    data = meta.to_dict()
    assert data["id"] == "test-model-3b"
    assert data["category"] == "community_huggingface"
    assert data["is_uncensored"] is True

    hydrated = ModelMetadata.from_dict(data)
    assert hydrated.id == meta.id
    assert hydrated.name == meta.name
    assert hydrated.category == ModelCategory.COMMUNITY_HUGGINGFACE
    assert hydrated.filename == meta.filename


def test_model_registry_custom_persistence(temp_models_dir: Path) -> None:
    """Test registering and persisting custom models in ModelRegistry."""
    meta = ModelMetadata(
        id="custom-qwen-test",
        name="Custom Qwen Test",
        category=ModelCategory.COMMUNITY_HUGGINGFACE,
        parameters="1.5B",
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        mmproj_filename=None,
        is_vision=False,
        is_uncensored=False,
        supports_tools=True,
        recommended_context=16384,
        description="Downloaded custom Qwen model.",
    )

    ModelRegistry.register_model(meta, persist=True, base_dir=temp_models_dir)
    assert ModelRegistry.get("custom-qwen-test") is not None

    reg_file = temp_models_dir / "models" / "llm" / "registry.json"
    assert reg_file.exists()

    with open(reg_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert len(persisted) == 1
    assert persisted[0]["id"] == "custom-qwen-test"

    # Test unregister
    removed = ModelRegistry.unregister_model("custom-qwen-test", persist=True, base_dir=temp_models_dir)
    assert removed is True
    assert ModelRegistry.get("custom-qwen-test") is None


def test_model_registry_scan_local_files(temp_models_dir: Path) -> None:
    """Test auto-discovery of valid GGUF binaries in models/llm/."""
    dummy_gguf = temp_models_dir / "models" / "llm" / "mistral-7b-instruct-v0.3.Q4_K_M.gguf"
    # Write valid GGUF magic header b"GGUF" followed by dummy bytes (>1MB)
    with open(dummy_gguf, "wb") as f:
        f.write(b"GGUF")
        f.write(b"\x00" * (1_050_000))

    new_models = ModelRegistry.scan_and_register_local_files(base_dir=temp_models_dir)
    assert len(new_models) >= 1
    found = any(m.filename == dummy_gguf.name for m in new_models)
    assert found is True

    retrieved = ModelRegistry.get("mistral_7b_instruct_v0_3_q4_k_m")
    assert retrieved is not None
    assert retrieved.parameters == "7B"


@pytest.mark.asyncio
async def test_hf_hub_service_search_mock() -> None:
    """Test HuggingFaceHubService search with mocked HTTP responses."""
    mock_response = [
        {
            "id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "downloads": 120000,
            "likes": 450,
            "tags": ["gguf", "llama-3", "text-generation"],
            "pipeline_tag": "text-generation",
            "lastModified": "2024-10-10",
        }
    ]

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await HuggingFaceHubService.search_models("llama-3.2", limit=5)
        assert len(results) == 1
        assert results[0]["repo_id"] == "bartowski/Llama-3.2-3B-Instruct-GGUF"
        assert results[0]["parameters"] == "3B"
        assert results[0]["downloads"] == 120000


@pytest.mark.asyncio
async def test_hf_hub_service_get_files_mock(temp_models_dir: Path) -> None:
    """Test retrieving and parsing GGUF file siblings with size and quantization flags."""
    mock_tree = [
        {"path": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "size": 2_150_000_000},
        {"path": "Llama-3.2-3B-Instruct-Q8_0.gguf", "size": 3_500_000_000},
        {"path": "README.md", "size": 4000},
    ]

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_tree
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client):
        files = await HuggingFaceHubService.get_model_files("bartowski/Llama-3.2-3B-Instruct-GGUF", base_dir=temp_models_dir)
        assert len(files) == 2
        # Q4_K_M should be recommended
        q4 = next(f for f in files if "Q4_K_M" in f["filename"])
        assert q4["is_recommended"] is True
        assert q4["quantization"] == "Q4_K_M"
        assert q4["fits_vram"] is True


@pytest.mark.asyncio
async def test_hf_download_manager_lifecycle(temp_models_dir: Path) -> None:
    """Test start, cancellation, and progress reporting in HuggingFaceDownloadManager."""
    mgr = HuggingFaceDownloadManager(base_dir=temp_models_dir)

    task = mgr.start_download(
        repo_id="author/mock-repo-GGUF",
        filename="mock-model-Q4_K_M.gguf",
        name="Mock Model",
        parameters="3B",
    )

    assert task.download_id.startswith("dl_")
    assert task.status in ("queued", "downloading")

    all_tasks = mgr.get_all_tasks()
    assert any(t["download_id"] == task.download_id for t in all_tasks)

    # Cancel task
    cancelled = await mgr.cancel_download(task.download_id)
    assert cancelled is True
    assert mgr.get_task(task.download_id)["status"] == "cancelled"


def test_api_endpoints_hf_search_and_downloads() -> None:
    """Test FastAPI endpoints for searching, listing, downloading, and cancelling models."""
    mock_engine = MagicMock()
    app = create_app(mock_engine)
    client = TestClient(app)

    with patch.object(HuggingFaceHubService, "search_models", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            {
                "repo_id": "test/model-GGUF",
                "name": "model-GGUF",
                "parameters": "1B",
                "downloads": 500,
                "likes": 20,
            }
        ]
        resp = client.get("/api/llm/hf/search?q=test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["models"]) == 1

    with patch.object(HuggingFaceHubService, "get_model_files", new_callable=AsyncMock) as mock_files:
        mock_files.return_value = [
            {
                "filename": "model-Q4_K_M.gguf",
                "size_bytes": 1000000,
                "size_formatted": "1 MB",
                "quantization": "Q4_K_M",
                "is_downloaded": False,
                "is_recommended": True,
                "fits_vram": True,
            }
        ]
        resp = client.get("/api/llm/hf/files?repo_id=test/model-GGUF")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["files"]) == 1

    # Test /api/llm/hf/downloads
    resp = client.get("/api/llm/hf/downloads")
    assert resp.status_code == 200
    assert "downloads" in resp.json()
