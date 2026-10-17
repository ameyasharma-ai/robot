"""Unit tests for FLUX, Stable Diffusion, GPU Handoff, and LLM Instruction Synthesis.

Verifies the ImageModelRegistry, ImagePromptEnhancer, GpuResourceHandoffManager,
UnifiedImageProvider, and corresponding API and tool handlers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from robot.api import create_app
from robot.image import (
    GpuHandoffState,
    GpuResourceHandoffManager,
    ImageGenerationInstruction,
    ImageGenerationService,
    ImageJob,
    ImageModelFamily,
    ImageModelMetadata,
    ImageModelRegistry,
    ImagePromptEnhancer,
    JobStatus,
    UnifiedImageProvider,
)
from robot.tools.builtins import register_builtin_tools


def test_image_model_registry_catalog_completeness() -> None:
    """Verify registry contains FLUX, Stable Diffusion, SDXL, and Procedural models."""
    models = ImageModelRegistry.list_models()
    assert len(models) >= 8, f"Expected at least 8 models, found {len(models)}"

    # Check FLUX models
    flux_models = ImageModelRegistry.list_flux_models()
    assert len(flux_models) >= 2
    flux_ids = [m.id for m in flux_models]
    assert "flux-schnell" in flux_ids
    assert "flux-dev" in flux_ids

    # Check SD models
    sd_models = ImageModelRegistry.list_sd_models()
    assert len(sd_models) >= 4
    sd_ids = [m.id for m in sd_models]
    assert "sd-turbo" in sd_ids
    assert "small-sd" in sd_ids
    assert "dreamshaper-v8" in sd_ids

    # Check default model resolution
    default_gpu = ImageModelRegistry.get_default_model(gpu_available=True)
    assert default_gpu.id == "sd-turbo"
    assert default_gpu.is_heavy is True

    default_cpu = ImageModelRegistry.get_default_model(gpu_available=False)
    assert default_cpu.id == "small-sd"


def test_image_model_metadata_serialization() -> None:
    """Verify ImageModelMetadata converts to dictionary and checks local cache."""
    meta = ImageModelRegistry.get("sd-turbo")
    assert meta is not None
    d = meta.to_dict()
    assert d["id"] == "sd-turbo"
    assert d["family"] == "stable_diffusion"
    assert d["recommended_steps"] == 4
    assert d["is_heavy"] is True
    assert isinstance(d["vram_mb_required"], int)


@pytest.mark.asyncio
async def test_prompt_enhancer_deterministic_fallback() -> None:
    """Verify ImagePromptEnhancer creates high-quality instructions when LLM is unavailable."""
    enhancer = ImagePromptEnhancer(engine=None)
    instruction = await enhancer.generate_instruction(
        user_query="cyberpunk cat in neon alley",
        target_model="sd-turbo",
        seed=42,
    )
    assert isinstance(instruction, ImageGenerationInstruction)
    assert "cyberpunk cat" in instruction.prompt
    assert "cinematic" in instruction.prompt or "lighting" in instruction.prompt
    assert instruction.seed == 42
    assert instruction.steps == 4
    assert instruction.model_id == "sd-turbo"


@pytest.mark.asyncio
async def test_prompt_enhancer_with_mock_llm() -> None:
    """Verify ImagePromptEnhancer queries the loaded LLM to synthesize JSON instructions."""
    mock_engine = MagicMock()
    mock_llm = MagicMock()

    json_payload = (
        '{"prompt": "A majestic cybernetic owl perched on a chrome branch, volumetric mist, 8k", '
        '"negative_prompt": "blurry, deformed, oversaturated", '
        '"width": 512, "height": 512, "steps": 4, "guidance_scale": 0.0, "seed": 9999}'
    )

    async def _mock_stream(*args: Any, **kwargs: Any):
        yield json_payload

    mock_llm.stream = _mock_stream
    mock_engine.llm = mock_llm

    enhancer = ImagePromptEnhancer(engine=mock_engine)
    instruction = await enhancer.generate_instruction(
        user_query="cybernetic owl",
        target_model="sd-turbo",
    )

    assert instruction is not None
    assert "cybernetic owl" in instruction.prompt
    assert instruction.steps == 4
    assert instruction.seed == 9999


@pytest.mark.asyncio
async def test_gpu_handoff_manager_lifecycle() -> None:
    """Verify GpuResourceHandoffManager offloads local LLM and restores it upon completion."""
    mock_engine = MagicMock()
    mock_engine.llm_provider_type = "local"
    mock_engine.switch_llm_provider = AsyncMock()
    mock_engine.switch_llm_provider.return_value = {"status": "ok"}
    mock_engine.voice_enabled = False
    mock_engine.is_chat_mode = True
    mock_engine.observability = MagicMock()

    handoff = GpuResourceHandoffManager(engine=mock_engine)

    # Patch is_cuda_available to simulate NVIDIA GPU presence
    with patch.object(handoff, "is_cuda_available", return_value=True), \
         patch.object(handoff, "get_free_vram_mb", return_value=3600.0), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_key"}):

        model_meta = ImageModelRegistry.get("sd-turbo")
        assert model_meta is not None

        # 1. Prepare GPU -> Offloads local LLM to Gemini
        prepared = await handoff.prepare_gpu_for_image_model(model_meta)
        assert prepared is True
        assert handoff.state == GpuHandoffState.GPU_CLEARED
        mock_engine.switch_llm_provider.assert_called_with("gemini")

        # 2. Cleanup GPU & Restore LLM -> Switches back to local
        await handoff.cleanup_gpu_and_restore_llm()
        assert handoff.state == GpuHandoffState.COMPLETED
        mock_engine.switch_llm_provider.assert_called_with("local")


@pytest.mark.asyncio
async def test_gpu_handoff_lightweight_model_skips_offload() -> None:
    """Verify lightweight/procedural models do not needlessly offload the local LLM."""
    mock_engine = MagicMock()
    mock_engine.llm_provider_type = "local"
    mock_engine.switch_llm_provider = AsyncMock()

    handoff = GpuResourceHandoffManager(engine=mock_engine)
    proc_meta = ImageModelRegistry.get("fast-procedural")
    assert proc_meta is not None

    prepared = await handoff.prepare_gpu_for_image_model(proc_meta)
    assert prepared is True
    # Should not call switch_llm_provider
    mock_engine.switch_llm_provider.assert_not_called()


@pytest.mark.asyncio
async def test_unified_provider_procedural_generation(tmp_path: Path) -> None:
    """Verify UnifiedImageProvider renders procedural art without errors."""
    mock_engine = MagicMock()
    mock_engine.llm = None
    mock_engine.observability = MagicMock()

    provider = UnifiedImageProvider(output_dir=str(tmp_path), engine=mock_engine)
    job = ImageJob(prompt="ambient crystal garden", model="fast-procedural", width=256, height=256)
    ev = asyncio.Event()

    out_path = await provider.generate(job, ev)
    assert job.status == JobStatus.COMPLETED
    assert job.progress == 100
    assert out_path.endswith(".png")
    assert (tmp_path / Path(out_path).name).exists()


def test_api_image_models_endpoint() -> None:
    """Verify GET /api/images/models returns model catalog and default configuration."""
    mock_engine = MagicMock()
    mock_engine.memory = MagicMock()
    mock_engine.telemetry = MagicMock()
    mock_engine.telemetry.snapshot.return_value = {}

    app = create_app(mock_engine)
    client = TestClient(app)

    response = client.get("/api/images/models")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert "total_models" in data
    assert "default_model" in data
    assert data["total_models"] >= 8
    assert data["default_model"] == "sd-turbo"

    first_model = next(m for m in data["models"] if m["id"] == "sd-turbo")
    assert first_model["family"] == "stable_diffusion"
    assert first_model["is_heavy"] is True


@pytest.mark.asyncio
async def test_image_generate_builtin_tool_registration() -> None:
    """Verify image.generate tool handler submits job with model parameters."""
    mock_engine = MagicMock()
    mock_engine.image_service = MagicMock()
    mock_job = ImageJob(job_id="test_job_123", prompt="cosmic galaxy")
    mock_engine.image_service.submit_job.return_value = mock_job
    mock_engine.tool_registry = MagicMock()

    registered_tools: dict[str, Any] = {}

    def _register(tool_def: Any) -> None:
        registered_tools[tool_def.name] = tool_def

    mock_engine.tool_registry.register = _register

    register_builtin_tools(mock_engine)

    assert "image.generate" in registered_tools
    tool_def = registered_tools["image.generate"]
    assert "properties" in tool_def.parameters
    assert "model" in tool_def.parameters["properties"]
    assert "negative_prompt" in tool_def.parameters["properties"]

    # Execute handler
    result = await tool_def.handler(prompt="cosmic galaxy", model="sd-turbo", steps=4)
    assert result["status"] == "queued"
    assert result["job_id"] == "test_job_123"
    assert result["model"] == "sd-turbo"
    mock_engine.image_service.submit_job.assert_called_once()
