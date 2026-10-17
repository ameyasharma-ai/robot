"""Image generation model catalog and registry for FLUX, Stable Diffusion, and Procedural models.

Architectural Rationale:
- NVIDIA RTX 3050 Laptop GPU has 4GB VRAM.
- Heavy models (FLUX.1 Schnell/Dev, SDXL-Turbo, SD 1.5, DreamShaper) require dedicated GPU VRAM.
- To prevent CUDA Out Of Memory (OOM) errors, heavy models trigger the GPU handoff pipeline:
  temporarily offloading the local LLM (llama-server) to Cloud API (Gemini / OpenAI-Compatible)
  and clearing CUDA cache before loading diffusion pipelines.
- Lightweight models (Small-SD, Fast-Procedural) can run on CPU or low-memory without LLM offload.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ImageModelFamily(str, Enum):
    """Categorical family classification of image synthesis models."""

    FLUX = "flux"
    STABLE_DIFFUSION = "stable_diffusion"
    SDXL = "sdxl"
    PROCEDURAL = "procedural"


@dataclass
class ImageModelMetadata:
    """Metadata specification for image generation models and pipeline requirements."""

    id: str
    name: str
    family: ImageModelFamily
    repo_id: str
    is_heavy: bool
    recommended_steps: int
    default_guidance_scale: float
    default_width: int
    default_height: int
    supports_negative_prompt: bool
    vram_mb_required: int
    pipeline_type: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary.

        Returns:
            dict[str, Any]: JSON-serializable model metadata.
        """
        data = asdict(self)
        data["family"] = self.family.value
        return data

    def is_cached_locally(self, cache_dir: Path | None = None) -> bool:
        """Check if model checkpoint files exist in local Hugging Face or models cache.

        Args:
            cache_dir: Optional root cache directory.

        Returns:
            bool: True if model is cached locally on disk.
        """
        if self.family == ImageModelFamily.PROCEDURAL:
            return True

        # Check Hugging Face hub cache
        hf_cache = cache_dir or Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
        repo_slug = "--".join(["models"] + self.repo_id.split("/"))
        target_dir = hf_cache / repo_slug
        if target_dir.exists() and any(target_dir.glob("snapshots/*")):
            return True

        # Check local models directory
        local_dir = Path("models") / "image" / self.id
        if local_dir.exists() and any(local_dir.iterdir()):
            return True

        return False


class ImageModelRegistry:
    """Central registry and capability directory for image generation models."""

    DEFAULT_GPU_MODEL_ID = "sd-turbo"
    DEFAULT_CPU_MODEL_ID = "small-sd"

    CATALOG: dict[str, ImageModelMetadata] = {
        # ─── FLUX Models ──────────────────────────────────────────────────────────
        "flux-schnell": ImageModelMetadata(
            id="flux-schnell",
            name="FLUX.1 Schnell",
            family=ImageModelFamily.FLUX,
            repo_id="black-forest-labs/FLUX.1-schnell",
            is_heavy=True,
            recommended_steps=4,
            default_guidance_scale=0.0,
            default_width=512,
            default_height=512,
            supports_negative_prompt=False,
            vram_mb_required=3800,
            pipeline_type="FluxPipeline",
            description="State-of-the-art 12B rectified flow transformer distilled for 4-step ultra-fidelity rendering.",
        ),
        "flux-dev": ImageModelMetadata(
            id="flux-dev",
            name="FLUX.1 Dev",
            family=ImageModelFamily.FLUX,
            repo_id="black-forest-labs/FLUX.1-dev",
            is_heavy=True,
            recommended_steps=24,
            default_guidance_scale=3.5,
            default_width=768,
            default_height=768,
            supports_negative_prompt=False,
            vram_mb_required=3900,
            pipeline_type="FluxPipeline",
            description="Guidance-distilled FLUX model for premier typography, photorealism, and complex scene composition.",
        ),
        "flux-schnell-gguf": ImageModelMetadata(
            id="flux-schnell-gguf",
            name="FLUX.1 Schnell GGUF",
            family=ImageModelFamily.FLUX,
            repo_id="city96/FLUX.1-schnell-gguf",
            is_heavy=True,
            recommended_steps=4,
            default_guidance_scale=0.0,
            default_width=512,
            default_height=512,
            supports_negative_prompt=False,
            vram_mb_required=3700,
            pipeline_type="FluxPipeline",
            description="4-bit quantized GGUF FLUX model for ultra-low memory visual generation on consumer GPUs.",
        ),

        # ─── Stable Diffusion Turbo & SDXL ─────────────────────────────────────────
        "sd-turbo": ImageModelMetadata(
            id="sd-turbo",
            name="Stable Diffusion Turbo",
            family=ImageModelFamily.STABLE_DIFFUSION,
            repo_id="stabilityai/sd-turbo",
            is_heavy=True,
            recommended_steps=4,
            default_guidance_scale=0.0,
            default_width=512,
            default_height=512,
            supports_negative_prompt=False,
            vram_mb_required=2600,
            pipeline_type="AutoPipelineForText2Image",
            description="Adversarial Diffusion Distillation (ADD) single-to-four step real-time generation on RTX 3050.",
        ),
        "sdxl-turbo": ImageModelMetadata(
            id="sdxl-turbo",
            name="SDXL Turbo",
            family=ImageModelFamily.SDXL,
            repo_id="stabilityai/sdxl-turbo",
            is_heavy=True,
            recommended_steps=4,
            default_guidance_scale=0.0,
            default_width=512,
            default_height=512,
            supports_negative_prompt=False,
            vram_mb_required=3600,
            pipeline_type="AutoPipelineForText2Image",
            description="High-resolution SDXL architecture distilled for rapid 1-4 step rendering.",
        ),
        "sdxl-lightning": ImageModelMetadata(
            id="sdxl-lightning",
            name="SDXL Lightning 4-Step",
            family=ImageModelFamily.SDXL,
            repo_id="ByteDance/SDXL-Lightning",
            is_heavy=True,
            recommended_steps=4,
            default_guidance_scale=0.0,
            default_width=1024,
            default_height=1024,
            supports_negative_prompt=False,
            vram_mb_required=3700,
            pipeline_type="StableDiffusionXLPipeline",
            description="ByteDance progressive adversarial distillation model generating 1024x1024 visuals in 4 steps.",
        ),

        # ─── Stable Diffusion 1.5 & Checkpoints ────────────────────────────────────
        "sd-1.5": ImageModelMetadata(
            id="sd-1.5",
            name="Stable Diffusion 1.5",
            family=ImageModelFamily.STABLE_DIFFUSION,
            repo_id="runwayml/stable-diffusion-v1-5",
            is_heavy=True,
            recommended_steps=18,
            default_guidance_scale=7.5,
            default_width=512,
            default_height=512,
            supports_negative_prompt=True,
            vram_mb_required=3200,
            pipeline_type="StableDiffusionPipeline",
            description="Universal standard Stable Diffusion v1.5 with full classifier-free guidance and negative prompts.",
        ),
        "dreamshaper-v8": ImageModelMetadata(
            id="dreamshaper-v8",
            name="DreamShaper v8",
            family=ImageModelFamily.STABLE_DIFFUSION,
            repo_id="Lykon/dreamshaper-8",
            is_heavy=True,
            recommended_steps=20,
            default_guidance_scale=7.5,
            default_width=512,
            default_height=512,
            supports_negative_prompt=True,
            vram_mb_required=3200,
            pipeline_type="StableDiffusionPipeline",
            description="Fine-tuned artistic and photorealistic community checkpoint optimized for cinematic composition.",
        ),
        "small-sd": ImageModelMetadata(
            id="small-sd",
            name="Small Stable Diffusion",
            family=ImageModelFamily.STABLE_DIFFUSION,
            repo_id="OFA-Sys/small-stable-diffusion-v0",
            is_heavy=False,
            recommended_steps=15,
            default_guidance_scale=7.0,
            default_width=512,
            default_height=512,
            supports_negative_prompt=True,
            vram_mb_required=1400,
            pipeline_type="StableDiffusionPipeline",
            description="Lightweight ~1GB diffusion model optimized for CPU or low-memory GPU fallback.",
        ),

        # ─── Procedural Fast Art ──────────────────────────────────────────────────
        "fast-procedural": ImageModelMetadata(
            id="fast-procedural",
            name="Fast Procedural Vector Art",
            family=ImageModelFamily.PROCEDURAL,
            repo_id="local/procedural",
            is_heavy=False,
            recommended_steps=4,
            default_guidance_scale=0.0,
            default_width=512,
            default_height=512,
            supports_negative_prompt=False,
            vram_mb_required=0,
            pipeline_type="FastProceduralImageProvider",
            description="Zero-VRAM CPU geometric card generator providing instant fallback without memory allocation.",
        ),
    }

    @classmethod
    def get(cls, model_id: str) -> ImageModelMetadata | None:
        """Retrieve image model metadata by identifier.

        Args:
            model_id: Model identifier key (case-insensitive).

        Returns:
            ImageModelMetadata | None: Matched metadata or None if not found.
        """
        clean_id = model_id.lower().strip()
        return cls.CATALOG.get(clean_id)

    @classmethod
    def list_models(cls, family: ImageModelFamily | None = None) -> list[ImageModelMetadata]:
        """List registered image models, optionally filtered by family.

        Args:
            family: Optional model family filter.

        Returns:
            list[ImageModelMetadata]: Registered model specifications.
        """
        if family is None:
            return list(cls.CATALOG.values())
        return [m for m in cls.CATALOG.values() if m.family == family]

    @classmethod
    def list_heavy_models(cls) -> list[ImageModelMetadata]:
        """List models that require dedicated GPU allocation and LLM offload.

        Returns:
            list[ImageModelMetadata]: Heavy models requiring GPU handoff.
        """
        return [m for m in cls.CATALOG.values() if m.is_heavy]

    @classmethod
    def list_flux_models(cls) -> list[ImageModelMetadata]:
        """List all FLUX architecture models.

        Returns:
            list[ImageModelMetadata]: Registered FLUX models.
        """
        return cls.list_models(ImageModelFamily.FLUX)

    @classmethod
    def list_sd_models(cls) -> list[ImageModelMetadata]:
        """List all Stable Diffusion and SDXL models.

        Returns:
            list[ImageModelMetadata]: Registered Stable Diffusion models.
        """
        return [m for m in cls.CATALOG.values() if m.family in (ImageModelFamily.STABLE_DIFFUSION, ImageModelFamily.SDXL)]

    @classmethod
    def get_default_model(cls, gpu_available: bool = True) -> ImageModelMetadata:
        """Return the default recommended model based on hardware availability.

        Args:
            gpu_available: Whether an NVIDIA GPU with CUDA is detected.

        Returns:
            ImageModelMetadata: Recommended model metadata.
        """
        if gpu_available:
            return cls.CATALOG[cls.DEFAULT_GPU_MODEL_ID]
        return cls.CATALOG[cls.DEFAULT_CPU_MODEL_ID]
