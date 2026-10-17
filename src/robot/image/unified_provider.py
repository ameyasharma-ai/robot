"""Unified multi-model image generation provider supporting FLUX, Stable Diffusion, and Procedural art.

Architectural Rationale:
- Coordinates multi-model diffusion inference across FLUX.1 (Schnell/Dev), SDXL,
  Stable Diffusion 1.5, SD-Turbo, Small-SD, and Fast-Procedural.
- Integrates with GpuResourceHandoffManager to execute the dynamic VRAM arbitration
  pipeline on the RTX 3050 (4GB):
    1. LLM Prompt Enhancement ->
    2. GPU VRAM Clearance / LLM Offload ->
    3. Pipeline Loading with Memory Optimizations ->
    4. Image Synthesis ->
    5. GPU Clearance & LLM Restoration ->
    6. LLM Spoken/Chat Notification.
"""

from __future__ import annotations

import asyncio
import gc
import os
import random
import time
from pathlib import Path
from typing import Any

from robot.image import FastProceduralImageProvider, ImageJob, ImageProvider, JobStatus
from robot.image.gpu_handoff import GpuResourceHandoffManager
from robot.image.model_registry import ImageModelFamily, ImageModelMetadata, ImageModelRegistry
from robot.image.prompt_enhancer import ImageGenerationInstruction, ImagePromptEnhancer
from robot.telemetry import get_logger

logger = get_logger("image.unified_provider")


class UnifiedImageProvider(ImageProvider):
    """Multi-architecture image synthesis provider with GPU handoff and active LLM integration."""

    def __init__(
        self,
        output_dir: str = "data/images",
        engine: Any | None = None,
        default_model_id: str | None = None,
    ) -> None:
        """Initialize unified image provider.

        Args:
            output_dir: Path to directory where generated images are persisted.
            engine: Optional reference to RobotEngine for LLM arbitration and events.
            default_model_id: Optional default model identifier.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.handoff_manager = GpuResourceHandoffManager(engine=engine)
        self.prompt_enhancer = ImagePromptEnhancer(engine=engine)
        self.procedural_provider = FastProceduralImageProvider(output_dir=str(self.output_dir))

        gpu_ok = self.handoff_manager.is_cuda_available()
        default_meta = ImageModelRegistry.get_default_model(gpu_available=gpu_ok)
        self.default_model_id = default_model_id or default_meta.id
        self._current_loaded_model_id: str | None = None
        self._pipe: Any | None = None

    def list_models(self) -> list[str]:
        """List all supported model identifiers.

        Returns:
            list[str]: Registered model keys.
        """
        return list(ImageModelRegistry.CATALOG.keys())

    async def health_check(self) -> bool:
        """Verify provider availability.

        Returns:
            bool: Always True (procedural fallback guarantees availability).
        """
        return True

    async def generate(self, job: ImageJob, interrupt: asyncio.Event) -> str:
        """Execute end-to-end image generation with LLM instruction synthesis and GPU handoff.

        Args:
            job: Image generation job specification.
            interrupt: Event signal to abort execution if requested by user or barge-in.

        Returns:
            str: Relative web URL path to generated image file.

        Raises:
            asyncio.CancelledError: If generation was cancelled.
            RuntimeError: If generation failed completely across all fallbacks.
        """
        # Resolve target model metadata
        requested_model = job.model if (job.model and job.model != "default") else self.default_model_id
        model_meta = ImageModelRegistry.get(requested_model) or ImageModelRegistry.get(self.default_model_id) or ImageModelRegistry.get("fast-procedural")

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.model = model_meta.id
        job.progress = 5

        filename = f"img_{job.job_id}_{int(time.time())}.png"
        full_path = self.output_dir / filename

        # Step 1: Synthesize complete visual instruction via loaded LLM
        instruction = await self.prompt_enhancer.generate_instruction(
            user_query=job.prompt,
            target_model=model_meta.id,
            width=job.width,
            height=job.height,
            steps=job.steps,
            seed=job.seed,
        )
        job.prompt = instruction.prompt
        job.negative_prompt = instruction.negative_prompt
        job.width = instruction.width
        job.height = instruction.height
        job.steps = instruction.steps
        job.seed = instruction.seed
        job.progress = 15

        # Step 2: Handle procedural art models immediately on CPU
        if model_meta.family == ImageModelFamily.PROCEDURAL:
            job.backend = "cpu_procedural"
            job.device = "Host CPU"
            out = await self.procedural_provider.generate(job, interrupt)
            await self.handoff_manager.notify_user_via_loaded_llm(instruction, out, job.duration_s)
            return out

        # Step 3: Heavy Model GPU Preparation & LLM Offload
        gpu_ready = await self.handoff_manager.prepare_gpu_for_image_model(model_meta)
        job.progress = 25

        # Check for user cancellation before heavy inference
        if interrupt.is_set():
            await self.handoff_manager.cleanup_gpu_and_restore_llm()
            job.status = JobStatus.CANCELLED
            job.error = "Interrupted by user or voice barge-in"
            raise asyncio.CancelledError("Image generation cancelled")

        # Step 4: Execute Diffusion Pipeline in dedicated thread
        try:
            out_path = await asyncio.to_thread(self._run_diffusion_inference, model_meta, instruction, full_path, job, interrupt)
        except Exception as diff_err:
            logger.warning("diffusion_inference_failed_falling_back_to_procedural", error=str(diff_err))
            # Clean up GPU before fallback
            await self.handoff_manager.cleanup_gpu_and_restore_llm()
            # Fall back to procedural provider to guarantee user receives an artwork
            job.backend = "procedural_fallback"
            job.device = "Host CPU (Fallback)"
            out_path = await self.procedural_provider.generate(job, interrupt)

        # Step 5: GPU Cleanup & LLM Restoration
        await self.handoff_manager.cleanup_gpu_and_restore_llm()
        job.progress = 100
        job.completed_at = time.time()
        job.duration_s = job.completed_at - job.started_at
        job.output_path = f"/data/images/{filename}"
        job.status = JobStatus.COMPLETED

        # Step 6: Notify user through whatever LLM is currently loaded
        await self.handoff_manager.notify_user_via_loaded_llm(instruction, job.output_path, job.duration_s)

        return job.output_path

    def _run_diffusion_inference(
        self,
        meta: ImageModelMetadata,
        instruction: ImageGenerationInstruction,
        full_path: Path,
        job: ImageJob,
        interrupt: asyncio.Event,
    ) -> str:
        """Synchronous worker thread executing diffusers pipeline.

        Args:
            meta: Model metadata.
            instruction: Synthesized generation parameters.
            full_path: Target disk filepath.
            job: Active ImageJob.
            interrupt: Cancellation event.

        Returns:
            str: Web-accessible output path.
        """
        import torch

        cuda_ok = self.handoff_manager.is_cuda_available()
        device_str = "cuda" if (cuda_ok and meta.is_heavy) else "cpu"
        torch_dtype = torch.float16 if device_str == "cuda" else torch.float32

        job.backend = f"diffusers_{meta.family.value}"
        job.device = f"NVIDIA RTX 3050 ({torch_dtype})" if device_str == "cuda" else "Host CPU"
        logger.info(
            "loading_diffusion_pipeline",
            model=meta.id,
            repo_id=meta.repo_id,
            device=job.device,
            steps=instruction.steps,
        )

        pipe = None
        try:
            pipe = self._build_pipeline(meta, device_str, torch_dtype)
            job.progress = 45

            if interrupt.is_set():
                raise InterruptedError("Cancelled by user")

            # Setup seed generator
            gen = None
            if instruction.seed >= 0:
                gen = torch.Generator(device=device_str).manual_seed(instruction.seed)

            # Step callback for live progress updates
            def _step_callback(step: int, timestep: int, latents: Any) -> None:
                if interrupt.is_set():
                    raise InterruptedError("Cancelled by user")
                pct = 45 + int((step / max(1, instruction.steps)) * 50)
                job.progress = min(95, pct)

            # Build generation kwargs
            gen_kwargs: dict[str, Any] = {
                "prompt": instruction.prompt,
                "num_inference_steps": instruction.steps,
                "width": instruction.width,
                "height": instruction.height,
                "generator": gen,
            }
            if meta.supports_negative_prompt and instruction.negative_prompt:
                gen_kwargs["negative_prompt"] = instruction.negative_prompt
            if instruction.guidance_scale > 0.0:
                gen_kwargs["guidance_scale"] = instruction.guidance_scale

            # Some pipelines support callback, others callback_on_step_end
            if hasattr(pipe, "callback"):
                gen_kwargs["callback"] = _step_callback
                gen_kwargs["callback_steps"] = 1

            result = pipe(**gen_kwargs)
            image = result.images[0]
            image.save(str(full_path), format="PNG")
            logger.info("diffusion_image_saved", path=str(full_path))

        finally:
            # Free pipeline memory immediately
            if pipe is not None:
                del pipe
            gc.collect()
            if cuda_ok:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return f"/data/images/{full_path.name}"

    def _build_pipeline(
        self,
        meta: ImageModelMetadata,
        device_str: str,
        torch_dtype: Any,
    ) -> Any:
        """Instantiate and optimize diffusers pipeline based on model architecture.

        Args:
            meta: Target model metadata.
            device_str: Target device string ('cuda' or 'cpu').
            torch_dtype: Precision dtype (torch.float16 or torch.float32).

        Returns:
            Any: Configured diffusers pipeline.
        """
        import torch

        # ─── 1. FLUX Pipeline ───────────────────────────────────────────────────
        if meta.family == ImageModelFamily.FLUX:
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                meta.repo_id,
                torch_dtype=torch_dtype,
            )
            if device_str == "cuda":
                # For FLUX on 4GB VRAM, enable sequential CPU offload
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                elif hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to(device_str)
            else:
                pipe = pipe.to(device_str)
            return pipe

        # ─── 2. SDXL & SDXL-Lightning ───────────────────────────────────────────
        if meta.family == ImageModelFamily.SDXL:
            from diffusers import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(
                meta.repo_id,
                torch_dtype=torch_dtype,
                variant="fp16" if device_str == "cuda" else None,
            )
            pipe = pipe.to(device_str)
            if device_str == "cuda" and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            return pipe

        # ─── 3. Stable Diffusion 1.5 & SD-Turbo ──────────────────────────────────
        from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

        pipe = AutoPipelineForText2Image.from_pretrained(
            meta.repo_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, steps_offset=1)
        except Exception:
            pass

        pipe = pipe.to(device_str)
        if device_str == "cuda" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        return pipe
