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
            job.error = f"Diffusion unavailable: {diff_err}"
            out_path = await self.procedural_provider.generate(job, interrupt)

        # Step 5: GPU Cleanup & LLM Restoration
        await self.handoff_manager.cleanup_gpu_and_restore_llm()
        job.progress = 100
        job.completed_at = time.time()
        job.duration_s = job.completed_at - job.started_at
        job.output_path = out_path
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
        """Execute diffusers image generation synchronously inside worker thread.

        Args:
            meta: Model metadata and resource specifications.
            instruction: Synthesized generation parameters.
            full_path: Target disk filepath for rendered PNG.
            job: Active ImageJob tracked for observability.
            interrupt: Cancellation event for early abort.

        Returns:
            str: Web-accessible output path.

        Raises:
            InterruptedError: If user cancels job or voice barge-in triggers.
            RuntimeError: If diffusers pipeline fails to execute.
        """
        import inspect
        import torch

        cuda_ok = self.handoff_manager.is_cuda_available()
        free_mb = self.handoff_manager.get_free_vram_mb() if cuda_ok else 0
        # Allocate to CUDA if available and free VRAM exceeds model threshold
        use_cuda = cuda_ok and (meta.is_heavy or free_mb >= meta.vram_mb_required)
        device_str = "cuda" if use_cuda else "cpu"
        torch_dtype = torch.float16 if device_str == "cuda" else torch.float32
        # ─── High-Efficiency GGUF Path via stable-diffusion.cpp ───────────────
        sd_cli = Path("/home/ameya/stable-diffusion.cpp/build/bin/sd-cli")
        flux_dir = Path("models/image/flux")
        if (
            meta.family == ImageModelFamily.FLUX
            and sd_cli.exists()
            and (flux_dir / "flux1-schnell-q2_k.gguf").exists()
        ):
            return self._run_sd_cpp_inference(meta, instruction, full_path, job, interrupt, sd_cli, flux_dir)

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

            # Setup seed generator (CPU generator is universally compatible with offload)
            gen = None
            if instruction.seed >= 0:
                gen = torch.Generator(device="cpu").manual_seed(instruction.seed)

            # Step callback for live progress updates across diffusers versions
            def _step_callback(step: int, timestep: int, latents: Any) -> None:
                """Report step progress for legacy diffusers pipelines."""
                if interrupt.is_set():
                    raise InterruptedError("Cancelled by user")
                pct = 45 + int(((step + 1) / max(1, instruction.steps)) * 50)
                job.progress = min(95, pct)

            def _step_end_callback(
                pipe_self: Any,
                step_idx: int,
                timestep: Any,
                callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                """Report step progress for modern diffusers pipelines."""
                if interrupt.is_set():
                    raise InterruptedError("Cancelled by user")
                pct = 45 + int(((step_idx + 1) / max(1, instruction.steps)) * 50)
                job.progress = min(95, pct)
                return callback_kwargs

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

            # Bind step callback dynamically based on pipeline signature
            try:
                call_params = inspect.signature(pipe.__call__).parameters
                if "callback_on_step_end" in call_params:
                    gen_kwargs["callback_on_step_end"] = _step_end_callback
                elif "callback" in call_params:
                    gen_kwargs["callback"] = _step_callback
                    gen_kwargs["callback_steps"] = 1
            except Exception as cb_err:
                logger.debug("callback_binding_skipped", error=str(cb_err))

            result = pipe(**gen_kwargs)
            image = result.images[0]
            image.save(str(full_path), format="PNG")
            logger.info("diffusion_image_saved", path=str(full_path))

        finally:
            # Free pipeline memory immediately to release RTX 3050 VRAM
            if pipe is not None:
                del pipe
            gc.collect()
            if cuda_ok:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return f"/data/images/{full_path.name}"

    def _run_sd_cpp_inference(
        self,
        meta: ImageModelMetadata,
        instruction: ImageGenerationInstruction,
        full_path: Path,
        job: ImageJob,
        interrupt: asyncio.Event,
        sd_cli: Path,
        flux_dir: Path,
    ) -> str:
        """Execute high-efficiency GGUF diffusion inference via stable-diffusion.cpp.

        Architectural Rationale:
            - NVIDIA RTX 3050 Laptop has 4GB VRAM and host has 14GB RAM with 0B swap.
            - Standard Diffusers PyTorch pipelines require >12GB RAM/VRAM even when quantized with bitsandbytes.
            - stable-diffusion.cpp with GGUF architecture enables hybrid CPU+CUDA execution:
                * Diffusion model (3.83GB Q2_K) runs on NVIDIA RTX 3050 CUDA cores.
                * T5-XXL text encoder (2.14GB Q3_K_M) runs on host Ryzen 7 CPU via AVX2 (--clip-on-cpu).
                * CLIP-L (0.23GB) runs on CPU.
                * VAE (0.31GB) handles latent decoding with micro-tiling.
            - Total memory fits comfortably within 3.8GB VRAM and 2.5GB RAM.

        Args:
            meta: Model metadata specification.
            instruction: Target generation parameters.
            full_path: Disk destination for rendered PNG.
            job: Observable tracking job.
            interrupt: Cancellation event for early abort.
            sd_cli: Path to compiled sd-cli binary.
            flux_dir: Root directory of GGUF model files.

        Returns:
            str: Web-accessible PNG URL.

        Raises:
            InterruptedError: If user cancels job or voice barge-in triggers.
            RuntimeError: If sd-cli process execution returns non-zero.
        """
        import subprocess

        job.backend = "stable-diffusion.cpp (GGUF)"
        job.device = "NVIDIA RTX 3050 (CUDA) + CPU RAM"
        job.progress = 30

        cmd: list[str] = [
            str(sd_cli),
            "--diffusion-model", str(flux_dir / "flux1-schnell-q2_k.gguf"),
            "--t5xxl", str(flux_dir / "t5-v1_1-xxl-encoder-Q3_K_M.gguf"),
            "--clip_l", str(flux_dir / "clip_l.safetensors"),
            "--vae", str(flux_dir / "ae.safetensors"),
            "--clip-on-cpu",
            "--offload-to-cpu",
            "--cfg-scale", "1.0",
            "--sampling-method", "euler",
            "--steps", str(instruction.steps or meta.recommended_steps),
            "-W", str(instruction.width),
            "-H", str(instruction.height),
            "-p", instruction.prompt,
            "-o", str(full_path),
        ]
        if instruction.seed >= 0:
            cmd.extend(["-s", str(instruction.seed)])

        logger.info(
            "executing_sd_cpp_inference",
            model=meta.id,
            device=job.device,
            steps=instruction.steps,
            width=instruction.width,
            height=instruction.height,
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in iter(proc.stdout.readline, ""):
                if interrupt.is_set():
                    proc.terminate()
                    raise InterruptedError("Cancelled by user")
                if "/4" in line or "/steps" in line or "sampling completed" in line:
                    job.progress = min(95, job.progress + 15)
            proc.stdout.close()
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"sd-cli process exited with code {returncode}")
        except Exception:
            if proc.poll() is None:
                proc.kill()
            raise

        job.progress = 100
        logger.info("sd_cpp_image_saved", path=str(full_path))
        return f"/data/images/{full_path.name}"

    def _build_pipeline(
        self,
        meta: ImageModelMetadata,
        device_str: str,
        torch_dtype: Any,
    ) -> Any:
        """Instantiate and optimize diffusers pipeline based on model architecture.

        Architectural Rationale:
            - NVIDIA RTX 3050 Laptop has 4GB VRAM.
            - SDXL UNet alone requires ~2.6GB FP16 parameters; full model with text encoders
              and VAE exceeds 6GB, risking CUDA OOM if loaded into VRAM all at once.
            - `enable_sequential_cpu_offload()` swaps modules layer-by-layer during inference,
              enabling SDXL and FLUX to execute smoothly within 4GB VRAM.

        Args:
            meta: Target model metadata.
            device_str: Target device string ('cuda' or 'cpu').
            torch_dtype: Precision dtype (torch.float16 or torch.float32).

        Returns:
            Any: Configured diffusers pipeline.

        Raises:
            RuntimeError: If model pipeline instantiation fails.
        """
        import torch

        # ─── 1. FLUX Pipeline ───────────────────────────────────────────────────
        if meta.family == ImageModelFamily.FLUX:
            try:
                from diffusers import FluxPipeline
            except ImportError as err:
                raise RuntimeError(
                    f"FLUX pipeline requires diffusers >= 0.30.0: {err}"
                ) from err

            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            flux_dtype = torch.bfloat16 if (device_str == "cuda" and torch.cuda.is_bf16_supported()) else torch_dtype
            try:
                pipe = FluxPipeline.from_pretrained(
                    meta.repo_id,
                    torch_dtype=flux_dtype,
                    token=hf_token,
                )
            except Exception as flux_err:
                if "401" in str(flux_err) or "gated" in str(flux_err).lower() or "restricted" in str(flux_err).lower():
                    raise RuntimeError(
                        f"Model {meta.repo_id} requires authentication or terms acceptance.\n"
                        "1. Accept terms on HuggingFace if the repository is gated.\n"
                        "2. Add your HuggingFace token to .env: HF_TOKEN=hf_...\n"
                        f"(Original error: {flux_err})"
                    ) from flux_err
                raise flux_err

            # Enable VAE tiling and slicing to prevent VRAM spikes during latent decode
            if hasattr(pipe, "vae") and pipe.vae is not None:
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                if hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()

            if device_str == "cuda":
                # For FLUX on 4GB VRAM, sequential CPU offload dynamically swaps layers
                # between host system RAM and NVIDIA RTX 3050 VRAM during execution
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                    logger.info("flux_sequential_cpu_offload_enabled", model=meta.id)
                elif hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                    logger.info("flux_model_cpu_offload_enabled", model=meta.id)
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
            if device_str == "cuda":
                # Enable sequential CPU offload to prevent 4GB VRAM OOM
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                elif hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to(device_str)
            else:
                pipe = pipe.to(device_str)

            if hasattr(pipe, "enable_attention_slicing"):
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            if hasattr(pipe, "enable_vae_tiling"):
                try:
                    pipe.enable_vae_tiling()
                except Exception:
                    pass
            return pipe

        # ─── 3. Stable Diffusion 1.5 & SD-Turbo ──────────────────────────────────
        from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

        # Models like small-sd lack explicit fp16 variant files, requiring fallback to standard weights
        variant = "fp16" if (device_str == "cuda" and meta.id != "small-sd") else None
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                meta.repo_id,
                torch_dtype=torch_dtype,
                variant=variant,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except Exception:
            if variant is not None:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    meta.repo_id,
                    torch_dtype=torch_dtype,
                    variant=None,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            else:
                raise
        try:
            if meta.id != "sd-turbo":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, steps_offset=1)
        except Exception:
            pass

        pipe = pipe.to(device_str)
        if device_str == "cuda" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        return pipe
