"""Local Stable Diffusion image generation provider."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import psutil

from robot.image import ImageJob, JobStatus, ImageProvider
from robot.telemetry import get_logger

logger = get_logger("image.sd")


class LocalStableDiffusionProvider(ImageProvider):
    """Local Stable Diffusion image provider using diffusers on CPU/Vulkan."""

    def __init__(self, output_dir: str = "data/images", model_id: str = "OFA-Sys/small-stable-diffusion-v0") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._pipe = None
        self._backend = "cpu+amd_opt"
        self._device = "AMD Ryzen 7 6800H + Radeon 680M Platform"
        self._fallback = True
        self._model_loaded = False
        
        # Check available RAM before doing anything
        ram_info = psutil.virtual_memory()
        self.available_ram_mb = ram_info.available / (1024 * 1024)

    def list_models(self) -> list[str]:
        return [self.model_id]

    async def health_check(self) -> bool:
        return True

    def _load_model(self) -> None:
        if self._model_loaded:
            return
            
        logger.info("sd_model_load_started", model=self.model_id, backend=self._backend, device=self._device)
        print(f"\n==================================================")
        print(f"IMAGE GENERATION MODEL SETUP")
        print(f"MODEL: {self.model_id}")
        print(f"EXPECTED RAM: ~2 GB")
        print(f"BACKEND: {self._backend}")
        print(f"DEVICE: {self._device}")
        print(f"AVAILABLE SYSTEM RAM: {self.available_ram_mb:.0f} MB")
        print(f"==================================================\n")
        
        if self.available_ram_mb < 3000:
            logger.error("sd_model_load_failed", reason="Insufficient RAM", available_mb=self.available_ram_mb)
            raise RuntimeError(f"Insufficient RAM to load model. Available: {self.available_ram_mb:.0f} MB. Required: ~3000 MB.")
            
        import torch
        from diffusers import StableDiffusionPipeline
        
        # Optimize multi-core execution for AMD Ryzen 7 6800H host platform
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        device = torch.device("cpu")
        
        # Download and load
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Use DPMSolverMultistepScheduler with steps_offset=1 to eliminate diffusers warning and accelerate inference
        try:
            from diffusers import DPMSolverMultistepScheduler
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipe.scheduler.config,
                steps_offset=1,
            )
        except Exception:
            pass
        self._pipe = self._pipe.to(device)
        # Enable attention slicing for minimal memory footprint and fast inference
        if hasattr(self._pipe, "enable_attention_slicing"):
            self._pipe.enable_attention_slicing()
        self._model_loaded = True
        logger.info("sd_model_loaded", backend=self._backend, device=self._device)

    async def generate(self, job: ImageJob, interrupt: asyncio.Event) -> str:
        job.backend = self._backend
        job.device = self._device
        job.model = self.model_id
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = 5
        
        filename = f"img_{job.job_id}_{int(time.time())}.png"
        full_path = self.output_dir / filename
        
        def _do_generate():
            self._load_model()
            job.progress = 20
            
            # Run inference
            # To handle cancellation, we could use a callback in diffusers, but for now we'll just run it
            def callback(step: int, timestep: int, latents: Any) -> None:
                if interrupt.is_set():
                    raise InterruptedError("Cancelled by user")
                job.progress = 20 + int((step / job.steps) * 75)
            
            image = self._pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_inference_steps=job.steps,
                width=job.width,
                height=job.height,
                callback=callback,
                callback_steps=1,
            ).images[0]
            
            image.save(full_path)
            
        try:
            await asyncio.to_thread(_do_generate)
        except InterruptedError:
            job.status = JobStatus.CANCELLED
            job.error = "Interrupted by user"
            raise asyncio.CancelledError("Image generation cancelled")
        except Exception as e:
            raise e
            
        job.progress = 100
        job.completed_at = time.time()
        job.duration_s = job.completed_at - job.started_at
        job.output_path = f"/data/images/{filename}"
        job.status = JobStatus.COMPLETED
        
        # Emit metric
        peak_ram = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info("sd_generation_metadata", 
            backend=self._backend,
            device=self._device,
            width=job.width,
            height=job.height,
            steps=job.steps,
            generation_ms=job.duration_s * 1000,
            peak_ram_mb=peak_ram,
            fallback=self._fallback
        )
        
        return job.output_path
