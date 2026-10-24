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

#: Ceiling on a base64 image payload accepted from a remote generation API.
MAX_REMOTE_IMAGE_B64_CHARS = 32_000_000
#: Ceiling on the decoded image written to the gallery, in bytes.
MAX_REMOTE_IMAGE_BYTES = 24_000_000


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
            
            # Clamp steps to 12-20 range for high fidelity and consistent generation time
            effective_steps = max(12, min(20, getattr(job, "steps", 15) or 15))
            job.steps = effective_steps
            
            image = self._pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_inference_steps=effective_steps,
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


class RemoteOpenAIImageProvider(ImageProvider):
    """Remote OpenAI-compatible image provider to offload compute (/v1/images/generations)."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "dall-e-3",
        output_dir: str = "data/images",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[str]:
        return [self.model]

    async def health_check(self) -> bool:
        return bool(self.base_url)

    async def generate(self, job: ImageJob, interrupt: asyncio.Event) -> str:
        """Call remote /images/generations endpoint, download resulting image, and save to output_dir."""
        import base64
        import httpx

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.backend = "remote_openai_compatible"
        job.device = f"Cloud Endpoint ({self.base_url})"
        job.model = self.model
        job.progress = 10

        filename = f"img_{job.job_id}_{int(time.time())}.png"
        full_path = self.output_dir / filename

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Standard OpenAI Images API payload
        payload = {
            "prompt": job.prompt,
            "n": 1,
            "size": f"{job.width}x{job.height}" if job.width in (256, 512, 1024) else "512x512",
            "response_format": "b64_json",
        }
        if self.model and self.model != "default":
            payload["model"] = self.model

        url = f"{self.base_url}/images/generations"
        logger.info("remote_image_generation_request", url=url, prompt=job.prompt[:40])

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err_msg = resp.text[:200]
                logger.error("remote_image_gen_error", status=resp.status_code, body=err_msg)
                raise RuntimeError(f"Remote Image API returned {resp.status_code}: {err_msg}")

            data = resp.json()
            images_data = data.get("data", [])
            if not images_data:
                raise RuntimeError("No image data returned from remote endpoint")

            first_item = images_data[0]
            raw_bytes: bytes | None = None
            if "b64_json" in first_item:
                encoded = str(first_item["b64_json"])
                if len(encoded) > MAX_REMOTE_IMAGE_B64_CHARS:
                    raise RuntimeError("Remote image payload exceeds the accepted size limit")
                raw_bytes = await asyncio.to_thread(base64.b64decode, encoded)
            elif "url" in first_item:
                img_url = str(first_item["url"])
                scheme = img_url.split(":", 1)[0].lower()
                if scheme not in ("http", "https"):
                    raise RuntimeError(f"Refusing to fetch image with unsupported scheme: {scheme!r}")
                dl_resp = await client.get(img_url)
                if dl_resp.status_code != 200:
                    raise RuntimeError(f"Failed to download image (HTTP {dl_resp.status_code})")
                if len(dl_resp.content) > MAX_REMOTE_IMAGE_BYTES:
                    raise RuntimeError("Downloaded image exceeds the accepted size limit")
                raw_bytes = dl_resp.content
            else:
                raise RuntimeError("Unrecognized image response format")

            if not raw_bytes:
                raise RuntimeError("Remote image response contained no bytes")
            await asyncio.to_thread(full_path.write_bytes, raw_bytes)

        job.progress = 100
        job.completed_at = time.time()
        job.duration_s = job.completed_at - job.started_at
        job.output_path = f"/data/images/{filename}"
        job.status = JobStatus.COMPLETED

        logger.info(
            "remote_image_generated",
            output_path=job.output_path,
            duration_s=round(job.duration_s, 2),
        )
        return job.output_path
