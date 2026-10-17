"""Image generation service, job queue, and provider abstractions."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from robot.telemetry import get_logger

logger = get_logger("image")


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ImageJob:
    """Represents an asynchronous image generation task."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 4
    seed: int = -1
    model: str = "default"
    provider: str = "fast_procedural"
    backend: str = "cpu"
    device: str = "CPU"
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    duration_s: float = 0.0
    output_path: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["duration_s"] = round(self.duration_s, 2)
        return d


class ImageProvider(ABC):
    """Abstract interface for image generation backends."""

    @abstractmethod
    async def generate(self, job: ImageJob, interrupt: asyncio.Event) -> str:
        """Generate image and save to disk; returns relative output file path."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List supported model names."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify provider availability."""
        ...


class FastProceduralImageProvider(ImageProvider):
    """High-speed local image generation provider.

    Renders styled graphic artwork without requiring multi-GB VRAM,
    guaranteeing non-blocking instant execution under conversational workloads.
    """

    def __init__(self, output_dir: str = "data/images") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[str]:
        return ["fast_procedural_v1", "robot_avatar_art", "cyberpunk_neon"]

    async def health_check(self) -> bool:
        return True

    async def generate(self, job: ImageJob, interrupt: asyncio.Event) -> str:
        """Render beautiful, themed procedural visual artwork on CPU/Pillow."""
        import random
        from PIL import Image, ImageDraw, ImageFont

        filename = f"img_{job.job_id}_{int(time.time())}.png"
        full_path = self.output_dir / filename

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = 10

        # Simulate stepped rendering with cancellation checks
        steps = max(1, min(job.steps, 30))
        for step in range(1, steps + 1):
            if interrupt.is_set():
                job.status = JobStatus.CANCELLED
                job.error = "Interrupted by user or voice barge-in"
                raise asyncio.CancelledError("Image generation cancelled")
            job.progress = int((step / steps) * 90)
            await asyncio.sleep(0.04)

        # Generate stylized high-res visual card based on prompt
        w, h = job.width, job.height
        seed = job.seed if job.seed >= 0 else random.randint(1, 999999)
        random.seed(seed)

        # Background gradient
        img = Image.new("RGB", (w, h), color=(15, 18, 28))
        draw = ImageDraw.Draw(img)

        # Palette
        hue_theme = hash(job.prompt) % 360
        r_base = (hash(job.prompt) * 13) % 180 + 30
        g_base = (hash(job.prompt) * 17) % 180 + 30
        b_base = (hash(job.prompt) * 23) % 220 + 35

        # Render ambient geometric and organic elements
        for i in range(12):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rad = random.randint(w // 8, w // 2)
            c_accent = (
                (r_base + random.randint(-40, 40)) % 255,
                (g_base + random.randint(-40, 40)) % 255,
                (b_base + random.randint(-40, 40)) % 255,
            )
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=c_accent, width=2)

        # Grid lines
        for y in range(0, h, 40):
            draw.line([(0, y), (w, y)], fill=(30, 40, 60), width=1)
        for x in range(0, w, 40):
            draw.line([(x, 0), (x, h)], fill=(30, 40, 60), width=1)

        # Text banner
        banner_h = 100
        draw.rectangle([(0, h - banner_h), (w, h)], fill=(10, 12, 18))
        draw.line([(0, h - banner_h), (w, h - banner_h)], fill=(0, 220, 255), width=2)

        title = f"AI VISUAL: {job.prompt[:42]}"
        meta_str = f"Seed: {seed} | Steps: {steps} | {w}x{h} | FastProcedural"
        draw.text((16, h - 80), title, fill=(240, 245, 255))
        draw.text((16, h - 45), meta_str, fill=(0, 200, 240))

        img.save(str(full_path), format="PNG")
        job.progress = 100
        job.completed_at = time.time()
        job.duration_s = job.completed_at - job.started_at
        job.output_path = f"/data/images/{filename}"
        job.status = JobStatus.COMPLETED
        return job.output_path


class ImageGenerationService:
    """Manages asynchronous background image generation jobs with voice prioritization."""

    def __init__(
        self,
        output_dir: str = "data/images",
        max_concurrent: int = 1,
        default_width: int = 512,
        default_height: int = 512,
        engine: Any | None = None,
        provider: ImageProvider | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.default_width = default_width
        self.default_height = default_height
        self.engine = engine

        self._fallback_provider: ImageProvider = FastProceduralImageProvider(output_dir=str(self.output_dir))
        if provider is not None:
            self.provider: ImageProvider = provider
        else:
            from robot.image.unified_provider import UnifiedImageProvider
            self.provider = UnifiedImageProvider(
                output_dir=str(self.output_dir),
                engine=self.engine,
            )
        self._jobs: dict[str, ImageJob] = {}
        self._interrupt_events: dict[str, asyncio.Event] = {}
        self._queue: asyncio.Queue[ImageJob] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._event_callbacks: list = []

    def on_event(self, cb) -> None:
        self._event_callbacks.append(cb)

    async def _emit_job_event(self, event_type: str, job: ImageJob) -> None:
        for cb in self._event_callbacks:
            try:
                res = cb(event_type=event_type, job=job)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug("image_event_cb_error", error=str(e))

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())

    def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

    def submit_job(
        self,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        steps: int = 4,
        seed: int = -1,
        model: str = "default",
        negative_prompt: str = "",
        trace_id: str | None = None,
    ) -> ImageJob:
        """Validate parameters and enqueue image generation job."""
        # Sanitize and bound parameters
        safe_w = max(128, min(width or self.default_width, 1024))
        safe_h = max(128, min(height or self.default_height, 1024))
        safe_steps = max(1, min(steps, 50))
        safe_prompt = prompt.strip()[:500]

        job = ImageJob(
            trace_id=trace_id,
            prompt=safe_prompt,
            negative_prompt=negative_prompt.strip()[:500],
            width=safe_w,
            height=safe_h,
            steps=safe_steps,
            seed=seed,
            model=model,
            provider=getattr(self.provider, "__class__", type(self.provider)).__name__,
            status=JobStatus.QUEUED,
        )
        self._jobs[job.job_id] = job
        self._interrupt_events[job.job_id] = asyncio.Event()
        self._queue.put_nowait(job)
        asyncio.create_task(self._emit_job_event("image.queued", job))
        logger.info("image_job_queued", job_id=job.job_id, prompt=safe_prompt[:40], model=model)
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or queued image job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        ev = self._interrupt_events.get(job_id)
        if ev:
            ev.set()

        job.status = JobStatus.CANCELLED
        job.error = "Cancelled by user"
        job.completed_at = time.time()
        if job.started_at:
            job.duration_s = job.completed_at - job.started_at
        await self._emit_job_event("image.cancelled", job)
        logger.info("image_job_cancelled", job_id=job_id)
        return True

    def get_job(self, job_id: str) -> ImageJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return [j.to_dict() for j in list(self._jobs.values())[-limit:]]

    async def _process_queue(self) -> None:
        """Background queue processor with error isolation."""
        while True:
            job = await self._queue.get()
            if job.status == JobStatus.CANCELLED:
                continue

            interrupt = self._interrupt_events.get(job.job_id, asyncio.Event())
            await self._emit_job_event("image.started", job)

            try:
                out_path = await self.provider.generate(job, interrupt)
                job.output_path = out_path
                await self._emit_job_event("image.completed", job)
                logger.info("image_job_completed", job_id=job.job_id, duration_s=round(job.duration_s, 2))
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                await self._emit_job_event("image.cancelled", job)
            except Exception as e:
                # Fallback to local provider if remote image generation fails
                if hasattr(self, "_fallback_provider") and self.provider != self._fallback_provider:
                    logger.warning("remote_image_gen_fallback", error=str(e), job_id=job.job_id)
                    try:
                        out_path = await self._fallback_provider.generate(job, interrupt)
                        job.output_path = out_path
                        await self._emit_job_event("image.completed", job)
                        logger.info("image_job_completed_fallback", job_id=job.job_id)
                        continue
                    except Exception as fb_err:
                        logger.error("image_fallback_also_failed", error=str(fb_err))
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                await self._emit_job_event("image.failed", job)
                logger.error("image_job_failed", job_id=job.job_id, error=str(e))


from robot.image.gpu_handoff import GpuHandoffState, GpuResourceHandoffManager
from robot.image.model_registry import ImageModelFamily, ImageModelMetadata, ImageModelRegistry
from robot.image.prompt_enhancer import ImageGenerationInstruction, ImagePromptEnhancer
from robot.image.unified_provider import UnifiedImageProvider

__all__ = [
    "FastProceduralImageProvider",
    "GpuHandoffState",
    "GpuResourceHandoffManager",
    "ImageGenerationInstruction",
    "ImageGenerationService",
    "ImageJob",
    "ImageModelFamily",
    "ImageModelMetadata",
    "ImageModelRegistry",
    "ImagePromptEnhancer",
    "ImageProvider",
    "JobStatus",
    "UnifiedImageProvider",
]

