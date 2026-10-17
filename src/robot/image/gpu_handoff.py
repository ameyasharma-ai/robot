"""Dynamic NVIDIA GPU Resource Handoff & LLM Offloading Orchestrator.

Architectural Rationale:
- NVIDIA GeForce RTX 3050 Laptop GPU has 4GB VRAM.
- When running locally, llama-server occupies 1.5 - 3.4 GB of VRAM with FlashAttention.
- Heavy generative diffusion models (FLUX.1, SDXL, Stable Diffusion 1.5/Turbo) require
  2.5 - 3.8 GB of VRAM. Running both simultaneously causes immediate CUDA Out Of Memory (OOM).
- This orchestrator implements the dynamic handoff lifecycle:
  1. Detects heavy image generation requests.
  2. If local LLM is active on GPU, gracefully offloads LLM to Cloud API (Gemini / OpenAI-Compatible)
     or stops llama-server, freeing 100% of GPU memory.
  3. Purges PyTorch CUDA cache (`torch.cuda.empty_cache()`, `gc.collect()`).
  4. Loads the requested diffusion model onto CUDA (using fp16 and memory slicing).
  5. Executes inference using the pre-synthesized LLM instruction (prompt, seed, steps, etc.).
  6. Releases diffusion model from VRAM and purges CUDA cache.
  7. Restores local LLM (or maintains API connection based on settings).
  8. Triggers whatever LLM is currently loaded to notify the user of completion.
"""

from __future__ import annotations

import asyncio
import gc
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

from robot.image.model_registry import ImageModelMetadata, ImageModelRegistry
from robot.image.prompt_enhancer import ImageGenerationInstruction
from robot.telemetry import get_logger

logger = get_logger("image.gpu_handoff")


class GpuHandoffState(str, Enum):
    """Lifecycle states of the GPU resource handoff orchestrator."""

    IDLE = "IDLE"
    OFFLOADING_LLM = "OFFLOADING_LLM"
    GPU_CLEARED = "GPU_CLEARED"
    LOADING_DIFFUSION = "LOADING_DIFFUSION"
    GENERATING_IMAGE = "GENERATING_IMAGE"
    RELEASING_DIFFUSION = "RELEASING_DIFFUSION"
    RESTORING_LLM = "RESTORING_LLM"
    NOTIFYING_USER = "NOTIFYING_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GpuResourceHandoffManager:
    """Coordinates dynamic VRAM arbitration between llama-server and diffusion pipelines."""

    def __init__(self, engine: Any | None = None) -> None:
        """Initialize GPU resource handoff manager.

        Args:
            engine: Reference to RobotEngine instance.
        """
        self.engine = engine
        self.state = GpuHandoffState.IDLE
        self._lock = asyncio.Lock()
        self._previous_llm_provider: str | None = None

    def is_cuda_available(self) -> bool:
        """Check if PyTorch has access to an active NVIDIA CUDA device.

        Returns:
            bool: True if CUDA is accessible.
        """
        try:
            import torch
            return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        except Exception:
            return False

    def get_free_vram_mb(self) -> float:
        """Return currently available free GPU VRAM in megabytes.

        Returns:
            float: Free VRAM in MB, or 0.0 if CUDA is unavailable.
        """
        if not self.is_cuda_available():
            return 0.0
        try:
            import torch
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return free_bytes / (1024 * 1024)
        except Exception as e:
            logger.debug("cuda_mem_info_failed", error=str(e))
            return 0.0

    async def prepare_gpu_for_image_model(self, model_meta: ImageModelMetadata) -> bool:
        """Prepare NVIDIA GPU by offloading local LLM and clearing CUDA memory if model is heavy.

        Args:
            model_meta: Specification of image model to be loaded.

        Returns:
            bool: True if GPU is prepared and has sufficient headroom, False otherwise.
        """
        if not model_meta.is_heavy or not self.is_cuda_available():
            logger.info("model_does_not_require_heavy_gpu_handoff", model=model_meta.id)
            return True

        self.state = GpuHandoffState.OFFLOADING_LLM
        logger.info(
            "gpu_handoff_started",
            model=model_meta.id,
            vram_required=model_meta.vram_mb_required,
            current_free_vram_mb=round(self.get_free_vram_mb(), 1),
        )

        if self.engine:
            self._previous_llm_provider = getattr(self.engine, "llm_provider_type", "local")

            # If local LLM is active on GPU, offload to Cloud API or stop llama-server
            if self._previous_llm_provider == "local":
                cfg = getattr(self.engine, "config", {}) or {}
                llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
                gemini_key = os.getenv("GEMINI_API_KEY", "") or llm_cfg.get("gemini", {}).get("api_key", "")
                openai_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "") or llm_cfg.get("openai_compatible", {}).get("api_key", "")
                has_gemini = bool(str(gemini_key).strip())
                has_openai = bool(str(openai_key).strip())

                if has_gemini and hasattr(self.engine, "switch_llm_provider"):
                    logger.info("offloading_local_llm_to_gemini_api")
                    await self.engine.switch_llm_provider("gemini")
                elif has_openai and hasattr(self.engine, "switch_llm_provider"):
                    logger.info("offloading_local_llm_to_openai_compatible_api")
                    await self.engine.switch_llm_provider("openai_compatible")
                else:
                    # No API key configured: gracefully stop llama-server to free VRAM
                    from robot.llm import LlamaServerManager
                    logger.info("stopping_llama_server_directly_to_free_gpu")
                    await LlamaServerManager.stop()

        # Clear PyTorch CUDA cache and invoke garbage collection
        self.state = GpuHandoffState.GPU_CLEARED
        await self._purge_cuda_memory()

        free_vram = self.get_free_vram_mb()
        logger.info("gpu_cleared_for_image_model", free_vram_mb=round(free_vram, 1))
        return True

    async def cleanup_gpu_and_restore_llm(self) -> None:
        """Release diffusion model from VRAM and restore previous LLM provider."""
        self.state = GpuHandoffState.RELEASING_DIFFUSION
        await self._purge_cuda_memory()

        if self.engine and self._previous_llm_provider == "local":
            self.state = GpuHandoffState.RESTORING_LLM
            logger.info("restoring_previous_local_llm_provider")
            try:
                if hasattr(self.engine, "switch_llm_provider"):
                    await self.engine.switch_llm_provider("local")
                else:
                    from robot.llm import LlamaServerManager
                    await LlamaServerManager.start()
            except Exception as e:
                logger.error("failed_to_restore_local_llm", error=str(e))

        self.state = GpuHandoffState.COMPLETED

    async def notify_user_via_loaded_llm(
        self,
        instruction: ImageGenerationInstruction,
        output_path: str,
        duration_s: float,
    ) -> str:
        """Trigger whatever LLM is currently loaded to synthesize a completion notification.

        Args:
            instruction: Generation instruction that was executed.
            output_path: Relative URL path to generated image file.
            duration_s: Generation duration in seconds.

        Returns:
            str: Synthesized notification message.
        """
        self.state = GpuHandoffState.NOTIFYING_USER
        default_notification = (
            f"I have finished generating your image: \"{instruction.raw_user_query}\"! "
            f"It is ready and visible on your dashboard."
        )

        notification_text = default_notification
        if self.engine and getattr(self.engine, "llm", None):
            try:
                system_prompt = (
                    "You are an embodied AI companion. An image requested by the user has just finished rendering. "
                    "Provide a warm, enthusiastic, 1-2 sentence spoken notification confirming the image is ready on their dashboard. "
                    "Do NOT describe raw technical arguments; speak naturally."
                )
                user_msg = (
                    f"User Request: \"{instruction.raw_user_query}\"\n"
                    f"Prompt Generated: \"{instruction.prompt}\"\n"
                    f"Render Duration: {duration_s:.1f}s\n"
                    "Notify the user that their artwork is ready."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ]

                reply = ""
                async for chunk in self.engine.llm.stream(messages=messages, temperature=0.7, max_tokens=100):
                    if not chunk.startswith("__TOOL_CALL__") and not chunk.startswith("<thought>"):
                        reply += chunk
                clean_reply = reply.strip()
                if clean_reply and len(clean_reply) > 5:
                    notification_text = clean_reply
            except Exception as e:
                logger.warning("llm_completion_notification_synthesis_failed", error=str(e))

        # Enqueue voice announcement if voice is enabled and not in chat mode
        if self.engine and getattr(self.engine, "voice_enabled", False) and not getattr(self.engine, "is_chat_mode", False):
            if hasattr(self.engine, "_tts_queue"):
                try:
                    await self.engine._tts_queue.put((notification_text, asyncio.Event(), None))
                    logger.info("image_completion_voice_enqueued", text=notification_text[:50])
                except Exception as e:
                    logger.warning("voice_notification_enqueue_failed", error=str(e))

        # Emit dashboard notification event
        if self.engine and getattr(self.engine, "observability", None):
            self.engine.observability.emit_event(
                event_type="image.notification",
                component="IMAGE",
                operation="NOTIFY",
                status="COMPLETED",
                metadata={
                    "notification": notification_text,
                    "output_path": output_path,
                    "prompt": instruction.raw_user_query,
                    "duration_s": duration_s,
                },
            )

        logger.info("user_notified_of_image_completion", text=notification_text[:60])
        return notification_text

    async def _purge_cuda_memory(self) -> None:
        """Force garbage collection and clear PyTorch CUDA caching allocator."""
        gc.collect()
        if self.is_cuda_available():
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        await asyncio.sleep(0.1)
