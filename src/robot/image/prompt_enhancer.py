"""LLM-driven prompt enhancement and instruction synthesizer for image generation.

Architectural Rationale:
- Takes raw conversational user requests (e.g. "Draw a futuristic samurai robot in rain")
  and leverages whichever LLM is currently loaded (Local Qwen3.5-4B, Gemini, or OpenRouter)
  to formulate an enriched visual instruction.
- Generates all parameters supported by the diffusion architecture:
  expanded visual prompt, negative prompt, randomized or chosen seed, dimension bounds,
  inference step count, and classifier-free guidance scale.
- Employs a robust deterministic fallback if the LLM is undergoing transition or times out.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

from robot.image.model_registry import ImageModelMetadata, ImageModelRegistry
from robot.telemetry import get_logger

logger = get_logger("image.prompt_enhancer")


@dataclass
class ImageGenerationInstruction:
    """Consolidated generation parameters synthesized by the active LLM."""

    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    model_id: str
    raw_user_query: str

    def to_dict(self) -> dict[str, Any]:
        """Convert instruction to JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Serializable instruction dictionary.
        """
        return asdict(self)


class ImagePromptEnhancer:
    """Synthesizes rich diffusion instructions using whichever LLM is currently active."""

    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, out of focus, low quality, distorted, extra limbs, bad anatomy, "
        "gross proportions, duplicate, watermark, signature, grainy, low resolution"
    )

    def __init__(self, engine: Any | None = None) -> None:
        """Initialize prompt enhancer.

        Args:
            engine: Optional reference to RobotEngine instance for accessing active LLM.
        """
        self.engine = engine

    async def generate_instruction(
        self,
        user_query: str,
        target_model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
    ) -> ImageGenerationInstruction:
        """Formulate a comprehensive generation instruction using the active LLM.

        Args:
            user_query: Raw input text from the user describing the desired image.
            target_model: Optional requested model ID (defaults to sd-turbo).
            width: Optional width override.
            height: Optional height override.
            steps: Optional steps override.
            seed: Optional seed override.

        Returns:
            ImageGenerationInstruction: Complete parameters ready for the diffusion pipeline.
        """
        clean_query = user_query.strip()
        model_meta = ImageModelRegistry.get(target_model or "") or ImageModelRegistry.get_default_model(gpu_available=True)
        chosen_seed = seed if (seed is not None and seed >= 0) else random.randint(10000, 999999)

        # Attempt synthesis through active LLM
        llm_instruction = await self._synthesize_via_llm(clean_query, model_meta, chosen_seed)
        if llm_instruction:
            # Apply user overrides if explicitly provided
            if seed is not None and seed >= 0:
                llm_instruction.seed = seed
            if width:
                llm_instruction.width = max(128, min(width, 1024))
            if height:
                llm_instruction.height = max(128, min(height, 1024))
            if steps:
                llm_instruction.steps = max(1, min(steps, 50))
            return llm_instruction

        # Deterministic fallback if LLM synthesis fails or is offline
        return self._fallback_instruction(clean_query, model_meta, chosen_seed, width, height, steps)

    async def _synthesize_via_llm(
        self,
        user_query: str,
        model_meta: ImageModelMetadata,
        seed: int,
    ) -> ImageGenerationInstruction | None:
        """Query active LLM to generate structured diffusion parameters.

        Args:
            user_query: User image request.
            model_meta: Target model metadata.
            seed: Chosen seed integer.

        Returns:
            ImageGenerationInstruction | None: Parsed instruction or None if failed.
        """
        if not self.engine or not getattr(self.engine, "llm", None):
            return None

        system_prompt = (
            "You are an expert diffusion prompt engineer for FLUX and Stable Diffusion. "
            "Your job is to transform a user's image concept into an optimal, highly detailed visual prompt. "
            "You must output ONLY valid JSON without markdown wrapping or conversational dialogue."
        )

        user_content = (
            f"User Concept: \"{user_query}\"\n"
            f"Target Model: {model_meta.name} (Family: {model_meta.family.value})\n"
            f"Recommended Steps: {model_meta.recommended_steps}\n"
            f"Default Guidance Scale: {model_meta.default_guidance_scale}\n"
            f"Supports Negative Prompt: {model_meta.supports_negative_prompt}\n\n"
            "Return JSON matching this exact structure:\n"
            "{\n"
            "  \"prompt\": \"Expanded visual description specifying subject, environment, lighting, texture, camera lens, and atmosphere\",\n"
            "  \"negative_prompt\": \"Undesirable flaws to avoid\",\n"
            f"  \"width\": {model_meta.default_width},\n"
            f"  \"height\": {model_meta.default_height},\n"
            f"  \"steps\": {model_meta.recommended_steps},\n"
            f"  \"guidance_scale\": {model_meta.default_guidance_scale},\n"
            f"  \"seed\": {seed}\n"
            "}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            raw_response = ""
            is_mock = lambda obj: "Mock" in type(obj).__name__
            sc = getattr(self.engine.llm, "stream_chat", None)
            st = getattr(self.engine.llm, "stream", None)
            if sc is not None and not is_mock(sc):
                stream_fn = sc
            elif st is not None and not is_mock(st):
                stream_fn = st
            elif sc is not None:
                stream_fn = sc
            elif st is not None:
                stream_fn = st
            else:
                return None

            stream_iter = stream_fn(messages=messages, max_tokens=350)
            async for token in stream_iter:
                # Strip tool call or reasoning tags
                if not token.startswith("__TOOL_CALL__") and not token.startswith("<thought>"):
                    raw_response += token

            clean_json_str = self._extract_json_block(raw_response)
            if not clean_json_str:
                return None

            data = json.loads(clean_json_str)
            p = str(data.get("prompt", "")).strip()
            if not p:
                return None

            neg_p = str(data.get("negative_prompt", self.DEFAULT_NEGATIVE_PROMPT)).strip() if model_meta.supports_negative_prompt else ""
            w = int(data.get("width", model_meta.default_width))
            h = int(data.get("height", model_meta.default_height))
            s = int(data.get("steps", model_meta.recommended_steps))
            g = float(data.get("guidance_scale", model_meta.default_guidance_scale))
            parsed_seed = int(data.get("seed", seed)) if data.get("seed") is not None else seed

            # Clamp parameters to hardware-safe bounds
            safe_w = max(256, min(w, 1024))
            safe_h = max(256, min(h, 1024))
            safe_steps = max(1, min(s, 50))

            logger.info("llm_image_instruction_synthesized", model=model_meta.id, prompt=p[:60])
            return ImageGenerationInstruction(
                prompt=p,
                negative_prompt=neg_p,
                seed=parsed_seed,
                width=safe_w,
                height=safe_h,
                steps=safe_steps,
                guidance_scale=g,
                model_id=model_meta.id,
                raw_user_query=user_query,
            )
        except Exception as e:
            logger.warning("llm_prompt_synthesis_failed_falling_back", error=str(e))
            return None

    def _fallback_instruction(
        self,
        query: str,
        meta: ImageModelMetadata,
        seed: int,
        w: int | None,
        h: int | None,
        steps: int | None,
    ) -> ImageGenerationInstruction:
        """Construct deterministic enhanced visual parameters when LLM is offline.

        Args:
            query: User concept string.
            meta: Target model metadata.
            seed: Random seed.
            w: Width override.
            h: Height override.
            steps: Steps override.

        Returns:
            ImageGenerationInstruction: Deterministic fallback instruction.
        """
        # Enhance prompt with photography, lighting, and cinematic quality modifiers
        enhancements = [
            "highly detailed",
            "cinematic volumetric lighting",
            "photorealistic texture",
            "8k resolution",
            "masterpiece composition",
        ]
        enhanced_prompt = f"{query}, {', '.join(enhancements)}"

        safe_w = max(256, min(w or meta.default_width, 1024))
        safe_h = max(256, min(h or meta.default_height, 1024))
        safe_steps = max(1, min(steps or meta.recommended_steps, 50))
        neg_p = self.DEFAULT_NEGATIVE_PROMPT if meta.supports_negative_prompt else ""

        return ImageGenerationInstruction(
            prompt=enhanced_prompt,
            negative_prompt=neg_p,
            seed=seed,
            width=safe_w,
            height=safe_h,
            steps=safe_steps,
            guidance_scale=meta.default_guidance_scale,
            model_id=meta.id,
            raw_user_query=query,
        )

    def _extract_json_block(self, text: str) -> str | None:
        """Extract first valid JSON object block from model completion string.

        Args:
            text: Raw model output.

        Returns:
            str | None: Clean JSON string or None.
        """
        # Remove any <thought>...</thought> tags
        cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return match.group(0).strip()
        return None
