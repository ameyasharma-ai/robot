"""Text-to-Speech provider interface and Piper ONNX engine."""

from __future__ import annotations

import io
import re
import wave

import numpy as np
import piper

from robot.telemetry import get_logger

logger = get_logger("tts")


def clean_text_for_speech(text: str) -> str:
    """Sanitize text for natural spoken TTS.

    Strips markdown syntax (**bold**, *italic*, backticks, headers, bullet points,
    blockquotes, and inline action tags) so TTS does not speak out asterisks or symbols.
    """
    if not text:
        return ""
    # Strip any XML tool call tags first before processing underscores and markdown
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?(?:tool_call|function|parameter)[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Strip markdown bold/italic asterisks and underscores: **word**, *word*, __word__, _word_
    cleaned = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_{1,3}([^_]+)_{1,3}(?!\w)", r"\1", cleaned)
    # Strip lingering raw asterisks or tildes
    cleaned = cleaned.replace("*", "").replace("~", "").replace("`", "")
    # Strip markdown headers (e.g. ## Title)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    # Strip markdown list bullets
    cleaned = re.sub(r"^\s*[-•*+]\s+", "", cleaned, flags=re.MULTILINE)
    # Strip motion tags like [motion:wave] or [action:nod] if any
    cleaned = re.sub(r"\[(motion|action):[^\]]+\]", "", cleaned, flags=re.IGNORECASE)
    # Collapse multiple whitespaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


def split_into_phrases(text: str) -> list[str]:
    """Split text at natural pause points for low-latency streaming TTS.

    Uses sentence-ending punctuation as split boundaries.
    Guards against tiny single-word fragments.
    """
    sanitized = clean_text_for_speech(text)
    parts = re.split(r"(?<=[.!?,;:\n])\s+", sanitized)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


def _shift_pitch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Pitch-shift float32 audio via fast linear resampling without altering duration."""
    if abs(factor - 1.0) < 0.01 or len(audio) == 0:
        return audio
    orig_len = len(audio)
    new_len = max(1, int(orig_len / factor))
    idx1 = np.linspace(0, orig_len - 1, new_len)
    resampled = np.interp(idx1, np.arange(orig_len), audio).astype(np.float32)
    idx2 = np.linspace(0, new_len - 1, orig_len)
    return np.interp(idx2, np.arange(new_len), resampled).astype(np.float32)


class TTSProvider:
    """Abstract TTS provider interface."""

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Returns:
            (audio_float32, sample_rate)
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        raise NotImplementedError


import os
from pathlib import Path

class PiperTTS(TTSProvider):
    """Piper ONNX TTS engine — local, low-latency, multi-voice synthesis.

    Produces WAV audio from text using ONNX Runtime with support for
    distinct voice models per personality, dynamic expressive prosody
    (length_scale, noise_scale, noise_w_scale, pitch shift), and an
    in-memory phrase cache for instant 0ms playback of acknowledgment phrases.
    """

    def __init__(self, model_path: str) -> None:
        self._voices: dict[str, piper.PiperVoice] = {}
        self._phrase_cache: dict[str, tuple[np.ndarray, int]] = {}
        self._default_model = self._resolve_model_path(model_path)
        self._active_model = self._default_model
        self._load_voice(self._default_model)
        logger.info("piper_initialized", default_model=self._default_model)

    def _resolve_model_path(self, model_path: str) -> str:
        if not model_path:
            return getattr(self, "_active_model", "models/tts/en-us-lessac-medium.onnx")
        p = Path(model_path)
        if p.exists():
            return str(p)
        candidates = [
            Path("models/tts") / model_path,
            Path("models/tts") / f"{model_path}.onnx",
            Path("models/tts") / f"{model_path.replace('_', '-')}.onnx",
            Path("models/tts") / f"{model_path.replace('-', '_')}.onnx",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return model_path

    def _load_voice(self, model_path: str) -> piper.PiperVoice:
        resolved = self._resolve_model_path(model_path)
        if resolved not in self._voices:
            logger.info("loading_piper_voice", model=resolved)
            try:
                self._voices[resolved] = piper.PiperVoice.load(resolved)
                logger.info("piper_voice_loaded", model=resolved)
            except Exception as e:
                logger.error(
                    "piper_voice_load_failed",
                    model=resolved,
                    error=str(e),
                    fallback=self._default_model,
                )
                # Fall back to the default model so the robot doesn't crash
                if resolved != self._default_model and self._default_model in self._voices:
                    return self._voices[self._default_model]
                raise
        return self._voices[resolved]

    def set_active_voice(self, model_path: str) -> None:
        """Dynamically switch active voice model."""
        resolved = self._resolve_model_path(model_path)
        self._load_voice(resolved)
        self._active_model = resolved
        logger.info("active_voice_changed", model=resolved)

    def _normalize_key(
        self,
        text: str,
        speed: float,
        pitch: float,
        voice_key: str,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> str:
        ns_str = f"{noise_scale:.2f}" if noise_scale is not None else "def"
        nw_str = f"{noise_w:.2f}" if noise_w is not None else "def"
        return f"{voice_key}|{text.strip().lower()}|s={speed:.2f}|p={pitch:.2f}|ns={ns_str}|nw={nw_str}"

    def _synthesize_sync(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        voice_model: str | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> tuple[np.ndarray, int]:
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), 16000

        target_voice_key = self._resolve_model_path(voice_model or self._active_model)
        norm_key = self._normalize_key(
            clean_text, speed, pitch, target_voice_key, noise_scale, noise_w
        )
        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        voice = self._load_voice(target_voice_key)

        # Piper length_scale: higher values = slower speech; lower values = faster speech
        length_scale = 1.0 / max(0.5, min(2.0, speed))
        syn_kwargs: dict[str, Any] = {"length_scale": length_scale}
        if noise_scale is not None:
            syn_kwargs["noise_scale"] = float(noise_scale)
        if noise_w is not None:
            syn_kwargs["noise_w_scale"] = float(noise_w)

        syn_config = piper.SynthesisConfig(**syn_kwargs)

        chunks = []
        sr = 16000
        for chunk in voice.synthesize(clean_text, syn_config=syn_config):
            chunks.append(chunk.audio_float_array)
            if chunk.sample_rate:
                sr = chunk.sample_rate

        if not chunks:
            raw_audio = np.array([], dtype=np.float32)
        elif len(chunks) == 1:
            raw_audio = chunks[0]
        else:
            raw_audio = np.concatenate(chunks)

        # Apply pitch shift if configured (e.g. thin high-pitched voice for Luna)
        if abs(pitch - 1.0) >= 0.02 and len(raw_audio) > 0:
            raw_audio = _shift_pitch(raw_audio, pitch)

        res = (raw_audio, sr)

        # Cache short phrases (< 120 chars) in memory
        if len(clean_text) < 120:
            self._phrase_cache[norm_key] = res
        return res

    def warm_phrase(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        voice_model: str | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Pre-synthesize and store phrase in memory cache for 0ms instant playback."""
        return self._synthesize_sync(
            text,
            speed=speed,
            pitch=pitch,
            voice_model=voice_model,
            noise_scale=noise_scale,
            noise_w=noise_w,
        )

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        voice_model: str | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> tuple[np.ndarray, int]:
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), 16000

        target_voice_key = self._resolve_model_path(voice_model or self._active_model)
        norm_key = self._normalize_key(
            clean_text, speed, pitch, target_voice_key, noise_scale, noise_w
        )
        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        import asyncio
        return await asyncio.to_thread(
            self._synthesize_sync,
            clean_text,
            speed,
            pitch,
            voice_model,
            noise_scale,
            noise_w,
        )

    async def health_check(self) -> bool:
        return bool(self._voices)

