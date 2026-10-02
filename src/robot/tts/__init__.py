"""Text-to-Speech provider interface and Piper ONNX engine."""

from __future__ import annotations

import io
import re
import wave

import numpy as np
import piper

from robot.telemetry import get_logger

logger = get_logger("tts")


def split_into_phrases(text: str) -> list[str]:
    """Split text at natural pause points for low-latency streaming TTS.

    Uses sentence-ending punctuation as split boundaries.
    Guards against tiny single-word fragments.
    """
    parts = re.split(r"(?<=[.!?,;:\n])\s+", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


class TTSProvider:
    """Abstract TTS provider interface."""

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Returns:
            (audio_float32, sample_rate)
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        raise NotImplementedError


class PiperTTS(TTSProvider):
    """Piper ONNX TTS engine — local, low-latency voice synthesis.

    Produces WAV audio from text using ONNX Runtime with an in-memory
    phrase cache for instant 0ms synthesis of repetitive/acknowledgment phrases.
    """

    def __init__(self, model_path: str) -> None:
        logger.info("loading_piper", model=model_path)
        self._voice = piper.PiperVoice.load(model_path)
        logger.info("piper_loaded")
        self._phrase_cache: dict[str, tuple[np.ndarray, int]] = {}

    def _normalize_key(self, text: str) -> str:
        return text.strip().lower()

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        norm_key = self._normalize_key(text)
        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        chunks = []
        sr = 16000
        for chunk in self._voice.synthesize(text):
            chunks.append(chunk.audio_float_array)
            if chunk.sample_rate:
                sr = chunk.sample_rate
        if not chunks:
            res = (np.array([], dtype=np.float32), sr)
        elif len(chunks) == 1:
            res = (chunks[0], sr)
        else:
            res = (np.concatenate(chunks), sr)

        # Cache short phrases (< 120 chars) in memory
        if len(norm_key) < 120:
            self._phrase_cache[norm_key] = res
        return res

    def warm_phrase(self, text: str) -> tuple[np.ndarray, int]:
        """Pre-synthesize and store phrase in memory cache for 0ms instant playback."""
        return self._synthesize_sync(text)

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        norm_key = self._normalize_key(text)
        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        import asyncio
        return await asyncio.to_thread(self._synthesize_sync, text)

    async def health_check(self) -> bool:
        return self._voice is not None

