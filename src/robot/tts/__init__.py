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

    Produces WAV audio from text using ONNX Runtime.
    """

    def __init__(self, model_path: str) -> None:
        logger.info("loading_piper", model=model_path)
        self._voice = piper.PiperVoice.load(model_path)
        logger.info("piper_loaded")

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        chunks = []
        sr = 16000
        for chunk in self._voice.synthesize(text):
            chunks.append(chunk.audio_float_array)
            if chunk.sample_rate:
                sr = chunk.sample_rate
        if not chunks:
            return np.array([], dtype=np.float32), sr
        if len(chunks) == 1:
            return chunks[0], sr
        return np.concatenate(chunks), sr

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import asyncio
        return await asyncio.to_thread(self._synthesize_sync, text)

    async def health_check(self) -> bool:
        return self._voice is not None
