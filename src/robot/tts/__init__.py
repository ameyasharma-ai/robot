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

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            self._voice.synthesize_wav(text, wf)
        bio.seek(0)

        with wave.open(bio, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        return audio, sr

    async def health_check(self) -> bool:
        return self._voice is not None
