"""Speech-to-Text provider interface and whisper.cpp client."""

from __future__ import annotations

import io
import re
import wave
from abc import ABC, abstractmethod

import numpy as np
import requests

from robot.telemetry import get_logger

logger = get_logger("stt")

# Hallucination patterns — Whisper commonly generates these on noise/silence
HALLUCINATION_PATTERNS = [
    r"^\[.*\]$",
    r"^\(.*\)$",
    r"^<.*>$",
    r"^(thank you|thanks for watching|you|bye|subtitles?|captions?|watching|"
    r"okay|yeah|amen|silence|bell rings?|music|applause|laughter|sighs?|coughs?|"
    r"hmm+|huh|uh+|um+|oh+|ah+|mhm+|tsk|shhh?|wow|ooh|aah|hey|hi|no|yes|"
    r"so|well|right|like|I mean|you know)\.?!?\s*$",
    r"^[.,?!\s\-—]+$",
    r"^.{1,2}$",
]


def clean_transcript(text: str) -> str:
    """Clean and validate STT transcript, rejecting hallucinations."""
    t = text.strip()
    t = re.sub(r'^[,.\-\s"—]+', "", t)
    t = re.sub(r'[,.\-\s"—]+$', "", t).strip()

    if len(t) < 3:
        return ""

    for pat in HALLUCINATION_PATTERNS:
        if re.match(pat, t, re.IGNORECASE):
            return ""

    # Reject single repeated words (e.g. "the the the")
    words = t.split()
    if len(words) >= 2 and len(set(w.lower() for w in words)) == 1:
        return ""

    return t


class STTProvider(ABC):
    """Abstract STT provider interface."""

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio to text."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the STT service is reachable."""
        ...


class WhisperServerSTT(STTProvider):
    """whisper.cpp HTTP server client.

    Sends PCM audio as WAV over multipart/form-data.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8081/inference",
        language: str = "en",
        temperature: float = 0.0,
    ) -> None:
        self.url = url
        self.language = language
        self.temperature = temperature

    @staticmethod
    def _pcm_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
        """Convert float32 PCM to normalized WAV bytes."""
        # 1. DC offset removal
        pcm = pcm - np.mean(pcm)

        # 2. Peak normalization (boost quiet speech up to target 0.9 peak without clipping)
        peak = np.max(np.abs(pcm))
        if peak > 1e-4:
            gain = min(0.9 / peak, 8.0)  # Max 8x gain boost
            pcm = pcm * gain

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(int16.tobytes())
        return buf.getvalue()

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        wav_bytes = self._pcm_to_wav(audio, sample_rate)
        audio_duration = len(audio) / sample_rate

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "temperature": str(self.temperature),
            "language": self.language,
            "no_timestamps": "true",
            "no_fallback": "true",
            "suppress_non_speech_tokens": "true",
            "prompt": "Voice conversation with a human user.",
        }

        try:
            resp = requests.post(self.url, files=files, data=data, timeout=10)
            if resp.status_code == 200:
                raw = resp.json().get("text", "").strip()
                cleaned = clean_transcript(raw)
                # Short audio (<1s) producing long text is hallucination
                if audio_duration < 1.0 and len(cleaned.split()) > 6:
                    return ""
                return cleaned
        except Exception as e:
            logger.error("stt_error", error=str(e))
        return ""

    async def health_check(self) -> bool:
        try:
            base = self.url.rsplit("/", 1)[0]
            resp = requests.get(f"{base}/", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
