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
        self._session = requests.Session()

    @staticmethod
    def _pcm_to_wav(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert float32 PCM array directly to 16-bit mono PCM WAV bytes in sub-millisecond time.

        Args:
            pcm: Audio sample buffer as 1D float32 NumPy array.
            sample_rate: Audio sampling frequency in Hz (typically 16000).

        Returns:
            bytes: Valid 44-byte RIFF/WAVE header followed by raw 16-bit PCM byte stream.
        """
        import struct

        # Peak normalization if signal is low
        peak = float(np.max(np.abs(pcm))) if len(pcm) > 0 else 0.0
        if 1e-4 < peak < 0.7:
            gain = min(0.9 / peak, 4.0)
            pcm = pcm * gain

        int16_data = (pcm * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
        num_bytes = len(int16_data)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            num_bytes + 36,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            num_bytes,
        )
        return header + int16_data

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe PCM audio buffer into text using the local whisper-server instance.

        Args:
            audio: 1D float32 PCM numpy array.
            sample_rate: Audio sample rate in Hz.

        Returns:
            str: Normalized, hallucination-filtered transcript string.
        """
        wav_bytes = self._pcm_to_wav(audio, sample_rate)
        audio_duration = len(audio) / sample_rate

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "temperature": "0.0",
            "language": self.language,
            "response_format": "json",
            "best_of": "1",
            "beam_size": "1",
        }

        try:
            import asyncio
            def _do_post() -> requests.Response:
                return self._session.post(self.url, files=files, data=data, timeout=5)

            resp = await asyncio.to_thread(_do_post)
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
            import asyncio
            base = self.url.rsplit("/", 1)[0]
            resp = await asyncio.to_thread(self._session.get, f"{base}/", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False


class StreamingSTT:
    """Manages streaming/chunked audio processing for low-latency parallel STT.

    Enables chunking speech while the user is actively talking, transcribing chunks
    in the background in parallel, and finalizing with only the short tail when speech ends.
    """

    def __init__(
        self,
        provider: STTProvider,
        sample_rate: int = 16000,
        chunk_seconds: float = 0.8,
    ) -> None:
        self.provider = provider
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self._current_buffer: list[float] = []
        self._completed_transcripts: list[str] = []
        self._in_flight_tasks: list[asyncio.Task[str]] = []
        self._is_active = False

    def start(self) -> None:
        """Initialize or reset state for a new speech turn."""
        self.reset()
        self._is_active = True

    def feed_frame(self, frame: np.ndarray | list[float]) -> asyncio.Task[str] | None:
        """Feed audio frame into streaming buffer. Dispatches chunk if threshold is reached."""
        if not self._is_active:
            self.start()

        if isinstance(frame, np.ndarray):
            self._current_buffer.extend(frame.flatten().tolist())
        else:
            self._current_buffer.extend(frame)

        if len(self._current_buffer) >= self.chunk_samples:
            return self._dispatch_chunk()
        return None

    def _dispatch_chunk(self) -> asyncio.Task[str]:
        """Find a clean energy dip split point and dispatch chunk transcription."""
        import asyncio
        buf = np.array(self._current_buffer, dtype=np.float32)
        search_window = int(self.sample_rate * 0.4)
        split_idx = len(buf)

        if len(buf) > search_window:
            recent_slice = buf[-search_window:]
            step = int(self.sample_rate * 0.02)  # 20ms steps
            min_energy = float("inf")
            best_offset = len(recent_slice)
            for i in range(0, len(recent_slice) - step, step):
                chunk_rms = float(np.sqrt(np.mean(recent_slice[i : i + step] ** 2)))
                if chunk_rms < min_energy:
                    min_energy = chunk_rms
                    best_offset = i + (step // 2)
            split_idx = len(buf) - search_window + best_offset

        chunk_audio = buf[:split_idx]
        self._current_buffer = buf[split_idx:].tolist()

        task = asyncio.create_task(self.provider.transcribe(chunk_audio, self.sample_rate))
        self._in_flight_tasks.append(task)
        return task

    def get_partial_transcript(self) -> str:
        """Return partial transcript of finished background chunks so far."""
        texts = list(self._completed_transcripts)
        for t in list(self._in_flight_tasks):
            if t.done() and not t.cancelled():
                try:
                    res = t.result()
                    if res:
                        texts.append(res)
                except Exception:
                    pass
        return " ".join(t for t in texts if t).strip()

    async def finalize(self, remaining_audio: np.ndarray | None = None) -> str:
        """Finalize the stream when speech ends, await in-flight tasks, transcribe tail, and merge."""
        import asyncio
        self._is_active = False
        parts: list[str] = list(self._completed_transcripts)

        # Collect in-flight tasks
        if self._in_flight_tasks:
            results = await asyncio.gather(*self._in_flight_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, str) and r:
                    parts.append(r)

        # Transcribe remaining audio tail
        tail_audio = None
        if remaining_audio is not None and len(remaining_audio) > 0:
            tail_audio = remaining_audio
        elif self._current_buffer:
            tail_audio = np.array(self._current_buffer, dtype=np.float32)

        if tail_audio is not None and len(tail_audio) >= int(self.sample_rate * 0.25):
            tail_text = await self.provider.transcribe(tail_audio, self.sample_rate)
            if tail_text:
                parts.append(tail_text)

        self.reset()
        merged = " ".join(parts).strip()
        return clean_transcript(merged) if merged else ""

    def reset(self) -> None:
        """Reset internal buffers and cancel pending tasks."""
        for t in self._in_flight_tasks:
            if not t.done():
                t.cancel()
        self._current_buffer.clear()
        self._completed_transcripts.clear()
        self._in_flight_tasks.clear()
        self._is_active = False


from robot.stt.manager import WhisperModelMetadata, WhisperServerManager

__all__ = [
    "STTProvider",
    "StreamingSTT",
    "WhisperModelMetadata",
    "WhisperServerManager",
    "WhisperServerSTT",
    "clean_transcript",
]

