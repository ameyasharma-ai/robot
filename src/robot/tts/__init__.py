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
    # Strip markdown bold/italic asterisks and underscores: **word**, *word*, __word__, _word_
    cleaned = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    cleaned = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", cleaned)
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

    def _normalize_key(self, text: str, speed: float, pitch: float) -> str:
        return f"{text.strip().lower()}|s={speed:.2f}|p={pitch:.2f}"

    def _synthesize_sync(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), 16000

        norm_key = self._normalize_key(clean_text, speed, pitch)
        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        # Piper length_scale: higher values = slower speech; lower values = faster speech
        # speed > 1.0 means faster speech, so length_scale = 1.0 / speed
        length_scale = 1.0 / max(0.5, min(2.0, speed))
        syn_config = piper.SynthesisConfig(length_scale=length_scale)

        chunks = []
        sr = 16000
        for chunk in self._voice.synthesize(clean_text, syn_config=syn_config):
            chunks.append(chunk.audio_float_array)
            if chunk.sample_rate:
                sr = chunk.sample_rate

        if not chunks:
            raw_audio = np.array([], dtype=np.float32)
        elif len(chunks) == 1:
            raw_audio = chunks[0]
        else:
            raw_audio = np.concatenate(chunks)

        # Apply pitch shift if configured (e.g. thin high-pitched voice for Cypher)
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
    ) -> tuple[np.ndarray, int]:
        """Pre-synthesize and store phrase in memory cache for 0ms instant playback."""
        return self._synthesize_sync(text, speed=speed, pitch=pitch)

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), 16000

        norm_key = self._normalize_key(clean_text, speed, pitch)
        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        import asyncio
        return await asyncio.to_thread(self._synthesize_sync, clean_text, speed, pitch)

    async def health_check(self) -> bool:
        return self._voice is not None

