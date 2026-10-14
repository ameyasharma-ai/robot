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
    # Strip all bracket tags like [motion:wave], [emotion: happy], [action:nod], etc.
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    # Strip any lingering unclosed or orphaned brackets e.g. [motion: or wave]
    cleaned = re.sub(r"\[\s*(?:motion|action|emotion)?:?[^\]\n]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:wave|point|kick|nod|smile|happy|sad|raise_hand|stop)\s*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("[", "").replace("]", "")
    # Strip common stage directions in asterisks e.g. *waves*, *smiles*, *nods*, *laughs*, *sighs*
    stage_dir_pattern = (
        r"\*(?:waves?|raises?\s+(?:her\s+|his\s+|right\s+|left\s+)?hand|points?|kicks?|nods?|"
        r"smiles?|laughs?|sighs?|stops?|leans?|looks?\s+around|shrugs?|pauses?|blinks?|winks?|"
        r"whispers?|gasps?|chuckles?|scoffs?|coughs?|yawns?)[^*]*\*"
    )
    cleaned = re.sub(stage_dir_pattern, "", cleaned, flags=re.IGNORECASE)
    # Strip parenthetical stage directions e.g. (waves hand), (smiling), (chuckles)
    cleaned = re.sub(r"\([^)]*(?:wave|smile|nod|laugh|sigh|point|kick|wink|shrug|pause|gasp)[^)]*\)", "", cleaned, flags=re.IGNORECASE)
    # Unpack markdown bold/italic asterisks: **word**, *word* -> word
    cleaned = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", cleaned)
    # Strip markdown bold/italic underscores: __word__, _word_ -> word
    cleaned = re.sub(r"(?<!\w)_{1,3}([^_]+)_{1,3}(?!\w)", r"\1", cleaned)
    # Strip lingering raw asterisks, tildes, backticks
    cleaned = cleaned.replace("*", "").replace("~", "").replace("`", "")
    # Strip markdown headers (e.g. ## Title)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    # Strip markdown list bullets
    cleaned = re.sub(r"^\s*[-•*+]\s+", "", cleaned, flags=re.MULTILINE)
    # Strip unicode emojis so TTS never speaks out emoji names (e.g. ⚡ -> "HIGH VOLTAGE", 😊 -> "SMILING FACE")
    emoji_pattern = (
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U00002300-\U000023FF"  # Miscellaneous Technical
        "]+"
    )
    cleaned = re.sub(emoji_pattern, "", cleaned)
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
from typing import Any

from robot.tts.prosody import EmotionCategory, ProsodyModulation, ProsodyModulator


class PiperTTS(TTSProvider):
    """Piper ONNX TTS engine — local, low-latency, multi-voice synthesis.

    Produces WAV audio from text using ONNX Runtime with support for
    distinct voice models per personality, dynamic expressive prosody
    (length_scale, noise_scale, noise_w_scale, pitch shift), dynamic sentiment
    analysis, and an in-memory phrase cache for instant 0ms playback.
    """

    def __init__(self, model_path: str, enable_prosody: bool = True) -> None:
        self._voices: dict[str, piper.PiperVoice] = {}
        self._phrase_cache: dict[str, tuple[np.ndarray, int]] = {}
        self._default_model = self._resolve_model_path(model_path)
        self._active_model = self._default_model
        self.prosody = ProsodyModulator(enabled=enable_prosody)
        self._load_voice(self._default_model)
        logger.info("piper_initialized", default_model=self._default_model, dynamic_prosody=enable_prosody)

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

    def modulate_prosody(
        self,
        text: str,
        base_speed: float = 1.0,
        base_pitch: float = 1.0,
        base_noise_scale: float | None = None,
        base_noise_w: float | None = None,
        personality_name: str | None = None,
    ) -> tuple[float, float, float | None, float | None, ProsodyModulation]:
        """Calculate dynamic sentiment-driven prosody modulation for a phrase.

        Args:
            text: Text phrase to analyze.
            base_speed: Baseline speech speed.
            base_pitch: Baseline speech pitch.
            base_noise_scale: Baseline noise scale.
            base_noise_w: Baseline noise_w phoneme duration.
            personality_name: Active personality name.

        Returns:
            Tuple of (modulated_speed, modulated_pitch, modulated_noise_scale, modulated_noise_w, modulation).
        """
        return self.prosody.modulate_prosody(
            text,
            base_speed=base_speed,
            base_pitch=base_pitch,
            base_noise_scale=base_noise_scale,
            base_noise_w=base_noise_w,
            personality_name=personality_name,
        )

    def _synthesize_sync(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        voice_model: str | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
        modulate: bool = False,
        personality_name: str | None = None,
    ) -> tuple[np.ndarray, int]:
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), 16000

        eff_speed = speed
        eff_pitch = pitch
        eff_ns = noise_scale
        eff_nw = noise_w

        if modulate:
            eff_speed, eff_pitch, eff_ns, eff_nw, _ = self.modulate_prosody(
                clean_text,
                base_speed=speed,
                base_pitch=pitch,
                base_noise_scale=noise_scale,
                base_noise_w=noise_w,
                personality_name=personality_name,
            )

        target_voice_key = self._resolve_model_path(voice_model or self._active_model)
        norm_key = self._normalize_key(
            clean_text, eff_speed, eff_pitch, target_voice_key, eff_ns, eff_nw
        )
        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        voice = self._load_voice(target_voice_key)

        # Piper length_scale: higher values = slower speech; lower values = faster speech
        length_scale = 1.0 / max(0.5, min(2.0, eff_speed))
        syn_kwargs: dict[str, Any] = {"length_scale": length_scale}
        if eff_ns is not None:
            syn_kwargs["noise_scale"] = float(eff_ns)
        if eff_nw is not None:
            syn_kwargs["noise_w_scale"] = float(eff_nw)

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
        if abs(eff_pitch - 1.0) >= 0.02 and len(raw_audio) > 0:
            raw_audio = _shift_pitch(raw_audio, eff_pitch)

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
        modulate: bool = False,
        personality_name: str | None = None,
    ) -> tuple[np.ndarray, int]:
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), 16000

        target_voice_key = self._resolve_model_path(voice_model or self._active_model)
        if not modulate:
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
            modulate,
            personality_name,
        )

    async def synthesize_with_prosody(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        voice_model: str | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
        personality_name: str | None = None,
    ) -> tuple[np.ndarray, int, ProsodyModulation]:
        """Synthesize text using dynamic sentiment-driven prosody modulation.

        Args:
            text: Conversational speech text to synthesize.
            speed: Baseline personality speed.
            pitch: Baseline personality pitch.
            voice_model: Optional voice ONNX model path.
            noise_scale: Baseline noise scale.
            noise_w: Baseline noise_w duration.
            personality_name: Name of active personality.

        Returns:
            Tuple of (audio_float32, sample_rate, ProsodyModulation).
        """
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), 16000, ProsodyModulation()

        eff_speed, eff_pitch, eff_ns, eff_nw, modulation = self.modulate_prosody(
            clean_text,
            base_speed=speed,
            base_pitch=pitch,
            base_noise_scale=noise_scale,
            base_noise_w=noise_w,
            personality_name=personality_name,
        )
        audio, sr = await self.synthesize(
            clean_text,
            speed=eff_speed,
            pitch=eff_pitch,
            voice_model=voice_model,
            noise_scale=eff_ns,
            noise_w=eff_nw,
        )
        return audio, sr, modulation

    async def health_check(self) -> bool:
        return bool(self._voices)


__all__ = [
    "clean_text_for_speech",
    "split_into_phrases",
    "TTSProvider",
    "PiperTTS",
    "EmotionCategory",
    "ProsodyModulation",
    "ProsodyModulator",
]

