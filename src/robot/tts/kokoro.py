"""Kokoro ONNX neural Text-to-Speech provider implementation.

Provides ultra-realistic, expressive neural voice synthesis using Kokoro-82M ONNX Runtime
weights and 54 pre-trained multi-accent voice embeddings at 24kHz sample rate.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np

from robot.telemetry import get_logger
from robot.tts.prosody import EmotionCategory, ProsodyModulation, ProsodyModulator

logger = get_logger("tts.kokoro")

# Language code mapping based on voice identifier prefixes
VOICE_LANG_MAP: dict[str, str] = {
    "af_": "en-us",
    "am_": "en-us",
    "bf_": "en-gb",
    "bm_": "en-gb",
    "ef_": "es",
    "em_": "es",
    "ff_": "fr-fr",
    "hf_": "hi",
    "hm_": "hi",
    "if_": "it",
    "im_": "it",
    "jf_": "ja",
    "jm_": "ja",
    "pf_": "pt-br",
    "pm_": "pt-br",
    "zf_": "cmn",
    "zm_": "cmn",
}

# Curated metadata dictionary for Kokoro voices
VOICE_METADATA: dict[str, dict[str, str]] = {
    # American Female
    "af_heart": {"name": "Heart", "gender": "female", "accent": "American", "style": "Warm & Natural (Recommended)"},
    "af_bella": {"name": "Bella", "gender": "female", "accent": "American", "style": "Energetic & Expressive"},
    "af_nicole": {"name": "Nicole", "gender": "female", "accent": "American", "style": "Crisp & Professional"},
    "af_sarah": {"name": "Sarah", "gender": "female", "accent": "American", "style": "Friendly & Conversational"},
    "af_sky": {"name": "Sky", "gender": "female", "accent": "American", "style": "Bright & Clear"},
    "af_alloy": {"name": "Alloy", "gender": "female", "accent": "American", "style": "Neutral & Balanced"},
    "af_aoede": {"name": "Aoede", "gender": "female", "accent": "American", "style": "Melodic & Elegant"},
    "af_jessica": {"name": "Jessica", "gender": "female", "accent": "American", "style": "Casual & Modern"},
    "af_kore": {"name": "Kore", "gender": "female", "accent": "American", "style": "Soft & Soothing"},
    "af_nova": {"name": "Nova", "gender": "female", "accent": "American", "style": "Dynamic & Confident"},
    "af_river": {"name": "River", "gender": "female", "accent": "American", "style": "Smooth & Calm"},
    # American Male
    "am_adam": {"name": "Adam", "gender": "male", "accent": "American", "style": "Deep & Confident"},
    "am_michael": {"name": "Michael", "gender": "male", "accent": "American", "style": "Authoritative & Classic"},
    "am_echo": {"name": "Echo", "gender": "male", "accent": "American", "style": "Warm & Resonance"},
    "am_eric": {"name": "Eric", "gender": "male", "accent": "American", "style": "Friendly & Casual"},
    "am_fenrir": {"name": "Fenrir", "gender": "male", "accent": "American", "style": "Intense & Bold"},
    "am_liam": {"name": "Liam", "gender": "male", "accent": "American", "style": "Young & Energetic"},
    "am_onyx": {"name": "Onyx", "gender": "male", "accent": "American", "style": "Grounded & Rich"},
    "am_puck": {"name": "Puck", "gender": "male", "accent": "American", "style": "Playful & Mischievous"},
    "am_santa": {"name": "Santa", "gender": "male", "accent": "American", "style": "Jovial & Deep"},
    # British Female
    "bf_emma": {"name": "Emma", "gender": "female", "accent": "British", "style": "Sophisticated & Polished"},
    "bf_isabella": {"name": "Isabella", "gender": "female", "accent": "British", "style": "Formal & Articulate"},
    "bf_alice": {"name": "Alice", "gender": "female", "accent": "British", "style": "Gentle & Clear"},
    "bf_lily": {"name": "Lily", "gender": "female", "accent": "British", "style": "Youthful & Bright"},
    # British Male
    "bm_george": {"name": "George", "gender": "male", "accent": "British", "style": "Refined & Distinguished"},
    "bm_lewis": {"name": "Lewis", "gender": "male", "accent": "British", "style": "Crisp & Intelligent"},
    "bm_daniel": {"name": "Daniel", "gender": "male", "accent": "British", "style": "Narrative & Deep"},
    "bm_fable": {"name": "Fable", "gender": "male", "accent": "British", "style": "Theatrical & Expressive"},
    # International Voices
    "hf_alpha": {"name": "Alpha (Hindi)", "gender": "female", "accent": "Hindi", "style": "Clear Hindi Female"},
    "hf_beta": {"name": "Beta (Hindi)", "gender": "female", "accent": "Hindi", "style": "Melodic Hindi Female"},
    "hm_omega": {"name": "Omega (Hindi)", "gender": "male", "accent": "Hindi", "style": "Deep Hindi Male"},
    "hm_psi": {"name": "Psi (Hindi)", "gender": "male", "accent": "Hindi", "style": "Conversational Hindi Male"},
    "jf_alpha": {"name": "Alpha (Japanese)", "gender": "female", "accent": "Japanese", "style": "Clear Japanese Female"},
    "jf_gongitsune": {"name": "Gongitsune", "gender": "female", "accent": "Japanese", "style": "Storyteller Japanese"},
    "jm_kumo": {"name": "Kumo (Japanese)", "gender": "male", "accent": "Japanese", "style": "Modern Japanese Male"},
    "ff_siwis": {"name": "Siwis (French)", "gender": "female", "accent": "French", "style": "Natural Parisian Female"},
    "ef_dora": {"name": "Dora (Spanish)", "gender": "female", "accent": "Spanish", "style": "Expressive Spanish Female"},
    "em_alex": {"name": "Alex (Spanish)", "gender": "male", "accent": "Spanish", "style": "Warm Spanish Male"},
    "if_sara": {"name": "Sara (Italian)", "gender": "female", "accent": "Italian", "style": "Vibrant Italian Female"},
    "im_nicola": {"name": "Nicola (Italian)", "gender": "male", "accent": "Italian", "style": "Classic Italian Male"},
    "zf_xiaobei": {"name": "Xiaobei (Mandarin)", "gender": "female", "accent": "Mandarin", "style": "Gentle Mandarin Female"},
    "zm_yunjian": {"name": "Yunjian (Mandarin)", "gender": "male", "accent": "Mandarin", "style": "Clear Mandarin Male"},
}


def _shift_pitch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Pitch-shift float32 audio via fast linear resampling without altering duration.

    Args:
        audio: 1D float32 numpy array representing audio waveform.
        factor: Resampling pitch shift factor (1.0 = unchanged).

    Returns:
        np.ndarray: Resampled float32 audio array.
    """
    if abs(factor - 1.0) < 0.01 or len(audio) == 0:
        return audio
    orig_len = len(audio)
    new_len = max(1, int(orig_len / factor))
    idx1 = np.linspace(0, orig_len - 1, new_len)
    resampled = np.interp(idx1, np.arange(orig_len), audio).astype(np.float32)
    idx2 = np.linspace(0, new_len - 1, orig_len)
    return np.interp(idx2, np.arange(new_len), resampled).astype(np.float32)


class KokoroTTS:
    """Kokoro-82M neural Text-to-Speech provider via ONNX Runtime.

    Features 54 pre-trained multi-accent voice embeddings at 24kHz sample rate,
    dynamic sentiment-driven prosody modulation, and an in-memory phrase cache.
    """

    SAMPLE_RATE: int = 24000

    def __init__(
        self,
        model_path: str | None = None,
        voices_path: str | None = None,
        default_voice: str = "af_heart",
        enable_prosody: bool = True,
    ) -> None:
        """Initialize Kokoro TTS provider.

        Args:
            model_path: Path to kokoro-v1.0.onnx model file.
            voices_path: Path to voices-v1.0.bin voice embeddings file.
            default_voice: Default voice identifier (e.g. 'af_heart').
            enable_prosody: Whether to enable dynamic sentiment prosody modulation.

        Raises:
            ImportError: If kokoro-onnx package is not installed.
            FileNotFoundError: If model or voices file cannot be found.
        """
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise ImportError(
                "kokoro-onnx package is not installed. Install via '.venv/bin/pip install kokoro-onnx'"
            ) from e

        self._model_path = self._resolve_path(
            model_path or "models/tts/kokoro/kokoro-v1.0.onnx",
            fallback_names=["kokoro-v1.0.onnx", "kokoro.onnx"],
        )
        self._voices_path = self._resolve_path(
            voices_path or "models/tts/kokoro/voices-v1.0.bin",
            fallback_names=["voices-v1.0.bin", "voices.bin"],
        )

        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"Kokoro model file not found: {self._model_path}")
        if not Path(self._voices_path).exists():
            raise FileNotFoundError(f"Kokoro voices file not found: {self._voices_path}")

        self._kokoro = Kokoro(self._model_path, self._voices_path)
        self._voices_list: list[str] = sorted(self._kokoro.get_voices())
        self._active_voice: str = default_voice if default_voice in self._voices_list else (
            self._voices_list[0] if self._voices_list else "af_heart"
        )
        self.prosody = ProsodyModulator(enabled=enable_prosody)
        self._phrase_cache: dict[str, tuple[np.ndarray, int]] = {}

        logger.info(
            "kokoro_initialized",
            model=self._model_path,
            voices_count=len(self._voices_list),
            active_voice=self._active_voice,
            sample_rate=self.SAMPLE_RATE,
        )

    def _resolve_path(self, target: str, fallback_names: list[str]) -> str:
        """Resolve path to local filesystem with fallback candidate checks.

        Args:
            target: Initial target path string.
            fallback_names: Alternative filename variants to inspect.

        Returns:
            str: Resolved absolute or relative path string.
        """
        p = Path(target)
        if p.exists():
            return str(p)
        candidates = [
            Path("models/tts/kokoro") / p.name,
            Path("models/tts") / p.name,
        ]
        for name in fallback_names:
            candidates.append(Path("models/tts/kokoro") / name)
            candidates.append(Path("models/tts") / name)

        for c in candidates:
            if c.exists():
                return str(c)
        return target

    def _resolve_voice(self, voice_model: str | None) -> str:
        """Resolve voice identifier, handling Piper ONNX model paths and aliases.

        Args:
            voice_model: Requested voice identifier or model filename.

        Returns:
            str: Validated Kokoro voice key (e.g. 'af_heart').
        """
        if not voice_model:
            return self._active_voice

        # Check if already a valid Kokoro voice
        if voice_model in self._voices_list:
            return voice_model

        # Handle voice names with prefix stripped (e.g. 'heart' -> 'af_heart')
        for v in self._voices_list:
            if v.endswith(f"_{voice_model.lower()}") or v == voice_model.lower():
                return v

        # Map legacy Piper voice models to best matching Kokoro counterpart
        lower_v = voice_model.lower()
        if "lessac" in lower_v or "ryan" in lower_v or "bryce" in lower_v:
            return "am_adam" if "am_adam" in self._voices_list else self._active_voice
        if "amy" in lower_v or "luna" in lower_v or "hfc" in lower_v or "kathleen" in lower_v:
            return "af_heart" if "af_heart" in self._voices_list else self._active_voice
        if "alan" in lower_v:
            return "bm_george" if "bm_george" in self._voices_list else self._active_voice

        return self._active_voice

    def _detect_lang(self, voice_id: str) -> str:
        """Determine ISO language code from Kokoro voice prefix.

        Args:
            voice_id: Kokoro voice identifier string.

        Returns:
            str: eSpeak language code (e.g. 'en-us', 'en-gb', 'ja', 'es').
        """
        for prefix, lang in VOICE_LANG_MAP.items():
            if voice_id.startswith(prefix):
                return lang
        return "en-us"

    def list_voices(self) -> list[dict[str, Any]]:
        """Return catalog of all available Kokoro voices with metadata.

        Returns:
            list[dict[str, Any]]: List of voice metadata dictionaries.
        """
        catalog: list[dict[str, Any]] = []
        for voice_id in self._voices_list:
            meta = VOICE_METADATA.get(voice_id, {})
            name = meta.get("name", voice_id.replace("_", " ").title())
            gender = meta.get("gender", "female" if voice_id.startswith(("af_", "bf_", "ef_", "ff_", "hf_", "if_", "jf_", "pf_", "zf_")) else "male")
            accent = meta.get("accent", "American" if voice_id.startswith(("af_", "am_")) else ("British" if voice_id.startswith(("bf_", "bm_")) else "International"))
            style = meta.get("style", f"{accent} {gender.title()}")

            catalog.append({
                "id": voice_id,
                "name": name,
                "engine": "kokoro",
                "gender": gender,
                "accent": accent,
                "style": style,
                "lang": self._detect_lang(voice_id),
                "is_active": (voice_id == self._active_voice),
            })
        return catalog

    def set_active_voice(self, voice_model: str) -> None:
        """Dynamically set the active voice model.

        Args:
            voice_model: Voice identifier to activate.
        """
        resolved = self._resolve_voice(voice_model)
        self._active_voice = resolved
        logger.info("active_kokoro_voice_changed", voice=resolved)

    @property
    def active_voice(self) -> str:
        """Return current active voice identifier."""
        return self._active_voice

    def _normalize_key(self, text: str, speed: float, pitch: float, voice_key: str) -> str:
        """Construct normalized deterministic cache key for phrase memoization.

        Args:
            text: Spoken text content.
            speed: Modulated speed multiplier.
            pitch: Modulated pitch multiplier.
            voice_key: Voice identifier string.

        Returns:
            str: Unique string key.
        """
        return f"kokoro|{voice_key}|{text.strip().lower()}|s={speed:.2f}|p={pitch:.2f}"

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
        """Synchronously synthesize text to float32 audio waveform at 24kHz.

        Args:
            text: Text string to synthesize.
            speed: Speech rate multiplier.
            pitch: Pitch shift multiplier.
            voice_model: Optional voice identifier override.
            noise_scale: Unused in Kokoro (retained for interface parity).
            noise_w: Unused in Kokoro (retained for interface parity).
            modulate: Whether to compute prosody modulation.
            personality_name: Optional personality name for sentiment bias.

        Returns:
            tuple[np.ndarray, int]: Tuple of (audio_float32, 24000).
        """
        from robot.tts import clean_text_for_speech

        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE

        eff_speed = speed
        eff_pitch = pitch

        if modulate:
            eff_speed, eff_pitch, _, _, _ = self.modulate_prosody(
                clean_text,
                base_speed=speed,
                base_pitch=pitch,
                personality_name=personality_name,
            )

        target_voice = self._resolve_voice(voice_model or self._active_voice)
        norm_key = self._normalize_key(clean_text, eff_speed, eff_pitch, target_voice)

        if norm_key in self._phrase_cache:
            audio, sr = self._phrase_cache[norm_key]
            return audio.copy(), sr

        lang = self._detect_lang(target_voice)

        # Kokoro clamps speed comfortably between 0.5 and 2.0
        clamped_speed = max(0.5, min(2.0, float(eff_speed)))

        try:
            samples, sr = self._kokoro.create(
                clean_text,
                voice=target_voice,
                speed=clamped_speed,
                lang=lang,
            )
            raw_audio = np.asarray(samples, dtype=np.float32)
        except Exception as e:
            logger.error(
                "kokoro_synthesis_failed",
                error=str(e),
                voice=target_voice,
                text=clean_text[:50],
            )
            return np.array([], dtype=np.float32), self.SAMPLE_RATE

        # Apply pitch shift if configured
        if abs(eff_pitch - 1.0) >= 0.02 and len(raw_audio) > 0:
            raw_audio = _shift_pitch(raw_audio, eff_pitch)

        res = (raw_audio, sr)

        # Cache short phrases (< 120 chars) in memory for instant replay
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
        """Pre-synthesize and store phrase in memory cache for 0ms instant playback.

        Args:
            text: Text phrase to warm up.
            speed: Speech speed.
            pitch: Speech pitch.
            voice_model: Optional voice identifier.
            noise_scale: Optional noise scale.
            noise_w: Optional noise_w parameter.

        Returns:
            tuple[np.ndarray, int]: Cached audio array and sample rate.
        """
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
        """Asynchronously synthesize text to audio waveform in worker thread.

        Args:
            text: Text string to synthesize.
            speed: Speech rate multiplier.
            pitch: Pitch shift multiplier.
            voice_model: Voice model identifier.
            noise_scale: Unused in Kokoro.
            noise_w: Unused in Kokoro.
            modulate: Whether to apply sentiment prosody modulation.
            personality_name: Active personality name.

        Returns:
            tuple[np.ndarray, int]: (audio_float32, 24000).
        """
        from robot.tts import clean_text_for_speech

        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE

        target_voice = self._resolve_voice(voice_model or self._active_voice)
        if not modulate:
            norm_key = self._normalize_key(clean_text, speed, pitch, target_voice)
            if norm_key in self._phrase_cache:
                audio, sr = self._phrase_cache[norm_key]
                return audio.copy(), sr

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
            voice_model: Optional voice identifier.
            noise_scale: Baseline noise scale.
            noise_w: Baseline noise_w duration.
            personality_name: Name of active personality.

        Returns:
            Tuple of (audio_float32, sample_rate, ProsodyModulation).
        """
        from robot.tts import clean_text_for_speech

        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE, ProsodyModulation()

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
        """Check if Kokoro TTS engine is healthy and ready to synthesize.

        Returns:
            bool: True if engine and voice embeddings are loaded.
        """
        return bool(self._kokoro is not None and self._voices_list)
