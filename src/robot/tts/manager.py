"""Unified Text-to-Speech coordinator and dynamic multi-engine manager.

Manages switching between Piper ONNX (ultra low-latency local) and Kokoro Neural
(ultra-realistic 24kHz neural voices), cataloging available voices, and state persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from robot.telemetry import get_logger
from robot.tts.prosody import EmotionCategory, ProsodyModulation, ProsodyModulator

logger = get_logger("tts.manager")

DATA_DIR = Path("data")
ACTIVE_ENGINE_FILE = DATA_DIR / "active_tts_engine.txt"
ACTIVE_VOICE_FILE = DATA_DIR / "active_tts_voice.txt"
CONFIG_FILE = Path("config/default.yaml")


class TTSManager:
    """Unified Text-to-Speech coordinator supporting multiple engines and dynamic switching."""

    def __init__(self, config: dict[str, Any] | None = None, enable_prosody: bool = True) -> None:
        """Initialize TTS manager and instantiate primary engine.

        Args:
            config: Optional system configuration dictionary.
            enable_prosody: Whether to enable sentiment-driven prosody modulation.
        """
        self.config = config or {}
        tts_cfg = self.config.get("tts", {})
        self.prosody = ProsodyModulator(enabled=enable_prosody)

        # Resolve persisted active engine
        persisted_engine = self._load_persisted_engine()
        default_engine = persisted_engine or tts_cfg.get("provider", "kokoro")
        self._active_engine_name: str = default_engine.lower()

        # Voice paths and instances dictionary
        self._piper_model_path: str = tts_cfg.get("model_path", "models/tts/en-us-lessac-medium.onnx")
        self._kokoro_model_path: str = tts_cfg.get("kokoro_model_path", "models/tts/kokoro/kokoro-v1.0.onnx")
        self._kokoro_voices_path: str = tts_cfg.get("kokoro_voices_path", "models/tts/kokoro/voices-v1.0.bin")

        self._engines: dict[str, Any] = {}
        self._init_primary_engine()

        # Load persisted active voice
        persisted_voice = self._load_persisted_voice()
        if persisted_voice:
            self.set_active_voice(persisted_voice)

        logger.info(
            "tts_manager_initialized",
            active_engine=self._active_engine_name,
            active_voice=self.get_active_voice(),
        )

    def _load_persisted_engine(self) -> str | None:
        """Read persisted active TTS engine name from data file.

        Returns:
            str | None: Name of persisted engine if file exists and non-empty.
        """
        if ACTIVE_ENGINE_FILE.exists():
            try:
                content = ACTIVE_ENGINE_FILE.read_text(encoding="utf-8").strip().lower()
                if content in ("piper", "kokoro"):
                    return content
            except Exception as e:
                logger.warning("failed_reading_active_tts_engine", error=str(e))
        return None

    def _load_persisted_voice(self) -> str | None:
        """Read persisted active TTS voice identifier from data file.

        Returns:
            str | None: Active voice key if file exists and non-empty.
        """
        if ACTIVE_VOICE_FILE.exists():
            try:
                content = ACTIVE_VOICE_FILE.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception as e:
                logger.warning("failed_reading_active_tts_voice", error=str(e))
        return None

    def _persist_state(self, engine_name: str, voice_name: str | None = None) -> None:
        """Save active TTS engine and voice to disk.

        Args:
            engine_name: Active engine identifier.
            voice_name: Optional voice identifier to save.
        """
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            ACTIVE_ENGINE_FILE.write_text(engine_name, encoding="utf-8")
            if voice_name:
                ACTIVE_VOICE_FILE.write_text(voice_name, encoding="utf-8")

            # Update default.yaml tts.provider key
            if CONFIG_FILE.exists():
                text = CONFIG_FILE.read_text(encoding="utf-8")
                import re
                new_text = re.sub(
                    r"(tts:\s*\n(?:\s+#[^\n]*\n)*\s+provider:\s*)[^\n]+",
                    r"\g<1>" + engine_name,
                    text,
                )
                if new_text != text:
                    CONFIG_FILE.write_text(new_text, encoding="utf-8")
        except Exception as e:
            logger.warning("failed_persisting_tts_state", error=str(e))

    def _init_primary_engine(self) -> None:
        """Attempt to instantiate active engine with fallback to available alternative."""
        target = self._active_engine_name
        try:
            self._get_or_load_engine(target)
        except Exception as e:
            logger.error("primary_tts_engine_failed", engine=target, error=str(e))
            fallback = "piper" if target == "kokoro" else "kokoro"
            try:
                self._get_or_load_engine(fallback)
                self._active_engine_name = fallback
                logger.info("fallback_tts_engine_activated", engine=fallback)
            except Exception as e2:
                logger.critical("all_tts_engines_failed", error=str(e2))

    def _get_or_load_engine(self, engine_name: str) -> Any:
        """Instantiate and cache TTS engine on demand.

        Args:
            engine_name: Target engine ('piper' or 'kokoro').

        Returns:
            Any: Instantiated TTS provider instance.

        Raises:
            ValueError: If engine_name is unsupported.
            RuntimeError: If engine fails to initialize.
        """
        name = engine_name.lower()
        if name in self._engines:
            return self._engines[name]

        if name == "kokoro":
            from robot.tts.kokoro import KokoroTTS
            inst = KokoroTTS(
                model_path=self._kokoro_model_path,
                voices_path=self._kokoro_voices_path,
                default_voice="af_heart",
            )
            self._engines["kokoro"] = inst
            return inst

        if name == "piper":
            from robot.tts import PiperTTS
            inst = PiperTTS(model_path=self._piper_model_path)
            self._engines["piper"] = inst
            return inst

        raise ValueError(f"Unsupported TTS engine: {engine_name}")

    def get_active_engine_name(self) -> str:
        """Return identifier of active TTS engine."""
        return self._active_engine_name

    def get_active_engine(self) -> Any:
        """Return active TTS provider instance."""
        return self._get_or_load_engine(self._active_engine_name)

    def get_active_voice(self) -> str:
        """Return currently active voice identifier for active engine."""
        eng = self.get_active_engine()
        if hasattr(eng, "active_voice"):
            return getattr(eng, "active_voice")
        if hasattr(eng, "_active_model"):
            p = Path(getattr(eng, "_active_model"))
            return p.stem
        return "default"

    def set_active_voice(self, voice_model: str) -> None:
        """Set active voice on current engine (with auto-detection if voice belongs to other engine).

        Args:
            voice_model: Voice identifier or model filename.
        """
        if not voice_model:
            return

        # Auto-detect if voice is Kokoro voice (starts with af_, am_, bf_, bm_, etc.)
        is_kokoro_voice = any(voice_model.startswith(p) for p in ("af_", "am_", "bf_", "bm_", "ef_", "em_", "ff_", "hf_", "hm_", "if_", "im_", "jf_", "jm_", "pf_", "pm_", "zf_", "zm_"))
        is_piper_voice = "piper" in voice_model.lower() or ".onnx" in voice_model.lower() or any(k in voice_model.lower() for k in ("lessac", "amy", "bryce", "kathleen", "hfc", "ryan", "alan"))

        if is_kokoro_voice and self._active_engine_name != "kokoro":
            try:
                self.switch_engine("kokoro", voice=voice_model)
                return
            except Exception:
                pass

        if is_piper_voice and self._active_engine_name != "piper" and not is_kokoro_voice:
            try:
                self.switch_engine("piper", voice=voice_model)
                return
            except Exception:
                pass

        eng = self.get_active_engine()
        if hasattr(eng, "set_active_voice"):
            eng.set_active_voice(voice_model)
            self._persist_state(self._active_engine_name, voice_name=self.get_active_voice())

    def switch_engine(self, engine_name: str, voice: str | None = None) -> dict[str, Any]:
        """Hot-swap active TTS engine and optional voice at runtime.

        Args:
            engine_name: Target engine ('piper' or 'kokoro').
            voice: Optional voice identifier to set after switching.

        Returns:
            dict[str, Any]: Status summary dictionary.

        Raises:
            ValueError: If target engine is unknown or cannot be loaded.
        """
        target = engine_name.lower().strip()
        if target not in ("piper", "kokoro"):
            raise ValueError(f"Invalid engine '{engine_name}'. Must be 'piper' or 'kokoro'.")

        eng = self._get_or_load_engine(target)
        self._active_engine_name = target

        if voice:
            if hasattr(eng, "set_active_voice"):
                eng.set_active_voice(voice)

        active_v = self.get_active_voice()
        self._persist_state(target, voice_name=active_v)

        logger.info("tts_engine_switched", engine=target, voice=active_v)
        return {
            "status": "switched",
            "engine": target,
            "voice": active_v,
            "sample_rate": getattr(eng, "SAMPLE_RATE", 16000),
        }

    def list_engines(self) -> list[dict[str, Any]]:
        """List status and capabilities of all supported TTS engines.

        Returns:
            list[dict[str, Any]]: List of engine metadata records.
        """
        kokoro_available = Path(self._kokoro_model_path).exists() and Path(self._kokoro_voices_path).exists()
        piper_available = Path("models/tts").exists() and any(Path("models/tts").glob("*.onnx"))

        return [
            {
                "id": "kokoro",
                "name": "Kokoro Neural",
                "description": "Ultra-realistic 82M neural voices with 54 expressive accents",
                "sample_rate": 24000,
                "is_available": kokoro_available,
                "is_active": (self._active_engine_name == "kokoro"),
                "total_voices": 54 if kokoro_available else 0,
                "recommended_voice": "af_heart",
            },
            {
                "id": "piper",
                "name": "Piper ONNX",
                "description": "Ultra low-latency local ONNX synthesis (<20ms TTFT)",
                "sample_rate": 16000,
                "is_available": piper_available,
                "is_active": (self._active_engine_name == "piper"),
                "total_voices": len(list(Path("models/tts").glob("*.onnx"))),
                "recommended_voice": "en-us-lessac-medium",
            },
        ]

    def list_voices(self, engine_name: str | None = None) -> list[dict[str, Any]]:
        """Return catalog of all available voices for specified engine or all engines.

        Args:
            engine_name: Optional filter ('piper' or 'kokoro').

        Returns:
            list[dict[str, Any]]: Comprehensive voice catalog.
        """
        results: list[dict[str, Any]] = []
        target = engine_name.lower().strip() if engine_name else None

        # 1. Kokoro Voices
        if target is None or target == "kokoro":
            try:
                kokoro_eng = self._get_or_load_engine("kokoro")
                if hasattr(kokoro_eng, "list_voices"):
                    results.extend(kokoro_eng.list_voices())
            except Exception as e:
                logger.warning("failed_listing_kokoro_voices", error=str(e))

        # 2. Piper Voices
        if target is None or target == "piper":
            tts_dir = Path("models/tts")
            if tts_dir.exists():
                active_piper = getattr(self._engines.get("piper"), "_active_model", "")
                for p in sorted(tts_dir.glob("*.onnx")):
                    stem = p.stem
                    is_active = (stem in active_piper) and (self._active_engine_name == "piper")
                    gender = "female" if any(w in stem for w in ("amy", "kathleen", "hfc", "luna")) else "male"
                    accent = "British" if "en-gb" in stem else "American"

                    clean_name = stem.replace("en-us-", "").replace("en-gb-", "").replace("-medium", "").replace("-low", "").replace("_", " ").title()

                    results.append({
                        "id": str(p),
                        "name": f"{clean_name} ({accent})",
                        "engine": "piper",
                        "gender": gender,
                        "accent": accent,
                        "style": f"{accent} {gender.title()}",
                        "lang": "en-gb" if "en-gb" in stem else "en-us",
                        "is_active": is_active,
                    })

        return results

    # ─── Transparent Delegation to Active Engine ──────────────────────────

    def modulate_prosody(
        self,
        text: str,
        base_speed: float = 1.0,
        base_pitch: float = 1.0,
        base_noise_scale: float | None = None,
        base_noise_w: float | None = None,
        personality_name: str | None = None,
    ) -> tuple[float, float, float | None, float | None, ProsodyModulation]:
        """Delegate prosody calculation to active engine."""
        eng = self.get_active_engine()
        if hasattr(eng, "modulate_prosody"):
            return eng.modulate_prosody(
                text,
                base_speed=base_speed,
                base_pitch=base_pitch,
                base_noise_scale=base_noise_scale,
                base_noise_w=base_noise_w,
                personality_name=personality_name,
            )
        return self.prosody.modulate_prosody(
            text,
            base_speed=base_speed,
            base_pitch=base_pitch,
            base_noise_scale=base_noise_scale,
            base_noise_w=base_noise_w,
            personality_name=personality_name,
        )

    def warm_phrase(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        voice_model: str | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Warm up phrase in memory cache on active engine."""
        eng = self.get_active_engine()
        if hasattr(eng, "warm_phrase"):
            return eng.warm_phrase(
                text,
                speed=speed,
                pitch=pitch,
                voice_model=voice_model,
                noise_scale=noise_scale,
                noise_w=noise_w,
            )
        return np.array([], dtype=np.float32), 16000

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
        """Synthesize text on active engine."""
        eng = self.get_active_engine()
        return await eng.synthesize(
            text,
            speed=speed,
            pitch=pitch,
            voice_model=voice_model,
            noise_scale=noise_scale,
            noise_w=noise_w,
            modulate=modulate,
            personality_name=personality_name,
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
        """Synthesize text with dynamic prosody modulation on active engine."""
        eng = self.get_active_engine()
        return await eng.synthesize_with_prosody(
            text,
            speed=speed,
            pitch=pitch,
            voice_model=voice_model,
            noise_scale=noise_scale,
            noise_w=noise_w,
            personality_name=personality_name,
        )

    async def health_check(self) -> bool:
        """Check operational health of active TTS provider."""
        eng = self.get_active_engine()
        if hasattr(eng, "health_check"):
            return await eng.health_check()
        return True
