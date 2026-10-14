"""Robot Engine — the central async orchestrator.

This is the brain of Robot Core. It wires together all subsystems and
runs the full-duplex voice loop:

    Mic → AEC/VAD → Whisper → Context Engine → Qwen → Piper → Speaker

The LLM reasons; Robot Core owns the robot.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import yaml

from robot.audio import AudioCapture, AudioPlayer, VoiceActivityDetector, WakeWordDetector
from robot.core.autonomy import AutonomyConfig, CuriosityEngine
from robot.core.context import ContextEngine, GROUNDING_POLICY, format_system_prompt
from robot.core.events import EventBus, Events
from robot.core.state import RobotMode, StateManager

from robot.config import load_system_config
from robot.expression import AsusAuraLighting
from robot.image import ImageGenerationService
from robot.llm import ClauseChunker, LlamaServerLLM, GeminiLLM, OpenAICompatibleLLM, LlamaServerManager
from robot.memory import MemoryManager
from robot.observability import ObservabilityHub
from robot.observability.models import SpanStatus
from robot.personality import PersonalityManager
from robot.perception import CameraCapture, PerceptionEngine
from robot.services import HealthMonitor
from robot.services.scheduler import TemporalScheduler
from robot.stt import StreamingSTT, WhisperServerSTT
from robot.telemetry import Telemetry, Timer, get_logger, setup_logging
from robot.tools import (
    CoordinatorState,
    PermissionManager,
    ToolCategory,
    ToolConversationCoordinator,
    ToolRegistry,
    ToolRouter,
)
from robot.tools.fast_path import FastPathRouter
from robot.tools.builtins import register_builtin_tools
from robot.tools.skills import ToolSkillManager
from robot.tts import PiperTTS

logger = get_logger("engine")


def _extract_motion_from_phrase(text: str) -> tuple[str, dict | None]:
    """Extract and strip predefined motion markers from LLM output for synchronized audio-motion execution.
    
    Supports:
    - [motion: wave] or [motion: raise_hand left]
    - Natural stage directions like *waves*, *raises right hand*, *points*, *kicks*
    """
    import re
    if not text:
        return text, None

    action_dict = None

    # 1. Explicit bracket marker: [motion: wave] or [character.execute_action action="wave"]
    m = re.search(r"\[(?:motion|action):\s*([a-zA-Z_]+)(?:\s+([a-zA-Z]+))?\]", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\[(?:character\.)?execute_action[\s\(]+(?:action=)?['\"]?([a-zA-Z_]+)['\"]?(?:[\s,]+(?:side=)?['\"]?([a-zA-Z]+)['\"]?)?", text, re.IGNORECASE)
    if m:
        act = m.group(1).lower()
        side = m.group(2).lower() if m.group(2) else "right"
        action_dict = {"action": act, "side": side}

    # 2. Asterisk stage directions: *waves*, *raises hand*, *points*
    if not action_dict:
        m2 = re.search(r"\*(waves?|raises?\s+(?:her\s+|his\s+|right\s+|left\s+)?hand|points?|kicks?|nods?|smiles?|stops?)\*", text, re.IGNORECASE)
        if m2:
            raw_act = m2.group(1).lower()
            act = "wave" if "wave" in raw_act else ("raise_hand" if "hand" in raw_act else ("point" if "point" in raw_act else ("kick" if "kick" in raw_act else ("nod" if "nod" in raw_act else "idle"))))
            side = "left" if "left" in raw_act else "right"
            action_dict = {"action": act, "side": side}

    # Clean out inner reasoning <thought>...</thought> blocks from reasoning models
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Clean out ALL stage directions, bracket tags (motions, emotions), and parentheses so TTS never speaks them
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    # Strip any lingering unclosed or orphaned brackets e.g. [motion: or wave]
    cleaned = re.sub(r"\[\s*(?:motion|action|emotion)?:?[^\]\n]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:wave|point|kick|nod|smile|happy|sad|raise_hand|stop)\s*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("[", "").replace("]", "")
    # Strip common physical stage directions in asterisks e.g. *waves*, *smiles*, *nods*
    stage_dir_pattern = (
        r"\*(?:waves?|raises?\s+(?:her\s+|his\s+|right\s+|left\s+)?hand|points?|kicks?|nods?|"
        r"smiles?|laughs?|sighs?|stops?|leans?|looks?\s+around|shrugs?|pauses?|blinks?|winks?|"
        r"whispers?|gasps?|chuckles?|scoffs?|coughs?|yawns?)[^*]*\*"
    )
    cleaned = re.sub(stage_dir_pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\([^)]*(?:wave|smile|nod|laugh|sigh|point|kick|wink|shrug|pause|gasp)[^)]*\)", "", cleaned, flags=re.IGNORECASE)
    # Unpack markdown bold/italic asterisks: **word**, *word* -> word (preserve actual spoken words)
    cleaned = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_{1,3}([^_]+)_{1,3}(?!\w)", r"\1", cleaned)
    cleaned = cleaned.replace("*", "").replace("~", "").replace("`", "")
    # Strip raw tool invocation code e.g. music.play(query="...") or system.calculate(...)
    cleaned = re.sub(r"\b[a-zA-Z_]+\.[a-zA-Z_]+\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()
    return cleaned, action_dict



class RobotEngine:
    """Main orchestration engine — owns all subsystems.

    Lifecycle:
    1. Load config
    2. Initialize subsystems
    3. Health check
    4. Start voice loop + API server
    5. Run until shutdown
    """

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        # Load configuration with .env overrides
        self.config = load_system_config(config_path)

        # Core subsystems
        self.state = StateManager()
        self.events = EventBus()
        self.telemetry = Telemetry()
        self.context = ContextEngine(self.config.get("context_budget"))

        # Providers (initialized in setup)
        self.llm_provider_type: str = self._get_persisted_llm_provider(
            self.config.get("llm", {}).get("provider", "local")
        )
        self.stt: WhisperServerSTT | None = None
        self.tts: PiperTTS | None = None
        self.llm: LlamaServerLLM | None = None
        self.memory: MemoryManager | None = None
        self.personalities = PersonalityManager(
            self.config.get("personality", {}).get("directory", "config/personalities")
        )

        # Audio
        self.capture: AudioCapture | None = None
        self.player: AudioPlayer | None = None
        self.vad: VoiceActivityDetector | None = None

        # Tools
        self.tool_registry = ToolRegistry()
        self.tool_router: ToolRouter | None = None
        self.skill_manager = ToolSkillManager()
        self.permissions = PermissionManager(event_bus=self.events)

        # Health
        self.health = HealthMonitor(self.state)

        # Observability & Metrics
        obs_cfg = self.config.get("observability", {})
        self.observability = ObservabilityHub(
            event_bus=self.events,
            max_history=obs_cfg.get("max_trace_history", 250),
            redact=obs_cfg.get("redact_sensitive_data", True),
        )

        # Image Generation Service
        img_cfg = self.config.get("image", {})
        self.image_service = ImageGenerationService(
            output_dir=img_cfg.get("output_dir", "data/images"),
            max_concurrent=img_cfg.get("max_concurrent_jobs", 1),
            default_width=img_cfg.get("default_width", 512),
            default_height=img_cfg.get("default_height", 512),
        )

        # Connect image service events to observability
        def _on_image_event(event_type, job):
            node_state = "RUNNING" if event_type in ("image.started", "image.queued") else ("COMPLETED" if event_type == "image.completed" else "IDLE")
            self.observability.set_pipeline_node("IMAGE", node_state)
            self.observability.emit_event(
                event_type=event_type,
                trace_id=job.trace_id,
                component="IMAGE",
                operation="GENERATE",
                status=job.status.value,
                duration_ms=job.duration_s * 1000,
                metadata={
                    "job_id": job.job_id,
                    "prompt": job.prompt,
                    "progress": job.progress,
                    "output_path": job.output_path,
                    "error": job.error,
                },
            )

        self.image_service.on_event(_on_image_event)

        # Runtime state
        self._session_id = str(uuid.uuid4())[:8]
        self._interrupt_event = asyncio.Event()
        self._speech_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        self._tts_queue: asyncio.Queue[tuple[str, asyncio.Event, dict | None] | None] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[tuple[Any, ...] | None] = asyncio.Queue(maxsize=16)
        self._is_speaking = False
        self._speech_ended_ts = 0.0
        self._last_bot_phrases: list[str] = []
        self.voice_enabled = True
        self.is_chat_mode = False
        self._pending_motion: dict | None = None
        self.tool_coordinator: ToolConversationCoordinator | None = None
        self._provider_switch_lock = asyncio.Lock()

        # Wake-Word & Proactive Autonomy
        audio_cfg = self.config.get("audio", {})
        self.wakeword = WakeWordDetector(
            wake_words=audio_cfg.get(
                "wake_words",
                ["hey lisa", "lisa", "hey robot", "wake up lisa", "wake up", "hi lisa"],
            ),
            sample_rate=audio_cfg.get("sample_rate", 16000),
        )
        autonomy_cfg = self.config.get("autonomy", {})
        self.curiosity = CuriosityEngine(
            engine=self,
            config=AutonomyConfig(
                enabled=autonomy_cfg.get("enabled", True),
                enable_static_timer=autonomy_cfg.get("enable_static_timer", False),
                idle_threshold_s=autonomy_cfg.get("idle_threshold_s", 300.0),
                cooldown_s=autonomy_cfg.get("cooldown_s", 240.0),
                check_interval_s=autonomy_cfg.get("check_interval_s", 3.0),
                quiet_after_user_activity_s=autonomy_cfg.get("quiet_after_user_activity_s", 120.0),
                arrival_absence_threshold_s=autonomy_cfg.get("arrival_absence_threshold_s", 180.0),
                arrival_debounce_s=autonomy_cfg.get("arrival_debounce_s", 3.0),
                enable_arrival_trigger=autonomy_cfg.get("enable_arrival_trigger", True),
            ),
        )
        sched_cfg = self.config.get("scheduler", {})
        self.scheduler = TemporalScheduler(
            engine=self,
            db_path=sched_cfg.get("database_path", "data/database/scheduler.db"),
        )


    async def setup(self) -> None:
        """Initialize all subsystems."""
        setup_logging(self.config.get("logging", {}).get("level", "INFO"))
        logger.info("engine_setup_start", session=self._session_id)

        # STT
        stt_cfg = self.config.get("stt", {})
        self.stt = WhisperServerSTT(
            url=stt_cfg.get("url", "http://127.0.0.1:8081/inference"),
            language=stt_cfg.get("language", "en"),
            temperature=stt_cfg.get("temperature", 0.0),
        )
        self.streaming_stt = StreamingSTT(
            provider=self.stt,
            sample_rate=self.config.get("audio", {}).get("sample_rate", 16000),
            chunk_seconds=stt_cfg.get("chunk_seconds", 0.8),
        )

        # TTS
        tts_cfg = self.config.get("tts", {})
        model_path = tts_cfg.get("model_path", "models/tts/en-us-lessac-medium.onnx")
        if Path(model_path).exists():
            self.tts = PiperTTS(model_path)
        else:
            logger.warning("tts_model_missing", path=model_path)

        # LLM (supports local llama-server on RTX 3050, Gemini Cloud 1M context, and OpenAI-Compatible)
        llm_cfg = self.config.get("llm", {})
        gemini_cfg = llm_cfg.get("gemini", {})
        openai_cfg = llm_cfg.setdefault("openai_compatible", {})
        persisted_openai_model = self._get_persisted_openai_model(openai_cfg.get("model", "gpt-4o"))
        openai_cfg["model"] = persisted_openai_model

        self.llm_provider_type = self._get_persisted_llm_provider(llm_cfg.get("provider", "local"))
        self._speech_started_ts = 0.0

        if self.llm_provider_type == "gemini":
            self.llm = GeminiLLM(
                api_key=gemini_cfg.get("api_key"),
                base_url=gemini_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                model=gemini_cfg.get("model", "gemini-2.0-flash"),
                fallback_models=gemini_cfg.get("fallback_models", ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-3.7-flash", "gemini-3.6-flash"]),
            )
            logger.info("gemini_llm_initialized_on_boot")
        elif self.llm_provider_type in ("openai_compatible", "openai"):
            resolved_base, resolved_key, resolved_model = self._resolve_openai_credentials(persisted_openai_model)
            self.llm = OpenAICompatibleLLM(
                api_key=resolved_key,
                base_url=resolved_base,
                model=resolved_model,
                context_window=openai_cfg.get("context_window", 128000),
                enable_prompt_caching=openai_cfg.get("enable_prompt_caching", True),
            )
            logger.info("openai_compatible_llm_initialized_on_boot", model=resolved_model, base_url=resolved_base)
        else:
            self.llm = LlamaServerLLM(
                base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
                model=llm_cfg.get("model", "Qwen3.5-4B-Q4_K_M"),
                enable_thinking=llm_cfg.get("enable_thinking", False),
            )
            logger.info("local_llama_server_llm_initialized_on_boot")

        # Memory
        mem_cfg = self.config.get("memory", {})
        history_path = self.config.get("history", {}).get(
            "database_path", "data/database/conversations.db"
        )
        self.memory = MemoryManager(mem_cfg, history_path)
        await self.memory.initialize()

        # Personalities
        self.personalities.load_all()
        default_personality = self.personalities.get_persisted_active(
            self.config.get("personality", {}).get("default", "neo")
        )
        self.personalities.set_active(default_personality)
        self.state.state.active_personality = default_personality
        pers = self.personalities.active
        if pers and getattr(pers, "voice_model", None) and hasattr(self.tts, "set_active_voice"):
            self.tts.set_active_voice(pers.voice_model)

        # Audio subsystem
        audio_cfg = self.config.get("audio", {})
        self.capture = AudioCapture(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            frame_size=audio_cfg.get("vad_frame_size", 512),
            pre_roll_ms=audio_cfg.get("pre_roll_ms", 500),
        )
        self.player = AudioPlayer(block_ms=audio_cfg.get("playback_block_ms", 40))
        self.vad = VoiceActivityDetector(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            normal_threshold=audio_cfg.get("normal_confidence", 0.50),
            barge_in_threshold=audio_cfg.get("barge_in_confidence", 0.55),
            echo_guard_threshold=audio_cfg.get("echo_guard_confidence", 0.85),
            energy_threshold=audio_cfg.get("energy_threshold", 0.005),
            barge_in_energy=audio_cfg.get("barge_in_energy", 0.008),
            echo_guard_energy=audio_cfg.get("echo_guard_energy", 0.02),
            echo_coupling_ratio=audio_cfg.get("echo_coupling_ratio", 0.40),
        )

        # EasyEffects background service & default robot equalizer profile
        try:
            from robot.tools.builtins.music import MusicPlayerService, ensure_easyeffects_running
            ensure_easyeffects_running()
            MusicPlayerService.get_instance().set_equalizer_preset("robot")
        except Exception as _e:
            logger.debug("easyeffects_init_warning", error=str(_e))

        # Tools
        self.tool_router = ToolRouter(self.tool_registry)
        self.fast_path_router = FastPathRouter(self.tool_registry, self.permissions)
        self.tool_coordinator = ToolConversationCoordinator(self, self.config)

        # Vision & Physical Perception
        vision_cfg = self.config.get("vision", {})
        if vision_cfg.get("enabled", True):
            camera = CameraCapture(
                device_index=vision_cfg.get("camera_index", 0),
                width=vision_cfg.get("width", 640),
                height=vision_cfg.get("height", 480),
                fps=vision_cfg.get("fps", 30),
            )
            self.perception = PerceptionEngine(
                camera=camera,
                yunet_model_path=vision_cfg.get("yunet_model_path", "models/vision/face_detection_yunet.onnx"),
                yolo_model_path=vision_cfg.get("yolo_model_path", "models/vision/yolov8n.pt"),
                target_fps=vision_cfg.get("perception_fps", 5.0),
                enable_motion=vision_cfg.get("enable_motion", True),
            )
        else:
            self.perception = None

        # Temporal Scheduler persistence setup
        await self.scheduler.setup()

        register_builtin_tools(self)
        if hasattr(self.llm, "set_canonical_tools") and hasattr(self, "tool_registry"):
            self.llm.set_canonical_tools(self.tool_registry.get_schemas())

        # Expression & RGB lighting (ASUS ROG Strix)
        expr_cfg = self.config.get("expression", {})
        lighting_cfg = expr_cfg.get("lighting", {})
        self.lighting = AsusAuraLighting(enabled=lighting_cfg.get("enabled", True))
        self.lighting.attach(self.state, self.player)

        # Health monitoring
        self.health.register("stt", self.stt)
        self.health.register("llm", self.llm)
        self.health.register("scheduler", self.scheduler)
        if self.tts:
            self.health.register("tts", self.tts)
        if self.memory and self.memory.mem0:
            self.health.register("memory", self.memory.mem0)

        # Initial health check
        health_results = await self.health.check_all()
        logger.info("initial_health", results=health_results)

        # Emit real-time state mode changes to WebSocket event stream
        async def _on_mode_change(old_mode, new_mode):
            try:
                await self.events.emit("mode", mode=new_mode.value)
                await self.events.emit("mode_changed", mode=new_mode.value, old_mode=old_mode.value)
            except Exception:
                pass
        self.state.subscribe(_on_mode_change)

        # Warm up subsystems for zero first-turn cold-start delay
        # CRITICAL: LLM KV cache warm-up MUST be the LAST step so its slot
        # prefix remains hot when the first user utterance arrives.
        try:
            # 1. Warm up Whisper STT connection & Vulkan backend
            if self.stt:
                try:
                    dummy_audio = np.zeros(1600, dtype=np.float32)
                    await self.stt.transcribe(dummy_audio, 16000)
                    logger.info("stt_whisper_warmed")
                except Exception as e:
                    logger.debug("stt_warmup_failed", error=str(e))

            # 2. Warm up Mem0 FastEmbed embedding ONNX model eagerly so turn 1 has 0ms model load delay
            if self.memory:
                t_m0 = time.perf_counter()
                await self.memory.get_relevant_memories("warmup", limit=1)
                m_ms = (time.perf_counter() - t_m0) * 1000
                logger.info("mem0_fastembed_warmed", duration_ms=round(m_ms, 1))

            # 3. Warm up Piper TTS and SoundDevice OutputStream with active personality voice
            if self.tts:
                t_t0 = time.perf_counter()
                pers = self.personalities.active
                voice_m = getattr(pers, "voice_model", None) if pers else None
                tts_cfg = getattr(pers, "tts_settings", {}) if pers else {}
                if voice_m and hasattr(self.tts, "set_active_voice"):
                    self.tts.set_active_voice(voice_m)
                audio_w, sr_w = await self.tts.synthesize(
                    "Online.",
                    speed=tts_cfg.get("speed", 1.0),
                    pitch=tts_cfg.get("pitch", 1.0),
                    voice_model=voice_m,
                    noise_scale=tts_cfg.get("noise_scale", 0.667),
                    noise_w=tts_cfg.get("noise_w", 0.8),
                )
                tts_ms = (time.perf_counter() - t_t0) * 1000
                logger.info("tts_piper_warmed", duration_ms=round(tts_ms, 1), voice=voice_m or "default")
                if self.player:
                    self.player._ensure_stream(sr_w)

                # Pre-warm acknowledgment cache (0ms latency on tool acknowledgments)
                ack_phrases = [
                    "Searching...",
                    "Looking that up for you...",
                    "One moment while I search.",
                    "Generating that image for you now.",
                    "Creating the image in the background.",
                    "Checking the clock...",
                    "Saving that to memory.",
                ]
                for ack in ack_phrases:
                    if hasattr(self.tts, "warm_phrase"):
                        self.tts.warm_phrase(ack)
                logger.info("tts_phrase_cache_prewarmed", count=len(ack_phrases))

            # 4. LAST: Warm up LLM KV cache slot with exact invariant system prompt prefix
            # This MUST be the final warmup step so the llama-server slot is freshly
            # loaded with the system prompt when the first user utterance arrives.
            # 4. LLM Warmup (only necessary for local llama-server KV cache slot 0)
            if self.llm_provider_type in ("gemini", "openai_compatible", "openai"):
                # Ensure local llama-server is stopped so 3.4GB VRAM is free
                if LlamaServerManager.is_running():
                    await LlamaServerManager.stop()
            else:
                await self.warm_up_llm()
        except Exception as e:
            logger.debug("subsystems_warmup_failed", error=str(e))

        logger.info("engine_setup_complete")

    def _get_persisted_llm_provider(self, default: str = "local") -> str:
        prov_file = Path("data/active_llm_provider.txt")
        if prov_file.exists():
            try:
                p = prov_file.read_text().strip().lower()
                if p in ("local", "gemini", "openai_compatible", "openai"):
                    return "openai_compatible" if p == "openai" else p
            except Exception:
                pass
        return default

    def _persist_llm_provider(self, provider: str) -> None:
        try:
            prov_file = Path("data/active_llm_provider.txt")
            prov_file.parent.mkdir(parents=True, exist_ok=True)
            prov_file.write_text(provider.strip().lower())
        except Exception as e:
            logger.warning("failed_to_persist_llm_provider", error=str(e))

    def _get_persisted_openai_model(self, default: str = "gpt-4o") -> str:
        model_file = Path("data/active_openai_model.txt")
        if model_file.exists():
            try:
                m = model_file.read_text(encoding="utf-8").strip().strip('"').strip("'")
                if m:
                    return m
            except Exception:
                pass
        env_m = os.environ.get("OPENAI_COMPATIBLE_MODEL", "").strip().strip('"').strip("'")
        if env_m:
            return env_m
        return default

    def _persist_openai_model(self, model: str) -> None:
        clean = model.strip().strip('"').strip("'")
        if not clean:
            return
        try:
            model_file = Path("data/active_openai_model.txt")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_text(clean, encoding="utf-8")
            if hasattr(self, "config") and isinstance(self.config, dict):
                self.config.setdefault("llm", {}).setdefault("openai_compatible", {})["model"] = clean
        except Exception as e:
            logger.warning("failed_to_persist_openai_model", error=str(e))

    def _resolve_openai_credentials(self, model: str | None = None) -> tuple[str, str, str]:
        """Resolve effective base_url, api_key, and model for OpenAI-compatible provider.

        Args:
            model: Optional model override.

        Returns:
            tuple[str, str, str]: (base_url, api_key, active_model)
        """
        openai_cfg = self.config.get("llm", {}).get("openai_compatible", {})
        active_model = model or self._get_persisted_openai_model(openai_cfg.get("model", "gpt-4o"))

        is_or_model = ":free" in active_model or any(
            active_model.startswith(p) for p in (
                "meta-llama/", "google/", "nvidia/", "liquid/", "apodex/", "thinkingmachines/", "mistralai/", "cohere/", "dots-studio/", "poolside/"
            )
        )
        is_ai_model = active_model.startswith("aion-labs/")

        if is_or_model:
            base_url = os.environ.get("OPENROUTER_BASE_URL") or openai_cfg.get("base_url") or "https://openrouter.ai/api/v1"
            api_key = os.environ.get("OPENROUTER_API_KEY") or openai_cfg.get("api_key", "")
        elif is_ai_model:
            base_url = os.environ.get("AICREDITS_BASE_URL") or openai_cfg.get("base_url") or "https://aicredits.in/v1"
            api_key = os.environ.get("AICREDITS_API_KEY") or openai_cfg.get("api_key", "")
        else:
            base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL") or openai_cfg.get("base_url") or "https://api.openai.com/v1"
            api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY") or openai_cfg.get("api_key", "")

        return base_url, api_key, active_model

    async def switch_llm_provider(self, target: str) -> dict:
        """Dynamically toggle between Local (RTX 3050), Cloud Gemini, and OpenAI-Compatible API.
        
        Args:
            target: Provider name ('local', 'gemini', or 'openai_compatible').

        Returns:
            dict: Status summary containing the active provider and metadata.
        """
        target = target.strip().lower()
        if target == "openai":
            target = "openai_compatible"
        if target not in ("local", "gemini", "openai_compatible"):
            return {
                "status": "error",
                "message": f"Invalid provider '{target}'. Must be 'local', 'gemini', or 'openai_compatible'.",
            }

        async with self._provider_switch_lock:
            if target == self.llm_provider_type:
                return {
                    "status": "ok",
                    "provider": self.llm_provider_type,
                    "mode": "low_latency" if target == "local" else "high_intelligence",
                    "already_active": True,
                }

            logger.info("switching_llm_provider", from_provider=self.llm_provider_type, to_provider=target)

            llm_cfg = self.config.get("llm", {})
            gemini_cfg = llm_cfg.get("gemini", {})
            openai_cfg = llm_cfg.get("openai_compatible", {})

            if target == "gemini":
                # 1. Kill local llama-server to free 3.4GB VRAM
                await LlamaServerManager.stop()
                # 2. Switch to Gemini provider
                self.llm = GeminiLLM(
                    api_key=gemini_cfg.get("api_key"),
                    base_url=gemini_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                    model=gemini_cfg.get("model", "gemini-2.0-flash"),
                    fallback_models=gemini_cfg.get("fallback_models", ["gemini-1.5-flash", "gemini-2.5-flash"]),
                )
                self.llm_provider_type = "gemini"
                self._persist_llm_provider("gemini")
                await self.events.emit("llm.provider_changed", provider="gemini", mode="high_intelligence")
                return {
                    "status": "ok",
                    "provider": "gemini",
                    "mode": "high_intelligence",
                    "model": getattr(self.llm, "model", "gemini-2.0-flash"),
                    "context_window": 1000000,
                    "vram_freed": True,
                }

            elif target == "openai_compatible":
                # 1. Kill local llama-server to free 3.4GB VRAM
                await LlamaServerManager.stop()
                # 2. Switch to OpenAI-Compatible provider
                resolved_base, resolved_key, resolved_model = self._resolve_openai_credentials()
                self.llm = OpenAICompatibleLLM(
                    api_key=resolved_key,
                    base_url=resolved_base,
                    model=resolved_model,
                    context_window=openai_cfg.get("context_window", 128000),
                    enable_prompt_caching=openai_cfg.get("enable_prompt_caching", True),
                )
                self.llm_provider_type = "openai_compatible"
                self._persist_llm_provider("openai_compatible")
                await self.events.emit("llm.provider_changed", provider="openai_compatible", mode="high_intelligence")
                return {
                    "status": "ok",
                    "provider": "openai_compatible",
                    "mode": "high_intelligence",
                    "model": resolved_model,
                    "context_window": openai_cfg.get("context_window", 128000),
                    "vram_freed": True,
                    "prompt_caching": openai_cfg.get("enable_prompt_caching", True),
                }

            else:
                # 1. Start local llama-server on RTX 3050
                started = await LlamaServerManager.start()
                if not started:
                    return {
                        "status": "error",
                        "message": "Failed to start local llama-server on port 8080. Check logs/llama-server.log.",
                    }
                # 2. Switch to LlamaServerLLM
                self.llm = LlamaServerLLM(
                    base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
                    model=llm_cfg.get("model", "Qwen3.5-4B-Q4_K_M"),
                    enable_thinking=llm_cfg.get("enable_thinking", False),
                )
                if hasattr(self, "tool_registry") and self.tool_registry and hasattr(self.llm, "set_canonical_tools"):
                    self.llm.set_canonical_tools(self.tool_registry.get_schemas())
                self.llm_provider_type = "local"
                self._persist_llm_provider("local")
                # 3. Warm up KV cache slot
                try:
                    await self.warm_up_llm()
                except Exception as e:
                    logger.warning("warmup_after_switch_failed", error=str(e))

                await self.events.emit("llm.provider_changed", provider="local", mode="low_latency")
                return {
                    "status": "ok",
                    "provider": "local",
                    "mode": "low_latency",
                    "model": getattr(self.llm, "model", "Qwen3.5-4B-Q4_K_M"),
                    "context_window": 16384,
                    "llama_running": True,
                }

    async def get_situation_context(self) -> dict[str, Any]:
        """Collect real-time situational telemetry for context grounding.

        Returns:
            Dictionary containing system battery, music player state, pending
            scheduler jobs, and compact physical world perception state.
        """
        situation: dict[str, Any] = {}
        try:
            from robot.tools.builtins.system import get_system_battery_status

            situation["battery"] = get_system_battery_status()
        except Exception:
            pass

        try:
            from robot.tools.builtins.music import MusicPlayerService

            situation["music_state"] = MusicPlayerService.get_instance().get_status()
        except Exception:
            pass

        try:
            if hasattr(self, "scheduler") and self.scheduler:
                situation["scheduler_jobs"] = await self.scheduler.list_jobs(status="pending")
        except Exception:
            pass

        if getattr(self, "perception", None) is not None:
            situation["world_state"] = self.perception.world_state.to_compact()

        return situation

    async def warm_up_llm(
        self,
        personality_name: str | None = None,
        silent_prompt: str | None = None,
    ) -> float:
        """Warm up llama-server slot KV cache with a silent contextually aware prompt.

        Ensures the very first user chat has a 100% prefix cache hit (f_keep > 0.95),
        reducing first-turn prompt evaluation from ~600ms down to ~120ms so TTFT
        matches subsequent chats. Emits a 'personality_ready' event upon completion.

        Args:
            personality_name: Name of the personality to warm up. If None, uses active.
            silent_prompt: Optional custom silent arrival/sync prompt. If None, uses
                a standard sensory synchronization banner.

        Returns:
            Duration in milliseconds taken to prime the KV cache slot.
        """
        if not self.llm:
            return 0.0

        pers = (
            self.personalities.get(personality_name)
            if personality_name
            else self.personalities.active
        )
        if not pers:
            pers = self.personalities.active

        pers_name = pers.name if pers else "default"
        sys_prompt = pers.system_prompt if pers else "You are a helpful robot."
        ns = getattr(pers, "memory_namespace", "default")
        user_profile = None
        all_facts = None
        mem = getattr(self, "memory", None)
        if mem and hasattr(mem, "get_core_user_profile"):
            try:
                user_profile = await mem.get_core_user_profile(user_id=ns)
            except Exception:
                pass
        if mem and hasattr(mem, "get_all_facts"):
            try:
                all_facts = await mem.get_all_facts(user_id=ns, limit=50)
            except Exception:
                pass

        ctx = getattr(self, "context", None)
        if not ctx:
            from robot.core.context import ContextEngine

            ctx = ContextEngine()

        situation = {}
        if hasattr(self, "get_situation_context") and callable(getattr(self, "get_situation_context")):
            sit_res = self.get_situation_context()
            if asyncio.iscoroutine(sit_res):
                situation = await sit_res
            elif isinstance(sit_res, dict):
                situation = sit_res
        silent_text = (
            silent_prompt
            or f"[System Context: Sensorial perception active and hardware pipeline initialized. Active persona: {pers_name}. Telemetry synchronized. Silent background warm-up verification.]"
        )

        warmup_messages = ctx.build_messages(
            system_prompt=sys_prompt,
            user_text=silent_text,
            user_profile=user_profile,
            all_facts=all_facts,
            situation=situation,
            high_intelligence=getattr(self, "high_intelligence_mode", False),
        )

        # Pre-warm system prompt prefix together with canonical tools schema
        t0 = time.perf_counter()
        try:
            reg = getattr(self, "tool_registry", None)
            tools_schema = reg.get_schemas() if reg is not None else None
            if hasattr(self.llm, "set_canonical_tools") and reg is not None:
                self.llm.set_canonical_tools(reg.get_schemas())
            async for _ in self.llm.stream_chat(warmup_messages, max_tokens=1, tools=tools_schema):
                pass
            w_ms = (time.perf_counter() - t0) * 1000
            logger.info("llm_kv_cache_warmed", personality=pers_name, duration_ms=round(w_ms, 1))
            await self.events.emit("personality_ready", personality=pers_name, duration_ms=round(w_ms, 1))
            return w_ms
        except Exception as e:
            logger.debug("llm_warmup_failed", error=str(e))
            return 0.0

    async def run(self) -> None:
        """Start the full-duplex voice loop and API server."""
        await self.setup()

        loop = asyncio.get_running_loop()
        audio_queue = await self.capture.start(loop)

        audio_cfg = self.config.get("audio", {})
        barge_in_active = audio_cfg.get("barge_in_enabled", False)
        barge_status = "\033[32mActive\033[0m" if barge_in_active else "\033[33mDeaf during speech (no self-interruption)\033[0m"

        print("\n" + "=" * 58)
        print("🤖  \033[1;35mROBOT CORE ONLINE — LIVE PIPELINE\033[0m")
        print("=" * 58)
        print(f" • \033[1mSession\033[0m:     {self._session_id}")
        print(f" • \033[1mPersona\033[0m:     {self.personalities.active.name if self.personalities.active else 'neo'}")
        print(f" • \033[1mServices\033[0m:    LLM: \033[32m✔\033[0m  STT: \033[32m✔\033[0m  TTS: \033[32m✔\033[0m")
        print(f" • \033[1mBarge-in\033[0m:    {barge_status}")
        print(f" • \033[1mLighting\033[0m:    \033[32mROG Aura (Active DBus Sync)\033[0m")
        print(f" • \033[1mDisplay UI\033[0m:  \033[1;36mhttp://127.0.0.1:8888\033[0m (Face + Dev Console)")
        print("=" * 58)
        print("🎙️  \033[2mListening for speech...\033[0m\n")

        # Start background image processing queue
        self.image_service.start()
        self.curiosity.start()
        await self.scheduler.start()

        tasks = [
            asyncio.create_task(self._mic_loop(audio_queue)),
            asyncio.create_task(self._response_loop()),
            asyncio.create_task(self._tts_synthesizer_loop()),
            asyncio.create_task(self._tts_player_loop()),
            asyncio.create_task(self.health.start()),
            asyncio.create_task(self._run_api()),
            asyncio.create_task(self.lighting.start()),
        ]

        if self.perception is not None:
            tasks.append(asyncio.create_task(self.perception.start()))

        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("engine_shutting_down")
        finally:
            self.curiosity.stop()
            self.image_service.stop()
            await self.scheduler.stop()

            if hasattr(self, "_api_server") and self._api_server:
                self._api_server.should_exit = True
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.capture.stop()
            self.player.stop()
            self.player.close()
            await self.lighting.stop()
            await self.health.stop()
            if self.perception is not None:
                await self.perception.stop()
            if self.memory:
                await self.memory.close()

    async def stop(self) -> None:
        """Signal the engine and embedded API server to gracefully shut down."""
        logger.info("engine_stop_signaled")
        self._running = False
        if hasattr(self, "_api_server") and self._api_server:
            self._api_server.should_exit = True

    def notify_user_activity(self) -> None:
        """Signal user interaction (chat text, voice, or UI action) to reset silence and quiet timers."""
        if hasattr(self, "curiosity") and self.curiosity:
            self.curiosity.notify_activity()

    async def _run_api(self) -> None:
        """Start the FastAPI server."""
        try:
            import uvicorn

            from robot.api import create_app

            app = create_app(self)
            api_cfg = self.config.get("api", {})
            config = uvicorn.Config(
                app,
                host=api_cfg.get("host", "127.0.0.1"),
                port=api_cfg.get("port", 8888),
                log_level="warning",
            )
            server = uvicorn.Server(config)
            self._api_server = server
            await server.serve()
        except asyncio.CancelledError:
            if hasattr(self, "_api_server") and self._api_server:
                self._api_server.should_exit = True
        except Exception as e:
            logger.error("api_server_error", error=str(e))

    # ─── Voice Pipeline ──────────────────────────────────────────────────

    async def _mic_loop(self, audio_queue: asyncio.Queue) -> None:
        """Main microphone processing loop with VAD and barge-in."""
        audio_cfg = self.config.get("audio", {})
        sample_rate = audio_cfg.get("sample_rate", 16000)
        silence_threshold = int(
            sample_rate * audio_cfg.get("silence_threshold_ms", 500) / 1000
            / audio_cfg.get("vad_frame_size", 512)
        )
        barge_in_enabled = audio_cfg.get("barge_in_enabled", False)
        barge_in_required = audio_cfg.get("barge_in_frames", 3)
        barge_in_cooldown = audio_cfg.get("barge_in_cooldown_s", 0.3)
        echo_guard_s = audio_cfg.get("echo_guard_s", 0.4)

        audio_buffer: list[float] = []
        is_speaking = False
        silence_frames = 0
        barge_in_consecutive = 0
        last_barge_in_time = 0.0
        was_barge_in = False

        while True:
            frame = await audio_queue.get()

            # If voice interaction is disabled (e.g. text-only mode via UI toggle), discard mic audio
            if not self.voice_enabled:
                is_speaking = False
                audio_buffer = []
                silence_frames = 0
                barge_in_consecutive = 0
                was_barge_in = False
                continue

            clean_chunk, rms = self.capture.process_frame(frame)

            # When the robot is speaking, if barge-in is disabled, stay completely deaf to all input
            if self._is_speaking or self.player.is_playing:
                if not barge_in_enabled:
                    # Drop frame and reset capture buffer to avoid picking up bot's voice
                    is_speaking = False
                    audio_buffer = []
                    silence_frames = 0
                    barge_in_consecutive = 0
                    was_barge_in = False
                    continue

            in_echo_guard = (time.time() - self._speech_ended_ts) < echo_guard_s

            ref_rms = self.player.current_rms
            is_voice, prob = self.vad.detect(
                clean_chunk,
                rms,
                (self._is_speaking or self.player.is_playing),
                in_echo_guard,
                ref_rms=ref_rms,
            )

            if is_voice:
                if self._is_speaking or self.player.is_playing:
                    # Barge-in detection (only reached if barge_in_enabled is True)
                    # When music is playing concurrently, speaker bleed from music into the microphone
                    # must not falsely interrupt the robot's speech.
                    music_active = False
                    try:
                        from robot.tools.builtins.music import MusicPlayerService
                        music_active = MusicPlayerService.get_instance().is_playing
                    except Exception:
                        pass

                    if music_active:
                        # Music is playing; do not trigger barge-in from music acoustics
                        barge_in_consecutive = 0
                    speech_age = time.time() - getattr(self, "_speech_started_ts", 0.0)
                    if speech_age < 0.45:
                        # Grace period: ignore acoustic onset bleed during the first 450ms of robot speech
                        # Online acoustic transfer calibration clamped to physical hardware bounds (<= 0.45)
                        if ref_rms > 0.04:
                            ratio = rms / max(0.01, ref_rms)
                            if ratio > self.vad._learned_coupling:
                                self.vad._learned_coupling = min(0.45, self.vad._learned_coupling * 0.85 + ratio * 0.15)
                        barge_in_consecutive = 0
                    elif (time.time() - last_barge_in_time) < barge_in_cooldown:
                        barge_in_consecutive = 0
                    else:
                        barge_in_consecutive += 1
                        if barge_in_consecutive >= barge_in_required:
                            print("\n⚡ \033[1;33m[BARGE-IN]\033[0m Interrupted bot playback!")
                            logger.info("barge_in_triggered")
                            self._interrupt_event.set()
                            self._is_speaking = False
                            self._speech_ended_ts = time.time()
                            self.player.stop()
                            # Propagate cancellation to observability traces
                            self.observability.cancel_active_traces()
                            await self.events.emit("barge_in", barge_in=True)
                            await self.state.set_mode(RobotMode.LISTENING)
                            barge_in_consecutive = 0
                            last_barge_in_time = time.time()
                            is_speaking = True
                            was_barge_in = True
                            audio_buffer = list(self.capture.get_pre_roll())
                else:
                    barge_in_consecutive = 0
                    if not is_speaking:
                        is_speaking = True
                        was_barge_in = False
                        pre_roll = list(self.capture.get_pre_roll())
                        audio_buffer = list(pre_roll)
                        self.streaming_stt.start()
                        self.streaming_stt.feed_frame(pre_roll)
                        await self.state.set_mode(RobotMode.LISTENING)

                if is_speaking:
                    audio_buffer.extend(frame.flatten().tolist())
                    silence_frames = 0
            else:
                barge_in_consecutive = max(0, barge_in_consecutive - 1)
                if is_speaking:
                    audio_buffer.extend(frame.flatten().tolist())
                    silence_frames += 1

                    if silence_frames >= silence_threshold:
                        end_time = time.time()
                        is_speaking = False
                        silence_frames = 0
                        is_barge = was_barge_in
                        was_barge_in = False

                        min_audio_len = int(sample_rate * 0.3)
                        if len(audio_buffer) >= min_audio_len:
                            audio_np = np.array(audio_buffer, dtype=np.float32)
                            buf_rms = float(np.sqrt(np.mean(audio_np**2)))
                            if buf_rms > 0.003:
                                # Start a new trace for this user interaction
                                trace = self.observability.start_trace(
                                    name="voice_turn",
                                    metadata={"barge_in": is_barge, "rms": round(buf_rms, 4)},
                                    )
                                self.observability.set_pipeline_node("MIC", "COMPLETED")
                                self.observability.set_pipeline_node("VAD", "COMPLETED")

                                stt_span = self.observability.start_span(
                                    component="STT",
                                    operation="TRANSCRIBE",
                                    trace_id=trace.trace_id,
                                    metadata={"audio_len_s": round(len(audio_buffer) / sample_rate, 2)},
                                )

                                with Timer(self.telemetry, "STT_FINAL_LATENCY"):
                                    text = await self.stt.transcribe(audio_np, sample_rate)

                                if text:
                                    stt_ms = self.telemetry.get_metric("STT_FINAL_LATENCY").last
                                    self.observability.finish_span(
                                        stt_span,
                                        metadata={"text": text, "latency_ms": round(stt_ms, 1)},
                                    )
                                    if not self._is_self_echo(text, is_barge_in=is_barge):
                                        if hasattr(self, "curiosity") and self.curiosity:
                                            self.curiosity.notify_activity()

                                        current_mode = self.state.state.mode
                                        is_wake, matched_word, remainder = self.wakeword.is_wake_word(text)
                                        is_standby = self.wakeword.is_standby_command(text)

                                        if current_mode == RobotMode.STANDBY:
                                            if is_wake:
                                                print(f"\n⚡ \033[1;32m[WAKE WORD DETECTED]\033[0m Matched '{matched_word}'!")
                                                await self.state.set_mode(RobotMode.LISTENING)
                                                await self.events.emit(Events.WAKE_WORD_DETECTED, wake_word=matched_word, text=text)
                                                if remainder:
                                                    print(f"\n💬 \033[1;36m[USER]\033[0m {remainder}  \033[90m(stt: {stt_ms:.0f}ms)\033[0m")
                                                    self.telemetry.record("STT_FINAL_LATENCY", stt_ms)
                                                    await self.events.emit("user_text", user_text=remainder, stt_ms=stt_ms)
                                                    await self._speech_queue.put((remainder, end_time))
                                                else:
                                                    await self._speech_queue.put(("[Wake word acknowledge]", end_time))
                                            else:
                                                print(f"💤 \033[90m[STANDBY IGNORED]\033[0m {text}")
                                                self.observability.finish_trace(status=SpanStatus.CANCELLED)
                                        elif is_standby:
                                            print(f"\n🛑 \033[1;33m[STANDBY REQUESTED]\033[0m {text}")
                                            await self.state.set_mode(RobotMode.STANDBY)
                                            await self.events.emit(Events.STANDBY, reason="user_command")
                                            if hasattr(self, "tool_coordinator") and self.tool_coordinator:
                                                self.tool_coordinator.reset()
                                            await self._speech_queue.put(("[Enter standby acknowledge]", end_time))
                                        else:
                                            clean_text = remainder if is_wake else text
                                            print(f"\n💬 \033[1;36m[USER]\033[0m {clean_text}  \033[90m(stt: {stt_ms:.0f}ms)\033[0m")
                                            self.telemetry.record(
                                                "STT_FINAL_LATENCY",
                                                stt_ms,
                                            )
                                            await self.events.emit("user_text", user_text=clean_text, stt_ms=stt_ms)
                                            await self._speech_queue.put((clean_text, end_time))
                                    else:
                                        print(f"🔇 \033[90m[ECHO FILTERED]\033[0m {text}")
                                        self.observability.finish_trace(status=SpanStatus.CANCELLED)
                                else:
                                    self.observability.finish_span(stt_span, status=SpanStatus.ERROR)
                                    self.observability.finish_trace(status=SpanStatus.ERROR)

                        audio_buffer = []

    async def _response_loop(self) -> None:
        """Process transcribed speech and generate responses."""
        while True:
            user_text, end_of_speech_time = await self._speech_queue.get()
            if not user_text:
                continue

            logger.info("user_speech", text=user_text)
            await self.state.set_mode(RobotMode.THINKING)
            turn = await self.state.increment_turn()

            # Create new interrupt event for this turn
            self._interrupt_event = asyncio.Event()

            # Stream LLM → TTS
            full_response = await self._stream_response(
                user_text, end_of_speech_time, self._interrupt_event
            )

            # Finish turn trace
            self.observability.finish_trace()

            # Store in memory isolated by personality workspace
            if full_response and self.memory:
                active_pers = self.personalities.active
                ns = getattr(active_pers, "memory_namespace", self.state.state.active_personality)
                await self.memory.store_turn(
                    self._session_id,
                    user_text,
                    full_response,
                    self.state.state.active_personality,
                    turn,
                    namespace=ns,
                    metadata={"provider": getattr(self, "llm_provider_type", "local")},
                )

            if self.state.state.mode != RobotMode.STANDBY:
                await self.state.set_mode(RobotMode.IDLE)

    async def _stream_response(
        self,
        user_text: str,
        end_of_speech_time: float,
        interrupt: asyncio.Event,
    ) -> str:
        """Stream LLM response to TTS as phrases form.

        Implements proactive tool execution: detects user intent for web search,
        memory write/recall, etc., executes the tool BEFORE calling the LLM, and
        injects tool results into context so the model speaks grounded answers.
        """
        # Fast system acknowledgment triggers
        if user_text == "[Wake word acknowledge]":
            ack = "I'm listening."
            await self._tts_queue.put((ack, interrupt, None))
            return ack
        if user_text == "[Enter standby acknowledge]":
            ack = "Entering standby mode. Say 'Hey Lisa' whenever you need me."
            await self._tts_queue.put((ack, interrupt, None))
            return ack
        if user_text.startswith("[Proactive check-in]: "):
            proactive_line = user_text[len("[Proactive check-in]: ") :].strip()
            await self._tts_queue.put((proactive_line, interrupt, None))
            await self._tts_queue.put(None)
            return proactive_line
        if user_text.startswith("[Scheduler alert]: "):
            alert_line = user_text[len("[Scheduler alert]: ") :].strip()
            await self._tts_queue.put((alert_line, interrupt, None))
            await self._tts_queue.put(None)
            return alert_line
        if (
            hasattr(self, "wakeword")
            and self.wakeword
            and self.wakeword.is_standby_command(user_text)
        ):
            await self.state.set_mode(RobotMode.STANDBY)
            await self.events.emit(Events.STANDBY, reason="user_command")
            if hasattr(self, "tool_coordinator") and self.tool_coordinator:
                self.tool_coordinator.reset()
            ack = "Entering standby mode. Say 'Hey Lisa' whenever you need me."
            await self._tts_queue.put((ack, interrupt, None))
            return ack

        # Check deterministic Fast-Path Router for sub-millisecond local utility commands (<50ms)
        if hasattr(self, "fast_path_router") and self.fast_path_router:
            intent = self.fast_path_router.classify(user_text)
            if intent:
                self.observability.set_trace_category("fast_path")
                fp_span = self.observability.start_span("ROUTING", "FAST_PATH", metadata={"tool": intent.tool_name})
                outcome, phrase = await self.fast_path_router.execute_intent(intent)
                self.observability.finish_span(
                    fp_span,
                    metadata={"tool": intent.tool_name, "status": outcome.get("status", "success")},
                )
                if phrase and not interrupt.is_set():
                    timing_ref = {"end_of_speech": end_of_speech_time, "t_tts_start": time.perf_counter()}
                    await self._tts_queue.put((phrase, interrupt, timing_ref, None))
                    await self._tts_queue.put(None)
                return phrase


        # Get personality
        # Restore speech volume in case it was ducked by music on a previous turn
        if self.player:
            self.player.unduck()
        personality = self.personalities.active
        system_prompt = personality.system_prompt if personality else ""
        ns = getattr(personality, "memory_namespace", "default")

        # In Cloud High-Intelligence mode (Gemini 1M / OpenAI-Compatible 128k), provide rich context (up to 80% tokens)
        # In Local mode, keep low-latency budget (3 turns, 5 memories)
        is_gemini = getattr(self, "llm_provider_type", "local") in ("gemini", "openai_compatible", "openai")

        # Adaptive Context Sizing:
        # Prevent TPM rate-limit throttling and high TTFT on simple conversational banter (e.g. "hi", "how are you").
        # For simple banter, maintain a fast, lean context (~1k tokens, TTFT < 1s).
        # For analytical, recall, search, or substantive queries, expand to full multi-session archive and facts (up to 800k tokens).
        clean_query = user_text.strip().lower()
        words = clean_query.split()
        banter_triggers = {"hi", "hello", "hey", "sup", "howdy", "good morning", "good evening", "good night", "thanks", "thank you", "bye", "ok", "okay", "yes", "no"}
        knowledge_triggers = ("remember", "recall", "past", "history", "search", "who", "what", "why", "how", "code", "file", "web", "weather", "news", "tell me", "explain", "analyze")
        is_banter = len(words) <= 3 and any(w in clean_query for w in banter_triggers) and not any(k in clean_query for k in knowledge_triggers)

        if is_gemini and is_banter:
            recent_turns_limit = 10
            memory_limit = 5
            deep_expansion = False
        elif is_gemini:
            recent_turns_limit = 100
            memory_limit = 50
            deep_expansion = True
        elif is_banter:
            recent_turns_limit = 6
            memory_limit = 3
            deep_expansion = False
        else:
            recent_turns_limit = 15
            memory_limit = 6
            deep_expansion = False

        # Retrieve relevant memories isolated by personality workspace
        memories = []
        all_facts = None
        conversation_archive = None
        if self.memory:
            with Timer(self.telemetry, "MEMORY_RETRIEVAL_LATENCY"):
                memories = await self.memory.get_relevant_memories(user_text, user_id=ns, limit=memory_limit)
                if is_gemini and deep_expansion:
                    try:
                        all_facts = await self.memory.get_all_facts(user_id=ns, limit=100)
                        conversation_archive = await self.memory.get_conversation_archive(limit=150, exclude_session_id=self._session_id)
                    except Exception as e:
                        logger.warning("gemini_context_expansion_failed", error=str(e))
                elif not is_gemini:
                    try:
                        all_facts = await self.memory.get_all_facts(user_id=ns, limit=50)
                    except Exception as e:
                        logger.debug("local_facts_retrieval_failed", error=str(e))

        # Get recent conversation
        recent = []
        if self.memory:
            recent = await self.memory.get_recent_conversation(self._session_id, recent_turns_limit)

        # Collect real-time situational telemetry
        situation = {}
        if hasattr(self, "get_situation_context") and callable(getattr(self, "get_situation_context")):
            sit_res = self.get_situation_context()
            if asyncio.iscoroutine(sit_res):
                situation = await sit_res
            elif isinstance(sit_res, dict):
                situation = sit_res
        world_state_compact = situation.get("world_state")

        # Retrieve Core User Profile Sheet
        user_profile = None
        if self.memory and hasattr(self.memory, "get_core_user_profile"):
            try:
                user_profile = await self.memory.get_core_user_profile(user_id=ns)
            except Exception as e:
                logger.debug("user_profile_fetch_failed", error=str(e))

        # Visual query image snapshot (set only if explicit image data provided)
        image_data_url = None

        # ─── Autonomous LLM Tool Selection ──────────────────────────────────
        # Provide tools to the LLM so it can autonomously decide when and how
        # to invoke tools with structured arguments, and read the outputs.
        tool_results = []
        selected_tool_names = self.tool_router.select_tools(user_text, recent_context=recent)
        logger.debug("tool_router_selected", tools=selected_tool_names, user_text=user_text[:80])

        # Classify request category for granular observability distributions
        network_tools = {
            "weather.forecast", "web.search", "news.headlines", "finance.quote",
            "knowledge.wikipedia", "translation.translate", "web.scrape_page", "network.speedtest",
        }
        if selected_tool_names and any(t in network_tools for t in selected_tool_names):
            self.observability.set_trace_category("network_dependent")
        elif selected_tool_names:
            self.observability.set_trace_category("tool_assisted")
        else:
            self.observability.set_trace_category("conversational")

        # ─── Build Messages with Tool Results ───────────────────────────────
        context_span = self.observability.start_span("CONTEXT", "BUILD")
        with Timer(self.telemetry, "CONTEXT_BUILD_LATENCY"):
            messages = self.context.build_messages(
                system_prompt=system_prompt,
                user_text=user_text,
                recent_conversation=recent,
                conversation_archive=conversation_archive,
                memories=memories,
                all_facts=all_facts,
                robot_state=self.state.state.to_compact(),
                world_state=world_state_compact,
                image_data_url=image_data_url,
                tool_results=tool_results if tool_results else None,
                high_intelligence=is_gemini,
                situation=situation,
                user_profile=user_profile,
            )
            estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages if isinstance(m.get("content"), str))
            self.telemetry.record("CONTEXT_TOKENS", estimated_tokens)
            self.telemetry.record("CONTEXT_INPUT_TOKENS", estimated_tokens)
            asyncio.create_task(
                self.events.emit(
                    "metrics",
                    metrics={
                        "tokens_in": estimated_tokens,
                        "tokens": f"{estimated_tokens} in / 0 out",
                    },
                )
            )
        self.observability.finish_span(context_span, metadata={"msg_count": len(messages), "tokens": estimated_tokens})

        # ─── Autonomous LLM Tool Invocation ────────────────────────────────
        # Tools are not called deterministically or via simple keyword regex logic.
        # Intelligence is routed through tools only when the LLM actively proposes a tool call.

        # If we have an image and the LLM explicitly supports multimodal (mmproj loaded),
        # convert the last user message to OpenAI vision content blocks
        llm_is_multimodal = self.config.get("llm", {}).get("multimodal", False)
        if llm_is_multimodal and image_data_url and messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user" and isinstance(messages[i].get("content"), str):
                    text_content = messages[i]["content"]
                    messages[i]["content"] = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ]
                    break

        # Stream LLM with eager first-clause release and background sentence chunking
        chunker = ClauseChunker(eager_first_chunk=True, min_first_words=1)
        full_response = ""
        ttft_recorded = False
        timing_ref = None
        t_llm_start = time.perf_counter()
        collected_tool_calls = []

        llm_span = self.observability.start_span(
            component="LLM",
            operation="GENERATE",
            metadata={"model": self.config.get("llm", {}).get("model", "Qwen3.5-4B-Q4_K_M")},
        )

        # Invariant tool schema preservation:
        # Retain canonical tools schema (~10,280 tokens at Token 0) on every turn in local mode
        # to guarantee 100% KV-cache prefix match (<250ms TTFT) while granting full autonomous agency.
        tools_schema = self.tool_registry.get_schemas() if getattr(self, "tool_registry", None) else None

        try:
            stream_max_tokens = 1024 if is_gemini else self.config["llm"].get("max_tokens", 300)
            async with self._provider_switch_lock:
                llm_instance = self.llm
            async for token in llm_instance.stream_chat(
                messages, 
                max_tokens=stream_max_tokens,
                tools=tools_schema
            ):
                if interrupt.is_set():
                    self.observability.finish_span(llm_span, status=SpanStatus.CANCELLED)
                    break

                if token.startswith("__TOOL_CALL__"):
                    try:
                        import json as _json
                        tc_data = _json.loads(token[len("__TOOL_CALL__"):])
                        if isinstance(tc_data, list):
                            collected_tool_calls.extend(tc_data)
                        elif isinstance(tc_data, dict):
                            collected_tool_calls.append(tc_data)
                    except Exception as e:
                        logger.warning("tool_call_parse_error", error=str(e))
                    continue

                if not ttft_recorded:
                    ttft = (time.perf_counter() - t_llm_start) * 1000
                    self.telemetry.record("LLM_TTFT", ttft)
                    ttft_recorded = True
                    timing_ref = {
                        "end_of_speech": end_of_speech_time,
                    }
                    self.observability.emit_event(
                        event_type="llm.first_token",
                        component="LLM",
                        operation="TTFT",
                        duration_ms=ttft,
                        metadata={"ttft_ms": round(ttft, 1)},
                    )

                full_response += token
                # Only stream phrases to TTS if the model is not emitting inline tool calls
                if "<tool_call>" not in full_response:
                    phrase = chunker.feed(token)
                    if phrase and not interrupt.is_set():
                        clean_phrase, inline_act = _extract_motion_from_phrase(phrase)
                        act_to_sync = inline_act or self._pending_motion
                        self._pending_motion = None
                        if clean_phrase:
                            if timing_ref:
                                timing_ref["t_tts_start"] = time.perf_counter()
                            await self._tts_queue.put((clean_phrase, interrupt, timing_ref, act_to_sync))
                            timing_ref = None

            # Check if model output inline <tool_call> tags or plain syntax e.g. music.play(query="...")
            if not collected_tool_calls:
                import re
                import json as _json
                if "<tool_call>" in full_response:
                    for match in re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", full_response, re.DOTALL):
                        try:
                            parsed = _json.loads(match)
                            collected_tool_calls.append({
                                "id": f"call_{len(collected_tool_calls)}_{int(time.time()*1000)}",
                                "type": "function",
                                "function": parsed if "name" in parsed else {"name": parsed.get("function", ""), "arguments": parsed.get("parameters", {})},
                            })
                        except Exception:
                            pass
                else:
                    # Fallback for plain syntax: tool.name(arg="val", ...)
                    known_tools = set(self.tool_registry.list_tools()) if getattr(self, "tool_registry", None) else set()
                    for t_name in known_tools:
                        pat = rf"(?:\[)?\b{re.escape(t_name)}\s*\((.*?)\)(?:\])?"
                        m = re.search(pat, full_response)
                        if m:
                            args_str = m.group(1).strip()
                            args_dict: dict[str, Any] = {}
                            if args_str:
                                for k, v in re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s\)]+))', args_str):
                                    val: Any = v[0] or v[1] or v[2]
                                    if isinstance(val, str) and val.lower() == "true":
                                        val = True
                                    elif isinstance(val, str) and val.lower() == "false":
                                        val = False
                                    elif isinstance(val, str) and val.isdigit():
                                        val = int(val)
                                    args_dict[k] = val
                            collected_tool_calls.append({
                                "id": f"call_inline_{int(time.time()*1000)}",
                                "type": "function",
                                "function": {"name": t_name, "arguments": args_dict},
                            })
                            logger.info("plaintext_tool_call_intercepted", tool=t_name, arguments=args_dict)
                            break

            if not interrupt.is_set():
                gen_ms = (time.perf_counter() - t_llm_start) * 1000
                self.observability.finish_span(
                    llm_span,
                    metadata={"response_len": len(full_response), "gen_ms": round(gen_ms, 1)},
                )

            # If the model autonomously proposed tool calls, coordinate holding conversation and execution
            if collected_tool_calls and not interrupt.is_set():
                # Execute any preceding tools in the chain before coordinating the primary/final tool
                for tc_item in collected_tool_calls[:-1]:
                    if interrupt.is_set():
                        break
                    f_info = tc_item.get("function", tc_item)
                    t_nm = f_info.get("name", "")
                    tc_id = tc_item.get("id", f"call_{int(time.time()*1000)}")
                    try:
                        import json as _json
                        r_args = f_info.get("arguments", "{}")
                        a_args = _json.loads(r_args) if isinstance(r_args, str) else (r_args or {})
                    except Exception:
                        a_args = {}
                    t_def = self.tool_registry.get(t_nm) if getattr(self, "tool_registry", None) else None
                    exec_result = None
                    if t_def and getattr(self, "permissions", None):
                        try:
                            exec_result = await self.permissions.execute(t_def, a_args)
                            logger.info("chained_tool_executed", tool=t_nm, result=str(exec_result)[:100])
                            # Sequential pacing so actions do not pop simultaneously
                            await asyncio.sleep(0.35)
                        except Exception as e:
                            exec_result = f"Error: {e}"
                            logger.warning("chained_tool_exec_error", tool=t_nm, error=str(e))

                    # Append this tool call and its execution result into messages for complete context
                    import json as _json
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": t_nm, "arguments": _json.dumps(a_args)},
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "name": t_nm,
                        "tool_call_id": tc_id,
                        "content": str(exec_result) if exec_result is not None else "Success",
                    })

                tc = collected_tool_calls[-1]
                func_info = tc.get("function", tc)
                tool_name = func_info.get("name", "")
                try:
                    import json as _json
                    raw_args = func_info.get("arguments", "{}")
                    args = _json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {}

                if self.tool_coordinator:
                    return await self.tool_coordinator.coordinate(
                        tool_name=tool_name,
                        tool_args=args,
                        category=ToolCategory.LLM_REQUIRED,
                        user_text=user_text,
                        end_of_speech_time=end_of_speech_time,
                        interrupt=interrupt,
                        messages=messages,
                    )

            # Flush remaining tokens only if no tool call pass was executed
            if not collected_tool_calls:
                remainder = chunker.flush()
                if remainder and not interrupt.is_set():
                    clean_rem, in_act = _extract_motion_from_phrase(remainder)
                    act_to_sync = in_act or self._pending_motion
                    self._pending_motion = None
                    if clean_rem:
                        if timing_ref and "t_tts_start" not in timing_ref:
                            timing_ref["t_tts_start"] = time.perf_counter()
                        await self._tts_queue.put((clean_rem, interrupt, timing_ref, act_to_sync))
                        timing_ref = None

            if not interrupt.is_set() and full_response:
                logger.info("bot_response", text=full_response.strip()[:100])

            # Sentinel to mark normal end of turn (only if not interrupted)
            if not interrupt.is_set():
                await self._tts_queue.put(None)

        except Exception as e:
            logger.error("llm_stream_error", error=str(e))
            if not interrupt.is_set():
                await self._tts_queue.put(None)

        e2e = (time.perf_counter() - t_llm_start) * 1000
        self.telemetry.record("LLM_GENERATION_LATENCY", e2e)

        output_tokens = max(1, len(full_response.split())) if full_response else 0
        self.telemetry.record("LLM_OUTPUT_TOKENS", output_tokens)
        asyncio.create_task(
            self.events.emit(
                "metrics",
                metrics={
                    "tokens_in": estimated_tokens,
                    "tokens_out": output_tokens,
                    "tokens": f"{estimated_tokens} in / {output_tokens} out",
                },
            )
        )

        return full_response.strip()

    async def _tts_synthesizer_loop(self) -> None:
        """Background pipeline worker: dequeue text phrases, synthesize with Piper, push to audio queue."""
        while True:
            item = await self._tts_queue.get()
            if item is None:
                await self._audio_queue.put(None)
                continue

            if len(item) == 4:
                phrase, interrupt, timing_ref, motion_action = item
            else:
                phrase, interrupt, timing_ref = item
                motion_action = None

            if interrupt.is_set() or not phrase or not self.tts:
                continue

            # Check if music has started playing — if so, apply speech fade-out
            try:
                from robot.tools.builtins.music import MusicPlayerService
                ms = MusicPlayerService.get_instance()
                if getattr(ms, "_music_active", False):
                    if self.player:
                        self.player.fade_out(rate=0.10)
                    logger.info("tts_skipped_music_playing", phrase=phrase[:40])
                    continue
            except Exception:
                pass

            self._last_bot_phrases.append(phrase)
            if len(self._last_bot_phrases) > 10:
                self._last_bot_phrases.pop(0)

            # Synthesize
            tts_span = self.observability.start_span(
                component="TTS",
                operation="SYNTHESIZE",
                metadata={"phrase_len": len(phrase)},
            )
            pers = self.personalities.active
            tts_cfg = pers.tts_settings if pers else {}
            speed = float(tts_cfg.get("speed", 1.0))
            pitch = float(tts_cfg.get("pitch", 1.0))
            voice_model = getattr(pers, "voice_model", None)
            noise_scale = tts_cfg.get("noise_scale", None)
            noise_w = tts_cfg.get("noise_w", None)
            pers_name = getattr(pers, "name", "neo") if pers else "neo"

            # Apply dynamic sentiment analysis and prosody modulation
            from robot.tts.prosody import ProsodyModulation
            eff_speed, eff_pitch, eff_ns, eff_nw = speed, pitch, noise_scale, noise_w
            detected_emotion = "neutral"
            modulation = ProsodyModulation()

            if hasattr(self.tts, "modulate_prosody"):
                try:
                    res = self.tts.modulate_prosody(
                        phrase,
                        base_speed=speed,
                        base_pitch=pitch,
                        base_noise_scale=noise_scale,
                        base_noise_w=noise_w,
                        personality_name=pers_name,
                    )
                    if isinstance(res, tuple) and len(res) == 5:
                        eff_speed, eff_pitch, eff_ns, eff_nw, modulation = res
                        detected_emotion = getattr(modulation, "emotion", "neutral")
                except Exception:
                    pass

            t_synth = time.perf_counter()
            audio, sr = await self.tts.synthesize(
                phrase,
                speed=eff_speed,
                pitch=eff_pitch,
                voice_model=voice_model,
                noise_scale=eff_ns,
                noise_w=eff_nw,
            )
            synth_ms = (time.perf_counter() - t_synth) * 1000

            self.telemetry.record("TTS_SYNTHESIS_LATENCY", synth_ms)
            self.observability.finish_span(
                tts_span,
                metadata={
                    "synth_ms": round(synth_ms, 1),
                    "emotion": detected_emotion,
                    "pitch_delta": modulation.pitch_delta,
                    "speed_delta": modulation.speed_delta,
                    "valence": modulation.valence,
                    "arousal": modulation.arousal,
                },
            )

            if interrupt.is_set():
                continue

            # If no explicit physical motion is queued, set avatar emotion from detected sentiment
            if motion_action is None and detected_emotion != "neutral":
                motion_action = {"action": "set_emotion", "emotion": detected_emotion}

            await self._audio_queue.put(
                (audio, sr, phrase, interrupt, timing_ref, motion_action, synth_ms, detected_emotion)
            )

    async def _tts_player_loop(self) -> None:
        """Continuous gapless playback worker from pre-synthesized audio queue."""
        while True:
            item = await self._audio_queue.get()

            if item is None:
                self._is_speaking = False
                self._speech_ended_ts = time.time()
                # If an interruption occurred and switched state to LISTENING, don't clobber it to IDLE
                if self.state.state.mode != RobotMode.LISTENING:
                    await self.state.set_mode(RobotMode.IDLE)
                await self.events.emit("robot_speech_ended", mode="idle")
                continue

            if len(item) >= 8:
                audio, sr, phrase, interrupt, timing_ref, motion_action, synth_ms, emotion = item[:8]
            else:
                audio, sr, phrase, interrupt, timing_ref, motion_action, synth_ms = item[:7]
                emotion = (motion_action or {}).get("emotion", "neutral")

            if interrupt.is_set():
                continue

            self._is_speaking = True
            if self._speech_started_ts == 0.0 or (time.time() - self._speech_ended_ts) > 1.0:
                self._speech_started_ts = time.time()

            await self.state.set_mode(RobotMode.SPEAKING)
            await self.events.emit("robot_phrase", robot_phrase=phrase, emotion=emotion)

            if timing_ref:
                t_start = timing_ref.get("t_tts_start", time.perf_counter() - synth_ms / 1000)
                first_audio_ms = (time.perf_counter() - t_start) * 1000
                e2e_ms = (time.time() - timing_ref["end_of_speech"]) * 1000
                self.telemetry.record("TTS_FIRST_AUDIO_LATENCY", first_audio_ms)
                self.telemetry.record("END_OF_SPEECH_TO_FIRST_AUDIO", e2e_ms)
                self.observability.emit_event(
                    event_type="tts.first_audio",
                    component="TTS",
                    operation="FIRST_AUDIO",
                    duration_ms=first_audio_ms,
                    metadata={"e2e_ms": round(e2e_ms, 1)},
                )
                logger.info(
                    "tts_timing",
                    synth_ms=round(synth_ms, 1),
                    first_audio_ms=round(first_audio_ms, 1),
                    e2e_ms=round(e2e_ms, 1),
                )
                print(f"🤖 \033[1;32m[ROBOT]\033[0m {phrase}  \033[90m(first audio: {e2e_ms:.0f}ms)\033[0m")
            else:
                print(f"   \033[32m...\033[0m {phrase}")

            # Trigger avatar motion tightly synchronized with speech playback
            if motion_action and not interrupt.is_set():
                try:
                    from robot.motion.planner import MotionPlanner
                    mp = MotionPlanner()
                    plan = mp.plan_action(motion_action)
                    avatar_span = self.observability.start_span(
                        component="AVATAR",
                        operation=motion_action.get("action", "motion").upper(),
                    )
                    await self.events.emit(
                        "character.action",
                        action=motion_action,
                        sequence=[a.to_dict() for a in plan],
                    )
                    self.observability.finish_span(avatar_span)
                    logger.info("audio_synced_motion_dispatched", action=motion_action)
                except Exception as e:
                    logger.debug("audio_synced_motion_error", error=str(e))

            completed = await self.player.play(audio, sr, interrupt)
            self._speech_ended_ts = time.time()

            if not completed:
                # Interrupted — drain both audio and TTS queues
                self._is_speaking = False
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                while not self._tts_queue.empty():
                    try:
                        self._tts_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if self.state.state.mode != RobotMode.LISTENING:
                    await self.state.set_mode(RobotMode.IDLE)
                await self.events.emit("robot_speech_ended", mode="idle")

    # ─── Echo Filter ─────────────────────────────────────────────────────

    def _is_self_echo(self, user_text: str, is_barge_in: bool = False) -> bool:
        """Detect if transcribed text is an echo of bot's own speech.

        If the user intentionally triggered a barge-in, never discard their utterance
        unless it is an exact, 100% duplicate of the phrase that was interrupted.
        """
        norm_user = re.sub(r"[^\w\s]", "", user_text.lower()).strip()
        if not norm_user:
            return True

        # If user explicitly interrupted, trust their speech unless it's pure bot playback
        if is_barge_in:
            for phrase in self._last_bot_phrases[-3:]:
                norm_bot = re.sub(r"[^\w\s]", "", phrase.lower()).strip()
                if norm_user == norm_bot:
                    return True
            return False

        for phrase in self._last_bot_phrases[-4:]:
            norm_bot = re.sub(r"[^\w\s]", "", phrase.lower()).strip()
            if not norm_bot:
                continue
            if norm_user == norm_bot:
                return True
            # Require strong phrase alignment (>85% matching words and at least 3 words)
            u_words = set(norm_user.split())
            b_words = set(norm_bot.split())
            if len(u_words) >= 3 and len(u_words & b_words) / len(u_words) >= 0.85:
                return True
        return False

    # ─── Text API ────────────────────────────────────────────────────────

    async def process_text_input(self, text: str, session_id: str | None = None) -> str:
        """Process text input (from API, non-voice)."""
        if not self.llm:
            return "LLM not available"

        target_sid = session_id or self._session_id or str(uuid.uuid4())[:8]

        # Start monotonic trace for API turn
        trace = self.observability.start_trace(
            name="text_turn",
            category="conversational",
            metadata={"session_id": target_sid},
        )

        print(f"\n💬 \033[1;36m[USER (TEXT)]\033[0m {text}")
        await self.events.emit("user_text", user_text=text, stt_ms=0.0)

        turn = await self.state.increment_turn()
        interrupt = asyncio.Event()  # No interrupt for text

        try:
            response = await self._stream_response(text, time.time(), interrupt)
        finally:
            self.observability.finish_trace()

        if self.memory:
            active_pers = self.personalities.active
            ns = getattr(active_pers, "memory_namespace", self.state.state.active_personality)
            await self.memory.store_turn(
                target_sid,
                text,
                response,
                self.state.state.active_personality,
                turn,
                namespace=ns,
                metadata={"provider": getattr(self, "llm_provider_type", "local")},
            )

        return response

    def _allocate_generation_budget(self, text: str, is_cloud: bool = False, has_tools: bool = True) -> int:
        """Dynamically allocate output token budget based on query complexity and requested length.

        Rationale:
        - Casual greetings/acknowledgments: 120-180 tokens (minimal latency, fast TTFT).
        - Standard conversational QA & Tool calls: 350-500 tokens.
        - Long-form explanations, essays, creative writing, stories, code: 800-1200 tokens (up to 2048 for cloud).
        """
        clean = (text or "").strip().lower()
        word_count = len(clean.split())

        long_triggers = (
            "essay", "explain in detail", "write a story", "detailed", "comprehensive",
            "compare and contrast", "write code", "analyze", "list all", "step by step",
            "outline", "break down", "report", "pros and cons", "summary of", "write a long"
        )
        is_long_form = any(trigger in clean for trigger in long_triggers) or word_count >= 30

        if is_long_form:
            return 2048 if is_cloud else 1024
        elif re.match(r"^(?:hi|hello|hey|sup|what's up|how are you|good morning)[!.,?\s]*$", clean):
            return 160
        elif has_tools:
            return 450
        else:
            return 350

    async def stream_text_chat(
        self,
        text: str,
        session_id: str | None = None,
        personality: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream conversational text turn with real-time tokens, token speed, tool events, and DB persistence.

        - Hardware microphone is automatically paused so it does not capture room audio.
        - Conversation history is isolated to session_id.
        - Persistent memory (Mem0 + SQLite facts) is shared across all instances.
        - Tool calls are executed and emitted live.
        - Saves to SQLite sessions and conversations tables upon completion.
        """
        if not self.llm:
            yield {"event": "error", "error": "LLM inference engine not available"}
            return

        target_session = session_id or self._session_id or str(uuid.uuid4())[:8]

        # Ensure mic is physically paused in text chat mode
        if hasattr(self, "capture") and self.capture and not self.capture.is_paused:
            self.capture.pause()

        # Switch personality if specified
        if personality and personality in self.personalities.list_personalities():
            if personality != self.state.state.active_personality:
                self.personalities.set_active(personality)
                self.state.state.active_personality = personality
                pers_obj = self.personalities.get(personality)
                if pers_obj and getattr(pers_obj, "voice_model", None) and hasattr(self.tts, "set_active_voice"):
                    self.tts.set_active_voice(pers_obj.voice_model)

        active_pers_name = self.state.state.active_personality
        active_pers = self.personalities.active
        ns = getattr(active_pers, "memory_namespace", active_pers_name)

        is_gemini = getattr(self, "llm_provider_type", "local") in ("gemini", "openai_compatible", "openai")

        # 1. Fetch recent conversation history strictly isolated to target_session
        recent_history = []
        if self.memory:
            recent_limit = 20 if is_gemini else 10
            recent_history = await self.memory.get_recent_conversation(target_session, limit=recent_limit)

        # 2. Shared persistent memory across all instances and voice
        retrieved_memories = []
        all_facts = None
        conversation_archive = None
        if self.memory:
            memory_limit = 20 if is_gemini else 5
            retrieved_memories = await self.memory.get_relevant_memories(text, user_id=ns, limit=memory_limit)
            try:
                all_facts = await self.memory.get_all_facts(user_id=ns, limit=100 if is_gemini else 5)
            except Exception:
                all_facts = None
            if is_gemini:
                try:
                    conversation_archive = await self.memory.get_conversation_archive(limit=150, exclude_session_id=target_session)
                except Exception:
                    conversation_archive = None

        # 3. World and situation telemetry grounding
        situation = {}
        if hasattr(self, "get_situation_context") and callable(getattr(self, "get_situation_context")):
            sit_res = self.get_situation_context()
            if asyncio.iscoroutine(sit_res):
                situation = await sit_res
            elif isinstance(sit_res, dict):
                situation = sit_res
        world_compact = situation.get("world_state")

        # Retrieve Core User Profile Sheet
        user_profile = None
        if self.memory and hasattr(self.memory, "get_core_user_profile"):
            try:
                user_profile = await self.memory.get_core_user_profile(user_id=ns)
            except Exception:
                user_profile = None

        pers_prompt = (
            active_pers.system_prompt
            if active_pers and getattr(active_pers, "system_prompt", None)
            else (getattr(self.personalities, "active_prompt", "") or "")
        )
        messages = self.context.build_messages(
            system_prompt=pers_prompt,
            user_text=text,
            recent_conversation=recent_history,
            conversation_archive=conversation_archive,
            memories=retrieved_memories,
            all_facts=all_facts,
            world_state=world_compact,
            situation=situation,
            high_intelligence=is_gemini,
            user_profile=user_profile,
        )

        tools = self.tool_registry.get_schemas() if hasattr(self, "tool_registry") and self.tool_registry else None
        if tools:
            # Chat mode is purely text-based without 3D humanoid avatar canvas;
            # filter out physical character and robotic movement tools.
            tools = [
                t for t in tools
                if not (
                    (isinstance(t, dict) and t.get("name", "").startswith("character."))
                    or (isinstance(t, dict) and t.get("function", {}).get("name", "").startswith("character."))
                    or (isinstance(t, dict) and t.get("family") in ("motion", "character", "avatar"))
                )
            ]
        clean_prompt = text.strip().lower()
        if re.match(r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|what's\s+up|yo)[!.,?\s]*$", clean_prompt):
            tools = []

        active_model_name = getattr(self.llm, "model", None)
        if not active_model_name and getattr(self, "llm_provider_type", "local") in ("openai_compatible", "openai"):
            active_model_name = getattr(self, "_get_persisted_openai_model", lambda d: d)("gpt-4o")

        yield {
            "event": "start",
            "session_id": target_session,
            "personality": active_pers_name,
            "provider": getattr(self, "llm_provider_type", "local"),
            "model": active_model_name,
        }

        dynamic_max_tokens = self._allocate_generation_budget(
            text,
            is_cloud=is_gemini or getattr(self, "llm_provider_type", "local") in ("openai_compatible", "openai"),
            has_tools=bool(tools),
        )

        t_start = time.perf_counter()
        ttft_ms = None
        token_count = 0
        full_text = ""
        tools_called = []
        collected_tool_calls = []

        async def _iterate_stream(gen):
            if hasattr(gen, "__aiter__"):
                async for item in gen:
                    yield item
            else:
                for item in gen:
                    yield item

        def _call_llm_stream(msgs, tools=None):
            import inspect
            sig = inspect.signature(self.llm.stream_chat)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            kwargs = {}
            if accepts_kwargs or "tools" in sig.parameters:
                kwargs["tools"] = tools
            if accepts_kwargs or "max_tokens" in sig.parameters:
                kwargs["max_tokens"] = dynamic_max_tokens
            return self.llm.stream_chat(msgs, **kwargs)

        try:
            # First pass generation
            async for chunk in _iterate_stream(_call_llm_stream(messages, tools=tools)):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000

                if chunk.startswith("__TOOL_CALL__"):
                    raw_tc = chunk[len("__TOOL_CALL__:"):] if chunk.startswith("__TOOL_CALL__:") else chunk[len("__TOOL_CALL__"):]
                    try:
                        import json as _json
                        tc_data = _json.loads(raw_tc)
                        if isinstance(tc_data, list):
                            collected_tool_calls.extend(tc_data)
                        else:
                            collected_tool_calls.append(tc_data)
                    except Exception:
                        pass
                    continue

                token_count += 1
                full_text += chunk
                elapsed = max(0.001, time.perf_counter() - t_start - (ttft_ms / 1000.0))
                tok_per_sec = round(token_count / elapsed, 1)

                yield {
                    "event": "token",
                    "token": chunk,
                    "token_index": token_count,
                    "ttft_ms": round(ttft_ms, 1),
                    "tok_per_sec": tok_per_sec,
                }

            # Check for <tool_call> tags in full_text if model produced text-based tool calls
            if not collected_tool_calls and "<tool_call>" in full_text:
                m_tc = re.search(r"<tool_call>(.*?)</tool_call>", full_text, re.DOTALL)
                if m_tc:
                    try:
                        import json as _json
                        parsed = _json.loads(m_tc.group(1).strip())
                        collected_tool_calls.append({
                            "id": f"call_{len(collected_tool_calls)}_{int(time.time()*1000)}",
                            "type": "function",
                            "function": parsed if "name" in parsed else {"name": parsed.get("function", ""), "arguments": parsed.get("parameters", {})},
                        })
                    except Exception:
                        pass

            # Check for bracketed tool calls e.g. [system.time], [perception.what_do_you_see]
            if not collected_tool_calls:
                bracket_matches = re.findall(r"\[\s*([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+)(?:\((.*?)\)|:\s*(.*?)|)\s*\]", full_text)
                for b_name, b_args1, b_args2 in bracket_matches:
                    if hasattr(self, "tool_registry") and self.tool_registry and self.tool_registry.get(b_name):
                        raw_arg_str = (b_args1 or b_args2 or "").strip()
                        parsed_args = {}
                        if raw_arg_str:
                            try:
                                import json as _json
                                parsed_args = _json.loads(raw_arg_str) if raw_arg_str.startswith("{") else {"query": raw_arg_str}
                            except Exception:
                                parsed_args = {"query": raw_arg_str}
                        import json as _json
                        collected_tool_calls.append({
                            "id": f"call_{len(collected_tool_calls)}_{int(time.time()*1000)}",
                            "type": "function",
                            "function": {"name": b_name, "arguments": _json.dumps(parsed_args) if isinstance(parsed_args, dict) else str(parsed_args)},
                        })

            # If tool calls were initiated, execute them and run second-pass continuation
            if collected_tool_calls:
                # Inform client that tools are executing so pre-tool preamble tokens are cleared
                yield {
                    "event": "tool_started",
                    "tools": [tc.get("function", tc).get("name", "") for tc in collected_tool_calls],
                    "clear_preamble": True,
                }

                # Cap consecutive tool calls to 2 per turn to prevent cascading runaway tool loops
                if len(collected_tool_calls) > 2:
                    logger.warning("capping_tool_calls_per_turn", original_count=len(collected_tool_calls), cap=2)
                    collected_tool_calls = collected_tool_calls[:2]

                for t_idx, tc in enumerate(collected_tool_calls):
                    if t_idx > 0:
                        # Sequential pacing so multiple tool calls do not pop simultaneously
                        await asyncio.sleep(0.35)

                    func_info = tc.get("function", tc)
                    t_name = func_info.get("name", "")
                    raw_args = func_info.get("arguments", "{}")
                    try:
                        import json as _json
                        t_args = _json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        t_args = {}

                    yield {
                        "event": "tool_call",
                        "tool": t_name,
                        "args": t_args,
                        "status": "running",
                    }

                    t_tool_start = time.perf_counter()
                    tool_def = self.tool_registry.get(t_name) if hasattr(self, "tool_registry") and self.tool_registry else None
                    if tool_def and hasattr(self, "permissions") and self.permissions:
                        try:
                            exec_res = await self.permissions.execute(tool_def, t_args)
                            t_val = exec_res.get("result")
                            if t_val is not None:
                                t_res = t_val
                            else:
                                t_res = exec_res.get("error") or "Tool completed with no output."
                        except Exception as e:
                            t_res = f"Error: {e}"
                    else:
                        t_res = "Tool executed."

                    t_dur_ms = (time.perf_counter() - t_tool_start) * 1000
                    tools_called.append({
                        "name": t_name,
                        "args": t_args,
                        "result": t_res,
                        "duration_ms": round(t_dur_ms, 1),
                    })

                    yield {
                        "event": "tool_result",
                        "tool": t_name,
                        "result": t_res,
                        "duration_ms": round(t_dur_ms, 1),
                        "status": "completed",
                    }

                # Second pass continuation with tool outputs — isolate assistant turn to tool proposal
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": collected_tool_calls,
                })
                import json as _json
                for tc, tool_info in zip(collected_tool_calls, tools_called):
                    res_raw = tool_info.get("result")
                    res_str = _json.dumps(res_raw) if isinstance(res_raw, (dict, list)) else str(res_raw or "Executed")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{int(time.time()*1000)}"),
                        "content": res_str,
                    })

                second_pass_text = ""
                async for chunk in _iterate_stream(_call_llm_stream(messages, tools=None)):
                    if chunk.startswith("__TOOL_CALL__"):
                        continue
                    token_count += 1
                    second_pass_text += chunk
                    elapsed = max(0.001, time.perf_counter() - t_start - ((ttft_ms or 0) / 1000.0))
                    tok_per_sec = round(token_count / elapsed, 1)

                    yield {
                        "event": "token",
                        "token": chunk,
                        "token_index": token_count,
                        "ttft_ms": round(ttft_ms or 0, 1),
                        "tok_per_sec": tok_per_sec,
                    }

                # Fallback: If second pass yielded no text, force conversational synthesis
                if not second_pass_text.strip():
                    tool_name_summary = ", ".join(t["name"] for t in tools_called) if tools_called else "system"
                    synth_prompt = (
                        f"You executed the following tools: {tool_name_summary}. "
                        "Synthesize the tool results now in character and answer the user's question directly. "
                        "Do not output raw JSON, code tags, or repeat tool parameters."
                    )
                    synth_messages = [
                        m for m in messages if m.get("role") != "tool"
                    ] + [{"role": "user", "content": synth_prompt}]
                    async for chunk in _iterate_stream(_call_llm_stream(synth_messages, tools=[])):
                        if chunk.startswith("__TOOL_CALL__"):
                            continue
                        token_count += 1
                        second_pass_text += chunk
                        elapsed = max(0.001, time.perf_counter() - t_start - ((ttft_ms or 0) / 1000.0))
                        tok_per_sec = round(token_count / elapsed, 1)
                        yield {
                            "event": "token",
                            "token": chunk,
                            "token_index": token_count,
                            "ttft_ms": round(ttft_ms or 0, 1),
                            "tok_per_sec": tok_per_sec,
                        }

                full_text = second_pass_text.strip() if second_pass_text else full_text.strip()

        except Exception as e:
            logger.error("stream_text_chat_error", error=str(e))
            yield {"event": "error", "error": str(e)}

        finally:
            t_end = time.perf_counter()
            total_dur_ms = (t_end - t_start) * 1000
            gen_time_s = max(0.001, (total_dur_ms - (ttft_ms or 0)) / 1000.0)
            final_speed = round(token_count / gen_time_s, 1)

            # Clean full_text of any raw tool names or bracketed metadata
            clean_resp = full_text or ""
            clean_resp = re.sub(r"\[\s*[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+(?:\([^\]]*\)|:[^\]]*|)\s*\]", "", clean_resp)
            clean_resp = re.sub(r"\[\s*(?:conversation|tools?|call|meta|internal|action)\b[^\]]*\]", "", clean_resp, flags=re.IGNORECASE)
            clean_resp = re.sub(r"^[A-Z_]+_HANDLING:\s*[^\n]*\n?", "", clean_resp, flags=re.MULTILINE)
            clean_resp = re.sub(r"\[\s*(?:motion|action|emotion|character\.)\s*:[^\]]*\]", "", clean_resp, flags=re.IGNORECASE)
            full_text = clean_resp.strip()

            # Store turn and shared memories in Database safely
            try:
                turn_num = await self.state.increment_turn()
                if self.memory:
                    meta = {
                        "ttft_ms": round(ttft_ms or 0, 1),
                        "tok_per_sec": final_speed,
                        "tokens": token_count,
                        "duration_ms": round(total_dur_ms, 1),
                        "provider": getattr(self, "llm_provider_type", "local"),
                        "model": active_model_name,
                        "tools": [t.get("name") for t in tools_called],
                    }
                    primary_tool = tools_called[0] if tools_called else None
                    await self.memory.store_turn(
                        session_id=target_session,
                        user_text=text,
                        bot_text=full_text,
                        personality=active_pers_name,
                        turn_number=turn_num,
                        namespace=ns,
                        tool_name=primary_tool.get("name") if primary_tool else None,
                        tool_result=str(primary_tool.get("result"))[:500] if primary_tool else None,
                        metadata=meta,
                    )
            except Exception as store_err:
                logger.warning("store_turn_failed_silently", error=str(store_err))

            yield {
                "event": "done",
                "session_id": target_session,
                "response": full_text,
                "ttft_ms": round(ttft_ms or 0, 1),
                "tok_per_sec": final_speed,
                "tokens": token_count,
                "duration_ms": round(total_dur_ms, 1),
                "tools": tools_called,
            }

    def trigger_interrupt(self) -> None:
        """Manually interrupt active LLM generation, tool execution, and audio playback."""
        logger.info("manual_interrupt_triggered")
        print("\n🛑 \033[1;31m[MANUAL STOP]\033[0m Interrupted active generation & speech playback!")
        self._interrupt_event.set()
        self._is_speaking = False
        self._speech_ended_ts = time.time()
        if self.player:
            self.player.stop()

        # Drain pending TTS and audio queue items
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

        # Cancel observability traces and reset pipeline nodes
        self.observability.cancel_active_traces()
        self.observability.set_pipeline_node("LLM", "CANCELLED")
        self.observability.set_pipeline_node("TTS", "CANCELLED")
        self.observability.set_pipeline_node("TOOLS", "CANCELLED")

        # Emit mode change to IDLE
        asyncio.create_task(self.state.set_mode(RobotMode.IDLE))
