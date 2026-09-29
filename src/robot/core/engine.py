"""Robot Engine — the central async orchestrator.

This is the brain of Robot Core. It wires together all subsystems and
runs the full-duplex voice loop:

    Mic → AEC/VAD → Whisper → Context Engine → Qwen → Piper → Speaker

The LLM reasons; Robot Core owns the robot.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot.audio import AudioCapture, AudioPlayer, VoiceActivityDetector
from robot.core.context import ContextEngine
from robot.core.events import EventBus, Events
from robot.core.state import RobotMode, StateManager
from robot.llm import ClauseChunker, LlamaServerLLM
from robot.memory import MemoryManager
from robot.personality import PersonalityManager
from robot.services import HealthMonitor
from robot.stt import WhisperServerSTT
from robot.telemetry import Telemetry, Timer, get_logger, setup_logging
from robot.tools import PermissionManager, ToolRegistry, ToolRouter
from robot.tools.builtins import register_builtin_tools
from robot.tts import PiperTTS

logger = get_logger("engine")


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
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Core subsystems
        self.state = StateManager()
        self.events = EventBus()
        self.telemetry = Telemetry()
        self.context = ContextEngine(self.config.get("context_budget"))

        # Providers (initialized in setup)
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
        self.permissions = PermissionManager()

        # Health
        self.health = HealthMonitor(self.state)

        # Runtime state
        self._session_id = str(uuid.uuid4())[:8]
        self._interrupt_event = asyncio.Event()
        self._speech_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        self._tts_queue: asyncio.Queue[tuple[str, asyncio.Event, dict | None] | None] = asyncio.Queue()
        self._is_speaking = False
        self._speech_ended_ts = 0.0
        self._last_bot_phrases: list[str] = []

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

        # TTS
        tts_cfg = self.config.get("tts", {})
        model_path = tts_cfg.get("model_path", "models/tts/en-us-lessac-medium.onnx")
        if Path(model_path).exists():
            self.tts = PiperTTS(model_path)
        else:
            logger.warning("tts_model_missing", path=model_path)

        # LLM
        llm_cfg = self.config.get("llm", {})
        self.llm = LlamaServerLLM(
            base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
            model=llm_cfg.get("model", "Qwen3.5-4B-Q4_K_M"),
            enable_thinking=llm_cfg.get("enable_thinking", False),
        )

        # Memory
        mem_cfg = self.config.get("memory", {})
        history_path = self.config.get("history", {}).get(
            "database_path", "data/database/conversations.db"
        )
        self.memory = MemoryManager(mem_cfg, history_path)
        await self.memory.initialize()

        # Personalities
        self.personalities.load_all()
        default_personality = self.config.get("personality", {}).get("default", "neo")
        self.personalities.set_active(default_personality)
        self.state.state.active_personality = default_personality

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
            normal_threshold=audio_cfg.get("normal_confidence", 0.40),
            barge_in_threshold=audio_cfg.get("barge_in_confidence", 0.60),
            echo_guard_threshold=audio_cfg.get("echo_guard_confidence", 0.60),
            energy_threshold=audio_cfg.get("energy_threshold", 0.002),
            barge_in_energy=audio_cfg.get("barge_in_energy", 0.006),
        )

        # Tools
        self.tool_router = ToolRouter(self.tool_registry)
        register_builtin_tools(self)

        # Health monitoring
        self.health.register("stt", self.stt)
        self.health.register("llm", self.llm)
        if self.tts:
            self.health.register("tts", self.tts)
        if self.memory and self.memory.mem0:
            self.health.register("memory", self.memory.mem0)

        # Initial health check
        health_results = await self.health.check_all()
        logger.info("initial_health", results=health_results)

        logger.info("engine_setup_complete")

    async def run(self) -> None:
        """Start the full-duplex voice loop and API server."""
        await self.setup()

        loop = asyncio.get_running_loop()
        audio_queue = await self.capture.start(loop)

        print("\n" + "=" * 58)
        print("🤖  \033[1;35mROBOT CORE ONLINE — FULL-DUPLEX LIVE\033[0m")
        print("=" * 58)
        print(f" • \033[1mSession\033[0m:     {self._session_id}")
        print(f" • \033[1mPersona\033[0m:     {self.personalities.active.name if self.personalities.active else 'neo'}")
        print(f" • \033[1mServices\033[0m:    LLM: \033[32m✔\033[0m  STT: \033[32m✔\033[0m  TTS: \033[32m✔\033[0m")
        print(f" • \033[1mBarge-in\033[0m:    Active (interrupt anytime to cut bot off)")
        print("=" * 58)
        print("🎙️  \033[2mListening for speech...\033[0m\n")

        tasks = [
            asyncio.create_task(self._mic_loop(audio_queue)),
            asyncio.create_task(self._response_loop()),
            asyncio.create_task(self._tts_player_loop()),
            asyncio.create_task(self.health.start()),
            asyncio.create_task(self._run_api()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("engine_shutting_down")
        finally:
            await self.capture.stop()
            self.player.stop()

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
            await server.serve()
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
        barge_in_required = audio_cfg.get("barge_in_frames", 10)
        barge_in_cooldown = audio_cfg.get("barge_in_cooldown_s", 1.5)
        echo_guard_s = audio_cfg.get("echo_guard_s", 1.0)

        audio_buffer: list[float] = []
        is_speaking = False
        silence_frames = 0
        barge_in_consecutive = 0
        last_barge_in_time = 0.0

        while True:
            frame = await audio_queue.get()
            clean_chunk, rms = self.capture.process_frame(frame)

            in_echo_guard = (time.time() - self._speech_ended_ts) < echo_guard_s

            is_voice, prob = self.vad.detect(
                clean_chunk, rms, self._is_speaking, in_echo_guard
            )

            if is_voice:
                if self._is_speaking:
                    # Barge-in detection
                    if (time.time() - last_barge_in_time) < barge_in_cooldown:
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
                            barge_in_consecutive = 0
                            last_barge_in_time = time.time()
                            is_speaking = True
                            audio_buffer = list(self.capture.get_pre_roll())
                else:
                    barge_in_consecutive = 0
                    if not is_speaking:
                        is_speaking = True
                        audio_buffer = list(self.capture.get_pre_roll())
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

                        min_audio_len = int(sample_rate * 0.4)
                        if len(audio_buffer) >= min_audio_len:
                            audio_np = np.array(audio_buffer, dtype=np.float32)
                            buf_rms = float(np.sqrt(np.mean(audio_np**2)))
                            if buf_rms > 0.003:
                                with Timer(self.telemetry, "STT_FINAL_LATENCY"):
                                    text = await self.stt.transcribe(audio_np, sample_rate)
                                if text:
                                    stt_ms = self.telemetry.get_metric("STT_FINAL_LATENCY").last
                                    if not self._is_self_echo(text):
                                        print(f"\n💬 \033[1;36m[USER]\033[0m {text}  \033[90m(stt: {stt_ms:.0f}ms)\033[0m")
                                        self.telemetry.record(
                                            "STT_FINAL_LATENCY",
                                            stt_ms,
                                        )
                                        await self._speech_queue.put((text, end_time))
                                    else:
                                        print(f"🔇 \033[90m[ECHO FILTERED]\033[0m {text}")

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

            # Store in memory
            if full_response and self.memory:
                await self.memory.store_turn(
                    self._session_id,
                    user_text,
                    full_response,
                    self.state.state.active_personality,
                    turn,
                )

            await self.state.set_mode(RobotMode.IDLE)

    async def _stream_response(
        self,
        user_text: str,
        end_of_speech_time: float,
        interrupt: asyncio.Event,
    ) -> str:
        """Stream LLM response to TTS as phrases form."""
        # Get personality
        personality = self.personalities.active
        system_prompt = personality.system_prompt if personality else ""

        # Retrieve relevant memories
        memories = []
        if self.memory:
            with Timer(self.telemetry, "MEMORY_RETRIEVAL_LATENCY"):
                memories = await self.memory.get_relevant_memories(user_text)

        # Get recent conversation
        recent = []
        if self.memory:
            recent = await self.memory.get_recent_conversation(self._session_id, 10)

        # Build context
        with Timer(self.telemetry, "CONTEXT_BUILD_LATENCY"):
            messages = self.context.build_messages(
                system_prompt=system_prompt,
                user_text=user_text,
                recent_conversation=recent,
                memories=memories,
                robot_state=self.state.state.to_compact(),
            )

        # Stream LLM
        chunker = ClauseChunker()
        full_response = ""
        ttft_recorded = False
        timing_ref = None
        t_llm_start = time.perf_counter()

        try:
            async for token in self.llm.stream_chat(messages, max_tokens=self.config["llm"].get("max_tokens", 300)):
                if interrupt.is_set():
                    break

                if token.startswith("__TOOL_CALL__"):
                    # Handle tool calls (future expansion)
                    continue

                if not ttft_recorded:
                    ttft = (time.perf_counter() - t_llm_start) * 1000
                    self.telemetry.record("LLM_TTFT", ttft)
                    ttft_recorded = True
                    timing_ref = {
                        "t_tts_start": time.perf_counter(),
                        "end_of_speech": end_of_speech_time,
                    }

                full_response += token
                phrase = chunker.feed(token)
                if phrase and not interrupt.is_set():
                    await self._tts_queue.put((phrase, interrupt, timing_ref))
                    timing_ref = None

            # Flush remaining
            remainder = chunker.flush()
            if remainder and not interrupt.is_set():
                await self._tts_queue.put((remainder, interrupt, timing_ref))

            if not interrupt.is_set() and full_response:
                logger.info("bot_response", text=full_response.strip()[:100])

            # Sentinel to mark end of turn
            await self._tts_queue.put(None)

        except Exception as e:
            logger.error("llm_stream_error", error=str(e))
            await self._tts_queue.put(None)

        e2e = (time.perf_counter() - t_llm_start) * 1000
        self.telemetry.record("LLM_GENERATION_LATENCY", e2e)

        return full_response.strip()

    async def _tts_player_loop(self) -> None:
        """Dequeue phrases, synthesize with Piper, play immediately."""
        while True:
            item = await self._tts_queue.get()

            if item is None:
                self._is_speaking = False
                self._speech_ended_ts = time.time()
                await self.state.set_mode(RobotMode.IDLE)
                continue

            phrase, interrupt, timing_ref = item

            if interrupt.is_set() or not phrase or not self.tts:
                continue

            self._last_bot_phrases.append(phrase)
            if len(self._last_bot_phrases) > 10:
                self._last_bot_phrases.pop(0)

            # Synthesize
            t_synth = time.perf_counter()
            audio, sr = await self.tts.synthesize(phrase)
            synth_ms = (time.perf_counter() - t_synth) * 1000

            if timing_ref:
                first_audio_ms = (time.perf_counter() - timing_ref["t_tts_start"]) * 1000
                e2e_ms = (time.time() - timing_ref["end_of_speech"]) * 1000
                self.telemetry.record("TTS_FIRST_AUDIO_LATENCY", first_audio_ms)
                self.telemetry.record("END_OF_SPEECH_TO_FIRST_AUDIO", e2e_ms)
                logger.info(
                    "tts_timing",
                    synth_ms=round(synth_ms, 1),
                    first_audio_ms=round(first_audio_ms, 1),
                    e2e_ms=round(e2e_ms, 1),
                )

            # Play
            self._is_speaking = True
            await self.state.set_mode(RobotMode.SPEAKING)

            if timing_ref:
                print(f"🤖 \033[1;32m[ROBOT]\033[0m {phrase}  \033[90m(first audio: {e2e_ms:.0f}ms)\033[0m")
            else:
                print(f"   \033[32m...\033[0m {phrase}")

            completed = await self.player.play(audio, sr, interrupt)

            if not completed:
                # Interrupted — drain TTS queue
                self._is_speaking = False
                self._speech_ended_ts = time.time()
                while not self._tts_queue.empty():
                    try:
                        leftover = self._tts_queue.get_nowait()
                        if leftover is None:
                            break
                    except asyncio.QueueEmpty:
                        break
            elif self._tts_queue.empty():
                self._speech_ended_ts = time.time()

    # ─── Echo Filter ─────────────────────────────────────────────────────

    def _is_self_echo(self, user_text: str) -> bool:
        """Detect if transcribed text is an echo of bot's own speech."""
        norm_user = re.sub(r"[^\w\s]", "", user_text.lower()).strip()
        if not norm_user:
            return True

        for phrase in self._last_bot_phrases[-6:]:
            norm_bot = re.sub(r"[^\w\s]", "", phrase.lower()).strip()
            if not norm_bot:
                continue
            if norm_user in norm_bot or norm_bot in norm_user:
                return True
            u_words = set(norm_user.split())
            b_words = set(norm_bot.split())
            if u_words and len(u_words & b_words) / len(u_words) >= 0.6:
                return True
        return False

    # ─── Text API ────────────────────────────────────────────────────────

    async def process_text_input(self, text: str) -> str:
        """Process text input (from API, non-voice)."""
        if not self.llm:
            return "LLM not available"

        turn = await self.state.increment_turn()
        interrupt = asyncio.Event()  # No interrupt for text

        response = await self._stream_response(text, time.time(), interrupt)

        if self.memory:
            await self.memory.store_turn(
                self._session_id,
                text,
                response,
                self.state.state.active_personality,
                turn,
            )

        return response
