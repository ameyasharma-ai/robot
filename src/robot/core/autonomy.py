"""Proactive Idle Autonomy & Curiosity Engine.

Provides autonomous background cognition, context-grounded observations,
and non-intrusive proactive interactions when the user is present but idle.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from robot.core.events import Events
from robot.core.state import RobotMode
from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("core.autonomy")


@dataclass
class AutonomyConfig:
    """Configuration settings for autonomous curiosity and proactive checks."""

    enabled: bool = True
    enable_static_timer: bool = False
    idle_threshold_s: float = 300.0
    cooldown_s: float = 240.0
    check_interval_s: float = 3.0
    require_human_presence: bool = False
    max_proactive_per_session: int = 15
    allowed_modes: list[str] = field(
        default_factory=lambda: ["idle", "standby"]
    )
    quiet_after_user_activity_s: float = 120.0
    arrival_absence_threshold_s: float = 180.0
    arrival_debounce_s: float = 3.0
    enable_arrival_trigger: bool = True


class CuriosityEngine:
    """Autonomous engine for proactive idle reasoning and dynamic contextual check-ins.

    Monitors system silence, user presence, time-of-day, and recent memory
    to formulate timely, natural observations without requiring direct user prompt.
    Suppresses interruptions when in chat mode or during recent user interaction.

    Attributes:
        engine: Reference to the parent RobotEngine.
        config: AutonomyConfig dataclass.
        last_activity_ts: Timestamp of the most recent user or bot activity.
        last_proactive_ts: Timestamp of the most recent proactive utterance.
        proactive_count: Number of proactive events initiated this session.
    """

    def __init__(
        self,
        engine: "RobotEngine",
        config: AutonomyConfig | None = None,
    ) -> None:
        """Initialize the CuriosityEngine.

        Args:
            engine: The central RobotEngine instance.
            config: Optional configuration overrides.
        """
        self.engine: "RobotEngine" = engine
        self.config: AutonomyConfig = config or AutonomyConfig()
        self.last_activity_ts: float = time.time()
        self.last_proactive_ts: float = 0.0
        self.proactive_count: int = 0
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None

        # Dynamic state tracking
        self._was_present: bool = False
        self._absence_start_ts: float | None = time.time()
        self._arrival_candidate_ts: float | None = None
        self._last_arrival_greeting_ts: float = 0.0

        logger.info(
            "curiosity_engine_initialized",
            enabled=self.config.enabled,
            enable_static_timer=self.config.enable_static_timer,
            enable_arrival_trigger=self.config.enable_arrival_trigger,
            cooldown=self.config.cooldown_s,
        )

    def notify_activity(self) -> None:
        """Record activity to reset the idle timer and quiet window."""
        self.last_activity_ts = time.time()

    def start(self) -> None:
        """Start the background autonomy loop."""
        if not self.config.enabled or self._running:
            return
        self._running = True
        self.last_activity_ts = time.time()
        self._task = asyncio.create_task(self._autonomy_loop())
        logger.info("curiosity_engine_started")

    def stop(self) -> None:
        """Stop the background autonomy loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("curiosity_engine_stopped")

    async def _autonomy_loop(self) -> None:
        """Evaluate dynamic situational conditions periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_s)
                if not self._running:
                    break

                now = time.time()

                # 1. Hard Suppression: Chat Mode or Voice Muted
                if (
                    not getattr(self.engine, "voice_enabled", True)
                    or getattr(self.engine, "is_chat_mode", False)
                ):
                    continue

                # 2. Hard Suppression: Recent User Activity (Typing, Voice, Tool)
                if (now - self.last_activity_ts) < self.config.quiet_after_user_activity_s:
                    continue

                # 3. Hard Suppression: Robot Speaking or Media Playing
                if (
                    getattr(self.engine, "_is_speaking", False)
                    or getattr(getattr(self.engine, "player", None), "is_playing", False)
                ):
                    continue

                # 4. Check Robot Mode
                current_mode = self.engine.state.state.mode.value
                if current_mode not in self.config.allowed_modes:
                    continue

                # 5. Persona Proactivity Behavioral Filtering
                active_pers = self.engine.personalities.active if hasattr(self.engine, "personalities") else None
                pers_proactivity = "medium"
                if active_pers and hasattr(active_pers, "behavior") and isinstance(active_pers.behavior, dict):
                    pers_proactivity = active_pers.behavior.get("proactivity", "medium")

                # Low proactivity personas (like GLaDOS) completely suppress proactive chatter
                if pers_proactivity == "low":
                    continue

                # 6. Session limit & global cooldown
                if (
                    self.proactive_count >= self.config.max_proactive_per_session
                    or (now - self.last_proactive_ts) < self.config.cooldown_s
                ):
                    continue

                # 7. Dynamic Condition A: Visual Arrival after Extended Absence
                if self.config.enable_arrival_trigger:
                    triggered_arrival = await self._check_visual_arrival(now, active_pers)
                    if triggered_arrival:
                        continue

                # 8. Dynamic Condition B: Static Timer (ONLY IF explicitly enabled by user)
                if self.config.enable_static_timer:
                    idle_duration = now - self.last_activity_ts
                    if idle_duration >= self.config.idle_threshold_s:
                        await self._trigger_proactive_thought(idle_duration)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("autonomy_loop_error", error=str(e))
                await asyncio.sleep(self.config.check_interval_s * 2)

    async def _check_visual_arrival(self, now: float, active_pers: Any) -> bool:
        """Check if a user has arrived in front of the camera after being absent.

        Args:
            now: Current timestamp.
            active_pers: Active personality object.

        Returns:
            bool: True if an arrival greeting was triggered, False otherwise.
        """
        if not hasattr(self.engine, "perception") or self.engine.perception is None:
            return False

        if hasattr(self.engine.perception, "get_latest_scene") and callable(self.engine.perception.get_latest_scene):
            scene = self.engine.perception.get_latest_scene()
        elif hasattr(self.engine.perception, "world_state"):
            ws = self.engine.perception.world_state
            scene = {
                "num_faces": len(getattr(ws, "people", [])),
                "objects": [{"label": getattr(o, "label", "")} for o in getattr(ws, "objects", [])],
            }
        else:
            return False

        has_human = scene.get("num_faces", 0) > 0 or any(
            obj.get("label") == "person" for obj in scene.get("objects", [])
        )

        if not has_human:
            # Human is absent
            if self._was_present:
                self._was_present = False
                self._absence_start_ts = now
                self._arrival_candidate_ts = None
            return False

        # Human is present in the scene
        if not self._was_present:
            # Transition from absent to present
            absence_duration = (now - self._absence_start_ts) if self._absence_start_ts else 0.0
            if absence_duration >= self.config.arrival_absence_threshold_s:
                if self._arrival_candidate_ts is None:
                    self._arrival_candidate_ts = now
                    return False

                # Debounce check: must remain present steadily for arrival_debounce_s
                if (now - self._arrival_candidate_ts) >= self.config.arrival_debounce_s:
                    if (now - self._last_arrival_greeting_ts) >= self.config.cooldown_s:
                        self._was_present = True
                        self._arrival_candidate_ts = None
                        self._last_arrival_greeting_ts = now
                        await self._trigger_arrival_greeting(absence_duration=absence_duration)
                        return True
            else:
                self._was_present = True
                self._arrival_candidate_ts = None
        else:
            self._absence_start_ts = None
            self._arrival_candidate_ts = None

        return False

    async def _trigger_arrival_greeting(self, absence_duration: float) -> None:
        """Formulate and express a context-aware welcome back observation.

        Args:
            absence_duration: Number of seconds the user was away from the camera.
        """
        mins_away = max(1, int(absence_duration // 60))
        logger.info("proactive_arrival_trigger", mins_away=mins_away, count=self.proactive_count + 1)

        active_pers = self.engine.personalities.active if hasattr(self.engine, "personalities") else None
        pers_name = active_pers.name if active_pers else "Lisa"
        hour = datetime.now().hour
        time_greeting = "morning" if hour < 12 else ("afternoon" if hour < 18 else "evening")

        arrival_prompt = (
            f"You are {pers_name}. The user just walked back and sat down at their desk after being away for about {mins_away} minutes this {time_greeting}. "
            f"Give one brief, natural welcome back (under 12 words). Do not use asterisks or markdown."
        )
        await self._synthesize_and_speak_proactive(arrival_prompt, reason=f"arrival_after_{mins_away}m")

    async def _trigger_proactive_thought(self, idle_duration: float) -> None:
        """Formulate and express a proactive contextual observation on idle timer.

        Args:
            idle_duration: Number of seconds the system has been idle.
        """
        active_pers = self.engine.personalities.active if hasattr(self.engine, "personalities") else None
        pers_name = active_pers.name if active_pers else "Lisa"
        hour = datetime.now().hour
        time_greeting = (
            "morning" if hour < 12 else ("afternoon" if hour < 18 else "evening")
        )

        recent_fact = ""
        if self.engine.memory:
            try:
                facts = await self.engine.memory.get_all_facts(limit=3)
                if facts:
                    recent_fact = random.choice(facts).get("content", "")
            except Exception:
                pass

        curiosity_prompts = [
            f"You are {pers_name}. The user has been focused and working quietly for {int(idle_duration // 60)} minutes this {time_greeting}. Say one brief, friendly, non-intrusive sentence (under 15 words) offering a quick check-in or gentle encouragement. Do not use asterisks.",
            f"You are {pers_name}. It's a calm {time_greeting}. Make a concise, curious 1-sentence observation or ask if there is anything you can help look up or calculate. Keep it natural and warm. Do not use asterisks.",
        ]
        if recent_fact:
            curiosity_prompts.append(
                f"You are {pers_name}. Remembering that the user mentioned: '{recent_fact}', make a short 1-sentence check-in related to their interests or focus. Under 15 words. Do not use asterisks."
            )

        selected_prompt = random.choice(curiosity_prompts)
        await self._synthesize_and_speak_proactive(selected_prompt, reason="idle_timer", idle_s=idle_duration)

    async def _synthesize_and_speak_proactive(
        self,
        prompt: str,
        reason: str = "dynamic",
        idle_s: float = 0.0,
    ) -> None:
        """Synthesize proactive line via LLM and output via audio pipeline.

        Args:
            prompt: Text prompt instruction for the LLM.
            reason: Identifier string describing the trigger origin.
            idle_s: Idle seconds duration if applicable.
        """
        self.last_proactive_ts = time.time()
        self.proactive_count += 1

        try:
            active_pers = self.engine.personalities.active if hasattr(self.engine, "personalities") else None
            pers_name = active_pers.name if active_pers else "Lisa"

            ctx = getattr(self.engine, "context", None)
            if not ctx:
                from robot.core.context import ContextEngine
                ctx = ContextEngine()

            sys_prompt = active_pers.system_prompt if active_pers else f"You are {pers_name}."
            ns = getattr(active_pers, "memory_namespace", "default")
            user_profile = None
            if self.engine.memory and hasattr(self.engine.memory, "get_core_user_profile"):
                try:
                    user_profile = await self.engine.memory.get_core_user_profile(user_id=ns)
                except Exception:
                    pass

            situation = {}
            if hasattr(self.engine, "get_situation_context") and callable(getattr(self.engine, "get_situation_context")):
                sit_res = self.engine.get_situation_context()
                if asyncio.iscoroutine(sit_res):
                    situation = await sit_res
                elif isinstance(sit_res, dict):
                    situation = sit_res

            messages = ctx.build_messages(
                system_prompt=sys_prompt,
                user_text=f"[Proactive check-in]: {prompt}",
                user_profile=user_profile,
                situation=situation,
                high_intelligence=getattr(self.engine, "high_intelligence_mode", False),
            )

            full_thought = ""
            async for token in self.engine.llm.stream_chat(messages, max_tokens=60):
                # Filter out internal reasoning tokens so thoughts are never spoken by TTS
                if token.startswith("__THOUGHT__"):
                    continue
                full_thought += token
                if len(full_thought) >= 40 and any(p in token for p in [".", "!", "?", ".\n", "!\n", "?\n"]):
                    break
                if len(full_thought) >= 250:
                    break

            import re
            cleaned = re.sub(r"<thought>.*?</thought>", "", full_thought, flags=re.DOTALL)
            cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"\[thought\].*?\[/thought\]", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r"\[think\].*?\[/think\]", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
            # Remove any trailing unclosed thinking blocks
            cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"<thought>.*", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"\[think\].*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r"\[thought\].*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r"__THOUGHT__.*", "", cleaned)
            cleaned_thought = cleaned.strip().replace('"', "").replace("\n", " ")
            if cleaned_thought and not cleaned_thought.endswith((".", "!", "?")):
                last_punc = max(cleaned_thought.rfind("."), cleaned_thought.rfind("!"), cleaned_thought.rfind("?"))
                if last_punc > 20:
                    cleaned_thought = cleaned_thought[: last_punc + 1].strip()
                else:
                    cleaned_thought = cleaned_thought + "."
            if not cleaned_thought or len(cleaned_thought) < 3:
                return

            print(
                f"\n✨ \033[1;35m[{pers_name.upper()} PROACTIVE - {reason.upper()}]\033[0m {cleaned_thought}"
            )

            await self.engine.events.emit(
                "proactive_thought",
                text=cleaned_thought,
                reason=reason,
                idle_s=idle_s,
            )

            if hasattr(self.engine, "_speech_queue"):
                self.notify_activity()
                asyncio.create_task(
                    self.engine._stream_response(
                        f"[Proactive check-in]: {cleaned_thought}",
                        time.time(),
                        asyncio.Event(),
                    )
                )

        except Exception as e:
            logger.error("proactive_thought_failed", error=str(e), reason=reason)
