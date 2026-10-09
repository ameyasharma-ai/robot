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
    idle_threshold_s: float = 90.0
    cooldown_s: float = 180.0
    check_interval_s: float = 5.0
    require_human_presence: bool = False
    max_proactive_per_session: int = 15
    allowed_modes: list[str] = field(
        default_factory=lambda: ["idle", "standby", "listening"]
    )


class CuriosityEngine:
    """Autonomous engine for proactive idle reasoning and ambient check-ins.

    Monitors system silence, user presence, time-of-day, and recent memory
    to formulate timely, natural observations without requiring direct user prompt.

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

        logger.info(
            "curiosity_engine_initialized",
            enabled=self.config.enabled,
            idle_threshold=self.config.idle_threshold_s,
            cooldown=self.config.cooldown_s,
        )

    def notify_activity(self) -> None:
        """Record activity to reset the idle timer."""
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
        """Main periodic evaluation loop for proactive curiosity."""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_s)
                if not self._running:
                    break

                now = time.time()
                idle_duration = now - self.last_activity_ts
                time_since_proactive = now - self.last_proactive_ts

                # Check basic cooldowns and session limits
                if (
                    idle_duration < self.config.idle_threshold_s
                    or time_since_proactive < self.config.cooldown_s
                    or self.proactive_count >= self.config.max_proactive_per_session
                ):
                    continue

                # Check robot state
                current_mode = self.engine.state.state.mode.value
                if current_mode not in self.config.allowed_modes:
                    continue

                # Check if robot is currently speaking or music is playing
                if (
                    getattr(self.engine, "_is_speaking", False)
                    or self.engine.player.is_playing
                ):
                    continue

                # Check human presence if perception is enabled
                if self.config.require_human_presence:
                    has_human = False
                    if (
                        hasattr(self.engine, "perception")
                        and self.engine.perception is not None
                    ):
                        scene = self.engine.perception.get_latest_scene()
                        has_human = scene.get("num_faces", 0) > 0 or any(
                            obj.get("label") == "person"
                            for obj in scene.get("objects", [])
                        )
                    if not has_human:
                        continue

                # Trigger proactive curiosity action
                await self._trigger_proactive_thought(idle_duration)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("autonomy_loop_error", error=str(e))
                await asyncio.sleep(self.config.check_interval_s * 2)

    async def _trigger_proactive_thought(self, idle_duration: float) -> None:
        """Formulate and express a proactive contextual observation.

        Args:
            idle_duration: Number of seconds the system has been idle.
        """
        self.last_proactive_ts = time.time()
        self.proactive_count += 1
        logger.info(
            "proactive_curiosity_trigger",
            idle_s=round(idle_duration, 1),
            count=self.proactive_count,
        )

        try:
            # Build situational context prompt
            active_pers = self.engine.personalities.active
            pers_name = active_pers.name if active_pers else "Lisa"
            hour = datetime.now().hour
            time_greeting = (
                "morning" if hour < 12 else ("afternoon" if hour < 18 else "evening")
            )

            # Sample recent user memories if available
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

            # Generate proactive line via local LLM
            messages = [
                {
                    "role": "system",
                    "content": f"You are {pers_name}, an intelligent helpful companion. Respond with exactly ONE concise, spoken sentence.",
                },
                {"role": "user", "content": selected_prompt},
            ]

            full_thought = ""
            async for token in self.engine.llm.stream_chat(messages):
                full_thought += token
                # Complete the sentence naturally upon punctuation boundary
                if len(full_thought) >= 60 and any(p in token for p in [".", "!", "?", ".\n", "!\n", "?\n"]):
                    break
                # Hard safety ceiling (250 chars) only if no punctuation is produced
                if len(full_thought) >= 250:
                    break

            cleaned_thought = full_thought.strip().replace('"', "").replace("\n", " ")
            # Ensure the thought ends with proper sentence punctuation
            if cleaned_thought and not cleaned_thought.endswith((".", "!", "?")):
                # If cut off before punctuation, try to trim to last valid punctuation
                last_punc = max(cleaned_thought.rfind("."), cleaned_thought.rfind("!"), cleaned_thought.rfind("?"))
                if last_punc > 20:
                    cleaned_thought = cleaned_thought[: last_punc + 1].strip()
                else:
                    cleaned_thought = cleaned_thought + "."
            if not cleaned_thought:
                return

            print(
                f"\n✨ \033[1;35m[{pers_name.upper()} PROACTIVE]\033[0m {cleaned_thought}"
            )

            # Execute through engine response pipeline / speech output
            await self.engine.events.emit(
                "proactive_thought",
                text=cleaned_thought,
                idle_s=idle_duration,
            )

            # Queue utterance to TTS / UI
            if hasattr(self.engine, "_speech_queue"):
                # Play speech through speech queue or direct synthesis
                self.notify_activity()
                asyncio.create_task(
                    self.engine._stream_response(
                        f"[Proactive check-in]: {cleaned_thought}",
                        time.time(),
                        asyncio.Event(),
                    )
                )

        except Exception as e:
            logger.error("proactive_thought_failed", error=str(e))
