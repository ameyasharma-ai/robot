"""Deterministic Fast-Path Router for sub-millisecond local utility intents.

Architectural Rationale:
1. Ultra-Low Latency (<50ms): Deterministic queries like local time, battery status, or explicit
   display brightness commands do not require a two-pass LLM inference roundtrip.
2. Safety & Security Invariance: The Fast-Path Router never executes actions directly; it produces
   a structured FastPathIntent and routes execution through the existing PermissionManager and
   tool input hardening layers, preserving all safety, permission, and logging invariants.
3. Graceful Ambiguity Fallback: Any ambiguous query (e.g., 'What time does the train arrive?' or
   'Is it time to wake up?') returns None and falls back cleanly to the primary LLM planner.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from robot.telemetry import get_logger
from robot.tools import PermissionManager, ToolRegistry

logger = get_logger("tools.fast_path")


@dataclass
class FastPathIntent:
    """Structured representation of an unambiguous local utility intent."""

    tool_name: str
    arguments: dict[str, Any]
    confidence: float
    matched_pattern: str
    category: str = "local_utility"


class FastPathRouter:
    """Deterministic intent classifier and router for unambiguous local actions."""

    # ─── Unambiguous Patterns ───────────────────────────────────────────────

    TIME_PATTERNS = [
        r"^what(?:'s| is) the time(?:\s+right now|\s+now)?\??$",
        r"^what time is it(?:\s+right now|\s+now)?\??$",
        r"^tell me the time\??$",
        r"^current time\??$",
        r"^what(?:'s| is) (?:today's |the )?date\??$",
        r"^what date is it\??$",
        r"^what day is it(?:\s+today)?\??$",
        r"^what day of the week is it\??$",
    ]

    HARDWARE_PATTERNS = [
        r"^what(?:'s| is) (?:the )?battery(?: level| status| percentage)?\??$",
        r"^battery(?: level| status| percentage)?\??$",
        r"^how much battery(?:\s+is left)?\??$",
        r"^system status\??$",
        r"^hardware status\??$",
        r"^system health\??$",
        r"^system specs\??$",
    ]

    BRIGHTNESS_GET_PATTERNS = [
        r"^what(?:'s| is) (?:the )?(?:screen|display)?\s*brightness\??$",
        r"^get (?:screen |display )?brightness\??$",
    ]

    BRIGHTNESS_SET_PATTERNS = [
        r"^set (?:the )?(?:display |screen )?brightness to (\d{1,3})%?\??$",
        r"^change (?:the )?(?:display |screen )?brightness to (\d{1,3})%?\??$",
        r"^turn (?:the )?(?:display |screen )?brightness to (\d{1,3})%?\??$",
        r"^adjust (?:the )?(?:display |screen )?brightness to (\d{1,3})%?\??$",
    ]

    def __init__(self, registry: ToolRegistry, permissions: PermissionManager) -> None:
        """Initialize the fast-path router.

        Args:
            registry: Central tool registry containing registered tool definitions.
            permissions: Security permission manager to enforce safety gates.
        """
        self.registry: ToolRegistry = registry
        self.permissions: PermissionManager = permissions

    def classify(self, user_text: str) -> FastPathIntent | None:
        """Classify user text into a deterministic FastPathIntent if unambiguous.

        Args:
            user_text: Raw incoming user utterance string.

        Returns:
            FastPathIntent if unambiguous match is identified, otherwise None.
        """
        cleaned = user_text.strip().lower()

        # 1. System Time & Date
        for pat in self.TIME_PATTERNS:
            if re.match(pat, cleaned):
                return FastPathIntent(
                    tool_name="system.time",
                    arguments={},
                    confidence=0.99,
                    matched_pattern=pat,
                )

        # 2. Hardware & Battery
        for pat in self.HARDWARE_PATTERNS:
            if re.match(pat, cleaned):
                return FastPathIntent(
                    tool_name="system.hardware",
                    arguments={},
                    confidence=0.98,
                    matched_pattern=pat,
                )

        # 3. Display Brightness Query
        for pat in self.BRIGHTNESS_GET_PATTERNS:
            if re.match(pat, cleaned):
                return FastPathIntent(
                    tool_name="os.display_brightness",
                    arguments={"action": "get"},
                    confidence=0.98,
                    matched_pattern=pat,
                )

        # 4. Display Brightness Set
        for pat in self.BRIGHTNESS_SET_PATTERNS:
            m = re.match(pat, cleaned)
            if m:
                level = int(m.group(1))
                if 0 <= level <= 100:
                    return FastPathIntent(
                        tool_name="os.display_brightness",
                        arguments={"action": "set", "level_percent": level},
                        confidence=0.99,
                        matched_pattern=pat,
                    )

        return None

    async def execute_intent(
        self, intent: FastPathIntent
    ) -> tuple[dict[str, Any], str]:
        """Execute a FastPathIntent through permission checks and generate spoken response.

        Args:
            intent: Validated FastPathIntent object.

        Returns:
            Tuple of (raw_execution_result, spoken_response_phrase).
        """
        t0 = time.perf_counter()
        tool_def = self.registry.get(intent.tool_name)
        if not tool_def:
            return {"error": f"Tool '{intent.tool_name}' not found"}, "I could not find the requested tool."

        # Execute through standard permission manager (enforcing hardening & safety)
        exec_outcome = await self.permissions.execute(tool_def, intent.arguments)
        dur_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "fast_path_executed",
            tool=intent.tool_name,
            duration_ms=round(dur_ms, 2),
            status=exec_outcome.get("status", "success"),
        )

        res = exec_outcome.get("result", exec_outcome)

        # Generate natural, character-aligned response phrase
        phrase = self.format_response(intent.tool_name, intent.arguments, res)
        return exec_outcome, phrase

    def format_response(
        self, tool_name: str, arguments: dict[str, Any], result: Any
    ) -> str:
        """Format an execution outcome into a natural, conversational spoken sentence.

        Args:
            tool_name: Name of the executed tool.
            arguments: Arguments passed to the tool.
            result: Result dictionary returned by the tool.

        Returns:
            Fluent sentence suitable for immediate TTS synthesis.
        """
        if isinstance(result, dict) and "error" in result:
            return f"I ran into an issue: {result['error']}"

        if tool_name == "system.time" and isinstance(result, dict):
            cur_time = result.get("current_time", "")
            cur_date = result.get("current_date", "")
            if cur_time:
                return f"It is {cur_time} on {cur_date}."

        if tool_name == "system.hardware" and isinstance(result, dict):
            bat = result.get("battery", {})
            pct = bat.get("percent", "unknown")
            plugged = "plugged in" if bat.get("power_plugged") else "on battery"
            cpu = result.get("cpu_percent", 0)
            return f"Battery is at {pct}% and {plugged}. System CPU load is {cpu}%."

        if tool_name == "os.display_brightness" and isinstance(result, dict):
            act = arguments.get("action", "get")
            pct = result.get("brightness_percent", arguments.get("level_percent"))
            if act == "set":
                return f"Display brightness set to {pct}%."
            return f"Display brightness is currently {pct}%."

        return "Command completed successfully."
