"""Robot Core state management.

The State Manager owns the robot's operational state machine.
The LLM proposes state changes; Robot Core decides.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from robot.telemetry import get_logger

logger = get_logger("state")


class RobotMode(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    TOOL_EXECUTING = "tool_executing"
    ERROR = "error"
    SLEEPING = "sleeping"
    DEGRADED = "degraded"
    STANDBY = "standby"



@dataclass
class ServiceHealth:
    name: str
    healthy: bool = False
    last_check: datetime | None = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class RobotState:
    """Complete robot state snapshot."""

    mode: RobotMode = RobotMode.IDLE
    active_personality: str = "neo"
    session_id: str = ""
    turn_count: int = 0
    uptime_s: float = 0.0

    # Service health
    services: dict[str, ServiceHealth] = field(default_factory=dict)

    # Audio state
    is_speaking: bool = False
    is_listening: bool = True
    last_user_speech: datetime | None = None
    last_bot_speech: datetime | None = None

    # Current turn
    current_transcript: str = ""
    current_response: str = ""

    def to_compact(self) -> dict[str, Any]:
        """Compact state for context injection (0.25–0.5K tokens)."""
        healthy = [s.name for s in self.services.values() if s.healthy]
        unhealthy = [s.name for s in self.services.values() if not s.healthy]
        return {
            "mode": self.mode.value,
            "personality": self.active_personality,
            "turn": self.turn_count,
            "services_up": healthy,
            "services_down": unhealthy if unhealthy else None,
        }


class StateManager:
    """Manages robot state transitions with event notification."""

    def __init__(self) -> None:
        self.state = RobotState()
        self._lock = asyncio.Lock()
        self._listeners: list = []

    def subscribe(self, listener) -> None:
        """Register a callback for state changes: fn(old_mode, new_mode)."""
        self._listeners.append(listener)

    async def set_mode(self, mode: RobotMode) -> None:
        async with self._lock:
            old = self.state.mode
            self.state.mode = mode
            if old != mode:
                logger.info("mode_change", old=old.value, new=mode.value)
                for listener in self._listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            asyncio.create_task(listener(old, mode))
                        else:
                            listener(old, mode)
                    except Exception as e:
                        logger.error("state_listener_error", error=str(e))

    async def update_service_health(
        self, name: str, healthy: bool, error: str | None = None, latency_ms: float = 0.0
    ) -> None:
        async with self._lock:
            self.state.services[name] = ServiceHealth(
                name=name,
                healthy=healthy,
                last_check=datetime.now(timezone.utc),
                error=error,
                latency_ms=latency_ms,
            )

    async def increment_turn(self) -> int:
        async with self._lock:
            self.state.turn_count += 1
            return self.state.turn_count

    @property
    def is_degraded(self) -> bool:
        """True if any critical service is down."""
        critical = {"llm", "stt"}
        for name in critical:
            svc = self.state.services.get(name)
            if svc and not svc.healthy:
                return True
        return False
