"""Event bus for async inter-subsystem communication."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from robot.telemetry import get_logger

logger = get_logger("events")

EventHandler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """Lightweight in-process async event bus.

    Replaces Kafka/RabbitMQ — one laptop doesn't need distributed messaging.
    Uses asyncio queues and callbacks for decoupled subsystem communication.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        self._handlers[event].append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        """Unregister a handler."""
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: str, **kwargs) -> None:
        """Emit an event to all registered handlers."""
        handlers = self._handlers.get(event, [])
        if not handlers:
            return

        tasks = [asyncio.create_task(h(**kwargs)) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "event_handler_error",
                    event=event,
                    handler=handlers[i].__name__,
                    error=str(result),
                )


# Predefined event names
class Events:
    # Audio pipeline
    SPEECH_START = "speech.start"
    SPEECH_END = "speech.end"
    BARGE_IN = "barge_in"

    # STT
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"

    # LLM
    LLM_TOKEN = "llm.token"
    LLM_COMPLETE = "llm.complete"
    LLM_ERROR = "llm.error"

    # TTS
    TTS_PHRASE_READY = "tts.phrase_ready"
    TTS_PLAYBACK_START = "tts.playback_start"
    TTS_PLAYBACK_DONE = "tts.playback_done"

    # Memory
    MEMORY_STORED = "memory.stored"
    MEMORY_RETRIEVED = "memory.retrieved"

    # System
    SERVICE_UP = "service.up"
    SERVICE_DOWN = "service.down"
    PERSONALITY_CHANGED = "personality.changed"
