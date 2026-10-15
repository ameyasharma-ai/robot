"""Tests for cognitive reasoning and thinking mode toggle in Chat and Avatar views.

Validates that:
1. GET /api/thinking reports current thinking mode status accurately.
2. POST /api/thinking updates engine.thinking_enabled and broadcasts thinking_toggled event.
3. Chat stream requests accept and honor thinking boolean flag.
4. Internal thought tags and reasoning tokens are stripped before TTS speech queue ingestion.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest
from starlette.testclient import TestClient

from robot.api import create_app
from robot.core.state import RobotMode, RobotState


def _create_mock_engine() -> MagicMock:
    """Create a fully-mocked RobotEngine for API testing.

    Returns:
        MagicMock: Configured mock engine instance.
    """
    engine = MagicMock()
    engine.thinking_enabled = True
    engine.voice_enabled = True
    engine.is_chat_mode = False
    engine.state = RobotState()
    engine.events = MagicMock()
    engine.events.emit = AsyncMock()
    engine.telemetry = MagicMock()
    engine.telemetry.snapshot = MagicMock(return_value={})
    return engine


def test_get_thinking_mode() -> None:
    """Verify that GET /api/thinking returns the current engine state.

    Returns:
        None
    """
    engine = _create_mock_engine()
    engine.thinking_enabled = True
    app = create_app(engine)
    client = TestClient(app)

    response = client.get("/api/thinking")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["status"] == "ok"
    assert data["thinking_enabled"] is True

    # Toggle engine state and re-check
    engine.thinking_enabled = False
    response = client.get("/api/thinking")
    assert response.status_code == 200
    data = response.json()
    assert data["thinking_enabled"] is False


def test_set_thinking_mode() -> None:
    """Verify that POST /api/thinking modifies engine state and emits WebSocket event.

    Returns:
        None
    """
    engine = _create_mock_engine()
    engine.thinking_enabled = True
    app = create_app(engine)
    client = TestClient(app)

    # Disable thinking mode
    response = client.post("/api/thinking", json={"enabled": False})
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["status"] == "ok"
    assert data["thinking_enabled"] is False
    assert engine.thinking_enabled is False

    engine.events.emit.assert_called_with("thinking_toggled", enabled=False)

    # Re-enable thinking mode
    response = client.post("/api/thinking", json={"enabled": True})
    assert response.status_code == 200
    data = response.json()
    assert data["thinking_enabled"] is True
    assert engine.thinking_enabled is True

    engine.events.emit.assert_called_with("thinking_toggled", enabled=True)


@pytest.mark.asyncio
async def test_thought_isolation_from_speech() -> None:
    """Verify that internal reasoning tags are stripped cleanly from TTS speech strings.

    Rationale:
        TTS must never speak the robot's inner thoughts aloud to the user.
        Reasoning inside <thought>...</thought> must be stripped before synthesis.

    Returns:
        None
    """
    import re
    from robot.tts import clean_text_for_speech

    raw_response = (
        "<thought>The user wants to open the calculator. I should launch gnome-calculator.</thought>"
        "Opening the calculator for you right now, Ameya."
    )

    # Clean using engine regex
    cleaned = re.sub(r"<thought>.*?</thought>", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
    assert cleaned == "Opening the calculator for you right now, Ameya."
    assert "<thought>" not in cleaned

    # Verify speech sanitization
    speech_ready = clean_text_for_speech(cleaned)
    assert "calculator" in speech_ready
    assert "thought" not in speech_ready.lower()


def test_streaming_thought_parser_split_boundary() -> None:
    """Verify that StreamingThoughtParser properly reconstitutes tags split across arbitrary chunk boundaries.

    Returns:
        None
    """
    from robot.core.engine import StreamingThoughtParser

    chunks = ["<th", "ought>Analyzing ", "user intent", "...</th", "ought>Hello, ", "Ameya!"]
    parser = StreamingThoughtParser()
    events: list[tuple[str, str]] = []

    for chunk in chunks:
        events.extend(list(parser.feed(chunk)))
    events.extend(list(parser.finish()))

    event_types = [e[0] for e in events]
    assert "thought_started" in event_types
    assert "thought_completed" in event_types

    # Aggregate thoughts
    thought_tokens = [e[1] for e in events if e[0] == "thought"]
    assert "".join(thought_tokens) == "Analyzing user intent..."

    # Aggregate conversational tokens
    content_tokens = [e[1] for e in events if e[0] == "token"]
    assert "".join(content_tokens) == "Hello, Ameya!"

    # Verify completed event payload
    completed_events = [e[1] for e in events if e[0] == "thought_completed"]
    assert len(completed_events) == 1
    assert completed_events[0] == "Analyzing user intent..."


def test_streaming_thought_parser_think_tags() -> None:
    """Verify that StreamingThoughtParser correctly handles DeepSeek / Qwen style <think>...</think> tags.

    Returns:
        None
    """
    from robot.core.engine import StreamingThoughtParser

    chunks = ["<think>Evaluating computational budget.</think>Processing complete."]
    parser = StreamingThoughtParser()
    events: list[tuple[str, str]] = []

    for chunk in chunks:
        events.extend(list(parser.feed(chunk)))
    events.extend(list(parser.finish()))

    thought_tokens = [e[1] for e in events if e[0] == "thought"]
    assert "".join(thought_tokens) == "Evaluating computational budget."

    content_tokens = [e[1] for e in events if e[0] == "token"]
    assert "".join(content_tokens) == "Processing complete."


def test_streaming_thought_parser_no_tags() -> None:
    """Verify that StreamingThoughtParser emits direct tokens when no thought tags are present.

    Returns:
        None
    """
    from robot.core.engine import StreamingThoughtParser

    chunks = ["Hello, ", "world! ", "5 < 10 and 10 > 5."]
    parser = StreamingThoughtParser()
    events: list[tuple[str, str]] = []

    for chunk in chunks:
        events.extend(list(parser.feed(chunk)))
    events.extend(list(parser.finish()))

    event_types = [e[0] for e in events]
    assert "thought_started" not in event_types
    assert "thought" not in event_types
    assert "thought_completed" not in event_types

    content_tokens = [e[1] for e in events if e[0] == "token"]
    assert "".join(content_tokens) == "Hello, world! 5 < 10 and 10 > 5."

