"""Unit tests for conversation context grounding, tool execution tracking, and intent routing."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest

from robot.core.context import ContextEngine, GROUNDING_POLICY
from robot.tools import ToolCategory, ToolDefinition, ToolRegistry
from robot.tools.coordinator import ToolConversationCoordinator
from robot.memory import ConversationHistory


def test_grounding_policy_covers_image_and_motion() -> None:
    """Verify GROUNDING_POLICY explicitly directs image generation and motion tools."""
    assert "image.generate" in GROUNDING_POLICY
    assert "character.execute_action" in GROUNDING_POLICY
    assert "Starting to render that image for you now" in GROUNDING_POLICY
    assert "raise my left hand" in GROUNDING_POLICY


def test_fast_route_conversational_motion() -> None:
    """Verify conversational motion prompts route with correct action and limb side."""
    mock_engine = MagicMock()
    mock_engine.config = {}
    coordinator = ToolConversationCoordinator(mock_engine)

    # 1. Raise left hand
    route1 = coordinator.fast_route("Can you raise your left hand?")
    assert route1 is not None
    assert route1[0] == "character.execute_action"
    assert route1[1] == {"action": "raise_hand", "side": "left"}
    assert route1[2] == ToolCategory.FORMAT

    # 2. Raise hand (default right)
    route2 = coordinator.fast_route("raise your hand")
    assert route2 is not None
    assert route2[0] == "character.execute_action"
    assert route2[1] == {"action": "raise_hand", "side": "right"}

    # 3. Wave your left hand
    route3 = coordinator.fast_route("wave your left hand")
    assert route3 is not None
    assert route3[0] == "character.execute_action"
    assert route3[1] == {"action": "wave", "side": "left"}

    # 4. Nod
    route4 = coordinator.fast_route("nod your head")
    assert route4 is not None
    assert route4[0] == "character.execute_action"
    assert route4[1] == {"action": "nod"}


def test_fast_route_conversational_image() -> None:
    """Verify conversational image generation prompts route cleanly."""
    mock_engine = MagicMock()
    mock_engine.config = {}
    coordinator = ToolConversationCoordinator(mock_engine)

    # Leading "and" + "make me a picture of"
    route1 = coordinator.fast_route("and make me a picture of a dog")
    assert route1 is not None
    assert route1[0] == "image.generate"
    assert route1[1] == {"prompt": "a dog"}
    assert route1[2] == ToolCategory.ASYNC

    # "can you draw a sunset"
    route2 = coordinator.fast_route("can you draw a sunset over mountains")
    assert route2 is not None
    assert route2[0] == "image.generate"
    assert route2[1] == {"prompt": "sunset over mountains"}

    # "make me a picture of a cat"
    route3 = coordinator.fast_route("make me a picture of a cat")
    assert route3 is not None
    assert route3[0] == "image.generate"
    assert route3[1] == {"prompt": "a cat"}


@pytest.mark.asyncio
async def test_coordinator_last_execution_tracking() -> None:
    """Verify ToolConversationCoordinator tracks last execution for persistence."""
    mock_engine = MagicMock()
    mock_engine.config = {}
    mock_engine._session_id = "test_sess"
    mock_engine.observability.current_trace_id = "test_trace"
    mock_engine.events.emit = AsyncMock()
    mock_engine.permissions.execute = AsyncMock(return_value={"result": "http://img.png", "status": "ok"})
    
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="image.generate",
            description="Generate image",
            parameters={"type": "object", "properties": {"prompt": {"type": "string"}}},
            handler=AsyncMock(),
        )
    )
    mock_engine.tool_registry = reg

    coord = ToolConversationCoordinator(mock_engine)
    coord._generate_continuation = AsyncMock(return_value="Here is your image!")

    ans = await coord.coordinate(
        tool_name="image.generate",
        tool_args={"prompt": "a dog"},
        category=ToolCategory.DIRECT,
        user_text="make a picture of a dog",
        end_of_speech_time=0.0,
        interrupt=asyncio.Event(),
        messages=[],
    )

    last_exec = coord.get_last_execution()
    assert last_exec is not None
    assert last_exec["tool_name"] == "image.generate"
    assert last_exec["tool_args"] == {"prompt": "a dog"}
    assert last_exec["tool_result"] == "http://img.png"


@pytest.mark.asyncio
async def test_conversation_history_tool_results_and_wake_filtering(tmp_path) -> None:
    """Verify ConversationHistory attaches tool results and filters wake word acknowledge."""
    db_path = str(tmp_path / "conv_test.db")
    history = ConversationHistory(db_path)
    await history.initialize()

    sess_1 = "sess_001"
    await history.create_session(sess_1, personality="luna")

    # Add wake word turns (which should be filtered)
    await history.add_message(sess_1, "user", "[Wake word acknowledge]", "luna", 1)
    await history.add_message(sess_1, "assistant", "I'm listening.", "luna", 1)

    # Add real turns with a tool result
    await history.add_message(sess_1, "user", "Make me a picture of a golden retriever", "luna", 2)
    await history.add_message(
        sess_1,
        "assistant",
        "Here is the picture of a golden retriever on the beach!",
        "luna",
        2,
        tool_name="image.generate",
        tool_result="/data/images/retriever.png",
    )

    recent = await history.get_recent(sess_1, limit=10, personality="luna")
    
    # Wake word turns must be filtered out
    assert not any(m["content"] == "[Wake word acknowledge]" for m in recent)
    assert not any(m["content"] == "I'm listening." for m in recent)

    # Real turn must be present and assistant turn must have tool result attached
    asst_turn = next(m for m in recent if m["role"] == "assistant")
    assert "/data/images/retriever.png" in asst_turn["content"]
    assert "[Tool Result: image.generate -> /data/images/retriever.png]" in asst_turn["content"]


@pytest.mark.asyncio
async def test_conversation_history_cross_session_fallback(tmp_path) -> None:
    """Verify fresh sessions fall back to the most recent conversation across sessions for the persona."""
    db_path = str(tmp_path / "cross_test.db")
    history = ConversationHistory(db_path)
    await history.initialize()

    # Session 1 with previous discussion
    await history.create_session("sess_old", personality="luna")
    await history.add_message("sess_old", "user", "Let's talk about the beach", "luna", 1)
    await history.add_message("sess_old", "assistant", "I love the ocean and sandy beaches!", "luna", 1)

    # Session 2 is brand new (0 messages)
    await history.create_session("sess_new", personality="luna")
    recent = await history.get_recent("sess_new", limit=10, personality="luna")

    # Must fall back to previous persona turns so continuity is preserved
    assert len(recent) == 2
    assert recent[0]["content"] == "Let's talk about the beach"
    assert recent[1]["content"] == "I love the ocean and sandy beaches!"
