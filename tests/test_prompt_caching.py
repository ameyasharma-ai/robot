"""Regression tests for local LLM prompt caching, KV-cache stability, and context management."""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from robot.core.context import (
    ContextEngine,
    format_system_prompt,
    format_situation_banner,
    format_user_core_profile,
    _trim_conversation,
    _estimate_tokens,
)
from robot.llm import LlamaServerLLM


def test_trim_conversation_preserves_turn_atomicity():
    """Verify _trim_conversation drops oldest turns without leaving orphan tool messages."""
    history = [
        {"role": "user", "content": "Set a timer"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "timer.set"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'},
        {"role": "assistant", "content": "Timer set for 10 seconds."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]
    # Restrict budget so oldest turns must be dropped
    trimmed = _trim_conversation(history, budget_tokens=15)
    assert len(trimmed) >= 2
    # First retained turn must be a clean user turn, never an orphan tool or assistant response
    assert trimmed[0]["role"] == "user"
    assert trimmed[-1]["role"] == "assistant"


def test_trim_conversation_sanitizes_historical_grounding_wrappers():
    """Verify legacy or embedded grounding wrappers in historical turns are sanitized."""
    history = [
        {
            "role": "user",
            "content": "[Environment & System Grounding]:\nTime is 11:00 AM\n\n[User Instruction]:\nTurn on lights",
        },
        {"role": "assistant", "content": "Turning on the lights."},
    ]
    trimmed = _trim_conversation(history, budget_tokens=100)
    assert len(trimmed) == 2
    assert trimmed[0]["content"] == "Turn on lights"


def test_context_engine_enforces_local_context_cap():
    """Verify ContextEngine prunes older turns when exceeding 14,000 token limit in local mode."""
    engine = ContextEngine()
    # Create very large message history (>15,000 tokens)
    huge_text = "This is a long historical message turn. " * 300  # ~3,000 tokens per turn
    large_history = []
    for i in range(8):
        large_history.append({"role": "user", "content": f"User turn {i}: {huge_text}"})
        large_history.append({"role": "assistant", "content": f"Bot turn {i}: {huge_text}"})

    messages = engine.build_messages(
        system_prompt="You are Lisa.",
        user_text="Current short request",
        recent_conversation=large_history,
        high_intelligence=False,
    )
    total_tokens = sum(_estimate_tokens(str(m.get("content") or "")) for m in messages)
    assert total_tokens <= 14000
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Current short request" in messages[-1]["content"]


def test_llama_server_llm_canonical_tools_memoization():
    """Verify LlamaServerLLM memoizes and consistently exposes canonical tools."""
    tools = [
        {"type": "function", "function": {"name": "test_tool_a", "description": "tool a", "parameters": {}}},
        {"type": "function", "function": {"name": "test_tool_b", "description": "tool b", "parameters": {}}},
    ]
    llm = LlamaServerLLM(use_invariant_tools=True)
    assert llm.canonical_tools is None

    llm.set_canonical_tools(tools)
    assert llm.canonical_tools == tools
    assert len(llm.canonical_tools) == 2


@pytest.mark.asyncio
async def test_llama_server_stream_chat_uses_canonical_tools_when_tools_none():
    """Verify stream_chat applies canonical tools even when caller passes tools=None."""
    tools = [{"type": "function", "function": {"name": "canonical_tool", "description": "desc", "parameters": {}}}]
    llm = LlamaServerLLM(use_invariant_tools=True, canonical_tools=tools)

    captured_payload = None

    class MockStream:
        def __init__(self, *args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "ok"}}]}'
            yield "data: [DONE]"

        @property
        def status_code(self):
            return 200

    with patch("httpx.AsyncClient.stream", side_effect=MockStream):
        tokens = []
        async for t in llm.stream_chat([{"role": "user", "content": "hi"}], tools=None):
            tokens.append(t)

        assert tokens == ["ok"]
        assert captured_payload is not None
        assert "tools" in captured_payload
        assert captured_payload["tools"][0]["function"]["name"] == "canonical_tool"


@pytest.mark.asyncio
async def test_llama_server_stream_chat_respects_explicit_empty_tools():
    """Verify stream_chat excludes tools when caller explicitly provides tools=[]."""
    tools = [{"type": "function", "function": {"name": "canonical_tool", "description": "desc", "parameters": {}}}]
    llm = LlamaServerLLM(use_invariant_tools=True, canonical_tools=tools)

    captured_payload = None

    class MockStream:
        def __init__(self, *args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "no tools"}}]}'
            yield "data: [DONE]"

        @property
        def status_code(self):
            return 200

    with patch("httpx.AsyncClient.stream", side_effect=MockStream):
        tokens = []
        async for t in llm.stream_chat([{"role": "user", "content": "hi"}], tools=[]):
            tokens.append(t)

        assert tokens == ["no tools"]
        assert captured_payload is not None
        assert "tools" not in captured_payload


def test_format_system_prompt_determinism():
    """Verify format_system_prompt is deterministic and stable across calls."""
    prompt_1 = format_system_prompt("You are Lisa.")
    prompt_2 = format_system_prompt("You are Lisa.")
    assert prompt_1 == prompt_2
    assert "Truth, Tools & Real-Time Action Directives" in prompt_1


def test_format_situation_banner_contains_telemetry():
    """Verify format_situation_banner structures all telemetry keys properly."""
    situation = {
        "battery": {"percent": 75, "power_plugged": True},
        "music_state": {"is_playing": True, "current_track": "Test Track", "volume": 0.8},
        "world_state": {"people_count": 2, "scene": "lab"},
        "scheduler_jobs": [{"name": "Tea Timer", "time": "5m"}],
    }
    banner = format_situation_banner(situation)
    assert "Current Time (Indian Standard Time / IST)" in banner
    assert "Tea Timer" in banner
    assert "Test Track" in banner
    assert "75%" in banner
    assert "2 person(s)" in banner


@pytest.mark.asyncio
async def test_get_slot_metrics_retrieval():
    """Verify get_slot_metrics queries /slots endpoint cleanly."""
    llm = LlamaServerLLM()
    mock_slots_data = [{"id": 0, "n_ctx": 16384, "n_prompt_tokens": 512, "is_processing": False}]

    class MockSlotsResp:
        status_code = 200
        def json(self):
            return mock_slots_data

    with patch("httpx.AsyncClient.get", return_value=MockSlotsResp()):
        metrics = await llm.get_slot_metrics()
        assert metrics["slot_id"] == 0
        assert metrics["n_ctx"] == 16384
        assert metrics["n_prompt_tokens"] == 512
        assert metrics["is_processing"] is False


@pytest.mark.asyncio
async def test_personality_switch_endpoint_triggers_warmup():
    """Verify switch_personality in API schedules warm_up_llm without NameError."""
    from robot.api import create_app
    from fastapi.testclient import TestClient

    mock_engine = MagicMock()
    mock_engine.personalities.list_personalities.return_value = ["neo", "glados"]
    mock_engine.personalities.set_active.return_value = True
    mock_engine.personalities.get.return_value = MagicMock(voice_model="test.onnx")
    mock_engine.tts.set_active_voice = MagicMock()
    mock_engine.events.emit = AsyncMock()
    mock_engine.warm_up_llm = AsyncMock(return_value=120.0)

    app = create_app(mock_engine)
    client = TestClient(app)

    resp = client.post("/personality/glados")
    assert resp.status_code == 200
    assert resp.json()["status"] == "switched"
    assert resp.json()["personality"] == "glados"

    # Give event loop a micro-tick for background task to resolve
    await asyncio.sleep(0.01)
    mock_engine.warm_up_llm.assert_called_once_with("glados")


@pytest.mark.asyncio
async def test_curiosity_engine_proactive_thought_prefix_invariance():
    """Verify CuriosityEngine constructs messages with the invariant system prompt."""
    from robot.core.autonomy import CuriosityEngine, AutonomyConfig

    mock_engine = MagicMock()
    mock_pers = MagicMock()
    mock_pers.name = "glados"
    mock_pers.system_prompt = "You are GLaDOS. Be sarcastic."
    mock_pers.memory_namespace = "glados"
    mock_engine.personalities.active = mock_pers
    mock_engine.context = ContextEngine()
    mock_engine.memory = None
    mock_engine.get_situation_context = MagicMock(return_value={})
    mock_engine.events.emit = AsyncMock()
    mock_engine.high_intelligence_mode = False
    mock_engine._stream_response = AsyncMock(return_value="Take a break.")

    captured_messages = []

    async def fake_stream_chat(messages, max_tokens=60, **kwargs):
        captured_messages.append(messages)
        yield "Take a break."

    mock_engine.llm.stream_chat = fake_stream_chat

    engine_autonomy = CuriosityEngine(mock_engine, AutonomyConfig(enabled=True))
    await engine_autonomy._trigger_proactive_thought(idle_duration=120.0)

    assert len(captured_messages) == 1
    msgs = captured_messages[0]
    assert msgs[0]["role"] == "system"
    # System prompt MUST match format_system_prompt verbatim for KV cache slot invariance
    assert msgs[0]["content"] == format_system_prompt("You are GLaDOS. Be sarcastic.", high_intelligence=False)
    assert msgs[-1]["role"] == "user"
    assert "[Proactive check-in]:" in msgs[-1]["content"]

