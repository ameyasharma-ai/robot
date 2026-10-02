"""Tests for context engine."""

from robot.core.context import ContextEngine, _estimate_tokens


def test_estimate_tokens():
    assert _estimate_tokens("Hello world") > 0
    assert _estimate_tokens("") == 0


def test_context_basic_build():
    engine = ContextEngine()
    messages = engine.build_messages(
        system_prompt="You are a helpful assistant.",
        user_text="What time is it?",
    )
    assert len(messages) >= 2  # system + user
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "What time is it?"


def test_context_with_memories():
    engine = ContextEngine()
    memories = [
        {"memory": "User prefers dark mode"},
        {"memory": "User's name is Ameya"},
    ]
    messages = engine.build_messages(
        system_prompt="You are Neo.",
        user_text="Hello",
        memories=memories,
    )
    # Memories are grounded either in system prompt or dynamically in the turn
    full_content = " ".join(m.get("content", "") for m in messages)
    assert "Ameya" in full_content or "dark mode" in full_content


def test_context_with_conversation():
    engine = ContextEngine()
    recent = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    messages = engine.build_messages(
        system_prompt="You are Neo.",
        user_text="Tell me more",
        recent_conversation=recent,
    )
    assert len(messages) >= 4  # system + 2 history + user
