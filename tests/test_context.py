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


def test_context_alias_keyword_arguments():
    """Verify build_messages accepts legacy / alias keyword arguments without throwing TypeError."""
    engine = ContextEngine()
    messages = engine.build_messages(
        personality_prompt="You are Luna.",
        user_input="Hello from chat interface",
        history=[{"role": "user", "content": "previous question"}],
        retrieved_memories=[{"memory": "favorite color is cyan"}],
    )
    assert len(messages) >= 3
    assert messages[0]["role"] == "system"
    assert "Luna" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "Hello from chat interface" in messages[-1]["content"]
    assert "cyan" in messages[-1]["content"]

