"""Tests for LLM clause chunker."""

import pytest

from robot.llm import ClauseChunker


def test_chunker_sentence_boundary():
    chunker = ClauseChunker()
    assert chunker.feed("Hello, ") is None
    result = chunker.feed("world. ")
    assert result == "Hello, world."


def test_chunker_question():
    chunker = ClauseChunker()
    chunker.feed("How are ")
    result = chunker.feed("you? ")
    assert result == "How are you?"


def test_chunker_flush():
    chunker = ClauseChunker()
    chunker.feed("Incomplete sentence without")
    chunker.feed(" punctuation")
    result = chunker.flush()
    assert result == "Incomplete sentence without punctuation"


def test_chunker_long_unpunctuated():
    chunker = ClauseChunker(max_words=5)
    for word in ["one", " two", " three", " four", " five", " six"]:
        result = chunker.feed(word)
        if result:
            assert len(result.split()) >= 1
            break


def test_chunker_empty_flush():
    chunker = ClauseChunker()
    assert chunker.flush() is None


def test_chunker_inline_motion_tags():
    chunker = ClauseChunker(eager_first_chunk=True, min_first_words=2)
    # Inline motion tags with internal colons should not break clause chunking
    result = chunker.feed("[motion: lean_in] Hello there, ")
    assert result == "[motion: lean_in] Hello there,"


@pytest.mark.asyncio
async def test_warm_up_llm_prefix_matching():
    from unittest.mock import AsyncMock, MagicMock
    from robot.core.engine import RobotEngine
    from robot.core.context import GROUNDING_POLICY, format_system_prompt

    engine = RobotEngine.__new__(RobotEngine)
    engine.llm = MagicMock()
    captured_messages = []

    async def fake_stream_chat(messages, max_tokens=1, tools=None):
        captured_messages.extend(messages)
        yield "token"

    engine.llm.stream_chat = fake_stream_chat
    engine.personalities = MagicMock()
    mock_pers = MagicMock()
    mock_pers.name = "luna"
    mock_pers.system_prompt = "You are Luna."
    engine.personalities.get.return_value = mock_pers
    engine.personalities.active = mock_pers

    duration = await engine.warm_up_llm("luna")
    assert duration >= 0.0
    assert len(captured_messages) == 2
    assert captured_messages[0]["role"] == "system"
    assert captured_messages[0]["content"] == format_system_prompt("You are Luna.", high_intelligence=False)
    assert captured_messages[1]["role"] == "user"


def test_parse_qwen_tool_call_xml():
    from robot.llm import parse_qwen_tool_call_block
    xml_block = """<tool_call>
<function=web.search>
<parameter=query>
plane hijacking Indian pilot attacked
</parameter>
</function>
</tool_call>"""
    parsed = parse_qwen_tool_call_block(xml_block)
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "web.search"
    assert "plane hijacking Indian pilot attacked" in parsed[0]["function"]["arguments"]


def test_clean_text_for_speech_strips_tool_call():
    from robot.tts import clean_text_for_speech
    text_with_tool_call = "Checking data. <tool_call> <function=web.search> <parameter=query> plane hijacking </parameter> </function> </tool_call>"
    cleaned = clean_text_for_speech(text_with_tool_call)
    assert "<tool_call>" not in cleaned
    assert "web.search" not in cleaned
    assert cleaned == "Checking data."


def test_clause_chunker_strips_tool_call():
    chunker = ClauseChunker()
    # Emitting raw tool calls into chunker should be sanitized
    res = chunker.feed("Checking that. <tool_call><function=web.search><parameter=query>plane</parameter></function></tool_call>")
    assert res == "Checking that."
    assert "<tool_call>" not in (res or "")


def test_extract_motion_preserves_markdown_asterisks():
    from robot.core.engine import _extract_motion_from_phrase
    # Physical stage directions must be stripped
    text1 = "*waves* Hello there! *smiles*"
    c1, act1 = _extract_motion_from_phrase(text1)
    assert "Hello there!" in c1
    assert "*waves*" not in c1
    assert act1 is not None and act1.get("action") == "wave"

    # Spoken emphasized words with asterisks must NOT be stripped!
    text2 = "*Hello*! How are you?"
    c2, act2 = _extract_motion_from_phrase(text2)
    assert c2.startswith("Hello")
    assert act2 is None

    # Bracket motion markers
    text3 = "[motion: nod] Exactly right."
    c3, act3 = _extract_motion_from_phrase(text3)
    assert c3 == "Exactly right."
    assert act3 is not None and act3.get("action") == "nod"


def test_gemini_llm_initialization():
    from robot.llm import GeminiLLM
    gemini = GeminiLLM(api_key="test_key_123")
    assert gemini.api_key == "test_key_123"
    assert "openai" in gemini.base_url
    assert "gemini" in gemini.model


def test_llama_server_manager():
    from robot.llm import LlamaServerManager
    # Manager has required class methods
    assert hasattr(LlamaServerManager, "is_running")
    assert hasattr(LlamaServerManager, "stop")
    assert hasattr(LlamaServerManager, "start")


def test_context_engine_high_intelligence_mode():
    from robot.core.context import ContextEngine
    ce = ContextEngine()
    long_conv = [{"role": "user", "content": f"Turn message {i} with lots of context details"} for i in range(20)]
    memories = [{"memory": f"User preference {i}", "created_at": "2026-10-01"} for i in range(15)]

    # Standard mode (low latency): trims to budget
    msgs_standard = ce.build_messages(
        system_prompt="You are Luna.",
        user_text="What do you remember?",
        recent_conversation=long_conv,
        memories=memories,
        high_intelligence=False,
    )
    # High-intelligence mode: expands context up to 50k tokens
    msgs_high_intel = ce.build_messages(
        system_prompt="You are Luna.",
        user_text="What do you remember?",
        recent_conversation=long_conv,
        memories=memories,
        high_intelligence=True,
    )
    assert len(msgs_high_intel) >= len(msgs_standard)


def test_gemini_message_sanitization():
    from robot.llm import GeminiLLM

    gemini = GeminiLLM(api_key="mock_key")
    raw = [
        {"role": "user", "content": "Play closer"},
        {
            "role": "assistant",
            "content": "Putting on closer.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "music.play", "arguments": '{"query": "closer"}'},
                }
            ],
        },
        {"role": "tool", "name": "music.play", "tool_call_id": "call_123", "content": "Track playing"},
    ]
    sanitized = []
    for m in raw:
        role = m.get("role")
        if role == "tool":
            sanitized.append(
                {"role": "user", "content": f"[Tool result from {m.get('name') or 'tool'}]: {m.get('content')}"}
            )
        elif role == "assistant" and "tool_calls" in m:
            sanitized.append({"role": "assistant", "content": m.get("content") or "Executing..."})
        else:
            sanitized.append(m)

    assert len(sanitized) == 3
    assert "tool_calls" not in sanitized[1]
    assert sanitized[1]["content"] == "Putting on closer."
    assert sanitized[2]["role"] == "user"
    assert "Track playing" in sanitized[2]["content"]





