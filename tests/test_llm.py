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
    from robot.core.context import GROUNDING_POLICY

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
    assert captured_messages[0]["content"] == f"You are Luna.\n{GROUNDING_POLICY}"
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




