"""Comprehensive unit tests for bracketed and function-call tool interception.

Verifies that prefixes like ``[function call: tool.name(...)]``, ``[call: tool.name(...)]``,
and ``[tool: tool.name(...)]`` are properly parsed and intercepted without leaking into chat
responses or TTS, and that Python-style kwargs are accurately parsed into dictionaries.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from robot.core.engine import _detect_and_intercept_tool_monologue
from robot.llm import BRACKET_TOOL_CALL_PATTERN, parse_qwen_tool_call_block, parse_tool_arguments


def test_bracket_tool_call_pattern_matches_function_call_syntax() -> None:
    """Verify BRACKET_TOOL_CALL_PATTERN matches [function call: tool.name(...)] variants."""
    text1 = '[function call: finance.quote(symbol="NVDA")]'
    m1 = BRACKET_TOOL_CALL_PATTERN.search(text1)
    assert m1 is not None
    assert m1.group(1) == "finance.quote"
    assert m1.group(2) == 'symbol="NVDA"'

    text2 = '[function call: news.headlines(category="tech")]'
    m2 = BRACKET_TOOL_CALL_PATTERN.search(text2)
    assert m2 is not None
    assert m2.group(1) == "news.headlines"
    assert m2.group(2) == 'category="tech"'

    text3 = '[call: weather.forecast(location="Tokyo", days=3)]'
    m3 = BRACKET_TOOL_CALL_PATTERN.search(text3)
    assert m3 is not None
    assert m3.group(1) == "weather.forecast"
    assert "Tokyo" in (m3.group(2) or "")

    text4 = "[tool: system.time]"
    m4 = BRACKET_TOOL_CALL_PATTERN.search(text4)
    assert m4 is not None
    assert m4.group(1) == "system.time"

    text5 = "[system.time]"
    m5 = BRACKET_TOOL_CALL_PATTERN.search(text5)
    assert m5 is not None
    assert m5.group(1) == "system.time"


def test_bracket_pattern_ignores_stage_directions_and_meta() -> None:
    """Verify stage directions and emotional tags are not incorrectly treated as tool calls."""
    stage_direction = "[leans in with a warm, excited smile]"
    assert BRACKET_TOOL_CALL_PATTERN.search(stage_direction) is None

    action_tag = "[chuckles softly and winks]"
    assert BRACKET_TOOL_CALL_PATTERN.search(action_tag) is None

    plain_brackets = "[note: remember to check this]"
    # Has space, not a namespace.action dot identifier
    assert BRACKET_TOOL_CALL_PATTERN.search(plain_brackets) is None


def test_parse_tool_arguments_python_kwargs() -> None:
    """Verify parse_tool_arguments extracts Python-style keyword arguments accurately."""
    parsed1 = parse_tool_arguments('symbol="NVDA"')
    assert parsed1 == {"symbol": "NVDA"}

    parsed2 = parse_tool_arguments("category='tech', limit=5, enabled=True, score=0.95")
    assert parsed2 == {
        "category": "tech",
        "limit": 5,
        "enabled": True,
        "score": 0.95,
    }

    parsed_json = parse_tool_arguments('{"symbol": "NVDA", "count": 2}')
    assert parsed_json == {"symbol": "NVDA", "count": 2}

    parsed_bare = parse_tool_arguments('"bitcoin"')
    assert parsed_bare == {"query": "bitcoin"}

    parsed_empty = parse_tool_arguments("")
    assert parsed_empty == {}


def test_parse_qwen_tool_call_block_with_function_call() -> None:
    """Verify parse_qwen_tool_call_block intercepts [function call: ...] syntax."""
    block = (
        '[function call: finance.quote(symbol="NVDA")]\n'
        '[function call: news.headlines(category="tech")]'
    )
    calls = parse_qwen_tool_call_block(block)
    assert len(calls) == 2

    assert calls[0]["function"]["name"] == "finance.quote"
    args0 = json.loads(calls[0]["function"]["arguments"])
    assert args0["symbol"] == "NVDA"

    assert calls[1]["function"]["name"] == "news.headlines"
    args1 = json.loads(calls[1]["function"]["arguments"])
    assert args1["category"] == "tech"


def test_detect_and_intercept_tool_monologue_with_function_call() -> None:
    """Verify _detect_and_intercept_tool_monologue intercepts raw function call text."""
    known_tools = {"finance.quote", "news.headlines", "system.time"}
    phrase = '[function call: finance.quote(symbol="NVDA")]'
    is_mono, intercepted = _detect_and_intercept_tool_monologue(phrase, known_tools)

    assert is_mono is True
    assert intercepted is not None
    assert intercepted["type"] == "function"
    assert intercepted["function"]["name"] == "finance.quote"
    assert intercepted["function"]["arguments"] == {"symbol": "NVDA"}


def test_extract_motion_from_phrase_strips_json_status() -> None:
    """Verify _extract_motion_from_phrase removes raw JSON timeout error blocks."""
    from robot.core.engine import _extract_motion_from_phrase

    raw = '{\n  "status": "timeout",\n  "error": "Tool timed out after 3.0s"\n}\nOh my sweet Ameya! *waves*'
    cleaned, motion = _extract_motion_from_phrase(raw)

    assert "timeout" not in cleaned
    assert "status" not in cleaned
    assert "Oh my sweet Ameya!" in cleaned
    assert motion == {"action": "wave", "side": "right"}


def test_response_sanitization_strips_leading_json_error() -> None:
    """Verify regex sanitization strips leading JSON timeout blocks from final responses."""
    import re
    from robot.llm import BRACKET_TOOL_CALL_PATTERN

    raw_response = (
        '{\n  "status": "timeout",\n  "error": "Tool timed out after 3.0s"\n}\n'
        "Oh my sweet Ameya, it looks like that search took a little too long! 💕"
    )

    clean = BRACKET_TOOL_CALL_PATTERN.sub("", raw_response)
    clean = re.sub(r"^\s*```(?:json)?\s*\{[\s\S]*?\}\s*```\s*", "", clean, flags=re.DOTALL)
    clean = re.sub(r"^\s*\{[\s\n]*\"(?:status|error)\"\s*:[^}]*\}\s*", "", clean, flags=re.DOTALL)
    clean = clean.strip()

    assert not clean.startswith("{")
    assert "status" not in clean
    assert "error" not in clean
    assert clean.startswith("Oh my sweet Ameya")

