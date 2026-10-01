"""Tests for memory subsystem (SQLite history)."""

import asyncio
import os
import tempfile

import pytest

from robot.memory import ConversationHistory


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.mark.asyncio
async def test_history_init(tmp_db):
    history = ConversationHistory(tmp_db)
    await history.initialize()
    assert await history.health_check() is True


@pytest.mark.asyncio
async def test_history_add_and_get(tmp_db):
    history = ConversationHistory(tmp_db)
    await history.initialize()

    await history.add_message("session-1", "user", "Hello robot")
    await history.add_message("session-1", "assistant", "Hello! How can I help?")

    recent = await history.get_recent("session-1")
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
    assert recent[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_history_fts_search(tmp_db):
    history = ConversationHistory(tmp_db)
    await history.initialize()

    await history.add_message("session-1", "user", "My favorite color is blue")
    await history.add_message("session-1", "assistant", "Noted!")

    results = await history.search("favorite color")
    assert len(results) >= 1
    assert "blue" in results[0]["content"]


@pytest.mark.asyncio
async def test_memory_manager_get_relevant_memories_formats(tmp_db):
    from unittest.mock import AsyncMock
    from robot.memory import MemoryManager
    from robot.core.context import _format_memories

    mgr = MemoryManager({}, tmp_db)
    await mgr.history.initialize()
    await mgr.history.add_fact("User loves Python", user_id="user")

    # Mock mem0.search returning various formats (dict with results, list with strings, list with dicts)
    mgr.mem0.search = AsyncMock(return_value=[
        {"id": "1", "memory": "User enjoys tea"},
        "Direct text string memory",
        {"id": "2", "text": "Alternative text field memory"}
    ])

    results = await mgr.get_relevant_memories("user", limit=5)
    assert len(results) == 4
    mem_texts = [r.get("memory") for r in results]
    assert "User loves Python" in mem_texts
    assert "User enjoys tea" in mem_texts
    assert "Direct text string memory" in mem_texts

    # Test context._format_memories
    formatted = _format_memories(results, budget_tokens=500)
    assert "User loves Python" in formatted
    assert "Direct text string memory" in formatted
