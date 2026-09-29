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
