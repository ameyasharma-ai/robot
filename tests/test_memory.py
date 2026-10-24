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


@pytest.mark.asyncio
async def test_history_search_broad_queries_and_deduplication(tmp_db):
    history = ConversationHistory(tmp_db)
    await history.initialize()

    # Add multiple near-duplicate facts and distinct facts
    await history.add_fact("Ameya's crush is Anubha Anand", user_id="user")
    await history.add_fact("Ameya's crush is Anubha Anand, and he finds her cold", user_id="user")
    await history.add_fact("User favorite food is spicy chicken ramen", user_id="user")
    await history.add_fact("User favorite color is electric blue", user_id="user")

    # Broad query: should retrieve distinct facts without redundant crush duplicates
    results = await history.search_facts("What durable facts and memories do you have saved about me?", user_id="user", limit=5)
    assert len(results) >= 2
    memories = [r["memory"] for r in results]
    # Check that diverse facts are retrieved
    assert any("ramen" in m for m in memories)
    assert any("blue" in m for m in memories)
    assert any("Anubha" in m for m in memories)
    # Check that crush is not duplicated multiple times
    anubha_matches = [m for m in memories if "Anubha" in m]
    assert len(anubha_matches) == 1

    # query="all" should also return all distinct facts
    all_res = await history.search_facts("all", user_id="user", limit=10)
    assert len(all_res) >= 3


@pytest.mark.asyncio
async def test_memory_all_and_search_tool_execution(tmp_db):
    from unittest.mock import MagicMock
    from robot.tools import ToolRegistry, PermissionManager
    from robot.tools.builtins import register_builtin_tools
    from robot.memory import MemoryManager

    mem = MemoryManager({}, tmp_db)
    await mem.history.initialize()
    await mem.history.add_fact("Ameya lives in Mumbai", user_id="default")
    await mem.history.add_fact("Ameya favorite food is ramen", user_id="default")

    mock_engine = MagicMock()
    mock_engine.tool_registry = ToolRegistry()
    mock_engine.memory = mem
    mock_engine.personalities.active = MagicMock(memory_namespace="default")

    register_builtin_tools(mock_engine)

    # Verify memory.all tool is registered
    tool_all = mock_engine.tool_registry.get("memory.all")
    assert tool_all is not None

    res_all = await tool_all.handler(limit=10)
    assert len(res_all) == 2
    facts = [f["fact"] for f in res_all]
    assert "Ameya lives in Mumbai" in facts
    assert "Ameya favorite food is ramen" in facts

    # Verify memory.search with query="all" or empty string
    tool_search = mock_engine.tool_registry.get("memory.search")
    res_search = await tool_search.handler(query="all")
    assert len(res_search) == 2



@pytest.mark.asyncio
async def test_archive_is_scoped_to_one_persona(tmp_db):
    """The conversation archive must not leak another persona's history."""
    history = ConversationHistory(tmp_db)
    await history.initialize()

    await history.add_message("s-neo", "user", "neo private note", personality="neo")
    await history.add_message("s-luna", "user", "luna private note", personality="luna")

    luna = await history.get_archive(personality="luna")
    assert [row["content"] for row in luna] == ["luna private note"]

    neo = await history.get_archive(personality="neo")
    assert [row["content"] for row in neo] == ["neo private note"]


@pytest.mark.asyncio
async def test_archive_row_limit_is_bounded(tmp_db):
    """A hostile limit cannot pull the entire history into one context build."""
    history = ConversationHistory(tmp_db)
    await history.initialize()
    for i in range(5):
        await history.add_message("s-1", "user", f"note {i}", personality="neo")

    capped = await history.get_archive(limit=10**9, personality="neo")
    assert len(capped) == 5


@pytest.mark.asyncio
async def test_persona_expression_indexes_exist(tmp_db):
    """Persona predicates are expression-based, so they need expression indexes."""
    import sqlite3

    history = ConversationHistory(tmp_db)
    await history.initialize()

    conn = sqlite3.connect(tmp_db)
    try:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    finally:
        conn.close()

    assert "idx_conversations_persona_norm" in names
    assert "idx_conversations_session_persona_norm" in names
    assert "idx_sessions_persona_norm" in names
    # Superseded plain-column composites must not survive: they cost writes, serve no read.
    assert "idx_conversations_session_persona" not in names


def test_mem0_client_calls_are_exclusive():
    """Timed-out mem0 calls keep talking to the client, so access is serialized."""
    import threading
    import time

    from robot.memory import Mem0Provider

    provider = Mem0Provider({"timeout_seconds": 5.0})
    active = []
    peak = []

    def work():
        active.append(1)
        peak.append(len(active))
        time.sleep(0.01)
        active.pop()
        return "done"

    threads = [threading.Thread(target=provider._call_client, args=(work,)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max(peak) == 1


def test_mem0_client_call_fails_fast_when_backend_is_busy():
    """A busy store cannot queue threads without bound; it reports a timeout."""
    from robot.memory import Mem0Provider

    provider = Mem0Provider({"timeout_seconds": 0.05})
    assert provider._client_lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            provider._call_client(lambda: "unreachable")
    finally:
        provider._client_lock.release()

    assert provider._call_client(lambda: "value") == "value"
