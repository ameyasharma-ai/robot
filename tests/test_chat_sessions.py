"""Tests for ChatGPT-like chat interface, isolated sessions, shared memory, and speech/music bug fixes."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from robot.api import create_app
from robot.core.state import RobotMode, RobotState
from robot.memory import ConversationHistory, MemoryManager


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.mark.asyncio
async def test_session_lifecycle_in_history(tmp_db):
    """Test creating, listing, reading messages, updating title, and deleting sessions in ConversationHistory."""
    history = ConversationHistory(tmp_db)
    await history.initialize()

    # Create session
    sess = await history.create_session("sess-andromeda", title="Project Andromeda")
    sess_id = sess["id"]
    assert sess["title"] == "Project Andromeda"

    # Add messages to this session
    await history.add_message(sess_id, "user", "What is Andromeda?")
    await history.add_message(
        sess_id,
        "assistant",
        "It is a spiral galaxy.",
        tool_name="web_search",
        tool_result={"query": "Andromeda", "results": "Spiral galaxy"},
        metadata={"tokens": 12, "tokens_per_sec": 48.0},
    )

    # List sessions
    sessions = await history.list_sessions()
    assert len(sessions) >= 1
    found = next((s for s in sessions if s["id"] == sess_id), None)
    assert found is not None
    assert found["title"] == "Project Andromeda"

    # Get session messages
    messages = await history.get_session_messages(sess_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is Andromeda?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_name"] == "web_search"
    assert isinstance(messages[1]["tool_result"], dict)
    assert messages[1]["metadata"]["tokens"] == 12

    # Update session title
    await history.update_session_title(sess_id, "Deep Space & Andromeda")
    sessions_updated = await history.list_sessions()
    found_updated = next((s for s in sessions_updated if s["id"] == sess_id), None)
    assert found_updated["title"] == "Deep Space & Andromeda"

    # Delete session
    deleted = await history.delete_session(sess_id)
    assert deleted is True
    sessions_after_del = await history.list_sessions()
    assert not any(s["id"] == sess_id for s in sessions_after_del)
    # Associated messages should also be cleared
    messages_after_del = await history.get_session_messages(sess_id)
    assert len(messages_after_del) == 0


@pytest.mark.asyncio
async def test_memory_manager_session_operations(tmp_db):
    """Test MemoryManager wraps session methods cleanly while keeping facts globally shared."""
    mgr = MemoryManager({}, tmp_db)
    await mgr.history.initialize()

    # Add shared fact
    await mgr.history.add_fact("User is Ameya", user_id="user")

    # Create two separate sessions
    s1 = await mgr.create_session("s1", title="Session Alpha")
    s2 = await mgr.create_session("s2", title="Session Beta")
    assert s1["id"] != s2["id"]

    # Store turn in s1
    await mgr.store_turn(
        s1["id"],
        "My code is in Python",
        "Great choice!",
        tool_name=None,
        tool_result=None,
        metadata={"speed": 50},
    )

    # Store turn in s2
    await mgr.store_turn(
        s2["id"],
        "Let's write Rust instead",
        "Rust is very fast.",
        tool_name=None,
        tool_result=None,
        metadata={"speed": 60},
    )

    # Verify session isolation: s1 only has Python turn, s2 only has Rust turn
    m1 = await mgr.get_session_messages(s1["id"])
    m2 = await mgr.get_session_messages(s2["id"])
    assert len(m1) == 2
    assert "Python" in m1[0]["content"]
    assert len(m2) == 2
    assert "Rust" in m2[0]["content"]

    # Verify shared memory: global facts are accessible regardless of session
    facts = await mgr.history.get_facts("user")
    assert len(facts) >= 1
    assert any("Ameya" in f["fact"] for f in facts)


def test_api_chat_endpoints(tmp_db):
    """Test FastAPI REST endpoints for chat sessions and streaming."""
    from robot.core.state import StateManager

    # Mock engine
    mock_engine = MagicMock()
    mock_engine.state = StateManager()
    mock_engine.memory = MemoryManager({}, tmp_db)
    asyncio.run(mock_engine.memory.history.initialize())
    mock_engine._is_speaking = False
    mock_engine.is_speaking = False
    mock_engine.is_music_playing = False
    mock_engine.player = MagicMock()
    mock_engine.player.is_playing = False
    mock_engine.capture = MagicMock()
    mock_engine.telemetry = MagicMock()
    mock_engine.telemetry.snapshot.return_value = {}

    app = create_app(mock_engine)
    client = TestClient(app)

    # 1. Create a session
    resp = client.post("/chat/sessions", json={"title": "Cosmic Chat"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["title"] == "Cosmic Chat"
    sess_id = data["session_id"]

    # 2. List sessions
    resp = client.get("/chat/sessions")
    assert resp.status_code == 200
    sessions_data = resp.json()
    assert "sessions" in sessions_data
    assert any(s["id"] == sess_id for s in sessions_data["sessions"])

    # 3. Get session messages
    resp = client.get(f"/chat/sessions/{sess_id}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Cosmic Chat"
    assert resp.json()["messages"] == []

    # 4. Status endpoint includes is_speaking and music_playing
    resp = client.get("/status")
    assert resp.status_code == 200
    st = resp.json()
    assert st["is_speaking"] is False
    assert st["music_playing"] is False

    # 5. Delete session
    resp = client.delete(f"/chat/sessions/{sess_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Confirm deleted
    resp = client.get(f"/chat/sessions/{sess_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_coordinator_music_play_enqueues_sentinel():
    """Verify that coordinator music.play enqueues None into _tts_queue so speaking mode never hangs."""
    from robot.tools.coordinator import ToolCategory, ToolContextState, ToolConversationCoordinator

    mock_engine = MagicMock()
    mock_engine._tts_queue = asyncio.Queue()
    mock_engine.observability = MagicMock()
    mock_engine.observability.current_trace_id = "test-trace"
    mock_engine._session_id = "test-sess"
    mock_engine.events = MagicMock()
    mock_engine.events.emit = AsyncMock()

    coordinator = ToolConversationCoordinator(mock_engine)

    ctx = ToolContextState(
        request_id="req-1",
        trace_id="trace-1",
        conversation_id="sess-1",
        tool_call_id="call-1",
        tool_name="music.play",
        tool_args={"query": "cyberpunk synthwave"},
        category=ToolCategory.LLM_REQUIRED,
    )
    ctx.tool_result = {"status": "playing", "title": "Cyberpunk Synthwave"}

    interrupt = asyncio.Event()
    res = await coordinator._generate_continuation(
        ctx=ctx,
        user_text="play music",
        end_of_speech_time=0.0,
        interrupt=interrupt,
        messages=[],
    )

    assert "Cyberpunk Synthwave" in res
    # First item is speech ack tuple, second item is None sentinel
    item1 = await asyncio.wait_for(mock_engine._tts_queue.get(), timeout=1.0)
    assert isinstance(item1, tuple)
    item2 = await asyncio.wait_for(mock_engine._tts_queue.get(), timeout=1.0)
    assert item2 is None


@pytest.mark.asyncio
async def test_historical_voice_session_backfill(tmp_db):
    """Test that historical voice chats stored in conversations table are automatically migrated/available in sessions list."""
    import sqlite3
    # Seed raw historical conversations before history or sessions table is created
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        """CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            personality TEXT DEFAULT 'neo',
            turn_number INTEGER DEFAULT 0,
            tool_name TEXT,
            tool_result TEXT,
            timestamp TEXT NOT NULL,
            metadata TEXT
        )"""
    )
    conn.execute(
        """INSERT INTO conversations (session_id, role, content, personality, turn_number, timestamp)
           VALUES ('voice_hist_1', 'user', 'Hello robot from voice', 'luna', 1, '2026-10-01T12:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO conversations (session_id, role, content, personality, turn_number, timestamp)
           VALUES ('voice_hist_1', 'assistant', 'Hello human!', 'luna', 1, '2026-10-01T12:00:05+00:00')"""
    )
    conn.commit()
    conn.close()

    # Initialize ConversationHistory
    history = ConversationHistory(tmp_db)
    await history.initialize()

    # Sessions list should contain the historical voice session
    sessions = await history.list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["id"] == "voice_hist_1"
    assert "Hello robot from voice" in s["title"]
    assert s["personality"] == "luna"

    # get_session and get_session_messages should return the historical voice turns
    sess_meta = await history.get_session("voice_hist_1")
    assert sess_meta is not None
    assert sess_meta["id"] == "voice_hist_1"

    msgs = await history.get_session_messages("voice_hist_1")
    assert len(msgs) == 2
    assert msgs[0]["content"] == "Hello robot from voice"
    assert msgs[1]["content"] == "Hello human!"


def test_personality_active_prompt():
    """Verify PersonalityManager exposes active_prompt property cleanly."""
    from robot.personality import Personality, PersonalityManager

    pm = PersonalityManager()
    p = Personality(name="test_bot", system_prompt="You are a helpful AI.")
    pm._personalities["test_bot"] = p
    pm._active = "test_bot"

    assert pm.active_prompt == "You are a helpful AI."


@pytest.mark.asyncio
async def test_engine_stream_text_chat():
    """Verify RobotEngine.stream_text_chat generates tokens without TypeError."""
    from robot.core.engine import RobotEngine
    from robot.core.context import ContextEngine
    from robot.personality import PersonalityManager

    engine = RobotEngine()
    engine.context = ContextEngine()
    engine.personalities = PersonalityManager()
    engine.state = MagicMock()
    engine.state.state.active_personality = "neo"
    engine.state.increment_turn = AsyncMock(return_value=1)
    engine.llm = MagicMock()
    engine.llm.stream_chat.return_value = ["Hello", " world"]
    engine.memory = None

    chunks = []
    async for event in engine.stream_text_chat("Hi there", session_id="test_sess"):
        chunks.append(event)

    events = [c["event"] for c in chunks]
    assert "start" in events
    assert "token" in events
    assert "done" in events
    done_event = next(c for c in chunks if c["event"] == "done")
    assert done_event["response"] == "Hello world"


