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

    # 6. Test /api/chat/speak synthesis endpoint (clause-concatenated with prosody)
    import json
    import numpy as np
    mock_tts = MagicMock()
    mock_tts.synthesize = AsyncMock(return_value=(np.zeros(1600, dtype=np.float32), 16000))
    mock_engine.tts = mock_tts

    resp = client.post("/api/chat/speak", json={"text": "Hello world from chat mode"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 44

    # Test multi-clause text
    resp_multi = client.post("/api/chat/speak", json={"text": "Hello world! How are you doing? I am feeling great."})
    assert resp_multi.status_code == 200
    assert resp_multi.headers["content-type"] == "audio/wav"

    # Empty text validation error
    resp_empty = client.post("/api/chat/speak", json={"text": "   "})
    assert resp_empty.status_code == 400

    # 7. Test /api/chat/speak/stream SSE streaming endpoint
    stream_resp = client.post(
        "/api/chat/speak/stream",
        json={"text": "Hello world! This is phrase two. And here is phrase three!"},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]

    events = []
    for line in stream_resp.iter_lines():
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            events.append(payload)

    # Verify phrase events received
    phrase_events = [e for e in events if e.get("event") == "phrase"]
    done_events = [e for e in events if e.get("event") == "done"]

    assert len(phrase_events) >= 2
    assert len(done_events) == 1
    assert "audio" in phrase_events[0]
    assert "emotion" in phrase_events[0]
    assert phrase_events[0]["index"] == 0

    # Stream empty text validation error
    stream_empty = client.post("/api/chat/speak/stream", json={"text": "   "})
    assert stream_empty.status_code == 400


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


@pytest.mark.asyncio
async def test_engine_stream_text_chat_async_generator():
    """Verify RobotEngine.stream_text_chat consumes real async generator without 'async_generator is not iterable'."""
    from robot.core.engine import RobotEngine
    from robot.core.context import ContextEngine
    from robot.personality import PersonalityManager

    engine = RobotEngine()
    engine.context = ContextEngine()
    engine.personalities = PersonalityManager()
    engine.state = MagicMock()
    engine.state.state.active_personality = "neo"
    engine.state.increment_turn = AsyncMock(return_value=1)

    async def mock_async_stream(messages, tools=None):
        for w in ["Async", " stream", " working", "!"]:
            yield w

    engine.llm = MagicMock()
    engine.llm.stream_chat = mock_async_stream
    engine.memory = None

    chunks = []
    async for event in engine.stream_text_chat("Testing async stream", session_id="test_sess_async"):
        chunks.append(event)

    events = [c["event"] for c in chunks]
    assert "start" in events
    assert "token" in events
    assert "done" in events
    done_event = next(c for c in chunks if c["event"] == "done")
    assert done_event["response"] == "Async stream working!"


@pytest.mark.asyncio
async def test_warm_up_llm_does_not_pollute_session_history():
    """Verify warm_up_llm only primes inference KV cache and does NOT insert dummy turn 0 into memory."""
    from robot.core.engine import RobotEngine
    from robot.personality import PersonalityManager

    engine = RobotEngine()
    engine.personalities = PersonalityManager()
    engine.memory = MagicMock()
    engine.memory.store_turn = AsyncMock()
    engine._session_id = "test_boot_sess"

    engine.llm = MagicMock()
    async def mock_stream(messages, max_tokens=1, tools=None):
        yield "["
    engine.llm.stream_chat = mock_stream

    duration = await engine.warm_up_llm("neo")
    assert duration >= 0.0
    # Crucial: store_turn must NOT have been called with dummy "["
    assert engine.memory.store_turn.call_count == 0


@pytest.mark.asyncio
async def test_get_recent_filters_single_bracket_garbage(tmp_path):
    """Verify ConversationHistory.get_recent filters out corrupted single brackets ([ or ]) from chat context."""
    from robot.memory import ConversationHistory
    db_file = str(tmp_path / "test_filter.db")
    hist = ConversationHistory(db_file)
    await hist.initialize()

    sid = "sess_filter_test"
    await hist.create_session(sid, personality="luna", title="Filter Test")
    await hist.add_message(sid, "user", "Hello robot", "luna", 0)
    await hist.add_message(sid, "assistant", "[", "luna", 0)  # Corrupted single bracket
    await hist.add_message(sid, "user", "How are you?", "luna", 1)
    await hist.add_message(sid, "assistant", "I am doing wonderful, Ameya!", "luna", 1)

    recent = await hist.get_recent(sid, limit=10)
    contents = [r["content"] for r in recent]
    assert "[" not in contents
    assert "Hello robot" in contents
    assert "How are you?" in contents
    assert "I am doing wonderful, Ameya!" in contents


@pytest.mark.asyncio
async def test_stream_text_chat_filters_tool_call_token():
    """Verify stream_text_chat does not leak __TOOL_CALL__ chunks into client SSE stream or persisted text."""
    import json
    from unittest.mock import AsyncMock, MagicMock
    from robot.core.engine import RobotEngine

    engine = MagicMock()
    engine.state = MagicMock()
    engine.state.state.active_personality = "luna"
    engine.state.get_personality.return_value = "luna"
    engine.state.increment_turn = AsyncMock(return_value=1)
    engine.personalities = MagicMock()
    engine.personalities.active = MagicMock()
    engine.personalities.active.system_prompt = "You are Luna."
    engine.memory = MagicMock()
    engine.memory.get_recent_conversation = AsyncMock(return_value=[])
    engine.memory.get_relevant_memories = AsyncMock(return_value=[])
    engine.memory.store_turn = AsyncMock()
    engine.perception = None
    engine.context = MagicMock()
    engine.context.build_messages.return_value = [{"role": "user", "content": "Remember my favorite color"}]
    engine.config = MagicMock()
    engine.config.voice.rate_limit_enabled = False
    engine.config.memory.enabled = False
    engine._session_id = "test_sess"
    engine.observability = MagicMock()
    engine.observability.current_trace_id = "test_trace"
    engine.events = MagicMock()
    engine.events.emit = AsyncMock()
    engine.tools = MagicMock()
    engine.tools.get_tool_definitions.return_value = [{"type": "function", "function": {"name": "memory.write"}}]
    engine.tools.execute = AsyncMock(return_value={"status": "saved"})

    # LLM yields __TOOL_CALL__[{"name": "memory.write", ...}] without colon
    tc_json = json.dumps([{"id": "call_123", "type": "function", "function": {"name": "memory.write", "arguments": '{"fact": "test"}'}}])
    tc_chunk = f"__TOOL_CALL__{tc_json}"
    second_pass_chunks = ["Got it, ", "I have remembered that for you!"]

    async def mock_stream_chat(messages, tools=None):
        if tools is not None:
            yield tc_chunk
        else:
            for c in second_pass_chunks:
                yield c

    engine.llm = MagicMock()
    engine.llm.stream_chat = mock_stream_chat

    events = []
    async for event in RobotEngine.stream_text_chat(engine, "Remember my favorite color", session_id="test_sess"):
        events.append(event)

    tokens = [e["token"] for e in events if e.get("event") == "token"]
    for t in tokens:
        assert "__TOOL_CALL__" not in t

    # Ensure second pass tokens were yielded
    assert "Got it, " in tokens
    assert "I have remembered that for you!" in tokens

    # Ensure stored turn in memory contains the clean response, not the raw tool call
    engine.memory.store_turn.assert_called_once()
    stored_kwargs = engine.memory.store_turn.call_args.kwargs
    assert "__TOOL_CALL__" not in stored_kwargs["bot_text"]
    assert stored_kwargs["bot_text"] == "Got it, I have remembered that for you!"
    assert stored_kwargs["tool_name"] == "memory.write"


@pytest.mark.asyncio
async def test_transcribe_audio_endpoint():
    """Verify /api/transcribe processes audio WAV data and returns transcribed text."""
    from unittest.mock import AsyncMock, MagicMock
    from starlette.testclient import TestClient
    from robot.api import create_app

    engine = MagicMock()
    engine.stt = MagicMock()
    engine.stt.transcribe = AsyncMock(return_value="Testing audio dictation")
    engine.player = MagicMock()
    engine.player.is_playing = False
    engine.capture = MagicMock()
    engine.telemetry = MagicMock()
    engine.telemetry.snapshot.return_value = {}

    app = create_app(engine)
    client = TestClient(app)

    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 3200)
    fake_wav = buf.getvalue()

    response = client.post(
        "/api/transcribe",
        content=fake_wav,
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Testing audio dictation"
    assert engine.stt.transcribe.call_count == 1


@pytest.mark.asyncio
async def test_transcribe_audio_empty():
    """Verify /api/transcribe handles empty audio payload gracefully."""
    from unittest.mock import MagicMock
    from starlette.testclient import TestClient
    from robot.api import create_app

    engine = MagicMock()
    engine.stt = MagicMock()
    engine.player = MagicMock()
    engine.player.is_playing = False
    engine.capture = MagicMock()
    engine.telemetry = MagicMock()
    engine.telemetry.snapshot.return_value = {}

    app = create_app(engine)
    client = TestClient(app)

    response = client.post(
        "/api/transcribe",
        content=b"",
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == ""







