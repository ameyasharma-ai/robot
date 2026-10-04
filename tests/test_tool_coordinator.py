"""Unit tests for ToolConversationCoordinator and Zero-Perceived-Latency Tool Calling."""

import asyncio
import time
import pytest

from robot.core.events import EventBus
from robot.observability import ObservabilityHub
from robot.tools import (
    CoordinatorState,
    HoldingStatus,
    PermissionLevel,
    PermissionManager,
    ToolCategory,
    ToolContextState,
    ToolConversationCoordinator,
    ToolDefinition,
    ToolExecutionStatus,
    ToolRegistry,
)


class DummyLLM:
    """Mock streaming LLM for testing holding and continuation streams."""

    def __init__(self, holding_tokens=None, continuation_tokens=None, delay=0.01):
        self.holding_tokens = holding_tokens or ["Sure, ", "checking ", "the latest ", "listings ", "now. "]
        self.continuation_tokens = continuation_tokens or ["The price ", "is ", "1999 dollars."]
        self.delay = delay
        self.call_history = []

    async def stream_chat(self, messages, max_tokens=300, tools=None):
        self.call_history.append(messages)
        # Check if this is holding prompt
        is_holding = any("conversational holding voice" in str(m.get("content", "")) for m in messages)
        tokens = self.holding_tokens if is_holding else self.continuation_tokens
        for token in tokens:
            await asyncio.sleep(self.delay)
            yield token


class DummyEngine:
    """Mock RobotEngine for testing coordinator in isolation."""

    def __init__(self, llm=None):
        self.events = EventBus()
        self.observability = ObservabilityHub(self.events)
        self.tool_registry = ToolRegistry()
        self.permissions = PermissionManager(event_bus=self.events)
        self.llm = llm or DummyLLM()
        self.config = {
            "tools": {"max_holding_duration_ms": 500, "holding_max_tokens": 15},
            "llm": {"max_tokens": 100},
        }
        self.personalities = DummyPersonalities()
        self._tts_queue = asyncio.Queue()
        self._session_id = "test_sess_123"

        class DummyTelemetry:
            def record(self, *args, **kwargs):
                pass

        self.telemetry = DummyTelemetry()


class DummyPersonalities:
    class DummyPers:
        system_prompt = "You are a test companion."
        tts_settings = {"speed": 1.0, "pitch": 1.0}
    active = DummyPers()


def test_fast_router_intent_extraction():
    engine = DummyEngine()
    coordinator = ToolConversationCoordinator(engine, engine.config)

    # 1. System time
    res = coordinator.fast_route("what time is it?")
    assert res is not None
    assert res[0] == "system.time"
    assert res[2] == ToolCategory.DIRECT

    # 2. Web search
    res = coordinator.fast_route("Search the web for latest RTX 5090 price")
    assert res is not None
    assert res[0] == "web.search"
    assert "RTX 5090" in res[1]["query"]
    assert res[2] == ToolCategory.LLM_REQUIRED

    # 3. Perception
    res = coordinator.fast_route("What do you see right now?")
    assert res is not None
    assert res[0] == "perception.what_do_you_see"
    assert res[2] == ToolCategory.DIRECT

    # 4. Image generate
    res = coordinator.fast_route("Generate an image of a neon cybernetic city")
    assert res is not None
    assert res[0] == "image.generate"
    assert "neon cybernetic city" in res[1]["prompt"]
    assert res[2] == ToolCategory.ASYNC

    # 5. Motion action
    res = coordinator.fast_route("Wave your hand")
    assert res is not None
    assert res[0] == "character.execute_action"
    assert res[1]["action"] == "wave"

    # 6. Memory search
    res = coordinator.fast_route("Who is Anubha?")
    assert res is not None
    assert res[0] == "memory.search"
    assert "Anubha" in res[1]["query"]

    # 7. Unmatched chit-chat
    res = coordinator.fast_route("Hi there, how are you?")
    assert res is None


@pytest.mark.asyncio
async def test_concurrent_tool_and_holding_conversation():
    llm = DummyLLM(
        holding_tokens=["Looking ", "that ", "up ", "online. "],
        continuation_tokens=["Found it. ", "The MSRP ", "is ", "$1999."],
        delay=0.01,
    )
    engine = DummyEngine(llm)
    coordinator = ToolConversationCoordinator(engine, engine.config)

    # Register slow tool (takes 60ms)
    async def slow_search(query: str):
        await asyncio.sleep(0.06)
        return [{"title": "RTX 5090", "snippet": "Starting at $1999"}]

    engine.tool_registry.register(
        ToolDefinition(
            name="web.search",
            description="Web search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=slow_search,
            family="web",
        )
    )

    interrupt = asyncio.Event()
    messages = [{"role": "system", "content": "You are a companion."}]

    t_start = time.perf_counter()
    response = await coordinator.coordinate(
        tool_name="web.search",
        tool_args={"query": "RTX 5090"},
        category=ToolCategory.LLM_REQUIRED,
        user_text="Search the web for RTX 5090",
        end_of_speech_time=time.time(),
        interrupt=interrupt,
        messages=messages,
    )
    duration = time.perf_counter() - t_start

    # Verify response contains continuation
    assert "MSRP" in response or "1999" in response
    # Verify holding phrases were queued to TTS
    assert not engine._tts_queue.empty()
    queued_phrases = []
    while not engine._tts_queue.empty():
        item = engine._tts_queue.get_nowait()
        if item and item[0]:
            queued_phrases.append(item[0])

    assert len(queued_phrases) >= 1
    assert coordinator.state == CoordinatorState.COMPLETED


@pytest.mark.asyncio
async def test_barge_in_cancellation_during_holding():
    llm = DummyLLM(delay=0.05)
    engine = DummyEngine(llm)
    coordinator = ToolConversationCoordinator(engine, engine.config)

    async def very_slow_tool(query: str):
        await asyncio.sleep(0.5)
        return "Done"

    engine.tool_registry.register(
        ToolDefinition(
            name="web.search",
            description="Search",
            parameters={},
            handler=very_slow_tool,
            family="web",
        )
    )

    interrupt = asyncio.Event()

    async def trigger_interrupt():
        await asyncio.sleep(0.04)
        interrupt.set()

    asyncio.create_task(trigger_interrupt())

    res = await coordinator.coordinate(
        tool_name="web.search",
        tool_args={"query": "test"},
        category=ToolCategory.LLM_REQUIRED,
        user_text="Search something",
        end_of_speech_time=time.time(),
        interrupt=interrupt,
        messages=[],
    )

    assert coordinator.state == CoordinatorState.CANCELLED


@pytest.mark.asyncio
async def test_tool_failure_graceful_handling():
    llm = DummyLLM(
        holding_tokens=["Checking ", "it ", "now. "],
        continuation_tokens=["I couldn't ", "reach the service."],
        delay=0.01,
    )
    engine = DummyEngine(llm)
    coordinator = ToolConversationCoordinator(engine, engine.config)

    async def failing_tool():
        await asyncio.sleep(0.02)
        raise ConnectionError("DNS failure")

    engine.tool_registry.register(
        ToolDefinition(
            name="system.test_fail",
            description="Failing tool",
            parameters={},
            handler=failing_tool,
            family="system",
        )
    )

    interrupt = asyncio.Event()
    res = await coordinator.coordinate(
        tool_name="system.test_fail",
        tool_args={},
        category=ToolCategory.LLM_REQUIRED,
        user_text="Test fail",
        end_of_speech_time=time.time(),
        interrupt=interrupt,
        messages=[],
    )

    assert coordinator.state == CoordinatorState.COMPLETED
    assert coordinator._active_context.tool_status == ToolExecutionStatus.FAILED
    assert "DNS failure" in str(coordinator._active_context.tool_error)


@pytest.mark.asyncio
async def test_direct_tool_fast_path():
    """Verify ultra-fast direct tools skip holding speech overhead."""
    llm = DummyLLM(
        continuation_tokens=["It is ", "currently ", "12:00 PM."],
        delay=0.01,
    )
    engine = DummyEngine(llm)
    coordinator = ToolConversationCoordinator(engine, engine.config)

    async def get_time():
        return {"time": "12:00 PM"}

    engine.tool_registry.register(
        ToolDefinition(
            name="system.time",
            description="Get time",
            parameters={},
            handler=get_time,
            family="system",
        )
    )

    interrupt = asyncio.Event()
    res = await coordinator.coordinate(
        tool_name="system.time",
        tool_args={},
        category=ToolCategory.DIRECT,
        user_text="What time is it?",
        end_of_speech_time=time.time(),
        interrupt=interrupt,
        messages=[],
    )

    assert "12:00 PM" in res
    assert coordinator._active_context.tool_status == ToolExecutionStatus.COMPLETED
    # In direct mode, holding status is NOT_STARTED because holding speech was bypassed
    assert coordinator._active_context.holding_status == HoldingStatus.NOT_STARTED


@pytest.mark.asyncio
async def test_max_holding_duration_cap():
    """Verify holding speech transitions to quiet waiting if tool exceeds max duration."""
    llm = DummyLLM(
        holding_tokens=["Still ", "checking... "],
        continuation_tokens=["Finally ", "got it."],
        delay=0.01,
    )
    engine = DummyEngine(llm)
    # Set cap to 80ms
    engine.config["tools"]["max_holding_duration_ms"] = 80
    coordinator = ToolConversationCoordinator(engine, engine.config)

    async def very_slow_tool():
        await asyncio.sleep(0.12)
        return "Complete"

    engine.tool_registry.register(
        ToolDefinition(
            name="system.slow",
            description="Slow tool",
            parameters={},
            handler=very_slow_tool,
            family="system",
        )
    )

    interrupt = asyncio.Event()
    res = await coordinator.coordinate(
        tool_name="system.slow",
        tool_args={},
        category=ToolCategory.LLM_REQUIRED,
        user_text="Slow task",
        end_of_speech_time=time.time(),
        interrupt=interrupt,
        messages=[],
    )

    assert coordinator.state == CoordinatorState.COMPLETED
    assert coordinator._active_context.tool_status == ToolExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_fast_router_time_and_timezone():
    """Verify fast router accurately captures time queries and timezones."""
    engine = DummyEngine()
    coordinator = ToolConversationCoordinator(engine, engine.config)

    # 1. "What is the time?"
    r1 = coordinator.fast_route("What is the time?")
    assert r1 is not None
    tool, args, cat = r1
    assert tool == "system.time"
    assert cat == ToolCategory.DIRECT

    # 2. "What is the time in India?"
    r2 = coordinator.fast_route("What is the time in India?")
    assert r2 is not None
    tool2, args2, cat2 = r2
    assert tool2 == "system.time"
    assert args2.get("timezone_str") == "india"
    assert cat2 == ToolCategory.DIRECT

    # 3. Perception "what do you see?"
    r3 = coordinator.fast_route("What do you see?")
    assert r3 is not None
    assert r3[0] == "perception.what_do_you_see"
    assert r3[2] == ToolCategory.DIRECT


@pytest.mark.asyncio
async def test_timezone_in_system_time():
    """Verify system.time correctly resolves timezones and formats response."""
    from robot.tools.builtins import register_builtin_tools
    engine = DummyEngine()
    register_builtin_tools(engine)

    time_tool = engine.tool_registry.get("system.time")
    assert time_tool is not None

    res_india = await time_tool.handler(timezone_str="India")
    assert "current_time" in res_india
    assert res_india["timezone"] == "Asia/Kolkata"

    res_utc = await time_tool.handler(timezone_str="UTC")
    assert "current_time" in res_utc
    assert res_utc["timezone"] == "UTC"


@pytest.mark.asyncio
async def test_fast_router_product_search():
    """Verify fast router accurately captures product/info search queries."""
    engine = DummyEngine()
    coordinator = ToolConversationCoordinator(engine, engine.config)

    # 1. "What is the new iPhone"
    r1 = coordinator.fast_route("What is the new iPhone")
    assert r1 is not None
    tool, args, cat = r1
    assert tool == "web.search"
    assert "iPhone" in args.get("query", "")
    assert cat == ToolCategory.LLM_REQUIRED

    # 2. "Tell me about the new RTX 5090"
    r2 = coordinator.fast_route("Tell me about the new RTX 5090")
    assert r2 is not None
    assert r2[0] == "web.search"
    assert "5090" in r2[1].get("query", "")

    # 3. "Who won the match"
    r3 = coordinator.fast_route("Who won the match")
    assert r3 is not None
    assert r3[0] == "web.search"


def test_clause_chunker_with_motion_tag():
    """Verify ClauseChunker splits eager clauses on comma even with leading [motion: ...]."""
    from robot.llm import ClauseChunker
    chunker = ClauseChunker(min_first_words=2)

    tokens = ["[motion: ", "lean_in] ", "The ", "latest ", "iPhone ", "is ", "the ", "iPhone ", "16, ", "dropping ", "next ", "month."]
    emitted = []
    for t in tokens:
        res = chunker.feed(t)
        if res:
            emitted.append(res)

    rem = chunker.flush()
    if rem:
        emitted.append(rem)

    assert len(emitted) >= 2
    assert "[motion: lean_in]" in emitted[0]
    assert "iPhone 16," in emitted[0]


def test_fast_router_event_search():
    """Verify historical / event queries route to web.search and not memory.search."""
    engine = DummyEngine()
    coordinator = ToolConversationCoordinator(engine, engine.config)

    res = coordinator.fast_route("What was the plane hijacking involving an Indian pilot who was attacked.")
    assert res is not None
    assert res[0] == "web.search"
    assert "hijacking" in res[1]["query"]


def test_fast_router_public_entity_vs_memory():
    """Verify public multi-word entities route to web.search while personal queries route to memory."""
    engine = DummyEngine()
    coordinator = ToolConversationCoordinator(engine, engine.config)

    # Public entity queries -> web.search
    for q in ["Who is Andrew Tate", "who is Andrew Tate?", "Who is Elon Musk", "Who is Donald Trump"]:
        route = coordinator.fast_route(q)
        assert route is not None, f"Failed for {q}"
        assert route[0] == "web.search", f"Expected web.search for {q}, got {route[0]}"

    # Personal queries -> memory.search
    for q in ["Who is my girlfriend", "What is my favorite food", "Who is Anubha", "Who is Anubha?"]:
        route = coordinator.fast_route(q)
        assert route is not None, f"Failed for {q}"
        assert route[0] == "memory.search", f"Expected memory.search for {q}, got {route[0]}"


@pytest.mark.asyncio
async def test_memory_miss_failover_to_web():
    """Verify that when memory.search returns empty for an entity, coordinator fails over to web.search."""
    llm = DummyLLM(
        holding_tokens=["Checking ", "my notes... "],
        continuation_tokens=["Andrew Tate is an internet personality and former kickboxer."],
        delay=0.01,
    )
    engine = DummyEngine(llm)
    coordinator = ToolConversationCoordinator(engine, engine.config)

    # memory.search returns empty list
    async def empty_memory(query: str):
        return []

    engine.tool_registry.register(
        ToolDefinition(
            name="memory.search",
            description="Search memory",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=empty_memory,
            permission_level=PermissionLevel.READ_ONLY,
        )
    )

    # web.search returns real results
    web_called = []
    async def mock_web_search(query: str):
        web_called.append(query)
        return [{"title": "Andrew Tate", "snippet": "Internet personality and former kickboxer"}]

    engine.tool_registry.register(
        ToolDefinition(
            name="web.search",
            description="Web search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=mock_web_search,
            permission_level=PermissionLevel.READ_ONLY,
        )
    )

    res = await coordinator.coordinate(
        tool_name="memory.search",
        tool_args={"query": "Andrew Tate"},
        category=ToolCategory.FORMAT,
        user_text="Who is Andrew Tate",
        end_of_speech_time=time.perf_counter(),
        interrupt=asyncio.Event(),
        messages=[],
    )

    assert len(web_called) == 1
    assert web_called[0] == "Andrew Tate"
    assert "Andrew Tate is an internet personality" in res





