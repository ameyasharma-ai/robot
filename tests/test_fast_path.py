"""Unit tests for deterministic FastPathRouter."""

import pytest
from robot.tools import PermissionManager, ToolRegistry
from robot.tools.builtins import register_builtin_tools
from robot.tools.fast_path import FastPathIntent, FastPathRouter


class DummyEngine:
    """Mock engine providing a clean ToolRegistry for testing."""

    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()


@pytest.fixture
def router() -> FastPathRouter:
    """Fixture providing initialized FastPathRouter with registered builtin tools."""
    engine = DummyEngine()
    register_builtin_tools(engine)
    permissions = PermissionManager()
    return FastPathRouter(registry=engine.tool_registry, permissions=permissions)


def test_classify_time_intents(router: FastPathRouter):
    """Verify unambiguous time queries classify directly to system.time."""
    queries = [
        "What is the time?",
        "what's the time right now",
        "what time is it",
        "tell me the time",
        "current time",
        "what is today's date",
        "what date is it?",
        "what day is it today",
    ]
    for q in queries:
        intent = router.classify(q)
        assert intent is not None, f"Failed to classify: {q}"
        assert intent.tool_name == "system.time"
        assert intent.confidence >= 0.95


def test_classify_hardware_intents(router: FastPathRouter):
    """Verify hardware and battery status queries classify to system.hardware."""
    queries = [
        "what is the battery level?",
        "battery status",
        "how much battery is left",
        "system status",
        "hardware status",
        "system specs",
    ]
    for q in queries:
        intent = router.classify(q)
        assert intent is not None, f"Failed to classify: {q}"
        assert intent.tool_name == "system.hardware"


def test_classify_display_brightness_intents(router: FastPathRouter):
    """Verify display brightness get and set commands extract levels accurately."""
    intent_get = router.classify("what is the display brightness?")
    assert intent_get is not None
    assert intent_get.tool_name == "os.display_brightness"
    assert intent_get.arguments.get("action") == "get"

    intent_set = router.classify("set display brightness to 60%")
    assert intent_set is not None
    assert intent_set.tool_name == "os.display_brightness"
    assert intent_set.arguments.get("action") == "set"
    assert intent_set.arguments.get("level_percent") == 60

    intent_change = router.classify("change screen brightness to 80")
    assert intent_change is not None
    assert intent_change.arguments.get("level_percent") == 80


def test_classify_ambiguity_rejection(router: FastPathRouter):
    """Verify ambiguous, complex, or conversational queries return None to fall back to LLM."""
    ambiguous = [
        "What time does the sunset occur?",
        "What is the best time to exercise?",
        "Tell me a story about time travel",
        "How is the weather today?",
        "Play some jazz music",
        "Can you remember that my favorite color is blue?",
    ]
    for q in ambiguous:
        intent = router.classify(q)
        assert intent is None, f"Incorrectly classified ambiguous query: {q}"


@pytest.mark.asyncio
async def test_fast_path_execution_and_response(router: FastPathRouter):
    """Verify fast-path intent executes through permission manager and returns natural speech."""
    intent = router.classify("What time is it right now?")
    assert intent is not None

    outcome, speech = await router.execute_intent(intent)
    assert "result" in outcome
    assert "It is" in speech
    assert ":" in speech  # Contains clock time
