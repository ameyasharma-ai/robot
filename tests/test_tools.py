"""Tests for tool registry, router, and permissions."""

import asyncio

from robot.tools import PermissionLevel, PermissionManager, ToolDefinition, ToolRegistry, ToolRouter


async def dummy_handler(**kwargs):
    return {"status": "ok"}


def test_tool_registration():
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="test.tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
        handler=dummy_handler,
        family="test",
    )
    registry.register(tool)
    assert "test.tool" in registry.list_tools()
    assert registry.get("test.tool") is not None


def test_tool_family_lookup():
    registry = ToolRegistry()
    for i in range(3):
        registry.register(
            ToolDefinition(
                name=f"memory.tool_{i}",
                description=f"Tool {i}",
                parameters={},
                handler=dummy_handler,
                family="memory",
            )
        )
    assert len(registry.get_family("memory")) == 3
    assert len(registry.get_family("nonexistent")) == 0


def test_tool_router_selection():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="memory.search", description="Search", parameters={},
            handler=dummy_handler, family="memory",
        )
    )
    router = ToolRouter(registry)
    selected = router.select_tools("Can you remember what I said?")
    assert "memory.search" in selected


def test_permission_check():
    pm = PermissionManager(max_auto_level=PermissionLevel.HARMLESS_LOCAL)
    safe_tool = ToolDefinition(
        name="safe", description="", parameters={}, handler=dummy_handler,
        permission_level=PermissionLevel.READ_ONLY,
    )
    dangerous_tool = ToolDefinition(
        name="dangerous", description="", parameters={}, handler=dummy_handler,
        permission_level=PermissionLevel.SYSTEM,
    )
    assert pm.check(safe_tool) is True
    assert pm.check(dangerous_tool) is False


def test_clean_text_for_speech():
    from robot.tts import clean_text_for_speech
    raw = "**Hello world!** Here is *some* text with `code` and [motion:wave]."
    clean = clean_text_for_speech(raw)
    assert "**" not in clean
    assert "*" not in clean
    assert "`" not in clean
    assert "Hello world! Here is some text with code and ." in clean or "Hello world! Here is some text with code and." in clean


def test_tool_router_motion_and_perception():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="character.execute_action", description="", parameters={}, handler=dummy_handler, family="motion")
    )
    registry.register(
        ToolDefinition(name="perception.inspect_view", description="", parameters={}, handler=dummy_handler, family="perception")
    )
    router = ToolRouter(registry)
    motion_selected = router.select_tools("wave your right hand")
    assert "character.execute_action" in motion_selected

    perception_selected = router.select_tools("what do you see in front of you")
    assert "perception.inspect_view" in perception_selected


def test_extract_motion_from_phrase():
    from robot.core.engine import _extract_motion_from_phrase

    # Explicit marker
    cleaned, motion = _extract_motion_from_phrase("Hello there! [motion: wave] How are you?")
    assert motion == {"action": "wave", "side": "right"}
    assert cleaned == "Hello there!  How are you?"

    # Asterisk stage direction
    cleaned2, motion2 = _extract_motion_from_phrase("I am so excited! *waves* Welcome!")
    assert motion2 == {"action": "wave", "side": "right"}
    assert cleaned2 == "I am so excited!  Welcome!"

    # No motion
    cleaned3, motion3 = _extract_motion_from_phrase("Just a normal sentence.")
    assert motion3 is None
    assert cleaned3 == "Just a normal sentence."


import pytest



@pytest.mark.asyncio
async def test_builtins_perception_what_do_you_see():
    from unittest.mock import MagicMock
    from robot.tools.builtins.perception import register_perception_tools

    engine = MagicMock()
    engine.tool_registry = ToolRegistry()
    engine.perception = MagicMock()
    engine.perception.world_state.to_compact.return_value = {
        "people": [{"position": "center", "distance": "near"}],
        "objects": [{"label": "cup", "position": "center"}],
        "lighting": "bright",
        "motion": "subtle",
    }

    register_perception_tools(engine)
    tool = engine.tool_registry.get("perception.what_do_you_see")
    assert tool is not None
    result = await tool.handler()
    assert "cup at center" in result["summary"]
    assert len(result["people"]) == 1
    assert len(result["objects"]) == 1


@pytest.mark.asyncio
async def test_all_15_tools_functional():
    import numpy as np
    from unittest.mock import MagicMock, AsyncMock
    from robot.tools.builtins import register_builtin_tools

    engine = MagicMock()
    engine.tool_registry = ToolRegistry()
    engine.personalities = MagicMock()
    engine.personalities.active = MagicMock(memory_namespace="test_ns")
    engine.state = MagicMock()
    engine.state.state.to_compact.return_value = {"mode": "idle"}
    engine.telemetry = MagicMock()
    engine.telemetry.snapshot.return_value = {}
    engine.memory = MagicMock()
    engine.memory.add_memory = AsyncMock(return_value=None)
    engine.memory.get_relevant_memories = AsyncMock(return_value=[])
    engine.memory.search_history = AsyncMock(return_value=[])
    engine.perception = MagicMock()
    engine.perception.camera.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.perception.world_state.to_compact.return_value = {
        "people": [{"position": "center", "distance": "near"}],
        "objects": [{"label": "bottle", "position": "right"}],
        "lighting": "normal",
        "motion": "none",
    }
    engine.events = MagicMock()
    engine.events.emit = AsyncMock(return_value=None)
    engine._pending_motion = None

    register_builtin_tools(engine)
    tools = engine.tool_registry.list_tools()
    assert len(tools) >= 13

    for name in tools:
        t = engine.tool_registry.get(name)
        assert t is not None
        if name == "memory.write":
            res = await t.handler(fact="Test fact")
            assert "Test fact" in str(res)
        elif name in ("memory.search", "history.search"):
            res = await t.handler(query="test")
            assert isinstance(res, list)
        elif name == "character.execute_action":
            res = await t.handler(action="wave")
            assert res.get("status") == "executing"
        elif name == "character.stop":
            res = await t.handler()
            assert res.get("status") == "stopped"
        elif name == "personality.switch":
            engine.personalities.set_active.return_value = True
            res = await t.handler(name="cypher")
            assert "cypher" in str(res)
        elif name == "system.time":
            res = await t.handler()
            assert "current_time" in res
        elif name == "system.status":
            res = await t.handler()
            assert "state" in res
        elif name in ("perception.what_do_you_see", "perception.inspect_view"):
            res = await t.handler()
            assert "summary" in res
        elif name == "perception.get_world_state":
            res = await t.handler()
            assert "people" in res
        elif name == "camera.capture_snapshot":
            res = await t.handler()
            assert res.get("format") == "jpeg"


