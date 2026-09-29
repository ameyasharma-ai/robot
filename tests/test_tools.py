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
