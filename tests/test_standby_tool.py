"""Unit tests for system.stop_listening tool and standby state transitions."""

from __future__ import annotations

import pytest

from robot.core.engine import RobotEngine
from robot.core.state import RobotMode
from robot.tools.builtins import register_builtin_tools


@pytest.mark.asyncio
async def test_system_stop_listening_tool() -> None:
    """Test that system.stop_listening sets mode to STANDBY and returns expected message."""
    engine = RobotEngine()
    register_builtin_tools(engine)

    tool_def = engine.tool_registry.get("system.stop_listening")
    assert tool_def is not None
    assert "standby" in tool_def.description.lower()

    # Engine initial mode
    await engine.state.set_mode(RobotMode.LISTENING)
    assert engine.state.state.mode == RobotMode.LISTENING

    # Execute tool handler
    result = await tool_def.handler()
    assert "standby" in result.lower()
    assert engine.state.state.mode == RobotMode.STANDBY
