"""Unit tests for AMD iGPU Dual-Model Local VLM Grounding and tabletop inspection."""

from __future__ import annotations

import numpy as np
import pytest

from robot.core.engine import RobotEngine
from robot.perception.vlm import LocalVLMGrounding
from robot.tools.builtins import register_builtin_tools


def test_vlm_grounding_initialization() -> None:
    """Test LocalVLMGrounding initialization with AMD iGPU device setting."""
    vlm = LocalVLMGrounding(enabled=True, device="amd_igpu")
    assert vlm.enabled is True
    assert vlm.device == "amd_igpu"


def test_vlm_synthetic_frame_inspection() -> None:
    """Test tabletop inspection on a synthetic multi-object image."""
    vlm = LocalVLMGrounding(enabled=True, device="amd_igpu")

    # Create a synthetic 640x480 test frame with a white background and colored tabletop objects
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 220  # light background

    # Draw a blue mug/box on the left
    frame[200:350, 100:220] = [200, 50, 20]  # BGR blue

    # Draw a green notebook on the right
    frame[250:400, 420:580] = [30, 180, 40]  # BGR green

    result = vlm.inspect_frame(frame, query="Inspect all tabletop items")

    assert "summary" in result
    assert result["items_count"] >= 2
    assert result["device"] == "amd_igpu"
    assert len(result["items"]) >= 2
    assert any("blue" in item["color"] or "cyan" in item["color"] for item in result["items"])


@pytest.mark.asyncio
async def test_inspect_tabletop_tool() -> None:
    """Test the perception.inspect_tabletop tool registration and execution."""
    engine = RobotEngine()
    register_builtin_tools(engine)

    tool_def = engine.tool_registry.get("perception.inspect_tabletop")
    assert tool_def is not None
    assert "tabletop" in tool_def.description.lower()

    # Execute tool handler
    res = await tool_def.handler(query="Inspect what is on the desk")
    assert "summary" in res
    assert res["device"] == "amd_igpu"
