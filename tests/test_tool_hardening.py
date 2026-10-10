"""Comprehensive unit and regression tests for tool input hardening and parameter normalization."""

import pytest
import asyncio
from robot.tools import (
    PermissionLevel,
    PermissionManager,
    ToolDefinition,
    ToolRegistry,
)
from robot.tools.hardening import (
    ParameterConstraint,
    ToolOutcomeStatus,
    ToolValidationError,
    coerce_value,
    harden_tool_arguments,
    harden_tool_inputs,
)


def test_coerce_value_strict_types():
    """Verify strict schema-driven type coercion across fundamental types."""
    # Integers
    assert coerce_value("42", int, "count") == 42
    assert coerce_value(" 75% ", int, "pct") == 75
    assert coerce_value("100px", int, "size") == 100
    assert coerce_value(3.0, int, "num") == 3
    with pytest.raises(ToolValidationError):
        coerce_value("not_a_number", int, "count")
    with pytest.raises(ToolValidationError):
        coerce_value(True, int, "flag")  # Booleans must not silently become 1

    # Floats
    assert coerce_value("0.85", float, "volume") == 0.85
    assert coerce_value("50%", float, "vol") == 50.0
    with pytest.raises(ToolValidationError):
        coerce_value(False, float, "flag")

    # Booleans
    assert coerce_value("true", bool, "enabled") is True
    assert coerce_value("YES", bool, "enabled") is True
    assert coerce_value("0", bool, "enabled") is False
    assert coerce_value("off", bool, "enabled") is False
    assert coerce_value(1, bool, "enabled") is True

    # Lists
    assert coerce_value("apple, banana, cherry", list, "items") == ["apple", "banana", "cherry"]
    assert coerce_value("[\"one\", \"two\"]", list, "items") == ["one", "two"]

    # Dicts
    assert coerce_value("{\"key\": \"val\"}", dict, "payload") == {"key": "val"}


def test_alias_resolution_and_conflict_detection():
    """Verify approved aliases resolve to canonical names, while conflicting values error."""
    # 1. Alias resolves to canonical
    raw = {"track": "Lucid Dreams", "artist": "Juice WRLD"}
    res = harden_tool_arguments(raw)
    assert res["query"] == "Lucid Dreams"

    # 2. Canonical and alias identical value -> ok
    raw_dup = {"query": "Lucid Dreams", "song": "Lucid Dreams"}
    res_dup = harden_tool_arguments(raw_dup)
    assert res_dup["query"] == "Lucid Dreams"

    # 3. Canonical and alias conflicting values -> raise ToolValidationError
    raw_conflict = {"query": "Lucid Dreams", "track": "All Girls Are The Same"}
    with pytest.raises(ToolValidationError) as exc_info:
        harden_tool_arguments(raw_conflict)
    assert exc_info.value.code == "conflicting_arguments"
    assert "conflicts with alias" in str(exc_info.value)


def test_parameter_constraints_ranges_and_enums():
    """Verify range limits (min/max), allowed choices, and required arguments."""
    constraints = {
        "level_percent": ParameterConstraint(target_type=int, min_val=0, max_val=100, required=True),
        "action": ParameterConstraint(target_type=str, allowed_choices=["get", "set"]),
    }

    # Valid
    valid = harden_tool_arguments({"level": "60%", "action": "set"}, constraints=constraints)
    assert valid["level_percent"] == 60
    assert valid["action"] == "set"

    # Below minimum
    with pytest.raises(ToolValidationError) as exc_min:
        harden_tool_arguments({"level": -5, "action": "set"}, constraints=constraints)
    assert exc_min.value.code == "value_below_minimum"

    # Above maximum
    with pytest.raises(ToolValidationError) as exc_max:
        harden_tool_arguments({"level": 150, "action": "set"}, constraints=constraints)
    assert exc_max.value.code == "value_above_maximum"

    # Invalid enum choice
    with pytest.raises(ToolValidationError) as exc_choice:
        harden_tool_arguments({"level": 50, "action": "destroy"}, constraints=constraints)
    assert exc_choice.value.code == "invalid_choice"

    # Missing required parameter
    with pytest.raises(ToolValidationError) as exc_req:
        harden_tool_arguments({"action": "set"}, constraints=constraints)
    assert exc_req.value.code == "missing_required_parameter"


@pytest.mark.asyncio
async def test_harden_tool_inputs_decorator_async_and_sync():
    """Verify @harden_tool_inputs decorator handles both async and sync tools gracefully."""

    @harden_tool_inputs(
        constraints={
            "volume": ParameterConstraint(target_type=float, min_val=0.0, max_val=1.0, default=0.8),
        }
    )
    async def async_sample_tool(volume: float = 0.8) -> dict:
        return {"status": "ok", "volume": volume}

    @harden_tool_inputs(
        constraints={
            "volume": ParameterConstraint(target_type=float, min_val=0.0, max_val=1.0, default=0.8),
        }
    )
    def sync_sample_tool(volume: float = 0.8) -> dict:
        return {"status": "ok", "volume": volume}

    # Async invocation with alias 'vol' and string '0.5'
    res_async = await async_sample_tool(vol="0.5")
    assert res_async["status"] == "ok"
    assert res_async["volume"] == 0.5

    # Sync invocation with default
    res_sync = sync_sample_tool()
    assert res_sync["status"] == "ok"
    assert res_sync["volume"] == 0.8

    # Validation error returned cleanly as structured dict without unhandled exception
    res_err = await async_sample_tool(volume=2.5)
    assert res_err["status"] == ToolOutcomeStatus.VALIDATION_ERROR.value
    assert res_err["code"] == "value_above_maximum"


@pytest.mark.asyncio
async def test_permission_manager_integration_with_hardening():
    """Verify PermissionManager enforces hardening and structured error outcomes."""
    pm = PermissionManager()

    async def sample_handler(duration_seconds: int, message: str) -> dict:
        return {"scheduled": True, "duration": duration_seconds, "msg": message}

    tool = ToolDefinition(
        name="test.timer",
        description="Schedule a test timer",
        parameters={
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                "message": {"type": "string"},
            },
            "required": ["duration_seconds", "message"],
        },
        handler=sample_handler,
        permission_level=PermissionLevel.READ_ONLY,
    )

    # 1. Clean execution with aliases and coercion ('duration': '60s', 'note': 'Take a break')
    out1 = await pm.execute(tool, {"duration": "60s", "note": "Take a break"})
    assert "result" in out1
    assert out1["result"]["duration"] == 60
    assert out1["result"]["msg"] == "Take a break"

    # 2. Conflicting alias arguments
    out2 = await pm.execute(tool, {"duration_seconds": 60, "duration": 120, "message": "Test"})
    assert "error" in out2
    assert "conflicting_arguments" in str(out2.get("code", out2.get("error", "")))
