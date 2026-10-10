"""Unit tests for SequentialPlannerExecutor (Phase 5D)."""

import asyncio
import pytest

from robot.tools import (
    ExecutionPlan,
    PermissionLevel,
    PermissionManager,
    PlanStatus,
    PlanStepStatus,
    SequentialPlannerExecutor,
    ToolDefinition,
    ToolRegistry,
)


@pytest.fixture
def planner_setup():
    registry = ToolRegistry()
    permissions = PermissionManager(max_auto_level=PermissionLevel.HARMLESS_LOCAL)

    async def get_weather(location="Pune", **kwargs):
        return {"temp": 24, "condition": "Sunny", "location": location}

    async def calculate(expression="", **kwargs):
        return {"result": 42}

    async def shell_exec(command="", **kwargs):
        return {"stdout": "mocked shell output", "exit_code": 0}

    registry.register(
        ToolDefinition(
            name="weather.forecast",
            description="Weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            handler=get_weather,
            permission_level=PermissionLevel.READ_ONLY,
            is_idempotent=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="system.calculate",
            description="Calculate",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
            handler=calculate,
            permission_level=PermissionLevel.READ_ONLY,
            is_idempotent=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="terminal.execute",
            description="Run command",
            parameters={},
            handler=shell_exec,
            permission_level=PermissionLevel.SYSTEM,
            is_idempotent=False,
        )
    )

    executor = SequentialPlannerExecutor(registry=registry, permissions=permissions, max_steps_ceiling=5)
    return executor, registry, permissions


def test_create_plan_valid(planner_setup):
    executor, _, _ = planner_setup
    raw_steps = [
        {"step_id": "step_1", "tool_name": "weather.forecast", "arguments": {"location": "Tokyo"}},
        {"step_id": "step_2", "tool_name": "system.calculate", "arguments": {"expression": "2+2"}, "depends_on": ["step_1"]},
    ]
    plan = executor.create_plan(goal="Check Tokyo weather and calculate", raw_steps=raw_steps)
    assert plan.plan_id.startswith("plan_")
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_name == "weather.forecast"
    assert plan.steps[1].depends_on == ["step_1"]
    assert plan.steps[1].requires_confirmation is False


def test_create_plan_step_limit_enforced(planner_setup):
    executor, _, _ = planner_setup
    # Propose 6 steps when ceiling is 5
    raw_steps = [{"tool_name": "system.calculate", "arguments": {}} for _ in range(6)]
    with pytest.raises(ValueError, match="exceeds maximum step limit"):
        executor.create_plan(goal="Too many steps", raw_steps=raw_steps)


def test_consequential_action_gating(planner_setup):
    executor, _, _ = planner_setup
    raw_steps = [
        {"step_id": "step_cmd", "tool_name": "terminal.execute", "arguments": {"command": "ls -la"}},
    ]
    plan = executor.create_plan(goal="Run terminal command", raw_steps=raw_steps)
    assert plan.steps[0].requires_confirmation is True
    assert plan.steps[0].confirmed is False


@pytest.mark.asyncio
async def test_execute_plan_success(planner_setup):
    executor, _, _ = planner_setup
    raw_steps = [
        {"step_id": "step_1", "tool_name": "weather.forecast", "arguments": {"location": "London"}},
        {"step_id": "step_2", "tool_name": "system.calculate", "arguments": {"expression": "10*10"}, "depends_on": ["step_1"]},
    ]
    plan = executor.create_plan(goal="Two step plan", raw_steps=raw_steps)
    executed = await executor.execute_plan(plan)

    assert executed.status == PlanStatus.COMPLETED
    assert executed.steps[0].status == PlanStepStatus.COMPLETED
    assert executed.steps[0].result == {"temp": 24, "condition": "Sunny", "location": "London"}
    assert executed.steps[1].status == PlanStepStatus.COMPLETED
    assert executed.steps[1].result == {"result": 42}


@pytest.mark.asyncio
async def test_execute_plan_dependency_failure_skips(planner_setup):
    executor, registry, _ = planner_setup

    async def failing_tool(**kwargs):
        raise RuntimeError("Service unavailable")

    registry.register(
        ToolDefinition(
            name="broken.tool",
            description="Broken",
            parameters={},
            handler=failing_tool,
            permission_level=PermissionLevel.READ_ONLY,
        )
    )

    raw_steps = [
        {"step_id": "step_1", "tool_name": "broken.tool", "arguments": {}},
        {"step_id": "step_2", "tool_name": "weather.forecast", "arguments": {}, "depends_on": ["step_1"]},
    ]
    plan = executor.create_plan(goal="Failing prerequisite", raw_steps=raw_steps)
    executed = await executor.execute_plan(plan)

    assert executed.status == PlanStatus.FAILED
    assert executed.steps[0].status == PlanStepStatus.FAILED
    assert executed.steps[1].status == PlanStepStatus.SKIPPED
    assert "failed" in executed.steps[1].error.lower()


@pytest.mark.asyncio
async def test_execute_plan_requires_confirmation_pauses(planner_setup):
    executor, _, _ = planner_setup
    raw_steps = [
        {"step_id": "step_1", "tool_name": "terminal.execute", "arguments": {"command": "echo hi"}},
    ]
    plan = executor.create_plan(goal="Privileged task", raw_steps=raw_steps)

    # Initial execution should pause and require confirmation
    executed = await executor.execute_plan(plan)
    assert executed.status == PlanStatus.AWAITING_CONFIRMATION
    assert executed.steps[0].status == PlanStepStatus.AWAITING_CONFIRMATION

    # Grant confirmation
    confirmed = executor.confirm_step(plan, "step_1")
    assert confirmed is True
    assert plan.steps[0].confirmed is True

    # Note: terminal.execute has permission_level SYSTEM (L4), and max_auto_level is HARMLESS_LOCAL (L1).
    # Since permissions manager check will deny without manual escalation, let's verify denial contract:
    executed_after = await executor.execute_plan(plan)
    assert executed_after.steps[0].status == PlanStepStatus.DENIAL


@pytest.mark.asyncio
async def test_execute_plan_cancellation(planner_setup):
    executor, _, _ = planner_setup
    interrupt = asyncio.Event()
    interrupt.set()  # Cancel immediately

    raw_steps = [
        {"step_id": "step_1", "tool_name": "weather.forecast", "arguments": {}},
    ]
    plan = executor.create_plan(goal="Cancelled plan", raw_steps=raw_steps)
    executed = await executor.execute_plan(plan, interrupt=interrupt)

    assert executed.status == PlanStatus.CANCELLED
    assert executed.steps[0].status == PlanStepStatus.CANCELLED


def test_revise_step(planner_setup):
    executor, _, _ = planner_setup
    raw_steps = [
        {"step_id": "step_1", "tool_name": "weather.forecast", "arguments": {"location": "Paris"}},
    ]
    plan = executor.create_plan(goal="Revise plan", raw_steps=raw_steps)
    success = executor.revise_step(plan, "step_1", new_tool="system.calculate", new_args={"expression": "1+1"})
    assert success is True
    assert plan.steps[0].tool_name == "system.calculate"
    assert plan.steps[0].arguments == {"expression": "1+1"}
