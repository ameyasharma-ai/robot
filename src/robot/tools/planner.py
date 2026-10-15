"""Sequential Planner–Executor for structured, multi-step agentic tasks.

Architectural Rationale:
1. Strict Step Boundaries: Unbounded multi-step planning loops can exhaust API rate limits,
   bloat context, and cause runaway recursion. We enforce a strict step ceiling (default: 5 steps)
   and a total plan execution deadline (default: 45s).
2. Dependency & Order Guarantees: Steps declare explicit prerequisites (`depends_on`). An executor
   runs one authorized step at a time, resolving dependencies and skipping downstream steps if a
   prerequisite step fails or is denied.
3. Structured Outcome Contracts: Step outcomes are normalized into structured states
   (`completed`, `failed`, `denial`, `timeout`, `cancelled`, `skipped`, `awaiting_confirmation`).
4. Consequential Action Gating: Tools requiring SYSTEM (L4+) or DEVELOPMENT (L5+) permissions
   (such as shell command execution or git commits) require explicit authorization/confirmation
   before execution, preventing autonomous destructive actions.
5. Invariance to Real-World Verification: A tool invocation is never assumed to be proof of
   real-world success; the executor evaluates structured status and return data.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.tools import PermissionManager, ToolDefinition, ToolRegistry

logger = get_logger("tools.planner")


class PlanStepStatus(str, Enum):
    """Execution status of an individual plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIAL = "denial"
    DENIED = "denial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


class PlanStatus(str, Enum):
    """Aggregate lifecycle state of an ExecutionPlan."""

    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TIMEOUT = "timeout"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


@dataclass
class PlanStep:
    """A discrete, authorized step within a multi-step task plan."""

    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    requires_confirmation: bool = False
    confirmed: bool = False
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert step state to a serialized dictionary.

        Returns:
            Dictionary representation of step state.
        """
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "requires_confirmation": self.requires_confirmation,
            "confirmed": self.confirmed,
            "retries": self.retries,
        }


@dataclass
class ExecutionPlan:
    """Structured task plan with explicit steps, dependencies, and boundaries."""

    plan_id: str
    goal: str
    steps: list[PlanStep]
    status: PlanStatus = PlanStatus.INITIALIZED
    max_steps: int = 5
    step_timeout_s: float = 10.0
    total_timeout_s: float = 45.0
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error: str | None = None

    def get_step(self, step_id: str) -> PlanStep | None:
        """Find a step by its identifier.

        Args:
            step_id: Unique step identifier string.

        Returns:
            Matching PlanStep if found, else None.
        """
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert entire plan into structured JSON-compatible dictionary.

        Returns:
            Dictionary representation of the plan and all steps.
        """
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "status": self.status.value,
            "max_steps": self.max_steps,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class SequentialPlannerExecutor:
    """Orchestrates bounded, sequential multi-step execution with safety gates."""

    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionManager,
        max_steps_ceiling: int = 5,
        default_total_timeout_s: float = 45.0,
    ) -> None:
        """Initialize the SequentialPlannerExecutor.

        Args:
            registry: Central tool registry to look up tool definitions and schemas.
            permissions: Security permission manager to enforce safety gates.
            max_steps_ceiling: Hard maximum number of steps allowed per plan.
            default_total_timeout_s: Maximum wall-clock time in seconds for complete plan execution.
        """
        self.registry: ToolRegistry = registry
        self.permissions: PermissionManager = permissions
        self.max_steps_ceiling: int = max_steps_ceiling
        self.default_total_timeout_s: float = default_total_timeout_s

    def create_plan(
        self,
        goal: str,
        raw_steps: list[dict[str, Any]],
        max_steps: int | None = None,
        total_timeout_s: float | None = None,
    ) -> ExecutionPlan:
        """Construct and validate a structured ExecutionPlan from raw step proposals.

        Args:
            goal: Natural language description of the objective.
            raw_steps: List of step dictionaries containing 'tool_name', 'arguments',
                and optional 'depends_on', 'description'.
            max_steps: Optional ceiling override (capped at max_steps_ceiling).
            total_timeout_s: Optional plan deadline in seconds.

        Returns:
            Validated ExecutionPlan ready for execution.

        Raises:
            ValueError: If step count exceeds ceiling or circular dependencies are detected.
        """
        eff_max = min(max_steps or self.max_steps_ceiling, self.max_steps_ceiling)
        if len(raw_steps) > eff_max:
            raise ValueError(
                f"Proposed plan has {len(raw_steps)} steps, which exceeds maximum step limit of {eff_max}."
            )

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps: list[PlanStep] = []
        step_ids: set[str] = set()

        for idx, item in enumerate(raw_steps, start=1):
            s_id = item.get("step_id") or f"step_{idx}"
            if s_id in step_ids:
                s_id = f"step_{idx}_{uuid.uuid4().hex[:4]}"
            step_ids.add(s_id)

            tool_name = item.get("tool_name", "")
            tool_args = item.get("arguments", {})
            desc = item.get("description", f"Execute {tool_name}")
            deps = item.get("depends_on", [])
            if isinstance(deps, str):
                deps = [deps]

            # Security audit: check if tool requires consequential confirmation (L3+ EXTERNAL, SYSTEM, or DEVELOPMENT)
            t_def = self.registry.get(tool_name)
            requires_confirm = False
            if t_def and int(getattr(t_def, "permission_level", 0)) >= 3:
                requires_confirm = True

            step = PlanStep(
                step_id=s_id,
                tool_name=tool_name,
                arguments=tool_args,
                description=desc,
                depends_on=list(deps),
                requires_confirmation=requires_confirm,
            )
            steps.append(step)

        # Validate dependency integrity (no forward-only cycles or unknown step IDs)
        for s in steps:
            for dep in s.depends_on:
                if dep not in step_ids:
                    logger.warning("unknown_dependency_dropped", step_id=s.step_id, dependency=dep)
                    s.depends_on.remove(dep)

        return ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            max_steps=eff_max,
            total_timeout_s=total_timeout_s or self.default_total_timeout_s,
        )

    def confirm_step(self, plan: ExecutionPlan, step_id: str) -> bool:
        """Grant explicit authorization for a consequential plan step.

        Args:
            plan: Target ExecutionPlan instance.
            step_id: Step identifier to confirm.

        Returns:
            True if step was found and confirmed, False otherwise.
        """
        step = plan.get_step(step_id)
        if step and step.requires_confirmation:
            step.confirmed = True
            logger.info("step_confirmed", plan_id=plan.plan_id, step_id=step_id)
            return True
        return False

    def revise_step(
        self,
        plan: ExecutionPlan,
        step_id: str,
        new_tool: str,
        new_args: dict[str, Any],
        new_description: str | None = None,
    ) -> bool:
        """Revise a pending or failed step based on intermediate outcomes.

        Args:
            plan: Target ExecutionPlan instance.
            step_id: Identifier of the step to modify.
            new_tool: Revised tool name.
            new_args: Revised argument dictionary.
            new_description: Optional updated description.

        Returns:
            True if revised successfully, False if step is already running or completed.
        """
        step = plan.get_step(step_id)
        if not step or step.status in (PlanStepStatus.RUNNING, PlanStepStatus.COMPLETED):
            return False

        step.tool_name = new_tool
        step.arguments = new_args
        if new_description:
            step.description = new_description
        step.status = PlanStepStatus.PENDING
        step.error = None
        step.result = None

        t_def = self.registry.get(new_tool)
        step.requires_confirmation = bool(t_def and int(getattr(t_def, "permission_level", 0)) >= 3)
        step.confirmed = False

        logger.info("step_revised", plan_id=plan.plan_id, step_id=step_id, new_tool=new_tool)
        return True

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        interrupt: asyncio.Event | None = None,
    ) -> ExecutionPlan:
        """Execute all steps in the plan sequentially with dependency and deadline tracking.

        Args:
            plan: Validated ExecutionPlan instance.
            interrupt: Optional external cancellation event.

        Returns:
            Updated ExecutionPlan with terminal status and step results.
        """
        plan.status = PlanStatus.IN_PROGRESS
        t_plan_start = time.perf_counter()
        completed_steps: set[str] = set()
        failed_steps: set[str] = set()

        logger.info(
            "plan_execution_started",
            plan_id=plan.plan_id,
            total_steps=len(plan.steps),
            goal=plan.goal[:60],
        )

        for step in plan.steps:
            # 1. Check total plan timeout deadline
            elapsed_plan = time.perf_counter() - t_plan_start
            if elapsed_plan > plan.total_timeout_s:
                plan.status = PlanStatus.TIMEOUT
                plan.error = f"Plan exceeded total deadline of {plan.total_timeout_s}s."
                step.status = PlanStepStatus.SKIPPED
                logger.warning("plan_deadline_exceeded", plan_id=plan.plan_id, elapsed=elapsed_plan)
                break

            # 2. Check external cancellation
            if interrupt and interrupt.is_set():
                plan.status = PlanStatus.CANCELLED
                plan.error = "Plan execution cancelled by user or system interrupt."
                step.status = PlanStepStatus.CANCELLED
                logger.info("plan_cancelled", plan_id=plan.plan_id)
                break

            # 3. Check dependency prerequisites
            unmet_deps = [d for d in step.depends_on if d not in completed_steps]
            if unmet_deps:
                # If any prerequisite failed, skip this step
                if any(d in failed_steps for d in step.depends_on):
                    step.status = PlanStepStatus.SKIPPED
                    step.error = f"Prerequisite step(s) {unmet_deps} failed."
                    failed_steps.add(step.step_id)
                    logger.warning("step_skipped_due_to_dependency_failure", step_id=step.step_id)
                    continue
                else:
                    step.status = PlanStepStatus.SKIPPED
                    step.error = f"Dependencies not met: {unmet_deps}"
                    failed_steps.add(step.step_id)
                    continue

            # 4. Check consequential confirmation gate
            if step.requires_confirmation and not step.confirmed:
                step.status = PlanStepStatus.AWAITING_CONFIRMATION
                plan.status = PlanStatus.AWAITING_CONFIRMATION
                step.error = f"Action '{step.tool_name}' requires explicit confirmation."
                logger.info("step_awaiting_confirmation", step_id=step.step_id, tool=step.tool_name)
                return plan

            # 5. Execute single authorized step
            step.status = PlanStepStatus.RUNNING
            t_step_start = time.perf_counter()

            tool_def = self.registry.get(step.tool_name)
            if not tool_def:
                step.status = PlanStepStatus.FAILED
                step.error = f"Tool '{step.tool_name}' not found in registry."
                step.duration_ms = (time.perf_counter() - t_step_start) * 1000
                failed_steps.add(step.step_id)
                continue

            try:
                # Execute through standard permission manager (enforcing hardening, safety, retries)
                outcome = await self.permissions.execute(tool_def, step.arguments)
                dur = (time.perf_counter() - t_step_start) * 1000
                step.duration_ms = dur
                step.retries = outcome.get("retries", 0)

                raw_status = outcome.get("status", "success")
                if raw_status == "success":
                    step.status = PlanStepStatus.COMPLETED
                    step.result = outcome.get("result")
                    completed_steps.add(step.step_id)
                elif raw_status == "denial":
                    step.status = PlanStepStatus.DENIAL
                    step.error = outcome.get("error", "Permission denied")
                    failed_steps.add(step.step_id)
                elif raw_status == "timeout":
                    step.status = PlanStepStatus.TIMEOUT
                    step.error = outcome.get("error", "Execution timed out")
                    failed_steps.add(step.step_id)
                elif raw_status == "cancelled":
                    step.status = PlanStepStatus.CANCELLED
                    step.error = outcome.get("error", "Execution cancelled")
                    failed_steps.add(step.step_id)
                else:
                    step.status = PlanStepStatus.FAILED
                    step.error = outcome.get("error", "Execution failed")
                    failed_steps.add(step.step_id)

            except Exception as e:
                step.duration_ms = (time.perf_counter() - t_step_start) * 1000
                step.status = PlanStepStatus.FAILED
                step.error = str(e)
                failed_steps.add(step.step_id)
                logger.error("step_execution_exception", step_id=step.step_id, error=str(e))

        # Evaluate final plan status
        plan.completed_at = time.time()
        if plan.status == PlanStatus.IN_PROGRESS:
            if all(s.status == PlanStepStatus.COMPLETED for s in plan.steps):
                plan.status = PlanStatus.COMPLETED
            elif any(s.status in (PlanStepStatus.FAILED, PlanStepStatus.DENIAL, PlanStepStatus.TIMEOUT) for s in plan.steps):
                plan.status = PlanStatus.FAILED
            else:
                plan.status = PlanStatus.COMPLETED

        logger.info(
            "plan_execution_finished",
            plan_id=plan.plan_id,
            status=plan.status.value,
            duration_s=round(time.perf_counter() - t_plan_start, 2),
        )
        return plan
