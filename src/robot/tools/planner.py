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
import heapq
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools.hardening import (
    FAILURE_STATUS_VALUES,
    bound_payload,
    bound_text,
    classify_tool_result,
)

if TYPE_CHECKING:
    from robot.tools import PermissionManager, ToolDefinition, ToolRegistry

logger = get_logger("tools.planner")

# Hard planner bounds.  These protect model context, memory, and execution
# time even if a caller raises the configurable ceiling.
MAX_PLAN_STEPS: int = 20
MAX_PLAN_JSON_BYTES: int = 128 * 1024
MAX_STEP_ID_LENGTH: int = 64
MAX_GOAL_LENGTH: int = 2_000
MAX_STEP_OUTPUT_CHARS: int = 8_000
MAX_RETAINED_PLANS: int = 32


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
    # Step id to resume from after an AWAITING_CONFIRMATION pause.  A caller
    # may also pass an explicit cursor to ``execute_plan``.
    resume_cursor: str | None = None

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
            "resume_cursor": self.resume_cursor,
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
        # Never allow configuration to raise the safety ceiling above the
        # hard planner limit.
        self.max_steps_ceiling: int = max(1, min(int(max_steps_ceiling), MAX_PLAN_STEPS))
        self.default_total_timeout_s: float = default_total_timeout_s
        # Retained per-plan step state keyed by plan id.  This is what allows a
        # second execution after confirmation to skip work that already ran.
        self._retained_plans: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()

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
            ValueError: If step count/size exceeds bounds, step ids are
                duplicated/malformed, or dependencies are unknown, self-
                referential, or cyclic.
        """
        if not isinstance(raw_steps, list):
            raise ValueError("Plan steps must be provided as a list.")
        eff_max = max(
            1,
            min(
                max_steps or self.max_steps_ceiling,
                self.max_steps_ceiling,
                MAX_PLAN_STEPS,
            ),
        )
        if len(raw_steps) > eff_max:
            raise ValueError(
                f"Proposed plan has {len(raw_steps)} steps, which exceeds maximum step limit of {eff_max}."
            )

        # Prompt/plan size ceiling: reject before constructing step objects.
        try:
            plan_json = json.dumps(raw_steps, default=str, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("Plan steps are not JSON-serializable.") from exc
        if len(plan_json.encode("utf-8")) > MAX_PLAN_JSON_BYTES:
            raise ValueError(
                f"Plan JSON payload exceeds maximum size of {MAX_PLAN_JSON_BYTES} bytes."
            )

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps: list[PlanStep] = []
        step_ids: set[str] = set()

        for idx, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Plan step {idx} must be an object.")
            raw_id = item.get("step_id") or f"step_{idx}"
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError(f"Plan step {idx} has an invalid step_id.")
            s_id = raw_id.strip()
            if len(s_id) > MAX_STEP_ID_LENGTH:
                raise ValueError(
                    f"Plan step_id '{s_id[:16]}...' exceeds maximum length of {MAX_STEP_ID_LENGTH}."
                )
            if s_id in step_ids:
                raise ValueError(f"Duplicate plan step_id '{s_id}'.")
            step_ids.add(s_id)

            tool_name = item.get("tool_name", "")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"Plan step '{s_id}' is missing a valid tool_name.")
            tool_name = tool_name.strip()
            tool_args = item.get("arguments", {})
            if tool_args is None:
                tool_args = {}
            if not isinstance(tool_args, dict):
                raise ValueError(f"Plan step '{s_id}' arguments must be an object.")
            desc = str(item.get("description") or f"Execute {tool_name}")[:MAX_GOAL_LENGTH]
            deps = item.get("depends_on", [])
            if deps is None:
                deps = []
            if isinstance(deps, str):
                deps = [deps]
            if not isinstance(deps, (list, tuple)):
                raise ValueError(
                    f"Plan step '{s_id}' depends_on must be a list of step ids."
                )
            clean_deps: list[str] = []
            for dep in deps:
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError(
                        f"Plan step '{s_id}' has a non-string dependency {dep!r}."
                    )
                clean_dep = dep.strip()
                if len(clean_dep) > MAX_STEP_ID_LENGTH:
                    raise ValueError(
                        f"Plan step '{s_id}' dependency id exceeds {MAX_STEP_ID_LENGTH} characters."
                    )
                if clean_dep not in clean_deps:
                    clean_deps.append(clean_dep)

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
                depends_on=clean_deps,
                requires_confirmation=requires_confirm,
            )
            steps.append(step)

        # Validate dependency integrity before ordering.  Unknown references
        # are rejected rather than silently dropped (which used to make a step
        # look dependency-free and run prematurely).
        for step in steps:
            for dep in step.depends_on:
                if dep == step.step_id:
                    raise ValueError(
                        f"Plan step '{step.step_id}' cannot depend on itself."
                    )
                if dep not in step_ids:
                    raise ValueError(
                        f"Plan step '{step.step_id}' depends on unknown step '{dep}'."
                    )

        ordered_steps = self._topological_sort_steps(steps)

        return ExecutionPlan(
            plan_id=plan_id,
            goal=str(goal)[:MAX_GOAL_LENGTH],
            steps=ordered_steps,
            max_steps=eff_max,
            total_timeout_s=total_timeout_s or self.default_total_timeout_s,
        )

    @staticmethod
    def _topological_sort_steps(steps: list[PlanStep]) -> list[PlanStep]:
        """Order steps so every dependency precedes its dependants.

        Forward references are resolved by sorting rather than being skipped.
        Cycles and self-references raise a clear ``ValueError``.

        Args:
            steps: Validated steps with unique ids and known dependencies.

        Returns:
            Topologically ordered list (stable for independent steps).

        Raises:
            ValueError: If a dependency cycle is detected.
        """
        original_index = {step.step_id: idx for idx, step in enumerate(steps)}
        by_id = {step.step_id: step for step in steps}
        adjacency: dict[str, list[str]] = {step.step_id: [] for step in steps}
        indegree: dict[str, int] = {step.step_id: 0 for step in steps}
        for step in steps:
            for dep in step.depends_on:
                if dep not in by_id:
                    raise ValueError(
                        f"Plan step '{step.step_id}' depends on unknown step '{dep}'."
                    )
                adjacency[dep].append(step.step_id)
                indegree[step.step_id] += 1

        ready: list[tuple[int, str]] = [
            (original_index[step_id], step_id)
            for step_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        ordered: list[PlanStep] = []
        while ready:
            _, step_id = heapq.heappop(ready)
            ordered.append(by_id[step_id])
            for child in adjacency[step_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, (original_index[child], child))

        if len(ordered) != len(steps):
            cyclic = sorted(step_id for step_id, degree in indegree.items() if degree > 0)
            raise ValueError(
                "Plan contains a cyclic dependency involving: " + ", ".join(cyclic)
            )
        return ordered

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
            # Keep the retained snapshot in sync: a later execute_plan() call may
            # restore state from it, and a lost confirmation would re-pause a
            # step the operator already authorized.
            self._retain_plan_state(plan)
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

    # ─── Plan state retention ──────────────────────────────────────────────

    def _retain_plan_state(self, plan: ExecutionPlan) -> None:
        """Persist per-step state keyed by plan id (bounded LRU)."""
        state: dict[str, dict[str, Any]] = {}
        for step in plan.steps:
            state[step.step_id] = {
                "status": step.status,
                "result": step.result,
                "error": step.error,
                "duration_ms": step.duration_ms,
                "retries": step.retries,
                "confirmed": step.confirmed,
            }
        self._retained_plans[plan.plan_id] = state
        self._retained_plans.move_to_end(plan.plan_id)
        while len(self._retained_plans) > MAX_RETAINED_PLANS:
            self._retained_plans.popitem(last=False)

    def _restore_plan_state(self, plan: ExecutionPlan) -> None:
        """Restore already-settled steps for a plan object with the same id.

        Only terminal/settled states are restored; failed steps remain
        retryable and caller revisions are not overwritten.
        """
        retained = self._retained_plans.get(plan.plan_id)
        if not retained:
            return
        for step in plan.steps:
            prior = retained.get(step.step_id)
            if not prior:
                continue
            prior_status = prior.get("status")
            if prior_status in (
                PlanStepStatus.COMPLETED,
                PlanStepStatus.SKIPPED,
                PlanStepStatus.AWAITING_CONFIRMATION,
            ) and step.status in (PlanStepStatus.PENDING, PlanStepStatus.AWAITING_CONFIRMATION):
                step.status = prior_status
                step.result = prior.get("result")
                step.error = prior.get("error")
                step.duration_ms = float(prior.get("duration_ms") or 0.0)
                step.retries = int(prior.get("retries") or 0)
                step.confirmed = step.confirmed or bool(prior.get("confirmed", False))

    @staticmethod
    def _bound_step_result(value: Any) -> Any:
        """Cap and sanitize a step result before retaining/injecting it."""
        if isinstance(value, str):
            return bound_text(value, MAX_STEP_OUTPUT_CHARS)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        sanitized = bound_payload(value)
        try:
            serialized = json.dumps(sanitized, default=str, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError):
            return bound_text(repr(sanitized), MAX_STEP_OUTPUT_CHARS)
        if len(serialized) <= MAX_STEP_OUTPUT_CHARS:
            return sanitized
        return {
            "truncated": True,
            "original_chars": len(serialized),
            "preview": bound_text(serialized, MAX_STEP_OUTPUT_CHARS),
        }

    @staticmethod
    def _classify_step_outcome(
        outcome: Any,
    ) -> tuple[str, Any, str | None]:
        """Normalize a PermissionManager envelope into a step outcome.

        A handler-returned failure dictionary (or an envelope explicitly marked
        ``success: False``) is never classified as completed.

        Args:
            outcome: Raw PermissionManager return value.

        Returns:
            Tuple ``(kind, payload, error)`` where kind is one of
            ``completed``, ``failed``, ``denial``, ``timeout``, ``cancelled``.
        """
        if not isinstance(outcome, dict):
            return "completed", outcome, None

        raw_status = str(outcome.get("status", "success")).strip().lower()
        payload = outcome.get("result")
        payload_failure, payload_status, payload_error = classify_tool_result(payload)
        explicit_failure = outcome.get("success") is False
        status_failure = raw_status in FAILURE_STATUS_VALUES

        if not explicit_failure and not payload_failure and not status_failure:
            return "completed", payload, None

        effective = payload_status or raw_status or "failure"
        if effective in ("denial", "denied", "permission_denied"):
            kind = "denial"
        elif effective in ("timeout", "timed_out"):
            kind = "timeout"
        elif effective in ("cancelled", "canceled"):
            kind = "cancelled"
        else:
            kind = "failed"
        error = payload_error or outcome.get("error") or f"Tool reported status '{raw_status}'"
        return kind, payload, str(error)

    @staticmethod
    async def _cancel_and_await(task: asyncio.Task[Any] | None) -> None:
        """Cancel a task (if needed) and consume its result/exception."""
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def _run_step_with_bounds(
        self,
        tool_def: ToolDefinition,
        arguments: dict[str, Any],
        interrupt: asyncio.Event | None,
        budget_s: float,
    ) -> tuple[Any, str | None, str | None]:
        """Execute one step racing completion against interrupt and timeout.

        Args:
            tool_def: Tool to invoke through the permission manager.
            arguments: Hardened/raw step arguments.
            interrupt: Optional external cancellation event.
            budget_s: Maximum wall-clock seconds for this step (already capped
                by the remaining plan deadline).

        Returns:
            Tuple ``(outcome, failure_kind, error)``.  ``failure_kind`` is
            ``None`` when the tool returned; otherwise ``timeout`` or
            ``cancelled``.
        """
        if budget_s <= 0:
            return None, "timeout", "Step budget exhausted."

        exec_task = asyncio.create_task(
            self.permissions.execute(tool_def, arguments)
        )
        interrupt_task = (
            asyncio.create_task(interrupt.wait()) if interrupt is not None else None
        )
        waiters: set[asyncio.Task[Any]] = {exec_task}
        if interrupt_task is not None:
            waiters.add(interrupt_task)

        done: set[asyncio.Task[Any]] = set()
        timed_out = False
        exec_consumed = False
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=budget_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                timed_out = True
                return None, "timeout", f"Step timed out after {budget_s:.2f}s"

            if exec_task in done:
                try:
                    outcome = exec_task.result()
                    exec_consumed = True
                    return outcome, None, None
                except asyncio.CancelledError:
                    exec_consumed = True
                    return None, "cancelled", "Step execution was cancelled."
                except Exception as exc:  # noqa: BLE001 - normalized into step state
                    exec_consumed = True
                    return None, "failed", str(exc)

            # Interrupt won the race.
            return None, "cancelled", "Plan execution cancelled by user or system interrupt."
        finally:
            # Always reap both children; never leave orphan tasks behind.
            if not exec_consumed:
                await self._cancel_and_await(exec_task)
            if interrupt_task is not None:
                await self._cancel_and_await(interrupt_task)

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        interrupt: asyncio.Event | None = None,
        resume_cursor: str | None = None,
        resume_from: str | None = None,
    ) -> ExecutionPlan:
        """Execute all steps sequentially with dependencies, timeouts, and resume.

        Steps already recorded as COMPLETED or SKIPPED are not re-executed,
        even when ``execute_plan`` is called again after a confirmation pause.
        An explicit ``resume_cursor`` (step id) may be supplied; otherwise the
        plan's persisted cursor and retained state are used.

        Args:
            plan: Validated ExecutionPlan instance.
            interrupt: Optional external cancellation event.  An interrupt
                racing a running step cancels that step's task.
            resume_cursor: Optional step id to start/resume from.
            resume_from: Alias for ``resume_cursor``.

        Returns:
            Updated ExecutionPlan with terminal status and step results.

        Raises:
            ValueError: If the requested resume cursor does not exist.
            asyncio.CancelledError: If the caller cancels execution.
        """
        self._restore_plan_state(plan)
        plan.status = PlanStatus.IN_PROGRESS
        t_plan_start = time.perf_counter()

        all_step_ids = [step.step_id for step in plan.steps]
        cursor = resume_cursor or resume_from or plan.resume_cursor
        start_index = 0
        if cursor:
            if cursor not in all_step_ids:
                raise ValueError(f"Unknown resume cursor '{cursor}'.")
            start_index = all_step_ids.index(cursor)

        completed_steps: set[str] = {
            step.step_id
            for step in plan.steps
            if step.status == PlanStepStatus.COMPLETED
        }
        failed_steps: set[str] = {
            step.step_id
            for step in plan.steps
            if step.status
            in (
                PlanStepStatus.FAILED,
                PlanStepStatus.DENIAL,
                PlanStepStatus.TIMEOUT,
                PlanStepStatus.CANCELLED,
                PlanStepStatus.SKIPPED,
            )
        }

        logger.info(
            "plan_execution_started",
            plan_id=plan.plan_id,
            total_steps=len(plan.steps),
            goal=plan.goal[:60],
            resume_from=cursor,
        )

        try:
            for index, step in enumerate(plan.steps):
                if index < start_index:
                    continue
                # Resume semantics: never re-run settled work.
                if step.status == PlanStepStatus.COMPLETED:
                    completed_steps.add(step.step_id)
                    continue
                if step.status == PlanStepStatus.SKIPPED:
                    failed_steps.add(step.step_id)
                    continue

                elapsed_plan = time.perf_counter() - t_plan_start
                remaining_plan = float(plan.total_timeout_s) - elapsed_plan

                # 1. Total plan deadline
                if remaining_plan <= 0:
                    plan.status = PlanStatus.TIMEOUT
                    plan.error = (
                        f"Plan exceeded total deadline of {plan.total_timeout_s}s."
                    )
                    step.status = PlanStepStatus.SKIPPED
                    logger.warning(
                        "plan_deadline_exceeded",
                        plan_id=plan.plan_id,
                        elapsed=elapsed_plan,
                    )
                    break

                # 2. External cancellation (between steps)
                if interrupt and interrupt.is_set():
                    plan.status = PlanStatus.CANCELLED
                    plan.error = "Plan execution cancelled by user or system interrupt."
                    step.status = PlanStepStatus.CANCELLED
                    logger.info("plan_cancelled", plan_id=plan.plan_id)
                    break

                # 3. Dependency prerequisites
                unmet_deps = [d for d in step.depends_on if d not in completed_steps]
                if unmet_deps:
                    if any(d in failed_steps for d in step.depends_on):
                        step.status = PlanStepStatus.SKIPPED
                        step.error = f"Prerequisite step(s) {unmet_deps} failed."
                        failed_steps.add(step.step_id)
                        logger.warning(
                            "step_skipped_due_to_dependency_failure",
                            step_id=step.step_id,
                        )
                        continue
                    step.status = PlanStepStatus.SKIPPED
                    step.error = f"Dependencies not met: {unmet_deps}"
                    failed_steps.add(step.step_id)
                    continue

                # 4. Consequential confirmation gate
                if step.requires_confirmation and not step.confirmed:
                    step.status = PlanStepStatus.AWAITING_CONFIRMATION
                    plan.status = PlanStatus.AWAITING_CONFIRMATION
                    plan.resume_cursor = step.step_id
                    step.error = (
                        f"Action '{step.tool_name}' requires explicit confirmation."
                    )
                    logger.info(
                        "step_awaiting_confirmation",
                        step_id=step.step_id,
                        tool=step.tool_name,
                    )
                    self._retain_plan_state(plan)
                    return plan

                # 5. Execute one authorized step
                step.status = PlanStepStatus.RUNNING
                step.error = None
                t_step_start = time.perf_counter()

                tool_def = self.registry.get(step.tool_name)
                if not tool_def:
                    step.status = PlanStepStatus.FAILED
                    step.error = f"Tool '{step.tool_name}' not found in registry."
                    step.duration_ms = (time.perf_counter() - t_step_start) * 1000
                    failed_steps.add(step.step_id)
                    continue

                # Per-step timeout, also capped by the remaining plan budget.
                step_budget = min(float(plan.step_timeout_s), remaining_plan)
                outcome, forced_failure, forced_error = await self._run_step_with_bounds(
                    tool_def,
                    step.arguments,
                    interrupt,
                    step_budget,
                )
                step.duration_ms = (time.perf_counter() - t_step_start) * 1000

                if forced_failure:
                    step.error = forced_error
                    if forced_failure == "timeout":
                        step.status = PlanStepStatus.TIMEOUT
                        failed_steps.add(step.step_id)
                        logger.warning(
                            "step_timeout",
                            plan_id=plan.plan_id,
                            step_id=step.step_id,
                            timeout_s=step_budget,
                        )
                        # If the total plan deadline is exhausted, stop.
                        if time.perf_counter() - t_plan_start >= float(plan.total_timeout_s):
                            plan.status = PlanStatus.TIMEOUT
                            plan.error = (
                                f"Plan exceeded total deadline of {plan.total_timeout_s}s."
                            )
                            break
                        continue
                    if forced_failure == "cancelled":
                        step.status = PlanStepStatus.CANCELLED
                        failed_steps.add(step.step_id)
                        plan.status = PlanStatus.CANCELLED
                        plan.error = forced_error or "Plan execution cancelled."
                        logger.info("plan_cancelled", plan_id=plan.plan_id)
                        break
                    step.status = PlanStepStatus.FAILED
                    failed_steps.add(step.step_id)
                    continue

                if not isinstance(outcome, dict):
                    step.status = PlanStepStatus.COMPLETED
                    step.result = self._bound_step_result(outcome)
                    completed_steps.add(step.step_id)
                    continue

                step.retries = int(outcome.get("retries", 0) or 0)
                kind, payload, error = self._classify_step_outcome(outcome)
                if kind == "completed":
                    step.status = PlanStepStatus.COMPLETED
                    step.result = self._bound_step_result(payload)
                    completed_steps.add(step.step_id)
                elif kind == "denial":
                    step.status = PlanStepStatus.DENIAL
                    step.result = self._bound_step_result(payload)
                    step.error = error or "Permission denied"
                    failed_steps.add(step.step_id)
                elif kind == "timeout":
                    step.status = PlanStepStatus.TIMEOUT
                    step.result = self._bound_step_result(payload)
                    step.error = error or "Execution timed out"
                    failed_steps.add(step.step_id)
                elif kind == "cancelled":
                    step.status = PlanStepStatus.CANCELLED
                    step.result = self._bound_step_result(payload)
                    step.error = error or "Execution cancelled"
                    failed_steps.add(step.step_id)
                else:
                    step.status = PlanStepStatus.FAILED
                    step.result = self._bound_step_result(payload)
                    step.error = error or "Execution failed"
                    failed_steps.add(step.step_id)

                self._retain_plan_state(plan)

        except asyncio.CancelledError:
            plan.status = PlanStatus.CANCELLED
            plan.error = "Plan execution cancelled."
            self._retain_plan_state(plan)
            raise

        # Evaluate final plan status
        plan.completed_at = time.time()
        if plan.status == PlanStatus.IN_PROGRESS:
            if all(s.status == PlanStepStatus.COMPLETED for s in plan.steps):
                plan.status = PlanStatus.COMPLETED
                plan.resume_cursor = None
            elif any(
                s.status
                in (
                    PlanStepStatus.FAILED,
                    PlanStepStatus.DENIAL,
                    PlanStepStatus.TIMEOUT,
                )
                for s in plan.steps
            ):
                plan.status = PlanStatus.FAILED
            elif any(s.status == PlanStepStatus.CANCELLED for s in plan.steps):
                plan.status = PlanStatus.CANCELLED
            else:
                # Only completed/skipped steps remain.
                plan.status = PlanStatus.COMPLETED

        self._retain_plan_state(plan)
        logger.info(
            "plan_execution_finished",
            plan_id=plan.plan_id,
            status=plan.status.value,
            duration_s=round(time.perf_counter() - t_plan_start, 2),
        )
        return plan
