"""Shared helpers for fire-and-forget asyncio work.

asyncio keeps only a weak reference to a running task, so the result of a bare
``asyncio.create_task(coro)`` can be garbage-collected mid-execution — the task
silently disappears and, because its exception is never retrieved, the failure is
invisible. Every detached task in this process goes through :func:`spawn`, which
retains the task until it finishes and logs whatever it raised.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from robot.telemetry import get_logger

logger = get_logger("concurrency")

#: Strong references to in-flight detached tasks, keyed by the task object.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def spawn(
    coro: Coroutine[Any, Any, Any],
    name: str = "background_task",
    on_error: Callable[[str, BaseException], None] | None = None,
) -> asyncio.Task:
    """Start a retained background task that reports its own failures.

    Args:
        coro: The coroutine to run detached.
        name: Task name, used in logs and for debugging.
        on_error: Optional ``(name, exception)`` callback; defaults to a warning
            on the module logger.

    Returns:
        The created task, already registered for retention until it completes.
    """
    task = asyncio.create_task(coro, name=name)
    _BACKGROUND_TASKS.add(task)

    def _done(finished: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is None:
            return
        if on_error is not None:
            on_error(name, exc)
        else:
            logger.warning("background_task_failed", task=name, error=str(exc))

    task.add_done_callback(_done)
    return task


def pending_count() -> int:
    """Return how many detached tasks are still in flight (diagnostics/tests)."""
    return len(_BACKGROUND_TASKS)
