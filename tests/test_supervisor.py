"""Tests for ProcessSupervisor and health watchdog."""

import asyncio
import os
import pytest
from robot.services.supervisor import ProcessSupervisor


def test_supervisor_lifecycle_and_hooks():
    supervisor = ProcessSupervisor(health_check_interval_s=0.5)

    hook_called = False

    def on_exit():
        nonlocal hook_called
        hook_called = True

    supervisor.on_shutdown(on_exit)
    import subprocess
    proc = subprocess.Popen(["sleep", "5"])
    supervisor.register_process("dummy_worker", proc.pid)

    supervisor.start()
    assert supervisor._running is True

    supervisor._check_memory_usage()
    supervisor._check_process_liveness()

    supervisor.stop()
    assert supervisor._running is False
    assert hook_called is True


def test_gc_is_rate_limited_between_watchdog_ticks():
    """Full GC is stop-the-world; it must not run on every 15s watchdog tick."""
    import gc
    from unittest.mock import patch

    supervisor = ProcessSupervisor(max_cpu_rss_mb=0.0)
    with patch.object(gc, "collect") as collect:
        supervisor._check_memory_usage()
        supervisor._check_memory_usage()
        assert collect.call_count == 1


@pytest.mark.asyncio
async def test_health_monitor_survives_a_failing_cycle():
    """One failed health sweep must not terminate the monitoring loop."""
    from robot.services import HealthMonitor

    monitor = HealthMonitor(state=object(), check_interval_s=0.001)
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return {}

    monitor.check_all = flaky  # type: ignore[method-assign]
    await monitor.start()
    await asyncio.sleep(0.05)
    await monitor.stop()
    assert calls["n"] >= 2
