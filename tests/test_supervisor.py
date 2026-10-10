"""Tests for ProcessSupervisor and health watchdog."""

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
