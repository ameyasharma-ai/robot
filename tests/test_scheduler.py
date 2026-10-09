"""Unit tests for TemporalScheduler, time parsing, and temporal builtin tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot.core.events import EventBus
from robot.services.scheduler import ScheduledJob, TemporalScheduler, parse_time_expression
from robot.tools import ToolRegistry
from robot.tools.builtins.scheduler import register_scheduler_tools


def test_parse_time_expression_relative() -> None:
    """Test relative duration string parsing."""
    # Pure seconds
    ts, iso, sec = parse_time_expression("30")
    assert round(sec) == 30

    # Human relative
    _, _, sec_5m = parse_time_expression("5m")
    assert round(sec_5m) == 300

    _, _, sec_15mins = parse_time_expression("15 minutes")
    assert round(sec_15mins) == 900

    _, _, sec_2h = parse_time_expression("in 2 hours")
    assert round(sec_2h) == 7200

    _, _, sec_combo = parse_time_expression("1 hour 30 mins")
    assert round(sec_combo) == 5400


def test_parse_time_expression_clock() -> None:
    """Test absolute clock time parsing."""
    ts, iso, sec = parse_time_expression("7:30 AM")
    assert ts > time.time()
    assert sec > 0

    ts_pm, _, _ = parse_time_expression("at 4:15 PM")
    assert ts_pm > time.time()


def test_parse_time_expression_invalid() -> None:
    """Test invalid string raises ValueError."""
    with pytest.raises(ValueError):
        parse_time_expression("invalid gibberish completely")


@pytest.mark.asyncio
async def test_scheduler_lifecycle_and_db() -> None:
    """Test TemporalScheduler setup, database operations, and job lifecycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_scheduler.db"
        mock_engine = MagicMock()
        mock_engine.events = EventBus()

        scheduler = TemporalScheduler(engine=mock_engine, db_path=str(db_path))
        await scheduler.setup()
        assert await scheduler.health_check() is True

        # 1. Add timer
        timer_job = await scheduler.add_timer(60.0, label="Test Timer")
        assert timer_job.job_type == "timer"
        assert timer_job.label == "Test Timer"
        assert timer_job.status == "pending"

        # 2. Add alarm
        alarm_job = await scheduler.add_alarm("8:00 AM", label="Morning Alarm", recurring=True)
        assert alarm_job.job_type == "alarm"
        assert alarm_job.recurring is True

        # 3. Add reminder
        reminder_job = await scheduler.add_reminder("in 10 minutes", message="Buy coffee")
        assert reminder_job.job_type == "reminder"
        assert reminder_job.label == "Buy coffee"

        # 4. List jobs
        pending = await scheduler.list_jobs(status="pending")
        assert len(pending) == 3

        timers_only = await scheduler.list_jobs(job_type="timer")
        assert len(timers_only) == 1
        assert timers_only[0].id == timer_job.id

        # 5. Get job
        retrieved = await scheduler.get_job(timer_job.id)
        assert retrieved is not None
        assert retrieved.label == "Test Timer"

        # 6. Cancel job
        cancelled = await scheduler.cancel_job("Test Timer")
        assert cancelled is True
        cancelled_job = await scheduler.get_job(timer_job.id)
        assert cancelled_job is not None
        assert cancelled_job.status == "cancelled"

        # 7. Daily briefing
        briefing = await scheduler.get_daily_briefing()
        assert "current_time" in briefing
        assert "system_status" in briefing
        assert briefing["upcoming_alarms_count"] == 1
        assert briefing["pending_reminders_count"] == 1


@pytest.mark.asyncio
async def test_scheduler_background_trigger() -> None:
    """Test background loop triggers due jobs and emits events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_trigger.db"
        mock_engine = MagicMock()
        event_bus = EventBus()
        mock_engine.events = event_bus

        triggered_events = []

        async def on_triggered(**kwargs):
            triggered_events.append(kwargs)

        event_bus.on("scheduler.job_triggered", on_triggered)

        scheduler = TemporalScheduler(engine=mock_engine, db_path=str(db_path))
        await scheduler.setup()
        await scheduler.start()

        # Add a sub-second timer
        job = await scheduler.add_timer(0.1, label="Quick Alert")
        # Give loop time to process
        await asyncio.sleep(1.2)

        await scheduler.stop()

        # Verify job marked completed
        updated = await scheduler.get_job(job.id)
        assert updated is not None
        assert updated.status == "completed"
        assert len(triggered_events) >= 1
        assert triggered_events[0]["job"]["id"] == job.id


@pytest.mark.asyncio
async def test_scheduler_builtin_tools() -> None:
    """Test all 10 registered scheduler tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tools.db"
        registry = ToolRegistry()
        mock_engine = MagicMock()
        mock_engine.tool_registry = registry
        mock_engine.events = EventBus()
        scheduler = TemporalScheduler(engine=mock_engine, db_path=str(db_path))
        await scheduler.setup()
        mock_engine.scheduler = scheduler

        register_scheduler_tools(mock_engine)

        async def run_tool(name: str, args: dict) -> Any:
            tool = registry.get(name)
            assert tool is not None, f"Tool {name} not found in registry"
            return await tool.handler(**args)

        # 1. timer.set
        set_res = await run_tool("timer.set", {"duration_seconds": 120, "label": "Pasta"})
        assert set_res["status"] == "success"
        timer_id = set_res["timer"]["id"]

        # 2. timer.list
        list_res = await run_tool("timer.list", {})
        assert list_res["status"] == "success"
        assert list_res["count"] == 1

        # 3. timer.cancel
        cancel_res = await run_tool("timer.cancel", {"identifier": timer_id})
        assert cancel_res["status"] == "success"

        # 4. alarm.set
        alarm_res = await run_tool("alarm.set", {"target_time": "7:00 AM", "label": "Gym"})
        assert alarm_res["status"] == "success"
        alarm_id = alarm_res["alarm"]["id"]

        # 5. alarm.list
        alarm_list_res = await run_tool("alarm.list", {})
        assert alarm_list_res["count"] == 1

        # 6. alarm.cancel
        alarm_cancel_res = await run_tool("alarm.cancel", {"identifier": alarm_id})
        assert alarm_cancel_res["status"] == "success"

        # 7. reminder.schedule
        rem_res = await run_tool("reminder.schedule", {"time_expression": "in 15 mins", "message": "Call Doctor"})
        assert rem_res["status"] == "success"
        rem_id = rem_res["reminder"]["id"]

        # 8. reminder.list
        rem_list_res = await run_tool("reminder.list", {})
        assert rem_list_res["count"] == 1

        # 9. reminder.cancel
        rem_cancel_res = await run_tool("reminder.cancel", {"identifier": rem_id})
        assert rem_cancel_res["status"] == "success"

        # 10. system.daily_briefing
        briefing_res = await run_tool("system.daily_briefing", {})
        assert briefing_res["status"] == "success"
        assert "current_time" in briefing_res
