"""Built-in temporal scheduling tools for timers, alarms, reminders, and daily briefings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot.services.scheduler import parse_time_expression
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


def register_scheduler_tools(engine: "RobotEngine") -> None:
    """Register all temporal and scheduling tools with the engine's tool registry.

    Args:
        engine: Reference to the RobotEngine instance.
    """

    # 1. timer.set
    async def timer_set(duration_seconds: int = 60, label: str = "Timer", **kwargs: Any) -> dict[str, Any]:
        """Schedule a countdown timer for a specified number of seconds."""
        eff_duration = duration_seconds or kwargs.get("duration") or kwargs.get("seconds") or kwargs.get("time") or 60
        # If user passed a string like "5m" or "10 minutes"
        if isinstance(eff_duration, str):
            try:
                _, _, eff_duration = parse_time_expression(eff_duration)
            except Exception:
                eff_duration = 60
        eff_label = (label or kwargs.get("name") or kwargs.get("title") or "Timer").strip()
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        job = await engine.scheduler.add_timer(float(eff_duration), label=eff_label)
        return {
            "status": "success",
            "message": f"Timer '{job.label}' set for {job.duration_seconds} seconds ({job.to_dict()['remaining_human']}).",
            "timer": job.to_dict(),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="timer.set",
            description="Set a countdown timer for a duration in seconds with a label. Spoken triggers: 'Set a timer for 5 minutes', 'Timer 30 seconds for tea'.",
            parameters={
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Duration in seconds (e.g. 60 for 1 min, 300 for 5 mins, 900 for 15 mins)",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short label or name describing what this timer is for (e.g. 'Tea', 'Oven', 'Focus session')",
                        "default": "Timer",
                    },
                },
                "required": ["duration_seconds"],
            },
            handler=timer_set,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="scheduler",
        )
    )

    # 2. timer.list
    async def timer_list(status: str = "pending", **kwargs: Any) -> dict[str, Any]:
        """List active or pending countdown timers."""
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        jobs = await engine.scheduler.list_jobs(job_type="timer", status=status)
        return {
            "status": "success",
            "count": len(jobs),
            "timers": [j.to_dict() for j in jobs],
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="timer.list",
            description="List all currently active or pending countdown timers. Spoken triggers: 'What timers are running?', 'Show active timers'.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter status ('pending', 'completed', 'cancelled', 'all')",
                        "default": "pending",
                    },
                },
            },
            handler=timer_list,
            permission_level=PermissionLevel.READ_ONLY,
            family="scheduler",
        )
    )

    # 3. timer.cancel
    async def timer_cancel(identifier: str = "", **kwargs: Any) -> dict[str, Any]:
        """Cancel a countdown timer by ID or matching label."""
        eff_id = (identifier or kwargs.get("timer_id") or kwargs.get("label") or kwargs.get("name") or "").strip()
        if not eff_id:
            return {"status": "error", "message": "Please specify a timer ID or label to cancel."}
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        cancelled = await engine.scheduler.cancel_job(eff_id)
        if cancelled:
            return {"status": "success", "message": f"Successfully cancelled timer matching '{eff_id}'."}
        return {"status": "not_found", "message": f"No active timer found matching '{eff_id}'."}

    engine.tool_registry.register(
        ToolDefinition(
            name="timer.cancel",
            description="Cancel an active countdown timer by its ID or friendly label name. Spoken triggers: 'Cancel the tea timer', 'Stop the 5 minute timer'.",
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "The timer ID (e.g. 'timer_abc123') or the label of the timer (e.g. 'tea')",
                    },
                },
                "required": ["identifier"],
            },
            handler=timer_cancel,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="scheduler",
        )
    )

    # 4. alarm.set
    async def alarm_set(
        target_time: str = "",
        label: str = "Alarm",
        recurring: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Schedule an absolute clock alarm."""
        eff_time = (target_time or kwargs.get("time") or kwargs.get("time_expression") or "").strip()
        if not eff_time:
            return {"status": "error", "message": "Please specify a target time for the alarm (e.g. '7:30 AM', '08:00')."}
        eff_label = (label or kwargs.get("name") or "Alarm").strip()
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        try:
            job = await engine.scheduler.add_alarm(eff_time, label=eff_label, recurring=recurring)
            return {
                "status": "success",
                "message": f"Alarm '{job.label}' set for {job.target_time_iso} ({job.to_dict()['remaining_human']}).",
                "alarm": job.to_dict(),
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to set alarm: {str(e)}"}

    engine.tool_registry.register(
        ToolDefinition(
            name="alarm.set",
            description="Set an alarm for a specific clock time. Spoken triggers: 'Wake me up at 7:30 AM', 'Set an alarm for 8:00 tomorrow', 'Set daily alarm at 6:30 AM'.",
            parameters={
                "type": "object",
                "properties": {
                    "target_time": {
                        "type": "string",
                        "description": "Clock time expression (e.g. '7:30 AM', '14:00', 'tomorrow at 8:00 AM')",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short description for the alarm (e.g. 'Wake up', 'Standup meeting')",
                        "default": "Alarm",
                    },
                    "recurring": {
                        "type": "boolean",
                        "description": "Whether this alarm should repeat every day at the same time",
                        "default": False,
                    },
                },
                "required": ["target_time"],
            },
            handler=alarm_set,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="scheduler",
        )
    )

    # 5. alarm.list
    async def alarm_list(status: str = "pending", **kwargs: Any) -> dict[str, Any]:
        """List scheduled alarms."""
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        jobs = await engine.scheduler.list_jobs(job_type="alarm", status=status)
        return {
            "status": "success",
            "count": len(jobs),
            "alarms": [j.to_dict() for j in jobs],
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="alarm.list",
            description="List all scheduled alarms. Spoken triggers: 'What alarms are set?', 'Show my alarms'.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter status ('pending', 'completed', 'cancelled', 'all')",
                        "default": "pending",
                    },
                },
            },
            handler=alarm_list,
            permission_level=PermissionLevel.READ_ONLY,
            family="scheduler",
        )
    )

    # 6. alarm.cancel
    async def alarm_cancel(identifier: str = "", **kwargs: Any) -> dict[str, Any]:
        """Cancel a scheduled alarm by ID or matching label."""
        eff_id = (identifier or kwargs.get("alarm_id") or kwargs.get("label") or kwargs.get("name") or "").strip()
        if not eff_id:
            return {"status": "error", "message": "Please specify an alarm ID or label to cancel."}
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        cancelled = await engine.scheduler.cancel_job(eff_id)
        if cancelled:
            return {"status": "success", "message": f"Successfully cancelled alarm matching '{eff_id}'."}
        return {"status": "not_found", "message": f"No active alarm found matching '{eff_id}'."}

    engine.tool_registry.register(
        ToolDefinition(
            name="alarm.cancel",
            description="Cancel a scheduled alarm by ID or label. Spoken triggers: 'Cancel my 7:30 alarm', 'Turn off the wake up alarm'.",
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "The alarm ID or matching label name",
                    },
                },
                "required": ["identifier"],
            },
            handler=alarm_cancel,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="scheduler",
        )
    )

    # 7. reminder.schedule
    async def reminder_schedule(
        time_expression: str = "",
        message: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Schedule a natural language reminder."""
        eff_time = (time_expression or kwargs.get("time") or kwargs.get("when") or "").strip()
        eff_msg = (message or kwargs.get("text") or kwargs.get("note") or kwargs.get("reminder") or "").strip()
        if not eff_time or not eff_msg:
            return {"status": "error", "message": "Please provide both a time expression and a reminder message."}
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        try:
            job = await engine.scheduler.add_reminder(eff_time, message=eff_msg)
            return {
                "status": "success",
                "message": f"Reminder scheduled for {job.target_time_iso} ({job.to_dict()['remaining_human']}): '{job.label}'.",
                "reminder": job.to_dict(),
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to schedule reminder: {str(e)}"}

    engine.tool_registry.register(
        ToolDefinition(
            name="reminder.schedule",
            description="Schedule a reminder for a future time. Spoken triggers: 'Remind me at 4 PM to call Alex', 'Remind me in 20 minutes to stretch'.",
            parameters={
                "type": "object",
                "properties": {
                    "time_expression": {
                        "type": "string",
                        "description": "When to remind the user (e.g. 'in 15 minutes', '4:00 PM', 'tomorrow at 10am')",
                    },
                    "message": {
                        "type": "string",
                        "description": "The exact reminder note or task message to announce",
                    },
                },
                "required": ["time_expression", "message"],
            },
            handler=reminder_schedule,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="scheduler",
        )
    )

    # 8. reminder.list
    async def reminder_list(status: str = "pending", **kwargs: Any) -> dict[str, Any]:
        """List upcoming reminders."""
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        jobs = await engine.scheduler.list_jobs(job_type="reminder", status=status)
        return {
            "status": "success",
            "count": len(jobs),
            "reminders": [j.to_dict() for j in jobs],
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="reminder.list",
            description="List all pending reminders. Spoken triggers: 'What are my reminders?', 'Show upcoming reminders'.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter status ('pending', 'completed', 'cancelled', 'all')",
                        "default": "pending",
                    },
                },
            },
            handler=reminder_list,
            permission_level=PermissionLevel.READ_ONLY,
            family="scheduler",
        )
    )

    # 9. reminder.cancel
    async def reminder_cancel(identifier: str = "", **kwargs: Any) -> dict[str, Any]:
        """Cancel a reminder by ID or matching message text."""
        eff_id = (identifier or kwargs.get("reminder_id") or kwargs.get("message") or kwargs.get("label") or "").strip()
        if not eff_id:
            return {"status": "error", "message": "Please specify a reminder ID or text to cancel."}
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        cancelled = await engine.scheduler.cancel_job(eff_id)
        if cancelled:
            return {"status": "success", "message": f"Successfully cancelled reminder matching '{eff_id}'."}
        return {"status": "not_found", "message": f"No active reminder found matching '{eff_id}'."}

    engine.tool_registry.register(
        ToolDefinition(
            name="reminder.cancel",
            description="Cancel an active reminder by ID or message text. Spoken triggers: 'Cancel my reminder to stretch', 'Delete the 4 PM reminder'.",
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "The reminder ID or keywords from the reminder message",
                    },
                },
                "required": ["identifier"],
            },
            handler=reminder_cancel,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="scheduler",
        )
    )

    # 10. system.daily_briefing
    async def daily_briefing(include_news: bool = False, include_weather: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Compile an aggregated executive daily briefing."""
        if not engine.scheduler:
            return {"status": "error", "message": "Scheduler service is not initialized."}
        briefing = await engine.scheduler.get_daily_briefing()
        briefing["status"] = "success"
        return briefing

    engine.tool_registry.register(
        ToolDefinition(
            name="system.daily_briefing",
            description="Provide an executive daily briefing covering current time, upcoming alarms, pending reminders, active timers, and system health. Spoken triggers: 'Good morning', 'Daily briefing', 'What is my schedule today?'.",
            parameters={
                "type": "object",
                "properties": {
                    "include_news": {
                        "type": "boolean",
                        "description": "Whether to include top news headlines",
                        "default": False,
                    },
                    "include_weather": {
                        "type": "boolean",
                        "description": "Whether to include weather forecast",
                        "default": False,
                    },
                },
            },
            handler=daily_briefing,
            permission_level=PermissionLevel.READ_ONLY,
            family="scheduler",
        )
    )
