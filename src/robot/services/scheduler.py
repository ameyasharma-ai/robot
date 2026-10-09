"""Temporal Scheduling & Proactive Alert Engine.

Provides persistent SQLite-backed job scheduling for:
- Relative countdown timers ("Set a timer for 12 minutes")
- Absolute scheduled alarms ("Wake me up tomorrow at 7:30 AM")
- Natural language reminders ("Remind me at 4 PM to check the test results")
- Proactive event dispatching and executive daily briefings
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, TYPE_CHECKING
import uuid

from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("scheduler")


@dataclass
class ScheduledJob:
    """Represents a scheduled timer, alarm, reminder, or cron job."""

    id: str
    job_type: str  # "timer", "alarm", "reminder", "cron"
    label: str
    target_time_iso: str
    target_timestamp: float
    duration_seconds: float
    recurring: bool
    status: str  # "pending", "completed", "cancelled"
    created_at_iso: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for JSON serialization and tool responses."""
        remaining_s = max(0.0, self.target_timestamp - time.time())
        d = asdict(self)
        d["remaining_seconds"] = round(remaining_s, 1)
        d["remaining_human"] = _format_remaining_human(remaining_s)
        return d


def _format_remaining_human(seconds: float) -> str:
    """Format remaining duration into a concise human-readable string."""
    if seconds <= 0:
        return "0s (due now)"
    total_sec = int(seconds)
    hours, remainder = divmod(total_sec, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def parse_time_expression(expr: str) -> tuple[float, str, float]:
    """Parse relative or absolute time expressions into target timestamps.

    Args:
        expr: Spoken or text time expression (e.g. "10s", "15 minutes", "2 hours",
            "7:30 AM", "14:00", "tomorrow at 8am", "in 5 mins").

    Returns:
        tuple of (target_timestamp, target_time_iso, duration_seconds)

    Raises:
        ValueError: If the expression cannot be parsed into a valid timestamp.
    """
    now_ts = time.time()
    now_dt = datetime.now()
    clean_expr = expr.strip().lower()

    # 1. Pure numbers -> assume seconds
    if clean_expr.isdigit():
        sec = float(clean_expr)
        target_ts = now_ts + sec
        target_dt = datetime.fromtimestamp(target_ts)
        return target_ts, target_dt.isoformat(), sec

    # 2. Relative durations: "in 10 minutes", "15m", "2 hours 30 mins", "45s", "10 seconds"
    duration_match = re.match(
        r"^(?:in\s+)?(?:(\d+)\s*(?:hours?|hrs?|h))?\s*(?:(\d+)\s*(?:minutes?|mins?|m))?\s*(?:(\d+)\s*(?:seconds?|secs?|s))?$",
        clean_expr,
    )
    if duration_match and any(duration_match.groups()):
        h = int(duration_match.group(1) or 0)
        m = int(duration_match.group(2) or 0)
        s = int(duration_match.group(3) or 0)
        total_sec = float(h * 3600 + m * 60 + s)
        if total_sec > 0:
            target_ts = now_ts + total_sec
            target_dt = datetime.fromtimestamp(target_ts)
            return target_ts, target_dt.isoformat(), total_sec

    # 3. Absolute clock times: "7:30 am", "19:45", "at 4 pm", "tomorrow at 8:00 am", "8am"
    clock_pattern = r"(?:(?:tomorrow|today|at)\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    clock_match = re.search(clock_pattern, clean_expr)
    if clock_match:
        raw_h = int(clock_match.group(1))
        raw_m = int(clock_match.group(2) or 0)
        meridiem = (clock_match.group(3) or "").lower()

        if meridiem == "pm" and raw_h < 12:
            raw_h += 12
        elif meridiem == "am" and raw_h == 12:
            raw_h = 0

        is_tomorrow = "tomorrow" in clean_expr
        target_dt = now_dt.replace(hour=raw_h, minute=raw_m, second=0, microsecond=0)

        # If time has already passed today and user didn't specify tomorrow, push to tomorrow
        if target_dt <= now_dt and not is_tomorrow:
            target_dt += timedelta(days=1)
        elif is_tomorrow and target_dt <= now_dt:
            target_dt += timedelta(days=1)

        total_sec = max(1.0, (target_dt - now_dt).total_seconds())
        return target_dt.timestamp(), target_dt.isoformat(), total_sec

    # 4. Standard ISO string fallback
    try:
        parsed_dt = datetime.fromisoformat(clean_expr)
        total_sec = max(1.0, (parsed_dt - now_dt).total_seconds())
        return parsed_dt.timestamp(), parsed_dt.isoformat(), total_sec
    except Exception:
        pass

    raise ValueError(f"Could not parse time expression: '{expr}'. Expected formats like '10 minutes', '7:30 AM', or 'in 2 hours'.")


class TemporalScheduler:
    """Persistent SQLite-backed temporal scheduler for timers, alarms, and reminders."""

    def __init__(self, engine: RobotEngine | None = None, db_path: str = "data/database/scheduler.db") -> None:
        """Initialize the temporal scheduler.

        Args:
            engine: Optional reference to RobotEngine for event and audio dispatch.
            db_path: Path to persistent SQLite database.
        """
        self._engine = engine
        self._db_path = Path(db_path)
        self._task: asyncio.Task | None = None
        self._running = False
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        """Initialize database tables and indexes."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_db_sync)
        logger.info("scheduler_database_initialized", db_path=str(self._db_path))

    def _init_db_sync(self) -> None:
        """Create database tables synchronously."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    target_time_iso TEXT NOT NULL,
                    target_timestamp REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    recurring INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at_iso TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduler_status_target
                ON scheduled_jobs(status, target_timestamp);
                """
            )
            conn.commit()

    async def start(self) -> None:
        """Start the background 1-second job evaluation loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop(), name="temporal_scheduler_loop")
        logger.info("temporal_scheduler_started")

    async def stop(self) -> None:
        """Stop background job evaluation loop cleanly."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("temporal_scheduler_stopped")

    async def health_check(self) -> bool:
        """Verify scheduler database availability and loop health."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._health_check_sync)
            return True
        except Exception as e:
            logger.error("scheduler_health_check_failed", error=str(e))
            return False

    def _health_check_sync(self) -> None:
        with sqlite3.connect(str(self._db_path), timeout=2.0) as conn:
            conn.execute("SELECT COUNT(*) FROM scheduled_jobs").fetchone()

    async def add_timer(
        self,
        duration_seconds: float,
        label: str = "Timer",
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Schedule a relative countdown timer.

        Args:
            duration_seconds: Number of seconds from now until trigger.
            label: Friendly description or label for the timer.
            metadata: Optional additional context parameters.

        Returns:
            The created ScheduledJob object.
        """
        now_ts = time.time()
        target_ts = now_ts + max(1.0, float(duration_seconds))
        target_iso = datetime.fromtimestamp(target_ts).isoformat()
        job = ScheduledJob(
            id=f"timer_{uuid.uuid4().hex[:8]}",
            job_type="timer",
            label=label or "Timer",
            target_time_iso=target_iso,
            target_timestamp=target_ts,
            duration_seconds=float(duration_seconds),
            recurring=False,
            status="pending",
            created_at_iso=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        await self._insert_job(job)
        logger.info("timer_scheduled", job_id=job.id, label=job.label, duration_s=duration_seconds)
        if self._engine and hasattr(self._engine, "events"):
            await self._engine.events.emit("scheduler.job_added", job=job.to_dict())
        return job

    async def add_alarm(
        self,
        target_time_expr: str,
        label: str = "Alarm",
        recurring: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Schedule an absolute clock alarm.

        Args:
            target_time_expr: Time string like '7:30 AM', '08:00', 'tomorrow at 7am'.
            label: Friendly name for alarm.
            recurring: Whether this alarm repeats daily.
            metadata: Optional dictionary metadata.

        Returns:
            The created ScheduledJob object.
        """
        target_ts, target_iso, duration_sec = parse_time_expression(target_time_expr)
        job = ScheduledJob(
            id=f"alarm_{uuid.uuid4().hex[:8]}",
            job_type="alarm",
            label=label or "Alarm",
            target_time_iso=target_iso,
            target_timestamp=target_ts,
            duration_seconds=duration_sec,
            recurring=recurring,
            status="pending",
            created_at_iso=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        await self._insert_job(job)
        logger.info("alarm_scheduled", job_id=job.id, label=job.label, target=target_iso, recurring=recurring)
        if self._engine and hasattr(self._engine, "events"):
            await self._engine.events.emit("scheduler.job_added", job=job.to_dict())
        return job

    async def add_reminder(
        self,
        time_expr: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Schedule a natural language reminder.

        Args:
            time_expr: Relative or absolute time expression ('in 15 mins', '4 PM').
            message: The reminder notice message to announce.
            metadata: Optional metadata.

        Returns:
            The created ScheduledJob object.
        """
        target_ts, target_iso, duration_sec = parse_time_expression(time_expr)
        job = ScheduledJob(
            id=f"reminder_{uuid.uuid4().hex[:8]}",
            job_type="reminder",
            label=message,
            target_time_iso=target_iso,
            target_timestamp=target_ts,
            duration_seconds=duration_sec,
            recurring=False,
            status="pending",
            created_at_iso=datetime.now().isoformat(),
            metadata=metadata or {"message": message},
        )
        await self._insert_job(job)
        logger.info("reminder_scheduled", job_id=job.id, message=message, target=target_iso)
        if self._engine and hasattr(self._engine, "events"):
            await self._engine.events.emit("scheduler.job_added", job=job.to_dict())
        return job

    async def list_jobs(
        self,
        job_type: str | None = None,
        status: str = "pending",
        limit: int = 50,
    ) -> list[ScheduledJob]:
        """List scheduled jobs filtered by status and job type.

        Args:
            job_type: Optional filter by "timer", "alarm", "reminder", or "cron".
            status: Status filter ("pending", "completed", "cancelled", or "all").
            limit: Maximum items to return.

        Returns:
            List of ScheduledJob instances matching query.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._list_jobs_sync, job_type, status, limit)

    def _list_jobs_sync(self, job_type: str | None, status: str, limit: int) -> list[ScheduledJob]:
        query = "SELECT id, job_type, label, target_time_iso, target_timestamp, duration_seconds, recurring, status, created_at_iso, metadata_json FROM scheduled_jobs WHERE 1=1"
        params: list[Any] = []
        if status != "all":
            query += " AND status = ?"
            params.append(status)
        if job_type:
            query += " AND job_type = ?"
            params.append(job_type)
        query += " ORDER BY target_timestamp ASC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            rows = cursor.execute(query, params).fetchall()
            jobs = []
            for r in rows:
                jobs.append(
                    ScheduledJob(
                        id=r[0],
                        job_type=r[1],
                        label=r[2],
                        target_time_iso=r[3],
                        target_timestamp=float(r[4]),
                        duration_seconds=float(r[5]),
                        recurring=bool(r[6]),
                        status=r[7],
                        created_at_iso=r[8],
                        metadata=json.loads(r[9] or "{}"),
                    )
                )
            return jobs

    async def get_job(self, job_id: str) -> ScheduledJob | None:
        """Retrieve a specific scheduled job by ID."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_job_sync, job_id)

    def _get_job_sync(self, job_id: str) -> ScheduledJob | None:
        with sqlite3.connect(str(self._db_path)) as conn:
            r = conn.execute(
                "SELECT id, job_type, label, target_time_iso, target_timestamp, duration_seconds, recurring, status, created_at_iso, metadata_json FROM scheduled_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not r:
                return None
            return ScheduledJob(
                id=r[0],
                job_type=r[1],
                label=r[2],
                target_time_iso=r[3],
                target_timestamp=float(r[4]),
                duration_seconds=float(r[5]),
                recurring=bool(r[6]),
                status=r[7],
                created_at_iso=r[8],
                metadata=json.loads(r[9] or "{}"),
            )

    async def cancel_job(self, identifier: str) -> bool:
        """Cancel a job by exact ID or case-insensitive label substring.

        Args:
            identifier: Job ID or friendly label name.

        Returns:
            True if one or more matching active jobs were cancelled, False otherwise.
        """
        loop = asyncio.get_running_loop()
        cancelled_ids = await loop.run_in_executor(None, self._cancel_job_sync, identifier)
        if cancelled_ids and self._engine and hasattr(self._engine, "events"):
            for jid in cancelled_ids:
                await self._engine.events.emit("scheduler.job_cancelled", job_id=jid)
        return len(cancelled_ids) > 0

    def _cancel_job_sync(self, identifier: str) -> list[str]:
        clean_id = identifier.strip()
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            # 1. Exact ID match
            rows = cursor.execute("SELECT id FROM scheduled_jobs WHERE id = ? AND status = 'pending'", (clean_id,)).fetchall()
            if not rows:
                # 2. Fuzzy label match
                rows = cursor.execute(
                    "SELECT id FROM scheduled_jobs WHERE status = 'pending' AND LOWER(label) LIKE ?",
                    (f"%{clean_id.lower()}%",),
                ).fetchall()

            if not rows:
                return []

            matched_ids = [r[0] for r in rows]
            placeholders = ",".join("?" for _ in matched_ids)
            cursor.execute(f"UPDATE scheduled_jobs SET status = 'cancelled' WHERE id IN ({placeholders})", matched_ids)
            conn.commit()
            return matched_ids

    async def get_daily_briefing(self) -> dict[str, Any]:
        """Compile an aggregated executive daily briefing.

        Returns:
            Dictionary containing current datetime, pending reminders, active timers,
            and system health overview.
        """
        now = datetime.now()
        pending_jobs = await self.list_jobs(status="pending", limit=20)
        timers = [j.to_dict() for j in pending_jobs if j.job_type == "timer"]
        alarms = [j.to_dict() for j in pending_jobs if j.job_type == "alarm"]
        reminders = [j.to_dict() for j in pending_jobs if j.job_type == "reminder"]

        system_status = "nominal"
        health_results = {"system": True}
        if self._engine and getattr(self._engine, "health", None) is not None:
            try:
                res = self._engine.health.check_all()
                if asyncio.iscoroutine(res):
                    health_results = await res
                elif isinstance(res, dict):
                    health_results = res
            except Exception:
                pass
        if not all(bool(v) for v in health_results.values()):
            system_status = "degraded"

        return {
            "current_time": now.strftime("%I:%M %p"),
            "current_date": now.strftime("%A, %B %d, %Y"),
            "system_status": system_status,
            "health_checks": health_results,
            "active_timers_count": len(timers),
            "upcoming_alarms_count": len(alarms),
            "pending_reminders_count": len(reminders),
            "timers": timers,
            "alarms": alarms,
            "reminders": reminders,
        }

    async def _insert_job(self, job: ScheduledJob) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._insert_job_sync, job)

    def _insert_job_sync(self, job: ScheduledJob) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO scheduled_jobs (
                    id, job_type, label, target_time_iso, target_timestamp,
                    duration_seconds, recurring, status, created_at_iso, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.job_type,
                    job.label,
                    job.target_time_iso,
                    job.target_timestamp,
                    job.duration_seconds,
                    1 if job.recurring else 0,
                    job.status,
                    job.created_at_iso,
                    json.dumps(job.metadata),
                ),
            )
            conn.commit()

    async def _scheduler_loop(self) -> None:
        """Background loop executing every 1 second to dispatch due jobs."""
        while self._running:
            try:
                now_ts = time.time()
                due_jobs = await self._fetch_and_mark_due_jobs(now_ts)
                for job in due_jobs:
                    asyncio.create_task(self._trigger_job(job))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduler_loop_iteration_error", error=str(e))
            await asyncio.sleep(1.0)

    async def _fetch_and_mark_due_jobs(self, current_ts: float) -> list[ScheduledJob]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_and_mark_due_jobs_sync, current_ts)

    def _fetch_and_mark_due_jobs_sync(self, current_ts: float) -> list[ScheduledJob]:
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                """
                SELECT id, job_type, label, target_time_iso, target_timestamp,
                       duration_seconds, recurring, status, created_at_iso, metadata_json
                FROM scheduled_jobs
                WHERE status = 'pending' AND target_timestamp <= ?
                ORDER BY target_timestamp ASC
                """,
                (current_ts,),
            ).fetchall()

            if not rows:
                return []

            due_jobs: list[ScheduledJob] = []
            for r in rows:
                job = ScheduledJob(
                    id=r[0],
                    job_type=r[1],
                    label=r[2],
                    target_time_iso=r[3],
                    target_timestamp=float(r[4]),
                    duration_seconds=float(r[5]),
                    recurring=bool(r[6]),
                    status=r[7],
                    created_at_iso=r[8],
                    metadata=json.loads(r[9] or "{}"),
                )
                due_jobs.append(job)

                # Mark completed or calculate next recurring timestamp
                if job.recurring and job.job_type == "alarm":
                    next_ts = job.target_timestamp + 86400.0  # +24 hours
                    next_iso = datetime.fromtimestamp(next_ts).isoformat()
                    cursor.execute(
                        """
                        UPDATE scheduled_jobs
                        SET target_timestamp = ?, target_time_iso = ?
                        WHERE id = ?
                        """,
                        (next_ts, next_iso, job.id),
                    )
                else:
                    cursor.execute(
                        "UPDATE scheduled_jobs SET status = 'completed' WHERE id = ?",
                        (job.id,),
                    )

            conn.commit()
            return due_jobs

    async def _trigger_job(self, job: ScheduledJob) -> None:
        """Dispatch a triggered job via EventBus, WebSocket, and vocal announcement."""
        logger.info("job_triggered", job_id=job.id, job_type=job.job_type, label=job.label)

        # 1. Emit events across EventBus and WebSockets
        if self._engine and hasattr(self._engine, "events"):
            try:
                await self._engine.events.emit(
                    "scheduler.job_triggered",
                    job=job.to_dict(),
                )
                await self._engine.events.emit(
                    f"{job.job_type}.triggered",
                    job=job.to_dict(),
                )
            except Exception as e:
                logger.warning("failed_to_emit_job_event", error=str(e))

        # 2. Spoken notification if engine voice pipeline is active
        if self._engine:
            try:
                spoken_text = ""
                if job.job_type == "timer":
                    spoken_text = f"Your timer for {job.label} is done!"
                elif job.job_type == "alarm":
                    spoken_text = f"Alarm: {job.label}! It is now {datetime.now().strftime('%I:%M %p')}."
                elif job.job_type == "reminder":
                    spoken_text = f"Reminder: {job.label}."
                else:
                    spoken_text = f"Alert: {job.label}."

                print(f"\n⏰ \033[1;33m[{job.job_type.upper()} ALERT]\033[0m {spoken_text}\n")

                if hasattr(self._engine, "_stream_response"):
                    asyncio.create_task(
                        self._engine._stream_response(
                            f"[Scheduler alert]: {spoken_text}",
                            time.time(),
                            asyncio.Event(),
                        )
                    )
                elif hasattr(self._engine, "_tts_queue"):
                    interrupt = asyncio.Event()
                    await self._engine._tts_queue.put((spoken_text, interrupt, None))
                    await self._engine._tts_queue.put(None)
            except Exception as e:
                logger.warning("failed_to_speak_job_alert", error=str(e))
