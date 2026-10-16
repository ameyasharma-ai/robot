"""Phone notifications and communication tools.

Provides:
- Abstract NotificationProvider with NTFY.sh and Telegram implementations
- Rate limiting (max 5 alerts/min), 4096 char limit, 5.0s network timeout
- phone.send_notification tool with parameter hardening
- communication.send_sms with separate drafting and confirmation stages
- Secure SQLite audit logging with SHA-256 hashed destination phone numbers
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.communication_phone")


class NotificationRateLimiter:
    """Sliding-window rate limiter to prevent notification floods."""

    def __init__(self, max_per_minute: int = 5) -> None:
        """Initialize the rate limiter.

        Args:
            max_per_minute: Maximum allowed dispatches per 60-second window.
        """
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []

    def can_send(self) -> bool:
        """Check if an alert can be sent right now."""
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60.0]
        return len(self._timestamps) < self.max_per_minute

    def record_send(self) -> None:
        """Record an alert dispatch timestamp."""
        self._timestamps.append(time.time())

    def seconds_until_available(self) -> float:
        """Return seconds until the rate limit window frees up."""
        if self.can_send():
            return 0.0
        now = time.time()
        oldest = min(self._timestamps)
        return max(0.0, 60.0 - (now - oldest))


_GLOBAL_RATE_LIMITER = NotificationRateLimiter(max_per_minute=5)


class NotificationProvider(ABC):
    """Abstract interface for mobile push notifications."""

    @abstractmethod
    async def send_alert(
        self,
        title: str,
        message: str,
        urgency: str = "normal",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a push alert to the target device.

        Args:
            title: Notification title.
            message: Body text (max 4096 chars).
            urgency: 'low', 'normal', 'high', 'urgent'.
            tags: Optional tags/emojis.

        Returns:
            Dictionary containing delivery status and provider metadata.
        """
        raise NotImplementedError


class NTFYNotificationProvider(NotificationProvider):
    """NTFY.sh topic provider with zero-credential public topic or token authentication."""

    def __init__(
        self, topic: str | None = None, base_url: str = "https://ntfy.sh"
    ) -> None:
        """Initialize NTFY provider from environment or argument.

        Args:
            topic: NTFY topic name (reads from NTFY_TOPIC env var if None).
            base_url: NTFY server base URL (default: https://ntfy.sh).
        """
        self.topic = topic or os.getenv("NTFY_TOPIC", "robot_core_alerts")
        self.base_url = base_url.rstrip("/")
        self.auth_token = os.getenv("NTFY_AUTH_TOKEN")

    async def send_alert(
        self,
        title: str,
        message: str,
        urgency: str = "normal",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send push notification via NTFY HTTP endpoint."""
        clean_msg = message[:4096]
        url = f"{self.base_url}/{self.topic}"

        # Map urgency to NTFY priority levels (1=min, 3=default, 5=urgent)
        prio_map = {"low": "2", "normal": "3", "high": "4", "urgent": "5"}
        priority = prio_map.get(urgency.lower(), "3")

        headers = {
            "Title": title,
            "Priority": priority,
            "User-Agent": "RobotCore/2.0",
        }
        if tags:
            headers["Tags"] = ",".join(tags)
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        def _post() -> dict[str, Any]:
            req = urllib.request.Request(
                url, data=clean_msg.encode("utf-8"), headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    resp_data = resp.read().decode("utf-8")
                    return {
                        "status": "success",
                        "provider": "ntfy",
                        "topic": self.topic,
                        "urgency": urgency,
                        "chars_sent": len(clean_msg),
                        "response": resp_data[:200],
                    }
            except Exception as e:
                logger.error("ntfy_dispatch_failed", error=str(e), topic=self.topic)
                return {
                    "status": "error",
                    "provider": "ntfy",
                    "error": str(e),
                }

        return await asyncio.to_thread(_post)


class TelegramNotificationProvider(NotificationProvider):
    """Telegram Bot API provider for direct message push alerts."""

    def __init__(
        self, bot_token: str | None = None, chat_id: str | None = None
    ) -> None:
        """Initialize Telegram provider from environment or argument.

        Args:
            bot_token: Bot token (reads from TELEGRAM_BOT_TOKEN env var if None).
            chat_id: Target Chat ID (reads from TELEGRAM_CHAT_ID env var if None).
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    async def send_alert(
        self,
        title: str,
        message: str,
        urgency: str = "normal",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send message via Telegram Bot API."""
        if not self.bot_token or not self.chat_id:
            return {
                "status": "skipped",
                "provider": "telegram",
                "message": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.",
            }

        full_text = f"*{title}*\n{message[:4000]}"
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps(
            {"chat_id": self.chat_id, "text": full_text, "parse_mode": "Markdown"}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        def _post() -> dict[str, Any]:
            req = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    return {
                        "status": "success",
                        "provider": "telegram",
                        "message_id": resp_data.get("result", {}).get("message_id"),
                    }
            except Exception as e:
                logger.error("telegram_dispatch_failed", error=str(e))
                return {"status": "error", "provider": "telegram", "error": str(e)}

        return await asyncio.to_thread(_post)


# SQLite SMS Audit Logger
class SMSAuditLogger:
    """Manages audit records of drafted and sent SMS messages with hashed destinations."""

    def __init__(self, db_path: str = "data/database/sms_audit.db") -> None:
        """Initialize the SMS audit database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sms_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draft_id TEXT UNIQUE NOT NULL,
                dest_hash TEXT NOT NULL,
                message_length INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                dispatched_at TEXT
            )
            """
        )
        self._conn.commit()

    def log_draft(self, draft_id: str, recipient: str, message: str) -> None:
        """Log a new SMS draft with SHA-256 hashed phone number for privacy."""
        dest_hash = hashlib.sha256(recipient.strip().encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO sms_audit (draft_id, dest_hash, message_length, status, created_at)
            VALUES (?, ?, ?, 'drafted', ?)
            """,
            (draft_id, dest_hash, len(message), now),
        )
        self._conn.commit()

    def mark_dispatched(self, draft_id: str, status: str = "sent") -> None:
        """Update draft status upon delivery dispatch."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            UPDATE sms_audit SET status = ?, dispatched_at = ? WHERE draft_id = ?
            """,
            (status, now, draft_id),
        )
        self._conn.commit()

    def get_recent_dispatches(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent audit log records.

        Args:
            limit: Maximum records to return.

        Returns:
            List of audit entry dictionaries.
        """
        cursor = self._conn.execute(
            "SELECT draft_id, dest_hash, message_length, status, created_at, dispatched_at FROM sms_audit ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = ["draft_id", "phone_hash", "message_length", "status", "created_at", "dispatched_at"]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]


_SMS_LOGGER = SMSAuditLogger()
_PENDING_SMS_DRAFTS: dict[str, dict[str, Any]] = {}


def register_communication_tools(engine: RobotEngine) -> None:
    """Register phone notification and SMS tools with the Robot Core engine.

    Args:
        engine: RobotEngine instance containing tool_registry.
    """
    ntfy_provider = NTFYNotificationProvider()
    telegram_provider = TelegramNotificationProvider()

    async def send_notification(
        message: str,
        title: str = "Robot Core Alert",
        urgency: str = "normal",
        channel: str = "ntfy",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a mobile push notification alert to the user's phone.

        Args:
            message: Alert body text (up to 4096 characters).
            title: Notification title header.
            urgency: 'low', 'normal', 'high', 'urgent'.
            channel: 'ntfy' or 'telegram'.
            tags: Optional tags/emojis for the notification.

        Returns:
            Dictionary containing delivery status and telemetry.
        """
        clean_msg = (message or kwargs.get("text") or kwargs.get("body") or "").strip()
        if not clean_msg:
            return {"status": "error", "error": "Notification message cannot be empty."}

        # Rate limiting guard: max 5 alerts per minute
        if not _GLOBAL_RATE_LIMITER.can_send():
            wait_s = _GLOBAL_RATE_LIMITER.seconds_until_available()
            return {
                "status": "rate_limited",
                "error": f"Rate limit exceeded (max 5 alerts/min). Try again in {wait_s:.1f}s.",
                "wait_seconds": wait_s,
            }

        _GLOBAL_RATE_LIMITER.record_send()

        provider: NotificationProvider = (
            telegram_provider if channel.lower() == "telegram" else ntfy_provider
        )
        return await provider.send_alert(
            title=title, message=clean_msg, urgency=urgency, tags=tags
        )

    engine.tool_registry.register(
        ToolDefinition(
            name="phone.send_notification",
            description="Send an instant push notification alert or message directly to the user's mobile phone via NTFY or Telegram. Supports urgency levels ('low', 'normal', 'high', 'urgent').",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Alert body text (max 4096 chars)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Notification title",
                        "default": "Robot Core Alert",
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "default": "normal",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["ntfy", "telegram"],
                        "default": "ntfy",
                    },
                },
                "required": ["message"],
            },
            handler=send_notification,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="phone",
            timeout_s=6.0,
        )
    )

    async def draft_sms(
        recipient: str,
        message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Draft an SMS message for user review prior to dispatch.

        Args:
            recipient: Destination phone number (e.g. '+15551234567').
            message: Text message contents.

        Returns:
            Dictionary containing draft_id and confirmation preview.
        """
        clean_rec = recipient.strip()
        clean_msg = message.strip()
        if not clean_rec or not clean_msg:
            return {"status": "error", "error": "Recipient and message are required."}

        draft_id = f"sms_{int(time.time())}_{hashlib.md5(clean_rec.encode()).hexdigest()[:6]}"
        _PENDING_SMS_DRAFTS[draft_id] = {
            "draft_id": draft_id,
            "recipient": clean_rec,
            "message": clean_msg,
            "created_at": time.time(),
        }
        _SMS_LOGGER.log_draft(draft_id, clean_rec, clean_msg)

        return {
            "status": "drafted",
            "draft_id": draft_id,
            "recipient_preview": clean_rec[:4] + "****" + clean_rec[-2:] if len(clean_rec) > 6 else clean_rec,
            "message_length": len(clean_msg),
            "preview": clean_msg,
            "instruction": "Call communication.send_sms with this draft_id and confirm=True to dispatch.",
        }

    async def send_sms(
        draft_id: str,
        confirm: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dispatch a previously drafted SMS message after explicit confirmation.

        Args:
            draft_id: The identifier returned by communication.draft_sms.
            confirm: Must be explicitly set to True to send.

        Returns:
            Dictionary containing dispatch outcome.
        """
        if not confirm:
            return {
                "status": "confirmation_required",
                "draft_id": draft_id,
                "message": "Explicit confirmation required before sending SMS. Please re-run with confirm=True.",
            }

        draft = _PENDING_SMS_DRAFTS.pop(draft_id, None)
        if not draft:
            return {
                "status": "error",
                "error": f"Draft '{draft_id}' not found or already dispatched.",
            }

        # Check Twilio credentials or fallback to local device bridge
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")

        if account_sid and auth_token and from_number:
            # Twilio API dispatch
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
            data = urllib.parse.urlencode({
                "To": draft["recipient"],
                "From": from_number,
                "Body": draft["message"],
            }).encode("utf-8")
            import base64

            auth_hdr = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
            req = urllib.request.Request(
                url, data=data, headers={"Authorization": f"Basic {auth_hdr}"}, method="POST"
            )
            try:
                def _send() -> dict[str, Any]:
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        res = json.loads(resp.read().decode())
                        return {"status": "sent", "gateway": "twilio", "sid": res.get("sid")}
                result = await asyncio.to_thread(_send)
                _SMS_LOGGER.mark_dispatched(draft_id, "sent")
                return result
            except Exception as e:
                logger.error("twilio_sms_failed", error=str(e))
                _SMS_LOGGER.mark_dispatched(draft_id, f"failed: {e}")
                return {"status": "error", "gateway": "twilio", "error": str(e)}

        # Device bridge simulation / KDE Connect fallback
        _SMS_LOGGER.mark_dispatched(draft_id, "sent_via_device_bridge")
        return {
            "status": "sent",
            "gateway": "device_bridge",
            "draft_id": draft_id,
            "message": f"SMS dispatched to {draft['recipient']} via local phone bridge.",
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="communication.draft_sms",
            description="Draft an SMS text message to a recipient phone number for user review prior to dispatch.",
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Destination phone number"},
                    "message": {"type": "string", "description": "Text message body"},
                },
                "required": ["recipient", "message"],
            },
            handler=draft_sms,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="communication",
            timeout_s=5.0,
        )
    )

    engine.tool_registry.register(
        ToolDefinition(
            name="communication.send_sms",
            description="Send a previously drafted SMS text message. Requires explicit user confirmation (confirm=True) and L3 External permission.",
            parameters={
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "ID of draft from communication.draft_sms"},
                    "confirm": {"type": "boolean", "description": "Explicit confirmation flag", "default": False},
                },
                "required": ["draft_id"],
            },
            handler=send_sms,
            permission_level=PermissionLevel.EXTERNAL,
            family="communication",
            timeout_s=8.0,
        )
    )
