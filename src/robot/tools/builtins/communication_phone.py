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
import secrets
import sqlite3
import time
import urllib.error
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

_TRUTHY = {"1", "true", "yes", "on"}
_SMS_DRAFT_TTL_SECONDS = 600.0

# Token-like substrings are scrubbed from logs and user-visible errors.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{25,}\b"),  # Telegram bot tokens
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)\b"
        r"\s*[:=]\s*([^\s,;]{4,})"
    ),
)


def _scrub_secrets(text: str) -> str:
    """Redact token-like substrings from logs and user-visible error text."""
    redacted = text or ""
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", redacted)
        elif pattern.groups == 1:
            redacted = pattern.sub(lambda m: f"{m.group(1)} [REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _env_truthy(name: str) -> bool:
    """Return True when an environment flag is explicitly enabled."""
    return (os.getenv(name, "") or "").strip().lower() in _TRUTHY


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
    """NTFY provider that fails closed without an explicit private topic and token."""

    def __init__(
        self,
        topic: str | None = None,
        base_url: str = "https://ntfy.sh",
        auth_token: str | None = None,
    ) -> None:
        """Initialize NTFY provider from explicit arguments or environment.

        There is deliberately no public default topic.  Both a non-empty topic
        and an auth token must be configured (constructor or ``NTFY_TOPIC`` /
        ``NTFY_AUTH_TOKEN``) before any egress is attempted.

        Args:
            topic: NTFY topic name (reads from NTFY_TOPIC env var if None).
            base_url: NTFY server base URL (default: https://ntfy.sh).
            auth_token: NTFY access token (reads from NTFY_AUTH_TOKEN env var if None).
        """
        self.topic = (topic or os.getenv("NTFY_TOPIC", "") or "").strip()
        self.base_url = base_url.rstrip("/")
        self.auth_token = (auth_token or os.getenv("NTFY_AUTH_TOKEN", "") or "").strip()

    async def send_alert(
        self,
        title: str,
        message: str,
        urgency: str = "normal",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send a push notification via NTFY, failing closed when unconfigured."""
        if not self.topic or not self.auth_token:
            logger.warning("ntfy_not_configured", has_topic=bool(self.topic), has_token=bool(self.auth_token))
            return {
                "status": "not_sent",
                "provider": "ntfy",
                "error": (
                    "NTFY egress is not configured. Set NTFY_TOPIC and NTFY_AUTH_TOKEN, "
                    "or pass an explicit topic and auth_token. Refusing the public default topic."
                ),
            }

        clean_msg = message[:4096]
        url = f"{self.base_url}/{urllib.parse.quote(self.topic, safe='')}"

        # Map urgency to NTFY priority levels (1=min, 3=default, 5=urgent)
        prio_map = {"low": "2", "normal": "3", "high": "4", "urgent": "5"}
        priority = prio_map.get(urgency.lower(), "3")

        headers = {
            "Title": title,
            "Priority": priority,
            "User-Agent": "RobotCore/2.0",
            "Authorization": f"Bearer {self.auth_token}",
        }
        if tags:
            headers["Tags"] = ",".join(tags)

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
            except urllib.error.HTTPError as exc:
                logger.error(
                    "ntfy_dispatch_failed",
                    host=urllib.parse.urlsplit(url).hostname,
                    status_code=exc.code,
                )
                return {
                    "status": "error",
                    "provider": "ntfy",
                    "status_code": exc.code,
                    "error": f"NTFY rejected the notification (HTTP {exc.code}).",
                }
            except Exception as exc:
                logger.error(
                    "ntfy_dispatch_failed",
                    host=urllib.parse.urlsplit(url).hostname,
                    error_type=type(exc).__name__,
                )
                return {
                    "status": "error",
                    "provider": "ntfy",
                    "error": _scrub_secrets(f"NTFY dispatch failed: {exc}"),
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
            except urllib.error.HTTPError as exc:
                # Never log the request URL: it contains the bot token.
                logger.error(
                    "telegram_dispatch_failed",
                    method="POST",
                    host="api.telegram.org",
                    status_code=exc.code,
                )
                return {
                    "status": "error",
                    "provider": "telegram",
                    "status_code": exc.code,
                    "error": f"Telegram rejected the message (HTTP {exc.code}).",
                }
            except Exception as exc:
                logger.error(
                    "telegram_dispatch_failed",
                    method="POST",
                    host="api.telegram.org",
                    error_type=type(exc).__name__,
                )
                return {
                    "status": "error",
                    "provider": "telegram",
                    "error": _scrub_secrets(f"Telegram dispatch failed: {exc}"),
                }

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
_SMS_APPROVED_DRAFTS: dict[str, float] = {}
#: Hard cap on concurrently held drafts. Each entry retains message plaintext,
#: so a chatty model must not be able to accumulate them indefinitely.
_MAX_PENDING_SMS_DRAFTS = 64


def _prune_sms_draft_state() -> None:
    """Drop expired drafts and approvals, then trim to the pending-draft cap.

    Drafts are otherwise only removed when the same ``draft_id`` is presented
    again, so abandoned drafts — and the message text they hold — would persist
    for the life of the process.
    """
    now = time.time()
    expired = [
        draft_id
        for draft_id, draft in _PENDING_SMS_DRAFTS.items()
        if now - float(draft.get("created_at", 0.0)) > _SMS_DRAFT_TTL_SECONDS
    ]
    for draft_id in expired:
        _PENDING_SMS_DRAFTS.pop(draft_id, None)
        _SMS_APPROVED_DRAFTS.pop(draft_id, None)
    for draft_id, approved_at in list(_SMS_APPROVED_DRAFTS.items()):
        if now - approved_at > _SMS_DRAFT_TTL_SECONDS or draft_id not in _PENDING_SMS_DRAFTS:
            _SMS_APPROVED_DRAFTS.pop(draft_id, None)
    overflow = len(_PENDING_SMS_DRAFTS) - _MAX_PENDING_SMS_DRAFTS
    if overflow > 0:
        oldest = sorted(
            _PENDING_SMS_DRAFTS,
            key=lambda d: float(_PENDING_SMS_DRAFTS[d].get("created_at", 0.0)),
        )[:overflow]
        for draft_id in oldest:
            _PENDING_SMS_DRAFTS.pop(draft_id, None)
            _SMS_APPROVED_DRAFTS.pop(draft_id, None)


def issue_sms_confirmation_token(draft_id: str) -> str | None:
    """Issue an out-of-band SMS confirmation token for a pending draft.

    This function is intended for the local UI / API layer (the user's trusted
    channel), not for the LLM.  Tool handlers never return the token in their
    model-visible result: the model cannot read it and therefore cannot
    self-confirm a drafted message.

    Args:
        draft_id: Draft identifier returned by ``communication.draft_sms``.

    Returns:
        A one-time token, or ``None`` when the draft does not exist.
    """
    draft = _PENDING_SMS_DRAFTS.get(draft_id)
    if not draft:
        return None
    token = draft.get("confirmation_token") or secrets.token_urlsafe(24)
    draft["confirmation_token"] = token
    return token


def approve_sms_draft(draft_id: str, token: str) -> bool:
    """Record an out-of-band approval for a pending SMS draft.

    Args:
        draft_id: Draft identifier.
        token: Token previously issued by :func:`issue_sms_confirmation_token`.

    Returns:
        True when the token matches and the approval was recorded.
    """
    draft = _PENDING_SMS_DRAFTS.get(draft_id)
    if not draft:
        return False
    expected = draft.get("confirmation_token") or ""
    if not expected or not secrets.compare_digest(str(token), str(expected)):
        return False
    _SMS_APPROVED_DRAFTS[draft_id] = time.time()
    return True


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
            # Push notifications are external egress and must not auto-run at
            # the default HARMLESS_LOCAL ceiling.
            permission_level=PermissionLevel.EXTERNAL,
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

        _prune_sms_draft_state()
        draft_id = f"sms_{int(time.time())}_{secrets.token_hex(4)}"
        _PENDING_SMS_DRAFTS[draft_id] = {
            "draft_id": draft_id,
            "recipient": clean_rec,
            "message": clean_msg,
            "created_at": time.time(),
            "confirmation_token": secrets.token_urlsafe(24),
        }
        _SMS_LOGGER.log_draft(draft_id, clean_rec, clean_msg)

        return {
            "status": "drafted",
            "draft_id": draft_id,
            "recipient_preview": clean_rec[:4] + "****" + clean_rec[-2:] if len(clean_rec) > 6 else clean_rec,
            "message_length": len(clean_msg),
            "preview": clean_msg,
            "confirmation_required": True,
            "instruction": (
                "Show this draft to the user. Dispatch requires out-of-band user approval "
                "(confirm=True plus a confirmation_token issued by the local UI); the model "
                "cannot approve its own draft."
            ),
        }

    async def send_sms(
        draft_id: str,
        confirm: bool = False,
        confirmation_token: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dispatch a previously drafted SMS only after out-of-band approval.

        ``confirm=True`` from the model is not sufficient.  The call must also
        present a confirmation token that was issued to the trusted UI/user via
        :func:`issue_sms_confirmation_token` (or an operator-configured
        ``ROBOT_SMS_CONFIRMATION_TOKEN``), or the operator must explicitly set
        ``ROBOT_SMS_ALLOW_UNVERIFIED_CONFIRM=1``.  Without a provider 2xx the
        result is ``not_sent``/``error`` — never ``sent``.

        Args:
            draft_id: The identifier returned by communication.draft_sms.
            confirm: Explicit confirmation flag (necessary but not sufficient).
            confirmation_token: Out-of-band token supplied by the UI/user.

        Returns:
            Dictionary containing dispatch outcome.
        """
        if not confirm:
            return {
                "status": "confirmation_required",
                "draft_id": draft_id,
                "message": "Explicit confirmation required before sending SMS.",
            }

        draft = _PENDING_SMS_DRAFTS.get(draft_id)
        if not draft:
            return {
                "status": "error",
                "error": f"Draft '{draft_id}' not found or already dispatched.",
            }
        if time.time() - float(draft.get("created_at", 0.0)) > _SMS_DRAFT_TTL_SECONDS:
            _PENDING_SMS_DRAFTS.pop(draft_id, None)
            _SMS_APPROVED_DRAFTS.pop(draft_id, None)
            return {"status": "not_sent", "draft_id": draft_id, "reason": "Draft expired."}

        provided_token = str(
            confirmation_token or kwargs.get("confirmation_token") or kwargs.get("token") or ""
        ).strip()
        env_token = (os.getenv("ROBOT_SMS_CONFIRMATION_TOKEN", "") or "").strip()

        approved = False
        approval_reason = ""
        expected = str(draft.get("confirmation_token") or "")
        if provided_token and expected and secrets.compare_digest(provided_token, expected):
            approved = True
            approval_reason = "out-of-band token"
        elif provided_token and env_token and secrets.compare_digest(provided_token, env_token):
            approved = True
            approval_reason = "operator-configured token"
        elif _SMS_APPROVED_DRAFTS.get(draft_id) and time.time() - _SMS_APPROVED_DRAFTS[draft_id] <= _SMS_DRAFT_TTL_SECONDS:
            approved = True
            approval_reason = "recorded UI approval"
        elif _env_truthy("ROBOT_SMS_ALLOW_UNVERIFIED_CONFIRM"):
            # Explicit machine-level opt-in set by the operator; the model
            # cannot modify the process environment.
            approved = True
            approval_reason = "operator opt-in (ROBOT_SMS_ALLOW_UNVERIFIED_CONFIRM)"

        if not approved:
            logger.warning("sms_confirmation_rejected", draft_id=draft_id)
            return {
                "status": "not_sent",
                "draft_id": draft_id,
                "reason": (
                    "Out-of-band user confirmation is required. The model cannot "
                    "self-confirm an SMS draft."
                ),
            }

        # Check Twilio credentials; no other provider is currently configured.
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")

        if not (account_sid and auth_token and from_number):
            logger.warning("sms_provider_not_configured", approval=approval_reason)
            return {
                "status": "not_sent",
                "draft_id": draft_id,
                "gateway": "none",
                "reason": (
                    "No SMS provider is configured (TWILIO_ACCOUNT_SID, "
                    "TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER). Nothing was dispatched."
                ),
            }

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

        def _send() -> tuple[bool, dict[str, Any], int | None]:
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    status_code = int(getattr(resp, "status", None) or resp.getcode() or 0)
                    if not (200 <= status_code < 300):
                        return False, {"error": f"Provider returned HTTP {status_code}."}, status_code
                    raw = resp.read().decode("utf-8", errors="replace")
                    parsed = json.loads(raw)
                    return True, parsed, status_code
            except urllib.error.HTTPError as exc:
                return False, {"error": f"Twilio rejected the message (HTTP {exc.code})."}, exc.code
            except Exception as exc:
                return False, {"error": _scrub_secrets(f"Twilio dispatch failed: {exc}")}, None

        ok, payload, status_code = await asyncio.to_thread(_send)
        if ok:
            _PENDING_SMS_DRAFTS.pop(draft_id, None)
            _SMS_APPROVED_DRAFTS.pop(draft_id, None)
            _SMS_LOGGER.mark_dispatched(draft_id, "sent")
            return {
                "status": "sent",
                "gateway": "twilio",
                "sid": payload.get("sid"),
                "status_code": status_code,
            }

        logger.error("twilio_sms_failed", status_code=status_code, approval=approval_reason)
        _SMS_LOGGER.mark_dispatched(draft_id, "failed")
        return {
            "status": "error",
            "gateway": "twilio",
            "draft_id": draft_id,
            "status_code": status_code,
            "error": payload.get("error", "SMS provider rejected the message."),
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
            description="Send a previously drafted SMS text message. Requires explicit user confirmation and an out-of-band confirmation token supplied by the UI; the model cannot self-confirm.",
            parameters={
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "ID of draft from communication.draft_sms"},
                    "confirm": {"type": "boolean", "description": "Explicit confirmation flag", "default": False},
                    "confirmation_token": {"type": "string", "description": "Out-of-band token issued to the local UI/user", "default": ""},
                },
                "required": ["draft_id"],
            },
            handler=send_sms,
            permission_level=PermissionLevel.EXTERNAL,
            family="communication",
            timeout_s=8.0,
        )
    )
