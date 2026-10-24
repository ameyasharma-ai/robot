"""Production hardening for the HTTP surface.

Everything here is transport-level (pure ASGI) so it applies uniformly to every
route — including 401s produced by the auth middleware and static file
responses — without each endpoint having to remember to set it:

* a request id (``X-Request-ID``) that is echoed to the client and attached to
  the ASGI state for error reporting,
* a hard request-body cap (``413``) enforced both from ``Content-Length`` and
  while streaming, so a chunked upload cannot exhaust memory,
* security headers (CSP, nosniff, frame denial, referrer policy, permissions
  policy, HSTS on TLS) and ``Cache-Control: no-store`` for API responses,
* a bounded, per-client failure throttle for the login endpoint.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import deque
from typing import Any, Callable

from robot.telemetry import get_logger

logger = get_logger("api.production")

#: Default maximum request body (32 MiB). Large enough for a multi-minute WAV
#: dictation upload or an image-generation reference, small enough that a
#: malicious local process cannot make the API allocate unbounded memory.
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024

#: Headers every response carries. CSP still allows the inline bootstrap script
#: and inline styles used by the console, but blocks third-party script/frame
#: origins; Google Fonts is the only external style/font origin the UI loads.
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (
        b"permissions-policy",
        b"camera=(), microphone=(self), geolocation=(), payment=(), usb=(), serial=(), hid=()",
    ),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (
        b"content-security-policy",
        (
            b"default-src 'self'; "
            b"script-src 'self' 'unsafe-inline'; "
            b"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            b"font-src 'self' data: https://fonts.gstatic.com; "
            b"img-src 'self' data: blob:; "
            b"media-src 'self' data: blob:; "
            b"connect-src 'self' ws: wss: blob: data:; "
            b"worker-src 'self' blob:; "
            b"object-src 'none'; base-uri 'self'; form-action 'self'; "
            b"frame-ancestors 'none'"
        ),
    ),
)

_RID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class _BodyTooLarge(Exception):
    """Raised from the wrapped ``receive`` when a chunked body exceeds the cap."""


def _headers(scope: dict[str, Any]) -> dict[str, str]:
    """Decode ASGI headers into a lower-cased mapping.

    Args:
        scope: ASGI connection scope.

    Returns:
        dict[str, str]: Header name to value.
    """
    result: dict[str, str] = {}
    for raw_key, raw_value in scope.get("headers") or []:
        try:
            result[raw_key.decode("latin-1").lower()] = raw_value.decode("latin-1")
        except Exception:
            continue
    return result


async def _send_json(send: Callable, status: int, payload: dict[str, Any], request_id: str) -> None:
    """Send a JSON error response, using the two required ASGI messages."""
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("latin-1")),
        (b"cache-control", b"no-store"),
        (b"x-request-id", request_id.encode("latin-1")),
    ]
    headers.extend((k, v) for k, v in SECURITY_HEADERS)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class ProductionMiddleware:
    """Add request ids, body limits and security headers to HTTP traffic."""

    def __init__(self, app: Any, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        """
        Args:
            app: Inner ASGI application.
            max_body_bytes: Maximum accepted request body size in bytes.
        """
        self.app = app
        self.max_body_bytes = max(1024, int(max_body_bytes))

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        """Process one ASGI connection."""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        incoming = _headers(scope).get("x-request-id", "")
        request_id = incoming if _RID_RE.match(incoming or "") else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        content_length = _headers(scope).get("content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                await _send_json(
                    send,
                    400,
                    {"detail": "Invalid Content-Length header.", "request_id": request_id},
                    request_id,
                )
                return
            if declared < 0 or declared > self.max_body_bytes:
                await _send_json(
                    send,
                    413,
                    {
                        "detail": f"Request body exceeds {self.max_body_bytes} bytes.",
                        "request_id": request_id,
                    },
                    request_id,
                )
                return

        started = False
        path = scope.get("path", "") or ""

        async def _send(message: dict[str, Any]) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
                headers = list(message.get("headers") or [])
                present = {key.lower() for key, _ in headers}
                if b"x-request-id" not in present:
                    headers.append((b"x-request-id", request_id.encode("latin-1")))
                for key, value in SECURITY_HEADERS:
                    if key not in present:
                        headers.append((key, value))
                # API and HTML responses must never be cached; static assets
                # carry their own long-lived/validator policy.
                if (
                    b"cache-control" not in present
                    and not path.startswith(("/static/", "/data/images/"))
                ):
                    headers.append((b"cache-control", b"no-store"))
                if scope.get("scheme") == "https" and b"strict-transport-security" not in present:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message = {**message, "headers": headers}
            await send(message)

        received = 0

        async def _receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, _receive, _send)
        except _BodyTooLarge:
            if not started:
                await _send_json(
                    send,
                    413,
                    {
                        "detail": f"Request body exceeds {self.max_body_bytes} bytes.",
                        "request_id": request_id,
                    },
                    request_id,
                )
            else:
                logger.warning("request_body_limit_exceeded_after_response", path=path)


class FailureThrottle:
    """Sliding-window failure counter with bounded memory.

    Used for ``POST /auth/login`` so a local script (or a compromised browser
    tab) cannot brute-force the API token. Only failures are counted; a success
    clears the client's history.
    """

    def __init__(self, max_failures: int = 5, window_s: float = 60.0, max_clients: int = 4096) -> None:
        """
        Args:
            max_failures: Failures allowed inside the window.
            window_s: Sliding window length in seconds.
            max_clients: Upper bound of tracked client keys.
        """
        self.max_failures = max(1, int(max_failures))
        self.window_s = max(1.0, float(window_s))
        self.max_clients = max(16, int(max_clients))
        self._failures: dict[str, deque[float]] = {}

    def _prune(self, key: str, now: float) -> deque[float]:
        """Drop expired timestamps for ``key``."""
        bucket = self._failures.get(key)
        if bucket is None:
            bucket = deque()
            self._failures[key] = bucket
        cutoff = now - self.window_s
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return bucket

    def retry_after(self, key: str) -> int:
        """Return seconds until the client may retry, or ``0`` when allowed."""
        now = time.monotonic()
        bucket = self._prune(key, now)
        if len(bucket) < self.max_failures:
            return 0
        return max(1, int(round(self.window_s - (now - bucket[0]))) + 1)

    def record_failure(self, key: str) -> None:
        """Record one failed authentication attempt."""
        now = time.monotonic()
        bucket = self._prune(key, now)
        bucket.append(now)
        if len(self._failures) > self.max_clients:
            # Evict the client with the oldest activity; bounded memory beats a
            # perfect ledger for a local single-owner service.
            oldest = min(
                self._failures.items(),
                key=lambda item: item[1][-1] if item[1] else 0.0,
            )[0]
            self._failures.pop(oldest, None)

    def reset(self, key: str) -> None:
        """Forget a client's failure history after a successful login."""
        self._failures.pop(key, None)
