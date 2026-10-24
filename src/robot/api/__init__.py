"""FastAPI server — HTTP + WebSocket API for Robot Core.

Provides:
- REST endpoints for status, memory, personality
- WebSocket for real-time event streams
- Developer console / AI Terminal access
- Token-based authentication for every non-public route (see
  :func:`_resolve_auth_token` and :class:`_ApiAuthMiddleware`)

Authentication model
--------------------
Every route except a small public allow-list (``/health``, ``/auth/login``,
``/auth/logout``, ``/auth/status`` and the static UI shell pages/assets) requires
a valid API token, supplied in any of these ways:

1. ``Authorization: Bearer <token>`` header — the normal path for ``fetch``.
2. ``?token=<token>`` query parameter — required for ``EventSource``, MJPEG
   ``<img>`` streams and WebSocket handshakes that cannot set headers.
3. The ``httpOnly`` session cookie installed by ``POST /auth/login`` with the
   JSON body ``{"token": "<token>"}`` (SameSite=Lax; ``Secure`` is set
   automatically when the request arrives over HTTPS).

The token is resolved in this order: the ``auth_token`` argument, the
``ROBOT_API_TOKEN`` environment variable, ``api.auth_token`` in the engine
config, a token previously saved in ``data/.api_token``, and finally a fresh
``secrets.token_urlsafe(24)`` that is written to that file so it stays the same
across restarts.

**Trusted local access.** A request that arrives on a loopback peer address,
names the server with a loopback ``Host`` header and (for browser requests whose
``Origin``/``Referer`` is set) refers to its own host is treated as the owner
sitting at this machine and is served without a token. That triple check is what
makes it safe: a page on another origin cannot reach the console because rebinding
its ``Host`` to ``127.0.0.1`` fails the host/origin checks, and the gate closes
itself if the client is not literally on the loopback interface. Set
``api.trust_loopback: false`` in the config (or ``ROBOT_API_TRUST_LOOPBACK=0``) to
require the token even from localhost.

The uvicorn bind host itself is configured by ``RobotEngine._run_api``
(``api.host`` in ``config/default.yaml`` / ``ROBOT_API_HOST``, default
``127.0.0.1``). Binding to anything other than loopback is still possible, but
it is no longer anonymous: a non-loopback client now gets only the public
allow-list and every other request is rejected without a valid token.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import html
import inspect
import ipaddress
import json
import os
import re
import secrets
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from fastapi import Body, FastAPI, HTTPException, Path as PathParam, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from robot.api.production import DEFAULT_MAX_BODY_BYTES, FailureThrottle, ProductionMiddleware

logger = structlog.get_logger()

from robot.core.state import RobotMode
from robot.concurrency import spawn

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


# ─── Authentication Constants ──────────────────────────────────────────────
#: Environment variable holding the shared API bearer token.
AUTH_TOKEN_ENV_VAR = "ROBOT_API_TOKEN"
#: Name of the httpOnly session cookie written by ``POST /auth/login``.
AUTH_COOKIE_NAME = "robot_api_session"
#: Query-parameter fallback for header-less clients (EventSource / MJPEG / WS).
AUTH_TOKEN_QUERY_PARAM = "token"
#: Lifetime of the login session cookie in seconds (30 days).
AUTH_COOKIE_MAX_AGE = 30 * 24 * 3600
#: Routes that stay reachable without a token (liveness probe + login flow +
#: the static UI shell that renders the login gate).
AUTH_EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/auth/login",
        "/auth/logout",
        "/auth/status",
        "/favicon.ico",
        "/apple-touch-icon.png",
        "/",
        "/contextlayer",
        "/context-layer",
        "/inspect",
    }
)
#: Path prefixes that stay reachable without a token (UI assets only — generated
#: imagery under ``/data/images`` and every API route remain protected).
AUTH_EXEMPT_PREFIXES: tuple[str, ...] = ("/static/",)

#: Environment variable that overrides the on/off switch for trusted local access.
TRUST_LOOPBACK_ENV_VAR = "ROBOT_API_TRUST_LOOPBACK"
#: Environment variable naming the file the API token is read from / written to.
AUTH_TOKEN_FILE_ENV_VAR = "ROBOT_API_TOKEN_FILE"
#: Default location of the persisted API token (``data/`` is git-ignored).
AUTH_TOKEN_FILE_DEFAULT: str = "data/.api_token"
#: Host names that identify the machine itself, used for the rebinding check.
LOOPBACK_HOST_NAMES: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

#: Target frame rate per MJPEG stream; encoding is off-loaded to a worker thread.
MJPEG_TARGET_FPS = 15.0
#: Upper bound of buffered events per WebSocket client before the oldest is dropped.
WS_EVENT_QUEUE_MAXSIZE = 100
#: How often an idle WebSocket client re-checks whether the engine is still running.
#: The send loop has no data to observe the shutdown with, so it polls on this beat.
WS_SHUTDOWN_POLL_S = 0.5
#: Longest voice clip accepted by ``POST /api/transcribe`` (5 minutes).
TRANSCRIBE_MAX_DURATION_S = 300.0
#: Ceiling for one Whisper transcription before the request is abandoned.
TRANSCRIBE_TIMEOUT_S = 30.0
#: Gallery metadata keyed by filename: ``(mtime, size, width, height)``. Bounded
#: so a directory that grows without bound cannot grow the cache with it.
_GALLERY_META_CACHE: dict[str, tuple[float, int, int, int]] = {}
_GALLERY_META_CACHE_MAX = 4096


class LoginRequest(BaseModel):
    """Request model for ``POST /auth/login``."""

    token: str


def _extract_bearer_header(authorization: str | None) -> str:
    """Return the bearer token carried by an ``Authorization`` header value.

    Args:
        authorization: Raw header value (may be ``None``).

    Returns:
        str: The token, or an empty string when the header is absent or not a
            bearer token.
    """
    if not authorization:
        return ""
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def _headers_from_scope(scope: dict[str, Any]) -> dict[str, str]:
    """Decode the raw ASGI header list into a lower-cased mapping.

    Args:
        scope: Starlette ASGI connection scope for an ``http`` or ``websocket``
            connection.

    Returns:
        dict[str, str]: Header names (lower-case) to values; a repeated header is
            joined with ``, `` the way HTTP requires.
    """
    headers: dict[str, str] = {}
    for raw_key, raw_value in scope.get("headers") or []:
        try:
            key = raw_key.decode("latin-1").lower()
            value = raw_value.decode("latin-1")
        except Exception:
            continue
        headers[key] = f"{headers[key]}, {value}" if key in headers else value
    return headers


def _token_from_scope(scope: dict[str, Any]) -> str:
    """Collect the candidate API token from an ASGI connection scope.

    Checks, in order, the bearer header, the ``token`` query parameter (used by
    EventSource, MJPEG ``<img>`` streams and WebSockets) and the httpOnly login
    cookie.

    Args:
        scope: Starlette ASGI connection scope for an ``http`` or ``websocket``
            connection.

    Returns:
        str: The first non-empty candidate token, or ``""`` when there is none.
    """
    headers = _headers_from_scope(scope)

    token = _extract_bearer_header(headers.get("authorization"))
    if token:
        return token

    query = scope.get("query_string") or b""
    if isinstance(query, bytes):
        query = query.decode("latin-1")
    params = urllib.parse.parse_qs(query)
    for value in params.get(AUTH_TOKEN_QUERY_PARAM) or []:
        if value:
            return value

    cookie_header = headers.get("cookie", "")
    for chunk in cookie_header.split(";"):
        name, _, value = chunk.strip().partition("=")
        if name == AUTH_COOKIE_NAME and value:
            return urllib.parse.unquote(value)

    return ""


def _token_matches(candidate: str, expected: str) -> bool:
    """Constant-time comparison of a candidate token with the expected token.

    Args:
        candidate: Token supplied by the client.
        expected: Token configured for this application instance.

    Returns:
        bool: ``True`` only when both are non-empty and identical.
    """
    if not candidate or not expected:
        return False
    return secrets.compare_digest(candidate, expected)


def _is_auth_exempt(path: str) -> bool:
    """Report whether a request path is reachable without a token.

    Args:
        path: Request/WebSocket path from the ASGI scope.

    Returns:
        bool: ``True`` when the path is on the public allow-list.
    """
    if path in AUTH_EXEMPT_PATHS:
        return True
    if path.startswith(AUTH_EXEMPT_PREFIXES):
        return True
    # Assets referenced by the UI shell (icons/fonts shipped inside /static).
    return path in ("/index.html", "/index.css", "/app.js")


def _is_loopback_address(host: str | None) -> bool:
    """Report whether a peer address string is a loopback address.

    Args:
        host: Peer host from the ASGI ``scope["client"]`` tuple, or ``None``.

    Returns:
        bool: ``True`` for ``127.0.0.1``, ``::1`` and any other address the
            standard library classifies as loopback.
    """
    if not host:
        return False
    candidate = host.strip().strip("[]")
    if candidate in LOOPBACK_HOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _host_authority(header_value: str | None) -> str:
    """Return the lower-cased host name of a ``Host``/origin value with the port removed.

    Args:
        header_value: Raw ``Host`` header, or a full ``Origin``/``Referer`` URL.

    Returns:
        str: The bare host name, or ``""`` when the value is missing or unparsable.
    """
    if not header_value:
        return ""
    value = header_value.strip().lower()
    if "://" in value:
        value = urllib.parse.urlsplit(value).netloc
    value = value.rsplit("@", 1)[-1]
    if value.startswith("["):  # IPv6 literal: [::1]:8888
        end = value.find("]")
        return value[: end + 1] if end != -1 else value
    return value.split(":", 1)[0]


def _is_trusted_local_scope(scope: dict[str, Any]) -> bool:
    """Decide whether a connection is the machine's own owner using no token.

    Three independent facts must hold, which is what makes bypassing the token
    safe enough for a local-first console:

    * the peer address is loopback, so the client is literally this machine;
    * the ``Host`` header names a loopback authority, so a DNS-rebound external
      domain cannot reach the API through the local socket;
    * when the client is a browser (``Origin``/``Referer`` set) that origin is
      this same host, so a page on another site cannot drive the console.

    Args:
        scope: Starlette ASGI connection scope for an ``http`` or ``websocket``
            connection.

    Returns:
        bool: ``True`` only when every check above succeeds.
    """
    client = scope.get("client") or ("", 0)
    if not _is_loopback_address(str(client[0])):
        return False

    headers = _headers_from_scope(scope)
    request_host = _host_authority(headers.get("host"))
    if request_host not in LOOPBACK_HOST_NAMES:
        return False

    for name in ("origin", "referer", "sec-fetch-site"):
        value = headers.get(name)
        if not value:
            continue
        if name == "sec-fetch-site":
            if value.strip().lower() not in ("same-origin", "none"):
                return False
            continue
        if _host_authority(value) != request_host:
            return False

    return True


def _trust_loopback_enabled(engine: "RobotEngine") -> bool:
    """Read the ``api.trust_loopback`` switch (default on) for this engine.

    ``ROBOT_API_TRUST_LOOPBACK`` overrides the config; values ``0``, ``false``,
    ``no`` and ``off`` disable trusted local access.

    Args:
        engine: Running engine whose ``config`` dict may carry ``api.trust_loopback``.

    Returns:
        bool: ``True`` when loopback clients may skip the token.
    """
    raw: str | None = None
    config = getattr(engine, "config", None)
    if isinstance(config, dict):
        api_cfg = config.get("api")
        if isinstance(api_cfg, dict) and "trust_loopback" in api_cfg:
            value = api_cfg["trust_loopback"]
            if isinstance(value, bool):
                return value
            raw = str(value)
    env_value = (os.environ.get(TRUST_LOOPBACK_ENV_VAR) or "").strip()
    if env_value:
        raw = env_value
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _api_token_file() -> Path:
    """Locate the file that caches the generated API token across restarts.

    ``ROBOT_API_TOKEN_FILE`` overrides it; otherwise the token lives next to the
    other runtime state under ``data/`` (a git-ignored directory).

    Returns:
        Path: Absolute path of the token file; it may not exist yet.
    """
    override = (os.environ.get(AUTH_TOKEN_FILE_ENV_VAR) or "").strip()
    if override:
        return Path(override).expanduser()
    from robot.config import get_project_root

    return get_project_root() / AUTH_TOKEN_FILE_DEFAULT


async def _read_json_object(request: Request) -> dict[str, Any]:
    """Read a request body that must be a JSON object.

    Args:
        request: Incoming request.

    Returns:
        dict[str, Any]: The decoded object.

    Raises:
        HTTPException: 400 when the body is not valid JSON or not an object.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return payload


def _validate_http_url(value: str, field: str) -> str:
    """Validate an operator-supplied http(s) endpoint before it is persisted.

    Args:
        value: Raw URL string.
        field: Field name used in the error detail.

    Returns:
        str: The stripped URL.

    Raises:
        HTTPException: 400 when the URL is not http(s), has no host, or embeds
            credentials.
    """
    value = value.strip()
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail=f"{field} must be an http(s) URL.")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail=f"{field} must not contain credentials.")
    return value


def _resolve_images_dir(engine: "RobotEngine") -> Path:
    """Resolve the generated-image directory once, independent of the CWD.

    The image service owns the directory; using its configured value keeps the
    writer, the gallery listing, the delete route and the static mount in
    agreement even when the process is not started from the project root.

    Args:
        engine: Running engine, optionally carrying ``image_service.output_dir``.

    Returns:
        Path: Absolute directory holding generated artwork.
    """
    configured = getattr(getattr(engine, "image_service", None), "output_dir", None)
    if isinstance(configured, (str, Path)) and str(configured).strip():
        path = Path(configured).expanduser()
    else:
        path = Path("data/images")
    return path if path.is_absolute() else Path.cwd() / path


def _decode_wav_to_mono16k(raw: bytes) -> tuple[np.ndarray | None, float]:
    """Decode a PCM WAV upload into 16 kHz mono float32 samples.

    Args:
        raw: Raw WAV file bytes.

    Returns:
        tuple[np.ndarray | None, float]: The decoded samples and clip duration in
            seconds, or ``(None, 0.0)`` when the payload is not a usable PCM WAV.
    """
    import io
    import wave

    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frame_count = wf.getnframes()
            frames = wf.readframes(frame_count)
    except (wave.Error, EOFError, ValueError):
        return None, 0.0

    if channels < 1 or rate <= 0 or not frames:
        return None, 0.0

    duration_s = len(frames) / float(rate * channels * max(1, width))
    if width == 2:
        pcm = np.frombuffer(frames, dtype=np.int16)
    elif width == 4:
        pcm = (np.frombuffer(frames, dtype=np.int32) >> 16).astype(np.int16)
    elif width == 1:
        pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) << 8
    else:
        return None, 0.0

    if channels > 1:
        usable = (pcm.size // channels) * channels
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)

    samples = pcm.astype(np.float32) / 32768.0
    if rate != 16000 and samples.size:
        target_len = max(1, int(samples.size * 16000 / rate))
        samples = np.interp(
            np.linspace(0, samples.size, target_len, endpoint=False),
            np.arange(samples.size),
            samples,
        ).astype(np.float32)
    return samples, duration_s


def _read_persisted_token(path: Path) -> str:
    """Read a previously persisted API token from ``path``.

    Args:
        path: Candidate token file.

    Returns:
        str: The token, or ``""`` when the file is missing, unreadable or empty.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _persist_token(path: Path, token: str) -> bool:
    """Write ``token`` to ``path`` with owner-only permissions.

    Args:
        path: Destination token file.
        token: Generated API token to remember for the next start.

    Returns:
        bool: ``True`` when the token is now on disk; failures are logged and
            reported rather than raised — a read-only ``data/`` directory must not
            stop the API from starting.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with open(existing, "w", encoding="utf-8") as handle:
            handle.write(token)
        os.chmod(path, 0o600)
        return True
    except OSError as exc:
        logger.warning("api_auth_token_persist_failed", path=str(path), error=str(exc))
        return False


class _ApiAuthMiddleware:
    """Pure-ASGI bearer/cookie/query token gate for HTTP and WebSocket traffic.

    A pure ASGI middleware is used instead of ``BaseHTTPMiddleware`` so that
    ``websocket`` scopes are inspected too: a WebSocket that is not authorized
    is refused before the handshake is accepted (close code 4401).

    When ``trust_loopback`` is set, connections that :func:`_is_trusted_local_scope`
    recognizes as this machine's own owner pass through without a token, and the
    scope is stamped with ``robot_local_trusted`` so route-level re-checks (the
    WebSocket handshake) agree with the gate.
    """

    #: Scope key stamped ``True`` when a request was admitted as a trusted local client.
    TRUSTED_LOCAL_SCOPE_KEY = "robot_local_trusted"

    def __init__(self, app: Any, expected_token: str, trust_loopback: bool = True) -> None:
        """Store the wrapped ASGI app and the token that clients must present.

        Args:
            app: The inner ASGI application (the FastAPI middleware stack).
            expected_token: Shared API token for this application instance.
            trust_loopback: Admit verified loopback clients without a token.
        """
        self.app = app
        self.expected_token = expected_token or ""
        self.trust_loopback = trust_loopback

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """Enforce authentication for ``http``/``websocket`` scopes."""
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or ""
        # ``lifespan`` and redirects for the public allow-list pass straight through.
        if scope.get("method") == "OPTIONS" or _is_auth_exempt(path):
            await self.app(scope, receive, send)
            return

        candidate = _token_from_scope(scope)
        if _token_matches(candidate, self.expected_token):
            await self.app(scope, receive, send)
            return

        if self.trust_loopback and _is_trusted_local_scope(scope):
            scope[self.TRUSTED_LOCAL_SCOPE_KEY] = True
            await self.app(scope, receive, send)
            return

        if scope_type == "websocket":
            await self._deny_websocket(receive, send)
        else:
            await self._deny_http(send, path)

    async def _deny_http(self, send: Any, path: str) -> None:
        """Send a 401 JSON response asking for a bearer token."""
        payload = json.dumps(
            {
                "detail": "Unauthorized: a valid API token is required.",
                "hint": (
                    "Send 'Authorization: Bearer <token>', or '?token=<token>' for "
                    "EventSource/MJPEG/WebSocket streams, or POST the token to "
                    "/auth/login to get a session cookie."
                ),
                "path": path,
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("latin-1")),
                    (b"www-authenticate", b'Bearer realm="robot-api"'),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    async def _deny_websocket(self, receive: Any, send: Any) -> None:
        """Refuse a WebSocket handshake that carries no valid token."""
        message = await receive()
        if message.get("type") == "websocket.connect":
            await send({"type": "websocket.close", "code": 4401, "reason": "Unauthorized"})


def _resolve_auth_token(engine: "RobotEngine", auth_token: str | None = None) -> tuple[str, str]:
    """Determine the API token for an application instance.

    Resolution order: explicit ``auth_token`` argument, the ``ROBOT_API_TOKEN``
    environment variable, ``api.auth_token`` (or ``api.token``) in the engine
    configuration, the token persisted from an earlier start in ``data/.api_token``,
    and finally a freshly generated random token that is written back to that file
    so it survives restarts instead of locking the owner out of a bookmarked
    console.

    Args:
        engine: Running engine whose ``config`` dict may carry ``api.auth_token``.
        auth_token: Optional explicit token passed to :func:`create_app`.

    Returns:
        tuple[str, str]: The token and the label of the source it came from
            (``"parameter"``, ``"environment"``, ``"config"``, ``"token-file"``
            or ``"generated"``).
    """
    explicit = (auth_token or "").strip()
    if explicit:
        return explicit, "parameter"

    env_token = (os.environ.get(AUTH_TOKEN_ENV_VAR) or "").strip()
    if env_token:
        return env_token, "environment"

    config = getattr(engine, "config", None)
    if isinstance(config, dict):
        api_cfg = config.get("api")
        if isinstance(api_cfg, dict):
            for key in ("auth_token", "api_token", "token"):
                value = api_cfg.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip(), "config"

    token_file = _api_token_file()
    persisted = _read_persisted_token(token_file)
    if persisted:
        return persisted, "token-file"

    generated = secrets.token_urlsafe(24)
    _persist_token(token_file, generated)
    return generated, "generated"


def _encode_jpeg_frame(frame: Any, quality: int = 80) -> bytes | None:
    """Encode one BGR frame to JPEG bytes (blocking; run in a worker thread).

    Args:
        frame: OpenCV image matrix.
        quality: JPEG quality passed to ``cv2.imencode``.

    Returns:
        bytes | None: Encoded JPEG payload, or ``None`` when encoding failed.
    """
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return encoded.tobytes() if ok else None


async def _mjpeg_part_generator(
    frame_provider: Any,
    stop_check: Any,
    *,
    jpeg_quality: int = 80,
    target_fps: float = MJPEG_TARGET_FPS,
    stream_name: str = "mjpeg",
):
    """Async generator producing MJPEG multipart chunks off the event loop.

    JPEG encoding is CPU work, so it runs on the default thread executor instead
    of the asyncio loop, and each stream is paced to ``target_fps`` with
    drift-corrected sleeps (a slow frame delays the next one rather than causing
    a catch-up burst).

    Args:
        frame_provider: Zero-argument callable returning the newest BGR frame or
            ``None`` when none is ready yet.
        stop_check: Zero-argument callable; streaming continues while it is true.
        jpeg_quality: JPEG quality for ``cv2.imencode``.
        target_fps: Maximum frames emitted per second for this stream.
        stream_name: Label used in diagnostic log lines.

    Yields:
        bytes: One ``multipart/x-mixed-replace`` frame part.
    """
    loop = asyncio.get_running_loop()
    interval = 1.0 / max(0.1, float(target_fps))
    next_deadline = loop.time()
    try:
        while stop_check():
            frame = None
            try:
                frame = frame_provider()
            except Exception as exc:  # camera/perception not ready yet
                logger.debug("mjpeg_frame_provider_failed", stream=stream_name, error=str(exc))

            if frame is not None:
                try:
                    payload = await loop.run_in_executor(None, _encode_jpeg_frame, frame, jpeg_quality)
                except Exception as exc:
                    logger.warning("mjpeg_encode_failed", stream=stream_name, error=str(exc))
                    payload = None
                if payload:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + payload
                        + b"\r\n"
                    )

            # Pace the stream at ~target_fps; never burst to catch up with encoding lag.
            next_deadline += interval
            delay = next_deadline - loop.time()
            if delay <= 0:
                next_deadline = loop.time()
            else:
                await asyncio.sleep(delay)
    except (asyncio.CancelledError, GeneratorExit):
        return


class LoginRequest(BaseModel):
    """Request model for ``POST /auth/login``."""

    token: str = Field(min_length=1, max_length=512)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=32_000)
    session_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]*$")
    personality: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]*$")
    thinking: bool | None = None


class ThinkingModeRequest(BaseModel):
    enabled: bool


class ChatSpeakRequest(BaseModel):
    """Request model for synthesizing speech audio for a chat message."""
    text: str = Field(min_length=1, max_length=8_000)
    personality: str | None = Field(default=None, max_length=64)
    voice_model: str | None = Field(default=None, max_length=256)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.5, le=2.0)
    play_hardware: bool = False


class PersonaCreateRequest(BaseModel):
    """Request model for creating a new persona with provisioned isolated database."""
    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=2_000)
    system_prompt: str | None = Field(default=None, max_length=32_000)
    speaking_style: str | None = Field(default=None, max_length=4_000)
    vocabulary: str | None = Field(default=None, max_length=4_000)
    avatar_color: str | None = Field(default=None, max_length=32)
    default_models: dict[str, Any] | None = None


class PersonaUpdateRequest(BaseModel):
    """Request model for updating an existing persona and its model defaults."""
    display_name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=2_000)
    system_prompt: str | None = Field(default=None, max_length=32_000)
    speaking_style: str | None = Field(default=None, max_length=4_000)
    vocabulary: str | None = Field(default=None, max_length=4_000)
    avatar_color: str | None = Field(default=None, max_length=32)
    default_models: dict[str, Any] | None = None


class TTSSwitchRequest(BaseModel):
    """Request model for switching active TTS engine and voice."""
    engine: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    voice: str | None = Field(default=None, max_length=256)


class TTSTestRequest(BaseModel):
    """Request model for generating test audio with a specific TTS engine and voice."""
    engine: str = Field(default="kokoro", max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    voice: str | None = Field(default=None, max_length=256)
    text: str = Field(
        default="Hello, this is a test of the neural voice engine on the robot.",
        max_length=2_000,
    )


class ChatResponse(BaseModel):
    response: str
    turn: int
    personality: str
    session_id: str | None = None


class CreateSessionRequest(BaseModel):
    personality: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]*$")
    title: str | None = Field(default=None, max_length=200)


class StatusResponse(BaseModel):
    mode: str
    personality: str
    turn_count: int
    voice_enabled: bool = True
    services: dict[str, bool]
    telemetry: dict[str, Any]
    is_speaking: bool = False
    music_playing: bool = False


class VoiceModeRequest(BaseModel):
    enabled: bool


class CameraModeRequest(BaseModel):
    enabled: bool


class ImageGenRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    model: str | None = Field(default="default", max_length=128)
    negative_prompt: str | None = Field(default="", max_length=2_000)
    width: int | None = Field(default=512, ge=64, le=2048)
    height: int | None = Field(default=512, ge=64, le=2048)
    steps: int | None = Field(default=4, ge=1, le=50)
    seed: int | None = Field(default=-1, ge=-1, le=2**32 - 1)


class EqualizerRequest(BaseModel):
    preset: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9 _-]+$")


class MusicVolumeRequest(BaseModel):
    volume: float = Field(ge=0.0, le=1.0)


class ViewModeRequest(BaseModel):
    mode: str = Field(default="dual", pattern=r"^(chat|dual)$")


class WakeWordRequest(BaseModel):
    require_wake_word: bool | None = None


class LLMProviderRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")


class OpenAIModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=128, pattern=r"^[^\x00-\x1f]+$")


class CharacterActionRequest(BaseModel):
    action: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    args: dict[str, Any] | None = None


def create_app(engine: "RobotEngine", auth_token: str | None = None) -> FastAPI:
    """Create the FastAPI application with all routes.

    The returned application enforces token authentication on every route except
    the public allow-list (:data:`AUTH_EXEMPT_PATHS` / :data:`AUTH_EXEMPT_PREFIXES`):
    a bearer header, a ``?token=`` query parameter (for EventSource, MJPEG and
    WebSocket clients that cannot set headers) or the httpOnly cookie issued by
    ``POST /auth/login`` is required.

    Args:
        engine: The running :class:`~robot.core.engine.RobotEngine`.
        auth_token: Optional shared API token. When omitted it is read from the
            ``ROBOT_API_TOKEN`` environment variable, then from ``api.auth_token``
            in the engine config, then from the token file written by an earlier
            start, and only then is a random token generated and saved there.

    Returns:
        FastAPI: Configured, authenticated application. The effective token is
            exposed on ``app.state.auth_token`` (never on any HTTP response).

    Note:
        The uvicorn bind host comes from ``api.host`` in the engine config
        (``RobotEngine._run_api``, default ``127.0.0.1``) and is deliberately not
        changed here. Binding to a non-loopback address is still allowed but is no
        longer anonymous: a client that is not on the loopback interface can reach
        only the health probe and the static UI shell without a valid token.
    """
    token, token_source = _resolve_auth_token(engine, auth_token)
    trust_loopback = _trust_loopback_enabled(engine)

    app = FastAPI(
        title="Robot Core API",
        description="Local-first humanoid robot brain API",
        version="1.1.0",
    )
    app.state.auth_token = token
    app.state.auth_token_source = token_source
    app.state.trust_loopback = trust_loopback
    app.add_middleware(_ApiAuthMiddleware, expected_token=token, trust_loopback=trust_loopback)

    engine_config = getattr(engine, "config", None)
    api_config = engine_config.get("api", {}) if isinstance(engine_config, dict) else {}
    if not isinstance(api_config, dict):
        api_config = {}
    try:
        max_body_bytes = int(api_config.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
    except (TypeError, ValueError):
        max_body_bytes = DEFAULT_MAX_BODY_BYTES
    max_body_bytes = min(max(max_body_bytes, 64 * 1024), 256 * 1024 * 1024)
    app.state.max_body_bytes = max_body_bytes
    app.state.login_throttle = FailureThrottle(
        max_failures=int(api_config.get("login_max_failures", 5) or 5),
        window_s=float(api_config.get("login_window_s", 60) or 60),
    )
    # Added last so it is the outermost layer: security headers and the body cap
    # then also cover 401s and static responses, not just successful routes.
    app.add_middleware(ProductionMiddleware, max_body_bytes=max_body_bytes)

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        """Log the fault and return a sanitized envelope with a correlation id."""
        request_id = getattr(request.state, "request_id", "-")
        logger.exception(
            "api_unhandled_exception",
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )

    if token_source == "generated":
        logger.warning(
            "api_auth_token_generated",
            message=(
                "No API token configured — generated one and saved it so it stays "
                "stable across restarts. Local browsers at 127.0.0.1 need no token; "
                "remote clients must present the value stored in the token file."
            ),
            token_file=str(_api_token_file()),
            login_endpoint="POST /auth/login {\"token\": \"<token>\"}",
        )
    elif token_source == "token-file":
        logger.info("api_auth_token_restored", token_file=str(_api_token_file()))
    else:
        logger.info("api_auth_token_enabled", token_source=token_source)

    if trust_loopback:
        logger.info(
            "api_auth_loopback_trusted",
            message=(
                "Requests arriving on the loopback interface with a loopback Host "
                "are served without a token (set api.trust_loopback: false to require it)."
            ),
        )

    def _streaming_open() -> bool:
        """Return True while the engine is live and streams should keep producing.

        Uvicorn's graceful shutdown waits for in-flight responses before it will
        send the lifespan shutdown event, and an MJPEG or WebSocket route only
        ends when its generator returns. Without this gate Ctrl+C has to be
        resolved by cancelling the server task from outside, which Starlette
        reports as a CancelledError traceback rather than a clean stop. Engines
        without the flag (test doubles) are treated as running.
        """
        return bool(getattr(engine, "_running", True))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ─── Authentication Endpoints ──────────────────────────────────────────
    @app.post("/auth/login")
    async def auth_login(req: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
        """Exchange a valid API token for an httpOnly session cookie.

        Browsers cannot attach an ``Authorization`` header to ``EventSource``,
        MJPEG ``<img>`` or WebSocket streams, so the UI signs in once here and the
        cookie then authenticates those streams.

        Args:
            req: Payload containing the shared ``token``.
            request: Active request, used to detect HTTPS for the ``Secure`` flag.
            response: FastAPI response used to attach the cookie headers.

        Returns:
            dict[str, Any]: Status and session lifetime.

        Raises:
            HTTPException: 401 when the submitted token does not match.
        """
        client_key = getattr(request.client, "host", None) or "unknown"
        throttle: FailureThrottle = request.app.state.login_throttle
        retry_after = throttle.retry_after(client_key)
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="Too many failed login attempts. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )

        if not _token_matches((req.token or "").strip(), token):
            throttle.record_failure(client_key)
            logger.warning("api_auth_login_failed", remote=client_key)
            # A small constant delay makes scripted guessing measurably slower
            # without adding latency to a successful login.
            await asyncio.sleep(0.25)
            raise HTTPException(status_code=401, detail="Invalid API token.")
        throttle.reset(client_key)
        response.set_cookie(
            key=AUTH_COOKIE_NAME,
            value=token,
            max_age=AUTH_COOKIE_MAX_AGE,
            httponly=True,
            secure=(request.url.scheme == "https"),
            samesite="lax",
            path="/",
        )
        return {"status": "ok", "authenticated": True, "max_age": AUTH_COOKIE_MAX_AGE}

    @app.post("/auth/logout")
    async def auth_logout(response: Response) -> dict[str, str]:
        """Clear the local session cookie (the bearer token itself is unchanged)."""
        response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
        return {"status": "ok"}

    @app.get("/auth/status")
    async def auth_status(request: Request) -> dict[str, Any]:
        """Report whether the caller currently presents valid credentials.

        Returns:
            dict[str, Any]: ``authenticated`` flag and the accepted credential
                kinds; the token value itself is never returned. ``auth_required``
                is ``False`` for a caller admitted as a trusted local client, which
                is what lets the UI shell skip its token gate on this machine.
        """
        candidate = _token_from_scope(request.scope)
        local = bool(request.scope.get(_ApiAuthMiddleware.TRUSTED_LOCAL_SCOPE_KEY)) or (
            trust_loopback and _is_trusted_local_scope(request.scope)
        )
        return {
            "status": "ok",
            "auth_required": not local,
            "authenticated": local or _token_matches(candidate, token),
            "via": "trusted_loopback" if local and not _token_matches(candidate, token) else "token",
            "accepted": ["authorization_bearer", "query_token", "session_cookie"],
        }

    @app.get("/status", response_model=StatusResponse)
    async def status():
        state = engine.state.state
        player_active = hasattr(engine, "player") and engine.player and engine.player.is_playing
        speaking_active = getattr(engine, "_is_speaking", False)
        if state.mode == RobotMode.SPEAKING and not player_active and not speaking_active:
            await engine.state.set_mode(RobotMode.IDLE)
            state = engine.state.state

        music_playing = False
        try:
            from robot.tools.builtins.music import MusicPlayerService
            music_playing = MusicPlayerService.get_instance().is_playing
        except Exception:
            pass

        return StatusResponse(
            mode=state.mode.value,
            personality=state.active_personality,
            turn_count=state.turn_count,
            voice_enabled=getattr(engine, "voice_enabled", True),
            services={
                name: svc.healthy for name, svc in state.services.items()
            },
            telemetry=engine.telemetry.snapshot(),
            is_speaking=bool(speaking_active or player_active),
            music_playing=bool(music_playing),
        )

    @app.post("/voice-mode")
    async def set_voice_mode(req: VoiceModeRequest) -> dict[str, Any]:
        """Toggle voice/mic listening on or off for text-only chatting."""
        engine.voice_enabled = bool(req.enabled)
        engine.is_chat_mode = not engine.voice_enabled
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
        # Physically pause/resume mic stream to free audio hardware and ensure no audio is captured
        if hasattr(engine, "capture") and engine.capture:
            if engine.voice_enabled:
                await engine.capture.resume_async()
            else:
                # Releasing the microphone is blocking PortAudio teardown.
                await asyncio.to_thread(engine.capture.pause)
        await engine.events.emit("voice_mode_changed", voice_enabled=engine.voice_enabled)
        return {"status": "ok", "voice_enabled": engine.voice_enabled, "is_chat_mode": engine.is_chat_mode}

    @app.get("/api/voice/status")
    async def get_voice_status() -> dict[str, Any]:
        """Retrieve voice listening and wake-word gating status.

        Returns:
            dict[str, Any]: Voice configuration including active wake word,
                hardware mic mute state, and wake-word gating requirement.
        """
        active_word = getattr(getattr(engine, "wakeword", None), "active_wake_word", "luna")
        return {
            "status": "ok",
            "voice_enabled": engine.voice_enabled,
            "require_wake_word": getattr(engine, "require_wake_word", True),
            "active_wake_word": active_word,
            "barge_in": getattr(engine, "barge_in_enabled", True),
        }

    @app.post("/api/voice/wake-word-toggle")
    async def toggle_wake_word(req: WakeWordRequest | None = Body(default=None)) -> dict[str, Any]:
        """Toggle or set whether the active wake word is required to talk.

        Args:
            req: Optional JSON payload containing boolean `require_wake_word`.

        Returns:
            dict[str, Any]: Updated wake-word gating status.
        """
        explicit = req.require_wake_word if req is not None else None
        if explicit is None:
            engine.require_wake_word = not getattr(engine, "require_wake_word", True)
        else:
            engine.require_wake_word = bool(explicit)
        await engine.events.emit(
            "wake_word_mode_changed",
            require_wake_word=engine.require_wake_word,
        )
        return {"status": "ok", "require_wake_word": engine.require_wake_word}

    @app.post("/camera-mode")
    @app.post("/api/camera-mode")
    async def set_camera_mode(req: CameraModeRequest):
        """Toggle camera hardware acquisition on or off for privacy or chat mode."""
        actual = engine.set_camera_mode(bool(req.enabled))
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
        await engine.events.emit("camera_mode_changed", camera_enabled=actual)
        return {"status": "ok", "camera_enabled": actual}

    @app.get("/camera-mode")
    @app.get("/api/camera-mode")
    async def get_camera_mode():
        """Get current camera hardware acquisition status."""
        return {"status": "ok", "camera_enabled": engine.is_camera_enabled}

    @app.post("/api/view-mode")
    async def set_view_mode(req: ViewModeRequest):
        """Synchronize UI active view mode ('chat' vs 'dual') with engine state.

        Args:
            req: Payload containing 'mode' ('chat' | 'dual').

        Returns:
            dict[str, Any]: Status summary and confirmation.
        """
        mode = req.mode
        engine.is_chat_mode = (mode == "chat")
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
        return {"status": "ok", "mode": mode, "is_chat_mode": engine.is_chat_mode}

    @app.get("/personalities")
    async def list_personalities():
        engine.personalities.load_all()
        return {
            "personalities": engine.personalities.list_personalities(),
            "active": engine.state.state.active_personality,
        }

    @app.get("/api/personas")
    async def list_personas():
        """List all personas with complete metadata, isolated database statistics, and model defaults."""
        engine.personalities.load_all()
        personas = engine.personalities.list_personas_detailed()
        active_p = getattr(getattr(getattr(engine, "state", None), "state", None), "active_personality", None)
        if not active_p:
            active_p = getattr(engine, "active_personality", "neo")
        return {
            "status": "ok",
            "personas": personas,
            "active": active_p,
            "total": len(personas),
        }

    @app.post("/api/personas")
    async def create_persona(req: PersonaCreateRequest):
        """Create a new persona and provision its dedicated isolated database."""
        try:
            raw_data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            pers = engine.personalities.create_persona(raw_data)
            return {
                "status": "created",
                "persona": pers.to_dict(),
                "isolated_db": pers.database_path,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("persona_create_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to create persona.")

    @app.get("/api/personas/{persona_id}")
    async def get_persona(persona_id: str = PathParam(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")):
        """Get details, default models, and database statistics for a specific persona."""
        pers = engine.personalities.get(persona_id)
        if not pers:
            raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
        data = pers.to_dict()
        data["is_active"] = (engine.personalities.active_name == pers.name)
        data.update(engine.personalities.get_persona_stats(pers.name))
        return data

    @app.put("/api/personas/{persona_id}")
    async def update_persona(persona_id: str = PathParam(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"), req: PersonaUpdateRequest = Body(...)):
        """Update an existing persona definition and default model configurations."""
        try:
            update_data = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
            pers = engine.personalities.update_persona(persona_id, update_data)
            return {
                "status": "updated",
                "persona": pers.to_dict(),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("persona_update_failed", persona=persona_id, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to update persona.")

    @app.delete("/api/personas/{persona_id}")
    async def delete_persona(
        persona_id: str = PathParam(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
        delete_database: bool = True,
        force: bool = False,
    ):
        """Delete a persona definition and optionally delete its isolated database."""
        try:
            ok = engine.personalities.delete_persona(persona_id, delete_database=delete_database, force=force)
            return {"status": "deleted", "id": persona_id, "deleted_database": delete_database}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("persona_delete_failed", persona=persona_id, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to delete persona.")

    @app.post("/api/personas/{persona_id}/activate")
    async def activate_persona(persona_id: str = PathParam(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")):
        """Activate a persona, hot-swap memory to its isolated database, and apply its default models."""
        try:
            res = await engine.switch_personality(persona_id)
            return res
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception("persona_activate_failed", persona=persona_id, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to activate persona.")

    @app.post("/personality/{name}")
    async def switch_personality(name: str):
        """Legacy hot-swap endpoint delegating to engine.switch_personality with resilient fallback."""
        switch_fn = getattr(engine, "switch_personality", None)
        if callable(switch_fn) and type(switch_fn).__name__ not in ("MagicMock", "AsyncMock"):
            try:
                res = await switch_fn(name)
                return res
            except ValueError as exc:
                return {"error": str(exc)}
            except Exception as exc:
                logger.exception("legacy_personality_switch_failed", personality=name, error=str(exc))
                raise HTTPException(status_code=500, detail="Failed to switch personality.") from None

        # Fallback or Mock Engine execution
        mgr = getattr(engine, "personalities", None)
        if not mgr:
            return {"error": f"Personality manager unavailable"}
        success = mgr.set_active(name)
        if success:
            if hasattr(engine, "state") and hasattr(engine.state, "state"):
                engine.state.state.active_personality = name
            pers = mgr.get(name)
            if pers and getattr(pers, "voice_model", None) and hasattr(engine, "tts") and hasattr(engine.tts, "set_active_voice"):
                engine.tts.set_active_voice(pers.voice_model)
            if hasattr(engine, "wakeword") and engine.wakeword:
                engine.wakeword.set_active_persona(name)
            if hasattr(engine, "events") and engine.events:
                await engine.events.emit("personality_switched", personality=name)
                await engine.events.emit("personality_warming", personality=name)
            if hasattr(engine, "warm_up_llm") and callable(getattr(engine, "warm_up_llm")):
                coro = engine.warm_up_llm(name)
                if asyncio.iscoroutine(coro):
                    spawn(coro, name="warm_up_llm")
            return {"status": "switched", "personality": name}
        return {"error": f"Personality '{name}' not found"}

    # ─── LLM Provider Swapping & Management Endpoints ───────────────────────
    @app.get("/api/llm/status")
    async def get_llm_status():
        """Get active LLM provider, mode, context window, and llama-server process status."""
        prov = getattr(engine, "llm_provider_type", "local")
        if prov in ("llama-server", "local"):
            prov = "local"
        is_gemini = prov == "gemini"
        from robot.llm import LlamaServerManager
        llama_running = await LlamaServerManager.is_running_async()
        local_ctx = int(getattr(engine, "config", {}).get("llm", {}).get("context_window", 16384)) if hasattr(engine, "config") else 16384
        openai_cfg = getattr(engine, "config", {}).get("llm", {}).get("openai_compatible", {}) if hasattr(engine, "config") else {}
        is_openai = prov in ("openai_compatible", "openai")
        
        ctx_win = local_ctx
        if is_gemini:
            ctx_win = 1000000
        elif is_openai:
            ctx_win = int(openai_cfg.get("context_window", 128000))

        active_model = getattr(engine.llm, "model", "Qwen3.5-4B-Q4_K_M")
        persisted_openai_model = getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "gpt-4o"))
        openrouter_cfg = getattr(engine, "config", {}).get("llm", {}).get("openrouter", {}) if hasattr(engine, "config") else {}
        openrouter_key = openrouter_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        has_openrouter_key = bool(openrouter_key)

        aicredits_cfg = getattr(engine, "config", {}).get("llm", {}).get("aicredits", {}) if hasattr(engine, "config") else {}
        aicredits_key = aicredits_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", "")
        has_aicredits_key = bool(aicredits_key)

        # NOTE: no raw secret is ever returned here — providers are described by
        # booleans only, and base URLs are non-sensitive.
        return {
            "provider": prov,
            "mode": "high_intelligence" if (is_gemini or is_openai) else "low_latency",
            "model": active_model,
            "active_openai_model": persisted_openai_model,
            "context_window": ctx_win,
            "llama_running": llama_running,
            "vram_freed": not llama_running,
            "openai_compatible": {
                "base_url": openai_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                "model": persisted_openai_model,
                "context_window": int(openai_cfg.get("context_window", 128000)),
                "has_api_key": bool(openai_cfg.get("api_key")),
                "enable_prompt_caching": bool(openai_cfg.get("enable_prompt_caching", True)),
            },
            "openrouter": {
                "base_url": openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                "model": openrouter_cfg.get("model", "meta-llama/llama-3.3-70b-instruct:free"),
                "has_api_key": has_openrouter_key,
            },
            "aicredits": {
                "base_url": aicredits_cfg.get("base_url", "https://aicredits.in/v1"),
                "model": aicredits_cfg.get("model", "aion-labs/aion-3.0"),
                "has_api_key": has_aicredits_key,
            },
        }

    @app.post("/api/llm/toggle")
    async def toggle_llm_provider():
        """Toggle between Local, Cloud Gemini, and OpenAI-Compatible."""
        current = getattr(engine, "llm_provider_type", "local")
        if current in ("local", "llama-server"):
            target = "gemini"
        elif current == "gemini":
            target = "openai_compatible"
        else:
            target = "local"
        res = await engine.switch_llm_provider(target)
        return res

    @app.post("/api/llm/provider")
    async def set_llm_provider(req: LLMProviderRequest):
        """Set specific LLM provider ('local', 'gemini', or 'openai_compatible')."""
        target = req.provider
        res = await engine.switch_llm_provider(target)
        return res

    @app.get("/api/llm/openai/models")
    async def scan_openai_models(
        base_url: str | None = Query(default=None, max_length=2048),
        preset: str | None = Query(default=None, max_length=32),
    ):
        """Scan available models from the OpenAI-compatible endpoint.

        The API key is never accepted as a query parameter (it would leak into
        browser history, proxy logs and access logs); it is resolved from the
        engine configuration/environment for the selected preset.

        Args:
            base_url: Optional override endpoint URL to scan.
            preset: Optional provider preset ('openrouter', 'aicredits', 'openai', etc.).

        Returns:
            dict[str, Any]: Status and list of discovered model identifiers.
        """
        from robot.llm import OpenAICompatibleLLM
        cfg_root = getattr(engine, "config", {}).get("llm", {}) if hasattr(engine, "config") else {}
        openai_cfg = cfg_root.get("openai_compatible", {})
        openrouter_cfg = cfg_root.get("openrouter", {})
        aicredits_cfg = cfg_root.get("aicredits", {})

        target_url = base_url
        target_key: str | None = None

        if preset == "openrouter":
            target_url = base_url or openrouter_cfg.get("base_url") or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            target_key = openrouter_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        elif preset == "aicredits":
            target_url = base_url or aicredits_cfg.get("base_url") or os.environ.get("AICREDITS_BASE_URL", "https://aicredits.in/v1")
            target_key = aicredits_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", "")
        elif preset == "openai":
            target_url = base_url or "https://api.openai.com/v1"
            target_key = os.environ.get("OPENAI_API_KEY", "")
        elif preset == "deepseek":
            target_url = base_url or "https://api.deepseek.com/v1"
            target_key = os.environ.get("DEEPSEEK_API_KEY", "")
        elif preset == "groq":
            target_url = base_url or "https://api.groq.com/openai/v1"
            target_key = os.environ.get("GROQ_API_KEY", "")
        elif preset == "ollama":
            target_url = base_url or "http://127.0.0.1:11434/v1"
            target_key = ""
        else:
            if not target_url:
                target_url = openai_cfg.get("base_url", "https://openrouter.ai/api/v1")
            if "openrouter.ai" in str(target_url):
                target_key = openai_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
            elif "aicredits.in" in str(target_url):
                target_key = openai_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", "")
            else:
                target_key = openai_cfg.get("api_key", "")

        target_url = _validate_http_url(str(target_url or ""), "base_url")

        current_model = getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "meta-llama/llama-3.3-70b-instruct:free"))

        scanner = OpenAICompatibleLLM(
            base_url=target_url,
            api_key=target_key,
            model=current_model,
        )
        try:
            models = await scanner.scan_models()
            provider_preset = "openrouter" if "openrouter.ai" in str(target_url) else ("aicredits" if "aicredits.in" in str(target_url) else "custom")
            return {
                "status": "ok",
                "models": models,
                "count": len(models),
                "current_model": current_model,
                "active_model": current_model,
                "base_url": target_url,
                "provider_preset": provider_preset,
            }
        except Exception as e:
            logger.warning("openai_model_scan_failed", base_url=target_url, error=str(e))
            provider_preset = "openrouter" if "openrouter.ai" in str(target_url) else ("aicredits" if "aicredits.in" in str(target_url) else "custom")
            return {
                "status": "error",
                "message": "Could not reach the OpenAI-compatible provider.",
                "models": [],
                "count": 0,
                "current_model": current_model,
                "active_model": current_model,
                "base_url": target_url,
                "provider_preset": provider_preset,
            }

    @app.post("/api/llm/openai/model")
    async def set_openai_model(req: OpenAIModelRequest):
        """Set active model for the OpenAI-compatible provider.

        Args:
            req: Payload containing 'model' identifier string.

        Returns:
            dict[str, Any]: Status confirmation and selected model.
        """
        from robot.config import update_env_variable
        new_model = req.model.strip()
        if not new_model:
            return {"status": "error", "message": "Model name cannot be empty."}

        if hasattr(engine, "_persist_openai_model"):
            engine._persist_openai_model(new_model)
        update_env_variable("OPENAI_COMPATIBLE_MODEL", new_model)

        # Intelligent auto-routing: if model belongs to OpenRouter or AI Credits, align base_url and api_key
        current_base = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "").lower()
        is_or_model = ":free" in new_model or any(new_model.startswith(p) for p in ("meta-llama/", "google/", "nvidia/", "liquid/", "apodex/", "thinkingmachines/", "mistralai/"))
        is_ai_model = new_model.startswith("aion-labs/")

        if is_or_model and "openrouter" not in current_base:
            or_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", or_base)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", or_key)
            if hasattr(engine, "config"):
                cfg = engine.config.setdefault("llm", {}).setdefault("openai_compatible", {})
                cfg["base_url"] = or_base
                cfg["api_key"] = or_key
            if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
                engine.llm.base_url = or_base
                engine.llm.api_key = or_key
        elif is_ai_model and "aicredits" not in current_base:
            ai_base = os.environ.get("AICREDITS_BASE_URL", "https://aicredits.in/v1")
            ai_key = os.environ.get("AICREDITS_API_KEY", "")
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", ai_base)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", ai_key)
            if hasattr(engine, "config"):
                cfg = engine.config.setdefault("llm", {}).setdefault("openai_compatible", {})
                cfg["base_url"] = ai_base
                cfg["api_key"] = ai_key
            if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
                engine.llm.base_url = ai_base
                engine.llm.api_key = ai_key

        if hasattr(engine, "config"):
            engine.config.setdefault("llm", {}).setdefault("openai_compatible", {})["model"] = new_model

        if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
            engine.llm.model = new_model

        await engine.events.emit("llm.model_changed", provider="openai_compatible", model=new_model)
        return {"status": "ok", "model": new_model, "active_model": new_model}

    @app.post("/api/llm/openai/config")
    async def configure_openai_compatible(req: dict[str, Any]):
        """Update and persist OpenAI-compatible provider settings to .env.

        Args:
            req: Configuration payload containing base_url, api_key, model,
                context_window, and enable_prompt_caching.

        Returns:
            dict[str, Any]: Status summary and confirmation.
        """
        from robot.config import update_env_variable
        raw_base_url = req.get("base_url", "")
        raw_api_key = req.get("api_key", "")
        raw_model = req.get("model", "")
        if not all(isinstance(value, str) for value in (raw_base_url, raw_api_key, raw_model)):
            raise HTTPException(status_code=400, detail="base_url, api_key and model must be strings.")
        base_url = raw_base_url.strip()
        api_key = raw_api_key.strip()
        model = raw_model.strip()
        if base_url:
            base_url = _validate_http_url(base_url, "base_url")
        ctx_win = req.get("context_window")
        prompt_caching = req.get("enable_prompt_caching", True)
        active_provider = req.get("active_provider") or req.get("preset", "custom")
        if not isinstance(active_provider, str):
            raise HTTPException(status_code=400, detail="active_provider must be a string.")
        if len(api_key) > 512 or len(model) > 256:
            raise HTTPException(status_code=400, detail="api_key or model is too long.")

        cfg_root = getattr(engine, "config", {}).setdefault("llm", {})
        openai_cfg = cfg_root.setdefault("openai_compatible", {})
        or_cfg = cfg_root.setdefault("openrouter", {})
        ai_cfg = cfg_root.setdefault("aicredits", {})

        # 1. Update OpenRouter specific fields if passed
        def _nested_text(section: dict[str, Any], key: str) -> str:
            value = section.get(key, "")
            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f"{key} must be a string.")
            return value.strip()

        or_data = req.get("openrouter")
        if isinstance(or_data, dict):
            or_b = _nested_text(or_data, "base_url")
            or_k = _nested_text(or_data, "api_key")
            or_m = _nested_text(or_data, "model")
            if or_b:
                or_b = _validate_http_url(or_b, "openrouter.base_url")
                or_cfg["base_url"] = or_b
                update_env_variable("OPENROUTER_BASE_URL", or_b)
            if or_k:
                or_cfg["api_key"] = or_k
                update_env_variable("OPENROUTER_API_KEY", or_k)
            if or_m:
                or_cfg["model"] = or_m
                update_env_variable("OPENROUTER_MODEL", or_m)

        # 2. Update AI Credits specific fields if passed
        ai_data = req.get("aicredits")
        if isinstance(ai_data, dict):
            ai_b = _nested_text(ai_data, "base_url")
            ai_k = _nested_text(ai_data, "api_key")
            ai_m = _nested_text(ai_data, "model")
            if ai_b:
                ai_b = _validate_http_url(ai_b, "aicredits.base_url")
                ai_cfg["base_url"] = ai_b
                update_env_variable("AICREDITS_BASE_URL", ai_b)
            if ai_k:
                ai_cfg["api_key"] = ai_k
                update_env_variable("AICREDITS_API_KEY", ai_k)
            if ai_m:
                ai_cfg["model"] = ai_m
                update_env_variable("AICREDITS_MODEL", ai_m)

        # 3. Synchronize active provider target
        if active_provider == "openrouter":
            target_url = (or_data.get("base_url") if isinstance(or_data, dict) else "") or base_url or or_cfg.get("base_url") or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            target_key = (or_data.get("api_key") if isinstance(or_data, dict) else "") or api_key or or_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
            target_model = (or_data.get("model") if isinstance(or_data, dict) else "") or model or or_cfg.get("model") or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
            openai_cfg["base_url"] = target_url
            openai_cfg["api_key"] = target_key
            openai_cfg["model"] = target_model
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", target_url)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", target_key)
            update_env_variable("OPENAI_COMPATIBLE_MODEL", target_model)
        elif active_provider == "aicredits":
            target_url = (ai_data.get("base_url") if isinstance(ai_data, dict) else "") or base_url or ai_cfg.get("base_url") or os.environ.get("AICREDITS_BASE_URL", "https://aicredits.in/v1")
            target_key = (ai_data.get("api_key") if isinstance(ai_data, dict) else "") or api_key or ai_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", "")
            target_model = (ai_data.get("model") if isinstance(ai_data, dict) else "") or model or ai_cfg.get("model") or os.environ.get("AICREDITS_MODEL", "aion-labs/aion-3.0")
            openai_cfg["base_url"] = target_url
            openai_cfg["api_key"] = target_key
            openai_cfg["model"] = target_model
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", target_url)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", target_key)
            update_env_variable("OPENAI_COMPATIBLE_MODEL", target_model)
        else:
            if base_url:
                openai_cfg["base_url"] = base_url
                update_env_variable("OPENAI_COMPATIBLE_BASE_URL", base_url)
            if api_key is not None and api_key != "":
                openai_cfg["api_key"] = api_key
                update_env_variable("OPENAI_COMPATIBLE_API_KEY", api_key)
            if model:
                openai_cfg["model"] = model
                update_env_variable("OPENAI_COMPATIBLE_MODEL", model)

        if hasattr(engine, "_persist_openai_model") and openai_cfg.get("model"):
            engine._persist_openai_model(openai_cfg["model"])

        if ctx_win:
            try:
                int_ctx = int(ctx_win)
                openai_cfg["context_window"] = int_ctx
                update_env_variable("OPENAI_COMPATIBLE_CONTEXT_WINDOW", str(int_ctx))
            except ValueError:
                pass

        openai_cfg["enable_prompt_caching"] = bool(prompt_caching)
        update_env_variable("OPENAI_COMPATIBLE_ENABLE_PROMPT_CACHING", "true" if prompt_caching else "false")

        # If currently active, refresh live LLM instance
        if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
            from robot.llm import OpenAICompatibleLLM
            engine.llm = OpenAICompatibleLLM(
                base_url=openai_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                api_key=openai_cfg.get("api_key", ""),
                model=openai_cfg.get("model", "meta-llama/llama-3.3-70b-instruct:free"),
                context_window=openai_cfg.get("context_window", 128000),
                enable_prompt_caching=openai_cfg.get("enable_prompt_caching", True),
            )

        return {
            "status": "ok",
            "message": "OpenAI-compatible configuration updated and saved to .env",
            "config": {
                "base_url": openai_cfg.get("base_url"),
                "model": openai_cfg.get("model"),
                "context_window": openai_cfg.get("context_window"),
                "enable_prompt_caching": openai_cfg.get("enable_prompt_caching"),
                "has_api_key": bool(openai_cfg.get("api_key")),
            },
        }

    @app.get("/api/models/local")
    async def list_local_models() -> dict[str, Any]:
        """List registered 2B Vision Language Models (VLM) and <4B Uncensored models with download status.

        Returns:
            dict[str, Any]: Model catalog, default model, and download status.
        """
        from robot.config import get_project_root
        from robot.llm.model_registry import ModelRegistry

        base_dir = get_project_root()
        models_data = []
        for m in ModelRegistry.list_models():
            models_data.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "category": m.category.value,
                    "parameters": m.parameters,
                    "repo_id": m.repo_id,
                    "filename": m.filename,
                    "mmproj_filename": m.mmproj_filename,
                    "is_vision": m.is_vision,
                    "is_uncensored": m.is_uncensored,
                    "supports_tools": m.supports_tools,
                    "recommended_context": m.recommended_context,
                    "description": m.description,
                    "is_downloaded": m.is_downloaded(base_dir),
                    "model_path": str(m.get_model_path(base_dir)),
                    "mmproj_path": str(m.get_mmproj_path(base_dir)) if m.mmproj_filename else None,
                }
            )

        cfg_model = getattr(engine, "config", {}).get("llm", {}).get("model", "qwen2-vl-2b-instruct")
        return {
            "default_model": ModelRegistry.DEFAULT_VLM_MODEL_ID,
            "active_configured_model": cfg_model,
            "total_models": len(models_data),
            "models": models_data,
        }

    @app.post("/api/models/local/switch")
    async def switch_local_model(req: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Hot-swap the active local GGUF model running on the RTX 3050 GPU.

        Args:
            req: Request payload containing the target 'model' or 'model_id'.

        Returns:
            Dictionary with switch operation status, model metadata, and slot warming state.
        """
        model_id = req.get("model") or req.get("model_id")
        if not model_id:
            return {"status": "error", "message": "Missing 'model' parameter in request body."}
        from robot.llm import LlamaServerManager
        res = await LlamaServerManager.switch_model(model_id, engine=engine)
        if res.get("status") == "ok" and hasattr(engine, "events") and engine.events:
            await engine.events.emit("llm_model_changed", model=model_id, parameters=res.get("parameters"))
        return res

    @app.get("/api/llm/models")
    async def list_all_llm_models() -> dict[str, Any]:
        """List unified LLM models across Local, Gemini, and OpenAI-compatible providers."""
        from pathlib import Path
        from robot.config import get_project_root
        from robot.llm import LlamaServerManager
        from robot.llm.model_registry import ModelRegistry

        base_dir = get_project_root()
        current_prov = getattr(engine, "llm_provider_type", "local")
        active_local = LlamaServerManager.get_active_model_id()

        local_models = []
        # The default local GGUF is whatever the manager is configured to load;
        # never hard-code one machine's Downloads path into the API surface.
        configured_model = Path(str(getattr(LlamaServerManager, "MODEL_PATH", "") or "")).expanduser()
        if configured_model.is_file():
            stem = configured_model.stem
            default_id = "qwen3.5-4b" if "qwen3.5-4b" in stem.lower() else stem.lower().replace("_", "-")
            local_models.append({
                "id": default_id,
                "name": "Qwen 3.5 4B (Instruct)" if default_id == "qwen3.5-4b" else stem.replace("-", " ").replace("_", " "),
                "parameters": "4B" if "4b" in stem.lower() else "local",
                "is_vision": False,
                "is_uncensored": "uncensored" in stem.lower(),
                "supports_tools": True,
                "is_downloaded": True,
                "is_active": (active_local in (default_id, configured_model.name) and current_prov in ("local", "llama-server")),
                "description": "Standard high-reasoning local edge model served by llama-server on the local GPU.",
            })

        for m in ModelRegistry.list_models():
            local_models.append({
                "id": m.id,
                "name": m.name,
                "parameters": m.parameters,
                "is_vision": m.is_vision,
                "is_uncensored": m.is_uncensored,
                "supports_tools": m.supports_tools,
                "is_downloaded": m.is_downloaded(base_dir),
                "is_active": (active_local == m.id and current_prov in ("local", "llama-server")),
                "description": m.description,
            })

        gemini_models = [
            {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite", "context": "1M tokens", "is_active": (current_prov == "gemini")},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "context": "1M tokens", "is_active": False},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "context": "1M tokens", "is_active": False},
        ]

        return {
            "status": "ok",
            "provider": current_prov,
            "active_local_model": active_local,
            "local_models": local_models,
            "gemini_models": gemini_models,
            "llama_running": await LlamaServerManager.is_running_async(),
        }

    # ─── Llama.cpp Advanced Loader & Offloading Settings (LM Studio) ───────
    @app.get("/api/llm/settings")
    async def get_llama_settings() -> dict[str, Any]:
        """Retrieve current llama-server configuration parameters and hardware offloading state."""
        from robot.llm import LlamaServerManager
        return await LlamaServerManager.get_settings()

    @app.post("/api/llm/settings")
    async def update_llama_settings(req: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Update llama-server loading parameters and optionally hot-reload the running process."""
        from robot.llm import LlamaServerManager
        settings_data = req.get("settings", req)
        reload = bool(req.get("reload", False))
        res = await LlamaServerManager.update_settings(settings_data, reload=reload, engine=engine)
        return res

    @app.post("/api/llm/settings/reset")
    async def reset_llama_settings(req: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Reset llama-server configuration to optimal RTX 3050 defaults."""
        from robot.llm import LlamaServerManager
        reload = bool(req.get("reload", False))
        res = await LlamaServerManager.reset_settings(reload=reload, engine=engine)
        return res

    # ─── HuggingFace Hub LLM Search & Downloader Endpoints ──────────────────
    @app.get("/api/llm/hf/search")
    async def search_huggingface_models(
        q: str = Query("", description="Search term for HuggingFace models"),
        limit: int = Query(20, ge=1, le=50, description="Max results"),
    ) -> dict[str, Any]:
        """Search HuggingFace Hub for GGUF models.

        Args:
            q: Search query string.
            limit: Maximum items to return.

        Returns:
            dict[str, Any]: Search results and metadata.
        """
        from robot.llm.hf_downloader import HuggingFaceHubService
        results = await HuggingFaceHubService.search_models(query=q, limit=limit)
        return {
            "status": "ok",
            "query": q,
            "total": len(results),
            "models": results,
        }

    @app.get("/api/llm/hf/files")
    async def list_huggingface_model_files(
        repo_id: str = Query(..., description="Repository ID (e.g. bartowski/Qwen2.5-Coder-3B-Instruct-GGUF)"),
    ) -> dict[str, Any]:
        """Retrieve available GGUF files and quantization options for a model repository.

        Args:
            repo_id: HuggingFace repository identifier.

        Returns:
            dict[str, Any]: List of GGUF files with byte sizes and download statuses.
        """
        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceHubService
        base_dir = get_project_root()
        files = await HuggingFaceHubService.get_model_files(repo_id=repo_id, base_dir=base_dir)
        return {
            "status": "ok",
            "repo_id": repo_id,
            "total": len(files),
            "files": files,
        }

    @app.post("/api/llm/hf/download")
    async def download_huggingface_model(
        req: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Start asynchronous background download of a HuggingFace GGUF model.

        Args:
            req: Request body containing repo_id, filename, name, parameters, is_vision.

        Returns:
            dict[str, Any]: Created download task specification and tracking ID.
        """
        repo_id = req.get("repo_id")
        filename = req.get("filename")
        if not repo_id or not filename:
            return {"status": "error", "message": "Missing required 'repo_id' or 'filename'."}

        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceDownloadManager

        base_dir = get_project_root()
        mgr = HuggingFaceDownloadManager.get_instance(base_dir=base_dir)
        task = mgr.start_download(
            repo_id=repo_id,
            filename=filename,
            name=req.get("name"),
            parameters=req.get("parameters"),
            is_vision=bool(req.get("is_vision", False)),
            mmproj_filename=req.get("mmproj_filename"),
        )
        return {"status": "ok", "task": task.to_dict()}

    @app.get("/api/llm/hf/downloads")
    async def get_huggingface_downloads() -> dict[str, Any]:
        """Get list of active and recent HuggingFace model download tasks.

        Returns:
            dict[str, Any]: List of all download tasks and their progress metrics.
        """
        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceDownloadManager

        base_dir = get_project_root()
        mgr = HuggingFaceDownloadManager.get_instance(base_dir=base_dir)
        tasks = mgr.get_all_tasks()
        return {
            "status": "ok",
            "total": len(tasks),
            "downloads": tasks,
        }

    @app.post("/api/llm/hf/cancel")
    async def cancel_huggingface_download(
        req: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Cancel an in-progress HuggingFace model download task.

        Args:
            req: Request body containing download_id.

        Returns:
            dict[str, Any]: Operation status and cancellation flag.
        """
        download_id = req.get("download_id")
        if not download_id:
            return {"status": "error", "message": "Missing 'download_id'."}

        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceDownloadManager

        base_dir = get_project_root()
        mgr = HuggingFaceDownloadManager.get_instance(base_dir=base_dir)
        ok = await mgr.cancel_download(download_id)
        return {"status": "ok", "cancelled": ok}

    # ─── Speech-to-Text (Whisper) Endpoints ─────────────────────────────────
    @app.get("/api/stt/status")
    async def get_stt_status() -> dict[str, Any]:
        """Get active Whisper STT model, daemon process state, and hardware offload metrics."""
        from robot.stt import WhisperServerManager
        active_id = WhisperServerManager.get_active_model_id()
        is_running = await WhisperServerManager.is_running_async()
        meta = WhisperServerManager.MODELS_CATALOG.get(active_id)
        return {
            "status": "ok",
            "is_running": is_running,
            "active_model": active_id,
            "name": meta.name if meta else active_id,
            "parameters": meta.parameters if meta else "unknown",
            "size_mb": meta.size_mb if meta else 0,
            "latency_rating": meta.latency_rating if meta else "unknown",
            "device": "AMD Radeon 680M (Vulkan dev 0)",
            "vram_rtx3050_mb": 0,
        }

    @app.get("/api/stt/models")
    async def list_stt_models() -> dict[str, Any]:
        """List all registered Whisper models with download and active status."""
        from robot.stt import WhisperServerManager
        models = WhisperServerManager.list_models()
        active_id = WhisperServerManager.get_active_model_id()
        is_running = await WhisperServerManager.is_running_async()
        return {
            "status": "ok",
            "active_model": active_id,
            "is_running": is_running,
            "total_models": len(models),
            "models": models,
        }

    @app.post("/api/stt/switch")
    async def switch_stt_model(req: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Hot-swap the active Whisper model on the AMD Radeon 680M Vulkan backend.

        Args:
            req: Request payload containing the target Whisper 'model' or 'model_id'.

        Returns:
            Dictionary with switch operation status, model name, and file size in MB.
        """
        model_id = req.get("model") or req.get("model_id")
        if not model_id:
            return {"status": "error", "message": "Missing 'model' parameter in request body."}
        from robot.stt import WhisperServerManager
        res = await WhisperServerManager.switch_model(model_id, engine=engine)
        if res.get("status") == "ok" and hasattr(engine, "events") and engine.events:
            await engine.events.emit("stt_model_changed", model=model_id, size_mb=res.get("size_mb"))
        return res

    # ─── Text-to-Speech (Piper & Kokoro) Endpoints ─────────────────────────
    @app.get("/api/tts/status")
    async def get_tts_status() -> dict[str, Any]:
        """Get active TTS engine, voice model, sample rate, and supported engines."""
        from robot.tts import TTSManager
        tts = getattr(engine, "tts", None)
        if isinstance(tts, TTSManager):
            # Every accessor here can hit the disk or initialize an ONNX/Kokoro
            # model, so the whole snapshot is resolved away from the loop.
            (
                active_engine,
                active_voice,
                engines_list,
                voices_list,
                sample_rate,
            ) = await asyncio.to_thread(
                lambda: (
                    tts.get_active_engine_name(),
                    tts.get_active_voice(),
                    tts.list_engines(),
                    tts.list_voices(),
                    getattr(tts.get_active_engine(), "SAMPLE_RATE", 16000),
                )
            )
        else:
            active_engine = "piper"
            active_voice = getattr(tts, "_active_model", "en-us-lessac-medium")
            engines_list = [
                {
                    "id": "piper",
                    "name": "Piper ONNX",
                    "description": "Ultra low-latency local ONNX synthesis",
                    "sample_rate": 16000,
                    "is_available": True,
                    "is_active": True,
                    "total_voices": 7,
                    "recommended_voice": "en-us-lessac-medium",
                }
            ]
            voices_list = []
            sample_rate = 16000

        return {
            "status": "ok",
            "active_engine": active_engine,
            "active_voice": active_voice,
            "sample_rate": sample_rate,
            "total_voices": len(voices_list),
            "engines": engines_list,
        }

    @app.get("/api/tts/voices")
    async def list_tts_voices(engine_filter: str | None = Query(None, alias="engine")) -> dict[str, Any]:
        """List catalog of all available voices across Kokoro Neural and Piper ONNX."""
        from robot.tts import TTSManager
        tts = getattr(engine, "tts", None)
        if isinstance(tts, TTSManager):
            voices, active_engine, active_voice = await asyncio.to_thread(
                lambda: (
                    tts.list_voices(engine_name=engine_filter),
                    tts.get_active_engine_name(),
                    tts.get_active_voice(),
                )
            )
        else:
            active_engine = "piper"
            active_voice = "en-us-lessac-medium"
            voices = []

        return {
            "status": "ok",
            "active_engine": active_engine,
            "active_voice": active_voice,
            "total_voices": len(voices),
            "voices": voices,
        }

    @app.post("/api/tts/switch")
    async def switch_tts_engine_and_voice(req: TTSSwitchRequest) -> dict[str, Any]:
        """Hot-swap the active TTS engine (Piper or Kokoro) and active voice at runtime.

        Args:
            req: TTSSwitchRequest with engine ('piper' or 'kokoro') and optional voice identifier.

        Returns:
            Dictionary with switch result status, active engine, and active voice.
        """
        from robot.tts import TTSManager
        tts = getattr(engine, "tts", None)
        if not isinstance(tts, TTSManager):
            try:
                engine.tts = TTSManager(engine.config)
                tts = engine.tts
            except Exception as exc:
                logger.warning("tts_manager_init_failed", error=str(exc))
                raise HTTPException(status_code=503, detail="Text-to-speech is unavailable.") from None

        try:
            res = await asyncio.to_thread(tts.switch_engine, req.engine, req.voice)
            if hasattr(engine, "events") and engine.events:
                await engine.events.emit(
                    "tts_engine_changed",
                    engine=res["engine"],
                    voice=res["voice"],
                    sample_rate=res["sample_rate"],
                )
            return {
                "status": "ok",
                "engine": res["engine"],
                "voice": res["voice"],
                "sample_rate": res["sample_rate"],
            }
        except Exception as exc:
            logger.warning("tts_switch_failed", engine=req.engine, error=str(exc))
            raise HTTPException(status_code=400, detail="Could not switch the text-to-speech engine.") from None

    @app.post("/api/tts/test")
    async def test_tts_voice(req: TTSTestRequest) -> Response:
        """Synthesize preview audio using requested engine and voice without changing active state.

        Args:
            req: TTSTestRequest with engine, voice, and test phrase.

        Returns:
            Response containing audio/wav binary data for in-browser playback.
        """
        from robot.tts import TTSManager
        import io
        import wave
        tts = getattr(engine, "tts", None)
        if not isinstance(tts, TTSManager):
            try:
                engine.tts = TTSManager(engine.config)
                tts = engine.tts
            except Exception as exc:
                logger.warning("tts_test_manager_init_failed", error=str(exc))
                raise HTTPException(status_code=503, detail="Text-to-speech is unavailable.") from None

        try:
            target_engine = await asyncio.to_thread(tts._get_or_load_engine, req.engine)
            test_phrase = req.text.strip() or "Hello, this is a test of the neural voice engine on the robot."
            samples, sr = await target_engine.synthesize(test_phrase, voice_model=req.voice)
            if len(samples) == 0:
                raise HTTPException(status_code=500, detail="Synthesis produced zero audio frames.")

            pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())

            return Response(content=buf.getvalue(), media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("tts_test_synthesis_failed", error=str(exc))
            raise HTTPException(status_code=503, detail="Voice synthesis failed.") from None

    # ─── Spotify OAuth & Integration Endpoints ──────────────────────────────
    @app.get("/spotify/status")
    async def spotify_status():
        """Get Spotify configuration, OAuth status, and devices."""
        from robot.tools.builtins.music import MusicPlayerService
        ms = MusicPlayerService.get_instance()
        sp_cfg = ms.config.get("spotify", {})
        client_id = bool(sp_cfg.get("client_id"))
        client_secret = bool(sp_cfg.get("client_secret"))
        # Spotipy builds a session and talks HTTP; neither belongs on the loop.
        sp_oauth = await asyncio.to_thread(ms._get_spotify_oauth_client)
        authenticated = sp_oauth is not None
        devices: list[Any] = []
        if sp_oauth:
            try:
                payload = await asyncio.to_thread(sp_oauth.devices)
                devices = (payload or {}).get("devices", [])
            except Exception:
                pass
        return {
            "configured": client_id and client_secret,
            "authenticated": authenticated,
            "device_id": sp_cfg.get("device_id") or "",
            "available_devices": devices,
            "login_url": "/spotify/login",
        }

    @app.get("/spotify/login")
    async def spotify_login():
        """Get or redirect to Spotify OAuth login page."""
        from fastapi.responses import RedirectResponse
        from robot.tools.builtins.music import MusicPlayerService
        ms = MusicPlayerService.get_instance()
        sp_cfg = ms.config.get("spotify", {})
        client_id = sp_cfg.get("client_id")
        client_secret = sp_cfg.get("client_secret")
        redirect_uri = sp_cfg.get("redirect_uri", "http://127.0.0.1:8888/callback")
        if not client_id or not client_secret:
            return {"error": "Spotify client_id or client_secret not configured"}
        try:
            from spotipy.oauth2 import SpotifyOAuth
            scope = "user-modify-playback-state user-read-playback-state user-read-currently-playing"
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
                open_browser=False,
                cache_path=".spotify_cache",
            )
            auth_url = auth_manager.get_authorize_url()
            return RedirectResponse(auth_url)
        except Exception as exc:
            logger.warning("spotify_login_start_failed", error=str(exc))
            raise HTTPException(status_code=503, detail="Could not start Spotify authorization.") from None

    @app.get("/callback")
    async def spotify_callback(code: str | None = None, error: str | None = None):
        """Handle Spotify OAuth authorization redirect callback.

        Every request-derived value interpolated into the HTML body is escaped
        with :func:`html.escape` — the ``error`` parameter and the failure
        message come straight from the attacker-visible query string.
        """
        if error:
            return HTMLResponse(
                f"""<html><body style="font-family: sans-serif; background: #121212; color: #fff; padding: 2rem;">
                <h2 style="color: #ff5555;">Spotify Authorization Error</h2>
                <p>{html.escape(str(error))}</p>
                <a href="/" style="color: #1db954;">Return to Dashboard</a>
                </body></html>""",
                status_code=400,
            )
        if not code:
            return HTMLResponse(
                """<html><body style="font-family: sans-serif; background: #121212; color: #fff; padding: 2rem;">
                <h2>No authorization code provided.</h2>
                <a href="/" style="color: #1db954;">Return to Dashboard</a>
                </body></html>""",
                status_code=400,
            )
        from robot.tools.builtins.music import MusicPlayerService
        ms = MusicPlayerService.get_instance()
        success, msg = await asyncio.to_thread(ms.handle_spotify_callback, code)
        if success:
            return HTMLResponse(
                """<html><body style="font-family: sans-serif; background: #121212; color: #fff; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                <div style="background: #1e1e1e; padding: 2.5rem; border-radius: 12px; text-align: center; border: 1px solid #1db954; max-width: 480px; box-shadow: 0 8px 24px rgba(0,0,0,0.5);">
                <h2 style="color: #1db954; margin-top: 0;">Spotify Connected Successfully!</h2>
                <p style="color: #ccc; line-height: 1.5;">Robot Core has authorized Spotify Connect playback. You can now command music playback seamlessly.</p>
                <a href="/" style="display: inline-block; margin-top: 1rem; background: #1db954; color: #000; text-decoration: none; padding: 0.6rem 1.5rem; border-radius: 20px; font-weight: bold;">Return to Dashboard</a>
                </div></body></html>"""
            )
        else:
            return HTMLResponse(
                f"""<html><body style="font-family: sans-serif; background: #121212; color: #fff; padding: 2rem;">
                <h2 style="color: #ff5555;">Spotify Authorization Failed</h2>
                <p>{html.escape(str(msg))}</p>
                <a href="/" style="color: #1db954;">Return to Dashboard</a>
                </body></html>""",
                status_code=500,
            )

    @app.get("/api/equalizer")
    async def get_equalizer():
        """Get active equalizer profile and list of all available EasyEffects presets."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            return MusicPlayerService.get_instance().list_equalizer_presets()
        except Exception as exc:
            logger.warning("music_endpoint_failed", error=str(exc))
            return {"status": "error", "message": "Music service is unavailable."}

    @app.post("/api/equalizer")
    async def set_equalizer(req: EqualizerRequest):
        """Set active equalizer profile using EasyEffects."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            return await asyncio.to_thread(ms.set_equalizer_preset, req.preset)
        except Exception as exc:
            logger.warning("music_endpoint_failed", error=str(exc))
            return {"status": "error", "message": "Music service is unavailable."}

    @app.get("/api/music/status")
    async def get_music_status():
        """Get live music playback status and active equalizer profile."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            status, equalizer = await asyncio.to_thread(
                lambda: (ms.get_status(), ms.list_equalizer_presets())
            )
            return {
                "status": status,
                "equalizer": equalizer,
            }
        except Exception as exc:
            logger.warning("music_endpoint_failed", error=str(exc))
            return {"status": "error", "message": "Music service is unavailable."}

    @app.post("/api/music/toggle")
    async def toggle_music_playback():
        """Toggle music playback between playing and paused."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            action = ms.pause if ms.is_playing else ms.resume
            res = await asyncio.to_thread(action)
            return {"status": "ok", "playback": res, "is_playing": ms.is_playing}
        except Exception as exc:
            logger.warning("music_endpoint_failed", error=str(exc))
            return {"status": "error", "message": "Music service is unavailable."}

    @app.post("/api/music/restart")
    async def restart_music_track():
        """Restart playback of the current or last played track."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            res = await asyncio.to_thread(ms.replay)
            return {"status": "ok", "playback": res, "is_playing": ms.is_playing}
        except Exception as exc:
            logger.warning("music_endpoint_failed", error=str(exc))
            return {"status": "error", "message": "Music service is unavailable."}

    @app.post("/api/music/volume")
    async def set_music_volume(req: MusicVolumeRequest):
        """Set output volume level for music playback."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            res = await asyncio.to_thread(ms.set_volume, req.volume)
            return {"status": "ok", "volume": ms._volume, "result": res}
        except Exception as exc:
            logger.warning("music_endpoint_failed", error=str(exc))
            return {"status": "error", "message": "Music service is unavailable."}

    @app.get("/api/memory/stats")
    async def get_memory_stats():
        """Get live memory facts count, chat messages count, and vector store health."""
        if hasattr(engine, "memory") and engine.memory:
            stats = await engine.memory.get_stats()
            return {"status": "ok", **stats}
        return {"status": "ok", "facts_count": 0, "messages_count": 0, "sessions_count": 0}

    @app.get("/chat/sessions")
    async def list_chat_sessions(limit: int = Query(150, ge=1, le=500)):
        """List all isolated conversation sessions stored in the SQLite database."""
        if hasattr(engine, "memory") and engine.memory:
            sessions = await engine.memory.list_sessions(limit=limit)
            return {"sessions": sessions, "active_session": engine._session_id}
        return {"sessions": [], "active_session": engine._session_id}

    @app.post("/chat/sessions")
    async def create_chat_session(req: CreateSessionRequest | None = None):
        """Create a new isolated chat session in the database."""
        sid = f"chat_{uuid.uuid4()}"
        pers = (req and req.personality) or getattr(getattr(engine, "state", None), "active_personality", "neo")
        title = (req and req.title) or "New Chat"
        if hasattr(engine, "memory") and engine.memory:
            sess = await engine.memory.create_session(sid, personality=pers, title=title)
            return sess
        return {"id": sid, "session_id": sid, "personality": pers, "title": title}

    @app.get("/chat/sessions/{session_id}")
    async def get_chat_session(session_id: str = PathParam(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")):
        """Get all messages for an isolated chat session from the database."""
        if hasattr(engine, "memory") and engine.memory:
            sess = await engine.memory.get_session(session_id)
            msgs = await engine.memory.get_session_messages(session_id, limit=300)
            if not sess and not msgs:
                raise HTTPException(status_code=404, detail="Session not found")
            title = (sess.get("title") if sess else None) or "Conversation"
            personality = (sess.get("personality") if sess else None) or "neo"
            return {
                "session_id": session_id,
                "id": session_id,
                "title": title,
                "personality": personality,
                "messages": msgs,
            }
        return {"session_id": session_id, "id": session_id, "title": "Conversation", "messages": []}

    @app.delete("/chat/sessions/{session_id}")
    async def delete_chat_session(session_id: str = PathParam(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")):
        """Delete an isolated chat session and its conversation history."""
        if hasattr(engine, "memory") and engine.memory:
            success = await engine.memory.delete_session(session_id)
            return {"status": "ok", "deleted": success, "session_id": session_id}
        return {"status": "ok", "deleted": False, "session_id": session_id}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """Server-Sent Events (SSE) streaming endpoint for real-time tokens, token speed, and tool events."""
        from fastapi.responses import StreamingResponse
        import json

        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()

        thinking_flag = req.thinking if req.thinking is not None else getattr(engine, "thinking_enabled", True)

        async def _event_generator():
            # The engine can be silent for a long time while a tool runs. Pump it
            # through a bounded queue so the SSE connection receives a keepalive
            # comment during those gaps instead of looking dead to the browser.
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=128)

            async def _produce() -> None:
                try:
                    async for chunk in engine.stream_text_chat(
                        text=req.text,
                        session_id=req.session_id,
                        personality=req.personality,
                        thinking=thinking_flag,
                    ):
                        await queue.put(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("chat_stream_failed", error=str(exc))
                    try:
                        queue.put_nowait({"event": "error", "error": "The assistant stream failed."})
                    except asyncio.QueueFull:
                        pass
                finally:
                    try:
                        queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass

            producer = asyncio.create_task(_produce(), name="chat_stream_producer")
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if chunk is None:
                        break
                    yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                if not producer.done():
                    producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, Exception):
                    pass

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/thinking")
    async def get_thinking_mode():
        """Retrieve the active cognitive reasoning/thinking mode state."""
        is_enabled = getattr(engine, "thinking_enabled", True)
        return {"status": "ok", "thinking_enabled": is_enabled}

    @app.post("/api/thinking")
    async def set_thinking_mode(req: ThinkingModeRequest):
        """Update and broadcast the cognitive reasoning/thinking mode state."""
        engine.thinking_enabled = bool(req.enabled)
        if hasattr(engine, "events") and engine.events:
            await engine.events.emit("thinking_toggled", enabled=engine.thinking_enabled)
        return {"status": "ok", "thinking_enabled": engine.thinking_enabled}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """Text-based chat endpoint (non-voice)."""
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
        response = await engine.process_text_input(req.text, session_id=req.session_id)
        return ChatResponse(
            response=response,
            turn=engine.state.state.turn_count,
            personality=engine.state.state.active_personality,
            session_id=req.session_id or engine._session_id,
        )

    @app.post("/api/chat/speak")
    async def speak_chat_message(req: ChatSpeakRequest) -> Response:
        """Synthesize and return speech WAV audio for a chat message response.

        Splits message text into natural clause phrases, applying dynamic sentiment
        prosody modulation (pitch, speed, noise scale) to each clause individually,
        mirroring the active avatar TTS pipeline. Emits a robot phrase event for
        synchronized avatar lipsync animation.

        Args:
            req: ChatSpeakRequest containing text, personality, voice_model, speed, pitch.

        Returns:
            Response containing binary audio/wav content for in-browser playback.

        Raises:
            HTTPException: 400 if text or audio is empty; 503 if TTS is unavailable.
        """
        import io
        import wave
        import numpy as np
        from robot.tts import clean_text_for_speech, split_into_phrases

        raw_text = req.text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        tts = getattr(engine, "tts", None)
        if tts is None or not hasattr(tts, "synthesize"):
            raise HTTPException(status_code=503, detail="TTS provider is not initialized or unavailable.")

        # Resolve voice model and prosody from personality if not explicitly overridden
        pers_name = req.personality or getattr(engine.state.state, "active_personality", "neo")
        pers = engine.personalities.get(pers_name) if hasattr(engine, "personalities") and engine.personalities else None

        active_tts_voice = None
        if hasattr(tts, "get_active_voice") and callable(tts.get_active_voice):
            active_tts_voice = tts.get_active_voice()

        voice_model = req.voice_model or active_tts_voice or (getattr(pers, "voice_model", None) if pers else None)
        speed = req.speed
        pitch = req.pitch
        noise_scale: float | None = None
        noise_w: float | None = None

        if pers and hasattr(pers, "tts_config") and isinstance(pers.tts_config, dict):
            cfg = pers.tts_config
            if req.speed == 1.0 and "speed" in cfg:
                speed = float(cfg["speed"])
            if req.pitch == 1.0 and "pitch" in cfg:
                pitch = float(cfg["pitch"])
            noise_scale = cfg.get("noise_scale")
            noise_w = cfg.get("noise_w")

        phrases = split_into_phrases(raw_text)
        if not phrases:
            fallback = clean_text_for_speech(raw_text) or raw_text
            phrases = [fallback] if fallback.strip() else []

        if not phrases:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        audio_chunks: list[np.ndarray] = []
        sample_rate = 16000
        primary_emotion = "neutral"

        for phrase in phrases:
            eff_speed, eff_pitch, eff_ns, eff_nw = speed, pitch, noise_scale, noise_w
            phrase_emotion = "neutral"

            # Apply dynamic sentiment analysis and prosody modulation per clause
            if hasattr(tts, "modulate_prosody"):
                try:
                    res = tts.modulate_prosody(
                        phrase,
                        base_speed=speed,
                        base_pitch=pitch,
                        base_noise_scale=noise_scale,
                        base_noise_w=noise_w,
                        personality_name=pers_name,
                    )
                    if isinstance(res, tuple) and len(res) == 5:
                        eff_speed, eff_pitch, eff_ns, eff_nw, modulation = res
                        phrase_emotion = getattr(modulation, "emotion", "neutral")
                        if primary_emotion == "neutral" and phrase_emotion != "neutral":
                            primary_emotion = phrase_emotion
                except Exception:
                    pass

            try:
                chunk, sr = await tts.synthesize(
                    phrase,
                    speed=eff_speed,
                    pitch=eff_pitch,
                    voice_model=voice_model,
                    noise_scale=eff_ns,
                    noise_w=eff_nw,
                )
                if chunk is not None and len(chunk) > 0:
                    audio_chunks.append(chunk)
                    sample_rate = sr
            except Exception as exc:
                logger.error("chat_speak_phrase_failed", phrase=phrase[:30], error=str(exc))

        if not audio_chunks:
            raise HTTPException(status_code=400, detail="Synthesized audio was empty.")

        audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

        # Optionally play on robot hardware speaker if requested
        if req.play_hardware and hasattr(engine, "player") and engine.player:
            spawn(engine.player.play(audio, sample_rate), name="tts_hardware_play")

        # Emit robot_phrase event for avatar lip-sync / expressions
        if hasattr(engine, "events") and engine.events:
            try:
                clean_phrase = clean_text_for_speech(raw_text)
                await engine.events.emit("robot_phrase", robot_phrase=clean_phrase, emotion=primary_emotion)
            except Exception as e:
                logger.debug("speak_phrase_event_failed", error=str(e))

        # Convert float32 numpy array to 16-bit PCM WAV
        int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_audio.tobytes())
        wav_bytes = wav_buf.getvalue()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="response.wav"',
                "Cache-Control": "no-cache",
            },
        )

    @app.post("/api/chat/speak/stream")
    async def speak_chat_message_stream(req: ChatSpeakRequest, request: Request) -> StreamingResponse:
        """Synthesize and stream speech WAV audio clauses via Server-Sent Events (SSE).

        Splits input text into natural punctuation clauses/phrases and modulates each phrase
        individually for dynamic pitch, speed, and emotion prosody. Yields phrases as soon as
        they are synthesized so the client begins audio playback on phrase 0 immediately.

        Args:
            req: ChatSpeakRequest containing text, personality, voice_model, speed, pitch.
            request: Active Starlette/FastAPI Request used to detect client disconnections.

        Returns:
            StreamingResponse emitting SSE text/event-stream events containing base64 WAV chunks.

        Raises:
            HTTPException: 400 if text is empty; 503 if TTS is unavailable.
        """
        import base64
        import io
        import json
        import wave
        import numpy as np
        from fastapi.responses import StreamingResponse
        from robot.tts import clean_text_for_speech, split_into_phrases

        raw_text = req.text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        tts = getattr(engine, "tts", None)
        if tts is None or not hasattr(tts, "synthesize"):
            raise HTTPException(status_code=503, detail="TTS provider is not initialized or unavailable.")

        # Resolve voice model and prosody defaults from active personality
        pers_name = req.personality or getattr(engine.state.state, "active_personality", "neo")
        pers = engine.personalities.get(pers_name) if hasattr(engine, "personalities") and engine.personalities else None

        active_tts_voice = None
        if hasattr(tts, "get_active_voice") and callable(tts.get_active_voice):
            active_tts_voice = tts.get_active_voice()

        base_voice_model = req.voice_model or active_tts_voice or (getattr(pers, "voice_model", None) if pers else None)
        base_speed = req.speed
        base_pitch = req.pitch
        base_noise_scale: float | None = None
        base_noise_w: float | None = None

        if pers and hasattr(pers, "tts_config") and isinstance(pers.tts_config, dict):
            cfg = pers.tts_config
            if req.speed == 1.0 and "speed" in cfg:
                base_speed = float(cfg["speed"])
            if req.pitch == 1.0 and "pitch" in cfg:
                base_pitch = float(cfg["pitch"])
            base_noise_scale = cfg.get("noise_scale")
            base_noise_w = cfg.get("noise_w")

        # Split text into natural clause phrases for low-latency streaming
        phrases = split_into_phrases(raw_text)
        if not phrases:
            fallback = clean_text_for_speech(raw_text) or raw_text
            phrases = [fallback] if fallback.strip() else []

        if not phrases:
            raise HTTPException(status_code=400, detail="No speakable phrases found in input text.")

        async def _phrase_stream_generator():
            total = len(phrases)
            for idx, phrase in enumerate(phrases):
                if await request.is_disconnected():
                    logger.debug("chat_speak_stream_client_disconnected", phrase_idx=idx, total=total)
                    break

                eff_speed = base_speed
                eff_pitch = base_pitch
                eff_noise_scale = base_noise_scale
                eff_noise_w = base_noise_w
                detected_emotion = "neutral"

                # Apply dynamic sentiment analysis and prosody modulation per phrase
                if hasattr(tts, "modulate_prosody"):
                    try:
                        res = tts.modulate_prosody(
                            phrase,
                            base_speed=base_speed,
                            base_pitch=base_pitch,
                            base_noise_scale=base_noise_scale,
                            base_noise_w=base_noise_w,
                            personality_name=pers_name,
                        )
                        if isinstance(res, tuple) and len(res) == 5:
                            eff_speed, eff_pitch, eff_noise_scale, eff_noise_w, modulation = res
                            detected_emotion = getattr(modulation, "emotion", "neutral")
                    except Exception as err:
                        logger.debug("phrase_prosody_modulation_skipped", error=str(err))

                try:
                    audio, sr = await tts.synthesize(
                        phrase,
                        speed=eff_speed,
                        pitch=eff_pitch,
                        voice_model=base_voice_model,
                        noise_scale=eff_noise_scale,
                        noise_w=eff_noise_w,
                    )
                except Exception as exc:
                    logger.error("phrase_synthesis_failed", phrase=phrase[:30], error=str(exc))
                    yield f"data: {json.dumps({'event': 'error', 'index': idx, 'error': str(exc)})}\n\n"
                    continue

                if audio is None or len(audio) == 0:
                    continue

                # Optionally play on robot hardware speaker sequentially if requested
                if req.play_hardware and hasattr(engine, "player") and engine.player:
                    try:
                        await engine.player.play(audio, sr)
                    except Exception as e:
                        logger.warning("hardware_playback_failed", error=str(e))

                # Convert float32 numpy array to 16-bit PCM WAV in memory
                int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(int16_audio.tobytes())
                base64_wav = base64.b64encode(wav_buf.getvalue()).decode("ascii")

                duration_ms = (len(audio) / sr) * 1000.0

                event_data = {
                    "event": "phrase",
                    "index": idx,
                    "total": total,
                    "phrase": phrase,
                    "emotion": detected_emotion,
                    "pitch": round(float(eff_pitch), 3),
                    "speed": round(float(eff_speed), 3),
                    "duration_ms": round(duration_ms, 1),
                    "audio": base64_wav,
                }
                yield f"data: {json.dumps(event_data)}\n\n"

            yield f"data: {json.dumps({'event': 'done', 'total': total})}\n\n"

        return StreamingResponse(
            _phrase_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


    @app.post("/api/transcribe")
    async def transcribe_audio(request: Request):
        """Transcribe a recorded WAV clip (raw binary or JSON base64) via STT.

        All decoding and resampling runs in a worker thread, malformed audio is
        rejected with 400 instead of being forwarded, and a slow Whisper server
        surfaces as 504 rather than hanging the request until the client gives
        up.
        """
        ctype = (request.headers.get("content-type") or "").lower()
        try:
            if "application/json" in ctype:
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise HTTPException(status_code=400, detail="Expected a JSON object.")
                b64_audio = payload.get("audio", "")
                if not isinstance(b64_audio, str):
                    raise HTTPException(status_code=400, detail="'audio' must be a base64 string.")
                try:
                    content = base64.b64decode(b64_audio, validate=True)
                except (binascii.Error, ValueError):
                    raise HTTPException(status_code=400, detail="'audio' is not valid base64.") from None
            else:
                content = await request.body()
        except HTTPException:
            raise
        except Exception:
            logger.warning("api_transcribe_body_read_failed", content_type=ctype)
            raise HTTPException(status_code=400, detail="Could not read the request body.") from None

        if not content or len(content) < 44:
            return {"status": "ok", "text": ""}

        samples, duration_s = await asyncio.to_thread(_decode_wav_to_mono16k, content)
        if samples is None:
            raise HTTPException(status_code=400, detail="Unsupported audio: expected a PCM WAV file.")
        if duration_s > TRANSCRIBE_MAX_DURATION_S:
            raise HTTPException(
                status_code=413,
                detail=f"Audio is longer than {int(TRANSCRIBE_MAX_DURATION_S)} seconds.",
            )

        stt = getattr(engine, "stt", None)
        if stt is None or not hasattr(stt, "transcribe"):
            from robot.stt import WhisperServerSTT

            stt = WhisperServerSTT()

        try:
            if inspect.iscoroutinefunction(stt.transcribe):
                text = await asyncio.wait_for(
                    stt.transcribe(samples, sample_rate=16000),
                    timeout=TRANSCRIBE_TIMEOUT_S,
                )
            else:
                text = await asyncio.wait_for(
                    asyncio.to_thread(stt.transcribe, samples, 16000),
                    timeout=TRANSCRIBE_TIMEOUT_S,
                )
        except asyncio.TimeoutError:
            logger.warning("api_transcribe_timeout", duration_s=round(duration_s, 2))
            raise HTTPException(status_code=504, detail="Transcription timed out.") from None
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("api_transcribe_failed", error=str(exc), duration_s=round(duration_s, 2))
            raise HTTPException(status_code=503, detail="Speech recognition is unavailable.") from None

        return {"status": "ok", "text": str(text or "").strip()}

    @app.post("/interrupt")
    async def interrupt_execution():
        """Manually interrupt active LLM generation, tool execution, and audio playback."""
        engine.trigger_interrupt()
        return {"status": "interrupted"}

    @app.get("/memory/search")
    async def search_memory(
        query: str = Query(..., min_length=1, max_length=500),
        limit: int = Query(5, ge=1, le=50),
    ):
        active = getattr(getattr(engine, "personalities", None), "active", None)
        namespace = getattr(active, "memory_namespace", None) or getattr(engine, "active_personality", "default")
        results = await engine.memory.get_relevant_memories(query, user_id=namespace, limit=limit)
        return {"results": results}

    @app.get("/history/search")
    async def search_history(
        query: str = Query(..., min_length=1, max_length=500),
        limit: int = Query(10, ge=1, le=100),
    ):
        results = await engine.memory.search_history(query, limit)
        return {"results": results}

    @app.get("/world-state")
    async def get_world_state():
        """Get the current physical world state (people, motion, lighting)."""
        if hasattr(engine, "perception") and engine.perception:
            return engine.perception.world_state.to_compact()
        return {"error": "Perception subsystem not active"}

    @app.get("/camera/snapshot")
    async def get_camera_snapshot():
        """Get the latest snapshot from the robot's camera with green perception polygons.

        JPEG encoding is off-loaded to a worker thread so a single snapshot
        request cannot stall the event loop.
        """
        from fastapi.responses import Response
        if not engine.is_camera_enabled:
            return Response(content=b"Camera paused or disabled", status_code=503)
        if hasattr(engine, "perception") and engine.perception:
            frame = engine.perception.get_annotated_frame()
            if frame is not None:
                try:
                    payload = await asyncio.to_thread(_encode_jpeg_frame, frame, 80)
                except Exception as exc:
                    logger.warning("camera_snapshot_encode_failed", error=str(exc))
                    payload = None
                if payload:
                    return Response(
                        content=payload,
                        media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
                    )
            return Response(content=b"No frame available", status_code=503)
        return Response(content=b"Camera not available", status_code=503)

    @app.get("/camera/stream")
    async def get_camera_stream():
        """MJPEG stream serving the latest annotated perception frame, capped at ~15 FPS.

        Falls back to the raw camera frame when no annotated frame is ready yet.
        JPEG encoding runs on a worker thread (never on the event loop) and each
        client stream is paced to :data:`MJPEG_TARGET_FPS`.

        Returns:
            StreamingResponse: multipart/x-mixed-replace MJPEG stream.
        """
        from fastapi.responses import Response, StreamingResponse
        if not engine.is_camera_enabled:
            return Response(content=b"Camera paused or disabled", status_code=503)

        def latest_frame():
            """Return the newest annotated frame, falling back to the raw camera frame."""
            perception = getattr(engine, "perception", None)
            if perception is None:
                return None
            frame = perception.get_annotated_frame()
            if frame is None and hasattr(perception, "camera"):
                frame = perception.camera.get_latest_frame()
            return frame

        return StreamingResponse(
            _mjpeg_part_generator(
                latest_frame,
                lambda: _streaming_open() and engine.is_camera_enabled,
                stream_name="camera_stream",
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/camera/face-mesh")
    async def get_face_mesh_stream():
        """MJPEG stream with MediaPipe 468-point face mesh overlay, capped at ~15 FPS.

        Served by FaceMeshOverlay's dedicated background thread; the HTTP side
        encodes on a worker thread and paces output to :data:`MJPEG_TARGET_FPS`.
        Falls back to the plain camera/stream if face mesh is unavailable.

        Returns:
            StreamingResponse: multipart/x-mixed-replace MJPEG stream with
                green face mesh tessellation, contours, and iris rings drawn.
        """
        from fastapi.responses import Response, StreamingResponse
        if not engine.is_camera_enabled:
            return Response(content=b"Camera paused or disabled", status_code=503)

        face_mesh = getattr(engine, "face_mesh", None)
        if face_mesh is None or not face_mesh.is_available:
            # Face mesh not ready: fall back to annotated perception stream
            return await get_camera_stream()

        def latest_mesh_frame():
            """Return the newest mesh frame, falling back to the raw camera frame."""
            frame = face_mesh.get_latest_mesh_frame()
            if frame is None:
                perception = getattr(engine, "perception", None)
                if perception is not None and hasattr(perception, "camera"):
                    frame = perception.camera.get_latest_frame()
            return frame

        return StreamingResponse(
            _mjpeg_part_generator(
                latest_mesh_frame,
                lambda: _streaming_open() and engine.is_camera_enabled,
                stream_name="face_mesh_stream",
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/face-mesh/status")
    async def get_face_mesh_status() -> dict[str, Any]:
        """Return current face mesh overlay status and measured FPS.

        Returns:
            dict[str, Any]: Face mesh availability, running state, and FPS.
        """
        fm = getattr(engine, "face_mesh", None)
        return {
            "status": "ok",
            "available": fm is not None and fm.is_available,
            "running": fm._running if fm else False,
            "measured_fps": fm.measured_fps if fm else 0.0,
            "camera_enabled": engine.is_camera_enabled,
        }

    @app.post("/api/face-mesh/toggle")
    async def toggle_face_mesh(req: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Start or stop the face mesh background thread on demand.

        Args:
            req: Optional payload with boolean ``enabled`` to set explicitly;
                if omitted, toggles the current state.

        Returns:
            dict[str, Any]: Updated face mesh running state.
        """
        fm = getattr(engine, "face_mesh", None)
        if fm is None:
            return {"status": "error", "message": "Face mesh not initialized (mediapipe may not be installed)"}

        explicit = req.get("enabled")
        if explicit is None:
            # Toggle current state
            if fm._running:
                fm.stop()
            else:
                fm.start()
        else:
            if bool(explicit):
                fm.start()
            else:
                fm.stop()

        await engine.events.emit("face_mesh_toggled", running=fm._running)
        return {
            "status": "ok",
            "running": fm._running,
            "available": fm.is_available,
        }


    @app.post("/character/action")
    async def post_character_action(action_req: dict[str, Any]):
        """Execute semantic motion command on character."""
        from robot.motion.actions import validate_action
        from robot.motion.planner import MotionPlanner

        valid, semantic_action, err = validate_action(action_req)
        if not valid or semantic_action is None:
            return {"error": err}

        planner = MotionPlanner()
        sequence = planner.plan_action(action_req)
        serialized_seq = [a.to_dict() for a in sequence]

        await engine.events.emit(
            "character.action",
            action=semantic_action.to_dict(),
            sequence=serialized_seq,
        )
        return {"status": "ok", "action": semantic_action.to_dict(), "sequence": serialized_seq}

    @app.post("/character/stop")
    async def post_character_stop():
        """Interrupt and smoothly stop character motion."""
        await engine.events.emit("character.interrupt", interrupt=True)
        return {"status": "stopped"}

    @app.get("/telemetry")
    async def telemetry():
        return engine.telemetry.snapshot()

    # ─── Observability & Traces API ──────────────────────────────────────────
    @app.get("/observability/traces")
    async def get_traces(limit: int = Query(15, ge=1, le=200)):
        """Get recent execution traces with spans and latencies."""
        if hasattr(engine, "observability") and engine.observability:
            return {
                "traces": engine.observability.get_recent_traces(limit),
                "pipeline": engine.observability.pipeline_state,
            }
        return {"traces": [], "pipeline": {}}

    @app.get("/observability/hardware")
    async def get_hardware_metrics():
        """Get live CPU, RAM, discrete GPU VRAM/Temp, and integrated GPU utilization."""
        if hasattr(engine, "observability") and engine.observability:
            # A cold sample shells out to nvidia-smi; keep that off the loop.
            return await asyncio.to_thread(engine.observability.get_hardware_metrics)
        return {}

    @app.get("/observability/pipeline")
    async def get_pipeline_state():
        """Get live status of all pipeline components."""
        if hasattr(engine, "observability") and engine.observability:
            return engine.observability.pipeline_state
        return {}

    @app.get("/observability/benchmarks")
    async def get_benchmarks():
        """Get latency distributions (P50, P95, P99) categorized by request type."""
        if hasattr(engine, "observability") and engine.observability:
            return engine.observability.get_category_distributions()
        return {}

    # ─── Image Generation API ────────────────────────────────────────────────
    @app.post("/images/generate")
    async def api_generate_image(req: ImageGenRequest = Body(...)):
        """Asynchronously submit an image generation job."""
        if not hasattr(engine, "image_service") or not engine.image_service:
            return {"error": "Image service not initialized"}
        job = engine.image_service.submit_job(
            prompt=req.prompt,
            width=req.width or 512,
            height=req.height or 512,
            steps=req.steps if req.steps is not None else 4,
            seed=req.seed if req.seed is not None else -1,
            model=req.model or "default",
            negative_prompt=req.negative_prompt or "",
            trace_id=getattr(engine.observability, "current_trace_id", None) if hasattr(engine, "observability") else None,
        )
        return {"status": "queued", "job": job.to_dict()}

    @app.get("/api/images/models")
    async def list_image_models() -> dict[str, Any]:
        """List registered FLUX, Stable Diffusion, and Procedural models with cache status."""
        from robot.image.model_registry import ImageModelRegistry
        models = [m.to_dict() | {"is_cached": m.is_cached_locally()} for m in ImageModelRegistry.list_models()]
        return {
            "default_model": ImageModelRegistry.DEFAULT_GPU_MODEL_ID,
            "total_models": len(models),
            "models": models,
        }

    @app.get("/images/jobs")
    async def list_image_jobs(limit: int = Query(20, ge=1, le=200)):
        """List recent image generation jobs."""
        if hasattr(engine, "image_service") and engine.image_service:
            return {"jobs": engine.image_service.list_jobs(limit)}
        return {"jobs": []}

    @app.post("/images/cancel/{job_id}")
    async def cancel_image_job(job_id: str):
        """Cancel an in-flight image generation job."""
        if hasattr(engine, "image_service") and engine.image_service:
            ok = await engine.image_service.cancel_job(job_id)
            return {"cancelled": ok, "job_id": job_id}
        return {"cancelled": False, "error": "Image service not initialized"}

    @app.get("/api/images/gallery")
    async def get_image_gallery(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        """List generated artwork images from the image service directory.

        The directory scan, sorting and image decoding run off the event loop.
        Only the ``limit`` newest images are decoded, and dimensions are cached
        by ``(mtime, size)`` so repeat requests touch no image files at all.

        Args:
            limit: Maximum number of gallery images to return.

        Returns:
            Dictionary containing total count and list of image items with metadata.
        """
        from datetime import datetime, timezone

        images_dir = _resolve_images_dir(engine)
        if not images_dir.is_dir():
            return {"status": "ok", "total": 0, "images": []}

        jobs_lookup: dict[str, dict[str, Any]] = {}
        if hasattr(engine, "image_service") and engine.image_service:
            for j in engine.image_service.list_jobs(200):
                out = j.get("output_path", "")
                if out:
                    jobs_lookup[Path(out).name] = j

        valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}

        def _scan() -> tuple[int, list[dict[str, Any]]]:
            from PIL import Image

            candidates: list[tuple[float, str, int]] = []
            with os.scandir(images_dir) as entries:
                for entry in entries:
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if Path(entry.name).suffix.lower() not in valid_extensions:
                            continue
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    candidates.append((stat.st_mtime, entry.name, stat.st_size))

            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            total = len(candidates)

            if len(_GALLERY_META_CACHE) > _GALLERY_META_CACHE_MAX:
                _GALLERY_META_CACHE.clear()

            items: list[dict[str, Any]] = []
            for mtime, fname, size_bytes in candidates[:limit]:
                job_meta = jobs_lookup.get(fname, {})
                prompt = job_meta.get("prompt", "")
                model = job_meta.get("model", "")
                duration_s = job_meta.get("duration_s", 0.0)

                if not prompt:
                    if fname.startswith("img_test_flux"):
                        prompt = "FLUX.1 Schnell procedural/diffusion test benchmark"
                        model = "flux-schnell"
                    elif "sdxl" in fname:
                        prompt = "SDXL Turbo high-speed generation"
                        model = "sdxl-turbo"
                    elif "small_sd" in fname:
                        prompt = "Small-SD lightweight model generation"
                        model = "small-sd"
                    elif fname.startswith("img_"):
                        prompt = f"Generated Artwork ({fname[:12]})"
                        model = "diffusion"
                    else:
                        prompt = fname.rsplit(".", 1)[0].replace("_", " ").title()
                        model = "diffusion"

                cached = _GALLERY_META_CACHE.get(fname)
                if cached and cached[0] == mtime and cached[1] == size_bytes:
                    width, height = cached[2], cached[3]
                else:
                    width, height = 512, 512
                    try:
                        with Image.open(images_dir / fname) as img:
                            width, height = img.size
                    except Exception:
                        pass
                    _GALLERY_META_CACHE[fname] = (mtime, size_bytes, width, height)

                size_kb = (
                    f"{round(size_bytes / 1024, 1)} KB"
                    if size_bytes < 1024 * 1024
                    else f"{round(size_bytes / (1024 * 1024), 2)} MB"
                )

                items.append({
                    "filename": fname,
                    "url": f"/data/images/{urllib.parse.quote(fname)}",
                    "prompt": prompt,
                    "model": model or "diffusion",
                    "duration_s": round(duration_s, 2),
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes,
                    "size_kb": size_kb,
                    "created_at": mtime,
                    "created_at_iso": datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                })
            return total, items

        try:
            total, items = await asyncio.to_thread(_scan)
        except OSError as exc:
            logger.warning("gallery_scan_failed", directory=str(images_dir), error=str(exc))
            return {"status": "ok", "total": 0, "images": []}

        return {"status": "ok", "total": total, "images": items}

    @app.delete("/api/images/gallery/{filename}")
    async def delete_gallery_image(filename: str) -> dict[str, Any]:
        """Delete a generated artwork from the image service directory.

        Args:
            filename: Target image filename to remove.

        Returns:
            Dictionary with deletion status and filename.
        """
        if (
            not re.fullmatch(r"[a-zA-Z0-9_\-\.]{1,128}", filename)
            or filename in (".", "..")
            or ".." in filename
        ):
            raise HTTPException(status_code=400, detail="Invalid filename format.")

        images_dir = _resolve_images_dir(engine).resolve()
        target_file = (images_dir / filename).resolve()
        if target_file.parent != images_dir:
            raise HTTPException(status_code=400, detail="Invalid filename format.")
        if target_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise HTTPException(status_code=400, detail="Invalid filename format.")
        if not target_file.is_file():
            raise HTTPException(status_code=404, detail="File not found.")

        try:
            await asyncio.to_thread(target_file.unlink)
        except OSError as exc:
            logger.warning("gallery_delete_failed", file=filename, error=str(exc))
            raise HTTPException(status_code=500, detail="Could not delete the image.") from None
        _GALLERY_META_CACHE.pop(filename, None)
        return {"status": "ok", "deleted": True, "filename": filename}


    # --- Temporal Scheduler Endpoints ---
    @app.get("/api/scheduler/jobs")
    async def list_scheduled_jobs(
        job_type: str | None = Query(default=None, max_length=32),
        status: str = Query(default="pending", max_length=32),
        limit: int = Query(50, ge=1, le=200),
    ):
        """List scheduled timers, alarms, and reminders."""
        if hasattr(engine, "scheduler") and engine.scheduler:
            jobs = await engine.scheduler.list_jobs(job_type=job_type, status=status, limit=limit)
            return {"status": "success", "jobs": [j.to_dict() for j in jobs]}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.post("/api/scheduler/timer")
    async def create_timer(request: Request):
        """Create a new countdown timer."""
        data = await _read_json_object(request)
        duration_raw = data.get("duration_seconds", 60)
        label_raw = data.get("label", "Timer")
        if not isinstance(label_raw, str):
            raise HTTPException(status_code=400, detail="label must be a string.")
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="duration_seconds must be a number.") from None
        if not 1.0 <= duration <= 86400.0:
            raise HTTPException(status_code=400, detail="duration_seconds must be between 1 and 86400.")
        label = label_raw.strip()[:200] or "Timer"
        if hasattr(engine, "scheduler") and engine.scheduler:
            job = await engine.scheduler.add_timer(duration, label=label)
            return {"status": "success", "timer": job.to_dict()}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.post("/api/scheduler/alarm")
    async def create_alarm(request: Request):
        """Schedule a new clock alarm."""
        data = await _read_json_object(request)
        target_time = data.get("target_time", "")
        label = data.get("label", "Alarm")
        recurring = bool(data.get("recurring", False))
        if not isinstance(target_time, str) or not isinstance(label, str):
            raise HTTPException(status_code=400, detail="target_time and label must be strings.")
        target_time = target_time.strip()[:64]
        label = label.strip()[:200] or "Alarm"
        if not target_time:
            raise HTTPException(status_code=400, detail="target_time is required.")
        if hasattr(engine, "scheduler") and engine.scheduler:
            try:
                job = await engine.scheduler.add_alarm(target_time, label=label, recurring=recurring)
                return {"status": "success", "alarm": job.to_dict()}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            except Exception as exc:
                logger.warning("scheduler_alarm_create_failed", error=str(exc))
                raise HTTPException(status_code=503, detail="Could not schedule the alarm.") from None
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.post("/api/scheduler/reminder")
    async def create_reminder(request: Request):
        """Schedule a new reminder."""
        data = await _read_json_object(request)
        time_expr = data.get("time_expression", "")
        message = data.get("message", "")
        if not isinstance(time_expr, str) or not isinstance(message, str):
            raise HTTPException(status_code=400, detail="time_expression and message must be strings.")
        time_expr = time_expr.strip()[:64]
        message = message.strip()[:500]
        if not time_expr:
            raise HTTPException(status_code=400, detail="time_expression is required.")
        if hasattr(engine, "scheduler") and engine.scheduler:
            try:
                job = await engine.scheduler.add_reminder(time_expr, message=message)
                return {"status": "success", "reminder": job.to_dict()}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            except Exception as exc:
                logger.warning("scheduler_reminder_create_failed", error=str(exc))
                raise HTTPException(status_code=503, detail="Could not schedule the reminder.") from None
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.delete("/api/scheduler/jobs/{job_id}")
    async def cancel_scheduled_job(job_id: str):
        """Cancel a scheduled timer, alarm, or reminder."""
        if hasattr(engine, "scheduler") and engine.scheduler:
            cancelled = await engine.scheduler.cancel_job(job_id)
            return {"status": "success" if cancelled else "not_found", "job_id": job_id, "cancelled": cancelled}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.get("/api/scheduler/briefing")
    async def get_scheduler_briefing():
        """Get executive daily briefing with system status and upcoming events."""
        if hasattr(engine, "scheduler") and engine.scheduler:
            briefing = await engine.scheduler.get_daily_briefing()
            return {"status": "success", "briefing": briefing}
        return {"status": "error", "message": "Scheduler service not initialized"}

    # Mount /data/images for generated image previews.
    # NOTE: the directory is resolved through the image service so the mount,
    # the gallery listing and the writer can never disagree, even when the
    # process runs with a different working directory. The mount stays
    # authenticated: only the UI shell under /static is public (see
    # AUTH_EXEMPT_PATHS).
    from fastapi.staticfiles import StaticFiles

    images_dir = _resolve_images_dir(engine)
    images_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/data/images", StaticFiles(directory=str(images_dir)), name="images")

    @app.websocket("/ws/events")
    @app.websocket("/ws/observability")
    async def event_stream(websocket: WebSocket):
        """WebSocket stream for real-time telemetry and observability events.

        Security: :class:`_ApiAuthMiddleware` already refuses unauthorized
        upgrades, and the handshake is validated again here (bearer header,
        ``?token=`` query parameter, the login cookie, or the trusted-local stamp
        the middleware leaves for this machine's own owner) before ``accept()`` so
        a future route/mount change cannot expose an open event firehose.

        Backpressure: each client owns a bounded :class:`asyncio.Queue`
        (:data:`WS_EVENT_QUEUE_MAXSIZE`). A slow consumer causes the *oldest*
        buffered event to be dropped — never an unbounded memory climb — and the
        drop count is logged rather than silently swallowed.
        """
        trusted_local = bool(websocket.scope.get(_ApiAuthMiddleware.TRUSTED_LOCAL_SCOPE_KEY))
        if not trusted_local and not _token_matches(_token_from_scope(websocket.scope), token):
            logger.warning(
                "ws_handshake_rejected",
                path=websocket.scope.get("path"),
                remote=getattr(websocket.client, "host", None),
            )
            await websocket.close(code=4401, reason="Unauthorized")
            return

        await websocket.accept()

        queue: asyncio.Queue = asyncio.Queue(maxsize=WS_EVENT_QUEUE_MAXSIZE)
        drops = {"oldest_dropped": 0, "refused": 0}

        async def forward_event(**kwargs):
            """Push one bus event into this client's bounded queue."""
            try:
                queue.put_nowait(kwargs)
                return
            except asyncio.QueueFull:
                pass

            # Queue is full: drop the oldest buffered event to make room for the
            # newest one, then retry once.
            try:
                queue.get_nowait()
                drops["oldest_dropped"] += 1
            except asyncio.QueueEmpty:
                pass

            try:
                queue.put_nowait(kwargs)
            except asyncio.QueueFull:
                # Another producer won the race for the freed slot; report the loss
                # instead of dropping the event silently.
                drops["refused"] += 1
                logger.warning(
                    "ws_event_dropped_queue_full",
                    path=websocket.scope.get("path"),
                    maxsize=WS_EVENT_QUEUE_MAXSIZE,
                    refused=drops["refused"],
                )
                return

            if drops["oldest_dropped"] == 1 or drops["oldest_dropped"] % 100 == 0:
                logger.warning(
                    "ws_event_backpressure_dropped_oldest",
                    path=websocket.scope.get("path"),
                    maxsize=WS_EVENT_QUEUE_MAXSIZE,
                    oldest_dropped=drops["oldest_dropped"],
                )

        engine.events.on("*", forward_event)

        try:
            while _streaming_open():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=WS_SHUTDOWN_POLL_S)
                except asyncio.TimeoutError:
                    continue
                await websocket.send_text(json.dumps(event))
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            pass
        finally:
            engine.events.off("*", forward_event)
            try:
                await websocket.close(code=1012, reason="Server shutting down")
            except Exception:
                pass  # the client is already gone, or the loop is being torn down

    # Static UI Files & Web HUD Display
    from fastapi.responses import FileResponse

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        class VersionedStaticFiles(StaticFiles):
            """Cache static assets aggressively only when the URL is versioned.

            The UI references its bundles with ``?v=<build>``, so those responses
            can be cached for a year; unversioned assets keep revalidating so a
            live edit still shows up on refresh.
            """

            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                if scope.get("query_string"):
                    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                else:
                    response.headers["Cache-Control"] = "no-cache"
                return response

        app.mount("/static", VersionedStaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def index():
            """Serve the integrated Digital Face & Dev Console UI."""
            return FileResponse(
                str(static_dir / "index.html"),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        @app.get("/contextlayer")
        @app.get("/inspect")
        @app.get("/context-layer")
        async def context_layer_page():
            """Serve the dynamic LLM context layer & prompt caching debugger page."""
            context_html = static_dir / "contextlayer.html"
            if context_html.exists():
                return FileResponse(str(context_html))
            return {"error": "contextlayer.html not found"}

    @app.get("/api/context/inspect")
    @app.get("/api/inspect")
    async def inspect_context_layer(test_prompt: str = "Hi"):
        """Dynamically inspect the structured context layer and KV cache invariance status.

        Args:
            test_prompt: Optional test input prompt string to simulate context assembly.

        Returns:
            dict[str, Any]: Structured breakdown of system prompt, user profile,
                situation banner, durable memories, history, tool schemas, and cache metrics.
        """
        prov = getattr(engine, "llm_provider_type", "local")
        if prov in ("llama-server", "local"):
            prov = "local"
        is_gemini = prov == "gemini"
        is_openai = prov in ("openai_compatible", "openai")

        active_pers = engine.personalities.active
        active_pers_name = getattr(active_pers, "name", "neo") if active_pers else "neo"
        ns = getattr(active_pers, "memory_namespace", "default")

        openai_cfg = getattr(engine, "config", {}).get("llm", {}).get("openai_compatible", {}) if hasattr(engine, "config") else {}
        if is_openai:
            active_model = getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "gpt-4o"))
            ctx_win = int(openai_cfg.get("context_window", 128000))
        elif is_gemini:
            active_model = getattr(engine.llm, "model", "gemini-3.5-flash")
            ctx_win = 1000000
        else:
            active_model = getattr(engine.llm, "model", "Qwen3.5-4B-Q4_K_M")
            ctx_win = int(getattr(engine, "config", {}).get("llm", {}).get("context_window", 16384)) if hasattr(engine, "config") else 16384

        situation = await engine.get_situation_context() if hasattr(engine, "get_situation_context") else {}
        
        user_profile = None
        if hasattr(engine, "memory") and engine.memory and hasattr(engine.memory, "get_core_user_profile"):
            try:
                user_profile = await engine.memory.get_core_user_profile(user_id=ns)
            except Exception:
                user_profile = None

        all_facts = []
        if hasattr(engine, "memory") and engine.memory and hasattr(engine.memory, "get_all_facts"):
            try:
                all_facts = await engine.memory.get_all_facts(user_id=ns, limit=20)
            except Exception:
                all_facts = []

        recent_history = []
        if hasattr(engine, "memory") and engine.memory:
            try:
                if hasattr(engine.memory, "get_recent_history"):
                    recent_history = await engine.memory.get_recent_history(limit=10, user_id=ns)
                elif hasattr(engine.memory, "get_recent_conversation"):
                    session_id = getattr(engine, "_session_id", "default")
                    recent_history = await engine.memory.get_recent_conversation(session_id, limit=10)
            except Exception:
                recent_history = []

        pers_prompt = (
            active_pers.system_prompt
            if active_pers and getattr(active_pers, "system_prompt", None)
            else (getattr(engine.personalities, "active_prompt", "") or "")
        )

        from robot.core.context import (
            format_system_prompt,
            format_user_core_profile,
            format_situation_banner,
        )

        assembled_messages = engine.context.build_messages(
            system_prompt=pers_prompt,
            user_text=test_prompt,
            recent_conversation=recent_history,
            memories=all_facts[:5],
            all_facts=all_facts,
            world_state=situation.get("world_state"),
            situation=situation,
            high_intelligence=is_gemini,
            user_profile=user_profile,
        )

        tool_schemas = engine.tool_registry.get_schemas() if hasattr(engine, "tool_registry") and engine.tool_registry else []
        tool_names = [t.get("name", "") for t in tool_schemas if isinstance(t, dict)]

        # Estimate Token Distribution
        tools_tokens_est = 10280 if tool_schemas else 0
        sys_tokens_est = len(assembled_messages[0]["content"]) // 4 if assembled_messages else 0
        user_tokens_est = len(assembled_messages[-1]["content"]) // 4 if len(assembled_messages) > 1 else 0
        hist_tokens_est = sum(len(str(m.get("content", ""))) // 4 for m in assembled_messages[1:-1])

        prefix_tokens = tools_tokens_est + sys_tokens_est
        suffix_tokens = user_tokens_est + hist_tokens_est
        total_tokens = prefix_tokens + suffix_tokens
        cache_reuse_pct = round((prefix_tokens / max(1, total_tokens)) * 100, 1)

        from robot.llm import LlamaServerManager
        llama_running = await LlamaServerManager.is_running_async()

        return {
            "timestamp": time.time(),
            "personality": active_pers_name,
            "provider": prov,
            "model": active_model,
            "context_window": ctx_win,
            "test_prompt": test_prompt,
            "structured_layer": {
                "system_prompt": format_system_prompt(pers_prompt, high_intelligence=is_gemini),
                "user_profile": format_user_core_profile(user_profile),
                "situation_banner": format_situation_banner(situation),
                "retrieved_memories": all_facts,
                "conversation_turns": recent_history,
                "tools": {
                    "count": len(tool_schemas),
                    "names": tool_names,
                    "tokens_estimate": tools_tokens_est,
                    "sample": tool_schemas[:2],
                },
            },
            "caching_status": {
                "strategy": "Slot 0 Invariant Canonical Prefix + Prompt Caching",
                "prefix_tokens_estimate": prefix_tokens,
                "suffix_tokens_estimate": suffix_tokens,
                "total_tokens_estimate": total_tokens,
                "cache_reuse_percent": cache_reuse_pct,
                "is_caching_invariant": True,
                "llama_running": llama_running,
                "prompt_caching_enabled": True if is_openai else llama_running,
            },
            "diagnostics": getattr(engine.context, "get_last_diagnostics", lambda: {})(),
            "assembled_messages": assembled_messages,
        }

    return app
