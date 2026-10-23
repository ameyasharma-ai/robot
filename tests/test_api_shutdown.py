"""Regression tests for shutting the embedded API server down cleanly.

Ctrl+C used to end with two ``CancelledError`` tracebacks (one from Starlette's
lifespan protocol, one labelled "Exception in ASGI application"). Neither was a
fault in application code: uvicorn's graceful shutdown waits for in-flight
responses *before* it sends the lifespan shutdown event, and an open console tab
holds two responses that never end by themselves — the MJPEG camera stream and
the ``/ws/events`` WebSocket. So the wait never finished and the only way out was
cancelling ``Server.serve()`` from outside, which uvicorn reports through its
error logger.

These tests pin the contract that replaced that: long-lived streams watch the
engine's running flag, ``serve()`` is never cancelled while it is shutting down,
and the shutdown produces no error-level log output.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
import uvicorn
import websockets
from websockets.exceptions import ConnectionClosed

from robot.api import create_app
from robot.core.engine import RobotEngine
from tests.conftest import TEST_API_TOKEN

#: How long the server is given to drain before a test declares a hang.
SHUTDOWN_TIMEOUT_S = 15.0


class _ErrorHandler(logging.Handler):
    """Collects error-level records emitted by a target logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage().strip())


class _EngineHost:
    """Drives the real :meth:`RobotEngine._run_api` / :meth:`_stop_api` pair.

    Those two methods only touch ``config``, the uvicorn handles they create and
    the module logger, so the production shutdown wiring can be tested without a
    camera, a voice stack or a model server. ``create_app`` is monkeypatched to
    return an application built from this double.

    Attributes:
        config: Engine config, pinned to an ephemeral loopback port.
        _running: The flag the stream generators watch.
        _api_server: uvicorn's server handle, set by ``_run_api``.
        _api_task: Unused here; the caller owns the ``serve()`` task.
    """

    _run_api = RobotEngine._run_api
    _stop_api = RobotEngine._stop_api

    def __init__(self) -> None:
        self.config: dict = {"api": {"host": "127.0.0.1", "port": 0}}
        self._running = True
        self._api_server: uvicorn.Server | None = None
        self._api_task: asyncio.Task | None = None
        self.is_camera_enabled = True
        self.perception = None
        self.events = MagicMock()

    @property
    def port(self) -> int:
        """Return the ephemeral port the bound server is listening on."""
        assert self._api_server is not None and self._api_server.servers
        return int(self._api_server.servers[0].sockets[0].getsockname()[1])

    async def serve(self) -> None:
        """Run ``_run_api`` with ``create_app`` pointed at this double.

        Returns:
            None
        """
        import robot.api as api_module

        # Production starts this coroutine as self._api_task; _stop_api() looks it
        # up by that name, so the double has to publish it the same way.
        self._api_task = asyncio.current_task()
        app = create_app(self, auth_token=TEST_API_TOKEN)
        original = api_module.create_app
        api_module.create_app = lambda _engine: app
        try:
            await self._run_api()
        finally:
            api_module.create_app = original

    async def start(self) -> None:
        """Wait until the server has bound its socket and finished startup.

        Raises:
            AssertionError: If the server never reports itself started.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10.0
        while not (self._api_server and self._api_server.started):
            if loop.time() > deadline:
                raise AssertionError("API server did not start")
            await asyncio.sleep(0.05)


async def _open_mjpeg_stream(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open the camera MJPEG endpoint and read only its response headers.

    The body is deliberately left unread: the point is a server-side response
    that is still streaming when shutdown starts.

    Args:
        port: Port the test server is listening on.

    Returns:
        tuple: ``(reader, writer)`` for the still-open connection.

    Raises:
        AssertionError: If the response is not a 200.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    request = (
        f"GET /camera/stream HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Authorization: Bearer {TEST_API_TOKEN}\r\n"
        f"Connection: keep-alive\r\n\r\n"
    ).encode()
    writer.write(request)
    await writer.drain()
    headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
    assert b" 200 " in headers, headers
    return reader, writer


async def test_streams_stop_and_server_shuts_down_without_errors() -> None:
    """An open console tab must not turn Ctrl+C into a cancelled server task."""
    host = _EngineHost()
    serve_task = asyncio.create_task(host.serve())
    await host.start()

    errors = _ErrorHandler()
    uvicorn_logger = logging.getLogger("uvicorn.error")
    # Uvicorn configures this logger while loading its config during startup, so
    # the level is only pinned here. Otherwise its log level would filter out the
    # exact records this test looks for.
    uvicorn_logger.addHandler(errors)
    uvicorn_logger.setLevel(logging.ERROR)
    port = host.port
    writer: asyncio.StreamWriter | None = None

    try:
        reader, writer = await _open_mjpeg_stream(port)
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws/events?token={TEST_API_TOKEN}") as ws:
            # Both streams are live: this is the state a browser leaves behind.
            host._running = False
            await asyncio.wait_for(host._stop_api(), timeout=SHUTDOWN_TIMEOUT_S)

            # serve() has to return by itself — cancelling it from outside is what
            # produced the original tracebacks.
            assert serve_task.done() and not serve_task.cancelled()

            # ...and the clients must see their streams end, not a hung socket.
            tail = await asyncio.wait_for(reader.read(-1), timeout=10)
            assert tail.endswith(b"0\r\n\r\n"), tail
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=10)
    finally:
        uvicorn_logger.removeHandler(errors)
        if writer is not None:
            writer.close()
        if not serve_task.done():
            serve_task.cancel()
            await asyncio.gather(serve_task, return_exceptions=True)

    assert errors.messages == []


class _StopApiHost:
    """Exposes :meth:`RobotEngine._stop_api` without constructing an engine."""

    _stop_api = RobotEngine._stop_api


async def test_stop_api_waits_for_a_server_that_drains() -> None:
    """A server that closes its own connections is awaited, not cancelled."""
    finished = asyncio.Event()

    async def _serve() -> None:
        await finished.wait()

    host = _StopApiHost()
    host._api_server = MagicMock()
    host._api_task = asyncio.create_task(_serve())
    await asyncio.sleep(0.05)
    finished.set()

    await host._stop_api(grace_s=5.0)

    assert host._api_task.done() and not host._api_task.cancelled()
    assert host._api_server.should_exit is True


async def test_stop_api_forces_a_server_that_ignores_the_exit_flag() -> None:
    """A wedged stream costs the grace window, never a hung shutdown."""

    async def _stuck() -> None:
        await asyncio.Event().wait()

    host = _StopApiHost()
    host._api_server = MagicMock()
    host._api_task = asyncio.create_task(_stuck())

    loop = asyncio.get_running_loop()
    started = loop.time()
    await host._stop_api(grace_s=0.3)
    elapsed = loop.time() - started

    assert host._api_task.cancelled()
    assert 0.25 <= elapsed < 2.0
