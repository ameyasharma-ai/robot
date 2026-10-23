"""Tests for the token-auth layer, XSS escaping and secret redaction in robot.api."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from robot.api import AUTH_COOKIE_NAME, AUTH_TOKEN_QUERY_PARAM, create_app


@pytest.fixture
def api_engine() -> MagicMock:
    """Minimal mock engine accepted by :func:`create_app`."""
    engine = MagicMock()
    engine.config = {}
    engine._is_speaking = False
    engine.player.is_playing = False
    engine.telemetry.snapshot.return_value = {}
    engine.state.state.services = {}
    return engine


def _bare_client(app) -> TestClient:
    """A client that presents no credentials at all."""
    return TestClient(app, headers={"Authorization": ""})


def test_status_requires_token(api_engine: MagicMock) -> None:
    """Every API route is closed until a token is presented."""
    app = create_app(api_engine)
    client = _bare_client(app)
    assert client.get("/status").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code in (200, 404)  # 404 only when static UI is absent


def test_bearer_header_authenticates(api_engine: MagicMock) -> None:
    """The standard Authorization bearer header unlocks protected routes."""
    app = create_app(api_engine, auth_token="tok-bearer")
    client = TestClient(app, headers={"Authorization": "Bearer tok-bearer"})
    assert client.get("/auth/status").json()["authenticated"] is True
    assert client.get("/health").status_code == 200


def test_query_param_authenticates_stream_clients(api_engine: MagicMock) -> None:
    """Header-less clients (EventSource / MJPEG) may use ?token=."""
    app = create_app(api_engine, auth_token="tok-query")
    client = _bare_client(app)
    resp = client.get(f"/auth/status?{AUTH_TOKEN_QUERY_PARAM}=tok-query")
    assert resp.status_code == 200
    assert resp.json()["authenticated"] is True


def test_login_sets_httponly_cookie_and_logout_clears_it(api_engine: MagicMock) -> None:
    """POST /auth/login installs an httpOnly SameSite=Lax session cookie."""
    app = create_app(api_engine, auth_token="tok-login")
    client = _bare_client(app)

    bad = client.post("/auth/login", json={"token": "wrong"})
    assert bad.status_code == 401

    good = client.post("/auth/login", json={"token": "tok-login"})
    assert good.status_code == 200
    cookie = good.headers["set-cookie"].lower()
    assert AUTH_COOKIE_NAME.lower() in cookie
    assert "httponly" in cookie
    assert "samesite=lax" in cookie

    # The cookie now authenticates the client for header-less streams.
    assert client.get("/auth/status").json()["authenticated"] is True

    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/status").json()["authenticated"] is False


def test_token_is_never_returned_by_any_endpoint(api_engine: MagicMock) -> None:
    """No response body may leak the API token or provider keys."""
    engine = api_engine
    engine.config = {
        "llm": {
            "openai_compatible": {"api_key": "sk-secret-openai"},
            "openrouter": {"api_key": "sk-secret-openrouter"},
            "aicredits": {"api_key": "sk-secret-aicredits"},
        }
    }
    app = create_app(engine, auth_token="tok-secret")
    client = TestClient(app, headers={"Authorization": "Bearer tok-secret"})
    body = client.get("/api/llm/status").text
    for secret in ("tok-secret", "sk-secret-openai", "sk-secret-openrouter", "sk-secret-aicredits"):
        assert secret not in body


def test_spotify_callback_escapes_reflected_error(api_engine: MagicMock) -> None:
    """The OAuth callback escapes attacker-controlled query values."""
    app = create_app(api_engine, auth_token="tok-cb")
    client = TestClient(app, headers={"Authorization": "Bearer tok-cb"})
    payload = "<script>alert(1)</script>"
    resp = client.get("/callback", params={"error": payload})
    assert resp.status_code == 400
    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in resp.text


def test_websocket_rejects_missing_token(api_engine: MagicMock) -> None:
    """WebSocket handshakes without a valid token are refused before accept()."""
    app = create_app(api_engine, auth_token="tok-ws")
    client = _bare_client(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/events"):
            pass


def test_websocket_accepts_query_token(api_engine: MagicMock) -> None:
    """A WebSocket handshake may authenticate with ?token= and registers its handler."""
    engine = api_engine
    registered: list = []
    unregistered: list = []

    engine.events.on = lambda event, handler: registered.append((event, handler))
    engine.events.off = lambda event, handler: unregistered.append((event, handler))

    app = create_app(engine, auth_token="tok-ws")
    client = _bare_client(app)
    # Entering the context without WebSocketDisconnect proves accept() succeeded.
    with client.websocket_connect(f"/ws/events?{AUTH_TOKEN_QUERY_PARAM}=tok-ws"):
        assert registered and registered[0][0] == "*"

    assert unregistered and unregistered[0][0] == "*"


def test_websocket_handler_validates_token_before_accept(api_engine: MagicMock) -> None:
    """The /ws/events handler refuses itself (4401) even without the middleware."""
    engine = api_engine
    accepts: list = []

    app = create_app(engine, auth_token="tok-ws")
    route = next(r for r in app.routes if getattr(r, "path", "") == "/ws/events")

    async def _drive(scope: dict) -> list[dict]:
        sent: list[dict] = []

        async def receive() -> dict:
            return {"type": "websocket.connect"}

        async def send(message: dict) -> None:
            sent.append(message)
            if message.get("type") == "websocket.accept":
                accepts.append(message)

        websocket = WebSocket(scope, receive, send)
        task = asyncio.create_task(route.endpoint(websocket=websocket))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return sent

    unauthorized = {"type": "websocket", "path": "/ws/events", "headers": [], "query_string": b""}
    sent = asyncio.run(_drive(unauthorized))
    assert sent and sent[0]["type"] == "websocket.close" and sent[0]["code"] == 4401
    assert not accepts
    assert engine.events.on.call_count == 0

    authorized = dict(unauthorized, query_string=f"{AUTH_TOKEN_QUERY_PARAM}=tok-ws".encode())
    sent = asyncio.run(_drive(authorized))
    assert sent and sent[0]["type"] == "websocket.accept"
