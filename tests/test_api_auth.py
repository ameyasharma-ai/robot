"""Tests for the token-auth layer, trusted local access, XSS escaping and secret redaction."""

from __future__ import annotations

import asyncio
import stat
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from robot.api import (
    AUTH_COOKIE_NAME,
    AUTH_TOKEN_QUERY_PARAM,
    TRUST_LOOPBACK_ENV_VAR,
    create_app,
)


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


def _bare_client(app, **kwargs) -> TestClient:
    """A client that presents no credentials at all."""
    kwargs["headers"] = {**(kwargs.get("headers") or {}), "Authorization": ""}
    return TestClient(app, **kwargs)


def _local_client(app, **kwargs) -> TestClient:
    """A client that looks exactly like a browser on this machine: loopback peer
    address and a loopback ``Host``."""
    kwargs["headers"] = {**(kwargs.get("headers") or {}), "Authorization": ""}
    return TestClient(
        app,
        base_url="http://127.0.0.1:8888",
        client=("127.0.0.1", 54321),
        **kwargs,
    )


def test_status_requires_token(api_engine: MagicMock) -> None:
    """Every API route is closed until a token is presented."""
    app = create_app(api_engine)
    client = _bare_client(app)
    assert client.get("/status").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code in (200, 404)  # 404 only when static UI is absent


def test_loopback_client_needs_no_token(api_engine: MagicMock) -> None:
    """The owner's own browser on 127.0.0.1 opens the console without a token."""
    app = create_app(api_engine, auth_token="tok-local")
    client = _local_client(app)
    status = client.get("/auth/status")
    assert status.status_code == 200
    body = status.json()
    assert body["authenticated"] is True
    assert body["auth_required"] is False
    assert body["via"] == "trusted_loopback"


def test_trusted_local_never_leaks_the_token(api_engine: MagicMock) -> None:
    """The trusted-local response carries no token value."""
    app = create_app(api_engine, auth_token="tok-local-secret")
    assert "tok-local-secret" not in _local_client(app).get("/auth/status").text


def test_non_loopback_peer_is_still_locked_out(api_engine: MagicMock) -> None:
    """A client arriving on another interface cannot use the local trust path."""
    app = create_app(api_engine, auth_token="tok-lan")
    client = TestClient(
        app,
        base_url="http://127.0.0.1:8888",
        client=("192.168.1.50", 54321),
        headers={"Authorization": ""},
    )
    assert client.get("/auth/status").json()["authenticated"] is False
    assert client.get("/status").status_code == 401


def test_rebound_host_is_denied(api_engine: MagicMock) -> None:
    """DNS rebinding: a loopback peer naming an external Host is refused."""
    app = create_app(api_engine, auth_token="tok-rebind")
    client = TestClient(
        app,
        base_url="http://evil.example",
        client=("127.0.0.1", 54321),
        headers={"Authorization": ""},
    )
    assert client.get("/status").status_code == 401


def test_cross_origin_loopback_request_is_denied(api_engine: MagicMock) -> None:
    """A page on another site cannot drive the local console."""
    app = create_app(api_engine, auth_token="tok-cors")
    client = _local_client(app, headers={"Authorization": "", "Origin": "http://evil.example"})
    assert client.get("/status").status_code == 401


def test_same_origin_loopback_fetch_is_trusted(api_engine: MagicMock) -> None:
    """An ``Origin`` that matches the loopback Host keeps the local trust."""
    app = create_app(api_engine, auth_token="tok-same")
    client = TestClient(
        app,
        base_url="http://127.0.0.1:8888",
        client=("127.0.0.1", 54321),
        headers={"Authorization": "", "Origin": "http://127.0.0.1:8888"},
    )
    assert client.get("/auth/status").json()["via"] == "trusted_loopback"


def test_trust_loopback_can_be_disabled(
    api_engine: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api.trust_loopback: false`` and the env override both re-lock localhost."""
    engine = api_engine
    engine.config = {"api": {"trust_loopback": False}}
    assert _local_client(create_app(engine)).get("/status").status_code == 401

    engine.config = {}
    monkeypatch.setenv(TRUST_LOOPBACK_ENV_VAR, "0")
    assert _local_client(create_app(engine)).get("/status").status_code == 401

    monkeypatch.setenv(TRUST_LOOPBACK_ENV_VAR, "1")
    assert _local_client(create_app(engine)).get("/auth/status").json()["authenticated"] is True


def test_loopback_websocket_is_accepted_without_token(api_engine: MagicMock) -> None:
    """The middleware's trusted-local stamp also unlocks the WebSocket route."""
    engine = api_engine
    registered: list = []
    engine.events.on = lambda event, handler: registered.append((event, handler))
    engine.events.off = lambda event, handler: None

    app = create_app(engine, auth_token="tok-ws-local")
    # TestClient's WebSocket session hardcodes ``host: testserver``, so the local
    # Host has to be restated for the handshake like a real browser would send it.
    with _local_client(app).websocket_connect(
        "/ws/events", headers={"Host": "127.0.0.1:8888"}
    ):
        assert registered and registered[0][0] == "*"


def test_generated_token_is_reused_across_restarts(
    api_engine: MagicMock, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token generated on first start is persisted and read back, so the console
    never changes the value the owner has bookmarked."""
    token_file = tmp_path / "api_token"
    monkeypatch.delenv("ROBOT_API_TOKEN", raising=False)
    monkeypatch.setenv("ROBOT_API_TOKEN_FILE", str(token_file))
    monkeypatch.setenv(TRUST_LOOPBACK_ENV_VAR, "0")

    first = create_app(api_engine)
    token = first.state.auth_token
    assert first.state.auth_token_source == "generated"
    assert token_file.read_text(encoding="utf-8").strip() == token
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600

    second = create_app(api_engine)
    assert second.state.auth_token == token
    assert second.state.auth_token_source == "token-file"


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
