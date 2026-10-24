"""Tests for transport-level production hardening: headers, limits and throttling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from robot.api import create_app


@pytest.fixture
def api_engine() -> MagicMock:
    """Minimal mock engine accepted by :func:`create_app`."""
    engine = MagicMock()
    engine.config = {"api": {"max_body_bytes": 65536, "login_max_failures": 3, "login_window_s": 60}}
    engine._is_speaking = False
    engine.player.is_playing = False
    engine.telemetry.snapshot.return_value = {}
    engine.state.state.services = {}
    return engine


def _client(app) -> TestClient:
    """A non-loopback client, so the token is actually required."""
    return TestClient(app, base_url="http://robot.test", client=("203.0.113.7", 40000))


def test_security_headers_and_request_id(api_engine: MagicMock) -> None:
    """Every response carries the console's hardening headers and a request id."""
    client = _client(create_app(api_engine))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"
    assert "microphone=(self)" in response.headers["permissions-policy"]
    assert response.headers["x-request-id"]


def test_client_request_id_is_echoed(api_engine: MagicMock) -> None:
    """A well-formed client request id is preserved for log correlation."""
    client = _client(create_app(api_engine))
    response = client.get("/health", headers={"X-Request-ID": "trace-abc_123"})
    assert response.headers["x-request-id"] == "trace-abc_123"


def test_oversized_declared_body_is_rejected(api_engine: MagicMock) -> None:
    """A Content-Length above the configured cap is refused before it is read."""
    client = _client(create_app(api_engine))
    response = client.post(
        "/chat",
        content=b"x" * 1024,
        headers={
            "Authorization": "Bearer pytest-local-api-token",
            "Content-Length": str(1024 * 1024 * 4),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 413
    assert response.json()["request_id"] == response.headers["x-request-id"]


def test_login_failures_are_throttled(api_engine: MagicMock) -> None:
    """Repeated bad logins from one client receive 429 with Retry-After."""
    client = _client(create_app(api_engine))
    for _ in range(3):
        assert client.post("/auth/login", json={"token": "wrong"}).status_code == 401

    blocked = client.post("/auth/login", json={"token": "wrong"})
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
