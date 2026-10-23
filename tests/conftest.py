"""Shared pytest wiring for the API test-suite.

``robot.api.create_app`` now enforces token authentication on every route except
the public allow-list. Rather than disabling that protection in tests, each test
application is created with a fixed local token through the supported
``ROBOT_API_TOKEN`` mechanism, and every ``TestClient`` built inside a test
presents that token using the standard ``Authorization: Bearer`` header (the
token is read back from ``app.state.auth_token``, so tests that let the API
generate a random token keep working too).

Tests that need to prove the 401 behaviour explicitly clear the header by passing
``headers={}``-style overrides or by using their own client instance.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

#: Token injected through ``ROBOT_API_TOKEN`` for every test application.
TEST_API_TOKEN = "pytest-local-api-token"


@pytest.fixture(autouse=True)
def _api_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose a deterministic API token to ``create_app`` for the whole test run."""
    monkeypatch.setenv("ROBOT_API_TOKEN", TEST_API_TOKEN)


@pytest.fixture(autouse=True)
def _test_client_presents_token(_api_auth_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :class:`TestClient` send the bearer token by default.

    Args:
        _api_auth_env: Ensures ``ROBOT_API_TOKEN`` is set first.
        monkeypatch: Pytest monkeypatch fixture used to wrap ``TestClient.__init__``.
    """
    original_init = TestClient.__init__

    def patched_init(self, app, *args, **kwargs):  # type: ignore[no-untyped-def]
        target_app = app if app is not None else kwargs.get("app")
        token = getattr(getattr(target_app, "state", None), "auth_token", None)
        headers = dict(kwargs.get("headers") or {})
        if isinstance(token, str) and token:
            headers.setdefault("Authorization", f"Bearer {token}")
        kwargs["headers"] = headers
        original_init(self, app, *args, **kwargs)

    monkeypatch.setattr(TestClient, "__init__", patched_init, raising=False)
