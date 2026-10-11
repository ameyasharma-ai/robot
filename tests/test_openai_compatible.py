"""Tests for OpenAI-compatible LLM provider, prompt caching, and environment configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.config import load_env_file, update_env_variable
from robot.llm import OpenAICompatibleLLM


class MockHttpxResponse:
    """Mock httpx response object for scan_models tests."""

    def __init__(self, status_code: int, json_data: dict[str, Any]) -> None:
        """Initialize mock response with status code and json payload.

        Args:
            status_code: HTTP response status code.
            json_data: Dictionary payload returned by .json().
        """
        self.status_code = status_code
        self._json_data = json_data
        self.text = json.dumps(json_data)

    def json(self) -> dict[str, Any]:
        """Return mock json data.

        Returns:
            Dictionary payload.
        """
        return self._json_data


class MockHttpxStreamContext:
    """Mock context manager for client.stream in httpx."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        """Initialize mock stream with text lines and status code.

        Args:
            lines: List of raw string lines for aiter_lines().
            status_code: HTTP response status code.
        """
        self.lines = lines
        self.status_code = status_code

    async def __aenter__(self) -> MockHttpxStreamContext:
        """Enter async context manager.

        Returns:
            Self as response stream.
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        return None

    async def aiter_lines(self) -> Any:
        """Asynchronously yield streamed lines.

        Yields:
            str: Next line in stream.
        """
        for line in self.lines:
            yield line

    async def aread(self) -> bytes:
        """Read full response body.

        Returns:
            Bytes response.
        """
        return "\n".join(self.lines).encode("utf-8")


def test_openai_compatible_initialization() -> None:
    """Verify that OpenAICompatibleLLM correctly initializes with supplied settings."""
    provider = OpenAICompatibleLLM(
        base_url="https://api.openai.com/v1",
        api_key="sk-test-token-12345",
        model="gpt-4o",
        context_window=64000,
        enable_prompt_caching=True,
    )
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider.api_key == "sk-test-token-12345"
    assert provider.model == "gpt-4o"
    assert provider.context_window == 64000
    assert provider.enable_prompt_caching is True


@pytest.mark.asyncio
async def test_openai_compatible_scan_models_data_format() -> None:
    """Test model scanning with standard OpenAI /v1/models response format."""
    mock_payload = {
        "data": [
            {"id": "gpt-4o", "created": 1234567},
            {"id": "gpt-4o-mini", "created": 1234568},
            {"id": "o1-preview", "created": 1234569},
        ]
    }

    provider = OpenAICompatibleLLM(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
    )

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=MockHttpxResponse(200, mock_payload))

    with patch("httpx.AsyncClient", return_value=mock_client):
        models = await provider.scan_models()

    assert len(models) == 3
    ids = [m["id"] for m in models]
    assert "gpt-4o" in ids
    assert "gpt-4o-mini" in ids
    assert "o1-preview" in ids


@pytest.mark.asyncio
async def test_openai_compatible_scan_models_fallback_format() -> None:
    """Test model scanning with alternative endpoint payload format."""
    mock_payload = {
        "models": [
            {"name": "deepseek-coder", "id": "deepseek-coder"},
            {"name": "deepseek-chat", "id": "deepseek-chat"},
        ]
    }

    provider = OpenAICompatibleLLM(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
    )

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=MockHttpxResponse(200, mock_payload))

    with patch("httpx.AsyncClient", return_value=mock_client):
        models = await provider.scan_models()

    assert len(models) == 2
    ids = [m["id"] for m in models]
    assert "deepseek-chat" in ids
    assert "deepseek-coder" in ids


@pytest.mark.asyncio
async def test_openai_compatible_stream_chat_tokens() -> None:
    """Verify that stream_chat parses SSE delta chunks and yields generated text."""
    provider = OpenAICompatibleLLM(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="gpt-4o",
        enable_prompt_caching=True,
    )

    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" robot"}}]}',
        'data: {"choices":[{"delta":{"content":"!"}}]}',
        "data: [DONE]",
    ]

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=MockHttpxStreamContext(sse_lines, status_code=200))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tokens: list[str] = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
        ):
            tokens.append(chunk)

    assert "".join(tokens) == "Hello robot!"


@pytest.mark.asyncio
async def test_openai_compatible_prompt_caching_injection() -> None:
    """Ensure system prompt caching structures are injected into payload."""
    provider = OpenAICompatibleLLM(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        model="gpt-4o",
        enable_prompt_caching=True,
    )

    captured_payload: dict[str, Any] = {}

    def mock_stream(method: str, url: str, json: dict[str, Any], headers: dict[str, str]) -> MockHttpxStreamContext:
        captured_payload.update(json)
        return MockHttpxStreamContext(["data: [DONE]"], status_code=200)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(side_effect=mock_stream)

    with patch("httpx.AsyncClient", return_value=mock_client):
        async for _ in provider.stream_chat(
            messages=[
                {"role": "system", "content": "System prompt to cache."},
                {"role": "user", "content": "Test"},
            ],
        ):
            pass

    # Inspect captured payload
    assert captured_payload["model"] == "gpt-4o"
    sys_msg = captured_payload["messages"][0]
    assert sys_msg["role"] == "system"
    # Ephemeral cache control injected in content blocks for prompt caching
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_env_file_update_and_load(tmp_path: Path) -> None:
    """Verify update_env_variable and load_env_file maintain configuration integrity.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ROBOT_LLM_PROVIDER=local\nOPENAI_COMPATIBLE_MODEL=gpt-4o\n",
        encoding="utf-8",
    )

    # Update variable
    update_env_variable("OPENAI_COMPATIBLE_MODEL", "deepseek-chat", env_path=env_file)
    update_env_variable("OPENAI_COMPATIBLE_BASE_URL", "https://api.deepseek.com/v1", env_path=env_file)

    load_env_file(env_file)
    assert os.environ.get("OPENAI_COMPATIBLE_MODEL") == "deepseek-chat"
    assert os.environ.get("OPENAI_COMPATIBLE_BASE_URL") == "https://api.deepseek.com/v1"
    raw_content = env_file.read_text(encoding="utf-8")
    assert 'OPENAI_COMPATIBLE_MODEL="deepseek-chat"' in raw_content or "OPENAI_COMPATIBLE_MODEL=deepseek-chat" in raw_content
