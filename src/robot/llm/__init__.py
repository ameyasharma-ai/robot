"""LLM provider interface and llama-server streaming client."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import AsyncGenerator

import httpx
import requests

from robot.telemetry import get_logger

logger = get_logger("llm")


class LLMProvider(ABC):
    """Abstract LLM provider — supports llama-server and LM Studio."""

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 300,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class LlamaServerLLM(LLMProvider):
    """llama-server OpenAI-compatible streaming client.

    Connects to the local llama-server running Qwen3.5-4B on the RTX 3050.
    Streams tokens for low-latency first-audio delivery.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "Qwen3.5-4B-Q4_K_M",
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enable_thinking = enable_thinking

    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 300,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            choice = data["choices"][0]

                            # Check for tool calls
                            if "tool_calls" in choice.get("delta", {}):
                                tool_calls = choice["delta"]["tool_calls"]
                                yield f"__TOOL_CALL__{json.dumps(tool_calls)}"
                                continue

                            token = choice["delta"].get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    async def health_check(self) -> bool:
        try:
            base = self.base_url.rsplit("/v1", 1)[0]
            resp = requests.get(f"{base}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


class ClauseChunker:
    """Buffers LLM tokens and emits speakable phrases at natural boundaries.

    Optimization: Sends partial phrases to TTS as soon as a clause boundary
    is detected, rather than waiting for the full response. This achieves
    the whitepaper's streaming pipeline requirement.
    """

    def __init__(self, min_words: int = 2, max_words: int = 12) -> None:
        self._buffer = ""
        self._min_words = min_words
        self._max_words = max_words

    def feed(self, token: str) -> str | None:
        """Feed a token, return a phrase if a boundary is reached."""
        self._buffer += token

        # Check for terminal sentence boundary first (.!?)
        sent_match = re.search(r"^(.*?[.!?])(?:\s+|$)", self._buffer, re.DOTALL)
        if sent_match:
            clause = sent_match.group(1).strip()
            self._buffer = self._buffer[sent_match.end() :]
            return clause

        # Check for natural clause boundary (,;:—\n) if minimum words reached
        clause_match = re.search(r"^(.*?[,;:—\n])(?:\s+|$)", self._buffer, re.DOTALL)
        if clause_match:
            clause = clause_match.group(1).strip()
            if len(clause.split()) >= self._min_words:
                self._buffer = self._buffer[clause_match.end() :]
                return clause

        # Safety fallback for long unpunctuated runs
        words = self._buffer.strip().split()
        if len(words) >= self._max_words:
            last_space = self._buffer.rfind(" ")
            if last_space != -1:
                clause = self._buffer[:last_space].strip()
                self._buffer = self._buffer[last_space:].lstrip()
                return clause

        return None

    def flush(self) -> str | None:
        """Flush remaining buffer content."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder if remainder else None
