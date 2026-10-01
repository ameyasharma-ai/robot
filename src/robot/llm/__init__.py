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
                if response.status_code != 200:
                    err_body = await response.aread()
                    logger.error("llm_stream_error", status=response.status_code, body=err_body.decode("utf-8", errors="replace"))
                    yield "I encountered an error connecting to my local inference model."
                    return
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
    """Buffers LLM tokens and emits speakable, complete sentences for TTS.

    Chunking on full sentence boundaries (.!?) prevents unnatural fragmentation
    and awkward phrasing caused by mid-sentence comma cuts, while still achieving
    streaming TTS with minimal TTFT.
    """

    def __init__(self, min_words: int = 4, max_words: int = 24) -> None:
        self._buffer = ""
        self._min_words = min_words
        self._max_words = max_words

    def feed(self, token: str) -> str | None:
        """Feed a token, return a complete sentence if a boundary is reached."""
        self._buffer += token

        # Sentence terminal detection:
        # Match from start of buffer up to a sentence terminator (. ! ?), followed by space or end
        match = re.search(r"^(.*?[.!?]+[\"'\)\]]*)(?:\s+|$)", self._buffer, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # Guard against common abbreviations (e.g., Dr., Mr., etc.) or decimal numbers
            last_word = candidate.split()[-1] if candidate.split() else ""
            is_abbreviation = any(
                last_word.lower().endswith(abbr)
                for abbr in ("mr.", "mrs.", "ms.", "dr.", "prof.", "vs.", "etc.", "e.g.", "i.e.")
            ) or (re.search(r"\d\.\d*$", last_word) is not None)

            if not is_abbreviation:
                # Any sentence terminal (. ! ?) with at least 1 word can be a complete utterance
                if len(candidate.split()) >= 1:
                    self._buffer = self._buffer[match.end():]
                    return candidate

        # Fallback for unusually long continuous runs without terminal punctuation
        words = self._buffer.strip().split()
        if len(words) >= self._max_words:
            # Look for a natural clause boundary (, ; — or newline) far enough into the buffer
            clause_match = re.search(r"^(.*?[\n,;—])(?:\s+|$)", self._buffer, re.DOTALL)
            if clause_match:
                clause = clause_match.group(1).strip()
                if len(clause.split()) >= self._min_words:
                    self._buffer = self._buffer[clause_match.end():]
                    return clause

            last_space = self._buffer.rfind(" ")
            if last_space != -1:
                chunk = self._buffer[:last_space].strip()
                self._buffer = self._buffer[last_space:].lstrip()
                return chunk

        return None

    def flush(self) -> str | None:
        """Flush remaining buffer content."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder if remainder else None
