"""LLM provider interface and llama-server streaming client."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import os
import random
import re
import shutil
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, TextIO

import httpx
import requests

try:  # Optional dependency: used to terminate tracked PID gracefully.
    import psutil
except ImportError:  # pragma: no cover - fallback path exercised without psutil
    psutil = None  # type: ignore[assignment]

from robot.concurrency import spawn
from robot.daemon_log import open_daemon_log
from robot.llm.model_registry import ModelCategory, ModelMetadata, ModelRegistry
from robot.llm.settings import LlamaServerSettings
from robot.telemetry import get_logger

logger = get_logger("llm")

#: Retry/backoff tuning for streamed completions. Retries are only permitted before the
#: first token is received, otherwise replaying the stream would duplicate spoken output.
STREAM_MAX_RETRIES = 2
STREAM_RETRY_BASE_DELAY = 0.5
STREAM_RETRY_DEADLINE_SECONDS = 30.0
STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)

#: Retry policy for the OpenAI-compatible cloud client.
CLOUD_MAX_RETRIES = 2
CLOUD_RETRY_MAX_DELAY = 10.0


class LLMStreamError(RuntimeError):
    """Typed streaming failure the engine can surface without replaying tokens.

    Raised when a provider fails *after* partial output has already been emitted. The
    engine must not fail over to another model or retry the same request in that case.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "local",
        status_code: int | None = None,
        partial_output: bool = True,
    ) -> None:
        """Initialize the streaming error.

        Args:
            message: Safe, user-facing description.
            provider: Provider identifier ('local', 'gemini', 'openai_compatible').
            status_code: HTTP status when the failure was an HTTP response.
            partial_output: Whether tokens were already yielded to the caller.
        """
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.partial_output = partial_output


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens for an entire message list.

    Serialises every message including ``tool_calls`` arguments and multimodal content
    blocks. Base64 image payloads are replaced by a fixed per-image token cost instead
    of counting every character, which would wildly overestimate the budget.

    Args:
        messages: Chat messages in OpenAI format.

    Returns:
        int: Estimated prompt token count.
    """
    total_chars = 0
    image_tokens = 0
    for message in messages:
        if not isinstance(message, dict):
            total_chars += len(str(message))
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    total_chars += len(str(part))
                    continue
                if part.get("type") in ("image_url", "input_image", "image"):
                    image_tokens += 1024
                    total_chars += len(str(part.get("type", "")))
                else:
                    total_chars += len(str(part.get("text", "")))
        else:
            total_chars += len(str(content or ""))
        metadata = {k: v for k, v in message.items() if k != "content"}
        total_chars += len(json.dumps(metadata, default=str))
    return max(1, total_chars // 4 + image_tokens)


def _jittered_delay(attempt: int, base: float = STREAM_RETRY_BASE_DELAY, cap: float = 8.0) -> float:
    """Compute an exponentially backed-off, jittered retry delay.

    Args:
        attempt: Zero-based attempt number.
        base: Base delay in seconds.
        cap: Maximum delay in seconds.

    Returns:
        float: Delay including full jitter.
    """
    exponential = min(cap, base * (2 ** max(0, attempt)))
    return exponential + random.uniform(0.0, exponential * 0.25)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header in seconds form.

    Args:
        value: Header value.

    Returns:
        float | None: Parsed delay, or None when absent/non-numeric.
    """
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if seconds < 0 or seconds != seconds:
        return None
    return seconds


def _project_root() -> Path:
    """Return the repository root independently of the process working directory.

    Returns:
        Path: Absolute project root.
    """
    try:
        from robot.config import get_project_root

        return get_project_root()
    except Exception:  # pragma: no cover - defensive fallback
        return Path(__file__).resolve().parents[3]


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


def parse_qwen_tool_call_block(block: str) -> list[dict]:
    """Parse Qwen-style XML or JSON tool calls from a <tool_call> or [tools] block."""
    content = re.sub(r"</?tool_call>", "", block).strip()
    if content.startswith("[tools]"):
        content = content[len("[tools]"):].strip()
    if not content:
        return []

    # Check for multiple JSON tool call candidates: {"name": ..., ...}
    json_candidates = re.findall(r"\{\s*\"name\"\s*:\s*\"[^\"]+\"[^}]*\}", content, re.DOTALL)
    if json_candidates:
        parsed_calls = []
        for cand in json_candidates:
            try:
                d = json.loads(cand)
                fn = d.get("name") or d.get("function", "")
                args = d.get("arguments", {})
                if fn:
                    parsed_calls.append({
                        "id": f"call_{int(time.time()*1000)}_{len(parsed_calls)}",
                        "type": "function",
                        "function": {"name": fn, "arguments": json.dumps(args) if isinstance(args, dict) else str(args)},
                    })
            except Exception:
                pass
        if parsed_calls:
            return parsed_calls

    if content.startswith("{") and content.endswith("}"):
        try:
            d = json.loads(content)
            fn = d.get("name") or d.get("function", "")
            args = d.get("arguments", {})
            return [{
                "id": f"call_{int(time.time()*1000)}",
                "type": "function",
                "function": {"name": fn, "arguments": json.dumps(args) if isinstance(args, dict) else str(args)},
            }]
        except Exception:
            pass

    # Check for bracketed tool calls e.g. [system.time], [perception.what_do_you_see]
    bracket_calls = re.findall(r"\[\s*([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+)(?:\((.*?)\)|:\s*(.*?)|)\s*\]", content)
    if bracket_calls:
        parsed_calls = []
        for b_name, b_args1, b_args2 in bracket_calls:
            raw_arg = (b_args1 or b_args2 or "").strip()
            args = {}
            if raw_arg:
                try:
                    args = json.loads(raw_arg) if raw_arg.startswith("{") else {"query": raw_arg}
                except Exception:
                    args = {"query": raw_arg}
            parsed_calls.append({
                "id": f"call_{int(time.time()*1000)}_{len(parsed_calls)}",
                "type": "function",
                "function": {"name": b_name, "arguments": json.dumps(args)},
            })
        if parsed_calls:
            return parsed_calls

    fn_match = re.search(r"<function=([^>]+)>(.*?)(?:</function>|$)", content, re.DOTALL)
    if fn_match:
        fn_name = fn_match.group(1).strip()
        body = fn_match.group(2)
        params = {}
        for p_match in re.finditer(r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", body, re.DOTALL):
            p_name = p_match.group(1).strip()
            p_val = p_match.group(2).strip()
            try:
                params[p_name] = json.loads(p_val)
            except Exception:
                params[p_name] = p_val
        return [{
            "id": f"call_{int(time.time()*1000)}",
            "type": "function",
            "function": {"name": fn_name, "arguments": json.dumps(params)},
        }]
    return []


class LlamaServerLLM(LLMProvider):
    """llama-server OpenAI-compatible streaming client with KV-cache prefix stability.

    Architectural Rationale:
    1. Hardware & Latency Budget: Connects to local llama-server running Qwen3.5-4B offloaded
       100% to NVIDIA RTX 3050 (4GB VRAM) via FlashAttention. Targets <450ms TTFT by reusing
       the KV-cache prefix across conversational turns.
    2. Invariant Tool Prefix: Qwen's chat template places the `# Tools` block at token 0.
       To maintain 100% Longest Common Prefix (LCP) cache hits, `canonical_tools` maintains
       an invariant tool schema prefix across multi-turn chats and second-pass continuations.
    3. Session Safety: Uses native LCP slot matching. Conversation branching or new sessions
       automatically invalidate KV slots past the common prefix without cross-user leakage.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "Qwen3.5-4B-Q4_K_M",
        enable_thinking: bool = False,
        canonical_tools: list[dict[str, Any]] | None = None,
        use_invariant_tools: bool = True,
    ) -> None:
        """Initialize the llama-server streaming client.

        Args:
            base_url: Base HTTP endpoint of llama-server OpenAI API.
            model: Model name string.
            enable_thinking: Whether to enable thinking token output.
            canonical_tools: Optional invariant list of all tool schemas.
            use_invariant_tools: Whether to prioritize invariant tool prefix for KV cache reuse.
        """
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.enable_thinking: bool = enable_thinking
        self.canonical_tools: list[dict[str, Any]] | None = canonical_tools
        self.use_invariant_tools: bool = use_invariant_tools
        self._last_prompt_tokens: int = 0
        self._last_ttft_ms: float = 0.0
        self._client: httpx.AsyncClient | None = None
        # Reference counting for in-flight streams: a config change may retire the
        # cached client while another turn is still streaming through it. The old
        # client is closed only once its last stream finishes.
        self._client_refs: dict[httpx.AsyncClient, int] = {}
        self._retired_clients: set[httpx.AsyncClient] = set()
        self._client_rebuild_lock = asyncio.Lock()

    def _get_client(self) -> httpx.AsyncClient:
        """Return a lazily created, cached httpx client reused across requests.

        Mirrors GeminiLLM._get_client: one persistent AsyncClient per instance keeps
        HTTP keep-alive connections warm instead of paying TLS/connect setup per turn.

        Returns:
            httpx.AsyncClient: Cached async client bound to this instance.
        """
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=120.0)
            self._client = httpx.AsyncClient(timeout=STREAM_TIMEOUT, limits=limits)
        return self._client

    @contextlib.asynccontextmanager
    async def _lease_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Lease the cached client for one request, tracking concurrent users.

        Yields:
            httpx.AsyncClient: Client that remains open for the duration of the lease.
        """
        client = self._get_client()
        self._client_refs[client] = self._client_refs.get(client, 0) + 1
        try:
            yield client
        finally:
            remaining = self._client_refs.get(client, 1) - 1
            if remaining > 0:
                self._client_refs[client] = remaining
            else:
                self._client_refs.pop(client, None)
                if client is not self._client:
                    self._retired_clients.discard(client)
                    try:
                        await client.aclose()
                    except Exception:
                        pass

    async def aclose(self) -> None:
        """Close the cached HTTP client (call on engine shutdown)."""
        clients: list[httpx.AsyncClient] = []
        if self._client is not None:
            clients.append(self._client)
        clients.extend(self._retired_clients)
        self._client = None
        self._retired_clients.clear()
        self._client_refs.clear()
        for client in clients:
            if not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    pass

    async def _rebuild_client(self) -> None:
        """Retire the cached client after a transport failure.

        The retired client is detached immediately so new requests use a fresh one, but
        it is only closed after its last concurrent stream lease ends.
        """
        async with self._client_rebuild_lock:
            client = self._client
            self._client = None
            if client is None:
                return
            if self._client_refs.get(client, 0) > 0:
                self._retired_clients.add(client)
                logger.debug("llm_client_retired_while_in_use")
                return
            if not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    pass

    def set_canonical_tools(self, tools: list[dict[str, Any]]) -> None:
        """Set or update the canonical invariant tool schema set.

        Args:
            tools: Full list of tool schemas for invariant prompt caching.
        """
        self.canonical_tools = tools
        logger.info("llama_canonical_tools_updated", tool_count=len(tools))

    async def get_slot_metrics(self) -> dict[str, Any]:
        """Fetch live slot memory and KV cache metrics from llama-server.

        Returns:
            Dictionary containing slot processing status and prompt token count.
        """
        try:
            async with self._lease_client() as client:
                resp = await client.get(f"{self.base_url.rsplit('/v1', 1)[0]}/slots", timeout=3.0)
            if resp.status_code == 200:
                slots = resp.json()
                if slots and isinstance(slots, list):
                    s0 = slots[0]
                    return {
                        "slot_id": s0.get("id", 0),
                        "n_ctx": s0.get("n_ctx", 16384),
                        "n_prompt_tokens": s0.get("n_prompt_tokens", 0),
                        "is_processing": s0.get("is_processing", False),
                    }
        except Exception as e:
            logger.debug("llm_slot_metrics_fetch_failed", error=str(e))
        return {}

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens from local llama-server with KV cache preservation.

        Transport retries are only performed before the first token is received. Once any
        token has been received, a failure raises :class:`LLMStreamError` instead of
        replaying the stream (which would duplicate spoken output).

        Args:
            messages: Formatted message turns.
            max_tokens: Maximum tokens to generate.
            tools: Optional tool schemas. When `use_invariant_tools` is True and `canonical_tools`
                is registered, the invariant canonical schemas are retained to preserve token 0.

        Yields:
            Generated text tokens or tool call JSON indicators.

        Raises:
            LLMStreamError: If the stream drops after partial output was received.
        """
        # Resolve effective tools:
        # Invariant tool set guarantees token 0 (# Tools) stays 100% byte-invariant across turns.
        # Explicit empty list [] or tool response continuation allows opt-out to synthesize text.
        is_tool_continuation = bool(messages and messages[-1].get("role") == "tool")
        if tools == [] or is_tool_continuation:
            effective_tools = None
        elif self.use_invariant_tools and self.canonical_tools and tools is None:
            effective_tools = self.canonical_tools
        elif tools is not None:
            effective_tools = tools
            if self.canonical_tools is None and len(tools) > 0:
                self.canonical_tools = tools
        else:
            effective_tools = self.canonical_tools if self.use_invariant_tools else None

        self._last_prompt_tokens = _estimate_messages_tokens(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "stream_options": {"include_usage": True},
        }
        if effective_tools:
            payload["tools"] = effective_tools

        deadline = time.monotonic() + STREAM_RETRY_DEADLINE_SECONDS
        received_output = False

        for attempt in range(STREAM_MAX_RETRIES + 1):
            try:
                async with self._lease_client() as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            err_body = await response.aread()
                            err_str = err_body.decode("utf-8", errors="replace")
                            logger.error(
                                "llm_stream_error",
                                status=response.status_code,
                                body=err_str[:300],
                            )
                            # Auto-recovery for context overflow: drop the oldest turns while
                            # preserving the system prompt and the final user/assistant pair.
                            if response.status_code == 400 and (
                                "exceeds" in err_str.lower() or "context" in err_str.lower()
                            ):
                                pruned = self._trim_messages_for_overflow(
                                    payload.get("messages", []), err_str
                                )
                                if pruned is not None and attempt < STREAM_MAX_RETRIES:
                                    payload["messages"] = pruned
                                    logger.warning(
                                        "llm_context_overflow_recovery",
                                        attempt=attempt,
                                        message_count=len(pruned),
                                        estimated_tokens=_estimate_messages_tokens(pruned),
                                    )
                                    continue
                                yield "The local model context window is full (HTTP 400)."
                                return
                            if response.status_code == 400:
                                yield "The local model rejected the request (HTTP 400)."
                                return
                            yield f"Local model request failed (HTTP {response.status_code})."
                            return

                        tool_calls_acc: dict[int, dict] = {}
                        in_qwen_tag = False
                        qwen_tag_buf = ""
                        pending_text_buf = ""

                        async for line in response.aiter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    usage = data.get("usage") or {}
                                    if usage.get("prompt_tokens"):
                                        self._last_prompt_tokens = int(usage["prompt_tokens"])
                                    choices = data.get("choices") or []
                                    if not choices:
                                        continue
                                    choice = choices[0]
                                    delta = choice.get("delta", {})

                                    # Accumulate native OpenAI tool calls delta chunks
                                    if "tool_calls" in delta and delta["tool_calls"]:
                                        received_output = True
                                        for tc in delta["tool_calls"]:
                                            idx = tc.get("index", 0)
                                            if idx not in tool_calls_acc:
                                                tool_calls_acc[idx] = {
                                                    "id": tc.get("id") or f"call_{idx}_{int(time.time()*1000)}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": tc.get("function", {}).get("name", ""),
                                                        "arguments": "",
                                                    },
                                                }
                                            if tc.get("id"):
                                                tool_calls_acc[idx]["id"] = tc["id"]
                                            fn = tc.get("function", {})
                                            if fn.get("name"):
                                                tool_calls_acc[idx]["function"]["name"] = fn["name"]
                                            if fn.get("arguments"):
                                                tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]
                                        continue

                                    token = delta.get("content", "")
                                    if not token:
                                        continue
                                    received_output = True

                                    # Intercept Qwen-style <tool_call> tags embedded in content
                                    if in_qwen_tag:
                                        qwen_tag_buf += token
                                        if "</tool_call>" in qwen_tag_buf:
                                            parsed = parse_qwen_tool_call_block(qwen_tag_buf)
                                            for p_tc in parsed:
                                                tool_calls_acc[len(tool_calls_acc)] = p_tc
                                            post = qwen_tag_buf.split("</tool_call>", 1)[1]
                                            in_qwen_tag = False
                                            qwen_tag_buf = ""
                                            pending_text_buf += post
                                    else:
                                        pending_text_buf += token
                                        if "<tool_call" in pending_text_buf:
                                            pre, post = pending_text_buf.split("<tool_call", 1)
                                            if pre:
                                                yield pre
                                            qwen_tag_buf = "<tool_call" + post
                                            in_qwen_tag = True
                                            pending_text_buf = ""
                                            if "</tool_call>" in qwen_tag_buf:
                                                parsed = parse_qwen_tool_call_block(qwen_tag_buf)
                                                for p_tc in parsed:
                                                    tool_calls_acc[len(tool_calls_acc)] = p_tc
                                                post_close = qwen_tag_buf.split("</tool_call>", 1)[1]
                                                in_qwen_tag = False
                                                qwen_tag_buf = ""
                                                pending_text_buf += post_close
                                        else:
                                            # Hold back if pending_text_buf ends with an unclosed angle or square bracket
                                            match = re.search(r"(?:<[^>]*$|\[[^\]]*$)", pending_text_buf)
                                            if match:
                                                safe_len = match.start()
                                                if safe_len > 0:
                                                    yield pending_text_buf[:safe_len]
                                                    pending_text_buf = pending_text_buf[safe_len:]
                                            else:
                                                # Check if pending_text_buf contains bracketed tool calls to intercept
                                                b_tool_match = re.search(r"\[\s*([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+)(?:\((.*?)\)|:\s*(.*?)|)\s*\]", pending_text_buf)
                                                if b_tool_match:
                                                    pre = pending_text_buf[:b_tool_match.start()]
                                                    if pre:
                                                        yield pre
                                                    t_name = b_tool_match.group(1).strip()
                                                    raw_a = (b_tool_match.group(2) or b_tool_match.group(3) or "").strip()
                                                    parsed_a = {}
                                                    if raw_a:
                                                        try:
                                                            parsed_a = json.loads(raw_a) if raw_a.startswith("{") else {"query": raw_a}
                                                        except Exception:
                                                            parsed_a = {"query": raw_a}
                                                    tool_calls_acc[len(tool_calls_acc)] = {
                                                        "id": f"call_{int(time.time()*1000)}_{len(tool_calls_acc)}",
                                                        "type": "function",
                                                        "function": {"name": t_name, "arguments": json.dumps(parsed_a)},
                                                    }
                                                    pending_text_buf = pending_text_buf[b_tool_match.end():]
                                                    continue

                                                # Strip meta category markers like [conversation] or [tools]
                                                b_meta_match = re.search(r"\[\s*(?:conversation|tools?|call|meta|internal|action)\b[^\]]*\]", pending_text_buf, re.IGNORECASE)
                                                if b_meta_match:
                                                    pre = pending_text_buf[:b_meta_match.start()]
                                                    if pre:
                                                        yield pre
                                                    pending_text_buf = pending_text_buf[b_meta_match.end():]
                                                    continue

                                                # Strip internal planning headers e.g. GREETING_HANDLING: ...
                                                b_plan_match = re.search(r"^[A-Z_]+_HANDLING:\s*[^\n]*\n?", pending_text_buf, re.MULTILINE)
                                                if b_plan_match:
                                                    pre = pending_text_buf[:b_plan_match.start()]
                                                    if pre:
                                                        yield pre
                                                    pending_text_buf = pending_text_buf[b_plan_match.end():]
                                                    continue

                                                yield pending_text_buf
                                                pending_text_buf = ""
                                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
                                    continue

                        if in_qwen_tag and qwen_tag_buf:
                            parsed = parse_qwen_tool_call_block(qwen_tag_buf)
                            for p_tc in parsed:
                                tool_calls_acc[len(tool_calls_acc)] = p_tc

                        if pending_text_buf and not pending_text_buf.startswith("<") and not re.match(r"^\[(?:[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+|conversation|tools?)[^\]]*\]", pending_text_buf):
                            yield pending_text_buf

                        if tool_calls_acc:
                            yield f"__TOOL_CALL__{json.dumps(list(tool_calls_acc.values()))}"
                    return
            except (httpx.TransportError, TimeoutError) as exc:
                if received_output:
                    # Never replay a stream that already produced output: the caller has
                    # spoken or displayed it and a retry would duplicate content.
                    logger.error("llm_stream_dropped_after_output", attempt=attempt, error=str(exc))
                    raise LLMStreamError(
                        "The local model stream was interrupted after partial output.",
                        provider="local",
                        partial_output=True,
                    ) from exc
                # A stale pooled connection after a transport error can poison retries;
                # drop the cached client so the next attempt gets a fresh one.
                await self._rebuild_client()
                if attempt < STREAM_MAX_RETRIES and time.monotonic() < deadline:
                    delay = _jittered_delay(attempt)
                    logger.warning(
                        "llm_connection_retry",
                        attempt=attempt,
                        delay_s=round(delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("llm_stream_connect_failed", error=str(exc))
                yield "I encountered an error connecting to my local inference model."
                return

    @staticmethod
    def _trim_messages_for_overflow(
        messages: list[dict[str, Any]],
        error_text: str,
    ) -> list[dict[str, Any]] | None:
        """Trim history after a context-overflow 400 while preserving the current turn.

        The oldest turns are dropped first, the system prompt is retained, and the final
        user message plus any assistant/tool messages that follow it are always kept so
        the server never receives a system-only payload. Returns ``None`` when no safe
        change is possible, which prevents repeated no-op prune attempts.

        Args:
            messages: Messages that triggered the overflow.
            error_text: Server error body (used to infer the context budget).

        Returns:
            list[dict[str, Any]] | None: Trimmed message list, or None if unchanged.
        """
        if not messages:
            return None
        system_message = next((m for m in messages if m.get("role") == "system"), None)
        non_system = [m for m in messages if m.get("role") != "system"]
        last_user_index = next(
            (i for i in range(len(non_system) - 1, -1, -1) if non_system[i].get("role") == "user"),
            None,
        )
        if last_user_index is None:
            return None

        tail = [dict(m) for m in non_system[last_user_index:]]
        user_message = tail[0]
        user_text = str(user_message.get("content", ""))
        if "[User Instruction]:\n" in user_text:
            user_message["content"] = user_text.split("[User Instruction]:\n", 1)[1].strip()
        elif "[User Instruction]:" in user_text:
            user_message["content"] = user_text.split("[User Instruction]:", 1)[1].strip()

        budget = 8192
        match = re.search(r"(\d{3,7})\s*tokens?", error_text, re.IGNORECASE)
        if match:
            budget = max(512, int(match.group(1)) * 3 // 4)

        head = [dict(m) for m in non_system[:last_user_index]]
        base_prefix = [system_message] if system_message else []

        def _candidate() -> list[dict[str, Any]]:
            """Build the candidate payload from the current head and fixed tail."""
            return base_prefix + head + tail

        while head and _estimate_messages_tokens(_candidate()) > budget:
            # Drop whole oldest turns (user+assistant) rather than orphaning a tool result.
            drop = 2 if len(head) >= 2 else 1
            head = head[drop:]

        candidate = _candidate()
        if candidate == messages:
            candidate = base_prefix + tail
        if candidate == messages or not any(m.get("role") == "user" for m in candidate):
            return None
        return candidate

    async def health_check(self) -> bool:
        """Async, non-blocking llama-server health probe.

        Safe to call from the event-loop HealthMonitor loop (previously used the
        blocking `requests` API, stalling the loop for up to 3s every 30s).

        Returns:
            bool: True if /health responds with HTTP 200.
        """
        try:
            base = self.base_url.rsplit("/v1", 1)[0]
            async with self._lease_client() as client:
                resp = await client.get(f"{base}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("llm_health_check_failed", error=str(exc))
            return False


class GeminiLLM(LLMProvider):
    """Google Gemini Cloud streaming client using the OpenAI-compatible endpoint.

    Provides high-intelligence capabilities with up to 1,000,000 tokens of context.
    Connects to the official Google Gemini OpenAI compatibility layer.
    """

    DEFAULT_API_KEY = "[REDACTED_API_KEY]"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        model: str = "gemini-3.1-flash-lite",
        fallback_models: list[str] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or self.DEFAULT_API_KEY
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_models = fallback_models if fallback_models is not None else [
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
        ]

        self._client: httpx.AsyncClient | None = None
        self._model_cooldown: dict[str, float] = {}
        self._last_prompt_tokens: int = 0

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0)
            self._client = httpx.AsyncClient(timeout=25.0, http2=True, limits=limits)
        return self._client

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        # Sanitize messages for Gemini compatibility:
        # Google's OpenAI-compatible endpoint strictly forbids assistant messages containing
        # synthetic tool_calls if they lack a Google-generated thought_signature (HTTP 400).
        # We transform tool turns and synthetic tool_calls into clean conversational turns.
        sanitized_messages = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                tool_name = m.get("name") or "tool"
                content = m.get("content") or ""
                sanitized_messages.append({
                    "role": "user",
                    "content": f"[Tool result from {tool_name}]: {content}",
                })
            elif role == "assistant" and "tool_calls" in m:
                content = m.get("content") or ""
                if not content:
                    names = [tc.get("function", {}).get("name", "tool") for tc in m.get("tool_calls", [])]
                    content = f"Executing {', '.join(names)}..."
                sanitized_messages.append({
                    "role": "assistant",
                    "content": content,
                })
            else:
                sanitized_messages.append(m)

        # Dynamic model routing with circuit-breaker cooldown:
        # Models in cooldown are *skipped for this request* instead of being appended and
        # retried, which previously defeated the circuit breaker after a 429/503.
        now = time.time()
        all_candidates = [self.model] + [m for m in self.fallback_models if m != self.model]
        models_to_try = [m for m in all_candidates if self._model_cooldown.get(m, 0) < now]
        self._last_prompt_tokens = _estimate_messages_tokens(sanitized_messages)

        if not models_to_try:
            logger.warning("gemini_all_models_in_cooldown", models=all_candidates)
            yield "All Gemini models are cooling down after recent rate limits. Please retry shortly."
            return

        client = self._get_client()

        for model_name in models_to_try:
            received_output = False
            payload = {
                "model": model_name,
                "messages": sanitized_messages,
                "stream": True,
                "max_tokens": max_tokens,
                "stream_options": {"include_usage": True},
            }
            if tools:
                payload["tools"] = tools

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code == 200:
                        # Clear cooldown on verified success
                        if model_name in self._model_cooldown:
                            del self._model_cooldown[model_name]

                        tool_calls_acc: dict[int, dict] = {}
                        async for line in response.aiter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    usage = data.get("usage") or {}
                                    if usage.get("prompt_tokens"):
                                        self._last_prompt_tokens = int(usage["prompt_tokens"])
                                    choices = data.get("choices", [])
                                    if not choices:
                                        continue
                                    delta = choices[0].get("delta", {})

                                    if "tool_calls" in delta and delta["tool_calls"]:
                                        received_output = True
                                        for tc in delta["tool_calls"]:
                                            idx = tc.get("index", 0)
                                            if idx not in tool_calls_acc:
                                                tool_calls_acc[idx] = {
                                                    "id": tc.get("id") or f"call_{idx}_{int(time.time()*1000)}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": tc.get("function", {}).get("name", ""),
                                                        "arguments": "",
                                                    },
                                                }
                                            if tc.get("id"):
                                                tool_calls_acc[idx]["id"] = tc["id"]
                                            fn = tc.get("function", {})
                                            if fn.get("name"):
                                                tool_calls_acc[idx]["function"]["name"] = fn["name"]
                                            if fn.get("arguments"):
                                                tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]
                                        continue

                                    reasoning = delta.get("reasoning_content") or delta.get("thought")
                                    if reasoning:
                                        received_output = True
                                        yield f"__THOUGHT__{reasoning}"

                                    token = delta.get("content", "")
                                    if token:
                                        received_output = True
                                        yield token
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    continue

                        if tool_calls_acc:
                            yield f"__TOOL_CALL__{json.dumps(list(tool_calls_acc.values()))}"
                        return
                    else:
                        err_body = await response.aread()
                        err_text = err_body.decode(errors="replace")
                        # If model hit 429, 503, or 404, place into 300s cooldown
                        if response.status_code in (429, 503, 404):
                            self._model_cooldown[model_name] = time.time() + 300.0
                            logger.warning(
                                "gemini_model_placed_in_cooldown",
                                model=model_name,
                                status=response.status_code,
                                cooldown_s=300,
                                error=err_text[:120],
                            )
                        else:
                            logger.warning(
                                "gemini_stream_model_fallback",
                                model=model_name,
                                status=response.status_code,
                                error=err_text[:120],
                            )
                        continue
            except Exception as exc:
                if received_output:
                    # Abort failover after partial output: switching models would replay
                    # and duplicate what the user already heard or saw.
                    logger.error(
                        "gemini_stream_failed_after_output",
                        model=model_name,
                        error=str(exc),
                    )
                    raise LLMStreamError(
                        "The Gemini stream was interrupted after partial output.",
                        provider="gemini",
                        partial_output=True,
                    ) from exc
                logger.warning("gemini_request_exception", model=model_name, error=str(exc))
                continue

        yield "I encountered an error connecting to the Gemini cloud API service. Please verify network access."

    async def health_check(self) -> bool:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                return resp.status_code == 200
        except Exception:
            return False


class OpenAICompatibleLLM(LLMProvider):
    """OpenAI-compatible streaming LLM client with prompt caching and model discovery.

    Supports OpenAI, OpenRouter, DeepSeek, Groq, Mistral, Ollama, vLLM, and LM Studio.
    Implements prefix-preserving prompt caching and dynamic /models querying.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-4o",
        context_window: int = 128000,
        max_tokens: int = 1024,
        enable_prompt_caching: bool = True,
        timeout: float = 45.0,
    ) -> None:
        """Initialize the OpenAI-compatible streaming client.

        Args:
            base_url: Base endpoint URL (e.g. 'https://api.openai.com/v1').
            api_key: Optional API key or bearer token.
            model: Target model name (e.g. 'gpt-4o', 'deepseek-chat').
            context_window: Maximum context window in tokens.
            max_tokens: Maximum tokens to generate per turn.
            enable_prompt_caching: Whether to enable prefix prompt caching optimization.
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        if not api_key:
            if "openrouter.ai" in self.base_url:
                api_key = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
            else:
                api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
        self.api_key = api_key
        self.model = model
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.enable_prompt_caching = enable_prompt_caching
        self.timeout = timeout
        self.last_cached_tokens: int = 0
        self._last_prompt_tokens: int = 0
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return a lazily created, cached AsyncClient with HTTP keep-alive.

        Reusing one client across turns avoids a fresh TCP/TLS handshake per request.

        Returns:
            httpx.AsyncClient: Cached client for this provider instance.
        """
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0)
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=limits,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the cached HTTP client (call when the engine replaces this provider)."""
        if self._client is not None and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = None

    def _supports_stream_usage(self) -> bool:
        """Return whether the endpoint is known to accept ``stream_options.include_usage``.

        Returns:
            bool: True for OpenAI-compatible providers that implement streaming usage.
        """
        base = self.base_url.lower()
        known = (
            "openai.com",
            "openrouter.ai",
            "deepseek.com",
            "groq.com",
            "mistral.ai",
            "together.ai",
            "together.xyz",
            "fireworks.ai",
            "aicredits.in",
            "localhost",
            "127.0.0.1",
        )
        if "anthropic" in base:
            return False
        return any(marker in base for marker in known)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens from the OpenAI-compatible endpoint.

        Args:
            messages: List of conversation messages adhering to chat format.
            max_tokens: Maximum tokens to generate for this turn.
            tools: Optional canonical tool schemas to pass to the model.

        Yields:
            str: Generated token fragments or '__TOOL_CALL__{...}' markers.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # OpenRouter attribution headers for optimized prompt routing
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/ameya/robot"
            headers["X-Title"] = "Robot Core Humanoid Brain"

        # Format tools schemas into OpenAI-standard function definitions
        formatted_tools: list[dict[str, Any]] | None = None
        if tools:
            formatted_tools = []
            for t in tools:
                if "function" in t:
                    formatted_tools.append({
                        "type": "function",
                        "function": t["function"],
                    })
                elif "name" in t:
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        },
                    })

        msgs_payload = copy.deepcopy(messages)

        # Prompt Caching optimization for Anthropic / OpenRouter / AICredits / Claude endpoints
        is_claude_or_caching_proxy = (
            "openrouter" in self.base_url
            or "anthropic" in self.base_url
            or "aicredits" in self.base_url
            or "claude" in (self.model or "").lower()
        )
        if self.enable_prompt_caching and is_claude_or_caching_proxy:
            if msgs_payload and msgs_payload[0].get("role") == "system":
                sys_content = msgs_payload[0].get("content")
                if isinstance(sys_content, str):
                    msgs_payload[0]["content"] = [
                        {
                            "type": "text",
                            "text": sys_content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs_payload,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        if formatted_tools:
            payload["tools"] = formatted_tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        err_str = err_body.decode(errors="replace")
                        logger.error(
                            "openai_compatible_http_error",
                            status=response.status_code,
                            body=err_str[:200],
                            model=self.model,
                        )
                        yield f"Error from OpenAI-compatible provider ({response.status_code}): {err_str[:150]}"
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            # Record prompt caching telemetry if returned
                            if "usage" in chunk and chunk["usage"]:
                                details = chunk["usage"].get("prompt_tokens_details") or {}
                                cached = details.get("cached_tokens", 0)
                                if cached:
                                    self.last_cached_tokens = cached
                                    logger.debug("openai_compatible_cache_hit", cached_tokens=cached)

                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            # Accumulate streaming tool call deltas
                            if delta.get("tool_calls"):
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_acc:
                                        tool_calls_acc[idx] = {
                                            "id": tc.get("id", f"call_{idx}"),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    fn = tc.get("function", {})
                                    if fn.get("name"):
                                        tool_calls_acc[idx]["function"]["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]
                                continue

                            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                            if reasoning:
                                yield f"__THOUGHT__{reasoning}"

                            token = delta.get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

            if tool_calls_acc:
                yield f"__TOOL_CALL__{json.dumps(list(tool_calls_acc.values()))}"

        except Exception as exc:
            logger.error("openai_compatible_stream_exception", model=self.model, error=str(exc))
            yield f"Error streaming from OpenAI-compatible provider: {exc}"

    async def scan_models(self) -> list[dict[str, Any]]:
        """Query the /models endpoint to discover all available models.

        Returns:
            list[dict[str, Any]]: List of discovered models with 'id', 'name',
                and metadata, sorted alphabetically.

        Raises:
            RuntimeError: If endpoint connection or authentication fails.
        """
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/ameya/robot"
            headers["X-Title"] = "Robot Core Humanoid Brain"

        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_list = data.get("data") or data.get("models") or []
                    models: list[dict[str, Any]] = []
                    for item in raw_list:
                        if isinstance(item, str):
                            m_id = item
                            is_free = ":free" in m_id.lower()
                        elif isinstance(item, dict):
                            m_id = item.get("id") or item.get("name") or ""
                            pricing = item.get("pricing", {})
                            prompt_price = pricing.get("prompt") if isinstance(pricing, dict) else None
                            is_free = bool(
                                ":free" in m_id.lower()
                                or m_id.lower().endswith("-free")
                                or (prompt_price in ("0", 0, "0.0", 0.0))
                            )
                        else:
                            continue
                        if m_id:
                            models.append({
                                "id": m_id,
                                "name": m_id,
                                "is_free": is_free,
                            })
                    # Sort so free models and active models are easily discoverable
                    models.sort(key=lambda x: (not x.get("is_free", False), str(x["id"]).lower()))
                    logger.info("openai_models_scanned", count=len(models), base_url=self.base_url)
                    return models
                else:
                    err_txt = resp.text[:200]
                    raise RuntimeError(f"HTTP {resp.status_code}: {err_txt}")
        except Exception as e:
            logger.warning("openai_models_scan_failed", error=str(e), url=url)
            raise RuntimeError(f"Failed to scan models from {url}: {e}")

    async def health_check(self) -> bool:
        """Check availability and connectivity of the OpenAI-compatible endpoint.

        Returns:
            bool: True if endpoint responds with HTTP 200, False otherwise.
        """
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/models", headers=headers)
                if resp.status_code == 200:
                    return True
                post_resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    },
                    headers=headers,
                )
                return post_resp.status_code == 200
        except Exception:
            return False

    chat = stream_chat


class LlamaServerManager:
    """Manages the local llama-server process lifecycle on the RTX 3050 GPU."""

    BINARY_PATH = "/home/ameya/llama.cpp/build/bin/llama-server"
    MODEL_PATH = "/home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf"
    HEALTH_URL = "http://127.0.0.1:8080/health"
    _active_model_id: str | None = None
    _process: subprocess.Popen[bytes] | None = None
    _log_file: TextIO | None = None

    @classmethod
    def is_running(cls) -> bool:
        try:
            resp = requests.get(cls.HEALTH_URL, timeout=0.8)
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    async def is_running_async(cls) -> bool:
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                resp = await client.get(cls.HEALTH_URL)
                return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def _close_log_file(cls) -> None:
        """Close the subprocess stdout/stderr log handle if open."""
        if cls._log_file is not None:
            try:
                cls._log_file.close()
            except Exception as e:
                logger.warning("llama_server_log_close_failed", error=str(e))
            cls._log_file = None

    @classmethod
    def _terminate_tracked(cls, pid: int) -> None:
        """Send SIGTERM to the tracked llama-server PID (preferring psutil).

        Args:
            pid: PID of the tracked llama-server process.
        """
        if psutil is not None:
            try:
                psutil.Process(pid).terminate()
                return
            except Exception:
                return  # Process already exited
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    @classmethod
    def _binary_path_matches(cls, candidate: str | None, binary: Path) -> bool:
        """Return True when a process executable/argv[0] resolves to the binary.

        Args:
            candidate: Process executable or argv[0], if readable.
            binary: Expected llama-server binary path.

        Returns:
            bool: Whether both paths resolve to the same executable.
        """
        if not candidate:
            return False
        try:
            return Path(candidate).resolve() == binary.resolve()
        except OSError:
            return False

    @classmethod
    def _discover_llama_pids(cls) -> list[int]:
        """Find llama-server processes this manager can safely identify.

        A process is a candidate only when its executable (or argv[0]) resolves
        to :attr:`BINARY_PATH`, or its cmdline contains the exact configured
        model path. Nothing else is ever signalled.

        Returns:
            list[int]: Matching PIDs, excluding this Python process.
        """
        if psutil is None:  # pragma: no cover - psutil is a hard dependency
            return []
        binary = Path(cls.BINARY_PATH)
        model_str = str(cls.MODEL_PATH or "")
        own_pid = os.getpid()
        matches: list[int] = []
        for proc in psutil.process_iter(["pid", "exe", "cmdline"]):
            try:
                pid = int(proc.info.get("pid") or 0)
                if pid <= 0 or pid == own_pid:
                    continue
                cmdline = [str(arg) for arg in (proc.info.get("cmdline") or []) if arg]
                matched = cls._binary_path_matches(proc.info.get("exe"), binary) or (
                    cls._binary_path_matches(cmdline[0] if cmdline else None, binary)
                )
                if not matched and model_str:
                    matched = any(model_str in arg for arg in cmdline)
                if matched:
                    matches.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                continue
        return matches

    @classmethod
    async def _terminate_discovered(cls) -> bool:
        """SIGTERM only the llama-server processes this manager can identify.

        Used when no tracked handle exists — for example a server left over
        from a previous run of this program. ``pkill -f`` is deliberately not
        used: it also kills unrelated processes sharing the name, including a
        second llama.cpp checkout the operator is developing against.

        Returns:
            bool: True when at least one candidate was identified and signalled.
        """
        pids = await asyncio.to_thread(cls._discover_llama_pids)
        if not pids:
            logger.warning("llama_server_stop_no_safe_target")
            return False

        logger.info("llama_server_terminating_discovered", pids=pids)
        processes = []
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                proc.terminate()  # SIGTERM only; escalated below
                processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        for _ in range(30):
            if not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in processes):
                break
            await asyncio.sleep(0.1)
        else:
            stragglers = [p for p in processes if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            logger.warning("llama_server_discovered_terminate_timeout", pids=[p.pid for p in stragglers])
            for proc in stragglers:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return bool(pids)

    @classmethod
    async def _release_tracked_process(cls) -> bool:
        """Terminate the tracked llama-server (SIGTERM, escalate to SIGKILL) and close fds.

        Returns:
            bool: True if a tracked process handle existed and was released.
        """
        proc = cls._process
        if proc is None:
            return False
        if proc.poll() is None:
            cls._terminate_tracked(proc.pid)
            for _ in range(30):  # Up to ~3s grace period for clean shutdown
                if proc.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            if proc.poll() is None:
                logger.warning("llama_server_terminate_grace_expired", pid=proc.pid)
                try:
                    proc.kill()
                except Exception as e:
                    logger.error("llama_server_kill_failed", error=str(e))
        cls._process = None
        cls._close_log_file()
        return True

    @classmethod
    async def stop(cls) -> bool:
        """Stop the llama-server to free 100% of GPU VRAM (3.4 GB).

        Terminates the tracked PID launched by this manager. When no tracked
        handle exists (e.g. a server left over from a previous program run),
        psutil is used to signal only those processes whose executable resolves
        to :attr:`BINARY_PATH` — unrelated processes sharing the name are never
        killed.

        Returns:
            bool: True if the health endpoint confirmed shutdown, False otherwise.
        """
        logger.info("stopping_llama_server_freeing_vram")
        try:
            if not await cls._release_tracked_process():
                await cls._terminate_discovered()

            for _ in range(25):
                if not await cls.is_running_async():
                    logger.info("llama_server_stopped_vram_released")
                    return True
                await asyncio.sleep(0.1)
            logger.error("llama_server_stop_not_confirmed")
            return False
        except Exception as e:
            logger.error("llama_server_stop_error", error=str(e))
            return False

    @classmethod
    async def start(
        cls,
        model_path: str | None = None,
        binary_path: str | None = None,
        mmproj_path: str | None = None,
        ngl: int | None = None,
        port: int = 8080,
        settings: LlamaServerSettings | None = None,
    ) -> bool:
        """Launch llama-server with FlashAttention offloaded to the RTX 3050 using advanced settings.

        Args:
            model_path: Path to GGUF model file.
            binary_path: Optional path to llama-server binary.
            mmproj_path: Optional path to multimodal vision projector file.
            ngl: Number of GPU layers to offload (overrides settings if provided).
            port: Port for the llama-server HTTP daemon.
            settings: Optional LlamaServerSettings configuration object.

        Returns:
            bool: True if server launched and passed health check, False otherwise.
        """
        if await cls.is_running_async():
            return True

        bin_path = binary_path or cls.BINARY_PATH
        m_path = model_path or cls.MODEL_PATH

        if not Path(bin_path).exists() or not Path(m_path).exists():
            logger.error("llama_server_binary_or_model_missing", bin=bin_path, model=m_path)
            return False

        active_settings = settings or LlamaServerSettings.load()
        if ngl is not None:
            active_settings.n_gpu_layers = ngl
        elif mmproj_path:
            active_settings.n_gpu_layers = min(active_settings.n_gpu_layers, 32)

        if mmproj_path and Path(mmproj_path).exists():
            active_settings.mmproj_path = str(mmproj_path)

        cmd = active_settings.build_cli_args(
            bin_path=bin_path,
            model_path=m_path,
            port=port,
        )

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            log_file = open_daemon_log(log_dir / "llama-server.log")
        except OSError as e:
            logger.error("llama_server_log_open_failed", error=str(e))
            return False

        logger.info("starting_llama_server", cmd=" ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)
        except Exception as e:
            logger.error("llama_server_spawn_failed", error=str(e))
            try:
                log_file.close()
            except Exception:
                pass
            return False

        # Track the handle and its log fd so stop()/start-timeout can reap exactly
        # this PID instead of leaking an orphan or pkill-ing unrelated processes.
        cls._process = proc
        cls._log_file = log_file

        for i in range(50):
            await asyncio.sleep(0.2)
            if proc.poll() is not None:
                logger.error("llama_server_exited_early", returncode=proc.returncode)
                await cls.stop()
                return False
            if await cls.is_running_async():
                logger.info("llama_server_ready", attempts=i + 1, pid=proc.pid)
                return True

        logger.error("llama_server_start_timeout", pid=proc.pid)
        # Start failed: terminate the tracked PID (then kill after grace) and close fds.
        await cls.stop()
        return False

    @classmethod
    async def get_settings(cls) -> dict[str, Any]:
        """Retrieve current llama-server configuration and live status.

        Returns:
            dict[str, Any]: Active settings dictionary and status metadata.
        """
        settings = await asyncio.to_thread(LlamaServerSettings.load)
        return {
            "settings": settings.to_dict(),
            "is_running": await cls.is_running_async(),
            "active_model": cls.get_active_model_id(),
        }

    @classmethod
    async def update_settings(
        cls,
        new_settings: dict[str, Any],
        reload: bool = False,
        engine: Any | None = None,
        port: int = 8080,
    ) -> dict[str, Any]:
        """Save updated llama-server settings and optionally hot-reload the running process.

        Args:
            new_settings: Dictionary of configuration parameters matching LlamaServerSettings.
            reload: Whether to restart llama-server immediately with new parameters.
            engine: Optional active RobotEngine to warm up after reload.
            port: Port for llama-server.

        Returns:
            dict[str, Any]: Status summary dictionary.
        """
        s = LlamaServerSettings.from_dict(new_settings)
        s.save()

        if reload:
            active_model = cls.get_active_model_id()
            res = await cls.switch_model(active_model, engine=engine, port=port)
            return {
                "status": "reloaded" if res.get("status") == "ok" else "error",
                "settings": s.to_dict(),
                "reload_result": res,
            }

        return {
            "status": "saved",
            "settings": s.to_dict(),
        }

    @classmethod
    async def reset_settings(
        cls,
        reload: bool = False,
        engine: Any | None = None,
        port: int = 8080,
    ) -> dict[str, Any]:
        """Reset llama-server configuration to optimal defaults for RTX 3050 4GB.

        Args:
            reload: Whether to restart llama-server immediately.
            engine: Optional active RobotEngine.
            port: Port for llama-server.

        Returns:
            dict[str, Any]: Status summary dictionary.
        """
        s = LlamaServerSettings()
        s.save()

        if reload:
            active_model = cls.get_active_model_id()
            res = await cls.switch_model(active_model, engine=engine, port=port)
            return {
                "status": "reloaded" if res.get("status") == "ok" else "error",
                "settings": s.to_dict(),
                "reload_result": res,
            }

        return {
            "status": "reset",
            "settings": s.to_dict(),
        }

    @classmethod
    def get_active_model_id(cls) -> str:
        """Determine currently active local model ID or filename.

        Returns:
            str: Model ID or filename string.
        """
        if cls._active_model_id:
            return cls._active_model_id

        # 1. Check live process
        try:
            res = subprocess.run(
                ["pgrep", "-a", "llama-server"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout:
                m = re.search(r"-m\s+([^\s]+)", res.stdout)
                if m:
                    stem = Path(m.group(1)).name
                    if stem == "Qwen3.5-4B-Q4_K_M.gguf":
                        return "qwen3.5-4b"
                    # Check registry
                    for reg_id, reg_meta in ModelRegistry.CATALOG.items():
                        if reg_meta.filename == stem:
                            return reg_id
                    return stem
        except Exception:
            pass

        # 2. Check persisted state file
        txt_path = Path("data/active_local_model.txt")
        if txt_path.exists():
            try:
                saved = txt_path.read_text(encoding="utf-8").strip()
                if saved:
                    return saved
            except Exception:
                pass

        # 3. Check config/default.yaml
        cfg_path = Path("config/default.yaml")
        if cfg_path.exists():
            try:
                import yaml
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                m = cfg.get("llm", {}).get("model", "")
                if m:
                    return m
            except Exception:
                pass

        return "qwen3.5-4b"

    @classmethod
    async def switch_model(
        cls,
        model_id_or_path: str,
        engine: Any | None = None,
        port: int = 8080,
    ) -> dict[str, Any]:
        """Hot-swap the local GGUF model on the RTX 3050 GPU.

        Args:
            model_id_or_path: Identifier from ModelRegistry or path to GGUF model file.
            engine: Optional active RobotEngine to warm up and re-bind.
            port: Port for the llama-server (default: 8080).

        Returns:
            dict[str, Any]: Result status dictionary.
        """
        logger.info("switching_local_llm_model", target=model_id_or_path)

        target_model_path: Path | None = None
        target_mmproj_path: Path | None = None
        is_vision = False
        resolved_id = model_id_or_path
        model_name = model_id_or_path
        ngl = 99

        # Case 1: Qwen 3.5 4B standard model
        if model_id_or_path in ("qwen3.5-4b", "Qwen3.5-4B", "Qwen3.5-4B-Q4_K_M.gguf"):
            resolved_id = "qwen3.5-4b"
            model_name = "Qwen 3.5 4B (Instruct)"
            default_qwen = Path("/home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf")
            if default_qwen.exists():
                target_model_path = default_qwen
        # Case 2: In registered model catalog
        elif model_id_or_path in ModelRegistry.CATALOG:
            meta = ModelRegistry.CATALOG[model_id_or_path]
            resolved_id = meta.id
            model_name = meta.name
            target_model_path = meta.get_model_path()
            if not target_model_path.exists():
                return {
                    "status": "error",
                    "message": f"Model '{meta.name}' is not downloaded yet. Please download it via ./scripts/download_models.sh --model {meta.id}.",
                }
            if meta.is_vision:
                is_vision = True
                target_mmproj_path = meta.get_mmproj_path()
                ngl = 32
        else:
            # Case 3: Direct filepath
            p = Path(model_id_or_path)
            if p.exists() and p.is_file():
                target_model_path = p
                resolved_id = p.stem
                model_name = p.stem

        if not target_model_path or not target_model_path.exists():
            return {
                "status": "error",
                "message": f"Could not find model file for '{model_id_or_path}' on disk.",
            }

        # Stop currently running server to evacuate VRAM
        await cls.stop()
        await asyncio.sleep(0.5)

        # Launch server with new model
        ok = await cls.start(
            model_path=str(target_model_path),
            mmproj_path=str(target_mmproj_path) if target_mmproj_path else None,
            ngl=ngl,
            port=port,
        )
        if not ok:
            return {
                "status": "error",
                "message": f"Failed to start llama-server with model '{resolved_id}'. Check logs/llama-server.log.",
            }

        cls._active_model_id = resolved_id

        # Update engine references and pre-warm slots if attached
        if engine:
            try:
                if hasattr(engine, "llm") and hasattr(engine.llm, "model"):
                    engine.llm.model = resolved_id
                if hasattr(engine, "llm_provider_type"):
                    engine.llm_provider_type = "local"
                if hasattr(engine, "_persist_llm_provider"):
                    engine._persist_llm_provider("local")
                if hasattr(engine, "warm_up_llm") and callable(engine.warm_up_llm):
                    p_name = getattr(engine, "active_personality_name", "neo")
                    coro = engine.warm_up_llm(p_name)
                    if asyncio.iscoroutine(coro):
                        spawn(coro, name="warm_up_llm")
                if hasattr(engine, "events") and hasattr(engine.events, "emit"):
                    emit_coro = engine.events.emit(
                        "llm.model_changed",
                        provider="local",
                        model=resolved_id,
                        model_name=model_name,
                        is_vision=is_vision,
                    )
                    if asyncio.iscoroutine(emit_coro):
                        await emit_coro
            except Exception as e:
                logger.warning("engine_local_model_rebind_error", error=str(e))

        cls._persist_local_model(resolved_id, target_model_path, target_mmproj_path)

        return {
            "status": "ok",
            "provider": "local",
            "model": resolved_id,
            "name": model_name,
            "model_path": str(target_model_path),
            "is_vision": is_vision,
            "vram_freed": False,
        }

    @classmethod
    def _persist_local_model(
        cls,
        model_id: str,
        model_path: Path,
        mmproj_path: Path | None = None,
    ) -> None:
        """Persist active local model selection to data/ and config/default.yaml.

        Args:
            model_id: Model identifier string.
            model_path: Resolved path to the GGUF model file.
            mmproj_path: Optional path to the mmproj file.
        """
        try:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "active_local_model.txt").write_text(model_id + "\n", encoding="utf-8")
        except Exception as e:
            logger.debug("failed_writing_active_local_model_file", error=str(e))

        try:
            import yaml
            cfg_path = Path("config/default.yaml")
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if "llm" not in cfg or not isinstance(cfg["llm"], dict):
                    cfg["llm"] = {}
                cfg["llm"]["model"] = model_id
                cfg["llm"]["model_path"] = str(model_path)
                if mmproj_path:
                    cfg["llm"]["multimodal"] = True
                    cfg["llm"]["mmproj_path"] = str(mmproj_path)
                else:
                    cfg["llm"]["multimodal"] = False
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.debug("failed_updating_default_yaml_local_model", error=str(e))


class ClauseChunker:
    """Buffers LLM tokens and emits speakable chunks for TTS.

    Design:
    1. First chunk: Emitted as soon as the first word or initial clause is ready
       (min_first_words=1..2) to initiate playback with near-zero latency.
    2. Subsequent chunks: Buffered and synthesized in the background on natural
       sentence terminals (.!?) or full semantic boundaries, preventing the choppy,
       unnatural rhythm of mid-sentence comma fragmentation.
    """

    def __init__(
        self,
        min_words: int = 4,
        max_words: int = 24,
        eager_first_chunk: bool = True,
        min_first_words: int = 2,
    ) -> None:
        self._buffer = ""
        self._min_words = min_words
        self._max_words = max_words
        self.eager_first_chunk = eager_first_chunk
        self.min_first_words = min_first_words
        self._first_chunk_emitted = False

    def feed(self, token: str) -> str | None:
        """Feed a token, return a complete speakable chunk if a boundary is reached."""
        self._buffer += token

        # Clean any raw tool call XML tags from buffer
        if "<tool_call" in self._buffer:
            self._buffer = re.sub(r"<tool_call>.*?</tool_call>", "", self._buffer, flags=re.DOTALL)
            self._buffer = re.sub(r"</?(?:tool_call|function|parameter)[^>]*>", "", self._buffer, flags=re.IGNORECASE)
            if "<tool_call" in self._buffer:
                return None

        # If buffer contains any unclosed bracket tag like "[motion: ", wait for tag to close
        # Guard against unbounded waiting if model outputs an unclosed bracket or long description (>35 chars)
        last_open = self._buffer.rfind("[")
        if last_open != -1 and self._buffer.find("]", last_open) == -1:
            if len(self._buffer) - last_open <= 35:
                return None

        # Separate ALL leading bracket/inline tags (e.g. "[emotion: happy] [motion: wave] ")
        prefix_tag = ""
        check_buf = self._buffer
        tag_match = re.match(r"^(\s*\[[^\]]+\]\s*)+", check_buf)
        if tag_match:
            prefix_tag = tag_match.group(0)
            check_buf = check_buf[len(prefix_tag):]

        # 1. Terminal sentence punctuation detection (.!?)
        match = re.search(r"^(.*?[.!?]+[\"'\)\]]*)(?:\s+|$)", check_buf, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # Guard against common abbreviations (e.g., Dr., Mr., etc.) or decimal numbers
            last_word = candidate.split()[-1] if candidate.split() else ""
            is_abbreviation = any(
                last_word.lower().endswith(abbr)
                for abbr in ("mr.", "mrs.", "ms.", "dr.", "prof.", "vs.", "etc.", "e.g.", "i.e.")
            ) or (re.search(r"\d\.\d*$", last_word) is not None)

            if not is_abbreviation:
                if len(candidate.split()) >= 1:
                    cut_len = len(prefix_tag) + match.end()
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    self._first_chunk_emitted = True
                    return full_chunk

        # 2. Eager first chunk clause release:
        # Check all early clause boundaries (, ; : \n —) that satisfy min_first_words
        if not self._first_chunk_emitted and self.eager_first_chunk:
            for m in re.finditer(r"([,;:\n—])(?:\s+|$)", check_buf):
                clause = check_buf[:m.end()].strip()
                if len(clause.split()) >= self.min_first_words:
                    cut_len = len(prefix_tag) + m.end()
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    self._first_chunk_emitted = True
                    return full_chunk

            # If 10 or more words accumulated without any punctuation, release early as a safety valve
            words = check_buf.strip().split()
            if len(words) >= 10:
                cut_pos = -1
                w_count = 0
                for idx, ch in enumerate(check_buf):
                    if ch in (" ", "\t"):
                        w_count += 1
                        if w_count == 8:
                            cut_pos = idx
                            break
                if cut_pos != -1:
                    cut_len = len(prefix_tag) + cut_pos
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    self._first_chunk_emitted = True
                    return full_chunk

        # 3. Subsequent chunks: prioritize natural multi-word phrases
        if self._first_chunk_emitted:
            # Check pause boundaries (, ; \n —)
            matches = list(re.finditer(r"([,;\n—])(?:\s+|$)", check_buf))
            for m in reversed(matches):
                clause = check_buf[:m.end()].strip()
                if len(clause.split()) >= self._min_words:
                    cut_len = len(prefix_tag) + m.end()
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    return full_chunk

            # Long unpunctuated phrase chunking (max_words buffer safety)
            words = check_buf.strip().split()
            if len(words) >= self._max_words:
                last_space = check_buf.rfind(" ")
                if last_space != -1:
                    cut_len = len(prefix_tag) + last_space
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    return full_chunk

        return None

    def flush(self) -> str | None:
        """Flush remaining buffer content."""
        remainder = re.sub(r"<tool_call>.*?</tool_call>", "", self._buffer, flags=re.DOTALL)
        remainder = re.sub(r"</?(?:tool_call|function|parameter)[^>]*>", "", remainder, flags=re.IGNORECASE).strip()
        self._buffer = ""
        self._first_chunk_emitted = False
        return remainder if remainder else None

