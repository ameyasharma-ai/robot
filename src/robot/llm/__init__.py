"""LLM provider interface and llama-server streaming client."""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

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


def parse_qwen_tool_call_block(block: str) -> list[dict]:
    """Parse Qwen-style XML or JSON tool calls from a <tool_call> or [tools] block."""
    import time as _time
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
                        "id": f"call_{int(_time.time()*1000)}_{len(parsed_calls)}",
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
                "id": f"call_{int(_time.time()*1000)}",
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
                "id": f"call_{int(_time.time()*1000)}_{len(parsed_calls)}",
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
            "id": f"call_{int(_time.time()*1000)}",
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
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url.rsplit('/v1', 1)[0]}/slots")
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
        except Exception:
            pass
        return {}

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 300,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens from local llama-server with KV cache preservation.

        Args:
            messages: Formatted message turns.
            max_tokens: Maximum tokens to generate.
            tools: Optional tool schemas. When `use_invariant_tools` is True and `canonical_tools`
                is registered, the invariant canonical schemas are retained to preserve token 0.

        Yields:
            Generated text tokens or tool call JSON indicators.
        """
        import time as _time

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

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if effective_tools:
            payload["tools"] = effective_tools

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            err_body = await response.aread()
                            err_str = err_body.decode("utf-8", errors="replace")
                            logger.error("llm_stream_error", status=response.status_code, body=err_str)
                            # Auto-recovery for context overflow: prune older messages from middle while preserving system and latest user turn
                            if response.status_code == 400 and ("exceeds" in err_str or "context" in err_str):
                                logger.warning("llm_context_overflow_recovery", attempt=attempt)
                                msgs = payload.get("messages", [])
                                system_msg = next((m for m in msgs if m.get("role") == "system"), None)
                                user_msg = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
                                last_msg = msgs[-1] if msgs else None
                                pruned: list[dict[str, Any]] = []
                                if system_msg:
                                    pruned.append(system_msg)
                                non_system = [m for m in msgs if m.get("role") != "system"]
                                if len(non_system) > 2 and attempt == 0:
                                    pruned.extend(non_system[-2:])
                                else:
                                    # Strip bloated environment telemetry/grounding and retain bare user instruction
                                    if user_msg:
                                        u_copy = dict(user_msg)
                                        u_text = str(u_copy.get("content", ""))
                                        if "[User Instruction]:\n" in u_text:
                                            u_copy["content"] = u_text.split("[User Instruction]:\n", 1)[1].strip()
                                        elif "[User Instruction]:" in u_text:
                                            u_copy["content"] = u_text.split("[User Instruction]:", 1)[1].strip()
                                        pruned.append(u_copy)
                                payload["messages"] = pruned or msgs
                                if attempt < max_retries:
                                    continue
                            yield "I encountered an error connecting to my local inference model."
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
                                    choice = data["choices"][0]
                                    delta = choice.get("delta", {})

                                    # Accumulate native OpenAI tool calls delta chunks
                                    if "tool_calls" in delta and delta["tool_calls"]:
                                        for tc in delta["tool_calls"]:
                                            idx = tc.get("index", 0)
                                            if idx not in tool_calls_acc:
                                                tool_calls_acc[idx] = {
                                                    "id": tc.get("id") or f"call_{idx}_{int(_time.time()*1000)}",
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
                                                        "id": f"call_{int(_time.time()*1000)}_{len(tool_calls_acc)}",
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
                                except (json.JSONDecodeError, KeyError, IndexError):
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
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < max_retries:
                    logger.warning("llm_connection_retry", attempt=attempt, error=str(exc))
                    await asyncio.sleep(0.3)
                else:
                    logger.error("llm_stream_connect_failed", error=str(exc))
                    yield "I encountered an error connecting to my local inference model."
                    return

    async def health_check(self) -> bool:
        try:
            base = self.base_url.rsplit("/v1", 1)[0]
            resp = requests.get(f"{base}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
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
        model: str = "gemini-2.0-flash",
        fallback_models: list[str] | None = None,
    ) -> None:
        import os
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or self.DEFAULT_API_KEY
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_models = fallback_models if fallback_models is not None else [
            "gemini-1.5-flash",
            "gemini-2.5-flash",
        ]

        self._client: httpx.AsyncClient | None = None
        self._model_cooldown: dict[str, float] = {}

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
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        import time as _time

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
        # Avoid wasting 2-3 seconds querying a model that hit 429 quota or 503 high demand.
        now = _time.time()
        all_candidates = [self.model] + [m for m in self.fallback_models if m != self.model]
        available_models = [m for m in all_candidates if self._model_cooldown.get(m, 0) < now]
        cooldown_models = [m for m in all_candidates if self._model_cooldown.get(m, 0) >= now]
        models_to_try = available_models + cooldown_models

        client = self._get_client()

        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": sanitized_messages,
                "stream": True,
                "max_tokens": max_tokens,
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
                                    choices = data.get("choices", [])
                                    if not choices:
                                        continue
                                    delta = choices[0].get("delta", {})

                                    if "tool_calls" in delta and delta["tool_calls"]:
                                        for tc in delta["tool_calls"]:
                                            idx = tc.get("index", 0)
                                            if idx not in tool_calls_acc:
                                                tool_calls_acc[idx] = {
                                                    "id": tc.get("id") or f"call_{idx}_{int(_time.time()*1000)}",
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
                                    if token:
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
                            self._model_cooldown[model_name] = _time.time() + 300.0
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
        import os as _os
        self.base_url = base_url.rstrip("/")
        if not api_key:
            if "openrouter.ai" in self.base_url:
                api_key = _os.environ.get("OPENROUTER_API_KEY", "") or _os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
            else:
                api_key = _os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
        self.api_key = api_key
        self.model = model
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.enable_prompt_caching = enable_prompt_caching
        self.timeout = timeout
        self.last_cached_tokens: int = 0

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
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
        import copy
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
    async def stop(cls) -> bool:
        """Kill the llama-server process to free 100% of GPU VRAM (3.4 GB)."""
        import subprocess
        logger.info("stopping_llama_server_freeing_vram")
        try:
            subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
            for _ in range(25):
                if not await cls.is_running_async():
                    logger.info("llama_server_stopped_vram_released")
                    return True
                await asyncio.sleep(0.1)
            return True
        except Exception as e:
            logger.error("llama_server_stop_error", error=str(e))
            return False

    @classmethod
    async def start(
        cls,
        model_path: str | None = None,
        binary_path: str | None = None,
        port: int = 8080,
    ) -> bool:
        """Launch llama-server with FlashAttention offloaded to the RTX 3050."""
        import subprocess
        from pathlib import Path

        if await cls.is_running_async():
            return True

        bin_path = binary_path or cls.BINARY_PATH
        m_path = model_path or cls.MODEL_PATH

        if not Path(bin_path).exists() or not Path(m_path).exists():
            logger.error("llama_server_binary_or_model_missing", bin=bin_path, model=m_path)
            return False

        cmd = [
            bin_path,
            "-m", m_path,
            "-ngl", "99",
            "-t", "8",
            "-tb", "8",
            "-fa", "on",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-c", "16384",
            "-np", "1",
            "-b", "4096",
            "-ub", "4096",
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "llama-server.log", "a")

        logger.info("starting_llama_server", cmd=" ".join(cmd))
        subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)

        for i in range(50):
            await asyncio.sleep(0.2)
            if await cls.is_running_async():
                logger.info("llama_server_ready", attempts=i + 1)
                return True

        logger.error("llama_server_start_timeout")
        return False


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

