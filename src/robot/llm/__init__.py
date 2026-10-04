"""LLM provider interface and llama-server streaming client."""

from __future__ import annotations

import asyncio
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


def parse_qwen_tool_call_block(block: str) -> list[dict]:
    """Parse Qwen-style XML or JSON tool calls from a <tool_call> block."""
    import time as _time
    content = re.sub(r"</?tool_call>", "", block).strip()
    if not content:
        return []
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
        import time as _time
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if tools:
            payload["tools"] = tools

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
                            # Auto-recovery for context overflow: strip tools and prune older messages
                            if response.status_code == 400 and ("exceeds" in err_str or "context" in err_str):
                                logger.warning("llm_context_overflow_recovery", attempt=attempt)
                                if "tools" in payload:
                                    del payload["tools"]
                                if len(payload.get("messages", [])) > 2:
                                    payload["messages"] = [payload["messages"][0], payload["messages"][-1]]
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
                                            # Hold back if pending_text_buf ends with an unclosed angle bracket
                                            match = re.search(r"<[^>]*$", pending_text_buf)
                                            if match:
                                                safe_len = match.start()
                                                if safe_len > 0:
                                                    yield pending_text_buf[:safe_len]
                                                    pending_text_buf = pending_text_buf[safe_len:]
                                            else:
                                                yield pending_text_buf
                                                pending_text_buf = ""
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    continue

                        if in_qwen_tag and qwen_tag_buf:
                            parsed = parse_qwen_tool_call_block(qwen_tag_buf)
                            for p_tc in parsed:
                                tool_calls_acc[len(tool_calls_acc)] = p_tc

                        if pending_text_buf and not pending_text_buf.startswith("<"):
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

        # If buffer starts with an unclosed bracket tag like "[motion: ", wait for tag to close
        if self._buffer.startswith("[") and "]" not in self._buffer:
            return None

        # Separate any leading bracket/inline motion tag (e.g. "[motion: lean_in] ")
        prefix_tag = ""
        check_buf = self._buffer
        tag_match = re.match(r"^(\s*\[[^\]]+\]\s*)", check_buf)
        if tag_match:
            prefix_tag = tag_match.group(1)
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

        # 2. Eager first chunk clause release: emit when an early clause boundary (, ; : \n —)
        # has accumulated at least min_first_words (e.g. 2 words) to kick off speech instantly
        if not self._first_chunk_emitted and self.eager_first_chunk:
            first_punct_match = re.search(r"^(.*?[,;:\n—])(?:\s+|$)", check_buf, re.DOTALL)
            if first_punct_match:
                clause = first_punct_match.group(1).strip()
                if len(clause.split()) >= self.min_first_words:
                    cut_len = len(prefix_tag) + first_punct_match.end()
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    self._first_chunk_emitted = True
                    return full_chunk

        # 3. For subsequent speech, prioritize natural multi-word phrases:
        # Only split on major pause boundaries (semicolon, newline, em-dash, or comma with sufficient length)
        # to ensure speech cadence sounds fluent and natural rather than fragmented.
        if self._first_chunk_emitted:
            # Paragraph or major clause break
            major_break_match = re.search(r"^(.*?[;\n—])(?:\s+|$)", check_buf, re.DOTALL)
            if major_break_match:
                clause = major_break_match.group(1).strip()
                if len(clause.split()) >= self._min_words:
                    cut_len = len(prefix_tag) + major_break_match.end()
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    return full_chunk

            # Comma boundary: require at least 6-8 words to prevent jerky comma stutters
            comma_match = re.search(r"^(.*?,)(?:\s+|$)", check_buf, re.DOTALL)
            if comma_match:
                clause = comma_match.group(1).strip()
                if len(clause.split()) >= max(6, self._min_words):
                    cut_len = len(prefix_tag) + comma_match.end()
                    full_chunk = self._buffer[:cut_len].strip()
                    self._buffer = self._buffer[cut_len:].lstrip()
                    return full_chunk

            # 4. Long unpunctuated phrase chunking (max_words buffer safety)
            words = check_buf.strip().split()
            if len(words) >= self._max_words:
                last_space = check_buf.rfind(" ")
                if last_space != -1 and len(check_buf[:last_space].split()) >= self._min_words:
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

