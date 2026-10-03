"""Tool-Conversation Coordinator — Zero-Perceived-Latency Tool Calling.

Coordinates concurrent tool execution (Stream B) and dynamic LLM holding
conversation (Stream A), seamlessly transitioning into final answer continuation
upon tool result arrival without awkward pauses, repeated phrases, or fabricated facts.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator

from robot.core.state import RobotMode
from robot.llm import ClauseChunker
from robot.observability.models import SpanStatus
from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine
    from robot.llm import LLMProvider
    from robot.observability import ObservabilityHub
    from robot.tools import PermissionManager, ToolDefinition, ToolRegistry
    from robot.tts import PiperTTS

logger = get_logger("tool_coordinator")


class CoordinatorState(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    TOOL_DETECTED = "TOOL_DETECTED"
    HOLDING_CONVERSATION = "HOLDING_CONVERSATION"
    TOOL_RUNNING = "TOOL_RUNNING"
    RESULT_RECEIVED = "RESULT_RECEIVED"
    TRANSITIONING_TO_RESULT = "TRANSITIONING_TO_RESULT"
    FINAL_RESPONSE = "FINAL_RESPONSE"
    COMPLETED = "COMPLETED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class ToolExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class HoldingStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    GENERATING = "GENERATING"
    SPEAKING = "SPEAKING"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class ToolCategory(str, Enum):
    DIRECT = "DIRECT"              # Ultra-fast (<15ms): execute immediately, skip holding
    FORMAT = "FORMAT"              # Simple structured formatting
    LLM_REQUIRED = "LLM_REQUIRED"  # Multi-step/slow/network search: full concurrent holding
    ASYNC = "ASYNC"                # Long background (image generation)


@dataclass
class ToolContextState:
    """State tracking for a single tool-conversation lifecycle."""

    request_id: str
    trace_id: str
    conversation_id: str
    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    category: ToolCategory = ToolCategory.LLM_REQUIRED
    tool_status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    holding_status: HoldingStatus = HoldingStatus.NOT_STARTED
    spoken_holding_text: str = ""
    unspoken_holding_text: str = ""
    tool_result: Any = None
    tool_error: str | None = None
    final_response_status: str = "PENDING"
    segments: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    tool_start_time: float = 0.0
    tool_end_time: float = 0.0
    holding_first_token_time: float = 0.0
    holding_first_audio_time: float = 0.0
    final_first_token_time: float = 0.0
    final_first_audio_time: float = 0.0


class ToolConversationCoordinator:
    """Orchestrates zero-perceived-latency tool calling with dynamic LLM holding."""

    HOLDING_SYSTEM_PROMPT = (
        "You are the conversational holding voice of a sharp, natural companion robot. "
        "An external tool is actively running in the background to fetch information for the user. "
        "Your task is to keep the conversation alive naturally while the tool finishes.\n\n"
        "Guidelines:\n"
        "1. Speak 1 short, natural, punchy sentence (6 to 14 words max per segment).\n"
        "2. Acknowledge what you are checking, looking up, or verifying with confidence.\n"
        "3. You DO NOT have the tool result yet. NEVER invent, fabricate, or guess any facts, prices, weather, dates, or answers.\n"
        "4. DO NOT say robotic phrases like 'tool executing' or 'API running'. Speak like a real crewmate.\n"
        "5. Avoid repeating yourself. If you already said something, add a fresh natural comment.\n"
        "6. Do not use asterisks, markdown, emojis, or bullet points."
    )

    def __init__(
        self,
        engine: RobotEngine,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or {}
        self.max_holding_duration_ms: float = float(
            self.config.get("tools", {}).get("max_holding_duration_ms", 4500)
        )
        self.holding_max_tokens: int = int(
            self.config.get("tools", {}).get("holding_max_tokens", 22)
        )
        self.state: CoordinatorState = CoordinatorState.IDLE
        self._active_context: ToolContextState | None = None
        self._active_turn_id: str | None = None

    # ─── Fast Intent Router (Phase 24) ──────────────────────────────────────

    def fast_route(self, user_text: str) -> tuple[str, dict[str, Any], ToolCategory] | None:
        """Rapid regex/keyword intent extraction for common tool queries.

        Bypasses first-pass tool selection latency (< 1ms).
        """
        text = user_text.strip().lower()

        # 1. System Time / Date (DIRECT: < 5ms)
        time_triggers = (
            "what time is it", "what's the time", "tell me the time", "current time",
            "what day is it", "what's today's date", "what is today's date", "what is the date",
        )
        if any(tr in text for tr in time_triggers):
            return "system.time", {}, ToolCategory.DIRECT

        # 2. System Status (DIRECT)
        if text in ("system status", "robot status", "check status", "system health"):
            return "system.status", {}, ToolCategory.DIRECT

        # 3. Web Search (LLM_REQUIRED: 800ms - 2500ms)
        web_match = re.search(
            r"^(?:search(?:\s+the\s+web|\s+online)?\s+for|look\s+up|google|browse\s+for|find\s+out|find(?:\s+online)?)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if web_match:
            query = web_match.group(1).strip(" .?!")
            return "web.search", {"query": query}, ToolCategory.LLM_REQUIRED

        if "latest" in text and ("price" in text or "news" in text or "release" in text):
            # E.g. "latest RTX 5090 price"
            return "web.search", {"query": user_text.strip(" .?!")}, ToolCategory.LLM_REQUIRED

        # 4. Perception / Camera View (DIRECT/FORMAT: ~2ms via active YOLOv8 state)
        perception_triggers = (
            "what do you see", "what are you seeing", "describe what you see",
            "who is here", "what's in front of you", "what is in front of you",
            "look at me", "what am i holding", "what i am holding",
        )
        if any(tr in text for tr in perception_triggers):
            return "perception.what_do_you_see", {}, ToolCategory.DIRECT

        # 5. Image Generation (ASYNC: 3000ms - 8000ms)
        img_match = re.search(
            r"^(?:generate(?:\s+an)?\s+image\s+of|draw(?:\s+me)?(?:\s+an?)?|create(?:\s+an)?\s+image\s+of|paint(?:\s+an?)?|generate(?:\s+a)?\s+picture\s+of)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if img_match:
            prompt = img_match.group(1).strip(" .?!")
            return "image.generate", {"prompt": prompt}, ToolCategory.ASYNC

        # 6. Physical / Avatar Motions (FORMAT: ~5ms)
        motion_triggers = {
            "wave": "wave",
            "wave your hand": "wave",
            "nod": "nod",
            "shake head": "shake_head",
            "dance": "dance",
            "bow": "bow",
            "cheer": "cheer",
            "shrug": "shrug",
        }
        for trig, act in motion_triggers.items():
            if text == trig or text.startswith(f"{trig} ") or text.endswith(f" {trig}"):
                return "character.execute_action", {"action": act}, ToolCategory.FORMAT

        # 7. Memory Write / Recall (FORMAT: 10ms - 40ms)
        mem_write_match = re.search(
            r"^(?:remember(?:\s+that)?|store(?:\s+it\s+in\s+your\s+memory\s+that|that)?|keep\s+in\s+mind\s+that)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if mem_write_match:
            fact = mem_write_match.group(1).strip(" .?!")
            return "memory.write", {"text": fact}, ToolCategory.FORMAT

        mem_recall_match = re.search(
            r"^(?:who\s+is|what\s+is\s+my|who's|do\s+you\s+remember|what\s+was)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if mem_recall_match:
            query = mem_recall_match.group(1).strip(" .?!")
            return "memory.search", {"query": query}, ToolCategory.FORMAT

        return None

    # ─── Main Coordination Lifecycle (Phase 2 & 3) ──────────────────────────

    async def coordinate(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        category: ToolCategory,
        user_text: str,
        end_of_speech_time: float,
        interrupt: asyncio.Event,
        messages: list[dict],
        turn_id: str | None = None,
    ) -> str:
        """Run concurrent tool execution and dynamic LLM holding conversation.

        Transitions seamlessly into the final answer continuation upon tool result.
        """
        request_id = str(uuid.uuid4())[:8]
        trace_id = self.engine.observability.current_trace_id or request_id
        tool_call_id = f"call_{int(time.time() * 1000)}"

        ctx = ToolContextState(
            request_id=request_id,
            trace_id=trace_id,
            conversation_id=self.engine._session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            category=category,
        )
        self._active_context = ctx
        self._active_turn_id = turn_id or request_id

        tool_def = self.engine.tool_registry.get(tool_name)
        if not tool_def:
            logger.warning("coordinator_tool_not_found", tool=tool_name)
            return ""

        self._set_state(CoordinatorState.TOOL_DETECTED)
        self._emit_obs_event("tool_call_detected", tool_name=tool_name, args=tool_args)

        # ─── Ultra-Fast / Direct Tools Optimization (Phase 24 & 25) ──────────
        # If tool execution takes < 15ms (e.g. system.time, perception), run directly
        # without holding speech overhead.
        if category == ToolCategory.DIRECT:
            return await self._handle_direct_tool(
                ctx, tool_def, tool_args, user_text, end_of_speech_time, interrupt, messages
            )

        # ─── Parallel Streams: Task A (Tool) & Task B (Holding Speech) ───────
        self._set_state(CoordinatorState.HOLDING_CONVERSATION)
        self.engine.observability.set_pipeline_node("TOOLS", "RUNNING")
        self.engine.observability.set_pipeline_node("LLM", "RUNNING")

        tool_completed_event = asyncio.Event()
        holding_stop_event = asyncio.Event()

        # Stream B: Tool Execution
        async def _run_tool() -> None:
            ctx.tool_status = ToolExecutionStatus.RUNNING
            ctx.tool_start_time = time.perf_counter()
            self._emit_obs_event("tool_started", tool_name=tool_name)
            try:
                result_data = await self.engine.permissions.execute(tool_def, tool_args)
                if "error" in result_data:
                    ctx.tool_error = str(result_data["error"])
                    ctx.tool_status = ToolExecutionStatus.FAILED
                else:
                    ctx.tool_result = result_data.get("result", "")
                    ctx.tool_status = ToolExecutionStatus.COMPLETED
            except asyncio.CancelledError:
                ctx.tool_status = ToolExecutionStatus.CANCELLED
                raise
            except Exception as e:
                logger.error("coordinator_tool_failed", tool=tool_name, error=str(e))
                ctx.tool_error = str(e)
                ctx.tool_status = ToolExecutionStatus.FAILED
            finally:
                ctx.tool_end_time = time.perf_counter()
                tool_duration_ms = (ctx.tool_end_time - ctx.tool_start_time) * 1000
                self.engine.telemetry.record("TOOL_DURATION", tool_duration_ms)
                self.engine.observability.set_pipeline_node(
                    "TOOLS", "COMPLETED" if ctx.tool_status == ToolExecutionStatus.COMPLETED else "ERROR"
                )
                self._emit_obs_event(
                    "tool_result_received",
                    tool_name=tool_name,
                    duration_ms=tool_duration_ms,
                    status=ctx.tool_status.value,
                )
                tool_completed_event.set()

        tool_task = asyncio.create_task(_run_tool())

        # Stream A: Conversational Holding Generation
        holding_task = asyncio.create_task(
            self._generate_holding_stream(
                ctx=ctx,
                user_text=user_text,
                end_of_speech_time=end_of_speech_time,
                interrupt=interrupt,
                tool_completed_event=tool_completed_event,
                holding_stop_event=holding_stop_event,
            )
        )

        # Wait for tool completion or interrupt
        try:
            await tool_completed_event.wait()
        except asyncio.CancelledError:
            tool_task.cancel()
            holding_task.cancel()
            self._set_state(CoordinatorState.CANCELLED)
            raise

        # Tool result has arrived! Signal holding stream to finish at safe boundary
        self._set_state(CoordinatorState.RESULT_RECEIVED)
        holding_stop_event.set()

        # Let holding task cleanly stop current generation
        try:
            await asyncio.wait_for(holding_task, timeout=0.8)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            holding_task.cancel()

        if interrupt.is_set():
            self._set_state(CoordinatorState.CANCELLED)
            self._emit_obs_event("request_cancelled")
            return ctx.spoken_holding_text

        # Yield a tiny slice to let any remaining socket cleanup finish
        await asyncio.sleep(0.015)

        # ─── Seamless Transition into Final Response Continuation (Phase 10 & 22)
        self._set_state(CoordinatorState.TRANSITIONING_TO_RESULT)
        self._emit_obs_event("result_context_injected", tool_name=tool_name)

        final_response = await self._generate_continuation(
            ctx=ctx,
            user_text=user_text,
            end_of_speech_time=end_of_speech_time,
            interrupt=interrupt,
            messages=messages,
        )

        self._set_state(CoordinatorState.COMPLETED)
        self._emit_obs_event(
            "request_completed",
            tool_name=tool_name,
            spoken_holding_len=len(ctx.spoken_holding_text),
            final_response_len=len(final_response),
        )
        return final_response

    # ─── Holding Speech Stream Generator (Phase 4, 7, 8, 18, 19) ─────────────

    async def _generate_holding_stream(
        self,
        ctx: ToolContextState,
        user_text: str,
        end_of_speech_time: float,
        interrupt: asyncio.Event,
        tool_completed_event: asyncio.Event,
        holding_stop_event: asyncio.Event,
    ) -> None:
        """Stream dynamic, LLM-generated holding segments while tool executes."""
        ctx.holding_status = HoldingStatus.GENERATING
        self._emit_obs_event("holding_llm_started", tool_name=ctx.tool_name)

        start_time = time.perf_counter()
        segment_index = 0
        chunker = ClauseChunker()
        first_audio_emitted = False

        while not tool_completed_event.is_set() and not interrupt.is_set():
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms >= self.max_holding_duration_ms:
                # Reached max holding duration — enter quiet waiting to prevent unbounded monologue
                logger.info("holding_duration_cap_reached", elapsed_ms=elapsed_ms)
                ctx.holding_status = HoldingStatus.PAUSED
                break

            segment_index += 1
            # Build compact holding prompt avoiding repetition
            holding_messages = [
                {"role": "system", "content": self.HOLDING_SYSTEM_PROMPT},
            ]
            user_prompt = f"The user asked: '{user_text}'. You are currently running the '{ctx.tool_name}' tool."
            if ctx.spoken_holding_text:
                user_prompt += f" You already said: '{ctx.spoken_holding_text.strip()}'. Generate the next natural spoken phrase without repeating yourself."
            else:
                user_prompt += " Acknowledge this naturally and mention you are checking the latest details."

            holding_messages.append({"role": "user", "content": user_prompt})

            segment_text = ""
            t_seg_start = time.perf_counter()

            try:
                # LLM streaming with small token budget (5-20 tokens per segment)
                async for token in self.engine.llm.stream_chat(
                    holding_messages,
                    max_tokens=self.holding_max_tokens,
                ):
                    if interrupt.is_set() or tool_completed_event.is_set() or holding_stop_event.is_set():
                        break

                    if not ctx.holding_first_token_time:
                        ctx.holding_first_token_time = time.perf_counter()
                        holding_ttft = (ctx.holding_first_token_time - ctx.start_time) * 1000
                        self.engine.telemetry.record("HOLDING_TTFT", holding_ttft)
                        self._emit_obs_event("holding_llm_first_token", duration_ms=holding_ttft)

                    segment_text += token
                    phrase = chunker.feed(token)
                    if phrase and not interrupt.is_set():
                        clean_phrase = phrase.strip()
                        if clean_phrase:
                            ctx.spoken_holding_text = (ctx.spoken_holding_text + " " + clean_phrase).strip()
                            timing_ref = None
                            if not first_audio_emitted:
                                timing_ref = {"end_of_speech": end_of_speech_time, "t_tts_start": time.perf_counter()}
                                first_audio_emitted = True
                                self._emit_obs_event("holding_tts_started")

                            # Send to TTS queue immediately
                            await self.engine._tts_queue.put((clean_phrase, interrupt, timing_ref, None))
                            self._emit_obs_event("holding_segment_spoken", phrase=clean_phrase)

                # Flush remainder if safe and not interrupted
                rem = chunker.flush()
                if rem and not interrupt.is_set() and not tool_completed_event.is_set():
                    clean_rem = rem.strip()
                    if clean_rem:
                        ctx.spoken_holding_text = (ctx.spoken_holding_text + " " + clean_rem).strip()
                        await self.engine._tts_queue.put((clean_rem, interrupt, None, None))
                        self._emit_obs_event("holding_segment_spoken", phrase=clean_rem)

                if segment_text:
                    ctx.segments.append(segment_text.strip())
                    self._emit_obs_event(
                        "holding_segment_generated",
                        segment_index=segment_index,
                        text=segment_text.strip(),
                        duration_ms=(time.perf_counter() - t_seg_start) * 1000,
                    )

            except Exception as e:
                logger.error("holding_generation_error", error=str(e))
                # Fall back to quick deterministic acknowledgment if LLM fails
                if not ctx.spoken_holding_text and not interrupt.is_set():
                    fallback_phrase = "Checking that for you right now."
                    ctx.spoken_holding_text = fallback_phrase
                    timing_ref = {"end_of_speech": end_of_speech_time, "t_tts_start": time.perf_counter()}
                    await self.engine._tts_queue.put((fallback_phrase, interrupt, timing_ref, None))
                break

            # Brief pause between segments if tool is still running
            if not tool_completed_event.is_set() and not holding_stop_event.is_set():
                await asyncio.sleep(0.3)

        ctx.holding_status = HoldingStatus.COMPLETED

    # ─── Final Answer Continuation (Phase 10, 22, 23) ───────────────────────

    async def _generate_continuation(
        self,
        ctx: ToolContextState,
        user_text: str,
        end_of_speech_time: float,
        interrupt: asyncio.Event,
        messages: list[dict],
    ) -> str:
        """Stream final grounded response from where holding speech left off."""
        self._set_state(CoordinatorState.FINAL_RESPONSE)
        self.engine.observability.set_pipeline_node("LLM", "RUNNING")
        self._emit_obs_event("final_llm_started", tool_name=ctx.tool_name)

        t_final_start = time.perf_counter()

        # Format tool result cleanly
        result_text = str(ctx.tool_result) if ctx.tool_result is not None else (ctx.tool_error or "No data available")
        if isinstance(ctx.tool_result, list):
            snippets = []
            for item in ctx.tool_result:
                if isinstance(item, dict):
                    snippets.append(f"• {item.get('title', '')}: {item.get('snippet', '')} ({item.get('url', '')})")
                else:
                    snippets.append(str(item))
            result_text = "\n".join(snippets)

        # Build seamless continuation messages
        personality = self.engine.personalities.active
        sys_prompt = personality.system_prompt if personality else ""

        # Seamless bridging prompt
        spoken_prefix = ctx.spoken_holding_text.strip()
        continuation_messages = list(messages)

        instruction = (
            f"\n\n[Tool Execution Completed]\n"
            f"Tool: {ctx.tool_name}\n"
            f"Result: {result_text}\n\n"
        )
        if spoken_prefix:
            instruction += (
                f"You already spoke this holding phrase to the user: \"{spoken_prefix}\"\n"
                f"Continue seamlessly into the answer from where you left off. "
                f"DO NOT repeat what you already said, and deliver the final grounded response naturally."
            )
        else:
            instruction += "Deliver the final grounded response to the user based on the tool result."

        # Grounding in user turn
        continuation_messages.append({
            "role": "user",
            "content": f"{user_text}\n\n{instruction}",
        })

        chunker = ClauseChunker()
        full_continuation = ""
        first_final_token = False

        try:
            async for token in self.engine.llm.stream_chat(
                continuation_messages,
                max_tokens=self.config.get("llm", {}).get("max_tokens", 300),
            ):
                if interrupt.is_set():
                    self._set_state(CoordinatorState.CANCELLED)
                    break

                if not first_final_token:
                    first_final_token = True
                    ctx.final_first_token_time = time.perf_counter()
                    result_to_final_ttft = (ctx.final_first_token_time - ctx.tool_end_time) * 1000
                    self.engine.telemetry.record("RESULT_TO_FINAL_TTFT", result_to_final_ttft)
                    self._emit_obs_event("final_llm_first_token", duration_ms=result_to_final_ttft)

                full_continuation += token
                phrase = chunker.feed(token)
                if phrase and not interrupt.is_set():
                    clean_phrase = phrase.strip()
                    if clean_phrase:
                        await self.engine._tts_queue.put((clean_phrase, interrupt, None, None))

            rem = chunker.flush()
            if rem and not interrupt.is_set():
                clean_rem = rem.strip()
                if clean_rem:
                    await self.engine._tts_queue.put((clean_rem, interrupt, None, None))

        except Exception as e:
            logger.error("final_continuation_error", error=str(e))
            fallback_ans = f"Based on the results: {result_text[:120]}."
            if not full_continuation and not interrupt.is_set():
                await self.engine._tts_queue.put((fallback_ans, interrupt, None, None))
                full_continuation = fallback_ans

        # Combined conversational response (spoken holding + final continuation)
        combined_response = (spoken_prefix + " " + full_continuation).strip()
        return combined_response or full_continuation.strip()

    # ─── Direct Tool Fast-Path (Phase 24 & 25) ──────────────────────────────

    async def _handle_direct_tool(
        self,
        ctx: ToolContextState,
        tool_def: ToolDefinition,
        tool_args: dict[str, Any],
        user_text: str,
        end_of_speech_time: float,
        interrupt: asyncio.Event,
        messages: list[dict],
    ) -> str:
        """Instant execution (< 15ms) for local deterministic tools."""
        self._set_state(CoordinatorState.TOOL_RUNNING)
        ctx.tool_start_time = time.perf_counter()
        self._emit_obs_event("tool_started", tool_name=ctx.tool_name)

        try:
            res = await self.engine.permissions.execute(tool_def, tool_args)
            ctx.tool_result = res.get("result", res.get("error", ""))
            ctx.tool_status = ToolExecutionStatus.COMPLETED
        except Exception as e:
            ctx.tool_error = str(e)
            ctx.tool_status = ToolExecutionStatus.FAILED
        finally:
            ctx.tool_end_time = time.perf_counter()
            dur = (ctx.tool_end_time - ctx.tool_start_time) * 1000
            self._emit_obs_event("tool_result_received", tool_name=ctx.tool_name, duration_ms=dur)

        # Synthesize direct answer via LLM
        return await self._generate_continuation(
            ctx=ctx,
            user_text=user_text,
            end_of_speech_time=end_of_speech_time,
            interrupt=interrupt,
            messages=messages,
        )

    # ─── State & Observability Helpers ──────────────────────────────────────

    def _set_state(self, state: CoordinatorState) -> None:
        self.state = state
        logger.debug("coordinator_state_transition", state=state.value)
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                asyncio.create_task(
                    self.engine.events.emit(
                        "coordinator_state",
                        state=state.value,
                        tool=self._active_context.tool_name if self._active_context else None,
                    )
                )
        except RuntimeError:
            pass

    def _emit_obs_event(self, event_type: str, duration_ms: float = 0.0, **meta: Any) -> None:
        if self._active_context:
            meta["request_id"] = self._active_context.request_id
            meta["trace_id"] = self._active_context.trace_id
            meta["tool_name"] = self._active_context.tool_name
        self.engine.observability.emit_event(
            event_type=f"coordinator.{event_type}",
            component="COORDINATOR",
            operation=event_type.upper(),
            duration_ms=duration_ms,
            metadata=meta,
        )
