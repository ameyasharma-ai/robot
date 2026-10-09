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


def _extract_motion_from_phrase(text: str) -> tuple[str, dict | None]:
    """Extract and strip predefined motion markers from LLM output for synchronized audio-motion execution."""
    if not text:
        return text, None
    action_dict = None
    m = re.search(r"\[(?:motion|action):\s*([a-zA-Z_]+)(?:\s+([a-zA-Z]+))?\]", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\[(?:character\.)?execute_action[\s\(]+(?:action=)?['\"]?([a-zA-Z_]+)['\"]?(?:[\s,]+(?:side=)?['\"]?([a-zA-Z]+)['\"]?)?", text, re.IGNORECASE)
    if m:
        act = m.group(1).lower()
        side = m.group(2).lower() if m.group(2) else "right"
        action_dict = {"action": act, "side": side}

    if not action_dict:
        m2 = re.search(r"\*(waves?|raises?\s+(?:her\s+|his\s+|right\s+|left\s+)?hand|points?|kicks?|nods?|smiles?|stops?)\*", text, re.IGNORECASE)
        if m2:
            raw_act = m2.group(1).lower()
            act = "wave" if "wave" in raw_act else ("raise_hand" if "hand" in raw_act else ("point" if "point" in raw_act else ("kick" if "kick" in raw_act else ("nod" if "nod" in raw_act else "idle"))))
            side = "left" if "left" in raw_act else "right"
            action_dict = {"action": act, "side": side}

    # Clean out ALL stage directions, bracket tags (motions, emotions), and parentheses so TTS never speaks them
    cleaned = re.sub(r"\[[^\]]*\]", "", text)
    # Strip any lingering unclosed or orphaned brackets e.g. [motion: or wave]
    cleaned = re.sub(r"\[\s*(?:motion|action|emotion)?:?[^\]\n]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:wave|point|kick|nod|smile|happy|sad|raise_hand|stop)\s*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = re.sub(r"\*[^*]*\*", "", cleaned)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()
    return cleaned, action_dict


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
        time_pattern = r"\b(?:what(?:'s|\s+is)?(?:\s+the)?\s+time|current\s+time|tell\s+(?:me\s+)?(?:the\s+)?time|what\s+time(?:\s+is\s+it)?|what\s+is\s+the\s+date|what(?:'s|\s+is)?\s+today(?:'s)?\s+date|what\s+day\s+is\s+it)\b"
        if re.search(time_pattern, text) or any(tr in text for tr in ("what time", "what is the time", "current time", "what day is it", "today's date", "time in")):
            tz_match = re.search(r"\bin\s+([a-zA-Z\s/_]+)(?:\?|$)", text)
            tz_arg = tz_match.group(1).strip() if tz_match else None
            args = {"timezone_str": tz_arg} if tz_arg else {}
            return "system.time", args, ToolCategory.DIRECT

        # 2. System Status (DIRECT)
        if text in ("system status", "robot status", "check status", "system health"):
            return "system.status", {}, ToolCategory.DIRECT

        # 3. Perception / Camera View (DIRECT/FORMAT: ~2ms via active YOLOv8 state)
        perception_triggers = (
            "what do you see", "what are you seeing", "describe what you see",
            "who is here", "what's in front of you", "what is in front of you",
            "look at me", "what am i holding", "what i am holding",
            "what do u see", "can you see me", "do you see me", "look around",
            "what's that in my hand", "what is in my hand",
        )
        if any(tr in text for tr in perception_triggers):
            return "perception.what_do_you_see", {}, ToolCategory.DIRECT

        # 4. Physical / Avatar Motions (FORMAT: ~5ms)
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

        # 5. Memory Write (FORMAT: 10ms - 40ms)
        mem_write_match = re.search(
            r"^(?:remember(?:\s+that)?|store(?:\s+it\s+in\s+your\s+memory\s+that|that)?|keep\s+in\s+mind\s+that)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if mem_write_match:
            fact = mem_write_match.group(1).strip(" .?!")
            return "memory.write", {"text": fact}, ToolCategory.FORMAT

        # 6. Personal Memory Recall (personal relationships, user facts, single contact names)
        personal_mem_match = re.search(
            r"^(?:who\s+is\s+(?:my|our)\s+(.+)|what\s+is\s+(?:my|our)\s+(.+)|what\s+was\s+(?:my|our)\s+(.+)|who's\s+(?:my|our)\s+(.+)|do\s+you\s+remember\s+(.+)|what\s+did\s+i\s+(?:say|tell\s+you)\s+(?:about\s+)?(.+)|what\s+do\s+you\s+remember\s+about\s+(.+))$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if personal_mem_match:
            query = next(g for g in personal_mem_match.groups() if g is not None).strip(" .?!")
            return "memory.search", {"query": query}, ToolCategory.FORMAT

        # Single given name personal query (e.g. "Who is Anubha?")
        single_name_mem_match = re.match(
            r"^(?:who\s+is|who's)\s+([A-Za-z]+)\??$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if single_name_mem_match:
            name = single_name_mem_match.group(1).strip(" .?!")
            return "memory.search", {"query": name}, ToolCategory.FORMAT

        # 7. Web Search (LLM_REQUIRED: 800ms - 2500ms)
        web_match = re.search(
            r"^(?:search(?:\s+the\s+web|\s+online)?\s+for|look\s+up|google|browse\s+for|find\s+out(?:\s+about)?|find(?:\s+online)?)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if web_match:
            query = web_match.group(1).strip(" .?!")
            return "web.search", {"query": query}, ToolCategory.LLM_REQUIRED

        # Broad informational search queries: public figures, new/latest products, news, prices, scores, weather, historical events, etc.
        info_search_match = re.search(
            r"^(?:what(?:'s|\s+is)?\s+(?:the\s+)?(?:new|newest|latest|current)\s+(.+)|tell\s+me\s+about\s+(.+)|who\s+won\s+(.+)|what(?:'s|\s+is)?\s+the\s+price\s+of\s+(.+)|what(?:'s|\s+is)?\s+the\s+weather\s+(?:like\s+)?(?:in\s+)?(.+)|what(?:'s|\s+is)?\s+the\s+score\s+(?:of\s+)?(.+)|what\s+happened\s+(?:with|to|in)\s+(.+)|when\s+did\s+(.+)|who\s+is\s+([A-Za-z]+\s+[A-Za-z]+.*)|who's\s+([A-Za-z]+\s+[A-Za-z]+.*)|what\s+was\s+(?:the\s+)?(?:incident|plane|flight|hijacking|attack|accident|event|story|scandal|war|history|pilot)\s+(.+))$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if info_search_match:
            return "web.search", {"query": user_text.strip(" .?!")}, ToolCategory.LLM_REQUIRED

        # Real-world event, news, and history keywords
        web_event_keywords = ("hijack", "plane hijack", "flight hijack", "attacked", "who died", "when did", "plane crash", "terrorist", "election")
        if any(k in text for k in web_event_keywords):
            return "web.search", {"query": user_text.strip(" .?!")}, ToolCategory.LLM_REQUIRED

        # Tech products/brands with new/latest/price/release/specs
        tech_products = ("iphone", "pixel", "samsung", "galaxy", "macbook", "ipad", "android", "gpu", "rtx", "cpu", "intel", "amd", "nvidia", "tesla", "phone", "laptop", "bitcoin", "crypto")
        if any(prod in text for prod in tech_products) and any(w in text for w in ("new", "latest", "newest", "price", "release", "launch", "announced", "spec", "specs", "upcoming")):
            return "web.search", {"query": user_text.strip(" .?!")}, ToolCategory.LLM_REQUIRED

        # Deep Search queries
        deep_search_match = re.search(
            r"^(?:search\s+deeply(?:\s+about|\s+for)?|deep\s+search(?:\s+for|\s+about)?|look\s+deeply\s+into)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if deep_search_match:
            return "web.search", {"query": deep_search_match.group(1).strip(" .?!"), "deep": True}, ToolCategory.LLM_REQUIRED

        # Music Playback & Control (Spotify, SoundCloud, Local)
        music_play_match = re.search(
            r"^(?:(?:can\s+you\s+|could\s+you\s+|please\s+)?play(?:\s+some)?\s+music(?:\s+on\s+(spotify|soundcloud|local))?(?:\s+please)?|(?:can\s+you\s+|could\s+you\s+|please\s+)?play\s+(.+?)(?:\s+on\s+(spotify|soundcloud|local))?(?:\s+please)?)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if music_play_match and not any(w in text for w in ("game", "role", "video", "guitar")):
            q = music_play_match.group(2) or "relaxing chill music"
            src = music_play_match.group(1) or music_play_match.group(3) or "auto"
            return "music.play", {"query": q.strip(" .?!"), "source": src.lower()}, ToolCategory.LLM_REQUIRED

        # Check volume adjustment or composite restart + volume
        vol_match = re.search(r"(?:volume\s+(?:to\s+)?(\d{1,3})%?|(\d{1,3})%\s+vol(?:ume)?)", text)
        extracted_vol = float(vol_match.group(1) or vol_match.group(2)) / 100.0 if vol_match else None

        if any(w in text for w in ("pause music", "pause the music", "pause playback", "pause the song", "pause song", "stop music", "stop the music", "stop playing", "stop the song", "stop song", "stop it", "stop that", "resume music", "resume playback", "resume the song", "resume song", "resume", "replay", "replay the song", "replay song", "play again", "repeat", "repeat the song", "repeat that", "unpause", "restart", "restart it", "restart the song")):
            act = "pause" if "pause" in text else ("stop" if any(s in text for s in ("stop", "halt")) else ("replay" if any(r in text for r in ("replay", "again", "repeat", "restart")) else "resume"))
            args = {"action": act}
            if extracted_vol is not None:
                args["volume"] = extracted_vol
            return "music.control", args, ToolCategory.DIRECT

        if vol_match and any(w in text for w in ("volume", "vol", "sound", "turn it up", "turn it down")):
            return "music.control", {"action": "volume", "volume": extracted_vol}, ToolCategory.DIRECT

        eq_match = re.search(r"(?:equalizer|equaliser|eq|sound\s+profile|audio\s+profile)\s+(?:to\s+)?([a-z0-9_]+)|(?:switch|change|set)\s+(?:the\s+)?(?:preset|profile)\s+to\s+([a-z0-9_]+)", text)
        if eq_match:
            preset_val = eq_match.group(1) or eq_match.group(2)
            return "music.equalizer", {"preset": preset_val}, ToolCategory.DIRECT

        if any(k in text for k in ("what is playing", "what's playing", "what song is playing", "what is currently playing", "did it finish playing", "id it finish playing", "music status")):
            return "music.status", {}, ToolCategory.DIRECT

        if any(k in text for k in ("latest news", "breaking news", "stock price", "release date", "current weather")):
            return "web.search", {"query": user_text.strip(" .?!")}, ToolCategory.LLM_REQUIRED

        # 8. Image Generation (ASYNC: 3000ms - 8000ms)
        img_match = re.search(
            r"^(?:generate(?:\s+an)?\s+image\s+of|draw(?:\s+me)?(?:\s+an?)?|create(?:\s+an)?\s+image\s+of|paint(?:\s+an?)?|generate(?:\s+a)?\s+picture\s+of)\s+(.+)$",
            user_text.strip(),
            re.IGNORECASE,
        )
        if img_match:
            prompt = img_match.group(1).strip(" .?!")
            return "image.generate", {"prompt": prompt}, ToolCategory.ASYNC

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
        # Auto-detect ultra-fast direct tools (<15ms execution)
        direct_tools = {
            "system.time",
            "system.status",
            "perception.what_do_you_see",
            "perception.inspect_view",
            "music.control",
            "music.equalizer",
            "music.status",
            "music.history",
            "character.execute_action",
            "character.stop",
        }
        if tool_name in direct_tools:
            category = ToolCategory.DIRECT

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

        # ─── Autonomous Failover: Memory Miss -> Web Search ──────────────────
        # If memory.search was invoked for an entity/fact and yielded 0 results,
        # fail over to web.search before generating final continuation.
        if ctx.tool_name == "memory.search" and not interrupt.is_set():
            is_empty = (
                not ctx.tool_result
                or ctx.tool_result == []
                or ctx.tool_result == "[]"
                or ctx.tool_result == "No results found."
            )
            if is_empty:
                query_str = ctx.tool_args.get("query") or user_text
                personal_pronouns = ("my ", "i ", "me ", "mine ", "myself ", "our ")
                if not any(query_str.lower().startswith(p) for p in personal_pronouns):
                    web_tool = self.engine.tool_registry.get("web.search")
                    if web_tool:
                        logger.info("memory_miss_failover_to_web_search", query=query_str)
                        self._emit_obs_event("memory_miss_web_failover", query=query_str)
                        try:
                            w_res = await self.engine.permissions.execute(web_tool, {"query": query_str})
                            if w_res and not w_res.get("error") and w_res.get("result"):
                                ctx.tool_name = "web.search"
                                ctx.tool_args = {"query": query_str}
                                ctx.tool_result = w_res["result"]
                                ctx.tool_status = ToolExecutionStatus.COMPLETED
                        except Exception as e:
                            logger.warning("memory_failover_web_search_failed", error=str(e))

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
        self.engine.observability.set_pipeline_node("LLM", "COMPLETED")
        self.engine.observability.set_pipeline_node("TOOLS", "COMPLETED")
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

        # Emit instantaneous (<10ms) zero-latency opening acknowledgment to TTS
        opening_phrases = {
            "web.search": "Checking the latest info on that for you.",
            "image.generate": "Starting to render that image for you now.",
            "memory.write": "Remembering that for you.",
            "memory.search": "Checking my memory banks.",
            "music.control": "Adjusting the music player.",
        }
        if not ctx.spoken_holding_text and not interrupt.is_set():
            if ctx.tool_name == "music.play":
                song_q = ctx.tool_args.get("query", "").strip() if ctx.tool_args else ""
                src = ctx.tool_args.get("source", "").lower() if ctx.tool_args else ""
                on_src = " on Spotify" if "spotify" in src else ""
                if song_q:
                    initial_phrase = f"Putting on {song_q}{on_src}."
                else:
                    initial_phrase = f"Putting on the track for you{on_src}."
            else:
                initial_phrase = opening_phrases.get(ctx.tool_name, "Checking that for you right now.")

            ctx.spoken_holding_text = initial_phrase
            first_audio_emitted = True
            timing_ref = {"end_of_speech": end_of_speech_time, "t_tts_start": time.perf_counter()}
            await self.engine._tts_queue.put((initial_phrase, interrupt, timing_ref, None))
            self._emit_obs_event("holding_segment_spoken", phrase=initial_phrase)

        # Wait cleanly for tool execution to complete.
        # CRITICAL: We do NOT run a while loop calling stream_chat here!
        # Calling stream_chat with short holding prompts evicts slot 0 in llama-server,
        # which destroys the KV cache and spikes subsequent turn TTFT to ~2.2s.
        # It also produces repetitive, unnatural monologues.
        try:
            await asyncio.wait_for(tool_completed_event.wait(), timeout=self.max_holding_duration_ms / 1000.0)
        except asyncio.TimeoutError:
            logger.info("holding_duration_cap_reached")
            ctx.holding_status = HoldingStatus.PAUSED
        except asyncio.CancelledError:
            pass

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

        # For music.play, avoid speaking over the music track once it is launched
        if ctx.tool_name == "music.play" and not ctx.tool_error:
            self.engine.observability.set_pipeline_node("LLM", "COMPLETED")
            if ctx.spoken_holding_text:
                logger.info("music_starting_skipping_continuation_speech", text=ctx.spoken_holding_text)
                if not interrupt.is_set():
                    await self.engine._tts_queue.put(None)
                return ctx.spoken_holding_text
            track_name = "music"
            if isinstance(ctx.tool_result, dict):
                track_name = ctx.tool_result.get("title") or ctx.tool_result.get("track") or "music"
            ack = f"Playing {track_name}."
            await self.engine._tts_queue.put((ack, interrupt, None, None))
            if not interrupt.is_set():
                await self.engine._tts_queue.put(None)
            return ack

        # Format tool result cleanly without bloated tracking URLs
        # In Gemini High-Intelligence mode (1M context), provide full extracted article contents and rich multi-source data.
        # In Local mode (4K context), keep compact 3-item snippets with 160-char truncation to preserve KV cache slot 0.
        result_text = str(ctx.tool_result) if ctx.tool_result is not None else (ctx.tool_error or "No data available")
        is_gemini = getattr(self.engine, "llm_provider_type", "local") == "gemini"
        if isinstance(ctx.tool_result, list):
            snippets = []
            max_items = 8 if is_gemini else 5
            for item in ctx.tool_result[:max_items]:
                if isinstance(item, dict):
                    t = item.get("title", "").strip()
                    s = item.get("snippet", "").strip()
                    u = item.get("url", "")
                    content = item.get("content", "").strip()
                    if is_gemini and content:
                        snippets.append(f"• Source: {t}\n  URL: {u}\n  Full Extracted Article Content:\n{content}")
                    else:
                        if not is_gemini and len(s) > 400:
                            s = s[:400] + "..."
                        if t or s:
                            snippets.append(f"• {t}: {s}")
                else:
                    s_str = str(item)
                    if not is_gemini and len(s_str) > 400:
                        s_str = s_str[:400]
                    snippets.append(s_str)
            result_text = "\n\n".join(snippets)
        elif isinstance(ctx.tool_result, dict):
            if "summary" in ctx.tool_result:
                result_text = ctx.tool_result["summary"]
            elif "current_time" in ctx.tool_result:
                tr = ctx.tool_result
                result_text = f"Exact Current Time: {tr.get('current_time')}, Date: {tr.get('current_date')}, Timezone: {tr.get('timezone')}"
            else:
                import json as _json
                result_text = _json.dumps(ctx.tool_result, ensure_ascii=False, indent=2 if is_gemini else None)

        spoken_prefix = ctx.spoken_holding_text.strip()
        continuation_messages = [dict(m) for m in messages]

        # Ensure user turn is present
        has_user_query = (
            len(continuation_messages) > 0 and continuation_messages[-1].get("role") == "user"
        )
        if not has_user_query:
            continuation_messages.append({"role": "user", "content": user_text})

        # Append assistant tool_call turn (with any holding speech already uttered)
        import json as _json
        tool_args_str = _json.dumps(ctx.tool_args or {})
        continuation_messages.append({
            "role": "assistant",
            "content": spoken_prefix or None,
            "tool_calls": [
                {
                    "id": ctx.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": ctx.tool_name,
                        "arguments": tool_args_str,
                    },
                }
            ],
        })

        # Append tool output turn
        # For music.play, inject an explicit instruction directly into the tool result
        # to not describe motions, match genre energy, and keep the response brief.
        # NOTE: Do NOT append a 'system' role message here as Qwen's Jinja template requires
        # system messages to only appear at index 0.
        if ctx.tool_name == "music.play":
            result_text = (
                f"{result_text}\n"
                "[Instruction: The music has begun. Your spoken response MUST be at most 1 short sentence (under 6 words). "
                "CRITICAL VIBE MATCHING: strictly match the song's genre and mood. "
                "For hip-hop, rap, trap, drill, or rock tracks: DO NOT use affectionate, romantic, or cutesy terms (NEVER say 'my love', 'sweetheart', 'darling', or romantic affection). Be chill and cool. "
                "For explicit love ballads: warm affection is permitted. "
                "NEVER describe physical motions, gestures, body movements, or actions. Do NOT use asterisks. "
                "Keep it punchy, then STOP.]"
            )

        continuation_messages.append({
            "role": "tool",
            "name": ctx.tool_name,
            "tool_call_id": ctx.tool_call_id,
            "content": result_text,
        })

        chunker = ClauseChunker(eager_first_chunk=True, min_first_words=1)
        full_continuation = ""
        first_final_token = False
        first_audio_queue_emitted = False

        collected_continuation_tool_calls = []

        try:
            tools_schema = self.engine.tool_registry.get_schemas()
            async for token in self.engine.llm.stream_chat(
                continuation_messages,
                tools=tools_schema if tools_schema else None,
                max_tokens=self.config.get("llm", {}).get("max_tokens", 300),
            ):
                if interrupt.is_set():
                    self._set_state(CoordinatorState.CANCELLED)
                    break

                if token.startswith("__TOOL_CALL__"):
                    try:
                        import json as _json
                        tc_data = _json.loads(token[len("__TOOL_CALL__"):])
                        if isinstance(tc_data, list):
                            collected_continuation_tool_calls.extend(tc_data)
                        elif isinstance(tc_data, dict):
                            collected_continuation_tool_calls.append(tc_data)
                    except Exception as e:
                        logger.warning("continuation_tool_call_parse_error", error=str(e))
                    continue

                if not first_final_token:
                    first_final_token = True
                    ctx.final_first_token_time = time.perf_counter()
                    result_to_final_ttft = (ctx.final_first_token_time - ctx.tool_end_time) * 1000
                    self.engine.telemetry.record("RESULT_TO_FINAL_TTFT", result_to_final_ttft)
                    self._emit_obs_event("final_llm_first_token", duration_ms=result_to_final_ttft)

                full_continuation += token
                phrase = chunker.feed(token)
                if phrase and not interrupt.is_set():
                    # If music is actively playing, skip TTS to let music take over
                    if ctx.tool_name == "music.play":
                        try:
                            from robot.tools.builtins.music import MusicPlayerService
                            ms = MusicPlayerService.get_instance()
                            if getattr(ms, "_music_active", False):
                                logger.info("skipping_tts_music_playing", phrase=phrase[:30])
                                continue
                        except Exception:
                            pass
                    clean_phrase, inline_act = _extract_motion_from_phrase(phrase)
                    # For music.play, strip inline motion actions entirely
                    if ctx.tool_name == "music.play":
                        inline_act = None
                    if clean_phrase:
                        t_ref = None
                        if not spoken_prefix and end_of_speech_time > 0 and not first_audio_queue_emitted:
                            first_audio_queue_emitted = True
                            t_ref = {"end_of_speech": end_of_speech_time, "t_tts_start": time.perf_counter()}
                        await self.engine._tts_queue.put((clean_phrase, interrupt, t_ref, inline_act))

            rem = chunker.flush()
            if rem and not interrupt.is_set():
                clean_rem, in_act = _extract_motion_from_phrase(rem)
                if clean_rem:
                    t_ref = None
                    if not spoken_prefix and end_of_speech_time > 0 and not first_audio_queue_emitted:
                        first_audio_queue_emitted = True
                        t_ref = {"end_of_speech": end_of_speech_time, "t_tts_start": time.perf_counter()}
                    await self.engine._tts_queue.put((clean_rem, interrupt, t_ref, in_act))

            # If the LLM requested a follow-up tool call (e.g. web search after memory check):
            if collected_continuation_tool_calls and not interrupt.is_set():
                for tc in collected_continuation_tool_calls:
                    fn_info = tc.get("function", {})
                    fn_name = fn_info.get("name")
                    fn_args = {}
                    if fn_info.get("arguments"):
                        try:
                            fn_args = _json.loads(fn_info["arguments"])
                        except Exception:
                            fn_args = {"query": str(fn_info["arguments"])}
                    tool_def = self.engine.tool_registry.get(fn_name)
                    if tool_def:
                        logger.info("executing_continuation_tool", tool=fn_name, args=fn_args)
                        t_res = await self.engine.permissions.execute(tool_def, fn_args)
                        continuation_messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tc],
                        })
                        res_str = ""
                        if isinstance(t_res, list):
                            snippets = []
                            for itm in t_res[:3]:
                                if isinstance(itm, dict):
                                    t = itm.get("title", "")
                                    s = itm.get("snippet", "")
                                    snippets.append(f"• {t}: {s[:160]}")
                            res_str = "\n".join(snippets)
                        else:
                            res_str = str(t_res)[:400]
                        continuation_messages.append({
                            "role": "tool",
                            "name": fn_name,
                            "tool_call_id": tc.get("id", "call_1"),
                            "content": res_str,
                        })
                        chunker2 = ClauseChunker()
                        async for tok in self.engine.llm.stream_chat(continuation_messages, max_tokens=250):
                            if tok.startswith("__TOOL_CALL__"):
                                continue
                            full_continuation += tok
                            ph = chunker2.feed(tok)
                            if ph and not interrupt.is_set():
                                c_ph, in_a = _extract_motion_from_phrase(ph)
                                if c_ph:
                                    await self.engine._tts_queue.put((c_ph, interrupt, None, in_a))
                        rem2 = chunker2.flush()
                        if rem2 and not interrupt.is_set():
                            c_rem, in_a = _extract_motion_from_phrase(rem2)
                            if c_rem:
                                await self.engine._tts_queue.put((c_rem, interrupt, None, in_a))

            # Sentinel to mark normal end of turn (only if not interrupted)
            if not interrupt.is_set():
                await self.engine._tts_queue.put(None)

        except Exception as e:
            logger.error("final_continuation_error", error=str(e))
            fallback_ans = f"Based on the results: {result_text[:120]}."
            if not full_continuation and not interrupt.is_set():
                await self.engine._tts_queue.put((fallback_ans, interrupt, None, None))
                await self.engine._tts_queue.put(None)
                full_continuation = fallback_ans

        # Combined conversational response (spoken holding + final continuation)
        self.engine.observability.set_pipeline_node("LLM", "COMPLETED")
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
        try:
            return await self._generate_continuation(
                ctx=ctx,
                user_text=user_text,
                end_of_speech_time=end_of_speech_time,
                interrupt=interrupt,
                messages=messages,
            )
        finally:
            self._set_state(CoordinatorState.COMPLETED)
            self.engine.observability.set_pipeline_node("LLM", "COMPLETED")
            self.engine.observability.set_pipeline_node("TOOLS", "COMPLETED")

    # ─── State & Observability Helpers ──────────────────────────────────────

    def reset(self) -> None:
        """Reset the conversation coordinator state to IDLE."""
        self._set_state(CoordinatorState.IDLE)
        self._active_context = None

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
