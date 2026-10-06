"""Context Engine — assembles the optimal 16K context window per turn.

For each inference turn, Robot Core assembles:
1. Active personality system prompt
2. Core robot/system policy
3. Current task state
4. Relevant Mem0 memories
5. Recent conversation
6. Current user input
7. Relevant tool definitions only
8. Current system/robot state
9. Relevant tool results
10. Compact visual/world state
"""

from __future__ import annotations

from typing import Any

from robot.telemetry import get_logger

logger = get_logger("context")

import re

GROUNDING_POLICY = (
    "Truth, Tools & Real-Time Action Directives:\n"
    "- Mandatory Tool Calling Directive: You possess real, executable tools for music, volume, time, memory, vision, motion, and web search. NEVER claim, roleplay, narrate, or promise in conversational text that you are performing an action without ACTIVELY invoking the corresponding tool via function calling in that very turn! Saying 'I am adjusting the volume for you' or 'Playing that song for you now' WITHOUT emitting the tool call is STRICTLY FORBIDDEN and considered a severe system malfunction.\n"
    "- Volume & Music Controls: Whenever the user asks to change volume ('set volume to 20%', 'louder', 'turn it down', 'mute'), replay or play again ('play it again', 'replay', 'play that song again'), pause, resume, or stop, you MUST immediately emit a `music.control` tool call. Never narrate volume adjustments in text.\n"
    "- Track Truth: When asked what song was last played or what song is playing, you MUST check `music.status` or `music.history`. NEVER guess, assume, or hallucinate a song title.\n"
    "- Natural Memory & Autonomous Persistence: When the user asks you to remember something, acknowledge it naturally. Do NOT call `memory.write` for casual conversational remarks. Call `memory.write` only when a vital enduring user preference or fact must be permanently stored across sessions.\n"
    "- Grounding Context Separation: Any environment grounding, player status, or tool results provided are background telemetry. Never repeat background labels to the user."
)


def format_system_prompt(personality_prompt: str, high_intelligence: bool = False) -> str:
    """Format canonical system prompt with Grounding Policy for 100% KV-cache invariance."""
    system_parts = [personality_prompt.strip(), GROUNDING_POLICY]
    if high_intelligence:
        system_parts.append(
            "High-Intelligence Cognitive Directives:\n"
            "- You are operating in High-Intelligence Deep Context Mode with comprehensive access to long-term memory, prior conversation archives, and deep real-time tool knowledge.\n"
            "- Seamlessly cross-reference past interactions, stored facts, and real-time grounding to provide deeply intelligent, thoughtful, and perceptive answers while staying natural."
        )
    return "\n\n".join(system_parts)


class ContextEngine:
    """Dynamic context assembly with budget-aware allocation.

    The 16K context window is divided into dynamic (not hard) partitions:
    - System/personality:    1–1.5K tokens
    - Relevant tool schemas: 1.5–2.5K tokens
    - Retrieved memories:    1.5–2.5K tokens
    - Recent conversation:   4–6K tokens
    - World/vision state:    0.5–1K tokens
    - Robot state:           0.25–0.5K tokens
    - User input:            Variable
    - Generation headroom:   3–4K tokens
    """

    def __init__(self, budget: dict[str, int] | None = None) -> None:
        self._budget = budget or {
            "system_personality": 400,
            "tool_schemas": 400,
            "retrieved_memories": 300,
            "recent_conversation": 450,
            "world_state": 150,
            "robot_state": 100,
            "generation_headroom": 300,
        }

    def build_messages(
        self,
        system_prompt: str | None = None,
        user_text: str | None = None,
        recent_conversation: list[dict] | None = None,
        conversation_archive: list[dict] | None = None,
        memories: list[dict] | None = None,
        all_facts: list[dict] | None = None,
        robot_state: dict | None = None,
        world_state: dict | None = None,
        tool_results: list[dict] | None = None,
        image_data_url: str | None = None,
        high_intelligence: bool = False,
        **kwargs,
    ) -> list[dict]:
        """Assemble the complete message list for LLM inference.

        In High-Intelligence Mode (Gemini 1M context):
        - Exploits up to 80% (800,000 tokens) of available context window.
        - Injects full multi-session conversation archive and all durable facts.
        - Formats deep structured knowledge hierarchies so the model is not overwhelmed.

        In Local Mode (RTX 3050 4K context):
        - Strict compact allocation (3 turns, 5 memories, compact tool snippets).
        - Preserves KV cache slot invariance for <250ms TTFT.
        """
        resolved_system = system_prompt or kwargs.get("personality_prompt") or ""
        resolved_user = user_text or kwargs.get("user_input") or ""
        resolved_recent = recent_conversation if recent_conversation is not None else kwargs.get("history")
        resolved_memories = memories if memories is not None else kwargs.get("retrieved_memories")

        messages = []

        # 1. System prompt (personality + policy)
        messages.append({"role": "system", "content": format_system_prompt(resolved_system, high_intelligence=high_intelligence)})
        grounding_parts = []

        # Local System Time & Date Grounding (Always present: Indian Standard Time / IST)
        try:
            from datetime import datetime
            import zoneinfo
            tz_ist = zoneinfo.ZoneInfo("Asia/Kolkata")
            now_ist = datetime.now(tz_ist)
        except Exception:
            from datetime import datetime
            now_ist = datetime.now()

        grounding_parts.append(
            f"[Current Local System Time (Indian Standard Time / IST)]:\n"
            f"  Current Time: {now_ist.strftime('%I:%M:%S %p IST')}\n"
            f"  Current Date: {now_ist.strftime('%A, %B %d, %Y')}\n"
            f"  ISO 8601: {now_ist.isoformat()}"
        )

        # Gate memories: do NOT inject irrelevant memories if user turn is an action/command
        action_patterns = [
            r"\b(?:volume|louder|quieter|mute|unmute)\b",
            r"\b(?:pause|resume|unpause|stop|replay|play\s+again)\b",
            r"^(?:play|put\s+on)\s+",
            r"\b(?:what\s+time|current\s+time|clock|time\s+is\s+it)\b",
            r"\b(?:system\s+status|health\s+check)\b",
        ]
        is_direct_action = any(re.search(pat, (resolved_user or "").lower()) for pat in action_patterns)

        # 2. Complete User Facts & Knowledge Base
        if high_intelligence and all_facts and not is_direct_action:
            facts_str = _format_all_facts(all_facts, budget_tokens=100000)
            if facts_str:
                grounding_parts.append(f"[Comprehensive User Knowledge Base & Memory Facts]:\n{facts_str}")
        elif resolved_memories and not is_direct_action:
            mem_budget = 50000 if high_intelligence else self._budget["retrieved_memories"]
            mem_str = _format_memories(resolved_memories, mem_budget)
            if mem_str:
                grounding_parts.append(f"[Relevant Memories]:\n{mem_str}")

        # 3. Physical World State & Music Player State
        if world_state and (world_state.get("people") or world_state.get("objects")):
            world_str = _format_world_state(world_state)
            if world_str:
                grounding_parts.append(f"[Physical World State]:\n{world_str}")

        music_state = kwargs.get("music_state")
        if not music_state:
            try:
                from robot.tools.builtins.music import MusicPlayerService
                ms = MusicPlayerService.get_instance()
                music_state = ms.get_status()
            except Exception:
                pass
        if music_state and (music_state.get("is_playing") or music_state.get("is_paused") or music_state.get("current_track", "None") != "None"):
            grounding_parts.append(
                f"[Active Music Player State]:\n"
                f"  is_playing: {music_state.get('is_playing', False)}\n"
                f"  is_paused: {music_state.get('is_paused', False)}\n"
                f"  current_track: {music_state.get('current_track', 'None')}\n"
                f"  source: {music_state.get('source', 'None')}\n"
                f"  volume: {int(music_state.get('volume', 0.8) * 100)}%"
            )

        # 4. Multi-Session Conversation Archive (Gemini only)
        if high_intelligence and conversation_archive:
            archive_str = _format_archive(conversation_archive, budget_tokens=400000)
            if archive_str:
                grounding_parts.append(f"[Historical Multi-Session Conversation Archive]:\n{archive_str}")

        # 5. Recent Conversation History (Current Session)
        conv_budget = 250000 if high_intelligence else self._budget["recent_conversation"]
        if resolved_recent:
            trimmed = _trim_conversation(
                resolved_recent, conv_budget
            )
            messages.extend(trimmed)

        # 6. Dynamic Context Grounding (Tool results from current turn / deep web extraction)
        if tool_results:
            tool_parts = []
            for tr in tool_results:
                tool_id = tr.get("id", "tool")
                result_text = str(tr.get("result", tr.get("error", "")))
                tool_parts.append(f"[{tool_id}]:\n{result_text}")
            grounding_parts.append("[Tool Results & Grounded Intelligence — use these to answer]:\n" + "\n\n".join(tool_parts))

        # 7. Current user input cleanly separated from background telemetry
        user_content = resolved_user
        if grounding_parts:
            user_content = (
                "[Environment & System Grounding]:\n"
                + "\n\n".join(grounding_parts)
                + "\n\n[User Instruction]:\n"
                + resolved_user
            )

        messages.append({"role": "user", "content": user_content})

        # Safeguard: ensure total context never exceeds 800,000 tokens (80% of 1M limit)
        token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))
        if high_intelligence and token_estimate > 800000:
            logger.info("enforcing_80pct_context_cap", estimated_tokens=token_estimate, target_cap=800000)
            # Cap excess from user_content if needed
            excess = token_estimate - 800000
            chars_to_trim = excess * 4
            if len(user_content) > chars_to_trim:
                messages[-1]["content"] = user_content[chars_to_trim:]
            token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))

        logger.info(
            "context_built",
            messages=len(messages),
            estimated_tokens=token_estimate,
            mode="high_intelligence" if high_intelligence else "low_latency",
        )

        return messages


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English."""
    return len(text) // 4


def _format_memories(memories: list[dict], budget_tokens: int) -> str:
    """Format retrieved memories within budget."""
    parts = []
    total = 0
    for mem in memories:
        if isinstance(mem, dict):
            text = mem.get("memory", mem.get("text", str(mem)))
        else:
            text = str(mem)
        text = text.strip()
        if not text:
            continue
        tokens = _estimate_tokens(text)
        if total + tokens > budget_tokens:
            break
        parts.append(f"- {text}")
        total += tokens
    return "\n".join(parts)


def _format_state(state: dict) -> str:
    """Format robot state compactly."""
    lines = []
    for key, value in state.items():
        if value is not None:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _format_world_state(world_state: dict) -> str:
    """Format physical world state compactly for LLM context."""
    lines = []
    # People
    people = world_state.get("people", [])
    people_count = world_state.get("people_count", len(people))
    lines.append(f"  people_present: {people_count}")
    if people:
        for idx, p in enumerate(people):
            lines.append(
                f"    - person_{idx+1}: pos={p.get('position', 'center')} "
                f"distance={p.get('distance', 'medium')} "
                f"attentive={p.get('looking_at_robot', True)}"
            )

    # Physical Objects (from continuous object detection & vision reasoning)
    objects = world_state.get("objects", [])
    objects_count = world_state.get("objects_count", len(objects))
    lines.append(f"  objects_detected: {objects_count}")
    if objects:
        for obj in objects:
            lines.append(
                f"    - {obj.get('label')}: pos={obj.get('position', 'center')} "
                f"conf={obj.get('confidence', 0.8)}"
            )

    lines.append(f"  motion_level: {world_state.get('motion', 'none')}")
    lines.append(f"  room_lighting: {world_state.get('lighting', 'normal')}")
    lines.append(f"  scene: {world_state.get('scene', 'indoor')}")
    return "\n".join(lines)


def _trim_conversation(messages: list[dict], budget_tokens: int) -> list[dict]:
    """Keep as many recent messages as fit within the token budget."""
    result = []
    total = 0
    for msg in reversed(messages):
        tokens = _estimate_tokens(msg.get("content", ""))
        if total + tokens > budget_tokens:
            break
        result.insert(0, {"role": msg["role"], "content": msg["content"]})
        total += tokens
    return result


def _format_all_facts(facts: list[dict], budget_tokens: int) -> str:
    """Format all durable user facts and preferences for high-intelligence context."""
    lines = []
    total = 0
    for f in facts:
        txt = f.get("memory", f.get("fact", str(f))).strip()
        if not txt:
            continue
        line = f"• {txt}"
        tokens = _estimate_tokens(line)
        if total + tokens > budget_tokens:
            break
        lines.append(line)
        total += tokens
    return "\n".join(lines)


def _format_archive(archive: list[dict], budget_tokens: int) -> str:
    """Format past multi-session conversation archive chronologically within budget."""
    turns = []
    total = 0
    for msg in reversed(archive):
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        session = msg.get("session_id", "")
        prefix = f"[{session[:6]}] " if session else ""
        line = f"{prefix}{role}: {content}"
        tokens = _estimate_tokens(line)
        if total + tokens > budget_tokens:
            break
        turns.insert(0, line)
        total += tokens
    return "\n".join(turns)
