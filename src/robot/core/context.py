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
    "- User Entity Ownership & Attribution Directive: The human speaking with you is your creator and conversational partner. All facts, likes, dislikes, family members, friends, and romantic interests (e.g. 'crush', 'partner', 'friends', 'goals') stored in user memories belong exclusively to the USER. NEVER claim or misattribute the user's personal relationships or crushes as your own! If a fact states 'Crush: Anubha', Anubha is the USER'S crush, not yours. You must refer to them accurately as 'your crush Anubha'.\n"
    "- Mandatory Tool Calling Directive: You possess real, executable tools for music, volume, time, memory, vision, motion, web search, timers, and alarms. NEVER claim, roleplay, narrate, or promise in conversational text that you are performing an action without ACTIVELY invoking the corresponding tool via function calling in that very turn! Saying 'I am adjusting the volume for you', 'Setting a timer for you now', or 'Playing that song for you now' WITHOUT emitting the tool call is STRICTLY FORBIDDEN and considered a severe system malfunction.\n"
    "- Timers, Alarms & Scheduling: When the user asks to set a timer ('set a timer for 10 seconds', '5 min timer for tea'), set an alarm ('wake me up at 7:30 AM', 'set daily alarm at 8 AM'), schedule a reminder ('remind me in 15 mins to stretch'), list active jobs, cancel a timer/alarm, or asks for a morning briefing, you MUST immediately invoke the corresponding tool (`timer.set`, `alarm.set`, `reminder.schedule`, `timer.list`, `alarm.list`, `reminder.list`, `timer.cancel`, `alarm.cancel`, `reminder.cancel`, `system.daily_briefing`). Never claim to set a timer or alarm without calling the tool.\n"
    "- Volume & Music Controls: Whenever the user asks to change volume ('set volume to 20%', 'louder', 'turn it down', 'mute'), replay or play again ('play it again', 'replay', 'play that song again'), pause, resume, or stop, you MUST immediately emit a `music.control` tool call. Never narrate volume adjustments in text.\n"
    "- Track Truth: When asked what song was last played or what song is playing, you MUST check `music.status` or `music.history`. NEVER guess, assume, or hallucinate a song title.\n"
    "- Memory Grounding & Awareness: You possess durable long-term memory across sessions. When asked about past facts, user preferences, prior conversations, or stored details, cross-reference the provided user profile and memories with 100% factual accuracy. Never claim to have forgotten something that is recorded in your memory facts.\n"
    "- Autonomous Persistence: When the user asks you to remember something ('remember that ...', 'keep in mind that ...'), acknowledge it naturally. Emit a `memory.write` call only when a vital enduring user preference or fact must be permanently stored across sessions.\n"
    "- Sequential Multi-Action Chaining Directive: When the user asks you to perform multiple actions or requests in a single turn (e.g. 'check what you see, play some music, and set volume to 50%'), you MUST invoke all required tools. NEVER stop after one tool and claim you did the others in text without calling the tool. In multi-step turns, address each step sequentially with real execution.\n"
    "- Factual Truth & Zero Hallucination: Ground every factual assertion directly in tool results, system telemetry, or memory facts provided. If information is absent or unverified, state clearly and concisely that you do not know or ask for clarification rather than inventing facts.\n"
    "- Grounding Context Separation: Any environment grounding, player status, or tool results provided are background telemetry. Never repeat raw background labels or formatting tags to the user."
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


def format_situation_banner(situation: dict | None = None) -> str:
    """Format real-time situational telemetry banner for zero-latency awareness.

    Args:
        situation: Optional dictionary containing battery, music_state,
            scheduler_jobs, and world_state.

    Returns:
        Formatted string summarizing current system situation.
    """
    sit = situation or {}
    parts = []

    # 1. Local Time (IST)
    try:
        from datetime import datetime
        import zoneinfo

        tz_ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        now_ist = datetime.now(tz_ist)
        # Bucket minute to 15-minute intervals to guarantee 100% KV cache prefix stability across multi-turn sessions
        bucketed_min = (now_ist.minute // 15) * 15
        time_str = f"{now_ist.replace(minute=bucketed_min, second=0).strftime('%I:%M %p IST')} ({now_ist.strftime('%A, %B %d, %Y')})"
    except Exception:
        from datetime import datetime

        now_ist = datetime.now()
        time_str = now_ist.strftime("%I:%M %p, %B %d, %Y")
    parts.append(f"• Current Time (Indian Standard Time / IST): {time_str}")

    # 2. Scheduler Jobs (timers, alarms, reminders)
    sched_jobs = sit.get("scheduler_jobs", [])
    if sched_jobs:
        job_strs = []
        for j in sched_jobs[:5]:
            if isinstance(j, dict):
                job_name = j.get("name", j.get("label", j.get("id", "job")))
                j_time = j.get("next_run", j.get("time", "pending"))
            elif hasattr(j, "to_dict"):
                j_d = j.to_dict()
                job_name = j_d.get("name", j_d.get("label", j_d.get("id", "job")))
                j_time = j_d.get("next_run", j_d.get("time", "pending"))
            else:
                job_name = getattr(j, "name", getattr(j, "label", getattr(j, "id", str(j))))
                j_time = getattr(j, "next_run", getattr(j, "time", "pending"))
            job_strs.append(f"{job_name} at {j_time}")
        parts.append(f"• Active Timers & Scheduled Jobs: {', '.join(job_strs)}")
    else:
        parts.append("• Active Timers & Scheduled Jobs: None")

    # 3. Active Music Track & Volume
    music = sit.get("music_state", {})
    if isinstance(music, object) and hasattr(music, "to_dict"):
        music = music.to_dict()
    elif not isinstance(music, dict):
        music = {}

    if music and (
        music.get("is_playing")
        or music.get("is_paused")
        or music.get("current_track", "None") != "None"
    ):
        status = (
            "Playing"
            if music.get("is_playing")
            else ("Paused" if music.get("is_paused") else "Idle")
        )
        vol_pct = int(music.get("volume", 0.8) * 100)
        parts.append(
            f"• Music Player: {status} | Track: {music.get('current_track', 'None')} | Vol: {vol_pct}%"
        )
    else:
        parts.append("• Music Player: Idle")

    # 4. Laptop Battery State
    battery = sit.get("battery", {})
    if isinstance(battery, object) and hasattr(battery, "to_dict"):
        battery = battery.to_dict()
    elif not isinstance(battery, dict):
        battery = {}

    if battery:
        percent = battery.get("percent", "Unknown")
        plugged = "Plugged In" if battery.get("power_plugged") else "Discharging"
        parts.append(f"• Laptop Battery: {percent}% ({plugged})")

    # 5. Visual Perception Scene
    world = sit.get("world_state", {})
    if isinstance(world, object) and hasattr(world, "to_dict"):
        world = world.to_dict()
    elif not isinstance(world, dict):
        world = {}
    if world:
        p_val = world.get("people_count", world.get("people", 0))
        people = len(p_val) if isinstance(p_val, (list, tuple, dict)) else p_val
        scene = world.get("scene", "indoor")
        parts.append(f"• Visual Scene: {people} person(s) present | Scene: {scene}")

    return "[Real-Time Situation Telemetry Banner]:\n" + "\n".join(parts)


def format_user_core_profile(user_profile: str | None = None) -> str:
    """Format non-evictable User Core Profile Sheet for context injection.

    Args:
        user_profile: Explicit user profile string or default profile template.

    Returns:
        Formatted user profile sheet string.
    """
    if user_profile and user_profile.strip():
        return user_profile.strip()

    return (
        "=== USER CORE IDENTITY PROFILE ===\n"
        "Identity & Role:\n"
        "• User Name: Ameya\n"
        "• User Role: AI Engineer & Robot Systems Creator\n\n"
        "Technical Profile:\n"
        "• Stack: Python, PyTorch, C++, WebSockets, FastAPI, Local LLMs, Web & Robotics\n\n"
        "Social Context & Relationships:\n"
        "• Primary Relationships belong exclusively to User (refer to them as 'your crush', 'your friend')\n\n"
        "Directives & Preferences:\n"
        "• Direct, intelligent, highly articulate, supportive companion tone"
    )


class ContextEngine:
    """Dynamic context assembly with budget-aware allocation.

    The 16K context window is divided into dynamic partitions:
    - System/personality:    0.5–1K tokens
    - Relevant tool schemas: 1.5–2K tokens
    - Retrieved memories:    2–3K tokens
    - Recent conversation:   8–10K tokens
    - World/vision state:    0.3–0.5K tokens
    - Robot state:           0.2–0.3K tokens
    - User input:            Variable
    - Generation headroom:   1.5–2K tokens
    """

    def __init__(self, budget: dict[str, int] | None = None) -> None:
        default_budget = {
            "system_personality": 600,
            "tool_schemas": 10500,
            "retrieved_memories": 800,
            "recent_conversation": 2500,
            "world_state": 200,
            "robot_state": 100,
            "generation_headroom": 1500,
        }
        if budget:
            default_budget.update(budget)
        self._budget = default_budget


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
        situation: dict | None = None,
        user_profile: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """Assemble the complete message list for LLM inference.

        In High-Intelligence Mode (Gemini 1M context):
        - Exploits up to 80% (800,000 tokens) of available context window.
        - Injects full multi-session conversation archive and all durable facts.
        - Formats deep structured knowledge hierarchies so the model is not overwhelmed.

        In Local Mode (RTX 3050 4K context):
        - Clean balanced allocation (up to 18-20 turns history, persistent facts & semantic recall).
        - Preserves KV cache slot invariance for <250ms TTFT.
        """
        resolved_system = system_prompt or kwargs.get("personality_prompt") or ""
        resolved_user = user_text or kwargs.get("user_input") or ""
        resolved_recent = recent_conversation if recent_conversation is not None else kwargs.get("history")
        resolved_memories = memories if memories is not None else kwargs.get("retrieved_memories")
        resolved_situation = situation or kwargs.get("situation")
        resolved_profile = user_profile or kwargs.get("user_profile")

        messages = []

        # 1. System prompt (personality + policy)
        messages.append({"role": "system", "content": format_system_prompt(resolved_system, high_intelligence=high_intelligence)})
        grounding_parts = []

        # 2. User Core Identity Profile Sheet (Static & Non-Evictable — placed first for 100% KV cache invariance)
        profile_sheet = format_user_core_profile(resolved_profile)
        grounding_parts.append(profile_sheet)

        # Gate memories: do NOT inject irrelevant memories if user turn is an action/command
        action_patterns = [
            r"\b(?:volume|louder|quieter|mute|unmute)\b",
            r"\b(?:pause|resume|unpause|stop|replay|play\s+again)\b",
            r"^(?:play|put\s+on)\s+",
            r"\b(?:what\s+time|current\s+time|clock|time\s+is\s+it)\b",
            r"\b(?:system\s+status|health\s+check)\b",
        ]
        is_direct_action = any(re.search(pat, (resolved_user or "").lower()) for pat in action_patterns)

        # 2. Complete User Facts & Knowledge Base (Durable facts placed first to ensure static KV-cache invariance)
        if high_intelligence and all_facts and not is_direct_action:
            facts_str = _format_all_facts(all_facts, budget_tokens=100000)
            if facts_str:
                grounding_parts.append(f"[Comprehensive User Knowledge Base & Memory Facts (Belonging to User)]:\n{facts_str}")
        elif (resolved_memories or all_facts) and not is_direct_action:
            combined_mems: list[dict] = []
            seen_texts = set()
            # Invariant durable facts come first, followed by query-specific recalled context
            for m in (all_facts or []) + (resolved_memories or []):
                if isinstance(m, dict):
                    txt = m.get("memory", m.get("fact", m.get("text", ""))).strip()
                    m_item = m
                else:
                    txt = str(m).strip()
                    m_item = {"memory": txt}
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    combined_mems.append(m_item)

            mem_budget = 50000 if high_intelligence else self._budget.get("retrieved_memories", 800)
            mem_str = _format_memories(combined_mems, mem_budget)
            if mem_str:
                grounding_parts.append(f"[User Personal Profile & Durable Facts (Belonging to User)]:\n{mem_str}")

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
        conv_budget = 250000 if high_intelligence else self._budget.get("recent_conversation", 2500)
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

        # 7. Real-Time Situation Telemetry Banner (IST Time, Pending Jobs, Track/Vol, Battery, Scene)
        # Placed at the end of grounding_parts so all preceding static context is 100% invariant for KV cache reuse
        sit_banner = format_situation_banner(resolved_situation)
        grounding_parts.append(sit_banner)

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

        # Safeguard for High-Intelligence Mode: ensure total context never exceeds 800,000 tokens (80% of 1M limit)
        token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))
        if high_intelligence and token_estimate > 800000:
            logger.info("enforcing_80pct_context_cap", estimated_tokens=token_estimate, target_cap=800000)
            excess = token_estimate - 800000
            chars_to_trim = excess * 4
            if len(user_content) > chars_to_trim:
                messages[-1]["content"] = user_content[chars_to_trim:]
            token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))

        # Safeguard for Local Mode (RTX 3050 16K context limit):
        # Tools take ~10,280 tokens. Capping messages at 4,800 tokens guarantees total prompt
        # stays at ~15,080 tokens, leaving >1,300 tokens headroom for generation.
        max_local_cap = 4800
        if not high_intelligence and token_estimate > max_local_cap:
            logger.warning("enforcing_local_context_cap", estimated_tokens=token_estimate, target_cap=max_local_cap)
            while len(messages) > 2 and token_estimate > max_local_cap:
                popped = messages.pop(1)
                token_estimate -= _estimate_tokens(str(popped.get("content") or ""))
            while len(messages) > 2 and messages[1].get("role") in ("tool", "assistant"):
                popped = messages.pop(1)
                token_estimate -= _estimate_tokens(str(popped.get("content") or ""))

            # If user message with grounding is still exceeding cap on its own, trim grounding from the TAIL
            # NEVER slice from the front, as front-slicing shifts offsets and destroys Token 0 KV-cache prefix invariance!
            if token_estimate > max_local_cap and len(messages) >= 2:
                u_text = str(messages[-1].get("content") or "")
                if "[User Instruction]:\n" in u_text:
                    parts = u_text.split("[User Instruction]:\n", 1)
                    grounding = parts[0]
                    instruction = parts[1]
                    excess_chars = (token_estimate - max_local_cap) * 4
                    if len(grounding) > excess_chars:
                        trimmed_grounding = grounding[:-excess_chars]
                        messages[-1]["content"] = f"{trimmed_grounding}[User Instruction]:\n{instruction}"
                    else:
                        messages[-1]["content"] = f"[User Instruction]:\n{instruction}"
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


def _trim_conversation(messages: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
    """Prune conversation turns to fit budget while preserving turn-pair integrity.

    Architectural Rationale:
    1. Turn-Pair & Tool Call Atomicity: In Qwen chat template, an assistant tool call
       (`<tool_call>`) must be immediately followed by its `<tool_response>` (`role: tool`).
       If history truncation chops off an assistant tool call or its corresponding tool result,
       the template encounters an invalid conversational state.
    2. Historical Purity: Any legacy grounding wrappers in historical user turns are sanitized
       to extract the canonical user instruction, ensuring historical turn token invariance.
    3. Oldest-First Sliding Window: Retains the most recent exchanges while dropping the oldest
       complete turns from the beginning of history when budget is exceeded.

    Args:
        messages: List of historical conversation messages.
        budget_tokens: Maximum token budget allocated for conversation history.

    Returns:
        List of trimmed, structurally valid conversation messages.
    """
    if not messages:
        return []

    # 1. Sanitize messages: ensure clean structure and strip any accidental historical grounding wrappers
    clean_msgs: list[dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, str) and m.get("role") == "user":
            if "[User Instruction]:\n" in content:
                m["content"] = content.split("[User Instruction]:\n", 1)[1].strip()
            elif "[User Instruction]:" in content:
                m["content"] = content.split("[User Instruction]:", 1)[1].strip()
        clean_msgs.append(m)

    # 2. Walk backwards to collect turns within budget
    result: list[dict[str, Any]] = []
    total = 0
    for msg in reversed(clean_msgs):
        content_str = str(msg.get("content") or "")
        tokens = _estimate_tokens(content_str)
        if total + tokens > budget_tokens:
            break
        result.insert(0, msg)
        total += tokens

    # 3. Ensure valid turn boundary:
    # If the first retained message is an orphan 'tool' or 'assistant' message, drop it and any immediate siblings
    # until the history starts cleanly with a 'user' turn.
    while result and result[0].get("role") in ("tool", "assistant"):
        result.pop(0)

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
