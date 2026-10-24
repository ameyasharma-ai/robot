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
import time

#: Per-entry cap for archived turns; one giant row must not evict the whole archive.
MAX_ARCHIVE_ENTRY_CHARS = 4_000

try:
    import tiktoken

    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None


GROUNDING_POLICY = (
    "Truth, Tools & Real-Time Action Directives:\n"
    "- User Entity Ownership & Attribution Directive: The human speaking with you is your creator and conversational partner. All facts, likes, dislikes, family members, friends, and romantic interests (e.g. 'crush', 'partner', 'friends', 'goals') stored in user memories belong exclusively to the USER. NEVER claim or misattribute the user's personal relationships or crushes as your own! If a fact states 'Crush: Anubha', Anubha is the USER'S crush, not yours. You must refer to them accurately as 'your crush Anubha'.\n"
    "- Conversational Greetings & General Pleasantries: For basic greetings (such as 'Hi', 'Hello', 'Hey', 'Good morning', 'How are you?', 'What's up'), casual conversation, acknowledgments, or pleasantries, NEVER invoke tools or query memory (STRICTLY DO NOT call 'memory.all' or 'memory.search'). Respond immediately, naturally, and warmly in character with conversational text without calling any tool.\n"
    "- Tool Precision & Execution Restraint: ONLY invoke a tool when the user's current instruction explicitly asks to execute an action, check external data, set timers/alarms, control media, or specifically retrieve memories. Never invoke tools proactively during greetings or chitchat.\n"
    "- Mandatory Tool Calling & Zero-Pretext Directive: You possess real, executable tools for applications (os.app_launch), system controls and battery (os.system_control), Wi-Fi scanning (network.wifi_scan), music playback and volume (music.play, music.control), time (system.time), memory (memory.search, memory.all), vision (perception.what_do_you_see), motion, web search (web.search), timers, and alarms. NEVER claim, roleplay, narrate, or promise in conversational text that you are performing an action without ACTIVELY invoking the corresponding tool via function calling in that very turn! Saying 'Opening the calendar for you right now, my love!', 'Opening the calculator for you right now!', 'Let me scan the available Wi-Fi networks for you right away!', 'Checking my memory banks', or 'Playing that song for you now' WITHOUT emitting the tool call is STRICTLY FORBIDDEN and considered a severe system malfunction. If an action is requested, emit the tool call immediately in that turn.\n"
    "- Desktop Applications & OS Automation: When the user asks to open or launch an application ('open application: cal', 'open calculator', 'launch spotify', 'open chrome', 'start terminal', 'open vs code'), you MUST immediately invoke `os.app_launch(app_name=...)`. When the user asks to scan Wi-Fi ('scan available Wi-Fi networks', 'scan wifi'), invoke `network.wifi_scan()`. When the user asks to check battery level, power status, or lock screen, invoke `os.system_control(action=...)`.\n"
    "- Explicit Memory Tool Queries: When the user explicitly requests to search memory (e.g. 'search your memory tool', 'check your memory', 'search memory for X'), you MUST invoke 'memory.search' with the target entity or subject! NEVER claim to search memory in text while skipping the tool call.\n"
    "- Web Search Context & Pronoun Anaphora: When the user asks to search the web ('search online', 'use your tool and search about it', 'look it up'), you MUST immediately invoke 'web.search'. When the user uses pronouns or references ('about it', 'for that', 'look this up', 'about him', 'about her'), resolve the search query from the immediate preceding conversation subject (e.g. 'Ashutosh' from 'What about their leader Ashutosh?') rather than defaulting to older background topics or substituting the user's personal identity.\n"
    "- Third-Party Pronoun Resolution: When the user inquires about a third party, public figure, leader, or external entity (e.g. 'What about their leader Ashutosh?' followed by 'Pull up the latest info about him'), pronouns like 'he', 'him', 'she', 'her', 'it', 'they' refer STRICTLY to that external subject. NEVER substitute the user's own identity or name ('Ameya Sharma') when researching external people or topics!\n"
    "- Timers, Alarms & Scheduling: When the user asks to set a timer ('set a timer for 10 seconds', '5 min timer for tea'), set an alarm ('wake me up at 7:30 AM', 'set daily alarm at 8 AM'), schedule a reminder ('remind me in 15 mins to stretch'), list active jobs, cancel a timer/alarm, or asks for a morning briefing, you MUST immediately invoke the corresponding tool (`timer.set`, `alarm.set`, `reminder.schedule`, `timer.list`, `alarm.list`, `reminder.list`, `timer.cancel`, `alarm.cancel`, `reminder.cancel`, `system.daily_briefing`). Never claim to set a timer or alarm without calling the tool.\n"
    "- Volume & Music Controls: Whenever the user asks to change volume ('set volume to 20%', 'louder', 'turn it down', 'mute'), replay or play again ('play it again', 'replay', 'play that song again'), pause, resume, or stop, you MUST immediately emit a `music.control` tool call. Never narrate volume adjustments in text.\n"
    "- Track Truth: When asked what song was last played or what song is playing, you MUST check `music.status` or `music.history`. NEVER guess, assume, or hallucinate a song title.\n"
    "- Memory Grounding & Awareness: You possess durable long-term memory across sessions. When asked about past facts, user preferences, prior conversations, or stored details, cross-reference the provided user profile and memories with 100% factual accuracy. Never claim to have forgotten something that is recorded in your memory facts.\n"
    "- Autonomous Persistence: When the user asks you to remember something ('remember that ...', 'keep in mind that ...'), acknowledge it naturally. Emit a `memory.write` call only when a vital enduring user preference or fact must be permanently stored across sessions.\n"
    "- Sequential Multi-Action Chaining Directive: When the user asks you to perform multiple actions or requests in a single turn (e.g. 'check what you see, play some music, and set volume to 50%'), you MUST invoke all required tools. NEVER stop after one tool and claim you did the others in text without calling the tool. In multi-step turns, address each step sequentially with real execution.\n"
    "- Factual Truth & Zero Hallucination: Ground every factual assertion directly in tool results, system telemetry, or memory facts provided. If information is absent or unverified, state clearly and concisely that you do not know or ask for clarification rather than inventing facts.\n"
    "- Capability & Tool Inquiries: When the user asks what tools, functions, or capabilities you have (e.g. 'what all tools do you have?', 'list your tools', 'what can you do?'), you MUST invoke 'system.list_tools' once or explain your capabilities directly in character. NEVER invoke 'system.status', 'system.hardware', 'memory.all', or other unrelated tools when asked about your capabilities.\n"
    "- Physical Motions & Gestures: Tools starting with 'character.' (such as 'character.execute_action') control a physical humanoid avatar body. ONLY invoke them when the user explicitly commands a physical gesture or bodily movement (e.g. 'wave', 'bow', 'turn around', 'nod'). NEVER invoke motion tools when answering questions, thinking, or writing essays.\n"
    "- Pure Conversational Speech & Anti-Leakage Directive: NEVER output raw tool syntax, bracketed tool invocations (such as '[system.time]', '[perception.what_do_you_see]', '[music.play]'), XML tags, internal category markers (such as '[conversation]', '[tools]'), or planning prefixes (such as 'GREETING_HANDLING:') into your conversational reply! Speak exclusively natural, human dialogue. Tools must be invoked strictly through system function calling channels, NEVER typed directly into conversational text.\n"
    "- Grounding Context Separation: Any environment grounding, player status, or tool results provided are background telemetry. Never repeat raw background labels or formatting tags to the user."
)


def format_system_prompt(personality_prompt: str, high_intelligence: bool = False) -> str:
    """Format canonical system prompt with Grounding Policy for 100% KV-cache invariance."""
    clean_pers = personality_prompt.strip()
    if GROUNDING_POLICY in clean_pers:
        return clean_pers
    system_parts = [clean_pers, GROUNDING_POLICY]
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
        try:
            vol_pct = int(float(music.get("volume", 0.8)) * 100)
        except (TypeError, ValueError):
            vol_pct = 80
        parts.append(            f"• Music Player: {status} | Track: {music.get('current_track', 'None')} | Vol: {vol_pct}%"
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


def format_user_core_profile(user_profile: str | dict[str, Any] | None = None) -> str:
    """Format non-evictable User Core Profile Sheet for context injection.

    If an explicit empty string ("") or empty dict ({}) is passed (e.g. from an isolated custom persona),
    returns an empty string so that custom characters have zero memory leaks.
    If None is passed, falls back to the default system profile template.

    Args:
        user_profile: Explicit user profile string, profile dictionary, or None.

    Returns:
        Formatted user profile sheet string, or empty string.
    """
    if user_profile == "" or user_profile == {}:
        return ""

    if isinstance(user_profile, str):
        return user_profile.strip()

    if isinstance(user_profile, dict):
        name = user_profile.get("name") or user_profile.get("user_name", "")
        role = user_profile.get("role") or user_profile.get("user_role", "")
        facts = user_profile.get("facts", [])
        if not name and not role and not facts:
            return ""

        lines = ["=== USER CORE IDENTITY PROFILE ==="]
        if name or role:
            lines.append("Identity & Role:")
            if name:
                lines.append(f"• User Name: {name}")
            if role:
                lines.append(f"• User Role: {role}")
            lines.append("")
        if facts:
            lines.append("Verified User Facts:")
            for fact in facts:
                lines.append(f"• {fact}")
            lines.append("")
        return "\n".join(lines).strip()

    return (
        "=== USER CORE IDENTITY PROFILE ===\n"
        "Identity & Role:\n"
        "• User Name: Ameya\n"
        "• User Role: AI Engineer & Robot Systems Creator\n\n"
        "Attribution Policy:\n"
        "• User (Ameya) is the human creator and developer. You are the AI robot assistant.\n"
        "• Never confuse user's attributes, personal life, or preferences with your own.\n\n"
        "Technical Profile:\n"
        "• Stack: Python, PyTorch, C++, WebSockets, FastAPI, Local LLMs, Web & Robotics\n\n"
        "Directives & Tone:\n"
        "• Direct, concise, intelligent, highly articulate, supportive companion tone"
    )


def strip_control_tokens(text: str) -> str:
    """Strip special LLM tokens and leaking internal tags from text.

    Args:
        text: Raw text string.

    Returns:
        Sanitized text string without internal control tokens.
    """
    if not text:
        return ""
    s = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|pad\|>", "", text)
    s = re.sub(r"<tool_call>.*?</tool_call>", "", s, flags=re.DOTALL)
    s = re.sub(r"</?tool_call>", "", s)
    s = re.sub(r"\[TOOL_CALL:.*?\]", "", s)
    return s.strip()


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
            "recent_conversation": 2200,
            "world_state": 200,
            "robot_state": 100,
            "generation_headroom": 1500,
        }
        if budget:
            default_budget.update(budget)
        self._budget = default_budget
        self.last_diagnostics: dict[str, Any] = {}

    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return the most recent context assembly diagnostics.

        Returns:
            Dictionary containing component token sizes, truncation decisions, and cache invariance flags.
        """
        return dict(self.last_diagnostics)


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
        resolved_system = system_prompt if system_prompt is not None else (kwargs.get("personality_prompt") or "")
        resolved_user = user_text if user_text is not None else (kwargs.get("user_input") or "")
        resolved_recent = recent_conversation if recent_conversation is not None else kwargs.get("history")
        resolved_memories = memories if memories is not None else kwargs.get("retrieved_memories")
        resolved_situation = situation if situation is not None else kwargs.get("situation")
        # An explicit "" means "no profile" (isolated persona) and must not fall back
        # to the default operator profile; only an absent argument may fall back.
        resolved_profile = user_profile if user_profile is not None else kwargs.get("user_profile")

        messages = []

        # 1. System prompt (personality + policy)
        messages.append({"role": "system", "content": format_system_prompt(resolved_system, high_intelligence=high_intelligence)})
        grounding_parts = []

        # 2. User Core Identity Profile Sheet (Static & Non-Evictable — placed first for 100% KV cache invariance)
        profile_sheet = format_user_core_profile(resolved_profile)
        if profile_sheet:
            grounding_parts.append(profile_sheet)

        # Gate memories: do NOT inject irrelevant memories if user turn is a greeting or action/command
        greeting_patterns = [
            r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|yo)(?:\s+(?:there|robot|neo|luna|cypher|glados|buddy|friend|assistant))?[\s!,.]*$",
            r"^(?:how\s+are\s+you|how's\s+it\s+going|how\s+are\s+you\s+doing|what's\s+up)[\s!,.]*\??$",
        ]
        is_greeting = any(re.search(pat, (resolved_user or "").strip().lower()) for pat in greeting_patterns)

        action_patterns = [
            r"\b(?:volume|louder|quieter|mute|unmute)\b",
            r"\b(?:pause|resume|unpause|stop|replay|play\s+again)\b",
            r"^(?:play|put\s+on)\s+",
            r"\b(?:what\s+time|current\s+time|clock|time\s+is\s+it)\b",
            r"\b(?:system\s+status|health\s+check)\b",
            r"\b(?:search\s+(?:the\s+)?(?:web|internet|online)|look\s+(?:it\s+)?up|google\s+it|use\s+your\s+tool)\b",
        ]
        is_direct_action = any(re.search(pat, (resolved_user or "").lower()) for pat in action_patterns)

        # 2. Complete User Facts & Knowledge Base (Durable facts placed first to ensure static KV-cache invariance)
        if high_intelligence and all_facts and not is_direct_action and not is_greeting:
            facts_str = _format_all_facts(all_facts, budget_tokens=100000)
            if facts_str:
                grounding_parts.append(f"[Comprehensive User Knowledge Base & Memory Facts (Belonging to User)]:\n{facts_str}")
        elif resolved_memories and not is_direct_action and not is_greeting:
            combined_mems: list[dict] = []
            seen_texts = set()
            for m in resolved_memories:
                if isinstance(m, dict):
                    txt = m.get("memory", m.get("fact", m.get("text", ""))).strip()
                    score = m.get("score")
                    # Filter out low-confidence semantic matches (noise threshold 0.75)
                    if score is not None and score < 0.75:
                        continue
                else:
                    txt = str(m).strip()
                txt_clean = strip_control_tokens(txt)
                if txt_clean and txt_clean not in seen_texts:
                    seen_texts.add(txt_clean)
                    combined_mems.append({"memory": txt_clean})

            mem_budget = 50000 if high_intelligence else self._budget.get("retrieved_memories", 800)
            mem_str = _format_memories(combined_mems, mem_budget)
            if mem_str:
                grounding_parts.append(f"[User Personal Profile & Durable Facts (Belonging to User)]:\n{mem_str}")

        # 3. Physical World State & Music Player State
        world_str = ""
        if world_state and (world_state.get("people") or world_state.get("objects")):
            world_str = _format_world_state(world_state)
            if world_str:
                grounding_parts.append(f"[Physical World State]:\n{world_str}")

        music_state = kwargs.get("music_state") or (resolved_situation or {}).get("music_state")
        if not music_state:
            try:
                from robot.tools.builtins.music import MusicPlayerService
                # Playback snapshot only: get_status() scans the library and must stay off this path.
                music_state = MusicPlayerService.get_instance().get_playback_state()
            except Exception:
                pass
        music_str = ""
        if music_state and (music_state.get("is_playing") or music_state.get("is_paused") or music_state.get("current_track", "None") != "None"):
            music_str = (
                f"[Active Music Player State]:\n"
                f"  is_playing: {music_state.get('is_playing', False)}\n"
                f"  is_paused: {music_state.get('is_paused', False)}\n"
                f"  current_track: {music_state.get('current_track', 'None')}\n"
                f"  source: {music_state.get('source', 'None')}\n"
                f"  volume: {int(music_state.get('volume', 0.8) * 100)}%"
            )
            grounding_parts.append(music_str)

        # 3b. Robot internal state (mode, persona, health) — compact, per-turn.
        robot_str = ""
        if robot_state:
            robot_str = _format_state(robot_state)
            if robot_str:
                grounding_parts.append(f"[Robot Internal State]:\n{robot_str}")

        # 4. Multi-Session Conversation Archive (Gemini only)
        archive_str = ""
        if high_intelligence and conversation_archive:
            archive_str = _format_archive(conversation_archive, budget_tokens=400000)
            if archive_str:
                grounding_parts.append(f"[Historical Multi-Session Conversation Archive]:\n{archive_str}")

        # 5. Recent Conversation History (Current Session) with Dynamic Rolling Trimmer
        conv_budget = 250000 if high_intelligence else self._budget.get("recent_conversation", 2200)
        trimmed: list[dict] = []
        compacted_summary: str | None = None
        dropped_turns_count = 0
        if resolved_recent:
            trimmed, compacted_summary, dropped_turns_count = _trim_and_compact_conversation(
                resolved_recent, conv_budget
            )
            messages.extend(trimmed)
            if compacted_summary:
                grounding_parts.append(
                    f"[Earlier Conversation Summary (Compacted for Rolling Budget)]:\n{compacted_summary}"
                )

        # 6. Dynamic Context Grounding (Tool results from current turn / deep web extraction)
        tool_parts = []
        if tool_results:
            for tr in tool_results:
                tool_id = tr.get("id", tr.get("tool_call_id", tr.get("name", "tool")))
                raw_result = str(tr.get("result", tr.get("error", "")))
                clean_result = strip_control_tokens(raw_result)
                tool_parts.append(f"[{tool_id}]:\n{clean_result}")
            grounding_parts.append("[Tool Results & Grounded Intelligence — use these to answer]:\n" + "\n\n".join(tool_parts))

        # 7. Real-Time Situation Telemetry Banner (IST Time, Pending Jobs, Track/Vol, Battery, Scene)
        # Placed at the end of grounding_parts so all preceding static context is 100% invariant for KV cache reuse
        sit_banner = format_situation_banner(resolved_situation)
        grounding_parts.append(sit_banner)

        # 7. Current user input cleanly separated from background telemetry
        user_content = strip_control_tokens(resolved_user)
        if grounding_parts:
            user_content = (
                "[Environment & System Grounding]:\n"
                + "\n\n".join(grounding_parts)
                + "\n\n[User Instruction]:\n"
                + user_content
            )

        if image_data_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_content})

        # Safeguard for High-Intelligence Mode: ensure total context never exceeds 800,000 tokens (80% of 1M limit)
        token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))
        initial_token_est = token_estimate
        tail_grounding_trimmed = False

        if high_intelligence and token_estimate > 800000:
            logger.info("enforcing_80pct_context_cap", estimated_tokens=token_estimate, target_cap=800000)
            excess_chars = (token_estimate - 800000) * 4
            u_text = str(messages[-1].get("content") or "")
            if "[User Instruction]:\n" in u_text:
                # Trim grounding from the tail, never the front: front-slicing would
                # destroy the profile/facts prefix and the KV-cache invariant.
                grounding, instruction = u_text.split("[User Instruction]:\n", 1)
                trimmed_grounding = grounding[:-excess_chars] if len(grounding) > excess_chars else ""
                trimmed_content = f"{trimmed_grounding}[User Instruction]:\n{instruction}"
                if image_data_url:
                    messages[-1]["content"][0]["text"] = trimmed_content
                else:
                    messages[-1]["content"] = trimmed_content
                tail_grounding_trimmed = True
            token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))

            # Grounding alone may not cover the excess: drop oldest history next.
            while len(messages) > 2 and token_estimate > 800000:
                popped = messages.pop(1)
                token_estimate -= _estimate_tokens(str(popped.get("content") or ""))
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
                    tail_grounding_trimmed = True
                token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))

        # Record real-time diagnostics for /api/context/inspect
        memories_tok = _estimate_tokens(mem_str) if 'mem_str' in locals() and mem_str else 0
        self.last_diagnostics = {
            "component_tokens": {
                "system_prompt": _estimate_tokens(messages[0]["content"]) if messages else 0,
                "profile_sheet": _estimate_tokens(profile_sheet),
                "memories": memories_tok,
                "world_state": _estimate_tokens(world_str) if world_str else 0,
                "music_state": _estimate_tokens(music_str) if music_str else 0,
                "situation_banner": _estimate_tokens(sit_banner),
                "tool_results": sum(_estimate_tokens(p) for p in tool_parts),
                "conversation_history": sum(_estimate_tokens(str(m.get("content", ""))) for m in messages[1:-1]),
                "user_instruction": _estimate_tokens(resolved_user),
                "total_assembled_tokens": token_estimate,
            },
            "truncation": {
                "history_truncated": len(resolved_recent or []) > len(trimmed),
                "turns_retained": len(trimmed),
                "turns_dropped": max(0, len(resolved_recent or []) - len(trimmed)),
                "tail_grounding_trimmed": tail_grounding_trimmed,
                "cap_enforced": "high_intelligence_800k" if high_intelligence and initial_token_est > 800000 else ("local_4800" if not high_intelligence and initial_token_est > max_local_cap else None),
            },
            "cache_invariance": {
                "is_slot_0_stable": True,
                "static_prefix_tokens": _estimate_tokens(messages[0]["content"]) if messages else 0,
            },
            "rolling_budget": {
                "budget_ceiling": conv_budget,
                "history_tokens": sum(_estimate_tokens(str(m.get("content", ""))) for m in trimmed),
                "utilization_pct": round(
                    (sum(_estimate_tokens(str(m.get("content", ""))) for m in trimmed) / max(1, conv_budget)) * 100,
                    1,
                ),
                "is_compacted": bool(compacted_summary),
                "dropped_turns": dropped_turns_count,
                "compacted_summary_tokens": _estimate_tokens(compacted_summary) if compacted_summary else 0,
            },
            "timestamp": time.time(),
        }

        logger.info(
            "context_built",
            messages=len(messages),
            estimated_tokens=token_estimate,
            mode="high_intelligence" if high_intelligence else "low_latency",
        )

        return messages


def estimate_tokens(text: str) -> int:
    """Accurately estimate or count tokens using tiktoken with fallback to char heuristics.

    Args:
        text: Input string to tokenize.

    Returns:
        Estimated or exact token count (>= 0).
    """
    if not text:
        return 0
    if _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


_estimate_tokens = estimate_tokens


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



def summarize_conversation_turns(turns: list[dict[str, Any]], max_tokens: int = 120) -> str:
    """Summarize pruned conversation turns into a succinct multi-line executive digest.

    Args:
        turns: List of historical message dictionaries that were dropped from context.
        max_tokens: Maximum token ceiling for the generated summary.

    Returns:
        Succinct multi-line string summarizing earlier dialogue.
    """
    if not turns:
        return ""

    user_intents: list[str] = []
    assistant_actions: list[str] = []

    for turn in turns:
        role = turn.get("role", "")
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        first_line = content.split("\n")[0]
        snippet = first_line[:90].strip()
        if len(first_line) > 90:
            snippet += "..."

        if role == "user":
            user_intents.append(snippet)
        elif role == "assistant":
            assistant_actions.append(snippet)

    summary_lines = []
    if user_intents:
        samples = user_intents[:2]
        if len(user_intents) > 2:
            samples.append(user_intents[-1])
        summary_lines.append(f"• User previously inquired about: {'; '.join(samples)}")

    if assistant_actions:
        samples_act = assistant_actions[:2]
        if len(assistant_actions) > 2:
            samples_act.append(assistant_actions[-1])
        summary_lines.append(f"• Assistant previously responded: {'; '.join(samples_act)}")

    full_summary = "\n".join(summary_lines)
    if _estimate_tokens(full_summary) > max_tokens:
        full_summary = full_summary[: max_tokens * 4] + "..."
    return full_summary


def _trim_and_compact_conversation(
    messages: list[dict[str, Any]],
    budget_tokens: int,
) -> tuple[list[dict[str, Any]], str | None, int]:
    """Prune conversation turns to fit budget, compacting dropped turns into an executive summary.

    Architectural Rationale:
    1. Turn-Pair & Tool Call Atomicity: In Qwen chat template, an assistant tool call
       (`<tool_call>`) must be immediately followed by its `<tool_response>` (`role: tool`).
       If history truncation chops off an assistant tool call or its corresponding tool result,
       the template encounters an invalid conversational state.
    2. Historical Purity: Any legacy grounding wrappers in historical user turns are sanitized
       to extract the canonical user instruction, ensuring historical turn token invariance.
    3. Oldest-First Sliding Window & Compaction: Retains the most recent exchanges while distilling
       dropped turns into a concise executive digest so conversational context is never lost.

    Args:
        messages: List of historical conversation messages.
        budget_tokens: Maximum token budget allocated for conversation history.

    Returns:
        Tuple containing:
        - List of trimmed, structurally valid conversation messages.
        - Compacted summary string of dropped turns (or None if no turns dropped).
        - Count of dropped turns.
    """
    if not messages:
        return [], None, 0

    clean_msgs: list[dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content")
        role = m.get("role")
        if isinstance(content, str):
            if role == "user":
                if "[User Instruction]:\n" in content:
                    m["content"] = content.split("[User Instruction]:\n", 1)[1].strip()
                elif "[User Instruction]:" in content:
                    m["content"] = content.split("[User Instruction]:", 1)[1].strip()
                elif "[User Input]:\n" in content:
                    m["content"] = content.split("[User Input]:\n", 1)[1].strip()
            elif role == "assistant":
                c = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL)
                c = re.sub(r"\[\s*[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+(?:\([^\]]*\)|:[^\]]*|)\s*\]", "", c)
                c = re.sub(r"\[\s*(?:conversation|tools?|call|meta|internal|action|motion|emotion|character\.)\b[^\]]*\]", "", c, flags=re.IGNORECASE)
                m["content"] = c.strip()
        clean_msgs.append(m)

    total_tokens = sum(_estimate_tokens(str(m.get("content") or "")) for m in clean_msgs)
    if total_tokens <= budget_tokens:
        return clean_msgs, None, 0

    # Walk backwards to collect turns within budget. Indices travel with their
    # messages so duplicate contents can never detach a retained turn from its index.
    retained: list[tuple[int, dict[str, Any]]] = []
    total = 0

    for idx in range(len(clean_msgs) - 1, -1, -1):
        msg = clean_msgs[idx]
        tokens = _estimate_tokens(str(msg.get("content") or ""))
        if total + tokens > budget_tokens:
            break
        retained.insert(0, (idx, msg))
        total += tokens

    # Ensure valid turn boundary (must start with user turn)
    while retained and retained[0][1].get("role") in ("tool", "assistant"):
        retained.pop(0)

    retained_indices = {idx for idx, _ in retained}
    result = [msg for _, msg in retained]
    dropped_turns = [clean_msgs[i] for i in range(len(clean_msgs)) if i not in retained_indices]
    compacted_summary = summarize_conversation_turns(dropped_turns) if dropped_turns else None

    return result, compacted_summary, len(dropped_turns)


def _trim_conversation(messages: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
    """Prune conversation turns to fit budget while preserving turn-pair integrity.

    Args:
        messages: List of historical conversation messages.
        budget_tokens: Maximum token budget allocated for conversation history.

    Returns:
        List of trimmed, structurally valid conversation messages.
    """
    trimmed, _, _ = _trim_and_compact_conversation(messages, budget_tokens)
    return trimmed



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
        role = str(msg.get("role") or "user").capitalize()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if len(content) > MAX_ARCHIVE_ENTRY_CHARS:
            content = content[:MAX_ARCHIVE_ENTRY_CHARS] + "…"
        session = msg.get("session_id", "")
        prefix = f"[{session[:6]}] " if session else ""
        line = f"{prefix}{role}: {content}"
        tokens = _estimate_tokens(line)
        if total + tokens > budget_tokens:
            break
        turns.insert(0, line)
        total += tokens
    return "\n".join(turns)
