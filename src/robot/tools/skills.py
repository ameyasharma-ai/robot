"""Tool Skill Manager — Provides structured capability guidance, few-shots, and usage rules to the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot.telemetry import get_logger

logger = get_logger("tools.skills")


@dataclass
class ToolSkill:
    """Detailed skill guide and capability specification for a tool."""

    tool_name: str
    summary: str
    capabilities: list[str]
    parameters_guide: dict[str, str]
    when_to_use: list[str]
    when_not_to_use: list[str]
    examples: list[dict[str, Any]] = field(default_factory=list)

    def format_guide(self) -> str:
        """Format this skill into a concise prompt block for the LLM."""
        lines = [f"### Skill: {self.tool_name}", f"**Summary**: {self.summary}"]
        if self.capabilities:
            lines.append("**Capabilities**:")
            for cap in self.capabilities:
                lines.append(f"- {cap}")
        if self.parameters_guide:
            lines.append("**Parameters**:")
            for p, desc in self.parameters_guide.items():
                lines.append(f"- `{p}`: {desc}")
        if self.when_to_use:
            lines.append("**When to use**:")
            for w in self.when_to_use:
                lines.append(f"- {w}")
        if self.when_not_to_use:
            lines.append("**When NOT to use**:")
            for w in self.when_not_to_use:
                lines.append(f"- {w}")
        if self.examples:
            lines.append("**Examples**:")
            for ex in self.examples:
                lines.append(f"- User: \"{ex.get('user', '')}\" -> Call: `{self.tool_name}({ex.get('args', {})})`")
        return "\n".join(lines)


class ToolSkillManager:
    """Manages skill definitions for tools and renders skill context for the LLM."""

    def __init__(self) -> None:
        self._skills: dict[str, ToolSkill] = {}
        self._register_default_skills()

    def register(self, skill: ToolSkill) -> None:
        self._skills[skill.tool_name] = skill
        logger.debug("tool_skill_registered", tool=skill.tool_name)

    def get_skill(self, tool_name: str) -> ToolSkill | None:
        return self._skills.get(tool_name)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def format_skills_prompt(self, tool_names: list[str] | None = None) -> str:
        """Render a formatted skills reference block for system prompt injection."""
        targets = tool_names if tool_names is not None else list(self._skills.keys())
        blocks = []
        for name in targets:
            skill = self._skills.get(name)
            if skill:
                blocks.append(skill.format_guide())
        if not blocks:
            return ""
        return "## Available Tool Skills & Guidelines\n\n" + "\n\n".join(blocks)

    def _register_default_skills(self) -> None:
        # 1. system.time
        self.register(
            ToolSkill(
                tool_name="system.time",
                summary="Retrieve current real-world date, time, weekday, and timezone.",
                capabilities=[
                    "Check local time defaulting to Indian Standard Time (IST / Asia/Kolkata).",
                    "Check current time in any country, city, or timezone (e.g. Tokyo, London, New York).",
                ],
                parameters_guide={
                    "timezone_str": "Country name, city, or IANA timezone (e.g. 'India', 'Tokyo', 'London', 'America/New_York'). Leave empty for default Indian time."
                },
                when_to_use=[
                    "User asks for the time, date, today's day, or current year.",
                    "User asks 'what time is it in [city/country]' (e.g. London, Japan, New York).",
                ],
                when_not_to_use=[
                    "Historical timestamps in the past.",
                    "General conversational queries not about the time or date.",
                ],
                examples=[
                    {"user": "What time is it?", "args": {}},
                    {"user": "What's the time in London?", "args": {"timezone_str": "London"}},
                    {"user": "Tell me the date in Tokyo.", "args": {"timezone_str": "Tokyo"}},
                ],
            )
        )

        # 2. web.search
        self.register(
            ToolSkill(
                tool_name="web.search",
                summary="Ultra-fast web search for real-world facts, current events, product releases, and news.",
                capabilities=[
                    "Fetch live information from DuckDuckGo and Bing RSS.",
                    "Answer questions about recent news, tech releases (e.g. latest iPhone), or world events.",
                    "Supports deep search queries when the user requests thorough or detailed exploration.",
                ],
                parameters_guide={
                    "query": "Precise search keywords without filler conversational words.",
                    "max_results": "Number of snippets to retrieve (default: 3; deep search: 6-8).",
                    "deep": "Set to true if user specifically asks for deep, extensive, or thorough search.",
                },
                when_to_use=[
                    "Queries about real-world current events, sports scores, politics, or breaking news.",
                    "Queries about newly released products, specs, prices, or release dates.",
                    "Any question where up-to-date external internet knowledge is required.",
                ],
                when_not_to_use=[
                    "Personal questions about the user's memories or private preferences.",
                    "Current time and date (use system.time instead).",
                ],
                examples=[
                    {"user": "What is the newest iPhone model available?", "args": {"query": "latest iPhone model released"}},
                    {"user": "Search deeply about the Artemis moon missions.", "args": {"query": "Artemis moon mission updates", "deep": True}},
                ],
            )
        )

        # 3. memory.write
        self.register(
            ToolSkill(
                tool_name="memory.write",
                summary="Persistently store durable facts, personal preferences, and details about the user.",
                capabilities=[
                    "Store facts into persistent semantic memory (Mem0 + SQLite).",
                    "Preserves user preferences across sessions (e.g. crush name, favorite food, location).",
                ],
                parameters_guide={
                    "fact": "A clear, declarative statement of the fact (e.g. 'The user favorite food is ramen').",
                },
                when_to_use=[
                    "User explicitly says 'remember that...', 'store in memory that...', or 'keep in mind that...'.",
                    "User shares key personal information like their name, partner, crush, home, or hobbies.",
                ],
                when_not_to_use=[
                    "Temporary conversational chit-chat.",
                    "External web facts or news.",
                ],
                examples=[
                    {"user": "Store it in your memory that my crush is Anubha Anand", "args": {"fact": "The user crush is Anubha Anand."}},
                    {"user": "Remember that I love dark chocolate", "args": {"fact": "The user loves dark chocolate."}},
                ],
            )
        )

        # 4. memory.search
        self.register(
            ToolSkill(
                tool_name="memory.search",
                summary="Search semantic memory to recall stored user facts, preferences, and personal details.",
                capabilities=[
                    "Semantic vector search and FTS lookup of stored user memories.",
                    "Retrieves answers to personal questions about the user.",
                ],
                parameters_guide={
                    "query": "The topic, entity, or preference to recall from memory.",
                    "limit": "Max number of memory records to return (default: 5).",
                },
                when_to_use=[
                    "User asks 'who is [person]?', 'what is my favorite [item]?', 'where do I live?'.",
                    "User asks 'do you remember what I told you about...?'",
                ],
                when_not_to_use=[
                    "World history, famous celebrities, news, or general web inquiries.",
                ],
                examples=[
                    {"user": "Who's Anubha?", "args": {"query": "Anubha"}},
                    {"user": "What is my favorite food?", "args": {"query": "favorite food"}},
                ],
            )
        )

        # 5. music.play
        self.register(
            ToolSkill(
                tool_name="music.play",
                summary="Play music on Spotify, SoundCloud, or from local library files.",
                capabilities=[
                    "Launch song or playlist playback on Spotify web/app.",
                    "Launch SoundCloud music playback.",
                    "Play local audio tracks (.mp3, .wav, .flac) asynchronously.",
                ],
                parameters_guide={
                    "query": "Song title, artist, playlist, or genre.",
                    "source": "Playback source: 'spotify', 'soundcloud', 'local', or 'auto' (default: 'spotify').",
                },
                when_to_use=[
                    "User says 'play music', 'play [song/artist] on Spotify', 'play something on SoundCloud'.",
                    "User asks to play local songs or chill music.",
                ],
                when_not_to_use=[
                    "Synthesizing robot spoken speech (TTS handles speech).",
                ],
                examples=[
                    {"user": "Play Bohemian Rhapsody on Spotify", "args": {"query": "Bohemian Rhapsody", "source": "spotify"}},
                    {"user": "Play lofi beats on SoundCloud", "args": {"query": "lofi hip hop beats", "source": "soundcloud"}},
                    {"user": "Play some relaxing music", "args": {"query": "relaxing instrumental", "source": "spotify"}},
                ],
            )
        )

        # 6. music.control
        self.register(
            ToolSkill(
                tool_name="music.control",
                summary="Control playback state and volume of the active music player.",
                capabilities=[
                    "Pause, resume, or stop current music playback.",
                    "Adjust music volume.",
                ],
                parameters_guide={
                    "action": "One of: 'pause', 'resume', 'stop', 'next', 'volume'.",
                    "volume": "Optional float between 0.0 and 1.0 (for action='volume').",
                },
                when_to_use=[
                    "User says 'pause the music', 'stop playing', 'resume song', 'turn up the music'.",
                ],
                when_not_to_use=[
                    "Stopping the robot from speaking (voice barge-in handles speech interruption).",
                ],
                examples=[
                    {"user": "Pause the music", "args": {"action": "pause"}},
                    {"user": "Stop the music", "args": {"action": "stop"}},
                ],
            )
        )

        # 7. perception.what_do_you_see
        self.register(
            ToolSkill(
                tool_name="perception.what_do_you_see",
                summary="Inspect the physical camera view to detect people, faces, and everyday objects.",
                capabilities=[
                    "Real-time face detection (OpenCV YuNet) and object detection (YOLOv8n).",
                    "Describes what the robot is currently looking at.",
                ],
                parameters_guide={},
                when_to_use=[
                    "User asks 'what do you see?', 'what am I holding?', 'look at this', 'who is in the room?'.",
                ],
                when_not_to_use=[
                    "When the user is asking about visual memory or past events.",
                ],
                examples=[
                    {"user": "What do you see right now?", "args": {}},
                    {"user": "Look at what I am holding", "args": {}},
                ],
            )
        )

        # 8. character.execute_action
        self.register(
            ToolSkill(
                tool_name="character.execute_action",
                summary="Trigger animated physical humanoid movements and avatar gestures.",
                capabilities=[
                    "Perform expressive animations: wave, nod, shake_head, tilt_head, shrug, bow, clap, think, cheer, hands_on_hips, heart_gesture, kick, walk.",
                ],
                parameters_guide={
                    "action": "Name of the action: 'wave', 'nod', 'shake_head', 'tilt_head', 'shrug', 'bow', 'clap', 'think', 'cheer', 'hands_on_hips', 'heart_gesture', 'kick', 'stop'.",
                },
                when_to_use=[
                    "Greeting the user with a physical gesture.",
                    "Expressing agreement, excitement, curiosity, or reverence physically.",
                ],
                when_not_to_use=[
                    "Internal software logic.",
                ],
                examples=[
                    {"user": "Wave at me", "args": {"action": "wave"}},
                    {"user": "Can you nod?", "args": {"action": "nod"}},
                    {"user": "Take a bow", "args": {"action": "bow"}},
                    {"user": "Give me a heart", "args": {"action": "heart_gesture"}},
                ],
            )
        )

        # 9. knowledge.wikipedia
        self.register(
            ToolSkill(
                tool_name="knowledge.wikipedia",
                summary="Fetch encyclopedic summaries, history, science, and biographical information from Wikipedia.",
                capabilities=[
                    "Instant overview of historical events, scientific theories, world leaders, artists, and geography.",
                ],
                parameters_guide={
                    "query": "The person, topic, historical event, or concept to look up.",
                },
                when_to_use=[
                    "User asks 'Who was Albert Einstein?', 'What is quantum entanglement?', 'Tell me about the Roman Empire'.",
                    "Encyclopedic, historical, or academic background questions.",
                ],
                when_not_to_use=[
                    "Current breaking news from today (use web.search instead).",
                    "User personal memories (use memory.search instead).",
                ],
                examples=[
                    {"user": "Who was Marie Curie?", "args": {"query": "Marie Curie"}},
                    {"user": "What is the theory of relativity?", "args": {"query": "Theory of relativity"}},
                ],
            )
        )

        # 10. system.calculate
        self.register(
            ToolSkill(
                tool_name="system.calculate",
                summary="Safely evaluate mathematical and scientific expressions accurately without hallucinating.",
                capabilities=[
                    "Arithmetic, powers, roots, trigonometry, logarithms, and percentages.",
                ],
                parameters_guide={
                    "expression": "The mathematical formula to calculate (e.g. 'sqrt(256) * 12 + 45').",
                },
                when_to_use=[
                    "User asks to compute or verify complex math, geometry, or unit conversions.",
                ],
                when_not_to_use=[
                    "Conversational questions.",
                ],
                examples=[
                    {"user": "What is the square root of 529 multiplied by 14?", "args": {"expression": "sqrt(529) * 14"}},
                ],
            )
        )

        # 11. system.hardware
        self.register(
            ToolSkill(
                tool_name="system.hardware",
                summary="Retrieve real-time hardware status: CPU usage, RAM, RTX 3050 GPU utilization, VRAM, and temperatures.",
                capabilities=[
                    "Check RTX 3050 VRAM usage and temperature.",
                    "Check system RAM and CPU load.",
                ],
                parameters_guide={},
                when_to_use=[
                    "User asks 'How is your GPU doing?', 'How much VRAM is left?', 'What is your CPU usage?'.",
                ],
                when_not_to_use=[
                    "General robot status (use system.status).",
                ],
                examples=[
                    {"user": "How is your GPU running?", "args": {}},
                ],
            )
        )

