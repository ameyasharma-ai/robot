"""Tool framework — registry, router, and permission gates.

Qwen proposes actions via tool_call; Robot Core validates and executes.
A model-generated tool call is a REQUEST, not authorization.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine

from robot.telemetry import get_logger
from robot.tools.coordinator import (
    CoordinatorState,
    HoldingStatus,
    ToolCategory,
    ToolContextState,
    ToolConversationCoordinator,
    ToolExecutionStatus,
)
from robot.tools.planner import (
    ExecutionPlan,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    SequentialPlannerExecutor,
)
from robot.tools.hardening import (
    ParameterConstraint,
    ToolOutcomeStatus,
    ToolValidationError,
    classify_tool_result,
    coerce_value,
    harden_tool_arguments,
    harden_tool_inputs,
    redact_secrets,
    redact_tool_arguments,
)

logger = get_logger("tools")

ToolHandler = Callable[..., Coroutine[Any, Any, Any]]

# Default wall-clock budget granted to a single tool invocation.  Individual
# ToolDefinitions may set a smaller or larger budget; only genuinely
# interactive/browser tools receive a floor above this default.
DEFAULT_TOOL_TIMEOUT_S: float = 30.0
#: Hard upper bound on any single tool budget. A per-tool value above this is
#: clamped down: during a live voice turn an unbounded tool is indistinguishable
#: from a hung process, and the whole pipeline waits on it.
MAX_TOOL_TIMEOUT_S: float = 45.0
TOOL_TIMEOUT_FLOORS: dict[str, float] = {
    # Headless navigation needs time for browser launch, JS hydration, and
    # screenshot capture.  The legacy 3 s web-family cap made this impossible.
    "web.browser_agent": 30.0,
}

# Search result fan-out bound.  Handlers (and their caches) must never be
# allowed to request or return an unbounded number of results.
MIN_SEARCH_RESULTS: int = 1
MAX_SEARCH_RESULTS: int = 10


class BoundedTTLCache:
    """Small LRU cache with a hard size ceiling and per-entry TTL.

    Used for framework-level bookkeeping (duplicate side-effect suppression,
    search result reuse) so long-running robot processes cannot accumulate an
    unbounded number of keys.
    """

    def __init__(self, maxsize: int = 128, ttl_s: float = 60.0) -> None:
        """Initialize the cache.

        Args:
            maxsize: Maximum number of retained entries.
            ttl_s: Entry time-to-live in seconds.
        """
        self.maxsize: int = max(1, int(maxsize))
        self.ttl_s: float = max(0.0, float(ttl_s))
        self._data: OrderedDict[Any, tuple[float, Any]] = OrderedDict()

    def get(self, key: Any) -> Any | None:
        """Return a live cached value, evicting it when expired.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None when absent/expired.
        """
        entry = self._data.get(key)
        if entry is None:
            return None
        created_at, value = entry
        if time.monotonic() - created_at > self.ttl_s:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: Any, value: Any) -> None:
        """Insert or refresh a value, evicting the least-recently-used entry.

        Args:
            key: Cache key.
            value: Value to retain.
        """
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def pop(self, key: Any, default: Any = None) -> Any:
        """Remove and return a key, mirroring ``dict.pop``."""
        entry = self._data.pop(key, None)
        return default if entry is None else entry[1]

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def effective_tool_timeout(tool: "ToolDefinition") -> float:
    """Return the per-tool wall-clock budget, honoring floors and the hard cap.

    Args:
        tool: Tool definition under execution.

    Returns:
        Timeout in seconds.  A missing/non-positive per-tool value falls back
        to :data:`DEFAULT_TOOL_TIMEOUT_S`; named floors (for example the
        browser agent) raise the budget to their minimum, and every budget is
        then clamped to :data:`MAX_TOOL_TIMEOUT_S`.
    """
    budget = float(tool.timeout_s) if tool.timeout_s and tool.timeout_s > 0 else DEFAULT_TOOL_TIMEOUT_S
    floor = TOOL_TIMEOUT_FLOORS.get(tool.name, 0.0)
    return min(max(budget, floor), MAX_TOOL_TIMEOUT_S)


class PermissionLevel(IntEnum):
    """Whitepaper permission levels L0-L6."""

    READ_ONLY = 0       # status, memory/history search
    HARMLESS_LOCAL = 1   # face, RGB, music
    ENVIRONMENT = 2      # smart-home controls, local apps
    EXTERNAL = 3         # send/post/submit
    SYSTEM = 4           # filesystem, terminal, packages
    DEVELOPMENT = 5      # modify Robot Core/app code
    CRITICAL = 6         # kernel, boot, security


@dataclass
class ToolDefinition:
    """Complete tool registration."""

    name: str                          # e.g., "memory.search"
    description: str
    parameters: dict[str, Any]         # JSON Schema
    handler: ToolHandler
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    risk_level: str = "low"
    provider: str = ""
    family: str = ""                   # e.g., "memory", "music", "system"
    is_idempotent: bool = True
    max_retries: int = 0


class ToolRegistry:
    """Central tool registration and lookup."""

    ALIASES: dict[str, str] = {
        "web_search": "web.search",
        "web-search": "web.search",
        "search": "web.search",
        "google": "web.search",
        "web_deep_search": "web.deep_search",
        "motion": "character.execute_action",
        "gesture": "character.execute_action",
        "action": "character.execute_action",
        "execute_action": "character.execute_action",
        "stop_motion": "character.stop",
        "music_play": "music.play",
        "play_music": "music.play",
        "music_control": "music.control",
        "control_music": "music.control",
        "volume": "music.control",
        "music_status": "music.status",
        "music_history": "music.history",
        "music_equalizer": "music.equalizer",
        "equalizer": "music.equalizer",
        "calculator": "system.calculate",
        "calculate": "system.calculate",
        "hardware": "system.hardware",
        "camera_snapshot": "perception.snapshot",
        "snapshot": "perception.snapshot",
        "world_state": "perception.world_state",
        "what_do_you_see": "perception.what_do_you_see",
        "vision": "perception.what_do_you_see",
        "see": "perception.what_do_you_see",
        "analyze_camera_frame": "perception.analyze_camera_frame",
        "analyze_frame": "perception.analyze_camera_frame",
        "vlm_inspect": "perception.analyze_camera_frame",
        "read_text": "perception.analyze_camera_frame",
        "inspect_tabletop": "perception.inspect_tabletop",
        "generate_image": "image.generate",
        "create_image": "image.generate",
        "wikipedia": "knowledge.wikipedia",
        "wiki": "knowledge.wikipedia",
        "set_timer": "timer.set",
        "timer": "timer.set",
        "list_timers": "timer.list",
        "cancel_timer": "timer.cancel",
        "set_alarm": "alarm.set",
        "alarm": "alarm.set",
        "list_alarms": "alarm.list",
        "cancel_alarm": "alarm.cancel",
        "set_reminder": "reminder.schedule",
        "reminder": "reminder.schedule",
        "list_reminders": "reminder.list",
        "cancel_reminder": "reminder.cancel",
        "daily_briefing": "system.daily_briefing",
        "briefing": "system.daily_briefing",
        "app_launch": "os.app_launch",
        "launch_app": "os.app_launch",
        "open_app": "os.app_launch",
        "system_control": "os.system_control",
        "lock_screen": "os.system_control",
        "clipboard_read": "os.clipboard_read",
        "read_clipboard": "os.clipboard_read",
        "clipboard_write": "os.clipboard_write",
        "write_clipboard": "os.clipboard_write",
        "notify_desktop": "os.notify_desktop",
        "desktop_notify": "os.notify_desktop",
        "notify": "os.notify_desktop",
        "weather": "weather.forecast",
        "get_weather": "weather.forecast",
        "forecast": "weather.forecast",
        "news": "news.headlines",
        "headlines": "news.headlines",
        "top_news": "news.headlines",
        "finance": "finance.quote",
        "stock": "finance.quote",
        "stock_price": "finance.quote",
        "crypto": "finance.quote",
        "bitcoin_price": "finance.quote",
        "homeassistant": "homeassistant.control",
        "smart_home": "homeassistant.control",
        "smart_light": "homeassistant.control",
        "lights": "homeassistant.control",
        "thermostat": "homeassistant.control",
        "switch_personality": "personality.switch",
        "character.switch": "personality.switch",
        "character_switch": "personality.switch",
        "switch_persona": "personality.switch",
        "persona.switch": "personality.switch",
        "execute_terminal": "terminal.execute",
        "terminal": "terminal.execute",
        "shell": "terminal.execute",
        "run_command": "terminal.execute",
        "exec": "terminal.execute",
        "git": "git.workspace_commit",
        "git_commit": "git.workspace_commit",
        "workspace_commit": "git.workspace_commit",
        "run_script": "code.run_script",
        "run_code": "code.run_script",
        "code_runner": "code.run_script",
        "notes": "notes.vault_search",
        "notes_vault": "notes.vault_search",
        "vault_search": "notes.vault_search",
        "translate": "translation.translate",
        "translation": "translation.translate",
        "scrape_page": "web.scrape_page",
        "scrape": "web.scrape_page",
        "capture_screen": "os.capture_screen",
        "screenshot": "os.capture_screen",
        "screen_capture": "os.capture_screen",
        "brightness": "os.display_brightness",
        "display_brightness": "os.display_brightness",
        "wifi": "network.wifi_scan",
        "wifi_scan": "network.wifi_scan",
        "speedtest": "network.speedtest",
        "network_status": "network.speedtest",
        "ping": "network.speedtest",
        "list_tools": "system.list_tools",
        "system_list_tools": "system.list_tools",
        "capabilities": "system.list_tools",
        "tools": "system.list_tools",
        "death_clock": "system.death_clock",
        "system_death_clock": "system.death_clock",
        "health.death_clock": "system.death_clock",
        "life_expectancy": "system.death_clock",
        "longevity": "system.death_clock",
        "music_generate": "music.generate",
        "generate_music": "music.generate",
        "compose_music": "music.generate",
        "synthesize_music": "music.generate",
        "send_notification": "phone.send_notification",
        "send_phone_notification": "phone.send_notification",
        "phone_notification": "phone.send_notification",
        "push_notification": "phone.send_notification",
        "phone_alert": "phone.send_notification",
        "send_sms": "communication.send_sms",
        "draft_sms": "communication.draft_sms",
        "sms": "communication.send_sms",
        "deep_research": "research.deep_research",
        "research": "research.deep_research",
        "browser_agent": "web.browser_agent",
        "web_agent": "web.browser_agent",
    }

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name, family=tool.family)

    def get(self, name: str) -> ToolDefinition | None:
        if name in self._tools:
            return self._tools[name]
        canonical = self.ALIASES.get(name) or self.ALIASES.get(name.lower())
        if canonical and canonical in self._tools:
            return self._tools[canonical]
        dotted = name.replace("_", ".")
        if dotted in self._tools:
            return self._tools[dotted]
        return None

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_family(self, family: str) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if t.family == family]

    def get_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_schemas(self, tool_names: list[str] | None = None) -> list[dict]:
        """Generate OpenAI-compatible tool schemas for selected tools."""
        tools = (
            [self._tools[n] for n in tool_names if n in self._tools]
            if tool_names
            else list(self._tools.values())
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]


class ToolRouter:
    """Contextual tool selection — injects only relevant tool schemas.

    Saves context window and reduces tool-selection errors by matching
    user intent to tool families rather than dumping all schemas.
    """

    FAMILY_KEYWORDS: dict[str, list[str]] = {
        "memory": ["remember", "recall", "forget", "memory", "memories", "facts", "fact", "durable facts", "preference", "favorite", "crush", "keep in mind", "store", "who is", "who's", "what is my", "did i tell you", "what was", "know about me", "remember about me", "saved about me", "all memories", "all facts"],
        "history": ["what did you say", "what did i say", "past conversation", "yesterday", "last time", "earlier conversation"],
        "system": [
            "system status", "robot status", "system health", "system uptime", "restart service",
            "what time", "what's the time", "current time", "what day", "what date", "clock",
            "calculate", "calculator", "math", "sqrt", "multiply", "divide", "plus", "minus", "sum", "equation", "formula",
            "hardware", "cpu", "gpu", "vram", "ram", "memory usage", "disk", "temperature", "specs",
            "death clock", "death date", "life expectancy", "longevity", "how long do i have to live", "how long will i live", "lifespan"
        ],
        "scheduler": [
            "timer", "timers", "set timer", "start timer", "cancel timer", "stop timer",
            "alarm", "alarms", "set alarm", "wake me up", "cancel alarm", "turn off alarm",
            "reminder", "reminders", "remind me", "cancel reminder", "schedule reminder",
            "daily briefing", "briefing", "morning brief", "what is my schedule", "schedule today"
        ],
        "personality": [
            "switch personality", "change personality", "switch character", "change character", "switch mode",
            "switch to cypher", "switch to luna", "switch to aria", "switch to glados", "switch to neo",
            "change to cypher", "change to luna", "change to aria", "change to glados",
            "be cypher", "be luna", "be aria", "be glados", "act like cypher", "act like luna", "act like glados"
        ],
        "knowledge": [
            "wikipedia", "wiki", "who was", "who is", "what is", "tell me about", "history of",
            "explain the", "biography", "biography of", "encyclopedia", "concept of", "theory of"
        ],
        "web": [
            "search online", "search web", "search for", "search", "look up", "find online", "browse web",
            "google", "duckduckgo", "latest news", "price of", "release date", "current weather", "weather in",
            "sports score", "stock price", "specs of", "specifications", "deep search", "search deeply",
            "date of", "when is", "tell me the date", "event", "news on", "look for", "find out", "lookup",
            "browser agent", "web agent", "headless browser", "browse page", "navigate url"
        ],
        "music": [
            "play", "song", "music", "track", "spotify", "soundcloud", "youtube", "ytdlp", "listen to", "put on",
            "pause music", "pause the music", "pause song", "pause the song", "pause",
            "resume music", "resume the music", "resume the song", "resume song", "resume", "continue",
            "replay", "replay the song", "replay song", "play again", "play it again", "play that song again",
            "play that song again on youtube", "play that again", "play it again on youtube",
            "repeat", "repeat the song", "repeat song", "repeat that", "loop", "loop the song",
            "unpause", "stop music", "stop the music", "stop the song", "stop song", "stop playing", "stop it", "stop that", "stop",
            "what is playing", "what's playing", "what song is this", "what track is this", "music status",
            "what song is playing", "what's currently playing", "what is currently playing", "is it playing", "is music playing",
            "did it finish playing", "id it finish playing", "did the song finish", "finished playing", "has it stopped", "stopped playing",
            "recently played", "song history", "music history", "last played", "last song",
            "what did you play", "what song did you play", "what song did you last play", "which song did you play",
            "local collection", "local songs", "local files", "downloaded songs", "my collection", "library", "what songs do you have", "what songs are there", "what music do you have",
            "volume", "set volume", "music volume", "turn it up", "turn it down", "louder", "quieter", "mute", "unmute", "audio", "vol",
            "restart", "restart it", "restart the song", "restart song", "restart music",
            "equalizer", "equaliser", "eq", "preset", "sound profile", "audio profile", "robot preset", "bass boost", "spatial audio", "sub bass", "reverb preset",
            "generate music", "make music", "compose music", "create a song", "generate song", "generate a beat", "synthesize music", "synthwave", "lofi beat", "cyberpunk beat"
        ],
        "image": [
            "generate image", "generate an image", "create image", "create an image", "draw an image", "paint an image", "illustration of", "render image", "generate picture",
            "make a picture", "make me a picture", "picture of", "photo of", "draw me a", "draw a", "make me a", "paint me a", "render a picture", "picture that you created", "image you created", "picture that you just created"
        ],
        "motion": [
            "wave", "raise hand", "kick", "walk", "turn around", "point at", "dance", "gesture", "stop motion", "reset pose", "nod head",
            "raise your hand", "raise your left hand", "raise your right hand", "raise left hand", "raise right hand",
            "wave your hand", "wave left hand", "wave right hand", "wave your left hand", "wave your right hand",
            "nod your head", "shake head", "shake your head", "clap", "bow", "cheer", "shrug"
        ],
        "perception": [
            "what do you see", "what do u see", "what are you seeing", "tell me what you see", "describe what you see", "describe what you're seeing",
            "what is this", "what is that", "what's this", "what's that", "what is this object", "what is that object", "what item is this",
            "look at this", "look around", "look at me", "who is here", "who is this", "can you see me", "do you see me", "can you see this", "do you see this",
            "what am i holding", "what i am holding", "what am i showing", "look at what i", "holding", "inspect scene", "check", "check this", "check that", "check what",
            "inspect", "look", "see", "camera", "vision", "describe scene", "video feed", "camera feed", "through the camera", "through your camera",
            "what color is it", "what color is this", "what color is that", "what color are", "what color", "color of this", "color of that", "color of",
            "what color is my shirt", "what am i wearing", "what color are my clothes", "shirt color",
            "what is in front of you", "what is in front of me", "what's in front of you", "what's in front of me", "in front of you",
            "what do you observe", "what do you notice", "how many people", "how many objects", "who do you see", "who is in front of you",
            "read this", "read paper", "read text", "read document", "read what is written", "what does this say", "what does it say", "what is written", "read label",
            "analyze frame", "vlm", "vlm inspect", "analyze camera", "inspect tabletop", "inspect desk", "check the table"
        ],
        "os": [
            "launch", "open", "start", "run", "start app", "open app", "launch app", "open vs code", "open spotify",
            "open chrome", "open firefox", "open terminal", "open files", "open calculator", "files", "file manager",
            "nautilus", "explorer", "calculator", "calc", "spotify", "chrome", "google chrome", "firefox", "mozilla",
            "terminal", "console", "vscode", "vs code", "code", "cursor", "browser", "web browser", "slack", "discord",
            "vlc", "obs", "settings", "text editor", "gedit", "kate", "notepad", "lock screen", "lock session", "lock",
            "sleep display", "wake display", "turn off screen", "screen off", "display sleep", "turn on screen", "screen on",
            "battery", "battery status", "power status", "clipboard", "copy to clipboard", "read clipboard",
            "paste clipboard", "notify", "desktop notification", "popup", "notify me", "mute audio", "unmute",
            "brightness", "screen brightness", "display brightness", "display_brightness", "os.display_brightness",
            "set brightness", "adjust brightness", "change brightness", "dim screen", "brighten screen", "brightnessctl",
            "ddcutil", "xbacklight", "monitor brightness", "screen level"
        ],
        "weather": [
            "weather", "forecast", "temperature", "temp", "temp outside", "is it raining", "rain", "sunny",
            "humid", "humidity", "snow", "cloudy", "weather forecast", "weather today", "weather in"
        ],
        "news": [
            "news", "headlines", "breaking news", "tech news", "world news", "top stories", "daily news",
            "latest news", "current events", "science news", "business news"
        ],
        "finance": [
            "stock", "stocks", "crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "doge",
            "dogecoin", "nvidia", "nvda", "apple", "aapl", "tesla", "tsla", "market", "sp500", "nasdaq",
            "share price", "price of bitcoin", "bitcoin price", "crypto price", "quote", "stock quote"
        ],
        "smart_home": [
            "turn on light", "turn off light", "lights", "light", "switch", "lamp", "desk lamp", "living room light",
            "thermostat", "ac temperature", "set thermostat", "smart home", "home assistant", "dim light",
            "fan", "turn on fan", "turn off fan", "smart device", "iot"
        ],
        "terminal": ["terminal", "bash", "shell command", "run command", "exec command", "cli", "shell", "execute terminal", "terminal access"],
        "git": ["git", "git status", "git diff", "git commit", "stage files", "commit changes", "workspace commit"],
        "code": ["run python", "execute python", "run script", "run code", "javascript snippet", "bash script", "code runner"],
        "calculator": ["calculate", "calculator", "eval", "math expression", "unit conversion", "financial formula"],
        "wikipedia": ["wikipedia", "wiki", "look up on wikipedia", "wiki summary"],
        "notes": ["note", "notes", "markdown note", "knowledge vault", "tag note", "search notes", "notes vault", "vault search"],
        "translation": ["translate", "translation", "translate to", "translate text", "translate from"],
        "network": ["wifi", "wi-fi", "wifi scan", "scan wifi", "access points", "speedtest", "network speed", "bandwidth test", "ping", "internet connection"],
        "communication": ["notification", "notify phone", "send to phone", "alert phone", "push alert", "phone alert", "sms", "text message", "send sms", "send text", "draft sms", "draft text"],
    }

    CHAINING_CONJUNCTIONS: list[str] = [
        "and then", "after that", "then next", "next", "then play", "then do", "then check",
        "and also", "as well as", "followed by", "and next", "first", "afterwards",
        "also", ". also", ";", "plus", "along with", "while", "as well", "and"
    ]

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def select_tools(
        self,
        user_text: str,
        all_tools: bool = False,
        recent_context: list[dict] | None = None,
    ) -> list[str]:
        """Select relevant tool families or return all tools so the LLM can autonomously route.

        Args:
            user_text: Current user input text.
            all_tools: If True, returns all registered tool names.
            recent_context: Optional list of recent message turns to resolve short conversational follow-ups.

        Returns:
            List of matching tool names.
        """
        if all_tools:
            return self._registry.list_tools()

        import re
        text_lower = user_text.lower()
        selected_families: set[str] = set()

        for family, keywords in self.FAMILY_KEYWORDS.items():
            for kw in keywords:
                if " " in kw:
                    if kw in text_lower:
                        selected_families.add(family)
                        break
                else:
                    if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                        selected_families.add(family)
                        break

        # Match explicit registered tool names or dotted names (e.g. os.display_brightness)
        for tool_def in self._registry.get_all():
            name_lower = tool_def.name.lower()
            dotted = name_lower.replace("_", ".")
            if name_lower in text_lower or dotted in text_lower or tool_def.name in text_lower:
                selected_families.add(tool_def.family)

        # Context-aware follow-up resolution:
        # If user utterance is short (e.g. "files", "1", "yes", "do it", "sure", "play it"),
        # check recent assistant context to retain active tool families
        if recent_context and (not selected_families or len(text_lower.split()) <= 3):
            recent_texts = []
            for turn in reversed(recent_context[-3:]):
                c = turn.get("content", "")
                if isinstance(c, str):
                    recent_texts.append(c.lower())
            combined_recent = " ".join(recent_texts)

            for family, keywords in self.FAMILY_KEYWORDS.items():
                for kw in keywords:
                    if " " in kw:
                        if kw in combined_recent:
                            selected_families.add(family)
                            break
                    else:
                        if re.search(rf"\b{re.escape(kw)}\b", combined_recent):
                            selected_families.add(family)
                            break

        # If chaining conjunctions are detected and at least one tool family matched,
        # add any families referenced in the chained sub-clauses without returning ALL tools blindly
        has_chaining = any(conj in text_lower for conj in self.CHAINING_CONJUNCTIONS)
        if has_chaining and selected_families:
            if any(w in text_lower for w in ["look", "see", "camera", "inspect", "check", "color", "wear", "shirt", "holding", "showing", "view"]):
                selected_families.add("perception")
            if any(w in text_lower for w in ["play", "song", "music", "volume", "sound", "track", "restart", "vol"]):
                selected_families.add("music")
            if any(w in text_lower for w in ["search", "find", "online", "web", "date", "when", "google"]):
                selected_families.add("web")
            if any(w in text_lower for w in ["wave", "gesture", "motion", "walk", "kick", "lean", "lean_in", "hand"]):
                selected_families.add("motion")
            if any(w in text_lower for w in ["open", "launch", "lock", "clipboard", "notify", "battery", "files", "calc", "terminal", "spotify"]):
                selected_families.add("os")
            if any(w in text_lower for w in ["weather", "forecast", "rain", "temperature"]):
                selected_families.add("weather")
            if any(w in text_lower for w in ["news", "headlines"]):
                selected_families.add("news")
            if any(w in text_lower for w in ["crypto", "bitcoin", "stock", "price"]):
                selected_families.add("finance")
            if any(w in text_lower for w in ["light", "lights", "lamp", "thermostat", "fan", "switch"]):
                selected_families.add("smart_home")

        if not selected_families:
            return []

        tool_names = []
        for family in selected_families:
            for tool in self._registry.get_family(family):
                if tool.name not in tool_names:
                    tool_names.append(tool.name)

        return tool_names


class PermissionManager:
    """Validates tool execution against permission policies.

    The LLM proposes; the PermissionManager decides.
    """

    # Confirmations are single-use and short-lived.  A new request supersedes
    # every older in-flight confirmation so a late/stale "yes" from the user
    # can never authorize a different action.
    CONFIRMATION_TTL_S: float = 120.0

    def __init__(
        self,
        max_auto_level: PermissionLevel = PermissionLevel.HARMLESS_LOCAL,
        event_bus: Any | None = None,
    ) -> None:
        self._max_auto = max_auto_level
        self._events = event_bus
        # Bounded TTL-LRU, not an unbounded dict: a 24/7 robot may execute an
        # arbitrary number of distinct non-idempotent calls.
        self._execution_history: BoundedTTLCache = BoundedTTLCache(maxsize=256, ttl_s=60.0)
        self._search_cache: BoundedTTLCache = BoundedTTLCache(maxsize=128, ttl_s=120.0)
        self._confirmation_seq = itertools.count(1)
        self._active_confirmation_id: str | None = None
        self._pending_confirmations: dict[str, dict[str, Any]] = {}

    def check(self, tool: ToolDefinition) -> bool:
        """Check if a tool can be auto-executed without user confirmation."""
        if tool.permission_level <= self._max_auto:
            return True
        logger.warning(
            "permission_denied",
            tool=tool.name,
            level=tool.permission_level.name,
            max_auto=self._max_auto.name,
        )
        return False

    # ─── Confirmation lifecycle ─────────────────────────────────────────────

    @property
    def active_confirmation_id(self) -> str | None:
        """Return the currently valid in-flight confirmation id, if any."""
        return self._active_confirmation_id

    def request_confirmation(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any] | None = None,
        ttl_s: float | None = None,
    ) -> str:
        """Create a new single-use, monotonically identified confirmation.

        Any previously pending confirmation is invalidated immediately.  This
        prevents a stale response for an earlier question from satisfying a
        newly requested action.

        Args:
            tool: Tool awaiting authorization.
            arguments: Arguments to associate with the request (log-redacted).
            ttl_s: Optional lifetime override in seconds.

        Returns:
            Monotonically increasing confirmation identifier.
        """
        self._pending_confirmations.clear()
        confirmation_id = f"confirm_{next(self._confirmation_seq)}"
        ttl = self.CONFIRMATION_TTL_S if ttl_s is None else max(0.0, float(ttl_s))
        self._pending_confirmations[confirmation_id] = {
            "tool": tool.name,
            "arguments": redact_tool_arguments(arguments or {}),
            "expires_at": time.monotonic() + ttl,
        }
        self._active_confirmation_id = confirmation_id
        logger.info(
            "confirmation_requested",
            confirmation_id=confirmation_id,
            tool=tool.name,
            ttl_s=ttl,
        )
        return confirmation_id

    def is_confirmation_valid(self, confirmation_id: str | None) -> bool:
        """Return whether a confirmation id is current, pending, and unexpired."""
        if not confirmation_id:
            return False
        entry = self._pending_confirmations.get(str(confirmation_id))
        if entry is None:
            return False
        if time.monotonic() > float(entry["expires_at"]):
            self._pending_confirmations.pop(str(confirmation_id), None)
            if self._active_confirmation_id == str(confirmation_id):
                self._active_confirmation_id = None
            return False
        return self._active_confirmation_id == str(confirmation_id)

    def resolve_confirmation(self, confirmation_id: str, approved: bool = True) -> bool:
        """Resolve a pending confirmation exactly once.

        Args:
            confirmation_id: Identifier returned by :meth:`request_confirmation`.
            approved: Whether the user authorized the action.

        Returns:
            True only when the id was the single active, unexpired confirmation
            and ``approved`` is True.  Unknown, superseded, or expired ids are
            rejected with a warning and return False.
        """
        cid = str(confirmation_id)
        entry = self._pending_confirmations.get(cid)
        if entry is None or self._active_confirmation_id != cid:
            logger.warning(
                "confirmation_rejected",
                confirmation_id=cid,
                reason="unknown_or_superseded",
            )
            return False
        if time.monotonic() > float(entry["expires_at"]):
            self._pending_confirmations.pop(cid, None)
            self._active_confirmation_id = None
            logger.warning("confirmation_rejected", confirmation_id=cid, reason="expired")
            return False

        # Consume the confirmation: it can never authorize a second action.
        self._pending_confirmations.pop(cid, None)
        self._active_confirmation_id = None
        logger.info("confirmation_resolved", confirmation_id=cid, approved=bool(approved))
        return bool(approved)

    def reject_confirmation(self, confirmation_id: str) -> bool:
        """Explicitly reject/consume a pending confirmation id."""
        return self.resolve_confirmation(confirmation_id, approved=False)

    async def execute(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool with permission checking, validation, retries, and timeout.

        Returned envelopes always carry ``success`` (bool).  A handler-returned
        dict with a failure ``status`` (error/failed/timeout/denied/not_sent/
        cancelled) or a non-empty ``error`` field is converted into a failure
        envelope while retaining the original payload in ``result``.

        Args:
            tool: Registered tool definition to execute.
            arguments: Raw model/user supplied arguments.

        Returns:
            Structured outcome with keys ``status``, ``tool``, ``result``,
            ``error``, ``duration_ms``, ``retries``, and ``success``.

        Raises:
            asyncio.CancelledError: If the caller cancels the execution.  The
                cancellation is propagated (after telemetry) so task owners can
                perform deterministic cleanup.
        """
        if not self.check(tool):
            return {
                "status": ToolOutcomeStatus.DENIED.value,
                "tool": tool.name,
                "result": None,
                "error": f"Permission denied: {tool.name} requires level {tool.permission_level.name}",
                "duration_ms": 0.0,
                "retries": 0,
                "success": False,
            }

        # Guard non-idempotent tools against duplicate side-effects
        try:
            args_json = json.dumps(arguments or {}, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_json = repr(arguments)
        exec_key = f"{tool.name}:{args_json}"
        if not tool.is_idempotent:
            last_outcome = self._execution_history.get(exec_key)
            if isinstance(last_outcome, dict) and last_outcome.get("status") == ToolOutcomeStatus.SUCCESS.value:
                logger.warning("duplicate_side_effect_prevented", tool=tool.name)
                return {
                    "status": ToolOutcomeStatus.DENIED.value,
                    "tool": tool.name,
                    "result": last_outcome.get("result"),
                    "error": f"Duplicate side-effect prevented: {tool.name} was already executed within 5.0s window.",
                    "duration_ms": 0.0,
                    "retries": 0,
                    "success": False,
                }

        if self._events:
            await self._events.emit(
                "tool_executing",
                tool=tool.name,
                arguments=redact_tool_arguments(arguments),
            )

        import inspect

        sig = inspect.signature(tool.handler)
        handler_params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in handler_params.values()
        )

        param_aliases = {
            "fact": ["text", "memory", "content", "note", "information", "info", "data", "message", "value"],
            "query": [
                "q", "search", "search_query", "term", "keyword", "keywords", "topic", "text",
                "song", "track", "title", "name", "prompt", "song_name", "track_name", "artist",
                "music", "audio", "video", "url", "target", "item", "subject"
            ],
            "action": ["act", "motion", "command", "type", "operation", "gesture", "movement", "mode"],
            "prompt": ["text", "description", "query", "image_prompt", "caption"],
            "name": ["personality", "character", "persona", "mode", "bot_name", "name_str"],
            "timezone_str": ["timezone", "tz", "country", "city", "location", "place", "region"],
            "source": ["platform", "src", "provider", "music_source"],
            "volume": ["vol", "level", "val"],
            "expression": ["expr", "calc", "formula", "math", "equation", "calculation", "query"],
            "job_id": ["id", "job", "task_id", "jobId"],
            "preset": ["preset_name", "profile", "eq", "name"],
            "duration_seconds": ["duration", "seconds", "sec", "secs", "time", "val", "minutes", "min", "mins"],
            "target_time": ["time", "time_expression", "target", "when", "target_time_expr", "at", "time_str"],
            "time_expression": ["target_time", "time", "when", "time_expr", "target", "at"],
            "message": ["note", "text", "content", "reminder", "task", "msg", "label", "body"],
            "identifier": ["id", "job_id", "timer_id", "alarm_id", "reminder_id", "name", "label", "title"],
            "app_name": ["app", "name", "application", "program", "command", "target", "app_name_str"],
            "title": ["summary", "header", "topic", "subject", "heading", "name"],
            "text": ["content", "data", "value", "string", "snippet", "body", "code"],
            "location": ["city", "place", "where", "locale", "area", "address", "town", "query"],
            "category": ["topic", "section", "type", "genre", "category_name", "feed"],
            "symbol": ["ticker", "coin", "stock", "crypto", "asset", "company", "token", "query", "target"],
            "entity_id": ["device", "target", "entity", "light", "switch", "lamp", "thermostat", "name", "id"],
            "domain": ["type", "filter", "category", "scope"],
        }

        # Resolve handler-level aliases *before* schema hardening.  Anything the
        # framework injects (aliases or the single-string fallback) must still
        # pass enum/range/type validation below.
        mapped_args: dict[str, Any] = dict(arguments or {})
        schema_properties: set[str] = set(
            (tool.parameters or {}).get("properties", {}).keys()
        )
        for target_param, aliases in param_aliases.items():
            if target_param in handler_params and target_param not in mapped_args:
                for alias in aliases:
                    if alias in mapped_args and alias not in schema_properties:
                        # Pop the alias so it cannot survive as an unknown key.
                        mapped_args[target_param] = mapped_args.pop(alias)
                        break

        # Handle composite artist + title for music tools
        if "artist" in mapped_args and "query" not in mapped_args:
            other_title = (
                mapped_args.get("track")
                or mapped_args.get("song")
                or mapped_args.get("title")
                or ""
            )
            if other_title:
                mapped_args["query"] = f"{mapped_args['artist']} - {other_title}"
            else:
                mapped_args["query"] = str(mapped_args["artist"])

        # Universal single-string fallback: bind an unknown string argument to
        # a required handler parameter that is still missing.  The value is
        # validated against the schema immediately afterwards.
        for p_name, p_param in handler_params.items():
            if p_name in ("self", "args", "kwargs"):
                continue
            if p_param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if p_param.default == inspect.Parameter.empty and p_name not in mapped_args:
                for key, value in list(mapped_args.items()):
                    if isinstance(value, str) and value.strip() and key not in handler_params:
                        mapped_args[p_name] = mapped_args.pop(key)
                        break

        def _validation_failure(err: ToolValidationError) -> dict[str, Any]:
            safe_error = redact_secrets(str(err))
            logger.warning("tool_validation_failed", tool=tool.name, error=safe_error)
            return err.to_dict()

        # Apply strict schema hardening, alias resolution, and conflict detection.
        try:
            normalized_args = harden_tool_arguments(mapped_args, schema=tool.parameters)
        except ToolValidationError as err:
            if self._events:
                await self._events.emit("tool_error", tool=tool.name, error=redact_secrets(str(err)))
            res = _validation_failure(err)
            res.update(
                {
                    "status": "failure",
                    "tool": tool.name,
                    "result": None,
                    "duration_ms": 0.0,
                    "retries": 0,
                    "success": False,
                }
            )
            return res

        # Clamp search fan-out even when the schema forgot to declare bounds.
        if "max_results" in normalized_args:
            try:
                requested = int(normalized_args["max_results"])
                normalized_args["max_results"] = max(
                    MIN_SEARCH_RESULTS, min(MAX_SEARCH_RESULTS, requested)
                )
            except (TypeError, ValueError):
                # Leave non-numeric values for the schema validator to reject.
                pass

        # Any post-hardening mapping (declared-property aliases) is revalidated.
        for target_param, aliases in param_aliases.items():
            if target_param in handler_params and target_param not in normalized_args:
                for alias in aliases:
                    if alias in normalized_args and alias != target_param:
                        normalized_args[target_param] = normalized_args.pop(alias)
                        break
        try:
            normalized_args = harden_tool_arguments(
                normalized_args,
                schema=tool.parameters,
            )
        except ToolValidationError as err:
            if self._events:
                await self._events.emit("tool_error", tool=tool.name, error=redact_secrets(str(err)))
            res = _validation_failure(err)
            res.update(
                {
                    "status": "failure",
                    "tool": tool.name,
                    "result": None,
                    "duration_ms": 0.0,
                    "retries": 0,
                    "success": False,
                }
            )
            return res

        # If handler does not accept **kwargs, prune unaccepted keys so
        # invocation cannot raise TypeError.  Schema-declared unknowns were
        # already rejected above.
        if not accepts_kwargs:
            filtered_args = {k: v for k, v in normalized_args.items() if k in handler_params}
        else:
            filtered_args = normalized_args

        # Bounded framework-level search reuse.  The handler still enforces
        # its own cache; this guarantees the framework's cache cannot grow
        # without bound either.
        cacheable_search = (
            tool.is_idempotent
            and "search" in tool.name
            and tool.family in ("web", "news", "finance", "knowledge", "network")
        )
        if cacheable_search:
            cached_outcome = self._search_cache.get(exec_key)
            if isinstance(cached_outcome, dict):
                cached_copy = dict(cached_outcome)
                cached_copy["cached"] = True
                return cached_copy

        t_start = time.perf_counter()
        max_attempts = 1 + (tool.max_retries if tool.is_idempotent else 0)
        retries_done = 0
        status = ToolOutcomeStatus.FAILURE.value
        err_msg: str | None = None

        # Per-tool budget.  Do not globally flatten web tools to 3 s: each
        # ToolDefinition declares its own budget and browser tools get a floor.
        effective_timeout = effective_tool_timeout(tool)
        deadline = time.perf_counter() + effective_timeout

        for attempt in range(max_attempts):
            if attempt > 0:
                retries_done += 1
                logger.info("tool_retry_attempt", tool=tool.name, attempt=attempt)
                await asyncio.sleep(0.1)

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                status = ToolOutcomeStatus.TIMEOUT.value
                err_msg = f"Tool timed out after {effective_timeout}s"
                logger.warning(
                    "tool_timeout",
                    tool=tool.name,
                    timeout=effective_timeout,
                    attempt=attempt,
                )
                break

            try:
                result = await asyncio.wait_for(
                    tool.handler(**filtered_args),
                    timeout=remaining,
                )
                dur_ms = (time.perf_counter() - t_start) * 1000

                # A returned failure dictionary must never become success.
                is_failure, failure_status, failure_error = classify_tool_result(result)
                if is_failure:
                    logger.warning(
                        "tool_handler_reported_failure",
                        tool=tool.name,
                        status=failure_status,
                        duration_ms=round(dur_ms, 2),
                    )
                    if self._events:
                        await self._events.emit(
                            "tool_error",
                            tool=tool.name,
                            error=redact_secrets(failure_error or "Tool reported failure"),
                        )
                    return {
                        "status": failure_status or ToolOutcomeStatus.FAILURE.value,
                        "tool": tool.name,
                        "result": result,
                        "error": failure_error,
                        "duration_ms": round(dur_ms, 2),
                        "retries": retries_done,
                        "success": False,
                        "handler_status": (
                            result.get("status") if isinstance(result, dict) else None
                        ),
                    }

                logger.info("tool_executed", tool=tool.name, duration_ms=round(dur_ms, 2))
                if self._events:
                    await self._events.emit(
                        "tool_completed",
                        tool=tool.name,
                        result=redact_secrets(str(result))[:100],
                    )

                outcome = {
                    "status": ToolOutcomeStatus.SUCCESS.value,
                    "tool": tool.name,
                    "result": result,
                    "duration_ms": round(dur_ms, 2),
                    "retries": retries_done,
                    "success": True,
                }
                if not tool.is_idempotent:
                    self._execution_history.set(exec_key, outcome)
                if cacheable_search:
                    self._search_cache.set(exec_key, outcome)
                return outcome

            except asyncio.CancelledError:
                dur_ms = (time.perf_counter() - t_start) * 1000
                logger.warning("tool_cancelled", tool=tool.name)
                if self._events:
                    try:
                        await self._events.emit(
                            "tool_error", tool=tool.name, error="Cancelled"
                        )
                    except Exception:
                        pass
                # Re-raise: swallowing cancellation hides barge-in from the
                # task owner and leaks half-cancelled work.
                raise
            except asyncio.TimeoutError:
                status = ToolOutcomeStatus.TIMEOUT.value
                err_msg = f"Tool timed out after {effective_timeout}s"
                logger.warning(
                    "tool_timeout",
                    tool=tool.name,
                    timeout=effective_timeout,
                    attempt=attempt,
                )
            except Exception as e:
                status = ToolOutcomeStatus.FAILURE.value
                err_msg = redact_secrets(str(e))
                logger.error("tool_error", tool=tool.name, error=err_msg, attempt=attempt)

        dur_ms = (time.perf_counter() - t_start) * 1000
        if self._events:
            await self._events.emit(
                "tool_error", tool=tool.name, error=err_msg
            )

        return {
            "status": status,
            "tool": tool.name,
            "result": {"status": status, "error": err_msg or "Unknown execution failure"},
            "error": err_msg or "Unknown execution failure",
            "duration_ms": round(dur_ms, 2),
            "retries": retries_done,
            "success": False,
        }

