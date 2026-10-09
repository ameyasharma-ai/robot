"""Tool framework — registry, router, and permission gates.

Qwen proposes actions via tool_call; Robot Core validates and executes.
A model-generated tool call is a REQUEST, not authorization.
"""

from __future__ import annotations

import asyncio
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

logger = get_logger("tools")

ToolHandler = Callable[..., Coroutine[Any, Any, Any]]


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
    timeout_s: float = 10.0
    risk_level: str = "low"
    provider: str = ""
    family: str = ""                   # e.g., "memory", "music", "system"


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
            "hardware", "cpu", "gpu", "vram", "ram", "memory usage", "disk", "temperature", "specs"
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
            "date of", "when is", "tell me the date", "event", "news on", "look for", "find out", "lookup"
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
            "equalizer", "equaliser", "eq", "preset", "sound profile", "audio profile", "robot preset", "bass boost", "spatial audio", "sub bass", "reverb preset"
        ],
        "image": ["generate image", "generate an image", "create image", "create an image", "draw an image", "paint an image", "illustration of", "render image", "generate picture"],
        "motion": ["wave", "raise hand", "kick", "walk", "turn around", "point at", "dance", "gesture", "stop motion", "reset pose", "nod head"],
        "perception": [
            "what do you see", "what do u see", "what are you seeing", "what is this", "what is that", "what's this", "what's that",
            "look at this", "look around", "look at me", "who is here", "who is this", "can you see me", "do you see me",
            "what am i holding", "what i am holding", "inspect scene", "check", "check this", "check that", "check what",
            "inspect", "look", "see", "camera", "vision", "describe scene", "describe what you see"
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
            if any(w in text_lower for w in ["look", "see", "camera", "inspect", "check"]):
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

    def __init__(
        self,
        max_auto_level: PermissionLevel = PermissionLevel.HARMLESS_LOCAL,
        event_bus: Any | None = None,
    ) -> None:
        self._max_auto = max_auto_level
        self._events = event_bus

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

    async def execute(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool with permission checking, robust argument normalization, and timeout."""
        if not self.check(tool):
            return {"error": f"Permission denied: {tool.name} requires level {tool.permission_level.name}"}

        if self._events:
            await self._events.emit("tool_executing", tool=tool.name, arguments=arguments)

        # Normalize common LLM parameter name discrepancies and aliases
        normalized_args = dict(arguments or {})
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

        # Handle composite artist + title for music tools
        if "artist" in normalized_args and "query" not in normalized_args:
            other_title = normalized_args.get("track") or normalized_args.get("song") or normalized_args.get("title") or ""
            if other_title:
                normalized_args["query"] = f"{normalized_args['artist']} - {other_title}"
            else:
                normalized_args["query"] = str(normalized_args["artist"])

        try:
            import inspect
            sig = inspect.signature(tool.handler)
            handler_params = sig.parameters

            # Check if handler accepts **kwargs
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in handler_params.values())

            # Map aliases if target parameter is missing in provided arguments
            for target_param, aliases in param_aliases.items():
                if target_param in handler_params and target_param not in normalized_args:
                    for alias in aliases:
                        if alias in normalized_args:
                            normalized_args[target_param] = normalized_args[alias]
                            break

            # Universal single-string fallback:
            # If the handler expects a required parameter without default that is still missing,
            # and normalized_args contains string values, bind the first available string value.
            for p_name, p_param in handler_params.items():
                if p_param.default == inspect.Parameter.empty and p_name not in normalized_args:
                    # Look for any string value in normalized_args that was not mapped
                    for k, v in list(normalized_args.items()):
                        if isinstance(v, str) and v.strip() and k not in handler_params:
                            normalized_args[p_name] = v
                            break

            # If handler does not accept **kwargs, prune unknown keys so invocation does not raise TypeError
            if not accepts_kwargs:
                filtered_args = {k: v for k, v in normalized_args.items() if k in handler_params}
            else:
                filtered_args = normalized_args

            result = await asyncio.wait_for(
                tool.handler(**filtered_args),
                timeout=tool.timeout_s,
            )
            logger.info("tool_executed", tool=tool.name)
            if self._events:
                await self._events.emit("tool_completed", tool=tool.name, result=str(result)[:100])
            return {"result": result}
        except asyncio.TimeoutError:
            logger.error("tool_timeout", tool=tool.name, timeout=tool.timeout_s)
            if self._events:
                await self._events.emit("tool_error", tool=tool.name, error=f"Timeout after {tool.timeout_s}s")
            return {"error": f"Tool timed out after {tool.timeout_s}s"}
        except Exception as e:
            logger.error("tool_error", tool=tool.name, error=str(e))
            if self._events:
                await self._events.emit("tool_error", tool=tool.name, error=str(e))
            return {"error": str(e)}

