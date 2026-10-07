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

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name, family=tool.family)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

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
        "personality": ["switch personality", "change personality", "switch character", "change character", "switch mode"],
        "knowledge": [
            "wikipedia", "wiki", "who was", "who is", "what is", "tell me about", "history of",
            "explain the", "biography", "biography of", "encyclopedia", "concept of", "theory of"
        ],
        "web": ["search online", "search web", "look up", "find online", "browse web", "google", "duckduckgo", "latest news", "price of", "release date", "current weather", "weather in", "sports score", "stock price", "specs of", "specifications", "deep search", "search deeply"],
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
            "volume", "set volume", "music volume", "turn it up", "turn it down", "louder", "quieter", "mute", "unmute", "audio"
        ],
        "image": ["generate image", "generate an image", "create image", "create an image", "draw an image", "paint an image", "illustration of", "render image", "generate picture"],
        "motion": ["wave", "raise hand", "kick", "walk", "turn around", "point at", "dance", "gesture", "stop motion", "reset pose", "nod head"],
        "perception": ["what do you see", "what do u see", "what are you seeing", "what is this", "what is that", "what's this", "what's that", "look at this", "look around", "look at me", "who is here", "who is this", "can you see me", "do you see me", "what am i holding", "what i am holding", "inspect scene"],
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def select_tools(self, user_text: str, all_tools: bool = False) -> list[str]:
        """Select relevant tool families or return all tools so the LLM can autonomously route."""
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
            "fact": ["text", "memory", "content", "note", "information", "info", "data", "message"],
            "query": ["q", "search", "search_query", "term", "keyword", "keywords", "topic", "text", "song", "title", "name", "prompt"],
            "action": ["act", "motion", "command", "type", "operation", "gesture", "movement"],
            "prompt": ["text", "description", "query", "image_prompt", "caption"],
            "name": ["personality", "character", "persona", "mode", "bot_name"],
            "timezone_str": ["timezone", "tz", "country", "city", "location", "place", "region"],
            "source": ["platform", "src", "provider", "music_source"],
            "volume": ["vol", "level", "val"],
            "expression": ["expr", "calc", "formula", "math", "equation"],
        }

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

