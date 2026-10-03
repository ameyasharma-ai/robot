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
        "memory": ["remember", "recall", "forget", "memory", "preference", "favorite", "crush", "keep in mind", "store", "who is", "who's", "what is my", "did i tell you", "what was"],
        "history": ["said", "told", "conversation", "yesterday", "last time", "before", "earlier"],
        "system": ["status", "health", "uptime", "service", "restart", "time", "date", "clock", "day of week", "what time", "what day"],
        "personality": ["personality", "switch", "mode", "character"],
        "web": ["search", "look up", "find online", "browse", "google", "duckduckgo", "latest", "news", "price", "release date", "current"],
        "image": ["generate image", "generate an image", "draw", "picture", "create image", "paint", "illustration", "generate picture", "photo", "render image", "image of"],
        "motion": ["wave", "raise your hand", "raise hand", "kick", "walk", "turn", "point", "sit", "stand", "dance", "move", "gesture", "look at", "stop motion", "reset pose", "body", "nod", "motion"],
        "perception": ["what do you see", "what do u see", "what are you seeing", "what is this", "what is that", "what's this", "what's that", "look at this", "look around", "look at me", "who is here", "who is this", "camera", "snapshot", "inspect", "can you see", "do you see", "see", "what am i holding", "what i am holding", "view"],
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def select_tools(self, user_text: str) -> list[str]:
        """Select relevant tool families based on user intent."""
        text_lower = user_text.lower()
        selected_families: set[str] = set()

        for family, keywords in self.FAMILY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                selected_families.add(family)

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
        """Execute a tool with permission checking and timeout."""
        if not self.check(tool):
            return {"error": f"Permission denied: {tool.name} requires level {tool.permission_level.name}"}

        if self._events:
            await self._events.emit("tool_executing", tool=tool.name, arguments=arguments)

        try:
            result = await asyncio.wait_for(
                tool.handler(**arguments),
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

