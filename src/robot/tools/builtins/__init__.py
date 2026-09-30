"""Built-in tools for memory, system status, and personality management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


def register_builtin_tools(engine: "RobotEngine") -> None:
    """Register all built-in tools with the engine's tool registry."""

    # --- Memory Tools ---
    async def memory_write(fact: str) -> str:
        await engine.memory.add_memory(fact, user_id="user")
        return f"Successfully remembered: '{fact}'"

    engine.tool_registry.register(
        ToolDefinition(
            name="memory.write",
            description="Store and remember an important fact, preference, or detail about the user for future turns",
            parameters={
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The exact fact or preference to remember"},
                },
                "required": ["fact"],
            },
            handler=memory_write,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="memory",
        )
    )

    async def memory_search(query: str, limit: int = 5) -> Any:
        return await engine.memory.get_relevant_memories(query, limit)

    engine.tool_registry.register(
        ToolDefinition(
            name="memory.search",
            description="Search persistent memories for relevant facts, preferences, or past events",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
            handler=memory_search,
            permission_level=PermissionLevel.READ_ONLY,
            family="memory",
        )
    )

    # --- Web Search Tools (DuckDuckGo) ---
    async def web_search(query: str, max_results: int = 3) -> list[dict]:
        """Perform a live DuckDuckGo web search for current real-world knowledge."""
        try:
            from ddgs import DDGS
            import asyncio

            def _search():
                results = DDGS().text(query, max_results=max_results)
                return list(results) if results else []

            results = await asyncio.to_thread(_search)
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
                for r in results
            ]
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

    engine.tool_registry.register(
        ToolDefinition(
            name="web.search",
            description="Search the web using DuckDuckGo for live facts, current events, products (e.g. latest iPhone), or online information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to look up on the web"},
                    "max_results": {"type": "integer", "description": "Maximum search results to return", "default": 3},
                },
                "required": ["query"],
            },
            handler=web_search,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="web",
            timeout_s=20.0,
        )
    )

    async def history_search(query: str, limit: int = 10) -> Any:
        return await engine.memory.search_history(query, limit)

    engine.tool_registry.register(
        ToolDefinition(
            name="history.search",
            description="Search past conversation history using full-text search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
            handler=history_search,
            permission_level=PermissionLevel.READ_ONLY,
            family="history",
        )
    )

    # --- System Tools ---
    async def system_status() -> dict:
        return {
            "state": engine.state.state.to_compact(),
            "telemetry": engine.telemetry.snapshot(),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="system.status",
            description="Get current robot state, service health, and telemetry metrics",
            parameters={"type": "object", "properties": {}},
            handler=system_status,
            permission_level=PermissionLevel.READ_ONLY,
            family="system",
        )
    )

    # --- Personality Tools ---
    async def switch_personality(name: str) -> str:
        if engine.personalities.set_active(name):
            engine.state.state.active_personality = name
            return f"Switched to {name} personality"
        available = engine.personalities.list_personalities()
        return f"Personality '{name}' not found. Available: {available}"

    engine.tool_registry.register(
        ToolDefinition(
            name="personality.switch",
            description="Switch to a different personality/character mode",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Personality name to switch to"},
                },
                "required": ["name"],
            },
            handler=switch_personality,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="personality",
        )
    )

    # --- Perception & Camera Tools ---
    from robot.tools.builtins.perception import register_perception_tools
    register_perception_tools(engine)

    # --- Character & Motion Tools ---
    from robot.tools.builtins.motion import register_motion_tools
    register_motion_tools(engine)
