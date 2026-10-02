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

    # Short-Lived in-memory search LRU cache (TTL: 10 minutes)
    _search_cache: dict[str, tuple[float, list[dict]]] = {}
    SEARCH_CACHE_TTL = 600.0

    # --- Web Search Tools (Ultra-fast Multi-engine with LRU Cache) ---
    async def web_search(query: str, max_results: int = 4) -> list[dict]:
        """Perform an ultra-fast, extensive web search with multi-engine failover."""
        import asyncio
        import html as html_lib
        import json
        import re
        import time as _time
        import urllib.parse
        import urllib.request

        # Clean search query: strip natural language conversational filler
        clean_q = query.strip()
        lower_q = clean_q.lower()
        for prefix in [
            "search for", "search online for", "search the web for",
            "search about", "look up", "find online", "find out about",
            "find out", "google", "tell me about", "what is", "who is",
            "what happened in", "search"
        ]:
            if lower_q.startswith(prefix) and len(clean_q) > len(prefix) + 3:
                clean_q = clean_q[len(prefix):].strip(" ?:.,!")
                break

        cache_key = f"{clean_q.lower()}::{max_results}"
        now = _time.time()
        if cache_key in _search_cache:
            ts, cached_res = _search_cache[cache_key]
            if now - ts < SEARCH_CACHE_TTL:
                return cached_res

        def _execute_search():
            # 1. Primary Engine: DuckDuckGo Lite with paired Title, URL, and Snippet extraction
            try:
                url = "https://lite.duckduckgo.com/lite/"
                data = urllib.parse.urlencode({"q": clean_q}).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                        "Referer": "https://lite.duckduckgo.com/",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    page_html = resp.read().decode("utf-8", errors="ignore")

                if "result-snippet" in page_html:
                    link_pattern = re.compile(
                        r'<a[^>]+href=[\x27\"]([^\x27\"]+)[\x27\"][^>]*class=[\x27\"]result-link[\x27\"][^>]*>(.*?)</a>',
                        re.DOTALL,
                    )
                    snippet_pattern = re.compile(
                        r'<td class=[\x27\"]result-snippet[\x27\"]>(.*?)</td>',
                        re.DOTALL,
                    )
                    links = link_pattern.findall(page_html)
                    snippets = snippet_pattern.findall(page_html)

                    results = []
                    for i in range(min(len(snippets), max_results)):
                        clean_snip = html_lib.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
                        if i < len(links):
                            u, t = links[i]
                            clean_title = html_lib.unescape(re.sub(r"<[^>]+>", "", t)).strip()
                        else:
                            clean_title = clean_q
                            u = "https://duckduckgo.com"
                        if clean_snip:
                            results.append({"title": clean_title, "snippet": clean_snip, "url": u})
                    if results:
                        return results
            except Exception:
                pass

            # 2. Secondary Engine: Bing News & Web RSS (ultra-fast, highly current, no bot wall)
            try:
                bing_url = f"https://www.bing.com/news/search?q={urllib.parse.quote(clean_q)}&format=rss"
                req = urllib.request.Request(
                    bing_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    xml_data = resp.read().decode("utf-8", errors="ignore")

                items = re.findall(r"<item>(.*?)</item>", xml_data, re.DOTALL)
                if items:
                    bing_results = []
                    for item in items[:max_results]:
                        title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
                        link_m = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
                        desc_m = re.search(r"<description>(.*?)</description>", item, re.DOTALL)

                        t = html_lib.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip() if title_m else ""
                        d = html_lib.unescape(re.sub(r"<[^>]+>", "", desc_m.group(1))).strip() if desc_m else ""
                        u = link_m.group(1).strip() if link_m else ""
                        if d or t:
                            bing_results.append({"title": t, "snippet": d, "url": u})
                    if bing_results:
                        return bing_results
            except Exception:
                pass

            # 3. Tertiary Engine: DuckDuckGo Instant Answers API
            try:
                api_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
                req = urllib.request.Request(api_url, headers={"User-Agent": "RobotCore/1.0"})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                results = []
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", clean_q),
                        "snippet": data.get("AbstractText"),
                        "url": data.get("AbstractURL", ""),
                    })
                if data.get("Answer"):
                    results.append({"title": "Direct Answer", "snippet": data.get("Answer"), "url": ""})
                for rel in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(rel, dict) and rel.get("Text"):
                        results.append({
                            "title": rel.get("Text").split(" - ")[0],
                            "snippet": rel.get("Text"),
                            "url": rel.get("FirstURL", ""),
                        })
                if results:
                    return results[:max_results]
            except Exception:
                pass

            # 4. Quaternary Engine: ddgs library with strict limit
            try:
                from ddgs import DDGS
                res = list(DDGS().text(clean_q, max_results=max_results))
                if res:
                    return [
                        {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                        for r in res
                    ]
            except Exception:
                pass

            return [{"error": f"No web search results could be retrieved for '{clean_q}'."}]

        results = await asyncio.to_thread(_execute_search)
        if results and isinstance(results, list) and "error" not in results[0]:
            _search_cache[cache_key] = (_time.time(), results)
        return results

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

    async def get_current_time(timezone_str: str | None = None) -> dict:
        """Get exact current time, date, day of week, and timezone."""
        import datetime
        now = datetime.datetime.now().astimezone()
        return {
            "current_time": now.strftime("%I:%M:%S %p"),
            "current_date": now.strftime("%A, %B %d, %Y"),
            "timezone": str(now.tzinfo),
            "iso": now.isoformat(),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="system.time",
            description="Get the current real-world time, date, day of week, and timezone",
            parameters={
                "type": "object",
                "properties": {
                    "timezone_str": {"type": "string", "description": "Optional timezone name (e.g. UTC, Asia/Kolkata)"},
                },
            },
            handler=get_current_time,
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

    # --- Image Generation Tools ---
    async def generate_image(prompt: str, width: int = 512, height: int = 512, steps: int = 4) -> dict:
        if not getattr(engine, "image_service", None):
            return {"error": "Image generation service is not initialized"}
        span = None
        if getattr(engine, "observability", None):
            span = engine.observability.start_span("IMAGE", "GENERATE", metadata={"prompt": prompt})

        job = engine.image_service.submit_job(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            trace_id=getattr(engine.observability, "current_trace_id", None) if getattr(engine, "observability", None) else None,
        )

        if span and getattr(engine, "observability", None):
            engine.observability.finish_span(span, metadata={"job_id": job.job_id})

        return {
            "status": "queued",
            "job_id": job.job_id,
            "message": f"Image generation started in background for '{prompt}'. Check dashboard or /images/jobs for status.",
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="image.generate",
            description="Generate a realistic or stylized image from a prompt asynchronously in the background",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Visual prompt describing the image to generate"},
                    "width": {"type": "integer", "description": "Image width in pixels (128-1024)", "default": 512},
                    "height": {"type": "integer", "description": "Image height in pixels (128-1024)", "default": 512},
                },
                "required": ["prompt"],
            },
            handler=generate_image,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="image",
        )
    )

    async def cancel_image(job_id: str) -> dict:
        if not getattr(engine, "image_service", None):
            return {"error": "Image generation service is not initialized"}
        ok = await engine.image_service.cancel_job(job_id)
        return {"cancelled": ok, "job_id": job_id}

    engine.tool_registry.register(
        ToolDefinition(
            name="image.cancel",
            description="Cancel an in-progress or queued background image generation task",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID returned by image.generate"},
                },
                "required": ["job_id"],
            },
            handler=cancel_image,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="image",
        )
    )
