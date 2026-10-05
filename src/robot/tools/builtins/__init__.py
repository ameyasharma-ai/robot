"""Built-in tools for memory, system status, and personality management."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot.tools import PermissionLevel, ToolDefinition
from robot.tools.builtins.music import MusicPlayerService

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


_tool_config_cache: dict[str, Any] | None = None


def get_tool_config() -> dict[str, Any]:
    global _tool_config_cache
    if _tool_config_cache is not None:
        return _tool_config_cache
    cfg_path = Path("config/tools/tools.yaml")
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                _tool_config_cache = yaml.safe_load(f) or {}
        except Exception:
            _tool_config_cache = {}
    else:
        _tool_config_cache = {}
    return _tool_config_cache


def register_builtin_tools(engine: "RobotEngine") -> None:
    """Register all built-in tools with the engine's tool registry."""

    # --- Memory Tools ---
    async def memory_write(fact: str) -> str:
        pers = engine.personalities.active
        ns = getattr(pers, "memory_namespace", "default")
        await engine.memory.add_memory(fact, user_id=ns)
        return f"Successfully remembered: '{fact}' in {ns} workspace"

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
        pers = engine.personalities.active
        ns = getattr(pers, "memory_namespace", "default")
        return await engine.memory.get_relevant_memories(query, user_id=ns, limit=limit)

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

    async def web_search(query: str, max_results: int = 3, deep: bool = False) -> list[dict]:
        """Perform an ultra-fast, extensive web search with multi-engine failover."""
        import asyncio
        import html as html_lib
        import json
        import re
        import time as _time
        import urllib.parse
        import urllib.request

        eff_max = 8 if deep else max_results

        # Clean search query: strip natural language conversational filler
        clean_q = query.strip()
        lower_q = clean_q.lower()
        for prefix in [
            "search deeply about", "search deeply for", "deep search for", "deep search",
            "search for", "search online for", "search the web for",
            "search about", "look up", "find online", "find out about",
            "find out", "google", "tell me about", "what is", "who is",
            "what happened in", "search"
        ]:
            if lower_q.startswith(prefix) and len(clean_q) > len(prefix) + 3:
                clean_q = clean_q[len(prefix):].strip(" ?:.,!")
                break

        cache_key = f"{clean_q.lower()}::{eff_max}::{deep}"
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
                    for i in range(min(len(snippets), max_results + 3)):
                        if len(results) >= max_results:
                            break
                        clean_snip = html_lib.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
                        u = "https://duckduckgo.com"
                        clean_title = clean_q
                        if i < len(links):
                            u, t = links[i]
                            clean_title = html_lib.unescape(re.sub(r"<[^>]+>", "", t)).strip()
                        # Skip sponsored advertising links
                        if "ad_domain=" in u or "aclick" in u or "duckduckgo.com/y.js" in u:
                            continue
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

        # Ultimate Web Mode for Gemini Cloud & Deep Search:
        # Concurrently fetch and extract comprehensive readable article/page text from top sources
        is_gemini = getattr(engine, "llm_provider_type", "local") == "gemini"
        if (is_gemini or deep) and results and isinstance(results, list) and "error" not in results[0]:
            try:
                import httpx
                from bs4 import BeautifulSoup

                async def _fetch_page_content(target_url: str) -> str:
                    if not target_url or not target_url.startswith("http") or "duckduckgo.com" in target_url:
                        return ""
                    try:
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                        }
                        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
                            resp = await client.get(target_url, headers=headers)
                            if resp.status_code == 200:
                                soup = BeautifulSoup(resp.text, "html.parser")
                                for bad in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "button", "form"]):
                                    bad.decompose()
                                node = soup.find("article") or soup.find("main") or soup.body
                                text = node.get_text(separator=" ", strip=True) if node else ""
                                import re as _re
                                clean = _re.sub(r"\s+", " ", text).strip()
                                return clean[:15000]
                    except Exception:
                        pass
                    return ""

                fetch_tasks = []
                for item in results[:5]:
                    if isinstance(item, dict) and item.get("url"):
                        fetch_tasks.append(_fetch_page_content(item["url"]))
                    else:
                        fetch_tasks.append(asyncio.sleep(0, result=""))

                page_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                for idx, content in enumerate(page_contents):
                    if isinstance(content, str) and content and idx < len(results):
                        results[idx]["content"] = content
            except Exception:
                pass

        if results and isinstance(results, list) and "error" not in results[0]:
            results.append({
                "meta": "search_guidance",
                "instruction": "Synthesize the facts and article contents deeply. State key findings clearly.",
            })
            _search_cache[cache_key] = (_time.time(), results)
        return results

    engine.tool_registry.register(
        ToolDefinition(
            name="web.search",
            description="Search the web for live facts, current events, products, or news. Supports deep search queries.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to look up on the web"},
                    "max_results": {"type": "integer", "description": "Maximum search results to return", "default": 3},
                    "deep": {"type": "boolean", "description": "Set to true to initiate an extensive deeper search", "default": False},
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
        """Get exact current time, date, day of week, and timezone.
        
        Defaults to Indian Standard Time (Asia/Kolkata), configurable via config/tools/tools.yaml.
        Supports looking up time in different countries, cities, or timezones.
        """
        import datetime
        import zoneinfo

        cfg = get_tool_config().get("system_time", {})
        default_tz_name = cfg.get("default_timezone", "Asia/Kolkata")
        default_country = cfg.get("default_country", "India")
        country_map = cfg.get("country_timezones", {})

        target_tz = None
        target_country = default_country
        tz_label = default_tz_name

        if timezone_str:
            clean = timezone_str.strip().lower()
            if clean in country_map:
                target_tz = country_map[clean]
                target_country = timezone_str.strip().title()
                tz_label = target_tz
            else:
                tz_alias = {
                    "india": "Asia/Kolkata", "ist": "Asia/Kolkata", "bharat": "Asia/Kolkata",
                    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "jst": "Asia/Tokyo",
                    "london": "Europe/London", "uk": "Europe/London", "gmt": "GMT", "utc": "UTC",
                    "new york": "America/New_York", "ny": "America/New_York", "est": "America/New_York",
                    "california": "America/Los_Angeles", "san francisco": "America/Los_Angeles", "pst": "America/Los_Angeles", "la": "America/Los_Angeles",
                    "paris": "Europe/Paris", "berlin": "Europe/Berlin", "sydney": "Australia/Sydney",
                }
                target_tz = tz_alias.get(clean, timezone_str.strip())
                target_country = timezone_str.strip().title()
                tz_label = target_tz
        else:
            target_tz = default_tz_name

        try:
            tz = zoneinfo.ZoneInfo(target_tz)
        except Exception:
            tz = zoneinfo.ZoneInfo(default_tz_name)
            tz_label = default_tz_name

        now = datetime.datetime.now(tz=tz)
        return {
            "current_time": now.strftime("%I:%M:%S %p"),
            "current_date": now.strftime("%A, %B %d, %Y"),
            "country": target_country,
            "timezone": tz_label,
            "iso": now.isoformat(),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="system.time",
            description="Get the current real-world time, date, day of week, and timezone. Defaults to Indian Standard Time (IST), configurable for different countries",
            parameters={
                "type": "object",
                "properties": {
                    "timezone_str": {"type": "string", "description": "Optional country, city, or timezone name (e.g. 'India', 'Japan', 'London', 'New York'). Defaults to India."},
                },
            },
            handler=get_current_time,
            permission_level=PermissionLevel.READ_ONLY,
            family="system",
        )
    )

    # --- Music Tools (Spotify, SoundCloud, Local audio) ---
    music_service = MusicPlayerService.get_instance()

    async def music_play(query: str, source: str = "auto") -> dict:
        clean_src = source.strip().lower()
        if clean_src == "auto":
            if any(w in query.lower() for w in ["local", "library", "offline", "stored"]):
                clean_src = "local"
            elif "soundcloud" in query.lower():
                clean_src = "soundcloud"
            elif "spotify" in query.lower():
                clean_src = "spotify"
            else:
                clean_src = music_service.config.get("default_source", "ytdlp")

        if clean_src in {"ytdlp", "yt-dlp", "youtube"}:
            return await asyncio.to_thread(music_service.play_ytdlp, query, wait_speech_engine=engine)
        elif clean_src == "spotify":
            return await asyncio.to_thread(music_service.play_spotify, query, wait_speech_engine=engine)
        elif clean_src == "soundcloud":
            return await asyncio.to_thread(music_service.play_soundcloud, query, wait_speech_engine=engine)
        elif clean_src == "local":
            return await asyncio.to_thread(music_service.play_local, query, wait_speech_engine=engine)
        return await asyncio.to_thread(music_service.play_ytdlp, query, wait_speech_engine=engine)

    engine.tool_registry.register(
        ToolDefinition(
            name="music.play",
            description="Play music tracks, songs, or artists on YouTube (yt-dlp), Spotify, SoundCloud, or from local library files",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Song name, artist, genre, or playlist to play"},
                    "source": {"type": "string", "description": "Playback source: 'ytdlp', 'spotify', 'soundcloud', 'local', or 'auto'", "default": "auto"},
                },
                "required": ["query"],
            },
            handler=music_play,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="music",
        )
    )

    async def music_control(action: str, volume: float | None = None) -> dict:
        act = action.strip().lower()
        if act == "pause":
            return music_service.pause()
        elif act == "resume":
            return music_service.resume()
        elif act == "stop":
            return music_service.stop()
        elif act == "volume" and volume is not None:
            return music_service.set_volume(volume)
        return {"status": "error", "message": f"Unknown music action '{action}'"}

    engine.tool_registry.register(
        ToolDefinition(
            name="music.control",
            description="Control active music playback (pause, resume, stop, set volume)",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action to perform: 'pause', 'resume', 'stop', 'volume'"},
                    "volume": {"type": "number", "description": "Optional volume between 0.0 and 1.0 (for action='volume')"},
                },
                "required": ["action"],
            },
            handler=music_control,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="music",
        )
    )

    # --- Personality Tools ---
    async def switch_personality(name: str) -> str:
        if engine.personalities.set_active(name):
            engine.state.state.active_personality = name
            pers = engine.personalities.get(name)
            if pers and getattr(pers, "voice_model", None) and hasattr(engine.tts, "set_active_voice"):
                engine.tts.set_active_voice(pers.voice_model)
            if hasattr(engine, "warm_up_llm") and callable(getattr(engine, "warm_up_llm")):
                coro = engine.warm_up_llm(name)
                if asyncio.iscoroutine(coro):
                    asyncio.create_task(coro)
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
    async def generate_image(prompt: str, width: int = 512, height: int = 512, steps: int = 15) -> dict:
        if not getattr(engine, "image_service", None):
            return {"error": "Image generation service is not initialized"}
        # Bound steps between 12 and 20 for optimal quality and responsiveness
        clamped_steps = max(12, min(20, steps)) if steps else 15
        span = None
        if getattr(engine, "observability", None):
            span = engine.observability.start_span("IMAGE", "GENERATE", metadata={"prompt": prompt})

        job = engine.image_service.submit_job(
            prompt=prompt,
            width=width,
            height=height,
            steps=clamped_steps,
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
                    "steps": {"type": "integer", "description": "Inference steps (recommended 12-20)", "default": 15},
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
