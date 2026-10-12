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
    async def memory_write(fact: str = "", **kwargs) -> str:
        target_fact = (fact or kwargs.get("text") or kwargs.get("content") or kwargs.get("note") or kwargs.get("message") or "").strip()
        if not target_fact:
            return "No fact text provided to remember."
        pers = engine.personalities.active
        ns = getattr(pers, "memory_namespace", "default")
        await engine.memory.add_memory(target_fact, user_id=ns)
        return f"Successfully remembered: '{target_fact}' in {ns} workspace"

    engine.tool_registry.register(
        ToolDefinition(
            name="memory.write",
            description="Autonomously record an essential, durable user fact or long-term preference into permanent memory. Do NOT call this for casual conversational context or temporary chit-chat.",
            parameters={
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The exact enduring fact or preference to remember for future sessions"},
                },
                "required": ["fact"],
            },
            handler=memory_write,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="memory",
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def memory_all(limit: int = 20) -> list[dict]:
        pers = engine.personalities.active
        ns = getattr(pers, "memory_namespace", "default")
        return await engine.memory.get_all_facts(user_id=ns, limit=limit)

    engine.tool_registry.register(
        ToolDefinition(
            name="memory.all",
            description="Retrieve all durable facts, memories, preferences, and personal details saved about the user across past sessions. Call this ONLY when the user explicitly asks what you remember about them, asks to see their saved memories/facts, or asks for a profile summary. NEVER call this on greetings ('Hi', 'Hello', 'Hey') or general conversational chitchat.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of distinct facts to return", "default": 20},
                },
            },
            handler=memory_all,
            permission_level=PermissionLevel.READ_ONLY,
            family="memory",
        )
    )

    async def memory_search(query: str = "all", limit: int = 5) -> Any:
        pers = engine.personalities.active
        ns = getattr(pers, "memory_namespace", "default")
        clean_q = (query or "").strip().lower()
        if clean_q in {"", "all", "*", "everything", "facts", "memories", "about me", "me", "preferences", "durable facts"}:
            return await engine.memory.get_all_facts(user_id=ns, limit=limit)
        return await engine.memory.get_relevant_memories(query, user_id=ns, limit=limit)

    engine.tool_registry.register(
        ToolDefinition(
            name="memory.search",
            description="Search persistent memories for a specific topic, keyword, person, or preference. Call this ONLY when the user asks a question about a past topic or saved preference. NEVER call this on greetings, pleasantries, or general conversational dialogue.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific topic or keyword to search for, or 'all' to retrieve all saved facts and memories", "default": "all"},
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

    async def web_search(query: str = "", max_results: int = 3, deep: bool = False, **kwargs) -> list[dict]:
        """Perform an ultra-fast, extensive web search with multi-engine failover."""
        import asyncio
        import html as html_lib
        import json
        import re
        import time as _time
        import urllib.parse
        import urllib.request

        eff_q = (query or kwargs.get("q") or kwargs.get("search") or kwargs.get("term") or kwargs.get("topic") or kwargs.get("keyword") or "").strip()
        if not eff_q:
            return [{"error": "No search query provided."}]

        eff_max = 8 if deep else max_results

        # Clean search query: strip natural language conversational filler
        clean_q = eff_q
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

        async def _execute_parallel_search() -> list[dict]:
            import httpx
            results_list: list[dict] = []
            seen_titles = set()

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }

            async with httpx.AsyncClient(headers=headers, timeout=4.5, follow_redirects=True) as client:
                # 1. Wikipedia Summary & Search API
                async def _search_wiki():
                    try:
                        w_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_q)}"
                        resp = await client.get(w_url)
                        if resp.status_code == 200:
                            data = resp.json()
                            ext = data.get("extract", "").strip()
                            if ext and data.get("type") != "disambiguation":
                                return [{
                                    "title": f"{data.get('title', clean_q)} (Wikipedia)",
                                    "snippet": ext,
                                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                }]
                    except Exception:
                        pass
                    return []

                # 2. Bing News & Web RSS
                async def _search_bing():
                    try:
                        b_url = f"https://www.bing.com/news/search?q={urllib.parse.quote(clean_q)}&format=rss"
                        resp = await client.get(b_url)
                        if resp.status_code == 200:
                            items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
                            b_res = []
                            for it in items[:eff_max]:
                                tm = re.search(r"<title>(.*?)</title>", it, re.DOTALL)
                                lm = re.search(r"<link>(.*?)</link>", it, re.DOTALL)
                                dm = re.search(r"<description>(.*?)</description>", it, re.DOTALL)
                                t = html_lib.unescape(re.sub(r"<[^>]+>", "", tm.group(1))).strip() if tm else ""
                                d = html_lib.unescape(re.sub(r"<[^>]+>", "", dm.group(1))).strip() if dm else ""
                                u = lm.group(1).strip() if lm else ""
                                if t or d:
                                    b_res.append({"title": t, "snippet": d, "url": u})
                            return b_res
                    except Exception:
                        pass
                    return []

                # 3. DuckDuckGo Instant Answers API
                async def _search_ddg_api():
                    try:
                        api_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
                        resp = await client.get(api_url)
                        if resp.status_code == 200:
                            data = resp.json()
                            d_res = []
                            if data.get("AbstractText"):
                                d_res.append({
                                    "title": data.get("Heading", clean_q),
                                    "snippet": data.get("AbstractText"),
                                    "url": data.get("AbstractURL", ""),
                                })
                            if data.get("Answer"):
                                d_res.append({"title": "Direct Answer", "snippet": data.get("Answer"), "url": ""})
                            for rel in data.get("RelatedTopics", [])[:eff_max]:
                                if isinstance(rel, dict) and rel.get("Text"):
                                    d_res.append({
                                        "title": rel.get("Text").split(" - ")[0],
                                        "snippet": rel.get("Text"),
                                        "url": rel.get("FirstURL", ""),
                                    })
                            return d_res
                    except Exception:
                        pass
                    return []

                # 4. DuckDuckGo Lite HTML Search
                async def _search_ddg_lite():
                    try:
                        lite_url = "https://lite.duckduckgo.com/lite/"
                        data_post = {"q": clean_q}
                        resp = await client.post(lite_url, data=data_post, headers={"Content-Type": "application/x-www-form-urlencoded"})
                        if resp.status_code == 200 and "result-snippet" in resp.text:
                            link_pattern = re.compile(
                                r'<a[^>]+href=[\x27\"]([^\x27\"]+)[\x27\"][^>]*class=[\x27\"]result-link[\x27\"][^>]*>(.*?)</a>',
                                re.DOTALL,
                            )
                            snippet_pattern = re.compile(
                                r'<td class=[\x27\"]result-snippet[\x27\"]>(.*?)</td>',
                                re.DOTALL,
                            )
                            links = link_pattern.findall(resp.text)
                            snippets = snippet_pattern.findall(resp.text)
                            dl_res = []
                            for i in range(min(len(snippets), eff_max + 2)):
                                clean_snip = html_lib.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
                                u = "https://duckduckgo.com"
                                clean_title = clean_q
                                if i < len(links):
                                    u, t = links[i]
                                    clean_title = html_lib.unescape(re.sub(r"<[^>]+>", "", t)).strip()
                                if "ad_domain=" in u or "aclick" in u or "duckduckgo.com/y.js" in u:
                                    continue
                                if clean_snip:
                                    dl_res.append({"title": clean_title, "snippet": clean_snip, "url": u})
                            return dl_res
                    except Exception:
                        pass
                    return []

                engine_tasks = [_search_wiki(), _search_bing(), _search_ddg_api(), _search_ddg_lite()]
                gathered = await asyncio.gather(*engine_tasks, return_exceptions=True)

                for item_list in gathered:
                    if isinstance(item_list, list):
                        for item in item_list:
                            if isinstance(item, dict):
                                title_key = item.get("title", "").lower().strip()
                                if title_key and title_key not in seen_titles:
                                    seen_titles.add(title_key)
                                    results_list.append(item)
                                    if len(results_list) >= eff_max:
                                        break
                    if len(results_list) >= eff_max:
                        break

            if results_list:
                return results_list[:eff_max]
            return [{"error": f"No web search results could be retrieved for '{clean_q}'."}]

        results = await _execute_parallel_search()

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
                        async with httpx.AsyncClient(timeout=3.0, follow_redirects=True) as client:
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
            timeout_s=7.0,
            is_idempotent=True,
            max_retries=1,
        )
    )

    async def history_search(query: str = "", limit: int = 10, **kwargs) -> Any:
        eff_q = (query or kwargs.get("q") or kwargs.get("search") or kwargs.get("term") or "").strip()
        return await engine.memory.search_history(eff_q, limit)

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

    async def system_list_tools(category: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """List all active capabilities and tools available to the robot in structured format."""
        all_defs = engine.tool_registry.get_all() if hasattr(engine, "tool_registry") else []
        tools_by_family: dict[str, list[dict[str, str]]] = {}
        for td in all_defs:
            fam = td.family or "general"
            if category and category.lower() not in (fam.lower(), td.name.lower()):
                continue
            if fam not in tools_by_family:
                tools_by_family[fam] = []
            tools_by_family[fam].append({
                "name": td.name,
                "description": td.description,
            })
        return {
            "status": "success",
            "total_tools": sum(len(v) for v in tools_by_family.values()),
            "categories": list(tools_by_family.keys()),
            "tools_by_category": tools_by_family,
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="system.list_tools",
            description="List all available robot capabilities, functions, and tools in structured format. Use when user asks what tools you have or what you can do.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g. 'system', 'weather', 'media', 'memory', 'web', 'network')",
                    },
                },
            },
            handler=system_list_tools,
            permission_level=PermissionLevel.READ_ONLY,
            family="system",
        )
    )

    async def get_current_time(timezone_str: str | None = None, **kwargs) -> dict:
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

        eff_tz = timezone_str or kwargs.get("tz") or kwargs.get("timezone") or kwargs.get("location") or kwargs.get("country") or kwargs.get("city")
        if eff_tz:
            clean = eff_tz.strip().lower()
            if clean in country_map:
                target_tz = country_map[clean]
                target_country = eff_tz.strip().title()
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
                target_tz = tz_alias.get(clean, eff_tz.strip())
                target_country = eff_tz.strip().title()
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

    # --- Wikipedia Knowledge Tool ---
    async def wikipedia_lookup(query: str = "", **kwargs) -> dict:
        """Lookup encyclopedic summary and facts from Wikipedia."""
        import urllib.parse
        clean_q = (query or kwargs.get("topic") or kwargs.get("entity") or kwargs.get("term") or kwargs.get("title") or "").strip()
        if not clean_q:
            return {"found": False, "message": "No topic provided for Wikipedia lookup."}
        encoded = urllib.parse.quote(clean_q)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        headers = {"User-Agent": "AntigravityRobot/2.0 (open-source; educational-robotics)"}

        try:
            import requests

            def _fetch():
                return requests.get(url, headers=headers, timeout=4.0)

            resp = await asyncio.to_thread(_fetch)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "found": True,
                    "title": data.get("title", clean_q),
                    "description": data.get("description", ""),
                    "summary": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
            elif resp.status_code == 404:
                search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded}&limit=1&namespace=0&format=json"

                def _s_fetch():
                    return requests.get(search_url, headers=headers, timeout=4.0)

                s_resp = await asyncio.to_thread(_s_fetch)
                if s_resp.status_code == 200:
                    s_data = s_resp.json()
                    if len(s_data) >= 2 and s_data[1]:
                        alt_title = s_data[1][0]
                        alt_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(alt_title)}"
                        r2 = await asyncio.to_thread(lambda: requests.get(alt_url, headers=headers, timeout=4.0))
                        if r2.status_code == 200:
                            d2 = r2.json()
                            return {
                                "found": True,
                                "title": d2.get("title", alt_title),
                                "description": d2.get("description", ""),
                                "summary": d2.get("extract", ""),
                                "url": d2.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            }
        except Exception as e:
            return {"found": False, "error": str(e), "query": clean_q}
        return {"found": False, "message": f"No Wikipedia article found for '{clean_q}'"}

    engine.tool_registry.register(
        ToolDefinition(
            name="knowledge.wikipedia",
            description="Lookup encyclopedic knowledge, historical events, scientific concepts, or biographies on Wikipedia",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic, entity, person, or concept to look up on Wikipedia"},
                },
                "required": ["query"],
            },
            handler=wikipedia_lookup,
            permission_level=PermissionLevel.READ_ONLY,
            family="knowledge",
            timeout_s=6.0,
        )
    )

    # --- Math & Calculation Tool ---
    async def system_calculate(expression: str = "", **kwargs) -> dict:
        """Safely calculate mathematical and scientific expressions without hallucinating."""
        import math

        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
        }
        raw_expr = (expression or kwargs.get("expr") or kwargs.get("calc") or kwargs.get("math") or kwargs.get("formula") or kwargs.get("equation") or "").strip()
        if not raw_expr:
            return {"status": "error", "message": "No mathematical expression provided."}
        clean_expr = raw_expr.replace("^", "**").replace("×", "*").replace("÷", "/")
        try:
            code = compile(clean_expr, "<string>", "eval")
            for name in code.co_names:
                if name not in safe_dict:
                    return {"status": "error", "message": f"Disallowed function or symbol: '{name}'"}
            result = eval(code, {"__builtins__": {}}, safe_dict)
            return {"status": "success", "expression": raw_expr, "result": result}
        except Exception as e:
            return {"status": "error", "expression": raw_expr, "error": str(e)}

    engine.tool_registry.register(
        ToolDefinition(
            name="system.calculate",
            description="Calculate mathematical, trigonometric, and scientific expressions accurately without hallucination",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g. 'sqrt(144) + 15 * 3')"},
                },
                "required": ["expression"],
            },
            handler=system_calculate,
            permission_level=PermissionLevel.READ_ONLY,
            family="system",
        )
    )

    # --- Hardware Telemetry Tool ---
    async def system_hardware() -> dict:
        """Get live hardware performance telemetry (CPU, RAM, Disk, RTX 3050 GPU VRAM)."""
        import psutil
        import shutil

        cpu_pct = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        disk = shutil.disk_usage("/")

        hw_info: dict = {
            "cpu": {
                "utilization_percent": cpu_pct,
                "cores": psutil.cpu_count(logical=True),
            },
            "ram": {
                "total_gb": round(ram.total / (1024**3), 2),
                "used_gb": round(ram.used / (1024**3), 2),
                "free_gb": round(ram.available / (1024**3), 2),
                "percent": ram.percent,
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 1),
                "free_gb": round(disk.free / (1024**3), 1),
            },
            "gpu": {"name": "NVIDIA GeForce RTX 3050 Laptop GPU (4GB)"},
        }

        try:
            cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout:
                parts = [p.strip() for p in stdout.decode().split(",")]
                if len(parts) >= 5:
                    hw_info["gpu"].update({
                        "utilization_percent": float(parts[0]),
                        "vram_total_mb": float(parts[1]),
                        "vram_used_mb": float(parts[2]),
                        "vram_free_mb": float(parts[3]),
                        "temperature_c": float(parts[4]),
                    })
        except Exception:
            pass

        return hw_info

    engine.tool_registry.register(
        ToolDefinition(
            name="system.hardware",
            description="Retrieve real-time hardware status: CPU usage, RAM, RTX 3050 GPU utilization, VRAM, and temperatures",
            parameters={"type": "object", "properties": {}},
            handler=system_hardware,
            permission_level=PermissionLevel.READ_ONLY,
            family="system",
        )
    )

    # --- Music Tools (Spotify, SoundCloud, Local audio) ---
    music_service = MusicPlayerService.get_instance()

    async def music_play(query: str | None = None, source: str = "ytdlp", **kwargs) -> dict:
        raw_q = (query or kwargs.get("song") or kwargs.get("track") or kwargs.get("title") or
                 kwargs.get("song_name") or kwargs.get("track_name") or kwargs.get("name") or
                 kwargs.get("youtube") or kwargs.get("url") or kwargs.get("q") or kwargs.get("search") or
                 kwargs.get("search_query") or kwargs.get("topic") or kwargs.get("text") or kwargs.get("value") or
                 kwargs.get("input") or kwargs.get("prompt") or "").strip()

        # Handle composite artist + song
        artist = kwargs.get("artist")
        if artist and not raw_q:
            raw_q = str(artist).strip()
        elif artist and str(artist).lower() not in raw_q.lower():
            raw_q = f"{artist} {raw_q}".strip()

        # Fallback scan across all kwargs for any string query parameter
        if not raw_q:
            for k, v in kwargs.items():
                if isinstance(v, str) and v.strip() and k.lower() not in {"source", "src", "platform", "provider", "ytdlp", "spotify"}:
                    raw_q = v.strip()
                    break

        # Fallback if no query provided: check if paused or replay available
        if not raw_q:
            st = music_service.get_status()
            if st.get("is_paused"):
                return await asyncio.to_thread(music_service.resume, wait_speech_engine=engine)
            elif st.get("last_played"):
                return await asyncio.to_thread(music_service.replay, wait_speech_engine=engine)
            return {"status": "error", "message": "Please specify a song title, track, or artist to play."}

        clean_q = raw_q.lower()
        # Auto-delegate relative replay references to replay
        replay_queries = {
            "that song again", "that song", "it again", "play it again", "play that again",
            "the same song", "same song", "previous song", "last song", "replay", "again",
            "that song again on youtube", "it again on youtube",
        }
        if clean_q in replay_queries:
            return await asyncio.to_thread(music_service.replay, wait_speech_engine=engine)

        src_arg = source or kwargs.get("platform") or kwargs.get("src") or "ytdlp"
        clean_src = str(src_arg).strip().lower()
        if clean_src in {"auto", "default", ""}:
            if any(w in raw_q.lower() for w in ["local", "library", "offline", "stored"]):
                clean_src = "local"
            elif "soundcloud" in raw_q.lower():
                clean_src = "soundcloud"
            elif "spotify" in raw_q.lower():
                clean_src = "spotify"
            else:
                clean_src = "ytdlp"

        if clean_src in {"ytdlp", "yt-dlp", "youtube"}:
            return await asyncio.to_thread(music_service.play_ytdlp, raw_q, wait_speech_engine=engine)
        elif clean_src == "spotify":
            return await asyncio.to_thread(music_service.play_spotify, raw_q, wait_speech_engine=engine)
        elif clean_src == "soundcloud":
            return await asyncio.to_thread(music_service.play_soundcloud, raw_q, wait_speech_engine=engine)
        elif clean_src == "local":
            return await asyncio.to_thread(music_service.play_local, raw_q, wait_speech_engine=engine)
        return await asyncio.to_thread(music_service.play_ytdlp, raw_q, wait_speech_engine=engine)

    engine.tool_registry.register(
        ToolDefinition(
            name="music.play",
            description="Play songs, music tracks, or artists via YouTube (yt-dlp, default for all personalities) or local files. If user asks to replay or play the previous song again, invoke music.control(action='replay').",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Song name, artist, genre, or playlist to play"},
                    "source": {"type": "string", "description": "Playback source: 'ytdlp' (default YouTube full song stream), 'spotify', 'soundcloud', or 'local'", "default": "ytdlp"},
                },
                "required": ["query"],
            },
            handler=music_play,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="music",
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def music_control(
        action: str | None = None,
        volume: float | str | None = None,
        query: str | None = None,
        preset: str | None = None,
        **kwargs,
    ) -> dict:
        # 1. Parse and extract volume if supplied anywhere
        v_raw = volume if volume is not None else (kwargs.get("level") or kwargs.get("vol") or kwargs.get("v"))
        parsed_vol: float | None = None
        if v_raw is not None:
            try:
                if isinstance(v_raw, str) and "%" in v_raw:
                    parsed_vol = float(v_raw.replace("%", "").strip()) / 100.0
                else:
                    v_num = float(v_raw)
                    if v_num > 1.0:
                        parsed_vol = v_num / 100.0
                    else:
                        parsed_vol = v_num
                parsed_vol = max(0.0, min(1.0, parsed_vol))
            except Exception:
                pass

        # 2. Extract action with fallbacks to kwargs and parameter inference
        act = (action or kwargs.get("command") or kwargs.get("act") or kwargs.get("mode") or kwargs.get("operation") or "").strip().lower()
        preset_target = preset or kwargs.get("preset_name") or kwargs.get("profile") or kwargs.get("eq")

        if not act:
            if preset_target:
                act = "equalizer"
            elif query:
                act = "replay"
            elif parsed_vol is not None:
                act = "volume"
            else:
                act = "status"

        # 3. Apply volume adjustment if provided (even when composite, e.g. restart + 40% vol)
        vol_result = None
        if parsed_vol is not None:
            vol_result = music_service.set_volume(parsed_vol)

        # 4. Handle playback and control actions
        if act in {"pause", "hold", "freeze"}:
            return music_service.pause()
        elif act in {"resume", "unpause", "continue"}:
            res = music_service.resume(wait_speech_engine=engine)
            if vol_result:
                res["volume"] = parsed_vol
            return res
        elif act in {"replay", "restart", "repeat", "again", "play again"}:
            res = music_service.replay(query=query, wait_speech_engine=engine)
            if vol_result:
                res["volume"] = parsed_vol
                res["message"] = f"Set volume to {int(parsed_vol * 100)}% and {res.get('message', 'replaying track')}."
            return res
        elif act in {"stop", "cancel", "kill"}:
            return music_service.stop()
        elif act in {"status", "current", "what is playing", "what's playing"}:
            return music_service.get_status()
        elif act in {"library", "collection", "local", "files", "local_tracks", "my collection"}:
            tracks = music_service.list_local_tracks()
            return {"status": "ok", "local_collection": tracks, "total_songs": len(tracks)}
        elif act in {"history", "recent", "queue", "list"}:
            return {"history": music_service.get_history(limit=10)}
        elif act in {"volume", "vol", "level", "sound"}:
            if vol_result is not None:
                return vol_result
            return {"status": "error", "message": "Volume action requires volume level (e.g. 0.2 for 20%, 0.4 for 40%, 1.0 for 100%)"}
        elif act in {"equalizer", "eq", "preset", "profile", "sound_profile"}:
            p_to_set = preset_target or query
            return music_service.set_equalizer_preset(p_to_set)
        elif act in {"equalizers", "list_eq", "eq_list", "presets", "list_presets"}:
            return music_service.list_equalizer_presets()

        # Check if action itself is a preset name (e.g. action="robot", action="spatial")
        if act in music_service.PRESETS or act in {"bass", "apple", "3d", "concert"}:
            return music_service.set_equalizer_preset(act)

        if vol_result is not None:
            return vol_result

        return {"status": "error", "message": f"Unknown music action '{action}'"}

    engine.tool_registry.register(
        ToolDefinition(
            name="music.control",
            description="Control music playback, volume, and sound profiles: action='replay'/'restart' (optionally with volume, e.g. 0.4 for 40%), action='volume' (volume between 0.0 and 1.0), action='equalizer' (preset='robot' [signature preset], 'bass_boost', 'spatial', 'sub_bass', 'reverb', 'immersive', 'list'), action='resume', action='pause', action='stop', action='status', action='library', action='history'.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: 'replay', 'restart', 'volume', 'equalizer', 'resume', 'pause', 'stop', 'status', 'library', 'history'",
                        "default": "status",
                    },
                    "volume": {"type": "number", "description": "Volume level between 0.0 and 1.0 (e.g. 0.4 for 40%, 1.0 for 100%)"},
                    "preset": {"type": "string", "description": "Equalizer preset: 'robot', 'bass_boost', 'spatial', 'sub_bass', 'reverb', 'immersive', or 'list'"},
                    "query": {"type": "string", "description": "Optional song name or query"},
                },
            },
            handler=music_control,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="music",
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def music_equalizer(preset: str = "robot", **kwargs) -> dict:
        target = preset or kwargs.get("name") or kwargs.get("profile") or "robot"
        return music_service.set_equalizer_preset(target)

    engine.tool_registry.register(
        ToolDefinition(
            name="music.equalizer",
            description="Change audio equalizer profiles via EasyEffects: preset='robot' (user's saved preset, default), 'bass_boost' (Max Bass Pristine Reverb), 'spatial' (Apple Spatial Max Bass Vocals), 'sub_bass' (Sub Bass 75Hz Clear Vocals), 'reverb' (Pristine Reverb), 'immersive' (Immersive Sub 75Hz Clear Vocals). Pass preset='list' to inspect available sound profiles.",
            parameters={
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "description": "Preset name: 'robot', 'bass_boost', 'spatial', 'sub_bass', 'reverb', 'immersive', or 'list'",
                        "default": "robot",
                    },
                },
            },
            handler=music_equalizer,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="music",
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def music_history(limit: int = 10) -> list[dict]:
        return music_service.get_history(limit=limit)

    engine.tool_registry.register(
        ToolDefinition(
            name="music.history",
            description="Get the list of recently played songs, tracks, and artists",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max recent songs to return", "default": 10},
                },
            },
            handler=music_history,
            permission_level=PermissionLevel.READ_ONLY,
            family="music",
        )
    )

    async def music_status() -> dict:
        return music_service.get_status()

    engine.tool_registry.register(
        ToolDefinition(
            name="music.status",
            description="Get the current music player playback state (is_playing, current_track, source, volume, is_paused, last_played, history)",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=music_status,
            permission_level=PermissionLevel.READ_ONLY,
            family="music",
        )
    )

    # --- Personality Tools ---
    async def switch_personality(name: str = "", **kwargs) -> str:
        target_name = (name or kwargs.get("personality") or kwargs.get("character") or kwargs.get("mode") or kwargs.get("persona") or "").strip()
        if not target_name:
            return f"No personality specified. Available: {engine.personalities.list_personalities()}"
        if engine.personalities.set_active(target_name):
            raw_active = getattr(engine.personalities, "active_name", target_name)
            resolved = raw_active if isinstance(raw_active, str) and raw_active else str(target_name)
            engine.state.state.active_personality = resolved
            pers = engine.personalities.get(resolved)
            if pers and getattr(pers, "voice_model", None) and hasattr(engine.tts, "set_active_voice"):
                engine.tts.set_active_voice(pers.voice_model)
            if hasattr(engine, "events") and engine.events:
                await engine.events.emit("personality_switched", personality=resolved)
                await engine.events.emit("personality_warming", personality=resolved)
            if hasattr(engine, "warm_up_llm") and callable(getattr(engine, "warm_up_llm")):
                coro = engine.warm_up_llm(resolved)
                if asyncio.iscoroutine(coro):
                    asyncio.create_task(coro)
            disp = "Cypher" if resolved in ("neo", "cypher") else resolved.title()
            return f"Switched to {disp} ({target_name}) personality mode."
        available = engine.personalities.list_personalities()
        return f"Personality '{target_name}' not found. Available: {available}"

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
            is_idempotent=False,
            max_retries=0,
        )
    )

    # --- Perception & Camera Tools ---
    from robot.tools.builtins.perception import register_perception_tools
    register_perception_tools(engine)

    # --- Character & Motion Tools ---
    from robot.tools.builtins.motion import register_motion_tools
    register_motion_tools(engine)

    # --- Image Generation Tools ---
    async def generate_image(prompt: str = "", width: int = 512, height: int = 512, steps: int = 15, **kwargs) -> dict:
        if not getattr(engine, "image_service", None):
            return {"error": "Image generation service is not initialized"}
        raw_p = (prompt or kwargs.get("text") or kwargs.get("description") or kwargs.get("query") or kwargs.get("caption") or "").strip()
        if not raw_p:
            return {"error": "No visual prompt provided for image generation."}
        # Bound steps between 12 and 20 for optimal quality and responsiveness
        clamped_steps = max(12, min(20, steps)) if steps else 15
        span = None
        if getattr(engine, "observability", None):
            span = engine.observability.start_span("IMAGE", "GENERATE", metadata={"prompt": raw_p})

        job = engine.image_service.submit_job(
            prompt=raw_p,
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
            "message": f"Image generation started in background for '{raw_p}'. Check dashboard or /images/jobs for status.",
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
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def cancel_image(job_id: str = "", **kwargs) -> dict:
        if not getattr(engine, "image_service", None):
            return {"error": "Image generation service is not initialized"}
        jid = (job_id or kwargs.get("id") or kwargs.get("task_id") or "").strip()
        if not jid:
            return {"error": "No job ID provided to cancel."}
        ok = await engine.image_service.cancel_job(jid)
        return {"cancelled": ok, "job_id": jid}

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
            is_idempotent=False,
            max_retries=0,
        )
    )

    # --- Standby & Voice Control Tools ---
    async def stop_listening(**kwargs) -> str:
        """Put the robot into standby mode waiting for the wake word."""
        from robot.core.state import RobotMode
        from robot.core.events import Events

        if hasattr(engine, "state"):
            await engine.state.set_mode(RobotMode.STANDBY)
        if hasattr(engine, "events"):
            await engine.events.emit(Events.STANDBY, reason="user_command")
        if hasattr(engine, "coordinator") and engine.coordinator:
            engine.coordinator.reset()

        return "Entering standby mode. I will wait for you to say 'Hey Lisa' before listening again."

    engine.tool_registry.register(
        ToolDefinition(
            name="system.stop_listening",
            description="Put the robot into standby mode and pause active listening until the wake word ('Hey Lisa') is spoken. Call this when the user says 'stop listening', 'go to sleep', 'standby', 'pause mic', or tells the robot to stop paying attention.",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=stop_listening,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="system",
            is_idempotent=False,
            max_retries=0,
        )
    )

    # --- Temporal & Scheduling Tools ---
    from robot.tools.builtins.scheduler import register_scheduler_tools
    register_scheduler_tools(engine)

    # --- OS & Workspace Automation Tools ---
    from robot.tools.builtins.system_os import register_system_os_tools
    register_system_os_tools(engine)

    # --- Real-Time Weather, News & Financial Feeds ---
    from robot.tools.builtins.weather_news import register_weather_news_tools
    register_weather_news_tools(engine)

    # --- Home Assistant & IoT Smart Home Tools ---
    from robot.tools.builtins.smart_home import register_smart_home_tools
    register_smart_home_tools(engine)

    # --- Developer & Terminal Tools ---
    from robot.tools.builtins.developer_terminal import register_developer_terminal_tools
    register_developer_terminal_tools(engine)

    # --- Productivity & Knowledge Vault Tools ---
    from robot.tools.builtins.productivity_knowledge import register_productivity_knowledge_tools
    register_productivity_knowledge_tools(engine)

    # --- Extended System & Network Tools ---
    from robot.tools.builtins.system_extended import register_system_extended_tools
    register_system_extended_tools(engine)

    # --- Actuarial Death Clock & Longevity Estimation Tools ---
    from robot.tools.builtins.death_clock import register_death_clock_tools
    register_death_clock_tools(engine)



