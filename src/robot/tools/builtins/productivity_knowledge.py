"""Productivity & Knowledge Vault Built-in Tools.

Provides Wikipedia factual lookups, local Markdown notes vault management,
language translation, and web page content scraping.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.productivity_knowledge")


async def wikipedia_lookup(query: str, lang: str = "en") -> dict[str, Any]:
    """Search Wikipedia and retrieve clean factual summary, extract, and page link.

    Args:
        query: Topic, person, event, or concept to search for.
        lang: 2-letter language code (default 'en').

    Returns:
        dict containing title, description, extract, and url.
    """
    clean_q = (query or "").strip()
    if not clean_q:
        return {"status": "error", "error": "No query string provided for Wikipedia lookup."}

    def _fetch() -> dict[str, Any]:
        encoded_q = urllib.parse.quote(clean_q.replace(" ", "_"))
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_q}"
        req = urllib.request.Request(url, headers={"User-Agent": "RobotCore/1.0 (Autonomous Assistant)"})
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "status": "success",
                    "title": data.get("title", clean_q),
                    "description": data.get("description", ""),
                    "extract": data.get("extract", "No extract available."),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://{lang}.wikipedia.org/wiki/{encoded_q}"),
                }
        except Exception:
            # Fallback search API
            search_url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&format=json"
            req_s = urllib.request.Request(search_url, headers={"User-Agent": "RobotCore/1.0"})
            with urllib.request.urlopen(req_s, timeout=6) as resp_s:
                s_data = json.loads(resp_s.read().decode("utf-8"))
                results = s_data.get("query", {}).get("search", [])
                if not results:
                    return {"status": "not_found", "message": f"No Wikipedia article found for '{clean_q}'."}
                first = results[0]
                clean_snippet = re.sub(r"<[^>]+>", "", first.get("snippet", ""))
                title = first.get("title", clean_q)
                page_link = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                return {
                    "status": "success",
                    "title": title,
                    "description": "Wikipedia Search Result",
                    "extract": clean_snippet,
                    "url": page_link,
                }

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        logger.error("wikipedia_lookup_failed", query=clean_q, error=str(e))
        return {"status": "error", "error": str(e), "query": clean_q}


class NotesVault:
    """Local Markdown Knowledge Vault (`data/notes/`)."""

    def __init__(self, vault_dir: str = "data/notes") -> None:
        self.vault_path = Path(vault_dir)
        self.vault_path.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, title: str) -> str:
        safe_name = re.sub(r"[^\w\s-]", "", title.strip()).strip().replace(" ", "_").lower()
        return safe_name or "note"

    def create_note(self, title: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        """Create a new Markdown note in the vault."""
        if not title or not title.strip():
            return {"status": "error", "error": "Note title is required."}
        fname = self._safe_filename(title) + ".md"
        fpath = self.vault_path / fname
        tag_header = f"Tags: {', '.join(tags)}\n\n" if tags else ""
        full_text = f"# {title.strip()}\n{tag_header}{content.strip()}\n"
        fpath.write_text(full_text, encoding="utf-8")
        return {"status": "success", "action": "create", "file": fname, "title": title.strip()}

    def append_note(self, title: str, content: str) -> dict[str, Any]:
        """Append content to an existing Markdown note."""
        if not title or not title.strip():
            return {"status": "error", "error": "Note title is required to append content."}
        fname = self._safe_filename(title) + ".md"
        fpath = self.vault_path / fname
        if not fpath.exists():
            return self.create_note(title, content)
        existing = fpath.read_text(encoding="utf-8")
        fpath.write_text(f"{existing}\n\n{content.strip()}\n", encoding="utf-8")
        return {"status": "success", "action": "append", "file": fname}

    def list_notes(self) -> dict[str, Any]:
        """List all Markdown notes stored in the vault."""
        files = list(self.vault_path.glob("*.md"))
        notes_list = []
        for f in files:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            t = lines[0].replace("#", "").strip() if lines else f.stem
            notes_list.append({"file": f.name, "title": t, "size_bytes": f.stat().st_size})
        return {"status": "success", "action": "list", "notes": notes_list, "count": len(notes_list), "total": len(notes_list)}

    def read_note(self, title: str) -> dict[str, Any]:
        """Read full markdown content of a note."""
        if not title or not title.strip():
            return {"status": "error", "error": "Title or filename is required."}
        fname = self._safe_filename(title) + ".md"
        fpath = self.vault_path / fname
        if not fpath.exists():
            fpath = self.vault_path / title.strip()
        if not fpath.exists():
            return {"status": "error", "error": f"Note '{title}' not found in vault."}
        return {"status": "success", "action": "read", "title": title, "content": fpath.read_text(encoding="utf-8")}

    def search(self, query: str) -> dict[str, Any]:
        """Search notes vault content for query term."""
        q_clean = (query or "").strip().lower()
        files = list(self.vault_path.glob("*.md"))
        matches = []
        for f in files:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            if not q_clean or q_clean in txt.lower() or q_clean in f.name.lower():
                matches.append({"file": f.name, "snippet": txt[:300]})
        return {"status": "success", "action": "search", "query": q_clean, "matches": matches[:10], "match_count": len(matches[:10])}

    async def manage_notes(
        self,
        action: str = "search",
        title: str = "",
        content: str = "",
        query: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform operations on the local Markdown notes vault.

        Actions: 'create', 'append', 'search', 'list', 'read', 'delete'.
        """
        act = (action or "search").strip().lower()

        def _execute() -> dict[str, Any]:
            if act in ("create", "write"):
                return self.create_note(title, content, tags)
            elif act == "append":
                return self.append_note(title, content)
            elif act in ("list", "all"):
                return self.list_notes()
            elif act in ("read", "get"):
                return self.read_note(title)
            elif act in ("search", "query"):
                return self.search(query or title)
            return {"status": "error", "error": f"Unknown notes action '{action}'."}

        return await asyncio.to_thread(_execute)


async def translation_translate(
    text: str,
    target_lang: str = "en",
    source_lang: str = "auto",
) -> dict[str, Any]:
    """Translate text between languages with clean local or HTTP fallback.

    Args:
        text: Input string to translate.
        target_lang: Destination language code (e.g. 'en', 'es', 'fr', 'de', 'hi', 'ja').
        source_lang: Source language code (default 'auto').

    Returns:
        dict containing original text, translated text, target_lang, and status.
    """
    clean_t = (text or "").strip()
    if not clean_t:
        return {"status": "error", "error": "No text provided for translation."}

    def _translate() -> dict[str, Any]:
        # Google Translate free endpoint fallback
        encoded_txt = urllib.parse.quote(clean_t)
        url = (
            f"https://translate.googleapis.com/translate_a/single?"
            f"client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={encoded_txt}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "RobotCore/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                raw_res = json.loads(resp.read().decode("utf-8"))
                translated_parts = []
                if raw_res and isinstance(raw_res, list) and raw_res[0]:
                    for part in raw_res[0]:
                        if part and isinstance(part, list) and part[0]:
                            translated_parts.append(part[0])
                result_text = "".join(translated_parts) if translated_parts else clean_t
                return {
                    "status": "success",
                    "original_text": clean_t,
                    "translated_text": result_text,
                    "target_lang": target_lang,
                }
        except Exception as e:
            return {
                "status": "partial",
                "original_text": clean_t,
                "translated_text": clean_t,
                "target_lang": target_lang,
                "note": f"Translation fallback active: {e}",
            }

    return await asyncio.to_thread(_translate)


async def web_scrape_page(url: str, max_chars: int = 4000) -> dict[str, Any]:
    """Fetch and clean HTML content from a URL into readable plain text for LLM ingestion.

    Args:
        url: Target HTTP/HTTPS URL.
        max_chars: Maximum characters of body text to extract (default 4000).

    Returns:
        dict containing page title, clean body text extract, and status.
    """
    clean_url = (url or "").strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    def _scrape() -> dict[str, Any]:
        req = urllib.request.Request(
            clean_url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract title
        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        page_title = title_m.group(1).strip() if title_m else clean_url

        # Strip scripts, styles, navs, headers, footers
        html = re.sub(r"<(?:script|style|svg|nav|header|footer|iframe)[^>]*>.*?</(?:script|style|svg|nav|header|footer|iframe)>", "", html, flags=re.IGNORECASE | re.DOTALL)
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Unescape HTML entities
        import html as html_lib
        text = html_lib.unescape(text)
        # Collapse whitespace
        clean_text = re.sub(r"\s+", " ", text).strip()

        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + " ... [Content Truncated]"

        return {
            "status": "success",
            "url": clean_url,
            "title": page_title,
            "text": clean_text,
            "length_chars": len(clean_text),
        }

    try:
        return await asyncio.to_thread(_scrape)
    except Exception as e:
        logger.error("web_scrape_failed", url=clean_url, error=str(e))
        return {"status": "error", "error": str(e), "url": clean_url}


def register_productivity_knowledge_tools(engine: RobotEngine) -> None:
    """Register Wikipedia, Notes Vault, Translation, and Web Scraper tools with engine.

    Args:
        engine: RobotEngine instance containing tool_registry.
    """

    async def _wiki_lookup(query: str = "", lang: str = "en", **kwargs: Any) -> dict[str, Any]:
        """Search Wikipedia and retrieve clean summary.

        Args:
            query: Search query or article topic.
            lang: Language code.

        Returns:
            dict containing Wikipedia summary and details.
        """
        q = (query or kwargs.get("topic") or kwargs.get("term") or kwargs.get("entity") or "").strip()
        if not q:
            return {"status": "error", "error": "No query provided for Wikipedia lookup."}
        return await wikipedia_lookup(q, lang=lang)

    engine.tool_registry.register(
        ToolDefinition(
            name="wikipedia.lookup",
            description="Search Wikipedia and extract clean, factual reference summaries and links.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic, entity, or search term to look up on Wikipedia"},
                    "lang": {"type": "string", "description": "Language code (default 'en')", "default": "en"},
                },
                "required": ["query"],
            },
            handler=_wiki_lookup,
            permission_level=PermissionLevel.READ_ONLY,
            family="wikipedia",
            timeout_s=8.0,
        )
    )

    notes_vault = NotesVault()

    async def _notes_vault_handler(
        action: str = "search",
        title: str = "",
        content: str = "",
        tags: list[str] | None = None,
        query: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Manage Markdown notes vault.

        Args:
            action: Action ('create', 'append', 'search', 'list', 'read').
            title: Note title.
            content: Note body content.
            tags: List of tag strings.
            query: Search query term.

        Returns:
            dict containing notes operation result.
        """
        act = (action or kwargs.get("op") or "search").strip()
        t = (title or kwargs.get("note_title") or "").strip()
        c = (content or kwargs.get("body") or kwargs.get("text") or "").strip()
        tg = tags or kwargs.get("tag_list")
        q = (query or kwargs.get("q") or kwargs.get("term") or "").strip()

        if act in ("search", "vault_search"):
            return notes_vault.search(query=q or t)
        elif act == "create":
            return notes_vault.create_note(title=t, content=c, tags=tg)
        elif act == "append":
            return notes_vault.append_note(title=t, content=c)
        elif act in ("list", "all"):
            return notes_vault.list_notes()
        elif act == "read":
            return notes_vault.read_note(title=t)
        else:
            return {"status": "error", "error": f"Unknown notes action '{action}'"}

    engine.tool_registry.register(
        ToolDefinition(
            name="notes.vault_search",
            description="Create, append, tag, and search markdown notes in a persistent local knowledge vault.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Notes operation: 'create', 'append', 'search', 'list', 'read'", "default": "search"},
                    "title": {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Note markdown content body"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags associated with note"},
                    "query": {"type": "string", "description": "Search query when action='search'"},
                },
            },
            handler=_notes_vault_handler,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="notes",
        )
    )

    async def _translate(text: str = "", target_lang: str = "en", source_lang: str = "auto", **kwargs: Any) -> dict[str, Any]:
        """Translate text offline or via fast translation service.

        Args:
            text: Text snippet to translate.
            target_lang: Destination language code.
            source_lang: Source language code.

        Returns:
            dict containing original and translated text.
        """
        t = (text or kwargs.get("input") or kwargs.get("content") or "").strip()
        tl = (target_lang or kwargs.get("to") or "en").strip()
        sl = (source_lang or kwargs.get("from") or "auto").strip()
        return await translation_translate(text=t, target_lang=tl, source_lang=sl)

    engine.tool_registry.register(
        ToolDefinition(
            name="translation.translate",
            description="Perform fast text translation between 30+ languages.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_lang": {"type": "string", "description": "Destination language code (e.g. 'en', 'es', 'fr', 'hi', 'ja')", "default": "en"},
                    "source_lang": {"type": "string", "description": "Source language code (default 'auto')", "default": "auto"},
                },
                "required": ["text"],
            },
            handler=_translate,
            permission_level=PermissionLevel.READ_ONLY,
            family="translation",
            timeout_s=8.0,
        )
    )

    async def _scrape(url: str = "", max_chars: int = 4000, **kwargs: Any) -> dict[str, Any]:
        """Fetch and clean HTML content from URL into plain text.

        Args:
            url: Target URL.
            max_chars: Maximum characters to extract.

        Returns:
            dict containing clean body text extract.
        """
        u = (url or kwargs.get("link") or kwargs.get("web_url") or "").strip()
        if not u:
            return {"status": "error", "error": "No URL provided for page scraping."}
        return await web_scrape_page(url=u, max_chars=max_chars)

    engine.tool_registry.register(
        ToolDefinition(
            name="web.scrape_page",
            description="Fetch and extract main article text from any URL for deep summaries.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Target HTTP/HTTPS web URL to scrape"},
                    "max_chars": {"type": "integer", "description": "Maximum characters of text to extract", "default": 4000},
                },
                "required": ["url"],
            },
            handler=_scrape,
            permission_level=PermissionLevel.READ_ONLY,
            family="web",
            timeout_s=12.0,
        )
    )

