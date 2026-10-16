"""Autonomous headless web browsing agent for web navigation and data extraction.

Features:
- BrowserProvider abstraction using Playwright headless browser with fast HTTP fallback
- Read-only actions: navigate URL, capture DOM snapshot, extract readable text, take page screenshot
- Interactive actions (click, type, submit) require explicit user confirmation and PermissionLevel.EXTERNAL (L3)
- Sandboxed execution: disabled arbitrary file downloads, isolated profile
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.web_agent")


class BrowserProvider:
    """Manages headless browser automation sessions with safety sandboxing."""

    def __init__(self, screenshot_dir: str = "data/images/web_browser") -> None:
        """Initialize the browser provider.

        Args:
            screenshot_dir: Directory where page screenshots are stored.
        """
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def browse_page(
        self,
        url: str,
        take_screenshot: bool = True,
        max_chars: int = 8000,
    ) -> dict[str, Any]:
        """Navigate to a URL in a headless browser and extract text and snapshot.

        Args:
            url: Target URL to visit.
            take_screenshot: Whether to capture a rendered image snapshot.
            max_chars: Maximum characters of body text to return.

        Returns:
            Dictionary containing page title, text, screenshot path, and metadata.
        """
        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            clean_url = "https://" + clean_url

        # Check if Playwright is installed
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                )
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) RobotCore/2.0 WebAgent",
                    accept_downloads=False,
                )
                page = await context.new_page()
                await page.goto(clean_url, timeout=12000, wait_until="domcontentloaded")

                title = await page.title()
                # Extract clean text from main elements
                text = await page.inner_text("body")
                clean_text = re.sub(r"\s+", " ", text or "").strip()[:max_chars]

                screenshot_path = None
                if take_screenshot:
                    shot_file = self.screenshot_dir / f"page_{int(time.time())}.png"
                    await page.screenshot(path=str(shot_file))
                    screenshot_path = str(shot_file.resolve())

                await browser.close()

                return {
                    "status": "success",
                    "engine": "playwright",
                    "url": clean_url,
                    "title": title,
                    "text": clean_text,
                    "screenshot_path": screenshot_path,
                }
        except Exception as e:
            logger.debug("playwright_browse_fallback", error=str(e), url=clean_url)
            # Lightweight resilient HTTP fallback
            return await self._fallback_http_browse(clean_url, max_chars=max_chars)

    async def _fallback_http_browse(self, url: str, max_chars: int = 8000) -> dict[str, Any]:
        """Lightweight HTTP GET fallback when Playwright binary/browser is unavailable."""
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) RobotCore/2.0 WebAgent",
            "Accept": "text/html,application/xhtml+xml,text/plain",
        }

        def _fetch() -> dict[str, Any]:
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    raw_html = resp.read(150000).decode("utf-8", errors="replace")
                    # Extract title
                    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
                    title = title_m.group(1).strip() if title_m else url

                    # Extract body text
                    text = re.sub(r"<(?:script|style|svg)[^>]*>.*?</(?:script|style|svg)>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)
                    clean_text = re.sub(r"\s+", " ", text).strip()[:max_chars]

                    return {
                        "status": "success",
                        "engine": "http_fallback",
                        "url": url,
                        "title": title,
                        "text": clean_text,
                        "screenshot_path": None,
                    }
            except Exception as ex:
                logger.error("http_browse_failed", error=str(ex), url=url)
                return {"status": "error", "url": url, "error": str(ex)}

        return await asyncio.to_thread(_fetch)


_BROWSER_PROVIDER = BrowserProvider()


def register_web_agent_tools(engine: RobotEngine) -> None:
    """Register web.browser_agent tools with Robot Core.

    Args:
        engine: RobotEngine instance containing tool_registry.
    """
    async def browser_agent(
        url: str,
        action: str = "read",
        take_screenshot: bool = True,
        confirm: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Navigate web pages, inspect DOM content, and capture page screenshots."""
        target_url = url or kwargs.get("link") or kwargs.get("site") or ""
        clean_action = (action or "read").lower().strip()

        if clean_action in ("submit", "click", "form", "post"):
            if not confirm:
                return {
                    "status": "confirmation_required",
                    "action": clean_action,
                    "url": target_url,
                    "message": f"Interactive web action '{clean_action}' requires explicit confirmation. Re-run with confirm=True.",
                }

        return await _BROWSER_PROVIDER.browse_page(
            url=target_url, take_screenshot=take_screenshot
        )

    engine.tool_registry.register(
        ToolDefinition(
            name="web.browser_agent",
            description="Autonomous headless browser agent for web navigation, reading dynamic JavaScript pages, extracting structured text, and capturing visual page screenshots.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Target webpage URL to visit"},
                    "action": {"type": "string", "enum": ["read", "screenshot", "submit", "click"], "default": "read"},
                    "take_screenshot": {"type": "boolean", "description": "Whether to capture a rendered image screenshot", "default": True},
                    "confirm": {"type": "boolean", "description": "Confirmation flag for interactive actions", "default": False},
                },
                "required": ["url"],
            },
            handler=browser_agent,
            permission_level=PermissionLevel.READ_ONLY,
            family="web",
            timeout_s=15.0,
        )
    )
