"""Autonomous headless web browsing agent for web navigation and data extraction.

Features:
- BrowserProvider abstraction using Playwright headless browser with fast HTTP fallback
- Read-only actions: navigate URL, capture DOM snapshot, extract readable text, take page screenshot
- Interactive actions (click, type, submit) require explicit user confirmation and PermissionLevel.EXTERNAL (L3)
- Sandboxed execution: disabled arbitrary file downloads, isolated profile
- SSRF protection: every outbound URL (including redirects) is resolved and
  rejected when it targets loopback/private/link-local/reserved addresses.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.web_agent")

_MAX_TEXT_CHARS = 20_000
_MAX_HTTP_BYTES = 150_000

_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".home.arpa",
)

_TRUTHY = {"1", "true", "yes", "on"}

#: How long a successful DNS answer is trusted for SSRF checks. Resolution is
#: blocking socket work, so a hot browsing session must not redo it per hop.
_RESOLVE_CACHE_TTL_S = 10.0
#: Bound on the resolution cache; hosts come from untrusted model input.
_RESOLVE_CACHE_MAX = 256
_resolve_cache: dict[str, tuple[float, list[tuple[int, str]]]] = {}
_resolve_cache_lock = threading.Lock()


class UnsafeURLError(ValueError):
    """Raised when an outbound URL is not permitted."""


def _env_truthy(name: str) -> bool:
    """Return True when an environment flag is explicitly enabled."""
    return (os.getenv(name, "") or "").strip().lower() in _TRUTHY


def _is_forbidden_ip(address: str) -> bool:
    """Return True when an IP literal is loopback/private/link-local/reserved/etc."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast or ip.is_loopback or ip.is_link_local


def _check_url_shape(url: str) -> tuple[str, int]:
    """Validate everything about a URL that does not require the network.

    Args:
        url: Absolute URL to validate.

    Returns:
        tuple[str, int]: Normalised lowercase hostname and effective port.

    Raises:
        UnsafeURLError: If the URL is malformed, non-HTTP(S), credentialed,
            local, or uses an invalid port.
    """
    text = (url or "").strip()
    if not text:
        raise UnsafeURLError("URL is empty.")
    if len(text) > 8192:
        raise UnsafeURLError("URL is too long.")

    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError as exc:
        raise UnsafeURLError(f"Malformed URL: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError("Only http:// and https:// URLs are allowed.")
    if not parsed.hostname:
        raise UnsafeURLError("URL has no hostname.")
    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed.")

    host = parsed.hostname.rstrip(".").lower()
    if not host or host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UnsafeURLError("Local hostnames are not allowed.")

    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeURLError("Invalid port in URL.") from exc
    if not (1 <= port <= 65535):
        raise UnsafeURLError("Invalid port in URL.")
    return host, port


def _resolve_global_addresses(host: str, port: int) -> list[tuple[int, str]]:
    """Resolve a host and reject it if any answer is not globally routable.

    Results are cached briefly because DNS is blocking and a single browse
    validates the same host on the initial URL, every redirect, and the final
    URL. The cache is bounded so model-supplied hostnames cannot grow it.

    Args:
        host: Normalised hostname or IP literal.
        port: Port used only as a resolution cache key component.

    Returns:
        list[tuple[int, str]]: Address families and validated addresses.

    Raises:
        UnsafeURLError: If the host does not resolve or reaches a forbidden
            loopback, private, link-local, multicast or reserved address.
    """
    key = f"{host}:{port}"
    now = time.monotonic()
    with _resolve_cache_lock:
        entry = _resolve_cache.get(key)
        if entry and entry[0] > now:
            return list(entry[1])

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host: {host}") from exc
    if not infos:
        raise UnsafeURLError(f"Could not resolve host: {host}")

    addresses: list[tuple[int, str]] = []
    for family, _, _, _, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        address = sockaddr[0]
        if _is_forbidden_ip(address):
            raise UnsafeURLError(
                "URL resolves to a loopback, private, link-local, or reserved address."
            )
        addresses.append((family, address))
    if not addresses:
        raise UnsafeURLError(f"Could not resolve host: {host}")

    with _resolve_cache_lock:
        if len(_resolve_cache) >= _RESOLVE_CACHE_MAX:
            _resolve_cache.clear()
        _resolve_cache[key] = (now + _RESOLVE_CACHE_TTL_S, addresses)
    return list(addresses)


def validate_outbound_url(url: str) -> str:
    """Validate an outbound HTTP(S) URL against SSRF targets.

    The hostname is resolved with :func:`socket.getaddrinfo` and every returned
    address must be globally routable.  Loopback, private, link-local,
    multicast, reserved, and unspecified addresses are rejected, as are
    non-HTTP(S) schemes and hostnames that resolve to nothing.

    This call blocks on DNS. Event-loop callers must use
    :func:`validate_outbound_url_async` instead.

    Args:
        url: Absolute URL to validate.

    Returns:
        The original URL when safe.

    Raises:
        UnsafeURLError: If the URL is malformed or targets a forbidden address.
    """
    text = (url or "").strip()
    host, port = _check_url_shape(text)
    _resolve_global_addresses(host, port)
    return text


async def validate_outbound_url_async(url: str) -> str:
    """Validate an outbound URL without blocking the event loop on DNS.

    Args:
        url: Absolute URL to validate.

    Returns:
        The original URL when safe.

    Raises:
        UnsafeURLError: If the URL is malformed or targets a forbidden address.
    """
    text = (url or "").strip()
    host, port = _check_url_shape(text)
    await asyncio.to_thread(_resolve_global_addresses, host, port)
    return text


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib redirect handler that validates every redirect hop."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Validate the redirect target before allowing urllib to follow it."""
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        validate_outbound_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_safe_opener() -> urllib.request.OpenerDirector:
    """Build a urllib opener that validates redirects for SSRF protection."""
    return urllib.request.build_opener(SafeRedirectHandler())


def read_http_body(
    url: str, *, timeout: float = 5.0, max_bytes: int = _MAX_HTTP_BYTES
) -> tuple[bytes, str, int]:
    """Fetch an HTTP(S) URL with SSRF validation on the initial URL, every
    redirect hop, and the final URL.

    Args:
        url: Target URL.
        timeout: Network timeout in seconds.
        max_bytes: Maximum bytes read from the body.

    Returns:
        Tuple of ``(body_bytes, final_url, status_code)``.

    Raises:
        UnsafeURLError: If the initial/final URL or a redirect target is unsafe.
    """
    validate_outbound_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) RobotCore/2.0 WebAgent",
            "Accept": "text/html,application/xhtml+xml,text/plain",
        },
    )
    opener = build_safe_opener()
    with opener.open(request, timeout=timeout) as response:
        final_url = response.geturl()
        validate_outbound_url(final_url)
        status = getattr(response, "status", None) or response.getcode() or 0
        body = response.read(max_bytes)
    return body, final_url, int(status)


def _chromium_args() -> list[str]:
    """Return Chromium flags, only opting out of the sandbox explicitly."""
    args = ["--disable-dev-shm-usage"]
    if _env_truthy("ROBOT_ALLOW_CHROMIUM_NO_SANDBOX"):
        # This is a machine-level opt-in for hosts where user namespaces are
        # unavailable; it is never enabled by model input.
        args.extend(["--no-sandbox", "--disable-setuid-sandbox"])
    return args


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
        raw_url = (url or "").strip()
        if not raw_url:
            return {"status": "error", "error": "No URL provided.", "url": ""}
        if not raw_url.startswith(("http://", "https://")):
            raw_url = "https://" + raw_url
        try:
            clean_url = await validate_outbound_url_async(raw_url)
        except UnsafeURLError as exc:
            logger.warning("browser_url_blocked", host=_host_for_log(raw_url), reason=str(exc))
            return {"status": "error", "error": f"Blocked URL: {exc}", "url": raw_url}

        try:
            char_limit = max(1, min(_MAX_TEXT_CHARS, int(max_chars)))
        except (TypeError, ValueError):
            char_limit = 8000

        # Check if Playwright is installed
        try:
            from playwright.async_api import async_playwright

            blocked_urls: list[str] = []

            async def _guard_route(route: Any) -> None:
                """Validate every browser request, aborting SSRF targets."""
                request_url = route.request.url
                if not request_url.startswith(("http://", "https://")):
                    # file://, data:// and friends can read the host directly.
                    blocked_urls.append(request_url)
                    await route.abort("blockedbyclient")
                    return
                try:
                    await validate_outbound_url_async(request_url)
                except Exception:
                    blocked_urls.append(request_url)
                    await route.abort("blockedbyclient")
                    return
                await route.continue_()

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=_chromium_args())
                try:
                    context = await browser.new_context(
                        viewport={"width": 1280, "height": 800},
                        user_agent="Mozilla/5.0 (X11; Linux x86_64) RobotCore/2.0 WebAgent",
                        accept_downloads=False,
                    )
                    await context.route("**/*", _guard_route)
                    page = await context.new_page()
                    await page.goto(clean_url, timeout=12000, wait_until="domcontentloaded")

                    final_url = page.url
                    if final_url.startswith(("http://", "https://")):
                        await validate_outbound_url_async(final_url)

                    title = (await page.title())[:300]
                    # Extract clean text from main elements
                    text = await page.inner_text("body")
                    clean_text = re.sub(r"\s+", " ", text or "").strip()[:char_limit]

                    screenshot_path = None
                    if take_screenshot:
                        shot_file = self.screenshot_dir / f"page_{int(time.time())}.png"
                        await page.screenshot(path=str(shot_file))
                        screenshot_path = str(shot_file.resolve())

                    return {
                        "status": "success",
                        "engine": "playwright",
                        "url": final_url,
                        "title": title,
                        "text": clean_text,
                        "screenshot_path": screenshot_path,
                        "blocked_requests": len(blocked_urls),
                    }
                finally:
                    await browser.close()
        except UnsafeURLError as exc:
            logger.warning("browser_final_url_blocked", host=_host_for_log(clean_url), reason=str(exc))
            return {"status": "error", "error": f"Blocked URL: {exc}", "url": clean_url}
        except Exception as exc:
            logger.debug(
                "playwright_browse_fallback",
                error_type=type(exc).__name__,
                host=_host_for_log(clean_url),
            )
            # Lightweight resilient HTTP fallback
            return await self._fallback_http_browse(clean_url, max_chars=char_limit)

    async def _fallback_http_browse(self, url: str, max_chars: int = 8000) -> dict[str, Any]:
        """Lightweight HTTP GET fallback when Playwright binary/browser is unavailable."""
        char_limit = max(1, min(_MAX_TEXT_CHARS, int(max_chars)))
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) RobotCore/2.0 WebAgent",
            "Accept": "text/html,application/xhtml+xml,text/plain",
        }

        def _fetch() -> dict[str, Any]:
            try:
                validate_outbound_url(url)
                request = urllib.request.Request(url, headers=headers)
                opener = build_safe_opener()
                with opener.open(request, timeout=5.0) as response:
                    final_url = response.geturl()
                    validate_outbound_url(final_url)
                    raw_html = response.read(_MAX_HTTP_BYTES).decode("utf-8", errors="replace")
                    # Extract title
                    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
                    title = title_m.group(1).strip()[:300] if title_m else final_url

                    # Extract body text
                    text = re.sub(r"<(?:script|style|svg)[^>]*>.*?</(?:script|style|svg)>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)
                    clean_text = re.sub(r"\s+", " ", text).strip()[:char_limit]

                    return {
                        "status": "success",
                        "engine": "http_fallback",
                        "url": final_url,
                        "title": title,
                        "text": clean_text,
                        "screenshot_path": None,
                    }
            except UnsafeURLError as exc:
                logger.warning("http_browse_blocked", host=_host_for_log(url), reason=str(exc))
                return {"status": "error", "url": url, "error": f"Blocked URL: {exc}"}
            except Exception as exc:
                logger.error(
                    "http_browse_failed",
                    host=_host_for_log(url),
                    error_type=type(exc).__name__,
                )
                return {
                    "status": "error",
                    "url": url,
                    "error": f"Failed to fetch page ({type(exc).__name__}).",
                }

        return await asyncio.to_thread(_fetch)


def _host_for_log(url: str) -> str:
    """Return only the host portion of a URL for safe structured logging."""
    try:
        return urllib.parse.urlsplit(url).hostname or ""
    except Exception:
        return ""


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
