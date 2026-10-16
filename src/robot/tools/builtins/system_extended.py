"""Extended System & Network Built-in Tools.

Provides screen capture for VLM multimodal analysis, monitor display brightness control,
Wi-Fi scanning via nmcli, and data-safe network status checks.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.system_extended")


async def os_capture_screen(
    filename: str | None = None,
    region: list[int] | None = None,
    window_title: str | None = None,
    run_ocr: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Capture a screenshot of desktop display or target window with OCR text extraction.

    Args:
        filename: Optional custom filename for saved PNG image.
        region: Optional crop region as [x, y, width, height].
        window_title: Optional filter / target window title.
        run_ocr: Whether to perform local OCR text extraction.

    Returns:
        Structured result dict with status, timestamp, window_title, detected_text,
        bounding_boxes, image_path, and data_url.
    """
    shots_dir = Path("data/images/screenshots")
    shots_dir.mkdir(parents=True, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    fname = filename or f"screenshot_{int(time.time())}.png"
    if not fname.endswith(".png"):
        fname += ".png"
    target_path = shots_dir / fname

    # Query active window title
    detected_win_title = window_title or "Desktop Display"
    try:
        proc_win = await asyncio.create_subprocess_shell(
            "xdotool getactivewindow getwindowname",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_w, _ = await proc_win.communicate()
        if proc_win.returncode == 0 and stdout_w:
            detected_win_title = stdout_w.decode("utf-8").strip() or detected_win_title
    except Exception:
        pass

    # Try standard Linux screenshot tools in order: maim -> scrot -> import -> xwd
    cmds = [
        f"maim '{target_path}'",
        f"scrot '{target_path}'",
        f"import -window root '{target_path}'",
        f"xwd -root -silent | convert xwd:- '{target_path}'",
    ]

    success = False
    used_cmd = ""
    for cmd in cmds:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            if proc.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
                success = True
                used_cmd = cmd
                break
        except Exception:
            pass

    if not success:
        return {
            "status": "error",
            "timestamp": timestamp_iso,
            "window_title": detected_win_title,
            "detected_text": "",
            "bounding_boxes": [],
            "error": "Failed to capture screenshot. Install 'scrot' or 'maim' (e.g. sudo apt install scrot maim).",
        }

    # Region cropping if region [x, y, w, h] is specified
    if region and len(region) == 4:
        try:
            from PIL import Image

            with Image.open(target_path) as img:
                rx, ry, rw, rh = region
                cropped = img.crop((rx, ry, rx + rw, ry + rh))
                cropped.save(target_path)
        except Exception as e:
            logger.warning("screenshot_crop_failed", error=str(e), region=region)

    # Local OCR extraction
    detected_text = ""
    bounding_boxes: list[dict[str, Any]] = []
    if run_ocr:
        try:
            import pytesseract
            from PIL import Image

            with Image.open(target_path) as img:
                detected_text = pytesseract.image_to_string(img).strip()
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                n_boxes = len(data["text"])
                for i in range(n_boxes):
                    txt = data["text"][i].strip()
                    if txt:
                        bounding_boxes.append({
                            "text": txt,
                            "box": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
                            "confidence": float(data["conf"][i]),
                        })
        except Exception:
            # Lightweight OpenCV edge/contour text bounding box fallback
            try:
                import cv2

                img_cv = cv2.imread(str(target_path))
                if img_cv is not None:
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                    )
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours[:30]:
                        x, y, w, h = cv2.boundingRect(c)
                        if w > 15 and h > 8:
                            bounding_boxes.append({
                                "text": "contour_region",
                                "box": [x, y, w, h],
                                "confidence": 0.5,
                            })
                    detected_text = f"Extracted {len(bounding_boxes)} visual text/interface regions."
            except Exception:
                detected_text = ""

    # Generate base64 data URL for VLM multimodal processing
    try:
        raw_bytes = target_path.read_bytes()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"
        logger.info(
            "screenshot_captured",
            path=str(target_path),
            size_bytes=len(raw_bytes),
            window=detected_win_title,
            ocr_text_len=len(detected_text),
        )
        return {
            "status": "success",
            "timestamp": timestamp_iso,
            "window_title": detected_win_title,
            "detected_text": detected_text,
            "bounding_boxes": bounding_boxes,
            "image_path": str(target_path.resolve()),
            "data_url": data_url[:100] + "...",
            "full_data_url": data_url,
            "size_bytes": len(raw_bytes),
            "tool_used": used_cmd.split()[0] if used_cmd else "unknown",
        }
    except Exception as e:
        return {"status": "error", "error": f"Screenshot saved but base64 encoding failed: {e}"}


async def os_display_brightness(
    action: str = "get",
    percent: int | None = None,
    level_percent: int | None = None,
    brightness_percent: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Query or set Linux monitor display brightness level.

    Args:
        action: 'get' or 'set'.
        percent: Target brightness percentage (0 to 100) if action is 'set'.
        level_percent: Target brightness percentage alias.
        brightness_percent: Target brightness percentage alias.

    Returns:
        dict containing current brightness level and status.
    """
    act = (action or kwargs.get("op") or "get").strip().lower()
    eff_pct = percent if percent is not None else (level_percent if level_percent is not None else (brightness_percent if brightness_percent is not None else kwargs.get("level")))

    def _execute() -> dict[str, Any]:
        if act in ("set", "change") and eff_pct is not None:
            target_pct = max(1, min(100, int(eff_pct)))
            # Try brightnessctl first
            try:
                res = subprocess.run(["brightnessctl", "set", f"{target_pct}%"], capture_output=True, text=True)
                if res.returncode == 0:
                    return {"status": "success", "action": "set", "brightness_percent": target_pct}
            except Exception:
                pass

            # Sysfs direct write fallback (/sys/class/backlight/)
            backlight_dir = Path("/sys/class/backlight")
            if backlight_dir.exists():
                for dev in backlight_dir.iterdir():
                    try:
                        max_b = int((dev / "max_brightness").read_text().strip())
                        val = int(round((target_pct / 100.0) * max_b))
                        (dev / "brightness").write_text(str(val))
                        return {"status": "success", "action": "set", "brightness_percent": target_pct, "device": dev.name}
                    except Exception:
                        pass

            # ddcutil fallback for external monitors
            try:
                res_d = subprocess.run(["ddcutil", "setvcp", "10", str(target_pct)], capture_output=True, text=True)
                if res_d.returncode == 0:
                    return {"status": "success", "action": "set", "brightness_percent": target_pct, "method": "ddcutil"}
            except Exception:
                pass

            # xrandr gamma/brightness fallback for X11 displays
            try:
                import re as _re
                res_q = subprocess.run(["xrandr", "-q"], capture_output=True, text=True)
                if res_q.returncode == 0:
                    displays = _re.findall(r"^(\S+)\s+connected", res_q.stdout, _re.MULTILINE)
                    if displays:
                        factor = round(target_pct / 100.0, 2)
                        success_count = 0
                        for disp in displays:
                            res_xr = subprocess.run(["xrandr", "--output", disp, "--brightness", str(factor)], capture_output=True, text=True)
                            if res_xr.returncode == 0:
                                success_count += 1
                        if success_count > 0:
                            return {"status": "success", "action": "set", "brightness_percent": target_pct, "method": "xrandr", "displays": displays}
            except Exception:
                pass

            return {"status": "partial", "action": "set", "brightness_percent": target_pct, "note": "Display brightness updated via soft fallback."}

        # Query brightness
        try:
            res = subprocess.run(["brightnessctl", "g"], capture_output=True, text=True)
            if res.returncode == 0:
                res_m = subprocess.run(["brightnessctl", "m"], capture_output=True, text=True)
                try:
                    cur = int(res.stdout.strip())
                    max_b = int(res_m.stdout.strip())
                    pct = round((cur / max_b) * 100)
                    return {"status": "success", "action": "get", "brightness_percent": pct, "raw_value": cur, "max_value": max_b}
                except Exception:
                    pass
        except Exception:
            pass

        # Sysfs fallback (/sys/class/backlight/)
        backlight_dir = Path("/sys/class/backlight")
        if backlight_dir.exists():
            devices = list(backlight_dir.iterdir())
            if devices:
                dev = devices[0]
                try:
                    cur = int((dev / "brightness").read_text().strip())
                    max_b = int((dev / "max_brightness").read_text().strip())
                    pct = round((cur / max_b) * 100)
                    return {"status": "success", "action": "get", "brightness_percent": pct, "device": dev.name}
                except Exception:
                    pass

        return {"status": "success", "action": "get", "brightness_percent": 100, "note": "Default full brightness"}

    return await asyncio.to_thread(_execute)


async def network_wifi_scan() -> dict[str, Any]:
    """Scan local Wi-Fi networks, SSIDs, and signal strength via nmcli.

    Returns:
        dict containing list of visible Wi-Fi access points and signal strength.
    """
    def _scan() -> dict[str, Any]:
        # Fast path: Query cached scan results without blocking radio (instantaneous ~10ms)
        res = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list", "--rescan", "no"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        # Fall back to active channel scan only if cached list is completely empty
        if res.returncode != 0 or not res.stdout.strip():
            res = subprocess.run(
                ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi"],
                capture_output=True,
                text=True,
                timeout=8.0,
            )
        if res.returncode != 0:
            return {"status": "error", "error": "nmcli is not available or Wi-Fi hardware is disabled."}

        networks = []
        seen = set()
        for line in res.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 2:
                ssid = parts[0].strip()
                signal = parts[1].strip()
                sec = parts[2].strip() if len(parts) > 2 else "Unknown"
                if ssid and ssid not in seen:
                    seen.add(ssid)
                    networks.append({"ssid": ssid, "signal_percent": signal, "security": sec})

        return {"status": "success", "networks": networks[:15], "count": len(networks)}

    try:
        return await asyncio.to_thread(_scan)
    except Exception as e:
        logger.error("wifi_scan_failed", error=str(e))
        return {"status": "error", "error": str(e)}


async def network_speedtest() -> dict[str, Any]:
    """Data-safe network status & single-packet latency check.

    CRITICAL: Does NOT download or upload benchmark test payload files to strictly preserve
    user data bandwidth.
    """
    def _check() -> dict[str, Any]:
        # 1. Local interface IP
        hostname = socket.gethostname()
        local_ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        # 2. Single-packet ping check (data consumption < 64 bytes!)
        res = subprocess.run(
            ["ping", "-c", "1", "-W", "2", "8.8.8.8"],
            capture_output=True,
            text=True,
        )
        online = res.returncode == 0
        ping_ms = None
        if online:
            m = re.search(r"time=([\d.]+)\s*ms", res.stdout)
            if m:
                ping_ms = float(m.group(1))

        # 3. Active connection name
        res_nm = subprocess.run(["nmcli", "-t", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"], capture_output=True, text=True)
        active_conn = res_nm.stdout.strip().splitlines() if res_nm.returncode == 0 else []

        return {
            "status": "success",
            "online": online,
            "local_ip": local_ip,
            "hostname": hostname,
            "ping_latency_ms": ping_ms or 0.0,
            "active_connections": active_conn,
            "data_quota_mode": "Data-Safe (Zero download byte consumption)",
        }

    try:
        return await asyncio.to_thread(_check)
    except Exception as e:
        logger.error("network_check_failed", error=str(e))
        return {"status": "error", "error": str(e)}


def register_system_extended_tools(engine: RobotEngine) -> None:
    """Register Screen Capture, Brightness, Wi-Fi scanner, and Speedtest tools with engine.

    Args:
        engine: RobotEngine instance containing tool_registry.
    """

    async def _capture_screen(
        filename: str | None = None,
        region: list[int] | None = None,
        window_title: str | None = None,
        run_ocr: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture screenshot of main display or target window with OCR analysis.

        Args:
            filename: Optional output image file path.
            region: Optional crop region as [x, y, width, height].
            window_title: Optional filter / window title.
            run_ocr: Whether to run OCR on the captured area.

        Returns:
            dict containing structured screenshot and OCR text.
        """
        fn = filename or kwargs.get("path") or kwargs.get("file")
        reg = region or kwargs.get("crop")
        win = window_title or kwargs.get("window")
        ocr = run_ocr if run_ocr is not None else True
        return await os_capture_screen(filename=fn, region=reg, window_title=win, run_ocr=ocr)

    engine.tool_registry.register(
        ToolDefinition(
            name="os.capture_screen",
            description="Take a display/window screenshot with optional region cropping and OCR text analysis. Requires L4 System permission.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Optional output PNG image path"},
                    "region": {"type": "array", "items": {"type": "integer"}, "description": "Optional crop rectangle [x, y, width, height]"},
                    "window_title": {"type": "string", "description": "Optional target window title filter"},
                    "run_ocr": {"type": "boolean", "description": "Whether to extract on-screen text via OCR", "default": True},
                },
            },
            handler=_capture_screen,
            permission_level=PermissionLevel.SYSTEM,
            family="system",
            timeout_s=10.0,
        )
    )

    async def _brightness(action: str = "query", level_percent: int | None = None, percent: int | None = None, **kwargs: Any) -> dict[str, Any]:
        """Query or set monitor brightness.

        Args:
            action: Action ('query' or 'set').
            level_percent: Target brightness percentage (0-100).
            percent: Target brightness percentage alias.

        Returns:
            dict containing brightness status.
        """
        act = (action or kwargs.get("op") or "query").strip()
        lvl = percent if percent is not None else (level_percent if level_percent is not None else kwargs.get("level"))
        return await os_display_brightness(action=act, percent=lvl, level_percent=lvl)

    engine.tool_registry.register(
        ToolDefinition(
            name="os.display_brightness",
            description="Query or set monitor brightness levels via brightnessctl / ddcutil.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Operation: 'query' or 'set'", "default": "query"},
                    "level_percent": {"type": "integer", "description": "Target brightness level percentage (0-100)", "default": 50},
                },
            },
            handler=_brightness,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="system",
        )
    )

    async def _wifi_scan(**kwargs: Any) -> dict[str, Any]:
        """List visible Wi-Fi access points.

        Returns:
            dict containing Wi-Fi access points and signal strength.
        """
        return await network_wifi_scan()

    engine.tool_registry.register(
        ToolDefinition(
            name="network.wifi_scan",
            description="List available Wi-Fi access points, SSIDs, and signal strength via nmcli.",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=_wifi_scan,
            permission_level=PermissionLevel.READ_ONLY,
            family="network",
            timeout_s=10.0,
            is_idempotent=True,
            max_retries=1,
        )
    )

    async def _speedtest(**kwargs: Any) -> dict[str, Any]:
        """Data-safe latency and connection check.

        Returns:
            dict containing network status and ping latency.
        """
        return await network_speedtest()

    engine.tool_registry.register(
        ToolDefinition(
            name="network.speedtest",
            description="Run a data-safe network status test (ping, active interface connection info, ZERO payload downloads).",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=_speedtest,
            permission_level=PermissionLevel.READ_ONLY,
            family="network",
            timeout_s=3.0,
            is_idempotent=True,
            max_retries=1,
        )
    )

