"""Built-in OS and workspace automation tools for app launching, system control, clipboard, and notifications.

Provides native Linux desktop integration for workstation automation:
- os.app_launch: Launch GUI or terminal desktop applications detached from Robot Core.
- os.system_control: Manage workstation screen lock, display power, audio sink volume, and battery status.
- os.clipboard_read: Read text from the system clipboard with multi-tier fallbacks.
- os.clipboard_write: Copy text to the system clipboard with multi-tier fallbacks.
- os.notify_desktop: Dispatch native desktop notification popups via notify-send.
"""

from __future__ import annotations

import asyncio
import glob
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.system_os")

# In-memory clipboard fallback when headless or external clipboard managers are missing
_IN_MEMORY_CLIPBOARD: str = ""

# Common application name aliases mapped to binary search candidates
COMMON_APP_MAP: dict[str, list[str]] = {
    "vscode": ["code", "codium", "cursor"],
    "vs code": ["code", "codium", "cursor"],
    "code": ["code", "codium", "cursor"],
    "visual studio code": ["code", "codium", "cursor"],
    "cursor": ["cursor", "code"],
    "chrome": ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"],
    "google chrome": ["google-chrome", "google-chrome-stable", "chromium"],
    "browser": ["google-chrome", "firefox", "chromium", "xdg-open"],
    "web browser": ["google-chrome", "firefox", "chromium"],
    "firefox": ["firefox", "firefox-esr"],
    "mozilla": ["firefox"],
    "spotify": ["spotify", "spotify-client"],
    "terminal": ["gnome-terminal", "konsole", "alacritty", "kitty", "xterm", "terminator", "foot", "xfce4-terminal", "tilix"],
    "console": ["gnome-terminal", "konsole", "alacritty", "kitty", "xterm"],
    "files": ["nautilus", "dolphin", "thunar", "pcmanfm", "nemo"],
    "file manager": ["nautilus", "dolphin", "thunar", "pcmanfm", "nemo"],
    "nautilus": ["nautilus"],
    "calculator": ["gnome-calculator", "kcalc", "galculator", "xcalc"],
    "calc": ["gnome-calculator", "kcalc", "galculator", "xcalc"],
    "settings": ["gnome-control-center", "systemsettings"],
    "text editor": ["gedit", "kate", "mousepad", "xed", "gnome-text-editor"],
    "editor": ["gedit", "kate", "mousepad", "code"],
    "slack": ["slack"],
    "discord": ["discord"],
    "vlc": ["vlc"],
    "obs": ["obs", "obs-studio"],
}


def resolve_application_binary(app_name: str) -> tuple[str | None, list[str]]:
    """Resolve an application name or alias to an executable binary path and default arguments.

    Args:
        app_name: Friendly name, alias, or command string (e.g. "VS Code", "google-chrome", "spotify").

    Returns:
        A tuple of (resolved_binary_path_or_command, default_args).
    """
    clean_name = app_name.strip().lower()
    if not clean_name:
        return None, []

    # 1. Check known alias dictionary
    candidates = COMMON_APP_MAP.get(clean_name, [clean_name])

    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found, []

    # 2. Check if clean_name is an exact binary in PATH
    direct_match = shutil.which(clean_name)
    if direct_match:
        return direct_match, []

    # 3. Check for flatpak / snap binary names
    for prefix in ["/var/lib/flatpak/exports/bin", "/snap/bin"]:
        candidate_path = os.path.join(prefix, clean_name)
        if os.path.isfile(candidate_path) and os.access(candidate_path, os.X_OK):
            return candidate_path, []

    # 4. Fallback to xdg-open if it's a URL or file path
    if clean_name.startswith(("http://", "https://", "file://", "/", "~")):
        xdg = shutil.which("xdg-open")
        if xdg:
            return xdg, [app_name]

    return None, []


async def launch_detached_process(binary: str, args: list[str] | None = None) -> tuple[bool, int | None, str]:
    """Launch a system desktop application asynchronously detached from Robot Core.

    Args:
        binary: Absolute or PATH-resolvable binary executable.
        args: Optional list of command-line arguments.

    Returns:
        Tuple of (success, pid, message).
    """
    cmd = [binary] + (args or [])
    try:
        # Spawn detached process in its own session so it survives robot restarts and does not block IO
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return True, proc.pid, f"Started process '{binary}' with PID {proc.pid}"
    except Exception as e:
        logger.error("app_launch_failed", command=cmd, error=str(e))
        return False, None, f"Failed to start process '{binary}': {e}"


def get_system_battery_status() -> dict[str, Any]:
    """Read local system battery status and capacity from Linux sysfs.

    Returns:
        Dictionary containing battery capacity, status, and health metrics.
    """
    bats = glob.glob("/sys/class/power_supply/BAT*")
    if not bats:
        # Check if AC power supply exists (desktop workstation mode)
        acs = glob.glob("/sys/class/power_supply/AC*") or glob.glob("/sys/class/power_supply/ADP*")
        if acs:
            return {
                "present": False,
                "type": "AC Desktop Workstation",
                "status": "Connected to continuous AC wall power",
                "percentage": 100,
            }
        return {"present": False, "status": "No battery or power supply detected"}

    bat_path = bats[0]
    info: dict[str, Any] = {"present": True, "battery_id": os.path.basename(bat_path)}

    # Read standard sysfs attributes
    for attr in ["capacity", "status", "health", "technology", "model_name"]:
        file_path = os.path.join(bat_path, attr)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                    info[attr] = int(val) if attr == "capacity" and val.isdigit() else val
            except Exception:
                pass

    pct = info.get("capacity", "Unknown")
    st = info.get("status", "Unknown")
    info["summary"] = f"Battery is at {pct}% ({st})"
    return info


async def execute_system_control(action: str, value: str | int | float | None = None) -> dict[str, Any]:
    """Execute workstation controls for lock, display, audio, and battery.

    Args:
        action: Target control action ("lock", "sleep_display", "wake_display", "battery", "volume_mute", etc.).
        value: Optional numeric or string value for setting volume/brightness.

    Returns:
        Dictionary describing execution outcome.
    """
    clean_action = action.strip().lower()

    # 1. Lock Screen
    if clean_action in {"lock", "lock_screen", "lock_session", "lock screen"}:
        lock_cmds = [
            ["loginctl", "lock-session"],
            ["xdg-screensaver", "lock"],
            ["gnome-screensaver-command", "-l"],
            ["xdotool", "key", "Ctrl+Alt+l"],
        ]
        for cmd in lock_cmds:
            if shutil.which(cmd[0]):
                try:
                    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    await proc.communicate()
                    return {"status": "success", "action": "lock_screen", "message": "Workstation screen locked successfully."}
                except Exception:
                    continue
        return {"status": "error", "message": "No supported lock session utility found on system."}

    # 2. Display Sleep / Screen Off
    if clean_action in {"sleep_display", "display_off", "screen_off", "turn_off_screen", "screen off", "display sleep"}:
        if shutil.which("xset"):
            try:
                proc = await asyncio.create_subprocess_exec("xset", "dpms", "force", "off", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()
                return {"status": "success", "action": "sleep_display", "message": "Display turned off."}
            except Exception as e:
                return {"status": "error", "message": f"Failed to turn off display: {e}"}
        return {"status": "error", "message": "xset utility is not available on this host."}

    # 3. Display Wake / Screen On
    if clean_action in {"wake_display", "screen_on", "display_on", "turn_on_screen", "screen on", "display on"}:
        if shutil.which("xset"):
            try:
                proc = await asyncio.create_subprocess_exec("xset", "dpms", "force", "on", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()
                return {"status": "success", "action": "wake_display", "message": "Display awakened."}
            except Exception as e:
                return {"status": "error", "message": f"Failed to awaken display: {e}"}
        return {"status": "error", "message": "xset utility is not available on this host."}

    # 4. Battery / Power Status
    if clean_action in {"battery", "battery_status", "power", "power_status", "battery level"}:
        bat_info = get_system_battery_status()
        return {"status": "success", "action": "battery_status", "battery": bat_info, "message": bat_info.get("summary", "Battery status retrieved.")}

    # 5. Audio Volume Controls (pactl / amixer)
    if clean_action in {"volume_mute", "mute", "unmute", "toggle_mute"}:
        if shutil.which("pactl"):
            try:
                proc = await asyncio.create_subprocess_exec("pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()
                return {"status": "success", "action": "volume_mute", "message": "Toggled master audio mute."}
            except Exception as e:
                return {"status": "error", "message": f"Failed to toggle audio mute: {e}"}

    if clean_action in {"volume_up", "volume up", "louder"}:
        step = f"+{int(value)}%" if value else "+10%"
        if shutil.which("pactl"):
            try:
                proc = await asyncio.create_subprocess_exec("pactl", "set-sink-volume", "@DEFAULT_SINK@", step, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()
                return {"status": "success", "action": "volume_up", "message": f"Master volume increased by {step}."}
            except Exception as e:
                return {"status": "error", "message": f"Failed to increase volume: {e}"}

    if clean_action in {"volume_down", "volume down", "quieter"}:
        step = f"-{int(value)}%" if value else "-10%"
        if shutil.which("pactl"):
            try:
                proc = await asyncio.create_subprocess_exec("pactl", "set-sink-volume", "@DEFAULT_SINK@", step, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()
                return {"status": "success", "action": "volume_down", "message": f"Master volume decreased by {step}."}
            except Exception as e:
                return {"status": "error", "message": f"Failed to decrease volume: {e}"}

    if clean_action in {"volume_set", "set_volume", "volume"}:
        if value is not None:
            val_pct = int(float(value) * 100) if float(value) <= 1.0 else int(float(value))
            val_pct = max(0, min(150, val_pct))
            if shutil.which("pactl"):
                try:
                    proc = await asyncio.create_subprocess_exec("pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{val_pct}%", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    await proc.communicate()
                    return {"status": "success", "action": "volume_set", "volume_percent": val_pct, "message": f"Master volume set to {val_pct}%."}
                except Exception as e:
                    return {"status": "error", "message": f"Failed to set volume: {e}"}

    # 6. Suspend / Sleep System
    if clean_action in {"suspend", "sleep", "system_sleep"}:
        for cmd in [["systemctl", "suspend"], ["loginctl", "suspend"]]:
            if shutil.which(cmd[0]):
                try:
                    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    await proc.communicate()
                    return {"status": "success", "action": "suspend", "message": "System suspended."}
                except Exception as e:
                    return {"status": "error", "message": f"Failed to suspend system: {e}"}

    return {"status": "error", "message": f"Unknown system control action '{action}'."}


def read_system_clipboard() -> str:
    """Read plain text content from the host system clipboard with cascading fallbacks.

    Returns:
        String content of clipboard or empty string if unavailable.
    """
    global _IN_MEMORY_CLIPBOARD

    # 1. Wayland wl-paste
    if shutil.which("wl-paste"):
        try:
            res = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass

    # 2. X11 xclip
    if shutil.which("xclip"):
        try:
            res = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass

    # 3. X11 xsel
    if shutil.which("xsel"):
        try:
            res = subprocess.run(["xsel", "--clipboard", "--output"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass

    # 4. Pyperclip library
    try:
        import pyperclip
        text = pyperclip.paste()
        if text:
            return text
    except Exception:
        pass

    # 5. In-memory fallback
    return _IN_MEMORY_CLIPBOARD


def write_system_clipboard(text: str) -> bool:
    """Write text to the host system clipboard with cascading fallbacks.

    Args:
        text: String content to copy to the clipboard.

    Returns:
        True if successfully written, False otherwise.
    """
    global _IN_MEMORY_CLIPBOARD
    _IN_MEMORY_CLIPBOARD = text
    written = False

    # 1. Wayland wl-copy
    if shutil.which("wl-copy"):
        try:
            res = subprocess.run(["wl-copy"], input=text, text=True, capture_output=True, timeout=1.0)
            if res.returncode == 0:
                written = True
        except Exception:
            pass

    # 2. X11 xclip
    if shutil.which("xclip"):
        try:
            res = subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, capture_output=True, timeout=1.0)
            if res.returncode == 0:
                written = True
        except Exception:
            pass

    # 3. X11 xsel
    if shutil.which("xsel"):
        try:
            res = subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, capture_output=True, timeout=1.0)
            if res.returncode == 0:
                written = True
        except Exception:
            pass

    # 4. Pyperclip library
    try:
        import pyperclip
        pyperclip.copy(text)
        written = True
    except Exception:
        pass

    return written or bool(_IN_MEMORY_CLIPBOARD)


def register_system_os_tools(engine: "RobotEngine") -> None:
    """Register all OS and workspace automation tools with the engine's tool registry.

    Args:
        engine: Reference to the RobotEngine instance.
    """

    # 1. os.app_launch
    async def app_launch(app_name: str = "", args: list[str] | str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Launch a desktop application or utility."""
        target_app = (app_name or kwargs.get("app") or kwargs.get("name") or kwargs.get("application") or kwargs.get("command") or "").strip()
        if not target_app:
            return {"status": "error", "message": "No application name provided to launch."}

        # Parse args if provided as string
        arg_list: list[str] = []
        if isinstance(args, str) and args.strip():
            arg_list = [args.strip()]
        elif isinstance(args, list):
            arg_list = [str(a) for a in args]

        binary, default_args = resolve_application_binary(target_app)
        if not binary:
            return {
                "status": "not_found",
                "message": f"Application '{target_app}' could not be located in system PATH or standard desktop paths.",
            }

        effective_args = default_args + arg_list
        success, pid, message = await launch_detached_process(binary, effective_args)
        if success:
            return {
                "status": "success",
                "app_name": target_app,
                "binary": binary,
                "pid": pid,
                "message": f"Launched '{target_app}' ({os.path.basename(binary)}).",
            }
        return {"status": "error", "app_name": target_app, "message": message}

    engine.tool_registry.register(
        ToolDefinition(
            name="os.app_launch",
            description="Launch or open a desktop application (e.g. 'VS Code', 'Spotify', 'Chrome', 'Firefox', 'Terminal', 'Files', 'Calculator'). Spoken triggers: 'Open VS Code', 'Launch Spotify', 'Open Google Chrome', 'Start terminal'.",
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name or alias of the application to launch (e.g. 'code', 'spotify', 'chrome', 'firefox', 'terminal', 'calculator')",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional command-line arguments, file paths, or URLs to open with the application",
                    },
                },
                "required": ["app_name"],
            },
            handler=app_launch,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="os",
            is_idempotent=False,
            max_retries=0,
        )
    )

    # 2. os.system_control
    async def system_control(action: str = "", value: str | int | float | None = None, **kwargs: Any) -> dict[str, Any]:
        """Control workstation display, audio, screen lock, or query battery status."""
        target_action = (action or kwargs.get("act") or kwargs.get("command") or kwargs.get("operation") or "").strip()
        target_value = value if value is not None else (kwargs.get("level") or kwargs.get("val") or kwargs.get("amount"))
        if not target_action:
            return {"status": "error", "message": "No system control action specified."}

        return await execute_system_control(target_action, target_value)

    engine.tool_registry.register(
        ToolDefinition(
            name="os.system_control",
            description="Control workstation and OS power/display/audio/battery states (e.g. 'lock screen', 'sleep display', 'wake display', 'battery status', 'mute', 'volume up', 'volume down', 'set volume'). Spoken triggers: 'Lock screen', 'Put display to sleep', 'Turn off monitor', 'How much battery is left?', 'Mute audio'.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "System control action: 'lock', 'sleep_display', 'wake_display', 'battery', 'volume_mute', 'volume_up', 'volume_down', 'volume_set', 'suspend'",
                    },
                    "value": {
                        "type": "string",
                        "description": "Optional numeric percentage or modifier (e.g. 50 for 50% volume, 10 for 10% step)",
                    },
                },
                "required": ["action"],
            },
            handler=system_control,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="os",
            is_idempotent=False,
            max_retries=0,
        )
    )

    # 3. os.clipboard_read
    async def clipboard_read(**kwargs: Any) -> dict[str, Any]:
        """Read plain text from system clipboard."""
        content = await asyncio.to_thread(read_system_clipboard)
        if content:
            return {
                "status": "success",
                "text": content,
                "character_count": len(content),
                "message": f"Clipboard contains {len(content)} characters.",
            }
        return {
            "status": "empty",
            "text": "",
            "character_count": 0,
            "message": "System clipboard is currently empty.",
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="os.clipboard_read",
            description="Read the current text content from the system clipboard. Spoken triggers: 'What is on my clipboard?', 'Read my clipboard', 'Paste what is in my clipboard'.",
            parameters={"type": "object", "properties": {}},
            handler=clipboard_read,
            permission_level=PermissionLevel.READ_ONLY,
            family="os",
        )
    )

    # 4. os.clipboard_write
    async def clipboard_write(text: str = "", **kwargs: Any) -> dict[str, Any]:
        """Write plain text into system clipboard."""
        target_text = (text or kwargs.get("content") or kwargs.get("data") or kwargs.get("value") or "").strip()
        if not target_text:
            return {"status": "error", "message": "No text provided to copy to clipboard."}

        ok = await asyncio.to_thread(write_system_clipboard, target_text)
        preview = target_text[:60] + "..." if len(target_text) > 60 else target_text
        if ok:
            return {
                "status": "success",
                "character_count": len(target_text),
                "preview": preview,
                "message": f"Successfully copied {len(target_text)} characters to clipboard.",
            }
        return {
            "status": "error",
            "message": "Failed to write to system clipboard.",
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="os.clipboard_write",
            description="Copy specified text into the system clipboard. Spoken triggers: 'Copy this to clipboard', 'Put this text on my clipboard', 'Copy code to clipboard'.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text or code snippet to copy to the clipboard",
                    },
                },
                "required": ["text"],
            },
            handler=clipboard_write,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="os",
            is_idempotent=False,
            max_retries=0,
        )
    )

    # 5. os.notify_desktop
    async def notify_desktop(
        title: str = "Robot Core Notification",
        message: str = "",
        urgency: str = "normal",
        icon: str = "dialog-information",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Display a native desktop notification popup on the Linux host."""
        eff_title = (title or kwargs.get("summary") or kwargs.get("header") or "Robot Core").strip()
        eff_msg = (message or kwargs.get("body") or kwargs.get("text") or kwargs.get("content") or "").strip()
        eff_urgency = (urgency or kwargs.get("priority") or "normal").strip().lower()
        eff_icon = (icon or "dialog-information").strip()

        if eff_urgency not in {"low", "normal", "critical"}:
            eff_urgency = "normal"

        if not eff_msg:
            return {"status": "error", "message": "No notification message body provided."}

        notify_bin = shutil.which("notify-send")
        if not notify_bin:
            return {"status": "error", "message": "notify-send utility is not installed on this host."}

        cmd = [
            notify_bin,
            "-u", eff_urgency,
            "-i", eff_icon,
            "-a", "Robot Core",
            eff_title,
            eff_msg,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return {
                    "status": "success",
                    "title": eff_title,
                    "message": eff_msg,
                    "urgency": eff_urgency,
                }
            return {
                "status": "error",
                "message": f"notify-send exited with code {proc.returncode}: {stderr.decode().strip()}",
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to dispatch desktop notification: {e}"}

    engine.tool_registry.register(
        ToolDefinition(
            name="os.notify_desktop",
            description="Display a native desktop notification popup on the host Linux desktop. Spoken triggers: 'Send a desktop notification', 'Show a desktop popup', 'Notify me on screen'.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title or headline of the notification popup",
                        "default": "Robot Notification",
                    },
                    "message": {
                        "type": "string",
                        "description": "Body text of the notification to display",
                    },
                    "urgency": {
                        "type": "string",
                        "description": "Notification urgency: 'low', 'normal', or 'critical'",
                        "default": "normal",
                    },
                    "icon": {
                        "type": "string",
                        "description": "Icon name or icon path (e.g. 'dialog-information', 'dialog-warning')",
                        "default": "dialog-information",
                    },
                },
                "required": ["message"],
            },
            handler=notify_desktop,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="os",
            is_idempotent=False,
            max_retries=0,
        )
    )
