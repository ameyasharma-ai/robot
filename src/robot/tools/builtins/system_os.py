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
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.system_os")

# In-memory clipboard fallback when headless or external clipboard managers are missing
_IN_MEMORY_CLIPBOARD: str = ""

# Shells and interpreters must never be launchable through os.app_launch.
_SHELL_BINARIES: frozenset[str] = frozenset(
    {
        "sh", "bash", "zsh", "dash", "ash", "ksh", "csh", "tcsh", "fish",
        "busybox", "env", "xargs", "sudo", "pkexec", "su", "doas", "sudoedit",
        "python", "python2", "python3", "pypy", "perl", "ruby", "node", "nodejs",
        "deno", "bun", "php", "lua", "tclsh", "wish", "expect", "script",
        "powershell", "pwsh", "cmd", "cmd.exe", "wine", "systemd-run", "nohup",
        "setsid", "timeout", "flock", "stdbuf", "nice", "ionice", "chrt",
        "at", "batch", "crontab", "ssh", "scp", "socat", "nc", "netcat",
    }
)

# Flags that turn a benign desktop application into a command interpreter.
_DANGEROUS_ARG_FLAGS: frozenset[str] = frozenset(
    {
        "-c", "--command", "-e", "--eval", "--exec", "--execute", "--shell",
        "--rcfile", "--init-file", "--startup", "--preload", "--require",
        "--import", "-exec", "-execdir", "--app", "--process",
    }
)

_SHELL_META_RE = re.compile(r"[;&|`$><\n\r\\\x00]")
_SAFE_FLAG_RE = re.compile(r"--?[A-Za-z0-9][A-Za-z0-9._=:/+-]*")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_APP_ARG_LENGTH = 2048

# Aliases that spawn an interactive terminal.  They are allowed to launch, but
# never with model-supplied arguments.
_TERMINAL_ALIASES: frozenset[str] = frozenset(
    {"terminal", "console", "gnome-terminal", "konsole", "alacritty", "kitty", "xterm", "terminator", "foot", "xfce4-terminal", "tilix"}
)

_CLIPBOARD_MAX_CHARS = 8192
_CLIPBOARD_TRUNCATION_MARKER = "\n... [clipboard truncated at 8 KiB] ..."

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{25,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd|pwd)\b"
        r"\s*[:=]\s*([^\s,;]{4,})"
    ),
)


def _scrub_secrets(text: str) -> str:
    """Redact obvious credentials from user-visible or logged text."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2 and pattern.pattern.startswith("(?i)"):
            redacted = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", redacted)
        elif pattern.groups >= 1 and "bearer" in pattern.pattern.lower():
            redacted = pattern.sub(lambda m: f"{m.group(1)} [REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_and_bound_clipboard(text: str) -> tuple[str, bool, bool]:
    """Redact credentials and cap clipboard text before returning it to the LLM.

    Args:
        text: Raw clipboard contents.

    Returns:
        Tuple of ``(safe_text, truncated, redacted)``.
    """
    raw = text or ""
    truncated = len(raw) > _CLIPBOARD_MAX_CHARS
    bounded = raw[:_CLIPBOARD_MAX_CHARS]
    redacted_text = _scrub_secrets(bounded)
    redacted = redacted_text != bounded
    if truncated:
        redacted_text = f"{redacted_text}{_CLIPBOARD_TRUNCATION_MARKER}"
    return redacted_text, truncated, redacted


async def _run_command_with_timeout(
    cmd: list[str], timeout: float = 5.0
) -> tuple[int | None, bytes, bytes, bool]:
    """Run a short local command with a timeout and process-group cleanup.

    Args:
        cmd: argv array (never a shell string).
        timeout: Wall-clock limit in seconds.

    Returns:
        Tuple of ``(returncode, stdout, stderr, timed_out)``.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout, stderr, False
    except asyncio.TimeoutError:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:  # pragma: no cover - non-POSIX fallback
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:  # pragma: no cover
            pass
        return None, b"", b"", True

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
    "calendar": ["gnome-calendar", "korganizer", "cal"],
    "cal": ["gnome-calendar", "korganizer", "cal", "gnome-calculator"],
    "settings": ["gnome-control-center", "systemsettings"],
    "text editor": ["gedit", "kate", "mousepad", "xed", "gnome-text-editor"],
    "editor": ["gedit", "kate", "mousepad", "code"],
    "slack": ["slack"],
    "discord": ["discord"],
    "vlc": ["vlc"],
    "obs": ["obs", "obs-studio"],
}


def _is_http_url(value: str) -> bool:
    """Return True for an absolute HTTP(S) URL."""
    return bool(re.match(r"^https?://[^\s]+$", value, flags=re.IGNORECASE))


def resolve_application_binary(app_name: str) -> tuple[str | None, list[str]]:
    """Resolve an allowlisted application alias to an executable binary.

    Only aliases present in :data:`COMMON_APP_MAP` are resolved.  Arbitrary
    PATH lookups (``bash``, ``python``, ...) and arbitrary flatpak/snap names
    are intentionally not supported.  An absolute HTTP(S) URL is opened with
    ``xdg-open`` for backward compatibility, but shell-sensitive schemes and
    filesystem paths are rejected.

    Args:
        app_name: Friendly name or alias (e.g. "VS Code", "spotify").

    Returns:
        A tuple of (resolved_binary_path_or_command, default_args); ``(None, [])``
        when the application is unknown or not installed.
    """
    text = (app_name or "").strip()
    clean_name = text.lower()
    if not clean_name:
        return None, []

    if _is_http_url(text):
        xdg = shutil.which("xdg-open")
        if xdg:
            return xdg, [text]
        return None, []

    # Unknown aliases fail closed; never fall back to a raw PATH lookup.
    candidates = COMMON_APP_MAP.get(clean_name)
    if not candidates:
        return None, []

    for candidate in candidates:
        if candidate in _SHELL_BINARIES:
            continue
        found = shutil.which(candidate)
        if found and Path(found).name not in _SHELL_BINARIES:
            return found, []

    # Flatpak exports are consulted only for known allowlisted candidate names.
    for prefix in ("/var/lib/flatpak/exports/bin", "/snap/bin"):
        for candidate in candidates:
            if candidate in _SHELL_BINARIES:
                continue
            candidate_path = os.path.join(prefix, candidate)
            if os.path.isfile(candidate_path) and os.access(candidate_path, os.X_OK):
                return candidate_path, []

    return None, []


def _permitted_arg_roots() -> tuple[Path, ...]:
    """Return directory trees a model-supplied launch path may point inside.

    Returns:
        tuple[Path, ...]: Resolved root paths for the user's home, the robot
            workspace, and the world-writable temp directory.
    """
    roots = [Path(os.path.expanduser("~")), Path("/tmp"), Path(os.getcwd())]
    try:
        from robot.config import get_project_root

        roots.append(Path(str(get_project_root())))
    except Exception:  # pragma: no cover - config unavailable at import time
        pass
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except OSError:  # pragma: no cover - unreadable root
            continue
    return tuple(resolved)


def _launch_path_is_permitted(arg: str) -> bool:
    """Return True when a launch argument stays inside the permitted roots.

    Args:
        arg: Non-flag launch argument that is treated as a filesystem path.

    Returns:
        bool: False for absolute or symlinked paths outside the user's home,
            the workspace, or ``/tmp``.
    """
    candidate = Path(arg).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:  # pragma: no cover - pathological path
        return False
    return any(resolved == root or resolved.is_relative_to(root) for root in _permitted_arg_roots())


def validate_app_arguments(app_alias: str, args: list[str]) -> list[str]:
    """Validate model-supplied launch arguments for an allowlisted application.

    Arguments are passed as an argv array (never a shell string), but are still
    restricted to plain flags, paths, and HTTP(S) URLs.  Shell metacharacters,
    control characters, interpreter flags (``-c``/``-e``/``--command``), and
    terminal command injection are rejected.  Path arguments are additionally
    confined to the user's home, the workspace, and ``/tmp`` so a launched
    application can never be pointed straight at system or other-user files.

    Args:
        app_alias: Lower-case allowlist key.
        args: Proposed argument strings.

    Returns:
        The validated argument list.

    Raises:
        ValueError: If any argument is unsafe or not appropriate for the app.
    """
    clean_args = [str(arg) for arg in (args or [])]
    if app_alias in _TERMINAL_ALIASES and clean_args:
        raise ValueError("Terminal applications cannot be launched with arguments.")
    if len(clean_args) > 16:
        raise ValueError("Too many launch arguments (max 16).")

    for arg in clean_args:
        if not arg:
            raise ValueError("Empty launch argument.")
        if len(arg) > _MAX_APP_ARG_LENGTH:
            raise ValueError("Launch argument is too long.")
        if _CONTROL_RE.search(arg) or _SHELL_META_RE.search(arg):
            raise ValueError(f"Unsafe characters in launch argument: {arg!r}")
        lowered = arg.lower()
        if lowered in _DANGEROUS_ARG_FLAGS or any(
            lowered.startswith(f"{flag}=") for flag in _DANGEROUS_ARG_FLAGS
        ):
            raise ValueError(f"Launch flag is not allowed: {arg!r}")
        if arg.startswith("-"):
            if not _SAFE_FLAG_RE.fullmatch(arg):
                raise ValueError(f"Launch flag is not allowed: {arg!r}")
            continue
        if _is_http_url(arg):
            continue
        if arg.lower().startswith("file://"):
            raise ValueError("file:// launch targets are not allowed.")
        path = Path(arg).expanduser()
        if any(part == ".." for part in path.parts):
            raise ValueError(f"Path traversal is not allowed in launch argument: {arg!r}")
        if not _launch_path_is_permitted(arg):
            raise ValueError(f"Launch path is outside the permitted workspace: {arg!r}")
    return clean_args


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
        # Spawn detached process in its own session so it survives robot
        # restarts and does not block IO. fork/exec is done with the asyncio
        # child process API: subprocess.Popen would block the loop on a fork of
        # a multi-gigabyte resident process.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, proc.pid, f"Started process '{Path(binary).name}' with PID {proc.pid}"
    except Exception as e:
        logger.error("app_launch_failed", binary=Path(binary).name, error=_scrub_secrets(str(e)))
        return False, None, f"Failed to start process '{Path(binary).name}'."


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


async def execute_system_control(
    action: str,
    value: str | int | float | None = None,
    *,
    allow_power: bool = False,
) -> dict[str, Any]:
    """Execute workstation controls for lock, display, audio, and battery.

    Destructive power actions (suspend/shutdown/reboot) are only executed when
    ``allow_power`` is True; that path is exposed as a separate SYSTEM-level
    tool and is never reachable from the HARMLESS_LOCAL system-control tool.

    Args:
        action: Target control action ("lock", "sleep_display", "wake_display", "battery", "volume_mute", etc.).
        value: Optional numeric or string value for setting volume/brightness.
        allow_power: Whether destructive power actions are permitted.

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
                rc, _, _, timed_out = await _run_command_with_timeout(cmd, timeout=5.0)
                if not timed_out and rc == 0:
                    return {"status": "success", "action": "lock_screen", "message": "Workstation screen locked successfully."}
        return {"status": "error", "message": "No supported lock session utility found on system."}

    # 2. Display Sleep / Screen Off
    if clean_action in {"sleep_display", "display_off", "screen_off", "turn_off_screen", "screen off", "display sleep"}:
        if shutil.which("xset"):
            rc, _, _, timed_out = await _run_command_with_timeout(
                ["xset", "dpms", "force", "off"], timeout=5.0
            )
            if not timed_out and rc == 0:
                return {"status": "success", "action": "sleep_display", "message": "Display turned off."}
            return {"status": "error", "message": "Failed to turn off display."}
        return {"status": "error", "message": "xset utility is not available on this host."}

    # 3. Display Wake / Screen On
    if clean_action in {"wake_display", "screen_on", "display_on", "turn_on_screen", "screen on", "display on"}:
        if shutil.which("xset"):
            rc, _, _, timed_out = await _run_command_with_timeout(
                ["xset", "dpms", "force", "on"], timeout=5.0
            )
            if not timed_out and rc == 0:
                return {"status": "success", "action": "wake_display", "message": "Display awakened."}
            return {"status": "error", "message": "Failed to awaken display."}
        return {"status": "error", "message": "xset utility is not available on this host."}

    # 4. Battery / Power Status
    if clean_action in {"battery", "battery_status", "power", "power_status", "battery level"}:
        bat_info = get_system_battery_status()
        return {"status": "success", "action": "battery_status", "battery": bat_info, "message": bat_info.get("summary", "Battery status retrieved.")}

    # 5. Audio Volume Controls (pactl / amixer)
    if clean_action in {"volume_mute", "mute", "unmute", "toggle_mute"}:
        if shutil.which("pactl"):
            rc, _, _, timed_out = await _run_command_with_timeout(
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], timeout=5.0
            )
            if not timed_out and rc == 0:
                return {"status": "success", "action": "volume_mute", "message": "Toggled master audio mute."}
            return {"status": "error", "message": "Failed to toggle audio mute."}

    if clean_action in {"volume_up", "volume up", "louder"}:
        try:
            step = f"+{int(value)}%" if value else "+10%"
        except (TypeError, ValueError):
            step = "+10%"
        if shutil.which("pactl"):
            rc, _, _, timed_out = await _run_command_with_timeout(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", step], timeout=5.0
            )
            if not timed_out and rc == 0:
                return {"status": "success", "action": "volume_up", "message": f"Master volume increased by {step}."}
            return {"status": "error", "message": "Failed to increase volume."}

    if clean_action in {"volume_down", "volume down", "quieter"}:
        try:
            step = f"-{int(value)}%" if value else "-10%"
        except (TypeError, ValueError):
            step = "-10%"
        if shutil.which("pactl"):
            rc, _, _, timed_out = await _run_command_with_timeout(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", step], timeout=5.0
            )
            if not timed_out and rc == 0:
                return {"status": "success", "action": "volume_down", "message": f"Master volume decreased by {step}."}
            return {"status": "error", "message": "Failed to decrease volume."}

    if clean_action in {"volume_set", "set_volume", "volume"}:
        if value is not None:
            try:
                numeric = float(value)
                val_pct = int(numeric * 100) if numeric <= 1.0 else int(numeric)
            except (TypeError, ValueError):
                return {"status": "error", "message": "Volume value must be numeric."}
            val_pct = max(0, min(150, val_pct))
            if shutil.which("pactl"):
                rc, _, _, timed_out = await _run_command_with_timeout(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{val_pct}%"], timeout=5.0
                )
                if not timed_out and rc == 0:
                    return {"status": "success", "action": "volume_set", "volume_percent": val_pct, "message": f"Master volume set to {val_pct}%."}
                return {"status": "error", "message": "Failed to set volume."}

    # 6. Destructive power actions: only via the SYSTEM-level os.power_control tool.
    if clean_action in {"suspend", "sleep", "system_sleep", "shutdown", "poweroff", "reboot", "restart"}:
        if not allow_power:
            return {
                "status": "error",
                "error": (
                    "Destructive power actions require the SYSTEM-level os.power_control tool "
                    "and explicit user confirmation."
                ),
                "action": clean_action,
            }
        if clean_action in {"suspend", "sleep", "system_sleep"}:
            candidates = [["systemctl", "suspend"], ["loginctl", "suspend"]]
            success_message = "System suspended."
        elif clean_action in {"shutdown", "poweroff"}:
            candidates = [["systemctl", "poweroff"]]
            success_message = "System shutting down."
        else:
            candidates = [["systemctl", "reboot"]]
            success_message = "System rebooting."
        for cmd in candidates:
            if shutil.which(cmd[0]):
                rc, _, _, timed_out = await _run_command_with_timeout(cmd, timeout=10.0)
                if not timed_out and rc == 0:
                    return {"status": "success", "action": clean_action, "message": success_message}
                return {"status": "error", "message": f"Failed to execute power action '{clean_action}'."}
        return {"status": "error", "message": "No supported power management utility found."}

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
        """Launch a desktop application or utility detached from Robot Core.

        Args:
            app_name: Name or alias of the application to launch (e.g. 'calendar', 'calculator', 'chrome', 'terminal').
            args: Optional command-line arguments, file paths, or URLs.
            **kwargs: Extra parameters for robust LLM tool call tolerance.

        Returns:
            Dictionary containing execution status, process ID, binary path, and user-facing message.
        """
        raw_target = (
            app_name
            or kwargs.get("app")
            or kwargs.get("name")
            or kwargs.get("application")
            or kwargs.get("command")
            or kwargs.get("target")
            or kwargs.get("query")
            or kwargs.get("input")
            or kwargs.get("val")
            or kwargs.get("value")
            or ""
        )
        target_app = str(raw_target).strip()
        if not target_app:
            return {
                "status": "error",
                "message": "No application name provided to launch. Please specify an application name (e.g. 'calendar', 'calculator', 'chrome', 'terminal').",
            }

        # Parse args if provided as string
        arg_list: list[str] = []
        if isinstance(args, str) and args.strip():
            arg_list = [args.strip()]
        elif isinstance(args, list):
            arg_list = [str(a) for a in args]

        # Fail closed on anything that is not an explicit allowlist alias (or
        # an HTTP(S) URL opened through xdg-open).
        alias = target_app.lower()
        if _is_http_url(target_app):
            alias = "browser"
        elif alias not in COMMON_APP_MAP:
            return {
                "status": "error",
                "app_name": target_app,
                "error": (
                    f"Application '{target_app}' is not allowlisted. "
                    "Only known desktop applications and HTTP(S) URLs can be launched."
                ),
                "message": (
                    f"Application '{target_app}' is not allowlisted. "
                    "Only known desktop applications and HTTP(S) URLs can be launched."
                ),
            }

        try:
            validated_args = validate_app_arguments(alias, arg_list)
        except ValueError as exc:
            logger.warning("app_launch_arguments_rejected", app=alias, reason=str(exc))
            return {"status": "error", "app_name": target_app, "error": str(exc), "message": str(exc)}

        binary, default_args = resolve_application_binary(target_app)
        if not binary or Path(binary).name.lower() in _SHELL_BINARIES:
            return {
                "status": "error",
                "app_name": target_app,
                "error": f"Application '{target_app}' could not be safely resolved.",
                "message": f"Application '{target_app}' could not be safely resolved.",
            }

        effective_args = default_args + validated_args
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
                        "description": "System control action: 'lock', 'sleep_display', 'wake_display', 'battery', 'volume_mute', 'volume_up', 'volume_down', 'volume_set'. Destructive power actions use os.power_control.",
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

    # 2b. os.power_control (SYSTEM): suspend/shutdown/reboot are destructive and
    # must never be reachable from the HARMLESS_LOCAL system-control tool.
    async def power_control(action: str = "", **kwargs: Any) -> dict[str, Any]:
        """Suspend, shut down, or reboot the workstation (SYSTEM permission)."""
        target_action = (action or kwargs.get("act") or kwargs.get("command") or "").strip()
        if not target_action:
            return {"status": "error", "message": "No power action specified."}
        return await execute_system_control(target_action, allow_power=True)

    engine.tool_registry.register(
        ToolDefinition(
            name="os.power_control",
            description="Perform destructive workstation power actions: suspend, shutdown, or reboot. Requires SYSTEM permission and explicit user confirmation.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["suspend", "shutdown", "reboot"],
                        "description": "Destructive power action to perform",
                    },
                },
                "required": ["action"],
            },
            handler=power_control,
            permission_level=PermissionLevel.SYSTEM,
            family="os",
            is_idempotent=False,
            max_retries=0,
        )
    )

    # 3. os.clipboard_read
    async def clipboard_read(**kwargs: Any) -> dict[str, Any]:
        """Read plain text from system clipboard with secret redaction."""
        content = await asyncio.to_thread(read_system_clipboard)
        if content:
            safe_text, truncated, redacted = redact_and_bound_clipboard(content)
            return {
                "status": "success",
                "text": safe_text,
                "character_count": len(content),
                "returned_characters": len(safe_text),
                "truncated": truncated,
                "redacted": redacted,
                "message": (
                    f"Clipboard contains {len(content)} characters"
                    + (" (truncated to 8 KiB)" if truncated else "")
                    + ("; credentials were redacted" if redacted else "")
                    + "."
                ),
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
            description="Read the current text content from the system clipboard. Credentials are redacted and output is capped at 8 KiB.",
            parameters={"type": "object", "properties": {}},
            handler=clipboard_read,
            # Clipboard contents routinely contain passwords/tokens: require a
            # confirmation step above the HARMLESS_LOCAL auto-execution ceiling.
            permission_level=PermissionLevel.ENVIRONMENT,
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
            rc, _, stderr, timed_out = await _run_command_with_timeout(cmd, timeout=5.0)
            if not timed_out and rc == 0:
                return {
                    "status": "success",
                    "title": eff_title,
                    "message": eff_msg,
                    "urgency": eff_urgency,
                }
            if timed_out:
                return {"status": "error", "message": "notify-send timed out."}
            detail = _scrub_secrets(stderr.decode("utf-8", errors="replace").strip())
            return {
                "status": "error",
                "message": f"notify-send exited with code {rc}: {detail}",
            }
        except Exception as exc:
            logger.error("notify_send_failed", error=_scrub_secrets(str(exc)))
            return {"status": "error", "message": "Failed to dispatch desktop notification."}

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
