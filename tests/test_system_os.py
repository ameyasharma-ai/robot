"""Unit tests for OS and workspace automation tools (system_os).

Validates:
- Application binary resolution and detached execution.
- Workstation system controls (screen lock, display power, audio sink volume, battery).
- Clipboard read and write with fallback cascades.
- Native desktop notifications.
- Permission levels and tool router registration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.tools import PermissionLevel, ToolRegistry
from robot.tools.builtins import register_builtin_tools
from robot.tools.builtins.system_os import (
    COMMON_APP_MAP,
    execute_system_control,
    get_system_battery_status,
    launch_detached_process,
    read_system_clipboard,
    register_system_os_tools,
    resolve_application_binary,
    validate_app_arguments,
    write_system_clipboard,
)


@pytest.fixture
def mock_engine() -> MagicMock:
    """Fixture providing a mock RobotEngine with tool registry."""
    engine = MagicMock()
    engine.tool_registry = ToolRegistry()
    engine.personalities = MagicMock()
    engine.personalities.active = MagicMock(memory_namespace="test")
    engine.state = MagicMock()
    engine.telemetry = MagicMock()
    engine.scheduler = MagicMock()
    return engine


def test_resolve_application_binary_known() -> None:
    """Verify known application name aliases map to search candidates."""
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda cmd: f"/usr/bin/{cmd}" if cmd in ["code", "google-chrome", "firefox"] else None

        bin_path, args = resolve_application_binary("VS Code")
        assert bin_path == "/usr/bin/code"
        assert args == []

        bin_path, args = resolve_application_binary("chrome")
        assert bin_path == "/usr/bin/google-chrome"
        assert args == []

        bin_path, args = resolve_application_binary("firefox")
        assert bin_path == "/usr/bin/firefox"
        assert args == []


def test_resolve_application_binary_urls_and_unknown() -> None:
    """Verify URL and unknown application fallbacks."""
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda cmd: "/usr/bin/xdg-open" if cmd == "xdg-open" else None

        bin_path, args = resolve_application_binary("https://github.com")
        assert bin_path == "/usr/bin/xdg-open"
        assert args == ["https://github.com"]

        mock_which.side_effect = None
        mock_which.return_value = None
        bin_path, args = resolve_application_binary("nonexistent_app_12345")
        assert bin_path is None
        assert args == []


@pytest.mark.asyncio
async def test_launch_detached_process() -> None:
    """Verify detached launching uses the non-blocking asyncio child-process API."""
    mock_proc = MagicMock()
    mock_proc.pid = 4242
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc

        success, pid, msg = await launch_detached_process("/usr/bin/code", ["--new-window"])
        assert success is True
        assert pid == 4242
        assert "Started process" in msg
        mock_exec.assert_called_once()
        args, kwargs = mock_exec.call_args
        assert args == ("/usr/bin/code", "--new-window")
        assert kwargs.get("start_new_session") is True


def test_validate_app_arguments_confines_paths(tmp_path: Path) -> None:
    """Path arguments must stay inside the permitted workspace roots."""
    inside = tmp_path / "notes.txt"
    inside.write_text("hello")
    assert validate_app_arguments("code", [str(inside)]) == [str(inside)]

    with pytest.raises(ValueError, match="outside the permitted workspace"):
        validate_app_arguments("code", ["/etc/shadow"])
    with pytest.raises(ValueError, match="outside the permitted workspace"):
        validate_app_arguments("nautilus", ["/root/.ssh"])


def test_validate_app_arguments_rejects_urlish_and_flags() -> None:
    """Interpreter flags and shell metacharacters stay rejected."""
    assert validate_app_arguments("code", ["--new-window"]) == ["--new-window"]
    with pytest.raises(ValueError, match="not allowed"):
        validate_app_arguments("code", ["-e"])
    with pytest.raises(ValueError, match="Unsafe characters"):
        validate_app_arguments("code", ["a;rm -rf /"])


def test_get_system_battery_status() -> None:
    """Verify battery status reading gracefully handles desktop or laptop sysfs."""
    info = get_system_battery_status()
    assert isinstance(info, dict)
    assert "present" in info
    assert "status" in info or "summary" in info


@pytest.mark.asyncio
async def test_execute_system_control_lock_and_display() -> None:
    """Verify screen lock and display sleep/wake commands."""
    with patch("shutil.which", return_value="/usr/bin/loginctl"), \
         patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_subproc.return_value = mock_proc

        res = await execute_system_control("lock")
        assert res["status"] == "success"
        assert res["action"] == "lock_screen"

    with patch("shutil.which", return_value="/usr/bin/xset"), \
         patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_subproc.return_value = mock_proc

        res = await execute_system_control("sleep_display")
        assert res["status"] == "success"
        assert res["action"] == "sleep_display"

        res = await execute_system_control("wake_display")
        assert res["status"] == "success"
        assert res["action"] == "wake_display"


@pytest.mark.asyncio
async def test_execute_system_control_volume() -> None:
    """Verify volume mute, up, down, and set actions via pactl."""
    with patch("shutil.which", return_value="/usr/bin/pactl"), \
         patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_subproc.return_value = mock_proc

        mute_res = await execute_system_control("volume_mute")
        assert mute_res["status"] == "success"
        assert mute_res["action"] == "volume_mute"

        up_res = await execute_system_control("volume_up", value=15)
        assert up_res["status"] == "success"
        assert up_res["action"] == "volume_up"

        set_res = await execute_system_control("volume_set", value=60)
        assert set_res["status"] == "success"
        assert set_res["volume_percent"] == 60


def test_clipboard_read_and_write() -> None:
    """Verify clipboard read and write with in-memory and fallback cascades."""
    test_text = "Sample Robot Automation Payload 123"
    ok = write_system_clipboard(test_text)
    assert ok is True

    read_back = read_system_clipboard()
    assert test_text in read_back or read_back == test_text


@pytest.mark.asyncio
async def test_tool_registration_and_execution(mock_engine: MagicMock) -> None:
    """Verify all OS tools are properly registered and callable via ToolRegistry."""
    register_system_os_tools(mock_engine)

    expected_tools = [
        "os.app_launch",
        "os.system_control",
        "os.clipboard_read",
        "os.clipboard_write",
        "os.notify_desktop",
    ]
    registered = mock_engine.tool_registry.list_tools()
    for tool_name in expected_tools:
        assert tool_name in registered
        tool_def = mock_engine.tool_registry.get(tool_name)
        assert tool_def is not None
        assert tool_def.family == "os"

    # Test os.clipboard_write and os.clipboard_read via registry
    write_tool = mock_engine.tool_registry.get("os.clipboard_write")
    assert write_tool is not None
    res_w = await write_tool.handler(text="Hello from Unit Test")
    assert res_w["status"] == "success"

    read_tool = mock_engine.tool_registry.get("os.clipboard_read")
    assert read_tool is not None
    res_r = await read_tool.handler()
    assert res_r["status"] == "success"
    assert "Hello from Unit Test" in res_r["text"]

    # Test os.notify_desktop
    with patch("shutil.which", return_value="/usr/bin/notify-send"), \
         patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_subproc.return_value = mock_proc

        notify_tool = mock_engine.tool_registry.get("os.notify_desktop")
        assert notify_tool is not None
        res_n = await notify_tool.handler(title="Alert", message="Task Complete")
        assert res_n["status"] == "success"
        assert res_n["title"] == "Alert"
        assert res_n["message"] == "Task Complete"


def test_resolve_application_binary_calendar() -> None:
    """Verify 'cal' and 'calendar' resolve to gnome-calendar."""
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda cmd: "/usr/bin/gnome-calendar" if cmd == "gnome-calendar" else None
        bin_path, args = resolve_application_binary("cal")
        assert bin_path == "/usr/bin/gnome-calendar"
        assert args == []

        bin_path_cal, args_cal = resolve_application_binary("calendar")
        assert bin_path_cal == "/usr/bin/gnome-calendar"
        assert args_cal == []


@pytest.mark.asyncio
async def test_app_launch_kwargs_tolerance(mock_engine: MagicMock) -> None:
    """Verify app_launch tolerates aliases, query/app kwargs, and missing parameters cleanly."""
    register_system_os_tools(mock_engine)
    launch_tool = mock_engine.tool_registry.get("os.app_launch")
    assert launch_tool is not None

    with patch("robot.tools.builtins.system_os.resolve_application_binary", return_value=("/usr/bin/gnome-calendar", [])), \
         patch("robot.tools.builtins.system_os.launch_detached_process", new_callable=AsyncMock) as mock_launch:
        mock_launch.return_value = (True, 12345, "Started")

        # Call with app_name directly
        res = await launch_tool.handler(app_name="cal")
        assert res["status"] == "success"
        assert res["app_name"] == "cal"
        assert res["pid"] == 12345

        # Call with fallback kwarg 'app'
        res_app = await launch_tool.handler(app="calendar")
        assert res_app["status"] == "success"

        # Call with fallback kwarg 'query'
        res_query = await launch_tool.handler(query="cal")
        assert res_query["status"] == "success"

        # Call with empty arguments returns clean error without crashing
        res_empty = await launch_tool.handler()
        assert res_empty["status"] == "error"
        assert "No application name provided" in res_empty["message"]
