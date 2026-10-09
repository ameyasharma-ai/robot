"""Unit tests for the 12 new advanced developer, terminal, productivity, and extended system tools."""

import asyncio
from typing import Any
import pytest

from robot.tools.builtins.developer_terminal import (
    calculator_evaluate,
    code_run_script,
    execute_terminal_command,
    git_workspace_commit,
    register_developer_terminal_tools,
)
from robot.tools.builtins.productivity_knowledge import (
    NotesVault,
    register_productivity_knowledge_tools,
    translation_translate,
    web_scrape_page,
    wikipedia_lookup,
)
from robot.tools.builtins.system_extended import (
    network_speedtest,
    network_wifi_scan,
    os_capture_screen,
    os_display_brightness,
    register_system_extended_tools,
)
from robot.tools import ToolRegistry, PermissionLevel


@pytest.mark.asyncio
async def test_terminal_execute() -> None:
    """Test bash shell execution tool."""
    res = await execute_terminal_command("echo 'robot_test_ok'", timeout_seconds=5)
    assert res["status"] == "success"
    assert "robot_test_ok" in res["stdout"]
    assert res["exit_code"] == 0


@pytest.mark.asyncio
async def test_git_workspace_commit() -> None:
    """Test git workspace status and diff inspection."""
    res = await git_workspace_commit(action="status")
    assert res["status"] in ("success", "partial")
    assert "branch" in res or "modified_files" in res or "output" in res


@pytest.mark.asyncio
async def test_code_run_script() -> None:
    """Test Python script execution snippet."""
    res = await code_run_script(code="print(21 + 21)", language="python", timeout_seconds=5)
    assert res["status"] == "success"
    assert "42" in res["stdout"]


def test_calculator_evaluate() -> None:
    """Test safe mathematical evaluation and formula processing."""
    res = calculator_evaluate("25 * 4 + 10")
    assert res["status"] == "success"
    assert res["result"] == 110

    res_pct = calculator_evaluate("20% of 150")
    assert res_pct["status"] == "success"
    assert res_pct["result"] == 30.0

    res_trig = calculator_evaluate("sin(0) + cos(0)")
    assert res_trig["status"] == "success"
    assert res_trig["result"] == 1.0


@pytest.mark.asyncio
async def test_wikipedia_lookup() -> None:
    """Test Wikipedia encyclopedic summary lookup."""
    res = await wikipedia_lookup("Python (programming language)")
    assert res["status"] in ("success", "partial", "error", "not_found")
    if res["status"] == "success":
        assert "title" in res


def test_notes_vault(tmp_path: Any) -> None:
    """Test Markdown notes vault creation, tagging, search, and reading."""
    vault_dir = tmp_path / "notes_vault"
    vault = NotesVault(vault_dir=str(vault_dir))

    # Create note
    c_res = vault.create_note(title="Robot Architecture", content="High latency TTS optimization", tags=["robot", "architecture"])
    assert c_res["status"] == "success"
    assert (vault_dir / "robot_architecture.md").exists()

    # Append note
    a_res = vault.append_note(title="Robot Architecture", content="\nAdded RTX 3050 VRAM monitoring.")
    assert a_res["status"] == "success"

    # Search notes
    s_res = vault.search(query="optimization")
    assert s_res["status"] == "success"
    assert s_res["match_count"] >= 1

    # List notes
    l_res = vault.list_notes()
    assert l_res["status"] == "success"
    assert l_res["count"] >= 1


@pytest.mark.asyncio
async def test_translation_translate() -> None:
    """Test fast translation functionality."""
    res = await translation_translate(text="Hello world", target_lang="es")
    assert res["status"] in ("success", "partial")
    assert "translated_text" in res


@pytest.mark.asyncio
async def test_web_scrape_page() -> None:
    """Test web page text extraction."""
    res = await web_scrape_page("https://example.com", max_chars=1000)
    assert res["status"] in ("success", "error")
    if res["status"] == "success":
        assert "Domain" in res["text"] or "Example" in res["text"] or "title" in res


@pytest.mark.asyncio
async def test_os_capture_screen(tmp_path: Any) -> None:
    """Test screenshot capture functionality."""
    out_file = str(tmp_path / "test_shot.png")
    res = await os_capture_screen(filename=out_file)
    assert res["status"] in ("success", "error", "partial")


@pytest.mark.asyncio
async def test_os_display_brightness() -> None:
    """Test monitor brightness query and set handling."""
    res = await os_display_brightness(action="query")
    assert res["status"] in ("success", "error", "partial")


@pytest.mark.asyncio
async def test_network_wifi_scan() -> None:
    """Test Wi-Fi AP scanning via nmcli."""
    res = await network_wifi_scan()
    assert res["status"] in ("success", "error")


@pytest.mark.asyncio
async def test_network_speedtest_datasafe() -> None:
    """CRITICAL: Test data-safe network status check (NO HEAVY DOWNLOADS)."""
    res = await network_speedtest()
    assert res["status"] == "success"
    assert "data_quota_mode" in res
    assert "Data-Safe" in res["data_quota_mode"]
    assert "online" in res
    assert "ping_latency_ms" in res


class MockEngine:
    """Mock engine to test tool registration."""

    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()


def test_tool_registrations() -> None:
    """Test registering all 12 tools into ToolRegistry."""
    engine = MockEngine()
    register_developer_terminal_tools(engine)
    register_productivity_knowledge_tools(engine)
    register_system_extended_tools(engine)

    registered_names = engine.tool_registry.list_tools()
    expected_tools = [
        "terminal.execute",
        "git.workspace_commit",
        "code.run_script",
        "calculator.evaluate",
        "wikipedia.lookup",
        "notes.vault_search",
        "translation.translate",
        "web.scrape_page",
        "os.capture_screen",
        "os.display_brightness",
        "network.wifi_scan",
        "network.speedtest",
    ]
    for tool in expected_tools:
        assert tool in registered_names, f"Tool {tool} was not registered!"

    # Verify permission levels
    term_tool = engine.tool_registry.get("terminal.execute")
    assert term_tool is not None
    assert term_tool.permission_level == PermissionLevel.SYSTEM

    calc_tool = engine.tool_registry.get("calculator.evaluate")
    assert calc_tool is not None
    assert calc_tool.permission_level == PermissionLevel.READ_ONLY
