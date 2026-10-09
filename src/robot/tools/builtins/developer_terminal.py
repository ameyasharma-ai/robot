"""Developer & Terminal Built-in Tools.

Provides shell command execution for agentic IDE tasks, git workspace automation,
script execution, and safe mathematical evaluation.
"""

from __future__ import annotations

import ast
import asyncio
import math
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.developer_terminal")


def _strip_ansi(text: str) -> str:
    """Strip ANSI control codes and color escape sequences from terminal output."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


async def execute_terminal_command(
    command: str,
    cwd: str = ".",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Execute a Linux bash shell command with timeout and output capture.

    Args:
        command: The exact shell command line string to execute.
        cwd: Relative or absolute working directory path (default ".").
        timeout_seconds: Maximum execution time before SIGTERM timeout.

    Returns:
        dict containing stdout, stderr, exit_code, status, and execution_time_ms.
    """
    clean_cmd = (command or "").strip()
    if not clean_cmd:
        return {"status": "error", "error": "No shell command provided.", "exit_code": -1}

    # Resolve target working directory safely
    resolved_cwd = Path(cwd).expanduser().resolve()
    if not resolved_cwd.exists() or not resolved_cwd.is_dir():
        resolved_cwd = Path(".").resolve()

    start_t = asyncio.get_event_loop().time()

    try:
        proc = await asyncio.create_subprocess_shell(
            clean_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_cwd),
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout_seconds)
            )
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            try:
                proc.terminate()
                await proc.wait()
            except Exception:
                pass
            return {
                "status": "timeout",
                "error": f"Command timed out after {timeout_seconds} seconds.",
                "exit_code": -1,
                "command": clean_cmd,
                "cwd": str(resolved_cwd),
            }

        elapsed_ms = round((asyncio.get_event_loop().time() - start_t) * 1000, 1)
        stdout_str = _strip_ansi(stdout_bytes.decode("utf-8", errors="replace")).strip()
        stderr_str = _strip_ansi(stderr_bytes.decode("utf-8", errors="replace")).strip()

        # Truncate extremely large terminal outputs for LLM context friendliness (max 4,000 chars)
        if len(stdout_str) > 4000:
            stdout_str = stdout_str[:2000] + "\n\n... [Output Truncated] ...\n\n" + stdout_str[-2000:]
        if len(stderr_str) > 4000:
            stderr_str = stderr_str[:2000] + "\n\n... [Output Truncated] ...\n\n" + stderr_str[-2000:]

        logger.info("terminal_command_executed", command=clean_cmd[:50], exit_code=exit_code, duration_ms=elapsed_ms)
        return {
            "status": "success" if exit_code == 0 else "failed",
            "command": clean_cmd,
            "cwd": str(resolved_cwd),
            "exit_code": exit_code,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "execution_time_ms": elapsed_ms,
        }
    except Exception as e:
        logger.error("terminal_command_failed", command=clean_cmd[:50], error=str(e))
        return {"status": "error", "error": str(e), "exit_code": -1, "command": clean_cmd}


async def git_workspace_commit(
    action: str = "status",
    message: str = "",
    commit_message: str = "",
    files: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Perform workspace git operations: status, diff, stage, or conventional commit.

    Args:
        action: Git action to perform ('status', 'diff', 'stage', 'commit').
        message: Conventional commit message if action is 'commit'.
        commit_message: Alias for commit message.
        files: Optional list of files to stage if action is 'stage'.

    Returns:
        dict containing operation result details and command output.
    """
    act = (action or kwargs.get("op") or "status").strip().lower()
    eff_msg = (message or commit_message or kwargs.get("msg") or "").strip()
    root_dir = Path(".").resolve()

    if act == "status":
        res = await execute_terminal_command("git status --short", cwd=str(root_dir))
        return {
            "status": "success",
            "action": "status",
            "branch": (await execute_terminal_command("git branch --show-current", cwd=str(root_dir))).get("stdout", "main"),
            "modified_files": res.get("stdout", "").splitlines(),
        }

    elif act == "diff":
        res = await execute_terminal_command("git diff --stat", cwd=str(root_dir))
        return {"status": "success", "action": "diff", "diff_summary": res.get("stdout", "No changes")}

    elif act == "stage":
        target_files = " ".join(files) if files else "."
        res = await execute_terminal_command(f"git add {target_files}", cwd=str(root_dir))
        return {"status": "success", "action": "stage", "staged": target_files, "exit_code": res.get("exit_code", 0)}

    elif act == "commit":
        if not eff_msg:
            return {"status": "error", "error": "A commit message is required to commit changes."}
        clean_msg = eff_msg.replace('"', '\\"')
        res = await execute_terminal_command(f'git commit -m "{clean_msg}"', cwd=str(root_dir))
        return {
            "status": "success" if res.get("exit_code") == 0 else "failed",
            "action": "commit",
            "message": eff_msg,
            "output": res.get("stdout") or res.get("stderr"),
        }

    return {"status": "error", "error": f"Unknown git action '{action}'. Use status, diff, stage, or commit."}


async def code_run_script(
    language: str = "python",
    code: str = "",
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """Execute a Python, Node.js, or Bash script snippet with output capture.

    Args:
        language: Programming language ('python', 'javascript' / 'node', 'bash').
        code: Code snippet text to execute.
        timeout_seconds: Maximum script execution time before termination.

    Returns:
        dict containing execution stdout, stderr, and status.
    """
    clean_code = (code or "").strip()
    if not clean_code:
        return {"status": "error", "error": "No code snippet provided."}

    lang = (language or "python").strip().lower()

    if lang in ("python", "python3", "py"):
        cmd = f"python3 -c {subprocess.list2cmdline([clean_code])}"
    elif lang in ("javascript", "js", "node"):
        cmd = f"node -e {subprocess.list2cmdline([clean_code])}"
    elif lang in ("bash", "sh", "shell"):
        cmd = f"bash -c {subprocess.list2cmdline([clean_code])}"
    else:
        return {"status": "error", "error": f"Unsupported language '{language}'. Use python, javascript, or bash."}

    return await execute_terminal_command(cmd, timeout_seconds=timeout_seconds)


def calculator_evaluate(expression: str) -> dict[str, Any]:
    """Safely evaluate mathematical expressions, unit conversions, and formulas.

    Args:
        expression: Mathematical string expression (e.g. '25 * 4 + sin(30)', '100 USD to EUR').

    Returns:
        dict containing expression, numeric result, and status.
    """
    clean_expr = (expression or "").strip()
    if not clean_expr:
        return {"status": "error", "error": "No mathematical expression provided."}

    # Safe math functions mapping
    safe_dict = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "pow": math.pow,
        "pi": math.pi,
        "e": math.e,
        "round": round,
        "min": min,
        "max": max,
    }

    try:
        # Pre-process percentage notation (e.g., 20% of 150 -> 0.20 * 150)
        expr_mod = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"(\1 / 100.0) * \2", clean_expr, flags=re.IGNORECASE)
        expr_mod = expr_mod.replace("^", "**")

        # Parse AST to ensure zero arbitrary code execution
        node = ast.parse(expr_mod, mode="eval")
        compiled = compile(node, "<string>", "eval")
        result = eval(compiled, {"__builtins__": {}}, safe_dict)

        if isinstance(result, (int, float)):
            if math.isnan(result) or math.isinf(result):
                return {"status": "error", "error": "Result is undefined or infinite.", "expression": clean_expr}
            formatted = round(result, 6) if isinstance(result, float) else result
            return {"status": "success", "expression": clean_expr, "result": formatted}

        return {"status": "success", "expression": clean_expr, "result": str(result)}
    except Exception as e:
        logger.debug("calc_eval_error", expr=clean_expr, error=str(e))
        return {"status": "error", "error": f"Invalid expression '{clean_expr}': {e}", "expression": clean_expr}


def register_developer_terminal_tools(engine: RobotEngine) -> None:
    """Register terminal execution, git workspace, script runner, and calculator tools.

    Args:
        engine: RobotEngine instance containing tool_registry.
    """

    async def _terminal_exec(command: str = "", cwd: str = ".", timeout_seconds: int = 30, **kwargs: Any) -> dict[str, Any]:
        """Execute bash terminal commands on host system.

        Args:
            command: Shell command to execute.
            cwd: Working directory path.
            timeout_seconds: Timeout in seconds.

        Returns:
            dict containing execution status, stdout, and stderr.
        """
        cmd = (command or kwargs.get("cmd") or kwargs.get("shell") or "").strip()
        if not cmd:
            return {"status": "error", "error": "No shell command provided."}
        return await execute_terminal_command(cmd, cwd=cwd, timeout_seconds=timeout_seconds)

    engine.tool_registry.register(
        ToolDefinition(
            name="terminal.execute",
            description="Execute Linux bash terminal shell commands on host system for agentic IDE tasks. Returns stdout, stderr, exit code.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The exact shell command line string to execute"},
                    "cwd": {"type": "string", "description": "Working directory path", "default": "."},
                    "timeout_seconds": {"type": "integer", "description": "Execution timeout in seconds", "default": 30},
                },
                "required": ["command"],
            },
            handler=_terminal_exec,
            permission_level=PermissionLevel.SYSTEM,
            family="terminal",
            timeout_s=35.0,
        )
    )

    async def _git_commit(action: str = "status", commit_message: str = "", files: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        """View git status/diffs, stage files, and create conventional commits.

        Args:
            action: Git action ('status', 'diff', 'stage_all', 'commit').
            commit_message: Message for git commit.
            files: Optional list of files to stage/diff.

        Returns:
            dict containing status output and result.
        """
        act = (action or kwargs.get("op") or kwargs.get("command") or "status").strip()
        msg = (commit_message or kwargs.get("message") or kwargs.get("msg") or "").strip()
        f_list = files or kwargs.get("paths") or kwargs.get("file_list")
        return await git_workspace_commit(action=act, commit_message=msg, files=f_list)

    engine.tool_registry.register(
        ToolDefinition(
            name="git.workspace_commit",
            description="View git status/diffs, stage files, and create conventional git commits automatically.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Git action: 'status', 'diff', 'stage_all', 'commit'", "default": "status"},
                    "commit_message": {"type": "string", "description": "Commit message when action='commit'"},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Optional list of file paths to stage/diff"},
                },
            },
            handler=_git_commit,
            permission_level=PermissionLevel.DEVELOPMENT,
            family="git",
        )
    )

    async def _code_run(code: str = "", language: str = "python", timeout_seconds: int = 15, **kwargs: Any) -> dict[str, Any]:
        """Execute script snippet in Python, JS, or Bash.

        Args:
            code: Script code snippet.
            language: Programming language name.
            timeout_seconds: Max execution duration.

        Returns:
            dict containing stdout and execution result.
        """
        c = (code or kwargs.get("script") or kwargs.get("snippet") or "").strip()
        lang = (language or kwargs.get("lang") or "python").strip()
        return await code_run_script(code=c, language=lang, timeout_seconds=timeout_seconds)

    engine.tool_registry.register(
        ToolDefinition(
            name="code.run_script",
            description="Execute Python, JavaScript, or Bash code snippets in a sandboxed process with stdout capture.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code snippet to run"},
                    "language": {"type": "string", "description": "Programming language: 'python', 'javascript', 'bash'", "default": "python"},
                    "timeout_seconds": {"type": "integer", "description": "Execution timeout in seconds", "default": 15},
                },
                "required": ["code"],
            },
            handler=_code_run,
            permission_level=PermissionLevel.DEVELOPMENT,
            family="code",
            timeout_s=20.0,
        )
    )

    async def _calc_eval(expression: str = "", **kwargs: Any) -> dict[str, Any]:
        """Evaluate mathematical expressions, conversions, and formulas.

        Args:
            expression: Mathematical expression string.

        Returns:
            dict containing expression evaluation result.
        """
        expr = (expression or kwargs.get("expr") or kwargs.get("formula") or kwargs.get("math") or "").strip()
        return calculator_evaluate(expression=expr)

    engine.tool_registry.register(
        ToolDefinition(
            name="calculator.evaluate",
            description="Evaluate complex mathematical expressions, percentages, unit conversions, and formulas accurately.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression string (e.g. '25 * 4 + sin(30)')"},
                },
                "required": ["expression"],
            },
            handler=_calc_eval,
            permission_level=PermissionLevel.READ_ONLY,
            family="calculator",
        )
    )

