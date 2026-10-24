"""Developer & Terminal Built-in Tools.

Provides argv-only command execution for agentic IDE tasks, git workspace
automation, Python script execution, and a strictly sandboxed mathematical
evaluator.

Security model:
- ``terminal.execute`` never invokes a shell.  Only an explicit allowlist of
  developer commands may run, with argv arrays and a workspace-confined cwd.
- ``git.workspace_commit`` uses argv arrays and validates every path; commit
  messages are fed through stdin (``git commit -F -``), never a shell string.
- ``calculator.evaluate`` walks the parsed AST against a node/operator
  allowlist and rejects attributes, subscripts, lambdas, comprehensions,
  f-strings, oversize collections, and unbounded exponents.
"""

from __future__ import annotations

import ast
import asyncio
import math
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.developer_terminal")

# Hard output cap per stream (bytes).  Output is additionally shortened before
# it is handed back to the LLM.
_MAX_STREAM_BYTES = 256 * 1024
_MAX_DISPLAY_CHARS = 4000
_MAX_COMMAND_LENGTH = 8192
_MAX_TIMEOUT_SECONDS = 300.0

# Shell metacharacters are rejected outright even though no shell is used.
_SHELL_META_RE = re.compile(r"[;&|`$><\n\r\\\x00]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Conservative developer command allowlist.
_ALLOWED_PROGRAMS: frozenset[str] = frozenset(
    {"pwd", "ls", "cat", "head", "tail", "echo", "git"}
)

_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {"status", "diff", "log", "add", "commit", "branch"}
)

_GIT_STATUS_FLAGS: frozenset[str] = frozenset(
    {"--short", "-s", "--porcelain", "-b", "--branch", "--long", "-v"}
)
_GIT_DIFF_FLAGS: frozenset[str] = frozenset(
    {"--stat", "--name-only", "--name-status", "--cached", "--staged", "--no-color", "--color=never"}
)
_GIT_LOG_FLAGS: frozenset[str] = frozenset(
    {"--oneline", "--stat", "--decorate", "--graph", "--all", "--no-color"}
)
_GIT_ADD_FLAGS: frozenset[str] = frozenset(
    {"--all", "-A", "--update", "-u", "-N", "--intent-to-add", "--"}
)
_GIT_BRANCH_FLAGS: frozenset[str] = frozenset(
    {"--show-current", "-a", "--all", "-r", "--remotes", "--list", "-v", "--verbose"}
)

# Reading these through the terminal would leak credentials.
_SENSITIVE_NAMES: frozenset[str] = frozenset(
    {".env", ".netrc", ".pgpass", "id_rsa", "id_ed25519", "credentials", "credentials.json"}
)
_SENSITIVE_SUFFIXES: tuple[str, ...] = (".pem", ".key", ".p12", ".pfx")


def _strip_ansi(text: str) -> str:
    """Strip ANSI control codes and color escape sequences from terminal output."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def _workspace_root() -> Path:
    """Return the resolved project/workspace root used for path confinement."""
    return Path.cwd().resolve()


def _resolve_cwd(cwd: str) -> Path:
    """Resolve and confine a requested working directory to the workspace.

    Raises:
        ValueError: If the path escapes the workspace or is not a directory.
    """
    raw = (cwd or ".").strip() or "."
    if "\x00" in raw:
        raise ValueError("Invalid working directory.")
    root = _workspace_root()
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Working directory escapes the workspace root.")
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("Working directory does not exist or is not a directory.")
    return resolved


def _normalize_workspace_path(raw: str, *, allow_root: bool = True) -> str:
    """Validate a user-supplied path and return it relative to the workspace.

    The path must be relative-friendly, free of ``..`` components, and resolve
    (including symlinks) inside the workspace.

    Raises:
        ValueError: If the path is absolute-escaping, traversal-based, or unsafe.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty path argument.")
    if len(text) > 4096:
        raise ValueError("Path argument is too long.")
    if _CONTROL_RE.search(text) or _SHELL_META_RE.search(text):
        raise ValueError(f"Unsafe characters in path argument: {text!r}")

    root = _workspace_root()
    parts = Path(text).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"Path traversal is not allowed: {text!r}")

    candidate = Path(text).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes the workspace: {text!r}")
    if resolved == root and not allow_root:
        raise ValueError("The workspace root is not a valid file argument here.")

    rel = resolved.relative_to(root)
    return "." if str(rel) == "." else str(rel)


def _is_sensitive_file(path: Path) -> bool:
    """Return True when a workspace file is likely to contain credentials."""
    lowered = [part.lower() for part in path.parts]
    if ".git" in lowered or ".ssh" in lowered:
        return True
    name = path.name.lower()
    return name in _SENSITIVE_NAMES or name.endswith(_SENSITIVE_SUFFIXES)


def _shorten(text: str) -> str:
    """Shorten display output to the LLM-friendly character cap."""
    if len(text) <= _MAX_DISPLAY_CHARS:
        return text
    return (
        text[: _MAX_DISPLAY_CHARS // 2]
        + "\n\n... [Output Truncated] ...\n\n"
        + text[-_MAX_DISPLAY_CHARS // 2:]
    )


async def _read_limited(
    stream: asyncio.StreamReader | None, cap: int = _MAX_STREAM_BYTES
) -> tuple[bytes, bool]:
    """Read a subprocess stream while enforcing a hard byte cap.

    Args:
        stream: The pipe to drain (may be ``None``).
        cap: Maximum number of bytes retained.

    Returns:
        Tuple of ``(retained_bytes, truncated)``.
    """
    if stream is None:
        return b"", False

    buffer = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        remaining = cap - len(buffer)
        if remaining > 0:
            buffer.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        else:
            truncated = True
    return bytes(buffer), truncated


def _kill_process_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Send a signal to the process group, falling back to the process itself."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), sig)
        else:  # pragma: no cover - non-POSIX fallback
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate a process group with SIGTERM, then SIGKILL after a grace period."""
    if proc.returncode is not None:
        return
    _kill_process_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.5)
        return
    except asyncio.TimeoutError:
        pass
    _kill_process_group(proc, signal.SIGKILL)
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:  # pragma: no cover - extreme kernel stall
        logger.error("process_group_kill_timeout", pid=proc.pid)


async def _run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    stdin_data: bytes | None = None,
) -> dict[str, Any]:
    """Run an argv array without a shell, with a timeout and bounded output.

    Args:
        argv: Command and arguments.  Never interpreted by a shell.
        cwd: Confined working directory.
        timeout_seconds: Wall-clock limit.
        stdin_data: Optional bytes written to the child's stdin.

    Returns:
        Dict with exit_code, stdout, stderr, timed_out, truncation flags and duration.
    """
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    stdout_task = asyncio.create_task(_read_limited(proc.stdout))
    stderr_task = asyncio.create_task(_read_limited(proc.stderr))

    if stdin_data is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_data)
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=max(0.1, timeout_seconds))
    except asyncio.TimeoutError:
        timed_out = True
        await _terminate_process_group(proc)

    stdout_bytes, stdout_truncated = await asyncio.wait_for(stdout_task, timeout=2.0)
    stderr_bytes, stderr_truncated = await asyncio.wait_for(stderr_task, timeout=2.0)

    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    stdout_str = _strip_ansi(stdout_bytes.decode("utf-8", errors="replace")).strip()
    stderr_str = _strip_ansi(stderr_bytes.decode("utf-8", errors="replace")).strip()

    return {
        "exit_code": -1 if timed_out else (proc.returncode if proc.returncode is not None else -1),
        "stdout": stdout_str,
        "stderr": stderr_str,
        "timed_out": timed_out,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "duration_ms": elapsed_ms,
    }


def _validate_git_subcommand(subcmd: str, args: list[str]) -> None:
    """Validate arguments for an allowlisted git subcommand.

    Raises:
        ValueError: If a flag or path is not allowed.
    """
    if subcmd == "branch":
        if not args:
            raise ValueError("git branch requires --show-current or a list flag.")
        for arg in args:
            if arg not in _GIT_BRANCH_FLAGS:
                raise ValueError(f"git branch argument not allowed: {arg!r}")
        return

    if subcmd == "status":
        for arg in args:
            if arg.startswith("-"):
                if arg not in _GIT_STATUS_FLAGS:
                    raise ValueError(f"git status flag not allowed: {arg!r}")
            else:
                _normalize_workspace_path(arg)
        return

    if subcmd == "diff":
        for arg in args:
            if arg == "--":
                continue
            if arg.startswith("-"):
                if arg in _GIT_DIFF_FLAGS or re.fullmatch(r"-U\d{1,3}", arg):
                    continue
                raise ValueError(f"git diff flag not allowed: {arg!r}")
            _normalize_workspace_path(arg)
        return

    if subcmd == "log":
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg in ("-n", "--max-count"):
                skip_next = True
                continue
            if arg.startswith("-"):
                if (
                    arg in _GIT_LOG_FLAGS
                    or re.fullmatch(r"-n\d{1,4}", arg)
                    or re.fullmatch(r"--max-count=\d{1,4}", arg)
                    or re.fullmatch(r"--since=[\w:+\- ]{1,64}", arg)
                    or re.fullmatch(r"--until=[\w:+\- ]{1,64}", arg)
                ):
                    continue
                raise ValueError(f"git log flag not allowed: {arg!r}")
            _normalize_workspace_path(arg)
        return

    if subcmd == "add":
        paths = [a for a in args if not a.startswith("-")]
        for arg in args:
            if arg.startswith("-") and arg not in _GIT_ADD_FLAGS:
                raise ValueError(f"git add flag not allowed: {arg!r}")
            if not arg.startswith("-"):
                _normalize_workspace_path(arg)
        if not paths and not any(a in ("--all", "-A", "--update", "-u") for a in args):
            raise ValueError("git add requires at least one workspace path or --all.")
        return

    if subcmd == "commit":
        msg_args = 0
        for idx, arg in enumerate(args):
            if arg in ("-m", "--message"):
                msg_args += 1
                if idx + 1 >= len(args) or args[idx + 1].startswith("-"):
                    raise ValueError("git commit -m requires a message.")
                continue
            if arg in ("-a", "--all", "--amend", "--no-edit", "--no-verify"):
                continue
            if arg.startswith("-"):
                raise ValueError(f"git commit flag not allowed: {arg!r}")
            _normalize_workspace_path(arg)
        if msg_args > 1:
            raise ValueError("Only one commit message is allowed.")
        return

    raise ValueError(f"git subcommand not allowed: {subcmd!r}")


def _validate_argv(argv: list[str]) -> list[str]:
    """Validate a tokenized command against the developer allowlist.

    Args:
        argv: Tokenized command (``argv[0]`` is a bare program name).

    Returns:
        Canonical argv with the resolved executable path first.

    Raises:
        ValueError: If the program, flags, or file paths are not permitted.
    """
    if not argv:
        raise ValueError("No command provided.")
    program = argv[0]
    if "/" in program or "\\" in program:
        raise ValueError("Executable paths are not allowed; use a command name.")
    if program not in _ALLOWED_PROGRAMS:
        raise ValueError(
            f"Command {program!r} is not allowlisted. Allowed: "
            + ", ".join(sorted(_ALLOWED_PROGRAMS))
        )

    resolved = shutil.which(program)
    if not resolved:
        raise ValueError(f"Command {program!r} is not installed.")
    if Path(resolved).name != program:
        raise ValueError(f"Resolved executable does not match {program!r}.")

    args = argv[1:]

    if program == "git":
        if not args:
            raise ValueError("A git subcommand is required.")
        subcmd = args[0]
        if subcmd not in _GIT_SUBCOMMANDS:
            raise ValueError(
                f"git subcommand {subcmd!r} is not allowlisted. Allowed: "
                + ", ".join(sorted(_GIT_SUBCOMMANDS))
            )
        _validate_git_subcommand(subcmd, args[1:])
        return [resolved, *args]

    if program == "pwd":
        if args and args not in (["-P"], ["-L"]):
            raise ValueError("pwd accepts no arguments (or -P/-L).")
        return [resolved, *args]

    if program == "echo":
        for arg in args:
            if _CONTROL_RE.search(arg) or _SHELL_META_RE.search(arg):
                raise ValueError("Unsafe characters in echo argument.")
        return [resolved, *args]

    if program in ("ls", "cat", "head", "tail"):
        for arg in args:
            if arg == "--":
                continue
            if arg.startswith("-"):
                if not re.fullmatch(r"-[A-Za-z0-9]{1,8}", arg):
                    raise ValueError(f"Flag not allowed: {arg!r}")
                continue
            normalized = _normalize_workspace_path(arg)
            if program in ("cat", "head", "tail"):
                if _is_sensitive_file(_workspace_root() / normalized):
                    raise ValueError(f"Reading {arg!r} is blocked because it may contain secrets.")
        return [resolved, *args]

    raise ValueError(f"Command {program!r} is not allowlisted.")


async def execute_terminal_command(
    command: str,
    cwd: str = ".",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Execute an allowlisted developer command without a shell.

    Args:
        command: Command line string.  It is tokenized with :mod:`shlex` and
            executed as an argv array; shell metacharacters are rejected.
        cwd: Working directory, which must resolve inside the workspace.
        timeout_seconds: Maximum execution time before process-group kill.

    Returns:
        Dict containing stdout, stderr, exit_code, status, and execution_time_ms.
    """
    clean_cmd = (command or "").strip()
    if not clean_cmd:
        return {"status": "error", "error": "No command provided.", "exit_code": -1}
    if len(clean_cmd) > _MAX_COMMAND_LENGTH:
        return {"status": "error", "error": "Command is too long.", "exit_code": -1}
    if _SHELL_META_RE.search(clean_cmd):
        return {
            "status": "error",
            "error": "Shell metacharacters are not allowed in terminal commands.",
            "exit_code": -1,
        }

    try:
        argv = shlex.split(clean_cmd, posix=True)
    except ValueError as exc:
        return {"status": "error", "error": f"Could not parse command: {exc}", "exit_code": -1}
    if not argv:
        return {"status": "error", "error": "No command provided.", "exit_code": -1}

    try:
        safe_argv = _validate_argv(argv)
        resolved_cwd = _resolve_cwd(cwd)
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "exit_code": -1, "command": argv[0]}

    timeout = max(1.0, min(_MAX_TIMEOUT_SECONDS, float(timeout_seconds or 30)))
    try:
        result = await _run_process(safe_argv, cwd=resolved_cwd, timeout_seconds=timeout)
    except Exception as exc:
        logger.error("terminal_command_failed", program=argv[0], error=str(exc))
        return {
            "status": "error",
            "error": "Failed to execute command.",
            "exit_code": -1,
            "command": argv[0],
        }

    if result["timed_out"]:
        logger.warning("terminal_command_timeout", program=argv[0], timeout_s=timeout)
        return {
            "status": "timeout",
            "error": f"Command timed out after {timeout:g} seconds.",
            "exit_code": -1,
            "command": argv[0],
            "cwd": str(resolved_cwd),
            "stdout": _shorten(result["stdout"]),
            "stderr": _shorten(result["stderr"]),
        }

    exit_code = int(result["exit_code"])
    stdout_str = _shorten(result["stdout"])
    stderr_str = _shorten(result["stderr"])
    if result["stdout_truncated"]:
        stdout_str += "\n\n... [Stream exceeded 256 KiB and was truncated] ..."
    if result["stderr_truncated"]:
        stderr_str += "\n\n... [Stream exceeded 256 KiB and was truncated] ..."

    logger.info(
        "terminal_command_executed",
        program=argv[0],
        argument_count=max(0, len(safe_argv) - 1),
        exit_code=exit_code,
        duration_ms=result["duration_ms"],
    )
    return {
        "status": "success" if exit_code == 0 else "failed",
        "command": argv[0],
        "cwd": str(resolved_cwd),
        "exit_code": exit_code,
        "stdout": stdout_str,
        "stderr": stderr_str,
        "execution_time_ms": result["duration_ms"],
    }


async def _run_git(args: list[str], *, stdin_data: bytes | None = None) -> dict[str, Any]:
    """Run git with an argv array in the workspace root.

    Args:
        args: Git arguments (subcommand first).
        stdin_data: Optional stdin payload (used for commit messages).

    Returns:
        Result dictionary from :func:`_run_process`.
    """
    git = shutil.which("git")
    if not git:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "git is not installed.",
            "timed_out": False,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "duration_ms": 0.0,
        }
    return await _run_process(
        [git, *args], cwd=_workspace_root(), timeout_seconds=30.0, stdin_data=stdin_data
    )


def _validate_commit_message(message: str) -> str:
    """Validate a commit message for safe stdin delivery.

    Newlines and tabs are allowed (multi-line messages); other control
    characters and NUL bytes are rejected.

    Raises:
        ValueError: If the message is empty, oversized, or contains controls.
    """
    clean = (message or "").strip()
    if not clean:
        raise ValueError("A commit message is required to commit changes.")
    if len(clean) > 10000:
        raise ValueError("Commit message is too long (max 10000 characters).")
    if _CONTROL_RE.search(clean):
        raise ValueError("Commit message contains control characters.")
    return clean


async def git_workspace_commit(
    action: str = "status",
    message: str = "",
    commit_message: str = "",
    files: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Perform workspace git operations: status, diff, stage, or commit.

    Args:
        action: Git action to perform ('status', 'diff', 'stage', 'commit').
        message: Conventional commit message if action is 'commit'.
        commit_message: Alias for commit message.
        files: Optional list of files to stage/diff.  Paths are validated and
            passed to git as an argv array after ``--``.

    Returns:
        Dict containing operation result details and command output.
    """
    act = (action or kwargs.get("op") or "status").strip().lower()
    eff_msg = (message or commit_message or kwargs.get("msg") or "").strip()

    if act == "status":
        status_res = await _run_git(["status", "--short"])
        branch_res = await _run_git(["branch", "--show-current"])
        if status_res["timed_out"] or branch_res["timed_out"]:
            return {"status": "error", "error": "git status timed out."}
        branch = branch_res["stdout"].strip() or "main"
        return {
            "status": "success",
            "action": "status",
            "branch": branch,
            "modified_files": status_res["stdout"].splitlines(),
        }

    if act == "diff":
        args = ["diff", "--stat"]
        if files:
            try:
                safe_files = [_normalize_workspace_path(str(f)) for f in files]
            except ValueError as exc:
                return {"status": "error", "error": str(exc)}
            args.extend(["--", *safe_files])
        res = await _run_git(args)
        if res["timed_out"]:
            return {"status": "error", "error": "git diff timed out."}
        if res["exit_code"] != 0:
            return {"status": "error", "error": res["stderr"].strip() or "git diff failed."}
        return {"status": "success", "action": "diff", "diff_summary": res["stdout"] or "No changes"}

    if act == "stage":
        try:
            safe_files = (
                [_normalize_workspace_path(str(f)) for f in files] if files else ["."]
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        res = await _run_git(["add", "--", *safe_files])
        if res["timed_out"]:
            return {"status": "error", "error": "git add timed out."}
        target = " ".join(safe_files)
        return {
            "status": "success" if res["exit_code"] == 0 else "failed",
            "action": "stage",
            "staged": target,
            "exit_code": res["exit_code"],
            "error": None if res["exit_code"] == 0 else (res["stderr"].strip() or "git add failed."),
        }

    if act == "commit":
        try:
            clean_msg = _validate_commit_message(eff_msg)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        # The message is delivered on stdin; arbitrary characters (including
        # $(...), backticks and quotes) are data, never shell syntax.
        res = await _run_git(["commit", "-F", "-"], stdin_data=clean_msg.encode("utf-8"))
        if res["timed_out"]:
            return {"status": "error", "error": "git commit timed out."}
        return {
            "status": "success" if res["exit_code"] == 0 else "failed",
            "action": "commit",
            "message": clean_msg,
            "output": res["stdout"] or res["stderr"],
        }

    return {
        "status": "error",
        "error": f"Unknown git action '{action}'. Use status, diff, stage, or commit.",
    }


async def code_run_script(
    language: str = "python",
    code: str = "",
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """Execute a Python snippet in a workspace temp file without a shell.

    Bash/JavaScript execution is disabled: there is no sandbox available for
    those runtimes, so this tool fails closed rather than becoming a shell
    escape.

    Args:
        language: Programming language ('python' only).
        code: Code snippet text to execute.
        timeout_seconds: Maximum script execution time before termination.

    Returns:
        Dict containing execution stdout, stderr, and status.
    """
    clean_code = (code or "").strip()
    if not clean_code:
        return {"status": "error", "error": "No code snippet provided."}

    lang = (language or "python").strip().lower()
    if lang not in ("python", "python3", "py"):
        return {
            "status": "error",
            "error": (
                f"Language '{language}' is disabled for security. "
                "Only Python snippets are supported."
            ),
        }

    if len(clean_code) > 200_000:
        return {"status": "error", "error": "Code snippet is too large (max 200000 characters)."}

    root = _workspace_root()
    temp_dir = root / ".robot_tmp"
    temp_path: Path | None = None
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        if not temp_dir.resolve().is_relative_to(root):
            return {"status": "error", "error": "Temporary directory escapes the workspace."}
        fd, raw_path = tempfile.mkstemp(prefix="snippet_", suffix=".py", dir=str(temp_dir))
        temp_path = Path(raw_path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(clean_code)

        timeout = max(1.0, min(_MAX_TIMEOUT_SECONDS, float(timeout_seconds or 15)))
        result = await _run_process(
            [sys.executable, str(temp_path)], cwd=root, timeout_seconds=timeout
        )
    except Exception as exc:
        logger.error("code_run_failed", error=str(exc))
        return {"status": "error", "error": "Failed to run Python snippet.", "exit_code": -1}
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    if result["timed_out"]:
        return {
            "status": "timeout",
            "error": f"Script timed out after {timeout:g} seconds.",
            "exit_code": -1,
            "stdout": _shorten(result["stdout"]),
            "stderr": _shorten(result["stderr"]),
        }

    exit_code = int(result["exit_code"])
    return {
        "status": "success" if exit_code == 0 else "failed",
        "language": "python",
        "exit_code": exit_code,
        "stdout": _shorten(result["stdout"]),
        "stderr": _shorten(result["stderr"]),
        "execution_time_ms": result["duration_ms"],
    }


# ---------------------------------------------------------------------------
# Calculator: strict AST allowlist
# ---------------------------------------------------------------------------

_MAX_EXPRESSION_LENGTH = 2000
_MAX_AST_NODES = 500
_MAX_POW_EXPONENT = 1000
_MAX_POW_BITS = 100_000
_MAX_COLLECTION_LENGTH = 64
_MAX_RESULT_BITS = 100_000

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)
_ALLOWED_COMPARE_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_ALLOWED_BOOL_OPS = (ast.And, ast.Or)

_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "pow": pow,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "ceil": math.ceil,
    "floor": math.floor,
}

_ALLOWED_CONSTANTS: dict[str, Any] = {"pi": math.pi, "e": math.e}

_ALLOWED_NODE_TYPES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Load,
    *_ALLOWED_BINOPS,
    *_ALLOWED_UNARY_OPS,
    *_ALLOWED_COMPARE_OPS,
    *_ALLOWED_BOOL_OPS,
)


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses a disallowed AST construct."""


def _const_number(node: ast.AST, depth: int = 0) -> int | float | None:
    """Statically evaluate a constant-only numeric subexpression.

    Returns ``None`` when the node is not a bounded constant expression.

    Args:
        node: AST node to inspect.
        depth: Recursion depth guard.
    """
    if depth > 20:
        return None
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
        inner = _const_number(node.operand, depth + 1)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _const_number(node.left, depth + 1)
        right = _const_number(node.right, depth + 1)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                if abs(right) > _MAX_POW_EXPONENT:
                    return None
                if isinstance(left, int) and isinstance(right, int):
                    if left.bit_length() * max(0, right) > _MAX_POW_BITS:
                        return None
                return left**right
        except (ArithmeticError, ValueError, OverflowError):
            return None
    return None


def _validate_math_node(node: ast.AST) -> None:
    """Validate one AST node against the calculator allowlist.

    Raises:
        UnsafeExpressionError: If the node or its operands are disallowed.
    """
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        raise UnsafeExpressionError(f"Disallowed syntax: {type(node).__name__}")

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise UnsafeExpressionError("Non-finite numeric constants are not allowed.")
            if isinstance(value, int) and value.bit_length() > 4096:
                raise UnsafeExpressionError("Numeric constant is too large.")
            return
        raise UnsafeExpressionError("Only numeric and boolean constants are allowed.")

    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise UnsafeExpressionError("Only reads of allowed names are permitted.")
        if node.id not in _ALLOWED_FUNCTIONS and node.id not in _ALLOWED_CONSTANTS:
            raise UnsafeExpressionError(f"Name '{node.id}' is not allowed.")
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPS):
            raise UnsafeExpressionError("Unary operator is not allowed.")
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError("Binary operator is not allowed.")
        if isinstance(node.op, ast.Pow):
            exponent = _const_number(node.right)
            if exponent is not None and abs(exponent) > _MAX_POW_EXPONENT:
                raise UnsafeExpressionError("Exponent is too large.")
            base = _const_number(node.left)
            if base is not None and exponent is not None:
                if isinstance(base, int) and isinstance(exponent, int):
                    if base.bit_length() * max(0, exponent) > _MAX_POW_BITS:
                        raise UnsafeExpressionError("Power result is too large.")
                elif abs(base) > 1e6 or abs(exponent) > _MAX_POW_EXPONENT:
                    raise UnsafeExpressionError("Power result is too large.")
        if isinstance(node.op, ast.Mult):
            for side, other in ((node.left, node.right), (node.right, node.left)):
                if isinstance(side, (ast.List, ast.Tuple)):
                    count = _const_number(other)
                    if not isinstance(count, int) or count < 0 or count > _MAX_COLLECTION_LENGTH:
                        raise UnsafeExpressionError("Sequence repetition is too large.")
        return

    if isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(op, _ALLOWED_COMPARE_OPS):
                raise UnsafeExpressionError("Comparison operator is not allowed.")
        return

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOL_OPS):
            raise UnsafeExpressionError("Boolean operator is not allowed.")
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise UnsafeExpressionError("Only whitelisted math functions may be called.")
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not allowed.")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise UnsafeExpressionError("Starred arguments are not allowed.")
        if len(node.args) > 32:
            raise UnsafeExpressionError("Too many function arguments.")
        if node.func.id == "round" and len(node.args) > 2:
            raise UnsafeExpressionError("round() accepts at most two arguments.")
        if node.func.id == "pow" and len(node.args) != 2:
            raise UnsafeExpressionError("pow() requires exactly two arguments.")
        return

    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > _MAX_COLLECTION_LENGTH:
            raise UnsafeExpressionError("Literal collection is too large.")
        return


def _validate_math_ast(tree: ast.AST) -> None:
    """Count and validate every node in an expression AST.

    Raises:
        UnsafeExpressionError: On disallowed syntax or excessive node count.
    """
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > _MAX_AST_NODES:
            raise UnsafeExpressionError("Expression is too complex.")
        _validate_math_node(node)


def _eval_math(expression: str) -> Any:
    """Parse, validate, and evaluate a mathematical expression.

    Args:
        expression: Raw expression string.

    Returns:
        Numeric or boolean result.

    Raises:
        UnsafeExpressionError: If validation fails.
    """
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError("Expression is too long (max 2000 characters).")
    tree = ast.parse(expression, mode="eval")
    _validate_math_ast(tree)
    compiled = compile(tree, "<calculator>", "eval")
    return eval(compiled, {"__builtins__": {}}, {**_ALLOWED_FUNCTIONS, **_ALLOWED_CONSTANTS})


def calculator_evaluate(expression: str) -> dict[str, Any]:
    """Safely evaluate mathematical expressions, unit conversions, and formulas.

    The expression is parsed to an AST and checked against a strict allowlist:
    numbers/booleans, whitelisted names, unary/binary operators, comparisons,
    boolean operators, conditionals, whitelisted calls, and small literal
    lists/tuples.  Attributes, subscripts, lambdas, comprehensions, f-strings,
    awaits, dictionaries, and sets are rejected.

    Args:
        expression: Mathematical string expression (e.g. '25 * 4 + sin(30)').

    Returns:
        Dict containing expression, numeric result, and status.
    """
    clean_expr = (expression or "").strip()
    if not clean_expr:
        return {"status": "error", "error": "No mathematical expression provided."}

    try:
        # Pre-process percentage notation (e.g., 20% of 150 -> 0.20 * 150)
        expr_mod = re.sub(
            r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
            r"(\1 / 100.0) * \2",
            clean_expr,
            flags=re.IGNORECASE,
        )
        expr_mod = expr_mod.replace("^", "**")
        result = _eval_math(expr_mod)

        if isinstance(result, bool):
            return {"status": "success", "expression": clean_expr, "result": result}
        if isinstance(result, int):
            if result.bit_length() > _MAX_RESULT_BITS:
                return {
                    "status": "error",
                    "error": "Result is too large.",
                    "expression": clean_expr,
                }
            return {"status": "success", "expression": clean_expr, "result": result}
        if isinstance(result, float):
            if math.isnan(result) or math.isinf(result):
                return {
                    "status": "error",
                    "error": "Result is undefined or infinite.",
                    "expression": clean_expr,
                }
            return {
                "status": "success",
                "expression": clean_expr,
                "result": round(result, 6),
            }

        return {
            "status": "error",
            "error": "Expression must produce a numeric or boolean result.",
            "expression": clean_expr,
        }
    except UnsafeExpressionError as exc:
        logger.warning("calc_unsafe_expression", expr=clean_expr[:120], reason=str(exc))
        return {"status": "error", "error": f"Invalid expression: {exc}", "expression": clean_expr}
    except Exception as exc:
        logger.debug("calc_eval_error", expr=clean_expr[:120], error=str(exc))
        return {
            "status": "error",
            "error": f"Invalid expression '{clean_expr}': {exc}",
            "expression": clean_expr,
        }


def register_developer_terminal_tools(engine: RobotEngine) -> None:
    """Register terminal, git, script runner, and calculator tools.

    Args:
        engine: RobotEngine instance containing tool_registry.
    """

    async def _terminal_exec(
        command: str = "", cwd: str = ".", timeout_seconds: int = 30, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute an allowlisted developer command without a shell.

        Args:
            command: Command to execute.
            cwd: Working directory confined to the workspace.
            timeout_seconds: Timeout in seconds.

        Returns:
            Dict containing execution status, stdout, and stderr.
        """
        cmd = (command or kwargs.get("cmd") or kwargs.get("shell") or "").strip()
        if not cmd:
            return {"status": "error", "error": "No command provided."}
        return await execute_terminal_command(cmd, cwd=cwd, timeout_seconds=timeout_seconds)

    engine.tool_registry.register(
        ToolDefinition(
            name="terminal.execute",
            description=(
                "Execute a safe developer command without a shell. Allowed commands: "
                "pwd, ls, cat, head, tail, echo, and git status/diff/log/add/commit/branch. "
                "Working directory is confined to the workspace. Returns stdout, stderr, exit code."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Allowlisted command line (no shell metacharacters)"},
                    "cwd": {"type": "string", "description": "Working directory inside the workspace", "default": "."},
                    "timeout_seconds": {"type": "integer", "description": "Execution timeout in seconds", "default": 30},
                },
                "required": ["command"],
            },
            handler=_terminal_exec,
            permission_level=PermissionLevel.SYSTEM,
            family="terminal",
            timeout_s=35.0,
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def _git_commit(
        action: str = "status",
        commit_message: str = "",
        files: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """View git status/diffs, stage files, and create conventional commits.

        Args:
            action: Git action ('status', 'diff', 'stage', 'commit').
            commit_message: Message for git commit.
            files: Optional list of files to stage/diff.

        Returns:
            Dict containing status output and result.
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
                    "action": {"type": "string", "description": "Git action: 'status', 'diff', 'stage', 'commit'", "default": "status"},
                    "commit_message": {"type": "string", "description": "Commit message when action='commit'"},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Optional list of workspace-relative file paths to stage/diff"},
                },
            },
            handler=_git_commit,
            permission_level=PermissionLevel.DEVELOPMENT,
            family="git",
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def _code_run(
        code: str = "", language: str = "python", timeout_seconds: int = 15, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute a Python snippet without a shell.

        Args:
            code: Script code snippet.
            language: Programming language name (Python only).
            timeout_seconds: Max execution duration.

        Returns:
            Dict containing stdout and execution result.
        """
        c = (code or kwargs.get("script") or kwargs.get("snippet") or "").strip()
        lang = (language or kwargs.get("lang") or "python").strip()
        return await code_run_script(code=c, language=lang, timeout_seconds=timeout_seconds)

    engine.tool_registry.register(
        ToolDefinition(
            name="code.run_script",
            description="Execute a Python code snippet in an isolated workspace temp file with stdout capture. Bash/JavaScript are disabled.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code snippet to run"},
                    "language": {"type": "string", "description": "Programming language (only 'python' is supported)", "default": "python"},
                    "timeout_seconds": {"type": "integer", "description": "Execution timeout in seconds", "default": 15},
                },
                "required": ["code"],
            },
            handler=_code_run,
            permission_level=PermissionLevel.DEVELOPMENT,
            family="code",
            timeout_s=20.0,
            is_idempotent=False,
            max_retries=0,
        )
    )

    async def _calc_eval(expression: str = "", **kwargs: Any) -> dict[str, Any]:
        """Evaluate mathematical expressions, conversions, and formulas.

        Args:
            expression: Mathematical expression string.

        Returns:
            Dict containing expression evaluation result.
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
