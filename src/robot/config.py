"""Configuration management and environment variable synchronization.

Loads baseline YAML settings from config/default.yaml and merges environment
variables from the root .env file. Provides atomic persistence for runtime
settings updates (e.g. OpenAI-compatible API base URL and keys).
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

from robot.telemetry import get_logger

logger = get_logger("config")

#: Serializes read-modify-write cycles on shared runtime files (.env, YAML).
_FILE_WRITE_LOCK = threading.Lock()

#: Environment variable names must be shell/POSIX identifiers; this also keeps a
#: caller-supplied key from injecting a second assignment into the .env file.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns:
        Path: Absolute path to the repository root directory.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_env_file(env_path: str | Path | None = None) -> Path:
    """Load the root .env file into os.environ.

    Args:
        env_path: Optional explicit path to the .env file. Defaults to
            the .env file in the repository root.

    Returns:
        Path: The resolved path to the loaded .env file.
    """
    root = get_project_root()
    target_path = Path(env_path) if env_path else root / ".env"
    if target_path.exists():
        load_dotenv(dotenv_path=target_path, override=True)
        logger.debug("env_file_loaded", path=str(target_path))
    else:
        logger.debug("env_file_not_found_skipped", path=str(target_path))
    return target_path


def load_system_config(config_path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load system configuration from YAML and override with environment variables.

    Args:
        config_path: Path to the base YAML configuration file.

    Returns:
        dict[str, Any]: Consolidated configuration dictionary with environment overrides.
    """
    load_env_file()
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = get_project_root() / cfg_file

    config: dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("config_file_not_found_using_defaults", path=str(cfg_file))

    # Apply environment variable overrides
    _apply_env_overrides(config)
    return config


def _env_int(name: str) -> int | None:
    """Read an integer environment override, logging malformed values.

    Args:
        name: Environment variable name.

    Returns:
        int | None: Parsed value, or ``None`` when unset or invalid.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("invalid_env_override_ignored", variable=name, value=raw[:64])
        return None


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Merge environment variables into the nested configuration dictionary.

    Args:
        config: Mutable configuration dictionary to update in-place.
    """
    # ─── LLM Overrides ──────────────────────────────────────────────────────────
    llm = config.setdefault("llm", {})
    if "LLM_PROVIDER" in os.environ:
        llm["provider"] = os.environ["LLM_PROVIDER"].strip().lower()
    if "LLAMA_SERVER_URL" in os.environ:
        llm["base_url"] = os.environ["LLAMA_SERVER_URL"].strip()
    if "LLM_MODEL" in os.environ:
        llm["model"] = os.environ["LLM_MODEL"].strip()
    if "LLM_CONTEXT_WINDOW" in os.environ:
        value = _env_int("LLM_CONTEXT_WINDOW")
        if value is not None:
            llm["context_window"] = value
    if "LLM_MAX_TOKENS" in os.environ:
        value = _env_int("LLM_MAX_TOKENS")
        if value is not None:
            llm["max_tokens"] = value
    if "LLM_MODEL_PATH" in os.environ:
        llm["model_path"] = os.environ["LLM_MODEL_PATH"].strip()
    if "LLM_MULTIMODAL" in os.environ:
        llm["multimodal"] = os.environ["LLM_MULTIMODAL"].strip().lower() in ("true", "1", "yes")
    if "LLM_MMPROJ_PATH" in os.environ:
        llm["mmproj_path"] = os.environ["LLM_MMPROJ_PATH"].strip()
    if "LLM_ENABLE_THINKING" in os.environ:
        llm["enable_thinking"] = os.environ["LLM_ENABLE_THINKING"].strip().lower() in ("true", "1", "yes")

    # Gemini Overrides
    gemini = llm.setdefault("gemini", {})
    if "GEMINI_API_KEY" in os.environ:
        gemini["api_key"] = os.environ["GEMINI_API_KEY"].strip()
    if "GEMINI_BASE_URL" in os.environ:
        gemini["base_url"] = os.environ["GEMINI_BASE_URL"].strip()
    if "GEMINI_MODEL" in os.environ:
        gemini["model"] = os.environ["GEMINI_MODEL"].strip()
    if "GEMINI_CONTEXT_WINDOW" in os.environ:
        value = _env_int("GEMINI_CONTEXT_WINDOW")
        if value is not None:
            gemini["context_window"] = value

    # OpenAI-Compatible Provider Overrides
    openai_comp = llm.setdefault("openai_compatible", {})
    if "OPENAI_COMPATIBLE_BASE_URL" in os.environ:
        openai_comp["base_url"] = os.environ["OPENAI_COMPATIBLE_BASE_URL"].strip()
    if "OPENAI_COMPATIBLE_API_KEY" in os.environ:
        openai_comp["api_key"] = os.environ["OPENAI_COMPATIBLE_API_KEY"].strip()
    if "OPENAI_COMPATIBLE_MODEL" in os.environ:
        openai_comp["model"] = os.environ["OPENAI_COMPATIBLE_MODEL"].strip().strip('"').strip("'")
    
    # Priority check for active_openai_model.txt persistence
    persisted_model_file = get_project_root() / "data" / "active_openai_model.txt"
    if persisted_model_file.exists():
        try:
            persisted_val = persisted_model_file.read_text(encoding="utf-8").strip().strip('"').strip("'")
            if persisted_val:
                openai_comp["model"] = persisted_val
        except Exception:
            pass

    if "OPENAI_COMPATIBLE_CONTEXT_WINDOW" in os.environ:
        value = _env_int("OPENAI_COMPATIBLE_CONTEXT_WINDOW")
        if value is not None:
            openai_comp["context_window"] = value
    if "OPENAI_COMPATIBLE_ENABLE_PROMPT_CACHING" in os.environ:
        openai_comp["enable_prompt_caching"] = os.environ["OPENAI_COMPATIBLE_ENABLE_PROMPT_CACHING"].strip().lower() in ("true", "1", "yes")

    # OpenRouter Provider Overrides
    openrouter = llm.setdefault("openrouter", {})
    if "OPENROUTER_BASE_URL" in os.environ:
        openrouter["base_url"] = os.environ["OPENROUTER_BASE_URL"].strip()
    if "OPENROUTER_API_KEY" in os.environ:
        openrouter["api_key"] = os.environ["OPENROUTER_API_KEY"].strip()
    if "OPENROUTER_MODEL" in os.environ:
        openrouter["model"] = os.environ["OPENROUTER_MODEL"].strip().strip('"').strip("'")

    # AICredits Provider Overrides
    aicredits = llm.setdefault("aicredits", {})
    if "AICREDITS_BASE_URL" in os.environ:
        aicredits["base_url"] = os.environ["AICREDITS_BASE_URL"].strip()
    if "AICREDITS_API_KEY" in os.environ:
        aicredits["api_key"] = os.environ["AICREDITS_API_KEY"].strip()
    if "AICREDITS_MODEL" in os.environ:
        aicredits["model"] = os.environ["AICREDITS_MODEL"].strip().strip('"').strip("'")

    # ─── STT Overrides ──────────────────────────────────────────────────────────
    stt = config.setdefault("stt", {})
    if "WHISPER_SERVER_URL" in os.environ:
        stt["url"] = os.environ["WHISPER_SERVER_URL"].strip()
    if "STT_MODEL" in os.environ:
        stt["model"] = os.environ["STT_MODEL"].strip()
    if "STT_MODEL_PATH" in os.environ:
        stt["model_path"] = os.environ["STT_MODEL_PATH"].strip()

    # ─── TTS Overrides ──────────────────────────────────────────────────────────
    tts = config.setdefault("tts", {})
    if "PIPER_MODEL_PATH" in os.environ:
        tts["model_path"] = os.environ["PIPER_MODEL_PATH"].strip()

    # ─── Audio Overrides ────────────────────────────────────────────────────────
    audio = config.setdefault("audio", {})
    if "AUDIO_SAMPLE_RATE" in os.environ:
        try:
            audio["sample_rate"] = int(os.environ["AUDIO_SAMPLE_RATE"])
        except ValueError:
            pass
    if "AUDIO_SILENCE_THRESHOLD_MS" in os.environ:
        try:
            audio["silence_threshold_ms"] = int(os.environ["AUDIO_SILENCE_THRESHOLD_MS"])
        except ValueError:
            pass
    if "AUDIO_BARGE_IN_ENABLED" in os.environ:
        audio["barge_in_enabled"] = os.environ["AUDIO_BARGE_IN_ENABLED"].strip().lower() in ("true", "1", "yes")

    # ─── Vision Overrides ───────────────────────────────────────────────────────
    vision = config.setdefault("vision", {})
    if "CAMERA_ENABLED" in os.environ:
        vision["enabled"] = os.environ["CAMERA_ENABLED"].strip().lower() in ("true", "1", "yes")
    if "CAMERA_INDEX" in os.environ:
        try:
            vision["camera_index"] = int(os.environ["CAMERA_INDEX"])
        except ValueError:
            pass

    # ─── API & Logging Overrides ────────────────────────────────────────────────
    api = config.setdefault("api", {})
    if "ROBOT_API_HOST" in os.environ:
        api["host"] = os.environ["ROBOT_API_HOST"].strip()
    if "ROBOT_API_PORT" in os.environ:
        try:
            api["port"] = int(os.environ["ROBOT_API_PORT"])
        except ValueError:
            pass

    logging_cfg = config.setdefault("logging", {})
    if "LOG_LEVEL" in os.environ:
        logging_cfg["level"] = os.environ["LOG_LEVEL"].strip().upper()


def atomic_write_text(path: str | Path, text: str, *, mode: int = 0o600) -> None:
    """Write ``text`` to ``path`` atomically with owner-only permissions.

    A crash mid-write must never leave a truncated ``.env`` or YAML file behind,
    and secrets files must not be world-readable for even a moment. The data is
    written to a sibling temporary file, fsynced, and moved into place with
    :func:`os.replace` (atomic on POSIX). Failures propagate to the caller.

    Args:
        path: Destination file.
        text: Complete new contents.
        mode: Permission bits applied before the file is moved into place.

    Raises:
        OSError: When the temporary file cannot be written or the replace fails.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_yaml_atomic(path: str | Path, data: dict[str, Any]) -> None:
    """Persist a configuration mapping as YAML without truncating in place.

    Args:
        path: Destination YAML file.
        data: Mapping to serialize.
    """
    rendered = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    with _FILE_WRITE_LOCK:
        atomic_write_text(path, rendered, mode=0o600)


def _escape_dotenv_value(value: str) -> str:
    """Escape a value so it survives a quoted ``.env`` assignment intact.

    Args:
        value: Raw value.

    Returns:
        str: Backslash/quote-escaped value safe to place between double quotes.

    Raises:
        ValueError: If the value contains NUL or a line break, which would allow
            a second assignment to be injected into the file.
    """
    if any(ch in value for ch in ("\x00", "\n", "\r")):
        raise ValueError("Environment values may not contain NUL or line breaks.")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def update_env_variable(key: str, value: str, env_path: str | Path | None = None) -> bool:
    """Update or append an environment variable in the root .env file.

    The file is updated atomically with ``0600`` permissions and the value is
    quoted/escaped so a key or model name containing quotes cannot corrupt the
    file or inject another assignment.

    Args:
        key: Environment variable name (e.g. 'OPENAI_COMPATIBLE_MODEL').
        value: New string value to set.
        env_path: Optional path to the .env file. Defaults to repository root .env.

    Returns:
        bool: True if successfully written, False otherwise.
    """
    if not _ENV_KEY_RE.match(key or ""):
        logger.error("invalid_env_variable_name", key=key)
        return False
    try:
        escaped = _escape_dotenv_value(str(value))
    except ValueError as exc:
        logger.error("invalid_env_variable_value", key=key, error=str(exc))
        return False

    root = get_project_root()
    target = Path(env_path) if env_path else root / ".env"
    os.environ[key] = str(value)

    try:
        with _FILE_WRITE_LOCK:
            lines: list[str] = []
            if target.exists():
                with open(target, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            replacement = f'{key}="{escaped}"\n'
            new_lines: list[str] = []
            key_found = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
                    new_lines.append(replacement)
                    key_found = True
                else:
                    new_lines.append(line)

            if not key_found:
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines.append("\n")
                new_lines.append(replacement)

            atomic_write_text(target, "".join(new_lines), mode=0o600)

        logger.info("env_variable_persisted", key=key, target=str(target))
        return True
    except Exception as e:
        logger.error("failed_to_update_env_variable", key=key, error=str(e))
        return False
