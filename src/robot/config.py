"""Configuration management and environment variable synchronization.

Loads baseline YAML settings from config/default.yaml and merges environment
variables from the root .env file. Provides atomic persistence for runtime
settings updates (e.g. OpenAI-compatible API base URL and keys).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

from robot.telemetry import get_logger

logger = get_logger("config")


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
        try:
            llm["context_window"] = int(os.environ["LLM_CONTEXT_WINDOW"])
        except ValueError:
            pass
    if "LLM_MAX_TOKENS" in os.environ:
        try:
            llm["max_tokens"] = int(os.environ["LLM_MAX_TOKENS"])
        except ValueError:
            pass
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
        try:
            gemini["context_window"] = int(os.environ["GEMINI_CONTEXT_WINDOW"])
        except ValueError:
            pass

    # OpenAI-Compatible Provider Overrides
    openai_comp = llm.setdefault("openai_compatible", {})
    if "OPENAI_COMPATIBLE_BASE_URL" in os.environ:
        openai_comp["base_url"] = os.environ["OPENAI_COMPATIBLE_BASE_URL"].strip()
    if "OPENAI_COMPATIBLE_API_KEY" in os.environ:
        openai_comp["api_key"] = os.environ["OPENAI_COMPATIBLE_API_KEY"].strip()
    if "OPENAI_COMPATIBLE_MODEL" in os.environ:
        openai_comp["model"] = os.environ["OPENAI_COMPATIBLE_MODEL"].strip()
    if "OPENAI_COMPATIBLE_CONTEXT_WINDOW" in os.environ:
        try:
            openai_comp["context_window"] = int(os.environ["OPENAI_COMPATIBLE_CONTEXT_WINDOW"])
        except ValueError:
            pass
    if "OPENAI_COMPATIBLE_ENABLE_PROMPT_CACHING" in os.environ:
        openai_comp["enable_prompt_caching"] = os.environ["OPENAI_COMPATIBLE_ENABLE_PROMPT_CACHING"].strip().lower() in ("true", "1", "yes")

    # ─── STT Overrides ──────────────────────────────────────────────────────────
    stt = config.setdefault("stt", {})
    if "WHISPER_SERVER_URL" in os.environ:
        stt["url"] = os.environ["WHISPER_SERVER_URL"].strip()

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


def update_env_variable(key: str, value: str, env_path: str | Path | None = None) -> bool:
    """Update or append an environment variable in the root .env file.

    Args:
        key: Environment variable name (e.g. 'OPENAI_COMPATIBLE_MODEL').
        value: New string value to set.
        env_path: Optional path to the .env file. Defaults to repository root .env.

    Returns:
        bool: True if successfully written, False otherwise.
    """
    root = get_project_root()
    target = Path(env_path) if env_path else root / ".env"
    os.environ[key] = value

    try:
        lines: list[str] = []
        key_found = False
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                lines = f.readlines()

        new_lines: list[str] = []
        prefix = f"{key}="
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(prefix) or stripped.startswith(f"export {prefix}"):
                new_lines.append(f'{key}="{value}"\n')
                key_found = True
            else:
                new_lines.append(line)

        if not key_found:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f'{key}="{value}"\n')

        with open(target, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        logger.info("env_variable_persisted", key=key, target=str(target))
        return True
    except Exception as e:
        logger.error("failed_to_update_env_variable", key=key, error=str(e))
        return False
