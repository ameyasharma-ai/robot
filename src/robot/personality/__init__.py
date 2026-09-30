"""Personality manager — hot-swappable persona system.

Personality is external to model weights. The same Qwen model switches
personalities instantly by changing the system prompt and behavior config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot.telemetry import get_logger

logger = get_logger("personality")


@dataclass
class Personality:
    """A complete personality definition."""

    name: str = ""
    system_prompt: str = ""
    speaking_style: str = ""
    vocabulary: str = ""
    behavior: dict[str, Any] = field(default_factory=dict)
    voice_model: str = ""
    tts_settings: dict[str, Any] = field(default_factory=dict)
    tool_permissions: dict[str, bool] = field(default_factory=dict)
    memory_namespace: str = "default"


class PersonalityManager:
    """Loads and manages personality definitions from YAML files."""

    def __init__(self, personality_dir: str = "config/personalities") -> None:
        p = Path(personality_dir)
        if not p.is_absolute():
            # Try current directory first, then robot root
            p = (Path.cwd() / p).resolve()
        self._dir = p
        self._personalities: dict[str, Personality] = {}
        self._active: str = ""

    def load_all(self) -> None:
        """Load all personality YAML files from the configured directory."""
        if not self._dir.exists():
            # Fallback to local package config if cwd differs
            fallback = Path(__file__).resolve().parent.parent.parent.parent / "config" / "personalities"
            if fallback.exists():
                self._dir = fallback
            else:
                logger.warning("personality_dir_missing", path=str(self._dir))
                return

        for yaml_file in self._dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                personality = Personality(
                    name=data.get("name", yaml_file.stem),
                    system_prompt=data.get("system_prompt", ""),
                    speaking_style=data.get("speaking_style", ""),
                    vocabulary=data.get("vocabulary", ""),
                    behavior=data.get("behavior", {}),
                    voice_model=data.get("voice_model", ""),
                    tts_settings=data.get("tts_settings", {}),
                    tool_permissions=data.get("tool_permissions", {}),
                    memory_namespace=data.get("memory_namespace", "default"),
                )
                self._personalities[personality.name] = personality
                logger.info("personality_loaded", name=personality.name)
            except Exception as e:
                logger.error("personality_load_failed", file=str(yaml_file), error=str(e))

    def set_active(self, name: str) -> bool:
        """Switch to a named personality."""
        if name in self._personalities:
            self._active = name
            logger.info("personality_switched", name=name)
            return True
        logger.warning("personality_not_found", name=name)
        return False

    @property
    def active(self) -> Personality | None:
        return self._personalities.get(self._active)

    @property
    def active_name(self) -> str:
        return self._active

    def list_personalities(self) -> list[str]:
        return list(self._personalities.keys())

    def get(self, name: str) -> Personality | None:
        return self._personalities.get(name)
