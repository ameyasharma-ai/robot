"""Personality manager — hot-swappable persona system with isolated databases.

Architectural Rationale:
- Personality is external to model weights. The same LLM switches personalities instantly
  by updating the system prompt, behavior config, and per-persona default model presets.
- Each persona owns a completely isolated SQLite database at data/memory/personas/{persona_id}/conversations.db.
  This guarantees absolute memory and conversation isolation between distinct personas,
  preventing data leakage across companion, hacker, assistant, or clinical personas.
- Personas can configure their own default models across all subsystems:
  Default LLM, TTS Engine & Voice, STT Model, Image Model, and Thinking Mode.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot.telemetry import get_logger

logger = get_logger("personality")

# Protected system personas that cannot be deleted without explicit force override
PROTECTED_SYSTEM_PERSONAS: set[str] = {"neo", "luna", "aria", "glados"}


def init_isolated_database(db_path: Path | str) -> Path:
    """Initialize an isolated SQLite database with all tables, triggers, and full-text search indexes.

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        Path: Resolved filesystem path of the initialized database.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            personality TEXT DEFAULT 'neo',
            turn_number INTEGER DEFAULT 0,
            tool_name TEXT,
            tool_result TEXT,
            timestamp TEXT NOT NULL,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp);

        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
            content,
            content=conversations,
            content_rowid=id
        );

        CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
            INSERT INTO conversations_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            personality TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            turn_count INTEGER DEFAULT 0,
            title TEXT DEFAULT 'New Chat',
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            fact TEXT NOT NULL,
            user_id TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
            fact,
            content=facts,
            content_rowid=id
        );

        CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts(rowid, fact) VALUES (new.id, new.fact);
        END;

        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            durable_preferences TEXT NOT NULL DEFAULT '{}',
            temporary_context TEXT NOT NULL DEFAULT '{}',
            inferred_facts TEXT NOT NULL DEFAULT '[]',
            summary_markdown TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()
    return path


@dataclass
class Personality:
    """A complete personality definition with isolated storage and model defaults."""

    name: str = ""
    display_name: str = ""
    system_prompt: str = ""
    speaking_style: str = ""
    vocabulary: str = ""
    behavior: dict[str, Any] = field(default_factory=dict)
    voice_model: str = ""
    tts_settings: dict[str, Any] = field(default_factory=dict)
    tool_permissions: dict[str, bool] = field(default_factory=dict)
    memory_namespace: str = "default"
    avatar_color: str = "#00ffcc"
    description: str = ""
    default_models: dict[str, Any] = field(default_factory=dict)
    is_system: bool = False
    database_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert personality metadata to a serializable dictionary.

        Returns:
            dict[str, Any]: Serialized dictionary of the personality definition.
        """
        return {
            "id": self.name,
            "name": self.name,
            "display_name": self.display_name or self.name.title(),
            "description": self.description,
            "system_prompt": self.system_prompt,
            "speaking_style": self.speaking_style,
            "vocabulary": self.vocabulary,
            "behavior": self.behavior,
            "voice_model": self.voice_model,
            "tts_settings": self.tts_settings,
            "tool_permissions": self.tool_permissions,
            "memory_namespace": self.memory_namespace or self.name,
            "avatar_color": self.avatar_color,
            "default_models": self.default_models,
            "is_system": self.is_system,
            "database_path": self.database_path,
        }


class PersonalityManager:
    """Loads and manages personality definitions and provisions isolated databases."""

    ALIASES: dict[str, str] = {
        "cypher": "neo",
        "matrix": "neo",
        "hacker": "neo",
        "cyber": "neo",
    }

    def __init__(
        self,
        personality_dir: str | Path = "config/personalities",
        config_dir: str | Path | None = None,
        memory_base_dir: str | Path | None = None,
    ) -> None:
        if config_dir is not None:
            personality_dir = config_dir
        p = Path(personality_dir)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        self._dir = p
        mem_dir = Path(memory_base_dir) if memory_base_dir else Path("data/memory")
        if not mem_dir.is_absolute():
            mem_dir = (Path.cwd() / mem_dir).resolve()
        self._memory_base_dir = mem_dir
        self._personalities: dict[str, Personality] = {}
        self._active: str = ""

    def load_all(self) -> None:
        """Load all personality YAML files from the configured directory."""
        if not self._dir.exists():
            fallback = Path(__file__).resolve().parent.parent.parent.parent / "config" / "personalities"
            if fallback.exists():
                self._dir = fallback
            else:
                logger.warning("personality_dir_missing", path=str(self._dir))
                return

        for yaml_file in self._dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f) or {}
                name = data.get("name", yaml_file.stem)
                clean_name = self._resolve_name(name)

                # Default display names and colors
                default_colors = {
                    "neo": "#00ffcc",
                    "luna": "#ff69b4",
                    "aria": "#aa77ff",
                    "glados": "#ffaa00",
                }
                avatar_col = data.get("avatar_color") or default_colors.get(clean_name, "#00ffcc")
                is_sys = data.get("is_system", clean_name in PROTECTED_SYSTEM_PERSONAS)

                # Resolve per-persona isolated database path and ensure initialization
                db_path = self.ensure_isolated_database(clean_name)

                # Resolve default models
                def_models = data.get("default_models", {})
                if not isinstance(def_models, dict):
                    def_models = {}

                # Fallback model defaults for built-ins if omitted
                if clean_name == "luna":
                    def_models.setdefault("llm", "qwen3.5-4b")
                    def_models.setdefault("tts_engine", "kokoro")
                    def_models.setdefault("tts_voice", "af_heart")
                    def_models.setdefault("stt", "base.en")
                    def_models.setdefault("image", "dreamshaper-v8")
                    def_models.setdefault("thinking_mode", False)
                elif clean_name == "neo":
                    def_models.setdefault("llm", "qwen3.5-4b")
                    def_models.setdefault("tts_engine", "piper")
                    def_models.setdefault("tts_voice", "en-us-amy-medium")
                    def_models.setdefault("stt", "base.en")
                    def_models.setdefault("image", "dreamshaper-v8")
                    def_models.setdefault("thinking_mode", True)
                elif clean_name == "aria":
                    def_models.setdefault("llm", "qwen3.5-4b")
                    def_models.setdefault("tts_engine", "kokoro")
                    def_models.setdefault("tts_voice", "af_bella")
                    def_models.setdefault("stt", "base.en")
                    def_models.setdefault("image", "dreamshaper-v8")
                    def_models.setdefault("thinking_mode", False)
                elif clean_name == "glados":
                    def_models.setdefault("llm", "qwen3.5-4b")
                    def_models.setdefault("tts_engine", "piper")
                    def_models.setdefault("tts_voice", "en_US-lessac-medium")
                    def_models.setdefault("stt", "base.en")
                    def_models.setdefault("image", "dreamshaper-v8")
                    def_models.setdefault("thinking_mode", True)

                personality = Personality(
                    name=clean_name,
                    display_name=data.get("display_name", clean_name.title()),
                    system_prompt=data.get("system_prompt", ""),
                    speaking_style=data.get("speaking_style", ""),
                    vocabulary=data.get("vocabulary", ""),
                    behavior=data.get("behavior", {}),
                    voice_model=data.get("voice_model", def_models.get("tts_voice", "")),
                    tts_settings=data.get("tts_settings", {}),
                    tool_permissions=data.get("tool_permissions", {}),
                    memory_namespace=data.get("memory_namespace", clean_name),
                    avatar_color=avatar_col,
                    description=data.get("description", ""),
                    default_models=def_models,
                    is_system=is_sys,
                    database_path=str(db_path),
                )
                self._personalities[clean_name] = personality
                logger.info("personality_loaded", name=clean_name, db=str(db_path))
            except Exception as e:
                logger.error("personality_load_failed", file=str(yaml_file), error=str(e))

    def _resolve_name(self, name: str) -> str:
        """Normalize personality name via alias mapping and case-insensitivity."""
        clean = (name or "").strip().lower()
        return self.ALIASES.get(clean, clean)

    def get_database_path(self, name: str) -> Path:
        """Resolve the path of the isolated database for a persona.

        Args:
            name: Personality name identifier.

        Returns:
            Path: Path to the persona's isolated conversations.db file.
        """
        clean = self._resolve_name(name)
        return self._memory_base_dir / "personas" / clean / "conversations.db"

    def ensure_isolated_database(self, name: str) -> Path:
        """Ensure the isolated SQLite database for a persona exists and is initialized.

        Args:
            name: Personality identifier.

        Returns:
            Path: Resolved path to the initialized database.
        """
        db_path = self.get_database_path(name)
        if not db_path.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # If neo and baseline conversations.db exists, migrate as starting point
            clean = self._resolve_name(name)
            legacy_db = Path("data/database/conversations.db")
            if clean == "neo" and legacy_db.exists():
                try:
                    shutil.copy2(str(legacy_db), str(db_path))
                    logger.info("migrated_legacy_database_for_neo", src=str(legacy_db), dst=str(db_path))
                except Exception as e:
                    logger.warning("legacy_db_migration_failed", error=str(e))
                    init_isolated_database(db_path)
            else:
                init_isolated_database(db_path)
                logger.info("isolated_database_created", persona=clean, path=str(db_path))
        return db_path

    def get_persona_stats(self, name: str) -> dict[str, Any]:
        """Query statistics from the persona's isolated database.

        Args:
            name: Personality identifier.

        Returns:
            dict[str, Any]: Dictionary containing db_path, size_bytes, turns, sessions, facts.
        """
        clean = self._resolve_name(name)
        db_path = self.get_database_path(clean)
        size_bytes = db_path.stat().st_size if db_path.exists() else 0
        turn_count = 0
        session_count = 0
        fact_count = 0

        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM conversations")
                turn_count = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM sessions")
                session_count = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM facts")
                fact_count = c.fetchone()[0]
                conn.close()
            except Exception as e:
                logger.debug("persona_stats_query_error", persona=clean, error=str(e))

        return {
            "database_path": str(db_path),
            "size_bytes": size_bytes,
            "turns": turn_count,
            "sessions": session_count,
            "facts": fact_count,
        }

    def create_persona(self, data: dict[str, Any]) -> Personality:
        """Create a new persona definition and provision its isolated database.

        Args:
            data: Dictionary containing persona fields:
                  id/name, display_name, description, system_prompt,
                  speaking_style, vocabulary, default_models, avatar_color.

        Returns:
            Personality: The created and initialized Personality instance.

        Raises:
            ValueError: If name is missing or invalid, or persona already exists.
        """
        raw_name = data.get("id") or data.get("name") or ""
        name = re.sub(r"[^a-zA-Z0-9_\-]", "", raw_name).strip().lower()
        if not name:
            raise ValueError("Persona identifier (name/id) is required and must contain alphanumeric characters.")

        if name in self._personalities:
            raise ValueError(f"Persona '{name}' already exists.")

        display_name = (data.get("display_name") or name.title()).strip()
        system_prompt = (data.get("system_prompt") or f"You are {display_name}, an AI assistant.").strip()
        speaking_style = (data.get("speaking_style") or "helpful, friendly, concise").strip()
        vocabulary = (data.get("vocabulary") or "clear, engaging").strip()
        avatar_color = (data.get("avatar_color") or "#00ffcc").strip()
        description = (data.get("description") or f"{display_name} custom personality").strip()

        # Model defaults
        raw_models = data.get("default_models", {})
        default_models = {
            "llm": raw_models.get("llm", "qwen3.5-4b"),
            "tts_engine": raw_models.get("tts_engine", "kokoro"),
            "tts_voice": raw_models.get("tts_voice", "af_heart"),
            "stt": raw_models.get("stt", "base.en"),
            "image": raw_models.get("image", "dreamshaper-v8"),
            "thinking_mode": bool(raw_models.get("thinking_mode", False)),
        }

        # 1. Provision isolated database immediately
        db_path = self.ensure_isolated_database(name)

        # 2. Build YAML payload
        yaml_data = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "system_prompt": system_prompt,
            "speaking_style": speaking_style,
            "vocabulary": vocabulary,
            "avatar_color": avatar_color,
            "voice_model": default_models["tts_voice"],
            "tts_settings": {
                "speed": 1.0,
                "pitch": 1.0,
            },
            "default_models": default_models,
            "memory_namespace": name,
            "is_system": False,
        }

        yaml_file = self._dir / f"{name}.yaml"
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(yaml_file, "w") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False)

        personality = Personality(
            name=name,
            display_name=display_name,
            system_prompt=system_prompt,
            speaking_style=speaking_style,
            vocabulary=vocabulary,
            voice_model=default_models["tts_voice"],
            tts_settings={"speed": 1.0, "pitch": 1.0},
            memory_namespace=name,
            avatar_color=avatar_color,
            description=description,
            default_models=default_models,
            is_system=False,
            database_path=str(db_path),
        )

        self._personalities[name] = personality
        logger.info("persona_created_successfully", name=name, db=str(db_path))
        return personality

    def update_persona(self, name: str, data: dict[str, Any]) -> Personality:
        """Update an existing persona definition and model defaults.

        Args:
            name: Persona identifier.
            data: Fields to update.

        Returns:
            Personality: Updated Personality instance.

        Raises:
            ValueError: If persona does not exist.
        """
        clean = self._resolve_name(name)
        pers = self._personalities.get(clean)
        if not pers:
            raise ValueError(f"Persona '{name}' does not exist.")

        if "display_name" in data:
            pers.display_name = data["display_name"].strip()
        if "description" in data:
            pers.description = data["description"].strip()
        if "system_prompt" in data:
            pers.system_prompt = data["system_prompt"].strip()
        if "speaking_style" in data:
            pers.speaking_style = data["speaking_style"].strip()
        if "vocabulary" in data:
            pers.vocabulary = data["vocabulary"].strip()
        if "avatar_color" in data:
            pers.avatar_color = data["avatar_color"].strip()
        if "default_models" in data and isinstance(data["default_models"], dict):
            pers.default_models.update(data["default_models"])
            if "tts_voice" in pers.default_models:
                pers.voice_model = pers.default_models["tts_voice"]

        # Write updated YAML
        yaml_file = self._dir / f"{clean}.yaml"
        yaml_data = {
            "name": clean,
            "display_name": pers.display_name,
            "description": pers.description,
            "system_prompt": pers.system_prompt,
            "speaking_style": pers.speaking_style,
            "vocabulary": pers.vocabulary,
            "avatar_color": pers.avatar_color,
            "voice_model": pers.voice_model,
            "tts_settings": pers.tts_settings,
            "default_models": pers.default_models,
            "memory_namespace": pers.memory_namespace or clean,
            "is_system": pers.is_system,
        }
        with open(yaml_file, "w") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False)

        logger.info("persona_updated", name=clean)
        return pers

    def delete_persona(self, name: str, delete_database: bool = True, force: bool = False) -> bool:
        """Delete a persona and its isolated database.

        Args:
            name: Personality identifier.
            delete_database: Whether to delete the isolated SQLite database file.
            force: Force deletion of protected system personas.

        Returns:
            bool: True if deletion succeeded.

        Raises:
            ValueError: If trying to delete active persona or protected system persona without force.
        """
        clean = self._resolve_name(name)
        if clean not in self._personalities:
            raise ValueError(f"Persona '{name}' does not exist.")

        if clean == self.active_name:
            raise ValueError(f"Cannot delete active persona '{name}'. Switch to another persona before deleting.")

        pers = self._personalities[clean]
        if pers.is_system and not force:
            raise ValueError(f"Cannot delete built-in system persona '{name}'.")

        # 1. Remove YAML file
        yaml_file = self._dir / f"{clean}.yaml"
        if yaml_file.exists():
            yaml_file.unlink()

        # 2. Delete isolated database directory if requested
        if delete_database:
            persona_dir = self._memory_base_dir / "personas" / clean
            if persona_dir.exists():
                try:
                    shutil.rmtree(str(persona_dir))
                    logger.info("isolated_database_deleted", persona=clean, dir=str(persona_dir))
                except Exception as e:
                    logger.warning("delete_database_dir_failed", dir=str(persona_dir), error=str(e))

        # 3. Remove from memory
        del self._personalities[clean]
        logger.info("persona_deleted", name=clean)
        return True

    def get_persisted_active(self, default: str = "neo") -> str:
        """Get persisted active personality name from data/active_personality.txt if available."""
        persisted_file = Path("data/active_personality.txt")
        if persisted_file.exists():
            try:
                raw_name = persisted_file.read_text().strip()
                name = self._resolve_name(raw_name)
                if name and name in self._personalities:
                    return name
            except Exception:
                pass
        return default

    def set_active(self, name: str) -> bool:
        """Switch to a named personality and persist selection."""
        target = self._resolve_name(name)
        if target in self._personalities:
            self._active = target
            try:
                p_file = Path("data/active_personality.txt")
                p_file.parent.mkdir(parents=True, exist_ok=True)
                p_file.write_text(target)
            except Exception as e:
                logger.debug("persist_personality_failed", error=str(e))
            logger.info("personality_switched", name=target)
            return True
        logger.warning("personality_not_found", name=name, target=target)
        return False

    @property
    def active(self) -> Personality | None:
        if not self._active:
            self._active = self.get_persisted_active("neo")
        return self._personalities.get(self._active)

    @property
    def active_name(self) -> str:
        if not self._active:
            self._active = self.get_persisted_active("neo")
        return self._active

    @property
    def active_prompt(self) -> str:
        """System prompt of the currently active personality, or empty string."""
        p = self.active
        return p.system_prompt if p else ""

    def list_personalities(self) -> list[str]:
        return list(self._personalities.keys())

    def list_personas_detailed(self) -> list[dict[str, Any]]:
        """List all personas with full metadata, isolated database statistics, and active status.

        Returns:
            list[dict[str, Any]]: List of persona dictionaries.
        """
        result = []
        active_key = self.active_name
        for name, pers in self._personalities.items():
            d = pers.to_dict()
            d["is_active"] = (name == active_key)
            stats = self.get_persona_stats(name)
            d.update(stats)
            result.append(d)
        return result

    def get(self, name: str) -> Personality | None:
        target = self._resolve_name(name)
        return self._personalities.get(target)
