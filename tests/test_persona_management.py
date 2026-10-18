"""Unit and integration tests for AI Persona Management and Isolated Databases.

Tests verification of persona CRUD lifecycle, automated SQLite database isolation per persona,
model default configurations, and API endpoints.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
import pytest
from fastapi.testclient import TestClient

from robot.personality import PersonalityManager, Personality
from robot.memory import ConversationHistory


@pytest.fixture
def temp_persona_env(tmp_path: Path):
    """Create isolated temporary personalities and memory directory fixtures.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Yields:
        Tuple containing config directory and memory directory.
    """
    config_dir = tmp_path / "config" / "personalities"
    config_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = tmp_path / "data" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Seed one test persona
    test_yaml = config_dir / "test_bot.yaml"
    test_yaml.write_text(
        """name: test_bot
display_name: Test Bot
description: Test companion persona
system_prompt: You are a friendly test assistant.
avatar_color: "#10b981"
voice_model: af_heart
default_models:
  llm: "qwen3.5-4b"
  tts_engine: "kokoro"
  tts_voice: "af_heart"
  stt: "base.en"
  image: "dreamshaper-v8"
  thinking_mode: false
"""
    )

    yield config_dir, memory_dir


def test_persona_manager_initialization_and_db_path(temp_persona_env: tuple[Path, Path]) -> None:
    """Verify that PersonalityManager loads personas and provisions isolated database paths.

    Args:
        temp_persona_env: Temporary directories tuple fixture.
    """
    config_dir, memory_dir = temp_persona_env
    mgr = PersonalityManager(config_dir=config_dir, memory_base_dir=memory_dir)
    mgr.load_all()

    assert "test_bot" in mgr.list_personalities()
    pers = mgr.get("test_bot")
    assert pers is not None
    assert pers.display_name == "Test Bot"
    assert pers.default_models["tts_voice"] == "af_heart"

    # Verify isolated database path
    expected_db = memory_dir / "personas" / "test_bot" / "conversations.db"
    assert pers.database_path == str(expected_db)
    assert expected_db.exists()

    # Verify SQLite schema tables are provisioned
    conn = sqlite3.connect(str(expected_db))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "conversations" in tables
    assert "facts" in tables
    assert "sessions" in tables


def test_persona_creation_and_database_isolation(temp_persona_env: tuple[Path, Path]) -> None:
    """Verify creating a new persona creates an isolated database with zero cross-talk.

    Args:
        temp_persona_env: Temporary directories tuple fixture.
    """
    config_dir, memory_dir = temp_persona_env
    mgr = PersonalityManager(config_dir=config_dir, memory_base_dir=memory_dir)
    mgr.load_all()

    new_persona_data: dict[str, Any] = {
        "id": "ruby",
        "display_name": "Ruby Assistant",
        "description": "Cybersecurity assistant",
        "system_prompt": "You are Ruby, an elite hacker AI.",
        "avatar_color": "#ff3366",
        "default_models": {
            "llm": "gemini-2.0-flash",
            "tts_engine": "piper",
            "tts_voice": "en-us-amy-medium",
            "stt": "tiny.en",
            "image": "flux-schnell",
            "thinking_mode": True,
        },
    }

    created = mgr.create_persona(new_persona_data)
    assert created.name == "ruby"
    assert created.display_name == "Ruby Assistant"
    assert created.default_models["llm"] == "gemini-2.0-flash"
    assert created.default_models["thinking_mode"] is True

    ruby_db = memory_dir / "personas" / "ruby" / "conversations.db"
    test_db = memory_dir / "personas" / "test_bot" / "conversations.db"
    assert ruby_db.exists()

    # Insert a message directly into Ruby's isolated database
    conn_ruby = sqlite3.connect(str(ruby_db))
    conn_ruby.execute(
        "INSERT INTO conversations (session_id, role, content, personality, timestamp) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        ("sess_ruby_1", "user", "Hello Ruby, this is a secret key 12345", "ruby"),
    )
    conn_ruby.commit()

    # Query Ruby's isolated database
    cur_ruby = conn_ruby.cursor()
    cur_ruby.execute("SELECT content FROM conversations WHERE session_id = 'sess_ruby_1'")
    ruby_rows = cur_ruby.fetchall()
    conn_ruby.close()

    assert len(ruby_rows) == 1
    assert "secret key 12345" in ruby_rows[0][0]

    # Query test_bot's isolated database - it must have 0 conversations (absolute isolation)
    conn_test = sqlite3.connect(str(test_db))
    cur_test = conn_test.cursor()
    cur_test.execute("SELECT content FROM conversations")
    test_rows = cur_test.fetchall()
    conn_test.close()

    assert len(test_rows) == 0


def test_persona_update_lifecycle(temp_persona_env: tuple[Path, Path]) -> None:
    """Verify updating persona metadata and default model presets.

    Args:
        temp_persona_env: Temporary directories tuple fixture.
    """
    config_dir, memory_dir = temp_persona_env
    mgr = PersonalityManager(config_dir=config_dir, memory_base_dir=memory_dir)
    mgr.load_all()

    updated = mgr.update_persona(
        "test_bot",
        {
            "display_name": "Test Bot Ultra",
            "default_models": {
                "llm": "gemini-3.1-flash-lite",
                "tts_engine": "kokoro",
                "tts_voice": "af_bella",
            },
        },
    )

    assert updated.display_name == "Test Bot Ultra"
    assert updated.default_models["llm"] == "gemini-3.1-flash-lite"
    assert updated.default_models["tts_voice"] == "af_bella"

    # Reload from disk and verify persistence
    mgr.load_all()
    pers = mgr.get("test_bot")
    assert pers.display_name == "Test Bot Ultra"
    assert pers.default_models["llm"] == "gemini-3.1-flash-lite"


def test_persona_deletion_and_db_cleanup(temp_persona_env: tuple[Path, Path]) -> None:
    """Verify deleting a custom persona removes YAML and isolated database directory.

    Args:
        temp_persona_env: Temporary directories tuple fixture.
    """
    config_dir, memory_dir = temp_persona_env
    mgr = PersonalityManager(config_dir=config_dir, memory_base_dir=memory_dir)
    mgr.load_all()

    mgr.create_persona({"id": "temporary_bot", "display_name": "Temp"})
    temp_db_dir = memory_dir / "personas" / "temporary_bot"
    assert temp_db_dir.exists()

    # Delete with database cleanup
    ok = mgr.delete_persona("temporary_bot", delete_database=True)
    assert ok is True
    assert "temporary_bot" not in mgr.list_personalities()
    assert not temp_db_dir.exists()


def test_system_persona_deletion_guard(temp_persona_env: tuple[Path, Path]) -> None:
    """Verify built-in system personas cannot be deleted without explicit force flag.

    Args:
        temp_persona_env: Temporary directories tuple fixture.
    """
    config_dir, memory_dir = temp_persona_env
    mgr = PersonalityManager(config_dir=config_dir, memory_base_dir=memory_dir)

    # Seed a system persona
    (config_dir / "neo.yaml").write_text("name: neo\ndisplay_name: Cypher\n")
    mgr.load_all()
    mgr.set_active("test_bot")  # ensure neo is not active

    with pytest.raises(ValueError, match="Cannot delete built-in system persona"):
        mgr.delete_persona("neo", force=False)


def test_api_personas_crud_and_activate_endpoints(temp_persona_env: tuple[Path, Path]) -> None:
    """Verify REST API persona routes for listing, creating, activating, and deleting personas.

    Args:
        temp_persona_env: Temporary directories tuple fixture.
    """
    from unittest.mock import AsyncMock, MagicMock
    from robot.api import create_app

    config_dir, memory_dir = temp_persona_env
    mgr = PersonalityManager(config_dir=config_dir, memory_base_dir=memory_dir)
    mgr.load_all()

    engine = MagicMock()
    engine.personalities = mgr
    engine.active_personality = "test_bot"
    engine.switch_personality = AsyncMock(return_value={"status": "ok", "personality": "sapphire"})

    app = create_app(engine)
    with TestClient(app) as client:
        # 1. GET /api/personas
        res = client.get("/api/personas")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert len(data["personas"]) >= 1

        # 2. POST /api/personas
        create_res = client.post(
            "/api/personas",
            json={
                "id": "sapphire",
                "display_name": "Sapphire",
                "description": "Crystalline intelligence",
                "system_prompt": "You are Sapphire.",
                "default_models": {
                    "llm": "qwen3.5-4b",
                    "tts_engine": "kokoro",
                    "tts_voice": "af_bella",
                },
            },
        )
        assert create_res.status_code == 200
        create_data = create_res.json()
        assert create_data["persona"]["id"] == "sapphire"
        assert (memory_dir / "personas" / "sapphire" / "conversations.db").exists()

        # 3. GET /api/personas/sapphire
        get_res = client.get("/api/personas/sapphire")
        assert get_res.status_code == 200
        assert get_res.json()["display_name"] == "Sapphire"

        # 4. POST /api/personas/sapphire/activate
        act_res = client.post("/api/personas/sapphire/activate")
        assert act_res.status_code == 200
        engine.switch_personality.assert_awaited_with("sapphire")

        # 5. DELETE /api/personas/sapphire
        del_res = client.delete("/api/personas/sapphire?delete_database=true")
        assert del_res.status_code == 200
        assert not (memory_dir / "personas" / "sapphire").exists()

