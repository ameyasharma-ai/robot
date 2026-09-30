"""Memory subsystem — Mem0 persistent memory + SQLite conversation archive.

The LLM does NOT own memory. Robot Core controls all storage, retrieval,
updates, deletion, merging, namespaces, and context injection.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot.telemetry import get_logger

logger = get_logger("memory")


class Mem0Provider:
    """Mem0 integration for persistent semantic memory.

    Stores durable facts, preferences, events, and user context.
    Configured for local self-hosted operation with Qdrant + local LLM.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._memory = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Mem0 with local configuration."""
        try:
            from mem0 import Memory

            mem0_config = {
                "vector_store": {
                    "provider": self._config.get("vector_store", {}).get("provider", "qdrant"),
                    "config": {
                        "host": self._config.get("vector_store", {}).get("host", "localhost"),
                        "port": self._config.get("vector_store", {}).get("port", 6333),
                        "collection_name": self._config.get("vector_store", {}).get(
                            "collection", "robot_memory"
                        ),
                        "embedding_model_dims": self._config.get("vector_store", {}).get(
                            "embedding_dims", 768
                        ),
                    },
                },
                "llm": {
                    "provider": self._config.get("llm", {}).get("provider", "openai"),
                    "config": {
                        "model": self._config.get("llm", {}).get("model", "Qwen3.5-4B-Q4_K_M"),
                        "openai_base_url": self._config.get("llm", {}).get(
                            "base_url", "http://127.0.0.1:8080/v1"
                        ),
                        "api_key": self._config.get("llm", {}).get("api_key", "not-needed"),
                    },
                },
                "embedder": {
                    "provider": self._config.get("embedder", {}).get("provider", "openai"),
                    "config": {
                        "model": self._config.get("embedder", {}).get(
                            "model", "Qwen3.5-4B-Q4_K_M"
                        ),
                        "openai_base_url": self._config.get("embedder", {}).get(
                            "base_url", "http://127.0.0.1:8080/v1"
                        ),
                        "api_key": self._config.get("embedder", {}).get("api_key", "not-needed"),
                    },
                },
            }

            self._memory = Memory.from_config(mem0_config)
            self._initialized = True
            logger.info("mem0_initialized")
        except ImportError:
            logger.warning("mem0_not_installed", hint="pip install mem0ai qdrant-client")
        except Exception as e:
            logger.error("mem0_init_failed", error=str(e))

    async def add(
        self,
        messages: list[dict],
        user_id: str = "user",
        metadata: dict | None = None,
    ) -> dict | None:
        """Store a memory from conversation messages."""
        if not self._initialized or not self._memory:
            return None
        try:
            result = await asyncio.to_thread(
                self._memory.add, messages, user_id=user_id, metadata=metadata or {}
            )
            logger.info("memory_stored", user_id=user_id)
            return result
        except Exception as e:
            logger.error("memory_store_failed", error=str(e))
            return None

    async def search(
        self,
        query: str,
        user_id: str = "user",
        limit: int = 5,
    ) -> list[dict]:
        """Search memories relevant to a query."""
        if not self._initialized or not self._memory:
            return []
        try:
            results = await asyncio.to_thread(
                self._memory.search, query, user_id=user_id, limit=limit
            )
            logger.info("memory_searched", query=query[:50], results=len(results))
            return results
        except Exception as e:
            logger.error("memory_search_failed", error=str(e))
            return []

    async def get_all(self, user_id: str = "user") -> list[dict]:
        """Get all memories for a user."""
        if not self._initialized or not self._memory:
            return []
        try:
            return await asyncio.to_thread(self._memory.get_all, user_id=user_id)
        except Exception as e:
            logger.error("memory_get_all_failed", error=str(e))
            return []

    async def health_check(self) -> bool:
        return self._initialized and self._memory is not None


class ConversationHistory:
    """SQLite + FTS conversation archive.

    Stores complete conversations, tool events, timestamps,
    sessions, and important system events.
    """

    def __init__(self, db_path: str = "data/database/conversations.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Create tables and FTS index."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript(
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
                turn_count INTEGER DEFAULT 0
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
        """
        )
        self._conn.commit()
        logger.info("history_initialized", db=str(self.db_path))

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        personality: str = "neo",
        turn_number: int = 0,
        tool_name: str | None = None,
        tool_result: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a conversation message."""
        if not self._conn:
            return
        await asyncio.to_thread(
            self._conn.execute,
            """INSERT INTO conversations
               (session_id, role, content, personality, turn_number,
                tool_name, tool_result, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                role,
                content,
                personality,
                turn_number,
                tool_name,
                tool_result,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata) if metadata else None,
            ),
        )
        await asyncio.to_thread(self._conn.commit)

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across conversation history."""
        if not self._conn:
            return []
        cursor = await asyncio.to_thread(
            self._conn.execute,
            """SELECT c.role, c.content, c.timestamp, c.session_id
               FROM conversations_fts f
               JOIN conversations c ON c.id = f.rowid
               WHERE f.content MATCH ?
               ORDER BY c.timestamp DESC
               LIMIT ?""",
            (query, limit),
        )
        rows = cursor.fetchall()
        return [
            {"role": r[0], "content": r[1], "timestamp": r[2], "session_id": r[3]} for r in rows
        ]

    async def get_recent(self, session_id: str, limit: int = 20) -> list[dict]:
        """Get recent messages from current session."""
        if not self._conn:
            return []
        cursor = await asyncio.to_thread(
            self._conn.execute,
            """SELECT role, content, timestamp FROM conversations
               WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit),
        )
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

    async def add_fact(self, fact: str, user_id: str = "user", key: str | None = None) -> None:
        """Store a durable fact in SQLite."""
        if not self._conn:
            return
        await asyncio.to_thread(
            self._conn.execute,
            """INSERT INTO facts (key, fact, user_id, created_at) VALUES (?, ?, ?, ?)""",
            (key, fact, user_id, datetime.now(timezone.utc).isoformat()),
        )
        await asyncio.to_thread(self._conn.commit)

    async def search_facts(self, query: str, limit: int = 5) -> list[dict]:
        """Search persistent facts using FTS5 and LIKE fallback."""
        if not self._conn:
            return []
        # Try FTS5 first
        clean_q = "".join(c for c in query if c.isalnum() or c.isspace()).strip()
        if clean_q:
            try:
                cursor = await asyncio.to_thread(
                    self._conn.execute,
                    """SELECT f.id, f.fact, f.created_at FROM facts_fts ft
                       JOIN facts f ON f.id = ft.rowid
                       WHERE ft.fact MATCH ?
                       ORDER BY f.id DESC LIMIT ?""",
                    (clean_q, limit),
                )
                rows = cursor.fetchall()
                if rows:
                    return [{"id": r[0], "memory": r[1], "created_at": r[2]} for r in rows]
            except Exception:
                pass

        # Fallback to LIKE search or recent facts
        like_pattern = f"%{clean_q}%" if clean_q else "%"
        cursor = await asyncio.to_thread(
            self._conn.execute,
            """SELECT id, fact, created_at FROM facts
               WHERE fact LIKE ?
               ORDER BY id DESC LIMIT ?""",
            (like_pattern, limit),
        )
        rows = cursor.fetchall()
        return [{"id": r[0], "memory": r[1], "created_at": r[2]} for r in rows]

    async def health_check(self) -> bool:
        return self._conn is not None


class MemoryManager:
    """Orchestrates Mem0 + SQLite for the whitepaper's memory architecture.

    Working State       → RAM / Robot Core
    Recent Conversation → Context Engine (from SQLite)
    Persistent Memory   → Mem0 (semantic facts)
    Raw Archive         → SQLite + FTS
    """

    def __init__(self, mem0_config: dict, history_path: str) -> None:
        self.mem0 = Mem0Provider(mem0_config)
        self.history = ConversationHistory(history_path)

    async def initialize(self) -> None:
        await self.history.initialize()
        await self.mem0.initialize()
        logger.info("memory_manager_initialized")

    async def store_turn(
        self,
        session_id: str,
        user_text: str,
        bot_text: str,
        personality: str = "neo",
        turn_number: int = 0,
    ) -> None:
        """Store a complete conversation turn in both systems."""
        # SQLite archive (always)
        await self.history.add_message(
            session_id, "user", user_text, personality, turn_number
        )
        await self.history.add_message(
            session_id, "assistant", bot_text, personality, turn_number
        )

        # Also store user statements that declare preferences or facts into SQLite facts
        lower_u = user_text.lower()
        remember_triggers = ["remember that", "remember this", "keep in mind that", "i like", "my favorite", "i am", "my name is", "i love", "my phone is"]
        if any(trig in lower_u for trig in remember_triggers):
            await self.history.add_fact(user_text, user_id="user")

        # Mem0 semantic memory (extract facts/preferences)
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ]
        await self.mem0.add(messages, user_id="user")

    async def add_memory(self, fact: str, user_id: str = "user") -> None:
        """Explicitly add a durable memory."""
        await self.history.add_fact(fact, user_id=user_id)
        if self.mem0._initialized:
            await self.mem0.add([{"role": "user", "content": fact}], user_id=user_id)

    async def get_relevant_memories(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve relevant memories for context injection."""
        mem0_results = await self.mem0.search(query, limit=limit)
        sqlite_facts = await self.history.search_facts(query, limit=limit)
        
        # Merge results, prioritizing SQLite facts if mem0 is empty or offline
        combined = list(sqlite_facts)
        seen_texts = set(f.get("memory", "") for f in combined)
        for m in mem0_results:
            txt = m.get("memory", m.get("text", ""))
            if txt and txt not in seen_texts:
                combined.append(m)
                seen_texts.add(txt)
        return combined[:limit]

    async def get_recent_conversation(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """Get recent conversation for context."""
        return await self.history.get_recent(session_id, limit)

    async def search_history(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across all conversations."""
        return await self.history.search(query, limit)
