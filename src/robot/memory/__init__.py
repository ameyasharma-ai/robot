"""Memory subsystem — Mem0 persistent memory + SQLite conversation archive.

The LLM does NOT own memory. Robot Core controls all storage, retrieval,
updates, deletion, merging, namespaces, and context injection.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
import uuid
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
        try:
            # Suppress spaCy missing warnings if spaCy NLP extra is not installed
            try:
                import mem0.utils.spacy_models as sm
                sm._load_failed_full = True
                sm._load_failed_lemma = True
            except Exception:
                pass

            from mem0 import Memory

            vector_store_cfg = self._config.get("vector_store", {})
            provider = vector_store_cfg.get("provider", "qdrant")
            host = vector_store_cfg.get("host")
            path = vector_store_cfg.get("path", "data/memory/qdrant")

            qdrant_subconfig = {
                "collection_name": vector_store_cfg.get("collection", "robot_memory"),
                "embedding_model_dims": vector_store_cfg.get("embedding_dims", 768),
            }
            if host:
                qdrant_subconfig["host"] = host
                qdrant_subconfig["port"] = vector_store_cfg.get("port", 6333)
            else:
                qdrant_subconfig["path"] = path

            embedder_cfg = self._config.get("embedder", {})
            embedder_provider = embedder_cfg.get("provider", "fastembed")
            cache_dir = embedder_cfg.get("cache_dir", "models/fastembed")
            if embedder_provider == "fastembed":
                # Ensure fastembed always uses persistent local model directory instead of /tmp
                resolved_cache = Path(cache_dir).resolve()
                resolved_cache.mkdir(parents=True, exist_ok=True)
                os.environ["FASTEMBED_CACHE_PATH"] = str(resolved_cache)

                embedder_subconfig = {
                    "model": embedder_cfg.get("model", "BAAI/bge-small-en-v1.5"),
                }
            else:
                embedder_subconfig = {
                    "model": embedder_cfg.get("model", "Qwen3.5-4B-Q4_K_M"),
                    "openai_base_url": embedder_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
                    "api_key": embedder_cfg.get("api_key", "not-needed"),
                }

            mem0_config = {
                "vector_store": {
                    "provider": provider,
                    "config": qdrant_subconfig,
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
                    "provider": embedder_provider,
                    "config": embedder_subconfig,
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
        infer: bool = False,
    ) -> dict | None:
        """Store a memory from conversation messages without LLM token explosion."""
        if not self._initialized or not self._memory:
            return None
        try:
            result = await asyncio.to_thread(
                self._memory.add,
                messages,
                user_id=user_id,
                metadata=metadata or {},
                infer=infer,
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
        """Search memories relevant to a query using Mem0 filters."""
        if not self._initialized or not self._memory:
            return []
        try:
            raw = await asyncio.to_thread(
                self._memory.search,
                query,
                filters={"user_id": user_id},
                top_k=limit,
            )
            # In Mem0 v1.1+, search returns {'results': [...]}. Handle both dict and list.
            if isinstance(raw, dict):
                results = raw.get("results", [])
            elif isinstance(raw, list):
                results = raw
            else:
                results = []

            score_threshold = float(self._config.get("score_threshold", 0.78))
            cleaned = []
            for item in results:
                if isinstance(item, dict):
                    # Filter out low-confidence semantic matches that dilute context
                    score = item.get("score")
                    if score is not None and score < score_threshold:
                        continue
                    cleaned.append(item)
                elif isinstance(item, str):
                    cleaned.append({"memory": item})
            logger.info("memory_searched", query=query[:50], results=len(cleaned))
            return cleaned
        except Exception as e:
            logger.error("memory_search_failed", error=str(e))
            return []

    async def get_all(self, user_id: str = "user") -> list[dict]:
        """Get all memories for a user using Mem0 filters."""
        if not self._initialized or not self._memory:
            return []
        try:
            raw = await asyncio.to_thread(
                self._memory.get_all, filters={"user_id": user_id}
            )
            if isinstance(raw, dict):
                results = raw.get("results", [])
            elif isinstance(raw, list):
                results = raw
            else:
                results = []

            cleaned = []
            for item in results:
                if isinstance(item, dict):
                    cleaned.append(item)
                elif isinstance(item, str):
                    cleaned.append({"memory": item})
            return cleaned
        except Exception as e:
            logger.error("memory_get_all_failed", error=str(e))
            return []

    async def health_check(self) -> bool:
        return self._initialized and self._memory is not None

    async def close(self) -> None:
        """Close vector store and embedder resources cleanly."""
        if self._memory:
            try:
                # Close vector client (Qdrant) if open
                vector_store = getattr(self._memory, "vector_store", None)
                if vector_store:
                    client = getattr(vector_store, "client", None)
                    if client and hasattr(client, "close"):
                        client.close()
            except Exception:
                pass
        self._initialized = False


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
        """
        )
        self._conn.commit()

        # Migrate existing sessions table if title or updated_at are missing
        try:
            cur = self._conn.execute("PRAGMA table_info(sessions)")
            existing_cols = {row[1] for row in cur.fetchall()}
            if "title" not in existing_cols:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT DEFAULT 'New Chat'")
            if "updated_at" not in existing_cols:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN updated_at TEXT")
            self._conn.commit()
        except Exception:
            pass

        # Clean up any legacy system event titles in sessions
        try:
            self._conn.execute("UPDATE sessions SET title = 'Voice Chat' WHERE title LIKE '[%'")
            self._conn.commit()
        except Exception:
            pass

        # Backfill any historical voice sessions from conversations into sessions
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO sessions (id, personality, started_at, updated_at, turn_count, title)
                SELECT 
                    c.session_id,
                    COALESCE(MAX(c.personality), 'neo') as pers,
                    MIN(c.timestamp) as started,
                    MAX(c.timestamp) as updated,
                    COUNT(*)/2 as turns,
                    COALESCE(
                        (SELECT SUBSTR(content, 1, 36) || (CASE WHEN LENGTH(content) > 36 THEN '...' ELSE '' END)
                         FROM conversations
                         WHERE session_id = c.session_id AND role = 'user' AND content NOT LIKE '[%'
                         ORDER BY id ASC LIMIT 1),
                        'Voice Chat'
                    ) as title
                FROM conversations c
                WHERE c.session_id NOT IN (SELECT id FROM sessions)
                GROUP BY c.session_id;
                """
            )
            self._conn.commit()
        except Exception as e:
            logger.warning("history_backfill_sessions_failed", error=str(e))

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
        t_res_str = json.dumps(tool_result) if isinstance(tool_result, (dict, list)) else tool_result
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
                t_res_str,
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
        return [
            {"role": r[0], "content": r[1], "timestamp": r[2]}
            for r in reversed(rows)
            if r[1] and r[1].strip() not in ("[", "]", "<", ">")
        ]

    async def clear_turn(self, session_id: str, turn_number: int = 0) -> None:
        """Clear a specific turn (e.g. turn 0 bootstrap) from session history."""
        if not self._conn:
            return
        await asyncio.to_thread(
            self._conn.execute,
            "DELETE FROM conversations WHERE session_id = ? AND turn_number = ?",
            (session_id, turn_number),
        )
        await asyncio.to_thread(self._conn.commit)

    async def create_session(
        self, session_id: str | None = None, personality: str = "neo", title: str = "New Chat"
    ) -> dict:
        """Create or register a new isolated conversation session."""
        if not session_id:
            session_id = f"chat_{str(uuid.uuid4())[:8]}"
        if not self._conn:
            return {"id": session_id, "session_id": session_id, "personality": personality, "title": title}
        now_iso = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(
            self._conn.execute,
            """INSERT OR IGNORE INTO sessions (id, personality, started_at, updated_at, turn_count, title)
               VALUES (?, ?, ?, ?, 0, ?)""",
            (session_id, personality, now_iso, now_iso, title),
        )
        await asyncio.to_thread(self._conn.commit)
        return {
            "id": session_id,
            "session_id": session_id,
            "personality": personality,
            "title": title,
            "started_at": now_iso,
            "updated_at": now_iso,
            "turn_count": 0,
        }

    async def list_sessions(self, limit: int = 150) -> list[dict]:
        """List past chat sessions ordered by most recently active."""
        if not self._conn:
            return []

        # Ensure any orphan conversations have session records
        try:
            await asyncio.to_thread(
                self._conn.execute,
                """
                INSERT OR IGNORE INTO sessions (id, personality, started_at, updated_at, turn_count, title)
                SELECT 
                    c.session_id,
                    COALESCE(MAX(c.personality), 'neo'),
                    MIN(c.timestamp),
                    MAX(c.timestamp),
                    COUNT(*)/2,
                    COALESCE(
                        (SELECT SUBSTR(content, 1, 36) || (CASE WHEN LENGTH(content) > 36 THEN '...' ELSE '' END)
                         FROM conversations
                         WHERE session_id = c.session_id AND role = 'user' AND content NOT LIKE '[%'
                         ORDER BY id ASC LIMIT 1),
                        'Voice Chat'
                    )
                FROM conversations c
                WHERE c.session_id NOT IN (SELECT id FROM sessions)
                GROUP BY c.session_id;
                """
            )
            await asyncio.to_thread(self._conn.commit)
        except Exception:
            pass

        cursor = await asyncio.to_thread(
            self._conn.execute,
            """SELECT s.id, s.personality, s.started_at, s.updated_at, s.turn_count, s.title,
                      (SELECT content FROM conversations WHERE session_id = s.id AND role = 'assistant' ORDER BY id DESC LIMIT 1) as last_assistant_msg,
                      (SELECT content FROM conversations WHERE session_id = s.id AND role = 'user' AND content NOT LIKE '[%' ORDER BY id ASC LIMIT 1) as first_user_msg
               FROM sessions s
               ORDER BY COALESCE(s.updated_at, s.started_at) DESC
               LIMIT ?""",
            (limit,),
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            sid, pers, started, updated, turns, title, last_bot, first_u = r
            if title and title not in ("New Chat", "Voice Chat", "Chat", "New Conversation") and not title.startswith("["):
                display_title = title
            elif first_u:
                display_title = (first_u[:36] + "...") if len(first_u) > 36 else first_u
            else:
                display_title = title or "Voice Chat"

            result.append({
                "id": sid,
                "personality": pers,
                "started_at": started,
                "updated_at": updated or started,
                "turn_count": turns or 0,
                "title": display_title,
                "last_message": (last_bot[:60] + "...") if last_bot else None,
            })
        return result

    async def get_session(self, session_id: str) -> dict | None:
        """Get session metadata by ID, falling back to conversations table if needed."""
        if not self._conn:
            return None
        cursor = await asyncio.to_thread(
            self._conn.execute,
            """SELECT id, personality, started_at, updated_at, turn_count, title
               FROM sessions
               WHERE id = ?""",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            cursor_c = await asyncio.to_thread(
                self._conn.execute,
                """SELECT c.session_id, COALESCE(MAX(c.personality), 'neo'), MIN(c.timestamp), MAX(c.timestamp), COUNT(*)/2,
                          COALESCE(
                            (SELECT SUBSTR(content, 1, 36) FROM conversations WHERE session_id = c.session_id AND role = 'user' AND content NOT LIKE '[%' ORDER BY id ASC LIMIT 1),
                            'Voice Chat'
                          )
                   FROM conversations c
                   WHERE c.session_id = ?
                   GROUP BY c.session_id""",
                (session_id,),
            )
            c_row = cursor_c.fetchone()
            if not c_row:
                return None
            return {
                "id": c_row[0],
                "personality": c_row[1] or "neo",
                "started_at": c_row[2],
                "updated_at": c_row[3],
                "turn_count": c_row[4] or 0,
                "title": c_row[5] or "Voice Chat",
            }
        return {
            "id": row[0],
            "personality": row[1],
            "started_at": row[2],
            "updated_at": row[3],
            "turn_count": row[4] or 0,
            "title": row[5] or "Conversation",
        }

    async def get_stats(self) -> dict[str, int]:
        """Return memory statistics including fact count and total conversation messages.

        Returns:
            dict[str, int]: Dictionary containing facts_count, messages_count, and sessions_count.
        """
        if not self._conn:
            return {"facts_count": 0, "messages_count": 0, "sessions_count": 0}

        def _query() -> dict[str, int]:
            f_cur = self._conn.execute("SELECT COUNT(*) FROM facts")
            f_count = int(f_cur.fetchone()[0])
            m_cur = self._conn.execute("SELECT COUNT(*) FROM conversations")
            m_count = int(m_cur.fetchone()[0])
            s_cur = self._conn.execute("SELECT COUNT(*) FROM sessions")
            s_count = int(s_cur.fetchone()[0])
            return {"facts_count": f_count, "messages_count": m_count, "sessions_count": s_count}

        return await asyncio.to_thread(_query)

    async def get_session_messages(self, session_id: str, limit: int = 200) -> list[dict]:
        """Get all messages for an isolated session, including tool call details and metrics."""
        if not self._conn:
            return []
        cursor = await asyncio.to_thread(
            self._conn.execute,
            """SELECT id, role, content, personality, turn_number, tool_name, tool_result, timestamp, metadata
               FROM conversations
               WHERE session_id = ?
               ORDER BY id ASC
               LIMIT ?""",
            (session_id, limit),
        )
        rows = cursor.fetchall()
        msgs = []
        for r in rows:
            mid, role, content, pers, turn_num, t_name, t_res, ts, meta_raw = r
            meta = {}
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    pass
            parsed_t_res = t_res
            if isinstance(t_res, str) and (t_res.startswith("{") or t_res.startswith("[")):
                try:
                    parsed_t_res = json.loads(t_res)
                except Exception:
                    pass
            msgs.append({
                "id": mid,
                "role": role,
                "content": content,
                "personality": pers,
                "turn_number": turn_num,
                "tool_name": t_name,
                "tool_result": parsed_t_res,
                "timestamp": ts,
                "metadata": meta,
            })
        return msgs

    async def delete_session(self, session_id: str) -> bool:
        """Delete an isolated session and all its messages."""
        if not self._conn:
            return False
        await asyncio.to_thread(
            self._conn.execute,
            "DELETE FROM conversations WHERE session_id = ?",
            (session_id,),
        )
        await asyncio.to_thread(
            self._conn.execute,
            "DELETE FROM sessions WHERE id = ?",
            (session_id,),
        )
        await asyncio.to_thread(self._conn.commit)
        return True

    async def update_session(
        self, session_id: str, title: str | None = None, turn_count: int | None = None
    ) -> None:
        """Update session metadata."""
        if not self._conn:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        if title is not None and turn_count is not None:
            await asyncio.to_thread(
                self._conn.execute,
                "UPDATE sessions SET title = ?, turn_count = ?, updated_at = ? WHERE id = ?",
                (title, turn_count, now_iso, session_id),
            )
        elif title is not None:
            await asyncio.to_thread(
                self._conn.execute,
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now_iso, session_id),
            )
        elif turn_count is not None:
            await asyncio.to_thread(
                self._conn.execute,
                "UPDATE sessions SET turn_count = ?, updated_at = ? WHERE id = ?",
                (turn_count, now_iso, session_id),
            )
        else:
            await asyncio.to_thread(
                self._conn.execute,
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now_iso, session_id),
            )
        await asyncio.to_thread(self._conn.commit)

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update session title."""
        await self.update_session(session_id, title=title)

    async def update_session_title_if_default(self, session_id: str, new_title: str) -> None:
        """Update session title only if it is currently a generic default or system event string."""
        if not self._conn:
            return
        await asyncio.to_thread(
            self._conn.execute,
            """UPDATE sessions 
               SET title = ? 
               WHERE id = ? AND (title IN ('New Chat', 'Voice Chat', 'Chat', 'New Conversation', '') OR title LIKE '[%')""",
            (new_title, session_id),
        )
        await asyncio.to_thread(self._conn.commit)

    async def add_fact(self, fact: str, user_id: str = "user", key: str | None = None) -> None:
        """Store a durable fact in SQLite, preventing duplicate entries."""
        if not self._conn:
            return
        clean_fact = fact.strip()
        if not clean_fact:
            return

        # Prevent exact duplicate insertions for the same user
        try:
            cursor = await asyncio.to_thread(
                self._conn.execute,
                """SELECT id FROM facts WHERE (user_id = ? OR user_id = 'default') AND LOWER(fact) = LOWER(?) LIMIT 1""",
                (user_id, clean_fact),
            )
            if cursor.fetchone():
                return
        except Exception:
            pass

        await asyncio.to_thread(
            self._conn.execute,
            """INSERT INTO facts (key, fact, user_id, created_at) VALUES (?, ?, ?, ?)""",
            (key, clean_fact, user_id, datetime.now(timezone.utc).isoformat()),
        )
        await asyncio.to_thread(self._conn.commit)

    async def search_facts(self, query: str, user_id: str = "user", limit: int = 5) -> list[dict]:
        """Search persistent facts for specific user/namespace using FTS5, LIKE fallback, and deduplication."""
        if not self._conn:
            return []

        import re
        clean_q = (query or "").strip().lower()
        broad_phrases = {
            "", "all", "*", "everything", "facts", "memories", "durable facts",
            "saved facts", "what do you know", "what do you remember",
            "what facts", "what memories", "preferences", "me", "about me",
            "user facts", "who am i", "all facts", "all memories",
        }
        if clean_q in broad_phrases:
            return await self.get_all_facts(user_id=user_id, limit=limit)

        stop_and_meta = {
            "where", "what", "who", "when", "why", "how", "do", "i", "you", "is", "are",
            "the", "a", "an", "my", "your", "in", "on", "at", "tell", "me", "about", "can",
            "remember", "know", "am", "was", "were", "did", "does", "have", "has", "had",
            "and", "or", "so", "with", "for", "to", "of", "from", "by", "that", "this",
            "durable", "facts", "fact", "memories", "memory", "saved", "stored", "remembered",
            "details", "info", "information", "everything", "all", "any", "get", "show", "list",
        }
        words = [w for w in re.findall(r"\w+", clean_q) if w not in stop_and_meta]
        if not words:
            # The query did not specify a specific topic or keyword; return all deduplicated facts
            return await self.get_all_facts(user_id=user_id, limit=limit)

        if words:
            # 1. Prefix FTS matching with stop-word removal
            fts_expr = " OR ".join(f"{w}*" for w in words)
            try:
                cursor = await asyncio.to_thread(
                    self._conn.execute,
                    """SELECT f.id, f.fact, f.created_at FROM facts_fts ft
                       JOIN facts f ON f.id = ft.rowid
                       WHERE ft.fact MATCH ? AND (f.user_id = ? OR f.user_id = 'default')
                       ORDER BY f.id DESC LIMIT ?""",
                    (fts_expr, user_id, limit * 3),
                )
                rows = cursor.fetchall()
                if rows:
                    return self._deduplicate_fact_rows(rows, limit)
            except Exception:
                pass

            # 2. Individual keyword LIKE fallback
            like_clauses = " OR ".join("fact LIKE ?" for _ in words)
            params = [f"%{w}%" for w in words] + [user_id, limit * 3]
            try:
                cursor = await asyncio.to_thread(
                    self._conn.execute,
                    f"""SELECT id, fact, created_at FROM facts
                        WHERE ({like_clauses}) AND (user_id = ? OR user_id = 'default')
                        ORDER BY id DESC LIMIT ?""",
                    params,
                )
                rows = cursor.fetchall()
                if rows:
                    return self._deduplicate_fact_rows(rows, limit)
            except Exception:
                pass

        # If query specified keywords but no fact matched, return empty list
        # to prevent irrelevant memory pollution during unrelated commands or tasks.
        return []

    @staticmethod
    def _deduplicate_fact_rows(rows: list[tuple], limit: int) -> list[dict]:
        """Filter out semantically redundant fact rows to maximize diverse memory recall."""
        import re
        stop = {
            "user", "the", "a", "an", "is", "in", "on", "at", "to", "for", "with",
            "about", "of", "me", "my", "and", "that", "i", "he", "she", "it", "his",
            "her", "asked", "asking", "asks", "know", "remember", "saved", "stored",
        }
        unique_facts: list[dict] = []
        seen_word_sets: list[set[str]] = []

        for r in rows:
            fact_text = str(r[1]).strip()
            words = set(w for w in re.findall(r"\w+", fact_text.lower()) if w not in stop)
            if not words:
                words = set(re.findall(r"\w+", fact_text.lower()))

            is_dup = False
            for seen in seen_word_sets:
                common = words & seen
                # High content word overlap indicates redundant phrasing of the same fact
                if len(common) >= 2 and (len(common) / min(len(words), len(seen))) >= 0.5:
                    is_dup = True
                    break
                elif len(common) >= 1 and len(words) == 1 and len(seen) == 1:
                    is_dup = True
                    break

            if not is_dup:
                seen_word_sets.append(words)
                unique_facts.append({"id": r[0], "memory": fact_text, "fact": fact_text, "created_at": r[2]})
                if len(unique_facts) >= limit:
                    break

        return unique_facts

    async def get_all_facts(self, user_id: str = "user", limit: int = 500) -> list[dict]:
        """Retrieve all durable facts for the user/personality across time, deduplicated."""
        if not self._conn:
            return []
        cursor = await asyncio.to_thread(
            self._conn.execute,
            """SELECT id, fact, created_at FROM facts
               WHERE user_id = ? OR user_id = 'default'
               ORDER BY id DESC LIMIT ?""",
            (user_id, max(limit * 3, 100)),
        )
        rows = cursor.fetchall()
        return self._deduplicate_fact_rows(rows, limit)

    async def get_facts(self, user_id: str = "user", limit: int = 500) -> list[dict]:
        """Alias for get_all_facts."""
        return await self.get_all_facts(user_id=user_id, limit=limit)

    async def get_archive(self, limit: int = 1000, exclude_session_id: str | None = None) -> list[dict]:
        """Retrieve chronological past conversation history across previous sessions."""
        if not self._conn:
            return []
        if exclude_session_id:
            cursor = await asyncio.to_thread(
                self._conn.execute,
                """SELECT role, content, timestamp, session_id FROM conversations
                   WHERE session_id != ?
                   ORDER BY id DESC LIMIT ?""",
                (exclude_session_id, limit),
            )
        else:
            cursor = await asyncio.to_thread(
                self._conn.execute,
                """SELECT role, content, timestamp, session_id FROM conversations
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2], "session_id": r[3]} for r in reversed(rows)]

    async def health_check(self) -> bool:
        return self._conn is not None

    async def close(self) -> None:
        """Close SQLite connection cleanly."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


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
        personality: str = "cypher",
        turn_number: int = 0,
        namespace: str | None = None,
        tool_name: str | None = None,
        tool_result: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Store a complete conversation turn in both systems, isolated by personality namespace."""
        active_ns = namespace or personality or "default"

        # Duplicate turn guard: prevent recording duplicate identical turns within cooldown window
        last_turn_key = f"{session_id}:{user_text.strip()}:{bot_text.strip()}"
        now_ts = time.time()
        if hasattr(self, "_last_stored_turns"):
            if last_turn_key in self._last_stored_turns:
                last_time = self._last_stored_turns[last_turn_key]
                if now_ts - last_time < 15.0:
                    logger.debug("duplicate_turn_storage_suppressed", session_id=session_id)
                    return
        else:
            self._last_stored_turns = {}
        self._last_stored_turns[last_turn_key] = now_ts
        if len(self._last_stored_turns) > 50:
            self._last_stored_turns = {k: v for k, v in self._last_stored_turns.items() if now_ts - v < 30.0}

        # Auto-create or update session record
        is_system = user_text.strip().startswith("[")
        clean_title = (
            (user_text.strip().replace("\n", " ")[:36] + "...")
            if (len(user_text.strip()) > 36 and not is_system)
            else (user_text.strip() if (user_text.strip() and not is_system) else "Voice Chat")
        )
        await self.history.create_session(session_id, personality=personality, title=clean_title)
        if not is_system and user_text.strip():
            await self.history.update_session_title_if_default(session_id, clean_title)
        await self.history.update_session(session_id, turn_count=turn_number)

        # SQLite archive (always recorded with personality tag)
        await self.history.add_message(
            session_id, "user", user_text, personality, turn_number
        )
        await self.history.add_message(
            session_id,
            "assistant",
            bot_text,
            personality,
            turn_number,
            tool_name=tool_name,
            tool_result=tool_result,
            metadata=metadata,
        )

        # Store user statements that declare preferences or biographical facts into isolated SQLite facts
        lower_u = user_text.lower().strip()
        is_command = any(cmd in lower_u for cmd in (
            "search", "find", "google", "look up", "play", "stop", "pause", "resume",
            "set a timer", "set an alarm", "use your tool", "using your tool", "check time",
            "volume to", "turn it up", "turn it down"
        ))

        # Distinct biographical declarations
        bio_patterns = [
            r"\bremember\s+(?:that|this)\b",
            r"\bkeep\s+in\s+mind\s+that\b",
            r"\bmy\s+favorite\b",
            r"\bmy\s+name\s+is\b",
            r"\bmy\s+phone\s+is\b",
            r"\bi\s+live\s+in\b",
            r"\bi\s+work\s+as\b",
            r"\bcall\s+me\b",
            r"\bmy\s+partner\s+is\b",
            r"\bmy\s+friend\s+is\b",
            r"\bmy\s+crush\s+is\b",
            r"\bi\s+prefer\b",
            r"\bmy\s+goal\s+is\b",
            r"\bdon't\s+forget\b",
            r"\bi\s+am\s+(?:a|an|the|\d+|allergic\s+to|from|working\s+as|interested\s+in)\b",
            r"\bi\s+have\s+(?:a|an)\s+(?:severe\s+)?(?:allergy|job|pet|car)\b",
        ]

        if not is_command and any(re.search(pat, lower_u) for pat in bio_patterns):
            await self.history.add_fact(user_text, user_id=active_ns)

        # Mem0 semantic memory (isolated per personality namespace)
        # Avoid storing commands, trivial utterances (hi, no, yes), or empty strings
        if bot_text and bot_text.strip() not in ("[", "]", "") and not is_command:
            if len(lower_u) > 3 and lower_u not in ("hi", "hello", "hey", "yes", "no", "ok", "okay", "bye", "thanks", "thank you"):
                messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": bot_text},
                ]
                await self.mem0.add(messages, user_id=active_ns, infer=False)

    async def create_session(
        self, session_id: str | None = None, personality: str = "neo", title: str = "New Chat"
    ) -> dict:
        """Create or register a new isolated conversation session."""
        return await self.history.create_session(session_id=session_id, personality=personality, title=title)

    async def list_sessions(self, limit: int = 150) -> list[dict]:
        """List past chat sessions ordered by most recently active."""
        return await self.history.list_sessions(limit)

    async def get_session(self, session_id: str) -> dict | None:
        """Get session metadata by ID."""
        return await self.history.get_session(session_id)

    async def get_session_messages(self, session_id: str, limit: int = 200) -> list[dict]:
        """Get all messages for an isolated session, including tool call details and metrics."""
        return await self.history.get_session_messages(session_id, limit)

    async def delete_session(self, session_id: str) -> bool:
        """Delete an isolated session and all its messages."""
        return await self.history.delete_session(session_id)

    async def add_memory(self, fact: str, user_id: str = "user") -> None:
        """Explicitly add a durable memory within a specific isolated personality workspace."""
        await self.history.add_fact(fact, user_id=user_id)
        if self.mem0._initialized:
            await self.mem0.add([{"role": "user", "content": fact}], user_id=user_id, infer=False)

    async def get_relevant_memories(self, query: str, user_id: str = "user", limit: int = 5) -> list[dict]:
        """Retrieve relevant memories strictly isolated within the personality workspace."""
        mem0_results = await self.mem0.search(query, user_id=user_id, limit=limit)
        sqlite_facts = await self.history.search_facts(query, user_id=user_id, limit=limit)
        
        # Merge results, prioritizing SQLite facts if mem0 is empty or offline
        combined = []
        seen_texts = set()

        for source_list in (sqlite_facts, mem0_results):
            if not isinstance(source_list, list):
                continue
            for item in source_list:
                if isinstance(item, dict):
                    txt = item.get("memory", item.get("text", "")).strip()
                    m_dict = item
                elif isinstance(item, str):
                    txt = item.strip()
                    m_dict = {"memory": txt}
                else:
                    txt = str(item).strip()
                    m_dict = {"memory": txt}

                if txt and txt not in seen_texts:
                    combined.append(m_dict)
                    seen_texts.add(txt)

        return combined[:limit]

    async def get_recent_conversation(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """Get recent conversation for context."""
        return await self.history.get_recent(session_id, limit)

    async def get_all_facts(self, user_id: str = "user", limit: int = 500) -> list[dict]:
        """Retrieve all durable facts for full context grounding."""
        return await self.history.get_all_facts(user_id=user_id, limit=limit)

    async def get_conversation_archive(
        self, limit: int = 1000, exclude_session_id: str | None = None
    ) -> list[dict]:
        """Retrieve historical multi-session conversations for deep context recall."""
        return await self.history.get_archive(limit=limit, exclude_session_id=exclude_session_id)

    async def clear_turn(self, session_id: str, turn_number: int = 0) -> None:
        """Clear a specific turn from session history."""
        await self.history.clear_turn(session_id, turn_number)

    async def search_history(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across all conversations."""
        return await self.history.search(query, limit)

    async def get_stats(self) -> dict[str, int]:
        """Retrieve aggregate memory and conversation statistics.

        Returns:
            dict[str, int]: Counts of persistent facts, stored messages, and sessions.
        """
        return await self.history.get_stats()

    async def get_core_user_profile(self, user_id: str = "user") -> str:
        """Extract and format core user profile sheet from durable memory facts."""
        facts = await self.get_all_facts(user_id=user_id, limit=50)
        identity_lines = ["• User Name: Ameya", "• User Role: AI Engineer & Robot Systems Creator"]
        tech_lines = ["• Tech Stack: Python, PyTorch, C++, AsyncIO, WebSockets, Local LLMs, Web & Robotics"]
        pref_lines = []
        rel_lines = []

        for f in facts:
            txt = f.get("fact", f.get("memory", "")).strip()
            if not txt:
                continue
            # Defend against bloated text dumps, scrapes, or web contents
            if len(txt) > 250 or "\n\n" in txt or any(b in txt.lower() for b in ["skip to content", "openrouter", "http://", "https://"]):
                continue
            txt_clean = " ".join(txt.split())[:180]
            txt_lower = txt_clean.lower()
            if any(k in txt_lower for k in ["name", "ameya", "engineer", "developer", "creator", "built"]):
                if not any(txt_clean.lower() in line.lower() for line in identity_lines):
                    identity_lines.append(f"• {txt_clean}")
            elif any(k in txt_lower for k in ["crush", "partner", "friend", "girlfriend", "anubha", "relationship"]):
                if not any(txt_clean.lower() in line.lower() for line in rel_lines):
                    rel_lines.append(f"• {txt_clean}")
            elif any(k in txt_lower for k in ["like", "love", "favorite", "prefer", "music", "volume"]):
                if not any(txt_clean.lower() in line.lower() for line in pref_lines):
                    pref_lines.append(f"• {txt_clean}")
            else:
                if not any(txt_clean.lower() in line.lower() for line in pref_lines):
                    pref_lines.append(f"• {txt_clean}")

        profile_sections = ["=== USER CORE IDENTITY PROFILE ==="]
        if identity_lines:
            profile_sections.append("Identity & Role:\n" + "\n".join(identity_lines[:5]))
        if rel_lines:
            profile_sections.append("Relationships & Social Context:\n" + "\n".join(rel_lines[:5]))
        if tech_lines:
            profile_sections.append("Technical Profile:\n" + "\n".join(tech_lines[:5]))
        if pref_lines:
            profile_sections.append("Preferences & Directives:\n" + "\n".join(pref_lines[:10]))

        return "\n\n".join(profile_sections)

    async def close(self) -> None:
        """Cleanly close database and vector store handles."""
        await self.history.close()
        await self.mem0.close()


def compress_conversation_history(history: list[dict], max_raw_turns: int = 15) -> list[dict]:
    """Compress older conversation turns into a compact summary block while preserving recent turns.

    Args:
        history: List of conversation turn dicts with 'role' and 'content'.
        max_raw_turns: Maximum raw turns to keep uncompressed at the end (default 15).

    Returns:
        Compressed list of message dicts with summary prefix if total turns > max_raw_turns.
    """
    if not history or len(history) <= max_raw_turns:
        return list(history)

    older_turns = history[:-max_raw_turns]
    recent_turns = history[-max_raw_turns:]

    topics = []
    for msg in older_turns:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role == "user" and content and not content.startswith("["):
            topics.append(content[:60])

    summary_text = (
        f"Prior conversation covered {len(older_turns)} turns including: "
        + "; ".join(topics[:5])
        + (f" and {len(topics) - 5} more topics." if len(topics) > 5 else ".")
    )

    summary_msg = {
        "role": "system",
        "content": f"[Executive Conversation History Summary]: {summary_text}",
    }

    return [summary_msg] + recent_turns
