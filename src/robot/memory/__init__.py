"""Memory subsystem — Mem0 persistent memory + SQLite conversation archive.

The LLM does NOT own memory. Robot Core controls all storage, retrieval,
updates, deletion, merging, namespaces, and context injection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from robot import config as robot_config
from robot.telemetry import get_logger

logger = get_logger("memory")

# Project root (repo root), resolved from this file: <root>/src/robot/memory/__init__.py.
# Kept as a module constant for backward compatibility; runtime path resolution uses
# ``robot.config.get_project_root()`` so CWD changes (and test monkeypatching) cannot
# move the data directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Canonical persona aliases. ``cypher`` is Neo's memory namespace and the historical
#: public name of the built-in assistant persona.
PERSONA_ALIASES: dict[str, str] = {
    "cypher": "neo",
    "matrix": "neo",
    "hacker": "neo",
    "cyber": "neo",
}

#: Persona identifiers that are allowed to use the shared system-user profile.
SYSTEM_PERSONA_NAMES: frozenset[str] = frozenset(
    {"user", "neo", "luna", "aria", "glados", "cypher", "matrix", "hacker", "cyber"}
)

#: Safety ceiling for FTS query tokens; keeps a hostile query from becoming a giant
#: MATCH expression.
_MAX_FTS_TOKENS = 16
_MAX_FTS_TOKEN_LEN = 64

#: Ceiling on archive rows read into one context build. Callers ask for tens of
#: turns; an unbounded ``limit`` would let a client pull the whole history.
MAX_ARCHIVE_ROWS = 2_000
#: Ceiling on rows scanned before fact de-duplication.
MAX_FACT_SCAN_ROWS = 2_000

#: Conversation metadata keys that contain hidden chain-of-thought / model reasoning.
#: They are stripped before persistence so private reasoning never becomes a
#: world-readable database row.
_REDACTED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "thought",
        "thoughts",
        "thinking",
        "reasoning",
        "reasoning_content",
        "reasoning_text",
        "chain_of_thought",
        "cot",
    }
)


def normalize_persona(name: str | None) -> str:
    """Normalize a persona identifier and resolve built-in aliases.

    Args:
        name: Raw persona identifier (case-insensitive, alias-aware).

    Returns:
        Canonical lowercase persona identifier; ``"neo"`` when empty.
    """
    clean = (name or "").strip().lower()
    if not clean:
        return "neo"
    return PERSONA_ALIASES.get(clean, clean)


def persona_variants(name: str | None) -> tuple[str, ...]:
    """Return every stored persona spelling that belongs to ``name``.

    Conversation rows historically used both the active persona name (``neo``) and
    the memory namespace (``cypher``). Reads must treat both as the same persona
    while still excluding unrelated personas.

    Args:
        name: Persona identifier.

    Returns:
        Tuple of lowercase persona identifiers accepted for this persona.
    """
    canonical = normalize_persona(name)
    variants = {canonical}
    for alias, target in PERSONA_ALIASES.items():
        if target == canonical:
            variants.add(alias)
    return tuple(sorted(variants))


def generate_session_id() -> str:
    """Create a collision-resistant chat session identifier.

    Returns:
        A full UUID-based session identifier prefixed with ``chat_``.
    """
    return f"chat_{uuid.uuid4()}"


def _persona_expr(column: str) -> str:
    """Return the SQL expression that normalizes a persona column.

    Every persona-scoped predicate and every persona index must spell this the
    same way: SQLite only uses an expression index when the queried expression
    is structurally identical to the one that was indexed.

    Args:
        column: Column reference, optionally table-qualified.

    Returns:
        Normalized, lower-cased persona expression with the default persona.
    """
    return f"LOWER(COALESCE(NULLIF(TRIM({column}), ''), 'neo'))"


def _clause_for(name: str | None, column: str = "personality") -> tuple[str, tuple[str, ...]]:
    """Return the persona SQL predicate and its positional parameters."""
    variants = persona_variants(name)
    placeholders = ", ".join("?" for _ in variants)
    sql = f"{_persona_expr(column)} IN ({placeholders})"
    return sql, variants


def _sanitize_fts_query(query: str) -> str:
    """Turn arbitrary user input into a safe FTS5 ``MATCH`` expression.

    Raw text is never passed to ``MATCH``: quotes, ``*``, ``:``, ``NEAR`` and
    boolean operators have FTS5 syntax meaning and raised ``OperationalError``
    (HTTP 500) for ordinary search queries. Tokens are split on non-word
    characters, quoted, length-capped, and AND-joined.

    Args:
        query: Raw user query.

    Returns:
        A quoted FTS expression, or ``""`` when no searchable token exists.
    """
    raw_tokens = re.findall(r"\w+", (query or "").lower(), flags=re.UNICODE)
    operators = {"and", "or", "not", "near"}
    tokens: list[str] = []
    for token in raw_tokens:
        token = token[:_MAX_FTS_TOKEN_LEN]
        if not token or token in operators:
            continue
        tokens.append(token)
        if len(tokens) >= _MAX_FTS_TOKENS:
            break
    if not tokens:
        return ""
    return " AND ".join(f'"{token}"' for token in tokens)


def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip private model reasoning from conversation metadata.

    Args:
        metadata: Optional metadata mapping supplied by the caller.

    Returns:
        A copy without chain-of-thought keys, or ``None`` when nothing remains.
    """
    if not isinstance(metadata, dict):
        return None
    clean = {
        key: value
        for key, value in metadata.items()
        if str(key).strip().lower() not in _REDACTED_METADATA_KEYS
    }
    return clean or None


def _harden_permissions(db_path: Path) -> None:
    """Restrict a conversation database (and its WAL sidecars) to the owner.

    Args:
        db_path: SQLite database file path.
    """
    try:
        os.chmod(db_path.parent, 0o700)
    except OSError:
        pass
    for candidate in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        try:
            if candidate.exists():
                os.chmod(candidate, 0o600)
        except OSError:
            pass


def _parse_json_field(raw: Any, expected_type: type, default: Any, *, user_id: str = "") -> Any:
    """Parse a JSON profile column defensively.

    Corrupted or type-mismatched JSON falls back to ``default`` instead of
    crashing every profile read.

    Args:
        raw: Raw database value.
        expected_type: ``dict`` or ``list``.
        default: Value returned when parsing fails.
        user_id: Namespace used only for structured logging.

    Returns:
        Parsed value with the expected container type, or ``default``.
    """
    if raw is None or raw == "":
        return default
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("profile_json_parse_failed", user_id=user_id, error=type(exc).__name__)
        return default
    if not isinstance(parsed, expected_type):
        logger.warning(
            "profile_json_type_mismatch",
            user_id=user_id,
            expected=expected_type.__name__,
            actual=type(parsed).__name__,
        )
        return default
    return parsed


class Mem0Provider:
    """Mem0 integration for persistent semantic memory.

    Stores durable facts, preferences, events, and user context.
    Configured for local self-hosted operation with Qdrant + local LLM.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._memory = None
        self._initialized = False
        try:
            self._timeout = float(config.get("timeout_seconds", 5.0))
        except (TypeError, ValueError):
            self._timeout = 5.0
        if self._timeout <= 0:
            self._timeout = None
        # A timed-out call cannot be cancelled: its worker thread keeps running
        # and keeps talking to the client. This lock lives in the worker, so the
        # orphan serializes against the next call instead of opening the same
        # Qdrant collection twice.
        self._client_lock = threading.Lock()

    def _call_client(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run one mem0 client call with exclusive access to the underlying store.

        Args:
            func: Bound mem0 method to invoke.
            *args: Positional arguments for the call.
            **kwargs: Keyword arguments for the call.

        Returns:
            Whatever the mem0 call returns.

        Raises:
            TimeoutError: If the store is still busy after the configured timeout.
        """
        budget = self._timeout if self._timeout else 5.0
        if not self._client_lock.acquire(timeout=budget):
            raise TimeoutError(f"memory backend busy for more than {budget:.1f}s")
        try:
            return func(*args, **kwargs)
        finally:
            self._client_lock.release()

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
            future = asyncio.to_thread(
                self._call_client,
                self._memory.add,
                messages,
                user_id=user_id,
                metadata={**(metadata or {}), "namespace": user_id},
                infer=infer,
            )
            result = (
                await asyncio.wait_for(future, timeout=self._timeout)
                if self._timeout
                else await future
            )
            logger.info("memory_stored", user_id=user_id)
            return result
        except asyncio.TimeoutError:
            logger.warning("memory_store_timeout", user_id=user_id, timeout_s=self._timeout)
            return None
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
            future = asyncio.to_thread(
                self._call_client,
                self._memory.search,
                query,
                filters={"user_id": user_id},
                top_k=limit,
            )
            raw = (
                await asyncio.wait_for(future, timeout=self._timeout)
                if self._timeout
                else await future
            )
            # In Mem0 v1.1+, search returns {'results': [...]}. Handle both dict and list.
            if isinstance(raw, dict):
                results = raw.get("results", [])
            elif isinstance(raw, list):
                results = raw
            else:
                results = []

            score_threshold = float(self._config.get("score_threshold", 0.82))
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
            # Do not log raw user queries: they frequently contain personal data and
            # the database/query log is world-readable on some installations.
            logger.info("memory_searched", query_chars=len(query or ""), results=len(cleaned))
            return cleaned
        except asyncio.TimeoutError:
            logger.warning("memory_search_timeout", timeout_s=self._timeout)
            return []
        except Exception as e:
            logger.error("memory_search_failed", error=str(e))
            return []

    async def get_all(self, user_id: str = "user") -> list[dict]:
        """Get all memories for a user using Mem0 filters."""
        if not self._initialized or not self._memory:
            return []
        try:
            raw = await asyncio.to_thread(
                self._call_client, self._memory.get_all, filters={"user_id": user_id}
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

    Every public read path is scoped to a persona. Rows written by another
    persona are never returned, even when they share the same database file
    (legacy archives did exactly that). The SQLite connection is a single
    ``check_same_thread=False`` handle guarded by an ``RLock``; all execution
    and row materialization happen under that lock inside worker threads.
    """

    #: Schema migration version applied once per database file.
    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str = "data/database/conversations.db",
        personality: str | None = None,
        *,
        retention_days: int | None = 180,
        max_messages: int | None = None,
    ) -> None:
        self._in_memory = str(db_path) == ":memory:"
        p = Path(db_path)
        if not self._in_memory and not p.is_absolute():
            # Anchor relative defaults to the project root, not the process CWD.
            p = robot_config.get_project_root() / p
        self.db_path = p
        if not self._in_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(self.db_path.parent, 0o700)
            except OSError:
                pass
        self._conn: sqlite3.Connection | None = None
        # Shared connection is used from asyncio.to_thread worker threads and
        # read on the event loop; every execute/commit/fetch sequence must
        # hold this lock to serialize access to the single sqlite3 connection.
        self._db_lock = threading.RLock()

        # Persona context. An explicit argument (or a persona-scoped path such
        # as .../personas/neo/conversations.db) pins the namespace; otherwise
        # legacy/test databases infer it from their session/personality rows.
        inferred = personality or self._infer_personality_from_path(p)
        self._personality: str | None = normalize_persona(inferred) if inferred else None
        self._personality_explicit = bool(inferred)

        self._busy_timeout_ms = 5000
        self._write_retries = 3
        self._retention_days = self._normalize_retention(retention_days)
        self._max_messages = max_messages if (max_messages and max_messages > 0) else None
        self._last_retention_cleanup = 0.0

    # ------------------------------------------------------------------
    # Persona resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_personality_from_path(path: Path) -> str | None:
        """Infer a persona id from ``.../personas/<id>/conversations.db``."""
        try:
            parts = path.resolve().parts
        except OSError:
            return None
        for idx, part in enumerate(parts):
            if part == "personas" and idx + 1 < len(parts):
                candidate = parts[idx + 1].strip().lower()
                if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", candidate):
                    return candidate
        return None

    @staticmethod
    def _normalize_retention(value: Any) -> int | None:
        """Normalize a retention setting; ``None``/``<=0`` explicitly disables it."""
        if value is None:
            return None
        try:
            days = int(value)
        except (TypeError, ValueError):
            return 180
        return days if days > 0 else None

    @property
    def personality(self) -> str:
        """Active persona namespace for reads and writes."""
        return self._personality or "neo"

    @personality.setter
    def personality(self, value: str | None) -> None:
        """Pin the active persona namespace explicitly."""
        self._personality = normalize_persona(value)
        self._personality_explicit = True

    @property
    def has_explicit_personality(self) -> bool:
        """Whether the persona was supplied/path-derived rather than inferred."""
        return self._personality_explicit and self._personality is not None

    def _resolve_personality_sync(self, session_id: str | None = None) -> str:
        """Resolve the persona for an unscoped legacy/test database."""
        if self._personality:
            return self._personality
        if session_id and self._conn is not None:
            row = self._run_locked(
                lambda conn: conn.execute(
                    "SELECT personality FROM sessions WHERE id = ? LIMIT 1", (session_id,)
                ).fetchone()
            )
            if row and row[0]:
                return normalize_persona(str(row[0]))
        rows = self._run_locked(
            lambda conn: conn.execute(
                "SELECT DISTINCT personality FROM conversations "
                "WHERE personality IS NOT NULL AND TRIM(personality) != '' LIMIT 2"
            ).fetchall()
        ) or []
        if len(rows) == 1 and rows[0][0]:
            return normalize_persona(str(rows[0][0]))
        return "neo"

    async def _persona_for(
        self, personality: str | None = None, session_id: str | None = None
    ) -> str:
        """Resolve a persona without blocking the event loop."""
        if personality:
            return normalize_persona(personality)
        if self._personality:
            return self._personality
        return await asyncio.to_thread(self._resolve_personality_sync, session_id)

    # ------------------------------------------------------------------
    # Thread-safe SQLite helpers (all DB I/O goes through these)
    # ------------------------------------------------------------------

    def _run_locked(
        self, operation: Callable[[sqlite3.Connection], Any], *, retries: int | None = None
    ) -> Any:
        """Run ``operation(conn)`` atomically under the shared DB lock.

        ``database is locked`` is retried with bounded exponential backoff
        (the connection also carries ``busy_timeout=5000``). Any exception
        rolls back a partially applied transaction before propagating, so a
        failed multi-statement write can never leave the connection in an
        open transaction.

        Args:
            operation: Callable receiving the live connection.
            retries: Optional attempt override.

        Returns:
            Whatever ``operation`` returns, or ``None`` when not connected.
        """
        attempts = max(1, retries if retries is not None else self._write_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            busy = False
            with self._db_lock:
                if self._conn is None:
                    return None
                try:
                    return operation(self._conn)
                except sqlite3.OperationalError as exc:
                    message = str(exc).lower()
                    busy = "locked" in message or "busy" in message
                    try:
                        self._conn.rollback()
                    except sqlite3.Error:
                        pass
                    last_error = exc
                    if not busy or attempt == attempts - 1:
                        raise
                except sqlite3.Error:
                    try:
                        self._conn.rollback()
                    except sqlite3.Error:
                        pass
                    raise
                except Exception:
                    try:
                        self._conn.rollback()
                    except sqlite3.Error:
                        pass
                    raise
            # The external writer may release while we sleep outside the lock.
            time.sleep(min(0.05 * (2 ** attempt), 0.5))
        if last_error is not None:
            raise last_error
        return None

    def _require_connection(self) -> sqlite3.Connection:
        """Return the live connection or raise instead of silently dropping a write."""
        if self._conn is None:
            raise RuntimeError("conversation database is not initialized")
        return self._conn

    def _query_rows(self, sql: str, params: tuple = ()) -> list:
        """Execute a SELECT and materialize all rows in one locked thread task."""
        def _op(conn: sqlite3.Connection) -> list:
            return conn.execute(sql, params).fetchall()

        rows = self._run_locked(_op)
        return rows if rows is not None else []

    def _query_one(self, sql: str, params: tuple = ()) -> Any:
        """Execute a SELECT and fetch one row in one locked thread task."""
        def _op(conn: sqlite3.Connection) -> Any:
            return conn.execute(sql, params).fetchone()

        return self._run_locked(_op)

    def _execute_commit(self, statements: list[tuple[str, tuple]]) -> None:
        """Execute one or more statements and commit atomically under the DB lock."""
        self._require_connection()

        def _op(conn: sqlite3.Connection) -> None:
            try:
                for sql, params in statements:
                    conn.execute(sql, params)
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise

        try:
            self._run_locked(_op)
        except Exception as exc:
            logger.error("history_write_failed", error=type(exc).__name__, statements=len(statements))
            raise
        if not self._in_memory:
            _harden_permissions(self.db_path)

    async def initialize(self) -> None:
        """Create tables, indexes and FTS triggers, then run one-time migrations."""
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        """Blocking schema setup; runs under the DB lock in a worker thread."""
        with self._db_lock:
            if self._conn is not None:
                return
            self._conn = sqlite3.connect(
                str(self.db_path) if not self._in_memory else ":memory:",
                timeout=5.0,
                check_same_thread=False,
            )
            conn = self._conn
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            conn.execute("PRAGMA foreign_keys=ON")

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

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    personality TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    turn_count INTEGER DEFAULT 0,
                    title TEXT DEFAULT 'New Chat',
                    updated_at TEXT
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

                CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                    INSERT INTO conversations_fts(rowid, content) VALUES (new.id, new.content);
                END;

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

                CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                    INSERT INTO facts_fts(facts_fts, rowid, fact)
                    VALUES ('delete', old.id, old.fact);
                END;

                CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                    INSERT INTO facts_fts(facts_fts, rowid, fact)
                    VALUES ('delete', old.id, old.fact);
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

            # Migrate existing sessions table if title or updated_at are missing.
            try:
                cur = conn.execute("PRAGMA table_info(sessions)")
                existing_cols = {row[1] for row in cur.fetchall()}
                if "title" not in existing_cols:
                    conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT DEFAULT 'New Chat'")
                if "updated_at" not in existing_cols:
                    conn.execute("ALTER TABLE sessions ADD COLUMN updated_at TEXT")
                conn.commit()
            except sqlite3.Error as exc:
                logger.warning("history_sessions_migration_failed", error=str(exc))

            self._apply_persona_indexes(conn)
            self._apply_migrations(conn)
            self._repair_fts_if_needed(conn)
            if not self._in_memory:
                _harden_permissions(self.db_path)
            logger.info(
                "history_initialized",
                db=str(self.db_path),
                personality=self.personality,
                explicit=self._personality_explicit,
            )

    def _apply_persona_indexes(self, conn: sqlite3.Connection) -> None:
        """Install the expression indexes that persona-scoped queries actually use.

        Every persona predicate normalizes the column
        (``LOWER(COALESCE(NULLIF(TRIM(personality), ''), 'neo'))``), and SQLite
        only uses an expression index when the queried expression matches the
        indexed one. The older plain-personality composites can never match, so
        they are dropped: they cost a write per message and serve no read.
        """
        normalized = _persona_expr("personality")
        statements = [
            # Superseded plain-column composites.
            "DROP INDEX IF EXISTS idx_conversations_session_persona",
            "DROP INDEX IF EXISTS idx_conversations_persona_timestamp",
            "DROP INDEX IF EXISTS idx_sessions_persona_updated",
            f"CREATE INDEX IF NOT EXISTS idx_conversations_persona_norm "
            f"ON conversations({normalized})",
            f"CREATE INDEX IF NOT EXISTS idx_conversations_session_persona_norm "
            f"ON conversations(session_id, {normalized})",
            f"CREATE INDEX IF NOT EXISTS idx_sessions_persona_norm "
            f"ON sessions({normalized}, updated_at)",
        ]
        for sql in statements:
            try:
                conn.execute(sql)
            except sqlite3.Error as exc:
                logger.warning("persona_index_failed", error=str(exc)[:200])
        conn.commit()

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply the one-time schema/data migration (idempotent, versioned)."""
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = int(row[0]) if row and row[0] is not None else 0
            if current >= self.SCHEMA_VERSION:
                conn.commit()
                return

            # 1. Legacy system-event titles.
            conn.execute("UPDATE sessions SET title = 'Voice Chat' WHERE title LIKE '[%'")

            # 2. Backfill session rows for historical conversations exactly once.
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions
                    (id, personality, started_at, updated_at, turn_count, title)
                SELECT
                    c.session_id,
                    LOWER(COALESCE(NULLIF(TRIM(c.personality), ''), 'neo')) AS pers,
                    MIN(c.timestamp) AS started,
                    MAX(c.timestamp) AS updated,
                    COUNT(*) / 2 AS turns,
                    COALESCE(
                        (SELECT SUBSTR(u.content, 1, 36) ||
                                (CASE WHEN LENGTH(u.content) > 36 THEN '...' ELSE '' END)
                         FROM conversations u
                         WHERE u.session_id = c.session_id
                           AND u.role = 'user'
                           AND u.content NOT LIKE '[%'
                         ORDER BY u.id ASC LIMIT 1),
                        'Voice Chat'
                    ) AS title
                FROM conversations c
                WHERE c.session_id NOT IN (SELECT id FROM sessions)
                GROUP BY c.session_id;
                """
            )

            # 3. Remove known STT/lyric poison once, then rebuild the FTS index.
            #    Every clause must match a distinctive multi-word fragment: a
            #    short pattern here silently destroys legitimate facts (an
            #    earlier "%cold.%" clause deleted anything mentioning a cold).
            poison_clauses = [
                "fact LIKE '%world would never ever be true%'",
                "fact LIKE '%never love my whole game%'",
                "fact LIKE '%point where I love and I hate you%'",
                "fact LIKE '%my name is logged out in jail%'",
                "fact LIKE '%current activity: going to school%'",
                "fact LIKE '%Dove Cameron%'",
                "fact LIKE '%skip to content%'",
            ]
            deleted = conn.execute(
                f"DELETE FROM facts WHERE {' OR '.join(poison_clauses)}"
            ).rowcount

            # 4. Deduplicate only exact normalized duplicates; distinct values
            #    (e.g. two favorite colors) are always preserved.
            conn.execute(
                "DELETE FROM facts WHERE id NOT IN ("
                "SELECT MIN(id) FROM facts GROUP BY user_id, LOWER(fact))"
            )
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_user_fact_norm "
                    "ON facts(user_id, LOWER(fact))"
                )
            except sqlite3.Error as exc:
                # A legacy duplicate that survived normalization must never
                # prevent startup; add_fact still performs lock-protected dedupe.
                logger.warning("facts_unique_index_failed", error=str(exc))

            # 5. Rebuild both external-content FTS indexes with the FTS5 rebuild
            #    command (DELETE FROM an external-content table is invalid).
            conn.execute("INSERT INTO conversations_fts(conversations_fts) VALUES('rebuild')")
            conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")

            conn.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            logger.info(
                "history_migration_applied",
                version=self.SCHEMA_VERSION,
                poison_deleted=deleted,
                db=str(self.db_path),
            )
        except sqlite3.Error as exc:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            logger.error("history_migration_failed", error=str(exc), db=str(self.db_path))

    @staticmethod
    def _repair_fts_if_needed(conn: sqlite3.Connection) -> None:
        """Verify FTS indexes and rebuild them when orphan/mismatched rows exist."""
        pairs = (
            ("conversations", "conversations_fts"),
            ("facts", "facts_fts"),
        )
        for base_table, fts_table in pairs:
            try:
                base_count = int(conn.execute(f"SELECT COUNT(*) FROM {base_table}").fetchone()[0])
                fts_count = int(conn.execute(f"SELECT COUNT(*) FROM {fts_table}").fetchone()[0])
                if base_count != fts_count:
                    logger.warning(
                        "history_fts_orphans_detected",
                        table=fts_table,
                        base_count=base_count,
                        fts_count=fts_count,
                    )
                    conn.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')")
                else:
                    try:
                        conn.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('integrity-check')")
                    except sqlite3.Error:
                        logger.warning("history_fts_integrity_failed", table=fts_table)
                        conn.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')")
                conn.commit()
            except sqlite3.Error as exc:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                logger.warning("history_fts_check_failed", table=fts_table, error=str(exc))

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def configure_retention(
        self, retention_days: Any = 180, max_messages: Any = None
    ) -> None:
        """Configure periodic archive retention without touching recent data."""
        self._retention_days = self._normalize_retention(retention_days)
        try:
            limit = int(max_messages) if max_messages is not None else 0
        except (TypeError, ValueError):
            limit = 0
        self._max_messages = limit if limit > 0 else None

    async def maybe_cleanup_retention(self, *, force: bool = False) -> int:
        """Run cheap bounded retention cleanup at most once per interval."""
        now = time.monotonic()
        if not force and now - self._last_retention_cleanup < 3600:
            return 0
        self._last_retention_cleanup = now
        return await asyncio.to_thread(self._cleanup_retention_sync, force)

    def _cleanup_retention_sync(self, force: bool = False) -> int:
        """Delete a bounded batch of old archive rows and vacuum occasionally."""
        if self._retention_days is None and self._max_messages is None:
            return 0
        deleted_total = 0
        cutoff = None
        if self._retention_days is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self._retention_days)
            ).isoformat()

        def _op(conn: sqlite3.Connection) -> int:
            deleted = 0
            try:
                if cutoff is not None:
                    cur = conn.execute(
                        "DELETE FROM conversations WHERE id IN ("
                        "SELECT id FROM conversations WHERE timestamp < ? LIMIT 500)",
                        (cutoff,),
                    )
                    deleted += max(0, cur.rowcount)
                    cur = conn.execute(
                        "DELETE FROM sessions WHERE id NOT IN "
                        "(SELECT DISTINCT session_id FROM conversations) "
                        "AND COALESCE(updated_at, started_at) < ?",
                        (cutoff,),
                    )
                    deleted += max(0, cur.rowcount)
                if self._max_messages is not None:
                    row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
                    total = int(row[0]) if row else 0
                    excess = max(0, total - self._max_messages)
                    if excess:
                        cut = min(excess, 1000)
                        cur = conn.execute(
                            "DELETE FROM conversations WHERE id IN ("
                            "SELECT id FROM conversations ORDER BY id ASC LIMIT ?)",
                            (cut,),
                        )
                        deleted += max(0, cur.rowcount)
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            return deleted

        try:
            deleted_total = int(self._run_locked(_op) or 0)
        except Exception as exc:
            logger.warning("history_retention_cleanup_failed", error=type(exc).__name__)
            return 0

        if deleted_total:
            logger.info("history_retention_cleaned", rows=deleted_total, db=str(self.db_path))
            # VACUUM at most once per forced/manual cycle; it is never part of
            # a normal turn's write path.
            if force:
                try:
                    with self._db_lock:
                        if self._conn is not None:
                            self._conn.execute("VACUUM")
                except sqlite3.Error as exc:
                    logger.warning("history_vacuum_failed", error=str(exc))
            if not self._in_memory:
                _harden_permissions(self.db_path)
        return deleted_total


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
        """Record a conversation message.

        Raises:
            RuntimeError: If the database has not been initialized. A write is
                never silently discarded.
        """
        self._require_connection()
        t_res_str = json.dumps(tool_result) if isinstance(tool_result, (dict, list)) else tool_result
        safe_metadata = _sanitize_metadata(metadata)
        await asyncio.to_thread(
            self._execute_commit,
            [
                (
                    """INSERT INTO conversations
                       (session_id, role, content, personality, turn_number,
                        tool_name, tool_result, timestamp, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        role,
                        content,
                        normalize_persona(personality),
                        turn_number,
                        tool_name,
                        t_res_str,
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps(safe_metadata) if safe_metadata else None,
                    ),
                )
            ],
        )

    async def search(
        self, query: str, limit: int = 10, personality: str | None = None
    ) -> list[dict]:
        """Full-text search scoped to one persona.

        Args:
            query: Raw user query; tokenized and quoted before reaching FTS5.
            limit: Maximum rows.
            personality: Persona namespace; defaults to the active persona.

        Returns:
            Matching messages, or an empty list for an unsearchable query.
        """
        if not self._conn:
            return []
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        persona = await self._persona_for(personality)
        clause, variants = _clause_for(persona, "c.personality")
        try:
            rows = await asyncio.to_thread(
                self._query_rows,
                f"""SELECT c.role, c.content, c.timestamp, c.session_id
                   FROM conversations_fts f
                   JOIN conversations c ON c.id = f.rowid
                   WHERE f.content MATCH ? AND {clause}
                   ORDER BY c.timestamp DESC
                   LIMIT ?""",
                (safe_query, *variants, limit),
            )
        except sqlite3.Error as exc:
            logger.warning("history_search_failed", error=str(exc)[:200], query_chars=len(query or ""))
            return []
        return [
            {"role": r[0], "content": r[1], "timestamp": r[2], "session_id": r[3]} for r in rows
        ]

    async def get_recent(
        self, session_id: str, limit: int = 20, personality: str | None = None
    ) -> list[dict]:
        """Get recent messages from the current session within one persona.

        Falls back to recent turns within the persona across sessions if the current
        session has fewer than 2 messages, preserving follow-up conversation context.
        Attaches executed tool results to assistant turns so the model remains aware
        of real-time actions and creations across turns.
        """
        if not self._conn:
            return []
        persona = await self._persona_for(personality, session_id)
        clause, variants = _clause_for(persona)
        rows = await asyncio.to_thread(
            self._query_rows,
            f"""SELECT role, content, timestamp, tool_name, tool_result FROM conversations
               WHERE session_id = ? AND {clause}
               ORDER BY id DESC LIMIT ?""",
            (session_id, *variants, limit),
        )
        if len(rows) < 2:
            cross_rows = await asyncio.to_thread(
                self._query_rows,
                f"""SELECT role, content, timestamp, tool_name, tool_result FROM conversations
                   WHERE {clause}
                   ORDER BY id DESC LIMIT ?""",
                (*variants, limit),
            )
            if cross_rows:
                rows = cross_rows

        res = []
        for r in reversed(rows):
            role, content, ts, t_name, t_res = r[0], r[1], r[2], r[3], r[4]
            if not content or content.strip() in ("[", "]", "<", ">") or content.strip() == "[Wake word acknowledge]":
                continue
            if role == "assistant" and content.strip() == "I'm listening.":
                continue
            if role == "assistant" and t_name and t_res:
                clean_res = str(t_res).strip()
                if clean_res and clean_res not in content:
                    content = f"{content}\n[Tool Result: {t_name} -> {clean_res}]"
            res.append({"role": role, "content": content, "timestamp": ts})
        return res

    async def clear_turn(
        self, session_id: str, turn_number: int = 0, personality: str | None = None
    ) -> None:
        """Clear a specific turn (e.g. turn 0 bootstrap) from session history."""
        self._require_connection()
        persona = await self._persona_for(personality, session_id)
        clause, variants = _clause_for(persona)
        await asyncio.to_thread(
            self._execute_commit,
            [
                (
                    "DELETE FROM conversations WHERE session_id = ? AND turn_number = ? "
                    f"AND {clause}",
                    (session_id, turn_number, *variants),
                )
            ],
        )

    async def create_session(
        self, session_id: str | None = None, personality: str = "neo", title: str = "New Chat"
    ) -> dict:
        """Create or register a new isolated conversation session."""
        if not session_id:
            session_id = generate_session_id()
        persona = normalize_persona(personality)
        if not self._conn:
            return {"id": session_id, "session_id": session_id, "personality": persona, "title": title}
        now_iso = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(
            self._execute_commit,
            [
                (
                    """INSERT OR IGNORE INTO sessions (id, personality, started_at, updated_at, turn_count, title)
                       VALUES (?, ?, ?, ?, 0, ?)""",
                    (session_id, persona, now_iso, now_iso, title),
                )
            ],
        )
        return {
            "id": session_id,
            "session_id": session_id,
            "personality": persona,
            "title": title,
            "started_at": now_iso,
            "updated_at": now_iso,
            "turn_count": 0,
        }

    async def list_sessions(
        self, limit: int = 150, personality: str | None = None
    ) -> list[dict]:
        """List past chat sessions for one persona, most recently active first.

        This is a pure read path: legacy session backfill happens once during
        initialize (schema migration), never on a GET request.
        """
        if not self._conn:
            return []
        persona = await self._persona_for(personality)
        sess_clause, sess_variants = _clause_for(persona, "s.personality")
        msg_clause, msg_variants = _clause_for(persona, "c.personality")
        rows = await asyncio.to_thread(
            self._query_rows,
            f"""SELECT s.id, s.personality, s.started_at, s.updated_at, s.turn_count, s.title,
                      (SELECT c.content FROM conversations c
                        WHERE c.session_id = s.id AND c.role = 'assistant' AND {msg_clause}
                        ORDER BY c.id DESC LIMIT 1) as last_assistant_msg,
                      (SELECT c.content FROM conversations c
                        WHERE c.session_id = s.id AND c.role = 'user' AND c.content NOT LIKE '[%'
                          AND {msg_clause}
                        ORDER BY c.id ASC LIMIT 1) as first_user_msg
               FROM sessions s
               WHERE {sess_clause}
               ORDER BY COALESCE(s.updated_at, s.started_at) DESC
               LIMIT ?""",
            (*msg_variants, *msg_variants, *sess_variants, limit),
        )
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

    async def get_session(
        self, session_id: str, personality: str | None = None
    ) -> dict | None:
        """Get session metadata by ID.

        Args:
            session_id: Unique session identifier.
            personality: Optional personality filter. If omitted, returns the session
                matching the ID regardless of persona.

        Returns:
            Dictionary with session metadata, or None if not found.
        """
        if not self._conn:
            return None
        if personality:
            persona = await self._persona_for(personality, session_id)
            sess_clause, sess_variants = _clause_for(persona, "personality")
            row = await asyncio.to_thread(
                self._query_one,
                f"""SELECT id, personality, started_at, updated_at, turn_count, title
                   FROM sessions
                   WHERE id = ? AND {sess_clause}""",
                (session_id, *sess_variants),
            )
        else:
            row = await asyncio.to_thread(
                self._query_one,
                """SELECT id, personality, started_at, updated_at, turn_count, title
                   FROM sessions
                   WHERE id = ?""",
                (session_id,),
            )
        if not row:
            if personality:
                persona = await self._persona_for(personality, session_id)
                msg_clause, msg_variants = _clause_for(persona, "c.personality")
                c_row = await asyncio.to_thread(
                    self._query_one,
                    f"""SELECT c.session_id, COALESCE(MAX(c.personality), 'neo'),
                               MIN(c.timestamp), MAX(c.timestamp), COUNT(*)/2,
                              COALESCE(
                                (SELECT SUBSTR(u.content, 1, 36)
                                 FROM conversations u
                                 WHERE u.session_id = c.session_id AND u.role = 'user'
                                   AND u.content NOT LIKE '[%'
                                 ORDER BY u.id ASC LIMIT 1),
                                'Voice Chat'
                              )
                       FROM conversations c
                       WHERE c.session_id = ? AND {msg_clause}
                       GROUP BY c.session_id""",
                    (session_id, *msg_variants),
                )
            else:
                c_row = await asyncio.to_thread(
                    self._query_one,
                    """SELECT c.session_id, COALESCE(MAX(c.personality), 'neo'),
                               MIN(c.timestamp), MAX(c.timestamp), COUNT(*)/2,
                              COALESCE(
                                (SELECT SUBSTR(u.content, 1, 36)
                                 FROM conversations u
                                 WHERE u.session_id = c.session_id AND u.role = 'user'
                                   AND u.content NOT LIKE '[%'
                                 ORDER BY u.id ASC LIMIT 1),
                                'Voice Chat'
                              )
                       FROM conversations c
                       WHERE c.session_id = ?
                       GROUP BY c.session_id""",
                    (session_id,),
                )
            if not c_row:
                return None
            return {
                "id": c_row[0],
                "personality": normalize_persona(str(c_row[1] or "neo")),
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

    async def get_stats(
        self, personality: str | None = None, user_id: str | None = None
    ) -> dict[str, int]:
        """Return persona-scoped memory statistics.

        Args:
            personality: Persona whose conversations/sessions are counted.
            user_id: Optional fact namespace filter.

        Returns:
            Dictionary containing facts_count, messages_count, and sessions_count.
        """
        if not self._conn:
            return {"facts_count": 0, "messages_count": 0, "sessions_count": 0}
        persona = await self._persona_for(personality)
        msg_clause, variants = _clause_for(persona, "personality")
        sess_clause, _ = _clause_for(persona, "personality")

        def _op(conn: sqlite3.Connection) -> dict[str, int]:
            if user_id:
                f_count = int(
                    conn.execute("SELECT COUNT(*) FROM facts WHERE user_id = ?", (user_id,)).fetchone()[0]
                )
            else:
                f_count = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            m_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM conversations WHERE {msg_clause}", variants
                ).fetchone()[0]
            )
            s_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM sessions WHERE {sess_clause}", variants
                ).fetchone()[0]
            )
            return {"facts_count": f_count, "messages_count": m_count, "sessions_count": s_count}

        return await asyncio.to_thread(self._run_locked, _op)

    async def get_session_messages(
        self, session_id: str, limit: int = 200, personality: str | None = None
    ) -> list[dict]:
        """Get all messages for an isolated session.

        Args:
            session_id: Unique session identifier.
            limit: Maximum count of messages to return.
            personality: Optional personality filter. If omitted, returns all messages
                for this session.

        Returns:
            List of message dictionaries.
        """
        if not self._conn:
            return []
        if personality:
            persona = await self._persona_for(personality, session_id)
            clause, variants = _clause_for(persona)
            query_sql = f"""SELECT id, role, content, personality, turn_number, tool_name,
                               tool_result, timestamp, metadata
                        FROM conversations
                        WHERE session_id = ? AND {clause}
                        ORDER BY id ASC
                        LIMIT ?"""
            params = (session_id, *variants, limit)
        else:
            query_sql = """SELECT id, role, content, personality, turn_number, tool_name,
                               tool_result, timestamp, metadata
                        FROM conversations
                        WHERE session_id = ?
                        ORDER BY id ASC
                        LIMIT ?"""
            params = (session_id, limit)
        rows = await asyncio.to_thread(self._query_rows, query_sql, params)
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

    async def delete_session(self, session_id: str, personality: str | None = None) -> bool:
        """Delete an isolated session and all its messages within one persona."""
        self._require_connection()
        persona = await self._persona_for(personality, session_id)
        msg_clause, msg_variants = _clause_for(persona)
        sess_clause, sess_variants = _clause_for(persona)
        await asyncio.to_thread(
            self._execute_commit,
            [
                (
                    f"DELETE FROM conversations WHERE session_id = ? AND {msg_clause}",
                    (session_id, *msg_variants),
                ),
                (
                    f"DELETE FROM sessions WHERE id = ? AND {sess_clause}",
                    (session_id, *sess_variants),
                ),
            ],
        )
        return True

    async def update_session(
        self,
        session_id: str,
        title: str | None = None,
        turn_count: int | None = None,
        personality: str | None = None,
    ) -> None:
        """Update session metadata within one persona."""
        self._require_connection()
        persona = await self._persona_for(personality, session_id)
        clause, variants = _clause_for(persona)
        now_iso = datetime.now(timezone.utc).isoformat()
        if title is not None and turn_count is not None:
            sql = (
                "UPDATE sessions SET title = ?, turn_count = ?, updated_at = ? "
                f"WHERE id = ? AND {clause}"
            )
            params: tuple = (title, turn_count, now_iso, session_id, *variants)
        elif title is not None:
            sql = f"UPDATE sessions SET title = ?, updated_at = ? WHERE id = ? AND {clause}"
            params = (title, now_iso, session_id, *variants)
        elif turn_count is not None:
            sql = f"UPDATE sessions SET turn_count = ?, updated_at = ? WHERE id = ? AND {clause}"
            params = (turn_count, now_iso, session_id, *variants)
        else:
            sql = f"UPDATE sessions SET updated_at = ? WHERE id = ? AND {clause}"
            params = (now_iso, session_id, *variants)
        await asyncio.to_thread(self._execute_commit, [(sql, params)])

    async def update_session_title(
        self, session_id: str, title: str, personality: str | None = None
    ) -> None:
        """Update session title."""
        await self.update_session(session_id, title=title, personality=personality)

    async def update_session_title_if_default(
        self, session_id: str, new_title: str, personality: str | None = None
    ) -> None:
        """Update session title only if it is currently a generic default or system event string."""
        self._require_connection()
        persona = await self._persona_for(personality, session_id)
        clause, variants = _clause_for(persona)
        await asyncio.to_thread(
            self._execute_commit,
            [
                (
                    f"""UPDATE sessions
                       SET title = ?
                       WHERE id = ? AND {clause}
                         AND (title IN ('New Chat', 'Voice Chat', 'Chat', 'New Conversation', '')
                              OR title LIKE '[%')""",
                    (new_title, session_id, *variants),
                )
            ],
        )

    async def add_fact(self, fact: str, user_id: str = "user", key: str | None = None) -> bool:
        """Store a durable fact in SQLite, preventing exact duplicates.

        The dedupe SELECT, INSERT, and commit run as one locked transaction so
        concurrent ``add_fact`` calls cannot race past the duplicate check.

        Returns:
            True when a new row was inserted, False for a duplicate or blank fact.
        """
        self._require_connection()
        clean_fact = (fact or "").strip()
        if not clean_fact:
            return False
        namespace = (user_id or "").strip() or "user"

        def _add_atomic(conn: sqlite3.Connection) -> bool:
            try:
                row = conn.execute(
                    "SELECT id FROM facts WHERE user_id = ? AND LOWER(fact) = LOWER(?) LIMIT 1",
                    (namespace, clean_fact),
                ).fetchone()
                if row:
                    conn.commit()
                    return False
                conn.execute(
                    """INSERT INTO facts (key, fact, user_id, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (key, clean_fact, namespace, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                return True
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise

        result = await asyncio.to_thread(self._run_locked, _add_atomic)
        if result and not self._in_memory:
            _harden_permissions(self.db_path)
        return bool(result)

    async def search_facts(self, query: str, user_id: str = "user", limit: int = 5) -> list[dict]:
        """Search persistent facts for specific user/namespace using FTS5, LIKE fallback, and deduplication.

        Zero-pollution design:
        - Casual greetings/pleasantries immediately return empty lists.
        - Short words (<4 chars) use exact string matching to prevent prefix poisoning ('hi*' -> 'his').
        - Empty keyword extractions return empty lists unless query matches broad phrases.
        """
        if not self._conn:
            return []

        clean_q = (query or "").strip().lower()

        # 1. Greeting & Pleasantry Guard: never pollute conversational openings with durable facts
        greeting_patterns = [
            r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|yo)(?:\s+(?:there|robot|neo|luna|cypher|glados|buddy|friend|assistant))?[\s!,.]*$",
            r"^(?:how\s+are\s+you|how's\s+it\s+going|how\s+are\s+you\s+doing|what's\s+up)[\s!,.]*\??$",
        ]
        if any(re.search(pat, clean_q) for pat in greeting_patterns):
            return []

        broad_phrases = {
            "all", "*", "everything", "facts", "memories", "durable facts",
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
            "hi", "hello", "hey", "good", "please", "thanks", "thank",
            "explain", "describe", "done", "much", "many",
            "s", "t", "d", "ll", "ve", "re",
        }
        words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", clean_q) if w not in stop_and_meta]
        if not words:
            # Check if query is an explicit inquiry about saved facts/memories/preferences
            clean_stripped = clean_q.rstrip("?!., ")
            meta_inquiry_terms = {"durable", "facts", "fact", "memories", "memory", "saved", "stored", "remembered", "preferences", "preference"}
            has_meta_inquiry = any(t in clean_q for t in meta_inquiry_terms)
            is_broad = clean_stripped in broad_phrases or any(clean_stripped.startswith(bp) for bp in broad_phrases) or has_meta_inquiry
            if is_broad:
                return await self.get_all_facts(user_id=user_id, limit=limit)
            return []

        if words:
            # 1. Prefix FTS matching with stop-word removal. For words >= 4 chars, use prefix matching to cover inflections (e.g. work* -> works, allerg* -> allergies)
            fts_terms = [f"{w[:4]}*" if len(w) >= 4 else f'"{w}"' for w in words]
            fts_expr = " OR ".join(fts_terms)
            try:
                rows = await asyncio.to_thread(
                    self._query_rows,
                    """SELECT f.id, f.fact, f.created_at FROM facts_fts ft
                       JOIN facts f ON f.id = ft.rowid
                       WHERE ft.fact MATCH ? AND f.user_id = ?
                       ORDER BY f.id DESC LIMIT ?""",
                    (fts_expr, user_id, limit * 3),
                )
                if rows:
                    return self._deduplicate_fact_rows(rows, limit)
            except sqlite3.Error as exc:
                logger.warning("facts_fts_search_failed", error=str(exc)[:200])

            # 2. Individual keyword LIKE fallback
            like_clauses = " OR ".join("fact LIKE ?" for _ in words)
            params = [f"%{w}%" for w in words] + [user_id, limit * 3]
            try:
                rows = await asyncio.to_thread(
                    self._query_rows,
                    f"""SELECT id, fact, created_at FROM facts
                        WHERE ({like_clauses}) AND user_id = ?
                        ORDER BY id DESC LIMIT ?""",
                    tuple(params),
                )
                if rows:
                    return self._deduplicate_fact_rows(rows, limit)
            except sqlite3.Error as exc:
                logger.warning("facts_like_search_failed", error=str(exc)[:200])

        # If query specified keywords but no fact matched, return empty list
        # to prevent irrelevant memory pollution during unrelated commands or tasks.
        return []

    @staticmethod
    def _fact_identity(fact_text: str) -> tuple[str, str] | None:
        """Extract a ``(predicate/entity, value)`` pair from a fact sentence.

        Two facts are redundant only when both the predicate and the value
        match. "favorite color is blue" and "favorite color is red" therefore
        remain distinct even though they share words.

        Args:
            fact_text: Normalized fact sentence.

        Returns:
            ``(key, value)`` or ``None`` when the sentence is not structured.
        """
        text = " ".join(fact_text.lower().split()).strip(" .!?;")
        if not text:
            return None
        text = re.sub(r"\buser's\b", "user", text)
        text = re.sub(r"\bameya's\b", "ameya", text)
        text = re.sub(r"'s\b", "", text)
        match = re.match(
            r"^(?P<key>.+?)\s+(?:is|are|was|were|likes?|loves?|prefers?|has|have|=|:)\s+"
            r"(?P<value>.+)$",
            text,
        )
        if not match:
            return None
        key = " ".join(match.group("key").split())
        value = match.group("value").strip()
        # Elaborations after a comma/semicolon add commentary, not a new value
        # ("crush is X, and ..." still records the same crush X).
        value = re.split(r"[,;]", value, maxsplit=1)[0].strip()
        if not key or not value:
            return None
        return key, value

    @staticmethod
    def _deduplicate_fact_rows(rows: list[tuple], limit: int) -> list[dict]:
        """Filter redundant facts without collapsing facts whose values differ."""
        unique_facts: list[dict] = []
        seen_exact: set[str] = set()
        seen_structured: dict[str, set[str]] = {}

        for r in rows:
            fact_text = str(r[1]).strip()
            normalized = " ".join(fact_text.lower().split())
            if not normalized:
                continue
            if normalized in seen_exact:
                continue

            identity = ConversationHistory._fact_identity(fact_text)
            if identity is not None:
                key, value = identity
                values = seen_structured.setdefault(key, set())
                if value in values:
                    # Same subject/predicate AND same value: redundant phrasing.
                    continue
                values.add(value)
            seen_exact.add(normalized)
            unique_facts.append(
                {"id": r[0], "memory": fact_text, "fact": fact_text, "created_at": r[2]}
            )
            if len(unique_facts) >= limit:
                break

        return unique_facts

    async def get_all_facts(self, user_id: str = "user", limit: int = 500) -> list[dict]:
        """Retrieve all durable facts for one explicit namespace, deduplicated."""
        if not self._conn:
            return []
        rows = await asyncio.to_thread(
            self._query_rows,
            """SELECT id, fact, created_at FROM facts
               WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, min(max(int(limit) * 3, 100), MAX_FACT_SCAN_ROWS)),
        )
        return self._deduplicate_fact_rows(rows, limit)

    async def get_facts(self, user_id: str = "user", limit: int = 500) -> list[dict]:
        """Alias for get_all_facts."""
        return await self.get_all_facts(user_id=user_id, limit=limit)

    async def get_user_profile(self, user_id: str = "user") -> dict[str, Any]:
        """Retrieve partitioned user profile from SQLite.

        Args:
            user_id: Personality namespace or user identifier.

        Returns:
            Dictionary with durable_preferences, temporary_context, inferred_facts, summary_markdown.
        """
        if not self._conn:
            return {
                "user_id": user_id,
                "durable_preferences": {},
                "temporary_context": {},
                "inferred_facts": [],
                "summary_markdown": "",
                "updated_at": "",
            }

        def _get() -> dict[str, Any]:
            def _op(conn: sqlite3.Connection) -> tuple | None:
                return conn.execute(
                    "SELECT user_id, durable_preferences, temporary_context, inferred_facts, summary_markdown, updated_at "
                    "FROM user_profiles WHERE user_id = ?",
                    (user_id,),
                ).fetchone()

            row = self._run_locked(_op)
            if not row:
                return {
                    "user_id": user_id,
                    "durable_preferences": {},
                    "temporary_context": {},
                    "inferred_facts": [],
                    "summary_markdown": "",
                    "updated_at": "",
                }
            import json
            return {
                "user_id": row[0],
                "durable_preferences": json.loads(row[1]) if row[1] else {},
                "temporary_context": json.loads(row[2]) if row[2] else {},
                "inferred_facts": json.loads(row[3]) if row[3] else [],
                "summary_markdown": row[4] or "",
                "updated_at": row[5] or "",
            }

        return await asyncio.to_thread(_get)

    async def update_user_profile(
        self,
        user_id: str,
        durable_preferences: dict[str, Any] | None = None,
        temporary_context: dict[str, Any] | None = None,
        inferred_facts: list[str] | None = None,
        summary_markdown: str = "",
    ) -> dict[str, Any]:
        """Update or insert partitioned user profile fields in SQLite.

        Args:
            user_id: Personality namespace or user identifier.
            durable_preferences: Long-term explicit user preferences.
            temporary_context: Current task or immediate topic state.
            inferred_facts: Inferred background facts.
            summary_markdown: Formatted markdown representation.

        Returns:
            Updated user profile dict.
        """
        if not self._conn:
            return {}

        import json
        now = datetime.now(timezone.utc).isoformat()
        current = await self.get_user_profile(user_id)
        d_pref = durable_preferences if durable_preferences is not None else current["durable_preferences"]
        t_ctx = temporary_context if temporary_context is not None else current["temporary_context"]
        i_facts = inferred_facts if inferred_facts is not None else current["inferred_facts"]
        s_md = summary_markdown or current["summary_markdown"]

        def _save() -> None:
            def _op(conn: sqlite3.Connection) -> None:
                conn.execute(
                    """
                    INSERT INTO user_profiles (user_id, durable_preferences, temporary_context, inferred_facts, summary_markdown, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        durable_preferences=excluded.durable_preferences,
                        temporary_context=excluded.temporary_context,
                        inferred_facts=excluded.inferred_facts,
                        summary_markdown=excluded.summary_markdown,
                        updated_at=excluded.updated_at
                    """,
                    (
                        user_id,
                        json.dumps(d_pref),
                        json.dumps(t_ctx),
                        json.dumps(i_facts),
                        s_md,
                        now,
                    ),
                )
                conn.commit()

            self._run_locked(_op)

        await asyncio.to_thread(_save)
        return {
            "user_id": user_id,
            "durable_preferences": d_pref,
            "temporary_context": t_ctx,
            "inferred_facts": i_facts,
            "summary_markdown": s_md,
            "updated_at": now,
        }

    async def get_archive(
        self,
        limit: int = 1000,
        exclude_session_id: str | None = None,
        personality: str | None = None,
    ) -> list[dict]:
        """Retrieve chronological past conversation history for one persona.

        Args:
            limit: Maximum rows to read (newest first, returned oldest first).
            exclude_session_id: Session to omit, usually the live one.
            personality: Persona to scope to; defaults to this database's persona.

        Returns:
            Archived messages as dicts with role, content, timestamp and session.
        """
        if not self._conn:
            return []
        persona = await self._persona_for(personality)
        clause, variants = _clause_for(persona, "personality")
        cap = max(1, min(int(limit), MAX_ARCHIVE_ROWS))
        if exclude_session_id:
            rows = await asyncio.to_thread(
                self._query_rows,
                f"""SELECT role, content, timestamp, session_id FROM conversations
                   WHERE session_id != ? AND {clause}
                   ORDER BY id DESC LIMIT ?""",
                (exclude_session_id, *variants, cap),
            )
        else:
            rows = await asyncio.to_thread(
                self._query_rows,
                f"""SELECT role, content, timestamp, session_id FROM conversations
                   WHERE {clause}
                   ORDER BY id DESC LIMIT ?""",
                (*variants, cap),
            )
        return [
            {"role": r[0], "content": r[1], "timestamp": r[2], "session_id": r[3]}
            for r in reversed(rows)
        ]

    async def health_check(self) -> bool:
        return self._conn is not None

    async def close(self) -> None:
        """Close SQLite connection cleanly under the DB lock."""
        with self._db_lock:
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

    async def switch_persona_database(self, db_path: str | Path) -> None:
        """Hot-swap the active SQLite database to an isolated persona database asynchronously.

        Args:
            db_path: Filesystem path to the target persona's SQLite database.
        """
        target = Path(db_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "history") and self.history and self.history._conn:
            try:
                with self.history._db_lock:
                    self.history._conn.close()
                    self.history._conn = None
            except Exception as e:
                logger.debug("history_close_error", error=str(e))
        self.history = ConversationHistory(str(target))
        await self.history.initialize()
        logger.info("switched_persona_database", path=str(target))

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

        # Sanitize assistant response from leaked internal tags/brackets before archiving
        clean_bot_text = re.sub(r"<thought>.*?</thought>", "", bot_text, flags=re.DOTALL)
        clean_bot_text = re.sub(r"\[\s*(?:(?:function\s+)?call\s*:\s*|(?:action|tool)\s*:\s*)?[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+(?:\([^\]]*\)|:[^\]]*|)\s*\]", "", clean_bot_text, flags=re.IGNORECASE)
        clean_bot_text = re.sub(r"\[\s*(?:conversation|tools?|call|meta|internal|action|motion|emotion|character\.)\b[^\]]*\]", "", clean_bot_text, flags=re.IGNORECASE)
        persisted_bot_text = clean_bot_text.strip() or bot_text.strip()

        # SQLite archive (always recorded with personality tag)
        await self.history.add_message(
            session_id, "user", user_text, personality, turn_number
        )
        await self.history.add_message(
            session_id,
            "assistant",
            persisted_bot_text,
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
        is_question = lower_u.endswith("?") or lower_u.startswith((
            "is ", "are ", "do ", "does ", "did ", "who ", "what ", "where ", "when ", "why ", "how ", "can ", "could "
        ))

        # Distinct biographical declarations (strictly guarded against questions and fleeting status)
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

        if not is_command and not is_question and any(re.search(pat, lower_u) for pat in bio_patterns):
            # Guard against fleeting conditions and song lyrics
            fleeting_markers = ("i am going", "i am standing", "i am sitting", "i am tired", "i am sleepy")
            if not any(fm in lower_u for fm in fleeting_markers):
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
        """Retrieve relevant memories strictly isolated within the personality workspace.

        Args:
            query: The user query string to search memories against.
            user_id: The personality namespace / user identifier.
            limit: Maximum count of memories to retrieve.

        Returns:
            list[dict]: Deduplicated relevant memories, or empty list for greetings/commands.
        """
        clean_q = (query or "").strip().lower()
        if not clean_q:
            return []

        # Zero-pollution gate: never query memory for greetings or direct actions
        greeting_patterns = [
            r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|yo)(?:\s+(?:there|robot|neo|luna|cypher|glados|buddy|friend|assistant))?[\s!,.]*$",
            r"^(?:how\s+are\s+you|how's\s+it\s+going|how\s+are\s+you\s+doing|what's\s+up)[\s!,.]*\??$",
        ]
        if any(re.search(pat, clean_q) for pat in greeting_patterns):
            return []

        action_patterns = [
            r"\b(?:volume|louder|quieter|mute|unmute)\b",
            r"\b(?:pause|resume|unpause|stop|replay|play\s+again)\b",
            r"^(?:play|put\s+on)\s+",
            r"\b(?:what\s+time|current\s+time|clock|time\s+is\s+it)\b",
            r"\b(?:system\s+status|health\s+check)\b",
        ]
        if any(re.search(pat, clean_q) for pat in action_patterns):
            return []

        # Zero-pollution gate: impersonal encyclopedic/factual questions without user reference
        objective_patterns = [
            r"^(?:how\s+(?:far|many|much|does|deep)|what\s+is\s+the\s+(?:capital|speed|atomic|chemical|deepest|half|boiling|currency|distance|primary)|who\s+(?:won|wrote|discovered)|what\s+causes|explain\s+(?:how|dijkstra|quantum)|can\s+quantum)\b",
        ]
        personal_pattern = r"\b(?:i|my|me|mine|myself|am\s+i|we|our|us|user|remember|memory|memories|preference|preferences|favorite|favourite|habit|routine)\b"
        has_personal_marker = bool(re.search(personal_pattern, clean_q))
        if not has_personal_marker and any(re.search(pat, clean_q) for pat in objective_patterns):
            return []

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
        self,
        limit: int = MAX_ARCHIVE_ROWS,
        exclude_session_id: str | None = None,
        personality: str | None = None,
    ) -> list[dict]:
        """Retrieve one persona's historical conversations for deep recall.

        Args:
            limit: Maximum messages to read; capped at MAX_ARCHIVE_ROWS.
            exclude_session_id: Session to omit, usually the live one.
            personality: Persona to scope to; defaults to the active persona.

        Returns:
            Archived messages, oldest first.
        """
        return await self.history.get_archive(
            limit=limit,
            exclude_session_id=exclude_session_id,
            personality=personality,
        )

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

    async def get_user_profile(self, user_id: str = "user") -> dict[str, Any]:
        """Retrieve partitioned user profile sheet."""
        return await self.history.get_user_profile(user_id=user_id)

    async def update_user_profile(
        self,
        user_id: str,
        durable_preferences: dict[str, Any] | None = None,
        temporary_context: dict[str, Any] | None = None,
        inferred_facts: list[str] | None = None,
        summary_markdown: str = "",
    ) -> dict[str, Any]:
        """Update partitioned user profile sheet."""
        return await self.history.update_user_profile(
            user_id=user_id,
            durable_preferences=durable_preferences,
            temporary_context=temporary_context,
            inferred_facts=inferred_facts,
            summary_markdown=summary_markdown,
        )

    async def summarize_user_profile(self, user_id: str = "user") -> dict[str, Any]:
        """Asynchronously analyze conversation history and facts to update partitioned User Profile Sheet.

        Extracts:
        - Durable preferences (favorite colors, foods, music, explicit preferences)
        - Temporary context (current session topics and tasks)
        - Inferred facts (technologies, dev habits, roles)

        Saves to SQLite user_profiles table and generates clean summary markdown.

        Args:
            user_id: Personality namespace / user identifier.

        Returns:
            Dictionary of updated profile partition fields.
        """
        facts = await self.get_all_facts(user_id=user_id, limit=50)
        recent_turns = await self.history.get_archive(limit=30)

        durable_prefs: dict[str, str] = {}
        temp_context: dict[str, str] = {}
        inferred: list[str] = []

        pref_keywords = ("favorite", "prefer", "likes", "loves", "dislikes", "hates")
        for f in facts:
            txt = f.get("fact", f.get("memory", "")).strip()
            if not txt:
                continue
            lower = txt.lower()
            if any(k in lower for k in pref_keywords):
                if not any(bad in lower for bad in ("crush", "telescope", "anubha", "allergy")):
                    durable_prefs[f"fact_{f.get('id', len(durable_prefs))}"] = txt
            else:
                if len(txt) < 120 and not any(bad in lower for bad in ("crush", "test")):
                    inferred.append(txt)

        # Inspect recent turns for immediate temporary topic
        for turn in reversed(recent_turns[-6:]):
            if turn.get("role") == "user":
                content = turn.get("content", "")
                if len(content) > 10 and not content.startswith("/"):
                    temp_context["last_active_topic"] = content[:80]
                    break

        # Assemble summary markdown
        lines = [
            "=== USER PROFILE SHEET ===",
            "• Identity: Ameya (AI Engineer & Robot Creator)",
        ]
        if durable_prefs:
            lines.append("Durable Preferences:")
            for p in list(durable_prefs.values())[:5]:
                lines.append(f"  • {p}")
        if inferred:
            lines.append("Inferred Context:")
            for inf in inferred[:4]:
                lines.append(f"  • {inf}")
        if temp_context.get("last_active_topic"):
            lines.append(f"Active Focus: {temp_context['last_active_topic']}")

        summary_md = "\n".join(lines)

        return await self.update_user_profile(
            user_id=user_id,
            durable_preferences=durable_prefs,
            temporary_context=temp_context,
            inferred_facts=inferred,
            summary_markdown=summary_md,
        )

    async def get_core_user_profile(self, user_id: str = "user") -> str:
        """Extract and format core user profile sheet from durable memory facts.

        Zero-poisoning & Persona-Isolation Design:
        - If user_id is a custom/isolated persona (not a default system persona)
          and no facts exist in its isolated database, returns an empty string ("")
          to guarantee a 100% clean slate with zero identity or memory leakage.
        - For default system personas, provides verified user identity and attribution policy.

        Args:
            user_id: Personality workspace namespace to pull facts from.

        Returns:
            str: Clean, grounded markdown profile block or empty string if clean-slate.
        """
        facts = await self.get_all_facts(user_id=user_id, limit=30)
        cached_profile = await self.get_user_profile(user_id)

        system_personas = {"user", "neo", "luna", "aria", "glados"}
        is_system = user_id.lower() in system_personas

        # Custom personas start with 100% clean slate unless explicit facts or profile exist
        if not is_system and not facts and not cached_profile.get("name") and not cached_profile.get("facts"):
            return ""

        user_name = cached_profile.get("name", "Ameya" if is_system else "")
        user_role = cached_profile.get("role", "AI Engineer & Robot Systems Creator" if is_system else "")

        if not user_name and not facts:
            return ""

        identity_lines = []
        if user_name:
            identity_lines.append(f"• User Name: {user_name}")
        if user_role:
            identity_lines.append(f"• User Role: {user_role}")

        attribution_lines = [
            f"• User{f' ({user_name})' if user_name else ''} is the human user. You are the AI robot assistant.",
            "• Never confuse user's attributes, personal life, or preferences with your own.",
        ]
        tech_lines = [
            "• Tech Stack: Python, PyTorch, C++, AsyncIO, WebSockets, Local LLMs, Web & Robotics",
        ] if is_system else []
        pref_lines: list[str] = []

        d_prefs = cached_profile.get("durable_preferences", {})
        if d_prefs:
            for v in d_prefs.values():
                pref_lines.append(f"• {v}")
                if len(pref_lines) >= 3:
                    break

        if not pref_lines and facts:
            pref_keywords = ("favorite food", "favorite color", "favorite genre", "favorite music", "prefer")
            for f in facts:
                txt = f.get("fact", f.get("memory", "")).strip()
                if not txt or len(txt) > 150:
                    continue
                txt_clean = " ".join(txt.split())
                txt_lower = txt_clean.lower()
                if any(k in txt_lower for k in pref_keywords):
                    if not any(bad in txt_lower for bad in ("crush", "telescope", "anubha", "allergy", "anand")):
                        clean_entry = f"• {txt_clean}"
                        if not any(txt_clean.lower() in p.lower() for p in pref_lines):
                            pref_lines.append(clean_entry)
                            if len(pref_lines) >= 3:
                                break

        profile_sections = [
            "=== USER CORE IDENTITY PROFILE ===",
        ]
        if identity_lines:
            profile_sections.append("Identity & Role:\n" + "\n".join(identity_lines))
        profile_sections.append("Attribution Policy:\n" + "\n".join(attribution_lines))
        if tech_lines:
            profile_sections.append("Technical Profile:\n" + "\n".join(tech_lines))
        if pref_lines:
            profile_sections.append("Established Preferences:\n" + "\n".join(pref_lines))

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
