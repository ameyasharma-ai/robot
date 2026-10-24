"""Whisper STT Server Process and Model Lifecycle Manager.

Architectural Rationale:
- Manages the local `whisper-server` process offloaded to the AMD Radeon 680M iGPU
  via Vulkan (`-dev 0 -fa`) on port 8081 by default.
- Isolates STT computation completely from the NVIDIA GeForce RTX 3050 (0MB VRAM used),
  preserving all 4GB GPU VRAM for the local LLM and multi-model diffusion engines.
- Provides dynamic model discovery, health checks, process lifecycle management,
  and hot-swapping between all 9 Whisper models with automatic persistence in
  `<project>/data/active_stt_model.txt` and `config/default.yaml`.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, TextIO

import httpx
import requests
import yaml

try:  # Optional dependency: used for safe process discovery/termination.
    import psutil
except ImportError:  # pragma: no cover - fallback path exercised without psutil
    psutil = None  # type: ignore[assignment]

from robot.config import atomic_write_text, get_project_root
from robot.daemon_log import open_daemon_log
from robot.telemetry import get_logger

logger = get_logger("stt.manager")

#: Root-relative paths so persistence and process supervision do not depend on
#: the caller's current working directory.
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = PROJECT_ROOT / "config" / "default.yaml"
LOG_DIR = PROJECT_ROOT / "logs"

#: Serializes read-modify-write cycles on the shared YAML state file between
#: the STT and TTS managers (TTS imports this same lock).
CONFIG_STATE_LOCK = threading.RLock()

#: How long a discovered model id is trusted before re-probing the system.
_ACTIVE_MODEL_CACHE_TTL_S = 2.0


class WhisperModelMetadata:
    """Metadata specification for a Whisper.cpp speech recognition model."""

    def __init__(
        self,
        id: str,
        name: str,
        parameters: str,
        size_mb: int,
        is_multilingual: bool,
        latency_rating: str,
        accuracy_rating: str,
        description: str,
    ) -> None:
        self.id = id
        self.name = name
        self.parameters = parameters
        self.size_mb = size_mb
        self.is_multilingual = is_multilingual
        self.latency_rating = latency_rating
        self.accuracy_rating = accuracy_rating
        self.description = description

    @property
    def filename(self) -> str:
        """Return the standard GGML binary filename.

        Returns:
            str: GGML model filename (e.g. 'ggml-base.en.bin').
        """
        return f"ggml-{self.id}.bin"

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of model metadata.
        """
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.filename,
            "parameters": self.parameters,
            "size_mb": self.size_mb,
            "is_multilingual": self.is_multilingual,
            "multilingual": self.is_multilingual,
            "latency_rating": self.latency_rating,
            "accuracy_rating": self.accuracy_rating,
            "description": self.description,
        }


class WhisperServerManager:
    """Manages the local whisper-server daemon process and model transitions."""

    DEFAULT_PORT = 8081
    HEALTH_URL = "http://127.0.0.1:8081/"
    INFERENCE_URL = "http://127.0.0.1:8081/inference"

    _active_model_id: str | None = None
    _process: subprocess.Popen[bytes] | None = None
    _log_file: TextIO | None = None

    #: Port the server launched by this manager is listening on. Health checks
    #: and stop() must target this port, not a hard-coded 8081.
    _active_port: int | None = None
    _tracked_port: int | None = None

    #: Per-event-loop lifecycle lock shared by start/stop/switch_model.
    _lifecycle_lock: asyncio.Lock | None = None
    _lifecycle_lock_loop: asyncio.AbstractEventLoop | None = None
    _lifecycle_owner: asyncio.Task[Any] | None = None
    _lifecycle_depth: int = 0
    _lifecycle_bookkeeping = threading.Lock()

    #: Short-lived active-model probe cache (pgrep/psutil/config reads are not free).
    _active_model_cache: str | None = None
    _active_model_cache_ts: float = 0.0
    _active_model_cache_lock = threading.Lock()

    BINARY_PATHS = [
        Path("/home/ameya/whisper.cpp/build_vk/bin/whisper-server"),  # Vulkan backend (Radeon 680M)
        Path("/home/ameya/whisper.cpp/build/bin/whisper-server"),     # CPU / Standard fallback
    ]

    MODELS_CATALOG: dict[str, WhisperModelMetadata] = {
        "tiny.en": WhisperModelMetadata(
            id="tiny.en",
            name="Whisper Tiny (English)",
            parameters="39M",
            size_mb=75,
            is_multilingual=False,
            latency_rating="~65–85 ms",
            accuracy_rating="Good",
            description="Ultra-low latency English model for instant voice command recognition.",
        ),
        "base.en": WhisperModelMetadata(
            id="base.en",
            name="Whisper Base (English)",
            parameters="74M",
            size_mb=142,
            is_multilingual=False,
            latency_rating="~90–120 ms",
            accuracy_rating="Great",
            description="Balanced low-latency English model with robust noise tolerance.",
        ),
        "base": WhisperModelMetadata(
            id="base",
            name="Whisper Base (Multilingual)",
            parameters="74M",
            size_mb=142,
            is_multilingual=True,
            latency_rating="~100–130 ms",
            accuracy_rating="Great",
            description="Multilingual base model supporting 99+ spoken languages.",
        ),
        "small.en": WhisperModelMetadata(
            id="small.en",
            name="Whisper Small (English)",
            parameters="244M",
            size_mb=466,
            is_multilingual=False,
            latency_rating="~160–200 ms",
            accuracy_rating="High",
            description="High accuracy English model with excellent vocabulary precision.",
        ),
        "small": WhisperModelMetadata(
            id="small",
            name="Whisper Small (Multilingual)",
            parameters="244M",
            size_mb=466,
            is_multilingual=True,
            latency_rating="~170–220 ms",
            accuracy_rating="High",
            description="High accuracy multilingual model for diverse accents and dialects.",
        ),
        "medium.en": WhisperModelMetadata(
            id="medium.en",
            name="Whisper Medium (English)",
            parameters="769M",
            size_mb=1500,
            is_multilingual=False,
            latency_rating="~240–300 ms",
            accuracy_rating="Superior",
            description="Professional-grade English transcription with complex terminology precision.",
        ),
        "medium": WhisperModelMetadata(
            id="medium",
            name="Whisper Medium (Multilingual)",
            parameters="769M",
            size_mb=1500,
            is_multilingual=True,
            latency_rating="~260–340 ms",
            accuracy_rating="Superior",
            description="Superior accuracy multilingual transcription across technical and regional speech.",
        ),
        "large-v3-turbo": WhisperModelMetadata(
            id="large-v3-turbo",
            name="Whisper Large v3 Turbo",
            parameters="809M",
            size_mb=1600,
            is_multilingual=True,
            latency_rating="~220–290 ms",
            accuracy_rating="Near-Perfect",
            description="Optimized Large v3 distillation for fast high-accuracy multi-lingual inference.",
        ),
        "large-v3": WhisperModelMetadata(
            id="large-v3",
            name="Whisper Large v3",
            parameters="1550M",
            size_mb=2900,
            is_multilingual=True,
            latency_rating="~400–550 ms",
            accuracy_rating="Maximum",
            description="State-of-the-art multilingual transcription with maximum benchmark precision.",
        ),
    }

    # ─── Lifecycle Lock ───────────────────────────────────────────────────────

    @classmethod
    def _get_lifecycle_lock(cls) -> asyncio.Lock:
        """Return the lifecycle lock bound to the current event loop.

        A single class-level ``asyncio.Lock`` would bind to the first loop that
        used it; tests and embedded runners may create several loops, so the
        lock is recreated when the loop identity changes.

        Returns:
            asyncio.Lock: Lock for the running loop.
        """
        loop = asyncio.get_running_loop()
        with cls._lifecycle_bookkeeping:
            if cls._lifecycle_lock is None or cls._lifecycle_lock_loop is not loop:
                cls._lifecycle_lock = asyncio.Lock()
                cls._lifecycle_lock_loop = loop
                cls._lifecycle_owner = None
                cls._lifecycle_depth = 0
            return cls._lifecycle_lock

    @classmethod
    @asynccontextmanager
    async def _lifecycle(cls) -> AsyncIterator[None]:
        """Re-entrant lifecycle guard shared by start, stop and switch_model.

        Re-entrancy is required because ``switch_model`` calls the public
        ``start``/``stop`` methods while already holding the guard (and those
        methods are part of the public, mockable surface).
        """
        task = asyncio.current_task()
        lock = cls._get_lifecycle_lock()
        if task is not None and cls._lifecycle_owner is task:
            cls._lifecycle_depth += 1
            try:
                yield
            finally:
                cls._lifecycle_depth -= 1
            return

        async with lock:
            cls._lifecycle_owner = task
            cls._lifecycle_depth = 1
            try:
                yield
            finally:
                cls._lifecycle_owner = None
                cls._lifecycle_depth = 0

    # ─── Path and Process Discovery ───────────────────────────────────────────

    @classmethod
    def resolve_binary(cls) -> Path | None:
        """Locate the best available executable whisper-server binary.

        Resolution order:
            1. ``WHISPER_SERVER_BINARY`` environment variable.
            2. ``<project>/bin/whisper-server``.
            3. ``shutil.which("whisper-server")``.
            4. Legacy build paths.

        Returns:
            Path | None: Executable binary path, or None if uninstalled.
        """
        candidates: list[Path] = []
        env_binary = os.environ.get("WHISPER_SERVER_BINARY", "").strip()
        if env_binary:
            candidates.append(Path(env_binary).expanduser())
        candidates.append(PROJECT_ROOT / "bin" / "whisper-server")
        which_binary = shutil.which("whisper-server")
        if which_binary:
            candidates.append(Path(which_binary))
        candidates.extend(cls.BINARY_PATHS)

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
            except OSError:
                continue
        return None

    @classmethod
    def resolve_model_path(cls, model_id_or_name: str, base_dir: Path | None = None) -> Path | None:
        """Resolve an absolute filepath to a ggml model binary on disk.

        Resolution order:
            1. ``WHISPER_MODELS_DIR`` (``os.pathsep``-separated directories).
            2. ``<project>/models/stt``.
            3. Legacy whisper.cpp / robot model directories.
            4. The explicit path itself.

        Args:
            model_id_or_name: Model identifier (e.g. 'base.en', 'ggml-base.en.bin').
            base_dir: Optional project directory override.

        Returns:
            Path | None: Path to model file if found, else None.
        """
        raw = str(model_id_or_name).strip()
        clean_id = raw
        if clean_id.startswith("ggml-"):
            clean_id = clean_id[5:]
        if clean_id.endswith(".bin"):
            clean_id = clean_id[:-4]

        direct = Path(raw).expanduser()
        if direct.is_file() and direct.stat().st_size > 1_000_000:
            return direct.resolve()

        candidate_filenames = [
            f"ggml-{clean_id}.bin",
            f"{clean_id}.bin",
            clean_id,
        ]

        candidate_dirs: list[Path] = []
        env_models = os.environ.get("WHISPER_MODELS_DIR", "").strip()
        if env_models:
            candidate_dirs.extend(
                Path(part).expanduser()
                for part in env_models.split(os.pathsep)
                if part.strip()
            )
        if base_dir is not None:
            candidate_dirs.append(Path(base_dir) / "models" / "stt")
        candidate_dirs.append(PROJECT_ROOT / "models" / "stt")
        candidate_dirs.extend(
            [
                Path("/home/ameya/whisper.cpp/models"),
                Path("/home/ameya/robot/models/stt"),
            ]
        )

        seen: set[str] = set()
        for c_dir in candidate_dirs:
            key = str(c_dir)
            if key in seen:
                continue
            seen.add(key)
            for fname in candidate_filenames:
                target = c_dir / fname
                try:
                    if target.is_file() and target.stat().st_size > 1_000_000:
                        return target.resolve()
                except OSError:
                    continue
        return None

    @classmethod
    def _canonical_model_id(cls, value: str | Path) -> str:
        """Convert a model id or path into a catalog id when possible.

        Args:
            value: Model identifier, filename, or full path.

        Returns:
            str: Catalog id, or the original value when it is not catalogued.
        """
        raw = str(value).strip()
        stem = Path(raw).stem if ("/" in raw or raw.endswith(".bin")) else raw
        if stem.startswith("ggml-"):
            stem = stem[5:]
        return stem if stem in cls.MODELS_CATALOG else raw

    @staticmethod
    def _cmdline_for_pid(pid: int) -> list[str]:
        """Read a process command line without invoking external tools.

        Args:
            pid: Process identifier.

        Returns:
            list[str]: Command line arguments, empty when unavailable.
        """
        if psutil is not None:
            try:
                return [str(arg) for arg in psutil.Process(pid).cmdline()]
            except Exception:
                pass
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            return [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        except OSError:
            return []

    @classmethod
    def _model_from_cmdline(cls, cmdline: list[str]) -> str | None:
        """Extract a catalog model id from a whisper-server command line.

        Args:
            cmdline: Process argv list.

        Returns:
            str | None: Catalog model id when one is present.
        """
        model_arg: str | None = None
        for index, arg in enumerate(cmdline):
            if arg in ("-m", "--model") and index + 1 < len(cmdline):
                model_arg = cmdline[index + 1]
                break
            if arg.startswith("--model="):
                model_arg = arg.split("=", 1)[1]
                break
        if not model_arg:
            return None
        candidate = cls._canonical_model_id(model_arg)
        return candidate if candidate in cls.MODELS_CATALOG else None

    @classmethod
    def _binary_path_matches(cls, candidate: str | None, binary: Path | None) -> bool:
        """Return True when a process executable/argv[0] resolves to the binary.

        Args:
            candidate: Process executable or argv[0].
            binary: Expected whisper-server binary.

        Returns:
            bool: Whether the paths resolve to the same executable.
        """
        if not candidate or binary is None:
            return False
        try:
            return Path(candidate).resolve() == binary.resolve()
        except OSError:
            return False

    @classmethod
    def _discover_whisper_pids(cls) -> list[int]:
        """Find whisper-server processes this manager can safely identify.

        A process is a candidate when its executable/argv[0] resolves to this
        manager's resolved binary, or when its cmdline contains the exact
        configured model path. Nothing else is ever signalled.

        Returns:
            list[int]: Matching PIDs (excluding this Python process).
        """
        if psutil is None:  # pragma: no cover - psutil is a hard dependency
            return []
        binary = cls.resolve_binary()
        try:
            model_path = cls.resolve_model_path(cls.get_active_model_id())
        except Exception:
            model_path = None
        model_str = str(model_path) if model_path is not None else None
        own_pid = os.getpid()
        matches: list[int] = []

        for proc in psutil.process_iter(["pid", "exe", "cmdline"]):
            try:
                pid = int(proc.info.get("pid") or 0)
                if pid <= 0 or pid == own_pid:
                    continue
                cmdline = [str(arg) for arg in (proc.info.get("cmdline") or []) if arg]
                exe = proc.info.get("exe")
                matched = cls._binary_path_matches(exe, binary) or cls._binary_path_matches(
                    cmdline[0] if cmdline else None, binary
                )
                if not matched and model_str:
                    matched = any(
                        arg == model_str or model_str in arg for arg in cmdline
                    )
                if matched:
                    matches.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                continue
        return matches

    @classmethod
    def _invalidate_active_model_cache(cls) -> None:
        """Discard the short-lived active-model probe cache."""
        with cls._active_model_cache_lock:
            cls._active_model_cache = None
            cls._active_model_cache_ts = 0.0

    @classmethod
    def get_active_model_id(cls) -> str:
        """Determine the ID of the model currently running or configured.

        Probes only the tracked PID when available (never arbitrary system
        processes via ``pgrep``), falls back to psutil discovery, and caches
        the result for ~2 seconds to keep API polling cheap.

        Returns:
            str: Model identifier string (e.g. 'base.en').
        """
        with cls._active_model_cache_lock:
            if (
                cls._active_model_cache is not None
                and (time.monotonic() - cls._active_model_cache_ts) < _ACTIVE_MODEL_CACHE_TTL_S
            ):
                return cls._active_model_cache

        resolved = cls._detect_active_model_id()
        with cls._active_model_cache_lock:
            cls._active_model_cache = resolved
            cls._active_model_cache_ts = time.monotonic()
        return resolved

    @classmethod
    def _detect_active_model_id(cls) -> str:
        """Perform the uncached active-model detection.

        Returns:
            str: Model identifier string.
        """
        if cls._active_model_id and cls._active_model_id in cls.MODELS_CATALOG:
            return cls._active_model_id

        # 1. Inspect only the process this manager spawned, when still alive.
        proc = cls._process
        if proc is not None and proc.poll() is None:
            detected = cls._model_from_cmdline(cls._cmdline_for_pid(proc.pid))
            if detected:
                return detected

        # 2. psutil discovery of whisper-server processes (binary or exact model path).
        for pid in cls._discover_whisper_pids():
            detected = cls._model_from_cmdline(cls._cmdline_for_pid(pid))
            if detected:
                return detected

        # 3. Persisted state file (project-root data directory).
        stt_file = DATA_DIR / "active_stt_model.txt"
        if stt_file.exists():
            try:
                saved = cls._canonical_model_id(stt_file.read_text(encoding="utf-8").strip())
                if saved in cls.MODELS_CATALOG:
                    return saved
            except Exception as e:
                logger.warning(
                    "stt_active_model_state_read_failed", error=str(e), path=str(stt_file)
                )

        # 4. config/default.yaml
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle) or {}
                model = cfg.get("stt", {}).get("model", "")
                canonical = cls._canonical_model_id(model)
                if canonical in cls.MODELS_CATALOG:
                    return canonical
            except Exception as e:
                logger.warning(
                    "stt_active_model_config_read_failed", error=str(e), path=str(CONFIG_FILE)
                )

        return "base.en"

    @classmethod
    def list_models(cls, base_dir: Path | None = None) -> list[dict[str, Any]]:
        """List all catalog models with local disk download and active status.

        Args:
            base_dir: Optional base project directory.

        Returns:
            list[dict[str, Any]]: List of serialized model details.
        """
        active_id = cls.get_active_model_id()
        results: list[dict[str, Any]] = []

        for m_id, meta in cls.MODELS_CATALOG.items():
            path = cls.resolve_model_path(m_id, base_dir)
            is_downloaded = path is not None
            data = meta.to_dict()
            data["is_downloaded"] = is_downloaded
            data["is_active"] = (m_id == active_id)
            data["file_path"] = str(path) if path else None
            data["device"] = "AMD Radeon 680M (Vulkan dev 0)"
            results.append(data)

        return results

    # ─── HTTP Health / URLs ───────────────────────────────────────────────────

    @classmethod
    def _health_url(cls, port: int | None = None) -> str:
        """Return the health URL for a port (active port by default).

        Args:
            port: Explicit port override.

        Returns:
            str: Fully-qualified health URL.
        """
        target = int(port) if port is not None else int(cls._active_port or cls.DEFAULT_PORT)
        return f"http://127.0.0.1:{target}/"

    @classmethod
    def _inference_url(cls, port: int | None = None) -> str:
        """Return the inference URL for a port (active port by default).

        Args:
            port: Explicit port override.

        Returns:
            str: Fully-qualified inference URL.
        """
        target = int(port) if port is not None else int(cls._active_port or cls.DEFAULT_PORT)
        return f"http://127.0.0.1:{target}/inference"

    @classmethod
    def is_running(cls, port: int | None = None) -> bool:
        """Check synchronously if whisper-server responds to its health check.

        Args:
            port: Optional explicit port (defaults to the tracked active port).

        Returns:
            bool: True if server is healthy and accepting requests.
        """
        try:
            resp = requests.get(cls._health_url(port), timeout=0.8)
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    async def is_running_async(cls, port: int | None = None) -> bool:
        """Check asynchronously if whisper-server is reachable.

        Args:
            port: Optional explicit port (defaults to the tracked active port).

        Returns:
            bool: True if server is reachable.
        """
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                resp = await client.get(cls._health_url(port))
                return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def _port_has_listener(cls, port: int) -> bool:
        """Probe whether something is already listening on a local TCP port.

        Args:
            port: TCP port to probe.

        Returns:
            bool: True when a connection can be established.
        """
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.25):
                return True
        except OSError:
            return False

    # ─── Process Lifecycle ────────────────────────────────────────────────────

    @classmethod
    def _close_log_file(cls) -> None:
        """Close the whisper-server stdout/stderr log handle if open."""
        if cls._log_file is not None:
            try:
                cls._log_file.close()
            except Exception as e:
                logger.warning("whisper_server_log_close_failed", error=str(e))
            cls._log_file = None

    @classmethod
    def _terminate_tracked(cls, pid: int) -> None:
        """Send SIGTERM to the tracked whisper-server PID (preferring psutil).

        Args:
            pid: PID of the tracked whisper-server process.
        """
        if psutil is not None:
            try:
                psutil.Process(pid).terminate()
            except Exception:
                pass  # Process already exited
            return
        try:
            os.kill(pid, 15)
        except Exception:
            pass

    @classmethod
    async def _release_tracked_process(cls) -> bool:
        """Terminate the tracked whisper-server (SIGTERM, escalate to SIGKILL) and close fds.

        Returns:
            bool: True if a tracked process handle existed and was released.
        """
        proc = cls._process
        if proc is None:
            return False
        if proc.poll() is None:
            cls._terminate_tracked(proc.pid)
            for _ in range(30):  # Up to ~3s grace period for clean shutdown
                if proc.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            if proc.poll() is None:
                logger.warning("whisper_server_terminate_grace_expired", pid=proc.pid)
                try:
                    proc.kill()
                except Exception as e:
                    logger.error("whisper_server_kill_failed", error=str(e))
        cls._process = None
        cls._tracked_port = None
        cls._close_log_file()
        return True

    @classmethod
    async def _terminate_discovered(cls) -> bool:
        """SIGTERM only whisper-server processes this manager can safely identify.

        When no tracked handle exists (e.g. the manager process restarted),
        psutil is used to find processes whose executable resolves to this
        manager's binary or whose cmdline contains the exact configured model
        path. ``pkill -f``/``killall`` are never used because they kill
        unrelated servers sharing the process name.

        Returns:
            bool: True when at least one candidate was identified and signalled.
        """
        pids = await asyncio.to_thread(cls._discover_whisper_pids)
        if not pids:
            logger.warning("whisper_server_stop_no_safe_target")
            return False

        logger.info("whisper_server_terminating_discovered", pids=pids)
        processes = []
        for pid in pids:
            if psutil is None:  # pragma: no cover
                continue
            try:
                proc = psutil.Process(pid)
                proc.terminate()  # SIGTERM only; never pkill/killall
                processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        for _ in range(30):
            if not any(proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE for proc in processes):
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning(
                "whisper_server_discovered_terminate_timeout",
                pids=[proc.pid for proc in processes if proc.is_running()],
            )
        return bool(pids)

    @classmethod
    async def stop(cls, port: int | None = None) -> bool:
        """Terminate the running whisper-server daemon safely.

        Args:
            port: Optional explicit port whose health should be confirmed down.

        Returns:
            bool: True if the health endpoint confirmed the server is down,
                False when nothing could be safely identified/stopped.
        """
        async with cls._lifecycle():
            return await cls._stop_locked(port=port)

    @classmethod
    async def _stop_locked(cls, port: int | None = None, log_missing: bool = True) -> bool:
        """Stop implementation; caller must hold the lifecycle guard.

        Args:
            port: Optional explicit health-check port.
            log_missing: Emit a warning when nothing safe can be identified.

        Returns:
            bool: True when the server is confirmed down.
        """
        logger.info("stopping_whisper_server", port=port)
        health_port = port if port is not None else (cls._active_port or cls._tracked_port)
        try:
            had_tracked = await cls._release_tracked_process()
            if not had_tracked:
                had_tracked = await cls._terminate_discovered()
            if not had_tracked and log_missing:
                logger.warning(
                    "whisper_server_stop_nothing_identified",
                    port=health_port,
                )
                return False
            if health_port is None:
                logger.info("whisper_server_stopped_cleanly")
                return True

            for _ in range(15):
                if not await cls.is_running_async(health_port):
                    logger.info("whisper_server_stopped_cleanly", port=health_port)
                    return True
                await asyncio.sleep(0.1)
            logger.error("whisper_server_stop_not_confirmed", port=health_port)
            return False
        except Exception as e:
            logger.error("whisper_server_stop_error", error=str(e))
            return False

    @classmethod
    async def start(
        cls,
        model_id_or_path: str | None = None,
        port: int = DEFAULT_PORT,
    ) -> bool:
        """Launch whisper-server with Vulkan GPU offload on the AMD Radeon 680M.

        Args:
            model_id_or_path: Target model identifier or explicit file path.
            port: Port for the HTTP server (default: 8081).

        Returns:
            bool: True if server launched and became healthy, False otherwise.
        """
        async with cls._lifecycle():
            return await cls._start_locked(model_id_or_path, port=port)

    @classmethod
    async def _start_locked(
        cls,
        model_id_or_path: str | None = None,
        port: int = DEFAULT_PORT,
    ) -> bool:
        """Start implementation; caller must hold the lifecycle guard.

        Args:
            model_id_or_path: Target model identifier or explicit file path.
            port: Server port.

        Returns:
            bool: True when healthy.
        """
        bin_path = cls.resolve_binary()
        if not bin_path:
            logger.error("whisper_server_binary_missing")
            return False

        target_model = model_id_or_path or cls.get_active_model_id()
        model_file = cls.resolve_model_path(target_model)
        if not model_file:
            logger.error("whisper_server_model_file_missing", target=target_model)
            return False

        model_id = cls._canonical_model_id(target_model)

        # Kill any zombie/stale instance this manager owns first, then make sure
        # the requested port is not held by an unrelated listener.
        await cls._stop_locked(log_missing=False)
        await asyncio.sleep(0.1)

        if cls._port_has_listener(port):
            logger.error(
                "whisper_server_port_in_use_by_foreign_listener",
                port=port,
                msg="Refusing to start; the port is bound by a process this manager does not own.",
            )
            return False

        cmd = [
            str(bin_path),
            "-m", str(model_file),
            "-dev", "0",
            "-fa",
            "-t", "8",
            "-bo", "1",
            "-bs", "1",
            "-nf",
            "-nt",
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            log_file = open_daemon_log(LOG_DIR / "whisper-server.log")
        except OSError as e:
            logger.error("whisper_server_log_open_failed", error=str(e))
            return False

        logger.info(
            "starting_whisper_server",
            model=str(model_file),
            binary=str(bin_path),
            port=port,
        )
        try:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)
        except Exception as e:
            logger.error("whisper_server_spawn_failed", error=str(e))
            try:
                log_file.close()
            except Exception:
                pass
            return False

        # Track handle + log fd so stop()/start-timeout reaps exactly this PID.
        cls._process = proc
        cls._log_file = log_file
        cls._tracked_port = int(port)

        for i in range(30):
            await asyncio.sleep(0.3)
            if proc.poll() is not None:
                logger.error("whisper_server_exited_early", returncode=proc.returncode)
                await cls._stop_locked(port=port, log_missing=False)
                return False
            if await cls.is_running_async(port):
                cls._active_port = int(port)
                cls._active_model_id = model_id
                cls._invalidate_active_model_cache()
                logger.info(
                    "whisper_server_ready",
                    attempts=i + 1,
                    pid=proc.pid,
                    model=str(model_file),
                    port=port,
                )
                cls._persist_model_selection(model_id, model_file)
                return True

        logger.error("whisper_server_launch_timed_out", pid=proc.pid, port=port)
        # Start failed: terminate tracked PID and close fds — no orphan, no fd leak.
        await cls._stop_locked(port=port, log_missing=False)
        return False

    @classmethod
    async def switch_model(
        cls,
        model_id_or_path: str,
        engine: Any | None = None,
        port: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Hot-swap the active Whisper STT model with rollback on failure.

        The new server is only considered active after it passes its health
        check; ``engine.stt`` is not rebound while the switch is in flight. If
        the new model fails to start, the previous model is restarted and the
        caller receives a clear error instead of a dead microphone.

        Args:
            model_id_or_path: Target model identifier (e.g. 'small.en') or path.
            engine: Optional active RobotEngine to re-bind STT and emit events.
            port: Target server port (defaults to the currently active port).
            force: If True, restart whisper-server even if target model is already running.

        Returns:
            dict[str, Any]: Status summary of the model switch operation.
        """
        async with cls._lifecycle():
            return await cls._switch_model_locked(model_id_or_path, engine=engine, port=port, force=force)

    @classmethod
    async def _switch_model_locked(
        cls,
        model_id_or_path: str,
        engine: Any | None = None,
        port: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Switch implementation; caller must hold the lifecycle guard."""
        logger.info("switching_whisper_model", target=model_id_or_path)
        resolved_path = cls.resolve_model_path(model_id_or_path)
        if not resolved_path:
            return {
                "status": "error",
                "message": f"Whisper model '{model_id_or_path}' is not downloaded on this machine.",
            }

        clean_id = cls._canonical_model_id(model_id_or_path)
        meta = cls.MODELS_CATALOG.get(clean_id)
        target_port = int(port) if port is not None else int(
            cls._active_port or cls._tracked_port or cls.DEFAULT_PORT
        )

        previous_id = cls.get_active_model_id()
        previous_path = cls.resolve_model_path(previous_id)
        previous_port = int(cls._active_port or cls._tracked_port or target_port)

        # Avoid redundant process restart if target model is already healthy and active
        if not force and clean_id == previous_id and await cls.is_running_async(target_port):
            logger.info("whisper_model_already_running", model=clean_id, port=target_port)
            return {
                "status": "ok",
                "model": clean_id,
                "name": meta.name if meta else clean_id,
                "parameters": meta.parameters if meta else "unknown",
                "size_mb": meta.size_mb if meta else 0,
                "latency_rating": meta.latency_rating if meta else "unknown",
                "device": "AMD Radeon 680M (Vulkan dev 0)",
                "model_path": str(resolved_path),
                "port": target_port,
                "already_active": True,
            }

        # Same-port hot swap: the old process must release the port before the
        # new one can bind. Keep the old client usable until the new model is
        # proven healthy (engine.stt is only rebound on success).
        await cls.stop()
        await asyncio.sleep(0.3)

        started = await cls.start(str(resolved_path), port=target_port)
        if not started:
            rolled_back = False
            if previous_path is not None:
                rolled_back = await cls.start(str(previous_path), port=previous_port)
                if rolled_back:
                    cls._active_model_id = previous_id
                    cls._invalidate_active_model_cache()
                    logger.warning(
                        "whisper_server_switch_rolled_back",
                        failed_model=clean_id,
                        restored_model=previous_id,
                        port=previous_port,
                    )
                else:
                    logger.error(
                        "whisper_server_switch_rollback_failed",
                        failed_model=clean_id,
                        previous_model=previous_id,
                    )
            message = (
                f"Failed to start whisper-server with model '{clean_id}'. "
                + (
                    f"Rolled back to '{previous_id}'."
                    if rolled_back
                    else "Previous model could not be restored. Check logs/whisper-server.log."
                )
            )
            return {
                "status": "error",
                "message": message,
                "model": clean_id,
                "rolled_back": rolled_back,
                "restored_model": previous_id if rolled_back else None,
            }

        cls._active_model_id = clean_id
        cls._invalidate_active_model_cache()

        # Rebind only after the server is healthy, then close the old HTTP
        # session so its sockets do not leak across model switches.
        if engine is not None and hasattr(engine, "stt"):
            old_stt = getattr(engine, "stt", None)
            try:
                from robot.stt import WhisperServerSTT

                new_stt = WhisperServerSTT(url=cls._inference_url(target_port))
                engine.stt = new_stt
            except Exception as e:
                logger.error("stt_engine_rebind_failed", error=str(e))
                rolled_back = False
                if previous_path is not None:
                    rolled_back = await cls.start(str(previous_path), port=previous_port)
                if rolled_back:
                    cls._active_model_id = previous_id
                    cls._invalidate_active_model_cache()
                return {
                    "status": "error",
                    "message": (
                        f"whisper-server started with '{clean_id}' but rebinding engine.stt failed. "
                        + (
                            f"Rolled back to '{previous_id}'."
                            if rolled_back
                            else "Previous model could not be restored."
                        )
                    ),
                    "model": clean_id,
                    "rolled_back": rolled_back,
                }
            else:
                if old_stt is not None and old_stt is not new_stt:
                    close_old = getattr(old_stt, "close", None)
                    if callable(close_old):
                        try:
                            close_old()
                        except Exception as e:
                            logger.debug("stt_old_client_close_failed", error=str(e))

        if engine is not None and hasattr(engine, "events") and hasattr(engine.events, "emit"):
            try:
                emit_coro = engine.events.emit(
                    "stt.model_changed",
                    model=clean_id,
                    model_name=meta.name if meta else clean_id,
                    model_path=str(resolved_path),
                    device="AMD Radeon 680M (Vulkan dev 0)",
                )
                if asyncio.iscoroutine(emit_coro):
                    await emit_coro
            except Exception as e:
                logger.warning("stt_model_changed_emit_error", error=str(e))

        return {
            "status": "ok",
            "model": clean_id,
            "name": meta.name if meta else clean_id,
            "parameters": meta.parameters if meta else "unknown",
            "size_mb": meta.size_mb if meta else 0,
            "latency_rating": meta.latency_rating if meta else "unknown",
            "device": "AMD Radeon 680M (Vulkan dev 0)",
            "model_path": str(resolved_path),
            "port": target_port,
        }

    @classmethod
    def _persist_model_selection(cls, model_id: str, model_path: Path) -> None:
        """Atomically persist the active model choice to the runtime state file.

        ``config/default.yaml`` is never rewritten: a load/dump cycle would drop
        its comments, and ``stt.model`` there is only a first-run default that
        ``_detect_active_model_id`` consults after this file. The read-modify-
        write cycle is serialized with the TTS manager through
        ``CONFIG_STATE_LOCK``.

        Args:
            model_id: Target model identifier.
            model_path: Resolved path to the binary.
        """
        clean_id = cls._canonical_model_id(model_id)

        try:
            with CONFIG_STATE_LOCK:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                atomic_write_text(
                    DATA_DIR / "active_stt_model.txt",
                    clean_id + "\n",
                    mode=0o600,
                )

        except Exception as e:
            logger.warning(
                "failed_persisting_stt_model",
                error=str(e),
                model=clean_id,
                path=str(model_path),
            )
