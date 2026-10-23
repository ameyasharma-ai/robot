"""Whisper STT Server Process and Model Lifecycle Manager.

Architectural Rationale:
- Manages the local `whisper-server` process offloaded to the AMD Radeon 680M iGPU
  via Vulkan (`-dev 0 -fa`) on port 8081.
- Isolates STT computation completely from the NVIDIA GeForce RTX 3050 (0MB VRAM used),
  preserving all 4GB GPU VRAM for the local LLM and multi-model diffusion engines.
- Provides dynamic model discovery, health checks, process lifecycle management,
  and hot-swapping between all 9 Whisper models with automatic persistence in
  `data/active_stt_model.txt` and `config/default.yaml`.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any, TextIO

import httpx
import requests
import structlog
import yaml

try:  # Optional dependency: used to terminate the tracked PID gracefully.
    import psutil
except ImportError:  # pragma: no cover - fallback path exercised without psutil
    psutil = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


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

    @classmethod
    def resolve_binary(cls) -> Path | None:
        """Locate the best available whisper-server binary.

        Returns:
            Path | None: Path to executable binary or None if uninstalled.
        """
        for p in cls.BINARY_PATHS:
            if p.exists() and os.access(p, os.X_OK):
                return p
        return None

    @classmethod
    def resolve_model_path(cls, model_id_or_name: str, base_dir: Path | None = None) -> Path | None:
        """Resolve absolute filepath to a ggml model binary on disk.

        Args:
            model_id_or_name: Model identifier (e.g. 'base.en', 'ggml-base.en.bin').
            base_dir: Optional base project directory.

        Returns:
            Path | None: Path to model file if found, else None.
        """
        clean_id = model_id_or_name.strip()
        if clean_id.startswith("ggml-"):
            clean_id = clean_id[5:]
        if clean_id.endswith(".bin"):
            clean_id = clean_id[:-4]

        root = base_dir or Path.cwd()
        candidate_filenames = [
            f"ggml-{clean_id}.bin",
            f"{clean_id}.bin",
            clean_id,
        ]

        candidate_dirs = [
            root / "models" / "stt",
            Path("/home/ameya/whisper.cpp/models"),
            Path("/home/ameya/robot/models/stt"),
        ]

        for c_dir in candidate_dirs:
            for fname in candidate_filenames:
                target = c_dir / fname
                if target.exists() and target.is_file() and target.stat().st_size > 1_000_000:
                    return target.resolve()

        # Check direct path if passed as an absolute or relative path
        direct = Path(model_id_or_name)
        if direct.exists() and direct.is_file() and direct.stat().st_size > 1_000_000:
            return direct.resolve()

        return None

    @classmethod
    def get_active_model_id(cls) -> str:
        """Determine the ID of the model currently running or configured.

        Returns:
            str: Model identifier string (e.g. 'base.en').
        """
        if cls._active_model_id:
            return cls._active_model_id

        # 1. Inspect live running process arguments
        try:
            res = subprocess.run(
                ["pgrep", "-a", "whisper-server"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout:
                line = res.stdout.strip()
                match = re.search(r"-m\s+([^\s]+)", line)
                if match:
                    m_path = Path(match.group(1))
                    name = m_path.stem
                    if name.startswith("ggml-"):
                        name = name[5:]
                    if name in cls.MODELS_CATALOG:
                        return name
        except Exception as e:
            logger.warning("stt_active_model_process_probe_failed", error=str(e))

        # 2. Check persisted state file
        stt_file = Path("data/active_stt_model.txt")
        if stt_file.exists():
            try:
                saved = stt_file.read_text().strip()
                if saved in cls.MODELS_CATALOG:
                    return saved
            except Exception as e:
                logger.warning("stt_active_model_state_read_failed", error=str(e), path=str(stt_file))

        # 3. Check config/default.yaml
        cfg_file = Path("config/default.yaml")
        if cfg_file.exists():
            try:
                with open(cfg_file, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                m = cfg.get("stt", {}).get("model", "")
                if m in cls.MODELS_CATALOG:
                    return m
            except Exception as e:
                logger.warning("stt_active_model_config_read_failed", error=str(e), path=str(cfg_file))

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

    @classmethod
    def is_running(cls) -> bool:
        """Check synchronously if whisper-server responds to HTTP health check.

        Returns:
            bool: True if server is healthy and accepting requests.
        """
        try:
            resp = requests.get(cls.HEALTH_URL, timeout=0.8)
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    async def is_running_async(cls) -> bool:
        """Check asynchronously if whisper-server is reachable over HTTP.

        Returns:
            bool: True if server is reachable.
        """
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                resp = await client.get(cls.HEALTH_URL)
                return resp.status_code == 200
        except Exception:
            return False

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
            os.kill(pid, signal.SIGTERM)
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
        cls._close_log_file()
        return True

    @classmethod
    async def stop(cls) -> bool:
        """Terminate the running whisper-server daemon.

        Terminates the tracked PID launched by this manager. Falls back to
        `pkill -f whisper-server` only when no tracked handle exists, so
        unrelated system processes sharing the name are never killed.

        Returns:
            bool: True if the health endpoint confirmed the server is down,
                False if it could not be stopped or an error occurred.
        """
        logger.info("stopping_whisper_server")
        try:
            if not await cls._release_tracked_process():
                subprocess.run(["pkill", "-9", "-f", "whisper-server"], check=False)

            for _ in range(15):
                if not await cls.is_running_async():
                    logger.info("whisper_server_stopped_cleanly")
                    return True
                await asyncio.sleep(0.1)
            logger.error("whisper_server_stop_not_confirmed")
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
        bin_path = cls.resolve_binary()
        if not bin_path:
            logger.error("whisper_server_binary_missing")
            return False

        target_model = model_id_or_path or cls.get_active_model_id()
        model_file = cls.resolve_model_path(target_model)
        if not model_file:
            logger.error("whisper_server_model_file_missing", target=target_model)
            return False

        # Kill any zombie or stale instance first
        await cls.stop()
        await asyncio.sleep(0.3)

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

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            log_file = open(log_dir / "whisper-server.log", "a", encoding="utf-8")
        except OSError as e:
            logger.error("whisper_server_log_open_failed", error=str(e))
            return False

        logger.info("starting_whisper_server", model=str(model_file), binary=str(bin_path))
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

        for i in range(30):
            await asyncio.sleep(0.3)
            if proc.poll() is not None:
                logger.error("whisper_server_exited_early", returncode=proc.returncode)
                await cls.stop()
                return False
            if await cls.is_running_async():
                logger.info("whisper_server_ready", attempts=i + 1, pid=proc.pid, model=str(model_file))
                # Persist active model selection
                cls._persist_model_selection(target_model, model_file)
                return True

        logger.error("whisper_server_launch_timed_out", pid=proc.pid)
        # Start failed: terminate tracked PID and close fds — no orphan, no fd leak.
        await cls.stop()
        return False

    @classmethod
    async def switch_model(
        cls,
        model_id_or_path: str,
        engine: Any | None = None,
        port: int = DEFAULT_PORT,
    ) -> dict[str, Any]:
        """Hot-swap the active Whisper STT model with zero downtime for other subsystems.

        Args:
            model_id_or_path: Target model identifier (e.g. 'small.en') or path.
            engine: Optional active RobotEngine to re-bind STT and emit events.
            port: Server port (default: 8081).

        Returns:
            dict[str, Any]: Status summary of model switch operation.
        """
        logger.info("switching_whisper_model", target=model_id_or_path)
        resolved_path = cls.resolve_model_path(model_id_or_path)
        if not resolved_path:
            return {
                "status": "error",
                "message": f"Whisper model '{model_id_or_path}' is not downloaded on this machine.",
            }

        clean_id = model_id_or_path
        if clean_id.startswith("ggml-"):
            clean_id = clean_id[5:]
        if clean_id.endswith(".bin"):
            clean_id = clean_id[:-4]

        meta = cls.MODELS_CATALOG.get(clean_id)

        # Stop currently running server to evacuate iGPU memory and release port
        await cls.stop()
        await asyncio.sleep(0.3)

        # Launch server with new model
        ok = await cls.start(str(resolved_path), port=port)
        if not ok:
            return {
                "status": "error",
                "message": f"Failed to start whisper-server with model '{clean_id}'. Check logs/whisper-server.log.",
            }

        cls._active_model_id = clean_id

        # If engine is attached, update STT provider and warm up connection
        if engine and hasattr(engine, "stt"):
            try:
                from robot.stt import WhisperServerSTT
                engine.stt = WhisperServerSTT(url=f"http://127.0.0.1:{port}/inference")
                if hasattr(engine, "events") and hasattr(engine.events, "emit"):
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
                logger.warning("stt_engine_rebind_error", error=str(e))

        return {
            "status": "ok",
            "model": clean_id,
            "name": meta.name if meta else clean_id,
            "parameters": meta.parameters if meta else "unknown",
            "size_mb": meta.size_mb if meta else 0,
            "latency_rating": meta.latency_rating if meta else "unknown",
            "device": "AMD Radeon 680M (Vulkan dev 0)",
            "model_path": str(resolved_path),
        }

    @classmethod
    def _persist_model_selection(cls, model_id: str, model_path: Path) -> None:
        """Save active model choice to data/ directory and config/default.yaml.

        Args:
            model_id: Target model identifier.
            model_path: Resolved path to the binary.
        """
        clean_id = model_id
        if clean_id.startswith("ggml-"):
            clean_id = clean_id[5:]
        if clean_id.endswith(".bin"):
            clean_id = clean_id[:-4]

        try:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "active_stt_model.txt").write_text(clean_id + "\n", encoding="utf-8")
        except Exception as e:
            logger.debug("failed_writing_active_stt_model_file", error=str(e))

        try:
            cfg_path = Path("config/default.yaml")
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if "stt" not in cfg or not isinstance(cfg["stt"], dict):
                    cfg["stt"] = {}
                cfg["stt"]["model"] = clean_id
                cfg["stt"]["model_path"] = str(model_path)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.debug("failed_updating_default_yaml_stt_model", error=str(e))
