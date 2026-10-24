"""HuggingFace Hub LLM search, inspection, and background streaming downloader.

Provides asynchronous searching of GGUF repositories on HuggingFace Hub, quantization
file tree inspection, resumable chunked downloads with real-time transfer metrics
(speed, ETA, percentage), binary GGUF verification, and dynamic local model registration.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from robot.llm.model_registry import ModelCategory, ModelMetadata, ModelRegistry
from robot.telemetry import get_logger

logger = get_logger("hf_downloader")

# Disk write batching for download workers: accumulate chunks in memory and flush
# to the .part file from a worker thread once this many bytes are buffered, so the
# event loop is never blocked by synchronous 512 KB f.write() calls.
DOWNLOAD_FLUSH_THRESHOLD_BYTES = 4 * 1024 * 1024

# Maximum number of finished download tasks retained in history.
MAX_RETAINED_DOWNLOAD_TASKS = 50

# Maximum number of concurrent file transfers across all download tasks.
DEFAULT_MAX_CONCURRENT_DOWNLOADS = 2

# Free-space headroom required before writing a file of known size.
DISK_SPACE_HEADROOM_BYTES = 64 * 1024 * 1024

#: GGUF magic header shared by model weights and multimodal projectors.
_GGUF_MAGIC = b"GGUF"

#: Repository identifiers must be two simple path segments; anything else could
#: smuggle traversal or a different URL path into the download request.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class DownloadError(RuntimeError):
    """Download failure carrying a message that is safe to expose through the API.

    Internal details (absolute paths, signed URLs, HTTP bodies) are logged only and must
    never be interpolated into ``task.error``; callers surface ``str(exc)`` verbatim.
    """


def _project_root() -> Path:
    """Return the repository root, independent of the current working directory.

    Returns:
        Path: Absolute project root.
    """
    try:
        from robot.config import get_project_root

        return get_project_root()
    except Exception:  # pragma: no cover - defensive fallback
        return Path(__file__).resolve().parents[3]


class _HTTPStatusError(DownloadError):
    """Remote endpoint returned a non-success status code."""

    def __init__(self, status_code: int) -> None:
        """Initialize the error.

        Args:
            status_code: HTTP status returned by the remote endpoint.
        """
        self.status_code = status_code
        super().__init__(_status_message(status_code))


def _status_message(status_code: int) -> str:
    """Map an HTTP status to a safe, specific user-facing message.

    Args:
        status_code: HTTP status code.

    Returns:
        str: Message without URL or filesystem details.
    """
    if status_code == 404:
        return "The requested model file was not found on HuggingFace (HTTP 404)."
    if status_code in (401, 403):
        return "HuggingFace denied access to this model file. Check HF_TOKEN permissions."
    if status_code == 429:
        return "HuggingFace rate-limited the download (HTTP 429). Try again shortly."
    if 500 <= status_code < 600:
        return f"HuggingFace is temporarily unavailable (HTTP {status_code}). Try again later."
    return f"HuggingFace rejected the download request (HTTP {status_code})."


def _validate_repo_id(repo_id: str) -> str:
    """Validate a HuggingFace repository identifier.

    Args:
        repo_id: Raw repository identifier such as ``author/repo-GGUF``.

    Returns:
        str: The validated identifier.

    Raises:
        DownloadError: If the identifier is malformed.
    """
    if not isinstance(repo_id, str) or "\x00" in repo_id or not _REPO_ID_RE.match(repo_id.strip()):
        raise DownloadError("Invalid HuggingFace repository identifier.")
    return repo_id.strip()


def _validate_gguf_filename(filename: str, kind: str = "model") -> str:
    """Validate a user-supplied GGUF filename before it touches the filesystem.

    Args:
        filename: Candidate filename.
        kind: Human readable file kind ('model' or 'mmproj').

    Returns:
        str: The validated filename.

    Raises:
        DownloadError: If the name is absolute, contains directory components, NUL
            bytes, or an unexpected extension.
    """
    if not isinstance(filename, str) or not filename:
        raise DownloadError(f"Missing {kind} filename.")
    if "\x00" in filename:
        raise DownloadError(f"Invalid {kind} filename.")
    if Path(filename).is_absolute() or Path(filename).name != filename:
        raise DownloadError(f"Invalid {kind} filename: directory components are not allowed.")
    if filename not in (".", "..") and not filename.lower().endswith(".gguf"):
        raise DownloadError(f"Invalid {kind} filename: expected a .gguf file.")
    if filename.startswith("."):
        raise DownloadError(f"Invalid {kind} filename: hidden files are not allowed.")
    return filename


def _resolve_inside(base_dir: Path, filename: str, kind: str = "model") -> Path:
    """Resolve ``base_dir / filename`` and guarantee it stays inside ``base_dir``.

    Args:
        base_dir: Directory that owns the file (models/llm or models/vision).
        filename: Validated plain filename.
        kind: Human readable file kind.

    Returns:
        Path: Resolved destination path.

    Raises:
        DownloadError: If the resolved path escapes the model directory or is a symlink.
    """
    base = base_dir.resolve()
    candidate = base / filename
    if candidate.is_symlink():
        raise DownloadError(f"Refusing to write the {kind} through a symbolic link.")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError:
        raise DownloadError(f"Refusing to write the {kind} outside the model directory.") from None
    if resolved.parent != base:
        raise DownloadError(f"Invalid {kind} destination.")
    return resolved


def _part_path(dest_path: Path) -> Path:
    """Return the sidecar partial-download path for a destination.

    Args:
        dest_path: Final destination path.

    Returns:
        Path: ``<destination>.part`` path.
    """
    return dest_path.with_name(dest_path.name + ".part")


def _meta_path(dest_path: Path) -> Path:
    """Return the resume metadata sidecar path for a destination.

    Args:
        dest_path: Final destination path.

    Returns:
        Path: ``<destination>.part.meta.json`` path.
    """
    return dest_path.with_name(dest_path.name + ".part.meta.json")


def _backup_path(dest_path: Path) -> Path:
    """Return the temporary backup path used while swapping in a verified model.

    Args:
        dest_path: Final destination path.

    Returns:
        Path: ``<destination>.bak`` path.
    """
    return dest_path.with_name(dest_path.name + ".bak")


def _fsync_path(path: Path) -> None:
    """Flush a file's contents to stable storage.

    Args:
        path: File to fsync.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Flush directory entries to stable storage after a rename.

    Args:
        path: Directory to fsync.
    """
    try:
        fd = os.open(path, os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    """Read a resume metadata sidecar.

    Args:
        path: Sidecar JSON path.

    Returns:
        dict[str, Any] | None: Parsed metadata, or None when absent/corrupt.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist resume metadata.

    Args:
        path: Sidecar JSON path.
        payload: Metadata mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    """Parse an HTTP ``Content-Range`` header.

    Args:
        value: Header value such as ``bytes 100-199/200``.

    Returns:
        tuple[int, int, int] | None: ``(start, end, total)`` with total 0 when unknown.
    """
    match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", (value or "").strip())
    if not match:
        return None
    total = 0 if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


@dataclass
class DownloadTask:
    """State tracking container for an asynchronous model download task."""

    download_id: str
    repo_id: str
    filename: str
    name: str
    parameters: str
    is_vision: bool
    status: str  # "queued" | "downloading" | "verifying" | "completed" | "failed" | "cancelled"
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    percent: float = 0.0
    eta_seconds: float = 0.0
    error: str | None = None
    registered_model_id: str | None = None
    file_path: str = ""
    mmproj_file_path: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert task state to JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Serialized download task status.
        """
        speed_mb = round(self.speed_bps / (1024 * 1024), 2) if self.speed_bps > 0 else 0.0
        return {
            "download_id": self.download_id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "name": self.name,
            "parameters": self.parameters,
            "is_vision": self.is_vision,
            "status": self.status,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "downloaded_formatted": self._format_size(self.downloaded_bytes),
            "total_formatted": self._format_size(self.total_bytes),
            "speed_bps": self.speed_bps,
            "speed_mb_s": speed_mb,
            "percent": round(self.percent, 1),
            "eta_seconds": round(self.eta_seconds, 1),
            "error": self.error,
            "registered_model_id": self.registered_model_id,
            "file_path": self.file_path,
            "mmproj_file_path": self.mmproj_file_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format raw byte counts into human-readable strings.

        Args:
            size_bytes: Raw size in bytes.

        Returns:
            str: Formatted string (e.g. '1.24 GB', '450.2 MB').
        """
        if size_bytes <= 0:
            return "0 B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        if size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class HuggingFaceHubService:
    """Service client for searching and inspecting HuggingFace Hub models."""

    BASE_API_URL = "https://huggingface.co/api"

    @classmethod
    def _get_headers(cls) -> dict[str, str]:
        """Build HTTP request headers including optional authorization token.

        Returns:
            dict[str, str]: HTTP request headers.
        """
        headers = {
            "User-Agent": "Robot-Core-LLM-Downloader/1.0",
            "Accept": "application/json",
        }
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token.strip()}"
        return headers

    @classmethod
    async def search_models(
        cls,
        query: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search HuggingFace Hub for GGUF models.

        Args:
            query: Search query text (e.g. 'qwen2.5-coder', 'llama-3.2', 'deepseek').
            limit: Maximum number of models to return.

        Returns:
            list[dict[str, Any]]: Parsed model card summaries.
        """
        cleaned_query = (query or "").strip()
        search_param = cleaned_query if cleaned_query else "instruct"

        params: dict[str, Any] = {
            "search": search_param,
            "filter": "gguf",
            "sort": "downloads",
            "direction": "-1",
            "limit": min(limit, 50),
        }

        url = f"{cls.BASE_API_URL}/models"
        results: list[dict[str, Any]] = []

        # If user searched for an exact repo name (contains '/' without spaces), try fetching direct repo
        if "/" in cleaned_query and " " not in cleaned_query:
            try:
                direct_model = await cls.get_model_details(cleaned_query)
                if direct_model:
                    results.append(direct_model)
            except Exception:
                pass

        try:
            async with httpx.AsyncClient(headers=cls._get_headers(), timeout=12.0) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    raw_items = resp.json()
                    for item in raw_items:
                        repo_id = item.get("id") or item.get("modelId")
                        if not repo_id or any(r["repo_id"] == repo_id for r in results):
                            continue

                        author = repo_id.split("/")[0] if "/" in repo_id else "unknown"
                        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                        
                        # Extract parameter estimation
                        param_match = re.search(r"(\d+(?:\.\d+)?[bB])", model_name)
                        params_est = param_match.group(1).upper() if param_match else "Unknown"

                        tags = item.get("tags", [])
                        pipeline = item.get("pipeline_tag", "")
                        is_vision = (
                            "image-text-to-text" in pipeline
                            or any("vision" in t.lower() or "vlm" in t.lower() for t in tags)
                            or "vl" in model_name.lower()
                        )
                        is_uncensored = any(
                            x in model_name.lower() or any(x in t.lower() for t in tags)
                            for x in ["uncensored", "abliterated", "refusal-free"]
                        )

                        results.append({
                            "repo_id": repo_id,
                            "author": author,
                            "name": model_name,
                            "parameters": params_est,
                            "downloads": item.get("downloads", 0),
                            "likes": item.get("likes", 0),
                            "last_modified": item.get("lastModified", ""),
                            "is_vision": is_vision,
                            "is_uncensored": is_uncensored,
                            "tags": tags[:10],
                        })
        except Exception as e:
            logger.error("hf_search_failed", query=query, error=str(e))

        return results[:limit]

    @classmethod
    async def get_model_details(cls, repo_id: str) -> dict[str, Any] | None:
        """Fetch high-level details for a single HuggingFace model repo.

        Args:
            repo_id: Repository ID (e.g. 'bartowski/Qwen2.5-Coder-3B-Instruct-GGUF').

        Returns:
            dict[str, Any] | None: Parsed model summary or None if not found.
        """
        url = f"{cls.BASE_API_URL}/models/{repo_id}"
        try:
            async with httpx.AsyncClient(headers=cls._get_headers(), timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    author = repo_id.split("/")[0] if "/" in repo_id else "unknown"
                    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                    param_match = re.search(r"(\d+(?:\.\d+)?[bB])", model_name)
                    params_est = param_match.group(1).upper() if param_match else "Unknown"
                    tags = data.get("tags", [])
                    pipeline = data.get("pipeline_tag", "")
                    is_vision = "image-text-to-text" in pipeline or "vl" in model_name.lower()
                    is_uncensored = "uncensored" in model_name.lower() or "abliterated" in model_name.lower()

                    return {
                        "repo_id": repo_id,
                        "author": author,
                        "name": model_name,
                        "parameters": params_est,
                        "downloads": data.get("downloads", 0),
                        "likes": data.get("likes", 0),
                        "last_modified": data.get("lastModified", ""),
                        "is_vision": is_vision,
                        "is_uncensored": is_uncensored,
                        "tags": tags[:10],
                    }
        except Exception as exc:
            logger.warning("hf_model_details_lookup_failed", error=str(exc))
        return None

    @classmethod
    async def get_model_files(
        cls,
        repo_id: str,
        base_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve all GGUF files and quantizations available in a repository.

        Args:
            repo_id: Repository identifier.
            base_dir: Optional base directory to check downloaded status.

        Returns:
            list[dict[str, Any]]: List of GGUF file entries with sizes, quantization info, and status.
        """
        root = base_dir or _project_root()
        llm_dir = root / "models" / "llm"
        vision_dir = root / "models" / "vision"

        url = f"{cls.BASE_API_URL}/models/{repo_id}/tree/main"
        files: list[dict[str, Any]] = []

        try:
            async with httpx.AsyncClient(headers=cls._get_headers(), timeout=12.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    tree = resp.json()
                    for item in tree:
                        path_str = item.get("path", "")
                        if not path_str.lower().endswith(".gguf"):
                            continue

                        filename = Path(path_str).name
                        size_bytes = int(item.get("size") or 0)
                        is_mmproj = "mmproj" in filename.lower()

                        # Check local download state
                        target_dir = vision_dir if is_mmproj else llm_dir
                        local_file = target_dir / filename
                        is_downloaded = (
                            local_file.exists()
                            and local_file.stat().st_size > 1_000_000
                            and (size_bytes <= 0 or local_file.stat().st_size >= int(size_bytes * 0.98))
                        )

                        # Parse quantization string (e.g. Q4_K_M, Q5_K_M, IQ3_M, FP16)
                        quant_match = re.search(
                            r"[-_\.](Q\d+_[A-Za-z0-9_]+|Q\d+_\d+|Q\d+|IQ\d+_[A-Za-z0-9_]+|IQ\d+|BF16|FP16|FP32|F16|F32)\.gguf$",
                            filename,
                            re.IGNORECASE,
                        )
                        quant = quant_match.group(1).upper() if quant_match else "GGUF"

                        # Determine if this is recommended for RTX 3050 (4GB VRAM)
                        is_recommended = quant in ("Q4_K_M", "Q4_0", "Q4_1") and not is_mmproj
                        # RTX 3050 has 4GB VRAM (~3.6GB usable)
                        fits_vram = size_bytes < 3_800_000_000 if size_bytes > 0 else True

                        files.append({
                            "filename": filename,
                            "path": path_str,
                            "size_bytes": size_bytes,
                            "size_formatted": DownloadTask._format_size(size_bytes),
                            "quantization": quant,
                            "is_mmproj": is_mmproj,
                            "is_downloaded": is_downloaded,
                            "is_recommended": is_recommended,
                            "fits_vram": fits_vram,
                        })
        except Exception as e:
            logger.error("hf_get_model_files_failed", repo_id=repo_id, error=str(e))

        # Sort: recommended first, then smaller sizes to larger sizes
        files.sort(key=lambda x: (not x["is_recommended"], x["is_mmproj"], x["size_bytes"]))
        return files


class HuggingFaceDownloadManager:
    """Manages background asynchronous streaming downloads of HuggingFace models."""

    _instance: HuggingFaceDownloadManager | None = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        base_dir: Path | None = None,
        max_retained_tasks: int = MAX_RETAINED_DOWNLOAD_TASKS,
        max_concurrent_downloads: int = DEFAULT_MAX_CONCURRENT_DOWNLOADS,
    ) -> None:
        """Initialize the download manager.

        Args:
            base_dir: Optional base project directory. Defaults to the repository root so
                downloads never depend on the process working directory.
            max_retained_tasks: Cap for finished task history; active tasks are
                never pruned. Keeps `_tasks` from growing unboundedly.
            max_concurrent_downloads: Maximum number of file transfers in flight.
        """
        self.base_dir = base_dir or _project_root()
        self.llm_dir = self.base_dir / "models" / "llm"
        self.vision_dir = self.base_dir / "models" / "vision"
        self.llm_dir.mkdir(parents=True, exist_ok=True)
        self.vision_dir.mkdir(parents=True, exist_ok=True)

        self._tasks: dict[str, DownloadTask] = {}
        self._async_tasks: dict[str, asyncio.Task[None]] = {}
        self._max_retained_tasks = max(1, int(max_retained_tasks))
        self._download_semaphore = asyncio.Semaphore(max(1, int(max_concurrent_downloads)))

    @classmethod
    def get_instance(cls, base_dir: Path | None = None) -> HuggingFaceDownloadManager:
        """Retrieve or create singleton download manager instance.

        Args:
            base_dir: Optional base project root directory.

        Returns:
            HuggingFaceDownloadManager: Singleton instance.
        """
        # The constructor creates directories and an asyncio.Semaphore; two
        # concurrent callers must not each build a manager (the second would
        # orphan the first's in-flight tasks).
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(base_dir=base_dir)
            return cls._instance

    def get_all_tasks(self) -> list[dict[str, Any]]:
        """Return serialized state of all active and historical download tasks.

        Returns:
            list[dict[str, Any]]: List of task status dictionaries.
        """
        return [task.to_dict() for task in self._tasks.values()]

    def get_task(self, download_id: str) -> dict[str, Any] | None:
        """Return serialized state of a specific download task.

        Args:
            download_id: Unique task identifier.

        Returns:
            dict[str, Any] | None: Task state dictionary or None if not found.
        """
        task = self._tasks.get(download_id)
        return task.to_dict() if task else None

    def _prune_finished_tasks(self) -> None:
        """Bound task history: drop oldest completed/failed/cancelled entries beyond the cap.

        Active tasks (queued/downloading/verifying) are never removed, so in-progress
        downloads and the most recent finished history entries remain queryable.
        """
        overflow = len(self._tasks) - self._max_retained_tasks
        if overflow <= 0:
            return
        finished = sorted(
            (task for task in self._tasks.values() if task.status in ("completed", "failed", "cancelled")),
            key=lambda task: task.updated_at,
        )
        for task in finished[:overflow]:
            self._tasks.pop(task.download_id, None)
            self._async_tasks.pop(task.download_id, None)
            logger.debug("download_task_history_pruned", download_id=task.download_id, status=task.status)

    @staticmethod
    def _cleanup_artifacts(dest_path: Path) -> None:
        """Remove partial files for a destination without touching a finished model.

        Args:
            dest_path: Final destination path.
        """
        for artifact in (_part_path(dest_path), _meta_path(dest_path)):
            try:
                artifact.unlink(missing_ok=True)
            except OSError:
                pass

    async def cancel_download(self, download_id: str) -> bool:
        """Cancel an in-progress download task and clean up temporary partial files.

        Args:
            download_id: Unique task identifier to cancel.

        Returns:
            bool: True if task was cancelled, False otherwise.
        """
        task = self._tasks.get(download_id)
        if not task:
            return False

        if task.status in ("completed", "failed", "cancelled"):
            return task.status == "cancelled"

        task._cancel_event.set()
        task.status = "cancelled"
        task.updated_at = time.time()

        # Cancel underlying asyncio coroutine
        async_task = self._async_tasks.get(download_id)
        if async_task and not async_task.done():
            async_task.cancel()

        # Remove incomplete .part files (model and mmproj) plus their metadata.
        for raw_path in (task.file_path, task.mmproj_file_path):
            if raw_path:
                self._cleanup_artifacts(Path(raw_path))

        logger.info("download_cancelled_by_user", download_id=download_id, filename=task.filename)
        return True

    @staticmethod
    def _public_error(exc: BaseException) -> str:
        """Translate an internal failure into an API-safe message.

        Args:
            exc: Caught exception.

        Returns:
            str: Message containing no absolute path or signed URL.
        """
        if isinstance(exc, DownloadError):
            return str(exc)
        if isinstance(exc, httpx.TransportError):
            return "Network error while contacting HuggingFace."
        return "Download failed; see server logs for details."

    def start_download(
        self,
        repo_id: str,
        filename: str,
        name: str | None = None,
        parameters: str | None = None,
        is_vision: bool = False,
        mmproj_filename: str | None = None,
    ) -> DownloadTask:
        """Enqueue and launch background download of a GGUF model from HuggingFace.

        The filename (and optional mmproj filename) is validated before any filesystem
        access so traversal, absolute paths and symlinks can never write outside the
        model directories. Invalid requests yield a ``failed`` task whose error is safe
        to display through the API.

        Args:
            repo_id: Repository ID (e.g. 'bartowski/Qwen2.5-Coder-3B-Instruct-GGUF').
            filename: Target GGUF filename (e.g. 'Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf').
            name: Optional human-readable model name.
            parameters: Optional parameter count string (e.g. '3B', '1.5B').
            is_vision: Whether the model supports vision VLM capabilities.
            mmproj_filename: Optional multimodal projector companion file.

        Returns:
            DownloadTask: Newly enqueued (or rejected) download task instance.
        """
        download_id = f"dl_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        display_name = name or Path(filename if isinstance(filename, str) else "model").stem.replace("-", " ").replace("_", " ").title()
        param_str = parameters
        if not param_str and isinstance(filename, str):
            param_match = re.search(r"(\d+(?:\.\d+)?[bB])", filename)
            param_str = param_match.group(1).upper() if param_match else "Custom"

        task = DownloadTask(
            download_id=download_id,
            repo_id=repo_id if isinstance(repo_id, str) else "",
            filename=filename if isinstance(filename, str) else "",
            name=display_name,
            parameters=param_str or "Custom",
            is_vision=is_vision,
            status="queued",
        )

        try:
            validated_repo = _validate_repo_id(repo_id)
            validated_name = _validate_gguf_filename(filename, "model")
            is_mmproj = "mmproj" in validated_name.lower()
            target_dir = self.vision_dir if is_mmproj else self.llm_dir
            dest_path = _resolve_inside(target_dir, validated_name, "model")
            mmproj_dest: Path | None = None
            if mmproj_filename:
                validated_mmproj = _validate_gguf_filename(mmproj_filename, "mmproj")
                mmproj_dest = _resolve_inside(self.vision_dir, validated_mmproj, "mmproj")
        except DownloadError as exc:
            task.status = "failed"
            task.error = str(exc)
            task.updated_at = time.time()
            self._tasks[download_id] = task
            self._prune_finished_tasks()
            logger.warning("download_request_rejected", repo_id=repo_id, reason=str(exc))
            return task

        task.filename = validated_name
        task.repo_id = validated_repo
        task.file_path = str(dest_path)
        task.mmproj_file_path = str(mmproj_dest) if mmproj_dest else ""

        # Deduplicate by (repo_id, filename): the same file from a different repository is
        # a distinct artefact and must not silently share a task.
        for existing in self._tasks.values():
            if (
                existing.repo_id == task.repo_id
                and existing.filename == task.filename
                and existing.status in ("queued", "downloading", "verifying")
            ):
                return existing

        active = [
            t for t in self._tasks.values() if t.status in ("queued", "downloading", "verifying")
        ]
        if len(active) >= self._max_retained_tasks:
            task.status = "failed"
            task.error = (
                f"Too many downloads are already queued (limit {self._max_retained_tasks}). "
                "Wait for one to finish and try again."
            )
            task.updated_at = time.time()
            self._tasks[download_id] = task
            self._prune_finished_tasks()
            logger.warning("download_queue_saturated", active=len(active))
            return task

        self._tasks[download_id] = task
        self._prune_finished_tasks()
        coro = self._download_worker(task, dest_path, mmproj_dest)
        self._async_tasks[download_id] = asyncio.create_task(coro)

        logger.info(
            "download_enqueued",
            download_id=download_id,
            repo=task.repo_id,
            file=task.filename,
            mmproj=mmproj_dest.name if mmproj_dest else None,
        )
        return task

    @staticmethod
    def _header_value(headers: Any, name: str) -> str:
        """Read a response header case-insensitively.

        Args:
            headers: Mapping or httpx.Headers instance.
            name: Header name.

        Returns:
            str: Header value or an empty string.
        """
        try:
            if name in headers:
                return str(headers[name])
        except Exception:
            pass
        lowered = name.lower()
        try:
            for key, value in headers.items():
                if str(key).lower() == lowered:
                    return str(value)
        except Exception:
            pass
        return ""

    @classmethod
    def _resume_metadata_matches(
        cls,
        meta: dict[str, Any] | None,
        repo_id: str,
        filename: str,
    ) -> bool:
        """Return whether a sidecar can safely authorise a Range resume.

        Args:
            meta: Parsed sidecar metadata.
            repo_id: Expected repository identifier.
            filename: Expected filename.

        Returns:
            bool: True when the sidecar belongs to this artefact and carries a validator.
        """
        if not meta:
            return False
        if meta.get("repo_id") != repo_id or meta.get("filename") != filename:
            return False
        return bool(meta.get("etag") or meta.get("last_modified"))

    @classmethod
    def _validators_match(cls, meta: dict[str, Any] | None, headers: Any) -> bool:
        """Return whether the server's ETag/Last-Modified still matches the sidecar.

        Args:
            meta: Parsed sidecar metadata.
            headers: Response headers from the ranged request.

        Returns:
            bool: True when the remote artefact revision is unchanged.
        """
        if not meta:
            return False
        stored_etag = str(meta.get("etag") or "")
        if stored_etag:
            return cls._header_value(headers, "etag") == stored_etag
        stored_lm = str(meta.get("last_modified") or "")
        if stored_lm:
            return cls._header_value(headers, "last-modified") == stored_lm
        return False

    @classmethod
    def _promote_verified(cls, part_path: Path, dest_path: Path, expected_total: int) -> None:
        """Swap a verified partial file into place, retaining the old file until verified.

        The previous model is moved to ``<dest>.bak`` first and restored if the swap or
        post-swap verification fails, so a corrupt download can never replace a working
        model. Only the part file is removed on failure; the destination is preserved.

        Args:
            part_path: Fully downloaded and verified partial file.
            dest_path: Final model destination.
            expected_total: Declared file size (0 when unknown).

        Raises:
            DownloadError: If the candidate is invalid or the swap fails.
        """
        actual_total = part_path.stat().st_size
        if expected_total > 0 and actual_total != expected_total:
            part_path.unlink(missing_ok=True)
            raise DownloadError(
                f"Download incomplete: received {actual_total} of {expected_total} bytes."
            )
        if not ModelRegistry.verify_gguf_header(part_path):
            part_path.unlink(missing_ok=True)
            raise DownloadError("Verification failed: downloaded file is not a valid GGUF model.")

        if dest_path.is_symlink():
            part_path.unlink(missing_ok=True)
            raise DownloadError("Refusing to replace a model through a symbolic link.")

        backup_path = _backup_path(dest_path)
        moved_previous = False
        if dest_path.exists():
            os.replace(dest_path, backup_path)
            moved_previous = True
            _fsync_dir(dest_path.parent)

        try:
            os.replace(part_path, dest_path)
            _fsync_path(dest_path)
            _fsync_dir(dest_path.parent)
            if expected_total > 0 and dest_path.stat().st_size != expected_total:
                raise DownloadError("Verification failed: final file size does not match the download.")
            if not ModelRegistry.verify_gguf_header(dest_path):
                raise DownloadError("Verification failed: downloaded file is not a valid GGUF model.")
        except BaseException:
            try:
                dest_path.unlink(missing_ok=True)
            except OSError:
                pass
            if moved_previous and backup_path.exists():
                os.replace(backup_path, dest_path)
            raise

        if moved_previous and backup_path.exists():
            try:
                backup_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def _download_one(
        self,
        task: DownloadTask,
        filename: str,
        dest_path: Path,
        progress_base: int,
        total_base: int,
    ) -> int:
        """Download and verify a single file, resuming only when the revision matches.

        Args:
            task: Parent download task.
            filename: Validated plain filename inside the destination directory.
            dest_path: Resolved destination path.
            progress_base: Bytes completed by earlier files in this task.
            total_base: Bytes declared by earlier files in this task.

        Returns:
            int: Final size of the downloaded file.

        Raises:
            DownloadError: On HTTP, validation, truncation, or disk-space failures.
        """
        part_path = _part_path(dest_path)
        meta_path = _meta_path(dest_path)
        url = f"https://huggingface.co/{task.repo_id}/resolve/main/{quote(filename)}"

        resume_bytes = 0
        meta = _read_sidecar(meta_path)
        if part_path.exists():
            if not self._resume_metadata_matches(meta, task.repo_id, filename):
                self._cleanup_artifacts(dest_path)
                meta = None
            else:
                assert meta is not None
                size = part_path.stat().st_size
                known_total = int(meta.get("total_bytes") or 0)
                if known_total > 0 and size == known_total:
                    # A previous run finished the transfer but died before the swap.
                    task.status = "verifying"
                    self._promote_verified(part_path, dest_path, known_total)
                    meta_path.unlink(missing_ok=True)
                    task.total_bytes = total_base + known_total
                    task.downloaded_bytes = progress_base + known_total
                    logger.info("download_resumed_verified_locally", filename=filename)
                    return known_total
                if size > 0 and (known_total == 0 or size < known_total):
                    resume_bytes = size
                else:
                    self._cleanup_artifacts(dest_path)
                    meta = None
        else:
            meta_path.unlink(missing_ok=True)
            meta = None

        task.status = "downloading"
        task.downloaded_bytes = progress_base + resume_bytes

        for attempt in range(2):
            use_resume = resume_bytes > 0 and attempt == 0
            headers = HuggingFaceHubService._get_headers()
            if use_resume:
                headers["Range"] = f"bytes={resume_bytes}-"
            timeout = httpx.Timeout(connect=15.0, read=120.0, write=90.0, pool=30.0)
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    status_code = resp.status_code
                    if status_code == 416:
                        # The partial file is at/over the remote size and is unusable.
                        self._cleanup_artifacts(dest_path)
                        resume_bytes = 0
                        meta = None
                        task.downloaded_bytes = progress_base
                        continue
                    if status_code not in (200, 206):
                        raise _HTTPStatusError(status_code)

                    total_for_file = 0
                    server_headers = resp.headers

                    if status_code == 200:
                        # The server ignored the Range request (or this is a fresh start):
                        # truncate and take the full body.
                        if resume_bytes and part_path.exists():
                            part_path.unlink(missing_ok=True)
                        resume_bytes = 0
                        task.downloaded_bytes = progress_base
                        total_for_file = int(self._header_value(server_headers, "content-length") or 0)
                        mode = "wb"
                    else:
                        if not use_resume:
                            raise DownloadError("HuggingFace returned an unexpected partial response.")
                        content_range = _parse_content_range(
                            self._header_value(server_headers, "content-range")
                        )
                        restart = content_range is None
                        if not restart:
                            assert content_range is not None
                            range_start, _range_end, range_total = content_range
                            meta_total = int(meta.get("total_bytes") or 0) if meta else 0
                            if range_start != resume_bytes:
                                restart = True
                            elif meta_total and range_total and meta_total != range_total:
                                restart = True
                            elif not self._validators_match(meta, server_headers):
                                restart = True
                            total_for_file = range_total
                        if restart:
                            self._cleanup_artifacts(dest_path)
                            resume_bytes = 0
                            meta = None
                            task.downloaded_bytes = progress_base
                            logger.warning(
                                "download_resume_invalidated",
                                filename=filename,
                                reason="revision or range mismatch",
                            )
                            continue
                        mode = "ab"

                    if total_for_file > 0:
                        required = max(0, total_for_file - resume_bytes)
                        try:
                            free_bytes = shutil.disk_usage(dest_path.parent).free
                        except OSError:
                            free_bytes = required
                        if free_bytes < required + DISK_SPACE_HEADROOM_BYTES:
                            raise DownloadError("Insufficient free disk space to download this file.")
                        task.total_bytes = total_base + total_for_file
                    elif total_base:
                        task.total_bytes = total_base

                    # Persist the revision/ETag before writing so an interrupted transfer can
                    # only be resumed when the remote artefact is provably unchanged.
                    _write_sidecar(
                        meta_path,
                        {
                            "repo_id": task.repo_id,
                            "filename": filename,
                            "etag": self._header_value(server_headers, "etag"),
                            "last_modified": self._header_value(server_headers, "last-modified"),
                            "total_bytes": total_for_file,
                            "revision": "main",
                        },
                    )

                    last_time = time.time()
                    last_bytes = task.downloaded_bytes
                    write_buf = bytearray()

                    async def _flush_buf() -> None:
                        """Flush the in-memory chunk buffer to the partial file."""
                        nonlocal write_buf
                        if not write_buf:
                            return
                        data = bytes(write_buf)
                        write_buf = bytearray()
                        await asyncio.to_thread(f.write, data)

                    with open(part_path, mode) as f:
                        async for chunk in resp.aiter_bytes(chunk_size=524288):  # 512 KB chunks
                            if task._cancel_event.is_set():
                                raise asyncio.CancelledError()

                            write_buf += chunk
                            task.downloaded_bytes += len(chunk)
                            now = time.time()
                            elapsed = now - last_time

                            if len(write_buf) >= DOWNLOAD_FLUSH_THRESHOLD_BYTES:
                                await _flush_buf()

                            # Update metrics every 300ms
                            if elapsed >= 0.3:
                                delta_bytes = task.downloaded_bytes - last_bytes
                                speed = delta_bytes / elapsed
                                # Exponential Moving Average smoothing
                                task.speed_bps = (
                                    (task.speed_bps * 0.7 + speed * 0.3) if task.speed_bps > 0 else speed
                                )

                                if task.total_bytes > 0:
                                    task.percent = min(
                                        100.0, (task.downloaded_bytes / task.total_bytes) * 100.0
                                    )
                                    remaining = task.total_bytes - task.downloaded_bytes
                                    task.eta_seconds = (
                                        remaining / task.speed_bps if task.speed_bps > 0 else 0.0
                                    )

                                task.updated_at = now
                                last_time = now
                                last_bytes = task.downloaded_bytes

                        # Drain the remaining buffer and durably flush before verification.
                        await _flush_buf()
                        await asyncio.to_thread(f.flush)
                        await asyncio.to_thread(os.fsync, f.fileno())

                    actual_size = part_path.stat().st_size
                    if total_for_file > 0 and actual_size != total_for_file:
                        self._cleanup_artifacts(dest_path)
                        raise DownloadError(
                            f"Download incomplete: received {actual_size} of {total_for_file} bytes."
                        )
                    if not ModelRegistry.verify_gguf_header(part_path):
                        self._cleanup_artifacts(dest_path)
                        raise DownloadError(
                            "Verification failed: downloaded file is not a valid GGUF model."
                        )

                    task.status = "verifying"
                    task.speed_bps = 0.0
                    task.eta_seconds = 0.0
                    task.updated_at = time.time()
                    try:
                        self._promote_verified(part_path, dest_path, total_for_file)
                    except DownloadError:
                        self._cleanup_artifacts(dest_path)
                        raise
                    meta_path.unlink(missing_ok=True)

                    task.downloaded_bytes = progress_base + actual_size
                    if total_for_file > 0:
                        task.total_bytes = total_base + total_for_file
                    task.percent = (
                        min(100.0, (task.downloaded_bytes / task.total_bytes) * 100.0)
                        if task.total_bytes > 0
                        else 100.0
                    )
                    return actual_size

        raise DownloadError("Download could not be resumed safely; please retry.")

    async def _download_worker(
        self,
        task: DownloadTask,
        dest_path: Path,
        mmproj_path: Path | None = None,
    ) -> None:
        """Asynchronous worker executing chunked download, resume, mmproj fetch and registration.

        Args:
            task: Task tracking object.
            dest_path: Resolved target destination path for the model GGUF.
            mmproj_path: Optional resolved destination for the vision projector.
        """
        task.status = "queued"
        task.updated_at = time.time()

        try:
            async with self._download_semaphore:
                task.status = "downloading"
                task.updated_at = time.time()
                model_size = await self._download_one(task, task.filename, dest_path, 0, task.total_bytes)
                total = task.total_bytes
                progress = model_size

                if mmproj_path is not None:
                    mmproj_size = await self._download_one(
                        task,
                        mmproj_path.name,
                        mmproj_path,
                        progress,
                        total,
                    )
                    progress += mmproj_size
                    total = task.total_bytes

                task.status = "verifying"
                task.downloaded_bytes = progress
                task.total_bytes = max(total, progress)
                task.percent = 100.0
                task.updated_at = time.time()

            if task._cancel_event.is_set():
                raise asyncio.CancelledError()

            # Never report completion for a vision model whose projector is missing.
            if not dest_path.exists() or dest_path.stat().st_size < 4:
                raise DownloadError("Verification failed: model file is missing after download.")
            if mmproj_path is not None and (
                not mmproj_path.exists() or mmproj_path.stat().st_size < 4
            ):
                raise DownloadError("Verification failed: multimodal projector file is missing.")

            model_id = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", dest_path.stem).lower().strip("_")
            is_uncensored = any(
                x in dest_path.name.lower() or x in task.repo_id.lower()
                for x in ["uncensored", "abliterated", "sft"]
            )

            metadata = ModelMetadata(
                id=model_id,
                name=task.name,
                category=ModelCategory.COMMUNITY_HUGGINGFACE,
                parameters=task.parameters,
                repo_id=task.repo_id,
                filename=dest_path.name,
                mmproj_filename=mmproj_path.name if mmproj_path is not None else None,
                is_vision=task.is_vision,
                is_uncensored=is_uncensored,
                supports_tools=True,
                recommended_context=16384,
                description=f"HuggingFace model downloaded from {task.repo_id} ({task.parameters}).",
            )

            ModelRegistry.register_model(metadata, persist=True, base_dir=self.base_dir)
            task.registered_model_id = model_id
            task.status = "completed"
            task.updated_at = time.time()

            logger.info(
                "download_completed_and_registered",
                download_id=task.download_id,
                model_id=model_id,
                filename=task.filename,
                mmproj=mmproj_path.name if mmproj_path is not None else None,
                size_mb=round(task.downloaded_bytes / (1024 * 1024), 2),
            )

        except asyncio.CancelledError:
            task.status = "cancelled"
            task.updated_at = time.time()
            for raw_path in (task.file_path, task.mmproj_file_path):
                if raw_path:
                    self._cleanup_artifacts(Path(raw_path))
        except Exception as e:
            # Full details stay in server logs; API-visible text is sanitized.
            logger.error("download_worker_failed", download_id=task.download_id, error=str(e))
            task.status = "failed"
            task.error = self._public_error(e)
            task.updated_at = time.time()
        finally:
            # Keep `_tasks` bounded once this task reaches a terminal status.
            self._prune_finished_tasks()
