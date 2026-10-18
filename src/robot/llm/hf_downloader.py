"""HuggingFace Hub LLM search, inspection, and background streaming downloader.

Provides asynchronous searching of GGUF repositories on HuggingFace Hub, quantization
file tree inspection, resumable chunked downloads with real-time transfer metrics
(speed, ETA, percentage), binary GGUF verification, and dynamic local model registration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from robot.llm.model_registry import ModelCategory, ModelMetadata, ModelRegistry

logger = logging.getLogger(__name__)


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
        except Exception:
            pass
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
        root = base_dir or Path.cwd()
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

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize the download manager.

        Args:
            base_dir: Optional base project directory.
        """
        self.base_dir = base_dir or Path.cwd()
        self.llm_dir = self.base_dir / "models" / "llm"
        self.vision_dir = self.base_dir / "models" / "vision"
        self.llm_dir.mkdir(parents=True, exist_ok=True)
        self.vision_dir.mkdir(parents=True, exist_ok=True)

        self._tasks: dict[str, DownloadTask] = {}
        self._async_tasks: dict[str, asyncio.Task[None]] = {}

    @classmethod
    def get_instance(cls, base_dir: Path | None = None) -> HuggingFaceDownloadManager:
        """Retrieve or create singleton download manager instance.

        Args:
            base_dir: Optional base project root directory.

        Returns:
            HuggingFaceDownloadManager: Singleton instance.
        """
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

        task._cancel_event.set()
        task.status = "cancelled"
        task.updated_at = time.time()

        # Cancel underlying asyncio coroutine
        async_task = self._async_tasks.get(download_id)
        if async_task and not async_task.done():
            async_task.cancel()

        # Remove incomplete .part file
        part_path = Path(task.file_path + ".part")
        if part_path.exists():
            try:
                part_path.unlink()
            except Exception:
                pass

        logger.info("download_cancelled_by_user", download_id=download_id, filename=task.filename)
        return True

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

        Args:
            repo_id: Repository ID (e.g. 'bartowski/Qwen2.5-Coder-3B-Instruct-GGUF').
            filename: Target GGUF filename (e.g. 'Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf').
            name: Optional human-readable model name.
            parameters: Optional parameter count string (e.g. '3B', '1.5B').
            is_vision: Whether the model supports vision VLM capabilities.
            mmproj_filename: Optional multimodal projector companion file.

        Returns:
            DownloadTask: Newly enqueued download task instance.
        """
        # Check if already running this exact file
        for existing in self._tasks.values():
            if existing.filename == filename and existing.status in ("queued", "downloading", "verifying"):
                return existing

        download_id = f"dl_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        is_mmproj = "mmproj" in filename.lower()
        target_dir = self.vision_dir if is_mmproj else self.llm_dir
        dest_path = target_dir / filename

        display_name = name or Path(filename).stem.replace("-", " ").replace("_", " ").title()
        param_str = parameters
        if not param_str:
            param_match = re.search(r"(\d+(?:\.\d+)?[bB])", filename)
            param_str = param_match.group(1).upper() if param_match else "Custom"

        task = DownloadTask(
            download_id=download_id,
            repo_id=repo_id,
            filename=filename,
            name=display_name,
            parameters=param_str,
            is_vision=is_vision,
            status="queued",
            file_path=str(dest_path),
        )

        self._tasks[download_id] = task
        coro = self._download_worker(task, dest_path, mmproj_filename)
        self._async_tasks[download_id] = asyncio.create_task(coro)

        logger.info("download_enqueued", download_id=download_id, repo=repo_id, file=filename)
        return task

    async def _download_worker(
        self,
        task: DownloadTask,
        dest_path: Path,
        mmproj_filename: str | None = None,
    ) -> None:
        """Asynchronous worker executing chunked download, resume, and model registration.

        Args:
            task: Task tracking object.
            dest_path: Resolved target destination path.
            mmproj_filename: Optional companion vision projector filename.
        """
        download_url = f"https://huggingface.co/{task.repo_id}/resolve/main/{task.filename}"
        part_path = Path(str(dest_path) + ".part")

        task.status = "downloading"
        task.updated_at = time.time()

        try:
            # Check existing partial size for HTTP Range resume
            resume_bytes = 0
            if part_path.exists():
                resume_bytes = part_path.stat().st_size
                task.downloaded_bytes = resume_bytes

            headers = HuggingFaceHubService._get_headers()
            if resume_bytes > 0:
                headers["Range"] = f"bytes={resume_bytes}-"

            timeout = httpx.Timeout(connect=15.0, read=90.0, write=90.0, pool=90.0)
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", download_url) as resp:
                    # Validate HTTP status
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP error {resp.status_code} downloading from HuggingFace.")

                    # Determine total size
                    content_length = int(resp.headers.get("content-length", 0))
                    if resp.status_code == 206 and "content-range" in resp.headers:
                        # e.g. bytes 1048576-2097151/1117320768
                        crange = resp.headers["content-range"]
                        if "/" in crange:
                            task.total_bytes = int(crange.split("/")[-1])
                        else:
                            task.total_bytes = resume_bytes + content_length
                    else:
                        task.total_bytes = content_length
                        if resp.status_code == 200:
                            # Server did not accept Range or starting fresh
                            resume_bytes = 0
                            task.downloaded_bytes = 0

                    mode = "ab" if (resp.status_code == 206 and resume_bytes > 0) else "wb"
                    last_time = time.time()
                    last_bytes = task.downloaded_bytes

                    with open(part_path, mode) as f:
                        async for chunk in resp.aiter_bytes(chunk_size=524288):  # 512 KB chunks
                            if task._cancel_event.is_set():
                                return

                            f.write(chunk)
                            task.downloaded_bytes += len(chunk)
                            now = time.time()
                            elapsed = now - last_time

                            # Update metrics every 300ms
                            if elapsed >= 0.3:
                                delta_bytes = task.downloaded_bytes - last_bytes
                                speed = delta_bytes / elapsed
                                # Exponential Moving Average smoothing
                                task.speed_bps = (task.speed_bps * 0.7 + speed * 0.3) if task.speed_bps > 0 else speed
                                
                                if task.total_bytes > 0:
                                    task.percent = min(100.0, (task.downloaded_bytes / task.total_bytes) * 100.0)
                                    remaining = task.total_bytes - task.downloaded_bytes
                                    task.eta_seconds = remaining / task.speed_bps if task.speed_bps > 0 else 0.0

                                task.updated_at = now
                                last_time = now
                                last_bytes = task.downloaded_bytes

            # Finalize download: rename .part to final file
            task.status = "verifying"
            task.percent = 100.0
            task.speed_bps = 0.0
            task.eta_seconds = 0.0
            task.updated_at = time.time()

            part_path.replace(dest_path)

            # Binary GGUF magic header check
            if not ModelRegistry.verify_gguf_header(dest_path):
                dest_path.unlink(missing_ok=True)
                raise RuntimeError("Verification failed: File is not a valid binary GGUF model.")

            # Register into ModelRegistry
            model_id = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", dest_path.stem).lower().strip("_")
            is_uncensored = any(x in dest_path.name.lower() or x in task.repo_id.lower() for x in ["uncensored", "abliterated", "sft"])

            metadata = ModelMetadata(
                id=model_id,
                name=task.name,
                category=ModelCategory.COMMUNITY_HUGGINGFACE,
                parameters=task.parameters,
                repo_id=task.repo_id,
                filename=dest_path.name,
                mmproj_filename=mmproj_filename,
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
                size_mb=round(task.downloaded_bytes / (1024 * 1024), 2),
            )

        except asyncio.CancelledError:
            task.status = "cancelled"
            task.updated_at = time.time()
            if part_path.exists():
                part_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error("download_worker_failed", download_id=task.download_id, error=str(e))
            task.status = "failed"
            task.error = str(e)
            task.updated_at = time.time()
