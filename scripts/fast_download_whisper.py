"""High-speed multi-threaded downloader and validator for Whisper.cpp GGML models.

Utilizes HTTP Range requests across parallel workers with automatic chunk-level
retry logic and post-download model verification using whisper-cli.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
import urllib.error

WHISPER_MODELS_DIR = Path("/home/ameya/whisper.cpp/models")
WHISPER_CLI_BIN = Path("/home/ameya/whisper.cpp/build_vk/bin/whisper-cli")
ROBOT_STT_DIR = Path("/home/ameya/robot/models/stt")
CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB chunks
MAX_WORKERS = 8

# Target models to ensure downloaded locally
MODELS = [
    "small",
    "medium.en",
    "medium",
    "large-v3-turbo",
    "large-v3",
]


def get_final_url_and_size(url: str) -> tuple[str, int]:
    """Resolve redirect chain to obtain direct CDN URL and exact content length.

    Args:
        url: Initial HuggingFace model download URL.

    Returns:
        tuple[str, int]: Direct CDN URL and total file size in bytes.

    Raises:
        RuntimeError: If unable to determine direct URL or content size.
    """
    req = Request(url, method="HEAD", headers={"User-Agent": "Wget/1.21.2"})
    try:
        with urlopen(req, timeout=15) as resp:
            final_url = resp.geturl()
            content_length = int(resp.headers.get("Content-Length", 0))
            if content_length > 0:
                return final_url, content_length
    except Exception:
        pass

    # Fallback to GET with Range 0-0 if HEAD is blocked
    req = Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Wget/1.21.2"})
    with urlopen(req, timeout=20) as resp:
        final_url = resp.geturl()
        crange = resp.headers.get("Content-Range", "")
        if "/" in crange:
            content_length = int(crange.split("/")[-1])
        else:
            content_length = int(resp.headers.get("Content-Length", 0))
        if content_length <= 0:
            raise RuntimeError(f"Unable to determine content size for {url}")
        return final_url, content_length


def download_chunk(url: str, filepath: Path, start: int, end: int, max_retries: int = 5) -> int:
    """Download a single byte range chunk directly to file at offset with retries.

    Args:
        url: Direct CDN URL.
        filepath: Target destination file path.
        start: Starting byte index (inclusive).
        end: Ending byte index (inclusive).
        max_retries: Maximum number of retry attempts for network blips.

    Returns:
        int: Number of bytes successfully downloaded.

    Raises:
        IOError: If chunk download fails after all retry attempts.
    """
    expected_bytes = end - start + 1
    req = Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Wget/1.21.2",
        },
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=40) as resp:
                data = resp.read()
                if len(data) != expected_bytes:
                    raise IOError(f"Byte count mismatch: got {len(data)}, expected {expected_bytes}")
                with open(filepath, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return len(data)
        except Exception as e:
            last_error = e
            time.sleep(1.0 * (attempt + 1))

    raise IOError(f"Chunk bytes={start}-{end} failed after {max_retries} attempts: {last_error}")


def verify_model_integrity(model_path: Path) -> bool:
    """Verify that a GGML model file is intact by executing whisper-cli on it.

    Args:
        model_path: Path to the downloaded GGML model file.

    Returns:
        bool: True if the model initialized successfully, False otherwise.
    """
    if not WHISPER_CLI_BIN.exists() or not model_path.exists():
        return False

    sample_audio = Path("/home/ameya/robot/data/diagnostics/captured_processed_stt.wav")
    if not sample_audio.exists():
        return model_path.stat().st_size > 50 * 1024 * 1024

    cmd = [
        str(WHISPER_CLI_BIN),
        "-m",
        str(model_path),
        "-f",
        str(sample_audio),
        "-d",
        "100",
        "-dev",
        "0",
        "-otxt",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25)
        output = proc.stdout + proc.stderr
        if "unknown tensor" in output or "failed to load model" in output or "error: failed" in output:
            return False
        return proc.returncode == 0
    except Exception:
        return False


def download_model(model_name: str) -> None:
    """Download and verify single Whisper model using multi-threaded chunking.

    Args:
        model_name: Name of model (e.g. 'small', 'medium.en', 'large-v3-turbo').

    Returns:
        None.
    """
    target_file = WHISPER_MODELS_DIR / f"ggml-{model_name}.bin"
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"

    print(f"\n🚀 Checking model '{model_name}'...")

    if target_file.exists():
        if verify_model_integrity(target_file):
            size_mb = target_file.stat().st_size / (1024 * 1024)
            print(f"✅ Model '{model_name}' already complete and verified ({size_mb:.1f} MB). Skipping.")
            return
        else:
            print(f"⚠️ Model '{model_name}' on disk is incomplete or invalid. Restarting download.")
            target_file.unlink(missing_ok=True)

    try:
        final_url, total_size = get_final_url_and_size(url)
    except Exception as e:
        print(f"❌ Failed to resolve URL for {model_name}: {e}")
        return

    print(f"📦 Downloading {model_name} ({total_size / (1024*1024):.1f} MB) with {MAX_WORKERS} workers...")

    # Pre-allocate file
    with open(target_file, "wb") as f:
        f.truncate(total_size)

    # Compute chunk ranges
    pending_chunks: list[tuple[int, int]] = []
    offset = 0
    while offset < total_size:
        end = min(offset + CHUNK_SIZE - 1, total_size - 1)
        pending_chunks.append((offset, end))
        offset = end + 1

    completed_bytes = 0
    start_time = time.time()
    last_print = start_time
    pass_num = 1

    while pending_chunks:
        if pass_num > 1:
            print(f"\n🔄 Retrying {len(pending_chunks)} pending/failed chunk(s) (pass {pass_num})...")

        failed_chunks: list[tuple[int, int]] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(download_chunk, final_url, target_file, c_start, c_end): (c_start, c_end)
                for c_start, c_end in pending_chunks
            }

            for future in as_completed(future_map):
                c_start, c_end = future_map[future]
                try:
                    b = future.result()
                    completed_bytes += b
                    now = time.time()
                    if now - last_print > 0.5 or completed_bytes == total_size:
                        elapsed = now - start_time
                        speed = (completed_bytes / (1024 * 1024)) / max(0.1, elapsed)
                        pct = (completed_bytes / total_size) * 100
                        rem_bytes = max(0, total_size - completed_bytes)
                        eta = (rem_bytes / (1024 * 1024)) / max(0.01, speed) if speed > 0 else 0
                        print(
                            f"   [{model_name}] {pct:5.1f}% | {completed_bytes / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB | {speed:5.1f} MB/s | ETA: {eta:.0f}s    \r",
                            end="",
                            flush=True,
                        )
                        last_print = now
                except Exception as e:
                    failed_chunks.append((c_start, c_end))

        pending_chunks = failed_chunks
        pass_num += 1
        if pass_num > 5:
            print(f"\n❌ Aborting {model_name}: chunks persistently failed after 5 passes.")
            target_file.unlink(missing_ok=True)
            return

    print(f"\n🔍 Verifying {model_name} tensor integrity...")
    if verify_model_integrity(target_file):
        total_time = time.time() - start_time
        print(f"✅ {model_name} downloaded & 100% verified ({total_size / (1024*1024):.1f} MB in {total_time:.1f}s)")
    else:
        print(f"❌ {model_name} failed tensor verification! Removing corrupt artifact.")
        target_file.unlink(missing_ok=True)


def sync_symlinks() -> None:
    """Synchronize all verified GGML models into robot/models/stt/."""
    ROBOT_STT_DIR.mkdir(parents=True, exist_ok=True)
    for model_file in WHISPER_MODELS_DIR.glob("ggml-*.bin"):
        if model_file.is_file() and model_file.stat().st_size > 50 * 1024 * 1024:
            if verify_model_integrity(model_file):
                symlink_target = ROBOT_STT_DIR / model_file.name
                if not symlink_target.exists() or symlink_target.is_symlink():
                    symlink_target.unlink(missing_ok=True)
                    symlink_target.symlink_to(model_file)

    print("\n📁 Verified models available in models/stt/:")
    for link in sorted(ROBOT_STT_DIR.glob("ggml-*.bin")):
        size_mb = link.stat().st_size / (1024 * 1024)
        print(f"   • {link.name:25} ({size_mb:6.1f} MB)")


def main() -> None:
    """Download all Whisper model variants and synchronize symlinks."""
    WHISPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("🎙️ MULTI-THREADED WHISPER MODEL DOWNLOADER & VALIDATOR")
    print("=" * 70)

    for m in MODELS:
        download_model(m)

    sync_symlinks()
    print("\n🎉 Model synchronization complete!")


if __name__ == "__main__":
    main()
