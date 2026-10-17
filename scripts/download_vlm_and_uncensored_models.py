"""High-speed multi-threaded downloader and validator for 2B VLM and <4B Uncensored LLMs.

Supports parallel chunk downloads via HTTP Range requests with resume capability,
automatic redirect resolution, GGUF binary magic header verification, and seamless
directory placement into models/llm/ and models/vision/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from robot.llm.model_registry import ModelCategory, ModelMetadata, ModelRegistry

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks for parallel download
MAX_WORKERS = 4  # Keep concurrency balanced with connection limits


def get_final_url_and_size(url: str, timeout: int = 15) -> tuple[str, int]:
    """Resolve redirect chain to obtain direct CDN URL and exact content length.

    Args:
        url: Initial HuggingFace model download URL.
        timeout: Network timeout in seconds.

    Returns:
        tuple[str, int]: Direct CDN URL and total file size in bytes.

    Raises:
        RuntimeError: If unable to determine direct URL or content size.
    """
    req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            content_length = int(resp.headers.get("Content-Length", 0))
            if content_length > 0:
                return final_url, content_length
    except Exception:
        pass

    # Fallback to GET with Range 0-0 if HEAD is blocked or lacks Content-Length
    req = Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout + 5) as resp:
        final_url = resp.geturl()
        crange = resp.headers.get("Content-Range", "")
        if "/" in crange:
            content_length = int(crange.split("/")[-1])
        else:
            content_length = int(resp.headers.get("Content-Length", 0))
        if content_length <= 0:
            raise RuntimeError(f"Unable to determine content size for {url}")
        return final_url, content_length


def download_chunk(
    url: str,
    filepath: Path,
    start: int,
    end: int,
    max_retries: int = 5,
) -> int:
    """Download a byte range slice and write directly to file at byte offset.

    Args:
        url: Direct CDN download URL.
        filepath: Local target file path.
        start: Inclusive start byte index.
        end: Inclusive end byte index.
        max_retries: Maximum allowable retry attempts for connection interruptions.

    Returns:
        int: Number of bytes downloaded.

    Raises:
        RuntimeError: If chunk cannot be downloaded after max retries.
    """
    chunk_len = end - start + 1
    req = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"})

    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) != chunk_len:
                    time.sleep(0.5 * attempt)
                    continue

                with open(filepath, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return chunk_len
        except Exception:
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))
            else:
                raise RuntimeError(f"Failed chunk {start}-{end} after {max_retries} attempts.")
    return 0


def download_file_fast(url: str, target_path: Path) -> bool:
    """Download a large GGUF file using parallel chunk workers with resume support.

    Args:
        url: Source download URL.
        target_path: Destination file path on local filesystem.

    Returns:
        bool: True if download succeeded and GGUF header is valid.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📡 Resolving URL: {url}")
    try:
        final_url, total_bytes = get_final_url_and_size(url)
    except Exception as e:
        print(f"❌ Failed to resolve URL: {e}")
        return False

    size_mb = total_bytes / (1024 * 1024)
    print(f"📦 Target: {target_path.name} ({size_mb:.1f} MB)")

    # Check if already fully downloaded
    if target_path.exists() and target_path.stat().st_size == total_bytes:
        if ModelRegistry.verify_gguf_header(target_path):
            print(f"✅ File {target_path.name} already fully downloaded and verified.")
            return True

    # Pre-allocate sparse file if size differs or missing
    if not target_path.exists() or target_path.stat().st_size != total_bytes:
        with open(target_path, "wb") as f:
            f.truncate(total_bytes)

    # Divide file into chunks
    chunks: list[tuple[int, int]] = []
    for s in range(0, total_bytes, CHUNK_SIZE):
        e = min(s + CHUNK_SIZE - 1, total_bytes - 1)
        chunks.append((s, e))

    t0 = time.time()
    bytes_done = 0
    total_chunks = len(chunks)

    print(f"⚡ Starting multi-threaded download ({total_chunks} chunks, {MAX_WORKERS} workers)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_chunk, final_url, target_path, s, e): (s, e)
            for s, e in chunks
        }
        for idx, fut in enumerate(as_completed(futures), start=1):
            try:
                b = fut.result()
                bytes_done += b
                pct = (bytes_done / total_bytes) * 100.0
                elapsed = max(0.001, time.time() - t0)
                speed_mb = (bytes_done / (1024 * 1024)) / elapsed
                print(
                    f"\r⏳ [{idx}/{total_chunks}] {pct:.1f}% ({bytes_done / (1024*1024):.1f}/{size_mb:.1f} MB) @ {speed_mb:.1f} MB/s",
                    end="",
                    flush=True,
                )
            except Exception as e:
                print(f"\n❌ Chunk failed: {e}")
                return False

    elapsed = max(0.001, time.time() - t0)
    avg_speed = size_mb / elapsed
    print(f"\n✨ Download finished in {elapsed:.1f}s ({avg_speed:.1f} MB/s).")

    # Verify GGUF Magic Header
    if ModelRegistry.verify_gguf_header(target_path):
        print(f"🔒 GGUF binary magic header verified ('GGUF').")
        return True
    else:
        print(f"⚠️ Warning: File {target_path.name} does not start with valid GGUF magic header.")
        return False


def download_model(model_meta: ModelMetadata, base_dir: Path | None = None) -> bool:
    """Download a complete model and its vision projector if applicable.

    Args:
        model_meta: ModelMetadata specification.
        base_dir: Optional base directory.

    Returns:
        bool: True if model (and mmproj if applicable) downloaded successfully.
    """
    root = base_dir or Path.cwd()
    print("\n" + "=" * 70)
    print(f"🤖 Processing Model: {model_meta.name} ({model_meta.parameters}) [{model_meta.id}]")
    print(f"   Category: {model_meta.category.value} | Vision: {model_meta.is_vision} | Tools: {model_meta.supports_tools}")
    print(f"   Description: {model_meta.description}")
    print("=" * 70)

    # 1. Download Primary GGUF Model
    model_path = model_meta.get_model_path(root)
    model_url = f"https://huggingface.co/{model_meta.repo_id}/resolve/main/{model_meta.filename}"
    ok_model = download_file_fast(model_url, model_path)
    if not ok_model:
        return False

    # 2. Download Multimodal Vision Projector (mmproj) if Vision model
    if model_meta.is_vision and model_meta.mmproj_filename:
        mmproj_path = model_meta.get_mmproj_path(root)
        if mmproj_path:
            mmproj_url = f"https://huggingface.co/{model_meta.repo_id}/resolve/main/{model_meta.mmproj_filename}"
            print(f"\n📷 Downloading Multimodal Vision Projector (mmproj)...")
            ok_mm = download_file_fast(mmproj_url, mmproj_path)
            if not ok_mm:
                return False

    print(f"🎉 Model {model_meta.id} is ready for local inference!")
    return True


def list_catalog(base_dir: Path | None = None) -> None:
    """Print formatted model catalog with local availability indicators.

    Args:
        base_dir: Optional base directory.
    """
    root = base_dir or Path.cwd()
    print("\n" + "=" * 80)
    print("📋 LOCAL MODEL CATALOG: 2B VISION (VLM) & <4B UNCENSORED MODELS")
    print("=" * 80)
    for m in ModelRegistry.list_models():
        downloaded = m.is_downloaded(root)
        status_icon = "✅ DOWNLOADED" if downloaded else "⏳ AVAILABLE"
        vision_icon = "👁️ Vision" if m.is_vision else "💬 Text"
        tools_icon = "🔧 Tools" if m.supports_tools else "🚫 NoTools"
        uncensored_icon = "🔓 Uncensored" if m.is_uncensored else "🛡️ Guarded"
        
        print(f"• \033[1m{m.id:<32}\033[0m [{m.parameters}] | {status_icon}")
        print(f"    {vision_icon} | {tools_icon} | {uncensored_icon} | Context: {m.recommended_context}")
        print(f"    File: models/llm/{m.filename}")
        if m.mmproj_filename:
            print(f"    Vision: models/vision/{m.mmproj_filename}")
        print(f"    {m.description}")
        print()


def main() -> int:
    """CLI Entry point for model downloading and validation.

    Returns:
        int: Exit status code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Download 2B Vision Models and <4B Uncensored LLMs.")
    parser.add_argument("--model", type=str, help="Specific model identifier to download (e.g. qwen2-vl-2b-instruct).")
    parser.add_argument("--default", action="store_true", help="Download default 2B Vision model (qwen2-vl-2b-instruct).")
    parser.add_argument("--vlm", action="store_true", help="Download all 2B vision language models.")
    parser.add_argument("--uncensored", action="store_true", help="Download all uncensored models < 4B.")
    parser.add_argument("--all", action="store_true", help="Download entire catalog of 12 models.")
    parser.add_argument("--list", action="store_true", help="List all catalog models and local download status.")
    parser.add_argument("--verify", action="store_true", help="Verify GGUF binary magic headers for all local files.")

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent

    if args.list:
        list_catalog(base_dir)
        return 0

    if args.verify:
        print("🔍 Verifying local GGUF models...")
        for m in ModelRegistry.list_models():
            m_path = m.get_model_path(base_dir)
            if m_path.exists():
                valid = ModelRegistry.verify_gguf_header(m_path)
                print(f"  {'✅' if valid else '❌'} {m_path.name}: {'Valid GGUF' if valid else 'Invalid'}")
            if m.is_vision and m.mmproj_filename:
                mm_path = m.get_mmproj_path(base_dir)
                if mm_path and mm_path.exists():
                    valid = ModelRegistry.verify_gguf_header(mm_path)
                    print(f"  {'✅' if valid else '❌'} {mm_path.name}: {'Valid GGUF' if valid else 'Invalid'}")
        return 0

    targets: list[ModelMetadata] = []
    if args.model:
        m = ModelRegistry.get(args.model)
        if not m:
            print(f"❌ Unknown model '{args.model}'. Use --list to see available models.")
            return 1
        targets.append(m)
    elif args.default:
        targets.append(ModelRegistry.get_default_model())
    elif args.vlm:
        targets.extend(ModelRegistry.list_vlm_2b_models())
    elif args.uncensored:
        targets.extend(ModelRegistry.list_uncensored_models())
    elif args.all:
        targets.extend(ModelRegistry.list_models())
    else:
        # Default behavior if run without args: download default 2B VLM + top uncensored model
        print("ℹ️ No specific option selected. Defaulting to downloading default 2B VLM model.")
        targets.append(ModelRegistry.get_default_model())

    success_count = 0
    for target in targets:
        ok = download_model(target, base_dir)
        if ok:
            success_count += 1

    print("\n" + "=" * 70)
    print(f"🏁 Download process finished: {success_count}/{len(targets)} models ready.")
    print("=" * 70)
    return 0 if success_count == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
