#!/usr/bin/env python3
"""Download GGUF weights for FLUX.1 Schnell execution via stable-diffusion.cpp.

Architectural Rationale:
- NVIDIA RTX 3050 Laptop GPU has 4GB VRAM and host has 14GB RAM with 0B swap.
- Standard Diffusers PyTorch pipelines require >12GB RAM/VRAM even when quantized with bitsandbytes.
- stable-diffusion.cpp with GGUF architecture enables hybrid CPU+CUDA execution:
    * Diffusion model (3.83GB Q2_K) runs on NVIDIA RTX 3050 CUDA cores.
    * T5-XXL text encoder (2.14GB Q3_K_M) runs on host Ryzen 7 CPU via AVX2 (--clip-on-cpu).
    * CLIP-L (0.23GB) runs on CPU.
    * VAE (0.31GB) handles latent decoding with micro-tiling.
- Total memory fits comfortably within 3.8GB VRAM and 2.5GB RAM.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

TARGET_DIR = Path(__file__).resolve().parent.parent / "models" / "image" / "flux"

FILES_TO_DOWNLOAD = [
    {
        "repo_id": "leejet/FLUX.1-schnell-gguf",
        "filename": "flux1-schnell-q2_k.gguf",
        "description": "FLUX.1 Schnell Q2_K DiT Transformer (CUDA)",
        "expected_gb": 3.83,
    },
    {
        "repo_id": "city96/t5-v1_1-xxl-encoder-gguf",
        "filename": "t5-v1_1-xxl-encoder-Q3_K_M.gguf",
        "description": "T5-XXL Q3_K_M Text Encoder (Host CPU / AVX2)",
        "expected_gb": 2.14,
    },
    {
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "clip_l.safetensors",
        "description": "CLIP-L Text Encoder (Host CPU / AVX2)",
        "expected_gb": 0.23,
    },
    {
        "repo_id": "lzyvegetable/FLUX.1-schnell",
        "filename": "ae.safetensors",
        "description": "FLUX Autoencoder / VAE",
        "expected_gb": 0.31,
    },
]


def download_flux_gguf_suite() -> None:
    """Download the complete GGUF FLUX model suite for stable-diffusion.cpp."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    total_gb = sum(f["expected_gb"] for f in FILES_TO_DOWNLOAD)

    print("=" * 80)
    print("🎨 DOWNLOADING FLUX.1 SCHNELL GGUF SUITE FOR STABLE-DIFFUSION.CPP")
    print(f"📁 Target Directory: {TARGET_DIR}")
    print(f"📦 Total Download Size: ~{total_gb:.2f} GB (4 components)")
    print("=" * 80)

    start_total = time.time()
    for idx, item in enumerate(FILES_TO_DOWNLOAD, 1):
        target_path = TARGET_DIR / item["filename"]
        if target_path.exists() and target_path.stat().st_size > 10 * 1024 * 1024:
            size_gb = target_path.stat().st_size / (1024 * 1024 * 1024)
            print(f"\n[{idx}/{len(FILES_TO_DOWNLOAD)}] ✅ ALREADY CACHED: {item['filename']} ({size_gb:.2f} GB)")
            continue

        print(f"\n[{idx}/{len(FILES_TO_DOWNLOAD)}] ⬇️ Downloading: {item['description']}")
        print(f"   Repo: {item['repo_id']} | File: {item['filename']} (~{item['expected_gb']:.2f} GB)")
        t0 = time.time()
        try:
            cached_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["filename"],
                local_dir=str(TARGET_DIR),
            )
            elapsed = time.time() - t0
            print(f"   ✓ Finished in {elapsed:.1f}s ({elapsed / 60:.1f} min) -> {cached_path}")
        except Exception as err:
            print(f"   ❌ Error downloading {item['filename']}: {err}", file=sys.stderr)
            sys.exit(1)

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"🎉 ALL GGUF FLUX COMPONENTS DOWNLOADED SUCCESSFULLY in {total_time:.1f}s ({total_time / 60:.1f} min)!")
    print(f"📁 Verified files in {TARGET_DIR}:")
    for f in TARGET_DIR.iterdir():
        print(f"   • {f.name:<35} ({f.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    download_flux_gguf_suite()
