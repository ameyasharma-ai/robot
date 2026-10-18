#!/usr/bin/env python3
"""Download and verify 4-bit NF4 FLUX.1 Schnell for RTX 3050 and RAM offloading.

Architectural Rationale:
- Standard FLUX.1 Schnell in FP16 is ~24GB, exceeding 4GB VRAM and causing OOM on 14GB RAM.
- 'magespace/FLUX.1-schnell-bnb-nf4' provides a complete 4-bit NF4 quantized pipeline (~12.5GB total).
- Using Hugging Face's snapshot_download caches all shards locally in ~/.cache/huggingface/hub/.
- Once cached, UnifiedImageProvider loads the pipeline 100% offline with sequential CPU offload,
  swapping layers between system RAM and NVIDIA RTX 3050 VRAM.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "magespace/FLUX.1-schnell-bnb-nf4"


def main() -> None:
    """Download and verify FLUX.1 Schnell NF4 weights."""
    print("=" * 75)
    print(f"🚀 DOWNLOADING FLUX.1 SCHNELL (4-BIT NF4) FOR NVIDIA RTX 3050 + RAM OFFLOAD")
    print(f"   Repository: {REPO_ID}")
    print(f"   Architecture: 12B DiT + T5-XXL + CLIP + VAE (NF4 4-bit quantized)")
    print(f"   Total Footprint: ~12.5 GB")
    print("=" * 75)

    start_time = time.time()
    try:
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            resume_download=True,
        )
        elapsed = time.time() - start_time
        print("\n" + "=" * 75)
        print(f"✅ DOWNLOAD COMPLETE in {elapsed:.1f}s ({elapsed / 60:.1f} minutes)!")
        print(f"📁 Local Snapshot Path: {local_dir}")
        print("=" * 75)
    except Exception as e:
        print(f"\n❌ Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
