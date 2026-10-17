#!/usr/bin/env python3
"""Image model downloader and cache manager for FLUX and Stable Diffusion.

Pre-downloads and verifies model checkpoints, schedules local storage caching,
and checks model readiness for the GPU handoff pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot.image.model_registry import ImageModelFamily, ImageModelMetadata, ImageModelRegistry


def print_banner() -> None:
    """Print standard CLI banner."""
    print("=" * 80)
    print("🎨 ROBOT CORE — IMAGE MODEL CATALOG & DOWNLOAD MANAGER")
    print("=" * 80)


def list_models() -> None:
    """Display all registered image models and local download/cache statuses."""
    print("\n📋 REGISTERED IMAGE SYNTHESIS MODELS:\n")
    for m in ImageModelRegistry.list_models():
        cached = m.is_cached_locally()
        status_str = "✅ CACHED LOCALLY" if cached else "⏳ NOT CACHED"
        heavy_str = "⚡ HEAVY (GPU Handoff)" if m.is_heavy else "🌱 LIGHTWEIGHT"
        neg_str = "Negative Prompt: Yes" if m.supports_negative_prompt else "Negative Prompt: No"

        print(f"• {m.id:<18} [{m.family.value.upper()}] | {status_str}")
        print(f"    Name: {m.name} | Repo: {m.repo_id}")
        print(f"    {heavy_str} | VRAM: ~{m.vram_mb_required} MB | Steps: {m.recommended_steps} | Guidance: {m.default_guidance_scale}")
        print(f"    Resolution: {m.default_width}x{m.default_height} | {neg_str}")
        print(f"    Description: {m.description}\n")


def download_model(meta: ImageModelMetadata) -> bool:
    """Download and cache weights for a specified model.

    Args:
        meta: Target model metadata.

    Returns:
        bool: True if downloaded and cached successfully, False otherwise.
    """
    if meta.family == ImageModelFamily.PROCEDURAL:
        print(f"ℹ️ Model '{meta.id}' is procedural (zero-weight, CPU-rendered). Nothing to download.")
        return True

    print(f"\n======================================================================")
    print(f"📦 DOWNLOADING MODEL: {meta.name} ({meta.id})")
    print(f"   Family: {meta.family.value} | Repo: {meta.repo_id}")
    print(f"   Target VRAM: ~{meta.vram_mb_required} MB | Pipeline: {meta.pipeline_type}")
    print(f"======================================================================")

    try:
        start_ts = time.time()
        print(f"📡 Resolving and downloading fp16 weights for: {meta.repo_id}...")
        import torch

        if meta.family == ImageModelFamily.FLUX:
            from diffusers import FluxPipeline
            FluxPipeline.from_pretrained(
                meta.repo_id,
                torch_dtype=torch.float16,
            )
        elif meta.family == ImageModelFamily.SDXL:
            from diffusers import AutoPipelineForText2Image
            AutoPipelineForText2Image.from_pretrained(
                meta.repo_id,
                torch_dtype=torch.float16,
                variant="fp16",
            )
        else:
            from diffusers import AutoPipelineForText2Image
            AutoPipelineForText2Image.from_pretrained(
                meta.repo_id,
                torch_dtype=torch.float16,
                variant="fp16" if meta.id != "small-sd" else None,
                safety_checker=None,
                requires_safety_checker=False,
            )
        duration = time.time() - start_ts
        print(f"✨ Successfully downloaded and verified '{meta.id}' in {duration:.1f}s!")
        return True
    except Exception as e:
        print(f"❌ Failed to download model '{meta.id}': {e}")
        return False


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Image model manager for FLUX and Stable Diffusion")
    parser.add_argument("--list", action="store_true", help="List registered models and local cache status")
    parser.add_argument("--model", type=str, help="Specific model ID to download")
    parser.add_argument("--default", action="store_true", help="Download default model (sd-turbo)")
    parser.add_argument("--sd", action="store_true", help="Download Stable Diffusion models (sd-turbo, small-sd)")
    parser.add_argument("--flux", action="store_true", help="Download FLUX.1 models (flux-schnell)")
    parser.add_argument("--all", action="store_true", help="Download all registered models")

    args = parser.parse_args()
    print_banner()

    if args.list or len(sys.argv) == 1:
        list_models()
        return

    targets: list[ImageModelMetadata] = []
    if args.default:
        targets.append(ImageModelRegistry.get_default_model(gpu_available=True))

    if args.model:
        m = ImageModelRegistry.get(args.model)
        if not m:
            print(f"❌ Error: Unknown model identifier '{args.model}'.")
            print("Run with --list to see available model identifiers.")
            sys.exit(1)
        targets.append(m)

    if args.sd:
        targets.extend([m for m in ImageModelRegistry.list_sd_models() if m.id in ("sd-turbo", "small-sd")])

    if args.flux:
        targets.extend(ImageModelRegistry.list_flux_models())

    if args.all:
        targets = ImageModelRegistry.list_models()

    # Deduplicate targets
    unique_targets = {t.id: t for t in targets}.values()
    success_count = 0
    total = len(unique_targets)

    for target in unique_targets:
        if download_model(target):
            success_count += 1

    print("\n" + "=" * 80)
    print(f"🏁 Download process concluded: {success_count}/{total} models ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()
