"""Downloader and validator for Text-to-Speech models (Kokoro Neural & Piper ONNX).

Ensures all neural model weights, voice embeddings, and ONNX checkpoints are downloaded
and verified locally for offline, multi-engine speech synthesis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

TTS_DIR = Path("models/tts")
KOKORO_DIR = TTS_DIR / "kokoro"

KOKORO_MODELS: dict[str, str] = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

PIPER_VOICES: dict[str, tuple[str, str]] = {
    "en-us-ryan-medium.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ),
    "en-gb-alan-medium.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    ),
}


def download_file(url: str, dest: Path) -> None:
    """Download a remote file with progress reporting if not already present.

    Args:
        url: Remote HTTP/HTTPS URL.
        dest: Local destination Path.
    """
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  ✓ Already exists: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    print(f"  ↓ Downloading {dest.name} from {url[:60]}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    with urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = (downloaded / total) * 100
                print(f"\r    {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({percent:.1f}%)", end="", flush=True)

    print()
    tmp_path.rename(dest)
    print(f"  ✓ Downloaded {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    """Entry point for TTS model downloader CLI."""
    parser = argparse.ArgumentParser(description="Download and verify TTS models (Kokoro & Piper).")
    parser.add_argument("--kokoro-only", action="store_true", help="Download only Kokoro neural models.")
    parser.add_argument("--piper-only", action="store_true", help="Download only Piper ONNX voices.")
    args = parser.parse_args()

    print("======================================================================")
    print("🎙️ DOWNLOADING & VALIDATING MULTI-ENGINE TTS MODELS (KOKORO & PIPER)")
    print("======================================================================")

    if not args.piper_only:
        print("\n[Phase 1/2] Kokoro Neural TTS Models & 54 Voice Embeddings:")
        for filename, url in KOKORO_MODELS.items():
            download_file(url, KOKORO_DIR / filename)

    if not args.kokoro_only:
        print("\n[Phase 2/2] Additional Piper ONNX Voice Checkpoints:")
        for filename, (model_url, json_url) in PIPER_VOICES.items():
            download_file(model_url, TTS_DIR / filename)
            download_file(json_url, TTS_DIR / f"{filename}.json")

    print("\n[Phase 3/3] Validating TTS Subsystem via TTSManager...")
    try:
        from robot.tts.manager import TTSManager
        mgr = TTSManager()
        engines = mgr.list_engines()
        voices = mgr.list_voices()
        print(f"  ✓ TTSManager online! Active: {mgr.get_active_engine_name()} ({mgr.get_active_voice()})")
        print(f"  ✓ Total engines available: {len(engines)}, Total voices: {len(voices)}")
        print("\n✅ All TTS engines and voices ready for speech synthesis!")
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
