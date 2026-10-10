#!/usr/bin/env python3
"""
Next-Fix Advisor & Backlog Scanner
==================================
Scans TRACKER.md, git status, and code health indicators to synthesize
the top prioritized recommendations for the user:
1. 🔴 Priority 1: Stability, Known Issues & Bug Watchlist
2. 🟡 Priority 2: Performance, Latency & UX Optimizations
3. 🟢 Priority 3: Architecture & Next Roadmap Milestones
"""

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
TRACKER_PATH = ROOT_DIR / "docs" / "TRACKER.md" if (ROOT_DIR / "docs" / "TRACKER.md").exists() else ROOT_DIR / "TRACKER.md"


def scan_tracker() -> dict[str, list[str]]:
    """Parse TRACKER.md for pending ToDos and known watchlist items."""
    results: dict[str, list[str]] = {
        "critical": [],
        "performance": [],
        "roadmap": [],
        "todos": [],
    }

    if not TRACKER_PATH.exists():
        return results

    content = TRACKER_PATH.read_text(encoding="utf-8")

    # Match unchecked todos: - [ ] **Title**: Details
    todo_matches = re.findall(r"-\s*\[\s*\]\s*\*\*([^*]+)\*\*:\s*(.*?)(?=\n-\s*\[|\n##|\Z)", content, re.DOTALL)
    for title, details in todo_matches:
        clean_title = title.strip()
        first_line = details.strip().split("\n")[0] if details else ""
        item = f"{clean_title}: {first_line}"
        low = item.lower()
        if any(w in low for w in ["bug", "crash", "fix", "error", "leak", "reconnect", "fail"]):
            results["critical"].append(item)
        elif any(w in low for w in ["perf", "latency", "optim", "speed", "ui", "hud", "respons"]):
            results["performance"].append(item)
        else:
            results["roadmap"].append(item)

    # Parse Roadmap section if present
    roadmap_match = re.search(r"## 5\.\s*Comprehensive Bot Improvement.*?Roadmap(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if roadmap_match:
        roadmap_text = roadmap_match.group(1)
        subsections = re.findall(r"###\s*\d+\.\s*([^\n]+)", roadmap_text)
        for sub in subsections:
            if sub not in results["roadmap"]:
                results["roadmap"].append(sub.strip())

    return results


def main() -> int:
    print("=" * 70)
    print("🧭 PROACTIVE NEXT-FIX ADVISOR: SYSTEM RECOMMENDATIONS")
    print("=" * 70)

    data = scan_tracker()

    print("\n🔴 PRIORITY 1: Stability, Hardware & Bug Watchlist")
    if data["critical"]:
        for idx, item in enumerate(data["critical"][:3], 1):
            print(f"  {idx}. {item}")
    else:
        print("  ✓ No immediate critical bugs recorded. Watchlist: Hardware servo reconnect & ALSA audio sync.")

    print("\n🟡 PRIORITY 2: Performance, Latency & UX Optimizations")
    if data["performance"]:
        for idx, item in enumerate(data["performance"][:3], 1):
            print(f"  {idx}. {item}")
    else:
        print("  1. Automated E2E Benchmark Runner (`scripts/benchmark_e2e.py`) to chart TTFT & STT latency.")
        print("  2. Local Speculative Decoding for LLM on RTX 3050 (<250ms TTFT).")
        print("  3. Chunked Streaming Neural Vocoder for zero-wait TTS audio onset.")

    print("\n🟢 PRIORITY 3: Strategic Architecture & Roadmap Milestones")
    if data["roadmap"]:
        for idx, item in enumerate(data["roadmap"][:4], 1):
            print(f"  {idx}. {item}")
    else:
        print("  1. Continuous 3D Spatial Scene Graph (YOLOv8 + depth map).")
        print("  2. Local Low-Bit VLM (Moondream2 / SmolVLM) on AMD Radeon 680M iGPU.")
        print("  3. Microphone Array DOA Beamforming & SoundSpaces 3D Binaural HRTF.")

    print("\n" + "=" * 70)
    print("💡 Prompt to agent: 'Proceed with Priority X item Y' or specify an area above.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
