---
name: suggest-next-fix
description: >-
  Proactively analyze system state, recent user interactions, bug watchlists, and the TRACKER.md roadmap to recommend high-impact next fixes, performance optimizations, and feature implementations.
---

# Proactive Next-Fix Advisor Skill

Use this skill at the conclusion of tasks, when resuming sessions, or when the user asks "what should we do next?", "what needs fixing?", or asks for recommendations on improving the robot system.

---

## 1. Automated Advisor Engine

Run the helper script to scan pending items across `TRACKER.md` and codebase status:
```bash
.venv/bin/python .agents/skills/suggest-next-fix/scripts/scan_next_fixes.py
```

This script parses:
- 🔴 **Priority 1 (Stability & Fixes)**: Unchecked bug watchlists, edge cases, error-prone hardware bridges (e.g. ALSA audio stream sync, serial reconnects).
- 🟡 **Priority 2 (Latency & Performance)**: TTFT reduction, streaming vocoder pipelines, memory caching, query classification speed.
- 🟢 **Priority 3 (Strategic Roadmap)**: Multi-modal vision enhancements, local VLMs, physical servo kinematics, continuous background memory consolidation.

---

## 2. Recommendation Framing Workflow

When presenting recommendations to the user, structure them in three clearly delineated tiers:

### Tier 1: Immediate Stability & Hardening (Critical)
- Address items that could cause unexpected crashes, dead air, or tool execution exceptions during live conversations.
- Example: Handling loose LLM tool argument schemas, auto-reconnecting lost serial bridges, ALSA ring-buffer underflow protection.

### Tier 2: Real-Time Performance & Polish (High ROI)
- Optimizations that noticeably reduce voice-to-voice latency, improve avatar fluidity, or streamline frontend responsiveness.
- Example: Local speculative decoding on RTX 3050, sub-100ms streaming TTS audio onset, mobile HUD touch interactions.

### Tier 3: Strategic Architectural Milestones (Roadmap)
- Step-change capabilities advancing Luna's autonomy and embodiment.
- Example: Local SmolVLM/Moondream2 on AMD Radeon 680M iGPU, 3D Spatial Scene Graph with YOLOv8, continuous sleep/wake memory synthesis.

---

## 3. Communication Guidelines
1. **Be Concise & Actionable**: Provide a clear 1-2 sentence rationale for *why* an item matters now.
2. **Present Clear Tradeoffs**: Note hardware resource usage (e.g., VRAM on RTX 3050 vs. system RAM vs. battery/thermal envelope).
3. **Offer Direct Action Steps**: Always end with a crisp prompt offering to immediately begin work on the user's preferred item.
