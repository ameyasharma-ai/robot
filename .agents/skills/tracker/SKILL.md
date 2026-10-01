---
name: tracker
description: Track features, ToDos, architecture changes, and issues in TRACKER.md
---

# Feature, ToDo & Issue Tracker Skill

Use this skill whenever you implement, optimize, debug, or evaluate features in this codebase.

## Workflow

1. **Before Starting**:
   - Read [`TRACKER.md`](file:///home/ameya/robot/TRACKER.md) to review active features, backlog ToDos, and known issues.
   - Run `graphify query "<feature>"` or inspect `graphify-out/` to trace code relationships.

2. **While Working**:
   - Respect hardware allocation (NVIDIA RTX 3050 for LLM, AMD Radeon 680M for STT/Rendering, CPU AVX2 for TTS/Vision).
   - Ensure latency targets remain under 0.8s end-to-end.

3. **After Completing Work**:
   - Update [`TRACKER.md`](file:///home/ameya/robot/TRACKER.md):
     - Mark completed items as ✅ **Active**.
     - Add new features to the **Feature Implementation Status Matrix**.
     - Document resolved bugs or new findings in the **Known Issues & Resolved Bugs Tracker**.
     - Add new backlog items to the **Active ToDos** section.
   - Run test verification:
     ```bash
     .venv/bin/pytest tests/
     ```
   - Update the knowledge graph:
     ```bash
     .venv/bin/graphify update .
     ```
