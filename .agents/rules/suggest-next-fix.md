---
trigger: always_on
description: Proactively recommend prioritized next fixes and roadmap milestones upon task completion.
---

# Proactive Next-Fix Advisor Rule

Whenever you complete a primary task, resolve a bug, or reach a logical milestone:

1. **Scan Ahead**:
   - Check `TRACKER.md` (Active ToDos, Known Issues Watchlist, Comprehensive Roadmap) or execute:
     ```bash
     .venv/bin/python .agents/skills/suggest-next-fix/scripts/scan_next_fixes.py
     ```

2. **Proactively Suggest Next Steps**:
   - In your summary response, include a brief, prioritized list of recommended next fixes or enhancements:
     - 🔴 **Stability**: Edge cases or hardware reconnection hardening.
     - 🟡 **Performance**: Voice latency, TTFT, or frontend fluidity.
     - 🟢 **Roadmap**: Next high-leverage architectural milestone from `TRACKER.md`.

3. **Keep Suggestions Actionable**:
   - Keep each item to 1-2 concise bullet points and ask the user which area they would like to tackle next.
