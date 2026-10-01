---
trigger: always_on
description: Always consult and update TRACKER.md alongside graphify for features, ToDos, and issue resolution.
---

## Feature, ToDo & Issue Tracker Guidelines

This project maintains a centralized system tracker at `TRACKER.md`.

Rules for every session:
1. **Always consult `TRACKER.md`** at the beginning of tasks involving architecture, new features, or bug fixes to avoid duplicated work or regressions.
2. **Always update `TRACKER.md`** whenever you:
   - Implement or modify a feature (update the Feature Implementation Status Matrix).
   - Resolve an issue or discover a new bug (update Known Issues & Resolved Bugs Tracker).
   - Complete or introduce tasks (update Active ToDos).
3. **Always run `graphify update .`** after modifying code files or documentation to keep the graph and tracker in sync.
4. **Always ensure tests pass**: run `.venv/bin/pytest tests/`.
