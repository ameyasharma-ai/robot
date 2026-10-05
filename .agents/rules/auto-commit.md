## Auto-Commit on Major Milestones Guidelines

This project requires automated git commits upon reaching major milestones.

Rules for every session:
1. **Trigger on Major Milestones**: Whenever you finish implementing a feature, resolving a critical issue, completing an architectural milestone, or fulfilling a `/goal` or user request:
   - Ensure tests pass: run `.venv/bin/pytest tests/`.
   - Ensure `TRACKER.md` is updated with feature status and bug resolutions.
   - Ensure knowledge graph is updated: run `.venv/bin/graphify update .`.
2. **Execute Git Commit**:
   - Check modified files with `git status -s`.
   - Stage the appropriate source, test, config, and documentation files (excluding `.env`, credentials, or temporary artifacts).
   - Create a clean git commit using Conventional Commits format (`feat(...)`, `fix(...)`, `perf(...)`, `refactor(...)`, etc.) with a descriptive summary of what was accomplished.
