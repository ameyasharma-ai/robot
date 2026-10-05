---
name: auto-commit
description: Automatically verify and commit codebase changes with conventional commit messages after completing major milestones, features, or bug fixes.
---

# Auto-Commit on Major Milestones Skill

Use this skill whenever you complete a major milestone, feature implementation, architectural refactoring, or significant bug fix in this repository.

## Milestone Definition

A **major milestone** is reached when:
- A user request, task, or `/goal` has been fulfilled.
- A new feature or subsystem is fully implemented and tested (e.g., new tool, API endpoint, UI view).
- A critical bug, performance bottleneck, or regression has been diagnosed, resolved, and verified.
- An architectural refactoring or multi-file configuration update is completed.

---

## Pre-Commit Verification Checklist

Before creating a commit, you **MUST** ensure the following 3 gates pass:

1. **Test Verification**:
   All unit and integration tests must pass with zero failures:
   ```bash
   .venv/bin/pytest tests/
   ```
   *Never commit code that breaks existing tests.*

2. **Tracker Synchronization**:
   [`TRACKER.md`](file:///home/ameya/robot/TRACKER.md) must be updated:
   - Mark completed feature/task in **Feature Implementation Status Matrix** or **Active ToDos**.
   - Document any bug fixes in **Known Issues & Resolved Bugs Tracker**.

3. **Knowledge Graph Synchronization**:
   Keep the AST knowledge graph up to date:
   ```bash
   .venv/bin/graphify update .
   ```

---

## Commit Workflow

1. **Inspect Working Tree**:
   Check modified and untracked files:
   ```bash
   git status -s
   ```
   Verify that temporary scratch files, `.env` files, credentials, or large binary model weights (`.bin`, `.gguf`, `.onnx`, `.glb`) are NOT staged.

2. **Stage Relevant Changes**:
   Stage code, tests, documentation, and configuration files:
   ```bash
   git add src/ tests/ config/ scripts/ TRACKER.md .agents/
   ```
   *(Or specify exact modified paths to ensure clean commits).*

3. **Formulate Conventional Commit Message**:
   Follow the format:
   ```text
   <type>(<scope>): <short imperative summary>

   - Detailed bullet points highlighting key additions or alterations
   - Root causes and resolutions for bug fixes
   - Test and verification status
   ```

   **Types**:
   - `feat`: New feature or user-facing functionality (e.g., `feat(chat): add ChatGPT-like workspace with SSE streaming and mic unloading`)
   - `fix`: Bug fix or error resolution (e.g., `fix(avatar): resolve false speaking and continuous motion during idle/music`)
   - `perf`: Performance improvement (e.g., `perf(llm): optimize KV cache pre-warming for sub-200ms TTFT`)
   - `refactor`: Code reorganization without functional changes
   - `test`: Adding or updating test suites
   - `docs`: Documentation updates only

4. **Execute Commit**:
   Commit the staged changes using bash:
   ```bash
   git commit -m "<commit message>"
   ```

5. **Verify Clean Tree**:
   ```bash
   git status
   ```
