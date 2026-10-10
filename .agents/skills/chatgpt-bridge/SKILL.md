---
name: chatgpt-bridge
description: >-
  Automatically orchestrate multi-agent collaboration between Antigravity IDE and ChatGPT running in Chrome. Streams architectural specifications, implements code, runs tests, and iterates back-and-forth until a target goal is achieved.
---

# ChatGPT AI Development Bridge Skill

This skill allows Antigravity IDE to pair-program and collaborate in real time with ChatGPT running in Google Chrome on Linux Mint. It enables an autonomous **Architect $\leftrightarrow$ Engineer** feedback loop where:
1. **ChatGPT (Architect)** drafts architecture specifications, defines requirements, and reviews pull requests/diffs.
2. **Antigravity (Engineer)** analyzes the specification, implements code in the robot repository, runs unit/integration tests, and reports outcomes.
3. The agents iterate back and forth until the user's specific goal is achieved with full architectural sign-off.

---

## 1. Quick Verification & Status Check

Check the real-time health of the local bridge server and Chrome extension:
```bash
.venv/bin/python -c "from .agents.skills.chatgpt_bridge.scripts.bridge_client import ChatGPTBridgeClient; print(ChatGPTBridgeClient().get_status())"
```
Or view the live telemetry dashboard at:
👉 **`http://127.0.0.1:8765`**

---

## 2. Automated Multi-Turn Collaboration Loop

To initiate a goal-driven back-and-forth session with ChatGPT:

```bash
.venv/bin/python .agents/skills/chatgpt-bridge/scripts/run_collaboration_loop.py "<GOAL_DESCRIPTION>"
```

### Example Goals:
- `"Design and implement an adaptive audio buffer manager for ALSA ring-buffers to prevent underflow clicks."`
- `"Create a lightweight temporal decay model for short-term visual working memory."`
- `"Refactor serial motion bridge with auto-reconnecting baud rate detection."`

### Autonomous Loop Execution Steps:
1. **Turn 1 (Architectural Specification)**:
   - Antigravity transmits the goal statement and robot hardware constraints (RTX 3050 4GB VRAM, AMD Radeon 680M iGPU, Ryzen 7 6800H CPU) into ChatGPT in Chrome.
   - ChatGPT streams the full architecture specification into Antigravity.
   - Saved to `.agents/artifacts/latest_chatgpt_spec.md`.

2. **Turn 2 (Engineering & Implementation)**:
   - Antigravity synthesizes the implementation code across the targeted modules.
   - Runs strict validation:
     ```bash
     .venv/bin/python .agents/skills/zero-import-errors/scripts/verify_imports.py
     .venv/bin/pytest tests/
     ```

3. **Turn 3 (Architectural Review & Sign-Off)**:
   - Antigravity packages the diff, test results, and benchmark timings into an **Engineering Report**.
   - Transmits the report back into ChatGPT in Chrome.
   - ChatGPT reviews the code against its original acceptance criteria and responds with feedback or architectural sign-off.

4. **Iterate**:
   - If changes are requested, the loop repeats with ChatGPT's specific feedback until full sign-off is granted.

---

## 3. Direct Interactive Prompts (Ad-Hoc Queries)

To send an ad-hoc prompt directly to ChatGPT in Chrome without a full loop:
```bash
/home/ameya/ai-dev-bridge/.venv/bin/python /home/ameya/ai-dev-bridge/bridge_cli.py ask "<PROMPT>"
```

To fetch whatever is currently on ChatGPT's screen without typing:
```bash
/home/ameya/ai-dev-bridge/.venv/bin/python /home/ameya/ai-dev-bridge/bridge_cli.py fetch
```

---

## 4. Hardware & Safety Boundaries
- **RTX 3050 (4GB VRAM)**: Reserved for local `llama-server` (Qwen3.5-4B). ChatGPT bridge operations run strictly over local WebSockets (`127.0.0.1:8765`) and do not consume GPU memory.
- **Chrome Privacy**: The bridge interacts solely with the active conversation tab via the unpacked extension in `~/ai-dev-bridge/extensions/chatgpt_bridge`. No credentials or browser session directories are scraped.
- **Git Discipline**: Every completed milestone must be checked with `verify_imports.py` and committed with conventional commits (`feat:`, `fix:`, `refactor:`).
