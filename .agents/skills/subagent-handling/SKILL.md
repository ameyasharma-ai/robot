---
name: subagent-handling
description: >-
  Standardized protocol for orchestrating, prompting, supervising, and synthesizing browser and worker subagents effectively without loss of context or drift.
---

# Subagent Handling & Orchestration Skill

Use this skill whenever delegating complex, visual, exploratory, or long-running sub-tasks to subagents (such as `browser_subagent` or worker subagents).

---

## 1. Subagent Lifecycle & Principles

Subagents are autonomous, one-shot execution environments. Because they execute in a separate sub-context, you must follow these four core principles:

1. **Comprehensive Task Framing**:
   - Never pass vague or underspecified prompts (e.g., "Check the UI").
   - Explicitly define:
     - URL or endpoint to navigate to.
     - Viewport dimensions (e.g., mobile 390x844 vs. desktop 1920x1080).
     - Specific DOM elements, buttons, and interactions to execute.
     - Success criteria and clear exit conditions ("Stop and report when...").
     - Exact required return payload (e.g. visual layout defects, console errors, button responsiveness).

2. **Descriptive Naming**:
   - `TaskName`: Short, human-readable title (e.g., "Inspecting Mobile Cyberdeck HUD").
   - `TaskSummary`: Concise 1-2 sentence overview for the user interface.
   - `RecordingName`: All lowercase with underscores, max 3 words (e.g., `mobile_ui_audit`).

3. **Autonomous Verification & Visual Feedback**:
   - After a subagent completes, read its report, inspect generated recordings or screenshots, and verify whether the objective was met.
   - Never assume success without checking the subagent's exit status and output.

4. **Synthesis & Context Preservation**:
   - Extract the concrete findings from the subagent's report.
   - Update the primary context, code, or `TRACKER.md` based on those findings.
   - Inform the user of key outcomes without overwhelming them with raw execution logs.

---

## 2. Browser Subagent Best Practices

When delegating UI and frontend validation to `browser_subagent`:

```python
browser_subagent(
    TaskName="Audit Mobile Responsive Layout",
    TaskSummary="Verify that the cyberdeck UI renders cleanly on mobile viewports without horizontal clipping or overlapping HUD cards.",
    RecordingName="mobile_audit",
    Task="""
    1. Navigate to http://localhost:8000.
    2. Resize viewport to 390x844 (simulating mobile portrait).
    3. Verify that the mobile tab bar ('AVATAR', 'CHAT', 'HUD') is visible.
    4. Click 'CHAT' tab and verify that the message container fills the screen cleanly.
    5. Click the microphone toggle button and verify that the status badge changes to 'LISTENING'.
    6. Resize viewport to 1440x900 (simulating desktop).
    7. Verify that all 3 columns render side-by-side with no overlapping text.
    8. Return a final summary detailing:
       - Any observed element overlaps
       - Button click responsiveness
       - Any browser console errors or failed network requests
    """,
)
```

---

## 3. Error Handling & Recovery

- If a browser subagent reports `open_browser_url failed`, verify that the backend server is running (`ps aux | grep uvicorn` or `curl -I http://localhost:8000`).
- If an element is not clickable, instruct the subagent to scroll into view or inspect CSS z-index and pointer-events.
- Do not poll subagents in an infinite loop; rely on reactive wakeups and process returns cleanly.
