---
trigger: always_on
description: Guidelines for orchestrating, prompting, supervising, and synthesizing subagent executions.
---

# Subagent Handling & Orchestration Rule

When using subagents (e.g. `browser_subagent` or autonomous sub-tasks):

1. **Unambiguous, Detailed Prompts**:
   - Every subagent invocation must include explicit goals, step-by-step actions, URLs, screen sizes, and unambiguous termination criteria. Avoid vague one-line prompts.

2. **Always Review Subagent Output**:
   - Thoroughly inspect the subagent's return report and artifacts (recordings/screenshots) before marking a task as done.
   - If an error occurred or an assertion failed in the subagent, address the root cause in the main conversation.

3. **No Redundant or Parallel Conflicts**:
   - Never launch concurrent subagents targeting identical files or overlapping browser states.

4. **Synthesize Findings**:
   - Integrate concrete results from the subagent into the main conversation and project tracker.
