# Lead Architect Directive — Phase 5 Implementation Sequence

**Decision**: Proceed with Phase 5.

### Implementation Order:
1. **Step 0 — Regression verification** *(COMPLETED)*:
   - Fixed personality switching background pre-warming (`import asyncio` in `src/robot/api/__init__.py`).
   - Aligned `CuriosityEngine` in `src/robot/core/autonomy.py` to route through `ContextEngine.build_messages()` with invariant system prompt and canonical tools schemas.
   - Tested runtime KV-cache reuse and verified 185 regression tests passing.
2. **Step 1 — Observability**:
   - Implement trace IDs and monotonic timestamps across routing, LLM generation, tool execution, post-tool generation, TTS synthesis, playback initiation, and task completion. Correct benchmark classification and reporting. Establish representative baseline distributions.
3. **Step 2 — Tool contracts and network reliability**:
   - Audit all 59 tools, consolidate canonical execution metadata, standardize structured outcomes, and test permissions, validation, timeouts (3.0s cap for weather/search), retries only on idempotent operations, cancellation, and duplicate-side-effect prevention.
4. **Step 3 — Sequential planner–executor**:
   - Implement bounded, authorized, sequential plan execution with structured results, dependencies, revision/termination logic, cancellation, deadlines, and step limits.
5. **Step 4 — Memory and context integrity**:
   - Test bounded context growth, duplicate-turn prevention, persistent-memory writes and retrieval, tool-call identity, and side-effect integrity across multi-step sessions.

PHASE 5B — TOOL EXECUTION CONTRACTS

Audit all 59 registered tools.

Define or consolidate canonical metadata for schemas, permissions, capabilities, timeouts, and retry behavior.

Normalize execution outcomes into structured success, failure, denial, timeout, and cancellation states.

Prevent unsafe duplicate execution when a request is retried.

Propagate cancellation wherever supported.

Preserve authorization and validation checks across every execution path.

PHASE 5C — NETWORK RELIABILITY

Investigate the 7.49-second weather request and 4.50-second search turn using stage-level traces.

Add bounded timeouts and explicit error reporting where missing.

Implement retries only for operations that are safe to repeat.

Record external-service latency separately from LLM latency.

Ensure slow external operations cannot block unrelated assistant functionality indefinitely.

PHASE 5D — SEQUENTIAL PLANNER–EXECUTOR

Implement a structured task plan with explicit steps and dependencies.

Execute one authorized step at a time initially.

Return structured tool results to the planner.

Allow the planner to continue, revise, or terminate the plan based on actual results.

Add maximum-step, timeout, cancellation, and permission boundaries.

Require appropriate confirmation for consequential actions.

Never treat a tool invocation as proof that the requested real-world action succeeded.

PHASE 5E — MEMORY AND CONTEXT INTEGRITY

Verify bounded context growth during long sessions.

Prevent duplicated historical turns and repeated side effects.

Test persistent-memory writes and retrieval.

Preserve tool-call identity and argument integrity across multi-turn execution.

TESTING AND RELEASE GATES

All existing tests continue to pass.

Add targeted tests for failure, cancellation, timeout, and retry behavior.

Benchmark at least 100 representative turns per category where practical.

Report P50/P95/P99 latency, tool selection accuracy, tool success rate, timeout rate, cancellation behavior, and memory/context growth.

Distinguish measured results from estimates.

Do not claim that a target is met unless the evidence supports it.

Deliver a concise engineering report containing changed files, architecture changes, test results, benchmark data, regressions, and remaining risks.

Do not add more tools or parallel agents merely to increase feature count. Prioritize reliability and measurable end-to-end behavior.