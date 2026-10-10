# Lead Architect Directive — Post Phase 1-4 Sign-Off

Implement Phase 5: Orchestration Intelligence and Reliability.

Preserve the existing architecture and inspect the current code before introducing new abstractions.

PHASE 5A — OBSERVABILITY

Add trace IDs and monotonic timestamps for pipeline stages.

Measure routing, LLM generation, tool execution, post-tool generation, TTS synthesis, playback initiation, and task completion separately.

Report P50, P95, and P99 latency by request category.

Verify the benchmark classification and averaging logic.

Preserve slow samples and failures in the report.

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