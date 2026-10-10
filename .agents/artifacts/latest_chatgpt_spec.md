# ARCHITECTURAL DIRECTIVE — ROBOT V2, DAY 2
**Lead Architect (ChatGPT in Chrome) $\leftrightarrow$ Engineer (Antigravity IDE)**
*Timestamp: 2026-10-10*

## Decision
**Execute Tool Suite Hardening before further audio optimization, DAG orchestration, or proactive autonomy.**

Repository: `/home/ameya/robot`

## Objective
Harden all 59 tools without changing their intended behavior or destabilizing the existing V2 architecture.

---

## Phase 1 — Audit
- Inventory all 59 tools and their signatures.
- Document required/optional arguments, aliases, defaults, ranges, side effects, timeouts, and dependencies.
- Classify tools by risk and execution type.
- Identify existing validation and result conventions.
- Record the baseline test results.
- **Do not modify the main project until the audit identifies a concrete implementation plan.**

---

## Phase 2 — Input Normalization
Implement a reusable `@harden_tool_inputs` mechanism that:
- Resolves explicitly approved aliases.
- Detects conflicting canonical and alias arguments.
- Performs strict, schema-driven type coercion.
- Validates ranges and required fields.
- Applies only documented, tool-specific defaults.
- Preserves synchronous and asynchronous behavior.
- Preserves function metadata and signatures.
- Propagates cancellation.
- Produces structured validation errors.
- **Rule**: Do not invent arguments, silently discard conflicting values, or coerce ambiguous inputs into potentially dangerous actions.

---

## Phase 3 — Driver Fallbacks
Implement capability-aware fallback for brightness, system audio, and network status.
For each driver:
- Check capability and availability.
- Distinguish unsupported operations from recoverable failures.
- Avoid implicit privilege escalation.
- Verify resulting state when possible.
- Do not blindly retry potentially non-idempotent operations.
- Do not report success solely because a command returned exit code zero.
- Do not assume brightness drivers are interchangeable: internal panel backlights and external monitor controls are distinct capabilities.

---

## Phase 4 — Standardized Outcomes
- Ensure tool results unambiguously distinguish success, validation errors, unsupported capabilities, timeouts, execution errors, partial success, and failed verification.
- Preserve compatibility with existing callers. Document any required migration.

---

## Phase 5 — Verification
Add tests for:
- All approved aliases.
- Conflicting arguments.
- Strict coercion and range validation.
- Async and sync tools.
- Driver fallback selection.
- Permission failures.
- Timeouts and cancellation.
- Failed post-operation verification.
- Duplicate or repeated side effects.
- Existing behavior and regression cases.
- **Run the complete existing test suite, including the 173-test regression baseline.**

---

## Constraints
- Do not introduce unrelated refactors.
- Do not change prompt caching or conversation serialization.
- Do not modify the LLM, STT, TTS, memory, or frontend architecture unless a verified dependency requires it.
- Do not apply a blanket decorator to all tools without auditing their signatures.
- Keep changes incremental and reviewable.
- Preserve a clear Git diff and rollback path.

---

## Deliverables
1. Tool inventory and risk classification.
2. Input-normalization implementation.
3. Driver capability/fallback report.
4. Test results and regression comparison.
5. Remaining known limitations.
6. Evidence that no unrelated behavior changed.
7. **Stop after Day 2 hardening. Do not begin DAG orchestration automatically.**
8. Return the report to ChatGPT for architectural review before advancing to the next workstream.
