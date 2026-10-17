NEXT ENGINEERING CYCLE — ROBOT INTELLIGENCE PLATFORM

Role: Lead Implementation & Testing Engineer
Architecture owner: Lead Systems & AI Architect
Priority: P0 — Safety and resilience

Mission

Implement and verify two work packages:

Independent actuator safety and watchdog assurance.

Audio-device fault containment and bounded recovery.

Do not add new product features or perform broad refactors during this cycle.

Work package A — Actuator safety

Inspect the existing SerialBridge, microcontroller protocol, watchdog, kinematics validator, motor-stall detection, and physical E-stop path.

Produce a concise map distinguishing host-enforced protections from hardware-enforced protections.

Verify that loss of host process, serial connectivity, or heartbeat results in the documented hardware safe state.

Reject malformed, stale, duplicate, and out-of-range motion commands.

Ensure that reconnecting a device never automatically resumes motion.

Add deterministic simulated-controller tests for timeout, disconnect, malformed packet, duplicate command, E-stop latching, and recovery.

Conduct guarded physical verification where hardware is available. Do not claim hardware safety from simulation alone.

Work package B — Audio resilience

Inspect the current capture adapter, AudioPlayer, playback lock, VAD pipeline, wake-word state machine, standby transition, and device configuration.

Add explicit device-health states and bounded recovery without creating a competing playback writer.

Handle capture and playback failure independently.

Ensure barge-in cancels the current playback generation and purges stale clauses.

Add deterministic tests for underflow, device disconnect, sample-rate change, retry exhaustion, concurrent earcons/TTS, and interruption during recovery.

Add lightweight, turn-correlated audio metrics without blocking the audio callback.

Verify recovery using simulated adapters first and hardware-in-the-loop tests second.

Non-negotiable constraints

Keep NVIDIA RTX 3050 VRAM strictly below 3.8 GB.

Do not change the model, context allocation, canonical tool prefix, or KV-cache configuration.

Preserve existing permission gates and the independent physical E-stop.

No automatic replay of potentially executed motion commands.

No unbounded retries, queues, or blocking device-recovery loops.

Preserve current wake-word gating, playback serialization, anti-pop fades, and standby behavior.

Reuse actual repository contracts and file locations; do not duplicate existing abstractions unnecessarily.

Verification and reporting

Run the existing test suite before and after changes. Add deterministic failure-injection tests, run the full pytest suite, run import verification, and execute the relevant soak tests. Report physical verification separately from simulated verification.

Deliver:

Files changed and architectural rationale.

New tests and exact pass/fail counts.

Hardware tests actually performed and their results.

Before/after P50/P95/P99 latency and resource measurements.

Maximum observed NVIDIA VRAM under the test workload.

Known limitations and any unresolved safety risks.

Do not mark the cycle complete until all applicable regression tests pass, no new import errors exist, the VRAM ceiling is respected, and all unverified hardware assumptions are explicitly documented.