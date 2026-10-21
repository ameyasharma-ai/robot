# Embodied AI Robot Companion — Production Roadmap

Principal Systems Architecture • Robot V2 • 4 GB VRAM ceiling • Local-first desktop deployment

Ameya, the next milestone for Robot is not adding more intelligence or more tools. It is making the intelligence you already have reliable, interruptible, resource-aware, private, and easy to install.

Your existing stack—FastAPI, WebSockets, Ollama, Whisper, Kokoro, Three.js, SQLite, and ChromaDB—is sufficient for a serious local-first product. Given your previous work on the context layer, tool execution contracts, network reliability, and automated regression tests, I would build on those foundations rather than redesign everything.

## 1. Executive priorities

P0 — Runtime reliability

Prevent VRAM exhaustion, audio deadlocks, orphaned subprocesses, and corrupted conversation state.

P1 — Real-time conversation

Streaming STT, first-audio latency, barge-in, cancellation, and synchronized speech and avatar motion.

P2 — Consumer installation

Signed application bundles, model downloads, dependency isolation, updates, and recovery.

P3 — Privacy and persistence

Encrypted local data, explicit retention policies, reproducible backup and restore, and zero-by-default telemetry.

Architectural rule: every AI engine must be replaceable independently, every expensive operation must be cancellable, and every persistent state change must have a defined owner.

## 2. Target production architecture

Desktop shell — Tauri recommended

Three.js avatar · Settings · Model manager · Privacy controls

Local application supervisor

Lifecycle · IPC authentication · Health checks · Resource scheduler

Realtime pipeline

VAD · STT · Turn manager · TTS · Audio output

Intelligence pipeline

LLM · Context engine · Tools · Vision

Local data plane

SQLite · Encrypted vector store · Model registry · Backup manifests

All application traffic remains local unless the user explicitly enables a remote feature.

### Core architectural decisions

| Concern            | Decision                                                    |
| ------------------ | ----------------------------------------------------------- |
| Desktop shell      | Tauri + existing Three.js frontend                          |
| Backend            | Python sidecar, supervised by the desktop application       |
| API                | Local authenticated IPC; WebSockets for streaming           |
| LLM                | Ollama initially, with a backend adapter for alternatives   |
| Real-time audio    | Dedicated audio pipeline and turn-state machine             |
| Persistence        | SQLite as canonical source of truth                         |
| Semantic retrieval | Local vector index derived from canonical records           |
| GPU management     | Central resource scheduler plus runtime-specific controls   |
| Distribution       | Signed native packages with separately managed model assets |

Important: Tauri does not eliminate the need to ship or install the Python backend. Package the backend as a self-contained executable using an appropriate bundler, or install a managed runtime. Keep it out of the frontend process.

## 3. Phase 0 — Establish a reproducible baseline

Start here

Goal: Make every subsequent optimization measurable and reversible.

### Engineering tasks

- Create a single `doctor` command that checks GPU availability, model files, audio devices, camera permissions, ports, database migrations, and backend health.
- Record a clean baseline for startup time, first-token latency, first-audio latency, GPU memory, system RAM, CPU utilization, and audio underruns.
- Run existing regression tests and add integration tests for a complete voice interaction.
- Introduce structured logs with a shared `trace_id`, `session_id`, and `turn_id`.
- Add fault-injection tests for model crashes, GPU allocation failures, camera disconnects, interrupted streams, and unavailable audio devices.
- Pin runtime versions and model checksums in a release manifest.

### Acceptance criteria

- All existing tests pass.
- A fresh installation can diagnose missing dependencies without requiring manual debugging.
- A failed model process can be restarted without corrupting the session or restarting the entire desktop application.

## 4. Phase 1 — VRAM and concurrency orchestration

This is the most important technical problem on your RTX 3050 Mobile.

### 4.1 Correct the memory model

Your stated allocations total:

\\[ 2.2+0.3+0.35+0.4=3.25\text{ GB} \\]

That leaves only 0.75 GB of a nominal 4 GB GPU budget. In practice, that is not a safe operating margin.

Those estimates also may not include CUDA context overhead, temporary inference buffers, model-specific workspaces, allocator fragmentation, and vision preprocessing. The operating system and display may consume GPU memory too.

Use measured peak allocations, not weight sizes, to establish admission limits.

| Component        | Initial deployment policy                                                       |
| ---------------- | ------------------------------------------------------------------------------- |
| Ollama 3B Q4_K_M | Primary GPU inference process; control context size and concurrency             |
| YOLOv8n          | GPU when permitted by the scheduler; lower resolution or CPU fallback otherwise |
| Faster-Whisper   | CPU initially, with GPU execution only when budget allows                       |
| Kokoro-82M       | CPU as the reliable baseline; GPU execution only after benchmarking             |
| Piper            | CPU fallback for degraded operation                                             |
| Avatar rendering | Avoid unnecessary GPU-heavy postprocessing and high-resolution effects          |

These are starting policies, not claims that CPU execution will meet your final latency target.

### 4.2 Implement a GPU Resource Manager

Create a dedicated module such as `runtime/resource_manager.py`.

Responsibilities:

1. Query actual GPU memory usage and available capacity.
2. Maintain reservations for inference and temporary allocations.
3. Grant or deny GPU execution based on priorities and measured peak memory.
4. Serialize operations that compete for scarce GPU capacity.
5. Demote lower-priority workloads to CPU, reduced resolution, or deferred execution.
6. Detect allocation failures and recover without entering an infinite retry loop.

Recommended priority order:

1. Active conversational response.
2. Speech synthesis for the current response.
3. Voice transcription.
4. User-facing face detection and gaze tracking.
5. Background object detection, indexing, and other maintenance.

Speech synthesis should not automatically outrank active transcription during an interruption. The scheduler must respond to the current turn state.

### 4.3 Use a state-driven resource policy

| Runtime state | LLM                 | Vision       | STT                    | TTS              |
| ------------- | ------------------- | ------------ | ---------------------- | ---------------- |
| Listening     | Ready               | Reduced rate | Active                 | Idle             |
| Thinking      | Active              | Low-rate     | Available for barge-in | Idle             |
| Speaking      | Streaming or paused | Low-rate     | VAD active             | Active           |
| Interrupted   | Cancel generation   | Low-rate     | Active                 | Stop immediately |
| Idle          | Unloaded or warm    | Reduced rate | Wake detection         | Idle             |

Do not unload and reload the LLM on every turn. If RAM permits, keep its weights resident while controlling its GPU layers and other workloads. Repeated model reloads can destroy conversational latency.

### 4.4 Define explicit resource budgets

- Establish a configurable VRAM admission ceiling below the observed OOM boundary.
- Reserve headroom for the display driver, CUDA context, and transient allocations.
- Measure `nvidia-smi` usage during simultaneous inference, not only isolated benchmarks.
- Bound concurrent inference jobs and queue lengths.
- Add hysteresis to model unloading so small fluctuations do not trigger repeated loading and unloading.
- Expose resource state in the HUD: `GPU_AVAILABLE`, `GPU_RESERVED`, `GPU_PRESSURE`, and `CPU_FALLBACK`.

Acceptance criteria: 30 minutes of sustained mixed workload, repeated barge-in, and vision activity without an OOM crash, deadlock, or unbounded queue growth.

## 5. Phase 2 — Latency engineering and natural conversation

### First, define the latency correctly

Your target of less than 400 ms needs two separate metrics:

- TTFT: time from the LLM request becoming ready to the first generated token.
- Voice-to-ear: time from the end of the user's utterance to the first audible response sample.

A 400 ms voice-to-ear target is an aggressive optimization goal, not a guaranteed outcome. STT finalization, LLM generation, text buffering, and TTS startup all contribute. A long or complex request will often exceed it.

Measure two paths independently: streaming transcription during speech and response generation after the end-of-turn decision.

### 5.1 Suggested latency budget

Initial engineering targets — to be validated on the actual machine

VAD and endpoint detection

Post-utterance finalization

### 50–100 ms

LLM first token

Warm model, short prompt

### 50–150 ms

First TTS chunk

Short text segment

### 80–180 ms

These are provisional component targets, not benchmark results. They do not fit a worst-case 400 ms budget when added at their upper bounds; profile and optimize the critical path.

### 5.2 Streaming pipeline

Implement the following flow:

1. Silero VAD detects speech onset and continuously feeds audio frames to STT.
2. Faster-Whisper produces partial hypotheses where supported by the integration.
3. The turn manager detects likely end-of-turn, with configurable silence thresholds and a maximum wait.
4. The LLM begins generation as soon as the finalized input is available.
5. A sentence or clause buffer sends stable text to Kokoro.
6. Kokoro emits PCM chunks directly into a bounded audio queue.
7. The audio player schedules chunks without blocking the event loop.
8. The avatar consumes the same turn ID and phoneme timing metadata for lip-sync.

Optimize first-audio latency separately from complete-response latency.

### 5.3 Barge-in: a mandatory state machine

Use explicit states:

`IDLE → LISTENING → THINKING → SPEAKING → INTERRUPTED → LISTENING`

The implementation must:

- Keep VAD active while TTS is speaking.
- Detect speech onset using hysteresis and noise-aware thresholds.
- Immediately stop scheduling new TTS chunks.
- Flush queued PCM audio and stop or cancel playback.
- Cancel the current LLM generation using the runtime's supported cancellation mechanism.
- Invalidate late chunks from the cancelled turn using a generation ID.
- Preserve the user's new utterance rather than losing it during cancellation.
- Prevent the avatar from continuing to lip-sync to stale audio.
- Resume listening without restarting the backend.

Critical test: the user interrupts during a sentence, and the system becomes silent promptly without playing stale audio or responding to the previous turn.

### 5.4 Micro-gestures and gaze

- Drive breathing, blinking, and subtle head movement from a deterministic animation controller.
- Use filtered face landmarks or gaze estimates with temporal smoothing.
- Introduce gaze-lock confidence thresholds and graceful fallback when the face disappears.
- Decouple animation updates from inference and audio threads.
- Synchronize visemes to actual playback timestamps rather than text-generation timestamps.
- Avoid repetitive blinking, rigid staring, and periodic movements that look mechanical.

Keep the existing Three.js avatar and wardrobe work intact. Improve the animation controller rather than rebuilding the appearance pipeline.

## 6. Phase 3 — Zero-configuration packaging

Recommendation: Tauri + a managed Python sidecar + on-demand model downloads.

Do not attempt to cram the application, every model, all runtimes, and all platform dependencies into one giant binary. Ship a small installer and manage large model assets separately.

### 6.1 Packaging strategy

| Component            | Distribution approach                                                |
| -------------------- | -------------------------------------------------------------------- |
| Desktop shell        | Tauri application bundle                                             |
| Frontend             | Bundled HTML, JavaScript, CSS and Three.js assets                    |
| Python backend       | Self-contained executable, with native libraries packaged and tested |
| Ollama               | Managed installation or a separately distributed compatible runtime  |
| Whisper / Kokoro     | Bundled lightweight assets or verified downloads                     |
| YOLO weights         | Download only when vision is enabled                                 |
| SQLite / vector data | User-data directory, never the application installation directory    |
| Models and caches    | Dedicated model directory with versioned manifests                   |

Use Linux AppImage for a portable Linux distribution where supported, plus a native package such as `.deb` where useful. Tauri can also target Windows and macOS, but each platform needs its own builds and testing.

### 6.2 First-run experience

1\. Install

Download and launch the application without a manually configured Python environment.

2\. Hardware check

Detect CPU, GPU, RAM, storage, audio devices, and camera availability.

3\. Download models

Show required disk space, license, checksum, progress, and cancellation controls.

4\. Verify and launch

Verify model hashes, run a local smoke test, and start the companion.

### 6.3 Model download manager

- Maintain a signed manifest containing model ID, revision, expected SHA-256 hash, size, license, and supported runtime versions.
- Download to a temporary file and verify integrity before atomic promotion.
- Support resumable downloads, retry limits, cancellation, and free-space checks.
- Prevent duplicate downloads through a process-safe lock.
- Keep previous known-good model versions until the replacement passes its smoke test.
- Display model provenance and licenses in the UI.
- Never execute downloaded model assets as code.

For Ollama, explicitly test how the chosen installation mode locates and manages its model store. Avoid downloading a second copy of model weights unnecessarily.

### 6.4 Security and lifecycle

- Bind the backend only to loopback when a local network socket is required.
- Authenticate local clients with an unpredictable session credential.
- Restrict WebSocket origins and validate messages against typed schemas.
- Start the backend as a child process with bounded shutdown and restart behavior.
- Ensure application updates cannot overwrite user data or encryption keys.
- Sign releases and verify updates before installation.

Acceptance criteria: a clean machine can install the app, acquire the minimum models, pass a microphone test, and complete its first conversation without manual Python, CUDA, or model-path configuration.

## 7. Phase 4 — Local-first persistence and privacy

The most important design decision here is to make SQLite the canonical record store. ChromaDB should be treated as a rebuildable search index, not an independent source of truth.

### 7.1 Data architecture

Application domain layer

Conversations · Preferences · Entities · Tool outcomes

SQLite — canonical source of truth

Transactions · Schema migrations · Append-only audit events where appropriate

Vector index

Embeddings and semantic retrieval

Backup engine

Encrypted snapshots and restore validation

### 7.2 SQLite and vector storage tasks

- Introduce explicit schema versions and tested migrations.
- Use transactions for each logical conversation turn.
- Store timestamps, stable record IDs, and conversation ordering.
- Use foreign keys and appropriate indexes.
- Define deletion semantics for conversations, memories, and derived embeddings.
- Add an indexing queue so embedding failures cannot corrupt canonical history.
- Rebuild ChromaDB from canonical records after index corruption.
- Test database integrity after forced termination and disk-full scenarios.

### 7.3 Encryption and key management

- Encrypt sensitive records at rest using a vetted authenticated-encryption implementation.
- Consider SQLCipher for SQLite and independently encrypted vector-store records or an encrypted storage layer.
- Store keys in the operating system's credential store where available.
- Never hardcode encryption keys in the source code or package them alongside encrypted data.
- Define what happens when the user loses the key: recovery, export, or unrecoverable encrypted data must be explicit.
- Keep diagnostic logs free of raw conversation text, audio, images, credentials, and decrypted memory content by default.

Encryption of the database alone is not enough if the vector index, temporary audio, crash dumps, or exported logs contain the same sensitive information.

### 7.4 CRDTs and end-to-end encrypted backup

Do not introduce CRDTs into the single-device MVP without a multi-device synchronization requirement. SQLite transactions are simpler and sufficient for local persistence.

When synchronization becomes necessary:

- Define conflict semantics for mutable preferences, conversation history, and deletions.
- Use stable IDs and versioned operations.
- Select a CRDT strategy only for data types that benefit from concurrent edits.
- Build an encrypted backup format with authenticated metadata, ciphertext, and versioned manifests.
- Encrypt on the client before upload, with keys unavailable to the storage provider.
- Test restore on a clean installation and verify database integrity after decryption.
- Implement deletion propagation and backup retention policies.

For a strictly local-first release, remote backup should be optional and disabled until the user enables it.

### 7.5 Define a zero-telemetry contract

- No analytics SDK or remote telemetry by default.
- No external model calls unless the user explicitly enables a cloud provider.
- No silent camera or microphone uploads.
- No DNS or outbound requests required for ordinary local inference after installation.
- Audit outbound connections and third-party libraries.
- Provide a visible network activity indicator and a testable offline mode.

Acceptance criteria: the application works without internet after installation, sensitive data remains encrypted at rest, and automated network tests detect unexpected external traffic.

## 8. Phase 5 — Production hardening and release engineering

This phase turns a working prototype into software that can survive real consumer usage.

### 8.1 Reliability checklist

- Implement a supervisor with health checks, bounded restarts, and graceful shutdown.
- Add timeouts and cancellation to every model, tool, network, and database operation.
- Use bounded queues with explicit backpressure; never buffer unlimited audio or video.
- Define recovery behavior for GPU OOM, model crashes, audio-device removal, camera loss, and database lock contention.
- Add a circuit breaker for repeatedly failing engines.
- Test upgrades from previous application versions, including database migrations.
- Add crash reporting that is local by default and redacts sensitive information.
- Create reproducible release builds, software bills of materials, and dependency vulnerability checks.
- Test uninstall and reinstall without accidental deletion of user data.

### 8.2 Automated test matrix

| Test                                   | Required outcome                                    |
| -------------------------------------- | --------------------------------------------------- |
| 30-minute mixed workload               | No OOM, deadlocks, or unbounded queue growth        |
| 100 consecutive conversation turns     | No progressive resource leak or stale-turn response |
| 50 interruption events                 | No stale PCM playback or cancelled-turn leakage     |
| Kill LLM process during generation     | Session recovers without application restart        |
| Disconnect microphone during speech    | Clear error and recoverable audio state             |
| Corrupt model download                 | Verification rejects the asset                      |
| Kill application during database write | Database integrity preserved                        |
| Disconnect internet after installation | Local conversation continues to work                |
| Restore encrypted backup               | Records and vector index recover correctly          |

These are proposed acceptance tests, not claims that the current implementation already passes them.

## 9. Phase 6 — UX, avatar realism, and adaptive intelligence

Only after runtime stability should you spend substantial engineering time on expressive behavior.

### Avatar and sensory pipeline

- Separate perception, attention, emotion/intent, animation, and rendering into distinct modules.
- Drive gestures from conversational intent and state rather than random animation triggers.
- Blend gaze targets smoothly between the user's face, the camera, and natural idle directions.
- Introduce confidence-aware gaze tracking, with head-pose fallback when eye-gaze estimation is unreliable.
- Schedule object detection at an adaptive frame rate based on resource pressure.
- Keep the avatar responsive even when the LLM or vision engine is busy.
- Add a calibration screen for camera placement, microphone gain, and audio output.

### Context and tool execution

Build on your existing Robot context layer and tool execution contracts:

- Use an explicit policy for which memories are retrieved, persisted, summarized, or forgotten.
- Separate untrusted observations from trusted system instructions.
- Require schema validation and authorization for every tool invocation.
- Require confirmation before consequential actions, including destructive file operations.
- Record tool outcomes separately from the model's claims about those outcomes.
- Add synthetic tests for prompt injection through camera observations, retrieved memories, and tool output.
- Test memory relevance and hallucination rates using a fixed evaluation dataset.

Your companion should be able to explain what it knows, distinguish an observed fact from an inference, and recover gracefully when it is uncertain.

## 10. Prioritized implementation schedule

This is a proposed engineering schedule for one primary developer, assuming the current application remains functional. Some workstreams can run in parallel.

Week 1

Stabilize and measure

Resource profiling, trace IDs, health checks, concurrency limits, crash recovery, and baseline tests.

Deliverable: repeatable local benchmark suite.

Week 2

Real-time interaction

Streaming STT, chunked TTS, cancellation, barge-in, bounded audio queues, and latency instrumentation.

Deliverable: reliable interruptible conversation loop.

Week 3

Desktop packaging

Tauri integration, managed backend, first-run setup, model manager, installer, and offline smoke tests.

Deliverable: clean-machine Linux installation.

Week 4

Privacy and persistence

Database migrations, encryption, deletion, backup and restore, and outbound-network audit.

Deliverable: verified local-first data lifecycle.

Weeks 5–6

Release candidate

Failure injection, security testing, gaze and animation refinement, performance tuning, and release packaging.

Deliverable: a reproducible release candidate with documented limitations.

Allow additional time for Windows and macOS support, code signing, installer edge cases, and testing on hardware other than your development machine.

## 11. Production KPIs

Instrument these from the first week. Do not declare production readiness solely because unit tests pass.

| Metric                                   | Initial target                                      |
| ---------------------------------------- | --------------------------------------------------- |
| Warm LLM TTFT                            | p95 ≤ 400 ms, where feasible                        |
| End-of-utterance to first audible sample | p50 ≤ 400 ms; optimize p95 separately               |
| Barge-in to silence                      | p95 ≤ 150 ms                                        |
| Unhandled application crashes            | Zero in the defined stress-test suite               |
| Memory growth                            | No sustained unbounded growth during long-run tests |
| Offline operation                        | No required network calls after model setup         |
| Database recovery                        | Integrity preserved across forced termination tests |
| Installation                             | No manual Python or CUDA setup                      |
| Model integrity                          | All downloaded assets verified before use           |

Treat these as engineering targets, not established performance guarantees. The 400 ms voice-to-ear objective may require carefully chosen short responses, aggressive streaming, warm models, and reduced contention. Measure the actual end-to-end path on the RTX 3050 before committing to a public SLA.

## 12. What not to do

These decisions will save you considerable engineering time.

- Don't load all four AI workloads onto the GPU simultaneously just because their estimated model sizes add up to less than 4 GB.
- Don't unload the LLM after every response or reload it on every interruption.
- Don't block FastAPI's event loop with synchronous model inference or audio playback.
- Don't add distributed CRDT synchronization before the local data model is stable.
- Don't embed multi-gigabyte model weights in every application installer.
- Don't assume a bundled Python executable removes native-library, GPU-driver, or audio-system compatibility issues.
- Don't let background vision or avatar rendering starve conversational inference.
- Don't spend another major sprint on avatar appearance before completing resource management and interruption tests.

## 13. Your immediate engineering sprint

Given the Robot V2 foundations and regression-testing work you've already completed, I would start with these five tasks in this exact order.&#x20;

## Sprint execution checklist

0 of 5

Build a repeatable GPU/CPU memory and latency benchmark.

Implement a central GPU admission controller and bounded inference queues.

Finish the turn-state machine, including immediate audio cancellation and barge-in.

Add end-to-end trace IDs and automated fault-injection tests.

Package the Linux application with a managed backend and verified model downloads.

&#x20; Copy sprint tasks

My final architectural recommendation: stabilize the local runtime first, make speech genuinely interruptible second, and package the proven system third. Privacy and recovery must be built into each stage, not added as a final feature.

That sequence gives Robot the best chance of becoming a dependable consumer companion rather than an impressive demo that only works on its developer's machine.