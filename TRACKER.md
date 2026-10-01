# Feature, ToDo & Issue Tracker

> **Notice for AI Agents & Developers**:
> This document is the single source of truth for all architectural features, implementation statuses, ongoing tasks, known issues, and future roadmaps across the robot intelligence platform.
> Whenever you implement features, fix bugs, or identify new requirements, **you must update this file immediately** and run `.venv/bin/graphify update .` to keep the codebase knowledge graph synchronized.

---

## 1. System Overview & Architecture Snapshot

The robot runs a heterogeneous, real-time distributed architecture designed for low-latency (<800ms) full-duplex conversational and embodied interaction:
- **LLM**: Local `llama-server` running `Qwen3.5-4B-Q4_K_M` (`-ngl 99 -fa on -c 2048`) offloaded 100% to **NVIDIA GeForce RTX 3050 Mobile** (0% system RAM).
- **STT**: `whisper.cpp` (`whisper-server`) offloaded to **AMD Radeon 680M iGPU** via Vulkan (`-dev 0 -fa`).
- **TTS**: `piper-tts` (`en-us-lessac-medium`) running on **AMD Ryzen 7 6800H CPU** via AVX2 / ONNX with sentence/clause streaming.
- **Vision & Perception**: OpenCV YuNet (Face) + YOLOv8n (Object detection @ 320x240) feeding into real-time `WorldState`.
- **Memory**: Mem0 persistent semantic memory with Qdrant vector store and `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dims) local ONNX embeddings + SQLite conversation archive.
- **Full-Duplex Audio & Barge-In**: Silero VAD (tuned 320ms pause) + self-speech suppression filter.
- **Avatars**: Three.js WebGL2 3D VRM character with 17 canonical bones, procedural gestures, and 4Hz Look-At face tracking.

---

## 2. Feature Implementation Status Matrix

| Feature Module | Component / Implementation | Status | Device / Hardware | Verification / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **LLM Inference** | `llama-server` + OpenAI-compatible API client | ✅ **Active** | NVIDIA RTX 3050 (4GB) | FlashAttention enabled; prompt eval ~138 tok/s, generation ~47 tok/s. |
| **STT Engine** | `whisper-server` HTTP stream | ✅ **Active** | AMD Radeon 680M (Vulkan) | Offloaded to Vulkan device 0 (`RADV REMBRANDT`); ~210–280ms latency. |
| **TTS Engine** | Piper ONNX + ClauseChunker | ✅ **Active** | CPU (AVX2) | Streaming phrase synthesis; first chunk ready in 69–95ms. |
| **Barge-in / Echo Suppression** | Silero VAD + `EchoGuard` RMS subtraction | ✅ **Active** | Host CPU | Dynamic RMS subtraction prevents robot hearing its own voice. |
| **Real-time Face Detection** | OpenCV DNN `FaceDetectorYN` (YuNet) | ✅ **Active** | Host CPU | Runs @ 320x240, ~17ms per frame (58 FPS). |
| **Object Detection** | Ultralytics YOLOv8n | ✅ **Active** | Host CPU | Runs @ 320x240, ~35ms per frame (28 FPS). |
| **Visual Grounding** | Perception-to-LLM Prompt Injector | ✅ **Active** | Host CPU | Bypasses multimodal mmproj missing error; injects `[Visual Stream]`. |
| **Internet Web Search** | Ultra-Fast DuckDuckGo Lite + Instant Answer | ✅ **Active** | Network / Thread Pool | Direct HTML parser (~500ms–1.5s vs 13s `ddgs`); plays voice ack first. |
| **Instant Spoken Acknowledgment**| TTS queue injection before search | ✅ **Active** | Core Engine | Speaks *"Searching..."* or *"Checking the web..."* before HTTP query. |
| **System Time Tool** | `system.time` built-in tool | ✅ **Active** | Local OS Clock | Returns exact local time, date, weekday, timezone in <1ms without web. |
| **Personality Switching** | Neo, Aria, GLaDOS YAML definitions | ✅ **Active** | Core Engine | Hot-swappable via tool or UI without server restart. |
| **Mem0 Persistent Memory** | Mem0 + Qdrant + FastEmbed | ✅ **Active** | Local ONNX (CPU) | Uses `BAAI/bge-small-en-v1.5` (384-dim); no 501 llama-server error. |
| **3D Avatar & Web HUD** | Three.js + VRM Humanoid Rig | ✅ **Active** | AMD Radeon 680M (Browser) | 60 FPS WebGL2 rendering, procedural breathing, blinking, and gaze. |
| **Physical Serial Motion** | `robot.motion.serial_bridge` | 🟡 **Emulated** | USB Serial / Microcontroller | Mock fallback active when physical hardware disconnected. |
| **Real-Time Observability** | `ObservabilityHub` (WebSockets) | ✅ **Active** | Core Engine | Live latency metrics, span hierarchy, pipeline flow visualization in HUD. |
| **Async Image Generation** | `ImageGenerationService` (Local SD) | ✅ **Active** | CPU Fallback (OFA-Sys SD v0) | Generates images natively via Hugging Face diffusers avoiding RTX 3050 blocking. |

Legend:
- ✅ **Active**: Fully implemented, tested, and passing all regression suites.
- 🟡 **Emulated / Partial**: Functional in emulation or fallback mode.
- 🔴 **Planned / Not Started**: Backlog item.

---

## 3. Active ToDos (Work In Progress & Prioritized Backlog)

### 🔴 High Priority
- [ ] **Physical Hardware Integration**:
  - Test and calibrate real physical servos over serial (`/dev/ttyUSB0` or `/dev/ttyACM0`) using `src/robot/motion/serial_bridge.py`.
  - Add failsafe torque disconnect on unexpected stall current.
- [ ] **Multi-turn Context Pruning**:
  - Implement rolling context budget trimmer in `ContextEngine` to ensure token budget never exceeds 2048 under prolonged conversations.
- [ ] **Multi-Modal Vision Encoder (`mmproj`)**:
  - Add optional local CLIP / MobileCLIP vision projector to `llama-server` when full visual question answering is demanded directly by the weights.

### 🟡 Medium Priority
- [ ] **Custom Wake-Word Detection**:
  - Add local wake-word engine (e.g. openWakeWord) preceding Silero VAD to allow hands-free wake-up (e.g., *"Hey Neo"*).
- [ ] **Voice Cloning / Multiple Piper Voices**:
  - Assign unique Piper ONNX voices to each personality (e.g., dedicated voice models for Neo, Aria, and GLaDOS).
- [ ] **Web Search Cache**:
  - Add short-lived in-memory LRU cache (TTL: 10 minutes) for repeated search queries.

### 🟢 Low Priority / Polishing
- [ ] **Automated Benchmark Runner**:
  - Create a single benchmark script `scripts/benchmark_e2e.py` that automatically records and plots TTFT, STT latency, and TTS RTF over 50 automated audio runs.
- [ ] **Docker / Containerization**:
  - Build a multi-device container configuration supporting both NVIDIA CUDA runtime and AMD ROCm/Vulkan drivers.

---

## 4. Known Issues & Resolved Bugs Tracker

### 🐛 Resolved Issues
1. **TTS Output Audio Slowed / Robotic**:
   - *Cause*: Resampling artifact from mismatch between Piper's native 22050Hz output and ALSA's default 44100/48000Hz device clock.
   - *Fix*: Locked `sample_rate: 22050` across `sounddevice.OutputStream` and enforced uniform chunk boundaries.
2. **Mem0 Embeddings 501 Error**:
   - *Cause*: `llama-server` was passed as embedding endpoint, but it was running `Qwen3.5-4B` exclusively in completion mode without `--embeddings`.
   - *Fix*: Switched Mem0 embedder to `fastembed` with local ONNX model `BAAI/bge-small-en-v1.5` (384 dims).
3. **Web Search Sluggish (13.3s latency) & DDG Lite Anomaly Blocking**:
   - *Cause*: Python `ddgs` library incurred heavy anti-bot delays; subsequently, standalone DuckDuckGo Lite intermittently triggered challenge forms (`anomaly.js`) on certain tech/product queries, while queries with conversational prefixes ("search online for...") degraded search precision.
   - *Fix*:
     - Built an automatic natural language query cleaner stripping speech filler ("search for", "look up", "tell me about").
     - Upgraded `web.search` to a 4-tier multi-engine architecture:
       1. Primary: DuckDuckGo Lite with paired Title, Snippet, and URL extraction.
       2. Secondary: Bing News & Web RSS (ultra-fast, real-time news/specs, immune to bot walls).
       3. Tertiary: DuckDuckGo Instant Answer & RelatedTopics JSON API.
       4. Quaternary: `ddgs` library fallback.
     - Enriched grounded context injection into LLM prompt with source titles, summaries, and URLs so the LLM speaks factual, current data immediately.
4. **TTFT Latency Increase (Multi-slot Cache Invalidation & Dynamic Header Jitter)**:
   - *Cause*: `llama-server` ran with default 4 parallel slots (`n_slots=4`) cycling incoming queries across different slots and invalidating cached KV prefixes. In addition, injecting dynamic tool blocks into the system prompt caused full re-evaluation of prompt tokens on every turn.
   - *Fix*:
     - Configured `llama-server` in [`scripts/run.sh`](file:///home/ameya/robot/scripts/run.sh) with single-slot continuous mode (`-np 1 -b 2048 -ub 2048`), dedicating 100% GPU compute and cache to the live conversation.
     - Preserved static personality prompt prefix at line 1 of messages, and moved dynamic turn grounding directly into the current turn.
     - Reset stale 768-dim Qdrant store from disk to eliminate embedding shape mismatch errors (`shapes (0,768) and (384,) not aligned`).
     - *Result*: TTFT dropped from ~580ms down to **167–183ms**.
5. **Self-Interruption / Feedback Loop**:
   - *Cause*: Microphone picking up robot's own speaker output and triggering Silero VAD.
   - *Fix*: Implemented dynamic energy subtraction via `EchoGuard` and temporarily suppressed user transcription during high-volume playback.
6. **Slow Object Detection (181ms)**:
   - *Cause*: YOLOv8n evaluating high-resolution (640x480) camera frames on CPU.
   - *Fix*: Downscaled inference resolution to 320x240, yielding 35ms inference (28.5 FPS).
7. **STT-LLM-TTS Turnaround Latency (First-Phrase Delay & STT Model Overhead)**:
   - *Cause*: `ClauseChunker` waited for full sentence termination (`.!?`) or 24 words before dispatching text to Piper, causing 400–500ms dead air on long opening sentences. In addition, `scripts/run.sh` prioritized `ggml-base.bin` (~240ms) and ran unquantized KV cache.
   - *Fix*:
     - Implemented eager first-clause boundary in [`ClauseChunker`](file:///home/ameya/robot/src/robot/llm/__init__.py) (`eager_first_chunk=True, min_first_words=2`), emitting conversational opening clauses immediately to Piper TTS (saving ~300–350ms TTFA).
     - Prioritized `ggml-tiny.en.bin` on AMD Radeon 680M Vulkan backend in [`scripts/run.sh`](file:///home/ameya/robot/scripts/run.sh), cutting STT latency from ~240ms to ~70ms.
     - Enabled quantized KV cache (`-ctk q8_0 -ctv q8_0`) in `llama-server` for reduced memory bandwidth load on the RTX 3050 Mobile.

8. **PortAudio ALSA Segfault during Barge-In Interrupt**:
   - *Cause*: Calling `stream.abort()` and `stream.close()` from the main event loop thread while a worker thread in `asyncio.to_thread` was actively inside `stream.write()` corrupted PortAudio's ALSA mmap buffer pointers (`alsa_snd_pcm_mmap_begin failed`), triggering a C-level segmentation fault.
   - *Fix*: Added `threading.Lock` across stream access in [`AudioPlayer`](file:///home/ameya/robot/src/robot/audio/playback.py); made `stop()` pause the stream non-destructively without tearing down active ALSA file descriptors.
9. **Mem0 Top-Level Entity & 8k Extraction Token Exhaustion**:
   - *Cause*: Mem0 deprecated top-level `user_id` in `search()` and `get_all()`, requiring `filters={'user_id': '...'}`. Additionally, `Mem0.add()` with default `infer=True` issued an 8,469-token prompt to `llama-server`, exceeding the 2048-token context window and causing HTTP 400 errors.
   - *Fix*: Updated [`Mem0Provider`](file:///home/ameya/robot/src/robot/memory/__init__.py) to use `filters={"user_id": user_id}` and `infer=False` for zero-overhead direct vector storage without LLM token explosion.
10. **`SpanStatus` NameError in `_mic_loop`**:
    - *Cause*: `SpanStatus` was imported locally inside `RobotEngine.__init__`, causing `NameError: name 'SpanStatus' is not defined` when an error path or echo cancellation path in `_mic_loop` attempted to invoke `self.observability.finish_span(stt_span, status=SpanStatus.ERROR)`.
    - *Fix*: Moved `ObservabilityHub` and `SpanStatus` imports to the top level of [`engine.py`](file:///home/ameya/robot/src/robot/core/engine.py#L31-L32).
11. **Mem0 Dict Return Format & AttributeError (`'str' object has no attribute 'get'`)**:
    - *Cause*: In Mem0 v1.1+, `Memory.search()` and `Memory.get_all()` return `{"results": [...]}` rather than a raw `list`. When iterating over the returned dictionary keys (`"results"` string), `m.get("memory", ...)` raised `AttributeError: 'str' object has no attribute 'get'`.
    - *Fix*: Normalized return values in [`Mem0Provider.search`](file:///home/ameya/robot/src/robot/memory/__init__.py#L119) and [`Mem0Provider.get_all`](file:///home/ameya/robot/src/robot/memory/__init__.py#L141) to cleanly unpack `raw.get("results", [])`, coerce any strings to `{"memory": ...}`, and added defensive type-checking in [`MemoryManager.get_relevant_memories`](file:///home/ameya/robot/src/robot/memory/__init__.py#L432) and [`_format_memories`](file:///home/ameya/robot/src/robot/core/context.py#L151).
12. **Default Tool Schema KV-Cache Invalidation (TTFT ~845ms)**:
    - *Cause*: [`ToolRouter.select_tools`](file:///home/ameya/robot/src/robot/tools/__init__.py#L106) defaulted to returning `{"memory", "system"}` families (4+ tools) on *every single conversational turn*. This injected ~450 extra JSON schema tokens into the LLM prompt on basic greetings and dialogue, continuously evicting the continuous KV cache prefix and driving prompt eval time from ~140ms to ~720ms (TTFT ~845ms).
    - *Fix*: Changed `ToolRouter.select_tools` to return an empty tool list for standard conversational queries, only injecting tool definitions when intent keywords explicitly match.
13. **Proactive LLM KV-Cache Warm-Up & Silence Threshold Tuning**:
    - *Optimization*: Added automatic pre-warming of the active personality's system prompt in [`RobotEngine.setup`](file:///home/ameya/robot/src/robot/core/engine.py#L239) and in [`scripts/run.sh`](file:///home/ameya/robot/scripts/run.sh#L52). Tuned `silence_threshold_ms` from 190ms to 150ms in [`config/default.yaml`](file:///home/ameya/robot/config/default.yaml#L26) to reduce turn-around latency.
14. **System Prompt Prefix Stabilization & History Compaction (<300ms TTFT Guarantee)**:
    - *Cause*: Mutating the `role: system` message on every turn with dynamic perception / facts constantly broke `llama-server`'s prefix KV-cache (`f_keep` fell to 0.23), and fetching 10 past conversation turns ballooned context to 900+ tokens, forcing 600–790ms prompt evaluations.
    - *Fix*: Kept `role: system` 100% invariant across turns during live conversation in [`ContextEngine`](file:///home/ameya/robot/src/robot/core/context.py#L70-L97), grounding dynamic facts/world state in the current turn. Compacting conversation history retrieval to the last 3 turns in [`RobotEngine`](file:///home/ameya/robot/src/robot/core/engine.py#L565) and tightening `context_budget` in [`config/default.yaml`](file:///home/ameya/robot/config/default.yaml#L84-L91) guarantees prompt token count remains under 250 tokens and slot reuse remains >90%, achieving stable **<200ms TTFT**.

### ⚠️ Open Watchlist
1. **Physical Servos Closed-Loop Feedback**:
   - *Symptom*: Hardware serial bridge is in emulation mode when physical USB microcontrollers are unattached.
   - *Mitigation*: Emulation layer responds cleanly and keeps VRM character in sync.
2. **Local Aider & llama-server Integration**:
   - *Documentation*: Added [`AIDER_HELP.md`](file:///home/ameya/robot/AIDER_HELP.md) covering `--chat-template chatml`, `--alias gpt-4o`, context window sizing (`-c 16384`), and `--map-tokens 0`.

---

## 5. Instructions for AI Agents Updating This Tracker

When you perform work on this repository:
1. **Consult this tracker** before designing new systems to understand existing capabilities and dependencies.
2. **Update this document** whenever you:
   - Implement a new feature or tool.
   - Fix a bug or performance bottleneck.
   - Add, complete, or re-prioritize a ToDo item.
3. **Run knowledge graph update**:
   ```bash
   .venv/bin/graphify update .
   ```
4. **Verify tests pass**:
   ```bash
   .venv/bin/pytest tests/
   ```
