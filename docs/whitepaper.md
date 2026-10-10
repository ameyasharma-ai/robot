# 🤖 Humanoid Robot --- Complete Technical Architecture

**Project:** Full-Stack Local-First Humanoid Robot\
**Architecture Version:** v1.0\
**Development OS:** Windows 11\
**Final Robot OS:** Linux Mint\
**Status:** Architecture Baseline --- Implementation Next\
**Design:** Local-first · Real-time · Multimodal · Agentic · Modular ·
Safety-gated

------------------------------------------------------------------------

## 1. Mission

Build a humanoid/companion robot whose laptop-class computer acts as the
first complete digital body and brain. The system must support natural
full-duplex conversation, persistent memory, continuous perception,
visual reasoning, personalities, tools, smart-home and browser control,
expressive screen/RGB behavior, and eventually safe physical actuation
and controlled self-development.

The central rule is:

> **The LLM reasons; Robot Core owns the robot.**

Qwen never directly owns memory, hardware, permissions, the filesystem,
motors, or deployment.

------------------------------------------------------------------------

## 2. Available Compute

  -----------------------------------------------------------------------
  Resource                Hardware                Primary Role
  ----------------------- ----------------------- -----------------------
  CPU                     AMD Ryzen 7 6800H,      Robot Core, CV, audio
                          8C/16T                  preprocessing, services

  Discrete GPU            NVIDIA RTX 3050 Laptop, Qwen language +
                          4 GB VRAM               on-demand visual
                                                  reasoning

  Integrated GPU          Radeon 680M, RDNA2, 12  Whisper/Piper
                          CUs                     acceleration where
                                                  benchmarks justify it

  RAM                     16 GB DDR5              OS, shared iGPU memory,
                                                  databases, services,
                                                  model spill

  Storage                 1 TB internal SSD       OS, models, source,
                                                  memory, logs, datasets

  Display                 1080p laptop display    Robot face + optional
                                                  developer UI

  Camera                  Integrated webcam       Continuous perception /
                                                  visual reasoning
                                                  snapshots

  Audio                   Integrated mic +        Full-duplex voice
                          speakers                

  Expression              4-zone ROG keyboard +   Nonverbal robot
                          lightbar/strip          state/expression
  -----------------------------------------------------------------------

### Resource policy

The RTX 3050 is reserved primarily for the main Qwen model. Continuous
perception must not steal large amounts of RTX VRAM. The Radeon 680M is
a secondary accelerator, not assumed to be faster until measured.
Expensive tasks such as browser automation, coding, VLM analysis, and
builds are scheduled rather than allowed to contend indefinitely.

------------------------------------------------------------------------

## 3. Top-Level Architecture

``` text
Human
 │
 ├──────────── Voice ─────────────┐
 │                                ▼
 │                         Audio Front End
 │                                │
 │                         VAD → Whisper
 │                                │
 │                                ▼
 │                         ┌──────────────┐
 ├──── Camera ────────────►│  ROBOT CORE  │◄──── World State
 │                         │              │
 │                         │ Context      │
 │                         │ State        │
 │                         │ Tools        │
 │                         │ Permissions  │
 │                         │ Personality  │
 │                         └──────┬───────┘
 │                                │
 │                         llama-server
 │                                │
 │                         Qwen3.5-4B
 │                                │
 │             ┌──────────────────┼───────────────────┐
 │             ▼                  ▼                   ▼
 │           Tools              Memory            Expression
 │             │                  │                   │
 │      MCP/native providers    Mem0/SQLite       Face + RGB
 │                                                     │
 └──────────────────────────── Speakers ◄──── Piper ◄──┘
```

------------------------------------------------------------------------

## 4. Primary AI Brain

### Locked model

**Qwen3.5-4B Q4_K_M**

-   Primary language/reasoning model.
-   Also provides on-demand multimodal visual reasoning.
-   Runs locally.
-   RTX 3050 is the primary inference accelerator.
-   Normal context: **16K**.
-   Extended/experimental context: **32K**, subject to VRAM/latency
    benchmarks.

### Context is not memory

Long-term memories are never kept permanently inside the model context.
The Context Engine retrieves only relevant information.

Approximate 16K working budget:

  Context component                     Target
  ------------------------------- ------------
  System/personality                   1--1.5K
  Relevant tool schemas              1.5--2.5K
  Retrieved memories                 1.5--2.5K
  Recent conversation                    4--6K
  World/vision state                   0.5--1K
  Robot state                       0.25--0.5K
  User input                          Variable
  Generation/reasoning headroom          3--4K

Budgets are dynamic rather than hard partitions.

------------------------------------------------------------------------

## 5. LLM Runtime

### Production: llama.cpp / llama-server

``` text
Robot Core
   │ OpenAI-compatible HTTP/streaming
   ▼
llama-server
   │
   ├── Qwen3.5-4B Q4_K_M
   ├── multimodal projector
   ├── 16K/32K context
   └── CUDA
          ▼
      RTX 3050
```

Robot Core is not coupled to C++. `llama-server` is an independent
process.

### Development: LM Studio

LM Studio is the model laboratory for: - GGUF/model management -
quantization comparisons - prompt experiments - context experiments -
sampling changes - tool-call experiments

Robot Core uses a provider abstraction:

``` text
LLMProvider
 ├── LM Studio       [development]
 └── llama-server    [production]
```

Changing the base URL should not require Robot Core changes.

### Linux lifecycle

`systemd` starts and monitors `llama-server`. If it fails, the service
can restart without crashing Robot Core.

------------------------------------------------------------------------

## 6. Context Engine

For each inference turn, Robot Core assembles:

1.  Active personality prompt
2.  Core robot grounding policy & strict User Entity Ownership & Attribution Directive
3.  Current task & conversation history
4.  Relevant Mem0 user memories & durable personal facts
5.  Recent conversation (compacted for local mode KV cache invariance)
6.  Current user input & telemetry separation
7.  Relevant tool definitions only (routed via ToolRouter)
8.  Current system & hardware telemetry metrics
9.  Relevant tool execution outputs
10. Compact visual world state
11. Actual image snapshots only when multimodal visual reasoning is actively required

### Adaptive Dual-Tier Context Architecture
- **Local Mode (RTX 3050)**: Strict budget enforcement keeping prompt tokens under 2,200 tokens (3 recent turns, 5 memories). This maintains 100% KV-cache invariance on llama-server with Time-to-First-Token (TTFT) < 250ms and zero VRAM exhaustion.
- **Cloud Mode (Gemini 1M Context)**: Deep multi-session archive expansion (up to 100 recent turns, 50 memories, and complete durable facts up to 800,000 tokens) with multi-session cross-referencing.
- **Entity Ownership & Attribution Directive**: Grounding policy strictly assigns user preferences, relationships, and romantic interests (e.g. crushes, family) exclusively to the human user, ensuring the robot assistant never claims or confuses user entities as its own.

------------------------------------------------------------------------

## 7. Speech-to-Text

### Locked stack & Streaming Architecture

**Whisper Small Multilingual / Tiny + whisper.cpp + StreamingSTT**

Target accelerator: Radeon 680M through Vulkan (`-dev 0 -fa`); CPU fallback remains valid.

``` text
Microphone
  ↓
Audio Front End
  ↓
Silero VAD (Speech Active)
  ↓
StreamingSTT (Background Chunking @ ~2.0s with Energy-Dip Splitting)
  ↓
whisper.cpp parallel workers
  ↓
Partial Transcripts (HUD Live Stream) & Fast Tail Finalization (<50ms turnaround)
  ↓
Robot Core
```

Whisper Base/Tiny is the low-latency target for real-time speech, supported by parallel chunking that transcribes speech while the user is actively speaking.

------------------------------------------------------------------------

## 8. Full-Duplex Audio and Barge-In

### Audio processing

Use WebRTC-style audio processing for: - Acoustic echo cancellation -
Noise suppression - Automatic gain control where appropriate

``` text
Piper output ─────────────► Speaker
     │                         │
     └── reference audio ─┐    │ acoustic leakage
                          ▼    ▼
Mic ───────────────► Echo Cancellation
                          │
                    Noise Suppression
                          │
                       VAD
                          │
                       STT
```

### Barge-in behavior

``` text
Robot speaking
    ↓
User begins speaking
    ↓
VAD detects speech
    ↓
cancel TTS playback
    ↓
cancel current LLM generation
    ↓
STT new utterance
    ↓
build fresh context
    ↓
new response
```

Target conversational behavior: robot starts responding roughly within
1--2 seconds after the user finishes, subject to actual benchmarks.

------------------------------------------------------------------------

## 9. Text-to-Speech

### Locked stack

**Piper-compatible ONNX TTS**

Requirements: - Local - Low latency - Custom trainable/fine-tunable
voices - Multiple personality voices - CPU fallback - Radeon 680M
acceleration only where runtime support/benchmarks justify it

``` text
Qwen token stream
     ↓
Phrase/Sentence Chunker
     ↓
TTSService
     ↓
Piper-compatible ONNX
     ↓
Audio Buffer
     ↓
Speaker
```

Voice training can occur on a stronger/cloud GPU and be exported for
local inference.

Optional Phase II experimental/high-quality engine: **F5-TTS**.

------------------------------------------------------------------------

## 10. Vision

### Two-level design

Continuous perception stays lightweight. Qwen is invoked only when
semantic visual reasoning is needed.

``` text
Webcam
  ↓
OpenCV perception
  ↓
World State
  ↓
Robot Core
  │
  └── visual reasoning needed?
          ↓ yes
      capture selected frame
          ↓
      Qwen3.5 multimodal
          ↓
      semantic answer
```

### Continuous perception

OpenCV is the foundation.

Proposed lightweight components: - NanoDet-class lightweight object
detection - YuNet INT8 face detection - SFace face recognition -
Lightweight tracking - MP-Pose only when needed

Scheduling concept:

  Operation                     Approx cadence
  ----------------------- --------------------
  Camera acquisition                    30 FPS
  Tracking                            \~15 FPS
  Object detection                     \~5 FPS
  Face detection                       \~5 FPS
  Face recognition          New face / \~1 FPS
  Pose                               On demand
  Qwen visual reasoning              On demand

Actual cadence is adaptive based on CPU load.

### World State

Do not feed raw frames continuously to Qwen.

Example:

``` yaml
people:
  - id: user
    position: center
    looking_at_robot: true
objects:
  - phone
  - bottle
  - laptop
motion: low
scene: desk
updated_at: ...
```

Only significant events are promoted to persistent memory.

------------------------------------------------------------------------

## 11. Memory

### Memory principle

**The LLM does not own memory.**

Robot Core controls storage, retrieval, updates, deletion, merging,
namespaces, and context injection.

### Persistent memory: Mem0

Used for relevant durable facts, preferences, events, and context.

### Raw history: SQLite

Stores: - Complete conversations - Tool events - timestamps - sessions -
important system events

SQLite FTS supports exact/keyword history search.

### Phase II: Graphiti

Adds temporal knowledge relationships:

``` text
Person ──knows──► Person
User ──owns─────► Device
Object ─located─► Room
Preference ─changed_at─► Time
```

### Memory layers

``` text
Working State       → RAM / Robot Core
Recent Conversation → Context Engine
Persistent Memory   → Mem0
Raw Archive         → SQLite + FTS
Temporal Graph      → Graphiti [Phase II]
World State         → Perception subsystem
```

Memory consolidation can run during idle periods.

------------------------------------------------------------------------

## 12. Personality System

Personality is external to model weights.

Each personality can define:

``` yaml
name:
system_prompt:
speaking_style:
vocabulary:
behavior:
voice_model:
tts_settings:
face_profile:
rgb_profile:
tool_permissions:
memory_namespace:
proactivity:
```

The same Qwen model can switch personalities instantly.

**Personality = how the robot behaves.**\
**Skills = what the robot can do.**

------------------------------------------------------------------------

## 13. Robot Core

### Locked implementation

**Python 3.12 + asyncio**

Robot Core is the authoritative orchestration layer.

``` text
Robot Core
│
├── Context Engine
├── Conversation Manager
├── State Manager
├── World State
├── Personality Manager
├── Tool Registry
├── Tool Router
├── Permission Manager
├── Provider Manager
├── Resource Scheduler
├── Expression Controller
├── Memory Manager
├── Event Logger
├── Service Health Manager
└── Development Controller
```

### API

**FastAPI + Pydantic**

Use: - HTTP for request/response services - WebSocket for
streams/events - asyncio queues for in-process events - MCP for external
tool/provider integration

No Kafka/RabbitMQ/NATS initially. One laptop does not need
distributed-system infrastructure.

------------------------------------------------------------------------

## 14. Provider Abstraction

``` text
LLMProvider
SpeechProvider
TTSProvider
MemoryProvider
VisionProvider
SearchProvider
BrowserProvider
MusicProvider
HomeProvider
LightingProvider
DevelopmentProvider
SandboxProvider
```

Current mappings:

  Interface             Provider
  --------------------- --------------------------
  LLMProvider           llama-server / LM Studio
  SpeechProvider        whisper.cpp
  TTSProvider           Piper-compatible ONNX
  MemoryProvider        Mem0
  VisionProvider        OpenCV + Qwen multimodal
  SearchProvider        SearXNG
  BrowserProvider       Browser Use
  MusicProvider         Spotipy + local provider
  HomeProvider          Home Assistant
  DevelopmentProvider   Aider
  SandboxProvider       Podman → Firecracker
  LightingProvider      OpenRGB → asusd/asusctl

Providers can be replaced without redesigning Robot Core.

------------------------------------------------------------------------

## 15. Tool Architecture

Qwen proposes actions; Robot Core executes approved actions.

``` text
Qwen
 ↓ tool_call
llama-server
 ↓
Robot Core
 ↓
Tool Router
 ↓
Permission Manager
 ↓
Schema Validation
 ↓
Provider/Handler
 ↓
Result
 ↓
Robot Core
 ↓
Qwen
```

### Tool registration

Each tool declares: - name - description - JSON input schema - handler -
permission level - timeout - risk level - logging policy - provider
dependency

Example namespaces:

``` text
memory.search
memory.write
history.search
search.query
browser.task
music.play
music.pause
home.control
vision.inspect
face.set
lighting.set
system.status
files.read
developer.request_change
```

------------------------------------------------------------------------

## 16. Tool Router

Do not inject every tool schema into every 16K context.

``` text
User request
   ↓
Tool Router
   ↓
Relevant families
   ↓
small schema set
   ↓
Qwen
```

Examples:

`"play music"` → music tools only.

`"why did vision crash?"` → service status, logs, vision, developer
inspection.

This saves context and reduces tool-selection errors.

------------------------------------------------------------------------

## 17. MCP

MCP is the standard external capability boundary.

``` text
Robot Core
  ↓
MCP Client
  ├── Search
  ├── Browser
  ├── Home
  ├── Files
  ├── Development
  └── future integrations
```

Latency-critical internal functions such as `cancel_tts`, `set_face`,
`update_world_state`, and `cancel_generation` remain direct internal
calls rather than being forced through MCP.

------------------------------------------------------------------------

## 18. Permission Architecture

  Level   Category               Examples
  ------- ---------------------- ---------------------------------
  L0      Read-only              status, memory/history search
  L1      Harmless local         face, RGB, music
  L2      Environment            smart-home controls, local apps
  L3      External consequence   send/post/submit
  L4      System                 filesystem, terminal, packages
  L5      Development            modify Robot Core/app code
  L6      Critical               kernel, boot, security

A model-generated tool call is a **request**, not authorization.

Physical actuation later receives an independent real-time safety layer.

------------------------------------------------------------------------

## 19. Search and Browser

### SearXNG

Search provider for information retrieval.

``` text
Qwen → search.query → SearXNG → normalized results → Qwen
```

### Browser Use

Used when the robot must interact with a website:

``` text
Robot Core
 ↓
Permission Manager
 ↓
Browser Use
 ↓
Browser
```

**Search finds information. Browser automation acts on web interfaces.**

------------------------------------------------------------------------

## 20. Music

### Spotify

Spotipy is the Spotify provider.

Capabilities: - search - play - pause - resume - skip - previous -
queue - playlist operations - volume, where API/device support allows it

Local files use a separate local media provider.

------------------------------------------------------------------------

## 21. Smart Home

Home Assistant is the smart-device abstraction.

``` text
Robot Core
  ↓
Home Assistant
  ├── lights
  ├── plugs
  ├── fans
  ├── displays/TV
  ├── sensors
  ├── environmental devices
  └── future IoT
```

Robot Core does not implement manufacturer-specific protocols unless
unavoidable.

------------------------------------------------------------------------

## 22. Face / Expression UI

### Rive-based face

The 1080p display becomes the primary digital face.

Expression states include: - neutral - happy - excited - thinking -
confused - concerned - listening - speaking - sleeping - error/alert

``` text
Robot state
  ↓
Expression Controller
  ↓
emotion + intensity + activity
  ↓
Rive State Machine
  ↓
Display
```

Qwen provides high-level expression intent; it does not animate pixels
directly.

------------------------------------------------------------------------

## 23. ROG Keyboard and Lightbar Expression

The ROG Strix hardware exposes all keyboard zones through Armoury Crate.
Independent multi-zone control is therefore a requirement for any
replacement backend.

### Provider abstraction

``` text
Expression Controller
       ↓
LightingProvider
       │
       ├── Windows 11
       │     └── OpenRGB [verify exact controller support]
       │
       └── Linux Mint
             └── asusd/asusctl
```

Required addressable hardware:

``` text
ROG Aura
├── Keyboard Zone 1
├── Keyboard Zone 2
├── Keyboard Zone 3
├── Keyboard Zone 4
└── Lightbar / strip
    ├── left       [if exposed separately]
    └── right      [if exposed separately]
```

If a backend cannot expose the zones Armoury Crate exposes, it does not
satisfy the provider contract.

### Semantic lighting

Lighting represents robot body language:

``` text
LISTENING → gentle listening pulse
THINKING  → slow breathing / motion
SPEAKING  → audio-reactive or speech animation
HAPPY     → personality-defined animation
EXCITED   → stronger animated profile
SLEEPING  → dim/off
ERROR     → warning profile
```

Personalities may override lighting profiles.

------------------------------------------------------------------------

## 24. AI Terminal / Developer Console

Separate interfaces:

**Face UI:** human-facing personality/expression.\
**AI Terminal:** developer-facing observability/control.

Terminal displays: - CPU utilization / temperature / RAM - RTX
utilization / VRAM / temperature / power - Radeon status -
model/context/tokens per second/TTFT - STT/TTS status and latency -
camera/vision state - memory retrievals - current personality -
session - tool events - errors - service health

Structured event tags:

``` text
[USER]
[STT]
[LLM]
[MEMORY]
[SEARCH]
[VISION]
[TOOL]
[SKILL]
[FACE]
[RGB]
[TTS]
[SYSTEM]
[ERROR]
```

Do not expose private chain-of-thought. Expose operational events,
timings, tool calls/results, context composition summaries, and
diagnostics.

Commands may include:

``` text
/status
/models
/memory search "..."
/personality <name>
/vision status
/listen on
/listen off
/speak <text>
/face happy
/rgb status
/tools
/skills
/restart stt
/restart tts
/restart vision
```

------------------------------------------------------------------------

## 25. Self-Development / Agentic IDE

### Coding agent

**Aider** is the initial repository-aware development agent.

``` text
Robot detects limitation
      ↓
Development Controller
      ↓
Permission Gate
      ↓
Aider
      ↓
Temporary Git Worktree
      ↓
Sandbox
      ↓
Tests / lint / benchmark
      ↓
Review Gate
      ↓
Deploy or discard
```

The running production tree is never the coding agent's unrestricted
workspace.

### Development memory

Store: - bug reports - proposed changes - experiments - benchmarks -
commits - tests - failed approaches - successful changes - rollbacks

### Improvement loop

``` text
observe
 ↓
detect limitation
 ↓
diagnose
 ↓
design
 ↓
code
 ↓
test
 ↓
benchmark
 ↓
review
 ↓
deploy
 ↓
monitor
 ↓
learn
```

Initial self-improvement focuses on prompts, tools, memory, perception,
orchestration, skills, and application code---not autonomous weight
training.

------------------------------------------------------------------------

## 26. Code Sandbox

### Phase I: Podman

Generated code runs with: - temporary Git worktree - restricted
filesystem - restricted network - CPU limit - RAM limit - execution
timeout - no production secrets - disposable environment

### Phase II: Firecracker

Use microVM isolation when autonomous execution becomes sufficiently
powerful/risky to justify a stronger boundary.

### Critical updates

Kernel/boot/security changes are never ordinary coding-agent operations.

``` text
candidate change
 ↓
build
 ↓
static/security checks
 ↓
isolated testing
 ↓
recovery point
 ↓
install as candidate boot target
 ↓
reboot
 ↓
health check
 ├── pass → promote
 └── fail → rollback
```

Maintain: - CURRENT - CANDIDATE - RECOVERY

------------------------------------------------------------------------

## 27. Service Architecture

Final Linux Mint deployment uses independent services.

``` text
systemd
│
├── robot-core.service
├── llama-server.service
├── stt.service
├── tts.service
├── vision.service
├── memory.service
├── expression.service
├── search.service
└── developer services as needed
```

Service failures should be isolated and restartable.

Robot Core performs health checks and degrades gracefully where
possible.

------------------------------------------------------------------------

## 28. Windows → Linux Strategy

### Current phase

Develop on Windows 11.

Windows-specific providers hide behind interfaces.

### Final phase

Migrate to Linux Mint.

``` text
Robot Core
  ↓
Platform Abstraction
  ├── Windows backend
  └── Linux backend
```

Platform-specific features include: - RGB hardware - service
management - audio devices - GPU/runtime configuration - filesystem
paths - system telemetry - process control

Application-level logic remains portable.

------------------------------------------------------------------------

## 29. Storage Architecture

The internal 1 TB SSD is primary.

Suggested logical layout:

``` text
/robot
├── core/
├── config/
├── models/
│   ├── llm/
│   ├── stt/
│   ├── tts/
│   └── vision/
├── personalities/
├── voices/
├── memory/
├── database/
├── logs/
├── telemetry/
├── media/
├── datasets/
├── worktrees/
├── sandbox/
├── backups/
└── recovery/
```

USB policy:

  Storage         Role
  --------------- -------------------------------------
  Internal SSD    Production
  USB 3.x SSD     Portable/test environment
  USB 3.x flash   Prototype/recovery
  USB 2.0 flash   Installer/recovery/diagnostics only

Do not run the production robot from USB 2.0.

------------------------------------------------------------------------

## 30. Resource Scheduler

Because this is a 16 GB / 4 GB VRAM system, Robot Core must coordinate
heavy work.

Priority concept:

1.  Audio/VAD/barge-in
2.  Core LLM conversation
3.  Safety/state
4.  Expression
5.  Continuous lightweight vision
6.  Browser tasks
7.  Coding/builds
8.  Background consolidation/indexing

Avoid simultaneously running: - large browser automation - code
compilation - Qwen image reasoning - heavy vision processing - memory
consolidation

when interactive latency matters.

------------------------------------------------------------------------

## 31. Boot Sequence

Final Linux target:

``` text
Power on
 ↓
Linux Mint
 ↓
drivers / hardware
 ↓
systemd
 ↓
core infrastructure
 ├── llama-server
 ├── STT
 ├── TTS
 ├── memory
 ├── vision
 └── search
 ↓
Robot Core health checks
 ↓
Expression system
 ↓
restore safe session state
 ↓
ROBOT ONLINE
```

If a noncritical subsystem fails, Robot Core should start in degraded
mode rather than preventing all operation.

------------------------------------------------------------------------

## 32. Conversation Lifecycle

``` text
1. microphone frames arrive
2. echo cancellation / cleanup
3. VAD detects user speech
4. Whisper streams transcript
5. utterance finalized
6. Context Engine gathers:
   - personality
   - recent conversation
   - Mem0 memories
   - robot state
   - world state
   - relevant tools
7. Qwen generation begins
8. tool call if necessary
9. Robot Core validates/executes
10. result returned to Qwen
11. text chunks sent to Piper
12. audio begins
13. face/RGB synchronize with state
14. user can interrupt at any moment
15. conversation/archive/memory events persisted
```

------------------------------------------------------------------------

## 33. Tool Example

User:

> Play my favorite track and make the lights match the mood.

Flow:

``` text
STT
 ↓
Robot Core
 ↓
Context Engine
 ├── personality
 ├── music preference memory
 └── music + lighting schemas
 ↓
Qwen
 ├── music.play(...)
 └── lighting.set_state(...)
 ↓
Permission Manager
 ↓
MusicProvider + LightingProvider
 ↓
results
 ↓
Qwen response
 ↓
Piper + face + RGB
```

Qwen never directly talks to Spotify or the ROG controller.

------------------------------------------------------------------------

## 34. Visual Example

User:

> What am I holding?

``` text
STT
 ↓
Robot Core sees visual question
 ↓
capture current/high-quality webcam frame
 ↓
Qwen3.5 multimodal
 ↓
visual answer
 ↓
Piper
```

For ordinary awareness, Qwen receives compact World State instead of
frames.

------------------------------------------------------------------------

## 35. Memory Example

User mentions a durable preference.

``` text
Conversation
 ↓
Memory Manager
 ↓
relevance/storage decision
 ↓
Mem0
```

Later:

``` text
new request
 ↓
Memory retrieval
 ↓
relevant memories only
 ↓
Context Engine
 ↓
Qwen
```

Complete original chat remains independently available in SQLite.

------------------------------------------------------------------------

## 36. Physical Humanoid Expansion

The digital architecture must remain usable when the laptop becomes the
robot computer.

### Critical rule

**The LLM never directly drives motors.**

``` text
Qwen
 ↓
high-level intent
 ↓
Task Planner
 ↓
Motion Planner
 ↓
Deterministic Safety Controller
 ↓
MCU
 ↓
Motor Drivers
 ↓
Actuators
```

Feedback:

``` text
Encoders
IMU
Force/Torque
Current
Temperature
Limit switches
       ↓
Safety Controller / State Estimator
       ↓
Robot Core
```

Physical emergency mechanisms must remain effective even if Linux, Robot
Core, or Qwen crashes.

Future physical stack may include ROS 2, MCU firmware, real-time motor
loops, navigation, SLAM, manipulation, balance, battery/BMS, and hard
emergency stop. These are Phase II/III and require hardware-specific
design.

------------------------------------------------------------------------

## 37. Safety Boundaries

Independent boundaries:

``` text
LLM boundary
  → model cannot directly execute

Tool boundary
  → schema + permission validation

Code boundary
  → sandbox + Git worktree

Deployment boundary
  → tests + review + rollback

OS boundary
  → privileged operations isolated

Physical boundary
  → deterministic controller + MCU + E-stop
```

No single model output should be sufficient to bypass all layers.

------------------------------------------------------------------------

## 38. Final Software Baseline

  -----------------------------------------------------------------------
  System                  Selection               Status
  ----------------------- ----------------------- -----------------------
  Development OS          Windows 11              Current

  Final OS                Linux Mint              🔒

  LLM                     Qwen3.5-4B Q4_K_M       🔒

  Context                 16K / 32K experimental  🔒

  Production inference    llama.cpp /             🔒
                          llama-server            

  Development model lab   LM Studio               🔒

  Visual reasoning        Qwen3.5 multimodal      🔒

  Vision foundation       OpenCV                  🔒

  Face detection          YuNet INT8              🔒

  Face recognition        SFace                   🔒

  Object detection        Lightweight             🔒
                          OpenCV/NanoDet-class    

  Pose                    Lightweight OpenCV      🔒
                          model, on demand        

  STT                     Whisper Small           🔒
                          Multilingual            

  STT runtime             whisper.cpp             🔒

  VAD                     Silero VAD              🔒

  Audio preprocessing     WebRTC-style AEC/NS     🔒

  TTS                     Piper-compatible ONNX   🔒

  Persistent memory       Mem0                    🔒

  Raw history             SQLite + FTS            🔒

  Temporal graph          Graphiti                Phase II

  Search                  SearXNG                 🔒

  Browser automation      Browser Use             🔒

  Smart home              Home Assistant          🔒

  Spotify                 Spotipy                 🔒

  Coding agent            Aider                   🔒

  Initial sandbox         Podman                  🔒

  Advanced sandbox        Firecracker             Phase II

  Robot Core              Python 3.12 + asyncio   🔒

  API                     FastAPI + Pydantic      🔒

  External tool protocol  MCP                     🔒

  Face UI                 Rive                    🔒

  Windows RGB             OpenRGB, hardware       🔒 architecture
                          validation required     

  Linux ROG RGB           asusd/asusctl           🔒 architecture

  Linux services          systemd                 🔒
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 39. What We Build Ourselves

Do **not** rebuild mature infrastructure.

Custom code should concentrate on:

1.  Robot Core
2.  Context Engine
3.  State Manager
4.  World State
5.  Tool Registry
6.  Tool Router
7.  Permission Manager
8.  Resource Scheduler
9.  Provider interfaces
10. Personality Manager
11. Expression Controller
12. Memory policy/orchestration
13. Service health/recovery
14. Development Controller
15. Physical hardware bridge later

Everything else should use mature projects where practical.

------------------------------------------------------------------------

## 40. Implementation Roadmap

### Phase 0 --- Hardware/Runtime Benchmarking

-   Benchmark Qwen 16K on RTX 3050.
-   Test multimodal projector.
-   Benchmark Whisper CPU vs Radeon Vulkan.
-   Benchmark Piper runtime.
-   Validate webcam/audio.
-   Validate OpenRGB access to all ROG zones.
-   Record RAM/VRAM budgets.

### Phase 1 --- Voice Loop

Build:

``` text
Mic → AEC/VAD → Whisper → Robot Core → Qwen → Piper → Speaker
```

Add cancellation/barge-in before moving on.

### Phase 2 --- Core State and Memory

-   SQLite archive
-   Mem0
-   Context Engine
-   personality loading
-   session state
-   observability

### Phase 3 --- Tool Framework

-   Tool Registry
-   Router
-   Permission Manager
-   MCP client
-   search
-   music
-   browser

### Phase 4 --- Expression

-   Rive face
-   RGB provider
-   semantic expression states
-   synchronize listening/thinking/speaking

### Phase 5 --- Vision

-   OpenCV camera service
-   detection/tracking
-   face recognition
-   World State
-   Qwen snapshot reasoning

### Phase 6 --- Environment

-   Home Assistant
-   application integrations
-   local files/media
-   additional MCP providers

### Phase 7 --- Developer Agent

-   Aider
-   temporary worktrees
-   Podman
-   automated tests
-   deployment gate
-   rollback

### Phase 8 --- Linux Migration

-   Linux Mint
-   NVIDIA/CUDA
-   Vulkan
-   asusctl/asusd
-   systemd
-   Podman rootless
-   complete hardware validation

### Phase 9 --- Advanced Intelligence

-   Graphiti
-   improved consolidation
-   proactive behavior
-   richer world modeling
-   optional F5-TTS
-   Firecracker

### Phase 10 --- Physical Humanoid

-   ROS 2 / robotics middleware as justified
-   MCU
-   sensors
-   motor control
-   deterministic safety
-   motion planning
-   battery/BMS
-   emergency stop
-   physical expression

------------------------------------------------------------------------

## 41. Definition of Digital v1.0 Success

The digital robot is considered operational when it can:

-   start automatically and report health
-   listen continuously without hearing itself
-   be interrupted while speaking
-   transcribe multilingual speech locally
-   answer through local Qwen
-   speak through a custom Piper-compatible voice
-   remember relevant information across sessions
-   retrieve complete past conversations
-   maintain current World State
-   recognize known faces with consent/configuration
-   reason over a webcam snapshot
-   search the web
-   operate supported websites
-   control Spotify/local media
-   control Home Assistant devices
-   display an animated face
-   control all supported ROG RGB zones
-   switch personalities
-   call tools through permissions
-   expose developer telemetry
-   propose code changes safely
-   test generated code in isolation
-   recover from subsystem failures without losing the whole robot

------------------------------------------------------------------------

## 42. Architectural Principles

1.  **Local first.**
2.  **LLM is intelligence, not authority.**
3.  **Memory lives outside model context.**
4.  **Continuous perception is cheap; deep reasoning is selective.**
5.  **Use providers so components can be replaced.**
6.  **Use mature open-source infrastructure instead of rebuilding it.**
7.  **Keep real-time paths short.**
8.  **Treat 4 GB VRAM and 16 GB RAM as hard constraints.**
9.  **Schedule heavy background workloads.**
10. **Every consequential action passes through permissions.**
11. **Generated code never executes directly in production.**
12. **Physical motion gets deterministic safety below the LLM.**
13. **Observability is a first-class subsystem.**
14. **Windows is development; Linux Mint is the final robot platform.**
15. **Benchmark actual hardware instead of assuming acceleration.**

------------------------------------------------------------------------

# Final Architecture Snapshot

``` text
                             HUMAN
                  ┌────────────┼─────────────┐
                  ▼            ▼             ▼
                Voice        Camera       Interaction
                  │            │
          WebRTC AEC/NS      OpenCV
                  │            │
             Silero VAD     Perception
                  │            │
           Whisper Small   World State
                  │            │
                  └──────┬─────┘
                         ▼
                ┌───────────────────┐
                │    ROBOT CORE     │
                │ Python + asyncio  │
                │                   │
                │ Context Engine    │
                │ State Manager     │
                │ Personality       │
                │ Tool Router       │
                │ Permissions       │
                │ Resource Manager  │
                │ Expression        │
                └─────────┬─────────┘
                          │
                     llama-server
                          │
                    Qwen3.5-4B
                 language + vision
                          │
        ┌─────────────────┼──────────────────────┐
        ▼                 ▼                      ▼
      MEMORY             TOOLS                OUTPUT
        │                 │                      │
   ┌────┼────┐     ┌──────┼─────────┐      ┌────┼─────┐
   ▼    ▼    ▼     ▼      ▼         ▼      ▼          ▼
 Mem0 SQLite Graphiti SearXNG Browser Home  Piper     Expression
             later          Use  Assistant   │           │
                                  │          Speaker   ┌──┴────┐
                               Spotipy                 ▼       ▼
                                                   Rive     RGB
                                                           │
                                               OpenRGB / asusd

Development path:
Robot Core → Development Controller → Aider → Git Worktree
                                     → Podman → tests → gate
                                     → Firecracker later

Physical path:
Robot Core → Task Planner → Motion Planner → Safety Controller
          → MCU → Motor Drivers → Actuators
                    ▲
             Sensors/Feedback
```

------------------------------------------------------------------------

**Baseline status:** Digital architecture decisions are sufficiently
complete to begin implementation. Future changes should be driven by
benchmarks, hardware discoveries, or clearly superior maintained
projects---not architecture churn.

---

## 43. Latency & Responsiveness Targets

The primary conversational latency metric is **end-of-user-speech → first audible robot speech**. Total answer completion time is secondary because LLM generation and TTS are streamed concurrently.

### Theoretical stage budget

| Stage | Optimistic | Realistic engineering target |
|---|---:|---:|
| VAD / end-of-turn detection | 100–200 ms | 200–350 ms |
| Whisper finalization | 50–150 ms | 100–300 ms |
| Context assembly + Mem0 retrieval | 20–80 ms | 50–150 ms |
| Qwen time-to-first-token (TTFT) | 80–200 ms | 150–400 ms |
| Generate first speakable phrase | 100–250 ms | 200–500 ms |
| Piper first audio | 20–80 ms | 40–120 ms |
| Audio buffering/output | 20–50 ms | 30–80 ms |
| **Total end-of-speech → audible response** | **~390–1,010 ms** | **~770–1,900 ms** |

These are architecture targets, not guaranteed measured results. Exact performance must be benchmarked on the Ryzen 7 6800H, RTX 3050 4 GB, Radeon 680M, and 16 GB DDR5 system.

### Locked performance goals

| Interaction | Target |
|---|---:|
| **Barge-in / stop robot speech** | **<200 ms** |
| **Normal conversational response** | **<1.5 s preferred** |
| **Excellent conversational result** | **<1.0 s** |
| **Acceptable normal-response ceiling** | **~2.0 s** |
| Memory-heavy response | ~1–2 s target |
| On-demand Qwen visual reasoning | ~1.5–4 s target |
| Simple local tool action | ~0.5–1.5 s |
| Web search | ~2–6+ s |
| Browser automation | Several seconds to tens of seconds |

External-network and website latency are not treated as local inference latency.

### Streaming pipeline

The system must avoid serial completion of STT → LLM → full TTS. Speech recognition occurs while the user is talking, LLM output streams immediately, and TTS starts as soon as a stable speakable phrase is available.

```text
USER SPEAKING
     │
     ├──── Whisper streams partial transcript
     │
USER FINISHES
     │
     ▼
VAD confirms end of turn
     │
     ▼
final STT + context assembly
     │
     ▼
Qwen starts streaming
     │
     ├── tokens
     ├── tokens
     ├── first stable phrase ─────► Piper ─────► Speaker
     │
     └── Qwen continues generating while phrase #1 is spoken
```

### Barge-in latency path

Barge-in must not wait for another LLM inference.

```text
Robot speaking
     │
User begins speaking
     │
     ▼
VAD detects speech
     │
     ▼
cancel audio playback
cancel TTS generation
cancel current Qwen generation
     │
     ▼
begin/continue STT for new turn
```

**Engineering target: robot speech should stop within 200 ms of confidently detecting the user's interruption.**

### Latency optimization rules

1. Keep Qwen resident whenever practical.
2. Keep speech services warm rather than spawning them per turn.
3. Stream Whisper partial transcripts.
4. Use aggressive but reliable end-of-turn detection.
5. Retrieve only relevant memories.
6. Inject only relevant tool schemas.
7. Send compact World State rather than continuous images.
8. Invoke Qwen vision only when required.
9. Begin TTS at stable phrase boundaries instead of waiting for the complete answer.
10. Allow Qwen generation and Piper playback to overlap.
11. Give interactive audio/LLM work priority over browser automation, builds, indexing, and memory consolidation.
12. Cancel obsolete generation immediately after barge-in.
13. Benchmark CPU vs Radeon 680M for Whisper/Piper rather than assuming GPU acceleration is faster.
14. Track TTFT, STT finalization, TTS first-audio, end-to-end response latency, and barge-in latency in the AI Terminal.

### Required telemetry

The AI Terminal should expose at minimum:

```text
STT_PARTIAL_LATENCY
STT_FINAL_LATENCY
TURN_END_LATENCY
MEMORY_RETRIEVAL_LATENCY
CONTEXT_BUILD_LATENCY
LLM_TTFT
LLM_TOKENS_PER_SECOND
TTS_FIRST_AUDIO_LATENCY
AUDIO_OUTPUT_BUFFER_LATENCY
END_OF_SPEECH_TO_FIRST_AUDIO
BARGE_IN_STOP_LATENCY
TOOL_EXECUTION_LATENCY
```

The final tuning criterion is **perceived conversational responsiveness**, not merely peak tokens per second.
