# 🤖 Robot Core

**Local-first humanoid robot brain** — full-duplex voice conversation, persistent memory, agentic tools, and modular architecture.

> **The LLM reasons; Robot Core owns the robot.**

---

## Architecture

```text
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
            │ Memory Manager    │
            └─────────┬─────────┘
                      │
                 llama-server
                      │
                Qwen3.5-4B
             language + vision
                      │
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
      MEMORY         TOOLS        OUTPUT
        │             │              │
   ┌────┼────┐     search       Piper TTS
   ▼         ▼     browser        │
  Mem0    SQLite    home        Speaker
          + FTS   assistant
```

## 🚀 Month 0 Iteration — Base

This release marks the completed **Month 0 Base Iteration** of Robot Core:
- **Full-Duplex Voice Engine**: Silero VAD + real-time barge-in cancellation and speaker echo isolation calibrated for physical laptop hardware.
- **Streaming Sentence TTS Chunker**: Coherent, whole-sentence streaming synthesis via Piper ONNX TTS with minimal TTFT and natural cadence.
- **Hot-Swappable Multi-Persona System**: Matrix hacker girl **Cypher** (sharp, street-smart operator — shipped as the default `neo` persona), **Aria** (empathetic companion), **GLaDOS** (clinical sarcasm), **Jarvis** (Iron Man-style assistant), and **Luna** (devoted companion).
- **Physical World State & Perception**: Continuous 60 FPS camera perception with YuNet face tracking and YOLOv8 object recognition with spatial HUD telemetry.
- **Dual 3D Humanoid Engine & Digital HUD**: WebGL2 humanoid avatar (cyber/anime & realistic portrait modes) with interactive semantic IK actions (wave, kick, raise hand, head tracking) and text/voice dual-mode console.
- **Proactive Agentic Tools & Memory**: Live DuckDuckGo web search, conversation history FTS, and long-term semantic memory retrieval.

---

## Key Features

- **Full-duplex voice** with barge-in interruption (< 200ms)
- **Streaming STT with Parallel Chunking**: Chunks speech during utterance with energy dip splitting, slashing final turnaround latency by up to 80% with live streaming captions.
- **Context Layer Intelligence & Strict Entity Attribution**: Context-aware budget management (lean <2,200 tokens for local Qwen on RTX 3050 vs 1M tokens for Gemini Cloud) with guaranteed user entity ownership.
- **Expressive 3D Avatar Kinematics**: 10 expressive semantic actions (`nod`, `shake_head`, `tilt_head`, `shrug`, `bow`, `clap`, `think`, `cheer`, `hands_on_hips`, `heart_gesture`) with Three.js inverse kinematics and natural language parsing.
- **Multi-Zone Keyboard RGB Lighting**: Differentiated ASUS ROG Aura lighting separating keyboard zones (WASD, Center, Right) from front underglow with traveling voice ripples, neural calculation scanners, and acoustic visualizers.
- **Extensible Tool & Skill Ecosystem**: Autonomous agentic tools including `web.search`, `knowledge.wikipedia`, `system.calculate`, `system.hardware`, `system.time`, `music.play`, and `character.execute_action`.
- **Persistent memory**: Mem0 for semantic recall + SQLite FTS for cross-session history
- **Streaming pipeline**: LLM tokens → phrase chunker → Piper TTS (< 150ms TTFA)
- **Personality system**: Hot-swappable YAML-defined personas (neo/Cypher, aria, glados, jarvis, luna)
- **FastAPI server & Cyberdeck HUD**: REST + WebSockets + 3D WebGL avatar + Tool Resource Monitor + Image Generation Studio
- **Health monitoring & Observability**: Graceful degradation, telemetry metrics, and 100% passing test suite (365 tests)

## Hardware

| Resource | Hardware | Role |
|---|---|---|
| CPU | AMD Ryzen 7 6800H, 8C/16T | Robot Core, audio, CV |
| dGPU | NVIDIA RTX 3050, 4GB VRAM | Qwen3.5-4B inference |
| iGPU | Radeon 680M, RDNA2 | Whisper acceleration |
| RAM | 16 GB DDR5 | OS, services, models |

## Quick Start

### Prerequisites

- Linux (Ubuntu 22.04+ / Linux Mint)
- Python 3.12+
- NVIDIA CUDA Toolkit
- Docker (for Qdrant / Mem0)

### Setup

```bash
# Clone
git clone https://github.com/ameyasharma-ai/robot.git
cd robot

# Install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
```

### Build External Services

#### llama-server (Qwen3.5-4B on RTX 3050)
```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc) --target llama-server
```

#### whisper-server (Whisper Small on Vulkan/CPU)
```bash
git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp && cmake -B build_vk -DGGML_VULKAN=ON
cmake --build build_vk --config Release -j$(nproc) --target whisper-server
bash ./models/download-ggml-model.sh small
```

#### Piper TTS Voice
```bash
mkdir -p models/tts
wget -O models/tts/en-us-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_us/lessac/medium/en_us-lessac-medium.onnx
wget -O models/tts/en-us-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_us/lessac/medium/en_us-lessac-medium.onnx.json
```

### Run

```bash
# All-in-one (starts all services)
bash scripts/run.sh

# Or manually
python -m robot
```

### API

```bash
# Health
curl http://127.0.0.1:8888/health

# Status
curl http://127.0.0.1:8888/status

# Chat (text)
curl -X POST http://127.0.0.1:8888/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, Neo"}'

# Search memories
curl "http://127.0.0.1:8888/memory/search?query=favorite+color"
```

### Console access & API token

Opening `http://127.0.0.1:8888` from this machine needs no token — the API only
treats a caller as local when it arrives on the loopback interface *and* names the
server with a loopback `Host` *and* (for browsers) refers to its own origin, so a
rebound domain or a page on another site cannot reach it.

Any other client (LAN, another host) must present the API token as
`Authorization: Bearer <token>`, `?token=<token>` (EventSource / MJPEG / WebSocket)
or the session cookie from `POST /auth/login {"token": "…"}`. The token is
`ROBOT_API_TOKEN` if set, else `api.auth_token` in `config/default.yaml`, else the
value remembered in `data/.api_token` — generated once and reused across restarts.
Set `api.trust_loopback: false` to require the token from localhost too.

## Project Structure

```
robot/
├── config/                  # YAML configurations
│   ├── default.yaml         # Main config
│   └── personalities/       # Persona definitions
├── src/robot/               # Main package
│   ├── core/                # Engine, context, state, events
│   ├── audio/               # Capture, VAD, playback
│   ├── stt/                 # Speech-to-text providers
│   ├── tts/                 # Text-to-speech providers
│   ├── llm/                 # LLM providers + clause chunker
│   ├── memory/              # Mem0 + SQLite FTS
│   ├── personality/         # Personality loader
│   ├── perception/          # Camera, face mesh, YOLO world state
│   ├── expression/          # Lighting (RGB) & face expression backends
│   ├── image/               # Image generation providers (SD/Flux/unified)
│   ├── motion/              # Robot motion/servo bridge
│   ├── observability/       # Traces, resource monitor, health metrics
│   ├── tools/               # Registry, router, permissions
│   ├── api/                 # FastAPI server
│   ├── telemetry/           # Logging + metrics
│   └── services/            # Health monitoring & process supervisor
├── docs/                    # Architecture whitepaper, tracker, and reports
│   ├── whitepaper.md        # Architecture specification
│   └── TRACKER.md           # Roadmap and issue tracker
├── scripts/                 # Operational scripts
├── tests/                   # Test suite
└── pyproject.toml           # Python packaging
```

## Latency Targets

| Interaction | Target |
|---|---|
| Barge-in stop | < 200 ms |
| Normal response | < 1.5 s |
| Excellent response | < 1.0 s |

## License

MIT
