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

## Key Features

- **Full-duplex voice** with barge-in interruption (< 200ms)
- **Streaming pipeline**: LLM tokens → phrase chunker → TTS (< 1.5s response)
- **Persistent memory**: Mem0 for semantic recall + SQLite FTS for history
- **Context engine**: Dynamic 16K token budget allocation
- **Tool framework**: Namespace-based routing with L0-L6 permission gates
- **Personality system**: Hot-swappable YAML-defined personas
- **FastAPI server**: REST + WebSocket for developer console
- **Health monitoring**: Graceful degradation when services fail
- **Structured telemetry**: All whitepaper metrics tracked

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
│   ├── tools/               # Registry, router, permissions
│   ├── api/                 # FastAPI server
│   ├── telemetry/           # Logging + metrics
│   └── services/            # Health monitoring
├── scripts/                 # Operational scripts
├── tests/                   # Test suite
├── whitepaper.md            # Architecture specification
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
