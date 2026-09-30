# Robot Core System Architecture & Benchmark Report

This document details the complete hardware profile, technology stack mapping, and performance benchmarks for the Robot Core full-duplex humanoid robotic intelligence platform.

---

## 1. Hardware Architecture & Device Mapping

The system operates across a heterogeneous multi-processor architecture combining high-efficiency CPU cores, an integrated GPU for visual display rendering, and a dedicated GPU for real-time neural language inference.

| Hardware Subsystem | Hardware Specification | Purpose / Workload | Driver / API |
| :--- | :--- | :--- | :--- |
| **Host CPU** | **AMD Ryzen 7 6800H** (8 Cores, 16 Threads @ up to 4.7 GHz, 16MB L3 Cache) | Speech Recognition (STT), Text-to-Speech (TTS), Face Detection, Object Detection, Barge-in Echo Subtraction, FastAPI Event Loop | Linux Native (`x86_64`), AVX2, FMA |
| **Discrete GPU (dGPU)** | **NVIDIA GeForce RTX 3050 Mobile** (4,096 MiB GDDR6 VRAM, 2048 CUDA Cores) | Real-time LLM inference (`llama-server` running Qwen 3.5 4B Q4_K_M) | NVIDIA Driver 595.91.07, CUDA 12.x, cuBLAS |
| **Integrated GPU (iGPU)** | **AMD Radeon 680M** (12 RDNA2 CUs @ up to 2.2 GHz, Shared RAM) | Browser HUD rendering, 3D VRM Humanoid character animation, WebGL2 / Three.js 60 FPS viewport | Vulkan / Mesa RADV, WebGL2 |
| **System Memory (RAM)** | **16 GB DDR5-4800 MHz** (Dual Channel) | Model weights buffering, audio circular buffers, perception frames, Mem0 vector caching | OS Page Cache, Zero-copy shared memory |
| **Host Operating System** | **Linux Mint 22.3 ("Zena")** / Ubuntu 24.04 LTS Base, Linux Kernel 6.8+ | System scheduling, low-latency audio via ALSA / PipeWire, ASUS ROG Aura DBus control | POSIX Real-Time FIFO / Round Robin |
| **Physical Audio I/O** | Dual Array Microphones + Realtek High-Definition Audio DAC | 16 kHz acoustic capture with pre-roll buffering, 22.05 kHz stereo speech playback | ALSA / PortAudio / `sounddevice` |
| **Physical Optical Sensor**| Integrated HD Webcam (USB UVC, Index 0) | Continuous 60 FPS 640x480 video capture for real-time face tracking & object perception | OpenCV VideoCapture / V4L2 |

---

## 2. Software Architecture & Technology Stack

The pipeline is organized into modular asynchronous microservices connected via high-speed HTTP/1.1 streaming and WebSockets:

```mermaid
graph TD
    User([User Voice & Visuals]) -->|UVC Camera 60 FPS| Vision[Perception Engine]
    User -->|Microphone 16kHz PCM| VAD[Silero VAD + Echo Guard]
    
    subgraph Host CPU: AMD Ryzen 7 6800H
        Vision -->|YuNet Face 58 FPS| WS[WorldState Model]
        Vision -->|YOLOv8n Object 5.5 FPS| WS
        VAD -->|Voice Energy > 0.45| Whisper[whisper.cpp Base Model]
        Whisper -->|Transcribed Text| Core[Robot Engine Coordinator]
        Core -->|Prompt Assembly| Context[Context Engine Budget-Aware]
        TTS_Piper[Piper TTS en-us-lessac] -->|22.05 kHz PCM| AudioOut[Audio Player DAC]
    end
    
    subgraph Dedicated GPU: NVIDIA RTX 3050 4GB
        Context -->|HTTP POST Stream| LlamaServer[llama-server Qwen3.5-4B Q4_K_M]
        LlamaServer -->|Token Stream 19.7 tok/s| Chunk[Clause Chunker]
    end
    
    Chunk -->|Phrases| TTS_Piper
    WS -->|HTTP JSON 4Hz| BrowserHUD
    Core -->|WebSocket Events| BrowserHUD
    
    subgraph Integrated GPU: AMD Radeon 680M
        BrowserHUD[Browser UI Dashboard] --> ThreeJS[Three.js WebGL2 60 FPS]
        ThreeJS --> Gaze[GazeSystem Face Tracking IK]
        ThreeJS --> Anim[AnimationSystem VRM Bones]
    end
```

---

## 3. Microservice Latency & Throughput Benchmarks

Empirical benchmarks collected directly on this host system under live operating loads:

| Pipeline Stage | Underlying Technology | Model / Weight Version | Compute Device | Benchmark Metric | Measured Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Speech-to-Text (STT)** | `whisper.cpp` (`whisper-server`) | `ggml-base.bin` (39M params) | CPU (4 Threads) | Latency for 2.0s audio | **660.5 ms** |
| **LLM Time-To-First-Token** | `llama-server` | `Qwen3.5-4B-Q4_K_M.gguf` | NVIDIA RTX 3050 (24 offloaded layers) | TTFT (Prompt eval + 1st token) | **359.8 ms** |
| **LLM Generation Speed** | `llama-server` | `Qwen3.5-4B-Q4_K_M.gguf` | NVIDIA RTX 3050 (24 offloaded layers) | Decoding Throughput | **19.7 tokens/sec** |
| **Text-to-Speech (TTS)** | `piper` (ONNX Runtime) | `en-us-lessac-medium.onnx` | CPU (AVX2) | Latency for 3.66s phrase | **104.5 ms** (RTF: **0.029**) |
| **Face Detection** | OpenCV DNN `FaceDetectorYN` | `face_detection_yunet.onnx` | CPU (INT8) | Single frame inference | **17.0 ms** (**58.7 FPS**) |
| **Object Detection** | Ultralytics YOLOv8 | `yolov8n.pt` (Nano) | CPU (Torch) | Single frame inference | **181.0 ms** (**5.5 FPS**) |
| **3D Avatar Rendering** | Three.js WebGL2 Engine | VRM Humanoid Rig (17 Canonical Bones) | AMD Radeon 680M | Viewport Frame Rate | **60.0 FPS** (**16.6 ms**) |
| **End-to-End Voice Turn** | Full-duplex streaming loop | STT + LLM (TTFT) + TTS (1st Chunk) | Distributed | Total turnaround time | **~1,124 ms** |

---

## 4. Personalities System

The dashboard supports multi-personality switching. Hot-swapping changes the system prompt, behavior profile, vocabulary, and response tone without restarting the server:

1. **Neo (`neo.yaml`)**:
   - *Archetype*: Confident, direct, grounded, philosophically alert.
   - *Physical awareness*: Directly moves and acknowledges his physical humanoid chassis.
2. **Aria (`aria.yaml`)**:
   - *Archetype*: Warm, empathetic, cheerful, highly conversational.
   - *Physical awareness*: Expresses emotion and joy in interacting with people in physical space.
3. **GLaDOS (`glados.yaml`)**:
   - *Archetype*: Sarcastic, dry, passive-aggressive, test-oriented.
   - *Physical awareness*: Moves with chilling precision; observes humans with dry condescension.

---

## 5. Vision-Grounded QA & Real-Time Face Tracking

- **Visual Grounding Resolution**:
  - The local `Qwen3.5-4B` weights do not use a vision encoder (`mmproj`). To enable questions such as *"what do you see?"*, *"what is this?"*, and *"what am I holding?"*, the system performs real-time visual grounding:
  - OpenCV YuNet face detection and YOLOv8n object detection continuously populate `WorldState`.
  - When the user asks a visual question, the active perceptual state is injected directly into the LLM context prompt under `[Visual Stream in Front of Robot]`, enabling prompt, reliable visual reasoning without HTTP 500 errors.
- **Head & Neck Face Tracking (Look-At IK)**:
  - Face tracking runs at 4 Hz via `/world-state`.
  - The user's normalized face center coordinates `(norm_x, norm_y)` are mapped into 3D world target space.
  - Head (`J_Bip_C_Head`) and Neck (`J_Bip_C_Neck`) are smoothly oriented using spherical linear interpolation (SLERPF) with natural micro-saccades and idle breathing.
