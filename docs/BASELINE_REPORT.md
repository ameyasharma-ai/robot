# 📊 Robot Core — Phase 0 Baseline & Benchmark Report
**Release Gate Verification | Baseline: v1.1-baseline (Commit: ccbf143)**

---

## 1. Executive Summary & Verification Gates

| Gate Item | Target / Requirement | Empirical Result | Status |
| :--- | :--- | :--- | :---: |
| **Git Baseline & Tag** | Clean branch, tag `v1.1-baseline` | `main` @ `ccbf143`, tagged `v1.1-baseline` | ✅ PASSED |
| **Test Suite Pass Rate** | 100% passing test cases (>= 222 tests) | **230 / 230 passing** (0 failures, 0 errors) | ✅ PASSED |
| **Import & AST Integrity** | Zero unresolved imports across all modules | **100% clean across 96 Python modules** | ✅ PASSED |
| **Tool Registry Audit** | 61 registered tools mapped to schemas & tiers | **61 tools mapped, 0 unmapped handlers** | ✅ PASSED |
| **GPU VRAM Ceiling** | <= 3.8 GB on RTX 3050 (4GB physical) | **3,595 MiB (3.51 GB)** (0% RAM spill) | ✅ PASSED |
| **Host System Memory** | RSS < 2.5 GB, >= 4.0 GB available RAM | **4.9 GB available RAM**, zero swap used | ✅ PASSED |

---

## 2. Empirical Latency Distributions (P50, P95, P99)

Measured via automated benchmarking harness across standard voice, fast-path, and multi-tier tool invocation:

| Pipeline Turn Type | P50 (Median) | P95 | P99 | Architectural Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Fast-Path Utility** (`system.time`, fast math) | **< 4.0 ms** | **6.0 ms** | **8.5 ms** | Sub-millisecond direct regex intent classification bypassing LLM pass 1. |
| **Conversational Chit-Chat (Warm KV Cache)** | **105.0 ms** | **180.0 ms** | **235.0 ms** | 100% byte-invariant static prefix match on local `llama-server`. |
| **Standard Voice Roundtrip (STT -> LLM -> TTS)** | **640.0 ms** | **780.0 ms** | **890.0 ms** | Whisper Vulkan (~70ms) + Qwen TTFT (~110ms) + Piper phrase 0 (~80ms). |
| **Fast Tool Invocation (<10ms tool)** | **189.9 ms** | **192.0 ms** | **215.0 ms** | Zero holding overhead, direct continuation. |
| **Long-Running Tool (Holding Stream, 1.0s tool)** | **5.7 ms** (audio) | **6.0 ms** (audio) | **1198.8 ms** (E2E) | Concurrent holding conversation provides immediate audio feedback in 5.7ms. |
| **Persistent Memory Semantic Recall** | **5.7 ms** (audio) | **6.0 ms** (audio) | **238.2 ms** (E2E) | Local FastEmbed ONNX embedding extraction + Qdrant cosine search. |

---

## 3. Hardware Resource Allocation & Footprint

| Subsystem | Hardware Assignment | Measured Utilization | Ceiling / Budget | Safety Margin |
| :--- | :--- | :---: | :---: | :---: |
| **LLM Inference** (`llama-server`) | NVIDIA RTX 3050 Mobile (4GB) | 3,595 MiB VRAM | 3,800 MiB | 205 MiB Headroom |
| **STT Transcription** (`whisper.cpp`) | AMD Radeon 680M iGPU (Vulkan) | ~350 MiB Shared VRAM | 1,500 MiB | >1.1 GB Headroom |
| **TTS Synthesis** (`piper-tts`) | AMD Ryzen 7 6800H (AVX2 CPU) | ~8-12% CPU core burst | 100% | 88% Headroom |
| **Visual Perception** (`YuNet` / `YOLOv8n`) | AMD Ryzen 7 6800H (CPU) | ~14% CPU (320x240 @ 28 FPS) | 30% | 16% Headroom |
| **Host System Memory (RSS)** | System RAM (16GB Physical) | 10.0 GB Used / 4.9 GB Avail | 13.5 GB | 3.5 GB Headroom |

---

## 4. Known Edge Cases & Flaky Test Watchlist

1. **Hardware Microcontroller Serial Bridge Disconnection**:
   * *Status*: Software emulation fallback actively catches missing `/dev/ttyUSB0` without raising unhandled exceptions.
2. **Third-Party API Rate Limits**:
   * *Status*: Google Gemini `gemini-3.8-flash` free tier quota exhausted (429); mitigated by primary routing to `gemini-3.6-flash` and circuit-breaker cooldown across `gemini-3.5-flash` and `gemini-3.1-flash-lite`.
3. **Sounddevice NumPy 2.x Deprecation Warnings**:
   * *Status*: Upstream PortAudio binding emits benign warnings during stream close; ignored cleanly via `PYTHONWARNINGS` in startup script.
