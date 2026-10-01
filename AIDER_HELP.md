# Aider with Local llama.cpp Setup Guide

This guide documents the setup, flags, troubleshooting, and optimization steps for running [Aider](https://aider.chat) with a local `llama.cpp` server (specifically tested with `Qwen3.5-4B-Q4_K_M.gguf` on Linux).

---

## 1. Quick Start Commands

### Step 1: Start `llama-server` in Background
```bash
# Terminate any existing instances
killall llama-server 2>/dev/null

# Start the server with proper flags
/home/ameya/llama.cpp/build/bin/llama-server \
  -m /home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8080 \
  -c 16384 \
  --alias gpt-4o \
  --chat-template chatml \
  > /home/ameya/robot/logs/llama-server.log 2>&1 &
```

### Step 2: Launch `aider`
```bash
aider \
  --openai-api-base http://127.0.0.1:8080/v1 \
  --openai-api-key "dummy-key" \
  --model openai/gpt-4o
```
*(If you need to reduce token consumption, append `--map-tokens 0` to disable the codebase repository map).*

---

## 2. Parameter & Flag Breakdown

### `llama-server` Flags
| Flag | Recommended Value | Why It Is Critical |
| :--- | :--- | :--- |
| `-m <path>` | `/path/to/model.gguf` | Path to the local quantized GGUF weights. |
| `--port 8080` | `8080` | Local port serving the OpenAI-compatible HTTP REST endpoint. |
| `-c <tokens>` | `16384` | **Context Size**: Setting 32768 causes CUDA OOM crashes when combined with full layer offloading. Setting 8192 is too small for Aider's prompt + repo map (~11.4k tokens). 16K gives the ideal balance. |
| `--alias gpt-4o` | `gpt-4o` | **Model Name Alias**: Aider does not know context/token limits for custom local models and defaults max tokens to `0`. Aliasing to `gpt-4o` allows Aider to utilize up to 128k context without token ceiling warnings. |
| `--chat-template chatml` | `chatml` | **Chat Template**: Qwen's default Jinja template strictly enforces that the system prompt is message #1. Aider interleaves prompts, throwing `Jinja Exception: System message must be at the beginning`. The standard `chatml` template handles Aider's message ordering smoothly. |
| Layer Offload | *(Auto-fit)* | Avoid hardcoding `-ngl 999` if VRAM is constrained (<8GB), as it prevents `llama.cpp` from auto-fitting layers into available memory. |

### `aider` Flags
| Flag | Recommended Value | Why It Is Critical |
| :--- | :--- | :--- |
| `--openai-api-base` | `http://127.0.0.1:8080/v1` | Directs Aider to the local endpoint instead of api.openai.com. |
| `--openai-api-key` | `"dummy-key"` | Aider requires a non-empty key string even if local server does not enforce auth. |
| `--model` | `openai/gpt-4o` | Directs Aider to use the OpenAI API client provider with known high context limits. |
| `--map-tokens` | `0` or `1024` *(optional)* | Controls tokens allocated to the repository map. If your context window is tight, set `--map-tokens 0` to reclaim ~4,000 tokens. |

---

## 3. Troubleshooting & Known Error Resolution

### 1. `Model openai/... has hit a token limit! Input tokens: ~X of 0`
- **Root Cause**: Aider's LiteLLM metadata dictionary does not contain specifications for custom model names, causing max tokens to default to `0`.
- **Solution**: Pass `--alias gpt-4o` to `llama-server` and use `--model openai/gpt-4o` in Aider.

### 2. `cudaMalloc failed: out of memory / Connection error`
- **Root Cause**: `llama-server` crashed on startup due to high context (`-c 32768`) and forced layer offloading (`-ngl 999`), overflowing GPU VRAM.
- **Solution**: Check `/home/ameya/robot/logs/llama-server.log`. Reduce `-c` to `16384` and allow `llama.cpp` to auto-fit layers rather than forcing `-ngl 999`.

### 3. `Jinja Exception: System message must be at the beginning`
- **Root Cause**: Qwen's embedded Jinja chat template contains `{{- raise_exception('System message must be at the beginning') -}}`. Aider generates dynamic conversational chains where system prompts appear downstream.
- **Solution**: Supply `--chat-template chatml` to `llama-server`.

### 4. `request (11406 tokens) exceeds the available context size (8192 tokens)`
- **Root Cause**: Aider's repository map (typically ~4096 tokens) plus prompt history exceeded an 8K server context window.
- **Solution**:
  - Increase server context to `-c 16384`, **or**
  - Launch Aider with `--map-tokens 0` to skip sending the repo map.

### 5. `No LLM model was specified and no API keys were provided`
- **Root Cause**: Running bare `aider` without arguments defaults to cloud provider lookup.
- **Solution**: Include the local `--openai-api-base`, `--openai-api-key`, and `--model` arguments, or persist them in environment variables:
  ```bash
  export AIDER_OPENAI_API_BASE=http://127.0.0.1:8080/v1
  export AIDER_OPENAI_API_KEY=dummy-key
  export AIDER_MODEL=openai/gpt-4o
  ```
