#!/bin/bash
# Robot Core — Production launcher
# Starts infrastructure services and launches the voice pipeline.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "🤖 Robot Core — Starting up..."

# ─── 1. Acoustic Echo Cancellation (PipeWire) ───────────────────────
if ! pactl list sources short 2>/dev/null | grep -q echo-cancel-source; then
    echo "[AEC] Loading PipeWire echo-cancel module..."
    pactl load-module module-echo-cancel 2>/dev/null || true
    pactl set-default-source echo-cancel-source 2>/dev/null || true
    pactl set-default-sink echo-cancel-sink 2>/dev/null || true
fi

# ─── 2. llama-server (RTX 3050, port 8080) ──────────────────────────
if ! curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "[LLM] Starting llama-server on RTX 3050..."
    CUDA_VISIBLE_DEVICES=0 /home/ameya/llama.cpp/build/bin/llama-server \
        -m /home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf \
        -ngl 99 \
        -c 16384 \
        -fa \
        --host 127.0.0.1 \
        --port 8080 > "$DIR/logs/llama-server.log" 2>&1 &

    echo -n "[LLM] Waiting for llama-server..."
    while ! curl -s http://127.0.0.1:8080/health | grep -q '"ok"'; do
        echo -n "."
        sleep 0.5
    done
    echo " ready"
fi

# ─── 3. whisper-server (Vulkan/CPU, port 8081) ──────────────────────
if ! curl -s http://127.0.0.1:8081/load > /dev/null 2>&1; then
    echo "[STT] Starting whisper-server..."
    /home/ameya/whisper.cpp/build_vk/bin/whisper-server \
        -m /home/ameya/whisper.cpp/models/ggml-small.bin \
        -t 4 \
        --host 127.0.0.1 \
        --port 8081 \
        --language en \
        -sns \
        --no-speech-thold 0.5 > "$DIR/logs/whisper-server.log" 2>&1 &

    echo -n "[STT] Waiting for whisper-server..."
    while ! curl -s http://127.0.0.1:8081/load > /dev/null 2>&1; do
        echo -n "."
        sleep 0.5
    done
    echo " ready"
fi

# ─── 4. Qdrant (for Mem0, port 6333) ────────────────────────────────
if ! curl -s http://127.0.0.1:6333/healthz > /dev/null 2>&1; then
    if command -v docker &> /dev/null; then
        echo "[MEM] Starting Qdrant vector store..."
        docker run -d --rm \
            -p 6333:6333 \
            -v "$DIR/data/memory/qdrant:/qdrant/storage" \
            --name robot-qdrant \
            qdrant/qdrant 2>/dev/null || true

        echo -n "[MEM] Waiting for Qdrant..."
        for i in $(seq 1 20); do
            if curl -s http://127.0.0.1:6333/healthz > /dev/null 2>&1; then
                break
            fi
            echo -n "."
            sleep 0.5
        done
        echo " ready"
    else
        echo "[MEM] Docker not found — Mem0 will use in-memory storage"
    fi
fi

# ─── 5. Launch Robot Core ────────────────────────────────────────────
echo ""
echo "🚀 Launching Robot Core..."
exec python -m robot "$@"
