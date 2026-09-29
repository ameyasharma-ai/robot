#!/bin/bash
# Robot Core — Production launcher
# Starts infrastructure services and launches the voice pipeline.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "🤖 Robot Core — Starting up..."

# ─── 0. Signal Trapping & Cleanup ────────────────────────────────────
cleanup() {
    echo ""
    echo "🛑 Shutting down Robot Core..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─── 1. Clean Hardware Mic Calibration ───────────────────────────────
echo "[AUDIO] Calibrating clean hardware microphone capture (zero clipping)..."
pactl unload-module module-echo-cancel 2>/dev/null || true
amixer -c 1 sset 'Internal Mic Boost' 0 2>/dev/null || true
amixer -c 1 sset 'Capture' 20 2>/dev/null || true
pactl set-source-volume alsa_input.pci-0000_05_00.6.analog-stereo 25% 2>/dev/null || true
pactl set-default-source alsa_input.pci-0000_05_00.6.analog-stereo 2>/dev/null || true

# ─── 2. llama-server (RTX 3050, port 8080) ──────────────────────────
if ! curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
    if [ -f "/home/ameya/llama.cpp/build/bin/llama-server" ] && [ -f "/home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf" ]; then
        echo "[LLM] Starting llama-server on RTX 3050..."
        /home/ameya/llama.cpp/build/bin/llama-server \
            -m /home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf \
            -ngl 24 \
            -c 2048 \
            --host 127.0.0.1 \
            --port 8080 > "$DIR/logs/llama-server.log" 2>&1 &

        echo -n "[LLM] Waiting for llama-server..."
        for i in $(seq 1 30); do
            if curl -s http://127.0.0.1:8080/health | grep -q '"ok"'; then
                break
            fi
            echo -n "."
            sleep 0.5
        done
        echo " ready"
    else
        echo "[LLM] llama-server binary or model not found at configured paths."
    fi
fi

# ─── 3. whisper-server (port 8081) ───────────────────────────────────
WHISPER_BIN=""
if [ -f "/home/ameya/whisper.cpp/build/bin/whisper-server" ]; then
    WHISPER_BIN="/home/ameya/whisper.cpp/build/bin/whisper-server"
elif [ -f "/home/ameya/whisper.cpp/build_vk/bin/whisper-server" ]; then
    WHISPER_BIN="/home/ameya/whisper.cpp/build_vk/bin/whisper-server"
fi

WHISPER_MODEL=""
if [ -f "/home/ameya/whisper.cpp/models/ggml-base.bin" ]; then
    WHISPER_MODEL="/home/ameya/whisper.cpp/models/ggml-base.bin"
elif [ -f "/home/ameya/whisper.cpp/models/ggml-small.bin" ]; then
    WHISPER_MODEL="/home/ameya/whisper.cpp/models/ggml-small.bin"
elif [ -f "/home/ameya/whisper.cpp/models/ggml-tiny.en.bin" ]; then
    WHISPER_MODEL="/home/ameya/whisper.cpp/models/ggml-tiny.en.bin"
fi

if ! curl -s http://127.0.0.1:8081/ > /dev/null 2>&1; then
    if [ -n "$WHISPER_BIN" ] && [ -n "$WHISPER_MODEL" ]; then
        echo "[STT] Starting whisper-server with $WHISPER_MODEL..."
        "$WHISPER_BIN" \
            -m "$WHISPER_MODEL" \
            -t 4 \
            --host 127.0.0.1 \
            --port 8081 > "$DIR/logs/whisper-server.log" 2>&1 &

        echo -n "[STT] Waiting for whisper-server..."
        for i in $(seq 1 30); do
            if curl -s http://127.0.0.1:8081/ > /dev/null 2>&1; then
                break
            fi
            echo -n "."
            sleep 0.5
        done
        echo " ready"
    else
        echo "[STT] whisper-server binary or model not found. STT will fall back gracefully or connect when started."
    fi
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
if [ -f "$DIR/.venv/bin/python" ]; then
    exec "$DIR/.venv/bin/python" -m robot "$@"
else
    exec python3 -m robot "$@"
fi
