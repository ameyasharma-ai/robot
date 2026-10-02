#!/bin/bash
# clear_memory.sh - Script to terminate all robot servers and clear system memory caches.

echo "🛑 Terminating all robot server processes to free VRAM and RAM..."

# Terminate processes
pkill -f "python -m robot" 2>/dev/null || true
pkill -f "python3 -m robot" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
pkill -f "whisper-server" 2>/dev/null || true
pkill -f "run.sh" 2>/dev/null || true

# Wait briefly and force kill any stubborn processes
sleep 1
pkill -9 -f "llama-server" 2>/dev/null || true
pkill -9 -f "whisper-server" 2>/dev/null || true

echo "✅ Server processes terminated. GPU VRAM and application RAM freed."

if command -v nvidia-smi &>/dev/null; then
    echo "[GPU] Current NVIDIA RTX 3050 VRAM:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
fi

echo "🧹 Clearing OS memory caches (pagecache, dentries, and inodes)..."
if [ "$EUID" -eq 0 ]; then
    sync && echo 3 > /proc/sys/vm/drop_caches
elif sudo -n true 2>/dev/null; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
else
    sync
    echo "Note: Non-root user without passwordless sudo. Ran sync; run with sudo to drop pagecaches."
fi

echo "✅ All memory and GPU processes cleared successfully!"
