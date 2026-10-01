#!/bin/bash
# clear_memory.sh - Script to terminate all robot servers and clear system memory caches.

echo "🛑 Terminating all robot server processes to free VRAM and RAM..."

# Terminate processes
pkill -f "python -m robot" 2>/dev/null || true
pkill -f "python3 -m robot" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
pkill -f "whisper-server" 2>/dev/null || true
pkill -f "run.sh" 2>/dev/null || true

# Wait briefly for processes to terminate and release resources
sleep 2

echo "✅ Server processes terminated. GPU VRAM and application RAM freed."

echo "🧹 Clearing OS memory caches (pagecache, dentries, and inodes)..."
echo "Note: This requires administrator privileges. You may be prompted for your password."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo "✅ System caches cleared. All memory successfully freed!"
