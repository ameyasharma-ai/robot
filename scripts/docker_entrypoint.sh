#!/usr/bin/env bash
# ==============================================================================
# Robot Core — Docker Container Entrypoint
# ==============================================================================
set -e

echo "🤖 Robot Core Container Starting..."

# Ensure volume mount directories exist with correct permissions
mkdir -p /app/data/memory/qdrant /app/data/memory/personas /app/data/database /app/data/images /app/logs /app/models

# Check if ALSA audio device /dev/snd is available
if [ -d "/dev/snd" ]; then
    echo "🔊 Host ALSA audio devices detected (/dev/snd mounted)."
else
    echo "⚠️ Warning: /dev/snd not mounted. Audio capture/playback will use dummy/fallback driver."
fi

# Export fastembed cache path
export FASTEMBED_CACHE_PATH="/app/models/fastembed"

# Execute requested container command
exec "$@"
