#!/bin/bash
# Download and configure multiple Whisper STT models for Robot Core
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_MODELS_DIR="/home/ameya/whisper.cpp/models"
ROBOT_STT_DIR="$DIR/models/stt"

mkdir -p "$WHISPER_MODELS_DIR"
mkdir -p "$ROBOT_STT_DIR"

MODELS=(
    "small.en"
    "small"
    "medium.en"
    "medium"
    "large-v3-turbo"
    "large-v3"
)

echo "🎙️ Downloading Whisper models to $WHISPER_MODELS_DIR..."

for model in "${MODELS[@]}"; do
    target_file="$WHISPER_MODELS_DIR/ggml-${model}.bin"
    if [ -f "$target_file" ] && [ $(stat -c%s "$target_file" 2>/dev/null || stat -f%z "$target_file") -gt 100000000 ]; then
        echo "✅ Model $model already exists ($(du -h "$target_file" | cut -f1)), skipping."
    else
        echo "⏳ Downloading $model..."
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${model}.bin"
        curl -L -C - --fail --show-error --progress-bar -o "$target_file" "$url"
        echo "✅ $model downloaded successfully."
    fi
done

echo ""
echo "🔗 Symlinking all models into $ROBOT_STT_DIR..."
for bin_file in "$WHISPER_MODELS_DIR"/ggml-*.bin; do
    if [ -f "$bin_file" ]; then
        base_name=$(basename "$bin_file")
        ln -sf "$bin_file" "$ROBOT_STT_DIR/$base_name"
    fi
done

echo "🎉 All Whisper STT models ready:"
ls -lh "$ROBOT_STT_DIR"/ggml-*.bin
