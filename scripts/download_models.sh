#!/bin/bash
# Download and configure 2B Vision Models (VLM) & <4B Uncensored LLMs for Robot Core
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "🤖 Robot Core — Local Model Manager"
echo "======================================"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    "$DIR/.venv/bin/python" "$DIR/scripts/download_vlm_and_uncensored_models.py" --help
    exit 0
fi

if [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
    "$DIR/.venv/bin/python" "$DIR/scripts/download_vlm_and_uncensored_models.py" --list
    exit 0
fi

# Execute python downloader
"$DIR/.venv/bin/python" "$DIR/scripts/download_vlm_and_uncensored_models.py" "$@"
