#!/usr/bin/env bash
# ==============================================================================
# Robot Core — Image Model Download & Cache Manager
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "❌ Error: Virtual environment python not found at $VENV_PYTHON"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
exec "$VENV_PYTHON" "$PROJECT_ROOT/scripts/download_image_models.py" "$@"
