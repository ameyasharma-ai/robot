#!/bin/bash
# Robot Core — Production launcher
# Starts infrastructure services and launches the voice pipeline.
set -e
export PYTHONWARNINGS="ignore::DeprecationWarning"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "🤖 Robot Core — Starting up..."

cleanup() {
    echo ""
    echo "🛑 Shutting down Robot Core and child servers..."
    kill $(jobs -p) 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
    pkill -f "whisper-server" 2>/dev/null || true
    echo "✅ All GPU VRAM and process memory released."
}
trap cleanup EXIT INT TERM

# ─── 1. Clean Hardware Mic Calibration & EasyEffects Daemon ───────────
echo "[AUDIO] Calibrating clean hardware microphone capture (35% volume)..."
pactl unload-module module-echo-cancel 2>/dev/null || true
pactl set-source-volume alsa_input.pci-0000_05_00.6.analog-stereo 35% 2>/dev/null || true
pactl set-default-source alsa_input.pci-0000_05_00.6.analog-stereo 2>/dev/null || true
amixer -c 1 sset 'Internal Mic Boost' 0 2>/dev/null || true

if command -v easyeffects >/dev/null 2>&1; then
    if ! pgrep -f "easyeffects" >/dev/null 2>&1; then
        echo "[AUDIO] Starting EasyEffects audio processing service in background..."
        nohup easyeffects --gapplication-service >/dev/null 2>&1 &
        sleep 0.6
        easyeffects -l robot >/dev/null 2>&1 || true
    else
        echo "[AUDIO] EasyEffects service already active. Loading robot equalizer preset..."
        easyeffects -l robot >/dev/null 2>&1 || true
    fi
fi

# ─── 2. llama-server (RTX 3050, port 8080) ──────────────────────────
ACTIVE_PROVIDER="local"
if [ -f "$DIR/data/active_llm_provider.txt" ]; then
    ACTIVE_PROVIDER=$(cat "$DIR/data/active_llm_provider.txt" 2>/dev/null | tr -d '[:space:]')
fi

if [ "$ACTIVE_PROVIDER" = "gemini" ]; then
    echo "[LLM] Active provider is Gemini Cloud API — local llama-server deferred until swapped."
else
    if ! curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
        if [ -f "/home/ameya/llama.cpp/build/bin/llama-server" ] && [ -f "/home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf" ]; then
            echo "[LLM] Starting llama-server (100% offload to RTX 3050, FlashAttention)..."
            MMPROJ_ARG=""
            NGL=99
            MULTIMODAL_ENABLED=$(grep -E '^\s*multimodal:\s*true' "$DIR/config/default.yaml" || true)
            if [ -n "$MULTIMODAL_ENABLED" ] && [ -f "$DIR/models/vision/mmproj-Qwen3.5-4B-F16.gguf" ]; then
                echo "[LLM] Enabling multimodal vision projector ($DIR/models/vision/mmproj-Qwen3.5-4B-F16.gguf)..."
                MMPROJ_ARG="--mmproj $DIR/models/vision/mmproj-Qwen3.5-4B-F16.gguf --mmproj-offload"
                NGL=28
            fi

            pkill -9 -f "llama-server" 2>/dev/null || true
            sleep 0.5

            /home/ameya/llama.cpp/build/bin/llama-server \
                -m /home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf \
                $MMPROJ_ARG \
                -ngl $NGL \
                -fa on \
                -ctk q8_0 \
                -ctv q8_0 \
                -c 4096 \
                -np 1 \
                -b 4096 \
                -ub 4096 \
                --host 127.0.0.1 \
                --port 8080 > "$DIR/logs/llama-server.log" 2>&1 &

            echo -n "[LLM] Waiting for llama-server..."
            READY=0
            for i in $(seq 1 40); do
                if curl -s http://127.0.0.1:8080/health | grep -q '"ok"'; then
                    READY=1
                    break
                fi
                echo -n "."
                sleep 0.5
            done
            if [ "$READY" -eq 1 ]; then
                echo " ready"
            else
                echo " FAILED (Check logs/llama-server.log)"
                exit 1
            fi
        else
            echo "[LLM] llama-server binary or model not found at configured paths."
        fi
    fi

    # Always pre-warm llama-server KV cache slot with active personality and tools schema
    echo "[LLM] Pre-warming llama-server KV cache slot with active personality & tools..."
    "$DIR/.venv/bin/python" -c "
import urllib.request, json
try:
    from robot.personality import PersonalityManager
    from robot.core.context import ContextEngine, format_system_prompt
    from robot.tools import ToolRegistry, ToolRouter
    from robot.tools.builtins import register_builtin_tools
    import yaml

    pm = PersonalityManager('config/personalities')
    pm.load_all()
    def_p = pm.get_persisted_active('neo')
    pm.set_active(def_p)
    pers = pm.active

    sys_prompt = pers.system_prompt if pers else 'You are a helpful robot.'
    system_content = format_system_prompt(sys_prompt, high_intelligence=False)

    class MockEngine:
        def __init__(self):
            self.tool_registry = ToolRegistry()
            self.config = {}

    mock = MockEngine()
    register_builtin_tools(mock)
    silent_arrival_prompt = (
        '[System Event: Robot Core online and sensors active. The user has arrived / entered the room. '
        'In your next response, warmly acknowledge and greet the user.]'
    )
    router = ToolRouter(mock.tool_registry)
    selected_tools = router.select_tools(silent_arrival_prompt)
    tools_schema = mock.tool_registry.get_schemas(selected_tools) if selected_tools else None
    msgs = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': silent_arrival_prompt},
    ]

    payload = json.dumps({
        'model': 'Qwen3.5-4B-Q4_K_M',
        'messages': msgs,
        'tools': tools_schema,
        'stream': True,
        'max_tokens': 15,
        'chat_template_kwargs': {'enable_thinking': False},
        'temperature': 0.7
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:8080/v1/chat/completions',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        for line in resp:
            pass
    print(f'[LLM] KV cache pre-warmed for personality \"{def_p}\" (silent arrival prompt primed)')
except Exception as e:
    print('[LLM] Warmup notice:', e)
" 2>/dev/null || true
fi

# ─── 3. whisper-server (AMD Radeon 680M Vulkan, port 8081) ──────────
WHISPER_BIN=""
if [ -f "/home/ameya/whisper.cpp/build_vk/bin/whisper-server" ]; then
    WHISPER_BIN="/home/ameya/whisper.cpp/build_vk/bin/whisper-server"
elif [ -f "/home/ameya/whisper.cpp/build/bin/whisper-server" ]; then
    WHISPER_BIN="/home/ameya/whisper.cpp/build/bin/whisper-server"
fi

WHISPER_MODEL=""
if [ -f "/home/ameya/whisper.cpp/models/ggml-tiny.en.bin" ]; then
    WHISPER_MODEL="/home/ameya/whisper.cpp/models/ggml-tiny.en.bin"
elif [ -f "/home/ameya/whisper.cpp/models/ggml-base.bin" ]; then
    WHISPER_MODEL="/home/ameya/whisper.cpp/models/ggml-base.bin"
elif [ -f "/home/ameya/whisper.cpp/models/ggml-small.bin" ]; then
    WHISPER_MODEL="/home/ameya/whisper.cpp/models/ggml-small.bin"
fi

if ! curl -s http://127.0.0.1:8081/ > /dev/null 2>&1; then
    if [ -n "$WHISPER_BIN" ] && [ -n "$WHISPER_MODEL" ]; then
        echo "[STT] Starting whisper-server on AMD Radeon 680M iGPU (Vulkan dev 0) with $WHISPER_MODEL..."
        "$WHISPER_BIN" \
            -m "$WHISPER_MODEL" \
            -dev 0 \
            -fa \
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

# ─── 4. Mem0 Vector Storage ─────────────────────────────────────────
mkdir -p "$DIR/data/memory/qdrant"
echo "[MEM] Persistent embedded Qdrant storage ready at data/memory/qdrant"
export FASTEMBED_CACHE_PATH="$DIR/models/fastembed"

# ─── 5. Launch Robot Core & Display UI ──────────────────────────────
echo ""
echo "🚀 Launching Robot Core..."

# Launch browser display in background once server is up
(
    for i in $(seq 1 30); do
        if curl -s http://127.0.0.1:8888/health > /dev/null 2>&1; then
            if command -v xdg-open &>/dev/null; then
                xdg-open "http://127.0.0.1:8888" >/dev/null 2>&1 &
            elif command -v google-chrome &>/dev/null; then
                google-chrome "http://127.0.0.1:8888" >/dev/null 2>&1 &
            fi
            break
        fi
        sleep 0.5
    done
) &

export PYTHONWARNINGS="ignore::DeprecationWarning"

if [ -f "$DIR/.venv/bin/python" ]; then
    "$DIR/.venv/bin/python" -m robot "$@"
else
    python3 -m robot "$@"
fi

