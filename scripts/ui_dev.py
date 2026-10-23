"""UI development server: serves the real frontend against a stubbed backend.

Lets browser-based UI work iterate without running llama-server, whisper, piper,
the camera or the engine. Run: ``.venv/bin/python scripts/ui_dev.py [port]``
then open ``http://127.0.0.1:8901/`` — like the real API it admits loopback
clients without a token. Set ``ROBOT_API_TRUST_LOOPBACK=0`` to exercise the
"Console locked" gate instead, and use ``?token=uidev`` for a non-loopback bind.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time

os.environ.setdefault("ROBOT_API_TOKEN", "uidev")

import numpy as np
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "robot", "api", "static")

app = FastAPI()

from robot.api import _is_trusted_local_scope, _trust_loopback_enabled  # noqa: E402

_PUBLIC_PREFIXES = ("/health", "/auth/", "/static/", "/favicon")


class _StubEngine:
    """Engine-shaped object so the dev harness reads the same trust switch as prod."""

    config: dict = {}


def _is_trusted_local(request: Request) -> bool:
    """Mirror production exactly: reuse the API's own trusted-local predicate."""
    return _trust_loopback_enabled(_StubEngine()) and _is_trusted_local_scope(request.scope)


def _presents_token(request: Request) -> bool:
    """True when the request carries the dev token in a header, cookie or query."""
    return (
        request.headers.get("authorization", "") == "Bearer uidev"
        or request.cookies.get("robot_api_session") == "uidev"
        or request.query_params.get("token") == "uidev"
    )


@app.middleware("http")
async def fake_auth(request: Request, call_next):
    """Mirror the production API: reject unauthenticated calls so the UI's
    cookie/bearer plumbing is exercised identically in dev."""
    path = request.url.path
    ok = (
        path == "/"
        or path.startswith(_PUBLIC_PREFIXES)
        or _presents_token(request)
        or _is_trusted_local(request)
    )
    if not ok:
        return JSONResponse({"detail": "Invalid API token."}, status_code=401)
    return await call_next(request)

_START = time.time()
_JOBS: dict[str, dict] = {}


def _telemetry() -> dict:
    def m(ms: float) -> dict:
        return {
            "last": round(ms + random.uniform(-2, 2), 1),
            "avg": round(ms, 1),
            "p50": round(ms * 0.9, 1),
            "p95": round(ms * 1.4, 1),
            "p99": round(ms * 1.8, 1),
            "target": round(ms * 2, 1),
            "status": "ok",
            "count": random.randint(40, 300),
        }

    return {
        "stt": m(240),
        "llm_first_token": m(620),
        "llm_total": m(2100),
        "tts_synthesis": m(85),
        "audio_playback_latency": m(40),
        "turn_total": m(3200),
    }


@app.get("/")
async def index() -> HTMLResponse:
    with open(os.path.join(STATIC_DIR, "index.html"), "rb") as fh:
        return HTMLResponse(fh.read())


@app.get("/contextlayer")
async def contextlayer() -> HTMLResponse:
    path = os.path.join(STATIC_DIR, "contextlayer.html")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return HTMLResponse(fh.read())
    return HTMLResponse("<h1>contextlayer stub</h1>")


from fastapi.staticfiles import StaticFiles  # noqa: E402

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "uptime_s": round(time.time() - _START)}


@app.get("/auth/status")
async def auth_status(request: Request) -> dict:
    local = _is_trusted_local(request)
    ok = local or _presents_token(request)
    return {
        "status": "ok",
        "auth_required": not local,
        "authenticated": ok,
        "via": "trusted_loopback" if local else "token",
        "accepted": ["authorization_bearer", "query_token", "session_cookie"],
    }


@app.post("/auth/login")
async def auth_login(response: Response) -> dict:
    response.set_cookie(key="robot_api_session", value="uidev", max_age=86400, samesite="lax", path="/")
    return {"status": "ok", "authenticated": True, "max_age": 86400}


@app.post("/auth/logout")
async def auth_logout() -> dict:
    return {"status": "ok"}


@app.get("/api/thinking")
@app.get("/api/llm/openai/models")
@app.get("/api/chat/speak/stream")
async def generic_get(request: Request) -> dict:
    return {"ok": True, "models": [], "enabled": True}


@app.get("/status")
async def status() -> dict:
    mode = "idle"
    elapsed = time.time() - _START
    if 8 <= elapsed % 20 < 12:
        mode = "speaking"
    elif 14 <= elapsed % 20 < 16:
        mode = "listening"
    return {
        "mode": mode,
        "personality": "neo",
        "turn_count": int(elapsed // 20),
        "voice_enabled": True,
        "services": {
            "llama-server": True,
            "whisper-server": True,
            "piper-tts": True,
            "camera": True,
            "memory": True,
            "spotify": False,
        },
        "telemetry": _telemetry(),
        "is_speaking": mode == "speaking",
        "music_playing": False,
    }


@app.get("/world-state")
async def world_state() -> dict:
    t = time.time() - _START
    return {
        "people_count": 1,
        "people": [
            {
                "id": "p1",
                "position": [0.5 + 0.2 * math.sin(t / 7), 0.45, 1.6],
                "distance": round(1.6 + 0.15 * math.sin(t / 5), 2),
                "looking_at_robot": True,
                "norm_x": round(0.5 + 0.18 * math.sin(t / 6), 3),
                "norm_y": 0.42,
            }
        ],
        "objects_count": 3,
        "objects": [
            {"label": "cup", "confidence": 0.91, "position": [0.2, 0.8]},
            {"label": "laptop", "confidence": 0.87, "position": [0.6, 0.7]},
            {"label": "chair", "confidence": 0.76, "position": [0.85, 0.5]},
        ],
        "tabletop_items": [],
        "tabletop_count": 0,
        "motion": "still",
        "lighting": "indoor-warm",
        "scene": "desk-room",
        "time_since_update_s": round(random.uniform(0.1, 0.9), 1),
    }


def _synthetic_jpeg(width: int = 640, height: int = 480) -> bytes:
    import cv2

    t = time.time()
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (24, 18, 16)
    cx = int(width * (0.5 + 0.18 * math.sin(t / 3)))
    cy = int(height * (0.42 + 0.05 * math.sin(t / 2.1)))
    r = 90
    cv2.ellipse(img, (cx, cy), (r, int(r * 1.3)), 0, 0, 360, (60, 90, 130), -1)
    cv2.circle(img, (cx - 32, cy - 22), 8, (200, 220, 240), -1)
    cv2.circle(img, (cx + 32, cy - 22), 8, (200, 220, 240), -1)
    cv2.ellipse(img, (cx, cy + 36), (26, 12), 0, 0, 180, (40, 40, 180), 3)
    cv2.putText(img, "UI DEV CAMERA STUB", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 120), 1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


async def _mjpeg() -> bytes:
    while True:
        frame = _synthetic_jpeg()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        await asyncio.sleep(1 / 15)


@app.get("/camera/stream")
async def camera_stream() -> StreamingResponse:
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera/face-mesh")
async def camera_face_mesh() -> StreamingResponse:
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/face-mesh/status")
async def face_mesh_status() -> dict:
    return {"enabled": True, "available": True, "fps": 29.6, "faces": 1, "last_error": None}


@app.post("/api/face-mesh/toggle")
async def face_mesh_toggle() -> dict:
    return {"enabled": True}


@app.get("/api/llm/status")
async def llm_status() -> dict:
    return {
        "provider": "llama-server",
        "running": True,
        "model": "qwen3.5-4b-q4_k_m.gguf",
        "models": ["qwen3.5-4b-q4_k_m.gguf", "llama-3.2-3b-instruct.gguf"],
        "openrouter": {"has_api_key": True, "model": "meta-llama/llama-3.3-70b"},
        "aicredits": {"has_api_key": False, "model": None},
        "slots": [{"id": 0, "n_ctx": 4096, "n_kv_used": 1420, "state": "idle"}],
        "context_window": {"used": 1420, "max": 4096},
        "ttft_ms": 658,
    }


@app.get("/api/llm/settings")
async def llm_settings() -> dict:
    return {
        "n_ctx": 4096,
        "n_gpu_layers": 99,
        "flash_attn": True,
        "threads": 8,
        "batch_size": 512,
        "temp": 0.7,
        "top_p": 0.9,
    }


@app.get("/api/llm/models")
async def llm_models() -> dict:
    return {
        "local": [
            {"name": "qwen3.5-4b-q4_k_m.gguf", "size_gb": 2.6, "active": True},
            {"name": "llama-3.2-3b-instruct.gguf", "size_gb": 2.0, "active": False},
        ],
        "provider": "llama-server",
    }


@app.post("/api/llm/provider")
@app.post("/api/llm/openai/model")
@app.post("/api/llm/settings")
@app.post("/api/llm/settings/reset")
@app.post("/api/models/local/switch")
@app.post("/api/llm/hf/download")
@app.post("/api/llm/hf/cancel")
async def generic_ok() -> dict:
    return {"ok": True}


@app.get("/api/llm/openai/config")
async def openai_config() -> dict:
    return {"base_url": "http://127.0.0.1:1234/v1", "model": "qwen3.5-4b", "has_api_key": False}


@app.get("/api/llm/hf/downloads")
async def hf_downloads() -> dict:
    return {
        "tasks": [
            {
                "id": "dl-1",
                "repo_id": "Qwen/Qwen3.5-4B-GGUF",
                "file": "qwen3.5-4b-q4_k_m.gguf",
                "status": "downloading",
                "progress": 0.62,
                "speed_mbps": 14.2,
            }
        ]
    }


@app.get("/api/llm/hf/search")
async def hf_search(q: str = "") -> dict:
    return {
        "results": [
            {"repo_id": f"SomeOrg/{q or 'model'}", "name": f"{q or 'model'}-GGUF", "downloads": 12000, "likes": 88}
        ]
    }


@app.get("/api/stt/models")
async def stt_models() -> dict:
    return {"models": ["ggml-large-v3-turbo.bin", "ggml-medium.bin"], "active": "ggml-large-v3-turbo.bin"}


@app.get("/api/tts/status")
async def tts_status() -> dict:
    return {"engine": "piper", "voice": "amy", "sample_rate": 22050, "running": True}


@app.get("/api/tts/voices")
async def tts_voices() -> dict:
    return {
        "voices": ["amy", "bryce", "kathleen", "hfc_female", "lessac"],
        "active": "amy",
        "engine": "piper",
        "kokoro_voices": ["af_heart", "am_adam"],
    }


@app.post("/api/tts/switch")
@app.post("/api/tts/test")
@app.post("/api/stt/switch")
@app.post("/api/thinking")
@app.post("/api/view-mode")
@app.post("/camera-mode")
@app.post("/voice-mode")
@app.post("/character/action")
@app.post("/interrupt")
@app.post("/personality/{name}")
@app.post("/api/equalizer")
@app.post("/api/music/toggle")
@app.post("/api/music/restart")
@app.post("/api/music/volume")
@app.post("/api/chat/speak/stream")
@app.post("/api/chat/speak")
@app.post("/chat/sessions")
@app.delete("/chat/sessions/{session_id}")
@app.patch("/chat/sessions/{session_id}")
@app.post("/chat/sessions/{session_id}")
async def generic_ok2(request: Request) -> dict:
    return {"ok": True}


@app.get("/api/personas")
async def personas() -> dict:
    return {
        "personas": [
            {
                "id": "neo",
                "name": "Cypher",
                "voice": "amy",
                "active": True,
                "turns": 341,
                "facts": 87,
                "sessions": 12,
                "description": "Precise, calm operator",
            },
            {"id": "aria", "name": "Aria", "voice": "bryce", "active": False, "turns": 92, "facts": 20, "sessions": 4, "description": "Warm and playful"},
            {"id": "glados", "name": "GLaDOS", "voice": "kathleen", "active": False, "turns": 55, "facts": 9, "sessions": 2, "description": "Sardonic AI"},
            {"id": "jarvis", "name": "Jarvis", "voice": "hfc_female", "active": False, "turns": 12, "facts": 3, "sessions": 1, "description": "Butler protocol"},
            {"id": "luna", "name": "Luna", "voice": "lessac", "active": False, "turns": 7, "facts": 2, "sessions": 1, "description": "Dreamy companion"},
        ]
    }


@app.post("/api/personas/{persona_id}/activate")
async def activate_persona(persona_id: str) -> dict:
    return {"ok": True, "active": persona_id}


@app.get("/api/memory/stats")
async def memory_stats() -> dict:
    return {"facts": 87, "messages": 4312, "sessions": 12, "db_size_mb": 8.4}


@app.get("/api/music/status")
async def music_status() -> dict:
    return {"playing": False, "track": None, "volume": 0.6}


@app.get("/chat/sessions")
async def chat_sessions() -> dict:
    now = time.time()
    return {
        "sessions": [
            {"id": f"sess-{i}", "title": t, "created_at": now - 86400 * (i + 1), "updated_at": now - 3600 * i, "turns": 8 - i}
            for i, t in enumerate(
                [
                    "Debugging serial bridge",
                    "Weekend project ideas",
                    "Camera FPS tuning",
                    "Persona switching test",
                    "Voice pipeline latency",
                ]
            )
        ]
    }


@app.get("/chat/sessions/{session_id}")
async def chat_session_detail(session_id: str) -> dict:
    return {
        "id": session_id,
        "messages": [
            {"role": "user", "content": "How is the KV cache reuse looking today?"},
            {"role": "assistant", "content": "Warm turns are reusing **96.8%** of the prefix cache; turn-1 TTFT stayed around *658 ms*. I trimmed two stale facts from memory during the last standby cycle."},
        ],
    }


@app.get("/history/search")
async def history_search(q: str = "") -> dict:
    return {"results": [{"role": "assistant", "content": f"...matching “{q}” in session sess-2", "created_at": time.time() - 7200}]}


@app.get("/api/images/gallery")
async def image_gallery() -> dict:
    return {
        "images": [
            {"name": f"gen_{i:03d}.png", "created_at": time.time() - 3600 * i, "prompt": f"cyberdeck concept art {i}"}
            for i in range(6)
        ]
    }


@app.get("/api/images/models")
async def image_models() -> dict:
    return {"models": ["sdxl-turbo", "flux-schnell"], "active": "sdxl-turbo"}


@app.get("/images/jobs")
async def image_jobs() -> dict:
    return {"jobs": list(_JOBS.values())}


@app.post("/images/generate")
async def image_generate() -> dict:
    jid = f"job-{int(time.time())}"
    _JOBS[jid] = {"id": jid, "status": "queued", "prompt": "stub", "created_at": time.time()}
    return {"job_id": jid}


@app.post("/images/cancel/{job_id}")
async def image_cancel(job_id: str) -> dict:
    return {"ok": True}


@app.get("/observability/traces")
async def traces(limit: int = 20) -> dict:
    now = time.time()
    out = []
    for i in range(min(limit, 8)):
        out.append(
            {
                "trace_id": f"tr-{i:04d}",
                "started_at": now - 60 * (i + 1),
                "duration_ms": random.randint(900, 4200),
                "status": "ok" if i % 4 else "error",
                "spans": [
                    {"name": "stt", "ms": 240},
                    {"name": "llm", "ms": 1800},
                    {"name": "tts", "ms": 90},
                ],
            }
        )
    return {"traces": out}


@app.get("/observability/hardware")
async def hardware() -> dict:
    t = time.time()
    return {
        "cpu_percent": round(35 + 20 * math.sin(t / 4), 1),
        "cpu_temp_c": round(61 + 4 * math.sin(t / 6), 1),
        "ram_used_gb": round(11.2 + math.sin(t / 8), 2),
        "ram_total_gb": 32,
        "gpu": {
            "name": "RTX 3050",
            "util_percent": round(44 + 30 * abs(math.sin(t / 5)), 1),
            "vram_used_gb": 3.7,
            "vram_total_gb": 4.0,
            "temp_c": 67,
        },
        "igpu": {"name": "Radeon 680M", "util_percent": round(20 + 15 * abs(math.sin(t / 7)), 1)},
        "disk_free_gb": 118,
    }


@app.get("/api/transcribe")
async def transcribe_stub() -> dict:
    return {"error": "not available in ui_dev"}


async def _sse_chunks(request: Request):
    for chunk in ["Hello ", "from ", "the ", "stub ", "backend."]:
        yield f"data: {json.dumps({'text': chunk})}\n\n"
        await asyncio.sleep(0.15)
    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
@app.get("/chat/stream")
async def chat_stream(request: Request):
    return StreamingResponse(_sse_chunks(request), media_type="text/event-stream")


@app.get("/api/equalizer/status")
async def equalizer_status() -> dict:
    return {"bands": [0.0] * 8, "enabled": True}


@app.post("/api/equalizer")
async def equalizer_post() -> dict:
    return {"ok": True}


@app.websocket("/ws/events")
@app.websocket("/ws/observability")
async def ws_fake(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(
                {
                    "type": "tick",
                    "ts": time.time(),
                    "payload": {"mode": "idle", "is_speaking": False},
                }
            )
            await asyncio.sleep(2)
    except Exception:
        return


if __name__ == "__main__":
    import sys

    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8901
    print(f"UI dev server: http://127.0.0.1:{port}/?token=uidev")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
