"""FastAPI server — HTTP + WebSocket API for Robot Core.

Provides:
- REST endpoints for status, memory, personality
- WebSocket for real-time event streams
- Developer console / AI Terminal access
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from robot.core.state import RobotMode

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None
    personality: str | None = None


class ChatResponse(BaseModel):
    response: str
    turn: int
    personality: str
    session_id: str | None = None


class CreateSessionRequest(BaseModel):
    personality: str | None = None
    title: str | None = None


class StatusResponse(BaseModel):
    mode: str
    personality: str
    turn_count: int
    voice_enabled: bool = True
    services: dict[str, bool]
    telemetry: dict[str, Any]
    is_speaking: bool = False
    music_playing: bool = False


class VoiceModeRequest(BaseModel):
    enabled: bool


def create_app(engine: "RobotEngine") -> FastAPI:
    """Create the FastAPI application with all routes."""
    app = FastAPI(
        title="Robot Core API",
        description="Local-first humanoid robot brain API",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/status", response_model=StatusResponse)
    async def status():
        state = engine.state.state
        player_active = hasattr(engine, "player") and engine.player and engine.player.is_playing
        speaking_active = getattr(engine, "_is_speaking", False)
        if state.mode == RobotMode.SPEAKING and not player_active and not speaking_active:
            await engine.state.set_mode(RobotMode.IDLE)
            state = engine.state.state

        music_playing = False
        try:
            from robot.tools.builtins.music import MusicPlayerService
            music_playing = MusicPlayerService.get_instance().is_playing
        except Exception:
            pass

        return StatusResponse(
            mode=state.mode.value,
            personality=state.active_personality,
            turn_count=state.turn_count,
            voice_enabled=getattr(engine, "voice_enabled", True),
            services={
                name: svc.healthy for name, svc in state.services.items()
            },
            telemetry=engine.telemetry.snapshot(),
            is_speaking=bool(speaking_active or player_active),
            music_playing=bool(music_playing),
        )

    @app.post("/voice-mode")
    async def set_voice_mode(req: VoiceModeRequest):
        """Toggle voice/mic listening on or off for text-only chatting."""
        engine.voice_enabled = bool(req.enabled)
        # Physically pause/resume mic stream to free audio hardware and ensure no audio is captured
        if hasattr(engine, "capture") and engine.capture:
            if engine.voice_enabled:
                engine.capture.resume()
            else:
                engine.capture.pause()
        await engine.events.emit("voice_mode_changed", voice_enabled=engine.voice_enabled)
        return {"status": "ok", "voice_enabled": engine.voice_enabled}

    @app.get("/personalities")
    async def list_personalities():
        engine.personalities.load_all()
        return {
            "personalities": engine.personalities.list_personalities(),
            "active": engine.state.state.active_personality,
        }

    @app.post("/personality/{name}")
    async def switch_personality(name: str):
        if name not in engine.personalities.list_personalities():
            engine.personalities.load_all()
        success = engine.personalities.set_active(name)
        if success:
            engine.state.state.active_personality = name
            pers = engine.personalities.get(name)
            if pers and getattr(pers, "voice_model", None) and hasattr(engine.tts, "set_active_voice"):
                engine.tts.set_active_voice(pers.voice_model)
            await engine.events.emit("personality_switched", personality=name)
            if hasattr(engine, "warm_up_llm") and callable(getattr(engine, "warm_up_llm")):
                try:
                    await engine.warm_up_llm(name)
                except Exception as e:
                    logger.debug("personality_warmup_error", error=str(e))
            return {"status": "switched", "personality": name}
        return {"error": f"Personality '{name}' not found"}

    # ─── LLM Provider Swapping & Management Endpoints ───────────────────────
    @app.get("/api/llm/status")
    async def get_llm_status():
        """Get active LLM provider, mode, context window, and llama-server process status."""
        prov = getattr(engine, "llm_provider_type", "local")
        if prov in ("llama-server", "local"):
            prov = "local"
        from robot.llm import LlamaServerManager
        llama_running = LlamaServerManager.is_running()
        is_gemini = prov == "gemini"
        return {
            "provider": prov,
            "mode": "high_intelligence" if is_gemini else "low_latency",
            "model": getattr(engine.llm, "model", "gemini-flash-latest" if is_gemini else "Qwen3.5-4B-Q4_K_M"),
            "context_window": 1000000 if is_gemini else 4096,
            "llama_running": llama_running,
            "vram_freed": not llama_running,
        }

    @app.post("/api/llm/toggle")
    async def toggle_llm_provider():
        """Toggle between Local (RTX 3050) and Cloud Gemini (1M context)."""
        current = getattr(engine, "llm_provider_type", "local")
        target = "gemini" if current in ("local", "llama-server") else "local"
        res = await engine.switch_llm_provider(target)
        return res

    @app.post("/api/llm/provider")
    async def set_llm_provider(req: dict[str, Any]):
        """Set specific LLM provider ('local' or 'gemini')."""
        target = req.get("provider", "local")
        res = await engine.switch_llm_provider(target)
        return res

    # ─── Spotify OAuth & Integration Endpoints ──────────────────────────────
    @app.get("/spotify/status")
    async def spotify_status():
        """Get Spotify configuration, OAuth status, and devices."""
        from robot.tools.builtins.music import MusicPlayerService
        ms = MusicPlayerService.get_instance()
        sp_cfg = ms.config.get("spotify", {})
        client_id = bool(sp_cfg.get("client_id"))
        client_secret = bool(sp_cfg.get("client_secret"))
        sp_oauth = ms._get_spotify_oauth_client()
        authenticated = sp_oauth is not None
        devices = []
        if sp_oauth:
            try:
                devices = sp_oauth.devices().get("devices", [])
            except Exception:
                pass
        return {
            "configured": client_id and client_secret,
            "authenticated": authenticated,
            "device_id": sp_cfg.get("device_id") or "",
            "available_devices": devices,
            "login_url": "/spotify/login",
        }

    @app.get("/spotify/login")
    async def spotify_login():
        """Get or redirect to Spotify OAuth login page."""
        from fastapi.responses import RedirectResponse
        from robot.tools.builtins.music import MusicPlayerService
        ms = MusicPlayerService.get_instance()
        sp_cfg = ms.config.get("spotify", {})
        client_id = sp_cfg.get("client_id")
        client_secret = sp_cfg.get("client_secret")
        redirect_uri = sp_cfg.get("redirect_uri", "http://127.0.0.1:8888/callback")
        if not client_id or not client_secret:
            return {"error": "Spotify client_id or client_secret not configured"}
        try:
            from spotipy.oauth2 import SpotifyOAuth
            scope = "user-modify-playback-state user-read-playback-state user-read-currently-playing"
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
                open_browser=False,
                cache_path=".spotify_cache",
            )
            auth_url = auth_manager.get_authorize_url()
            return RedirectResponse(auth_url)
        except Exception as e:
            return {"error": str(e)}

    @app.get("/callback")
    async def spotify_callback(code: str | None = None, error: str | None = None):
        """Handle Spotify OAuth authorization redirect callback."""
        from fastapi.responses import HTMLResponse
        if error:
            return HTMLResponse(
                f"""<html><body style="font-family: sans-serif; background: #121212; color: #fff; padding: 2rem;">
                <h2 style="color: #ff5555;">Spotify Authorization Error</h2>
                <p>{error}</p>
                <a href="/" style="color: #1db954;">Return to Dashboard</a>
                </body></html>""",
                status_code=400,
            )
        if not code:
            return HTMLResponse(
                """<html><body style="font-family: sans-serif; background: #121212; color: #fff; padding: 2rem;">
                <h2>No authorization code provided.</h2>
                <a href="/" style="color: #1db954;">Return to Dashboard</a>
                </body></html>""",
                status_code=400,
            )
        from robot.tools.builtins.music import MusicPlayerService
        ms = MusicPlayerService.get_instance()
        success, msg = ms.handle_spotify_callback(code)
        if success:
            return HTMLResponse(
                """<html><body style="font-family: sans-serif; background: #121212; color: #fff; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                <div style="background: #1e1e1e; padding: 2.5rem; border-radius: 12px; text-align: center; border: 1px solid #1db954; max-width: 480px; box-shadow: 0 8px 24px rgba(0,0,0,0.5);">
                <h2 style="color: #1db954; margin-top: 0;">Spotify Connected Successfully!</h2>
                <p style="color: #ccc; line-height: 1.5;">Robot Core has authorized Spotify Connect playback. You can now command music playback seamlessly.</p>
                <a href="/" style="display: inline-block; margin-top: 1rem; background: #1db954; color: #000; text-decoration: none; padding: 0.6rem 1.5rem; border-radius: 20px; font-weight: bold;">Return to Dashboard</a>
                </div></body></html>"""
            )
        else:
            return HTMLResponse(
                f"""<html><body style="font-family: sans-serif; background: #121212; color: #fff; padding: 2rem;">
                <h2 style="color: #ff5555;">Spotify Authorization Failed</h2>
                <p>{msg}</p>
                <a href="/" style="color: #1db954;">Return to Dashboard</a>
                </body></html>""",
                status_code=500,
            )

    @app.get("/chat/sessions")
    async def list_chat_sessions():
        """List all isolated conversation sessions stored in the SQLite database."""
        if hasattr(engine, "memory") and engine.memory:
            sessions = await engine.memory.list_sessions(limit=50)
            return {"sessions": sessions, "active_session": engine._session_id}
        return {"sessions": [], "active_session": engine._session_id}

    @app.post("/chat/sessions")
    async def create_chat_session(req: CreateSessionRequest | None = None):
        """Create a new isolated chat session in the database."""
        sid = f"chat_{str(uuid.uuid4())[:8]}"
        pers = (req and req.personality) or getattr(getattr(engine, "state", None), "active_personality", "neo")
        title = (req and req.title) or "New Chat"
        if hasattr(engine, "memory") and engine.memory:
            sess = await engine.memory.create_session(sid, personality=pers, title=title)
            return sess
        return {"id": sid, "session_id": sid, "personality": pers, "title": title}

    @app.get("/chat/sessions/{session_id}")
    async def get_chat_session(session_id: str):
        """Get all messages for an isolated chat session from the database."""
        if hasattr(engine, "memory") and engine.memory:
            sessions = await engine.memory.list_sessions(limit=100)
            matching = next((s for s in sessions if s["id"] == session_id), None)
            if not matching:
                raise HTTPException(status_code=404, detail="Session not found")
            msgs = await engine.memory.get_session_messages(session_id, limit=200)
            return {
                "session_id": session_id,
                "id": session_id,
                "title": matching.get("title", "Conversation"),
                "messages": msgs,
            }
        return {"session_id": session_id, "id": session_id, "title": "Conversation", "messages": []}

    @app.delete("/chat/sessions/{session_id}")
    async def delete_chat_session(session_id: str):
        """Delete an isolated chat session and its conversation history."""
        if hasattr(engine, "memory") and engine.memory:
            success = await engine.memory.delete_session(session_id)
            return {"status": "ok", "deleted": success, "session_id": session_id}
        return {"status": "ok", "deleted": False, "session_id": session_id}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """Server-Sent Events (SSE) streaming endpoint for real-time tokens, token speed, and tool events."""
        from fastapi.responses import StreamingResponse
        import json

        async def _event_generator():
            async for chunk in engine.stream_text_chat(
                text=req.text,
                session_id=req.session_id,
                personality=req.personality,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """Text-based chat endpoint (non-voice)."""
        response = await engine.process_text_input(req.text, session_id=req.session_id)
        return ChatResponse(
            response=response,
            turn=engine.state.state.turn_count,
            personality=engine.state.state.active_personality,
            session_id=req.session_id or engine._session_id,
        )

    @app.post("/interrupt")
    async def interrupt_execution():
        """Manually interrupt active LLM generation, tool execution, and audio playback."""
        engine.trigger_interrupt()
        return {"status": "interrupted"}

    @app.get("/memory/search")
    async def search_memory(query: str, limit: int = 5):
        results = await engine.memory.get_relevant_memories(query, limit)
        return {"results": results}

    @app.get("/history/search")
    async def search_history(query: str, limit: int = 10):
        results = await engine.memory.search_history(query, limit)
        return {"results": results}

    @app.get("/world-state")
    async def get_world_state():
        """Get the current physical world state (people, motion, lighting)."""
        if hasattr(engine, "perception") and engine.perception:
            return engine.perception.world_state.to_compact()
        return {"error": "Perception subsystem not active"}

    @app.get("/camera/snapshot")
    async def get_camera_snapshot():
        """Get the latest snapshot from the robot's camera with green perception polygons."""
        if hasattr(engine, "perception") and engine.perception:
            import cv2
            from fastapi.responses import Response

            frame = engine.perception.get_annotated_frame()
            if frame is not None:
                success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    return Response(content=enc.tobytes(), media_type="image/jpeg")
            return Response(content=b"No frame available", status_code=503)
        return Response(content=b"Camera not available", status_code=503)

    @app.get("/camera/stream")
    async def get_camera_stream():
        """Ultra-low latency 60 FPS continuous MJPEG video stream with green polygons."""
        from fastapi.responses import StreamingResponse
        import asyncio
        import cv2

        async def frame_generator():
            while True:
                if hasattr(engine, "perception") and engine.perception:
                    frame = engine.perception.get_annotated_frame()
                    if frame is not None:
                        success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                        if success:
                            yield (b"--frame\r\n"
                                   b"Content-Type: image/jpeg\r\n\r\n" + enc.tobytes() + b"\r\n")
                await asyncio.sleep(1.0 / 60.0)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )


    @app.post("/character/action")
    async def post_character_action(action_req: dict[str, Any]):
        """Execute semantic motion command on character."""
        from robot.motion.actions import validate_action
        from robot.motion.planner import MotionPlanner

        valid, semantic_action, err = validate_action(action_req)
        if not valid or semantic_action is None:
            return {"error": err}

        planner = MotionPlanner()
        sequence = planner.plan_action(action_req)
        serialized_seq = [a.to_dict() for a in sequence]

        await engine.events.emit(
            "character.action",
            action=semantic_action.to_dict(),
            sequence=serialized_seq,
        )
        return {"status": "ok", "action": semantic_action.to_dict(), "sequence": serialized_seq}

    @app.post("/character/stop")
    async def post_character_stop():
        """Interrupt and smoothly stop character motion."""
        await engine.events.emit("character.interrupt", interrupt=True)
        return {"status": "stopped"}

    @app.get("/telemetry")
    async def telemetry():
        return engine.telemetry.snapshot()

    # ─── Observability & Traces API ──────────────────────────────────────────
    @app.get("/observability/traces")
    async def get_traces(limit: int = 15):
        """Get recent execution traces with spans and latencies."""
        if hasattr(engine, "observability") and engine.observability:
            return {
                "traces": engine.observability.get_recent_traces(limit),
                "pipeline": engine.observability.pipeline_state,
            }
        return {"traces": [], "pipeline": {}}

    @app.get("/observability/hardware")
    async def get_hardware_metrics():
        """Get live CPU, RAM, RTX 3050 VRAM/Temp, and AMD 680M utilization."""
        if hasattr(engine, "observability") and engine.observability:
            return engine.observability.get_hardware_metrics()
        return {}

    @app.get("/observability/pipeline")
    async def get_pipeline_state():
        """Get live status of all pipeline components."""
        if hasattr(engine, "observability") and engine.observability:
            return engine.observability.pipeline_state
        return {}

    # ─── Image Generation API ────────────────────────────────────────────────
    class ImageGenRequest(BaseModel):
        prompt: str
        width: int | None = 512
        height: int | None = 512
        steps: int | None = 4
        seed: int | None = -1

    @app.post("/images/generate")
    async def api_generate_image(req: ImageGenRequest):
        """Asynchronously submit an image generation job."""
        if not hasattr(engine, "image_service") or not engine.image_service:
            return {"error": "Image service not initialized"}
        job = engine.image_service.submit_job(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.steps or 4,
            seed=req.seed or -1,
            trace_id=getattr(engine.observability, "current_trace_id", None) if hasattr(engine, "observability") else None,
        )
        return {"status": "queued", "job": job.to_dict()}

    @app.get("/images/jobs")
    async def list_image_jobs(limit: int = 20):
        """List recent image generation jobs."""
        if hasattr(engine, "image_service") and engine.image_service:
            return {"jobs": engine.image_service.list_jobs(limit)}
        return {"jobs": []}

    @app.post("/images/cancel/{job_id}")
    async def cancel_image_job(job_id: str):
        """Cancel an in-flight image generation job."""
        if hasattr(engine, "image_service") and engine.image_service:
            ok = await engine.image_service.cancel_job(job_id)
            return {"cancelled": ok, "job_id": job_id}
        return {"cancelled": False, "error": "Image service not initialized"}

    # Mount /data/images for generated image previews
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles

    images_dir = Path("data/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/data/images", StaticFiles(directory=str(images_dir)), name="images")

    @app.websocket("/ws/events")
    @app.websocket("/ws/observability")
    async def event_stream(websocket: WebSocket):
        """WebSocket stream for real-time telemetry and observability events."""
        await websocket.accept()
        import asyncio
        import json

        queue: asyncio.Queue = asyncio.Queue()

        async def forward_event(**kwargs):
            await queue.put(kwargs)

        engine.events.on("*", forward_event)

        try:
            while True:
                event = await queue.get()
                await websocket.send_text(json.dumps(event))
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            pass
        finally:
            engine.events.off("*", forward_event)

    # Static UI Files & Web HUD Display
    from fastapi.responses import FileResponse

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def index():
            """Serve the integrated Digital Face & Dev Console UI."""
            return FileResponse(str(static_dir / "index.html"))

    return app
