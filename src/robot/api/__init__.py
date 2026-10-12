"""FastAPI server — HTTP + WebSocket API for Robot Core.

Provides:
- REST endpoints for status, memory, personality
- WebSocket for real-time event streams
- Developer console / AI Terminal access
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import Body, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = structlog.get_logger()

from robot.core.state import RobotMode

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None
    personality: str | None = None


class ChatSpeakRequest(BaseModel):
    """Request model for synthesizing speech audio for a chat message."""
    text: str
    personality: str | None = None
    voice_model: str | None = None
    speed: float = 1.0
    pitch: float = 1.0
    play_hardware: bool = False


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


class ImageGenRequest(BaseModel):
    prompt: str
    width: int | None = 512
    height: int | None = 512
    steps: int | None = 4
    seed: int | None = -1


class EqualizerRequest(BaseModel):
    preset: str


class MusicVolumeRequest(BaseModel):
    volume: float


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
            await engine.events.emit("personality_warming", personality=name)
            if hasattr(engine, "warm_up_llm") and callable(getattr(engine, "warm_up_llm")):
                coro = engine.warm_up_llm(name)
                if asyncio.iscoroutine(coro):
                    asyncio.create_task(coro)
            return {"status": "switched", "personality": name}
        return {"error": f"Personality '{name}' not found"}

    # ─── LLM Provider Swapping & Management Endpoints ───────────────────────
    @app.get("/api/llm/status")
    async def get_llm_status():
        """Get active LLM provider, mode, context window, and llama-server process status."""
        prov = getattr(engine, "llm_provider_type", "local")
        if prov in ("llama-server", "local"):
            prov = "local"
        is_gemini = prov == "gemini"
        from robot.llm import LlamaServerManager
        llama_running = LlamaServerManager.is_running()
        local_ctx = int(getattr(engine, "config", {}).get("llm", {}).get("context_window", 16384)) if hasattr(engine, "config") else 16384
        openai_cfg = getattr(engine, "config", {}).get("llm", {}).get("openai_compatible", {}) if hasattr(engine, "config") else {}
        is_openai = prov in ("openai_compatible", "openai")
        
        ctx_win = local_ctx
        if is_gemini:
            ctx_win = 1000000
        elif is_openai:
            ctx_win = int(openai_cfg.get("context_window", 128000))

        active_model = getattr(engine.llm, "model", "Qwen3.5-4B-Q4_K_M")
        return {
            "provider": prov,
            "mode": "high_intelligence" if (is_gemini or is_openai) else "low_latency",
            "model": active_model,
            "context_window": ctx_win,
            "llama_running": llama_running,
            "vram_freed": not llama_running,
            "openai_compatible": {
                "base_url": openai_cfg.get("base_url", "https://api.openai.com/v1"),
                "model": getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "gpt-4o")),
                "context_window": int(openai_cfg.get("context_window", 128000)),
                "has_api_key": bool(openai_cfg.get("api_key")),
                "enable_prompt_caching": bool(openai_cfg.get("enable_prompt_caching", True)),
            },
        }

    @app.post("/api/llm/toggle")
    async def toggle_llm_provider():
        """Toggle between Local, Cloud Gemini, and OpenAI-Compatible."""
        current = getattr(engine, "llm_provider_type", "local")
        if current in ("local", "llama-server"):
            target = "gemini"
        elif current == "gemini":
            target = "openai_compatible"
        else:
            target = "local"
        res = await engine.switch_llm_provider(target)
        return res

    @app.post("/api/llm/provider")
    async def set_llm_provider(req: dict[str, Any]):
        """Set specific LLM provider ('local', 'gemini', or 'openai_compatible')."""
        target = req.get("provider", "local")
        res = await engine.switch_llm_provider(target)
        return res

    @app.get("/api/llm/openai/models")
    async def scan_openai_models(base_url: str | None = None, api_key: str | None = None):
        """Scan available models from the OpenAI-compatible endpoint.

        Args:
            base_url: Optional override endpoint URL to scan.
            api_key: Optional override API key.

        Returns:
            dict[str, Any]: Status and list of discovered model identifiers.
        """
        from robot.llm import OpenAICompatibleLLM
        openai_cfg = getattr(engine, "config", {}).get("llm", {}).get("openai_compatible", {}) if hasattr(engine, "config") else {}
        target_url = base_url or openai_cfg.get("base_url", "https://api.openai.com/v1")
        target_key = api_key if api_key is not None else openai_cfg.get("api_key", "")
        current_model = getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "gpt-4o"))

        scanner = OpenAICompatibleLLM(
            base_url=target_url,
            api_key=target_key,
            model=current_model,
        )
        try:
            models = await scanner.scan_models()
            return {
                "status": "ok",
                "models": models,
                "count": len(models),
                "current_model": current_model,
                "active_model": current_model,
                "base_url": target_url,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "models": [],
                "count": 0,
                "current_model": current_model,
                "active_model": current_model,
                "base_url": target_url,
            }

    @app.post("/api/llm/openai/model")
    async def set_openai_model(req: dict[str, Any]):
        """Set active model for the OpenAI-compatible provider.

        Args:
            req: Payload containing 'model' identifier string.

        Returns:
            dict[str, Any]: Status confirmation and selected model.
        """
        from robot.config import update_env_variable
        new_model = req.get("model", "").strip()
        if not new_model:
            return {"status": "error", "message": "Model name cannot be empty."}

        if hasattr(engine, "_persist_openai_model"):
            engine._persist_openai_model(new_model)
        update_env_variable("OPENAI_COMPATIBLE_MODEL", new_model)

        if hasattr(engine, "config"):
            engine.config.setdefault("llm", {}).setdefault("openai_compatible", {})["model"] = new_model

        if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
            engine.llm.model = new_model

        await engine.events.emit("llm.model_changed", provider="openai_compatible", model=new_model)
        return {"status": "ok", "model": new_model, "active_model": new_model}

    @app.post("/api/llm/openai/config")
    async def configure_openai_compatible(req: dict[str, Any]):
        """Update and persist OpenAI-compatible provider settings to .env.

        Args:
            req: Configuration payload containing base_url, api_key, model,
                context_window, and enable_prompt_caching.

        Returns:
            dict[str, Any]: Status summary and confirmation.
        """
        from robot.config import update_env_variable
        base_url = req.get("base_url", "").strip()
        api_key = req.get("api_key", "").strip()
        model = req.get("model", "").strip()
        ctx_win = req.get("context_window")
        prompt_caching = req.get("enable_prompt_caching", True)

        openai_cfg = getattr(engine, "config", {}).setdefault("llm", {}).setdefault("openai_compatible", {})

        if base_url:
            openai_cfg["base_url"] = base_url
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", base_url)
        if api_key is not None and api_key != "":
            openai_cfg["api_key"] = api_key
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", api_key)
        if model:
            openai_cfg["model"] = model
            if hasattr(engine, "_persist_openai_model"):
                engine._persist_openai_model(model)
            update_env_variable("OPENAI_COMPATIBLE_MODEL", model)
        if ctx_win:
            try:
                int_ctx = int(ctx_win)
                openai_cfg["context_window"] = int_ctx
                update_env_variable("OPENAI_COMPATIBLE_CONTEXT_WINDOW", str(int_ctx))
            except ValueError:
                pass
        openai_cfg["enable_prompt_caching"] = bool(prompt_caching)
        update_env_variable("OPENAI_COMPATIBLE_ENABLE_PROMPT_CACHING", "true" if prompt_caching else "false")

        # If currently active, refresh live LLM instance
        if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
            from robot.llm import OpenAICompatibleLLM
            engine.llm = OpenAICompatibleLLM(
                base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
                api_key=openai_cfg.get("api_key", ""),
                model=openai_cfg.get("model", "gpt-4o"),
                context_window=openai_cfg.get("context_window", 128000),
                enable_prompt_caching=openai_cfg.get("enable_prompt_caching", True),
            )

        return {
            "status": "ok",
            "message": "OpenAI-compatible configuration updated and saved to .env",
            "config": {
                "base_url": openai_cfg.get("base_url"),
                "model": openai_cfg.get("model"),
                "context_window": openai_cfg.get("context_window"),
                "enable_prompt_caching": openai_cfg.get("enable_prompt_caching"),
                "has_api_key": bool(openai_cfg.get("api_key")),
            },
        }

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

    @app.get("/api/equalizer")
    async def get_equalizer():
        """Get active equalizer profile and list of all available EasyEffects presets."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            return MusicPlayerService.get_instance().list_equalizer_presets()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/api/equalizer")
    async def set_equalizer(req: EqualizerRequest):
        """Set active equalizer profile using EasyEffects."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            return MusicPlayerService.get_instance().set_equalizer_preset(req.preset)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.get("/api/music/status")
    async def get_music_status():
        """Get live music playback status and active equalizer profile."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            return {
                "status": ms.get_status(),
                "equalizer": ms.list_equalizer_presets(),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/api/music/toggle")
    async def toggle_music_playback():
        """Toggle music playback between playing and paused."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            if ms.is_playing:
                res = ms.pause()
            else:
                res = ms.resume()
            return {"status": "ok", "playback": res, "is_playing": ms.is_playing}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/api/music/restart")
    async def restart_music_track():
        """Restart playback of the current or last played track."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            res = ms.replay()
            return {"status": "ok", "playback": res, "is_playing": ms.is_playing}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/api/music/volume")
    async def set_music_volume(req: MusicVolumeRequest):
        """Set output volume level for music playback."""
        try:
            from robot.tools.builtins.music import MusicPlayerService
            ms = MusicPlayerService.get_instance()
            res = ms.set_volume(req.volume)
            return {"status": "ok", "volume": ms._volume, "result": res}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.get("/api/memory/stats")
    async def get_memory_stats():
        """Get live memory facts count, chat messages count, and vector store health."""
        if hasattr(engine, "memory") and engine.memory:
            stats = await engine.memory.get_stats()
            return {"status": "ok", **stats}
        return {"status": "ok", "facts_count": 0, "messages_count": 0, "sessions_count": 0}

    @app.get("/chat/sessions")
    async def list_chat_sessions(limit: int = 150):
        """List all isolated conversation sessions stored in the SQLite database."""
        if hasattr(engine, "memory") and engine.memory:
            sessions = await engine.memory.list_sessions(limit=limit)
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
            sess = await engine.memory.get_session(session_id)
            msgs = await engine.memory.get_session_messages(session_id, limit=300)
            if not sess and not msgs:
                raise HTTPException(status_code=404, detail="Session not found")
            title = (sess.get("title") if sess else None) or "Conversation"
            personality = (sess.get("personality") if sess else None) or "neo"
            return {
                "session_id": session_id,
                "id": session_id,
                "title": title,
                "personality": personality,
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

    @app.post("/api/chat/speak")
    async def speak_chat_message(req: ChatSpeakRequest) -> Response:
        """Synthesize and return speech WAV audio for a chat message response.

        Converts message text into 16-bit PCM WAV audio using the active or specified
        Piper TTS voice model and prosody configuration. Emits a robot phrase event
        for synchronized avatar lipsync animation.

        Args:
            req: ChatSpeakRequest containing text, personality, voice_model, speed, pitch.

        Returns:
            Response containing binary audio/wav content for in-browser playback.

        Raises:
            HTTPException: 400 if text or audio is empty; 503 if TTS is unavailable.
        """
        import io
        import wave
        import numpy as np

        raw_text = req.text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        tts = getattr(engine, "tts", None)
        if tts is None or not hasattr(tts, "synthesize"):
            raise HTTPException(status_code=503, detail="TTS provider is not initialized or unavailable.")

        # Resolve voice model and prosody from personality if not explicitly overridden
        pers_name = req.personality or getattr(engine.state.state, "active_personality", "neo")
        pers = engine.personalities.get(pers_name) if hasattr(engine, "personalities") and engine.personalities else None

        voice_model = req.voice_model or (getattr(pers, "voice_model", None) if pers else None)
        speed = req.speed
        pitch = req.pitch
        noise_scale: float | None = None
        noise_w: float | None = None

        if pers and hasattr(pers, "tts_config") and isinstance(pers.tts_config, dict):
            cfg = pers.tts_config
            if req.speed == 1.0 and "speed" in cfg:
                speed = float(cfg["speed"])
            if req.pitch == 1.0 and "pitch" in cfg:
                pitch = float(cfg["pitch"])
            noise_scale = cfg.get("noise_scale")
            noise_w = cfg.get("noise_w")

        try:
            audio, sr = await tts.synthesize(
                raw_text,
                speed=speed,
                pitch=pitch,
                voice_model=voice_model,
                noise_scale=noise_scale,
                noise_w=noise_w,
            )
        except Exception as exc:
            logger.error("chat_speak_synthesis_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}")

        if audio is None or len(audio) == 0:
            raise HTTPException(status_code=400, detail="Synthesized audio was empty.")

        # Optionally play on robot hardware speaker if requested
        if req.play_hardware and hasattr(engine, "player") and engine.player:
            asyncio.create_task(engine.player.play(audio, sr))

        # Emit robot_phrase event for avatar lip-sync / expressions
        if hasattr(engine, "events") and engine.events:
            try:
                from robot.tts import clean_text_for_speech
                clean_phrase = clean_text_for_speech(raw_text)
                await engine.events.emit("robot_phrase", robot_phrase=clean_phrase)
            except Exception as e:
                logger.debug("speak_phrase_event_failed", error=str(e))

        # Convert float32 numpy array to 16-bit PCM WAV
        int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int16_audio.tobytes())
        wav_bytes = wav_buf.getvalue()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="response.wav"',
                "Cache-Control": "no-cache",
            },
        )

    @app.post("/api/transcribe")
    async def transcribe_audio(request: Request):
        """Transcribe uploaded voice recording (WAV binary or JSON base64) via Whisper STT for chat dictation."""
        try:
            ctype = request.headers.get("content-type", "")
            if "application/json" in ctype:
                data = await request.json()
                import base64
                b64_audio = data.get("audio", "")
                content = base64.b64decode(b64_audio) if b64_audio else b""
            else:
                content = await request.body()

            if not content or len(content) < 44:
                return {"status": "ok", "text": ""}

            from robot.stt import STTProvider, WhisperServerSTT
            stt = getattr(engine, "stt", None)
            if stt is None or not hasattr(stt, "transcribe"):
                stt = WhisperServerSTT()

            import io
            import wave
            import numpy as np

            try:
                with wave.open(io.BytesIO(content), "rb") as wf:
                    channels = wf.getnchannels()
                    width = wf.getsampwidth()
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())

                    if width == 2:
                        pcm_int16 = np.frombuffer(frames, dtype=np.int16)
                    elif width == 4:
                        pcm_int16 = (np.frombuffer(frames, dtype=np.int32) >> 16).astype(np.int16)
                    else:
                        pcm_int16 = np.frombuffer(frames, dtype=np.int8).astype(np.int16) << 8

                    if channels > 1:
                        pcm_int16 = pcm_int16.reshape(-1, channels).mean(axis=1).astype(np.int16)

                    pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

                    # Resample if sample rate != 16000
                    if rate != 16000 and len(pcm_float32) > 0:
                        target_len = int(len(pcm_float32) * 16000 / rate)
                        pcm_float32 = np.interp(
                            np.linspace(0, len(pcm_float32), target_len, endpoint=False),
                            np.arange(len(pcm_float32)),
                            pcm_float32,
                        ).astype(np.float32)

                    import inspect
                    res = stt.transcribe(pcm_float32, sample_rate=16000)
                    text = await res if inspect.isawaitable(res) else str(res)
                    return {"status": "ok", "text": text}
            except wave.Error:
                import asyncio
                import requests
                def _forward():
                    files = {"file": ("audio.wav", content, "audio/wav")}
                    data = {"temperature": "0.0", "language": getattr(stt, "language", "en"), "response_format": "json"}
                    url = getattr(stt, "url", "http://127.0.0.1:8081/inference")
                    return requests.post(url, files=files, data=data, timeout=8)
                resp = await asyncio.to_thread(_forward)
                if resp.status_code == 200:
                    raw_text = resp.json().get("text", "").strip()
                    from robot.stt import clean_transcript
                    return {"status": "ok", "text": clean_transcript(raw_text)}
                return {"status": "error", "error": f"Whisper server returned {resp.status_code}", "text": ""}
        except Exception as e:
            logger.error("api_transcribe_error", error=str(e))
            return {"status": "error", "error": str(e), "text": ""}

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
            try:
                while True:
                    if hasattr(engine, "perception") and engine.perception:
                        frame = engine.perception.get_annotated_frame()
                        if frame is not None:
                            success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            if success:
                                yield (b"--frame\r\n"
                                       b"Content-Type: image/jpeg\r\n\r\n" + enc.tobytes() + b"\r\n")
                    await asyncio.sleep(1.0 / 60.0)
            except (asyncio.CancelledError, GeneratorExit):
                return

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

    @app.get("/observability/benchmarks")
    async def get_benchmarks():
        """Get latency distributions (P50, P95, P99) categorized by request type."""
        if hasattr(engine, "observability") and engine.observability:
            return engine.observability.get_category_distributions()
        return {}

    # ─── Image Generation API ────────────────────────────────────────────────
    @app.post("/images/generate")
    async def api_generate_image(req: ImageGenRequest = Body(...)):
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

    # --- Temporal Scheduler Endpoints ---
    @app.get("/api/scheduler/jobs")
    async def list_scheduled_jobs(job_type: str | None = None, status: str = "pending", limit: int = 50):
        """List scheduled timers, alarms, and reminders."""
        if hasattr(engine, "scheduler") and engine.scheduler:
            jobs = await engine.scheduler.list_jobs(job_type=job_type, status=status, limit=limit)
            return {"status": "success", "jobs": [j.to_dict() for j in jobs]}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.post("/api/scheduler/timer")
    async def create_timer(request: Request):
        """Create a new countdown timer."""
        data = await request.json()
        duration = float(data.get("duration_seconds", 60))
        label = data.get("label", "Timer")
        if hasattr(engine, "scheduler") and engine.scheduler:
            job = await engine.scheduler.add_timer(duration, label=label)
            return {"status": "success", "timer": job.to_dict()}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.post("/api/scheduler/alarm")
    async def create_alarm(request: Request):
        """Schedule a new clock alarm."""
        data = await request.json()
        target_time = data.get("target_time", "")
        label = data.get("label", "Alarm")
        recurring = bool(data.get("recurring", False))
        if hasattr(engine, "scheduler") and engine.scheduler:
            try:
                job = await engine.scheduler.add_alarm(target_time, label=label, recurring=recurring)
                return {"status": "success", "alarm": job.to_dict()}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.post("/api/scheduler/reminder")
    async def create_reminder(request: Request):
        """Schedule a new reminder."""
        data = await request.json()
        time_expr = data.get("time_expression", "")
        message = data.get("message", "")
        if hasattr(engine, "scheduler") and engine.scheduler:
            try:
                job = await engine.scheduler.add_reminder(time_expr, message=message)
                return {"status": "success", "reminder": job.to_dict()}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.delete("/api/scheduler/jobs/{job_id}")
    async def cancel_scheduled_job(job_id: str):
        """Cancel a scheduled timer, alarm, or reminder."""
        if hasattr(engine, "scheduler") and engine.scheduler:
            cancelled = await engine.scheduler.cancel_job(job_id)
            return {"status": "success" if cancelled else "not_found", "job_id": job_id, "cancelled": cancelled}
        return {"status": "error", "message": "Scheduler service not initialized"}

    @app.get("/api/scheduler/briefing")
    async def get_scheduler_briefing():
        """Get executive daily briefing with system status and upcoming events."""
        if hasattr(engine, "scheduler") and engine.scheduler:
            briefing = await engine.scheduler.get_daily_briefing()
            return {"status": "success", "briefing": briefing}
        return {"status": "error", "message": "Scheduler service not initialized"}

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
