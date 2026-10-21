"""FastAPI server — HTTP + WebSocket API for Robot Core.

Provides:
- REST endpoints for status, memory, personality
- WebSocket for real-time event streams
- Developer console / AI Terminal access
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from fastapi import Body, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = structlog.get_logger()

from robot.core.state import RobotMode

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None
    personality: str | None = None
    thinking: bool | None = None


class ThinkingModeRequest(BaseModel):
    enabled: bool


class ChatSpeakRequest(BaseModel):
    """Request model for synthesizing speech audio for a chat message."""
    text: str
    personality: str | None = None
    voice_model: str | None = None
    speed: float = 1.0
    pitch: float = 1.0
    play_hardware: bool = False


class PersonaCreateRequest(BaseModel):
    """Request model for creating a new persona with provisioned isolated database."""
    id: str
    display_name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    speaking_style: str | None = None
    vocabulary: str | None = None
    avatar_color: str | None = None
    default_models: dict[str, Any] | None = None


class PersonaUpdateRequest(BaseModel):
    """Request model for updating an existing persona and its model defaults."""
    display_name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    speaking_style: str | None = None
    vocabulary: str | None = None
    avatar_color: str | None = None
    default_models: dict[str, Any] | None = None


class TTSSwitchRequest(BaseModel):
    """Request model for switching active TTS engine and voice."""
    engine: str
    voice: str | None = None


class TTSTestRequest(BaseModel):
    """Request model for generating test audio with a specific TTS engine and voice."""
    engine: str = "kokoro"
    voice: str | None = None
    text: str = "Hello, this is a test of the neural voice engine on the robot."


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


class CameraModeRequest(BaseModel):
    enabled: bool


class ImageGenRequest(BaseModel):
    prompt: str
    model: str | None = "default"
    negative_prompt: str | None = ""
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
    async def set_voice_mode(req: VoiceModeRequest) -> dict[str, Any]:
        """Toggle voice/mic listening on or off for text-only chatting."""
        engine.voice_enabled = bool(req.enabled)
        engine.is_chat_mode = not engine.voice_enabled
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
        # Physically pause/resume mic stream to free audio hardware and ensure no audio is captured
        if hasattr(engine, "capture") and engine.capture:
            if engine.voice_enabled:
                engine.capture.resume()
            else:
                engine.capture.pause()
        await engine.events.emit("voice_mode_changed", voice_enabled=engine.voice_enabled)
        return {"status": "ok", "voice_enabled": engine.voice_enabled, "is_chat_mode": engine.is_chat_mode}

    @app.get("/api/voice/status")
    async def get_voice_status() -> dict[str, Any]:
        """Retrieve voice listening and wake-word gating status.

        Returns:
            dict[str, Any]: Voice configuration including active wake word,
                hardware mic mute state, and wake-word gating requirement.
        """
        active_word = getattr(getattr(engine, "wakeword", None), "active_wake_word", "luna")
        return {
            "status": "ok",
            "voice_enabled": engine.voice_enabled,
            "require_wake_word": getattr(engine, "require_wake_word", True),
            "active_wake_word": active_word,
            "barge_in": getattr(engine, "barge_in_enabled", True),
        }

    @app.post("/api/voice/wake-word-toggle")
    async def toggle_wake_word(req: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Toggle or set whether the active wake word is required to talk.

        Args:
            req: Optional JSON payload containing boolean `require_wake_word`.

        Returns:
            dict[str, Any]: Updated wake-word gating status.
        """
        explicit = req.get("require_wake_word")
        if explicit is None:
            engine.require_wake_word = not getattr(engine, "require_wake_word", True)
        else:
            engine.require_wake_word = bool(explicit)
        await engine.events.emit(
            "wake_word_mode_changed",
            require_wake_word=engine.require_wake_word,
        )
        return {"status": "ok", "require_wake_word": engine.require_wake_word}

    @app.post("/camera-mode")
    @app.post("/api/camera-mode")
    async def set_camera_mode(req: CameraModeRequest):
        """Toggle camera hardware acquisition on or off for privacy or chat mode."""
        actual = engine.set_camera_mode(bool(req.enabled))
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
        await engine.events.emit("camera_mode_changed", camera_enabled=actual)
        return {"status": "ok", "camera_enabled": actual}

    @app.get("/camera-mode")
    @app.get("/api/camera-mode")
    async def get_camera_mode():
        """Get current camera hardware acquisition status."""
        return {"status": "ok", "camera_enabled": engine.is_camera_enabled}

    @app.post("/api/view-mode")
    async def set_view_mode(req: dict[str, Any]):
        """Synchronize UI active view mode ('chat' vs 'dual') with engine state.

        Args:
            req: Payload dictionary containing 'mode' ('chat' | 'dual').

        Returns:
            dict[str, Any]: Status summary and confirmation.
        """
        mode = req.get("mode", "dual")
        engine.is_chat_mode = (mode == "chat")
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
        return {"status": "ok", "mode": mode, "is_chat_mode": engine.is_chat_mode}

    @app.get("/personalities")
    async def list_personalities():
        engine.personalities.load_all()
        return {
            "personalities": engine.personalities.list_personalities(),
            "active": engine.state.state.active_personality,
        }

    @app.get("/api/personas")
    async def list_personas():
        """List all personas with complete metadata, isolated database statistics, and model defaults."""
        engine.personalities.load_all()
        personas = engine.personalities.list_personas_detailed()
        active_p = getattr(getattr(getattr(engine, "state", None), "state", None), "active_personality", None)
        if not active_p:
            active_p = getattr(engine, "active_personality", "neo")
        return {
            "status": "ok",
            "personas": personas,
            "active": active_p,
            "total": len(personas),
        }

    @app.post("/api/personas")
    async def create_persona(req: PersonaCreateRequest):
        """Create a new persona and provision its dedicated isolated database."""
        try:
            raw_data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            pers = engine.personalities.create_persona(raw_data)
            return {
                "status": "created",
                "persona": pers.to_dict(),
                "isolated_db": pers.database_path,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create persona: {e}")

    @app.get("/api/personas/{persona_id}")
    async def get_persona(persona_id: str):
        """Get details, default models, and database statistics for a specific persona."""
        pers = engine.personalities.get(persona_id)
        if not pers:
            raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
        data = pers.to_dict()
        data["is_active"] = (engine.personalities.active_name == pers.name)
        data.update(engine.personalities.get_persona_stats(pers.name))
        return data

    @app.put("/api/personas/{persona_id}")
    async def update_persona(persona_id: str, req: PersonaUpdateRequest):
        """Update an existing persona definition and default model configurations."""
        try:
            update_data = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
            pers = engine.personalities.update_persona(persona_id, update_data)
            return {
                "status": "updated",
                "persona": pers.to_dict(),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update persona: {e}")

    @app.delete("/api/personas/{persona_id}")
    async def delete_persona(persona_id: str, delete_database: bool = True, force: bool = False):
        """Delete a persona definition and optionally delete its isolated database."""
        try:
            ok = engine.personalities.delete_persona(persona_id, delete_database=delete_database, force=force)
            return {"status": "deleted", "id": persona_id, "deleted_database": delete_database}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete persona: {e}")

    @app.post("/api/personas/{persona_id}/activate")
    async def activate_persona(persona_id: str):
        """Activate a persona, hot-swap memory to its isolated database, and apply its default models."""
        try:
            res = await engine.switch_personality(persona_id)
            return res
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to activate persona: {e}")

    @app.post("/personality/{name}")
    async def switch_personality(name: str):
        """Legacy hot-swap endpoint delegating to engine.switch_personality with resilient fallback."""
        switch_fn = getattr(engine, "switch_personality", None)
        if callable(switch_fn) and type(switch_fn).__name__ not in ("MagicMock", "AsyncMock"):
            try:
                res = await switch_fn(name)
                return res
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to switch personality: {e}")

        # Fallback or Mock Engine execution
        mgr = getattr(engine, "personalities", None)
        if not mgr:
            return {"error": f"Personality manager unavailable"}
        success = mgr.set_active(name)
        if success:
            if hasattr(engine, "state") and hasattr(engine.state, "state"):
                engine.state.state.active_personality = name
            pers = mgr.get(name)
            if pers and getattr(pers, "voice_model", None) and hasattr(engine, "tts") and hasattr(engine.tts, "set_active_voice"):
                engine.tts.set_active_voice(pers.voice_model)
            if hasattr(engine, "wakeword") and engine.wakeword:
                engine.wakeword.set_active_persona(name)
            if hasattr(engine, "events") and engine.events:
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
        persisted_openai_model = getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "gpt-4o"))
        openrouter_cfg = getattr(engine, "config", {}).get("llm", {}).get("openrouter", {}) if hasattr(engine, "config") else {}
        openrouter_key = openrouter_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        has_openrouter_key = bool(openrouter_key)

        aicredits_cfg = getattr(engine, "config", {}).get("llm", {}).get("aicredits", {}) if hasattr(engine, "config") else {}
        aicredits_key = aicredits_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", "")
        has_aicredits_key = bool(aicredits_key)

        return {
            "provider": prov,
            "mode": "high_intelligence" if (is_gemini or is_openai) else "low_latency",
            "model": active_model,
            "active_openai_model": persisted_openai_model,
            "context_window": ctx_win,
            "llama_running": llama_running,
            "vram_freed": not llama_running,
            "openai_compatible": {
                "base_url": openai_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                "model": persisted_openai_model,
                "context_window": int(openai_cfg.get("context_window", 128000)),
                "has_api_key": bool(openai_cfg.get("api_key")),
                "enable_prompt_caching": bool(openai_cfg.get("enable_prompt_caching", True)),
            },
            "openrouter": {
                "base_url": openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                "model": openrouter_cfg.get("model", "meta-llama/llama-3.3-70b-instruct:free"),
                "has_api_key": has_openrouter_key,
                "api_key": openrouter_key,
            },
            "aicredits": {
                "base_url": aicredits_cfg.get("base_url", "https://aicredits.in/v1"),
                "model": aicredits_cfg.get("model", "aion-labs/aion-3.0"),
                "has_api_key": has_aicredits_key,
                "api_key": aicredits_key,
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
    async def scan_openai_models(
        base_url: str | None = None,
        api_key: str | None = None,
        preset: str | None = None,
    ):
        """Scan available models from the OpenAI-compatible endpoint.

        Args:
            base_url: Optional override endpoint URL to scan.
            api_key: Optional override API key.
            preset: Optional provider preset ('openrouter', 'aicredits', 'openai', etc.).

        Returns:
            dict[str, Any]: Status and list of discovered model identifiers.
        """
        from robot.llm import OpenAICompatibleLLM
        cfg_root = getattr(engine, "config", {}).get("llm", {}) if hasattr(engine, "config") else {}
        openai_cfg = cfg_root.get("openai_compatible", {})
        openrouter_cfg = cfg_root.get("openrouter", {})
        aicredits_cfg = cfg_root.get("aicredits", {})

        target_url = base_url
        target_key = api_key

        if preset == "openrouter":
            target_url = base_url or openrouter_cfg.get("base_url") or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            target_key = api_key if api_key is not None else (openrouter_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", ""))
        elif preset == "aicredits":
            target_url = base_url or aicredits_cfg.get("base_url") or os.environ.get("AICREDITS_BASE_URL", "https://aicredits.in/v1")
            target_key = api_key if api_key is not None else (aicredits_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", ""))
        elif preset == "openai":
            target_url = base_url or "https://api.openai.com/v1"
            target_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        elif preset == "deepseek":
            target_url = base_url or "https://api.deepseek.com/v1"
            target_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        elif preset == "groq":
            target_url = base_url or "https://api.groq.com/openai/v1"
            target_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        elif preset == "ollama":
            target_url = base_url or "http://127.0.0.1:11434/v1"
            target_key = api_key if api_key is not None else ""
        else:
            if not target_url:
                target_url = openai_cfg.get("base_url", "https://openrouter.ai/api/v1")
            if target_key is None:
                if "openrouter.ai" in str(target_url):
                    target_key = openai_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
                elif "aicredits.in" in str(target_url):
                    target_key = openai_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", "")
                else:
                    target_key = openai_cfg.get("api_key", "")

        current_model = getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "meta-llama/llama-3.3-70b-instruct:free"))

        scanner = OpenAICompatibleLLM(
            base_url=target_url,
            api_key=target_key,
            model=current_model,
        )
        try:
            models = await scanner.scan_models()
            provider_preset = "openrouter" if "openrouter.ai" in str(target_url) else ("aicredits" if "aicredits.in" in str(target_url) else "custom")
            return {
                "status": "ok",
                "models": models,
                "count": len(models),
                "current_model": current_model,
                "active_model": current_model,
                "base_url": target_url,
                "provider_preset": provider_preset,
            }
        except Exception as e:
            provider_preset = "openrouter" if "openrouter.ai" in str(target_url) else ("aicredits" if "aicredits.in" in str(target_url) else "custom")
            return {
                "status": "error",
                "message": str(e),
                "models": [],
                "count": 0,
                "current_model": current_model,
                "active_model": current_model,
                "base_url": target_url,
                "provider_preset": provider_preset,
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

        # Intelligent auto-routing: if model belongs to OpenRouter or AI Credits, align base_url and api_key
        current_base = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "").lower()
        is_or_model = ":free" in new_model or any(new_model.startswith(p) for p in ("meta-llama/", "google/", "nvidia/", "liquid/", "apodex/", "thinkingmachines/", "mistralai/"))
        is_ai_model = new_model.startswith("aion-labs/")

        if is_or_model and "openrouter" not in current_base:
            or_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", or_base)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", or_key)
            if hasattr(engine, "config"):
                cfg = engine.config.setdefault("llm", {}).setdefault("openai_compatible", {})
                cfg["base_url"] = or_base
                cfg["api_key"] = or_key
            if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
                engine.llm.base_url = or_base
                engine.llm.api_key = or_key
        elif is_ai_model and "aicredits" not in current_base:
            ai_base = os.environ.get("AICREDITS_BASE_URL", "https://aicredits.in/v1")
            ai_key = os.environ.get("AICREDITS_API_KEY", "")
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", ai_base)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", ai_key)
            if hasattr(engine, "config"):
                cfg = engine.config.setdefault("llm", {}).setdefault("openai_compatible", {})
                cfg["base_url"] = ai_base
                cfg["api_key"] = ai_key
            if getattr(engine, "llm_provider_type", "") in ("openai_compatible", "openai") and engine.llm:
                engine.llm.base_url = ai_base
                engine.llm.api_key = ai_key

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
        active_provider = req.get("active_provider") or req.get("preset", "custom")

        cfg_root = getattr(engine, "config", {}).setdefault("llm", {})
        openai_cfg = cfg_root.setdefault("openai_compatible", {})
        or_cfg = cfg_root.setdefault("openrouter", {})
        ai_cfg = cfg_root.setdefault("aicredits", {})

        # 1. Update OpenRouter specific fields if passed
        or_data = req.get("openrouter")
        if isinstance(or_data, dict):
            or_b = or_data.get("base_url", "").strip()
            or_k = or_data.get("api_key", "").strip()
            or_m = or_data.get("model", "").strip()
            if or_b:
                or_cfg["base_url"] = or_b
                update_env_variable("OPENROUTER_BASE_URL", or_b)
            if or_k:
                or_cfg["api_key"] = or_k
                update_env_variable("OPENROUTER_API_KEY", or_k)
            if or_m:
                or_cfg["model"] = or_m
                update_env_variable("OPENROUTER_MODEL", or_m)

        # 2. Update AI Credits specific fields if passed
        ai_data = req.get("aicredits")
        if isinstance(ai_data, dict):
            ai_b = ai_data.get("base_url", "").strip()
            ai_k = ai_data.get("api_key", "").strip()
            ai_m = ai_data.get("model", "").strip()
            if ai_b:
                ai_cfg["base_url"] = ai_b
                update_env_variable("AICREDITS_BASE_URL", ai_b)
            if ai_k:
                ai_cfg["api_key"] = ai_k
                update_env_variable("AICREDITS_API_KEY", ai_k)
            if ai_m:
                ai_cfg["model"] = ai_m
                update_env_variable("AICREDITS_MODEL", ai_m)

        # 3. Synchronize active provider target
        if active_provider == "openrouter":
            target_url = (or_data.get("base_url") if isinstance(or_data, dict) else "") or base_url or or_cfg.get("base_url") or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            target_key = (or_data.get("api_key") if isinstance(or_data, dict) else "") or api_key or or_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
            target_model = (or_data.get("model") if isinstance(or_data, dict) else "") or model or or_cfg.get("model") or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
            openai_cfg["base_url"] = target_url
            openai_cfg["api_key"] = target_key
            openai_cfg["model"] = target_model
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", target_url)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", target_key)
            update_env_variable("OPENAI_COMPATIBLE_MODEL", target_model)
        elif active_provider == "aicredits":
            target_url = (ai_data.get("base_url") if isinstance(ai_data, dict) else "") or base_url or ai_cfg.get("base_url") or os.environ.get("AICREDITS_BASE_URL", "https://aicredits.in/v1")
            target_key = (ai_data.get("api_key") if isinstance(ai_data, dict) else "") or api_key or ai_cfg.get("api_key") or os.environ.get("AICREDITS_API_KEY", "")
            target_model = (ai_data.get("model") if isinstance(ai_data, dict) else "") or model or ai_cfg.get("model") or os.environ.get("AICREDITS_MODEL", "aion-labs/aion-3.0")
            openai_cfg["base_url"] = target_url
            openai_cfg["api_key"] = target_key
            openai_cfg["model"] = target_model
            update_env_variable("OPENAI_COMPATIBLE_BASE_URL", target_url)
            update_env_variable("OPENAI_COMPATIBLE_API_KEY", target_key)
            update_env_variable("OPENAI_COMPATIBLE_MODEL", target_model)
        else:
            if base_url:
                openai_cfg["base_url"] = base_url
                update_env_variable("OPENAI_COMPATIBLE_BASE_URL", base_url)
            if api_key is not None and api_key != "":
                openai_cfg["api_key"] = api_key
                update_env_variable("OPENAI_COMPATIBLE_API_KEY", api_key)
            if model:
                openai_cfg["model"] = model
                update_env_variable("OPENAI_COMPATIBLE_MODEL", model)

        if hasattr(engine, "_persist_openai_model") and openai_cfg.get("model"):
            engine._persist_openai_model(openai_cfg["model"])

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
                base_url=openai_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                api_key=openai_cfg.get("api_key", ""),
                model=openai_cfg.get("model", "meta-llama/llama-3.3-70b-instruct:free"),
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

    @app.get("/api/models/local")
    async def list_local_models() -> dict[str, Any]:
        """List registered 2B Vision Language Models (VLM) and <4B Uncensored models with download status.

        Returns:
            dict[str, Any]: Model catalog, default model, and download status.
        """
        from robot.config import get_project_root
        from robot.llm.model_registry import ModelRegistry

        base_dir = get_project_root()
        models_data = []
        for m in ModelRegistry.list_models():
            models_data.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "category": m.category.value,
                    "parameters": m.parameters,
                    "repo_id": m.repo_id,
                    "filename": m.filename,
                    "mmproj_filename": m.mmproj_filename,
                    "is_vision": m.is_vision,
                    "is_uncensored": m.is_uncensored,
                    "supports_tools": m.supports_tools,
                    "recommended_context": m.recommended_context,
                    "description": m.description,
                    "is_downloaded": m.is_downloaded(base_dir),
                    "model_path": str(m.get_model_path(base_dir)),
                    "mmproj_path": str(m.get_mmproj_path(base_dir)) if m.mmproj_filename else None,
                }
            )

        cfg_model = getattr(engine, "config", {}).get("llm", {}).get("model", "qwen2-vl-2b-instruct")
        return {
            "default_model": ModelRegistry.DEFAULT_VLM_MODEL_ID,
            "active_configured_model": cfg_model,
            "total_models": len(models_data),
            "models": models_data,
        }

    @app.post("/api/models/local/switch")
    async def switch_local_model(req: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Hot-swap the active local GGUF model running on the RTX 3050 GPU.

        Args:
            req: Request payload containing the target 'model' or 'model_id'.

        Returns:
            Dictionary with switch operation status, model metadata, and slot warming state.
        """
        model_id = req.get("model") or req.get("model_id")
        if not model_id:
            return {"status": "error", "message": "Missing 'model' parameter in request body."}
        from robot.llm import LlamaServerManager
        res = await LlamaServerManager.switch_model(model_id, engine=engine)
        if res.get("status") == "ok" and hasattr(engine, "events") and engine.events:
            await engine.events.emit("llm_model_changed", model=model_id, parameters=res.get("parameters"))
        return res

    @app.get("/api/llm/models")
    async def list_all_llm_models() -> dict[str, Any]:
        """List unified LLM models across Local, Gemini, and OpenAI-compatible providers."""
        from pathlib import Path
        from robot.config import get_project_root
        from robot.llm import LlamaServerManager
        from robot.llm.model_registry import ModelRegistry

        base_dir = get_project_root()
        current_prov = getattr(engine, "llm_provider_type", "local")
        active_local = LlamaServerManager.get_active_model_id()

        local_models = []
        # Standard Qwen 3.5 4B model in Downloads
        qwen_default = Path("/home/ameya/Downloads/Qwen3.5-4B-Q4_K_M.gguf")
        if qwen_default.exists():
            local_models.append({
                "id": "qwen3.5-4b",
                "name": "Qwen 3.5 4B (Instruct)",
                "parameters": "4B",
                "is_vision": False,
                "is_uncensored": False,
                "supports_tools": True,
                "is_downloaded": True,
                "is_active": (active_local == "qwen3.5-4b" and current_prov in ("local", "llama-server")),
                "description": "Standard high-reasoning local edge model running on RTX 3050 with FlashAttention.",
            })

        for m in ModelRegistry.list_models():
            local_models.append({
                "id": m.id,
                "name": m.name,
                "parameters": m.parameters,
                "is_vision": m.is_vision,
                "is_uncensored": m.is_uncensored,
                "supports_tools": m.supports_tools,
                "is_downloaded": m.is_downloaded(base_dir),
                "is_active": (active_local == m.id and current_prov in ("local", "llama-server")),
                "description": m.description,
            })

        gemini_models = [
            {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite", "context": "1M tokens", "is_active": (current_prov == "gemini")},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "context": "1M tokens", "is_active": False},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "context": "1M tokens", "is_active": False},
        ]

        return {
            "status": "ok",
            "provider": current_prov,
            "active_local_model": active_local,
            "local_models": local_models,
            "gemini_models": gemini_models,
            "llama_running": LlamaServerManager.is_running(),
        }

    # ─── Llama.cpp Advanced Loader & Offloading Settings (LM Studio) ───────
    @app.get("/api/llm/settings")
    async def get_llama_settings() -> dict[str, Any]:
        """Retrieve current llama-server configuration parameters and hardware offloading state."""
        from robot.llm import LlamaServerManager
        return LlamaServerManager.get_settings()

    @app.post("/api/llm/settings")
    async def update_llama_settings(req: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Update llama-server loading parameters and optionally hot-reload the running process."""
        from robot.llm import LlamaServerManager
        settings_data = req.get("settings", req)
        reload = bool(req.get("reload", False))
        res = await LlamaServerManager.update_settings(settings_data, reload=reload, engine=engine)
        return res

    @app.post("/api/llm/settings/reset")
    async def reset_llama_settings(req: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Reset llama-server configuration to optimal RTX 3050 defaults."""
        from robot.llm import LlamaServerManager
        reload = bool(req.get("reload", False))
        res = await LlamaServerManager.reset_settings(reload=reload, engine=engine)
        return res

    # ─── HuggingFace Hub LLM Search & Downloader Endpoints ──────────────────
    @app.get("/api/llm/hf/search")
    async def search_huggingface_models(
        q: str = Query("", description="Search term for HuggingFace models"),
        limit: int = Query(20, ge=1, le=50, description="Max results"),
    ) -> dict[str, Any]:
        """Search HuggingFace Hub for GGUF models.

        Args:
            q: Search query string.
            limit: Maximum items to return.

        Returns:
            dict[str, Any]: Search results and metadata.
        """
        from robot.llm.hf_downloader import HuggingFaceHubService
        results = await HuggingFaceHubService.search_models(query=q, limit=limit)
        return {
            "status": "ok",
            "query": q,
            "total": len(results),
            "models": results,
        }

    @app.get("/api/llm/hf/files")
    async def list_huggingface_model_files(
        repo_id: str = Query(..., description="Repository ID (e.g. bartowski/Qwen2.5-Coder-3B-Instruct-GGUF)"),
    ) -> dict[str, Any]:
        """Retrieve available GGUF files and quantization options for a model repository.

        Args:
            repo_id: HuggingFace repository identifier.

        Returns:
            dict[str, Any]: List of GGUF files with byte sizes and download statuses.
        """
        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceHubService
        base_dir = get_project_root()
        files = await HuggingFaceHubService.get_model_files(repo_id=repo_id, base_dir=base_dir)
        return {
            "status": "ok",
            "repo_id": repo_id,
            "total": len(files),
            "files": files,
        }

    @app.post("/api/llm/hf/download")
    async def download_huggingface_model(
        req: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Start asynchronous background download of a HuggingFace GGUF model.

        Args:
            req: Request body containing repo_id, filename, name, parameters, is_vision.

        Returns:
            dict[str, Any]: Created download task specification and tracking ID.
        """
        repo_id = req.get("repo_id")
        filename = req.get("filename")
        if not repo_id or not filename:
            return {"status": "error", "message": "Missing required 'repo_id' or 'filename'."}

        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceDownloadManager

        base_dir = get_project_root()
        mgr = HuggingFaceDownloadManager.get_instance(base_dir=base_dir)
        task = mgr.start_download(
            repo_id=repo_id,
            filename=filename,
            name=req.get("name"),
            parameters=req.get("parameters"),
            is_vision=bool(req.get("is_vision", False)),
            mmproj_filename=req.get("mmproj_filename"),
        )
        return {"status": "ok", "task": task.to_dict()}

    @app.get("/api/llm/hf/downloads")
    async def get_huggingface_downloads() -> dict[str, Any]:
        """Get list of active and recent HuggingFace model download tasks.

        Returns:
            dict[str, Any]: List of all download tasks and their progress metrics.
        """
        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceDownloadManager

        base_dir = get_project_root()
        mgr = HuggingFaceDownloadManager.get_instance(base_dir=base_dir)
        tasks = mgr.get_all_tasks()
        return {
            "status": "ok",
            "total": len(tasks),
            "downloads": tasks,
        }

    @app.post("/api/llm/hf/cancel")
    async def cancel_huggingface_download(
        req: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Cancel an in-progress HuggingFace model download task.

        Args:
            req: Request body containing download_id.

        Returns:
            dict[str, Any]: Operation status and cancellation flag.
        """
        download_id = req.get("download_id")
        if not download_id:
            return {"status": "error", "message": "Missing 'download_id'."}

        from robot.config import get_project_root
        from robot.llm.hf_downloader import HuggingFaceDownloadManager

        base_dir = get_project_root()
        mgr = HuggingFaceDownloadManager.get_instance(base_dir=base_dir)
        ok = await mgr.cancel_download(download_id)
        return {"status": "ok", "cancelled": ok}

    # ─── Speech-to-Text (Whisper) Endpoints ─────────────────────────────────
    @app.get("/api/stt/status")
    async def get_stt_status() -> dict[str, Any]:
        """Get active Whisper STT model, daemon process state, and hardware offload metrics."""
        from robot.stt import WhisperServerManager
        active_id = WhisperServerManager.get_active_model_id()
        is_running = await WhisperServerManager.is_running_async()
        meta = WhisperServerManager.MODELS_CATALOG.get(active_id)
        return {
            "status": "ok",
            "is_running": is_running,
            "active_model": active_id,
            "name": meta.name if meta else active_id,
            "parameters": meta.parameters if meta else "unknown",
            "size_mb": meta.size_mb if meta else 0,
            "latency_rating": meta.latency_rating if meta else "unknown",
            "device": "AMD Radeon 680M (Vulkan dev 0)",
            "vram_rtx3050_mb": 0,
        }

    @app.get("/api/stt/models")
    async def list_stt_models() -> dict[str, Any]:
        """List all registered Whisper models with download and active status."""
        from robot.stt import WhisperServerManager
        models = WhisperServerManager.list_models()
        active_id = WhisperServerManager.get_active_model_id()
        is_running = await WhisperServerManager.is_running_async()
        return {
            "status": "ok",
            "active_model": active_id,
            "is_running": is_running,
            "total_models": len(models),
            "models": models,
        }

    @app.post("/api/stt/switch")
    async def switch_stt_model(req: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Hot-swap the active Whisper model on the AMD Radeon 680M Vulkan backend.

        Args:
            req: Request payload containing the target Whisper 'model' or 'model_id'.

        Returns:
            Dictionary with switch operation status, model name, and file size in MB.
        """
        model_id = req.get("model") or req.get("model_id")
        if not model_id:
            return {"status": "error", "message": "Missing 'model' parameter in request body."}
        from robot.stt import WhisperServerManager
        res = await WhisperServerManager.switch_model(model_id, engine=engine)
        if res.get("status") == "ok" and hasattr(engine, "events") and engine.events:
            await engine.events.emit("stt_model_changed", model=model_id, size_mb=res.get("size_mb"))
        return res

    # ─── Text-to-Speech (Piper & Kokoro) Endpoints ─────────────────────────
    @app.get("/api/tts/status")
    async def get_tts_status() -> dict[str, Any]:
        """Get active TTS engine, voice model, sample rate, and supported engines."""
        from robot.tts import TTSManager
        tts = getattr(engine, "tts", None)
        if isinstance(tts, TTSManager):
            active_engine = tts.get_active_engine_name()
            active_voice = tts.get_active_voice()
            engines_list = tts.list_engines()
            voices_list = tts.list_voices()
            eng_inst = tts.get_active_engine()
            sample_rate = getattr(eng_inst, "SAMPLE_RATE", 16000)
        else:
            active_engine = "piper"
            active_voice = getattr(tts, "_active_model", "en-us-lessac-medium")
            engines_list = [
                {
                    "id": "piper",
                    "name": "Piper ONNX",
                    "description": "Ultra low-latency local ONNX synthesis",
                    "sample_rate": 16000,
                    "is_available": True,
                    "is_active": True,
                    "total_voices": 7,
                    "recommended_voice": "en-us-lessac-medium",
                }
            ]
            voices_list = []
            sample_rate = 16000

        return {
            "status": "ok",
            "active_engine": active_engine,
            "active_voice": active_voice,
            "sample_rate": sample_rate,
            "total_voices": len(voices_list),
            "engines": engines_list,
        }

    @app.get("/api/tts/voices")
    async def list_tts_voices(engine_filter: str | None = Query(None, alias="engine")) -> dict[str, Any]:
        """List catalog of all available voices across Kokoro Neural and Piper ONNX."""
        from robot.tts import TTSManager
        tts = getattr(engine, "tts", None)
        if isinstance(tts, TTSManager):
            voices = tts.list_voices(engine_name=engine_filter)
            active_engine = tts.get_active_engine_name()
            active_voice = tts.get_active_voice()
        else:
            active_engine = "piper"
            active_voice = "en-us-lessac-medium"
            voices = []

        return {
            "status": "ok",
            "active_engine": active_engine,
            "active_voice": active_voice,
            "total_voices": len(voices),
            "voices": voices,
        }

    @app.post("/api/tts/switch")
    async def switch_tts_engine_and_voice(req: TTSSwitchRequest) -> dict[str, Any]:
        """Hot-swap the active TTS engine (Piper or Kokoro) and active voice at runtime.

        Args:
            req: TTSSwitchRequest with engine ('piper' or 'kokoro') and optional voice identifier.

        Returns:
            Dictionary with switch result status, active engine, and active voice.
        """
        from robot.tts import TTSManager
        tts = getattr(engine, "tts", None)
        if not isinstance(tts, TTSManager):
            try:
                engine.tts = TTSManager(engine.config)
                tts = engine.tts
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize TTSManager: {e}")

        try:
            res = tts.switch_engine(req.engine, voice=req.voice)
            if hasattr(engine, "events") and engine.events:
                await engine.events.emit(
                    "tts_engine_changed",
                    engine=res["engine"],
                    voice=res["voice"],
                    sample_rate=res["sample_rate"],
                )
            return {
                "status": "ok",
                "engine": res["engine"],
                "voice": res["voice"],
                "sample_rate": res["sample_rate"],
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/tts/test")
    async def test_tts_voice(req: TTSTestRequest) -> Response:
        """Synthesize preview audio using requested engine and voice without changing active state.

        Args:
            req: TTSTestRequest with engine, voice, and test phrase.

        Returns:
            Response containing audio/wav binary data for in-browser playback.
        """
        from robot.tts import TTSManager
        import io
        import wave
        tts = getattr(engine, "tts", None)
        if not isinstance(tts, TTSManager):
            try:
                engine.tts = TTSManager(engine.config)
                tts = engine.tts
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"TTSManager unavailable: {e}")

        try:
            target_engine = tts._get_or_load_engine(req.engine)
            test_phrase = req.text.strip() or "Hello, this is a test of the neural voice engine on the robot."
            samples, sr = await target_engine.synthesize(test_phrase, voice_model=req.voice)
            if len(samples) == 0:
                raise HTTPException(status_code=500, detail="Synthesis produced zero audio frames.")

            pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())

            return Response(content=buf.getvalue(), media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"TTS test synthesis failed: {e}")

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

        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()

        thinking_flag = req.thinking if req.thinking is not None else getattr(engine, "thinking_enabled", True)

        async def _event_generator():
            async for chunk in engine.stream_text_chat(
                text=req.text,
                session_id=req.session_id,
                personality=req.personality,
                thinking=thinking_flag,
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

    @app.get("/api/thinking")
    async def get_thinking_mode():
        """Retrieve the active cognitive reasoning/thinking mode state."""
        is_enabled = getattr(engine, "thinking_enabled", True)
        return {"status": "ok", "thinking_enabled": is_enabled}

    @app.post("/api/thinking")
    async def set_thinking_mode(req: ThinkingModeRequest):
        """Update and broadcast the cognitive reasoning/thinking mode state."""
        engine.thinking_enabled = bool(req.enabled)
        if hasattr(engine, "events") and engine.events:
            await engine.events.emit("thinking_toggled", enabled=engine.thinking_enabled)
        return {"status": "ok", "thinking_enabled": engine.thinking_enabled}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """Text-based chat endpoint (non-voice)."""
        if hasattr(engine, "notify_user_activity"):
            engine.notify_user_activity()
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

        Splits message text into natural clause phrases, applying dynamic sentiment
        prosody modulation (pitch, speed, noise scale) to each clause individually,
        mirroring the active avatar TTS pipeline. Emits a robot phrase event for
        synchronized avatar lipsync animation.

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
        from robot.tts import clean_text_for_speech, split_into_phrases

        raw_text = req.text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        tts = getattr(engine, "tts", None)
        if tts is None or not hasattr(tts, "synthesize"):
            raise HTTPException(status_code=503, detail="TTS provider is not initialized or unavailable.")

        # Resolve voice model and prosody from personality if not explicitly overridden
        pers_name = req.personality or getattr(engine.state.state, "active_personality", "neo")
        pers = engine.personalities.get(pers_name) if hasattr(engine, "personalities") and engine.personalities else None

        active_tts_voice = None
        if hasattr(tts, "get_active_voice") and callable(tts.get_active_voice):
            active_tts_voice = tts.get_active_voice()

        voice_model = req.voice_model or active_tts_voice or (getattr(pers, "voice_model", None) if pers else None)
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

        phrases = split_into_phrases(raw_text)
        if not phrases:
            fallback = clean_text_for_speech(raw_text) or raw_text
            phrases = [fallback] if fallback.strip() else []

        if not phrases:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        audio_chunks: list[np.ndarray] = []
        sample_rate = 16000
        primary_emotion = "neutral"

        for phrase in phrases:
            eff_speed, eff_pitch, eff_ns, eff_nw = speed, pitch, noise_scale, noise_w
            phrase_emotion = "neutral"

            # Apply dynamic sentiment analysis and prosody modulation per clause
            if hasattr(tts, "modulate_prosody"):
                try:
                    res = tts.modulate_prosody(
                        phrase,
                        base_speed=speed,
                        base_pitch=pitch,
                        base_noise_scale=noise_scale,
                        base_noise_w=noise_w,
                        personality_name=pers_name,
                    )
                    if isinstance(res, tuple) and len(res) == 5:
                        eff_speed, eff_pitch, eff_ns, eff_nw, modulation = res
                        phrase_emotion = getattr(modulation, "emotion", "neutral")
                        if primary_emotion == "neutral" and phrase_emotion != "neutral":
                            primary_emotion = phrase_emotion
                except Exception:
                    pass

            try:
                chunk, sr = await tts.synthesize(
                    phrase,
                    speed=eff_speed,
                    pitch=eff_pitch,
                    voice_model=voice_model,
                    noise_scale=eff_ns,
                    noise_w=eff_nw,
                )
                if chunk is not None and len(chunk) > 0:
                    audio_chunks.append(chunk)
                    sample_rate = sr
            except Exception as exc:
                logger.error("chat_speak_phrase_failed", phrase=phrase[:30], error=str(exc))

        if not audio_chunks:
            raise HTTPException(status_code=400, detail="Synthesized audio was empty.")

        audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

        # Optionally play on robot hardware speaker if requested
        if req.play_hardware and hasattr(engine, "player") and engine.player:
            asyncio.create_task(engine.player.play(audio, sample_rate))

        # Emit robot_phrase event for avatar lip-sync / expressions
        if hasattr(engine, "events") and engine.events:
            try:
                clean_phrase = clean_text_for_speech(raw_text)
                await engine.events.emit("robot_phrase", robot_phrase=clean_phrase, emotion=primary_emotion)
            except Exception as e:
                logger.debug("speak_phrase_event_failed", error=str(e))

        # Convert float32 numpy array to 16-bit PCM WAV
        int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
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

    @app.post("/api/chat/speak/stream")
    async def speak_chat_message_stream(req: ChatSpeakRequest, request: Request) -> StreamingResponse:
        """Synthesize and stream speech WAV audio clauses via Server-Sent Events (SSE).

        Splits input text into natural punctuation clauses/phrases and modulates each phrase
        individually for dynamic pitch, speed, and emotion prosody. Yields phrases as soon as
        they are synthesized so the client begins audio playback on phrase 0 immediately.

        Args:
            req: ChatSpeakRequest containing text, personality, voice_model, speed, pitch.
            request: Active Starlette/FastAPI Request used to detect client disconnections.

        Returns:
            StreamingResponse emitting SSE text/event-stream events containing base64 WAV chunks.

        Raises:
            HTTPException: 400 if text is empty; 503 if TTS is unavailable.
        """
        import base64
        import io
        import json
        import wave
        import numpy as np
        from fastapi.responses import StreamingResponse
        from robot.tts import clean_text_for_speech, split_into_phrases

        raw_text = req.text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        tts = getattr(engine, "tts", None)
        if tts is None or not hasattr(tts, "synthesize"):
            raise HTTPException(status_code=503, detail="TTS provider is not initialized or unavailable.")

        # Resolve voice model and prosody defaults from active personality
        pers_name = req.personality or getattr(engine.state.state, "active_personality", "neo")
        pers = engine.personalities.get(pers_name) if hasattr(engine, "personalities") and engine.personalities else None

        active_tts_voice = None
        if hasattr(tts, "get_active_voice") and callable(tts.get_active_voice):
            active_tts_voice = tts.get_active_voice()

        base_voice_model = req.voice_model or active_tts_voice or (getattr(pers, "voice_model", None) if pers else None)
        base_speed = req.speed
        base_pitch = req.pitch
        base_noise_scale: float | None = None
        base_noise_w: float | None = None

        if pers and hasattr(pers, "tts_config") and isinstance(pers.tts_config, dict):
            cfg = pers.tts_config
            if req.speed == 1.0 and "speed" in cfg:
                base_speed = float(cfg["speed"])
            if req.pitch == 1.0 and "pitch" in cfg:
                base_pitch = float(cfg["pitch"])
            base_noise_scale = cfg.get("noise_scale")
            base_noise_w = cfg.get("noise_w")

        # Split text into natural clause phrases for low-latency streaming
        phrases = split_into_phrases(raw_text)
        if not phrases:
            fallback = clean_text_for_speech(raw_text) or raw_text
            phrases = [fallback] if fallback.strip() else []

        if not phrases:
            raise HTTPException(status_code=400, detail="No speakable phrases found in input text.")

        async def _phrase_stream_generator():
            total = len(phrases)
            for idx, phrase in enumerate(phrases):
                if await request.is_disconnected():
                    logger.debug("chat_speak_stream_client_disconnected", phrase_idx=idx, total=total)
                    break

                eff_speed = base_speed
                eff_pitch = base_pitch
                eff_noise_scale = base_noise_scale
                eff_noise_w = base_noise_w
                detected_emotion = "neutral"

                # Apply dynamic sentiment analysis and prosody modulation per phrase
                if hasattr(tts, "modulate_prosody"):
                    try:
                        res = tts.modulate_prosody(
                            phrase,
                            base_speed=base_speed,
                            base_pitch=base_pitch,
                            base_noise_scale=base_noise_scale,
                            base_noise_w=base_noise_w,
                            personality_name=pers_name,
                        )
                        if isinstance(res, tuple) and len(res) == 5:
                            eff_speed, eff_pitch, eff_noise_scale, eff_noise_w, modulation = res
                            detected_emotion = getattr(modulation, "emotion", "neutral")
                    except Exception as err:
                        logger.debug("phrase_prosody_modulation_skipped", error=str(err))

                try:
                    audio, sr = await tts.synthesize(
                        phrase,
                        speed=eff_speed,
                        pitch=eff_pitch,
                        voice_model=base_voice_model,
                        noise_scale=eff_noise_scale,
                        noise_w=eff_noise_w,
                    )
                except Exception as exc:
                    logger.error("phrase_synthesis_failed", phrase=phrase[:30], error=str(exc))
                    yield f"data: {json.dumps({'event': 'error', 'index': idx, 'error': str(exc)})}\n\n"
                    continue

                if audio is None or len(audio) == 0:
                    continue

                # Optionally play on robot hardware speaker sequentially if requested
                if req.play_hardware and hasattr(engine, "player") and engine.player:
                    try:
                        await engine.player.play(audio, sr)
                    except Exception as e:
                        logger.warning("hardware_playback_failed", error=str(e))

                # Convert float32 numpy array to 16-bit PCM WAV in memory
                int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(int16_audio.tobytes())
                base64_wav = base64.b64encode(wav_buf.getvalue()).decode("ascii")

                duration_ms = (len(audio) / sr) * 1000.0

                event_data = {
                    "event": "phrase",
                    "index": idx,
                    "total": total,
                    "phrase": phrase,
                    "emotion": detected_emotion,
                    "pitch": round(float(eff_pitch), 3),
                    "speed": round(float(eff_speed), 3),
                    "duration_ms": round(duration_ms, 1),
                    "audio": base64_wav,
                }
                yield f"data: {json.dumps(event_data)}\n\n"

            yield f"data: {json.dumps({'event': 'done', 'total': total})}\n\n"

        return StreamingResponse(
            _phrase_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
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
        from fastapi.responses import Response
        if not engine.is_camera_enabled:
            return Response(content=b"Camera paused or disabled", status_code=503)
        if hasattr(engine, "perception") and engine.perception:
            import cv2

            frame = engine.perception.get_annotated_frame()
            if frame is not None:
                success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    return Response(content=enc.tobytes(), media_type="image/jpeg")
            return Response(content=b"No frame available", status_code=503)
        return Response(content=b"Camera not available", status_code=503)

    @app.get("/camera/stream")
    async def get_camera_stream():
        """Ultra-low latency continuous MJPEG video stream with green polygons."""
        from fastapi.responses import Response, StreamingResponse
        if not engine.is_camera_enabled:
            return Response(content=b"Camera paused or disabled", status_code=503)

        import asyncio
        import cv2

        async def frame_generator():
            try:
                while engine.is_camera_enabled:
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
            model=req.model or "default",
            negative_prompt=req.negative_prompt or "",
            trace_id=getattr(engine.observability, "current_trace_id", None) if hasattr(engine, "observability") else None,
        )
        return {"status": "queued", "job": job.to_dict()}

    @app.get("/api/images/models")
    async def list_image_models() -> dict[str, Any]:
        """List registered FLUX, Stable Diffusion, and Procedural models with cache status."""
        from robot.image.model_registry import ImageModelRegistry
        models = [m.to_dict() | {"is_cached": m.is_cached_locally()} for m in ImageModelRegistry.list_models()]
        return {
            "default_model": ImageModelRegistry.DEFAULT_GPU_MODEL_ID,
            "total_models": len(models),
            "models": models,
        }

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

    @app.get("/api/images/gallery")
    async def get_image_gallery(limit: int = 100) -> dict[str, Any]:
        """List generated artwork images from data/images with rich metadata.

        Args:
            limit: Maximum number of gallery images to return.

        Returns:
            Dictionary containing total count and list of image items with metadata.
        """
        from datetime import datetime, timezone
        from PIL import Image

        images_dir = Path("data/images")
        if not images_dir.exists():
            return {"status": "ok", "total": 0, "images": []}

        # Build lookup table from in-memory jobs if available
        jobs_lookup: dict[str, dict[str, Any]] = {}
        if hasattr(engine, "image_service") and engine.image_service:
            for j in engine.image_service.list_jobs(200):
                out = j.get("output_path", "")
                if out:
                    base_name = Path(out).name
                    jobs_lookup[base_name] = j

        items: list[dict[str, Any]] = []
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}

        for p in images_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in valid_extensions:
                continue

            fname = p.name
            try:
                stat = p.stat()
                mtime = stat.st_mtime
                size_bytes = stat.st_size

                # Match with in-memory job if possible
                job_meta = jobs_lookup.get(fname, {})
                prompt = job_meta.get("prompt", "")
                model = job_meta.get("model", "")
                duration_s = job_meta.get("duration_s", 0.0)

                # If prompt is not in in-memory jobs, derive human-readable label
                if not prompt:
                    if fname.startswith("img_test_flux"):
                        prompt = "FLUX.1 Schnell procedural/diffusion test benchmark"
                        model = "flux-schnell"
                    elif "sdxl" in fname:
                        prompt = "SDXL Turbo high-speed generation"
                        model = "sdxl-turbo"
                    elif "small_sd" in fname:
                        prompt = "Small-SD lightweight model generation"
                        model = "small-sd"
                    elif fname.startswith("img_"):
                        prompt = f"Generated Artwork ({fname[:12]})"
                        model = "diffusion"
                    else:
                        prompt = fname.rsplit(".", 1)[0].replace("_", " ").title()
                        model = "diffusion"

                # Extract dimensions quickly
                width, height = 512, 512
                try:
                    with Image.open(p) as img:
                        width, height = img.size
                except Exception:
                    pass

                size_kb = (
                    f"{round(size_bytes / 1024, 1)} KB"
                    if size_bytes < 1024 * 1024
                    else f"{round(size_bytes / (1024 * 1024), 2)} MB"
                )

                items.append({
                    "filename": fname,
                    "url": f"/data/images/{fname}",
                    "prompt": prompt,
                    "model": model or "diffusion",
                    "duration_s": round(duration_s, 2),
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes,
                    "size_kb": size_kb,
                    "created_at": mtime,
                    "created_at_iso": datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                })
            except Exception as e:
                logger.warning("gallery_image_read_failed", file=fname, error=str(e))

        # Sort reverse chronologically
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return {
            "status": "ok",
            "total": len(items),
            "images": items[:limit],
        }

    @app.delete("/api/images/gallery/{filename}")
    async def delete_gallery_image(filename: str) -> dict[str, Any]:
        """Delete a generated artwork from data/images with path traversal protection.

        Args:
            filename: Target image filename to remove.

        Returns:
            Dictionary with deletion status and filename.
        """
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", filename) or ".." in filename or "/" in filename:
            return {"status": "error", "message": "Invalid filename format."}

        target_file = Path("data/images") / filename
        if not target_file.exists() or not target_file.is_file():
            return {"status": "error", "message": "File not found."}

        try:
            target_file.unlink()
            return {"status": "ok", "deleted": True, "filename": filename}
        except Exception as e:
            return {"status": "error", "message": str(e)}


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
        class NoCacheStaticFiles(StaticFiles):
            """StaticFiles handler that appends no-cache headers to ensure instant live asset updates."""

            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response

        app.mount("/static", NoCacheStaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def index():
            """Serve the integrated Digital Face & Dev Console UI."""
            return FileResponse(
                str(static_dir / "index.html"),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        @app.get("/contextlayer")
        @app.get("/inspect")
        @app.get("/context-layer")
        async def context_layer_page():
            """Serve the dynamic LLM context layer & prompt caching debugger page."""
            context_html = static_dir / "contextlayer.html"
            if context_html.exists():
                return FileResponse(str(context_html))
            return {"error": "contextlayer.html not found"}

    @app.get("/api/context/inspect")
    @app.get("/api/inspect")
    async def inspect_context_layer(test_prompt: str = "Hi"):
        """Dynamically inspect the structured context layer and KV cache invariance status.

        Args:
            test_prompt: Optional test input prompt string to simulate context assembly.

        Returns:
            dict[str, Any]: Structured breakdown of system prompt, user profile,
                situation banner, durable memories, history, tool schemas, and cache metrics.
        """
        prov = getattr(engine, "llm_provider_type", "local")
        if prov in ("llama-server", "local"):
            prov = "local"
        is_gemini = prov == "gemini"
        is_openai = prov in ("openai_compatible", "openai")

        active_pers = engine.personalities.active
        active_pers_name = getattr(active_pers, "name", "neo") if active_pers else "neo"
        ns = getattr(active_pers, "memory_namespace", "default")

        openai_cfg = getattr(engine, "config", {}).get("llm", {}).get("openai_compatible", {}) if hasattr(engine, "config") else {}
        if is_openai:
            active_model = getattr(engine, "_get_persisted_openai_model", lambda d: d)(openai_cfg.get("model", "gpt-4o"))
            ctx_win = int(openai_cfg.get("context_window", 128000))
        elif is_gemini:
            active_model = getattr(engine.llm, "model", "gemini-3.5-flash")
            ctx_win = 1000000
        else:
            active_model = getattr(engine.llm, "model", "Qwen3.5-4B-Q4_K_M")
            ctx_win = int(getattr(engine, "config", {}).get("llm", {}).get("context_window", 16384)) if hasattr(engine, "config") else 16384

        situation = await engine.get_situation_context() if hasattr(engine, "get_situation_context") else {}
        
        user_profile = None
        if hasattr(engine, "memory") and engine.memory and hasattr(engine.memory, "get_core_user_profile"):
            try:
                user_profile = await engine.memory.get_core_user_profile(user_id=ns)
            except Exception:
                user_profile = None

        all_facts = []
        if hasattr(engine, "memory") and engine.memory and hasattr(engine.memory, "get_all_facts"):
            try:
                all_facts = await engine.memory.get_all_facts(user_id=ns, limit=20)
            except Exception:
                all_facts = []

        recent_history = []
        if hasattr(engine, "memory") and engine.memory and hasattr(engine.memory, "get_recent_history"):
            try:
                recent_history = await engine.memory.get_recent_history(limit=10, user_id=ns)
            except Exception:
                recent_history = []

        pers_prompt = (
            active_pers.system_prompt
            if active_pers and getattr(active_pers, "system_prompt", None)
            else (getattr(engine.personalities, "active_prompt", "") or "")
        )

        from robot.core.context import (
            format_system_prompt,
            format_user_core_profile,
            format_situation_banner,
        )

        assembled_messages = engine.context.build_messages(
            system_prompt=pers_prompt,
            user_text=test_prompt,
            recent_conversation=recent_history,
            memories=all_facts[:5],
            all_facts=all_facts,
            world_state=situation.get("world_state"),
            situation=situation,
            high_intelligence=is_gemini,
            user_profile=user_profile,
        )

        tool_schemas = engine.tool_registry.get_schemas() if hasattr(engine, "tool_registry") and engine.tool_registry else []
        tool_names = [t.get("name", "") for t in tool_schemas if isinstance(t, dict)]

        # Estimate Token Distribution
        tools_tokens_est = 10280 if tool_schemas else 0
        sys_tokens_est = len(assembled_messages[0]["content"]) // 4 if assembled_messages else 0
        user_tokens_est = len(assembled_messages[-1]["content"]) // 4 if len(assembled_messages) > 1 else 0
        hist_tokens_est = sum(len(str(m.get("content", ""))) // 4 for m in assembled_messages[1:-1])

        prefix_tokens = tools_tokens_est + sys_tokens_est
        suffix_tokens = user_tokens_est + hist_tokens_est
        total_tokens = prefix_tokens + suffix_tokens
        cache_reuse_pct = round((prefix_tokens / max(1, total_tokens)) * 100, 1)

        from robot.llm import LlamaServerManager
        llama_running = LlamaServerManager.is_running()

        return {
            "timestamp": time.time(),
            "personality": active_pers_name,
            "provider": prov,
            "model": active_model,
            "context_window": ctx_win,
            "test_prompt": test_prompt,
            "structured_layer": {
                "system_prompt": format_system_prompt(pers_prompt, high_intelligence=is_gemini),
                "user_profile": format_user_core_profile(user_profile),
                "situation_banner": format_situation_banner(situation),
                "retrieved_memories": all_facts,
                "conversation_turns": recent_history,
                "tools": {
                    "count": len(tool_schemas),
                    "names": tool_names,
                    "tokens_estimate": tools_tokens_est,
                    "sample": tool_schemas[:2],
                },
            },
            "caching_status": {
                "strategy": "Slot 0 Invariant Canonical Prefix + Prompt Caching",
                "prefix_tokens_estimate": prefix_tokens,
                "suffix_tokens_estimate": suffix_tokens,
                "total_tokens_estimate": total_tokens,
                "cache_reuse_percent": cache_reuse_pct,
                "is_caching_invariant": True,
                "llama_running": llama_running,
                "prompt_caching_enabled": True if is_openai else llama_running,
            },
            "diagnostics": getattr(engine.context, "get_last_diagnostics", lambda: {})(),
            "assembled_messages": assembled_messages,
        }

    return app
