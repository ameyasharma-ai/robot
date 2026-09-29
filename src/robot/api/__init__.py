"""FastAPI server — HTTP + WebSocket API for Robot Core.

Provides:
- REST endpoints for status, memory, personality
- WebSocket for real-time event streams
- Developer console / AI Terminal access
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


class ChatRequest(BaseModel):
    text: str
    personality: str | None = None


class ChatResponse(BaseModel):
    response: str
    turn: int
    personality: str


class StatusResponse(BaseModel):
    mode: str
    personality: str
    turn_count: int
    services: dict[str, bool]
    telemetry: dict[str, Any]


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
        return StatusResponse(
            mode=state.mode.value,
            personality=state.active_personality,
            turn_count=state.turn_count,
            services={
                name: svc.healthy for name, svc in state.services.items()
            },
            telemetry=engine.telemetry.snapshot(),
        )

    @app.get("/personalities")
    async def list_personalities():
        return {"personalities": engine.personalities.list_personalities()}

    @app.post("/personality/{name}")
    async def switch_personality(name: str):
        success = engine.personalities.set_active(name)
        if success:
            engine.state.state.active_personality = name
            return {"status": "switched", "personality": name}
        return {"error": f"Personality '{name}' not found"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """Text-based chat endpoint (non-voice)."""
        response = await engine.process_text_input(req.text)
        return ChatResponse(
            response=response,
            turn=engine.state.state.turn_count,
            personality=engine.state.state.active_personality,
        )

    @app.get("/memory/search")
    async def search_memory(query: str, limit: int = 5):
        results = await engine.memory.get_relevant_memories(query, limit)
        return {"results": results}

    @app.get("/history/search")
    async def search_history(query: str, limit: int = 10):
        results = await engine.memory.search_history(query, limit)
        return {"results": results}

    @app.get("/telemetry")
    async def telemetry():
        return engine.telemetry.snapshot()

    @app.websocket("/ws/events")
    async def event_stream(websocket: WebSocket):
        """WebSocket stream for real-time telemetry events."""
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
        except WebSocketDisconnect:
            engine.events.off("*", forward_event)

    return app
