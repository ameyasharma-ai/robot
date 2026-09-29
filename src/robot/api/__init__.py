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

    # Static UI Files & Web HUD Display
    from pathlib import Path
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def index():
            """Serve the integrated Digital Face & Dev Console UI."""
            return FileResponse(str(static_dir / "index.html"))

    return app
