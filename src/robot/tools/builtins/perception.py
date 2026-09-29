"""Visual perception tools for LLM reasoning and camera snapshots."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import cv2

from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


def register_perception_tools(engine: "RobotEngine") -> None:
    """Register camera and perception tools with the engine's tool registry."""

    async def get_world_state() -> dict[str, Any]:
        """Return the current continuous perception world state."""
        if hasattr(engine, "perception") and engine.perception:
            return engine.perception.world_state.to_compact()
        return {"error": "Perception subsystem not active"}

    engine.tool_registry.register(
        ToolDefinition(
            name="perception.get_world_state",
            description="Get the current physical world state: people present, position, distance, motion level, and room lighting.",
            parameters={"type": "object", "properties": {}},
            handler=get_world_state,
            permission_level=PermissionLevel.READ_ONLY,
            family="perception",
        )
    )

    async def capture_snapshot(format: str = "base64", quality: int = 80) -> dict[str, Any]:
        """Capture a snapshot from the robot's camera for inspection or visual reasoning."""
        if not hasattr(engine, "perception") or not engine.perception or not engine.perception.camera:
            return {"error": "Camera not available"}

        frame = engine.perception.camera.get_latest_frame()
        if frame is None:
            return {"error": "No frame captured from camera yet"}

        h, w = frame.shape[:2]
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, quality))]
        success, encimg = cv2.imencode(".jpg", frame, encode_param)
        if not success:
            return {"error": "Failed to encode frame"}

        b64_str = base64.b64encode(encimg.tobytes()).decode("utf-8")
        return {
            "width": w,
            "height": h,
            "format": "jpeg",
            "image_base64": b64_str if format == "base64" else None,
            "data_url": f"data:image/jpeg;base64,{b64_str}" if format == "data_url" else None,
            "size_bytes": len(encimg),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="camera.capture_snapshot",
            description="Capture a current frame snapshot from the robot's physical camera for visual reasoning.",
            parameters={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["base64", "data_url"],
                        "default": "base64",
                        "description": "Encoding format for the captured image",
                    },
                    "quality": {
                        "type": "integer",
                        "default": 80,
                        "description": "JPEG compression quality (10-100)",
                    },
                },
            },
            handler=capture_snapshot,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="perception",
        )
    )

    async def inspect_view() -> dict[str, Any]:
        """Deep visual inspection of current scene, objects, and people."""
        if not hasattr(engine, "perception") or not engine.perception:
            return {"error": "Perception subsystem not active"}

        ws = engine.perception.world_state.to_compact()
        people_desc = []
        for p in ws.get("people", []):
            people_desc.append(f"Person at {p.get('position')} ({p.get('distance')} range)")

        obj_desc = []
        for o in ws.get("objects", []):
            obj_desc.append(f"{o.get('label')} at {o.get('position')}")

        return {
            "summary": f"Detected {len(people_desc)} people and {len(obj_desc)} physical objects in camera view.",
            "people": people_desc,
            "objects": obj_desc,
            "motion": ws.get("motion"),
            "lighting": ws.get("lighting"),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="perception.inspect_view",
            description="Perform a visual inspection through the camera to identify what objects, people, and scene conditions are in front of the robot.",
            parameters={"type": "object", "properties": {}},
            handler=inspect_view,
            permission_level=PermissionLevel.READ_ONLY,
            family="perception",
        )
    )
