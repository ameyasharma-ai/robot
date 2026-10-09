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

    async def get_world_state(**kwargs) -> dict[str, Any]:
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

    async def capture_snapshot(format: str = "base64", quality: int = 80, **kwargs) -> dict[str, Any]:
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
        eff_format = format or kwargs.get("encoding", "base64")
        return {
            "width": w,
            "height": h,
            "format": "jpeg",
            "image_base64": b64_str if eff_format == "base64" else None,
            "data_url": f"data:image/jpeg;base64,{b64_str}" if eff_format == "data_url" else None,
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

    async def what_do_you_see(**kwargs) -> dict[str, Any]:
        """Perform visual inspection through the camera to report what the robot sees."""
        if not hasattr(engine, "perception") or not engine.perception:
            return {
                "summary": "Camera perception system is currently inactive.",
                "people": [],
                "objects": [],
                "motion": "none",
                "lighting": "normal",
            }

        ws = engine.perception.world_state.to_compact()
        people_desc = []
        for p in ws.get("people", []):
            dist = p.get("distance", "mid")
            pos = p.get("position", "center")
            people_desc.append(f"Person at {pos} ({dist} distance)")

        obj_desc = []
        for o in ws.get("objects", []):
            lbl = o.get("label", "item")
            pos = o.get("position", "center")
            obj_desc.append(f"{lbl} at {pos}")

        details = []
        if people_desc:
            details.append("People detected: " + ", ".join(people_desc))
        if obj_desc:
            details.append("Objects detected: " + ", ".join(obj_desc))
        if not details:
            details.append("No distinct people or objects clearly recognized in the immediate camera frame. Suggest the user hold the object directly toward the camera.")

        summary = f"Visual camera feed: {'; '.join(details)}. Ambient lighting: {ws.get('lighting', 'normal')}. Motion: {ws.get('motion', 'none')}."
        return {
            "summary": summary,
            "people": people_desc,
            "objects": obj_desc,
            "lighting": ws.get("lighting", "normal"),
            "motion": ws.get("motion", "none"),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="perception.what_do_you_see",
            description="Look through the physical camera and inspect the surroundings: identifies people, faces, objects being held or placed in front of the robot, distance, lighting, and movement. Call this tool when asked 'what do you see', 'what am I holding', 'what is this', 'who is here', or 'look around'.",
            parameters={"type": "object", "properties": {}},
            handler=what_do_you_see,
            permission_level=PermissionLevel.READ_ONLY,
            family="perception",
        )
    )

    engine.tool_registry.register(
        ToolDefinition(
            name="perception.inspect_view",
            description="Alias for perception.what_do_you_see. Inspect current camera view for objects, people, and scene conditions.",
            parameters={"type": "object", "properties": {}},
            handler=what_do_you_see,
            permission_level=PermissionLevel.READ_ONLY,
            family="perception",
        )
    )

    async def inspect_tabletop(query: str = "Describe all items on the table", **kwargs) -> dict[str, Any]:
        """Perform deep visual inspection of tabletop objects via AMD iGPU VLM."""
        target_query = (query or kwargs.get("text") or kwargs.get("prompt") or kwargs.get("question") or "Describe all items on the table").strip()
        
        # Check perception engine and camera
        frame = None
        if hasattr(engine, "perception") and engine.perception and engine.perception.camera:
            frame = engine.perception.camera.get_latest_frame()
        
        from robot.perception.vlm import LocalVLMGrounding
        vlm = getattr(engine.perception, "vlm", None) if hasattr(engine, "perception") and engine.perception else None
        if vlm is None:
            vlm = LocalVLMGrounding(enabled=True, device="amd_igpu")

        if frame is None:
            # Fallback when physical camera is in standby or offline
            return {
                "summary": f"AMD iGPU VLM inspection: Tabletop camera feed is in standby mode. Query: '{target_query}'. Ensure the camera is connected and facing the workspace.",
                "items_count": 0,
                "items": [],
                "device": "amd_igpu",
            }

        return vlm.inspect_frame(frame, query=target_query)

    engine.tool_registry.register(
        ToolDefinition(
            name="perception.inspect_tabletop",
            description="Perform deep visual inspection of the tabletop, desk, or workspace using the AMD Radeon 680M iGPU VLM. Identifies fine-grained object shapes, colors, spatial positions, and items on the table. Call this when asked 'what is on the table', 'inspect the desk', 'describe what items are in front of you', or 'check the workspace'.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific visual question or tabletop inspection prompt", "default": "Describe all items on the table"},
                },
            },
            handler=inspect_tabletop,
            permission_level=PermissionLevel.READ_ONLY,
            family="perception",
        )
    )

