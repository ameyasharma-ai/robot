"""Visual perception tools for LLM reasoning and camera snapshots."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

#: Where camera snapshots are persisted for the vision tooling and the gallery.
SNAPSHOT_DIR = Path("data/images/camera")
#: URL prefix the API serves :data:`SNAPSHOT_DIR` under.
SNAPSHOT_URL_PREFIX = "/data/images/camera"
#: Newest snapshots kept on disk; the tool writes one per call.
MAX_SNAPSHOT_FILES = 24
#: OCR text is untrusted, page-length output: keep only what a spoken answer needs.
MAX_OCR_CHARS = 2_000


def _store_snapshot(encoded: bytes, width: int, height: int) -> dict[str, Any]:
    """Write an encoded JPEG frame to disk and return model-safe metadata.

    Raw base64 is never returned: a camera JPEG is tens of kilobytes of text
    that the language model cannot read, and injecting it wastes the entire
    context window. Callers get a path the vision pipeline or UI can use.

    Args:
        encoded: JPEG-encoded frame bytes.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        dict[str, Any]: Path, URL, byte size and dimensions of the snapshot.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"snapshot_{int(time.time() * 1000)}.jpg"
    path = SNAPSHOT_DIR / filename
    path.write_bytes(encoded)

    # Only keep the newest frames; the robot takes snapshots on demand.
    snapshots = sorted(SNAPSHOT_DIR.glob("snapshot_*.jpg"), key=lambda p: p.name, reverse=True)
    for stale in snapshots[MAX_SNAPSHOT_FILES:]:
        try:
            stale.unlink()
        except OSError:
            pass

    return {
        "width": width,
        "height": height,
        "format": "jpeg",
        "size_bytes": len(encoded),
        "image_path": str(path.resolve()),
        "url": f"{SNAPSHOT_URL_PREFIX}/{filename}",
    }


def register_perception_tools(engine: "RobotEngine") -> None:
    """Register camera and perception tools with the engine's tool registry."""

    vlm_holder: dict[str, Any] = {}
    vlm_lock = threading.Lock()

    def _resolve_vlm() -> Any:
        """Return a single shared VLM instance for standalone inspection.

        Loading a vision model costs seconds and hundreds of megabytes, so the
        fallback instance is created once per registration instead of once per
        tool call.

        Returns:
            Any: The engine's VLM, or a cached standalone instance.
        """
        perception = getattr(engine, "perception", None)
        vlm = getattr(perception, "vlm", None) if perception else None
        if vlm is not None:
            return vlm
        with vlm_lock:
            if "vlm" not in vlm_holder:
                from robot.perception.vlm import LocalVLMGrounding

                vlm_holder["vlm"] = LocalVLMGrounding(enabled=True, device="amd_igpu")
            return vlm_holder["vlm"]

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

    async def capture_snapshot(quality: int = 80, **kwargs) -> dict[str, Any]:
        """Capture a snapshot from the robot's camera for inspection or visual reasoning."""
        if not hasattr(engine, "perception") or not engine.perception or not engine.perception.camera:
            return {"error": "Camera not available"}

        frame = engine.perception.camera.get_latest_frame()
        if frame is None:
            return {"error": "No frame captured from camera yet"}

        try:
            level = max(10, min(100, int(quality)))
        except (TypeError, ValueError):
            level = 80

        def _encode() -> dict[str, Any]:
            # JPEG encoding a full frame is CPU-bound work for hundreds of
            # milliseconds; it must not run on the event loop.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), level]
            success, encimg = cv2.imencode(".jpg", frame, encode_param)
            if not success:
                return {"error": "Failed to encode frame"}
            h, w = frame.shape[:2]
            return _store_snapshot(encimg.tobytes(), w, h)

        return await asyncio.to_thread(_encode)

    engine.tool_registry.register(
        ToolDefinition(
            name="camera.capture_snapshot",
            description="Capture a current frame snapshot from the robot's physical camera and return its stored path and dimensions.",
            parameters={
                "type": "object",
                "properties": {
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
            dist = p.get("distance", "near")
            pos = p.get("position", "center")
            cloth = f" (wearing {p['clothing_color']})" if p.get("clothing_color") else ""
            people_desc.append(f"Person at {pos}{cloth} ({dist} distance)")

        obj_desc = []
        for o in ws.get("objects", []):
            lbl = o.get("label", "item")
            pos = o.get("position", "center")
            col = f"{o['color']} " if o.get("color") else ""
            obj_desc.append(f"{col}{lbl} at {pos}")

        table_desc = []
        for t in ws.get("tabletop_items", []):
            lbl = t.get("object_name", "item")
            clk = t.get("position_clock", "12 o'clock")
            col = f"{t['color']} " if t.get("color") else ""
            table_desc.append(f"{col}{lbl} at {clk}")

        details = []
        if people_desc:
            details.append("People detected: " + ", ".join(people_desc))
        if obj_desc:
            details.append("Objects detected: " + ", ".join(obj_desc))
        elif table_desc:
            details.append("Tabletop items: " + ", ".join(table_desc))

        if not details:
            details.append("No distinct people or objects clearly recognized in the immediate camera frame. Suggest the user hold the object directly toward the camera.")

        summary = f"Visual camera feed: {'; '.join(details)}. Ambient lighting: {ws.get('lighting', 'normal')}. Motion: {ws.get('motion', 'none')}."
        return {
            "summary": summary,
            "people": people_desc,
            "objects": obj_desc,
            "tabletop": table_desc,
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

        if frame is None:
            # Fallback when physical camera is in standby or offline
            return {
                "summary": f"AMD iGPU VLM inspection: Tabletop camera feed is in standby mode. Query: '{target_query}'. Ensure the camera is connected and facing the workspace.",
                "items_count": 0,
                "items": [],
                "device": "amd_igpu",
            }

        # Model construction and inference both block; run them off the loop so
        # a spoken "what's on my table" never stalls audio or the API.
        return await asyncio.to_thread(lambda: _resolve_vlm().inspect_frame(frame, query=target_query))

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

    async def analyze_camera_frame(
        prompt: str = "Describe what you see in the camera frame in full detail",
        detail_level: str = "normal",
        extract_text: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform comprehensive visual question answering, OCR, and scene analysis on the camera view.

        Args:
            prompt: Visual query or instruction (e.g. 'Read the text on this paper', 'What brand is this?').
            detail_level: Granularity level ('concise', 'normal', 'high').
            extract_text: Whether to perform local OCR text extraction.

        Returns:
            Structured dictionary with analysis summary, detected objects, extracted text,
            spatial layout, lighting conditions, and the stored snapshot path.
        """
        target_prompt = (
            prompt
            or kwargs.get("query")
            or kwargs.get("question")
            or kwargs.get("text")
            or "Describe what you see in the camera frame in full detail"
        ).strip()
        eff_detail = str(detail_level or kwargs.get("detail", "normal")).lower()
        do_ocr = bool(extract_text if extract_text is not None else kwargs.get("ocr", True))

        frame = None
        if hasattr(engine, "perception") and engine.perception and engine.perception.camera:
            frame = engine.perception.camera.get_latest_frame()

        if frame is None or getattr(frame, "size", 0) == 0:
            return {
                "status": "standby",
                "summary": (
                    f"Local VLM inspection: Tabletop/robot camera is in standby mode. "
                    f"Prompt '{target_prompt}' could not be visually resolved. "
                    f"Please ensure the physical camera is connected and unmuted."
                ),
                "prompt": target_prompt,
                "detail_level": eff_detail,
                "detected_objects": [],
                "detected_text": "",
                "spatial_layout": "standby",
                "lighting": "unknown",
                "dominant_color": "unknown",
                "image_path": None,
                "device": "amd_igpu",
            }

        # 1. Encode and persist the exact analysed frame off the event loop
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, encimg = cv2.imencode(".jpg", frame, encode_param)
        stored: dict[str, Any] = {}
        if success:
            stored = _store_snapshot(encimg.tobytes(), frame.shape[1], frame.shape[0])

        # 2. Local optical grounding and OCR
        def _local_ground() -> dict[str, Any]:
            vlm_res = _resolve_vlm().inspect_frame(frame, query=target_prompt)
            detected_text = ""
            if do_ocr and success:
                try:
                    import io
                    import pytesseract
                    from PIL import Image

                    pil_img = Image.open(io.BytesIO(encimg.tobytes()))
                    detected_text = pytesseract.image_to_string(pil_img).strip()
                except Exception:
                    detected_text = ""
            detected_text = detected_text[:MAX_OCR_CHARS]
            return {"vlm_res": vlm_res, "detected_text": detected_text}

        local_analysis = await asyncio.to_thread(_local_ground)
        vlm_res = local_analysis["vlm_res"]
        detected_text = local_analysis["detected_text"]

        # 3. Cloud Frontier VLM Grounding (if Gemini configured and not forced local)
        gemini_summary: str | None = None
        gemini_cfg = engine.config.get("llm", {}).get("gemini", {}) if getattr(engine, "config", None) else {}
        gemini_key = gemini_cfg.get("api_key")
        if gemini_key and len(str(gemini_key)) > 10 and success and not kwargs.get("_local_only"):
            try:
                import base64
                from robot.llm import GeminiLLM

                b64_frame = base64.b64encode(encimg.tobytes()).decode("utf-8")
                vlm_client = GeminiLLM(
                    api_key=gemini_key,
                    base_url=gemini_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                    model=gemini_cfg.get("model", "gemini-3.1-flash-lite"),
                )
                vlm_msgs = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"You are the visual cortex of an intelligent humanoid robot. "
                                    f"User asks: '{target_prompt}'. Inspect the camera frame and answer accurately and concisely in 1-2 direct sentences, "
                                    f"specifically detailing colors, objects, text, physical position, and scene appearance."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"},
                            },
                        ],
                    }
                ]
                vlm_chunks = []
                async for chunk in vlm_client.stream_chat(vlm_msgs, max_tokens=150):
                    vlm_chunks.append(chunk)
                if vlm_chunks:
                    gemini_summary = "".join(vlm_chunks).strip()
            except Exception as exc:
                logger.debug("gemini_vlm_inspection_failed", error=str(exc))

        # 4. Synthesize natural language result
        if gemini_summary and eff_detail != "concise":
            combined_summary = gemini_summary
            if detected_text and detected_text not in combined_summary:
                combined_summary += f" Readable text: '{detected_text[:120]}'."
        else:
            base_summary = str(vlm_res.get("summary", ""))
            summary_parts = [base_summary]
            if detected_text:
                clean_snippet = detected_text.replace("\n", " ").strip()
                if len(clean_snippet) > 160:
                    clean_snippet = clean_snippet[:160] + "..."
                summary_parts.append(f"Readable text extracted from camera: '{clean_snippet}'.")

            if eff_detail == "concise" and vlm_res.get("items"):
                items = vlm_res.get("items", [])
                summary_parts = [
                    f"Camera view: {len(items)} item(s) present "
                    f"({', '.join(str(i.get('color', '')) + ' ' + str(i.get('shape', '')) for i in items[:3])})."
                ]
                if detected_text:
                    summary_parts.append(f"Text detected: '{detected_text[:80]}'.")

            combined_summary = " ".join(summary_parts)

        return {
            "status": "success",
            "summary": combined_summary,
            "prompt": target_prompt,
            "detail_level": eff_detail,
            "detected_objects": vlm_res.get("items", []),
            "detected_text": detected_text,
            "lighting": vlm_res.get("lighting", "normal"),
            "dominant_color": vlm_res.get("dominant_color", "neutral"),
            "resolution": vlm_res.get("resolution", f"{frame.shape[1]}x{frame.shape[0]}"),
            "duration_ms": vlm_res.get("duration_ms", 0.0),
            "image_path": stored.get("image_path"),
            "url": stored.get("url"),
            "device": "gemini_vlm" if gemini_summary else "amd_igpu",
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="perception.analyze_camera_frame",
            description=(
                "Perform comprehensive visual inspection, OCR text extraction, and scene analysis through the robot's physical camera using the Local VLM. "
                "Answers visual questions, reads text from physical documents, books, medicine labels, whiteboard notes, and identifies object shapes/colors. "
                "Call this when asked 'read this paper', 'what text is on this', 'analyze what you see', 'what brand/type is this object', or 'describe the scene in detail'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Visual question or inspection prompt (e.g. 'Read the text on the document', 'What item is in front of me?')",
                        "default": "Describe what you see in the camera frame in full detail",
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["concise", "normal", "high"],
                        "default": "normal",
                        "description": "Desired detail granularity of visual report",
                    },
                    "extract_text": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to perform optical character recognition (OCR) on the camera frame",
                    },
                },
            },
            handler=analyze_camera_frame,
            permission_level=PermissionLevel.READ_ONLY,
            family="perception",
        )
    )

    engine.tool_registry.register(
        ToolDefinition(
            name="perception.vlm_inspect",
            description="Alias for perception.analyze_camera_frame. Perform visual analysis and OCR text reading on camera frame.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "default": "Describe what you see"},
                    "detail_level": {"type": "string", "enum": ["concise", "normal", "high"], "default": "normal"},
                    "extract_text": {"type": "boolean", "default": True},
                },
            },
            handler=analyze_camera_frame,
            permission_level=PermissionLevel.READ_ONLY,
            family="perception",
        )
    )


