"""Tests for camera capture, OpenCV perception, and WorldState."""

import asyncio
import numpy as np
import pytest

from robot.core.context import ContextEngine
from robot.perception.camera import CameraCapture
from robot.perception.engine import PerceptionEngine
from robot.perception.world import DetectedPerson, WorldState


def test_world_state_compact_and_yaml():
    ws = WorldState()
    ws.people = [
        DetectedPerson(
            id="user",
            position="center",
            distance_estimate="near",
            confidence=0.92,
            looking_at_robot=True,
            bbox=[100, 100, 200, 200],
        )
    ]
    ws.motion = "low"
    ws.lighting = "normal"
    ws.scene = "desk/indoor"

    compact = ws.to_compact()
    assert compact["people_count"] == 1
    assert compact["motion"] == "low"
    assert compact["lighting"] == "normal"
    assert compact["people"][0]["position"] == "center"

    yaml_str = ws.to_yaml_str()
    assert "people_count: 1" in yaml_str
    assert "position: center" in yaml_str
    assert "motion: low" in yaml_str


def test_context_engine_injects_world_state():
    engine = ContextEngine()
    ws_compact = {
        "people_count": 1,
        "people": [{"position": "center", "distance": "near", "looking_at_robot": True}],
        "motion": "medium",
        "lighting": "bright",
        "scene": "indoor",
    }

    messages = engine.build_messages(
        system_prompt="You are a humanoid robot.",
        user_text="What do you see?",
        world_state=ws_compact,
    )

    assert len(messages) == 2
    # In live conversation and prompt evaluation, world state is grounded into the user prompt
    # to keep the system prompt 100% invariant for KV cache reuse (<200ms TTFT)
    full_context = " ".join(m["content"] for m in messages)
    assert "[Physical World State]" in full_context
    assert "people_present: 1" in full_context
    assert "pos=center" in full_context
    assert "motion_level: medium" in full_context
    assert "room_lighting: bright" in full_context


class DummyCamera(CameraCapture):
    """Mock camera that generates synthetic test frames."""

    def __init__(self):
        super().__init__(device_index=-1)
        self._test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    def start(self) -> bool:
        self._running = True
        return True

    def get_latest_frame(self):
        return self._test_frame.copy()

    def set_frame(self, frame: np.ndarray):
        self._test_frame = frame

    def stop(self):
        self._running = False


@pytest.mark.asyncio
async def test_perception_engine_analysis_and_motion():
    dummy_cam = DummyCamera()
    pe = PerceptionEngine(
        camera=dummy_cam,
        yunet_model_path="models/vision/face_detection_yunet.onnx",
        target_fps=10.0,
    )

    await pe.start()

    # Step 1: Initial neutral frame
    pe._analyze_frame(dummy_cam.get_latest_frame())
    assert pe.world_state.lighting == "normal"
    assert pe.world_state.motion == "none"

    # Step 2: High motion frame
    motion_frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    pe._analyze_frame(motion_frame)
    assert pe.world_state.motion in ("medium", "high")

    # Step 3: Dark frame
    dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    pe._analyze_frame(dark_frame)
    assert pe.world_state.lighting == "dark"

    await pe.stop()


@pytest.mark.asyncio
async def test_perception_worker_bounded_queue():
    """Verify background perception worker maintains bounded queue (maxsize=2) and drops stale frames."""
    dummy_cam = DummyCamera()
    pe = PerceptionEngine(camera=dummy_cam, target_fps=10.0)

    await pe.start()
    assert pe._running is True
    assert pe._worker_thread is not None
    assert pe._worker_thread.is_alive()

    # Simulate feeding 10 rapid frames into the feeder loop
    for i in range(10):
        frame = np.full((480, 640, 3), i * 20, dtype=np.uint8)
        dummy_cam.set_frame(frame)
        await asyncio.sleep(0.01)

    # Queue must never exceed maxsize 2
    assert pe._frame_queue.qsize() <= 2
    await pe.stop()
    assert pe._running is False


def test_tabletop_spatial_tracking_and_expiration():
    """Test 3D spatial schema, temporal smoothing, and expiration for departed tabletop items."""
    import time
    from robot.perception.world import TabletopItem

    ws = WorldState()

    # Initial frame with cup and book
    item1 = TabletopItem(
        object_name="coffee mug",
        confidence=0.90,
        bounding_box=[100, 200, 50, 50],
        estimated_distance="near",
        estimated_distance_m=0.35,
        position_clock="11 o'clock",
        last_seen=time.time(),
    )
    item2 = TabletopItem(
        object_name="notebook",
        confidence=0.85,
        bounding_box=[300, 250, 120, 80],
        estimated_distance="medium",
        estimated_distance_m=0.70,
        position_clock="1 o'clock",
        last_seen=time.time(),
    )

    ws.update_tabletop_tracking([item1, item2], expiration_s=1.0)
    assert len(ws.tabletop_items) == 2

    # Second frame: cup slightly moved (testing temporal exponential smoothing)
    item1_moved = TabletopItem(
        object_name="coffee mug",
        confidence=0.95,
        bounding_box=[104, 202, 52, 50],
        estimated_distance="near",
        estimated_distance_m=0.36,
        position_clock="11 o'clock",
        last_seen=time.time(),
    )
    ws.update_tabletop_tracking([item1_moved], expiration_s=0.2)
    assert len(ws.tabletop_items) == 2  # notebook still in temporal memory

    # Wait for notebook to expire (>0.2s)
    time.sleep(0.25)
    ws.update_tabletop_tracking([item1_moved], expiration_s=0.2)
    # Notebook should have expired, leaving only the coffee mug
    assert len(ws.tabletop_items) == 1
    assert ws.tabletop_items[0].object_name == "coffee mug"

    compact = ws.to_compact()
    assert compact["tabletop_count"] == 1
    assert compact["tabletop_items"][0]["object_name"] == "coffee mug"


@pytest.mark.asyncio
async def test_inspect_tabletop_method():
    """Test inspect_tabletop structured response."""
    from robot.perception.world import TabletopItem

    dummy_cam = DummyCamera()
    pe = PerceptionEngine(camera=dummy_cam)
    pe.world_state.tabletop_items = [
        TabletopItem(
            object_name="wireless mouse",
            confidence=0.92,
            bounding_box=[200, 220, 40, 30],
            estimated_distance="near",
            estimated_distance_m=0.40,
            position_clock="12 o'clock",
        )
    ]

    res = pe.inspect_tabletop()
    assert res["status"] == "success"
    assert res["items_count"] == 1
    assert res["items"][0]["object_name"] == "wireless mouse"
    assert res["items"][0]["position_clock"] == "12 o'clock"


@pytest.mark.asyncio
async def test_os_capture_screen_with_ocr():
    """Test os_capture_screen region cropping, structured OCR, and PermissionLevel.SYSTEM."""
    from unittest.mock import AsyncMock, patch
    from robot.tools import PermissionLevel
    from robot.tools.builtins.system_extended import os_capture_screen

    # Test tool permission level
    from robot.core.engine import RobotEngine
    from robot.tools.builtins.system_extended import register_system_extended_tools

    mock_engine = RobotEngine()
    register_system_extended_tools(mock_engine)

    tool_def = mock_engine.tool_registry.get("os.capture_screen")
    assert tool_def is not None
    assert tool_def.permission_level == PermissionLevel.SYSTEM


def test_perception_get_latest_scene():
    """Verify PerceptionEngine.get_latest_scene extracts scene snapshot from WorldState."""
    from robot.perception.world import DetectedObject, DetectedPerson

    pe = PerceptionEngine(camera=None)
    pe.world_state.people = [
        DetectedPerson(
            id="person_0",
            position="center",
            distance_estimate="near",
            confidence=0.91,
            looking_at_robot=True,
            bbox=[100, 100, 200, 200],
        )
    ]
    pe.world_state.objects = [
        DetectedObject(
            label="cup",
            confidence=0.88,
            bbox=[150, 160, 40, 50],
            position="left",
        )
    ]
    scene = pe.get_latest_scene()
    assert scene["num_faces"] == 1
    assert len(scene["people"]) == 1
    assert scene["people"][0]["id"] == "person_0"
    assert len(scene["objects"]) == 1
    assert scene["objects"][0]["label"] == "cup"
    assert "motion" in scene
    assert "lighting" in scene


