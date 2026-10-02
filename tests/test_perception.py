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
