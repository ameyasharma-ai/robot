"""Unit and integration tests for universal visual intelligence, optical colorimetry, and multimodal grounding."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from robot.core.engine import RobotEngine
from robot.perception.engine import extract_roi_color
from robot.perception.world import DetectedObject, DetectedPerson, TabletopItem, WorldState
from robot.tools import ToolRouter
from robot.tools.builtins import register_builtin_tools


def test_extract_roi_color_primary_colors() -> None:
    """Verify extract_roi_color correctly identifies primary and neutral colors in HSV space."""
    # Test Red (BGR: [0, 0, 255])
    img_red_bgr = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)
    img_red_hsv = cv2.cvtColor(img_red_bgr, cv2.COLOR_BGR2HSV)
    assert extract_roi_color(img_red_hsv, 0, 0, 100, 100) == "red"

    # Test Green (BGR: [0, 255, 0])
    img_green_bgr = np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8)
    img_green_hsv = cv2.cvtColor(img_green_bgr, cv2.COLOR_BGR2HSV)
    assert extract_roi_color(img_green_hsv, 0, 0, 100, 100) == "green"

    # Test Blue (BGR: [255, 0, 0])
    img_blue_bgr = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
    img_blue_hsv = cv2.cvtColor(img_blue_bgr, cv2.COLOR_BGR2HSV)
    assert extract_roi_color(img_blue_hsv, 0, 0, 100, 100) == "blue"

    # Test White (BGR: [255, 255, 255])
    img_white_bgr = np.full((100, 100, 3), (255, 255, 255), dtype=np.uint8)
    img_white_hsv = cv2.cvtColor(img_white_bgr, cv2.COLOR_BGR2HSV)
    assert extract_roi_color(img_white_hsv, 0, 0, 100, 100) == "white"

    # Test Black (BGR: [10, 10, 10])
    img_black_bgr = np.full((100, 100, 3), (10, 10, 10), dtype=np.uint8)
    img_black_hsv = cv2.cvtColor(img_black_bgr, cv2.COLOR_BGR2HSV)
    assert extract_roi_color(img_black_hsv, 0, 0, 100, 100) == "black"

    # Test Yellow (BGR: [0, 255, 255])
    img_yellow_bgr = np.full((100, 100, 3), (0, 255, 255), dtype=np.uint8)
    img_yellow_hsv = cv2.cvtColor(img_yellow_bgr, cv2.COLOR_BGR2HSV)
    assert extract_roi_color(img_yellow_hsv, 0, 0, 100, 100) == "yellow"


def test_world_state_color_serialization() -> None:
    """Verify WorldState serializes clothing and object color metadata accurately."""
    ws = WorldState()
    ws.people = [
        DetectedPerson(
            id="user",
            position="center",
            distance_estimate="near",
            confidence=0.95,
            looking_at_robot=True,
            clothing_color="dark blue",
            bbox=[100, 100, 200, 200],
        )
    ]
    ws.objects = [
        DetectedObject(
            label="bottle",
            confidence=0.88,
            bbox=[150, 150, 50, 120],
            color="red",
        )
    ]
    ws.tabletop_items = [
        TabletopItem(
            object_name="notebook",
            color="green",
            position_clock="2 o'clock",
            confidence=0.91,
            bounding_box=[200, 300, 100, 150],
        )
    ]

    compact = ws.to_compact()
    assert compact["people"][0]["clothing_color"] == "dark blue"
    assert compact["objects"][0]["color"] == "red"
    assert compact["tabletop_items"][0]["color"] == "green"

    yaml_str = ws.to_yaml_str()
    assert "user (dark blue)" in yaml_str
    assert "bottle (red)" in yaml_str
    assert "notebook (green)" in yaml_str


def test_visual_query_recognition() -> None:
    """Test RobotEngine._is_visual_query detection across varied natural phrasing."""
    engine = RobotEngine()

    positive_queries: list[str] = [
        "What are you seeing?",
        "what color is it?",
        "What color is this cup?",
        "What color is my shirt?",
        "What am I wearing?",
        "What do you observe in front of you?",
        "Look at this and tell me what it is",
        "What am I holding?",
        "Read this page",
        "What does this text say?",
        "Describe what's in the camera feed",
    ]

    for query in positive_queries:
        assert engine._is_visual_query(query) is True, f"Failed on: {query}"

    negative_queries: list[str] = [
        "How are you doing today?",
        "Tell me a joke",
        "What is the capital of France?",
        "Play some music",
        "Set a timer for 10 minutes",
    ]

    for query in negative_queries:
        assert engine._is_visual_query(query) is False, f"False positive on: {query}"


def test_tool_router_perception_selection() -> None:
    """Test ToolRouter resolves perception tools for visual reasoning queries."""
    engine = RobotEngine()
    register_builtin_tools(engine)
    router = ToolRouter(engine.tool_registry)

    queries: list[str] = [
        "What color is my shirt?",
        "What are you seeing right now?",
        "What color is it?",
        "What am I holding in my hand?",
        "Can you read this document?",
    ]

    for q in queries:
        selected = router.select_tools(q)
        assert any(
            t in selected for t in ["perception.what_do_you_see", "perception.analyze_camera_frame", "perception.inspect_tabletop"]
        ), f"Failed routing for '{q}', got: {selected}"
