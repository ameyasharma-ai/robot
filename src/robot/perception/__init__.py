"""Perception subsystem package for Robot Core.

Exports:
- CameraCapture: OpenCV-based asynchronous webcam capture
- PerceptionEngine: Continuous lightweight face detection, motion, and scene illumination analysis
- WorldState, DetectedPerson: Physical world state data structures
"""

from robot.perception.camera import CameraCapture
from robot.perception.engine import PerceptionEngine
from robot.perception.vlm import LocalVLMGrounding
from robot.perception.world import DetectedObject, DetectedPerson, TabletopItem, WorldState

__all__ = [
    "CameraCapture",
    "PerceptionEngine",
    "LocalVLMGrounding",
    "WorldState",
    "DetectedPerson",
    "DetectedObject",
    "TabletopItem",
]

