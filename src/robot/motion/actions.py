"""Motion and kinematic animation models for the humanoid character.

Provides:
- Semantic Action schema (raise_hand, wave, kick, walk, turn, point, look_at, sit, stand, set_emotion, idle)
- Action sequence definition and priority levels
- Kinematic joints and bone specifications
- Validation rules for physical and virtual humanoid movement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class ActionPriority(IntEnum):
    LOW = 0          # Subtle procedural shifts, breathing
    NORMAL = 1       # Idle gestures, looks
    HIGH = 2         # User commands, purposeful gestures
    INTERRUPT = 3    # Emergency stop, safety cancellation


class KinematicChain(str, Enum):
    FULL_BODY = "full_body"
    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    HEAD = "head"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    FACE = "face"


@dataclass
class SemanticAction:
    """High-level semantic action dispatched to CharacterEngine or Physical Robot."""

    action: str  # e.g. "raise_hand", "wave", "kick", "walk", "point", "look_at", "sit", "stand", "set_emotion", "idle", "stop"
    side: Optional[str] = None  # "left", "right", "both"
    target: Optional[dict[str, float]] = None  # e.g. {"x": 0.0, "y": 1.5, "z": 2.0}
    direction: str = "forward"  # "forward", "backward", "left", "right"
    duration: float = 1.0  # seconds
    intensity: float = 1.0  # 0.0 to 1.0
    repetitions: int = 1
    angle: float = 0.0  # degrees for turn
    distance: float = 1.0  # meters for walk
    emotion: Optional[str] = None  # "neutral", "happy", "sad", "surprised", "thinking", "angry", "confused"
    priority: int = ActionPriority.HIGH
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for WebSocket/REST transfer."""
        d: dict[str, Any] = {"action": self.action}
        if self.side is not None:
            d["side"] = self.side
        if self.target is not None:
            d["target"] = self.target
        if self.direction != "forward":
            d["direction"] = self.direction
        if self.duration != 1.0:
            d["duration"] = self.duration
        if self.intensity != 1.0:
            d["intensity"] = self.intensity
        if self.repetitions != 1:
            d["repetitions"] = self.repetitions
        if self.angle != 0.0:
            d["angle"] = self.angle
        if self.distance != 1.0:
            d["distance"] = self.distance
        if self.emotion is not None:
            d["emotion"] = self.emotion
        if self.priority != ActionPriority.HIGH:
            d["priority"] = self.priority
        if self.parameters:
            d["parameters"] = self.parameters
        return d


def validate_action(action_dict: dict[str, Any]) -> tuple[bool, Optional[SemanticAction], Optional[str]]:
    """Validate and parse a raw semantic action request."""
    if not isinstance(action_dict, dict):
        return False, None, "Action must be a JSON object"

    name = action_dict.get("action")
    if not name or not isinstance(name, str):
        return False, None, "Missing or invalid 'action' name"

    name = name.lower().strip()
    action_aliases = {
        "lean_in": "think",
        "lean_forward": "think",
        "listening": "nod",
        "listen": "nod",
        "yes": "nod",
        "no": "shake_head",
        "look": "look_at",
        "wave_hand": "wave",
        "raise_arm": "raise_hand",
    }
    if name in action_aliases:
        name = action_aliases[name]

    valid_actions = {
        "idle",
        "raise_hand",
        "wave",
        "kick",
        "walk",
        "turn",
        "point",
        "look_at",
        "sit",
        "stand",
        "nod",
        "shake_head",
        "tilt_head",
        "shrug",
        "bow",
        "clap",
        "think",
        "cheer",
        "hands_on_hips",
        "heart_gesture",
        "set_emotion",
        "speak_start",
        "speak_stop",
        "stop",
        "reset",
        "reset_position",
    }
    if name not in valid_actions:
        return False, None, f"Unknown action '{name}'. Valid: {sorted(list(valid_actions))}"

    side = action_dict.get("side")
    if side is not None:
        side = str(side).lower()
        if side not in ("left", "right", "both"):
            return False, None, f"Invalid side '{side}'. Must be 'left', 'right', or 'both'"

    action = SemanticAction(
        action=name,
        side=side,
        target=action_dict.get("target"),
        direction=str(action_dict.get("direction", "forward")).lower(),
        duration=max(0.1, min(30.0, float(action_dict.get("duration", 1.0)))),
        intensity=max(0.0, min(1.0, float(action_dict.get("intensity", 1.0)))),
        repetitions=max(1, min(20, int(action_dict.get("repetitions", 1)))),
        angle=float(action_dict.get("angle", 0.0)),
        distance=max(0.0, min(20.0, float(action_dict.get("distance", 1.0)))),
        emotion=action_dict.get("emotion"),
        priority=int(action_dict.get("priority", ActionPriority.HIGH)),
        parameters=action_dict.get("parameters", {}),
    )
    return True, action, None
