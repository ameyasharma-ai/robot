"""Deterministic high-level semantic motion planner and sequence generator.

Features:
- Translates high-level natural intents into validated kinematic animation sequences
- Enforces natural humanoid balance, weight shifting, and stabilization
- Supports parallel compatible motions (e.g. walk + look_at + wave + speak)
- Resolves conflicts and coordinates limbs
- Immediate cancellation and smooth blend-to-idle for interruptions
"""

from __future__ import annotations

import re
from typing import Any, Optional

from robot.motion.actions import ActionPriority, SemanticAction, validate_action


class MotionPlanner:
    """Plans, validates, and sequences humanoid motions."""

    def __init__(self) -> None:
        self._current_sequence: list[SemanticAction] = []
        self._is_active: bool = False

    def plan_action(self, action_dict: dict[str, Any]) -> list[SemanticAction]:
        """Convert a semantic action request into an articulated motion plan.

        For example:
        'kick' with right leg:
          1. Shift weight to left leg & stabilize pelvis
          2. Lift right leg
          3. Extend leg forward
          4. Retract leg
          5. Place foot & restore balance to idle
        """
        valid, action, err = validate_action(action_dict)
        if not valid or action is None:
            raise ValueError(f"Invalid motion request: {err}")

        plan: list[SemanticAction] = []

        if action.action == "kick":
            side = action.side or "right"
            plan.append(SemanticAction(
                action="weight_shift",
                side="left" if side == "right" else "right",
                duration=0.25,
                parameters={"target_support": "contralateral"}
            ))
            plan.append(SemanticAction(
                action="kick",
                side=side,
                direction=action.direction,
                duration=action.duration,
                intensity=action.intensity,
            ))
            plan.append(SemanticAction(
                action="balance_restore",
                duration=0.3,
            ))
            plan.append(SemanticAction(action="idle", duration=0.5))

        elif action.action == "raise_hand":
            side = action.side or "right"
            plan.append(SemanticAction(
                action="raise_hand",
                side=side,
                duration=action.duration,
                intensity=action.intensity,
            ))

        elif action.action == "wave":
            side = action.side or "right"
            plan.append(SemanticAction(
                action="raise_hand",
                side=side,
                duration=0.3,
            ))
            plan.append(SemanticAction(
                action="wave",
                side=side,
                duration=action.duration,
                repetitions=action.repetitions,
            ))
            plan.append(SemanticAction(action="idle", duration=0.4))

        elif action.action == "point":
            side = action.side or "right"
            plan.append(SemanticAction(
                action="point",
                side=side,
                target=action.target or {"x": 0.5, "y": 1.2, "z": 1.5},
                duration=action.duration,
            ))

        elif action.action == "walk":
            plan.append(SemanticAction(
                action="walk",
                direction=action.direction,
                distance=action.distance,
                duration=action.duration,
            ))

        elif action.action == "turn":
            plan.append(SemanticAction(
                action="turn",
                angle=action.angle or 180.0,
                duration=action.duration,
            ))

        elif action.action == "stop":
            plan.append(SemanticAction(
                action="stop",
                priority=ActionPriority.INTERRUPT,
            ))
            plan.append(SemanticAction(action="idle", duration=0.3))

        elif action.action in ("reset", "reset_position"):
            plan.append(SemanticAction(
                action="reset",
                priority=ActionPriority.INTERRUPT,
            ))
            plan.append(SemanticAction(action="idle", duration=0.3))

        else:
            plan.append(action)

        return plan

    def parse_natural_command(self, text: str) -> list[dict[str, Any]]:
        """Extract semantic actions from natural language user commands."""
        text_lower = text.lower().strip()
        actions: list[dict[str, Any]] = []

        # 0. Reset / Recenter / Default pose
        if re.search(r"\b(reset|recenter|default pose|stand straight|straighten up|back to normal)\b", text_lower):
            return [{"action": "reset"}]

        # 1. Stop / Freeze / Cancel
        if re.search(r"\b(stop|halt|freeze|cancel|pause)\b", text_lower):
            return [{"action": "stop"}]

        # 2. Raise hand / arm
        if "raise" in text_lower or "lift" in text_lower:
            if "left" in text_lower and ("hand" in text_lower or "arm" in text_lower):
                actions.append({"action": "raise_hand", "side": "left", "duration": 1.2})
            if "right" in text_lower and ("hand" in text_lower or "arm" in text_lower):
                actions.append({"action": "raise_hand", "side": "right", "duration": 1.2})
            if not ("left" in text_lower or "right" in text_lower) and ("hand" in text_lower or "arm" in text_lower):
                actions.append({"action": "raise_hand", "side": "right", "duration": 1.2})

        # 3. Wave
        if "wave" in text_lower:
            side = "left" if "left" in text_lower else "right"
            actions.append({"action": "wave", "side": side, "repetitions": 3, "duration": 2.0})

        # 4. Kick
        if "kick" in text_lower:
            side = "left" if "left" in text_lower else "right"
            actions.append({"action": "kick", "side": side, "direction": "forward", "intensity": 0.85})

        # 5. Point
        if "point" in text_lower:
            side = "left" if "left" in text_lower else "right"
            actions.append({"action": "point", "side": side, "target": {"x": 0.8, "y": 1.4, "z": 2.0}})

        # 6. Walk
        if "walk" in text_lower or "step" in text_lower:
            direction = "forward"
            if "backward" in text_lower or "back" in text_lower:
                direction = "backward"
            actions.append({"action": "walk", "direction": direction, "distance": 1.5, "duration": 3.0})

        # 7. Turn
        if "turn" in text_lower or "rotate" in text_lower:
            angle = 180.0 if ("around" in text_lower or "180" in text_lower) else 90.0
            if "left" in text_lower:
                angle = -angle
            actions.append({"action": "turn", "angle": angle, "duration": 2.0})

        # 8. Look at me / Look around
        if "look" in text_lower:
            if "me" in text_lower or "camera" in text_lower or "user" in text_lower:
                actions.append({"action": "look_at", "target": {"x": 0.0, "y": 1.5, "z": 2.0}})
            else:
                actions.append({"action": "look_at", "target": {"x": 1.0, "y": 1.4, "z": 1.5}})

        # 9. Sit / Stand
        if "sit" in text_lower:
            actions.append({"action": "sit", "duration": 2.0})
        elif "stand" in text_lower:
            actions.append({"action": "stand", "duration": 2.0})

        # 10. Expressive gestures
        if re.search(r"\b(nod|agree|yes)\b", text_lower) and not re.search(r"\b(say yes|yesterday)\b", text_lower):
            actions.append({"action": "nod", "duration": 1.4, "repetitions": 2})
        elif re.search(r"\b(shake head|disagree|no)\b", text_lower) and not re.search(r"\b(say no|not)\b", text_lower):
            actions.append({"action": "shake_head", "duration": 1.4, "repetitions": 2})
        elif re.search(r"\b(tilt head|curious|wonder)\b", text_lower):
            actions.append({"action": "tilt_head", "duration": 1.5})
        elif re.search(r"\b(shrug|dunno|who knows|i don'?t know)\b", text_lower):
            actions.append({"action": "shrug", "duration": 1.6})
        elif re.search(r"\b(bow|take a bow|curtsey|respect)\b", text_lower):
            actions.append({"action": "bow", "duration": 2.2})
        elif re.search(r"\b(clap|applaud|applause|bravo)\b", text_lower):
            actions.append({"action": "clap", "duration": 2.2, "repetitions": 4})
        elif re.search(r"\b(think|ponder|contemplate|hmm)\b", text_lower):
            actions.append({"action": "think", "duration": 2.5})
        elif re.search(r"\b(cheer|hooray|celebrate|yay)\b", text_lower):
            actions.append({"action": "cheer", "duration": 2.0})
        elif re.search(r"\b(hands on hips|pose|confident)\b", text_lower):
            actions.append({"action": "hands_on_hips", "duration": 2.5})
        elif re.search(r"\b(heart|love you|affection)\b", text_lower):
            actions.append({"action": "heart_gesture", "duration": 2.2})

        # 11. Emotions
        for emotion in ["happy", "sad", "surprised", "angry", "confused", "thinking", "neutral"]:
            if emotion in text_lower:
                actions.append({"action": "set_emotion", "emotion": emotion})
                break

        return actions
