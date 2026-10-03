"""Built-in character and robotic motion tools for LLM reasoning and physical avatar actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot.motion.actions import validate_action
from robot.motion.planner import MotionPlanner
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine


def register_motion_tools(engine: "RobotEngine") -> None:
    """Register character and motion tools with the engine's tool registry."""
    planner = MotionPlanner()

    async def execute_action(
        action: str,
        side: str | None = None,
        duration: float = 1.0,
        intensity: float = 1.0,
        repetitions: int = 1,
        direction: str = "forward",
        angle: float = 0.0,
        distance: float = 1.0,
        emotion: str | None = None,
    ) -> dict[str, Any]:
        """Execute a validated semantic movement on the 3D character and future physical robot."""
        req = {
            "action": action,
            "side": side,
            "duration": duration,
            "intensity": intensity,
            "repetitions": repetitions,
            "direction": direction,
            "angle": angle,
            "distance": distance,
            "emotion": emotion,
        }
        valid, semantic_action, err = validate_action(req)
        if not valid or semantic_action is None:
            return {"error": err}

        # Plan articulated sequence (weight shift, balance, etc.)
        sequence = planner.plan_action(req)
        serialized_seq = [a.to_dict() for a in sequence]

        # Stage on engine so motion triggers in exact synchrony with upcoming speech audio
        if hasattr(engine, "_pending_motion"):
            engine._pending_motion = semantic_action.to_dict()
        else:
            # Fallback to direct EventBus emission if called outside engine conversational loop
            await engine.events.emit(
                "character.action",
                action=semantic_action.to_dict(),
                sequence=serialized_seq,
            )

        return {
            "status": "executing",
            "action": semantic_action.action,
            "side": semantic_action.side,
            "sequence_length": len(sequence),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="character.execute_action",
            description="Control the robot's physical visual body to perform gestures and movements (e.g. raise_hand, wave, kick, walk, turn, point, look_at, sit, stand, set_emotion, stop).",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
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
                            "set_emotion",
                            "stop",
                        ],
                        "description": "Semantic action to execute",
                    },
                    "side": {
                        "type": "string",
                        "enum": ["left", "right", "both"],
                        "description": "Limb side if applicable",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds (default 1.0)",
                    },
                    "intensity": {
                        "type": "number",
                        "description": "Intensity between 0.0 and 1.0",
                    },
                    "repetitions": {
                        "type": "integer",
                        "description": "Repetitions for cyclic motions like waving",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "left", "right"],
                    },
                    "angle": {
                        "type": "number",
                        "description": "Turn angle in degrees",
                    },
                    "distance": {
                        "type": "number",
                        "description": "Walk distance in meters",
                    },
                    "emotion": {
                        "type": "string",
                        "enum": ["neutral", "happy", "sad", "surprised", "thinking", "angry", "confused", "love", "affectionate", "shy"],
                    },
                },
                "required": ["action"],
            },
            handler=execute_action,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="motion",
        )
    )

    async def stop_motion() -> dict[str, Any]:
        """Emergency stop all ongoing character motions and blend gracefully to neutral idle."""
        await engine.events.emit("character.interrupt", interrupt=True)
        return {"status": "stopped", "mode": "idle"}

    engine.tool_registry.register(
        ToolDefinition(
            name="character.stop",
            description="Immediately stop current gesture or motion and return safely to natural idle.",
            parameters={"type": "object", "properties": {}},
            handler=stop_motion,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="motion",
        )
    )
