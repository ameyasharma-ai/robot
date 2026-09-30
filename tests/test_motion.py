"""Unit tests for character motion planner, semantic action validation, and sequencing."""

import pytest
from robot.motion.actions import ActionPriority, SemanticAction, validate_action
from robot.motion.planner import MotionPlanner


def test_action_validation_success():
    valid, action, err = validate_action({
        "action": "raise_hand",
        "side": "right",
        "duration": 1.5,
        "intensity": 0.9,
    })
    assert valid is True
    assert err is None
    assert action is not None
    assert action.action == "raise_hand"
    assert action.side == "right"
    assert action.duration == 1.5
    assert action.intensity == 0.9


def test_action_validation_invalid_action():
    valid, action, err = validate_action({"action": "fly_to_moon"})
    assert valid is False
    assert action is None
    assert "Unknown action" in err


def test_action_validation_invalid_side():
    valid, action, err = validate_action({"action": "raise_hand", "side": "upside_down"})
    assert valid is False
    assert "Invalid side" in err


def test_kick_motion_planning_sequence():
    planner = MotionPlanner()
    plan = planner.plan_action({
        "action": "kick",
        "side": "right",
        "direction": "forward",
        "intensity": 0.85,
    })

    # Should contain: weight shift to left -> kick right -> restore balance -> idle
    assert len(plan) == 4
    assert plan[0].action == "weight_shift"
    assert plan[0].side == "left"  # Contralateral weight support
    assert plan[1].action == "kick"
    assert plan[1].side == "right"
    assert plan[2].action == "balance_restore"
    assert plan[3].action == "idle"


def test_wave_motion_planning_sequence():
    planner = MotionPlanner()
    plan = planner.plan_action({
        "action": "wave",
        "side": "left",
        "repetitions": 3,
    })
    assert len(plan) == 3
    assert plan[0].action == "raise_hand"
    assert plan[0].side == "left"
    assert plan[1].action == "wave"
    assert plan[1].side == "left"
    assert plan[2].action == "idle"


def test_natural_language_command_parsing():
    planner = MotionPlanner()

    # 1. Raise right hand
    res1 = planner.parse_natural_command("Could you raise your right hand?")
    assert len(res1) == 1
    assert res1[0]["action"] == "raise_hand"
    assert res1[0]["side"] == "right"

    # 2. Wave at me
    res2 = planner.parse_natural_command("Wave at me!")
    assert len(res2) == 1
    assert res2[0]["action"] == "wave"

    # 3. Kick left leg
    res3 = planner.parse_natural_command("Kick with your left leg now")
    assert len(res3) == 1
    assert res3[0]["action"] == "kick"
    assert res3[0]["side"] == "left"

    # 4. Simultaneous combination: Raise left hand and kick right leg
    res4 = planner.parse_natural_command("Raise your left hand and kick with your right leg")
    actions = [a["action"] for a in res4]
    assert "raise_hand" in actions
    assert "kick" in actions

    # 5. Stop command
    res5 = planner.parse_natural_command("Emergency stop immediately!")
    assert len(res5) == 1
    assert res5[0]["action"] == "stop"


def test_motion_stop_interruption_priority():
    planner = MotionPlanner()
    plan = planner.plan_action({"action": "stop"})
    assert len(plan) == 2
    assert plan[0].action == "stop"
    assert plan[0].priority == ActionPriority.INTERRUPT
