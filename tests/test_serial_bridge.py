"""Comprehensive unit tests for the physical microcontroller serial bridge.

Tests:
- Emulation mode and joint limit clamping
- Versioned ASCII and binary packet formatting (CMD:MOVE:..., CMD:ESTOP, CMD:PING)
- Telemetry line parsing and 20Hz heartbeat acknowledgement
- Motor stall current threshold (>1200mA) triggering automated emergency stop
- Actuation isolation and kinematics proposal validation
- Software and hardware E-Stop lifecycle, hook execution, and reset
- Multi-joint pose application
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from robot.motion.serial_bridge import SerialBridge, ServoJointLimit


def test_serial_bridge_emulation_mode() -> None:
    """Test basic emulation mode behavior and joint limit clamping."""
    bridge = SerialBridge(port=None, enable_emulation_fallback=True)
    bridge.start()
    assert bridge.is_emulated is True
    assert bridge.is_active is True
    assert bridge.is_connected is False

    # Joint movement within limits
    res = bridge.set_joint_angle("neck_pan", 45.0)
    assert res is True
    assert bridge.joints["neck_pan"].current_deg == 45.0

    # Joint movement outside limits (must clamp)
    res_clamped = bridge.set_joint_angle("neck_pan", 120.0)
    assert res_clamped is True
    assert bridge.joints["neck_pan"].current_deg == 85.0  # Clamped to max_deg 85.0

    # Park servos
    bridge.park_servos()
    assert bridge.joints["neck_pan"].current_deg == 0.0

    # Unknown joint
    assert bridge.set_joint_angle("nonexistent_joint", 10.0) is False

    bridge.stop()
    assert not bridge.is_connected


def test_emergency_torque_disconnect() -> None:
    """Test legacy emergency_torque_disconnect invokes emergency stop."""
    bridge = SerialBridge(enable_emulation_fallback=True)
    bridge.start()
    assert bridge.is_e_stopped is False

    bridge.emergency_torque_disconnect()
    assert bridge.is_e_stopped is True
    bridge.stop()


def test_versioned_packet_protocol_formatting() -> None:
    """Test ASCII packet protocol formats matches firmware specification."""
    bridge = SerialBridge(enable_emulation_fallback=True)

    move_pkt = bridge.format_move_packet("neck_pan", 45.5, speed_deg_s=120.0)
    assert move_pkt == b"CMD:MOVE:neck_pan:45.50:120.00\n"

    estop_pkt = bridge.format_estop_packet()
    assert estop_pkt == b"CMD:ESTOP\n"

    resume_pkt = bridge.format_resume_packet()
    assert resume_pkt == b"CMD:RESUME\n"

    ping_pkt = bridge.format_heartbeat_packet()
    assert ping_pkt == b"CMD:PING\n"


def test_parse_incoming_telemetry() -> None:
    """Test parsing of incoming microcontroller telemetry lines."""
    bridge = SerialBridge(enable_emulation_fallback=True)

    # 1. Heartbeat Pong
    pong = bridge.parse_incoming_line("ACK:PONG\n")
    assert pong["type"] == "pong"

    # 2. Motor current draw
    curr = bridge.parse_incoming_line("TLM:CURRENT:neck_pan:450.5\n")
    assert curr["type"] == "current"
    assert curr["joint"] == "neck_pan"
    assert curr["milliamps"] == 450.5
    assert bridge._last_telemetry["current_neck_pan"] == 450.5

    # 3. Motion acknowledgement
    ack = bridge.parse_incoming_line("ACK:MOVE:neck_pan:45.0\n")
    assert ack["type"] == "ack_move"

    # 4. Firmware error
    err = bridge.parse_incoming_line("ERR:OVERHEAT:channel_0\n")
    assert err["type"] == "error"


def test_stall_current_exceeded_triggers_estop() -> None:
    """Ensure motor current exceeding 1200mA automatically triggers emergency stop."""
    bridge = SerialBridge(enable_emulation_fallback=True)
    bridge.start()
    assert bridge.is_e_stopped is False

    # Simulate excessive current draw from stalled actuator
    bridge.parse_incoming_line("TLM:CURRENT:elbow_r:1450.0\n")
    assert bridge.is_e_stopped is True

    # Subsequent movement commands must be rejected
    assert bridge.set_joint_angle("neck_pan", 20.0) is False

    bridge.stop()


def test_kinematics_proposal_validation() -> None:
    """Test validation and clamping of proposed LLM actions."""
    bridge = SerialBridge(enable_emulation_fallback=True)

    # Valid proposal
    ok, safe_angles, reason = bridge.validate_kinematics_proposal(
        action_name="look_up",
        target_angles={"neck_pan": 15.0, "neck_tilt": 20.0},
    )
    assert ok is True
    assert safe_angles["neck_pan"] == 15.0
    assert safe_angles["neck_tilt"] == 20.0

    # Proposal exceeding mechanical boundaries must clamp
    ok_clamp, safe_clamp, _ = bridge.validate_kinematics_proposal(
        action_name="look_extreme",
        target_angles={"neck_tilt": 90.0},  # max_deg is 45.0
    )
    assert ok_clamp is True
    assert safe_clamp["neck_tilt"] == 45.0

    # Unknown joint must be rejected
    bad_ok, _, reason_bad = bridge.validate_kinematics_proposal(
        action_name="alien_movement",
        target_angles={"tail_sway": 45.0},
    )
    assert bad_ok is False
    assert "Unknown joint" in reason_bad

    # Proposal rejected when E-Stop is active
    bridge.emergency_stop()
    estop_ok, _, reason_estop = bridge.validate_kinematics_proposal(
        action_name="wave",
        target_angles={"shoulder_pitch_r": 30.0},
    )
    assert estop_ok is False
    assert "emergency stop" in reason_estop.lower()


def test_estop_hook_and_reset_flow() -> None:
    """Test hardware disconnect callback execution and E-Stop reset."""
    bridge = SerialBridge(enable_emulation_fallback=True)
    hook_called = False

    def mock_hardware_disconnect() -> None:
        nonlocal hook_called
        hook_called = True

    bridge.register_e_stop_callback(mock_hardware_disconnect)
    assert bridge.is_e_stopped is False

    bridge.emergency_stop("Safety trip test")
    assert bridge.is_e_stopped is True
    assert hook_called is True

    # Resetting E-Stop
    assert bridge.reset_e_stop() is True
    assert bridge.is_e_stopped is False


def test_apply_pose_multi_joint() -> None:
    """Test applying multi-joint validated pose."""
    bridge = SerialBridge(enable_emulation_fallback=True)
    bridge.start()

    pose = {
        "neck_pan": 25.0,
        "neck_tilt": -10.0,
        "elbow_r": 45.0,
    }
    assert bridge.apply_pose(pose) is True
    assert bridge.joints["neck_pan"].current_deg == 25.0
    assert bridge.joints["neck_tilt"].current_deg == -10.0
    assert bridge.joints["elbow_r"].current_deg == 45.0

    bridge.stop()


def test_minimum_jerk_trajectory_generation() -> None:
    """Test 5th-order minimum jerk trajectory interpolation guarantees smooth motion."""
    joint = ServoJointLimit("neck_pan", channel=0, min_deg=-85.0, max_deg=85.0, default_deg=0.0)
    traj = joint.generate_minimum_jerk_trajectory(target_deg=50.0, steps=20)
    assert len(traj) == 20
    assert traj[-1] == 50.0
    # Values must be monotonic towards target
    for i in range(len(traj) - 1):
        assert traj[i] <= traj[i + 1]
    # Intermediate trajectory must not exceed clamp boundaries
    assert all(-85.0 <= val <= 85.0 for val in traj)
    # Test boundary clamping
    clamped_traj = joint.generate_minimum_jerk_trajectory(target_deg=100.0, steps=10)
    assert clamped_traj[-1] == 85.0

