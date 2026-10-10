"""Tests for physical servo serial bridge and software emulation."""

import pytest
from robot.motion.serial_bridge import SerialBridge, ServoJointLimit


def test_serial_bridge_emulation_mode():
    bridge = SerialBridge(port=None, enable_emulation_fallback=True)
    bridge.start()
    assert bridge.is_emulated is True
    assert bridge.is_active is True
    assert bridge.is_connected is False  # Real hardware is not present

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


def test_emergency_torque_disconnect():
    bridge = SerialBridge(enable_emulation_fallback=True)
    bridge.start()
    bridge.emergency_torque_disconnect()
    bridge.stop()
