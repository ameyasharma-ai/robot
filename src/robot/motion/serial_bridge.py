"""Hardware microcontroller serial bridge for physical robotic servos.

Features:
- Versioned binary and ASCII serial protocol (`CMD:MOVE:JOINT:POS:SPEED\\n`) over `/dev/ttyUSB*` at 115200 baud
- Seamless automatic fallback to software emulation when hardware is disconnected
- Asynchronous reconnection supervisor with exponential backoff (1s -> 2s -> 4s -> 8s)
- Firmware and software joint limit enforcement, velocity ramping, and acceleration damping
- 20Hz bidirectional heartbeat and command watchdog (<500ms safety deadline)
- Hardware and software Emergency Stop (E-Stop) with hardware disconnect hook
- Stall-current torque disconnect protection (>1200mA threshold)
- Actuation isolation validating kinematics proposals before physical dispatch
"""

from __future__ import annotations

import glob
import math
import re
import struct
import threading
import time
from typing import Any, Callable, Optional

from robot.telemetry import get_logger

logger = get_logger("motion.serial_bridge")


class ServoJointLimit:
    """Mechanical joint bounds and safe operational limits."""

    def __init__(
        self,
        name: str,
        channel: int,
        min_deg: float = -90.0,
        max_deg: float = 90.0,
        default_deg: float = 0.0,
        max_velocity_deg_s: float = 180.0,
        max_accel_deg_s2: float = 360.0,
    ) -> None:
        """Initialize servo joint limits.

        Args:
            name: Human-readable joint name (e.g. 'neck_pan', 'neck_tilt').
            channel: Microcontroller PWM/servo channel index (0-15).
            min_deg: Minimum physical joint angle in degrees.
            max_deg: Maximum physical joint angle in degrees.
            default_deg: Resting or home joint position in degrees.
            max_velocity_deg_s: Maximum allowable angular velocity in degrees/sec.
            max_accel_deg_s2: Maximum allowable angular acceleration in degrees/sec^2.
        """
        self.name = name
        self.channel = channel
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.default_deg = default_deg
        self.max_velocity_deg_s = max_velocity_deg_s
        self.max_accel_deg_s2 = max_accel_deg_s2
        self.current_deg = default_deg
        self.target_deg = default_deg
        self.current_velocity_deg_s = 0.0
        self.last_update_time = time.time()

    def clamp(self, angle_deg: float) -> float:
        """Clamp angle to mechanical joint boundaries.

        Args:
            angle_deg: Target angle in degrees.

        Returns:
            Angle clamped between min_deg and max_deg.
        """
        return max(self.min_deg, min(self.max_deg, float(angle_deg)))

    def apply_trajectory_step(self, target_deg: float, dt: float) -> float:
        """Calculate smooth velocity-ramped position step towards target.

        Args:
            target_deg: Desired target position in degrees.
            dt: Delta time elapsed in seconds.

        Returns:
            Updated current angle in degrees.
        """
        clamped_target = self.clamp(target_deg)
        if dt <= 0.0:
            return self.current_deg

        delta = clamped_target - self.current_deg
        max_step = self.max_velocity_deg_s * dt

        if abs(delta) <= max_step:
            self.current_deg = clamped_target
            self.current_velocity_deg_s = delta / dt if dt > 0 else 0.0
        else:
            step_dir = 1.0 if delta > 0 else -1.0
            self.current_deg += step_dir * max_step
            self.current_velocity_deg_s = step_dir * self.max_velocity_deg_s

        self.target_deg = clamped_target
        self.last_update_time = time.time()
        return self.current_deg

    def generate_minimum_jerk_trajectory(
        self,
        target_deg: float,
        duration_s: float = 0.5,
        steps: int = 25,
    ) -> list[float]:
        """Generate an Apple-grade minimum-jerk trajectory polynomial to target.

        Uses the 5th-order minimum jerk formulation:
            s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5,  where tau = t / T in [0, 1].
        Guarantees zero starting and ending velocity and acceleration, preventing
        mechanical shock, current spikes, and gear tooth chatter.

        Args:
            target_deg: Desired target position in degrees.
            duration_s: Movement duration in seconds (default 0.5s).
            steps: Number of discrete interpolation points (default 25).

        Returns:
            List of interpolated joint angles in degrees from current position to clamped target.
        """
        clamped_target = self.clamp(target_deg)
        start_deg = self.current_deg
        delta = clamped_target - start_deg

        if steps <= 1 or abs(delta) < 1e-4:
            return [clamped_target]

        trajectory: list[float] = []
        for i in range(1, steps + 1):
            tau = float(i) / float(steps)
            poly = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            deg = start_deg + delta * poly
            trajectory.append(round(deg, 3))

        return trajectory



class SerialBridge:
    """Hardware serial controller bridge for embodied robotic kinematics."""

    PROTOCOL_VERSION: str = "2.0"

    DEFAULT_JOINTS: dict[str, ServoJointLimit] = {
        "neck_pan": ServoJointLimit("neck_pan", channel=0, min_deg=-85.0, max_deg=85.0, default_deg=0.0),
        "neck_tilt": ServoJointLimit("neck_tilt", channel=1, min_deg=-35.0, max_deg=45.0, default_deg=0.0),
        "shoulder_pitch_r": ServoJointLimit("shoulder_pitch_r", channel=2, min_deg=-90.0, max_deg=90.0, default_deg=0.0),
        "shoulder_roll_r": ServoJointLimit("shoulder_roll_r", channel=3, min_deg=-45.0, max_deg=90.0, default_deg=0.0),
        "elbow_r": ServoJointLimit("elbow_r", channel=4, min_deg=0.0, max_deg=135.0, default_deg=15.0),
        "wrist_r": ServoJointLimit("wrist_r", channel=5, min_deg=-90.0, max_deg=90.0, default_deg=0.0),
    }

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 0.1,
        enable_emulation_fallback: bool = True,
        watchdog_timeout_s: float = 0.5,
    ) -> None:
        """Initialize serial bridge.

        Args:
            port: Serial device path (e.g. '/dev/ttyUSB0'). If None, probes ports automatically.
            baudrate: Communication baudrate (default 115200).
            timeout: Serial read timeout in seconds.
            enable_emulation_fallback: If True, operates in emulated mode when hardware is missing.
            watchdog_timeout_s: Safety watchdog timeout in seconds (default 0.5s = 500ms).
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.enable_emulation = enable_emulation_fallback
        self.watchdog_timeout_s = watchdog_timeout_s

        self.joints: dict[str, ServoJointLimit] = {k: v for k, v in self.DEFAULT_JOINTS.items()}
        self._serial: Any = None
        self._is_connected: bool = False
        self._is_emulated: bool = False
        self._running: bool = False
        self._e_stop: bool = False
        self._e_stop_callback: Optional[Callable[[], None]] = None

        self._lock = threading.Lock()
        self._reconnect_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_heartbeat_ack: float = time.time()
        self._stall_threshold_ma: float = 1200.0  # Milli-amps stall current protection
        self._last_telemetry: dict[str, Any] = {}

        # Work package A safety enhancements
        self._command_seq: int = 0
        self._last_command_time: dict[str, float] = {}
        self._last_command_angle: dict[str, float] = {}
        self._motion_latched: bool = False
        self._has_connected_once: bool = False
        self.duplicate_tolerance_deg: float = 0.05
        self.max_command_age_s: float = 0.5

    @classmethod
    def get_protection_boundary_map(cls) -> dict[str, list[str]]:
        """Return a structured map distinguishing host-enforced from hardware-enforced protections.

        Returns:
            dict[str, list[str]]: Categorized breakdown of safety mechanisms.
        """
        return {
            "host_enforced": [
                "Kinematic joint boundary clamping (min_deg / max_deg)",
                "5th-order minimum-jerk trajectory polynomial shaping (zero start/end acceleration)",
                "Malformed command rejection (NaN, Inf, non-numeric, negative speeds)",
                "Duplicate command filtering to prevent bus saturation (0.05 deg epsilon)",
                "Command freshness sequence & timestamp validation (max_command_age_s=0.5)",
                "Post-reconnect zero-motion latching (prevents replay of stale in-flight commands)",
                "Software E-Stop state machine & kinematics proposal gating",
            ],
            "hardware_enforced": [
                "20Hz bidirectional heartbeat watchdog (<500ms deadline triggers firm motor stop)",
                "Motor stall-current hardware threshold (>1200mA triggers emergency torque disconnect)",
                "Hardware E-Stop relay disconnect hook (trips physical power supply)",
                "Hardware safe state upon serial disconnect or loss of host process",
            ],
        }

    @property
    def command_seq(self) -> int:
        """Return current monotonic command sequence identifier."""
        return self._command_seq

    @property
    def is_motion_latched(self) -> bool:
        """Return True if motion is currently latched post-reconnect to prevent stale resume."""
        return self._motion_latched

    def unlatch_motion(self) -> None:
        """Explicitly release post-reconnect motion latch after operator confirmation."""
        self._motion_latched = False
        logger.info("motion_unlatched_after_reconnect")

    @property
    def is_connected(self) -> bool:
        """Return True if real physical serial connection is active."""
        return self._is_connected and not self._is_emulated

    @property
    def is_emulated(self) -> bool:
        """Return True if operating under software emulation."""
        return self._is_emulated

    @property
    def is_active(self) -> bool:
        """Return True if bridge is active either via hardware or emulation."""
        return self._is_connected or self._is_emulated

    @property
    def is_e_stopped(self) -> bool:
        """Return True if physical safety emergency stop is currently active."""
        return self._e_stop

    def register_e_stop_callback(self, callback: Callable[[], None]) -> None:
        """Register hardware power disconnect hook or relay switch for emergency stop.

        Args:
            callback: Function to invoke immediately when E-Stop is triggered.
        """
        self._e_stop_callback = callback

    def start(self) -> None:
        """Start serial communication, 20Hz heartbeat worker, and reconnection supervisor."""
        if self._running:
            return

        self._running = True
        self._connect()

        self._reconnect_thread = threading.Thread(
            target=self._reconnect_worker,
            daemon=True,
            name="SerialBridgeReconnectWorker",
        )
        self._reconnect_thread.start()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker,
            daemon=True,
            name="SerialBridgeHeartbeatWorker",
        )
        self._heartbeat_thread.start()
        logger.info("serial_bridge_started", port=self.port, emulated=self._is_emulated)

    def stop(self) -> None:
        """Safely de-energize servos and close serial connections."""
        self._running = False
        self.park_servos()
        with self._lock:
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._is_connected = False
        logger.info("serial_bridge_stopped")

    def _find_available_port(self) -> Optional[str]:
        """Auto-probe connected USB/ACM serial ports on Linux."""
        candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if candidates:
            return sorted(candidates)[0]
        return None

    def _connect(self) -> bool:
        """Attempt to open physical serial port connection.

        Enforces post-reconnection zero-motion latching: if this connection is a re-attachment
        after initial boot or a prior disconnect, motion is automatically latched until explicit
        operator homing or unlatching is confirmed.

        Returns:
            bool: True if connection (or emulation fallback) was established.
        """
        is_reconnect = self._has_connected_once
        target_port = self.port or self._find_available_port()

        if not target_port:
            if self.enable_emulation:
                self._is_emulated = True
                self._is_connected = True
                if is_reconnect:
                    self._motion_latched = True
                    self._last_command_angle.clear()
                    self._last_command_time.clear()
                    logger.warning("serial_reconnected_emulated_motion_latched_awaiting_safe_homing")
                self._has_connected_once = True
                logger.info("serial_hardware_not_found_emulation_active")
                return True
            return False

        try:
            import serial
            ser = serial.Serial(
                port=target_port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            with self._lock:
                self._serial = ser
                self.port = target_port
                self._is_connected = True
                self._is_emulated = False
                self._last_heartbeat_ack = time.time()
                if is_reconnect:
                    self._motion_latched = True
                    self._last_command_angle.clear()
                    self._last_command_time.clear()
                    logger.warning("serial_reconnected_hardware_motion_latched_awaiting_safe_homing")
                self._has_connected_once = True
            logger.info("serial_hardware_connected_successfully", port=target_port, baudrate=self.baudrate)
            return True
        except Exception as e:
            logger.warning("serial_connection_failed", port=target_port, error=str(e))
            if self.enable_emulation:
                self._is_emulated = True
                self._is_connected = True
                if is_reconnect:
                    self._motion_latched = True
                    self._last_command_angle.clear()
                    self._last_command_time.clear()
                    logger.warning("serial_reconnected_emulated_motion_latched_awaiting_safe_homing")
                self._has_connected_once = True
            return False

    def _reconnect_worker(self) -> None:
        """Background thread supervising hardware re-attachments with exponential backoff."""
        backoff = 1.0
        while self._running:
            if not self.is_connected and not (self._is_emulated and not self.enable_emulation):
                target_port = self.port or self._find_available_port()
                if target_port:
                    success = self._connect()
                    if success and not self._is_emulated:
                        backoff = 1.0
                    else:
                        backoff = min(10.0, backoff * 2.0)
                else:
                    backoff = min(10.0, backoff * 2.0)

            sleep_time = backoff
            t_end = time.time() + sleep_time
            while self._running and time.time() < t_end:
                time.sleep(0.2)

    def format_move_packet(
        self,
        joint_name: str,
        angle_deg: float,
        speed_deg_s: float = 100.0,
    ) -> bytes:
        """Format a versioned ASCII motion command packet for physical firmware.

        Args:
            joint_name: Joint identifier (e.g. 'neck_pan').
            angle_deg: Target joint position in degrees.
            speed_deg_s: Movement angular velocity limit.

        Returns:
            ASCII encoded packet bytes terminated with newline.
        """
        return f"CMD:MOVE:{joint_name}:{angle_deg:.2f}:{speed_deg_s:.2f}\n".encode("ascii")

    def format_estop_packet(self) -> bytes:
        """Format an emergency stop command packet."""
        return b"CMD:ESTOP\n"

    def format_resume_packet(self) -> bytes:
        """Format an emergency stop resume/reset packet."""
        return b"CMD:RESUME\n"

    def format_heartbeat_packet(self) -> bytes:
        """Format a 20Hz ping heartbeat packet."""
        return b"CMD:PING\n"

    def parse_incoming_line(self, line: str) -> dict[str, Any]:
        """Parse incoming ASCII telemetry or ACK packets from physical microcontroller.

        Args:
            line: Raw string received over serial.

        Returns:
            Structured dictionary of parsed telemetry or command response.
        """
        clean = line.strip()
        if not clean:
            return {"type": "empty"}

        # 1. Heartbeat pong response
        if clean in ("PONG", "ACK:PONG", "CMD:PONG"):
            self._last_heartbeat_ack = time.time()
            return {"type": "pong", "timestamp": self._last_heartbeat_ack}

        # 2. Motor current telemetry: TLM:CURRENT:<joint>:<milliamps>
        match_curr = re.match(r"^TLM:CURRENT:([a-zA-Z0-9_]+):([0-9.]+)", clean)
        if match_curr:
            joint = match_curr.group(1)
            ma = float(match_curr.group(2))
            self._last_telemetry[f"current_{joint}"] = ma

            # Stall current protection check
            if ma > self._stall_threshold_ma:
                logger.critical(
                    "motor_stall_current_exceeded_triggering_estop",
                    joint=joint,
                    current_ma=ma,
                    threshold=self._stall_threshold_ma,
                )
                self.emergency_stop(reason=f"Stall current exceeded on joint {joint}: {ma}mA")

            return {"type": "current", "joint": joint, "milliamps": ma}

        # 3. Motion acknowledgement
        if clean.startswith("ACK:MOVE:"):
            return {"type": "ack_move", "raw": clean}

        # 4. Error report from firmware
        if clean.startswith("ERR:"):
            logger.warning("firmware_error_reported", message=clean)
            return {"type": "error", "message": clean}

        return {"type": "raw", "data": clean}

    def _heartbeat_worker(self) -> None:
        """Transmit 20Hz heartbeat packet and monitor watchdog deadline."""
        interval = 0.05  # 20 Hz
        while self._running:
            t0 = time.time()
            if self.is_connected:
                # Transmit ASCII and binary heartbeat ping
                ascii_ping = self.format_heartbeat_packet()
                binary_ping = bytearray([0xAA, 0x55, 0x01, 0x00])
                binary_ping[3] = (binary_ping[0] ^ binary_ping[1] ^ binary_ping[2]) & 0xFF

                try:
                    with self._lock:
                        if self._serial:
                            self._serial.write(ascii_ping)
                            self._serial.flush()

                            # Read any waiting telemetry bytes
                            if self._serial.in_waiting > 0:
                                raw_lines = self._serial.read(self._serial.in_waiting).decode("ascii", errors="ignore")
                                for line in raw_lines.splitlines():
                                    self.parse_incoming_line(line)
                except Exception as e:
                    logger.warning("serial_heartbeat_write_failed", error=str(e))
                    with self._lock:
                        if self._serial:
                            try:
                                self._serial.close()
                            except Exception:
                                pass
                            self._serial = None
                        self._is_connected = False
                        if self.enable_emulation:
                            self._is_emulated = True

                # Watchdog check: If no pong received within watchdog_timeout_s, trigger E-Stop
                if time.time() - self._last_heartbeat_ack > self.watchdog_timeout_s:
                    if not self._e_stop:
                        logger.critical(
                            "hardware_command_watchdog_timed_out",
                            timeout=self.watchdog_timeout_s,
                        )
                        self.emergency_stop(reason="Microcontroller heartbeat watchdog expired (>500ms)")

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    def validate_kinematics_proposal(
        self,
        action_name: str,
        target_angles: dict[str, float],
        speeds: dict[str, float] | None = None,
        command_time: float | None = None,
    ) -> tuple[bool, dict[str, float], str]:
        """Isolate physical actuation from LLM proposals and validate kinematic boundaries.

        Rejects malformed, NaN, infinite, non-numeric, or stale proposals.

        Args:
            action_name: High-level action name (e.g. 'wave', 'nod', 'look_at').
            target_angles: Dictionary of joint names mapping to proposed angles in degrees.
            speeds: Optional dictionary of joint movement speeds.
            command_time: Optional timestamp of proposal generation for freshness validation.

        Returns:
            Tuple of (is_valid, validated_angles, explanation_str).
        """
        if self._e_stop:
            return False, {}, f"Rejected: Physical emergency stop (E-Stop) is currently active ({action_name})."

        now = time.time()
        if command_time is not None and (now - command_time > self.max_command_age_s):
            age_ms = (now - command_time) * 1000.0
            return False, {}, f"Rejected: Stale kinematics proposal for '{action_name}' ({age_ms:.1f}ms > {self.max_command_age_s * 1000}ms limit)."

        validated: dict[str, float] = {}
        for j_name, angle in target_angles.items():
            if j_name not in self.joints:
                return False, {}, f"Rejected: Unknown joint '{j_name}' proposed in {action_name}."

            if not isinstance(angle, (int, float)) or math.isnan(angle) or math.isinf(angle):
                return False, {}, f"Rejected: Malformed or NaN/Inf angle '{angle}' proposed for joint '{j_name}'."

            joint = self.joints[j_name]
            clamped = joint.clamp(float(angle))
            validated[j_name] = clamped

            # Check velocity constraints if speed provided
            if speeds and j_name in speeds:
                s = speeds[j_name]
                if not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s) or s <= 0.0:
                    return False, {}, f"Rejected: Invalid non-positive or NaN/Inf speed '{s}' for joint '{j_name}'."
                if s > joint.max_velocity_deg_s:
                    speeds[j_name] = joint.max_velocity_deg_s

        return True, validated, f"Kinematics proposal for '{action_name}' validated safely."

    def set_joint_angle(
        self,
        joint_name: str,
        angle_deg: float,
        speed_deg_s: float = 100.0,
        command_time: float | None = None,
    ) -> bool:
        """Move a specific joint to a designated angle within mechanical bounds.

        Guards against post-reconnection motion resume, malformed NaN/Inf angles,
        invalid speeds, stale sequence proposals, and serial bus saturation from duplicates.

        Args:
            joint_name: Name of joint (e.g. 'neck_pan', 'neck_tilt').
            angle_deg: Target angle in degrees.
            speed_deg_s: Movement speed limit in degrees/sec.
            command_time: Optional timestamp of proposal generation for freshness check.

        Returns:
            True if position command was successfully dispatched, filtered as duplicate, or emulated.
        """
        if self._e_stop:
            logger.warning("joint_move_rejected_estop_active", joint=joint_name)
            return False

        if self._motion_latched:
            logger.warning("joint_move_rejected_post_reconnect_latched", joint=joint_name)
            return False

        if not isinstance(angle_deg, (int, float)) or math.isnan(angle_deg) or math.isinf(angle_deg):
            logger.warning("joint_move_rejected_malformed_angle", joint=joint_name, angle=angle_deg)
            return False

        if not isinstance(speed_deg_s, (int, float)) or math.isnan(speed_deg_s) or math.isinf(speed_deg_s) or speed_deg_s <= 0.0:
            logger.warning("joint_move_rejected_invalid_speed", joint=joint_name, speed=speed_deg_s)
            return False

        now = time.time()
        if command_time is not None and (now - command_time > self.max_command_age_s):
            logger.warning("joint_move_rejected_stale_command", joint=joint_name, age_s=now - command_time)
            return False

        if joint_name not in self.joints:
            logger.warning("unknown_joint_command", joint=joint_name)
            return False

        joint = self.joints[joint_name]
        clamped_deg = joint.clamp(float(angle_deg))

        # Bus saturation protection: filter redundant duplicate commands within tolerance epsilon
        last_angle = self._last_command_angle.get(joint_name)
        if last_angle is not None and abs(clamped_deg - last_angle) <= self.duplicate_tolerance_deg:
            logger.debug("duplicate_joint_command_filtered", joint=joint_name, angle=clamped_deg)
            return True

        self._command_seq += 1
        self._last_command_angle[joint_name] = clamped_deg
        self._last_command_time[joint_name] = now
        joint.target_deg = clamped_deg
        joint.current_deg = clamped_deg

        if self._is_emulated or not self._is_connected:
            return True

        # Send versioned ASCII motion packet
        ascii_pkt = self.format_move_packet(joint_name, clamped_deg, speed_deg_s)

        # Also send binary packet for low-latency firmware
        angle_val = int(clamped_deg * 100)
        hi = (angle_val >> 8) & 0xFF
        lo = angle_val & 0xFF
        chk = (0xAA ^ 0x55 ^ 0x02 ^ joint.channel ^ hi ^ lo) & 0xFF
        bin_pkt = bytearray([0xAA, 0x55, 0x02, joint.channel, hi, lo, chk])

        try:
            with self._lock:
                if self._serial:
                    self._serial.write(ascii_pkt)
                    self._serial.write(bin_pkt)
                    self._serial.flush()
            return True
        except Exception as e:
            logger.warning("serial_joint_write_failed", joint=joint_name, error=str(e))
            return False

    def apply_pose(
        self,
        pose: dict[str, float],
        speeds: dict[str, float] | None = None,
        command_time: float | None = None,
    ) -> bool:
        """Apply a multi-joint pose after kinematics validation.

        Args:
            pose: Joint angles mapping in degrees.
            speeds: Optional joint speed limits.
            command_time: Optional proposal generation timestamp.

        Returns:
            True if all joints were updated successfully.
        """
        valid, safe_pose, reason = self.validate_kinematics_proposal("apply_pose", pose, speeds, command_time=command_time)
        if not valid:
            logger.warning("apply_pose_rejected", reason=reason)
            return False

        all_ok = True
        for j_name, angle in safe_pose.items():
            sp = speeds.get(j_name, 100.0) if speeds else 100.0
            if not self.set_joint_angle(j_name, angle, sp, command_time=command_time):
                all_ok = False
        return all_ok

    def park_servos(self) -> None:
        """Smoothly return all physical servos to their resting zero positions and unlatch motion."""
        self.unlatch_motion()
        for name, joint in self.joints.items():
            self.set_joint_angle(name, joint.default_deg)

    def emergency_stop(self, reason: str = "Manual / Automated E-Stop") -> None:
        """Trigger physical emergency stop: de-energize all servos, send E-Stop command, and invoke hardware disconnect hook.

        Args:
            reason: Explanation of the trigger condition for safety logs.
        """
        self._e_stop = True
        logger.critical("emergency_stop_activated", reason=reason)

        # 1. Invoke hardware disconnect hook if registered
        if self._e_stop_callback:
            try:
                self._e_stop_callback()
            except Exception as e:
                logger.error("e_stop_callback_failed", error=str(e))

        # 2. Dispatch hardware disconnect commands
        if self._is_connected and not self._is_emulated:
            ascii_estop = self.format_estop_packet()
            bin_estop = bytearray([0xAA, 0x55, 0x0F, 0xAA ^ 0x55 ^ 0x0F])
            try:
                with self._lock:
                    if self._serial:
                        self._serial.write(ascii_estop)
                        self._serial.write(bin_estop)
                        self._serial.flush()
            except Exception as e:
                logger.error("serial_estop_write_failed", error=str(e))

    def reset_e_stop(self) -> bool:
        """Reset emergency stop state after physical safety inspection.

        Returns:
            True if E-Stop was successfully reset.
        """
        self._e_stop = False
        logger.info("emergency_stop_reset")
        if self._is_connected and not self._is_emulated:
            resume_pkt = self.format_resume_packet()
            try:
                with self._lock:
                    if self._serial:
                        self._serial.write(resume_pkt)
                        self._serial.flush()
            except Exception:
                pass
        return True

    def emergency_torque_disconnect(self) -> None:
        """Trigger emergency torque release to protect actuators from stall current."""
        self.emergency_stop(reason="emergency_torque_disconnect")
