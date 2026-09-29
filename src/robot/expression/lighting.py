"""ROG Strix Aura RGB Lighting Controller via asusd / asusctl.

Controls keyboard zones and front lightbar with reactive animations for:
- IDLE: Gentle low-speed cyan/deep-blue breathing
- LISTENING: Bright turquoise / emerald responsive to microphone energy
- THINKING: Amber / gold pulsing animation
- SPEAKING: Dynamic audio-reactive lighting modulating brightness and color with TTS voice RMS
- ERROR: Crimson warning pulse
- SLEEPING: Dim dark indigo or LEDs off
"""

from __future__ import annotations

import asyncio
import colorsys
import math
import subprocess
import time
from typing import TYPE_CHECKING

from robot.core.state import RobotMode
from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.audio.playback import AudioPlayer
    from robot.core.state import StateManager

logger = get_logger("expression.lighting")


class AsusAuraLighting:
    """Lighting controller for ASUS ROG Strix laptops (G513RC and similar).

    Uses direct DBus (or asusctl CLI fallback) for sub-10ms lighting modulation.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._dbus_props = None
        self._current_mode: RobotMode = RobotMode.IDLE
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._voice_rms: float = 0.0
        self._smooth_energy: float = 0.0
        self._current_rgb: tuple[float, float, float] = (0.0, 80.0, 160.0)
        self._last_voice_update: float = 0.0
        self._last_set_time: float = 0.0

        if not self.enabled:
            return

        # Attempt to connect to asusd DBus directly for minimal latency
        try:
            import dbus

            bus = dbus.SystemBus()
            aura_obj = bus.get_object("xyz.ljones.Asusd", "/xyz/ljones/aura/19b6_2_3")
            self._dbus_props = dbus.Interface(aura_obj, "org.freedesktop.DBus.Properties")
            logger.info("asus_aura_dbus_connected")
        except Exception as e:
            logger.warning("asus_aura_dbus_fallback_to_cli", error=str(e))
            self._dbus_props = None

    def set_color(self, r: int, g: int, b: int, zone: int = 0) -> None:
        """Set RGB color for a zone (zone 0 = all zones, 1..4 = individual zones)."""
        if not self.enabled:
            return

        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        if self._dbus_props is not None:
            try:
                import dbus

                val = dbus.Struct(
                    (
                        dbus.UInt32(0),  # Static mode
                        dbus.UInt32(zone),
                        dbus.Struct((dbus.Byte(r), dbus.Byte(g), dbus.Byte(b))),
                        dbus.Struct((dbus.Byte(0), dbus.Byte(0), dbus.Byte(0))),
                        dbus.String("Med"),
                        dbus.String("Right"),
                    ),
                    signature="uu(yyy)(yyy)ss",
                )
                self._dbus_props.Set("xyz.ljones.Aura", "LedModeData", val)
                return
            except Exception as e:
                logger.debug("dbus_set_color_failed", error=str(e))

        # CLI fallback
        hex_color = f"{r:02x}{g:02x}{b:02x}"
        cmd = ["asusctl", "aura", "effect", "static", "-c", hex_color]
        if zone > 0:
            cmd.extend(["--zone", str(zone - 1)])
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def set_brightness(self, level: int) -> None:
        """Set keyboard & strip brightness (0=off, 1=low, 2=med, 3=high)."""
        if not self.enabled:
            return
        level = max(0, min(3, level))
        if self._dbus_props is not None:
            try:
                import dbus

                self._dbus_props.Set("xyz.ljones.Aura", "Brightness", dbus.UInt32(level))
                return
            except Exception:
                pass

        levels = ["off", "low", "med", "high"]
        try:
            subprocess.run(
                ["asusctl", "leds", "set", levels[level]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def on_voice_energy(self, rms: float) -> None:
        """Callback invoked by AudioPlayer on every TTS audio block."""
        self._voice_rms = rms
        self._last_voice_update = time.time()

    def attach(self, state_manager: StateManager, player: AudioPlayer) -> None:
        """Connect lighting controller to state changes and audio playback."""
        state_manager.subscribe(self._on_state_change)
        player.on_energy(self.on_voice_energy)

    async def _on_state_change(self, old_mode: RobotMode, new_mode: RobotMode) -> None:
        """React to RobotMode transitions."""
        self._current_mode = new_mode

    async def start(self) -> None:
        """Start animation loop."""
        if not self.enabled:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._animation_loop())
        logger.info("aura_lighting_started")

    async def stop(self) -> None:
        """Stop animation loop and return LEDs to idle."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.set_brightness(3)
        self.set_color(0, 160, 255)

    async def _animation_loop(self) -> None:
        """Vibrant, high-contrast, silky-smooth lighting animation loop."""
        t_start = time.time()
        # Always enforce 100% High brightness
        self.set_brightness(3)

        cur_r, cur_g, cur_b = self._current_rgb

        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - t_start
            mode = self._current_mode

            if mode == RobotMode.SPEAKING:
                # ── Voice Audio-Reactive Modulation (50% more amplified) ──
                # Ultra-sensitive energy tracking: speech volume mapped aggressively to [0.0, 1.0]
                raw_target = min(1.0, max(0.0, (self._voice_rms - 0.002) / 0.038))
                if raw_target > self._smooth_energy:
                    self._smooth_energy += (raw_target - self._smooth_energy) * 0.55
                else:
                    self._smooth_energy += (raw_target - self._smooth_energy) * 0.14

                # High-contrast hue sweep across the entire spectrum:
                # Base quiet tone = Electric Neon Cyan (hue 0.50 -> RGB 0, 255, 255)
                # Syllable accent = Electric Violet (hue 0.78 -> RGB 180, 0, 255)
                # Peak volume = Hot Electric Pink / Crimson (hue 0.95 -> RGB 255, 0, 90)
                hue = (0.50 + 0.45 * self._smooth_energy) % 1.0
                sat = 1.0
                val = 1.0

                target_r, target_g, target_b = [c * 255.0 for c in colorsys.hsv_to_rgb(hue, sat, val)]
                lerp_speed = 0.35

            elif mode == RobotMode.LISTENING:
                # ── Deep Emerald to Neon Lime Pulse (0.8 Hz) ──
                # Very distinct vivid electric green clearly different from all other modes
                pulse = 0.5 + 0.5 * math.sin(elapsed * 2.8)
                target_r = 10.0 + 80.0 * pulse   # Adds electric lime tint at peak
                target_g = 255.0                  # Pure max green
                target_b = 40.0 * (1.0 - pulse)  # Drops blue to zero at peak for max green contrast
                lerp_speed = 0.22

            elif mode == RobotMode.THINKING:
                # ── Burning Solar Flare (Orange -> Pure Golden Yellow) ──
                pulse = 0.5 + 0.5 * math.sin(elapsed * 4.0)
                target_r = 255.0
                target_g = 30.0 + 190.0 * pulse  # Swells between deep lava orange (255, 30, 0) and golden yellow (255, 220, 0)
                target_b = 0.0
                lerp_speed = 0.25

            elif mode == RobotMode.ERROR:
                # ── Pure Warning Crimson Flash ──
                flash = 1.0 if (int(elapsed * 6.0) % 2 == 0) else 0.15
                target_r = 255.0 * flash
                target_g = 0.0
                target_b = 0.0
                lerp_speed = 0.40

            elif mode == RobotMode.SLEEPING:
                # ── Deep Midnight Indigo ──
                pulse = 0.5 + 0.5 * math.sin(elapsed * 1.0)
                target_r = 40.0 * pulse
                target_g = 0.0
                target_b = 180.0 + 75.0 * pulse
                lerp_speed = 0.12

            else:
                # ── IDLE: Deep Royal Blue into Electric Cyan Swell ──
                pulse = 0.5 + 0.5 * math.sin(elapsed * 2.0)
                target_r = 0.0
                target_g = 20.0 + 200.0 * pulse   # Sweeps from pure Royal Blue (0, 20, 255) to Electric Cyan (0, 220, 255)
                target_b = 255.0
                lerp_speed = 0.18

            # Smooth exponential interpolation (lerp) towards target color
            cur_r += (target_r - cur_r) * lerp_speed
            cur_g += (target_g - cur_g) * lerp_speed
            cur_b += (target_b - cur_b) * lerp_speed
            self._current_rgb = (cur_r, cur_g, cur_b)

            self.set_color(int(cur_r), int(cur_g), int(cur_b))
            await asyncio.sleep(0.035)  # Responsive ~28 FPS update rate
