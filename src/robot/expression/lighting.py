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
from concurrent.futures import ThreadPoolExecutor
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
        self._last_zone_rgb: dict[int, tuple[int, int, int]] = {}
        self._player: AudioPlayer | None = None

        # asusctl CLI-fallback rate limiting: the animation loop runs at ~26 Hz
        # across 4 zones, which previously spawned ~100 subprocesses/sec. Now a
        # minimum inter-dispatch interval coalesces bursts down to <=10 Hz of
        # latest-wins colors, executed off the event loop via a single-worker
        # executor so subprocess.run never blocks asyncio.
        self._cli_min_interval_s: float = 0.1
        self._color_change_threshold: int = 2  # max per-channel RGB delta treated as "unchanged"
        self._last_cli_dispatch: float = 0.0
        self._pending_zone_rgb: dict[int, tuple[int, int, int]] = {}
        self._cli_executor: ThreadPoolExecutor | None = None

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
        """Set RGB color for a zone (zone 0 = all zones, 1=WASD/Left, 2=Center, 3=Right, 4=Front Strip)."""
        if not self.enabled:
            return

        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        # Cache check to avoid redundant DBus / IPC messages (skip when the
        # computed color is unchanged beyond a small threshold from last send)
        last = self._last_zone_rgb.get(zone)
        if last and abs(r - last[0]) < self._color_change_threshold and abs(g - last[1]) < self._color_change_threshold and abs(b - last[2]) < self._color_change_threshold:
            return
        self._last_zone_rgb[zone] = (r, g, b)

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

        # CLI fallback: enforce a minimum inter-dispatch interval, coalescing
        # pending zone colors (latest wins), then dispatch via a single-worker
        # executor so subprocess.run never blocks the asyncio event loop.
        now = time.monotonic()
        self._pending_zone_rgb[zone] = (r, g, b)
        if now - self._last_cli_dispatch < self._cli_min_interval_s:
            return
        self._last_cli_dispatch = now
        items = self._pending_zone_rgb
        self._pending_zone_rgb = {}
        self._submit_cli(items)

    def _submit_cli(self, items: dict[int, tuple[int, int, int]]) -> None:
        """Run queued asusctl CLI effects off the event loop (or inline if none)."""
        if not items:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # No event loop (sync caller / tests): dispatch inline.
            for zone, rgb in items.items():
                self._run_asusctl(zone, rgb)
            return

        if self._cli_executor is None:
            self._cli_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-cli")
        for zone, rgb in items.items():
            loop.run_in_executor(self._cli_executor, self._run_asusctl, zone, rgb)

    @staticmethod
    def _run_asusctl(zone: int, rgb: tuple[int, int, int]) -> None:
        """Execute one asusctl static-effect subprocess call (worker-thread safe)."""
        r, g, b = rgb
        hex_color = f"{r:02x}{g:02x}{b:02x}"
        cmd = ["asusctl", "aura", "effect", "static", "-c", hex_color]
        if zone > 0:
            cmd.extend(["--zone", str(zone - 1)])
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0)
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
        self._player = player
        player.on_energy(self.on_voice_energy)

    def detach(self) -> None:
        """Disconnect from the audio player's per-block energy callback."""
        if self._player is not None:
            self._player.off_energy(self.on_voice_energy)
            self._player = None

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
        self.detach()
        self.set_brightness(3)
        self.set_color(0, 160, 255)
        if self._cli_executor is not None:
            self._cli_executor.shutdown(wait=False)
            self._cli_executor = None

    async def _animation_loop(self) -> None:
        """Vibrant, multi-zone ROG lighting animation separating keyboard zones from front lightbar."""
        t_start = time.time()
        # Always enforce 100% High brightness
        self.set_brightness(3)

        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - t_start
            mode = self._current_mode

            if mode == RobotMode.SPEAKING:
                # ── Voice Audio-Reactive Modulation ──
                raw_target = min(1.0, max(0.0, (self._voice_rms - 0.002) / 0.038))
                if raw_target > self._smooth_energy:
                    self._smooth_energy += (raw_target - self._smooth_energy) * 0.55
                else:
                    self._smooth_energy += (raw_target - self._smooth_energy) * 0.14

                # Front Underglow Lightbar (Zone 4): Solid reactive grounding bar
                strip_hue = (0.50 + 0.45 * self._smooth_energy) % 1.0
                strip_r, strip_g, strip_b = [c * 255.0 for c in colorsys.hsv_to_rgb(strip_hue, 1.0, 1.0)]

                # Keyboard (Zones 1, 2, 3): Traveling Voice Ripple wave across keys
                ripple = elapsed * 6.5
                p1 = 0.5 + 0.5 * math.sin(ripple)
                p2 = 0.5 + 0.5 * math.sin(ripple - 1.2)
                p3 = 0.5 + 0.5 * math.sin(ripple - 2.4)

                h1 = (0.52 + 0.40 * (self._smooth_energy * 0.6 + p1 * 0.4)) % 1.0
                h2 = (0.58 + 0.40 * (self._smooth_energy * 0.6 + p2 * 0.4)) % 1.0
                h3 = (0.64 + 0.40 * (self._smooth_energy * 0.6 + p3 * 0.4)) % 1.0

                z1_r, z1_g, z1_b = [c * 255.0 for c in colorsys.hsv_to_rgb(h1, 1.0, 1.0)]
                z2_r, z2_g, z2_b = [c * 255.0 for c in colorsys.hsv_to_rgb(h2, 1.0, 1.0)]
                z3_r, z3_g, z3_b = [c * 255.0 for c in colorsys.hsv_to_rgb(h3, 1.0, 1.0)]

            elif mode == RobotMode.THINKING:
                # Front Underglow Lightbar (Zone 4): Gentle golden breathing aura
                b_pulse = 0.5 + 0.5 * math.sin(elapsed * 2.5)
                strip_r, strip_g, strip_b = 255.0, 90.0 + 80.0 * b_pulse, 0.0

                # Keyboard (Zones 1, 2, 3): High-tech Neural Calculation ping-pong scanner!
                scan = 0.5 + 0.5 * math.sin(elapsed * 6.0)
                w1 = max(0.0, 1.0 - abs(scan - 0.0) * 2.5)
                w2 = max(0.0, 1.0 - abs(scan - 0.5) * 2.5)
                w3 = max(0.0, 1.0 - abs(scan - 1.0) * 2.5)

                z1_r, z1_g, z1_b = 255.0, 30.0 + 210.0 * w1, 20.0 * w1
                z2_r, z2_g, z2_b = 255.0, 30.0 + 210.0 * w2, 20.0 * w2
                z3_r, z3_g, z3_b = 255.0, 30.0 + 210.0 * w3, 20.0 * w3

            elif mode == RobotMode.LISTENING:
                # Front Underglow Lightbar (Zone 4): Steady vigilant electric emerald indicator
                strip_r, strip_g, strip_b = 10.0, 255.0, 40.0

                # Keyboard (Zones 1, 2, 3): Acoustic focus wave
                pulse = 0.5 + 0.5 * math.sin(elapsed * 3.0)
                z1_r, z1_g, z1_b = 0.0, 200.0 + 40.0 * pulse, 120.0 * (1.0 - pulse)
                z2_r, z2_g, z2_b = 20.0 + 60.0 * pulse, 255.0, 30.0
                z3_r, z3_g, z3_b = 0.0, 180.0 + 40.0 * pulse, 160.0 * (1.0 - pulse)

            elif mode == RobotMode.ERROR:
                flash = 1.0 if (int(elapsed * 6.0) % 2 == 0) else 0.15
                strip_r, strip_g, strip_b = 200.0, 0.0, 0.0
                z1_r = z2_r = z3_r = 255.0 * flash
                z1_g = z2_g = z3_g = 0.0
                z1_b = z2_b = z3_b = 0.0

            elif mode == RobotMode.SLEEPING:
                pulse = 0.5 + 0.5 * math.sin(elapsed * 1.0)
                strip_r, strip_g, strip_b = 20.0 * pulse, 0.0, 80.0 + 40.0 * pulse
                z1_r = z2_r = z3_r = 10.0 * pulse
                z1_g = z2_g = z3_g = 0.0
                z1_b = z2_b = z3_b = 25.0 * pulse

            else:
                # ── IDLE: Front Underglow = Cyberpunk Deep Indigo, Keyboard = Chromatic Aurora Flow ──
                pulse = 0.5 + 0.5 * math.sin(elapsed * 2.0)
                strip_r, strip_g, strip_b = 0.0, 40.0 + 80.0 * pulse, 255.0

                # Flowing aurora across keyboard keys
                flow = elapsed * 1.4
                h1 = (0.50 + 0.15 * math.sin(flow)) % 1.0
                h2 = (0.58 + 0.15 * math.sin(flow - 0.8)) % 1.0
                h3 = (0.66 + 0.15 * math.sin(flow - 1.6)) % 1.0

                z1_r, z1_g, z1_b = [c * 255.0 for c in colorsys.hsv_to_rgb(h1, 1.0, 1.0)]
                z2_r, z2_g, z2_b = [c * 255.0 for c in colorsys.hsv_to_rgb(h2, 1.0, 1.0)]
                z3_r, z3_g, z3_b = [c * 255.0 for c in colorsys.hsv_to_rgb(h3, 1.0, 1.0)]

            # Dispatch multi-zone colors
            self.set_color(int(z1_r), int(z1_g), int(z1_b), zone=1)
            self.set_color(int(z2_r), int(z2_g), int(z2_b), zone=2)
            self.set_color(int(z3_r), int(z3_g), int(z3_b), zone=3)
            self.set_color(int(strip_r), int(strip_g), int(strip_b), zone=4)
            await asyncio.sleep(0.038)
