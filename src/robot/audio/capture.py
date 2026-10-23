"""Audio capture with DC offset removal and pre-roll buffering."""

from __future__ import annotations

import asyncio
import time
from collections import deque

import warnings

import numpy as np
import sounddevice as sd

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*Setting the shape on a NumPy array.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sounddevice")

from dataclasses import dataclass

from robot.audio.playback import AudioDeviceState
from robot.telemetry import get_logger

logger = get_logger("audio.capture")

# Max queued mic blocks (~100 * 512 samples ≈ 3.2s at 16kHz). A stale mic
# backlog is worse than dropped frames for a live VAD pipeline, so the queue
# is bounded with explicit drop-oldest policy.
_MIC_QUEUE_MAX_BLOCKS = 100


@dataclass
class AudioCaptureMetrics:
    """Non-blocking telemetry metrics for audio input subsystem."""

    frames_captured: int = 0
    overflows: int = 0
    dropouts: int = 0
    errors: int = 0
    sample_rate_mismatches: int = 0
    queue_drops: int = 0


class AudioCapture:
    """Microphone input stream with DC offset removal.

    Feeds raw float32 frames into an asyncio queue for the VAD pipeline.
    Maintains a pre-roll buffer so speech onset isn't lost.
    Tracks explicit hardware health states and capture metrics.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        pre_roll_ms: int = 500,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._pre_roll_len = int(sample_rate * pre_roll_ms / 1000)
        self._pre_roll: deque[float] = deque(maxlen=self._pre_roll_len)
        self._dc_mean = 0.0
        self._queue: asyncio.Queue[np.ndarray] | None = None
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._actual_rate: int = sample_rate
        self._last_queue_warn_ts: float = 0.0

        # Work Package B resilience enhancements
        self._state: AudioDeviceState = AudioDeviceState.HEALTHY
        self._metrics: AudioCaptureMetrics = AudioCaptureMetrics()
        self._recovery_attempts: int = 0
        self._max_recovery_attempts: int = 3

    @property
    def state(self) -> AudioDeviceState:
        """Return the current capture device health state."""
        return self._state

    @property
    def metrics(self) -> AudioCaptureMetrics:
        """Return operational capture telemetry metrics."""
        return self._metrics

    def _open_stream(self) -> None:
        """Open and start sounddevice InputStream with bounded resilience."""
        rates_to_try = [self.sample_rate]
        try:
            dev_info = sd.query_devices(sd.default.device[0], "input")
            native_rate = int(dev_info.get("default_samplerate", 44100))
            if native_rate != self.sample_rate:
                rates_to_try.append(native_rate)
        except Exception:
            rates_to_try.extend([44100, 48000])

        last_error = None
        for rate in rates_to_try:
            try:
                self._stream = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.frame_size,
                    callback=self._make_callback(rate),
                )
                self._stream.start()
                if rate != self.sample_rate:
                    logger.warning(
                        "capture_rate_fallback",
                        requested=self.sample_rate,
                        actual=rate,
                        msg="Using device native rate with software resample",
                    )
                    self._actual_rate = rate
                    self._metrics.sample_rate_mismatches += 1
                else:
                    self._actual_rate = self.sample_rate

                self._recovery_attempts = 0
                self._state = AudioDeviceState.HEALTHY
                break
            except Exception as e:
                last_error = e
                self._recovery_attempts += 1
                self._metrics.errors += 1
                self._state = AudioDeviceState.RECOVERING
                logger.warning("capture_rate_failed", rate=rate, error=str(e))
                continue
        else:
            self._state = AudioDeviceState.DEGRADED
            self._metrics.errors += 1
            raise RuntimeError(
                f"Could not open audio input at any sample rate. "
                f"Last error: {last_error}. "
                f"Try: systemctl --user restart pipewire wireplumber pipewire-pulse"
            )

        # Enforce clean hardware capture gain and guarantee microphone is unmuted
        try:
            import subprocess
            subprocess.run(["pactl", "set-source-mute", "@DEFAULT_SOURCE@", "0"], capture_output=True)
            subprocess.run(["pactl", "set-source-volume", "@DEFAULT_SOURCE@", "100%"], capture_output=True)
            subprocess.run(["amixer", "-c", "1", "sset", "Internal Mic Boost", "0"], capture_output=True)
            subprocess.run(["amixer", "-c", "1", "sset", "Capture", "35", "cap"], capture_output=True)
        except Exception:
            pass

    async def start(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue[np.ndarray]:
        """Start capturing audio, return queue of frames.

        Tries the configured sample rate first (16 kHz), then falls back to
        the device's native rate with software resampling. This prevents
        crashes when PipeWire is down and ALSA rejects non-native rates.

        The whole PortAudio open plus pactl/amixer device-configuration
        sequence is blocking I/O, so it is offloaded to a worker thread to
        keep the event loop responsive.
        """
        self._loop = loop
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=_MIC_QUEUE_MAX_BLOCKS)

        await asyncio.to_thread(self._open_stream)
        logger.info("capture_started", sample_rate=self._actual_rate, frame_size=self.frame_size)
        return self._queue

    def _offer_frame(self, data: np.ndarray) -> None:
        """Enqueue a mic frame on the event loop with drop-oldest backpressure.

        Runs on the event loop via call_soon_threadsafe. A bounded queue with
        explicit drops is safer than an unbounded backlog of stale audio.
        """
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop the oldest stale block to make room for live audio.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._metrics.queue_drops += 1
            self._metrics.dropouts += 1
            now = time.monotonic()
            if now - self._last_queue_warn_ts >= 5.0:
                self._last_queue_warn_ts = now
                logger.warning(
                    "capture_queue_overflow_drop_oldest",
                    max_blocks=_MIC_QUEUE_MAX_BLOCKS,
                    total_drops=self._metrics.queue_drops,
                )
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def _make_callback(self, actual_rate: int):
        """Create an audio callback, with optional resampling if rates differ.

        When software resampling is active, a continuous fractional-position
        resampler keeps phase continuity across 512-sample block boundaries
        (per-block np.linspace restarts caused clicks and spectral artifacts).
        """
        need_resample = actual_rate != self.sample_rate

        if need_resample:
            # Mutable carry state: leftover tail sample for interpolation and
            # the absolute fractional read position within the input stream.
            resample_state: dict[str, object] = {
                "tail": np.zeros(0, dtype=np.float32),
                "consumed": 0,       # total input samples seen so far
                "pos_abs": 0.0,      # absolute fractional read index
            }
            ratio = float(actual_rate) / float(self.sample_rate)  # input samples per output sample

            def _resample_block(block: np.ndarray) -> np.ndarray:
                tail = resample_state["tail"]
                flat = block.astype(np.float32, copy=False).flatten()
                if tail.size:
                    buf = np.concatenate((tail, flat))
                else:
                    buf = flat
                n = len(buf)
                base = int(resample_state["consumed"]) - tail.size  # absolute index of buf[0]
                rel_pos = float(resample_state["pos_abs"]) - base
                consumed_next = int(resample_state["consumed"]) + len(flat)

                if n >= 2 and rel_pos <= n - 2:
                    count = int((n - 1 - rel_pos) / ratio) + 1
                    indices = rel_pos + ratio * np.arange(count, dtype=np.float64)
                    out = np.interp(indices, np.arange(n, dtype=np.float64), buf).astype(np.float32)
                    # Advance fractional position past the last emitted sample
                    # so the next block continues seamlessly.
                    resample_state["pos_abs"] = base + rel_pos + ratio * count
                else:
                    out = np.zeros(0, dtype=np.float32)
                    resample_state["pos_abs"] = max(
                        float(resample_state["pos_abs"]), base + n - 1
                    )

                # Carry the final input sample so the next block can
                # interpolate across the block boundary.
                resample_state["tail"] = buf[-1:].copy()
                resample_state["consumed"] = consumed_next
                return out

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                if getattr(status, "input_overflow", False):
                    self._metrics.overflows += 1
                if getattr(status, "input_underflow", False):
                    self._metrics.dropouts += 1
            self._metrics.frames_captured += 1

            if self._loop and self._queue:
                data = indata.copy()
                if need_resample:
                    out = _resample_block(data.flatten())
                    if out.size == 0:
                        return
                    data = out.reshape(-1, 1)
                self._loop.call_soon_threadsafe(self._offer_frame, data)

        return _callback

    def pause(self) -> None:
        """Pause audio input capture, close stream, and release hardware microphone."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("capture_stream_pause_error", error=str(e))
            self._stream = None
            logger.info("capture_stream_paused_and_hardware_released")
        self._pre_roll.clear()
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break

    def resume(self) -> None:
        """Resume audio input capture and recreate microphone stream."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=_MIC_QUEUE_MAX_BLOCKS)

        if self._stream is None:
            try:
                self._open_stream()
                logger.info("capture_stream_resumed", sample_rate=getattr(self, "_actual_rate", self.sample_rate))
            except Exception as e:
                self._state = AudioDeviceState.DEGRADED
                self._metrics.errors += 1
                logger.warning("capture_stream_resume_error", error=str(e))
        elif self._stream.stopped:
            try:
                self._stream.start()
                self._state = AudioDeviceState.HEALTHY
                logger.info("capture_stream_restarted")
            except Exception as e:
                self._state = AudioDeviceState.DEGRADED
                self._metrics.errors += 1
                logger.debug("capture_stream_restart_error", error=str(e))

    @property
    def is_paused(self) -> bool:
        return self._stream is None or self._stream.stopped

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("capture_stopped")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply DC offset removal and compute RMS.

        Returns:
            (clean_chunk, rms_energy)
        """
        raw = frame.flatten()

        # Direct zero-mean centering for exact DC offset removal
        frame_mean = float(np.mean(raw))
        clean = raw - frame_mean
        rms = float(np.sqrt(np.mean(clean**2)))

        # Update pre-roll buffer with clean zero-centered audio
        self._pre_roll.extend(clean.tolist())

        return clean, rms

    def get_pre_roll(self) -> np.ndarray:
        """Return the pre-roll buffer as a numpy array."""
        return np.array(list(self._pre_roll), dtype=np.float32)
