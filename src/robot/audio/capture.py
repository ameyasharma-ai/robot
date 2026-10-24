"""Audio capture with DC offset removal and pre-roll buffering."""

from __future__ import annotations

import asyncio
import subprocess
import threading
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

#: A running PortAudio stream that stops delivering callbacks for this long is
#: treated as a dead/unplugged device (USB unplug leaves the stream "active"
#: but silent forever).
_STALE_CALLBACK_TIMEOUT_S = 2.0

#: Minimum spacing between automatic reopen attempts in ``ensure_healthy``.
_RECOVERY_BACKOFF_S = 1.0


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

        # Written by the PortAudio callback thread, read by the event loop.
        self._last_callback_ts: float = 0.0
        self._last_recovery_attempt_ts: float = 0.0
        self._stale_callback_timeout_s: float = _STALE_CALLBACK_TIMEOUT_S
        self._recovery_backoff_s: float = _RECOVERY_BACKOFF_S

        # Metric counters are mutated both from the PortAudio callback thread
        # and from the asyncio loop; every mutation goes through this lock so
        # increments are never lost.
        self._metrics_lock = threading.Lock()

    @property
    def state(self) -> AudioDeviceState:
        """Return the current capture device health state."""
        return self._state

    @property
    def metrics(self) -> AudioCaptureMetrics:
        """Return operational capture telemetry metrics."""
        return self._metrics

    def _bump_metrics(self, **deltas: int) -> None:
        """Atomically add deltas to callback/loop-shared metric counters.

        Args:
            **deltas: Metric attribute names mapped to integer increments.
        """
        if not deltas:
            return
        with self._metrics_lock:
            for name, delta in deltas.items():
                setattr(self._metrics, name, getattr(self._metrics, name) + delta)

    @staticmethod
    def _configure_device_gain() -> None:
        """Best-effort capture gain/unmute configuration with hard time bounds.

        PortAudio callbacks and the event loop must never be held hostage by a
        dead PipeWire/ALSA daemon, so every external mixer command is capped at
        one second and a missing utility is never treated as a fatal error.
        """
        commands = [
            ["pactl", "set-source-mute", "@DEFAULT_SOURCE@", "0"],
            ["pactl", "set-source-volume", "@DEFAULT_SOURCE@", "100%"],
            ["amixer", "-c", "1", "sset", "Internal Mic Boost", "0"],
            ["amixer", "-c", "1", "sset", "Capture", "35", "cap"],
        ]
        for command in commands:
            try:
                subprocess.run(
                    command,
                    capture_output=True,
                    timeout=1.0,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.debug("capture_mixer_command_failed", command=command[0], error=str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("capture_mixer_command_unexpected", command=command[0], error=str(exc))

    def _close_stream(self) -> None:
        """Stop and close the current PortAudio stream, if any.

        Blocking PortAudio calls stay confined to the caller; ``pause()``,
        ``resume()`` and ``stop()`` are offloaded to worker threads by the API
        layer where responsiveness matters.
        """
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as exc:
            logger.debug("capture_stream_stop_error", error=str(exc))
        try:
            stream.close()
        except Exception as exc:
            logger.debug("capture_stream_close_error", error=str(exc))

    def _open_stream(self) -> None:
        """Open and start sounddevice InputStream with bounded resilience.

        Any previously open stream is closed before a replacement is assigned,
        so a failed reopen can never leak a PortAudio device handle or leave a
        second callback thread enqueuing frames.
        """
        # Release the old handle *before* attempting a new one. Assigning only
        # after a successful start guarantees ``self._stream`` never points at
        # a half-open or definitively failed stream.
        self._close_stream()

        rates_to_try = [self.sample_rate]
        try:
            dev_info = sd.query_devices(sd.default.device[0], "input")
            native_rate = int(dev_info.get("default_samplerate", 44100))
            if native_rate != self.sample_rate:
                rates_to_try.append(native_rate)
        except Exception:
            rates_to_try.extend([44100, 48000])

        last_error: Exception | None = None
        for rate in rates_to_try:
            candidate: sd.InputStream | None = None
            try:
                candidate = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.frame_size,
                    callback=self._make_callback(rate),
                )
                candidate.start()
                self._stream = candidate
                if rate != self.sample_rate:
                    logger.warning(
                        "capture_rate_fallback",
                        requested=self.sample_rate,
                        actual=rate,
                        msg="Using device native rate with software resample",
                    )
                    self._actual_rate = rate
                    self._bump_metrics(sample_rate_mismatches=1)
                else:
                    self._actual_rate = self.sample_rate

                self._recovery_attempts = 0
                self._state = AudioDeviceState.HEALTHY
                self._last_callback_ts = time.monotonic()
                break
            except Exception as e:
                last_error = e
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                self._recovery_attempts += 1
                self._bump_metrics(errors=1)
                self._state = AudioDeviceState.RECOVERING
                logger.warning("capture_rate_failed", rate=rate, error=str(e))
                continue
        else:
            self._state = AudioDeviceState.DEGRADED
            self._bump_metrics(errors=1)
            raise RuntimeError(
                f"Could not open audio input at any sample rate. "
                f"Last error: {last_error}. "
                f"Try: systemctl --user restart pipewire wireplumber pipewire-pulse"
            )

        # Enforce clean hardware capture gain and guarantee microphone is unmuted.
        self._configure_device_gain()

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
            self._bump_metrics(queue_drops=1, dropouts=1)
            dropouts = self._metrics.dropouts
            now = time.monotonic()
            if now - self._last_queue_warn_ts >= 5.0:
                self._last_queue_warn_ts = now
                logger.warning(
                    "capture_queue_overflow_drop_oldest",
                    max_blocks=_MIC_QUEUE_MAX_BLOCKS,
                    total_drops=dropouts,
                )
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def _make_callback(self, actual_rate: int):
        """Create an audio callback that always emits exact VAD-sized frames.

        When software resampling is active, a continuous fractional-position
        resampler keeps phase continuity across block boundaries (per-block
        ``np.linspace`` restarts caused clicks and spectral artifacts). The
        resampled output is accumulated in a carry buffer and only complete
        ``frame_size`` frames are enqueued, because Silero VAD requires exactly
        512 samples at 16 kHz and raises ``ValueError`` for any other length.

        Args:
            actual_rate: Sample rate the PortAudio stream was opened at.

        Returns:
            Callable suitable for ``sounddevice.InputStream(callback=...)``.
        """
        need_resample = actual_rate != self.sample_rate

        if need_resample:
            # Mutable carry state: leftover tail sample for interpolation,
            # absolute fractional read position, and the partial output frame
            # waiting for enough resampled samples to reach frame_size.
            resample_state: dict[str, object] = {
                "tail": np.zeros(0, dtype=np.float32),
                "consumed": 0,       # total input samples seen so far
                "pos_abs": 0.0,      # absolute fractional read index
                "carry": np.zeros(0, dtype=np.float32),
                "generated": 0,      # total resampled samples produced
            }
            ratio = float(actual_rate) / float(self.sample_rate)  # input samples per output sample
            frame_size = int(self.frame_size)

            def _resample_block(block: np.ndarray) -> np.ndarray:
                """Resample one device block and return complete output frames."""
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
                resample_state["generated"] = int(resample_state["generated"]) + out.size

                # Accumulate resampled output and emit only whole VAD frames.
                carry = resample_state["carry"]
                combined = np.concatenate((carry, out)) if carry.size else out
                frame_count = combined.size // frame_size
                if frame_count:
                    frames = combined[: frame_count * frame_size].reshape(frame_count, frame_size)
                    resample_state["carry"] = combined[frame_count * frame_size :].copy()
                    return frames
                resample_state["carry"] = combined.copy()
                return np.zeros((0, frame_size), dtype=np.float32)

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            # Liveness is recorded even when this block only fills the
            # resampler carry buffer: the stream is demonstrably alive.
            self._last_callback_ts = time.monotonic()

            if status:
                overflows = 1 if getattr(status, "input_overflow", False) else 0
                underflows = 1 if getattr(status, "input_underflow", False) else 0
                if overflows or underflows:
                    self._bump_metrics(overflows=overflows, dropouts=underflows)
            self._bump_metrics(frames_captured=1)

            loop = self._loop
            if loop is None or self._queue is None:
                return

            data = indata.copy()
            if need_resample:
                frames_out = _resample_block(data.flatten())
                if frames_out.size == 0:
                    return
            else:
                if data.shape[0] != self.frame_size:
                    # VAD is frame-size sensitive; never enqueue a partial frame
                    # on the native-rate path either (PortAudio blocksize should
                    # already guarantee this, but a backend may coalesce blocks).
                    logger.debug(
                        "capture_frame_size_mismatch_dropped",
                        expected=self.frame_size,
                        actual=data.shape[0],
                    )
                    return
                frames_out = data.reshape(1, -1)

            # Teardown can close the loop while the PortAudio thread is still
            # draining; never let that abort the audio stream from its own
            # callback thread.
            if loop.is_closed():
                return
            try:
                for frame in frames_out:
                    loop.call_soon_threadsafe(self._offer_frame, frame.reshape(-1, 1))
            except RuntimeError as exc:
                # Event loop was closed between the check and the call.
                logger.debug("capture_loop_closed_frame_dropped", error=str(exc))

        return _callback

    def is_stale(self, now: float | None = None) -> bool:
        """Report whether the live stream has stopped delivering callbacks.

        Deliberately cheap and side-effect free so the audio hot path can call
        it once per frame on the event loop. Reopening the device is blocking
        PortAudio and mixer work and therefore belongs in
        :meth:`recover_async`, never here.

        Args:
            now: Optional monotonic timestamp override for deterministic tests.

        Returns:
            bool: True when a reopen attempt is due.
        """
        if self._stream is None or self._last_callback_ts <= 0.0:
            return False
        if self._recovery_attempts >= self._max_recovery_attempts:
            return False
        current = time.monotonic() if now is None else float(now)
        if (
            self._last_recovery_attempt_ts > 0.0
            and current - self._last_recovery_attempt_ts < self._recovery_backoff_s
        ):
            return False
        return current - self._last_callback_ts >= self._stale_callback_timeout_s

    async def recover_async(self) -> bool:
        """Reopen a stale capture device off the event loop.

        Returns:
            bool: True when the device is healthy (or no reopen was due).
        """
        if not self.is_stale():
            return True
        return await asyncio.to_thread(self.ensure_healthy)

    def ensure_healthy(self, now: float | None = None) -> bool:
        """Detect a silently dead capture stream and attempt one bounded reopen.

        A USB microphone that is unplugged (or a PipeWire node that dies) can
        leave the PortAudio stream nominally ``active`` while callbacks never
        fire again. This performs blocking device I/O: callers on the event loop
        must go through :meth:`recover_async`, which runs it in a worker thread.

        Args:
            now: Optional monotonic timestamp override for deterministic tests.

        Returns:
            bool: True when the stream is healthy (or intentionally paused),
                False when the device is stale/unrecoverable.
        """
        if self._stream is None:
            # Intentional pause (stream torn down by pause()/stop()) is not a
            # device failure. A failed startup leaves DEGRADED/ERROR state.
            return self._state not in (AudioDeviceState.DEGRADED, AudioDeviceState.ERROR)

        current = time.monotonic() if now is None else float(now)
        if self._last_callback_ts <= 0.0:
            self._last_callback_ts = current
            return True
        if current - self._last_callback_ts < self._stale_callback_timeout_s:
            return True

        # Stream is running but silent: treat as a dead device.
        self._state = AudioDeviceState.RECOVERING
        if self._recovery_attempts >= self._max_recovery_attempts:
            self._state = AudioDeviceState.ERROR
            logger.error(
                "capture_device_unrecoverable",
                recovery_attempts=self._recovery_attempts,
                max_attempts=self._max_recovery_attempts,
            )
            return False
        if (
            self._last_recovery_attempt_ts > 0.0
            and current - self._last_recovery_attempt_ts < self._recovery_backoff_s
        ):
            return False

        self._last_recovery_attempt_ts = current
        self._recovery_attempts += 1
        self._bump_metrics(errors=1)
        logger.warning(
            "capture_device_stale_reopening",
            silent_for_s=round(current - self._last_callback_ts, 3),
            attempt=self._recovery_attempts,
            max_attempts=self._max_recovery_attempts,
        )
        try:
            self._open_stream()
        except Exception as exc:
            self._state = (
                AudioDeviceState.ERROR
                if self._recovery_attempts >= self._max_recovery_attempts
                else AudioDeviceState.RECOVERING
            )
            logger.error(
                "capture_device_reopen_failed",
                attempt=self._recovery_attempts,
                error=str(exc),
            )
            return False

        self._last_callback_ts = time.monotonic()
        logger.info("capture_device_recovered", sample_rate=self._actual_rate)
        return True

    def pause(self) -> None:
        """Pause audio input capture, close stream, and release hardware microphone."""
        if self._stream is not None:
            try:
                self._close_stream()
            except Exception as e:  # pragma: no cover - _close_stream is defensive
                logger.debug("capture_stream_pause_error", error=str(e))
            logger.info("capture_stream_paused_and_hardware_released")
        self._pre_roll.clear()
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break

    def _attach_loop_and_queue(self) -> None:
        """Bind the running loop and create the capture queue if absent."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=_MIC_QUEUE_MAX_BLOCKS)

    def _resume_device(self) -> None:
        """Reopen or restart the PortAudio stream (blocking device I/O)."""
        if self._stream is None:
            try:
                self._open_stream()
                logger.info("capture_stream_resumed", sample_rate=getattr(self, "_actual_rate", self.sample_rate))
            except Exception as e:
                self._state = AudioDeviceState.DEGRADED
                self._bump_metrics(errors=1)
                logger.warning("capture_stream_resume_error", error=str(e))
        elif getattr(self._stream, "stopped", False):
            try:
                self._stream.start()
                self._state = AudioDeviceState.HEALTHY
                self._last_callback_ts = time.monotonic()
                logger.info("capture_stream_restarted")
            except Exception as e:
                self._state = AudioDeviceState.DEGRADED
                self._bump_metrics(errors=1)
                logger.debug("capture_stream_restart_error", error=str(e))

    def resume(self) -> None:
        """Resume audio input capture and recreate microphone stream.

        Blocking: callers running on the event loop must use
        :meth:`resume_async` instead.
        """
        self._attach_loop_and_queue()
        self._resume_device()

    async def resume_async(self) -> None:
        """Resume capture with the PortAudio/mixer work off the event loop."""
        self._attach_loop_and_queue()
        await asyncio.to_thread(self._resume_device)

    @property
    def is_paused(self) -> bool:
        return self._stream is None or getattr(self._stream, "stopped", True)

    async def stop(self) -> None:
        """Stop capture and release the PortAudio stream off the loop."""
        if self._stream is not None:
            await asyncio.to_thread(self._close_stream)
            logger.info("capture_stopped")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply DC offset removal and compute RMS.

        Pure DSP: the capture consumer in the engine calls
        :meth:`recover_async` for device liveness, because a reopen must never
        happen inside this hot path.

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
