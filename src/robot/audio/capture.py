"""Audio capture with DC offset removal and pre-roll buffering."""

from __future__ import annotations

import asyncio
from collections import deque

import warnings

import numpy as np
import sounddevice as sd

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*Setting the shape on a NumPy array.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sounddevice")

from robot.telemetry import get_logger

logger = get_logger("audio.capture")


class AudioCapture:
    """Microphone input stream with DC offset removal.

    Feeds raw float32 frames into an asyncio queue for the VAD pipeline.
    Maintains a pre-roll buffer so speech onset isn't lost.
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

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — runs in audio thread."""
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy())

    async def start(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue[np.ndarray]:
        """Start capturing audio, return queue of frames.

        Tries the configured sample rate first (16 kHz), then falls back to
        the device's native rate with software resampling. This prevents
        crashes when PipeWire is down and ALSA rejects non-native rates.
        """
        self._loop = loop
        self._queue = asyncio.Queue()

        # Try sample rates in preference order
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
                else:
                    self._actual_rate = self.sample_rate
                break
            except Exception as e:
                last_error = e
                logger.warning("capture_rate_failed", rate=rate, error=str(e))
                continue
        else:
            raise RuntimeError(
                f"Could not open audio input at any sample rate. "
                f"Last error: {last_error}. "
                f"Try: systemctl --user restart pipewire wireplumber pipewire-pulse"
            )

        # Enforce clean hardware capture gain (boost 0)
        try:
            import subprocess
            subprocess.run(["amixer", "-c", "1", "sset", "Internal Mic Boost", "0"], capture_output=True)
        except Exception:
            pass

        logger.info("capture_started", sample_rate=self._actual_rate, frame_size=self.frame_size)
        return self._queue

    def _make_callback(self, actual_rate: int):
        """Create an audio callback, with optional resampling if rates differ."""
        need_resample = actual_rate != self.sample_rate

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if self._loop and self._queue:
                data = indata.copy()
                if need_resample:
                    # Simple linear interpolation resample (low overhead)
                    ratio = self.sample_rate / actual_rate
                    n_out = int(len(data) * ratio)
                    indices = np.linspace(0, len(data) - 1, n_out)
                    data = np.interp(indices, np.arange(len(data)), data.flatten()).astype(np.float32).reshape(-1, 1)
                self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

        return _callback

    def pause(self) -> None:
        """Pause audio input capture and release/stop the microphone stream."""
        if self._stream and not self._stream.stopped:
            try:
                self._stream.stop()
                logger.info("capture_stream_paused")
            except Exception as e:
                logger.debug("capture_stream_pause_error", error=str(e))
        self._pre_roll.clear()

    def resume(self) -> None:
        """Resume audio input capture and restart microphone stream."""
        if self._stream and self._stream.stopped and not getattr(self._stream, "closed", False):
            try:
                self._stream.start()
                logger.info("capture_stream_resumed")
            except Exception as e:
                logger.debug("capture_stream_resume_error", error=str(e))

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
