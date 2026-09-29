"""Audio capture with DC offset removal and pre-roll buffering."""

from __future__ import annotations

import asyncio
from collections import deque

import numpy as np
import sounddevice as sd

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
        """Start capturing audio, return queue of frames."""
        self._loop = loop
        self._queue = asyncio.Queue()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("capture_started", sample_rate=self.sample_rate, frame_size=self.frame_size)
        return self._queue

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
