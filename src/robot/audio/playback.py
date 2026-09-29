"""Audio playback with interrupt support for barge-in."""

from __future__ import annotations

import asyncio

import numpy as np
import sounddevice as sd

from robot.telemetry import get_logger

logger = get_logger("audio.playback")


class AudioPlayer:
    """Non-blocking audio playback with instant interrupt capability.

    Plays audio in small blocks (40ms) so barge-in cancellation
    takes effect within a single block boundary (~40ms latency).
    """

    def __init__(self, block_ms: int = 40) -> None:
        self.block_ms = block_ms
        self._stream: sd.OutputStream | None = None
        self._stream_sr: int | None = None
        self._is_playing: bool = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def _ensure_stream(self, sample_rate: int) -> sd.OutputStream:
        if self._stream is None or sample_rate != self._stream_sr:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
            self._stream = sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="float32"
            )
            self._stream.start()
            self._stream_sr = sample_rate
        return self._stream

    async def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        interrupt: asyncio.Event,
    ) -> bool:
        """Play audio with interrupt checking per block.

        Returns:
            True if playback completed, False if interrupted.
        """
        stream = self._ensure_stream(sample_rate)
        block_len = int(sample_rate * self.block_ms / 1000)
        self._is_playing = True

        try:
            for idx in range(0, len(audio), block_len):
                if interrupt.is_set():
                    logger.info("playback_interrupted")
                    try:
                        stream.abort()
                        stream.start()
                    except Exception:
                        pass
                    return False
                chunk = audio[idx : idx + block_len]
                if chunk.ndim == 1:
                    chunk = chunk.reshape(-1, 1)
                stream.write(chunk)
                await asyncio.sleep(self.block_ms * 0.001 * 0.875)  # Slight under-sleep for smooth flow
            return True
        finally:
            self._is_playing = False

    def stop(self) -> None:
        """Immediately stop and release the output stream."""
        self._is_playing = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._stream_sr = None

    def __del__(self) -> None:
        self.stop()
