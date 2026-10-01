"""Audio playback with interrupt support for barge-in."""

from __future__ import annotations

import asyncio
from collections import deque
import threading

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
        self._current_rms: float = 0.0
        self._lock = threading.Lock()
        # Rolling buffer of recent block RMS values (covers ~120ms acoustic travel)
        self._recent_rms: deque[float] = deque(maxlen=3)
        self._energy_callbacks: list = []

    def on_energy(self, callback) -> None:
        """Register callback for each playback audio frame: fn(rms: float)."""
        self._energy_callbacks.append(callback)

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_rms(self) -> float:
        if not self._is_playing or not self._recent_rms:
            return 0.0
        return max(self._recent_rms)

    def _ensure_stream(self, sample_rate: int) -> sd.OutputStream:
        with self._lock:
            needs_new = (
                self._stream is None
                or sample_rate != self._stream_sr
                or getattr(self._stream, "closed", True)
            )
            if needs_new:
                if self._stream is not None:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        pass
                self._stream = sd.OutputStream(
                    samplerate=sample_rate, channels=1, dtype="float32", latency="low"
                )
                self._stream.start()
                self._stream_sr = sample_rate
            elif self._stream.stopped:
                try:
                    self._stream.start()
                except Exception:
                    pass
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
        block_len = int(sample_rate * self.block_ms / 1000)
        self._is_playing = True

        try:
            stream = self._ensure_stream(sample_rate)
            for idx in range(0, len(audio), block_len):
                if interrupt.is_set():
                    logger.info("playback_interrupted")
                    return False

                chunk = audio[idx : idx + block_len]
                block_rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
                self._current_rms = block_rms
                self._recent_rms.append(block_rms)
                for cb in self._energy_callbacks:
                    try:
                        cb(block_rms)
                    except Exception:
                        pass
                if chunk.ndim == 1:
                    chunk = chunk.reshape(-1, 1)

                try:
                    if interrupt.is_set():
                        return False
                    
                    def _write_block():
                        with self._lock:
                            if self._stream is not None and not interrupt.is_set():
                                try:
                                    self._stream.write(chunk)
                                except Exception as e:
                                    logger.debug("stream_write_error", error=str(e))

                    await asyncio.to_thread(_write_block)
                except Exception as exc:
                    logger.debug("audio_write_aborted_or_interrupted", error=str(exc))
                    return False
            return True
        finally:
            self._is_playing = False
            self._current_rms = 0.0
            self._recent_rms.clear()

    def stop(self) -> None:
        """Immediately stop playback without crashing underlying ALSA mmap buffers."""
        self._is_playing = False
        self._current_rms = 0.0
        self._recent_rms.clear()
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass

    def close(self) -> None:
        """Release PortAudio resources on shutdown."""
        self.stop()
        with self._lock:
            stream = self._stream
            self._stream = None
            self._stream_sr = None
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def __del__(self) -> None:
        self.close()
