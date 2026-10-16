"""Audio playback with interrupt support for barge-in."""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from collections import deque

import numpy as np
import sounddevice as sd

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*Setting the shape on a NumPy array.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sounddevice")

from robot.telemetry import get_logger

logger = get_logger("audio.playback")


class AudioPlayer:
    """Non-blocking audio playback with instant interrupt capability.

    Plays audio in small blocks (40ms) so barge-in cancellation
    takes effect within a single block boundary (~40ms latency).
    """

    def __init__(self, block_ms: int = 60) -> None:
        self.block_ms = block_ms
        self._stream: sd.OutputStream | None = None
        self._stream_sr: int | None = None
        self._is_playing: bool = False
        self._current_rms: float = 0.0
        self._volume: float = 1.0
        self._duck_target: float = 1.0
        self._duck_rate: float = 0.15
        self._lock = threading.Lock()
        # Rolling buffer of recent block RMS values (covers ~320ms acoustic travel & hardware DAC ring buffer)
        self._recent_rms: deque[float] = deque(maxlen=8)
        self._energy_callbacks: list = []
        self._last_play_time: float = 0.0

    def duck(self, target_volume: float = 1.0, rate: float = 0.12) -> None:
        """Disabled: speech plays at full volume unattenuated."""
        self._duck_target = 1.0
        self._volume = 1.0

    def fade_out(self, rate: float = 0.15) -> None:
        """Disabled: speech plays at full volume unattenuated."""
        self._duck_target = 1.0
        self._volume = 1.0

    def unduck(self) -> None:
        """Restore TTS volume back to 1.0."""
        self._duck_target = 1.0
        self._volume = 1.0

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
                    samplerate=sample_rate, channels=1, dtype="float32", latency=0.08
                )
                self._stream.start()
                self._stream_sr = sample_rate
                # Pre-roll 60ms silence to absorb DAC power-up / anti-pop ramp-up
                lead_in = np.zeros(int(sample_rate * 0.06), dtype=np.float32).reshape(-1, 1)
                try:
                    self._stream.write(lead_in)
                except Exception:
                    pass
            elif self._stream.stopped:
                try:
                    self._stream.start()
                    lead_in = np.zeros(int(sample_rate * 0.04), dtype=np.float32).reshape(-1, 1)
                    self._stream.write(lead_in)
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

        # Prepend 25ms silence only if starting fresh from idle (>0.8s gap) to prevent DAC ramp from eating the first syllable
        is_fresh_start = (time.time() - self._last_play_time) > 0.8 or not self._stream or getattr(self._stream, "stopped", True)
        if is_fresh_start:
            pre_silence = np.zeros(int(sample_rate * 0.025), dtype=np.float32)
            audio = np.concatenate([pre_silence, audio])

        self._is_playing = True
        self._duck_target = 1.0
        self._volume = 1.0

        try:
            self._ensure_stream(sample_rate)

            def _play_blocking() -> bool:
                for idx in range(0, len(audio), block_len):
                    if interrupt.is_set():
                        logger.info("playback_interrupted")
                        return False

                    chunk = audio[idx : idx + block_len]
                    # Smoothly adjust volume towards target (ducking fade-out/in)
                    if abs(self._volume - self._duck_target) > 0.01:
                        if self._volume > self._duck_target:
                            self._volume = max(self._duck_target, self._volume - self._duck_rate)
                        else:
                            self._volume = min(self._duck_target, self._volume + self._duck_rate)
                    else:
                        self._volume = self._duck_target

                    chunk = (chunk * self._volume).astype(np.float32)

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

                    if interrupt.is_set():
                        logger.info("playback_interrupted_barge_in")
                        with self._lock:
                            if self._stream is not None:
                                try:
                                    self._stream.abort()
                                except Exception:
                                    pass
                        return False

                    with self._lock:
                        if self._stream is not None and not interrupt.is_set():
                            try:
                                self._stream.write(chunk)
                            except Exception as e:
                                logger.debug("stream_write_error", error=str(e))
                                return False
                return True

            return await asyncio.to_thread(_play_blocking)
        finally:
            self._is_playing = False
            self._current_rms = 0.0
            self._recent_rms.clear()
            self._last_play_time = time.time()

    def stop(self) -> None:
        """Immediately stop playback and purge sounddevice/ALSA ring buffer on barge-in."""
        self._is_playing = False
        self._current_rms = 0.0
        self._recent_rms.clear()
        self._volume = 1.0
        self._duck_target = 1.0
        with self._lock:
            if self._stream is not None:
                try:
                    # abort() immediately discards unplayed frames in hardware buffer (<40ms)
                    self._stream.abort()
                except Exception:
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
