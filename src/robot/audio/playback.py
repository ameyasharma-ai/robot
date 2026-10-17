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


from dataclasses import dataclass
from enum import Enum

class AudioDeviceState(str, Enum):
    """Operational health state of audio hardware input/output devices."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    ERROR = "error"


@dataclass
class AudioPlaybackMetrics:
    """Non-blocking telemetry metrics for audio output subsystem."""

    blocks_played: int = 0
    underflows: int = 0
    stream_restarts: int = 0
    errors: int = 0
    dropouts: int = 0
    total_playback_time_s: float = 0.0
    last_block_latency_ms: float = 0.0


class AudioPlayer:
    """Non-blocking audio playback with instant interrupt capability.

    Plays audio in small blocks (40ms) so barge-in cancellation
    takes effect within a single block boundary (~40ms latency).
    Enforces bounded hardware recovery, generation tracking to purge
    stale synthesized clauses, and explicit device health states.
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
        self._playback_lock: asyncio.Lock | None = None
        # Rolling buffer of recent block RMS values (covers ~320ms acoustic travel & hardware DAC ring buffer)
        self._recent_rms: deque[float] = deque(maxlen=8)
        self._energy_callbacks: list = []
        self._last_play_time: float = 0.0

        # Work Package B resilience enhancements
        self._state: AudioDeviceState = AudioDeviceState.HEALTHY
        self._metrics: AudioPlaybackMetrics = AudioPlaybackMetrics()
        self._generation_id: int = 0
        self._recovery_attempts: int = 0
        self._max_recovery_attempts: int = 3

    @property
    def state(self) -> AudioDeviceState:
        """Return the current device health state."""
        return self._state

    @property
    def metrics(self) -> AudioPlaybackMetrics:
        """Return operational playback telemetry metrics."""
        return self._metrics

    @property
    def generation_id(self) -> int:
        """Return the current monotonic playback generation ID."""
        return self._generation_id

    def new_generation(self) -> int:
        """Increment and return a new generation ID to isolate audio turn synthesis.

        Returns:
            int: The newly assigned monotonic generation ID.
        """
        self._generation_id += 1
        return self._generation_id

    @property
    def playback_lock(self) -> asyncio.Lock:
        """Return the asyncio.Lock used to serialize audio playback streams.

        Guarantees that earcon chimes and speech synthesis phrases never concurrently
        clobber the underlying PortAudio ALSA output stream or race during sample-rate shifts.

        Returns:
            asyncio.Lock instance bound to the active event loop.
        """
        if self._playback_lock is None:
            self._playback_lock = asyncio.Lock()
        return self._playback_lock

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

    def _ensure_stream(self, sample_rate: int) -> sd.OutputStream | None:
        """Acquire or recreate PortAudio output stream with bounded backoff and error recovery.

        Args:
            sample_rate: Required output audio sampling rate in Hz.

        Returns:
            sd.OutputStream instance if healthy/recovered, or None if hardware is degraded.
        """
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

                # Attempt stream instantiation with bounded exponential backoff
                for attempt in range(1, self._max_recovery_attempts + 1):
                    try:
                        self._stream = sd.OutputStream(
                            samplerate=sample_rate, channels=1, dtype="float32", latency=0.08
                        )
                        self._stream.start()
                        self._stream_sr = sample_rate
                        self._recovery_attempts = 0
                        self._state = AudioDeviceState.HEALTHY
                        # Pre-roll 60ms silence to absorb DAC power-up / anti-pop ramp-up
                        lead_in = np.zeros(int(sample_rate * 0.06), dtype=np.float32).reshape(-1, 1)
                        try:
                            self._stream.write(lead_in)
                        except Exception:
                            pass
                        return self._stream
                    except Exception as e:
                        self._recovery_attempts += 1
                        self._metrics.errors += 1
                        self._metrics.stream_restarts += 1
                        self._state = AudioDeviceState.RECOVERING
                        logger.warning(
                            "audio_output_stream_open_retry",
                            attempt=attempt,
                            max_attempts=self._max_recovery_attempts,
                            error=str(e),
                        )
                        if attempt < self._max_recovery_attempts:
                            time.sleep(0.04 * (2 ** (attempt - 1)))

                self._state = AudioDeviceState.DEGRADED
                logger.error("audio_playback_device_degraded_stream_creation_failed")
                return None

            elif self._stream.stopped or not getattr(self._stream, "active", True):
                try:
                    self._stream.start()
                    lead_in = np.zeros(int(sample_rate * 0.04), dtype=np.float32).reshape(-1, 1)
                    self._stream.write(lead_in)
                    self._state = AudioDeviceState.HEALTHY
                except Exception:
                    # Cleanly close and recreate if stream failed to restart from abort
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    try:
                        self._stream = sd.OutputStream(
                            samplerate=sample_rate, channels=1, dtype="float32", latency=0.08
                        )
                        self._stream.start()
                        self._stream_sr = sample_rate
                        self._state = AudioDeviceState.HEALTHY
                    except Exception as e:
                        self._metrics.errors += 1
                        self._state = AudioDeviceState.DEGRADED
                        logger.error("audio_output_stream_restart_failed", error=str(e))
                        return None
            return self._stream

    async def play_earcon(
        self,
        earcon_type: Any,
        sample_rate: int | None = None,
        volume: float = 0.65,
        interrupt: asyncio.Event | None = None,
        generation_id: int | None = None,
    ) -> bool:
        """Play a synthesized Apple-grade acoustic earcon chime.

        Args:
            earcon_type: Target earcon signature to synthesize and play.
            sample_rate: Audio sampling frequency in Hz (defaults to current stream rate or 22050).
            volume: Master volume multiplier between 0.0 and 1.0 (default 0.65).
            interrupt: Optional event to interrupt earcon playback.
            generation_id: Optional generation token for barge-in isolation.

        Returns:
            True if earcon finished playing, False if interrupted.
        """
        from robot.audio.earcon import AcousticEarconEngine

        sr = sample_rate or self._stream_sr or 22050
        waveform = AcousticEarconEngine.generate(earcon_type, sample_rate=sr, volume=volume)
        evt = interrupt or asyncio.Event()
        return await self.play(waveform, sr, evt, generation_id=generation_id)

    async def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        interrupt: asyncio.Event,
        generation_id: int | None = None,
    ) -> bool:
        """Play audio with interrupt checking per block, anti-pop fading, and soft limiting.

        Guarantees serial playback through the instance's asynchronous playback lock so
        earcons, wake cues, and speech synthesis phrases never clobber each other's stream.
        Rejects stale synthesis from previous generations upon barge-in interruptions.

        Args:
            audio: 1D or 2D NumPy array containing float32 audio samples.
            sample_rate: Audio sampling frequency in Hz.
            interrupt: Event signaling immediate barge-in interrupt.
            generation_id: Optional generation token to purge obsolete speech synthesis.

        Returns:
            True if playback completed, False if interrupted or rejected as stale.
        """
        if len(audio) == 0:
            return True

        if interrupt.is_set() or (generation_id is not None and generation_id != self._generation_id):
            return False

        async with self.playback_lock:
            if interrupt.is_set() or (generation_id is not None and generation_id != self._generation_id):
                return False

            # Apple CoreAudio-grade anti-pop: 5ms cosine micro-fade on onset and offset
            fade_len = min(int(sample_rate * 0.005), len(audio))
            if fade_len > 0:
                audio = audio.copy()
                fade_in = (np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2).astype(np.float32)
                audio[:fade_len] *= fade_in
                fade_out = (np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2).astype(np.float32)
                audio[-fade_len:] *= fade_out

            # CoreAudio-grade soft saturation peak limiter: transparent for |x| <= 0.95, smoothly saturates above
            peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            if peak > 0.95:
                audio = (np.tanh(audio / 0.95) * 0.95).astype(np.float32)

            block_len = int(sample_rate * self.block_ms / 1000)

            # Prepend 25ms silence only if starting fresh from idle (>0.8s gap) to prevent DAC ramp from eating the first syllable
            is_fresh_start = (time.time() - self._last_play_time) > 0.8 or not self._stream or getattr(self._stream, "stopped", True)
            if is_fresh_start:
                pre_silence = np.zeros(int(sample_rate * 0.025), dtype=np.float32)
                audio = np.concatenate([pre_silence, audio])

            # Append 40ms silence so PortAudio latency buffer / ALSA DAC flushes the final word completely
            post_silence = np.zeros(int(sample_rate * 0.04), dtype=np.float32)
            audio = np.concatenate([audio, post_silence])

            self._is_playing = True
            self._duck_target = 1.0
            self._volume = 1.0

            try:
                stream = self._ensure_stream(sample_rate)
                if stream is None:
                    logger.warning("playback_failed_hardware_degraded")
                    return False
                self._stream = stream

                def _play_blocking() -> bool:
                    t_play_start = time.time()
                    for idx in range(0, len(audio), block_len):
                        if interrupt.is_set() or (generation_id is not None and generation_id != self._generation_id):
                            logger.info("playback_interrupted_or_generation_stale")
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

                        if interrupt.is_set() or (generation_id is not None and generation_id != self._generation_id):
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
                                if generation_id is not None and generation_id != self._generation_id:
                                    try:
                                        self._stream.abort()
                                    except Exception:
                                        pass
                                    return False
                                try:
                                    t0 = time.time()
                                    self._stream.write(chunk)
                                    latency_ms = (time.time() - t0) * 1000.0
                                    self._metrics.blocks_played += 1
                                    self._metrics.last_block_latency_ms = latency_ms
                                except Exception as e:
                                    self._metrics.errors += 1
                                    self._metrics.underflows += 1
                                    logger.debug("stream_write_error", error=str(e))
                                    return False
                    self._metrics.total_playback_time_s += (time.time() - t_play_start)
                    return True

                return await asyncio.to_thread(_play_blocking)
            finally:
                self._is_playing = False
                self._current_rms = 0.0
                self._recent_rms.clear()
                self._last_play_time = time.time()

    def stop(self) -> None:
        """Immediately stop playback, invalidate active generation ID, and purge ALSA ring buffer."""
        self._generation_id += 1
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
