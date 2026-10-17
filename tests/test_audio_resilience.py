"""Unit tests for audio device resilience, fault isolation, and barge-in generation purging.

Validates:
- Explicit device health states (HEALTHY, DEGRADED, RECOVERING, ERROR)
- Monotonic generation tracking to purge stale speech synthesis on barge-in
- Bounded stream recovery with exponential backoff without infinite loops
- Fault isolation: capture failure does not crash playback, playback failure does not crash capture
- Non-blocking telemetry metrics collection for both capture and playback pipelines
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot.audio import (
    AudioCapture,
    AudioCaptureMetrics,
    AudioDeviceState,
    AudioPlaybackMetrics,
    AudioPlayer,
)


@pytest.mark.asyncio
async def test_audio_device_health_states() -> None:
    """Verify explicit audio device health state initialization and enum membership."""
    player = AudioPlayer()
    capture = AudioCapture()

    assert player.state == AudioDeviceState.HEALTHY
    assert capture.state == AudioDeviceState.HEALTHY
    assert AudioDeviceState.HEALTHY.value == "healthy"
    assert AudioDeviceState.DEGRADED.value == "degraded"
    assert AudioDeviceState.RECOVERING.value == "recovering"
    assert AudioDeviceState.ERROR.value == "error"


@pytest.mark.asyncio
async def test_audio_player_generation_tracking_and_stale_clause_purging() -> None:
    """Verify barge-in stop increments generation ID and immediately purges stale playback."""
    player = AudioPlayer()
    initial_gen = player.generation_id

    # Create generation 1
    gen1 = player.new_generation()
    assert gen1 == initial_gen + 1

    # Stop invalidates current generation by advancing generation_id
    player.stop()
    assert player.generation_id == gen1 + 1

    # Stale generation proposal must be rejected immediately without entering lock
    dummy_audio = np.zeros(100, dtype=np.float32)
    interrupt = asyncio.Event()

    result = await player.play(dummy_audio, 22050, interrupt, generation_id=gen1)
    assert result is False

    # Current matching generation can proceed
    current_gen = player.generation_id
    with patch.object(player, "_ensure_stream", return_value=MagicMock()):
        with patch("asyncio.to_thread", return_value=True):
            ok = await player.play(dummy_audio, 22050, interrupt, generation_id=current_gen)
            assert ok is True


@pytest.mark.asyncio
async def test_audio_player_bounded_recovery_without_infinite_loops() -> None:
    """Verify stream recovery bounds retries to max 3 attempts before entering DEGRADED state."""
    player = AudioPlayer()
    player._max_recovery_attempts = 3

    # Simulate PortAudio failure on stream creation
    with patch("sounddevice.OutputStream", side_effect=Exception("PortAudio device error: ALSA overrun")):
        stream = player._ensure_stream(22050)
        assert stream is None
        assert player.state == AudioDeviceState.DEGRADED
        assert player.metrics.errors >= 3
        assert player.metrics.stream_restarts >= 3

        # Subsequent play attempt must fail fast without crashing or hanging the event loop
        dummy_audio = np.ones(200, dtype=np.float32)
        interrupt = asyncio.Event()

        res = await player.play(dummy_audio, 22050, interrupt)
        assert res is False


@pytest.mark.asyncio
async def test_audio_playback_metrics_collection() -> None:
    """Verify non-blocking metrics update during playback execution."""
    player = AudioPlayer(block_ms=20)
    mock_stream = MagicMock()

    with patch.object(player, "_ensure_stream", return_value=mock_stream):
        dummy_audio = np.ones(882, dtype=np.float32)  # 20ms at 44100Hz is 882 samples
        interrupt = asyncio.Event()

        ok = await player.play(dummy_audio, 44100, interrupt)
        assert ok is True
        assert player.metrics.blocks_played > 0
        assert player.metrics.total_playback_time_s >= 0.0
        assert player.metrics.last_block_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_capture_and_playback_fault_isolation() -> None:
    """Verify that capture and playback failures remain strictly isolated from each other."""
    player = AudioPlayer()
    capture = AudioCapture()

    # 1. Simulate capture failure (microphone unplugged / ALSA error)
    capture._state = AudioDeviceState.DEGRADED
    capture.metrics.errors += 1

    # Playback must remain healthy and operational despite microphone failure
    assert player.state == AudioDeviceState.HEALTHY
    with patch.object(player, "_ensure_stream", return_value=MagicMock()):
        with patch("asyncio.to_thread", return_value=True):
            played = await player.play(np.zeros(100, dtype=np.float32), 22050, asyncio.Event())
            assert played is True

    # 2. Simulate playback failure (speaker unplugged / DAC error)
    player._state = AudioDeviceState.DEGRADED
    player.metrics.errors += 1

    # Capture must remain operational to process audio frames and VAD
    assert capture.state == AudioDeviceState.DEGRADED  # Capture has its own state
    clean, rms = capture.process_frame(np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32))
    assert len(clean) == 4
    assert rms > 0.0
    assert len(capture.get_pre_roll()) > 0


@pytest.mark.asyncio
async def test_audio_capture_metrics_and_recovery() -> None:
    """Verify capture metrics tracking and safe resumption."""
    capture = AudioCapture(sample_rate=16000)

    # Status handling in callback
    status_mock = MagicMock()
    status_mock.input_overflow = True
    status_mock.input_underflow = False

    cb = capture._make_callback(actual_rate=16000)
    cb(np.zeros((512, 1), dtype=np.float32), 512, None, status_mock)

    assert capture.metrics.frames_captured == 1
    assert capture.metrics.overflows == 1
    assert capture.metrics.dropouts == 0

    # Resume failure handling
    with patch.object(capture, "_open_stream", side_effect=Exception("ALSA busy")):
        capture.resume()
        assert capture.state == AudioDeviceState.DEGRADED
        assert capture.metrics.errors >= 1
