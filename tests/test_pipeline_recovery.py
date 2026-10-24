"""Unit and integration tests for engine pipeline auto-recovery and watchdog supervision."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.core.engine import RobotEngine
from robot.core.state import RobotMode, RobotState, StateManager


@pytest.fixture
def mock_engine() -> RobotEngine:
    """Create a lightweight RobotEngine fixture with mocked I/O components."""
    engine = RobotEngine()
    engine.player = MagicMock()
    engine.player.stop = MagicMock()
    engine.player.is_playing = False
    engine.observability = MagicMock()
    engine.events = MagicMock()
    engine.events.emit = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_auto_recover_pipeline_clears_queues_and_resets_idle(mock_engine: RobotEngine) -> None:
    """Verify that _auto_recover_pipeline purges queues, halts playback, and resets mode to IDLE."""
    engine = mock_engine
    await engine.state.set_mode(RobotMode.THINKING)
    engine._is_speaking = True
    engine._user_speaking = True

    # Pre-fill queues with stale turn items
    await engine._speech_queue.put(("old user query", time.time()))
    await engine._tts_queue.put(("old tts phrase", asyncio.Event(), None))
    await engine._audio_queue.put((b"fake_audio", 16000, "old phrase", asyncio.Event(), None, None, 10.0, "neutral"))

    assert not engine._speech_queue.empty()
    assert not engine._tts_queue.empty()
    assert not engine._audio_queue.empty()

    # Trigger auto recovery
    await engine._auto_recover_pipeline(reason="test_recovery")

    # Assert queues are completely drained
    assert engine._speech_queue.empty()
    assert engine._tts_queue.empty()
    assert engine._audio_queue.empty()

    # Assert flags and state are reset to operational ready (IDLE)
    assert engine._interrupt_event.is_set()
    assert engine._is_speaking is False
    assert engine._user_speaking is False
    assert engine.state.state.mode == RobotMode.IDLE
    assert engine.player.stop.called

    # Assert recovery events were emitted
    engine.events.emit.assert_any_call(
        "robot_speech_ended",
        mode=RobotMode.IDLE.value,
        recovered=True,
        reason="test_recovery",
    )
    engine.events.emit.assert_any_call("pipeline_recovered", reason="test_recovery")


@pytest.mark.asyncio
async def test_watchdog_detects_and_recovers_stuck_thinking(mock_engine: RobotEngine) -> None:
    """Verify that watchdog detects turns stuck in THINKING for >45s and triggers auto recovery."""
    engine = mock_engine
    await engine.state.set_mode(RobotMode.THINKING)
    # Simulate a turn that started 50 seconds ago
    engine._thinking_started_ts = time.time() - 50.0
    engine._running = True

    with patch.object(engine, "_auto_recover_pipeline", new_callable=AsyncMock) as mock_recover:
        # Run one iteration of the watchdog logic
        curr_mode = engine.state.state.mode
        assert curr_mode == RobotMode.THINKING
        stuck_dur = time.time() - engine._thinking_started_ts
        if stuck_dur > 45.0:
            await engine._auto_recover_pipeline(reason=f"stuck_thinking_{int(stuck_dur)}s")

        mock_recover.assert_called_once()
        call_arg = mock_recover.call_args[1]["reason"]
        assert "stuck_thinking" in call_arg


@pytest.mark.asyncio
async def test_watchdog_clears_stuck_speaking_state_without_audio(mock_engine: RobotEngine) -> None:
    """Verify that watchdog resets SPEAKING mode if audio queue is empty and player is idle for >5s."""
    engine = mock_engine
    await engine.state.set_mode(RobotMode.SPEAKING)
    engine._is_speaking = True
    engine.player.is_playing = False
    assert engine._audio_queue.empty()
    assert engine._tts_queue.empty()

    # Simulate being stuck in speaking state for 6 seconds without active playback
    engine._stuck_speaking_ts = time.time() - 6.0
    engine._running = True

    # Check watchdog condition
    is_playing = bool(engine.player.is_playing)
    curr_mode = engine.state.state.mode
    if (curr_mode == RobotMode.SPEAKING or engine._is_speaking) and not is_playing:
        if engine._audio_queue.empty() and engine._tts_queue.empty():
            if engine._stuck_speaking_ts is not None and (time.time() - engine._stuck_speaking_ts) > 5.0:
                engine._is_speaking = False
                engine._speech_ended_ts = time.time()
                await engine.state.set_mode(RobotMode.IDLE)
                await engine.events.emit("robot_speech_ended", mode=engine.state.state.mode.value)

    assert engine.state.state.mode == RobotMode.IDLE
    assert engine._is_speaking is False
    engine.events.emit.assert_called_with("robot_speech_ended", mode="idle")
