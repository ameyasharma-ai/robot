"""Unit tests for CuriosityEngine and proactive idle autonomy."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot.core.autonomy import AutonomyConfig, CuriosityEngine
from robot.core.state import RobotMode, RobotState, StateManager


@pytest.mark.asyncio
async def test_curiosity_engine_initialization() -> None:
    """Test curiosity engine initialization and configuration."""
    mock_engine = MagicMock()
    config = AutonomyConfig(
        enabled=True,
        idle_threshold_s=5.0,
        cooldown_s=10.0,
        check_interval_s=0.1,
    )
    curiosity = CuriosityEngine(mock_engine, config)
    assert curiosity.config.idle_threshold_s == 5.0
    assert curiosity.proactive_count == 0
    assert curiosity._running is False


@pytest.mark.asyncio
async def test_curiosity_engine_activity_reset() -> None:
    """Test that notify_activity resets the idle timestamp."""
    mock_engine = MagicMock()
    curiosity = CuriosityEngine(mock_engine)
    old_ts = curiosity.last_activity_ts
    await asyncio.sleep(0.01)
    curiosity.notify_activity()
    assert curiosity.last_activity_ts > old_ts


@pytest.mark.asyncio
async def test_curiosity_engine_start_stop() -> None:
    """Test lifecycle start and stop."""
    mock_engine = MagicMock()
    config = AutonomyConfig(enabled=True, check_interval_s=0.05)
    curiosity = CuriosityEngine(mock_engine, config)

    curiosity.start()
    assert curiosity._running is True
    assert curiosity._task is not None

    curiosity.stop()
    assert curiosity._running is False


@pytest.mark.asyncio
async def test_curiosity_chat_mode_suppression() -> None:
    """Verify that chat mode or disabled voice completely suppresses proactive check-ins."""
    mock_engine = MagicMock()
    mock_engine.voice_enabled = False
    mock_engine.is_chat_mode = True
    mock_engine.state.state.mode = RobotMode.IDLE
    mock_engine._is_speaking = False
    mock_engine.player.is_playing = False

    config = AutonomyConfig(
        enabled=True,
        check_interval_s=0.01,
        quiet_after_user_activity_s=0.0,
        enable_static_timer=True,
        idle_threshold_s=0.0,
        cooldown_s=0.0,
    )
    curiosity = CuriosityEngine(mock_engine, config)
    curiosity._trigger_proactive_thought = AsyncMock()

    # Run one evaluation loop step
    curiosity.start()
    await asyncio.sleep(0.05)
    curiosity.stop()

    # Must NOT have triggered thought
    curiosity._trigger_proactive_thought.assert_not_called()


@pytest.mark.asyncio
async def test_curiosity_user_activity_quiet_window() -> None:
    """Verify that recent user activity suppresses proactive interruptions."""
    import time

    mock_engine = MagicMock()
    mock_engine.voice_enabled = True
    mock_engine.is_chat_mode = False
    mock_engine.state.state.mode = RobotMode.IDLE
    mock_engine._is_speaking = False
    mock_engine.player.is_playing = False

    config = AutonomyConfig(
        enabled=True,
        check_interval_s=0.01,
        quiet_after_user_activity_s=120.0,  # 2 minute quiet window
        enable_static_timer=True,
        idle_threshold_s=0.0,
        cooldown_s=0.0,
    )
    curiosity = CuriosityEngine(mock_engine, config)
    curiosity.last_activity_ts = time.time()  # Active just now
    curiosity._trigger_proactive_thought = AsyncMock()

    curiosity.start()
    await asyncio.sleep(0.05)
    curiosity.stop()

    curiosity._trigger_proactive_thought.assert_not_called()


@pytest.mark.asyncio
async def test_curiosity_persona_low_proactivity_suppression() -> None:
    """Verify that low proactivity personas like GLaDOS do not emit ambient chitchat."""
    mock_engine = MagicMock()
    mock_engine.voice_enabled = True
    mock_engine.is_chat_mode = False
    mock_engine.state.state.mode = RobotMode.IDLE
    mock_engine._is_speaking = False
    mock_engine.player.is_playing = False

    mock_pers = MagicMock()
    mock_pers.name = "glados"
    mock_pers.behavior = {"proactivity": "low"}
    mock_engine.personalities.active = mock_pers

    config = AutonomyConfig(
        enabled=True,
        check_interval_s=0.01,
        quiet_after_user_activity_s=0.0,
        enable_static_timer=True,
        idle_threshold_s=0.0,
        cooldown_s=0.0,
    )
    curiosity = CuriosityEngine(mock_engine, config)
    curiosity.last_activity_ts = 0.0
    curiosity._trigger_proactive_thought = AsyncMock()

    curiosity.start()
    await asyncio.sleep(0.05)
    curiosity.stop()

    curiosity._trigger_proactive_thought.assert_not_called()


@pytest.mark.asyncio
async def test_curiosity_visual_arrival_trigger() -> None:
    """Verify that visual arrival triggers a welcome greeting after absence."""
    import time

    mock_engine = MagicMock()
    mock_engine.voice_enabled = True
    mock_engine.is_chat_mode = False
    mock_engine.state.state.mode = RobotMode.IDLE
    mock_engine._is_speaking = False
    mock_engine.player.is_playing = False

    # Mock perception with a detected face
    mock_perception = MagicMock()
    mock_perception.get_latest_scene.return_value = {
        "num_faces": 1,
        "objects": [{"label": "person", "confidence": 0.95}],
    }
    mock_engine.perception = mock_perception

    config = AutonomyConfig(
        enabled=True,
        check_interval_s=0.01,
        quiet_after_user_activity_s=0.0,
        arrival_absence_threshold_s=10.0,
        arrival_debounce_s=0.01,
        enable_arrival_trigger=True,
        cooldown_s=0.0,
    )
    curiosity = CuriosityEngine(mock_engine, config)
    curiosity.last_activity_ts = 0.0

    # User was absent for 30s
    now = time.time()
    curiosity._was_present = False
    curiosity._absence_start_ts = now - 30.0
    curiosity._trigger_arrival_greeting = AsyncMock()

    # Step 1: candidate detected
    res1 = await curiosity._check_visual_arrival(now, None)
    assert res1 is False
    assert curiosity._arrival_candidate_ts is not None

    # Step 2: debounced after 0.02s
    res2 = await curiosity._check_visual_arrival(now + 0.05, None)
    assert res2 is True
    curiosity._trigger_arrival_greeting.assert_called_once()

