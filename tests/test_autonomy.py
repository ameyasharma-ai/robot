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
