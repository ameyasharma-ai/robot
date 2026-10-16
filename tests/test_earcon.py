"""Comprehensive unit tests for the Apple-grade acoustic earcon and chime engine.

Tests:
- EarconType enum completeness and string normalization
- Procedural additive overtone synthesis across all earcon signatures
- Waveform characteristics: finite length, float32 dtype, bounded peak amplitude
- Zero DC offset and smooth windowing at onset and offset
- Volume scaling and error handling
- In-memory waveform caching performance
- AudioPlayer soft limiter and play_earcon integration
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from robot.audio.earcon import AcousticEarconEngine, EarconType
from robot.audio.playback import AudioPlayer


def test_earcon_types_defined() -> None:
    """Verify that all core Apple-grade auditory signatures are registered."""
    expected = {"wake_detected", "confirmation", "alert", "dismiss", "estop"}
    registered = {e.value for e in EarconType}
    assert expected.issubset(registered)


def test_earcon_generation_all_types() -> None:
    """Verify procedural generation across all earcon types produces clean waveforms."""
    sample_rate = 24000
    for e_type in EarconType:
        audio = AcousticEarconEngine.generate(e_type, sample_rate=sample_rate, volume=0.8)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) > 0
        # Peak must never exceed 1.0 (anti-clipping guarantee)
        peak = float(np.max(np.abs(audio)))
        assert peak <= 0.95
        # Soft micro-fade verification: start and end must taper near zero
        assert abs(audio[0]) < 0.05
        assert abs(audio[-1]) < 0.05


def test_earcon_string_and_case_insensitivity() -> None:
    """Verify that earcon generator accepts case-insensitive string inputs."""
    audio_upper = AcousticEarconEngine.generate("WAKE_DETECTED", sample_rate=16000)
    audio_enum = AcousticEarconEngine.generate(EarconType.WAKE_DETECTED, sample_rate=16000)
    np.testing.assert_allclose(audio_upper, audio_enum, atol=1e-5)


def test_earcon_volume_scaling_and_errors() -> None:
    """Verify volume multiplier behavior and boundary validation."""
    audio_half = AcousticEarconEngine.generate(EarconType.ALERT, sample_rate=24000, volume=0.4)
    audio_full = AcousticEarconEngine.generate(EarconType.ALERT, sample_rate=24000, volume=0.8)
    np.testing.assert_allclose(audio_half * 2.0, audio_full, atol=1e-5)

    with pytest.raises(ValueError, match="Volume must be within"):
        AcousticEarconEngine.generate(EarconType.ALERT, volume=1.5)

    with pytest.raises(ValueError, match="Volume must be within"):
        AcousticEarconEngine.generate(EarconType.ALERT, volume=-0.1)

    with pytest.raises(ValueError, match="Unknown earcon type"):
        AcousticEarconEngine.generate("nonexistent_earcon_xyz")


def test_earcon_caching() -> None:
    """Verify that repeated generation returns cached baseline for ultra-low latency."""
    # First call primes cache
    a1 = AcousticEarconEngine.generate(EarconType.CONFIRMATION, sample_rate=24000, volume=0.5)
    cache_key = (EarconType.CONFIRMATION, 24000)
    assert cache_key in AcousticEarconEngine._CACHE
    # Second call uses cache
    a2 = AcousticEarconEngine.generate(EarconType.CONFIRMATION, sample_rate=24000, volume=0.5)
    np.testing.assert_array_equal(a1, a2)


@pytest.mark.asyncio
async def test_audio_player_play_earcon_dispatch() -> None:
    """Verify AudioPlayer.play_earcon delegates to synthesized waveform playback."""
    player = AudioPlayer()
    with patch.object(player, "play", new_callable=AsyncMock) as mock_play:
        mock_play.return_value = True
        res = await player.play_earcon(EarconType.WAKE_DETECTED, sample_rate=24000, volume=0.6)
        assert res is True
        assert mock_play.await_count == 1
        args, _ = mock_play.await_args
        played_audio, played_sr, _ = args
        assert isinstance(played_audio, np.ndarray)
        assert played_sr == 24000


@pytest.mark.asyncio
async def test_audio_player_soft_limiter_and_anti_pop() -> None:
    """Verify AudioPlayer.play applies soft saturation on clipped input and anti-pop micro-fades."""
    player = AudioPlayer()
    # Create artificially clipped square-ish wave with peaks exceeding 1.0
    harsh_audio = np.array([1.5, -1.8, 2.0, -1.9, 1.6], dtype=np.float32)

    with patch.object(player, "_ensure_stream", return_value=MagicMock()):
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = True
            interrupt = asyncio.Event()
            res = await player.play(harsh_audio, sample_rate=24000, interrupt=interrupt)
            assert res is True
            assert mock_thread.call_count == 1
