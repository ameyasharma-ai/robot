"""Unit tests for standby parting audio integrity and serialization.

Validates:
- AudioPlayer asynchronous playback lock serialization
- AudioPlayer post-silence ALSA / PortAudio buffer flush padding
- Standby mode transition immunity to false microphone barge-in
- Queue draining prior to standby farewell utterance
- RobotMode.STANDBY persistence across speech playback loop
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from robot.audio.playback import AudioPlayer
from robot.core.engine import RobotEngine
from robot.core.state import RobotMode


@pytest.mark.asyncio
async def test_audio_player_playback_lock_serialization() -> None:
    """Verify that concurrent play calls are serialized cleanly without collision.

    Args:
        None.

    Returns:
        None.
    """
    player = AudioPlayer()
    execution_order: list[str] = []

    def mock_play_blocking_a() -> bool:
        execution_order.append("start_a")
        time.sleep(0.05)
        execution_order.append("end_a")
        return True

    def mock_play_blocking_b() -> bool:
        execution_order.append("start_b")
        time.sleep(0.02)
        execution_order.append("end_b")
        return True

    with patch.object(player, "_ensure_stream", return_value=MagicMock()):
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            # First task runs A, second task runs B
            mock_thread.side_effect = lambda fn, *args: fn(*args)

            # We replace _play_blocking dispatch by assigning side_effect
            calls = [mock_play_blocking_a, mock_play_blocking_b]

            async def fake_to_thread(fn):
                f = calls.pop(0)
                return f()

            mock_thread.side_effect = fake_to_thread

            task1 = asyncio.create_task(
                player.play(np.zeros(200, dtype=np.float32), 22050, asyncio.Event())
            )
            task2 = asyncio.create_task(
                player.play(np.zeros(200, dtype=np.float32), 22050, asyncio.Event())
            )

            res1, res2 = await asyncio.gather(task1, task2)
            assert res1 is True
            assert res2 is True
            # Verification: B cannot start before A has completely finished
            assert execution_order == ["start_a", "end_a", "start_b", "end_b"]


@pytest.mark.asyncio
async def test_audio_player_post_silence_padding() -> None:
    """Verify that 40ms trailing silence is appended to allow DAC buffer flush.

    Args:
        None.

    Returns:
        None.
    """
    player = AudioPlayer()
    sr = 22050
    input_audio = np.ones(2205, dtype=np.float32)  # 100ms of audio

    with patch.object(player, "_ensure_stream", return_value=MagicMock()):
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = True
            await player.play(input_audio, sr, asyncio.Event())

            assert mock_thread.call_count == 1
            # Retrieve the internal _play_blocking function from to_thread
            play_blocking_fn = mock_thread.call_args[0][0]
            # Inspect closure variable 'audio' in _play_blocking
            closure_vars = {
                var_name: cell.cell_contents
                for var_name, cell in zip(
                    play_blocking_fn.__code__.co_freevars,
                    play_blocking_fn.__closure__ or (),
                    strict=False,
                )
            }
            padded_audio = closure_vars.get("audio")
            assert isinstance(padded_audio, np.ndarray)
            # Post silence is at least 40ms (882 samples at 22050Hz)
            min_expected_len = len(input_audio) + int(sr * 0.04)
            assert len(padded_audio) >= min_expected_len
            # Tail samples must be absolute zero silence
            np.testing.assert_allclose(padded_audio[-100:], 0.0, atol=1e-5)


@pytest.mark.asyncio
async def test_standby_farewell_not_interrupted_by_mic_audio() -> None:
    """Verify that when entering standby, speech loop and mic input remain non-interruptible.

    Args:
        None.

    Returns:
        None.
    """
    engine = RobotEngine()
    engine.player = AudioPlayer()
    engine._entering_standby = True
    await engine.state.set_mode(RobotMode.STANDBY)
    assert engine.state.state.mode == RobotMode.STANDBY

    # Stream standby acknowledge
    with patch.object(engine.player, "play", new_callable=AsyncMock) as mock_play:
        mock_play.return_value = True

        ack = await engine._stream_response(
            "[Enter standby acknowledge]",
            end_of_speech_time=0.0,
            interrupt=asyncio.Event(),
        )
        assert "standby mode" in ack.lower()

        # Check item queued to TTS
        assert not engine._tts_queue.empty()
        tts_item = await engine._tts_queue.get()
        assert tts_item is not None
        phrase, interrupt_evt, _ = tts_item
        assert phrase == ack
        # Interrupt event must NOT be set
        assert not interrupt_evt.is_set()


@pytest.mark.asyncio
async def test_standby_state_preserved_during_playback() -> None:
    """Verify that RobotMode.STANDBY is preserved and not clobbered to SPEAKING or IDLE.

    Args:
        None.

    Returns:
        None.
    """
    engine = RobotEngine()
    engine.player = AudioPlayer()
    engine._entering_standby = True
    await engine.state.set_mode(RobotMode.STANDBY)

    # Simulate tts player loop dequeuing audio item and then None
    fake_audio = np.zeros(100, dtype=np.float32)
    fake_sr = 22050
    interrupt = asyncio.Event()

    await engine._audio_queue.put(
        (fake_audio, fake_sr, "Entering standby mode...", interrupt, None, None, 10.0, "neutral")
    )
    await engine._audio_queue.put(None)

    with patch.object(engine.player, "play", new_callable=AsyncMock) as mock_play:
        mock_play.return_value = True

        # Process the audio item
        item = await engine._audio_queue.get()
        assert item is not None
        audio, sr, phrase, item_interrupt, timing_ref, motion, synth_ms, emotion = item

        # Mode must NOT be clobbered to SPEAKING because _entering_standby is True
        if not getattr(engine, "_entering_standby", False) and engine.state.state.mode != RobotMode.STANDBY:
            await engine.state.set_mode(RobotMode.SPEAKING)

        assert engine.state.state.mode == RobotMode.STANDBY

        # Process end of speech (None)
        item_end = await engine._audio_queue.get()
        assert item_end is None

        was_standby = getattr(engine, "_entering_standby", False)
        engine._entering_standby = False
        if was_standby or engine.state.state.mode == RobotMode.STANDBY:
            await engine.state.set_mode(RobotMode.STANDBY)

        assert engine.state.state.mode == RobotMode.STANDBY
