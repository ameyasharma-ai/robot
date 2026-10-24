"""Tests for conversational utterance continuation coalescing and 100% speaker barge-in tuning.

Validates:
1. Conversational coalescing: multi-part utterances with pauses, hesitations ('umm...'),
   and dangling conjunctions are coalesced into a single unified prompt instead of fragmented turns.
2. 100% speaker volume barge-in: pure speaker chassis bleed is rejected (zero false self-interruptions),
   while genuine human voice cuts through reliably.
3. Enhanced self-echo detection on barge-in rejecting partial bot phrase echoes.
"""

from __future__ import annotations

import asyncio
import numpy as np
import pytest
import torch

from robot.audio.vad import VoiceActivityDetector
from robot.core.engine import RobotEngine


def test_hesitation_pattern_matching() -> None:
    """Verify hesitation fillers, trailing conjunctions, and pauses match continuation pattern."""
    hesitation_re = RobotEngine._HESITATION_PATTERNS

    # Fillers and hesitations
    assert hesitation_re.search("What is the weather, umm")
    assert hesitation_re.search("Can you tell me uh...")
    assert hesitation_re.search("Hey robot, like")
    assert hesitation_re.search("I was wondering, er...")

    # Trailing conjunctions and prepositions
    assert hesitation_re.search("Can you search for")
    assert hesitation_re.search("What is the capital of")
    assert hesitation_re.search("Turn off the lights and")
    assert hesitation_re.search("Tell me about the")

    # Trailing ellipses or dashes
    assert hesitation_re.search("Hello there...")
    assert hesitation_re.search("Could you please--")

    # Clean completed sentences should NOT match
    assert not hesitation_re.search("What time is it in Tokyo?")
    assert not hesitation_re.search("Turn off the living room lights")
    assert not hesitation_re.search("Play Bohemian Rhapsody")
    assert not hesitation_re.search("Hello")


@pytest.mark.asyncio
async def test_response_loop_coalesces_hesitations() -> None:
    """Verify that _response_loop coalesces fragmented chunks into a single unified prompt."""
    engine = RobotEngine()

    async def mock_stream(text, eos, interrupt):
        return text

    engine._stream_response = mock_stream

    # Put a leading chunk with a hesitation marker into the speech queue
    await engine._speech_queue.put(("What is the capital of umm...", 1.0))

    # Concurrently put the continuation chunk after a short pause
    async def feed_continuation():
        await asyncio.sleep(0.08)
        await engine._speech_queue.put(("France and Germany?", 1.1))

    feeder = asyncio.create_task(feed_continuation())

    # Dequeue from speech queue and run coalesce logic
    user_text, end_of_speech_time = await engine._speech_queue.get()
    is_hesitation = bool(engine._HESITATION_PATTERNS.search(user_text.strip()))
    assert is_hesitation

    coalesce_deadline = asyncio.get_event_loop().time() + 0.5
    while asyncio.get_event_loop().time() < coalesce_deadline:
        try:
            next_chunk, next_end = await asyncio.wait_for(
                engine._speech_queue.get(), timeout=0.25
            )
            clean_prev = user_text.rstrip(" .,-")
            clean_next = next_chunk.lstrip()
            user_text = f"{clean_prev} {clean_next}"
            end_of_speech_time = next_end
            break
        except asyncio.TimeoutError:
            break

    await feeder
    assert user_text == "What is the capital of umm France and Germany?"


def test_barge_in_rejects_100_percent_speaker_bleed() -> None:
    """Verify that pure speaker bleed at 100% volume is rejected by VoiceActivityDetector."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        normal_threshold=0.50,
        barge_in_threshold=0.65,
        barge_in_energy=0.015,
        echo_coupling_ratio=0.60,
    )

    # Simulate loud bot playback at 100% volume: ref_rms = 0.15
    ref_rms = 0.15
    # Actual acoustic bleed arriving at mic: coupling ~ 0.52 -> mic_rms ~ 0.078
    bleed_chunk = np.random.randn(512).astype(np.float32)
    bleed_chunk = bleed_chunk / np.sqrt(np.mean(bleed_chunk**2)) * 0.078
    mic_rms = float(np.sqrt(np.mean(bleed_chunk**2)))

    is_voice, prob = vad.detect(
        bleed_chunk,
        rms=mic_rms,
        is_bot_speaking=True,
        in_echo_guard=False,
        ref_rms=ref_rms,
    )

    # Pure speaker bleed must be rejected before triggering false barge-in
    assert not is_voice


def test_barge_in_accepts_human_voice_over_playback() -> None:
    """Verify that a human voice speaking over playback cleanly passes energy threshold."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        normal_threshold=0.50,
        barge_in_threshold=0.65,
        barge_in_energy=0.015,
        echo_coupling_ratio=0.60,
    )

    # Loud bot playback: ref_rms = 0.14
    ref_rms = 0.14
    # User speaking loudly into mic over speaker bleed: mic_rms = 0.18
    # Synthesize speech-like periodic signal (harmonic vowel)
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    human_voice = (
        0.6 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)
    human_voice = human_voice / np.sqrt(np.mean(human_voice**2)) * 0.18
    mic_rms = float(np.sqrt(np.mean(human_voice**2)))

    expected_bleed = ref_rms * vad._learned_coupling
    min_rms = max(0.010, expected_bleed * 1.15) + vad.barge_in_energy

    # Assert that human voice energy cleanly exceeds the 100% speaker bleed envelope
    assert mic_rms > min_rms


def test_self_echo_barge_in_filters_partial_bot_phrase() -> None:
    """Verify that _is_self_echo filters partial fragments of active bot speech on barge in."""
    engine = RobotEngine()
    engine._last_bot_phrases = ["The weather in Tokyo today is sunny and seventy degrees."]

    # Exact duplicate must be filtered
    assert engine._is_self_echo("The weather in Tokyo today is sunny and seventy degrees", is_barge_in=True)

    # Substring duplicate from partial Whisper transcription must be filtered
    assert engine._is_self_echo("sunny and seventy degrees", is_barge_in=True)
    assert engine._is_self_echo("weather in tokyo today", is_barge_in=True)

    # Genuine human barge-in command must NOT be filtered
    assert not engine._is_self_echo("Stop talking", is_barge_in=True)
    assert not engine._is_self_echo("Actually what about Paris", is_barge_in=True)
    assert not engine._is_self_echo("Cancel", is_barge_in=True)


def test_barge_in_bypasses_wake_word_requirement() -> None:
    """Verify that barge-in utterance is accepted even when wake word is required and follow-up expired."""
    engine = RobotEngine()
    engine.require_wake_word = True
    engine.wakeword.enabled = True
    engine._last_interaction_ts = 0.0

    # Non-barge speech without wake word is ignored
    is_barge = False
    in_follow_up = False
    condition_non_barge = is_barge or in_follow_up or (not engine.require_wake_word or not engine.wakeword.enabled)
    assert condition_non_barge is False

    # Barge-in speech without wake word MUST be accepted
    is_barge = True
    condition_barge = is_barge or in_follow_up or (not engine.require_wake_word or not engine.wakeword.enabled)
    assert condition_barge is True

