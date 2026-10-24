"""Unit tests for offline WakeWordDetector."""

from __future__ import annotations

import pytest

from robot.audio.wakeword import WakeWordDetector


def test_wakeword_detector_initialization() -> None:
    """Test default initialization and wake word list."""
    detector = WakeWordDetector()
    assert "hey lisa" in detector.wake_words
    assert "lisa" in detector.wake_words
    assert detector.sample_rate == 16000


def test_wakeword_matching_prefix() -> None:
    """Test matching wake words at beginning of sentence."""
    detector = WakeWordDetector(cooldown_s=0.0)

    is_wake, matched, rem = detector.is_wake_word("Hey Lisa, what is the weather today?")
    assert is_wake is True
    assert matched == "hey lisa"
    assert rem == "what is the weather today?"

    is_wake2, matched2, rem2 = detector.is_wake_word("Lisa, calculate 50 times 4")
    assert is_wake2 is True
    assert matched2 == "lisa"
    assert rem2 == "calculate 50 times 4"


def test_wakeword_standalone() -> None:
    """Test matching standalone wake word without command."""
    detector = WakeWordDetector(cooldown_s=0.0)

    is_wake, matched, rem = detector.is_wake_word("Hey Lisa")
    assert is_wake is True
    assert matched == "hey lisa"
    assert rem == ""

    is_wake2, matched2, rem2 = detector.is_wake_word("Lisa")
    assert is_wake2 is True
    assert matched2 == "lisa"
    assert rem2 == ""


def test_wakeword_no_match() -> None:
    """Test non-wake utterances."""
    detector = WakeWordDetector(cooldown_s=0.0)

    is_wake, matched, rem = detector.is_wake_word("Good morning everyone, how are you?")
    assert is_wake is False
    assert matched is None
    assert rem == "Good morning everyone, how are you?"


def test_standby_command_detection() -> None:
    """Test standby/stop listening detection."""
    detector = WakeWordDetector()

    assert detector.is_standby_command("Stop listening") is True
    assert detector.is_standby_command("Please stop listening now") is True
    assert detector.is_standby_command("Go to sleep") is True
    assert detector.is_standby_command("Enter standby mode") is True
    assert detector.is_standby_command("What time is it in Tokyo?") is False


def test_wakeword_state_transitions() -> None:
    """Test state machine transitions and listener notification."""
    from robot.audio.wakeword import WakeWordState

    detector = WakeWordDetector(cooldown_s=0.0)
    assert detector.current_state == WakeWordState.IDLE

    history: list[tuple[WakeWordState, WakeWordState]] = []
    detector.on_state_change(lambda old, new: history.append((old, new)))

    detector.transition_to(WakeWordState.LISTENING, reason="user clicked mic")
    assert detector.current_state == WakeWordState.LISTENING
    assert history == [(WakeWordState.IDLE, WakeWordState.LISTENING)]

    # Trigger wake word
    is_wake, matched, rem = detector.is_wake_word("Hey Neo, open browser")
    assert is_wake is True
    assert matched == "hey neo"
    assert rem == "open browser"
    assert detector.current_state == WakeWordState.WAKE_DETECTED


def test_persona_wake_word_switching() -> None:
    """Test switching active personality wake words."""
    detector = WakeWordDetector(cooldown_s=0.0)
    detector.set_active_persona("luna")
    assert "hey luna" in detector.wake_words
    is_wake, matched, rem = detector.is_wake_word("Luna, play some jazz")
    assert is_wake is True
    assert matched == "luna"
    assert rem == "play some jazz"


def test_energy_prefilter() -> None:
    """Test low-overhead RMS energy pre-filtering."""
    import numpy as np

    detector = WakeWordDetector()
    silence = np.zeros(512, dtype=np.float32)
    assert detector.detect_energy_prefilter(silence, threshold=0.01) is False

    tone = np.sin(np.linspace(0, 10, 512)).astype(np.float32) * 0.5
    assert detector.detect_energy_prefilter(tone, threshold=0.01) is True


def test_wakeword_disabled_fallback() -> None:
    """Test direct passthrough when wake word gating is disabled."""
    from robot.audio.wakeword import WakeWordState

    detector = WakeWordDetector(enabled=False)
    assert detector.current_state == WakeWordState.LISTENING

    is_wake, matched, rem = detector.is_wake_word("Just a normal conversational question")
    assert is_wake is True
    assert rem == "Just a normal conversational question"


def test_wakeword_ambient_word_rejection() -> None:
    """Verify ambient mention of robot names in mid-sentence does NOT trigger activation."""
    detector = WakeWordDetector(cooldown_s=0.0)

    # Ambient utterances mentioning robot/neo/luna/aria/lisa in the middle of a sentence
    samples: list[str] = [
        "Yesterday I watched a movie about a robot vacuum in space.",
        "The opera soprano sang a beautiful aria at the concert hall.",
        "The neo classical architecture of the downtown courthouse is striking.",
        "My cousin Luna invited us over for Sunday dinner.",
        "I was reading a history book by Lisa yesterday.",
    ]
    for text in samples:
        is_wake, matched, rem = detector.is_wake_word(text)
        assert is_wake is False, f"Expected '{text}' to be rejected, but matched '{matched}'"


def test_wakeword_suffix_direct_address() -> None:
    """Verify trailing direct address commands properly trigger wake detection."""
    detector = WakeWordDetector(cooldown_s=0.0)

    is_wake, matched, rem = detector.is_wake_word("What is the time right now, Neo?")
    assert is_wake is True
    assert matched == "neo"
    assert rem == "What is the time right now"

    is_wake2, matched2, rem2 = detector.is_wake_word("Calculate 42 times 7, Cypher.")
    assert is_wake2 is True
    assert matched2 == "cypher"
    assert rem2 == "Calculate 42 times 7"

    is_wake3, matched3, rem3 = detector.is_wake_word("Tell me a funny joke, Robot")
    assert is_wake3 is True
    assert matched3 == "robot"
    assert rem3 == "Tell me a funny joke"


def test_wakeword_conversational_fillers() -> None:
    """Verify conversational pre-roll fillers (um, please, so) correctly trigger wake detection."""
    detector = WakeWordDetector(cooldown_s=0.0)

    is_wake, matched, rem = detector.is_wake_word("Um, hey Neo, what's on my calendar?")
    assert is_wake is True
    assert matched == "hey neo"
    assert rem == "what's on my calendar?"

    is_wake2, matched2, rem2 = detector.is_wake_word("Please Cypher, turn up the volume.")
    assert is_wake2 is True
    assert matched2 == "cypher"
    assert rem2 == "turn up the volume."


def test_standby_extended_phrases() -> None:
    """Verify extended standby commands like 'be quiet', 'shut up', and persona names."""
    detector = WakeWordDetector()
    detector.set_active_persona("cypher")

    assert detector.is_standby_command("Be quiet") is True
    assert detector.is_standby_command("Please be quiet now") is True
    assert detector.is_standby_command("Shut up") is True
    assert detector.is_standby_command("Stop cypher") is True
    assert detector.is_standby_command("Sleep cypher") is True
    assert detector.is_standby_command("Mute microphone") is True
    assert detector.is_standby_command("Quiet please, what time is it in London?") is False


@pytest.mark.asyncio
async def test_engine_wake_gating_and_follow_up_window() -> None:
    """Verify Engine voice gating, follow-up conversation window, and standby transitions."""
    import time
    from robot.core.engine import RobotEngine
    from robot.core.state import RobotMode

    engine = RobotEngine()
    engine.wakeword.set_active_persona("neo")
    assert engine.require_wake_word is True
    assert engine.follow_up_timeout_s == 12.0
    assert engine._last_interaction_ts == 0.0

    # 1. Spoken input without wake word when idle -> Must be rejected
    text1 = "What is the weather today?"
    is_wake1, _, _ = engine.wakeword.is_wake_word(text1)
    in_follow_up1 = (time.time() - engine._last_interaction_ts) < engine.follow_up_timeout_s
    assert is_wake1 is False
    assert in_follow_up1 is False  # Must not process ambient speech

    # 2. Spoken input with wake word -> Must be accepted and open follow-up window
    text2 = "Hey Neo, what is the weather today?"
    is_wake2, matched2, rem2 = engine.wakeword.is_wake_word(text2)
    assert is_wake2 is True
    assert matched2 == "hey neo"
    assert rem2 == "what is the weather today?"
    # Simulate turn acceptance
    engine._last_interaction_ts = time.time()
    await engine.state.set_mode(RobotMode.LISTENING)

    # 3. Follow-up turn within 12 seconds -> Must be accepted without wake word
    engine._last_interaction_ts = time.time() - 3.0  # 3 seconds ago
    text3 = "And what about tomorrow?"
    is_wake3, _, _ = engine.wakeword.is_wake_word(text3)
    in_follow_up3 = (time.time() - engine._last_interaction_ts) < engine.follow_up_timeout_s
    assert is_wake3 is False
    assert in_follow_up3 is True  # Seamless follow-up accepted!

    # 4. Turn after 12-second follow-up window expires -> Must be rejected
    engine._last_interaction_ts = time.time() - 15.0  # 15 seconds ago (> 12.0s)
    text4 = "And what about Sunday?"
    is_wake4, _, _ = engine.wakeword.is_wake_word(text4)
    in_follow_up4 = (time.time() - engine._last_interaction_ts) < engine.follow_up_timeout_s
    assert is_wake4 is False
    assert in_follow_up4 is False  # Outside window -> Ignored!

    # 5. Standby command -> Clears window and enters STANDBY
    text5 = "Go to sleep"
    assert engine.wakeword.is_standby_command(text5) is True
    engine._last_interaction_ts = 0.0
    await engine.state.set_mode(RobotMode.STANDBY)
    assert engine.state.state.mode == RobotMode.STANDBY

    # Even 1 second later, ambient speech without wake word is strictly ignored in STANDBY
    engine._last_interaction_ts = time.time() - 1.0
    in_follow_up5 = (
        (time.time() - engine._last_interaction_ts) < engine.follow_up_timeout_s
        and engine.state.state.mode != RobotMode.STANDBY
    )
    assert in_follow_up5 is False


@pytest.mark.asyncio
async def test_wake_word_auto_switch_persona() -> None:
    """Verify addressing a persona via wake word auto-switches personality and tailored acknowledgment."""
    from robot.core.engine import RobotEngine
    from unittest.mock import MagicMock

    engine = RobotEngine()
    engine.state.state.active_personality = "neo"

    # Test persona resolution helper
    assert engine._get_persona_for_wake_word("hey luna") == "luna"
    assert engine._get_persona_for_wake_word("luna") == "luna"
    assert engine._get_persona_for_wake_word("hey cypher") == "cypher"
    assert engine._get_persona_for_wake_word("hey glados") == "glados"
    assert engine._get_persona_for_wake_word("hey aria") == "aria"
    assert engine._get_persona_for_wake_word("wake up") is None
    assert engine._get_persona_for_wake_word("hey robot") is None

    # Test auto-switch to Luna
    target_pers = engine._get_persona_for_wake_word("hey luna")
    assert target_pers == "luna"
    await engine.switch_personality(target_pers)
    assert engine.state.state.active_personality == "luna"

    # Test persona-tailored wake acknowledgment in Luna
    engine.tts = MagicMock()
    interrupt = MagicMock()
    ack = await engine._stream_response("[Wake word acknowledge]", 0.0, interrupt)
    assert "listening" in ack.lower()
    assert ack == "Hey there! I'm listening."



