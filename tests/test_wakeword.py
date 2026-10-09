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
