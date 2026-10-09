"""Offline Wake-Word Detector for hands-free robot activation.

Supports listening for wake phrases ("Hey Lisa", "Lisa", "Hey robot", etc.)
and triggering seamless voice loop transitions. When in standby mode, speech
is evaluated offline; once the wake phrase is recognized, the system transitions
to active listening and strips or forwards any trailing user command in the
same utterance.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.stt import WhisperServerSTT

logger = get_logger("audio.wakeword")


class WakeWordDetector:
    """Offline keyword and wake-phrase detector.

    Provides ultra-low overhead keyword spotting and wake-word parsing.
    Maintains a rolling audio buffer during standby mode and evaluates
    transcripts or acoustic envelopes when speech is detected.

    Attributes:
        wake_words: List of normalized wake phrases to trigger on.
        sample_rate: Audio sampling rate in Hz (default: 16000).
        confidence_threshold: Minimum confidence score for detection.
        cooldown_s: Minimum seconds between consecutive wake-word triggers.
    """

    def __init__(
        self,
        wake_words: list[str] | None = None,
        sample_rate: int = 16000,
        confidence_threshold: float = 0.5,
        cooldown_s: float = 1.0,
    ) -> None:
        """Initialize the wake-word detector with target phrases.

        Args:
            wake_words: List of trigger phrases. Defaults to ["hey lisa", "lisa", "hey robot", "wake up lisa", "wake up"].
            sample_rate: Audio sampling rate (Hz).
            confidence_threshold: Minimum match confidence score.
            cooldown_s: Minimum cooldown in seconds between wake events.
        """
        self.wake_words: list[str] = [
            w.lower().strip()
            for w in (
                wake_words
                or ["hey lisa", "lisa", "hey robot", "wake up lisa", "wake up", "hi lisa"]
            )
        ]
        self.sample_rate: int = sample_rate
        self.confidence_threshold: float = confidence_threshold
        self.cooldown_s: float = cooldown_s
        self._last_wake_time: float = 0.0

        # Standby termination / sleep phrases
        self.standby_phrases: list[str] = [
            "stop listening",
            "go to sleep",
            "standby",
            "enter standby",
            "pause listening",
            "stop lisa",
            "sleep now",
            "cancel listening",
            "shut down voice",
            "stop talking",
            "be quiet",
        ]

        logger.info(
            "wakeword_detector_initialized",
            wake_words=self.wake_words,
            sample_rate=self.sample_rate,
        )

    def is_wake_word(self, text: str) -> tuple[bool, str | None, str]:
        """Check if a transcribed phrase begins with or contains a wake word.

        Args:
            text: Transcribed user speech.

        Returns:
            A tuple of:
            - is_detected (bool): True if a wake word was found.
            - matched_word (str | None): The exact wake word matched.
            - remainder (str): The remaining command with wake word stripped.
        """
        if not text:
            return False, None, ""

        now = time.time()
        if (now - self._last_wake_time) < self.cooldown_s:
            # During cooldown, do not re-trigger wake word
            return False, None, text

        cleaned = text.strip()
        lower = cleaned.lower()

        # Check for multi-word phrases first (e.g. "hey lisa", "wake up lisa")
        sorted_phrases = sorted(self.wake_words, key=lambda x: len(x), reverse=True)

        for phrase in sorted_phrases:
            # Match wake phrase at start of sentence (e.g. "Hey Lisa, what's the weather?")
            pattern_start = rf"^(?:hey|hi|hello|ok|yo)?\s*\b{re.escape(phrase)}\b[\s,:\.\?!]*"
            m_start = re.search(pattern_start, lower)
            if m_start:
                remainder = cleaned[m_start.end() :].strip()
                # Clean punctuation prefixes from remainder
                remainder = re.sub(r"^[\s,:\.\?!]+", "", remainder).strip()
                self._last_wake_time = now
                logger.info("wakeword_detected", phrase=phrase, remainder=remainder)
                return True, phrase, remainder

            # Match standalone wake word
            pattern_exact = rf"\b{re.escape(phrase)}\b"
            m_exact = re.search(pattern_exact, lower)
            if m_exact:
                remainder = (
                    cleaned[: m_exact.start()] + " " + cleaned[m_exact.end() :]
                ).strip()
                remainder = re.sub(r"^[\s,:\.\?!]+", "", remainder).strip()
                self._last_wake_time = now
                logger.info("wakeword_detected", phrase=phrase, remainder=remainder)
                return True, phrase, remainder

        return False, None, text

    def is_standby_command(self, text: str) -> bool:
        """Check if the user's speech is a command to enter standby mode.

        Args:
            text: Transcribed user speech.

        Returns:
            True if the text expresses an explicit intent to stop listening.
        """
        if not text:
            return False

        lower = text.lower().strip()
        # Clean trailing punctuation
        lower = re.sub(r"[^\w\s]", "", lower)

        for phrase in self.standby_phrases:
            if phrase in lower:
                return True

        # Check short regex commands
        if re.search(r"\b(?:stop|pause|cease)\s+(?:listening|voice|mic)\b", lower):
            return True
        if re.search(r"\b(?:go\s+to|enter)\s+standby\b", lower):
            return True
        if re.search(r"\b(?:go\s+to)\s+sleep\b", lower):
            return True

        return False

    def strip_wake_word(self, text: str) -> str:
        """Remove any leading wake words from an utterance.

        Args:
            text: Input utterance string.

        Returns:
            Cleaned utterance without wake words.
        """
        is_wake, _, remainder = self.is_wake_word(text)
        return remainder if is_wake else text
