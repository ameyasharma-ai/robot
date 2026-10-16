"""Offline Wake-Word Detector for hands-free robot activation.

Provides ultra-low overhead keyword spotting, acoustic energy pre-filtering,
and full state machine orchestration:
IDLE -> WAKE_DETECTED -> LISTENING -> RESPONDING -> STANDBY.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from robot.telemetry import get_logger

if TYPE_CHECKING:
    from robot.stt import WhisperServerSTT

logger = get_logger("audio.wakeword")


class WakeWordState(str, Enum):
    """Wake-word system states for voice cycle orchestration."""

    IDLE = "IDLE"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    RESPONDING = "RESPONDING"
    STANDBY = "STANDBY"


class WakeWordDetector:
    """Offline keyword and wake-phrase detector with state machine.

    Provides ultra-low overhead keyword spotting, acoustic energy gating,
    and voice loop state transitions. Runs locally (<5% CPU) preceding
    the STT and VAD pipelines.

    Attributes:
        wake_words: Normalized list of wake phrases.
        sample_rate: Audio sampling rate in Hz (default: 16000).
        confidence_threshold: Minimum match confidence score.
        cooldown_s: Minimum seconds between consecutive wake triggers.
        enabled: Whether wake-word requirement is active (falls back to direct VAD if False).
        current_state: Current WakeWordState of the audio cycle.
    """

    DEFAULT_WAKE_WORDS: list[str] = [
        "hey neo",
        "neo",
        "hey cypher",
        "cypher",
        "hey luna",
        "luna",
        "hey aria",
        "aria",
        "hey glados",
        "glados",
        "hey lisa",
        "lisa",
        "hey robot",
        "robot",
        "wake up lisa",
        "wake up",
        "hi lisa",
    ]

    PERSONA_WAKE_MAP: dict[str, list[str]] = {
        "neo": ["hey neo", "neo", "hey cypher", "cypher", "wake up"],
        "cypher": ["hey cypher", "cypher", "hey neo", "neo", "wake up"],
        "luna": ["hey luna", "luna", "wake up"],
        "aria": ["hey aria", "aria", "wake up"],
        "glados": ["hey glados", "glados", "wake up"],
    }

    def __init__(
        self,
        wake_words: list[str] | None = None,
        sample_rate: int = 16000,
        confidence_threshold: float = 0.5,
        cooldown_s: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Initialize the wake-word detector with target phrases and state machine.

        Args:
            wake_words: List of trigger phrases. If None, uses DEFAULT_WAKE_WORDS.
            sample_rate: Audio sampling rate (Hz).
            confidence_threshold: Minimum match confidence score.
            cooldown_s: Minimum cooldown in seconds between wake events.
            enabled: Initial enabled state for wake-word gating.
        """
        raw_words = wake_words or self.DEFAULT_WAKE_WORDS
        self.wake_words: list[str] = [w.lower().strip() for w in raw_words]
        self.sample_rate: int = sample_rate
        self.confidence_threshold: float = confidence_threshold
        self.cooldown_s: float = cooldown_s
        self.enabled: bool = enabled
        self._last_wake_time: float = 0.0
        self._current_state: WakeWordState = (
            WakeWordState.IDLE if enabled else WakeWordState.LISTENING
        )
        self._state_callbacks: list[Callable[[WakeWordState, WakeWordState], None]] = []

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
            enabled=self.enabled,
            initial_state=self._current_state.value,
        )

    @property
    def current_state(self) -> WakeWordState:
        """Return the current state of the wake-word state machine."""
        return self._current_state

    def on_state_change(
        self, callback: Callable[[WakeWordState, WakeWordState], None]
    ) -> None:
        """Register a callback invoked when the state machine transitions.

        Args:
            callback: Function taking (old_state, new_state).
        """
        self._state_callbacks.append(callback)

    def transition_to(self, new_state: WakeWordState, reason: str = "") -> None:
        """Transition the state machine to a new state and notify listeners.

        Args:
            new_state: Target state to transition to.
            reason: Optional context or explanation for telemetry logging.
        """
        old_state = self._current_state
        if old_state == new_state:
            return

        self._current_state = new_state
        logger.info(
            "wakeword_state_transition",
            old_state=old_state.value,
            new_state=new_state.value,
            reason=reason,
        )
        for cb in self._state_callbacks:
            try:
                cb(old_state, new_state)
            except Exception as e:
                logger.warning("wakeword_callback_error", error=str(e))

    def set_active_persona(self, persona: str) -> None:
        """Update active wake phrases to match the currently selected personality.

        Args:
            persona: Name of the persona (e.g. 'neo', 'luna', 'aria', 'glados').
        """
        key = persona.lower().strip()
        custom = self.PERSONA_WAKE_MAP.get(key)
        if custom:
            # Place specific persona wake words first, followed by general robot words
            combined = list(custom)
            for w in self.DEFAULT_WAKE_WORDS:
                if w not in combined:
                    combined.append(w)
            self.wake_words = combined
            logger.info("wakeword_persona_configured", persona=persona, wake_words=combined)

    def detect_energy_prefilter(
        self, chunk: np.ndarray, threshold: float = 0.005
    ) -> bool:
        """Perform low-overhead RMS energy check on audio chunk to avoid idle CPU usage.

        Args:
            chunk: Raw float32 or int16 audio array.
            threshold: Minimum RMS energy to qualify as potential speech.

        Returns:
            True if chunk energy exceeds threshold, False otherwise.
        """
        if len(chunk) == 0:
            return False
        # Calculate RMS energy quickly
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        return rms >= threshold

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

        if not self.enabled:
            # If wake word engine is disabled, pass speech directly
            return True, None, text.strip()

        now = time.time()
        if (now - self._last_wake_time) < self.cooldown_s:
            # During cooldown, do not re-trigger wake word
            return False, None, text

        cleaned = text.strip()
        lower = cleaned.lower()

        # Check for multi-word phrases first (e.g. "hey neo", "wake up")
        sorted_phrases = sorted(self.wake_words, key=lambda x: len(x), reverse=True)

        for phrase in sorted_phrases:
            pattern_start = (
                rf"^(?:hey|hi|hello|ok|yo)?\s*\b{re.escape(phrase)}\b[\s,:\.\?!]*"
            )
            m_start = re.search(pattern_start, lower)
            if m_start:
                remainder = cleaned[m_start.end() :].strip()
                remainder = re.sub(r"^[\s,:\.\?!]+", "", remainder).strip()
                self._last_wake_time = now
                self.transition_to(
                    WakeWordState.WAKE_DETECTED, reason=f"matched '{phrase}'"
                )
                logger.info("wakeword_detected", phrase=phrase, remainder=remainder)
                return True, phrase, remainder

            pattern_exact = rf"\b{re.escape(phrase)}\b"
            m_exact = re.search(pattern_exact, lower)
            if m_exact:
                remainder = (
                    cleaned[: m_exact.start()] + " " + cleaned[m_exact.end() :]
                ).strip()
                remainder = re.sub(r"^[\s,:\.\?!]+", "", remainder).strip()
                self._last_wake_time = now
                self.transition_to(
                    WakeWordState.WAKE_DETECTED, reason=f"matched '{phrase}'"
                )
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
        lower = re.sub(r"[^\w\s]", "", lower)

        for phrase in self.standby_phrases:
            if phrase in lower:
                self.transition_to(WakeWordState.STANDBY, reason="standby_phrase_match")
                return True

        if re.search(r"\b(?:stop|pause|cease)\s+(?:listening|voice|mic)\b", lower):
            self.transition_to(WakeWordState.STANDBY, reason="regex_stop_listening")
            return True
        if re.search(r"\b(?:go\s+to|enter)\s+standby\b", lower):
            self.transition_to(WakeWordState.STANDBY, reason="regex_enter_standby")
            return True
        if re.search(r"\b(?:go\s+to)\s+sleep\b", lower):
            self.transition_to(WakeWordState.STANDBY, reason="regex_go_to_sleep")
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
