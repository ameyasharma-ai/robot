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

# Tokens that carry no direct-address information on their own. Used to
# reject trailing bare-name matches like "so robot" / "that's so robot"
# where the name is just the last word of an ambient sentence.
_NON_ADDRESS_TOKENS: frozenset[str] = frozenset(
    {
        "um", "uh", "so", "well", "please", "and", "but", "or", "like", "just",
        "the", "a", "an", "this", "that", "thats", "that's", "it", "its", "is",
        "am", "are", "was", "were", "be", "been", "being", "very", "really",
        "kind", "sort", "type", "of", "to", "he", "she", "they", "i", "you",
        "we", "my", "me", "his", "her", "about", "such",
        # Clitic fragments produced when '\w+' splits apostrophes (that's, it's…)
        "s", "t", "d", "m", "ll", "ve", "re",
    }
)


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
        """Update active wake phrases and standby commands for the selected personality.

        Args:
            persona: Name of the persona (e.g. 'neo', 'cypher', 'luna', 'aria', 'glados').
        """
        key = persona.lower().strip()
        custom = self.PERSONA_WAKE_MAP.get(
            key, [f"hey {key}", key, f"wake up {key}", "wake up", f"hi {key}"]
        )
        combined = list(custom)
        # Always include universal robot wake words
        for w in ["hey robot", "robot", "wake up", "hey lisa", "lisa"]:
            if w not in combined:
                combined.append(w)
        self.wake_words = combined

        # Dynamically add persona-specific standby phrases
        persona_standbys = [
            f"stop {key}",
            f"quiet {key}",
            f"sleep {key}",
        ]
        if key in ("neo", "cypher"):
            persona_standbys.extend(["stop cypher", "stop neo"])
        for p in persona_standbys:
            if p not in self.standby_phrases:
                self.standby_phrases.insert(0, p)

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
        """Check if a transcribed phrase begins with or addresses the robot by a wake word.

        Supports prefix triggers, conversational fillers, trailing direct-address triggers,
        and multi-word invocation phrases while rejecting ambient mention of single words.

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

        cleaned = text.strip()
        lower = cleaned.lower()
        word_count = len(lower.split())

        # Sort phrases longest first so "hey cypher" matches before "cypher"
        sorted_phrases = sorted(self.wake_words, key=lambda x: len(x), reverse=True)

        # 1. Prefix match: start of utterance (with optional conversational fillers)
        for phrase in sorted_phrases:
            pattern_prefix = (
                rf"^(?:um|uh|so|well|please|yo|ok|okay)?[\s,:\.\?!]*(?:hey|hi|hello|ok|okay|yo)?[\s,:\.\?!]*\b{re.escape(phrase)}\b[\s,:\.\?!]*"
            )
            m_start = re.search(pattern_prefix, lower)
            if m_start:
                # Guard: the matched name must actually start near the top of the
                # utterance (first ~40% of the text, always allowing the short
                # single-filler prefix the regex permits).
                name_pos = m_start.group().find(phrase)
                if name_pos > 12 and name_pos > 0.4 * len(lower):
                    continue
                remainder = cleaned[m_start.end() :].strip()
                remainder = re.sub(r"^[\s,:\.\?!]+", "", remainder).strip()
                # A bare single-name wake ("robot") preceded only by a filler and
                # with no command after it ("so robot") is conversational tail
                # speech, not a direct-address wake invocation.
                if " " not in phrase and name_pos > 0 and not remainder:
                    continue
                self._last_wake_time = time.time()
                self.transition_to(
                    WakeWordState.WAKE_DETECTED, reason=f"matched '{phrase}'"
                )
                logger.info("wakeword_detected", phrase=phrase, remainder=remainder)
                return True, phrase, remainder

        # 2. Suffix address match: robot addressed at the end of sentence (e.g. "What time is it, Neo?")
        for phrase in sorted_phrases:
            if phrase in ("wake up", "wake up lisa", "wake up neo", "wake up cypher"):
                continue
            pattern_suffix = rf"[\s,]+(?:hey|hi|hello)?\s*\b{re.escape(phrase)}\b[\s\.\?!]*$"
            m_suffix = re.search(pattern_suffix, lower)
            if m_suffix:
                suffix_match = m_suffix.group()
                remainder = cleaned[: m_suffix.start()].strip()
                remainder = re.sub(r"[\s,:\.\?!]+$", "", remainder).strip()
                # A bare single name ending a sentence ("...so robot") is only a
                # direct address when punctuated (", robot") or greeted ("hey
                # robot"); otherwise require a short utterance whose remainder
                # still carries a substantive word beyond fillers/stopwords.
                has_greeting = bool(re.search(r"\b(?:hey|hi|hello)\b", suffix_match))
                has_comma = "," in suffix_match
                if " " not in phrase and not has_greeting and not has_comma:
                    rem_tokens = [
                        t
                        for t in re.findall(r"\w+", remainder.lower())
                        if t not in _NON_ADDRESS_TOKENS
                    ]
                    if word_count > 4 or not rem_tokens:
                        continue
                self._last_wake_time = time.time()
                self.transition_to(
                    WakeWordState.WAKE_DETECTED, reason=f"suffix matched '{phrase}'"
                )
                logger.info("wakeword_detected_suffix", phrase=phrase, remainder=remainder)
                return True, phrase, remainder

        # 3. Multi-word phrases inside the sentence (e.g. "hey neo", "wake up", "hey cypher")
        for phrase in sorted_phrases:
            if " " in phrase:
                pattern_multi = rf"\b{re.escape(phrase)}\b"
                m_multi = re.search(pattern_multi, lower)
                if m_multi:
                    remainder = (
                        cleaned[: m_multi.start()] + " " + cleaned[m_multi.end() :]
                    ).strip()
                    remainder = re.sub(r"\s+", " ", remainder).strip()
                    self._last_wake_time = time.time()
                    self.transition_to(
                        WakeWordState.WAKE_DETECTED, reason=f"multi-word matched '{phrase}'"
                    )
                    logger.info("wakeword_detected_multi", phrase=phrase, remainder=remainder)
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
        if re.search(r"\b(?:be\s+quiet|shut\s+up|silence|mute\s+(?:mic|microphone))\b", lower):
            self.transition_to(WakeWordState.STANDBY, reason="regex_quiet")
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
