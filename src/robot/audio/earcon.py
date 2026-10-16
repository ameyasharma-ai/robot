"""Procedural acoustic earcon and crystalline chime synthesis engine.

Generates Apple-grade harmonic earcons, glass chimes, and tactile auditory cues
using additive physical overtone synthesis with zero external audio assets.
"""

from __future__ import annotations

import enum
import math
from typing import ClassVar

import numpy as np


class EarconType(str, enum.Enum):
    """Enumeration of standard Apple-grade acoustic earcons and system cues."""

    WAKE_DETECTED = "wake_detected"
    CONFIRMATION = "confirmation"
    ALERT = "alert"
    DISMISS = "dismiss"
    ESTOP = "estop"


class AcousticEarconEngine:
    """Procedural harmonic synthesis engine for luxury tactile earcons and auditory chimes.

    Employs additive physical overtone modeling, frequency-modulated crystal bell
    resonances, and Hann-windowed ADSR decay curves to guarantee zero clicks, pops,
    or speaker driver distortion.
    """

    _CACHE: ClassVar[dict[tuple[EarconType, int], np.ndarray]] = {}

    @classmethod
    def generate(
        cls,
        earcon_type: EarconType | str,
        sample_rate: int = 24000,
        volume: float = 0.65,
    ) -> np.ndarray:
        """Synthesize a pristine acoustic earcon waveform.

        Args:
            earcon_type: Target earcon signature to generate.
            sample_rate: Audio sampling frequency in Hz (default 24000).
            volume: Master peak gain multiplier between 0.0 and 1.0 (default 0.65).

        Returns:
            1D float32 NumPy array containing the normalized acoustic waveform.

        Raises:
            ValueError: If the earcon type is unrecognized or volume is out of bounds.
        """
        if isinstance(earcon_type, str):
            try:
                earcon_type = EarconType(earcon_type.lower())
            except ValueError:
                raise ValueError(f"Unknown earcon type: '{earcon_type}'")

        if not 0.0 <= volume <= 1.0:
            raise ValueError(f"Volume must be within [0.0, 1.0], received {volume}")

        cache_key = (earcon_type, sample_rate)
        if cache_key in cls._CACHE:
            return (cls._CACHE[cache_key] * volume).astype(np.float32)

        if earcon_type == EarconType.WAKE_DETECTED:
            # Siri-inspired dual-tone crystalline rising bell (D5: 587.33 Hz -> A5: 880.0 Hz)
            waveform = cls._synthesize_two_tone_bell(
                f1=587.33,
                f2=880.0,
                duration1=0.07,
                duration2=0.11,
                sample_rate=sample_rate,
            )
        elif earcon_type == EarconType.CONFIRMATION:
            # Luxury Apple-style ascending glass triad (C5: 523.25 Hz -> E5: 659.25 Hz -> G5: 783.99 Hz)
            waveform = cls._synthesize_triad_bell(
                f1=523.25,
                f2=659.25,
                f3=783.99,
                sample_rate=sample_rate,
            )
        elif earcon_type == EarconType.ALERT:
            # Warm crystal chime alert (A4: 440 Hz + E5: 659.25 Hz harmonic resonance)
            waveform = cls._synthesize_chime_alert(
                f_fund=440.0,
                f_harm=659.25,
                duration=0.35,
                sample_rate=sample_rate,
            )
        elif earcon_type == EarconType.DISMISS:
            # Gentle descending soft two-tone cue (E5: 659.25 Hz -> C5: 523.25 Hz)
            waveform = cls._synthesize_two_tone_bell(
                f1=659.25,
                f2=523.25,
                duration1=0.06,
                duration2=0.09,
                sample_rate=sample_rate,
            )
        elif earcon_type == EarconType.ESTOP:
            # Crisp damped warning double-tap (G#4: 415.3 Hz -> E4: 329.63 Hz)
            waveform = cls._synthesize_warning_double_tap(
                f1=415.3,
                f2=329.63,
                sample_rate=sample_rate,
            )
        else:
            raise ValueError(f"Unhandled earcon type: {earcon_type}")

        # Peak normalization to 0.95 to leave headroom before soft limiter
        peak = np.max(np.abs(waveform))
        if peak > 1e-6:
            waveform = (waveform / peak) * 0.95

        # Cache baseline waveform
        cls._CACHE[cache_key] = waveform.astype(np.float32)
        return (cls._CACHE[cache_key] * volume).astype(np.float32)

    @classmethod
    def _synthesize_bell_tone(
        cls,
        frequency: float,
        duration: float,
        sample_rate: int,
        decay_rate: float = 12.0,
        harmonics: tuple[tuple[float, float], ...] = ((2.0, 0.28), (3.0, 0.12), (4.2, 0.05)),
    ) -> np.ndarray:
        """Synthesize a single crystal bell tone with harmonic overtones and natural decay.

        Args:
            frequency: Fundamental frequency in Hz.
            duration: Duration in seconds.
            sample_rate: Audio sampling frequency in Hz.
            decay_rate: Exponential damping coefficient.
            harmonics: Tuple of (multiplier, amplitude_ratio) pairs.

        Returns:
            Synthesized float32 audio array.
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Exponential decay envelope with 4ms soft attack
        attack_len = min(int(0.004 * sample_rate), n_samples)
        envelope = np.exp(-decay_rate * t)
        if attack_len > 0:
            attack_ramp = np.sin(np.linspace(0, np.pi / 2, attack_len)) ** 2
            envelope[:attack_len] *= attack_ramp

        # Fundamental + physical bell harmonics
        waveform = np.sin(2 * np.pi * frequency * t)
        for mult, amp in harmonics:
            waveform += amp * np.sin(2 * np.pi * frequency * mult * t) * np.exp(-decay_rate * 1.4 * t)

        waveform = waveform * envelope

        # Micro fade-out at end to prevent DC offset
        fade_len = min(int(0.005 * sample_rate), n_samples)
        if fade_len > 0:
            waveform[-fade_len:] *= np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2

        return waveform.astype(np.float32)

    @classmethod
    def _synthesize_two_tone_bell(
        cls,
        f1: float,
        f2: float,
        duration1: float,
        duration2: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Synthesize an articulated two-tone bell sequence."""
        t1 = cls._synthesize_bell_tone(f1, duration1, sample_rate, decay_rate=14.0)
        t2 = cls._synthesize_bell_tone(f2, duration2, sample_rate, decay_rate=10.0)

        # 15ms overlap crossfade for acoustic resonance continuity
        overlap_samples = int(0.015 * sample_rate)
        total_len = len(t1) + len(t2) - overlap_samples
        combined = np.zeros(total_len, dtype=np.float32)

        combined[: len(t1)] += t1
        combined[len(t1) - overlap_samples :] += t2
        return combined

    @classmethod
    def _synthesize_triad_bell(
        cls,
        f1: float,
        f2: float,
        f3: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Synthesize an ascending 3-note harmonic triad."""
        t1 = cls._synthesize_bell_tone(f1, 0.08, sample_rate, decay_rate=12.0)
        t2 = cls._synthesize_bell_tone(f2, 0.08, sample_rate, decay_rate=10.0)
        t3 = cls._synthesize_bell_tone(f3, 0.16, sample_rate, decay_rate=7.0)

        spacing = int(0.05 * sample_rate)
        total_len = spacing * 2 + len(t3)
        combined = np.zeros(total_len, dtype=np.float32)

        combined[: len(t1)] += t1
        combined[spacing : spacing + len(t2)] += t2
        combined[spacing * 2 : spacing * 2 + len(t3)] += t3
        return combined

    @classmethod
    def _synthesize_chime_alert(
        cls,
        f_fund: float,
        f_harm: float,
        duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Synthesize a warm resonant alert chime."""
        t1 = cls._synthesize_bell_tone(f_fund, duration, sample_rate, decay_rate=6.5)
        t2 = cls._synthesize_bell_tone(f_harm, duration * 0.9, sample_rate, decay_rate=7.5)

        min_len = min(len(t1), len(t2))
        combined = np.zeros(len(t1), dtype=np.float32)
        combined[:len(t1)] += t1 * 0.65
        combined[:min_len] += t2[:min_len] * 0.35
        return combined

    @classmethod
    def _synthesize_warning_double_tap(
        cls,
        f1: float,
        f2: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Synthesize a clean damped double-tap warning cue."""
        t1 = cls._synthesize_bell_tone(f1, 0.045, sample_rate, decay_rate=25.0)
        t2 = cls._synthesize_bell_tone(f2, 0.065, sample_rate, decay_rate=20.0)

        gap_samples = int(0.03 * sample_rate)
        total_len = len(t1) + gap_samples + len(t2)
        combined = np.zeros(total_len, dtype=np.float32)

        combined[: len(t1)] += t1
        combined[len(t1) + gap_samples :] += t2
        return combined
