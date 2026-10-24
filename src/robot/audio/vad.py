"""Silero VAD wrapper with dynamic thresholding for barge-in."""

from __future__ import annotations

import torch
from silero_vad import load_silero_vad

from robot.telemetry import get_logger

logger = get_logger("audio.vad")


class VoiceActivityDetector:
    """Silero VAD with context-aware confidence thresholds.

    When the robot is speaking, barge-in requires higher confidence
    and energy to avoid triggering on speaker bleed.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        normal_threshold: float = 0.50,
        barge_in_threshold: float = 0.55,
        echo_guard_threshold: float = 0.85,
        energy_threshold: float = 0.005,
        barge_in_energy: float = 0.008,
        echo_guard_energy: float = 0.02,
        echo_coupling_ratio: float = 0.40,
    ) -> None:
        """Initialize VoiceActivityDetector with adaptive echo suppression thresholds.

        Args:
            sample_rate: Audio sampling frequency in Hz (default: 16000).
            normal_threshold: Silero probability threshold during normal listening (default: 0.50).
            barge_in_threshold: Probability threshold required to barge in over playback (default: 0.55).
            echo_guard_threshold: Probability threshold during post-speech echo tail (default: 0.85).
            energy_threshold: Minimal RMS floor for ambient voice detection (default: 0.005).
            barge_in_energy: Excess RMS margin required above speaker bleed (default: 0.008).
            echo_guard_energy: RMS floor required during post-speech guard window (default: 0.02).
            echo_coupling_ratio: Baseline acoustic leakage ratio from laptop speakers to mic (default: 0.40).
        """
        self.sample_rate: int = sample_rate
        self.normal_threshold: float = normal_threshold
        self.barge_in_threshold: float = barge_in_threshold
        self.echo_guard_threshold: float = echo_guard_threshold
        self.energy_threshold: float = energy_threshold
        self.barge_in_energy: float = barge_in_energy
        self.echo_guard_energy: float = echo_guard_energy
        self.echo_coupling_ratio: float = echo_coupling_ratio
        self._learned_coupling: float = echo_coupling_ratio
        self._max_ref_rms: float = 0.0

        logger.info("loading_silero_vad")
        self._model = load_silero_vad()
        logger.info("silero_vad_loaded")

    def detect(
        self,
        chunk: torch.Tensor | __import__('numpy').ndarray,
        rms: float,
        is_bot_speaking: bool = False,
        in_echo_guard: bool = False,
        ref_rms: float = 0.0,
    ) -> tuple[bool, float]:
        """Run VAD on a single audio frame with dynamic reference echo subtraction.

        Tracks the acoustic playback envelope with a decay constant that spans
        the ~200ms audio output ring buffer and hardware DAC latency, dynamically
        adapting to physical speaker bleed on laptop chassis.

        Args:
            chunk: Audio frame as float32 mono array or torch.Tensor.
            rms: Pre-computed RMS energy of the zero-centered audio frame.
            is_bot_speaking: Whether the robot is currently playing TTS audio.
            in_echo_guard: Whether the pipeline is in the post-speech echo guard window.
            ref_rms: Real-time RMS of the TTS audio currently being written to speakers.

        Returns:
            Tuple of (is_voice, speech_probability).
        """
        # Slow decay (0.94 per 32ms frame) spans the 200ms PipeWire/ALSA latency
        self._max_ref_rms = max(ref_rms, self._max_ref_rms * 0.94)

        # Dynamic threshold selection based on duplex state
        if is_bot_speaking:
            threshold = self.barge_in_threshold
            # Acoustic speaker bleed scales directly with playback reference envelope.
            # At 100% speaker volume, laptop microphone captures direct chassis and airborne speaker bleed.
            # To allow barge-in, mic RMS must exceed the acoustic speaker bleed (with 15% safety margin)
            # plus an excess vocal energy threshold corresponding to human voice.
            expected_bleed = self._max_ref_rms * self._learned_coupling
            min_rms = max(0.008, expected_bleed * 1.10) + self.barge_in_energy
        elif in_echo_guard:
            threshold = self.echo_guard_threshold
            min_rms = self.echo_guard_energy
        else:
            threshold = self.normal_threshold
            min_rms = self.energy_threshold

        # Fast path: if microphone RMS doesn't exceed acoustic speaker bleed, reject
        if rms <= min_rms:
            return False, 0.0

        if not isinstance(chunk, torch.Tensor):
            chunk = torch.from_numpy(chunk)

        prob = float(self._model(chunk, self.sample_rate).item())
        is_voice = prob > threshold

        # Smoothly track physical acoustic chassis leakage during active playback when no human voice is detected
        if is_bot_speaking and self._max_ref_rms > 0.015 and prob < 0.35:
            instant_leakage = min(0.95, max(0.10, rms / self._max_ref_rms))
            # Exponential moving average (alpha=0.03) adapts to room acoustics and speaker volume
            self._learned_coupling = min(0.90, max(0.20, 0.97 * self._learned_coupling + 0.03 * instant_leakage))

        return is_voice, prob
