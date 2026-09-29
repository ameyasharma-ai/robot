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
        normal_threshold: float = 0.55,
        barge_in_threshold: float = 0.92,
        echo_guard_threshold: float = 0.85,
        energy_threshold: float = 0.008,
        barge_in_energy: float = 0.04,
        echo_guard_energy: float = 0.02,
    ) -> None:
        self.sample_rate = sample_rate
        self.normal_threshold = normal_threshold
        self.barge_in_threshold = barge_in_threshold
        self.echo_guard_threshold = echo_guard_threshold
        self.energy_threshold = energy_threshold
        self.barge_in_energy = barge_in_energy
        self.echo_guard_energy = echo_guard_energy

        logger.info("loading_silero_vad")
        self._model = load_silero_vad()
        logger.info("silero_vad_loaded")

    def detect(
        self,
        chunk: "torch.Tensor | __import__('numpy').ndarray",
        rms: float,
        is_bot_speaking: bool = False,
        in_echo_guard: bool = False,
        ref_rms: float = 0.0,
    ) -> tuple[bool, float]:
        """Run VAD on a single audio frame with reference echo subtraction.

        Args:
            chunk: Audio frame (float32, mono)
            rms: Pre-computed RMS energy of the frame
            is_bot_speaking: Whether the robot is currently playing TTS audio
            in_echo_guard: Whether we're in the post-speech echo guard window
            ref_rms: Real-time RMS of the TTS audio currently being played through speakers

        Returns:
            (is_voice, speech_probability)
        """
        # When bot is playing, subtract acoustic speaker bleed from mic energy
        # Laptop internal speaker to mic coupling is typically 0.10 - 0.25 (-12dB to -20dB)
        effective_rms = rms
        if is_bot_speaking and ref_rms > 0:
            estimated_leakage = ref_rms * 0.25
            effective_rms = max(0.0, rms - estimated_leakage)

        # Dynamic threshold selection based on duplex state
        if is_bot_speaking:
            threshold = self.barge_in_threshold
            min_rms = self.barge_in_energy
        elif in_echo_guard:
            threshold = self.echo_guard_threshold
            min_rms = self.echo_guard_energy
        else:
            threshold = self.normal_threshold
            min_rms = self.energy_threshold

        # Fast path: if effective energy does not cross min_rms, it's just speaker bleed / silence
        if effective_rms <= min_rms:
            return False, 0.0

        if not isinstance(chunk, torch.Tensor):
            chunk = torch.from_numpy(chunk)

        prob = self._model(chunk, self.sample_rate).item()
        is_voice = prob > threshold
        return is_voice, prob
