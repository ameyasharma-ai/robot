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
        echo_coupling_ratio: float = 0.20,
    ) -> None:
        self.sample_rate = sample_rate
        self.normal_threshold = normal_threshold
        self.barge_in_threshold = barge_in_threshold
        self.echo_guard_threshold = echo_guard_threshold
        self.energy_threshold = energy_threshold
        self.barge_in_energy = barge_in_energy
        self.echo_guard_energy = echo_guard_energy
        self.echo_coupling_ratio = echo_coupling_ratio
        self._learned_coupling = echo_coupling_ratio
        self._max_ref_rms = 0.0

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
        """Run VAD on a single audio frame with dynamic reference echo subtraction.

        Args:
            chunk: Audio frame (float32, mono)
            rms: Pre-computed RMS energy of the frame
            is_bot_speaking: Whether the robot is currently playing TTS audio
            in_echo_guard: Whether we're in the post-speech echo guard window
            ref_rms: Real-time RMS of the TTS audio currently being played through speakers

        Returns:
            (is_voice, speech_probability)
        """
        # Decay max reference envelope over time (~90ms decay)
        self._max_ref_rms = max(ref_rms, self._max_ref_rms * 0.70)

        # Dynamic threshold selection based on duplex state
        if is_bot_speaking:
            threshold = self.barge_in_threshold
            # Acoustic speaker bleed scales directly with playback reference envelope.
            # During playback, microphone captures (speaker_bleed + user_voice).
            # To allow barge-in, mic RMS must exceed the acoustic speaker bleed
            # plus a minimal excess energy threshold corresponding to speech.
            expected_bleed = self._max_ref_rms * self.echo_coupling_ratio
            min_rms = max(0.005, expected_bleed) + self.barge_in_energy
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

        prob = self._model(chunk, self.sample_rate).item()
        is_voice = prob > threshold
        return is_voice, prob
