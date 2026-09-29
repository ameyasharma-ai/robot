"""Tests for VAD module."""

import numpy as np
import pytest


def test_vad_silent_audio():
    """VAD should not trigger on silence."""
    from robot.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(sample_rate=16000)
    silence = np.zeros(512, dtype=np.float32)
    is_voice, prob = vad.detect(silence, rms=0.001)
    assert not is_voice


def test_vad_high_energy_random():
    """VAD with loud random noise should have some probability."""
    from robot.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(sample_rate=16000)
    noise = np.random.randn(512).astype(np.float32) * 0.5
    rms = float(np.sqrt(np.mean(noise**2)))
    is_voice, prob = vad.detect(noise, rms=rms)
    # Prob will vary, but should be a valid float
    assert 0.0 <= prob <= 1.0
