"""Comprehensive unit test suite for dynamic sentiment analysis and prosody modulation."""

from __future__ import annotations

import pytest

from robot.tts import EmotionCategory, PiperTTS, ProsodyModulation, ProsodyModulator


def test_prosody_modulator_initialization() -> None:
    """Verify that ProsodyModulator initializes properly."""
    modulator = ProsodyModulator(enabled=True)
    assert modulator.enabled is True

    disabled_modulator = ProsodyModulator(enabled=False)
    assert disabled_modulator.enabled is False
    s, p, ns, nw, mod = disabled_modulator.modulate_prosody("Hello!", 1.0, 1.0)
    assert s == 1.0
    assert p == 1.0
    assert mod.emotion == "neutral"


def test_emotion_lexicon_classification() -> None:
    """Verify accurate emotional classification across lexical categories."""
    modulator = ProsodyModulator()

    # Happy / joyful
    cat, val, aro, conf = modulator.analyze_emotion("I am so delighted and happy with this wonderful outcome!")
    assert cat == EmotionCategory.HAPPY
    assert val > 0.5
    assert conf > 0.6

    # Excited / enthusiastic
    cat, val, aro, conf = modulator.analyze_emotion("This is absolutely incredible and epic! Let's go!!")
    assert cat == EmotionCategory.EXCITED
    assert aro > 0.6

    # Affectionate / tender
    cat, val, aro, conf = modulator.analyze_emotion("I truly adore you, sweetheart. You bring so much love and warmth.")
    assert cat == EmotionCategory.AFFECTIONATE
    assert val > 0.6

    # Sad / sorrowful
    cat, val, aro, conf = modulator.analyze_emotion("I am so terribly sorry for your loss. It is tragic and heartbreaking.")
    assert cat == EmotionCategory.SAD
    assert val < -0.5

    # Angry / urgent
    cat, val, aro, conf = modulator.analyze_emotion("Warning! Danger! Critical violation and emergency alert!")
    assert cat == EmotionCategory.ANGRY
    assert aro > 0.6

    # Surprised
    cat, val, aro, conf = modulator.analyze_emotion("Whoa! What?! That is completely unexpected and unbelievable!")
    assert cat in (EmotionCategory.SURPRISED, EmotionCategory.EXCITED)

    # Curious
    cat, val, aro, conf = modulator.analyze_emotion("I wonder why that happened? How does that work, I am curious.")
    assert cat == EmotionCategory.CURIOUS

    # Deadpan / clinical
    cat, val, aro, conf = modulator.analyze_emotion("Obviously. Your performance was predictably mediocre and pointless.")
    assert cat == EmotionCategory.DEADPAN
    assert val <= 0.0

    # Neutral
    cat, val, aro, conf = modulator.analyze_emotion("The current system time is 4:15 PM.")
    assert cat == EmotionCategory.NEUTRAL


def test_prosody_hard_limits_clamping() -> None:
    """Verify that pitch is strictly clamped to ±15% and speed is strictly clamped to ±10%."""
    modulator = ProsodyModulator()

    test_phrases = [
        "I am so wildly excited and thrilled!!",
        "I love you so much sweetheart...",
        "I am deeply sorry for your sad tragic loss.",
        "Warning! Critical alert! Danger!",
        "Obviously pathetic and completely pointless.",
        "Just a plain neutral sentence.",
    ]

    base_speed = 1.10
    base_pitch = 1.20

    for phrase in test_phrases:
        eff_speed, eff_pitch, eff_ns, eff_nw, modulation = modulator.modulate_prosody(
            phrase,
            base_speed=base_speed,
            base_pitch=base_pitch,
            base_noise_scale=0.667,
            base_noise_w=0.80,
        )

        # Pitch delta must strictly be within [-0.15, +0.15]
        assert -0.15 <= modulation.pitch_delta <= 0.15, f"Pitch delta {modulation.pitch_delta} exceeded ±15%"
        # Speed delta must strictly be within [-0.10, +0.10]
        assert -0.10 <= modulation.speed_delta <= 0.10, f"Speed delta {modulation.speed_delta} exceeded ±10%"

        # Resulting pitch must be within base_pitch * [0.85, 1.15]
        min_pitch = round(base_pitch * 0.85, 3) - 0.001
        max_pitch = round(base_pitch * 1.15, 3) + 0.001
        assert min_pitch <= eff_pitch <= max_pitch, f"eff_pitch {eff_pitch} outside [{min_pitch}, {max_pitch}]"

        # Resulting speed must be within base_speed * [0.90, 1.10]
        min_speed = round(base_speed * 0.90, 3) - 0.001
        max_speed = round(base_speed * 1.10, 3) + 0.001
        assert min_speed <= eff_speed <= max_speed, f"eff_speed {eff_speed} outside [{min_speed}, {max_speed}]"

        # Noise scales must remain positive and within bounds
        assert 0.05 <= eff_ns <= 1.20
        assert 0.10 <= eff_nw <= 1.20


def test_personality_prosody_bias() -> None:
    """Verify personality-specific behavioral bias (e.g. Luna vs GLaDOS)."""
    modulator = ProsodyModulator()

    # Ambiguous phrase with mild affection
    phrase = "It is nice seeing you today."

    # Under Luna, should lean affectionate
    cat_luna, _, _, _ = modulator.analyze_emotion(phrase, personality_name="luna")
    assert cat_luna in (EmotionCategory.AFFECTIONATE, EmotionCategory.HAPPY)

    # Under GLaDOS, excitement/affection pitch peaks are suppressed
    excited_phrase = "Amazing! Incredible result!"
    _, pitch_glados, _, _, mod_glados = modulator.modulate_prosody(
        excited_phrase,
        base_speed=1.0,
        base_pitch=1.0,
        personality_name="glados",
    )
    _, pitch_aria, _, _, mod_aria = modulator.modulate_prosody(
        excited_phrase,
        base_speed=1.0,
        base_pitch=1.0,
        personality_name="aria",
    )
    # GLaDOS should have a lower pitch shift than optimistic Aria
    assert pitch_glados <= pitch_aria


def test_inline_emotion_tag_extraction() -> None:
    """Verify that stage directions and inline emotion tags are captured."""
    modulator = ProsodyModulator()

    cat, _, _, conf = modulator.analyze_emotion("[emotion: happy] Welcome home!")
    assert cat == EmotionCategory.HAPPY
    assert conf >= 0.90

    cat_smile, _, _, _ = modulator.analyze_emotion("*smiles* It is great to see you.")
    assert cat_smile == EmotionCategory.HAPPY

    cat_sigh, _, _, _ = modulator.analyze_emotion("*sighs* I wish things were different.")
    assert cat_sigh == EmotionCategory.SAD


def test_piper_tts_prosody_integration() -> None:
    """Verify PiperTTS prosody modulation methods."""
    tts = PiperTTS(model_path="models/tts/en-us-lessac-medium.onnx")

    assert hasattr(tts, "prosody")
    assert isinstance(tts.prosody, ProsodyModulator)

    speed, pitch, ns, nw, mod = tts.modulate_prosody(
        "I am so thrilled and happy today!",
        base_speed=1.0,
        base_pitch=1.0,
    )
    assert mod.emotion in ("happy", "excited")
    assert pitch > 1.0
    assert speed >= 1.0
