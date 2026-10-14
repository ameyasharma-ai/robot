"""Dynamic Sentiment Analysis and Expressive Prosody Modulation for Piper TTS.

Calculates real-time conversational valence, arousal, and discrete emotional states
from synthesized text, dynamically modulating Piper TTS parameters:
- Pitch modulation: ±15% (e.g. higher in joy/excitement, tender in affection, lower in melancholy/deadpan).
- Speed modulation: ±10% (e.g. lively/faster in excitement/urgency, gentle/slower in tenderness/sadness).
- Noise scale & noise_w: Inflection variance and phoneme duration adjustments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from robot.telemetry import get_logger

logger = get_logger("prosody")


class EmotionCategory(str, Enum):
    """Canonical emotion categories for conversational prosody and avatar expressions."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    AFFECTIONATE = "affectionate"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    DEADPAN = "deadpan"


@dataclass(frozen=True)
class ProsodyModulation:
    """Calculated prosody deltas and emotional telemetry.

    Attributes:
        emotion: Canonical emotion classification string.
        pitch_delta: Fractional pitch shift in [-0.15, +0.15] (±15%).
        speed_delta: Fractional speed change in [-0.10, +0.10] (±10%).
        noise_scale_delta: Additive shift applied to Piper noise_scale.
        noise_w_delta: Additive shift applied to Piper noise_w phoneme duration.
        valence: Sentiment polarity score from -1.0 (negative) to +1.0 (positive).
        arousal: Emotional energy/activation from -1.0 (calm/subdued) to +1.0 (high energy).
        confidence: Confidence score of the emotion detection from 0.0 to 1.0.
    """

    emotion: str = "neutral"
    pitch_delta: float = 0.0
    speed_delta: float = 0.0
    noise_scale_delta: float = 0.0
    noise_w_delta: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 1.0


class ProsodyModulator:
    """Evaluates conversational text sentiment and applies dynamic prosody scaling."""

    # Curated emotional lexicon for sub-millisecond deterministic sentiment scoring
    _AFFECTIONATE_WORDS: frozenset[str] = frozenset({
        "love", "adore", "sweetheart", "darling", "honey", "cherish", "dear",
        "sweet", "fond", "tender", "hug", "cuddle", "kiss", "warmth", "gently",
        "heart", "precious", "beloved", "caring", "comfort", "holding", "miss you",
    })

    _HAPPY_WORDS: frozenset[str] = frozenset({
        "happy", "glad", "joy", "joyful", "great", "wonderful", "delighted", "delight",
        "fantastic", "brilliant", "awesome", "radiant", "smile", "smiles", "smiling",
        "laugh", "laughs", "laughing", "giggle", "giggles", "cheerful", "pleasure",
        "yay", "hooray", "blessed", "splendid", "marvelous", "chuckle", "warm",
        "nice", "good", "pleased", "kind", "sweet", "enjoy", "lovely", "welcome",
    })

    _EXCITED_WORDS: frozenset[str] = frozenset({
        "excited", "exciting", "thrilled", "amazing", "incredible", "spectacular",
        "unbelievable", "super", "omg", "wow", "epic", "let's go", "eager",
        "phenomenal", "astonishing", "hyped", "pumped", "electrifying", "wild",
    })

    _SAD_WORDS: frozenset[str] = frozenset({
        "sad", "sorrow", "grief", "depressed", "unfortunate", "tragic", "crying",
        "tears", "apologies", "apologize", "sorry", "forgive", "regret", "painful",
        "hurts", "heartbroken", "loss", "mourn", "condolences", "melancholy",
        "downcast", "lonely", "hopeless", "sigh", "sighs", "alas",
    })

    _ANGRY_WORDS: frozenset[str] = frozenset({
        "angry", "mad", "furious", "hate", "irritated", "annoyed", "frustrated",
        "rage", "outraged", "terrible", "horrible", "unacceptable", "stop", "danger",
        "warning", "threat", "hazard", "hostile", "critical", "violation", "alert",
    })

    _SURPRISED_WORDS: frozenset[str] = frozenset({
        "surprised", "surprise", "shock", "shocked", "astonished", "whoa", "gasp",
        "unexpected", "unanticipated", "stunned", "bewildered", "what?", "really?",
        "no way", "impossible", "heavens",
    })

    _CURIOUS_WORDS: frozenset[str] = frozenset({
        "curious", "wonder", "wondering", "why", "how", "what if", "inquire",
        "investigate", "perhaps", "fascinating", "ponder", "pondering", "puzzle",
        "intriguing", "interesting", "could it be",
    })

    _DEADPAN_WORDS: frozenset[str] = frozenset({
        "obviously", "pathetic", "hardly", "disappointing", "predictable", "pointless",
        "tedious", "mundane", "test subject", "as expected", "mediocre", "futile",
        "clinical", "monotonous", "unimpressed", "inevitable",
    })

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the ProsodyModulator engine.

        Args:
            enabled: Master switch enabling or bypassing dynamic prosody calculation.
        """
        self.enabled = enabled

    def analyze_emotion(
        self,
        text: str,
        personality_name: str | None = None,
    ) -> tuple[EmotionCategory, float, float, float]:
        """Analyze conversational text to identify dominant emotion, valence, and arousal.

        Args:
            text: Raw or pre-sanitized text phrase.
            personality_name: Name of active personality (e.g. 'luna', 'glados', 'neo').

        Returns:
            Tuple of (EmotionCategory, valence [-1.0..1.0], arousal [-1.0..1.0], confidence [0.0..1.0]).
        """
        if not text or not text.strip():
            return EmotionCategory.NEUTRAL, 0.0, 0.0, 1.0

        raw_lower = text.lower()

        # 1. Check for explicit bracket/inline tags e.g. [emotion: happy], *smiles*, *sighs*
        tag_match = re.search(r"\[\s*emotion\s*:\s*([a-zA-Z_\-]+)\s*\]", raw_lower)
        if tag_match:
            tag_emo = tag_match.group(1).strip()
            for emo_enum in EmotionCategory:
                if emo_enum.value == tag_emo:
                    return emo_enum, 0.5, 0.5, 0.95

        if "*smiles*" in raw_lower or "*giggles*" in raw_lower or "*chuckles*" in raw_lower:
            return EmotionCategory.HAPPY, 0.7, 0.4, 0.90
        if "*sighs*" in raw_lower or "*cries*" in raw_lower:
            return EmotionCategory.SAD, -0.6, -0.3, 0.90
        if "*waves*" in raw_lower or "*leans in*" in raw_lower or "*blushes*" in raw_lower:
            return EmotionCategory.AFFECTIONATE, 0.8, 0.3, 0.90

        # 2. Tokenize words for lexicon matching
        clean_words = re.findall(r"\b[a-zA-Z']+\b", raw_lower)
        if not clean_words:
            return EmotionCategory.NEUTRAL, 0.0, 0.0, 1.0

        scores: dict[EmotionCategory, float] = {cat: 0.0 for cat in EmotionCategory}

        # Syntactic arousal indicators: exclamation marks and question marks
        exclamation_count = text.count("!")
        question_count = text.count("?")
        has_caps = any(w.isupper() and len(w) > 1 for w in text.split())

        base_arousal = 0.0
        if exclamation_count >= 2 or (exclamation_count >= 1 and has_caps):
            base_arousal += 0.4
            scores[EmotionCategory.EXCITED] += 1.5
        elif exclamation_count == 1:
            base_arousal += 0.2
            scores[EmotionCategory.HAPPY] += 0.8

        if question_count >= 1:
            scores[EmotionCategory.CURIOUS] += 1.2
            if question_count >= 2 or (question_count >= 1 and "!" in text):
                scores[EmotionCategory.SURPRISED] += 1.2

        # 3. Lexical matching against emotional word banks
        for word in clean_words:
            if word in self._AFFECTIONATE_WORDS:
                scores[EmotionCategory.AFFECTIONATE] += 2.0
            if word in self._HAPPY_WORDS:
                scores[EmotionCategory.HAPPY] += 1.5
            if word in self._EXCITED_WORDS:
                scores[EmotionCategory.EXCITED] += 1.8
            if word in self._SAD_WORDS:
                scores[EmotionCategory.SAD] += 2.0
            if word in self._ANGRY_WORDS:
                scores[EmotionCategory.ANGRY] += 2.0
            if word in self._SURPRISED_WORDS:
                scores[EmotionCategory.SURPRISED] += 1.5
            if word in self._CURIOUS_WORDS:
                scores[EmotionCategory.CURIOUS] += 1.2
            if word in self._DEADPAN_WORDS:
                scores[EmotionCategory.DEADPAN] += 1.5

        # 4. Personality-specific bias
        pers = (personality_name or "").lower()
        if "luna" in pers:
            # Luna naturally leans affectionate and tender
            scores[EmotionCategory.AFFECTIONATE] += 0.8
            scores[EmotionCategory.HAPPY] += 0.5
        elif "glados" in pers:
            # GLaDOS has a default deadpan sarcastic baseline
            scores[EmotionCategory.DEADPAN] += 1.0
            scores[EmotionCategory.AFFECTIONATE] = 0.0
            scores[EmotionCategory.EXCITED] *= 0.3
        elif "aria" in pers:
            # Aria is bright, helpful, optimistic
            scores[EmotionCategory.HAPPY] += 0.6

        best_category = EmotionCategory.NEUTRAL
        best_score = 0.0

        for cat, sc in scores.items():
            if cat != EmotionCategory.NEUTRAL and sc > best_score:
                best_score = sc
                best_category = cat

        # If score is too weak, remain neutral
        if best_score < 0.9:
            best_category = EmotionCategory.NEUTRAL
            confidence = 0.6
        else:
            confidence = min(1.0, 0.5 + (best_score / 4.0))

        # Valence & Arousal computation
        valence_map = {
            EmotionCategory.HAPPY: 0.75,
            EmotionCategory.EXCITED: 0.85,
            EmotionCategory.AFFECTIONATE: 0.90,
            EmotionCategory.SURPRISED: 0.30,
            EmotionCategory.CURIOUS: 0.20,
            EmotionCategory.NEUTRAL: 0.0,
            EmotionCategory.DEADPAN: -0.20,
            EmotionCategory.SAD: -0.75,
            EmotionCategory.ANGRY: -0.70,
        }
        arousal_map = {
            EmotionCategory.EXCITED: 0.85,
            EmotionCategory.ANGRY: 0.75,
            EmotionCategory.SURPRISED: 0.70,
            EmotionCategory.HAPPY: 0.50,
            EmotionCategory.CURIOUS: 0.30,
            EmotionCategory.AFFECTIONATE: 0.15,
            EmotionCategory.NEUTRAL: 0.0,
            EmotionCategory.DEADPAN: -0.35,
            EmotionCategory.SAD: -0.50,
        }

        valence = valence_map.get(best_category, 0.0)
        arousal = max(-1.0, min(1.0, arousal_map.get(best_category, 0.0) + base_arousal))

        return best_category, valence, arousal, confidence

    def calculate_modulation(
        self,
        emotion: EmotionCategory,
        valence: float,
        arousal: float,
        confidence: float,
        personality_name: str | None = None,
    ) -> ProsodyModulation:
        """Map emotional category, valence, and arousal into pitch, speed, and noise scale deltas.

        Enforces strict hard limits:
        - Pitch delta: exactly bounded within [-0.15, +0.15] (±15%).
        - Speed delta: exactly bounded within [-0.10, +0.10] (±10%).

        Args:
            emotion: Detected EmotionCategory.
            valence: Sentiment polarity score.
            arousal: Emotional energy score.
            confidence: Analysis confidence score.
            personality_name: Name of active personality.

        Returns:
            ProsodyModulation with calculated parameters.
        """
        pers = (personality_name or "").lower()

        # Base profiles per emotion
        if emotion == EmotionCategory.EXCITED:
            p_delta = 0.14
            s_delta = 0.08
            ns_delta = 0.12
            nw_delta = 0.10
        elif emotion == EmotionCategory.HAPPY:
            p_delta = 0.11
            s_delta = 0.05
            ns_delta = 0.10
            nw_delta = 0.06
        elif emotion == EmotionCategory.AFFECTIONATE:
            # Tender and loving: sweet higher pitch (+8%), gentle and slower speed (-6%), lingering vowels
            p_delta = 0.08
            s_delta = -0.06
            ns_delta = 0.06
            nw_delta = 0.14
        elif emotion == EmotionCategory.SAD:
            # Sorrowful/melancholy: lower pitch (-12%), slower tempo (-8%), subdued inflection
            p_delta = -0.12
            s_delta = -0.08
            ns_delta = -0.10
            nw_delta = 0.04
        elif emotion == EmotionCategory.ANGRY:
            # Firm/urgent: slightly deeper pitch (-6%), faster tempo (+8%), high inflection variance
            p_delta = -0.06
            s_delta = 0.08
            ns_delta = 0.12
            nw_delta = -0.06
        elif emotion == EmotionCategory.SURPRISED:
            p_delta = 0.14
            s_delta = 0.04
            ns_delta = 0.10
            nw_delta = 0.08
        elif emotion == EmotionCategory.CURIOUS:
            p_delta = 0.07
            s_delta = 0.02
            ns_delta = 0.05
            nw_delta = 0.04
        elif emotion == EmotionCategory.DEADPAN:
            # GLaDOS-like deadpan: lower flat pitch (-8%), slower/deliberate tempo (-5%), robotic inflection
            p_delta = -0.08
            s_delta = -0.05
            ns_delta = -0.16
            nw_delta = -0.08
        else:
            p_delta = 0.0
            s_delta = 0.0
            ns_delta = 0.0
            nw_delta = 0.0

        # GLaDOS constraints: prevent cheerful pitch peaks
        if "glados" in pers:
            p_delta = min(0.02, p_delta)
            s_delta = min(0.03, s_delta)

        # Scale by confidence factor
        eff_p = p_delta * confidence
        eff_s = s_delta * confidence
        eff_ns = ns_delta * confidence
        eff_nw = nw_delta * confidence

        # Strict clamping to specification: Pitch ±15%, Speed ±10%
        clamped_p = max(-0.15, min(0.15, eff_p))
        clamped_s = max(-0.10, min(0.10, eff_s))

        return ProsodyModulation(
            emotion=emotion.value,
            pitch_delta=round(clamped_p, 3),
            speed_delta=round(clamped_s, 3),
            noise_scale_delta=round(eff_ns, 3),
            noise_w_delta=round(eff_nw, 3),
            valence=round(valence, 2),
            arousal=round(arousal, 2),
            confidence=round(confidence, 2),
        )

    def modulate_prosody(
        self,
        text: str,
        base_speed: float = 1.0,
        base_pitch: float = 1.0,
        base_noise_scale: float | None = None,
        base_noise_w: float | None = None,
        personality_name: str | None = None,
    ) -> tuple[float, float, float | None, float | None, ProsodyModulation]:
        """Apply dynamic sentiment-driven prosody modulation to baseline Piper settings.

        Args:
            text: Conversational speech text to analyze.
            base_speed: Baseline speech speed configured for personality (e.g. 1.10).
            base_pitch: Baseline speech pitch configured for personality (e.g. 1.20).
            base_noise_scale: Baseline phoneme noise scale (or None for Piper default).
            base_noise_w: Baseline phoneme duration scale (or None for Piper default).
            personality_name: Name of active personality.

        Returns:
            Tuple of:
            - target_speed: Modulated speed within base_speed * [0.90..1.10]
            - target_pitch: Modulated pitch within base_pitch * [0.85..1.15]
            - target_noise_scale: Modulated noise scale clamped to [0.05..1.20]
            - target_noise_w: Modulated noise_w clamped to [0.10..1.20]
            - modulation: ProsodyModulation object containing emotion and telemetry
        """
        if not self.enabled:
            mod = ProsodyModulation()
            return base_speed, base_pitch, base_noise_scale, base_noise_w, mod

        emotion, valence, arousal, confidence = self.analyze_emotion(text, personality_name)
        modulation = self.calculate_modulation(emotion, valence, arousal, confidence, personality_name)

        # Apply pitch modulation (base * (1 + delta)), bounded to base ±15% and absolute [0.60, 1.60]
        min_p = base_pitch * 0.85
        max_p = base_pitch * 1.15
        calc_pitch = base_pitch * (1.0 + modulation.pitch_delta)
        target_pitch = max(min_p, min(max_p, calc_pitch))
        target_pitch = max(0.60, min(1.60, target_pitch))

        # Apply speed modulation (base * (1 + delta)), bounded to base ±10% and absolute [0.50, 1.80]
        min_s = base_speed * 0.90
        max_s = base_speed * 1.10
        calc_speed = base_speed * (1.0 + modulation.speed_delta)
        target_speed = max(min_s, min(max_s, calc_speed))
        target_speed = max(0.50, min(1.80, target_speed))

        # Apply noise scale & noise_w deltas if provided or default
        def_ns = base_noise_scale if base_noise_scale is not None else 0.667
        target_ns = max(0.05, min(1.20, def_ns + modulation.noise_scale_delta))

        def_nw = base_noise_w if base_noise_w is not None else 0.80
        target_nw = max(0.10, min(1.20, def_nw + modulation.noise_w_delta))

        logger.debug(
            "prosody_modulated",
            emotion=modulation.emotion,
            pitch=f"{base_pitch:.2f}->{target_pitch:.2f}",
            speed=f"{base_speed:.2f}->{target_speed:.2f}",
            noise_scale=f"{def_ns:.2f}->{target_ns:.2f}",
            noise_w=f"{def_nw:.2f}->{target_nw:.2f}",
        )

        return (
            round(target_speed, 3),
            round(target_pitch, 3),
            round(target_ns, 3),
            round(target_nw, 3),
            modulation,
        )
