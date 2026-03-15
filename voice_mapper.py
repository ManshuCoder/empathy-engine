"""
voice_mapper.py - Emotion → Voice Parameter Mapping for The Empathy Engine

Maps a detected emotion (+ confidence score) to ElevenLabs voice settings.

Design:
  • Each emotion has a BASE parameter set (full-confidence values).
  • Parameters are BLENDED between neutral and the target emotion proportional
    to the confidence score (intensity scaling):
        param = neutral + confidence × (target − neutral)
  • Values are clamped to safe ranges defined in config.py.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass

import config
from emotion_detector import EmotionResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Emotion → Voice Parameter Table
# ─────────────────────────────────────────────
#
# pitch          – semitone offset (float)
# speaking_rate  – speed multiplier (1.0 = normal)
# volume         – amplitude multiplier (1.0 = normal)
# tone           – descriptive label for SSML / logs
# stability      – ElevenLabs: 0 = expressive, 1 = stable
# similarity_boost – ElevenLabs: voice clarity
# style          – ElevenLabs: style exaggeration (0-1)
#
EMOTION_VOICE_MAP: dict[str, dict] = {
    "happy": {
        "pitch":            3.0,
        "speaking_rate":    1.25,
        "volume":           1.10,
        "tone":             "cheerful",
        "stability":        0.30,
        "similarity_boost": 0.75,
        "style":            0.40,
        "description": (
            "Higher pitch, faster rate, brighter tone — conveys joy and excitement."
        ),
    },
    "sad": {
        "pitch":            -2.5,
        "speaking_rate":    0.80,
        "volume":           0.85,
        "tone":             "melancholic",
        "stability":        0.70,
        "similarity_boost": 0.70,
        "style":            0.10,
        "description": (
            "Lower pitch, slower rate, softer volume — conveys sorrow and grief."
        ),
    },
    "angry": {
        "pitch":            4.0,
        "speaking_rate":    1.15,
        "volume":           1.25,
        "tone":             "sharp",
        "stability":        0.20,
        "similarity_boost": 0.80,
        "style":            0.60,
        "description": (
            "Raised pitch, louder volume, assertive tone — conveys frustration and anger."
        ),
    },
    "concerned": {
        "pitch":            -1.0,
        "speaking_rate":    0.90,
        "volume":           0.92,
        "tone":             "worried",
        "stability":        0.55,
        "similarity_boost": 0.72,
        "style":            0.20,
        "description": (
            "Slightly lower pitch, slower rate, softer tone — conveys worry and care."
        ),
    },
    "neutral": {
        "pitch":            0.0,
        "speaking_rate":    1.00,
        "volume":           1.00,
        "tone":             "neutral",
        "stability":        0.50,
        "similarity_boost": 0.75,
        "style":            0.00,
        "description": (
            "Flat, clear delivery — no emotional colouring."
        ),
    },
}


# ─────────────────────────────────────────────
# Data Schema
# ─────────────────────────────────────────────

@dataclass
class VoiceParameters:
    """Concrete voice parameters passed to the TTS engine."""
    emotion:          str
    pitch:            float
    speaking_rate:    float
    volume:           float
    tone:             str
    stability:        float
    similarity_boost: float
    style:            float
    confidence:       float
    description:      str = ""

    def to_dict(self) -> dict:
        return {
            "emotion":          self.emotion,
            "pitch":            round(self.pitch, 3),
            "speaking_rate":    round(self.speaking_rate, 3),
            "volume":           round(self.volume, 3),
            "tone":             self.tone,
            "stability":        round(self.stability, 3),
            "similarity_boost": round(self.similarity_boost, 3),
            "style":            round(self.style, 3),
            "confidence":       round(self.confidence, 4),
            "description":      self.description,
        }


# ─────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────

class VoiceMapper:
    """
    Maps detected emotion + confidence → VoiceParameters.

    Intensity scaling formula:
        param = neutral_value + confidence × (target_value − neutral_value)
    """

    def __init__(self) -> None:
        self._neutral = EMOTION_VOICE_MAP["neutral"]
        logger.info("VoiceMapper initialised.")

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _blend(self, neutral_val: float, target_val: float, intensity: float) -> float:
        """Linear interpolation between neutral and target, capped at 0.95."""
        capped = min(intensity, 0.95)
        return neutral_val + capped * (target_val - neutral_val)

    def map(self, emotion_result: EmotionResult) -> VoiceParameters:
        """
        Convert an EmotionResult into VoiceParameters.

        Args:
            emotion_result: Output from EmotionDetector.detect()

        Returns:
            VoiceParameters ready to be passed to the TTS engine.
        """
        emotion    = emotion_result.emotion
        confidence = emotion_result.confidence

        if emotion not in EMOTION_VOICE_MAP:
            logger.warning(
                f"Emotion '{emotion}' not in map; falling back to "
                f"'{config.DEFAULT_EMOTION}'."
            )
            emotion = config.DEFAULT_EMOTION

        target  = deepcopy(EMOTION_VOICE_MAP[emotion])
        neutral = self._neutral

        # Blend numeric parameters
        pitch            = self._blend(neutral["pitch"],            target["pitch"],            confidence)
        speaking_rate    = self._blend(neutral["speaking_rate"],    target["speaking_rate"],    confidence)
        volume           = self._blend(neutral["volume"],           target["volume"],           confidence)
        stability        = self._blend(neutral["stability"],        target["stability"],        confidence)
        similarity_boost = self._blend(neutral["similarity_boost"], target["similarity_boost"], confidence)
        style            = self._blend(neutral["style"],            target["style"],            confidence)

        # Clamp to safe ranges
        pitch            = self._clamp(pitch,         config.PITCH_MIN,  config.PITCH_MAX)
        speaking_rate    = self._clamp(speaking_rate, config.RATE_MIN,   config.RATE_MAX)
        volume           = self._clamp(volume,        config.VOLUME_MIN, config.VOLUME_MAX)
        stability        = self._clamp(stability,        0.0, 1.0)
        similarity_boost = self._clamp(similarity_boost, 0.0, 1.0)
        style            = self._clamp(style,            0.0, 1.0)

        params = VoiceParameters(
            emotion=emotion,
            pitch=pitch,
            speaking_rate=speaking_rate,
            volume=volume,
            tone=target["tone"],
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            confidence=confidence,
            description=target.get("description", ""),
        )

        logger.info(
            f"Voice params for '{emotion}' @ {confidence:.0%} confidence: "
            f"pitch={pitch:+.2f}st, rate={speaking_rate:.2f}×, "
            f"stability={stability:.2f}, style={style:.2f}"
        )
        return params

    def get_emotion_map(self) -> dict:
        """Return the raw emotion-to-param map (used by the API for documentation)."""
        return deepcopy(EMOTION_VOICE_MAP)


# ─────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────
voice_mapper = VoiceMapper()


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from emotion_detector import EmotionResult

    test_cases = [
        EmotionResult("happy",    0.92, "joy",      {}),
        EmotionResult("sad",      0.55, "sadness",  {}),
        EmotionResult("angry",    0.88, "anger",    {}),
        EmotionResult("concerned",0.70, "fear",     {}),
        EmotionResult("neutral",  0.60, "neutral",  {}),
    ]
    for er in test_cases:
        vp = voice_mapper.map(er)
        print(f"\nEmotion : {er.emotion} ({er.confidence:.0%})")
        print(f"Params  : {vp.to_dict()}")
