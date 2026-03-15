"""
emotion_detector.py - Emotion Detection via HuggingFace InferenceClient

Sends text to the HuggingFace Inference API using the model:
    j-hartmann/emotion-english-distilroberta-base

NOTE: The old api-inference.huggingface.co endpoint returned 410 Gone (deprecated).
      This module now uses huggingface_hub.InferenceClient, which routes through
      https://router.huggingface.co/hf-inference — the current supported endpoint.

This model classifies text into 7 emotions:
    anger | disgust | fear | joy | neutral | sadness | surprise

These are then normalised to the 5 canonical emotions the system uses:
    happy | sad | angry | concerned | neutral

No local model download required — all inference runs on HuggingFace servers.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Schema
# ─────────────────────────────────────────────

@dataclass
class EmotionResult:
    """
    Represents the output of the emotion detection step.

    Attributes:
        emotion     – one of 5 canonical emotions: happy | sad | angry | concerned | neutral
        confidence  – model confidence score in [0, 1]
        raw_label   – the original label returned by the HF model (before normalisation)
        all_scores  – full probability distribution over canonical emotions (for UI chart)
    """
    emotion:    str
    confidence: float
    raw_label:  str
    all_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "emotion":    self.emotion,
            "confidence": round(self.confidence, 4),
            "raw_label":  self.raw_label,
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }


# ─────────────────────────────────────────────
# Emotion Detector
# ─────────────────────────────────────────────

class EmotionDetector:
    """
    Calls the HuggingFace Inference API to classify text emotions.

    Uses model: j-hartmann/emotion-english-distilroberta-base

    Migration note:
        The legacy api-inference.huggingface.co endpoint now returns 410 Gone.
        We now use huggingface_hub.InferenceClient, which targets:
            https://router.huggingface.co/hf-inference
        This is the officially supported endpoint as of 2025.

    The API key is read from config.HUGGINGFACE_API_KEY (env var).
    A valid HuggingFace token with 'Inference' permissions is required.
    """

    def __init__(self) -> None:
        self._client = self._build_client()
        logger.info(
            f"EmotionDetector ready. Model: {config.EMOTION_MODEL_NAME} "
            f"(using huggingface_hub.InferenceClient)"
        )

    def _build_client(self):
        """
        Build the InferenceClient.
        Requires huggingface_hub >= 0.23.0
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed. "
                "Run: pip install huggingface_hub>=0.23.0"
            )

        token = config.HUGGINGFACE_API_KEY or None
        if not token:
            logger.warning(
                "HUGGINGFACE_API_KEY not set. "
                "The HuggingFace Inference API now requires a valid token. "
                "Set HUGGINGFACE_API_KEY in your .env file. "
                "Get a free token at: https://huggingface.co/settings/tokens"
            )
        else:
            logger.info("HuggingFace API key loaded.")

        # provider="hf-inference" routes to router.huggingface.co/hf-inference
        return InferenceClient(
            model=config.EMOTION_MODEL_NAME,
            token=token,
            provider="hf-inference",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_label(self, raw_label: str) -> str:
        """
        Map a raw HuggingFace label → one of the 5 canonical system emotions.

        Example:
            "joy"     → "happy"
            "sadness" → "sad"
            "fear"    → "concerned"
        """
        normalised = config.EMOTION_LABEL_MAP.get(raw_label.lower())
        if normalised is None:
            logger.warning(
                f"Unknown emotion label '{raw_label}' – falling back to "
                f"'{config.DEFAULT_EMOTION}'"
            )
            return config.DEFAULT_EMOTION
        return normalised

    def _call_hf_api(self, text: str) -> list[dict]:
        """
        Call HuggingFace InferenceClient for text classification.

        Returns a list of {label, score} dicts sorted by score descending.
        Handles transient errors with exponential back-off (up to 3 attempts).
        """
        last_exc: Optional[Exception] = None

        for attempt in range(3):
            try:
                # InferenceClient.text_classification returns a list of
                # ClassificationOutput objects with .label and .score attributes.
                results = self._client.text_classification(text)

                # Normalise to list of dicts for consistent downstream handling
                scores: list[dict] = [
                    {"label": r.label, "score": r.score}
                    for r in results
                ]

                # Sort by score descending
                return sorted(scores, key=lambda x: x["score"], reverse=True)

            except Exception as exc:
                err_str = str(exc)
                logger.error(
                    f"HuggingFace API request failed (attempt {attempt + 1}/3): {exc}"
                )

                # If model is loading, wait and retry
                if "loading" in err_str.lower() or "503" in err_str:
                    wait_sec = min(20 * (attempt + 1), 60)
                    logger.info(f"Model loading, waiting {wait_sec}s…")
                    time.sleep(wait_sec)
                    last_exc = exc
                    continue

                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)  # exponential back-off: 1s, 2s
                    continue

        raise RuntimeError(
            f"HuggingFace Inference API call failed after 3 attempts: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> EmotionResult:
        """
        Analyse text and return an EmotionResult.

        Args:
            text: The input sentence / paragraph to classify.

        Returns:
            EmotionResult with the dominant canonical emotion and confidence score.

        Raises:
            ValueError:   If text is empty or only whitespace.
            RuntimeError: If the HuggingFace API call fails.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text must not be empty.")

        logger.debug(
            f"Detecting emotion via HF API: "
            f"'{text[:80]}{'…' if len(text) > 80 else ''}'"
        )

        # Call HuggingFace Inference API
        scores_sorted = self._call_hf_api(text)

        # Top prediction
        top            = scores_sorted[0]
        raw_label: str = top["label"]
        confidence: float = top["score"]
        canonical: str = self._normalise_label(raw_label)

        # Build canonical score map (sum scores that map to the same canonical label)
        canonical_scores: dict[str, float] = {}
        for entry in scores_sorted:
            c_label = self._normalise_label(entry["label"])
            canonical_scores[c_label] = (
                canonical_scores.get(c_label, 0.0) + entry["score"]
            )

        result = EmotionResult(
            emotion=canonical,
            confidence=confidence,
            raw_label=raw_label,
            all_scores=canonical_scores,
        )

        logger.info(
            f"Emotion detected: '{canonical}' "
            f"(confidence={confidence:.2%}, raw='{raw_label}')"
        )
        return result


# ─────────────────────────────────────────────
# Module-level singleton (imported by app.py)
# ─────────────────────────────────────────────
emotion_detector = EmotionDetector()


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        "I just got promoted! This is the best day of my life!",
        "I can't believe they cancelled the project after all our hard work.",
        "This is absolutely unacceptable! I am furious!",
        "Are you sure everything is going to be alright? I'm a bit worried.",
        "The meeting is scheduled for 3 PM tomorrow.",
    ]
    for sample in samples:
        result = emotion_detector.detect(sample)
        print(f"\nText   : {sample}")
        print(f"Result : {result.to_dict()}")
