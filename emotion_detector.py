"""
emotion_detector.py - Emotion Detection Module for The Empathy Engine

Uses a pretrained HuggingFace Transformers model (DistilBERT fine-tuned on the
emotion dataset) to classify the dominant emotional tone of incoming text.

Pipeline:
  raw text → tokenisation → transformer inference → top-label extraction
           → canonical label normalisation → EmotionResult dataclass
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import pipeline, Pipeline

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
        emotion     – one of the 5 canonical emotions the system understands
        confidence  – model confidence score in [0, 1]
        raw_label   – the original label returned by the HF model (for logging)
        all_scores  – full probability distribution over all raw labels (bonus)
    """
    emotion: str
    confidence: float
    raw_label: str
    all_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "confidence": round(self.confidence, 4),
            "raw_label": self.raw_label,
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }


# ─────────────────────────────────────────────
# Classifier Singleton
# ─────────────────────────────────────────────

class EmotionDetector:
    """
    Wraps a HuggingFace text-classification pipeline.

    The model is loaded once at instantiation (lazy if you prefer) and reused
    for every subsequent call – important so the large model weights are not
    re-loaded on each request.
    """

    def __init__(self, model_name: str = config.EMOTION_MODEL_NAME) -> None:
        self.model_name = model_name
        self._classifier: Optional[Pipeline] = None
        logger.info(f"EmotionDetector initialised with model: {model_name}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load (or reload) the HF pipeline. Safe to call multiple times."""
        logger.info(f"Loading HuggingFace model '{self.model_name}' …")
        self._classifier = pipeline(
            task="text-classification",
            model=self.model_name,
            top_k=None,          # return ALL label scores for the bonus viz
            truncation=True,
            max_length=512,
        )
        logger.info("Model loaded successfully.")

    @property
    def classifier(self) -> Pipeline:
        """Lazy-load the model on first access."""
        if self._classifier is None:
            self._load_model()
        return self._classifier

    def _normalise_label(self, raw_label: str) -> str:
        """
        Map raw HuggingFace label → one of the 5 canonical system emotions.

        The DistilBERT emotion model uses labels like 'sadness', 'joy', etc.
        We standardise these so the rest of the pipeline always works with
        the same restricted vocabulary.
        """
        normalised = config.EMOTION_LABEL_MAP.get(raw_label.lower())
        if normalised is None:
            logger.warning(
                f"Unknown emotion label '{raw_label}' – falling back to "
                f"'{config.DEFAULT_EMOTION}'"
            )
            return config.DEFAULT_EMOTION
        return normalised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> EmotionResult:
        """
        Analyse *text* and return an EmotionResult.

        Args:
            text: The input sentence / paragraph to classify.

        Returns:
            EmotionResult with the dominant canonical emotion and confidence.

        Raises:
            ValueError: If text is empty or only whitespace.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text must not be empty.")

        logger.debug(f"Detecting emotion for: '{text[:80]}{'…' if len(text) > 80 else ''}'")

        # Run inference – returns a list-of-lists when top_k=None
        raw_output: list[list[dict]] = self.classifier(text)  # type: ignore[arg-type]

        # Flatten: pipeline(top_k=None) → [[{label, score}, …]]
        scores: list[dict] = raw_output[0] if isinstance(raw_output[0], list) else raw_output

        # Sort descending by score
        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
        top = scores_sorted[0]

        raw_label: str  = top["label"]
        confidence: float = top["score"]
        canonical: str  = self._normalise_label(raw_label)

        # Build the full score map keyed by *canonical* labels for visualisation
        canonical_scores: dict[str, float] = {}
        for entry in scores_sorted:
            c_label = self._normalise_label(entry["label"])
            # Accumulate scores when multiple raw labels map to the same canonical
            canonical_scores[c_label] = canonical_scores.get(c_label, 0.0) + entry["score"]

        result = EmotionResult(
            emotion=canonical,
            confidence=confidence,
            raw_label=raw_label,
            all_scores=canonical_scores,
        )

        logger.info(
            f"Emotion detected: '{canonical}' (confidence={confidence:.2%}, "
            f"raw='{raw_label}')"
        )
        return result


# ─────────────────────────────────────────────
# Module-level singleton (imported by other modules)
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
