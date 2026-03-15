"""
emotion_detector.py - Emotion Detection via HuggingFace Inference API

Sends text to the HuggingFace hosted Inference API using the model:
    j-hartmann/emotion-english-distilroberta-base

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

import requests

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
    Endpoint:   https://api-inference.huggingface.co/models/<model>

    The API key is read from config.HUGGINGFACE_API_KEY (env var).
    If no key is set, HF allows a limited number of free calls per hour.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
        })
        # Attach the API key only if available
        if config.HUGGINGFACE_API_KEY:
            self._session.headers["Authorization"] = (
                f"Bearer {config.HUGGINGFACE_API_KEY}"
            )
            logger.info("HuggingFace API key loaded.")
        else:
            logger.warning(
                "HUGGINGFACE_API_KEY not set — using unauthenticated access "
                "(rate-limited). Set the key in your .env file for production use."
            )
        logger.info(
            f"EmotionDetector ready. Model: {config.EMOTION_MODEL_NAME}"
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
        POST to the HuggingFace Inference API and return the list of
        {label, score} dicts sorted by score descending.

        Handles the model loading wait (503 → retry once after warm-up).
        """
        payload = {
            "inputs": text,
            "parameters": {"return_all_scores": True},
            "options":    {"wait_for_model": True},
        }

        for attempt in range(3):
            try:
                resp = self._session.post(
                    config.HF_INFERENCE_URL,
                    json=payload,
                    timeout=30,
                )

                # Model is still loading — wait and retry
                if resp.status_code == 503:
                    estimated_wait = resp.json().get("estimated_time", 20)
                    wait_sec = min(float(estimated_wait), 30)
                    logger.info(
                        f"HF model loading, waiting {wait_sec:.0f}s … "
                        f"(attempt {attempt + 1}/3)"
                    )
                    time.sleep(wait_sec)
                    continue

                resp.raise_for_status()

                # HF returns: [[{label, score}, …]] (list-of-lists)
                raw: list = resp.json()
                scores: list[dict] = raw[0] if isinstance(raw[0], list) else raw
                return sorted(scores, key=lambda x: x["score"], reverse=True)

            except requests.RequestException as exc:
                logger.error(f"HuggingFace API request failed (attempt {attempt + 1}): {exc}")
                if attempt == 2:
                    raise RuntimeError(
                        f"HuggingFace Inference API call failed after 3 attempts: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)   # exponential back-off

        raise RuntimeError("HuggingFace API did not respond after retries.")

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
