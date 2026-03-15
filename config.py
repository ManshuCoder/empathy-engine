"""
config.py - Central Configuration for The Empathy Engine

All API keys, model identifiers, file paths, and system-wide constants live here.
Modify this file to switch TTS providers or swap the emotion model.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "generated_audio"

# Ensure audio output folder exists at startup
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# API Keys  (set via environment variables for security;
#            fall back to the hard-coded defaults for dev convenience)
# ─────────────────────────────────────────────
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")

# ─────────────────────────────────────────────
# Emotion Detection Model
# ─────────────────────────────────────────────
# Primary  – fine-tuned DistilBERT on the "emotion" dataset (6 classes)
EMOTION_MODEL_NAME: str = "bhadresh-savani/distilbert-base-uncased-emotion"

# Label mapping: the HF model returns these raw labels; we normalise them
# to the 5 canonical emotions the system understands.
EMOTION_LABEL_MAP: dict[str, str] = {
    "joy": "happy",
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "fear": "concerned",
    "concerned": "concerned",
    "surprise": "happy",      # surprise reads as positive excitement
    "disgust": "angry",       # disgust maps to the sharpest tone
    "love": "happy",
    "neutral": "neutral",
}

# Fallback when no label maps cleanly
DEFAULT_EMOTION: str = "neutral"

# ─────────────────────────────────────────────
# TTS Provider Selection
# ─────────────────────────────────────────────
# Supported values: "elevenlabs" | "gtts" | "coqui"
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "elevenlabs")

# ElevenLabs voice IDs (https://api.elevenlabs.io/v1/voices)
ELEVENLABS_VOICE_MAP: dict[str, str] = {
    "happy":    "EXAVITQu4vr4xnSDxMaL",   # Sarah  – bright, upbeat
    "sad":      "onwK4e9ZLuTAKqWW03F9",   # Daniel – warm, measured
    "angry":    "pNInz6obpgDQGcFmaJgB",   # Adam   – strong, assertive
    "concerned":"XB0fDUnXU5powFXDhCwa",   # Charlotte – gentle, concerned
    "neutral":  "21m00Tcm4TlvDq8ikWAM",   # Rachel – clear, professional
}

# ElevenLabs model (use the latest multilingual v2 for best quality)
ELEVENLABS_MODEL: str = "eleven_multilingual_v2"

# ElevenLabs API base URL
ELEVENLABS_API_URL: str = "https://api.elevenlabs.io/v1"

# ─────────────────────────────────────────────
# Voice Parameter Ranges & Defaults
# ─────────────────────────────────────────────
# These bounds are enforced by the voice mapper to prevent extreme values.
PITCH_MIN: float = -10.0
PITCH_MAX: float = +10.0
RATE_MIN: float  = 0.5
RATE_MAX: float  = 2.0
VOLUME_MIN: float = 0.5
VOLUME_MAX: float = 1.5

# Default (neutral) voice parameters
DEFAULT_VOICE_PARAMS: dict = {
    "pitch": 0.0,
    "speaking_rate": 1.0,
    "volume": 1.0,
    "tone": "neutral",
    "stability": 0.5,        # ElevenLabs-specific: 0 = very expressive
    "similarity_boost": 0.75,# ElevenLabs-specific: voice clarity
    "style": 0.0,            # ElevenLabs-specific: style exaggeration
}

# ─────────────────────────────────────────────
# FastAPI / Server Settings
# ─────────────────────────────────────────────
APP_TITLE: str    = "The Empathy Engine"
APP_VERSION: str  = "1.0.0"
APP_DESCRIPTION: str = (
    "Converts text into emotionally expressive speech by detecting "
    "the emotional tone and dynamically adjusting voice parameters."
)
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8001"))

# Audio file settings
AUDIO_FORMAT: str = "mp3"          # "mp3" or "wav"
AUDIO_SAMPLE_RATE: int = 22_050    # Hz – used by gTTS / Coqui fallback
