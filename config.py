"""
config.py - Central Configuration for The Empathy Engine

All API keys, model identifiers, file paths, and system-wide constants live here.
Modify this file to switch TTS providers or swap the emotion model.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv(Path(__file__).parent / ".env")

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR  = STATIC_DIR / "generated_audio"

# Ensure audio output folder exists at startup
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# API Keys  (always from environment variables)
# ─────────────────────────────────────────────
ELEVENLABS_API_KEY: str  = os.getenv("ELEVENLABS_API_KEY", "")
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")

# ─────────────────────────────────────────────
# Emotion Detection — HuggingFace Inference API
# ─────────────────────────────────────────────
# Model: j-hartmann/emotion-english-distilroberta-base
# Labels: anger | disgust | fear | joy | neutral | sadness | surprise
EMOTION_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"

# HuggingFace Inference API endpoint
HF_INFERENCE_URL: str = (
    f"https://api-inference.huggingface.co/models/{EMOTION_MODEL_NAME}"
)

# Map raw HF labels → the 5 canonical emotions the system understands
EMOTION_LABEL_MAP: dict[str, str] = {
    # joy / happiness → happy
    "joy":      "happy",
    "happy":    "happy",
    "happiness":"happy",
    # sadness → sad
    "sadness":  "sad",
    "sad":      "sad",
    # anger / disgust → angry
    "anger":    "angry",
    "angry":    "angry",
    "disgust":  "angry",
    # fear / concern → concerned
    "fear":     "concerned",
    "concerned":"concerned",
    # surprise → happy (positive excitement)
    "surprise": "happy",
    # neutral
    "neutral":  "neutral",
}

DEFAULT_EMOTION: str = "neutral"

# ─────────────────────────────────────────────
# TTS Provider Selection
# ─────────────────────────────────────────────
# Supported values: "elevenlabs" | "gtts" | "pyttsx3"
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "elevenlabs")

# ElevenLabs — one voice per emotion
ELEVENLABS_VOICE_MAP: dict[str, str] = {
    "happy":    "EXAVITQu4vr4xnSDxMaL",   # Sarah  – bright, upbeat
    "sad":      "onwK4e9ZLuTAKqWW03F9",   # Daniel – warm, measured
    "angry":    "pNInz6obpgDQGcFmaJgB",   # Adam   – assertive
    "concerned":"XB0fDUnXU5powFXDhCwa",   # Charlotte – gentle
    "neutral":  "21m00Tcm4TlvDq8ikWAM",   # Rachel – clear, professional
}

ELEVENLABS_MODEL:   str = "eleven_multilingual_v2"
ELEVENLABS_API_URL: str = "https://api.elevenlabs.io/v1"

# ─────────────────────────────────────────────
# Voice Parameter Ranges & Defaults
# ─────────────────────────────────────────────
PITCH_MIN:   float = -10.0
PITCH_MAX:   float = +10.0
RATE_MIN:    float = 0.5
RATE_MAX:    float = 2.0
VOLUME_MIN:  float = 0.5
VOLUME_MAX:  float = 1.5

DEFAULT_VOICE_PARAMS: dict = {
    "pitch": 0.0,
    "speaking_rate": 1.0,
    "volume": 1.0,
    "tone": "neutral",
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
}

# ─────────────────────────────────────────────
# FastAPI / Server Settings
# ─────────────────────────────────────────────
APP_TITLE:       str = "The Empathy Engine"
APP_VERSION:     str = "2.0.0"
APP_DESCRIPTION: str = (
    "Converts text into emotionally expressive speech by detecting "
    "the emotional tone via HuggingFace API and synthesising voice via ElevenLabs."
)
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8001"))

# Audio file settings
AUDIO_FORMAT:      str = "mp3"
AUDIO_SAMPLE_RATE: int = 22_050
