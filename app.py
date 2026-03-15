"""
app.py - FastAPI Server for The Empathy Engine

This is the entry point and web server for the service.

Endpoints:
  POST /generate-voice   → full emotion detection + TTS pipeline
  GET  /emotion-map      → returns the emotion → voice-param mapping table
  GET  /health           → liveness probe
  GET  /                 → serves the web UI (index.html)

Static Assets:
  /static/generated_audio/  → generated audio files
  /static/                  → UI assets (HTML, CSS, JS)
"""

import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import config
from emotion_detector import emotion_detector
from tts_engine import tts_engine
from voice_mapper import voice_mapper

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────
app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow cross-origin requests (needed when the frontend is on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (generated audio + frontend assets)
app.mount(
    "/static",
    StaticFiles(directory=str(config.STATIC_DIR)),
    name="static",
)


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class GenerateVoiceRequest(BaseModel):
    """Request body for POST /generate-voice."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text to convert to speech.",
        examples=["I can't believe this happened. It's absolutely devastating."],
    )
    voice_style: str | None = Field(
        default=None,
        description=(
            "Override the auto-detected emotion. "
            "One of: happy, sad, angry, concerned, neutral"
        ),
    )
    provider: str | None = Field(
        default=None,
        description="TTS provider override: elevenlabs | gtts | pyttsx3",
    )

    @field_validator("voice_style")
    @classmethod
    def validate_voice_style(cls, v: str | None) -> str | None:
        allowed = {"happy", "sad", "angry", "concerned", "neutral", None}
        if v not in allowed:
            raise ValueError(
                f"voice_style must be one of {sorted(allowed - {None})} or null"
            )
        return v

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str | None) -> str | None:
        allowed_providers = {"elevenlabs", "gtts", "pyttsx3", None}
        if v not in allowed_providers:
            raise ValueError(
                f"provider must be one of {sorted(allowed_providers - {None})} or null"
            )
        return v


class GenerateVoiceResponse(BaseModel):
    """Response body for POST /generate-voice."""
    text: str
    emotion: str
    confidence: float
    raw_label: str
    all_scores: dict[str, float]
    voice_params: dict
    audio_url: str
    ssml: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    tts_provider: str


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui(request: Request) -> HTMLResponse:
    """Serve the web frontend."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if not ui_path.exists():
        return HTMLResponse(
            content="<h1>Frontend not found – check static/index.html</h1>",
            status_code=404,
        )
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Simple liveness probe."""
    return HealthResponse(
        status="ok",
        version=config.APP_VERSION,
        tts_provider=config.TTS_PROVIDER,
    )


@app.get("/emotion-map", tags=["Reference"])
async def get_emotion_map() -> JSONResponse:
    """
    Return the full emotion-to-voice-parameter mapping table.
    Useful for frontend visualisations and API consumers.
    """
    return JSONResponse(content=voice_mapper.get_emotion_map())


@app.post(
    "/generate-voice",
    response_model=GenerateVoiceResponse,
    tags=["Core"],
    summary="Convert text to emotionally expressive speech",
)
async def generate_voice(body: GenerateVoiceRequest) -> GenerateVoiceResponse:
    """
    Full pipeline:
      1. Detect emotion from text (unless voice_style override supplied)
      2. Map emotion → voice parameters (with intensity scaling)
      3. Synthesise speech via the configured TTS provider
      4. Return audio URL + full metadata

    **voice_style** lets the caller force a specific emotion regardless of the
    detected one – useful for demo and testing scenarios.
    """
    t_start = time.perf_counter()

    try:
        # Step 1 – Emotion Detection
        emotion_result = emotion_detector.detect(body.text)

        # Apply voice_style override if provided
        if body.voice_style:
            logger.info(
                f"Overriding detected emotion '{emotion_result.emotion}' "
                f"→ '{body.voice_style}' (manual selection)"
            )
            emotion_result.emotion = body.voice_style

        # Step 2 – Voice Mapping
        voice_params = voice_mapper.map(emotion_result)

        # Step 3 – TTS Synthesis
        audio_path, audio_url = tts_engine.generate(
            text=body.text,
            params=voice_params,
            preferred_provider=body.provider,
        )

        # Step 4 – SSML (bonus feature)
        ssml = tts_engine.get_ssml(body.text, voice_params)

        processing_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Request completed in {processing_ms:.0f} ms — "
            f"emotion='{emotion_result.emotion}', audio='{audio_url}'"
        )

        return GenerateVoiceResponse(
            text=body.text,
            emotion=emotion_result.emotion,
            confidence=round(emotion_result.confidence, 4),
            raw_label=emotion_result.raw_label,
            all_scores={k: round(v, 4) for k, v in emotion_result.all_scores.items()},
            voice_params=voice_params.to_dict(),
            audio_url=audio_url,
            ssml=ssml,
            processing_time_ms=round(processing_ms, 1),
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during voice generation")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(exc).__name__}: {exc}",
        ) from exc


# ─────────────────────────────────────────────
# Server Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(
        f"Starting {config.APP_TITLE} v{config.APP_VERSION} "
        f"on http://{config.HOST}:{config.PORT}"
    )
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info",
    )
