"""
tts_engine.py - Text-to-Speech Engine for The Empathy Engine

Primary provider:  ElevenLabs API  (high-quality, expressive)
Fallback 1:        gTTS            (free Google TTS, internet required)
Fallback 2:        pyttsx3         (offline, no internet)

The active provider is chosen by config.TTS_PROVIDER (env var).
Voice parameters from VoiceMapper are applied to each provider.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests

import config
from voice_mapper import VoiceParameters

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SSML Builder
# ─────────────────────────────────────────────

def build_ssml(text: str, params: VoiceParameters) -> str:
    """
    Wrap text in SSML prosody tags for TTS engines that support it.
    ElevenLabs does not use SSML natively, but we generate it here
    as the assignment bonus feature so developers can inspect it.
    """
    pitch_pct  = params.pitch * 6          # 1 semitone ≈ 6% relative pitch
    pitch_str  = f"{pitch_pct:+.1f}%"
    rate_str   = f"{params.speaking_rate:.2f}"
    volume_db  = f"{(params.volume - 1.0) * 6:+.1f}dB"

    return (
        "<speak>"
        f'<prosody pitch="{pitch_str}" rate="{rate_str}" volume="{volume_db}">'
        f"{text}"
        "</prosody>"
        "</speak>"
    )


# ─────────────────────────────────────────────
# Abstract TTS Provider
# ─────────────────────────────────────────────

class TTSProvider(ABC):
    """Abstract base class for all TTS back-ends."""

    @abstractmethod
    def synthesise(
        self,
        text: str,
        params: VoiceParameters,
        output_path: Path,
    ) -> Path:
        """
        Synthesise speech and write it to output_path.

        Returns:
            The actual path of the written audio file (extension may vary).
        """


# ─────────────────────────────────────────────
# Provider 1 – ElevenLabs (Primary)
# ─────────────────────────────────────────────

class ElevenLabsProvider(TTSProvider):
    """
    Uses the ElevenLabs v1 REST API.

    Endpoint: POST /v1/text-to-speech/{voice_id}

    Voice settings applied per emotion:
      stability        → voice variability  (low = expressive)
      similarity_boost → voice clarity
      style            → style exaggeration
      use_speaker_boost → additional clarity

    Each emotion maps to a different voice_id for maximum expressiveness.
    """

    def __init__(self) -> None:
        self.api_key = config.ELEVENLABS_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "xi-api-key":     self.api_key,
            "Content-Type":   "application/json",
            "Accept":         "audio/mpeg",
        })
        logger.info("ElevenLabsProvider initialised.")

    def _get_voice_id(self, emotion: str) -> str:
        return config.ELEVENLABS_VOICE_MAP.get(
            emotion,
            config.ELEVENLABS_VOICE_MAP["neutral"],
        )

    def synthesise(
        self,
        text: str,
        params: VoiceParameters,
        output_path: Path,
    ) -> Path:
        voice_id = self._get_voice_id(params.emotion)
        url      = f"{config.ELEVENLABS_API_URL}/text-to-speech/{voice_id}"

        payload = {
            "text":     text,
            "model_id": config.ELEVENLABS_MODEL,
            "voice_settings": {
                "stability":        round(params.stability, 3),
                "similarity_boost": round(params.similarity_boost, 3),
                "style":            round(params.style, 3),
                "use_speaker_boost": True,
            },
        }

        logger.info(
            f"ElevenLabs → voice_id={voice_id}, emotion={params.emotion}, "
            f"stability={params.stability:.2f}, style={params.style:.2f}"
        )

        resp = self._session.post(url, json=payload, timeout=30)

        if resp.status_code == 401:
            raise RuntimeError(
                "ElevenLabs API returned 401 Unauthorized. "
                "Check that ELEVENLABS_API_KEY is correct."
            )
        if resp.status_code == 422:
            raise RuntimeError(
                f"ElevenLabs API returned 422 Unprocessable Entity: {resp.text[:300]}"
            )
        if not resp.ok:
            logger.error(f"ElevenLabs error {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()

        # Response body is raw MP3 bytes
        mp3_path = output_path.with_suffix(".mp3")
        mp3_path.write_bytes(resp.content)
        logger.info(f"ElevenLabs audio saved: {mp3_path} ({len(resp.content):,} bytes)")
        return mp3_path


# ─────────────────────────────────────────────
# Provider 2 – gTTS (Free fallback)
# ─────────────────────────────────────────────

class GTTSProvider(TTSProvider):
    """
    Uses the gTTS library (Google Translate TTS).
    Limited customisation — speaking rate via the slow flag only.
    """

    def synthesise(
        self,
        text: str,
        params: VoiceParameters,
        output_path: Path,
    ) -> Path:
        try:
            from gtts import gTTS  # type: ignore
        except ImportError:
            raise RuntimeError("gTTS is not installed. Run: pip install gtts") from None

        slow    = params.speaking_rate < 0.85
        tts     = gTTS(text=text, lang="en", slow=slow)
        mp3_path = output_path.with_suffix(".mp3")
        tts.save(str(mp3_path))
        logger.info(f"gTTS audio saved: {mp3_path}")
        return mp3_path


# ─────────────────────────────────────────────
# Provider 3 – pyttsx3 (Offline fallback)
# ─────────────────────────────────────────────

class Pyttsx3Provider(TTSProvider):
    """
    Offline TTS using pyttsx3 (wraps eSpeak / SAPI / nsss depending on OS).
    Supports rate and volume natively. Output is WAV.
    """

    def synthesise(
        self,
        text: str,
        params: VoiceParameters,
        output_path: Path,
    ) -> Path:
        try:
            import pyttsx3  # type: ignore
        except ImportError:
            raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3") from None

        engine = pyttsx3.init()
        engine.setProperty("rate",   int(200 * params.speaking_rate))
        engine.setProperty("volume", params.volume)

        wav_path = output_path.with_suffix(".wav")
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()
        engine.stop()
        logger.info(f"pyttsx3 audio saved: {wav_path}")
        return wav_path


# ─────────────────────────────────────────────
# TTS Engine Facade
# ─────────────────────────────────────────────

class TTSEngine:
    """
    High-level facade that:
      1. Selects the configured TTS provider.
      2. Generates a unique filename for each audio file.
      3. Delegates synthesis to the provider with automatic fallback.
      4. Returns the relative URL path for the API response.

    Fallback chain: ElevenLabs → gTTS → pyttsx3
    """

    PROVIDERS: dict[str, type[TTSProvider]] = {
        "elevenlabs": ElevenLabsProvider,
        "gtts":       GTTSProvider,
        "pyttsx3":    Pyttsx3Provider,
    }

    FALLBACK_ORDER = ["elevenlabs", "gtts", "pyttsx3"]

    def __init__(self, provider: str = config.TTS_PROVIDER) -> None:
        self.provider_name = provider
        self._providers: dict[str, TTSProvider] = {}
        logger.info(f"TTSEngine initialised → primary provider='{provider}'")

    def _get_provider(self, name: str) -> TTSProvider:
        if name not in self._providers:
            cls = self.PROVIDERS.get(name)
            if cls is None:
                raise ValueError(f"Unknown TTS provider: '{name}'")
            self._providers[name] = cls()
        return self._providers[name]

    def _unique_path(self) -> Path:
        """Generate a unique output file path inside the audio directory."""
        stem = f"audio_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return config.AUDIO_DIR / stem   # provider appends the extension

    def generate(
        self,
        text: str,
        params: VoiceParameters,
        preferred_provider: Optional[str] = None,
    ) -> tuple[Path, str]:
        """
        Synthesise speech for text using params.

        Args:
            text               : Input text.
            params             : VoiceParameters from the mapper.
            preferred_provider : Override the default provider for this request.

        Returns:
            (audio_file_path, relative_url_for_api_response)
        """
        output_base   = self._unique_path()
        provider_name = preferred_provider or self.provider_name

        # Build fallback chain: preferred provider first, then the rest
        chain = [provider_name] + [
            p for p in self.FALLBACK_ORDER if p != provider_name
        ]

        last_exc: Optional[Exception] = None
        for name in chain:
            try:
                provider   = self._get_provider(name)
                audio_path = provider.synthesise(text, params, output_base)
                rel_url    = f"/static/generated_audio/{audio_path.name}"
                logger.info(f"TTS success via '{name}': {rel_url}")
                return audio_path, rel_url
            except Exception as exc:
                logger.warning(f"Provider '{name}' failed: {exc}. Trying next…")
                last_exc = exc

        raise RuntimeError(
            f"All TTS providers failed. Last error: {last_exc}"
        ) from last_exc

    def get_ssml(self, text: str, params: VoiceParameters) -> str:
        """Return SSML for the given text and voice params (bonus feature)."""
        return build_ssml(text, params)


# ─────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────
tts_engine = TTSEngine()


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from emotion_detector import EmotionResult
    from voice_mapper import voice_mapper

    er = EmotionResult("happy", 0.91, "joy", {})
    vp = voice_mapper.map(er)
    path, url = tts_engine.generate("I just got some amazing news!", vp)
    print(f"Audio file : {path}")
    print(f"Audio URL  : {url}")
    print(f"SSML       : {tts_engine.get_ssml('I just got some amazing news!', vp)}")
