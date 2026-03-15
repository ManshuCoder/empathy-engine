"""
tts_engine.py - Text-to-Speech Engine for The Empathy Engine

Provides a unified TTS interface with three PROVIDER strategies:
  1. ElevenLabs API  (primary – highest quality, voice style control)
  2. gTTS            (fallback – free, requires internet)
  3. pyttsx3         (offline fallback – no internet needed)

The active provider is chosen by config.TTS_PROVIDER.

Voice parameters from VoiceMapper are applied to each provider as best
the API allows:
  • ElevenLabs supports stability, similarity_boost, style, speaking_rate
  • gTTS supports speaking_rate (via tempo adjustment in post-processing)
  • pyttsx3 supports rate and volume natively
"""

import io
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
    Wrap plain text in SSML prosody tags so TTS engines that support SSML
    (like Google Cloud TTS) can interpret pitch and rate directly.

    Even for engines that ignore SSML we return valid XML so developers can
    inspect what parameters would have been applied.
    """
    # Convert semitone offset to percentage for SSML prosody pitch attribute
    # Rule of thumb: each semitone ≈ ~6% relative pitch change
    pitch_pct = params.pitch * 6
    pitch_str = f"{pitch_pct:+.1f}%"

    rate_str = f"{params.speaking_rate:.2f}"
    volume_db = f"{(params.volume - 1.0) * 6:+.1f}dB"   # ±6 dB corresponds to ±50% amplitude

    ssml = (
        "<speak>"
        f'<prosody pitch="{pitch_str}" rate="{rate_str}" volume="{volume_db}">'
        f"{text}"
        "</prosody>"
        "</speak>"
    )
    return ssml


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
        Synthesise speech and write it to *output_path*.

        Args:
            text        : Raw text to convert to speech.
            params      : Voice parameters from VoiceMapper.
            output_path : Where to write the audio file.

        Returns:
            The actual path of the written audio file (may differ in extension).
        """


# ─────────────────────────────────────────────
# Provider 1 – ElevenLabs
# ─────────────────────────────────────────────

class ElevenLabsProvider(TTSProvider):
    """
    Uses the ElevenLabs v1 REST API to generate high-quality, expressive speech.

    Voice settings applied per request:
      stability        → how variable the voice is (lower = more expressive)
      similarity_boost → how closely it matches the trained voice
      style            → exaggeration of the voice style
      use_speaker_boost → additional clarity enhancement

    Speaking rate is applied via the `speed` field available in the
    text-to-speech-with-timestamps endpoint (experimental) or via SSML prosody.
    """

    def __init__(self) -> None:
        self.api_key = config.ELEVENLABS_API_KEY
        self.base_url = config.ELEVENLABS_API_URL
        self.model_id = config.ELEVENLABS_MODEL
        self._session = requests.Session()
        self._session.headers.update({
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
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
        url = f"{self.base_url}/text-to-speech/{voice_id}"

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": round(params.stability, 3),
                "similarity_boost": round(params.similarity_boost, 3),
                "style": round(params.style, 3),
                "use_speaker_boost": True,
            },
        }

        logger.info(
            f"ElevenLabs → voice_id={voice_id}, "
            f"stability={params.stability:.2f}, style={params.style:.2f}"
        )

        resp = self._session.post(url, json=payload, timeout=30)

        if resp.status_code != 200:
            logger.error(
                f"ElevenLabs API error {resp.status_code}: {resp.text[:300]}"
            )
            resp.raise_for_status()

        # API returns raw MP3 bytes
        mp3_path = output_path.with_suffix(".mp3")
        mp3_path.write_bytes(resp.content)
        logger.info(f"Audio saved to {mp3_path} ({len(resp.content):,} bytes)")
        return mp3_path


# ─────────────────────────────────────────────
# Provider 2 – gTTS (Google Text-to-Speech, free)
# ─────────────────────────────────────────────

class GTTSProvider(TTSProvider):
    """
    Uses the public gTTS library (Google Translate TTS endpoint).
    Limited voice customisation – we control speaking speed (slow flag only).
    Post-processing with pydub can be added for pitch shifting.
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
            raise RuntimeError(
                "gTTS is not installed. Run: pip install gtts"
            ) from None

        slow = params.speaking_rate < 0.85   # gTTS only has a boolean slow flag
        tts = gTTS(text=text, lang="en", slow=slow)

        mp3_path = output_path.with_suffix(".mp3")
        tts.save(str(mp3_path))
        logger.info(f"gTTS audio saved to {mp3_path}")
        return mp3_path


# ─────────────────────────────────────────────
# Provider 3 – pyttsx3 (offline)
# ─────────────────────────────────────────────

class Pyttsx3Provider(TTSProvider):
    """
    Offline TTS using pyttsx3 (wraps eSpeak / SAPI / nsss depending on OS).
    Supports rate and volume natively; pitch is system-dependent.
    Output is WAV on most platforms.
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
            raise RuntimeError(
                "pyttsx3 is not installed. Run: pip install pyttsx3"
            ) from None

        engine = pyttsx3.init()

        # Default WPM ≈ 200; scale by speaking_rate
        base_wpm = 200
        engine.setProperty("rate", int(base_wpm * params.speaking_rate))
        engine.setProperty("volume", params.volume)

        wav_path = output_path.with_suffix(".wav")
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()
        engine.stop()
        logger.info(f"pyttsx3 audio saved to {wav_path}")
        return wav_path


# ─────────────────────────────────────────────
# TTS Engine Facade
# ─────────────────────────────────────────────

class TTSEngine:
    """
    High-level facade that:
      1. Selects the configured TTS provider.
      2. Generates a unique filename for each request.
      3. Delegates synthesis to the provider.
      4. Returns the relative URL path for the API response.

    Falls back through ElevenLabs → gTTS → pyttsx3 if the preferred
    provider raises an exception.
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
        logger.info(f"TTSEngine initialised with provider='{provider}'")

    def _get_provider(self, name: str) -> TTSProvider:
        if name not in self._providers:
            cls = self.PROVIDERS.get(name)
            if cls is None:
                raise ValueError(f"Unknown TTS provider: '{name}'")
            self._providers[name] = cls()
        return self._providers[name]

    def _unique_path(self) -> Path:
        """Generate a unique output file stem inside the audio directory."""
        stem = f"audio_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return config.AUDIO_DIR / stem   # extension added by provider

    def generate(
        self,
        text: str,
        params: VoiceParameters,
        preferred_provider: Optional[str] = None,
    ) -> tuple[Path, str]:
        """
        Synthesise speech for *text* using *params*.

        Args:
            text               : Input text.
            params             : VoiceParameters from the mapper.
            preferred_provider : Override the default provider for this call.

        Returns:
            (audio_file_path, relative_url_path)
        """
        output_base = self._unique_path()
        provider_name = preferred_provider or self.provider_name

        # Build fallback chain: preferred first, then the rest in order
        chain = [provider_name] + [
            p for p in self.FALLBACK_ORDER if p != provider_name
        ]

        last_exc: Optional[Exception] = None
        for name in chain:
            try:
                provider = self._get_provider(name)
                audio_path = provider.synthesise(text, params, output_base)
                # Build the relative URL served by FastAPI's StaticFiles mount
                rel_url = f"/static/generated_audio/{audio_path.name}"
                logger.info(f"TTS success via '{name}': {rel_url}")
                return audio_path, rel_url
            except Exception as exc:
                logger.warning(f"Provider '{name}' failed: {exc}. Trying next…")
                last_exc = exc

        raise RuntimeError(
            f"All TTS providers failed. Last error: {last_exc}"
        ) from last_exc

    def get_ssml(self, text: str, params: VoiceParameters) -> str:
        """Return the SSML representation for the given text and params (bonus feature)."""
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
