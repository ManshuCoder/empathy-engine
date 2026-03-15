# 🎙️ The Empathy Engine
### Challenge 1 — Giving AI a Human Voice

> An AI service that converts text into **emotionally expressive speech** by detecting the emotional tone of input text using HuggingFace and synthesising natural-sounding audio via ElevenLabs.

---

## 🏗️ Architecture

```
User Text Input
      │
      ▼
┌─────────────────────────────────────┐
│  POST /generate-voice  (FastAPI)    │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  emotion_detector.py                │
│  HuggingFace Inference API          │
│  Model: j-hartmann/emotion-english- │
│         distilroberta-base          │
│  → anger | disgust | fear | joy     │
│    neutral | sadness | surprise     │
│  → normalised to 5 canonical labels │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  voice_mapper.py                    │
│  Emotion → Voice Parameters         │
│  Intensity scaling via confidence   │
│  ─────────────────────────────────  │
│  happy    → pitch ↑, rate ↑, lively │
│  sad      → pitch ↓, rate ↓, soft  │
│  angry    → pitch ↑, loud, sharp   │
│  concerned→ rate ↓, gentle tone    │
│  neutral  → default voice          │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  tts_engine.py                      │
│  ElevenLabs API                     │
│  POST /v1/text-to-speech/{voice_id} │
│  Parameters: stability,             │
│  similarity_boost, style            │
│  Fallback: gTTS → pyttsx3          │
└─────────────────────────────────────┘
      │
      ▼
    🔊 Audio File (MP3)
    ← Served via /static/generated_audio/
```

---

## 📁 Project Structure

```
empathy-engine/
├── app.py                   # FastAPI server + API endpoints
├── emotion_detector.py      # HuggingFace Inference API client
├── voice_mapper.py          # Emotion → voice parameter mapping
├── tts_engine.py            # ElevenLabs / gTTS / pyttsx3 providers
├── config.py                # All configuration, API keys, constants
│
├── static/
│   ├── index.html           # Web UI (HTML + CSS + Vanilla JS)
│   └── generated_audio/     # Generated MP3 files
│
├── .env                     # API keys (git-ignored)
├── .env.example             # Template — copy to .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔌 APIs Used

### 1. HuggingFace Inference API — Emotion Detection

- **Model**: [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- **Client**: `huggingface_hub.InferenceClient` (routed via `router.huggingface.co/hf-inference`)
- **Output**: 7 probability scores — `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`
- **Normalised to**: 5 canonical labels — `happy`, `sad`, `angry`, `concerned`, `neutral`

> **Note**: The legacy `api-inference.huggingface.co` endpoint was deprecated in 2024/2025
> and now returns **410 Gone**. This project uses `huggingface_hub.InferenceClient`
> which targets the current `router.huggingface.co/hf-inference` endpoint.

```python
# Using the new InferenceClient (huggingface_hub >= 0.23.0)
from huggingface_hub import InferenceClient
client = InferenceClient(
    model="j-hartmann/emotion-english-distilroberta-base",
    token="<HF_API_KEY>",
    provider="hf-inference",
)
results = client.text_classification("I can't believe this happened. I'm devastated.")
# → [ClassificationOutput(label='sadness', score=0.87), ...]
```

### 2. ElevenLabs API — Voice Synthesis

- **Endpoint**: `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`
- **Model**: `eleven_multilingual_v2`
- **Parameters**: `stability`, `similarity_boost`, `style`, `use_speaker_boost`
- **Voice per emotion**: Different voice IDs mapped to each emotion for maximum expressiveness

```python
# Example request
headers = {"xi-api-key": "<ELEVENLABS_KEY>", "Content-Type": "application/json"}
payload = {
    "text": "I'm so happy today!",
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.30,
        "similarity_boost": 0.75,
        "style": 0.40,
        "use_speaker_boost": True
    }
}
# Response: raw MP3 bytes
```

---

## 🎭 Emotion → Voice Mapping

| Emotion   | Pitch     | Rate      | Stability | Style | Description                        |
|-----------|-----------|-----------|-----------|-------|------------------------------------|
| happy     | +3.0 st   | 1.25×     | 0.30      | 0.40  | Bright, energetic, upbeat          |
| sad       | −2.5 st   | 0.80×     | 0.70      | 0.10  | Low, slow, melancholic             |
| angry     | +4.0 st   | 1.15×     | 0.20      | 0.60  | Sharp, assertive, high intensity   |
| concerned | −1.0 st   | 0.90×     | 0.55      | 0.20  | Gentle, careful, measured          |
| neutral   | ±0 st     | 1.00×     | 0.50      | 0.00  | Professional, clear, flat          |

**Intensity Scaling**: Voice parameters are blended between neutral and target values proportional to the confidence score:
```
param = neutral + confidence × (target − neutral)
```

---

## 🚀 Setup & Running

### 1. Clone & enter directory
```bash
git clone https://github.com/ManshuCoder/empathy-engine.git
cd empathy-engine
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
```bash
cp .env.example .env
# Edit .env and fill in your keys:
# HUGGINGFACE_API_KEY=hf_...
# ELEVENLABS_API_KEY=...
# TTS_PROVIDER=elevenlabs
```

### 5. Run the server
```bash
python app.py
```

Open **http://localhost:8001** in your browser.

---

## 🌐 API Reference

### `POST /generate-voice`
Convert text to emotionally expressive speech.

**Request:**
```json
{
  "text": "I can't believe this happened. I'm really upset.",
  "voice_style": null,
  "provider": null
}
```
- `voice_style` (optional): Force an emotion — `happy | sad | angry | concerned | neutral`
- `provider` (optional): Override TTS engine — `elevenlabs | gtts | pyttsx3`

**Response:**
```json
{
  "text": "I can't believe this happened...",
  "emotion": "sad",
  "confidence": 0.8734,
  "raw_label": "sadness",
  "all_scores": { "sad": 0.87, "neutral": 0.08, ... },
  "voice_params": { "pitch": -2.37, "speaking_rate": 0.81, "stability": 0.68, ... },
  "audio_url": "/static/generated_audio/audio_1234567890_abcd1234.mp3",
  "ssml": "<speak><prosody pitch='-14.2%' rate='0.81'>...</prosody></speak>",
  "processing_time_ms": 1240.5
}
```

### `GET /health`
```json
{ "status": "ok", "version": "2.0.0", "tts_provider": "elevenlabs", "hf_model": "j-hartmann/..." }
```

### `GET /emotion-map`
Returns the full emotion → voice parameter reference table.

### `GET /docs`
Interactive Swagger UI for all endpoints.

---

## ✨ Bonus Features Implemented

- ✅ **Emotion confidence score** displayed with animated progress bar
- ✅ **Emotion distribution chart** — bar chart for all 5 canonical emotions
- ✅ **Emotion highlight UI** — glowing active-state buttons
- ✅ **SSML output** — full Speech Synthesis Markup Language display
- ✅ **Voice style control** — per-emotion stability, style, similarity_boost
- ✅ **Waveform visualisation** — animated audio waveform bars
- ✅ **Automatic fallback** — ElevenLabs → gTTS → pyttsx3
- ✅ **TTS provider selector** — switch live between providers
- ✅ **Keyboard shortcut** — Ctrl+Enter to generate
- ✅ **Audio download** — built-in download button

---

## 🔒 Security

- All API keys are loaded via environment variables — never hardcoded
- `.env` is listed in `.gitignore` and never committed
- Keys default to empty string if not set — app degrades gracefully

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `requests` | ElevenLabs HTTP calls |
| `huggingface_hub` | HuggingFace InferenceClient (new endpoint) |
| `python-dotenv` | .env loading |
| `gtts` | Free Google TTS fallback |
| `pyttsx3` | Offline TTS fallback |
| `pydantic` | Request/response validation |
| `aiofiles` | Async static file serving |

---

*The Empathy Engine — Challenge 1 Submission*
