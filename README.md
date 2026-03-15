# The Empathy Engine 🎙️
### *Giving AI a Human Voice*

> **Convert text into emotionally expressive speech** by detecting emotional tone and dynamically adjusting pitch, speaking rate, volume, and voice character before synthesis.

---

## 🌟 Project Overview

The Empathy Engine bridges the gap between flat, robotic TTS output and truly expressive, human-like speech. Instead of treating all text the same, the system:

1. **Detects emotion** in the input text using a fine-tuned DistilBERT model
2. **Maps the emotion** to specific voice parameters (pitch, rate, volume, tone)
3. **Synthesises speech** via ElevenLabs with emotion-appropriate voice settings
4. **Serves the audio** through a clean REST API and browser-based UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     POST /generate-voice                        │
│                        (app.py)                                 │
└─────────────────┬───────────────────────────────────────────────┘
                  │  text
                  ▼
┌─────────────────────────────────┐
│      Emotion Detector           │
│   (emotion_detector.py)         │
│                                 │
│  DistilBERT fine-tuned on       │
│  the 'emotion' dataset          │
│  → raw label + confidence       │
│  → canonical emotion label      │
└─────────────────┬───────────────┘
                  │  EmotionResult
                  ▼
┌─────────────────────────────────┐
│       Voice Mapper              │
│    (voice_mapper.py)            │
│                                 │
│  emotion + confidence           │
│  → intensity-scaled params      │
│  → pitch, rate, volume,         │
│    stability, style             │
└─────────────────┬───────────────┘
                  │  VoiceParameters
                  ▼
┌─────────────────────────────────┐
│        TTS Engine               │
│      (tts_engine.py)            │
│                                 │
│  Provider cascade:              │
│  1. ElevenLabs API  (primary)   │
│  2. gTTS            (fallback)  │
│  3. pyttsx3         (offline)   │
│                                 │
│  + SSML generation              │
└─────────────────┬───────────────┘
                  │  audio file (.mp3/.wav)
                  ▼
         /static/generated_audio/
              audio_<id>.mp3
```

---

## ⚡ Emotion → Voice Parameter Mapping

The core insight is that different emotions require different *acoustic characteristics* to sound authentic:

| Emotion   | Pitch       | Rate        | Volume      | Tone        | ElevenLabs Stability | Notes                          |
|-----------|-------------|-------------|-------------|-------------|----------------------|--------------------------------|
| 😊 Happy  | +3 semitones| 1.25×       | 1.1×        | Cheerful    | Low (0.30)           | Fast, bright, expressive       |
| 😢 Sad    | −2.5 st     | 0.80×       | 0.85×       | Melancholic | High (0.70)          | Slow, flat, hushed             |
| 😠 Angry  | +4 st       | 1.15×       | 1.25×       | Sharp       | Very low (0.20)      | Loud, tense, variable          |
| 😟 Concerned | −1 st  | 0.90×       | 0.92×       | Worried     | Medium (0.55)        | Careful, measured, soft        |
| 😐 Neutral | 0 st       | 1.00×       | 1.00×       | Neutral     | 0.50                 | Balanced, clear, professional  |

### 🔀 Intensity Scaling (Bonus Feature)

Parameters are not simply switched on/off — they are **blended** based on the model's confidence score:

```
param = neutral_value + confidence × (target_value − neutral_value)
```

- **Low confidence (40%)** → subtle emotional colouring
- **High confidence (90%)** → fully committed to the emotion
- **Capped at 95%** to retain naturalness even at maximum confidence

---

## 📁 Project Structure

```
empathy-engine/
│
├── app.py                    ← FastAPI server & API endpoints
├── emotion_detector.py       ← HuggingFace DistilBERT wrapper
├── voice_mapper.py           ← Emotion → voice parameter mapping
├── tts_engine.py             ← Multi-provider TTS (ElevenLabs/gTTS/pyttsx3)
├── config.py                 ← All configuration, keys, constants
├── requirements.txt          ← Python dependencies
│
├── static/
│   ├── index.html            ← Web UI (HTML+CSS+JS, no framework)
│   └── generated_audio/      ← Generated .mp3 / .wav files (auto-created)
│
└── README.md                 ← This file
```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+
- ~4 GB free disk space (for the DistilBERT model weights on first run)
- Internet connection (for ElevenLabs API and model download)

### 1. Clone / Navigate to the project

```bash
cd empathy-engine
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `torch` is a large package (~2 GB). If you only need CPU inference
> (no GPU), the standard PyTorch wheel works fine.

### 4. Configure API keys (optional — defaults are pre-set in config.py)

Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=sk_your_key_here
HUGGINGFACE_API_KEY=hf_your_key_here
TTS_PROVIDER=elevenlabs        # elevenlabs | gtts | pyttsx3
```

### 5. Run the server

```bash
python app.py
```

Or with Uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at **http://localhost:8000**.

---

## 🌐 Using the Web Interface

Open **http://localhost:8000** in your browser.

1. Type or paste any text into the input box (or click a sample chip)
2. Optionally select a **Voice Override** to force a specific emotion
3. Optionally select a **TTS Engine** (default: ElevenLabs)
4. Click **Generate Voice** (or press `Ctrl+Enter`)
5. The UI shows:
   - Detected emotion + confidence
   - Emotion distribution bar chart
   - Voice parameters used
   - Audio player with download link
   - Generated SSML markup

---

## 📡 API Reference

### `POST /generate-voice`

Convert text to emotionally expressive speech.

**Request Body:**

```json
{
  "text": "I can't believe this happened. It's absolutely devastating.",
  "voice_style": null,
  "provider": null
}
```

| Field         | Type   | Required | Description                                          |
|---------------|--------|----------|------------------------------------------------------|
| `text`        | string | ✅       | Text to synthesise (1–5000 characters)               |
| `voice_style` | string | ❌       | Force emotion: `happy`\|`sad`\|`angry`\|`concerned`\|`neutral` |
| `provider`    | string | ❌       | TTS engine: `elevenlabs`\|`gtts`\|`pyttsx3`          |

**Response:**

```json
{
  "text": "I can't believe this happened. It's absolutely devastating.",
  "emotion": "sad",
  "confidence": 0.9123,
  "raw_label": "sadness",
  "all_scores": {
    "sad": 0.9123,
    "angry": 0.0412,
    "neutral": 0.0231,
    "concerned": 0.0178,
    "happy": 0.0056
  },
  "voice_params": {
    "emotion": "sad",
    "pitch": -2.317,
    "speaking_rate": 0.818,
    "volume": 0.886,
    "tone": "melancholic",
    "stability": 0.682,
    "similarity_boost": 0.704,
    "style": 0.091,
    "confidence": 0.9123,
    "description": "Lower pitch, slower rate, softer volume – conveys sorrow and grief."
  },
  "audio_url": "/static/generated_audio/audio_1710234567_a1b2c3d4.mp3",
  "ssml": "<speak><prosody pitch=\"-13.9%\" rate=\"0.82\" volume=\"-0.7dB\">I can't believe...</prosody></speak>",
  "processing_time_ms": 1243.6
}
```

### `GET /emotion-map`

Returns the full emotion → voice parameter mapping table.

### `GET /health`

Liveness probe. Returns `{"status": "ok", "version": "1.0.0", "tts_provider": "elevenlabs"}`.

### Interactive API Docs

- Swagger UI: **http://localhost:8000/docs**
- ReDoc:       **http://localhost:8000/redoc**

---

## 🧪 Example API Requests

Using `curl`:

```bash
# Happy sentence
curl -X POST http://localhost:8000/generate-voice \
  -H "Content-Type: application/json" \
  -d '{"text": "I just got promoted! This is the best day of my life!"}'

# Force sad emotion
curl -X POST http://localhost:8000/generate-voice \
  -H "Content-Type: application/json" \
  -d '{"text": "Everything feels grey today.", "voice_style": "sad"}'

# Use gTTS as the engine
curl -X POST http://localhost:8000/generate-voice \
  -H "Content-Type: application/json" \
  -d '{"text": "Are you okay? I am worried about you.", "provider": "gtts"}'
```

Using Python:

```python
import requests

resp = requests.post(
    "http://localhost:8000/generate-voice",
    json={"text": "This is absolutely unacceptable! I demand an explanation!"}
)
data = resp.json()
print(f"Emotion : {data['emotion']} ({data['confidence']:.0%})")
print(f"Audio   : http://localhost:8000{data['audio_url']}")
```

---

## 🎁 Bonus Features Implemented

| Feature                     | Location               | Description                                          |
|-----------------------------|------------------------|------------------------------------------------------|
| Emotion confidence scores   | API response           | Full probability distribution over all emotions      |
| Intensity scaling           | `voice_mapper.py`      | Params blended with confidence for natural gradation |
| SSML generation             | `tts_engine.py`        | Full SSML prosody markup in every response           |
| Emotion distribution chart  | Web UI                 | Bar chart of all emotion probabilities               |
| Voice style override        | API + UI               | Force a specific emotion regardless of detection     |
| TTS provider selection      | API + UI               | Switch engines per request                           |
| Provider fallback cascade   | `tts_engine.py`        | Auto-retry with next provider on failure             |
| Offline TTS (pyttsx3)       | `tts_engine.py`        | Works without internet or API key                    |

---

## ⚙️ Configuration Reference (`config.py`)

| Key                     | Default                              | Description                   |
|-------------------------|--------------------------------------|-------------------------------|
| `ELEVENLABS_API_KEY`    | (pre-set)                            | ElevenLabs API key            |
| `HUGGINGFACE_API_KEY`   | (pre-set)                            | HuggingFace API key           |
| `TTS_PROVIDER`          | `"elevenlabs"`                       | Active TTS engine             |
| `EMOTION_MODEL_NAME`    | `"bhadresh-savani/distilbert-base-uncased-emotion"` | HF model |
| `AUDIO_FORMAT`          | `"mp3"`                              | Output format                 |
| `PORT`                  | `8000`                               | Server port                   |

---

## 🔒 Security Note

API keys in `config.py` are provided for convenience during evaluation. In production, always load secrets from environment variables or a secrets manager — never hard-code them in source files.

---

## 📄 License

MIT — free to use, modify, and distribute.
