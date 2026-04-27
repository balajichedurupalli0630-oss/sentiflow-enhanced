# 🌊 SentiFlow Enhanced 4.0

![SentiFlow Intelligence Dashboard](/Users/chinnu/.gemini/antigravity/brain/aa91d97a-8243-4a21-a19c-53705712cf11/sentiflow_header_1777277251007.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-05998b.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed.svg)](https://www.docker.com/)

**Real-time emotion intelligence for professional communication.** SentiFlow Enhanced is a state-of-the-art multi-label emotion analysis system designed to detect nuance in corporate and professional dialogue.

---

## 🚀 Key Features

- **🧠 DeBERTa-v3 Multi-Label Engine**: Transitioned from ensembles to a unified, fine-tuned DeBERTa-v3 model for superior accuracy and lower latency.
- **🎯 Calibrated Confidence**: Features temperature scaling and per-emotion activation thresholds to eliminate false positives in professional feedback.
- **💼 Corporate Tone Tuning**: Specialized training to detect frustration and urgency in polite, non-profane language (e.g., "I'm concerned about the timeline" vs "I'm angry").
- **⚡ Real-Time WebSockets**: Instant intelligence as you type in any browser input field via the integrated extension.
- **🛡️ Privacy First**: Local infrastructure analysis; sensitive fields (passwords, cards) are automatically protected.

---

## 🏗️ Architecture

SentiFlow 4.0 centers on a **Unified Multi-Label Pipeline** that detects up to 8 distinct emotional dimensions simultaneously.

### The Model: DeBERTa-v3 Multi-Label
Our flagship model is a fine-tuned **DeBERTa-v3** architecture trained on an oversampled corpus of corporate communications and the GoEmotions dataset.

| Feature | Implementation | Benefit |
|---|---|---|
| **Multi-Label** | BCE Loss with Logits | Detects mixed emotions (e.g., Joy + Surprise) |
| **Calibration** | Temperature Scaling | Mathematically grounded confidence scores |
| **Normalization** | L1 Normalization | Realistic probability spread across labels |
| **Penalty Layer** | Mixed Emotion Penalty | Reduces confidence when signals are conflicting |

### Emotion Dimensions
The system maps text onto the **Plutchik-inspired** SentiFlow 8:
- **Joy** 😊 | **Trust** 🤝 | **Anticipation** ⏳ | **Surprise** 😲
- **Sadness** 😢 | **Anger** 😡 | **Fear** 😨 | **Disgust** 🤢

---

## 🛠️ Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Start with DeBERTa Multi-Label (Recommended)
SENTIFLOW_ANALYZER=deberta_multilabel python main.py
```
The API starts at `http://localhost:8000`.

### 2. Browser Extension
1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the `browser-extension/` directory.

### 3. Docker Deployment
```bash
cd docker
docker compose up --build
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `SENTIFLOW_ANALYZER` | `default` | Set to `deberta_multilabel` for high accuracy |
| `USE_GPU` | `false` | Enable CUDA/MPS inference |
| `RATE_LIMIT_RPM` | `60` | Max requests per minute per IP |
| `REDIS_URL` | `""` | Redis connection string for production rate limiting |

---

## 📊 Performance
SentiFlow 4.0 achieves a **Macro F1 of 0.6297**, significantly outperforming baseline models on professional communication datasets.

- **Latency**: <45ms (GPU) / <120ms (CPU)
- **Calibration**: Temperature-fitted on validation split for reliable probability estimates.

---

## 📜 License
MIT License - Open for modification and professional use.

---
*Maintained by the SentiFlow Team*
