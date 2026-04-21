# 🌊 SentiFlow Enhanced

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-05998b.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed.svg)](https://www.docker.com/)

**Real-time emotion intelligence for professional communication.** SentiFlow Enhanced is a highly accurate, multi-model ensemble system that analyzes text sentiment, formality, and clarity across any website via a seamless browser extension.

---

## 🚀 Key Features

- **🧠 Triple-Model Ensemble**: Uses DistilRoBERTa, GoEmotions, and Twitter-RoBERTa for ultra-precise emotion and sentiment mapping.
- **⚡ Real-Time WebSockets**: Instant feedback as you type in any input field or text area.
- **🛡️ Privacy First**: Analysis is done on your own infrastructure; sensitive fields (passwords, cards) are automatically excluded.
- **🧩 Browser Integration**: Floating intelligence panel that works on Gmail, LinkedIn, Slack, and more.
- **🐳 Production Ready**: Fully containerized with Redis-backed rate limiting and Nginx reverse proxy.

---

## 🏗️ Architecture

SentiFlow's intelligence is powered by a weighted ensemble pipeline:

| Model | Weight | Purpose |
|---|---|---|
| **DistilRoBERTa (j-hartmann)** | 60% | Primary 6-emotion classification |
| **RoBERTa-Base (GoEmotions)** | 40% | Granular detection (Trust/Anticipation) |
| **Twitter-RoBERTa** | 100% | Dedicated sentiment (Positive/Negative/Neutral) |

### Intelligence Features:
- **Negation Handling**: Automatically flips emotion scores when negations ("not happy", "never liked") are detected.
- **Confidence Ceiling**: Applies a dynamic ceiling based on text length (prevents over-confidence on 1-2 word snippets).
- **Tone Advice**: Provides actionable suggestions to reframe messages that read as angry, critical, or concerned.

---

## 🛠️ Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```
The API starts at `http://localhost:8000`.

### 2. Browser Extension
1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and select the `/browser-extension` folder in this repo.

### 3. Docker Deployment (Recommended)
```bash
cd docker
docker compose up --build
```

---

## ⚙️ Configuration

Set these in `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `USE_GPU` | `false` | Enable CUDA inference (requires NVIDIA GPU) |
| `ALLOWED_ORIGIN` | `*` | CORS origin for the extension |
| `RATE_LIMIT_RPM` | `60` | Max requests per minute per IP |
| `REDIS_URL`| `""` | Connection string for Redis rate-limiting |

---

## 📂 Project Structure

- `/backend`: FastAPI server & ML pipeline (`emotion_analyzer.py`)
- `/browser-extension`: JavaScript/CSS for site-wide integration
- `/docker`: Production-grade deployment files
- `fineTune.py`: Scripts for refining models on custom datasets

---

## 📜 License

MIT License - feel free to use, modify, and distribute.

---
*Created by [balajichedurupalli0630-oss](https://github.com/balajichedurupalli0630-oss)*
