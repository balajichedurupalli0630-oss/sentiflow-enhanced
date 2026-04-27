# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

SentiFlow is a real-time emotion intelligence system consisting of:
- **FastAPI backend** — analyzes text for emotion, sentiment, formality, and clarity
- **Chrome browser extension** — detects text inputs on any website and shows a floating analysis panel
- **Docker deployment** — backend with Redis for rate limiting

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The server starts on `http://localhost:8000`. Health check at `GET /health`.

## Running with Docker

```bash
cd docker
docker compose up --build
```

This starts both `sentiflow_backend` and `redis` containers. The backend connects to Redis automatically when `REDIS_URL` is set.

## ML Model Architecture

The emotion analyzer (`emotion_analyzer.py`) has transitioned from a 3-model ensemble to a **unified DeBERTa-v3 Multi-Label pipeline** for higher accuracy and better calibration in corporate contexts:

1. **Model**: Fine-tuned `DeBERTa-v3-base` (or small) multi-label classifier.
2. **Calibration**: Temperature scaling applied to logits before L1 normalization.
3. **Labels**: Plutchik-inspired 8-label space (Joy, Sadness, Anger, Fear, Surprise, Disgust, Trust, Anticipation).
4. **Corporate Tuning**: Specialized weights for professional feedback (e.g., detecting frustration in polite language).

Calibration rules:
- **Temperature Scaling**: Fitted on validation data to ensure scores reflect real probabilities.
- **L1 Normalization**: Used instead of Softmax to allow for multi-label intensity detection.
- **Confidence Ceiling**: Capped at 88% to prevent over-confidence on short snippets.
- **Mixed Emotion Penalty**: Scores are penalized by 35% when conflicting signals (e.g., Joy vs Anger) are detected simultaneously.
- **Active Thresholds**: Per-emotion thresholds defined in `calibration.json`.

## Key API Endpoints

- `POST /analyze` — HTTP analysis (rate-limited per IP)
- `GET /ws/analyze` — WebSocket for real-time streaming (max 5 connections per IP, 90s ping timeout)
- `GET /health` — Health check

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `USE_GPU` | `false` | Enable GPU inference |
| `ALLOWED_ORIGIN` | `*` | CORS origin (required in production) |
| `RATE_LIMIT_RPM` | `60` | Requests per minute per IP |
| `MAX_WS_PER_IP` | `5` | Max WebSocket connections per IP |
| `REDIS_URL` | `""` | Redis URL for distributed rate limiting |
| `LOG_LEVEL` | `warning` | Logging level |

## Rate Limiting

- Redis-backed when `REDIS_URL` is set (multi-worker production via Gunicorn)
- In-memory fallback when Redis is unavailable (dev mode)
- Fails open — Redis outages allow requests through

## Docker Model Caching

ML model weights are baked into the Docker image at build time (`docker/Dockerfile` layer 2). This avoids ~10 minute cold-start downloads. Rebuild the image to update models.

## Browser Extension

- Loads as an **unpacked Chrome extension** from `browser-extension/`
- `content.js` injects a floating panel into all pages
- Connects via WebSocket to `ws://localhost:8000/ws/analyze`
- Detects `INPUT`, `TEXTAREA`, and `contenteditable` elements
- Guards against password/secret fields via blocklist

## Key File Locations

- Backend entry: `backend/main.py`
- ML pipeline: `backend/emotion_analyzer.py`
- Docker compose: `docker/docker-compose.yml`
- Nginx config: `docker/nginx.conf`
- Extension manifest: `browser-extension/manifest.json`
