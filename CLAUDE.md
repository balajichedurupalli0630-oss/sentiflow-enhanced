# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

The emotion analyzer (`emotion_analyzer.py`) uses an **ensemble of 3 transformer models**:

1. `j-hartmann/emotion-english-distilroberta-base` — 6 emotions, primary source (60% weight)
2. `SamLowe/roberta-base-go_emotions` — 28 emotions, used for Trust/Anticipation derivation (40% weight)
3. `cardiffnlp/twitter-roberta-base-sentiment-latest` — independent sentiment score (decoupled from emotion)

Blending rules:
- Trust and Anticipation are model-derived (not heuristic): 30% Hartmann + 70% go_emotions
- All other emotions: 60% Hartmann + 40% go_emotions
- Short-text confidence ceiling applied (55% for <3 words → 97% for 15+ words)
- Negation detection flips affected emotion scores to opposites
- Intensifiers/diminishers adjust scores within confidence ceiling

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
