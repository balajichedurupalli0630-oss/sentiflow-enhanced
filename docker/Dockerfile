# SentiFlow 3.0 — Production Dockerfile
#
# Models are downloaded during BUILD, not at runtime.
# Cold-start time: ~4s (image load) vs ~10min (downloading at startup).
#
# Build: docker build -t sentiflow:3.0 .
# Run:   docker run -p 8000:8000 --env-file .env sentiflow:3.0

FROM python:3.11-slim

# System deps for torch (libgomp needed for CPU inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Python dependencies (cached unless requirements.txt changes) ──────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: Bake model weights into the image ─────────────────────────────────
# All three models are downloaded to /app/.cache so they're available at startup
# without any network access.
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

RUN python -c "\
from transformers import pipeline; \
print('Downloading j-hartmann model...'); \
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None, device=-1); \
print('Downloading go_emotions model...'); \
pipeline('text-classification', model='SamLowe/roberta-base-go_emotions', top_k=None, device=-1); \
print('Downloading sentiment model...'); \
pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-sentiment-latest', top_k=None, device=-1); \
print('All models cached successfully.')"

# ── Layer 3: Application code ──────────────────────────────────────────────────
COPY . .

# Non-root user — security best practice
RUN useradd -m -u 1001 sentiflow && chown -R sentiflow:sentiflow /app
USER sentiflow

EXPOSE 8000

# Healthcheck — nginx and orchestrators use this
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Gunicorn with uvicorn workers.
# Keep workers low for ML models — each worker loads a full copy into RAM.
# Rule of thumb: 1-2 workers per model replica.
CMD ["gunicorn", "main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--max-requests", "500", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]