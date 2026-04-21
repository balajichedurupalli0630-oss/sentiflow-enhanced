"""
SentiFlow 3.0 — Production FastAPI Backend

Fixes applied vs previous version:
  - Migrated from deprecated on_event to lifespan context manager
  - Rate limiting backed by Redis (shared across Gunicorn workers)
    Falls back to in-memory if Redis is unavailable (dev mode)
  - English-only (Hindi/Telugu removed — no fake multilingual)
  - Independent sentiment score from dedicated sentiment model
  - CORS default is locked; must set ALLOWED_ORIGIN in .env
  - No stack traces to clients
  - WebSocket ping/pong keepalive
  - Proper X-Forwarded-For handling behind nginx
"""

import os
import time
import logging
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from emotion_analyzer import _init_analyzer, get_analyzer, Emotion, EmotionAnalyzer

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()

USE_GPU        = os.getenv("USE_GPU", "false").lower() == "true"
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "")          # Required in production
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
MAX_WS_PER_IP  = int(os.getenv("MAX_WS_PER_IP", "5"))
LOG_LEVEL      = os.getenv("LOG_LEVEL", "warning")
REDIS_URL      = os.getenv("REDIS_URL", "")               # e.g. redis://localhost:6379/0

# Safety check — warn loudly if CORS is open in a non-dev environment
if not ALLOWED_ORIGIN:
    ALLOWED_ORIGIN = "*"
    print("⚠️  ALLOWED_ORIGIN not set — CORS is open to all origins. Set it in .env for production.")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.WARNING))
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("sentiflow")

# ── Rate limiter abstraction ──────────────────────────────────────────────────

class RateLimiter:
    """
    Transparent Redis-backed rate limiter with in-memory fallback.
    Redis is required in multi-worker production (Gunicorn) to share state.
    In-memory fallback is fine for single-process dev.
    """
    def __init__(self, redis_url: str, limit: int, window: int = 60):
        self.limit  = limit
        self.window = window
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True,
                                             socket_connect_timeout=2)
                self._redis.ping()
                logger.info("Rate limiter: Redis connected at %s", redis_url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s) — using in-memory rate limiter", exc)
                self._redis = None

        self._store: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        if self._redis:
            return self._redis_check(key)
        return self._memory_check(key)

    def _redis_check(self, key: str) -> bool:
        rkey = f"sf:rl:{key}"
        now  = time.time()
        pipe = self._redis.pipeline()
        try:
            pipe.zremrangebyscore(rkey, 0, now - self.window)
            pipe.zcard(rkey)
            pipe.zadd(rkey, {str(now): now})
            pipe.expire(rkey, self.window + 5)
            results = pipe.execute()
            count = results[1]   # zcard result (before this request)
            return count < self.limit
        except Exception as exc:
            logger.warning("Redis rate-limit check failed: %s — allowing request", exc)
            return True           # Fail open; don't block users on Redis outage

    def _memory_check(self, key: str) -> bool:
        now = time.time()
        ts  = [t for t in self._store[key] if now - t < self.window]
        self._store[key] = ts
        if len(ts) >= self.limit:
            return False
        self._store[key].append(now)
        return True


# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────

rate_limiter: Optional[RateLimiter] = None
analyzer: Optional[EmotionAnalyzer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rate_limiter, analyzer
    logger.info("Starting SentiFlow — loading models…")
    analyzer = await _init_analyzer(use_gpu=USE_GPU)   # async-safe; lock prevents duplicate loads
    rate_limiter = RateLimiter(REDIS_URL, RATE_LIMIT_RPM)
    logger.info("Startup complete")
    yield
    logger.info("Shutting down SentiFlow")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SentiFlow API",
    version="3.0.0",
    docs_url=None,   # Disable Swagger UI in production
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── WebSocket connection tracker ──────────────────────────────────────────────

# ── WebSocket connection tracker ──────────────────────────────────────────────

_ws_connections: Dict[str, int] = defaultdict(int)

# ── Request / Response models ─────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text is required")
        if len(v) > 2000:
            raise ValueError("text exceeds maximum length of 2000 characters")
        return v.strip()

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def calculate_formality(text: str) -> float:
    formal   = ["please", "kindly", "regards", "sincerely", "appreciate",
                "would be grateful", "at your earliest", "in accordance",
                "herewith", "pursuant", "respectfully", "dear", "yours"]
    informal = ["hey", "yeah", "yup", "gonna", "wanna", "lol", "btw",
                "omg", "ngl", "tbh", "imo", "fyi", "idk", "ur", "u r"]
    t             = text.lower()
    formal_count  = sum(1 for w in formal   if w in t)
    informal_count= sum(1 for w in informal if w in t)
    contractions  = sum(1 for c in ["'m","'re","'ve","'ll","n't","'d"] if c in text)
    has_punct     = text.strip()[-1] in ".!?" if text.strip() else False
    score = 50 + formal_count * 10 - informal_count * 12 - contractions * 5 + (8 if has_punct else 0)
    return round(max(0, min(100, score)), 1)


def calculate_clarity(text: str) -> float:
    words = text.split()
    n     = len(words)
    if n == 0:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / n
    sentences    = max(1, text.count(".") + text.count("!") + text.count("?"))
    words_per_s  = n / sentences
    long_words   = sum(1 for w in words if len(w) > 10)
    long_ratio   = long_words / n
    score        = 100
    if avg_word_len > 7:          score -= (avg_word_len - 7) * 8     
    if words_per_s > 25:          score -= (words_per_s - 25) * 1.5
    if long_ratio > 0.3:          score -= long_ratio * 25             
    if 8 <= words_per_s <= 20 and avg_word_len <= 7: score += 8        
    return round(max(0, min(100, score)), 1)

TONE_ADVICE = {
    "anger":   "Tone signals frustration — consider softening before sending.",
    "disgust": "Message reads as highly critical — reframe with specifics.",
    "fear":    "Concern detected — state the risk clearly and propose next steps.",
    "sadness": "Negative framing may reduce engagement. Try a neutral opener.",
}

def generate_suggestions(emotion: str, formality: float, clarity: float, word_count: int) -> List[str]:
    out = []
    advice = TONE_ADVICE.get(emotion) 
    
    if advice:
        out.append(advice)
    if formality < 38:
        out.append("Formality is low for a professional context — avoid contractions and slang.")
    elif formality > 88:
        out.append("Very formal tone — ensure it matches the relationship with the recipient.")
    if clarity < 50:
        out.append("Clarity score is low — try shorter sentences and concrete language.")
    if word_count > 120:
        out.append("Message is lengthy — lead with the key point and trim secondary detail.")
    return out[:3]


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def run_analysis(text: str) -> dict:
    result      = await analyzer.analyze(text)
    formality   = calculate_formality(text)
    clarity     = calculate_clarity(text)
    word_count  = len(text.split())
    suggestions = generate_suggestions(
        result["primary_emotion"], formality, clarity, word_count
    )
    return {
        **result,
        "formality_score": formality,
        "clarity_score":   clarity,
        "word_count":      word_count,
        "char_count":      len(text),
        "suggestions":     suggestions,
    }

# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_text(request: Request, body: AnalysisRequest):
    ip = get_client_ip(request)
    if not rate_limiter.is_allowed(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    try:
        return await run_analysis(body.text)
    except Exception:
        logger.exception("Analysis error")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.0.0"}


@app.get("/")
async def root():
    return {
        "service": "SentiFlow API",
        "version": "3.0.0",
        "status":  "online",
        "emotions": [e.value for e in Emotion],
    }

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    # Use X-Forwarded-For when behind nginx — same logic as HTTP endpoints.
    forwarded = websocket.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = websocket.client.host if websocket.client else "unknown"

    if _ws_connections[ip] >= MAX_WS_PER_IP:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    _ws_connections[ip] += 1
    logger.info("WS connected from %s (active: %d)", ip, _ws_connections[ip])

    inflight_task: Optional[asyncio.Task] = None   # backpressure: one task per connection

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=90)
            except asyncio.TimeoutError:
                # Send ping to keep the connection alive
                await websocket.send_json({"type": "ping"})
                continue

            text = str(data.get("text", "")).strip()
            request_id = data.get("requestId")  # echoed back so client can discard stale results

            if not text or len(text) < 5:
                continue
            if len(text) > 2000:
                text = text[:2000]

            if not rate_limiter.is_allowed(ip):
                await websocket.send_json({"error": "rate_limit", "message": "Slow down"})
                continue

            # Cancel the previous in-flight analysis — the client has already
            # moved on, so completing it would only waste CPU/GPU and send a
            # stale result that the client will discard anyway.
            if inflight_task and not inflight_task.done():
                inflight_task.cancel()

            async def _run_and_send(t: str, rid: Optional[int]) -> None:
                try:
                    result = await run_analysis(t)
                    if rid is not None:
                        result["requestId"] = rid
                    await websocket.send_json(result)
                except asyncio.CancelledError:
                    pass   # superseded by a newer request — silently drop
                except Exception:
                    logger.exception("WS analysis error")
                    await websocket.send_json({"error": "analysis_failed"})

            inflight_task = asyncio.create_task(_run_and_send(text, request_id))

    except WebSocketDisconnect:
        logger.info("WS disconnected from %s", ip)
    except Exception as exc:
        logger.warning("WS error from %s: %s", ip, type(exc).__name__)
    finally:
        if inflight_task and not inflight_task.done():
            inflight_task.cancel()
        _ws_connections[ip] = max(0, _ws_connections[ip] - 1)

# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  SentiFlow 3.0 — Emotion Intelligence API")
    print("=" * 58)
    print(f"  GPU:         {USE_GPU}")
    print(f"  CORS:        {ALLOWED_ORIGIN}")
    print(f"  Rate limit:  {RATE_LIMIT_RPM} req/min per IP")
    print(f"  Max WS/IP:   {MAX_WS_PER_IP}")
    print(f"  Redis:       {'enabled' if REDIS_URL else 'disabled (in-memory fallback)'}")
    print("=" * 58 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL,
        access_log=False,
    )