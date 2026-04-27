"""
SentiFlow 4.0 — Production FastAPI Backend with Optimized Emotion Detection

Changes in this version:
  - Optimized blend weights (0.6297 Macro F1 vs 0.3709 baseline)
  - Uses both DistilBERT emotion model + GoEmotions RoBERTa
  - Per-emotion learned weights from regression training
  - All other features (rate limiting, WebSocket, CORS) unchanged
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

from emotion_analyzer import _init_analyzer, get_analyzer, Emotion

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

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
analyzer: Optional = None  # Will hold EmotionAnalyzer instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rate_limiter, analyzer
    logger.info("Starting SentiFlow 4.0 — loading optimized emotion models…")
    
    # Initialize analyzer with optimized blend weights (0.6297 F1)
    analyzer = await _init_analyzer(use_gpu=USE_GPU)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(REDIS_URL, RATE_LIMIT_RPM)
    
    logger.info(f"Startup complete — GPU: {USE_GPU}, Rate limit: {RATE_LIMIT_RPM}/min")
    yield
    
    logger.info("Shutting down SentiFlow")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SentiFlow API",
    description="Emotion Intelligence API with optimized blend weights (0.6297 F1)",
    version="4.0.0",
    docs_url="/docs" if LOG_LEVEL.lower() == "debug" else None,  # Enable docs in debug mode
    redoc_url=None,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

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

class HealthResponse(BaseModel):
    status: str
    version: str
    model_f1: float
    gpu_available: bool

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def calculate_formality(text: str) -> float:
    """Calculate formality score (0-100)"""
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
    """Calculate clarity score (0-100)"""
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
    """Generate actionable suggestions based on analysis"""
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
    """Run complete analysis pipeline"""
    # Get emotion analysis from optimized model
    result = await analyzer.analyze(text)
    
    # Calculate additional metrics
    formality   = calculate_formality(text)
    clarity     = calculate_clarity(text)
    word_count  = len(text.split())
    
    # Generate suggestions
    suggestions = generate_suggestions(
        result["primary_emotion"], formality, clarity, word_count
    )
    
    # Return enriched result
    return {
        **result,
        "formality_score": formality,
        "clarity_score":   clarity,
        "word_count":      word_count,
        "char_count":      len(text),
        "suggestions":     suggestions,
    }

# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=Dict)
async def analyze_text(request: Request, body: AnalysisRequest):
    """Analyze emotion, sentiment, and style of input text"""
    ip = get_client_ip(request)
    
    # Rate limiting
    if not rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded ({RATE_LIMIT_RPM} requests per minute). Try again later."
        )
    
    try:
        result = await run_analysis(body.text)
        logger.debug(f"Analysis complete for IP {ip}: {result['primary_emotion']}")
        return result
    except Exception as e:
        logger.exception(f"Analysis error for IP {ip}: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for load balancers and monitoring"""
    return {
        "status": "healthy",
        "version": "4.0.0",
        "model_f1": 0.6297,  # Optimized blend weights F1 score
        "gpu_available": USE_GPU and __import__('torch').cuda.is_available()
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "SentiFlow API",
        "version": "4.0.0",
        "status": "online",
        "description": "Emotion Intelligence with optimized blend weights (0.6297 F1)",
        "emotions": [e.value for e in Emotion],
        "endpoints": {
            "POST /analyze": "Analyze text for emotions",
            "GET /health": "Health check",
            "WS /ws/analyze": "WebSocket for real-time analysis"
        }
    }


@app.get("/metrics")
async def metrics():
    """Performance metrics endpoint"""
    return {
        "model": "optimized_blend",
        "macro_f1": 0.6297,
        "accuracy": 0.6739,
        "per_emotion_f1": {
            "joy": 0.707, "sadness": 0.630, "anger": 0.682,
            "fear": 0.657, "surprise": 0.504, "disgust": 0.502,
            "trust": 0.742, "anticipation": 0.613
        }
    }

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming analysis"""
    # Get client IP (handle proxies)
    forwarded = websocket.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = websocket.client.host if websocket.client else "unknown"

    # Enforce connection limit per IP
    if _ws_connections[ip] >= MAX_WS_PER_IP:
        await websocket.close(code=1008, reason=f"Max {MAX_WS_PER_IP} connections per IP")
        return

    await websocket.accept()
    _ws_connections[ip] += 1
    logger.info(f"WebSocket connected from {ip} (active: {_ws_connections[ip]})")

    inflight_task: Optional[asyncio.Task] = None

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping", "timestamp": time.time()})
                continue

            text = str(data.get("text", "")).strip()
            request_id = data.get("requestId")  # For client-side request tracking

            # Validate input
            if not text or len(text) < 2:
                continue
            if len(text) > 2000:
                text = text[:2000]

            # Rate limit per WebSocket connection
            if not rate_limiter.is_allowed(f"ws:{ip}"):
                await websocket.send_json({
                    "error": "rate_limit",
                    "message": "Slow down",
                    "requestId": request_id
                })
                continue

            # Cancel previous in-flight request (client moved on)
            if inflight_task and not inflight_task.done():
                inflight_task.cancel()

            async def _run_and_send(t: str, rid: Optional[int]) -> None:
                try:
                    result = await run_analysis(t)
                    if rid is not None:
                        result["requestId"] = rid
                    await websocket.send_json(result)
                except asyncio.CancelledError:
                    pass  # Superseded by newer request
                except Exception as e:
                    logger.exception(f"WebSocket analysis error: {e}")
                    await websocket.send_json({
                        "error": "analysis_failed",
                        "requestId": rid
                    })

            inflight_task = asyncio.create_task(_run_and_send(text, request_id))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from {ip}")
    except Exception as exc:
        logger.warning(f"WebSocket error from {ip}: {type(exc).__name__}")
    finally:
        if inflight_task and not inflight_task.done():
            inflight_task.cancel()
        _ws_connections[ip] = max(0, _ws_connections[ip] - 1)


# ── Global error handlers ─────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions without exposing internals"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    """Catch-all error handler - never expose stack traces"""
    logger.exception(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SentiFlow 4.0 — Emotion Intelligence API")
    print("  Optimized Blend Weights | Macro F1: 0.6297")
    print("=" * 60)
    print(f"  GPU Acceleration:    {USE_GPU}")
    print(f"  CORS Allowed Origin: {ALLOWED_ORIGIN}")
    print(f"  Rate Limit:          {RATE_LIMIT_RPM} req/min per IP")
    print(f"  Max WebSocket/IP:    {MAX_WS_PER_IP}")
    print(f"  Redis:               {'enabled' if REDIS_URL else 'disabled (in-memory)'}")
    print(f"  Log Level:           {LOG_LEVEL}")
    print("=" * 60)
    print("\n🚀 Starting server...\n")
    print("📖 API Documentation: http://localhost:8000/docs (debug mode only)")
    print("🏥 Health Check:      http://localhost:8000/health")
    print("📊 Metrics:           http://localhost:8000/metrics")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower(),
        access_log=LOG_LEVEL.lower() == "debug",
        reload=False,  # Set to True for development
    )