"""
SentiFlow 3.0 — Emotion Analyzer (English-only, production)

Changes in this version:
  - Blend weights are now loaded from blend_weights.json (produced by
    train_blend_weights.py) instead of hardcoded 60/40 constants.
  - Falls back to the original fixed weights if no JSON file is found.
  - BLEND_WEIGHTS_PATH env var overrides the default path.
  - Everything else (negation, intensifiers, keyword boost, ceiling) unchanged.
"""

import re
import asyncio
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline
import os
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger("sentiflow.analyzer")


# ── Emotion / Sentiment enums ──────────────────────────────────────────────────

class Emotion(str, Enum):
    JOY          = "joy"
    SADNESS      = "sadness"
    ANGER        = "anger"
    FEAR         = "fear"
    SURPRISE     = "surprise"
    DISGUST      = "disgust"
    TRUST        = "trust"
    ANTICIPATION = "anticipation"

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


EMOTION_TO_SENTIMENT: Dict[str, Sentiment] = {
    Emotion.JOY:          Sentiment.POSITIVE,
    Emotion.TRUST:        Sentiment.POSITIVE,
    Emotion.ANTICIPATION: Sentiment.POSITIVE,
    Emotion.SURPRISE:     Sentiment.NEUTRAL,
    Emotion.SADNESS:      Sentiment.NEGATIVE,
    Emotion.ANGER:        Sentiment.NEGATIVE,
    Emotion.FEAR:         Sentiment.NEGATIVE,
    Emotion.DISGUST:      Sentiment.NEGATIVE,
}

EMOTION_EMOJI: Dict[str, str] = {
    Emotion.JOY:          "😊",
    Emotion.SADNESS:      "😢",
    Emotion.ANGER:        "😠",
    Emotion.FEAR:         "😨",
    Emotion.SURPRISE:     "😲",
    Emotion.DISGUST:      "🤢",
    Emotion.TRUST:        "🤝",
    Emotion.ANTICIPATION: "🤗",
}

EMOTION_COLOR: Dict[str, str] = {
    Emotion.JOY:          "#d97706",
    Emotion.SADNESS:      "#3b82f6",
    Emotion.ANGER:        "#ef4444",
    Emotion.FEAR:         "#a855f7",
    Emotion.SURPRISE:     "#db2777",
    Emotion.DISGUST:      "#84cc16",
    Emotion.TRUST:        "#22c55e",
    Emotion.ANTICIPATION: "#f97316",
}


# ── Model label mappings ───────────────────────────────────────────────────────

HARTMANN_LABEL_MAP: Dict[str, str] = {
    "joy":      Emotion.JOY,
    "sadness":  Emotion.SADNESS,
    "anger":    Emotion.ANGER,
    "fear":     Emotion.FEAR,
    "surprise": Emotion.SURPRISE,
    "disgust":  Emotion.DISGUST,
    "love":     Emotion.JOY,
    "neutral":  Emotion.SURPRISE,
}

GO_EMOTIONS_MAP: Dict[str, Tuple[str, float]] = {
    "admiration":    (Emotion.TRUST,        0.85),
    "approval":      (Emotion.TRUST,        0.75),
    "caring":        (Emotion.TRUST,        0.65),
    "gratitude":     (Emotion.TRUST,        0.80),
    "relief":        (Emotion.TRUST,        0.60),
    "excitement":    (Emotion.ANTICIPATION, 0.85),
    "optimism":      (Emotion.ANTICIPATION, 0.90),
    "desire":        (Emotion.ANTICIPATION, 0.70),
    "curiosity":     (Emotion.ANTICIPATION, 0.60),
    "joy":           (Emotion.JOY,          0.90),
    "amusement":     (Emotion.JOY,          0.75),
    "pride":         (Emotion.JOY,          0.70),
    "love":          (Emotion.JOY,          0.85),
    "sadness":       (Emotion.SADNESS,      0.90),
    "grief":         (Emotion.SADNESS,      0.90),
    "disappointment":(Emotion.SADNESS,      0.80),
    "remorse":       (Emotion.SADNESS,      0.75),
    "anger":         (Emotion.ANGER,        0.90),
    "annoyance":     (Emotion.ANGER,        0.75),
    "disgust":       (Emotion.DISGUST,      0.90),
    "fear":          (Emotion.FEAR,         0.90),
    "nervousness":   (Emotion.FEAR,         0.80),
    "surprise":      (Emotion.SURPRISE,     0.90),
    "confusion":     (Emotion.SURPRISE,     0.60),
    "embarrassment": (Emotion.FEAR,         0.55),
    "realization":   (Emotion.SURPRISE,     0.65),
}


# ── NLP helpers ────────────────────────────────────────────────────────────────

NEGATION_WORDS = frozenset({
    "not", "no", "never", "nobody", "nothing", "neither", "nowhere",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
    "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
})

INTENSIFIERS: Dict[str, float] = {
    "very": 1.25, "extremely": 1.45, "absolutely": 1.45, "totally": 1.30,
    "incredibly": 1.40, "really": 1.20, "so": 1.15, "deeply": 1.30,
    "utterly": 1.40, "completely": 1.35, "quite": 1.10, "rather": 1.10,
    "truly": 1.20, "highly": 1.20, "terribly": 1.35, "awfully": 1.35,
    "super": 1.25, "damn": 1.30, "bloody": 1.30,
}

DIMINISHERS: Dict[str, float] = {
    "slightly": 0.70, "a bit": 0.75, "a little": 0.75, "somewhat": 0.80,
    "kind of": 0.80, "kinda": 0.80, "sort of": 0.80, "barely": 0.65,
    "hardly": 0.60, "mildly": 0.75, "partially": 0.80,
}

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    Emotion.JOY: [
        "happy", "joy", "excited", "glad", "pleased", "delighted", "wonderful",
        "amazing", "great", "excellent", "love", "fantastic", "thrilled", "ecstatic",
        "cheerful", "blissful", "elated", "jubilant", "overjoyed", "grateful",
    ],
    Emotion.SADNESS: [
        "sad", "unhappy", "disappointed", "depressed", "miserable", "sorry",
        "regret", "unfortunate", "terrible", "awful", "heartbroken", "grief",
        "sorrow", "mourn", "despair", "hopeless", "lonely", "down",
    ],
    Emotion.ANGER: [
        "angry", "furious", "mad", "outraged", "frustrated", "annoyed",
        "irritated", "unacceptable", "ridiculous", "pathetic", "hate",
        "infuriated", "livid", "enraged", "irate", "hostile",
    ],
    Emotion.FEAR: [
        "scared", "afraid", "worried", "anxious", "nervous", "concerned",
        "frightened", "terrified", "panic", "dread", "apprehensive",
        "uneasy", "alarmed", "stressed", "overwhelmed", "threatened",
    ],
    Emotion.SURPRISE: [
        "surprised", "shocked", "amazed", "astonished", "unexpected",
        "unbelievable", "wow", "incredible", "cant believe",
        "mind blown", "stunned", "speechless",
    ],
    Emotion.DISGUST: [
        "disgusting", "awful", "horrible", "terrible", "nasty", "gross",
        "repulsive", "revolting", "appalling", "vile", "dreadful", "ghastly",
    ],
    Emotion.TRUST: [
        "trust", "reliable", "confident", "believe", "faith", "assured",
        "dependable", "honest", "credible", "secure", "safe", "certain",
        "count on", "loyal", "committed", "transparent", "consistent",
    ],
    Emotion.ANTICIPATION: [
        "excited", "eager", "looking forward", "expect", "anticipate",
        "hope", "await", "can't wait", "upcoming", "soon", "planning",
        "preparing", "ready", "hopeful", "optimistic", "future",
    ],
}

EMOTION_OPPOSITES: Dict[str, str] = {
    Emotion.JOY:          Emotion.SADNESS,
    Emotion.SADNESS:      Emotion.JOY,
    Emotion.ANGER:        Emotion.TRUST,
    Emotion.FEAR:         Emotion.TRUST,
    Emotion.TRUST:        Emotion.FEAR,
    Emotion.ANTICIPATION: Emotion.SADNESS,
    Emotion.DISGUST:      Emotion.TRUST,
    Emotion.SURPRISE:     Emotion.SURPRISE,
}


# ── Adapter / weights paths ────────────────────────────────────────────────────

ADAPTER_PATH      = os.getenv("SENTIFLOW_ADAPTER", "")
BLEND_WEIGHTS_PATH = os.getenv("SENTIFLOW_BLEND_WEIGHTS", "blend_weights.json")

# ── Fixed fallback weights (used when no JSON is found) ───────────────────────
# These replicate the original hardcoded behavior exactly.
_FIXED_FALLBACK: Dict[str, Dict[str, float]] = {
    "joy":          {"hartmann": 0.60, "go": 0.40},
    "sadness":      {"hartmann": 0.60, "go": 0.40},
    "anger":        {"hartmann": 0.60, "go": 0.40},
    "fear":         {"hartmann": 0.60, "go": 0.40},
    "surprise":     {"hartmann": 0.60, "go": 0.40},
    "disgust":      {"hartmann": 0.60, "go": 0.40},
    "trust":        {"hartmann": 0.30, "go": 0.70},
    "anticipation": {"hartmann": 0.30, "go": 0.70},
}


def _load_blend_weights() -> Dict[str, Dict[str, float]]:
    """
    Load per-emotion blend weights from blend_weights.json.
    Falls back to fixed weights if file is missing or malformed.
    Expected JSON format from train_blend_weights.py:
        { "weights": { "joy": 0.55, "sadness": 0.62, ... } }
    where each value is the Hartmann weight (go_weight = 1 - hartmann_weight).
    """
    path = Path(BLEND_WEIGHTS_PATH)
    if not path.exists():
        logger.info(
            "blend_weights.json not found at %s — using fixed fallback weights. "
            "Run train_blend_weights.py to generate learned weights.",
            BLEND_WEIGHTS_PATH,
        )
        return _FIXED_FALLBACK

    try:
        data = json.loads(path.read_text())
        raw  = data.get("weights", {})
        if not raw:
            raise ValueError("'weights' key is empty")

        loaded: Dict[str, Dict[str, float]] = {}
        for emotion, hw in raw.items():
            hw = float(hw)
            loaded[emotion] = {"hartmann": hw, "go": round(1.0 - hw, 4)}

        # Fill any missing emotions with fallback
        for em, fb in _FIXED_FALLBACK.items():
            if em not in loaded:
                loaded[em] = fb

        macro_f1 = data.get("metrics", {}).get("learned_macro_f1", "?")
        method   = data.get("method", "?")
        logger.info(
            "Loaded regression blend weights from %s (method=%s macro_F1=%s)",
            BLEND_WEIGHTS_PATH, method, macro_f1,
        )
        return loaded

    except Exception as exc:
        logger.warning(
            "Failed to parse %s (%s) — using fixed fallback weights",
            BLEND_WEIGHTS_PATH, exc,
        )
        return _FIXED_FALLBACK


# Loaded once at module import time
BLEND_WEIGHTS: Dict[str, Dict[str, float]] = _load_blend_weights()


# ── Main analyzer class ────────────────────────────────────────────────────────

class EmotionAnalyzer:
    """
    Production emotion analyzer — English only.

    Primary model:   j-hartmann/emotion-english-distilroberta-base  (6 emotions)
    Secondary model: SamLowe/roberta-base-go_emotions               (28 emotions)
    Sentiment model: cardiffnlp/twitter-roberta-base-sentiment-latest

    Blend weights are loaded from blend_weights.json (generated by
    train_blend_weights.py). Falls back to fixed 60/40 if file is missing.
    """

    def __init__(self, use_gpu: bool = False):
        device      = 0 if use_gpu and torch.cuda.is_available() else -1
        device_name = "cuda" if device == 0 else "cpu"
        logger.info("Loading models on %s…", device_name)

        self.hartmann_pipe = self._load_hartmann_pipe(device)
        self.go_pipe = self._load_pipe(
            "text-classification",
            "SamLowe/roberta-base-go_emotions",
            device, top_k=None,
        )
        self.sentiment_pipe = self._load_pipe(
            "text-classification",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            device, top_k=None,
        )

        # Log which blend weights are active
        sample = BLEND_WEIGHTS.get("joy", {})
        logger.info(
            "Blend weights active — joy example: Hartmann=%.2f GoEmotions=%.2f",
            sample.get("hartmann", 0.60),
            sample.get("go", 0.40),
        )
        logger.info("All models loaded")

    @staticmethod
    def _load_pipe(task: str, model: str, device: int, **kwargs):
        try:
            p = pipeline(task, model=model, device=device,
                         truncation=True, max_length=512, **kwargs)
            logger.info("Loaded %s", model)
            return p
        except Exception as exc:
            logger.error("Failed to load %s: %s", model, exc)
            return None

    def _load_hartmann_pipe(self, device: int):
        """
        Loads Hartmann model — fine-tuned adapter if SENTIFLOW_ADAPTER is set,
        otherwise the base HuggingFace model.
        """
        adapter = ADAPTER_PATH.strip()
        if adapter and Path(adapter).exists():
            try:
                logger.info("Loading LoRA adapter from %s", adapter)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    "j-hartmann/emotion-english-distilroberta-base",
                    num_labels=8,
                    ignore_mismatched_sizes=True,
                )
                model     = PeftModel.from_pretrained(base_model, adapter)
                tokenizer = AutoTokenizer.from_pretrained(adapter)
                if device >= 0:
                    model = model.to("cuda")
                pipe = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
                logger.info("LoRA adapter loaded successfully")
                return pipe
            except Exception as exc:
                logger.warning(
                    "Failed to load LoRA adapter (%s) — falling back to base model", exc
                )
        return self._load_pipe(
            "text-classification",
            "j-hartmann/emotion-english-distilroberta-base",
            device,
            top_k=None,
        )

    # ── Analysis pipeline ──────────────────────────────────────────────────────

    async def analyze(self, text: str) -> Dict:
        clean = self._preprocess(text)

        loop = asyncio.get_event_loop()
        hartmann_task  = loop.run_in_executor(None, self._run_hartmann,     clean)
        go_task        = loop.run_in_executor(None, self._run_go_emotions,  clean)
        sentiment_task = loop.run_in_executor(None, self._run_sentiment,    clean)

        hartmann_scores, go_scores, sentiment_result = await asyncio.gather(
            hartmann_task, go_task, sentiment_task
        )

        scores = self._blend_emotion_scores(hartmann_scores, go_scores)

        negated   = self._find_negated_emotions(clean)
        intensity = self._find_intensities(clean)

        for emotion in negated:
            spill = scores.get(emotion, 0) * 0.55
            scores[emotion] = scores.get(emotion, 0) * 0.15
            opposite = EMOTION_OPPOSITES.get(emotion, Emotion.SURPRISE)
            scores[opposite] = min(scores.get(opposite, 0) + spill, 0.98)

        for emotion, multiplier in intensity.items():
            scores[emotion] = min(scores.get(emotion, 0) * multiplier, 0.98)

        scores = self._keyword_boost(clean, scores)

        word_count = len(clean.split())
        ceiling    = self._confidence_ceiling(word_count)

        for e in Emotion:
            scores.setdefault(e.value, 0.04)

        primary       = max(scores, key=scores.__getitem__)
        primary_score = min(scores[primary], ceiling)

        pct_scores      = {
            e: round(min(v * 100, ceiling * 100), 1)
            for e, v in scores.items()
        }
        sorted_emotions = sorted(pct_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "primary_emotion":  primary,
            "emotion_score":    round(primary_score * 100, 1),
            "sentiment":        sentiment_result["label"],
            "sentiment_score":  round(sentiment_result["score"] * 100, 1),
            "emotion_scores":   pct_scores,
            "top_3_emotions":   sorted_emotions[:3],
            "emoji":            EMOTION_EMOJI.get(primary, "😐"),
            "color":            EMOTION_COLOR.get(primary, "#888"),
        }

    # ── Model runners ──────────────────────────────────────────────────────────

    def _run_hartmann(self, text: str) -> Dict[str, float]:
        scores: Dict[str, float] = {e.value: 0.04 for e in Emotion}
        if not self.hartmann_pipe:
            return scores
        try:
            results = self.hartmann_pipe(text[:512])
            items   = results[0] if isinstance(results[0], list) else results
            for item in items:
                label = item["label"].lower()
                score = float(item["score"])
                if label in HARTMANN_LABEL_MAP:
                    emotion = HARTMANN_LABEL_MAP[label]
                    scores[emotion] = max(scores.get(emotion, 0), score)
        except Exception as exc:
            logger.exception("Hartmann inference error: %s", exc)
        return scores

    def _run_go_emotions(self, text: str) -> Dict[str, float]:
        scores: Dict[str, float] = {e.value: 0.0 for e in Emotion}
        if not self.go_pipe:
            return scores
        try:
            results = self.go_pipe(text[:512])
            items   = results[0] if isinstance(results[0], list) else results
            for item in items:
                label = item["label"].lower()
                raw   = float(item["score"])
                if label in GO_EMOTIONS_MAP:
                    target, weight = GO_EMOTIONS_MAP[label]
                    scores[target] = max(scores.get(target, 0), raw * weight)
        except Exception as exc:
            logger.exception("go_emotions inference error: %s", exc)
        return scores

    def _run_sentiment(self, text: str) -> Dict:
        fallback = {"label": "neutral", "score": 0.5}
        if not self.sentiment_pipe:
            return fallback
        try:
            results = self.sentiment_pipe(text[:512])
            items   = results[0] if isinstance(results[0], list) else results
            best    = max(items, key=lambda x: x["score"])
            label   = best["label"].lower()
            if label in ("positive", "pos", "label_2"):
                label = "positive"
            elif label in ("negative", "neg", "label_0"):
                label = "negative"
            else:
                label = "neutral"
            return {"label": label, "score": float(best["score"])}
        except Exception as exc:
            logger.exception("Sentiment inference error: %s", exc)
            return fallback

    # ── Blending — uses regression-learned weights ─────────────────────────────

    def _blend_emotion_scores(
        self,
        hartmann: Dict[str, float],
        go_em:    Dict[str, float],
    ) -> Dict[str, float]:
        """
        Blends Hartmann and GoEmotions scores using per-emotion weights
        loaded from blend_weights.json (produced by train_blend_weights.py).

        If blend_weights.json is missing, falls back to the original
        fixed 60/40 (or 30/70 for trust/anticipation) weights.
        """
        blended: Dict[str, float] = {}
        all_emotions = set(hartmann) | set(go_em)
        for emotion in all_emotions:
            w  = BLEND_WEIGHTS.get(emotion, {"hartmann": 0.60, "go": 0.40})
            h  = hartmann.get(emotion, 0.0)
            g  = go_em.get(emotion, 0.0)
            blended[emotion] = h * w["hartmann"] + g * w["go"]
        return blended

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 3:
            context = " ".join(sentences[-3:])
            words   = context.split()
            if len(words) > 120:
                context = " ".join(words[-100:])
            return context.strip()
        return text

    # ── Negation / intensifiers / keywords ────────────────────────────────────

    def _find_negated_emotions(self, text: str) -> List[str]:
        words   = text.lower().split()
        negated: List[str] = []
        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w']", "", word)
            if clean_word not in NEGATION_WORDS:
                continue
            window = " ".join(words[i + 1: i + 5])
            for emotion, keywords in EMOTION_KEYWORDS.items():
                if any(kw in window for kw in keywords):
                    negated.append(emotion)
        return negated

    def _find_intensities(self, text: str) -> Dict[str, float]:
        text_lower  = text.lower()
        multipliers: Dict[str, float] = {}
        for intensifier, factor in INTENSIFIERS.items():
            for match in re.finditer(r"\b" + re.escape(intensifier) + r"\b", text_lower):
                window = text_lower[match.end(): match.end() + 40]
                for emotion, keywords in EMOTION_KEYWORDS.items():
                    if any(kw in window for kw in keywords):
                        multipliers[emotion] = max(multipliers.get(emotion, 1.0), factor)
        for diminisher, factor in DIMINISHERS.items():
            for match in re.finditer(re.escape(diminisher), text_lower):
                window = text_lower[match.end(): match.end() + 40]
                for emotion, keywords in EMOTION_KEYWORDS.items():
                    if any(kw in window for kw in keywords):
                        if multipliers.get(emotion, 1.0) == 1.0:
                            multipliers[emotion] = factor
        return multipliers

    def _keyword_boost(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        text_lower      = text.lower()
        top_model_score = max(scores.values()) if scores else 0.0
        boost_ceiling   = min(top_model_score + 0.10, 0.98)
        for emotion, keywords in EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                boost = min(matches * 0.03, 0.08)
                new_score = scores.get(emotion, 0) + boost
                scores[emotion] = min(new_score, boost_ceiling)
        return scores

    @staticmethod
    def _confidence_ceiling(word_count: int) -> float:
        if word_count < 3:  return 0.55
        if word_count < 5:  return 0.70
        if word_count < 8:  return 0.80
        if word_count < 15: return 0.90
        return 0.97

    def emoji(self, emotion: str) -> str:
        return EMOTION_EMOJI.get(emotion, "😐")

    def color(self, emotion: str) -> str:
        return EMOTION_COLOR.get(emotion, "#888888")


# ── Singleton helpers ──────────────────────────────────────────────────────────

_analyzer: Optional[EmotionAnalyzer] = None
_analyzer_lock = asyncio.Lock()


def get_analyzer(use_gpu: bool = False) -> EmotionAnalyzer:
    if _analyzer is None:
        raise RuntimeError("Analyzer not initialized. Call _init_analyzer() first.")
    return _analyzer


async def _init_analyzer(use_gpu: bool = False) -> EmotionAnalyzer:
    global _analyzer
    async with _analyzer_lock:
        if _analyzer is None:
            loop      = asyncio.get_event_loop()
            _analyzer = await loop.run_in_executor(
                None, lambda: EmotionAnalyzer(use_gpu=use_gpu)
            )
    return _analyzer