import os

import torch
from transformers import pipeline
from typing import Dict, List, Optional
import asyncio
from enum import Enum

# ── Required by main.py ────────────────────────────────────────────────────────

class Emotion(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

class EmotionAnalyzer:
    """Simplified version that matches the expected interface"""
    
    def __init__(self, use_gpu: bool = True):
        self._delegate = None
        backend = os.getenv("SENTIFLOW_ANALYZER", "").strip().lower()
        if backend in {"deberta", "deberta_multilabel"}:
            from deberta_multilabel_analyzer import DebertaMultilabelAnalyzer

            self._delegate = DebertaMultilabelAnalyzer(
                model_path=os.getenv("SENTIFLOW_DEBERTA_MODEL", "./deberta_multilabel/model"),
                calibration_path=os.getenv(
                    "SENTIFLOW_DEBERTA_CALIBRATION",
                    "./deberta_multilabel/calibration.json",
                ),
                use_gpu=use_gpu,
                max_length=int(os.getenv("SENTIFLOW_MAX_LENGTH", "160")),
            )
            return

        # Explicitly set device to avoid ambiguity
        device = 0 if use_gpu and torch.cuda.is_available() else -1
        print(f"Loading emotion model on {'GPU' if device == 0 else 'CPU'}...")
        
        self.model = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            device=device,
            top_k=None,
            truncation=True,
            max_length=512
        )
        
        # Map model emotions to our categories
        self.emotion_map = {
            "joy": "joy",
            "sadness": "sadness",
            "anger": "anger", 
            "fear": "fear",
            "surprise": "surprise",
            "love": "joy",
            "optimism": "anticipation",
        }
        
        # All 8 emotions expected by the frontend/main.py
        self.all_emotions = [e.value for e in Emotion]
        
        print("✅ Analyzer ready!")
    
    async def analyze(self, text: str) -> Dict:
        """Analyze emotion in text (Async)"""
        if self._delegate is not None:
            return await self._delegate.analyze(text)
        # Run synchronous pipeline in a thread to avoid blocking the event loop
        return await asyncio.to_thread(self._sync_analyze, text)

    def _sync_analyze(self, text: str) -> Dict:
        """Synchronous core logic"""
        # Get predictions
        results = self.model(text)[0]
        
        # Initialize scores
        scores = {emotion: 0.0 for emotion in self.all_emotions}
        
        # Map predictions
        for pred in results:
            label = pred['label']
            score = pred['score']
            
            if label in self.emotion_map:
                mapped = self.emotion_map[label]
                scores[mapped] = max(scores[mapped], score)
            elif label in self.all_emotions:
                scores[label] = max(scores[label], score)
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        # Get primary emotion
        primary = max(scores, key=scores.get)
        
        return {
            "primary_emotion": primary,
            "emotion_score": round(scores[primary] * 100, 1),
            "emotion_scores": {k: round(v * 100, 1) for k, v in scores.items()},
            "top_3_emotions": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3],
            "text": text[:100]
        }

# ── Singleton Pattern for main.py ──────────────────────────────────────────────

_analyzer_instance: Optional[EmotionAnalyzer] = None
_lock = asyncio.Lock()

async def _init_analyzer(use_gpu: bool = False):
    """Initialize or return the existing analyzer instance"""
    global _analyzer_instance
    async with _lock:
        if _analyzer_instance is None:
            analyzer_type = os.environ.get("SENTIFLOW_ANALYZER", "default")
            if analyzer_type == "deberta_multilabel":
                from deberta_multilabel_analyzer import DebertaMultilabelAnalyzer
                _analyzer_instance = await asyncio.to_thread(DebertaMultilabelAnalyzer, use_gpu=use_gpu)
            else:
                _analyzer_instance = await asyncio.to_thread(EmotionAnalyzer, use_gpu=use_gpu)
        return _analyzer_instance

def get_analyzer() -> Optional[EmotionAnalyzer]:
    """Return the loaded analyzer (if initialized)"""
    return _analyzer_instance

# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def test():
        analyzer = await _init_analyzer(use_gpu=False)
        test_text = "I'm absolutely thrilled with this solution!"
        result = await analyzer.analyze(test_text)
        print(f"\n📝 Text: {test_text}")
        print(f"🎭 Primary: {result['primary_emotion']} ({result['emotion_score']}%)")
        print(f"📊 Top 3: {result['top_3_emotions']}")

    asyncio.run(test())
