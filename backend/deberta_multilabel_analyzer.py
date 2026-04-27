"""Runtime analyzer for calibrated SentiFlow DeBERTa multi-label models."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import numpy as np

try:
    from peft import PeftModel, PeftConfig
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from sentiflow_labels import EMOTION_COLOR, EMOTION_EMOJI, LABELS, sigmoid

logger = logging.getLogger("sentiflow.deberta_analyzer")

EMOTION_TO_SENTIMENT = {
    "joy": "positive",
    "trust": "positive",
    "anticipation": "positive",
    "surprise": "neutral",
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "disgust": "negative",
}


class DebertaMultilabelAnalyzer:
    """Load a trained DeBERTa multi-label model plus calibrated thresholds."""

    def __init__(
        self,
        model_path: str = "./deberta_multilabel_corporate/model",
        calibration_path: str = "./deberta_multilabel_corporate/calibration.json",
        use_gpu: bool = False,
        max_length: int = 160,
    ):
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.max_length = max_length
        self.device = self._resolve_device(use_gpu)
        self.temperature, self.thresholds = self._load_calibration(calibration_path)

        logger.info("Loading DeBERTa multi-label model from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # Check if this is a PEFT (LoRA) model
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.exists() and HAS_PEFT:
            logger.info("Detected PEFT/LoRA adapter. Loading base model and applying adapter...")
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=len(LABELS),
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
        self.model.to(self.device)
        self.model.eval()
        logger.info("DeBERTa analyzer ready on %s", self.device)

    async def analyze(self, text: str) -> Dict:
        return await asyncio.to_thread(self._sync_analyze, text)

    def _sync_analyze(self, text: str) -> Dict:
        clean = " ".join(text.strip().split())
        encoded = self.tokenizer(
            clean,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits.detach().cpu().numpy()[0]
            
        # --- PURE MATH CALIBRATION LAYER ---
        trust_idx = LABELS.index("trust")
        anger_idx = LABELS.index("anger")
        sadness_idx = LABELS.index("sadness")
        fear_idx = LABELS.index("fear")
        disgust_idx = LABELS.index("disgust")
        
        max_negative_logit = max(logits[anger_idx], logits[sadness_idx], logits[fear_idx], logits[disgust_idx])
        
        if max_negative_logit > -2.0:
            if logits[trust_idx] > 0:
                logits[trust_idx] *= 0.50
                
            if logits[anger_idx] > -2.0: logits[anger_idx] += 2.0
            if logits[sadness_idx] > -2.0: logits[sadness_idx] += 2.0
            if logits[fear_idx] > -2.0: logits[fear_idx] += 2.0
            if logits[disgust_idx] > -2.0: logits[disgust_idx] += 2.0

        # We use L1 Normalization instead of Softmax!
        raw_probs = sigmoid(logits / self.temperature)
        scores = {label: float(raw_probs[idx]) for idx, label in enumerate(LABELS)}

        # --- FIX 5: L1 Normalize Emotion Spread ---
        # Filter out tiny noise before normalizing
        for k in scores:
            if scores[k] < 0.05:
                scores[k] = 0.0
                
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total
                
        pct_scores = {label: round(value * 100, 1) for label, value in scores.items()}
        sorted_scores = sorted(pct_scores.items(), key=lambda item: item[1], reverse=True)
        primary = sorted_scores[0][0]
        primary_score = sorted_scores[0][1]
        
        # --- FIX 1: Mixed Emotion Confidence Penalty ---
        if len(sorted_scores) > 1:
            second_score = sorted_scores[1][1]
            if second_score > primary_score * 0.60:
                primary_score *= 0.65
                
        # --- FIX 4: Lower Confidence Ceiling (Cap at 88%) ---
        primary_score = min(primary_score, 88.0)
        primary_score = round(primary_score, 1)
        active = [
            label
            for idx, label in enumerate(LABELS)
            if scores[label] >= self.thresholds.get(label, 0.5)
        ]

        return {
            "primary_emotion": primary,
            "emotion_score": primary_score,
            "sentiment": EMOTION_TO_SENTIMENT.get(primary, "neutral"),
            "sentiment_score": primary_score,
            "emotion_scores": pct_scores,
            "top_3_emotions": sorted_scores[:3],
            "active_emotions": active,
            "uncertain": len(active) == 0,
            "calibrated": True,
            "emoji": EMOTION_EMOJI.get(primary, ":"),
            "color": EMOTION_COLOR.get(primary, "#888888"),
        }

    @staticmethod
    def _resolve_device(use_gpu: bool) -> str:
        if not use_gpu:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            logger.warning("MPS (Mac GPU) detected, but DeBERTa-v3 gradients are numerically unstable on MPS.")
            logger.warning("Falling back to CPU to prevent NaN corruption. Training will be slower but mathematically safe.")
            return "cpu"
        return "cpu"

    @staticmethod
    def _load_calibration(path: str) -> tuple[float, Dict[str, float]]:
        calibration_file = Path(path)
        if not calibration_file.exists():
            logger.warning("Calibration file not found at %s; using 0.5 thresholds", path)
            return 1.0, {label: 0.5 for label in LABELS}

        payload = json.loads(calibration_file.read_text())
        temperature = float(payload.get("temperature", 1.0))
        raw_thresholds = payload.get("thresholds", {})
        thresholds = {label: float(raw_thresholds.get(label, 0.5)) for label in LABELS}
        return temperature, thresholds
