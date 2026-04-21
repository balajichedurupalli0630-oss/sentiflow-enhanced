"""
emotion_analyzer.py — patch for LoRA adapter loading
=====================================================
Replace the _load_pipe() and __init__() section in your existing
emotion_analyzer.py with the version below.

Only the Hartmann model loading changes — go_pipe and sentiment_pipe
remain identical. Everything else (blending, negation, keywords, etc.)
is untouched.
"""

# ── Add this import at the top of emotion_analyzer.py ─────────────────────────

import os
from pathlib import Path
from peft import PeftModel

# from peft import PeftModel                    # uncomment after pip install peft
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     pipeline,
# )


# ── Replace __init__ with this version ────────────────────────────────────────

ADAPTER_PATH = os.getenv("SENTIFLOW_ADAPTER", "")   # e.g. ./adapter/adapter


def __init__(self, use_gpu: bool = False):
    device      = 0 if use_gpu and torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"
    logger.info("Loading models on %s…", device_name)

    # ── Hartmann: load fine-tuned adapter if available, else base model ──────
    self.hartmann_pipe = self._load_hartmann_pipe(device)

    # ── go_emotions and sentiment: unchanged ─────────────────────────────────
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
    logger.info("All models loaded")


# ── Add this new method to EmotionAnalyzer ────────────────────────────────────

def _load_hartmann_pipe(self, device: int):
    """
    Loads the Hartmann model.
    - If SENTIFLOW_ADAPTER env var points to a valid adapter directory,
      loads the LoRA fine-tuned version.
    - Falls back to the original HuggingFace model if no adapter is found.
    """
    adapter = ADAPTER_PATH.strip()

    if adapter and Path(adapter).exists():
        try:
            from peft import PeftModel
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

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
                "Failed to load LoRA adapter (%s) — falling back to base model",
                exc,
            )

    # Fallback: original base model
    return self._load_pipe(
        "text-classification",
        "j-hartmann/emotion-english-distilroberta-base",
        device,
        top_k=None,
    )