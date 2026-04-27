"""
fine_tune.py — SentiFlow LoRA Fine-Tuning v2 (Improved)
=========================================================
Changes from v1 that improve accuracy:

  FIX 1 — Class-weighted loss
          Minority emotions (disgust, fear) were being ignored.
          WeightedTrainer penalises the model more for getting them wrong.
          Expected gain: +3–6 F1 points on disgust/surprise/fear.

  FIX 2 — Expanded label mappings
          Added "disapproval" → disgust, "nervousness" → fear,
          "awe" → surprise. More data for weak classes.

  FIX 3 — Keep ALL mappable labels per row (not just first)
          Old code took only the first mappable label per example,
          discarding valid signal. Now every mappable label becomes
          its own training row.

  FIX 4 — Higher LoRA rank (r=16) + key/query/value modules
          r=8 with only query+value is too constrained for 8-class
          classification. r=16 + key gives the adapter more capacity.

  FIX 5 — Label smoothing (0.1)
          Prevents overconfidence, improves generalisation, especially
          for minority classes.

  FIX 6 — Early stopping patience raised to 3
          Patience=2 stopped at epoch 4 even though the model was
          still improving. Patience=3 gives it more runway.

  FIX 7 — Oversampling for minority classes
          fear (620) and disgust (858) are resampled to match
          the median class size before training.

Usage:
    python fineTune.py                          # default (CPU)
    python fineTune.py --epochs 10 --batch 16  # custom
    python fineTune.py --gpu                    # use CUDA / MPS
    python fineTune.py --train-blend --gpu      # full pipeline (adapter + blend)

Output:
    ./adapter_v3/adapter/   — LoRA adapter weights
    ./adapter_v3/metrics.json
    ./blend_weights.json    — optional (when --train-blend is used)
"""

import argparse
import inspect
import json
import logging
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentiflow.finetune")

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL = "j-hartmann/emotion-english-distilroberta-base"
OUTPUT_DIR = "./adapter_v3"

LABELS   = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# ── FIX 2: Expanded mappings — more coverage for weak classes ─────────────────
GO_TO_SENTIFLOW: Dict[str, str] = {
    # Joy (7 labels — well covered, unchanged)
    "joy":              "joy",
    "amusement":        "joy",
    "excitement":       "joy",
    "love":             "joy",
    "pride":            "joy",
    "relief":           "joy",
    "optimism":         "joy",

    # Sadness (4 labels — unchanged)
    "sadness":          "sadness",
    "grief":            "sadness",
    "disappointment":   "sadness",
    "remorse":          "sadness",

    # Anger (3 labels — added disapproval)
    "anger":            "anger",
    "annoyance":        "anger",
    "disapproval":      "anger",       # ← NEW: disapproval is close to anger

    # Fear (3 labels — added nervousness explicitly + apprehension)
    "fear":             "fear",
    "nervousness":      "fear",
    "apprehension":     "fear",        # ← NEW: if present in dataset

    # Surprise (4 labels — added awe)
    "surprise":         "surprise",
    "realization":      "surprise",
    "confusion":        "surprise",
    "awe":              "surprise",    # ← NEW: awe is a form of surprise

    # Disgust (3 labels — added contempt)
    "disgust":          "disgust",
    "embarrassment":    "disgust",
    "contempt":         "disgust",     # ← NEW: contempt is close to disgust

    # Trust (4 labels — unchanged)
    "admiration":       "trust",
    "approval":         "trust",
    "caring":           "trust",
    "gratitude":        "trust",

    # Anticipation (3 labels — added interest)
    "desire":           "anticipation",
    "curiosity":        "anticipation",
    "interest":         "anticipation", # ← NEW: interest = anticipation
}


# ── FIX 3: Keep ALL mappable labels per row ────────────────────────────────────

def load_and_map() -> DatasetDict:
    """
    Downloads GoEmotions (simplified), maps ALL mappable labels per row
    (not just the first one). This increases training data, especially
    for minority classes that appear as secondary labels.
    """
    log.info("Downloading GoEmotions…")
    raw     = load_dataset("google-research-datasets/go_emotions", "simplified")
    int2str = raw["train"].features["labels"].feature.int2str

    mapped_splits = {}
    for split_name, split_data in raw.items():
        texts, label_ids = [], []
        for row in split_data:
            seen_sf = set()
            for lid in row["labels"]:
                go_label = int2str(lid)
                sf_label = GO_TO_SENTIFLOW.get(go_label)
                # ── FIX 3: add every unique mappable label, not just first ──
                if sf_label is not None and sf_label not in seen_sf:
                    texts.append(row["text"])
                    label_ids.append(LABEL2ID[sf_label])
                    seen_sf.add(sf_label)

        mapped_splits[split_name] = Dataset.from_dict({"text": texts, "label": label_ids})
        log.info("  %-12s → %d samples (was %d)", split_name, len(texts), len(split_data))

    return DatasetDict(mapped_splits)


# ── FIX 7: Oversample minority classes in training split ──────────────────────

def oversample_minority(ds: Dataset, seed: int = 42) -> Dataset:
    """
    Brings every class up to the median class count by repeating
    minority examples (with shuffle). Applied to train split only.
    """
    counts  = Counter(ds["label"])
    median  = int(np.median(list(counts.values())))
    log.info("Oversampling to median count = %d", median)

    per_class = {lid: [] for lid in counts}
    for i, lbl in enumerate(ds["label"]):
        per_class[lbl].append(i)

    all_indices = []
    rng = random.Random(seed)
    for lid, indices in per_class.items():
        if len(indices) < median:
            # Repeat + sample up to median
            repeated = (indices * ((median // len(indices)) + 1))[:median]
            rng.shuffle(repeated)
            all_indices.extend(repeated)
            log.info("  %-15s %d → %d", LABELS[lid], len(indices), median)
        else:
            all_indices.extend(indices)

    rng.shuffle(all_indices)
    return ds.select(all_indices)


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_dataset(ds: DatasetDict, tokenizer) -> DatasetDict:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )
    return ds.map(_tok, batched=True, remove_columns=["text"])


# ── FIX 4: Proper 7→8 class head expansion + higher LoRA rank ────────────────

def build_lora_model(device_str: str):
    log.info("Loading base model: %s", BASE_MODEL)

    # Step 1 — Load with ORIGINAL 7 classes so no weights are thrown away
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=7,               # keep original — no MISMATCH warning
        ignore_mismatched_sizes=False,
    )

    # Step 2 — Expand classifier head 7 → 8 using meaningful initialization.
    # Without this the 8th class starts as random noise and LoRA has to
    # relearn all 7 original classes from scratch (huge wasted epochs).
    with torch.no_grad():
        old_weight = model.classifier.out_proj.weight.data   # (7, 768)
        old_bias   = model.classifier.out_proj.bias.data     # (7,)

        # 8th row = average of all 7 existing trained rows (neutral start)
        new_row    = old_weight.mean(dim=0, keepdim=True)    # (1, 768)
        new_bias_v = old_bias.mean().unsqueeze(0)            # (1,)

        new_weight = torch.cat([old_weight, new_row],    dim=0)  # (8, 768)
        new_bias   = torch.cat([old_bias,   new_bias_v], dim=0)  # (8,)

        model.classifier.out_proj = torch.nn.Linear(768, 8, bias=True)
        model.classifier.out_proj.weight = torch.nn.Parameter(new_weight)
        model.classifier.out_proj.bias   = torch.nn.Parameter(new_bias)

    # Step 3 — Update config so Trainer and pipeline know about 8 classes
    model.config.num_labels = len(LABELS)
    model.num_labels        = len(LABELS)
    model.config.id2label   = ID2LABEL
    model.config.label2id   = LABEL2ID

    log.info("Classifier head expanded 7 → 8 classes (7 original weights preserved)")

    # Step 4 — Wrap with LoRA AFTER head expansion
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if device_str in ("cuda", "mps"):
        model = model.to(device_str)
        log.info("Model moved to %s", device_str)

    return model


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds     = np.argmax(logits, axis=-1)
    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    acc       = float(np.mean(preds == labels))
    return {"f1_macro": round(macro_f1, 4), "accuracy": round(acc, 4)}


# ── FIX 1: Weighted loss trainer ──────────────────────────────────────────────

def make_weighted_trainer(class_weights: torch.Tensor):
    """
    Returns a Trainer subclass that applies per-class weights to cross-entropy.
    Minority classes (disgust, fear) get a higher loss penalty so the model
    cannot ignore them.
    """
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels  = inputs.get("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            weights = class_weights.to(logits.device)
            loss    = F.cross_entropy(logits, labels, weight=weights,
                                      # FIX 5: label smoothing ────────────────
                                      label_smoothing=0.1)
            
            # For Transformers 4.46+ / 5.x, handle loss scaling with num_items_in_batch
            if num_items_in_batch is not None:
                loss /= num_items_in_batch
                
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# ── Training ───────────────────────────────────────────────────────────────────

def get_device(use_gpu: bool) -> tuple:
    """
    Returns (device_str, hf_device_int) for the best available accelerator.
      - CUDA GPU  → ("cuda",  0)
      - Apple MPS → ("mps",  -1)   HF Trainer handles MPS via device_map
      - CPU       → ("cpu",  -1)
    """
    if not use_gpu:
        return "cpu", -1
    if torch.cuda.is_available():
        log.info("Using CUDA GPU")
        return "cuda", 0
    if torch.backends.mps.is_available():
        log.info("Using Apple MPS (Metal) GPU")
        return "mps", -1
    log.warning("GPU requested but no CUDA or MPS found — using CPU")
    return "cpu", -1


def _build_training_args(
    *,
    output_dir: str,
    epochs: int,
    batch: int,
    lr: float,
    grad_accum: int,
    device_str: str,
) -> TrainingArguments:
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=(device_str == "cuda"),
        report_to="none",
    )

    args_sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "evaluation_strategy" if "evaluation_strategy" in args_sig else "eval_strategy"
    kwargs[eval_key] = "epoch"

    cuda_bf16 = (
        device_str == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    kwargs["bf16"] = cuda_bf16
    kwargs["fp16"] = (device_str == "cuda") and not cuda_bf16

    if "optim" in args_sig:
        kwargs["optim"] = "adamw_torch_fused" if device_str == "cuda" else "adamw_torch"
    if "gradient_checkpointing" in args_sig:
        kwargs["gradient_checkpointing"] = True

    return TrainingArguments(**kwargs)


def _run_blend_training(
    *,
    adapter_path: str,
    use_gpu: bool,
    method: str,
    split: str,
    limit: Optional[int],
) -> None:
    cmd = [
        sys.executable,
        "train_blend_weights.py",
        "--adapter",
        adapter_path,
        "--method",
        method,
        "--split",
        split,
    ]
    if use_gpu:
        cmd.append("--gpu")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    log.info("Running blend-weight training: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train(
    epochs:     int   = 10,       # was 5 — more room with patience=3
    batch:      int   = 16,
    grad_accum: int   = 1,
    lr:         float = 1e-4,
    use_gpu:    bool  = False,
    seed:       int   = 42,
    train_blend: bool = False,
    blend_method: str = "both",
    blend_split: str = "validation",
    blend_limit: Optional[int] = None,
    output_dir: str   = OUTPUT_DIR,
):
    device_str, device = get_device(use_gpu)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds        = load_and_map()

    # ── Oversample minorities before tokenising ──────────────────────────────
    ds["train"] = oversample_minority(ds["train"])

    ds = tokenize_dataset(ds, tokenizer)

    # ── Compute class weights for loss ───────────────────────────────────────
    counts  = Counter(ds["train"]["label"])
    total   = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABELS) * counts[i]) for i in range(len(LABELS))],
        dtype=torch.float,
    )
    log.info("Class weights:")
    for i, (lbl, w) in enumerate(zip(LABELS, weights)):
        log.info("  %-15s %.3f  (n=%d)", lbl, w.item(), counts[i])

    model = build_lora_model(device_str)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    args = _build_training_args(
        output_dir=output_dir,
        epochs=epochs,
        batch=batch,
        lr=lr,
        grad_accum=grad_accum,
        device_str=device_str,
    )
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device_str == "cuda" else None,
    )

    WeightedTrainer = make_weighted_trainer(weights)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": ds["train"],
        "eval_dataset": ds["validation"],
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }

    # FIX: Compatibility for Transformers v5+ (which renamed 'tokenizer' to 'processing_class')
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    log.info("Starting training…")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = Path(output_dir) / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    log.info("Adapter saved → %s", adapter_path)

    # ── Final evaluation ──────────────────────────────────────────────────────
    log.info("Evaluating on test set…")
    test_preds  = trainer.predict(ds["test"])
    preds       = np.argmax(test_preds.predictions, axis=-1)
    true_labels = test_preds.label_ids

    report = classification_report(
        true_labels, preds,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Test results:")
    log.info("  Macro F1 : %.4f", report["macro avg"]["f1-score"])
    log.info("  Accuracy : %.4f", report["accuracy"])
    log.info("  Per-emotion F1:")
    for emotion in LABELS:
        log.info("    %-15s %.3f", emotion, report[emotion]["f1-score"])

    log.info("Metrics saved → %s", metrics_path)
    if train_blend:
        _run_blend_training(
            adapter_path=str(adapter_path),
            use_gpu=use_gpu,
            method=blend_method,
            split=blend_split,
            limit=blend_limit,
        )
    return str(adapter_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentiFlow LoRA fine-tuning v2")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--gpu",     action="store_true")
    parser.add_argument("--train-blend", action="store_true",
                        help="Run train_blend_weights.py after adapter training")
    parser.add_argument("--blend-method", type=str, default="both",
                        choices=["grid", "stacking", "both"])
    parser.add_argument("--blend-split", type=str, default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--blend-limit", type=int, default=None)
    parser.add_argument("--output",  type=str,   default=OUTPUT_DIR)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch=args.batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        use_gpu=args.gpu,
        seed=args.seed,
        train_blend=args.train_blend,
        blend_method=args.blend_method,
        blend_split=args.blend_split,
        blend_limit=args.blend_limit,
        output_dir=args.output,
    )
