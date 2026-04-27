"""
Train a calibrated multi-label DeBERTa emotion model for SentiFlow.

The output model predicts all 8 SentiFlow emotions independently. Validation
logits are temperature-scaled, then each emotion gets its own threshold chosen
to maximize validation F1. This is the path to use when confidence quality
matters as much as raw accuracy.

Example:
    python train_deberta_multilabel.py --gpu --epochs 4 --batch 8 --grad-accum 2

Fast smoke test:
    python train_deberta_multilabel.py --limit 200 --epochs 1 --batch 4
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sentiflow_labels import ID2LABEL, LABEL2ID, LABELS, go_row_to_multihot, sigmoid

log = logging.getLogger("sentiflow.deberta_multilabel")


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    max_length: int
    epochs: int
    batch: int
    grad_accum: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    use_gpu: bool
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    label_smoothing: float
    pos_weight_cap: float
    max_grad_norm: float
    limit: Optional[int]


class MultiLabelCollator:
    """Pad tokenized rows and keep labels as float tensors."""

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[dict]) -> dict:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        log.warning("MPS (Mac GPU) detected, but DeBERTa-v3 gradients are fundamentally unstable on MPS.")
        log.warning("Forcing CPU fallback to prevent permanent model corruption via NaN gradients.")
        return "cpu"
    log.warning("GPU requested but no CUDA device was found; using CPU")
    return "cpu"


from datasets import concatenate_datasets

def get_synthetic_support_data():
    """Generates synthetic corporate support emails to override Twitter bias based on user templates."""
    
    # Label index mapping for quick lookup:
    # 0: joy, 1: sadness, 2: anger, 3: fear, 4: surprise, 5: disgust, 6: trust, 7: anticipation
    label_map = {"joy": 0, "sadness": 1, "anger": 2, "fear": 3, "surprise": 4, "disgust": 5, "trust": 6, "anticipation": 7}
    
    raw_templates = [
        ("I am frustrated because this issue keeps happening repeatedly.", ["anger", "sadness"]),
        ("This problem wastes time and interrupts important tasks.", ["anger", "sadness"]),
        ("I expected better quality and reliability from this experience.", ["sadness", "trust"]),
        ("The system performance has become slower after recent changes.", ["sadness", "anger"]),
        ("I feel disappointed because this feature no longer works correctly.", ["sadness", "anger"]),
        ("I am concerned that these recurring errors may impact future work.", ["fear", "sadness"]),
        ("This update introduced unnecessary confusion and instability.", ["surprise", "sadness"]),
        ("The experience feels unreliable and difficult to trust.", ["fear", "disgust"]),
        ("The frequent crashes make the product difficult to use.", ["anger", "disgust"]),
        ("I regret spending time trying to make this work properly.", ["sadness", "disgust"]),
        ("The product has potential but still needs refinement.", ["anticipation"]),
        ("I am curious to see how future changes improve performance.", ["anticipation", "trust"]),
        ("I am looking forward to the upcoming update because it promises several useful improvements.", ["anticipation", "joy"]),
        ("I cannot wait to see how the new features will improve productivity.", ["anticipation", "joy"]),
        ("We are excited about the upcoming launch and eager to test everything.", ["anticipation", "joy"]),
        ("I hope the next release solves the performance issues we have been facing.", ["anticipation", "trust"]),
        ("The announced changes sound promising, and I am interested in trying them soon.", ["anticipation", "trust"]),
        ("I am preparing for the next phase of the project and feel optimistic about the outcome.", ["anticipation", "joy"]),
        ("Looking forward to hearing your feedback after you review the proposal.", ["anticipation", "trust"]),
        ("I am eager to begin using the upgraded version once it becomes available.", ["anticipation", "joy"]),
        ("The future roadmap looks interesting, and I am excited about what comes next.", ["anticipation", "joy"]),
        ("I expect the new improvements to make the workflow smoother and faster.", ["anticipation", "trust"]),
        ("I am curious to see how the system performs after the recent enhancements.", ["anticipation", "surprise"]),
        ("We are waiting for the official announcement before making a final decision.", ["anticipation", "trust"]),
        ("I feel hopeful that upcoming updates will improve stability and performance.", ["anticipation", "joy"]),
        ("The preview looks impressive, and I am excited to explore it in detail.", ["anticipation", "joy"]),
        ("I am anticipating positive changes based on the recent development progress.", ["anticipation", "trust"]),
        ("I am worried that the upcoming update may introduce more bugs into the system.", ["anticipation", "fear"]),
        ("I expect delays because the project has already missed several deadlines.", ["anticipation", "sadness"]),
        ("I am concerned that future changes could make the workflow more complicated.", ["anticipation", "fear"]),
        ("I feel uneasy about the upcoming release because previous updates caused issues.", ["anticipation", "fear"]),
        ("I am expecting problems during deployment due to unresolved technical errors.", ["anticipation", "fear"]),
        ("I worry that the next version may remove important features again.", ["anticipation", "sadness"]),
        ("I am nervous about how the system will perform under heavy usage next week.", ["anticipation", "fear"]),
        ("I fear that upcoming deadlines may be difficult to achieve with current issues.", ["anticipation", "sadness"]),
        ("I am anxious about the results because the testing phase revealed several problems.", ["anticipation", "fear"]),
        ("I expect customer complaints if these issues are not resolved soon.", ["anticipation", "anger"]),
        ("I am doubtful that the next update will fix all existing bugs.", ["anticipation", "sadness"]),
        ("I feel concerned about the future stability of the platform.", ["anticipation", "fear"]),
        ("I worry that upcoming changes may negatively affect performance.", ["anticipation", "fear"]),
        ("I expect complications during implementation because the process is unclear.", ["anticipation", "fear"]),
        ("I am uncertain whether the planned improvements will actually work as intended.", ["anticipation", "sadness"]),
        ("I appreciate the improvements, but several issues still remain unresolved.", ["trust", "sadness"]),
        ("The concept is promising, but performance still feels inconsistent.", ["anticipation", "sadness"]),
        ("I enjoy the design, although navigation remains confusing.", ["joy", "sadness"]),
        ("The feature is useful, but reliability remains a concern.", ["trust", "fear"]),
        ("I am optimistic about future updates despite current problems.", ["anticipation", "sadness"]),
        ("The interface looks modern, but loading speed is disappointing.", ["joy", "sadness"]),
        ("I appreciate the effort, although more optimization is needed.", ["trust", "sadness"]),
        ("The recent changes are interesting but introduced instability.", ["surprise", "sadness"]),
        ("The workflow is smoother now, though occasional bugs still occur.", ["trust", "sadness"]),
        ("The update improved usability but removed useful options.", ["trust", "sadness"]),
    ]
    
    data = []
    # Loop each template 400 times to generate a robust training weight (20,800 total rows)
    for text, emotion_labels in raw_templates:
        for _ in range(400):
            vec = [0.0] * 8
            for label in emotion_labels:
                vec[label_map[label]] = 1.0
            data.append({"text": text, "labels": vec, "label_names": emotion_labels})
            
    return data

def load_combined_dataset(limit: Optional[int] = None) -> DatasetDict:
    """Load GoEmotions, Dair-AI, and synthetic corporate datasets and combine them."""
    log.info("Loading GoEmotions simplified dataset")
    raw_go = load_dataset("google-research-datasets/go_emotions", "simplified")
    int2str = raw_go["train"].features["labels"].feature.int2str

    mapped_go = {"train": [], "validation": [], "test": []}
    for split_name, split in raw_go.items():
        rows = {"text": [], "labels": [], "label_names": []}
        for row in split:
            converted = go_row_to_multihot(row, int2str)
            if converted is None:
                continue
            multi_hot, names = converted
            rows["text"].append(row["text"])
            rows["labels"].append(multi_hot)
            rows["label_names"].append(names)
        mapped_go[split_name] = Dataset.from_dict(rows)
        log.info("  %-10s (Go) -> %d mapped rows", split_name, len(mapped_go[split_name]))

    log.info("Loading dair-ai/emotion dataset")
    # dair-ai default config is "unsplit" or we just load default
    raw_dair = load_dataset("dair-ai/emotion")
    
    # dair-ai labels: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    # We map 'love' (2) to 'trust' to match SentiFlow's schema.
    dair_mapping = {0: "sadness", 1: "joy", 2: "trust", 3: "anger", 4: "fear", 5: "surprise"}
    
    mapped_dair = {"train": [], "validation": [], "test": []}
    for split_name in ["train", "validation", "test"]:
        if split_name not in raw_dair:
            continue
        rows = {"text": [], "labels": [], "label_names": []}
        for row in raw_dair[split_name]:
            sentiflow_label = dair_mapping[row["label"]]
            sentiflow_id = LABEL2ID[sentiflow_label]
            vec = [0.0] * len(LABELS)
            vec[sentiflow_id] = 1.0
            rows["text"].append(row["text"])
            rows["labels"].append(vec)
            rows["label_names"].append([sentiflow_label])
        mapped_dair[split_name] = Dataset.from_dict(rows)
        log.info("  %-10s (Dair) -> %d mapped rows", split_name, len(mapped_dair[split_name]))

    log.info("Loading Synthetic Corporate Support dataset")
    synthetic_data = get_synthetic_support_data()
    rows = {"text": [d["text"] for d in synthetic_data], "labels": [d["labels"] for d in synthetic_data], "label_names": [d["label_names"] for d in synthetic_data]}
    mapped_synthetic = {"train": Dataset.from_dict(rows)}
    log.info("  %-10s (Synthetic) -> %d mapped rows", "train", len(mapped_synthetic["train"]))

    combined = {}
    for split in ["train", "validation", "test"]:
        # Only combine if both exist
        to_concat = []
        if len(mapped_go.get(split, [])) > 0: to_concat.append(mapped_go[split])
        if len(mapped_dair.get(split, [])) > 0: to_concat.append(mapped_dair[split])
        if split == "train": to_concat.append(mapped_synthetic["train"])
        
        combined[split] = concatenate_datasets(to_concat).shuffle(seed=42)
        if limit is not None:
            combined[split] = combined[split].select(range(min(limit, len(combined[split]))))
        log.info("  %-10s (Total) -> %d rows", split, len(combined[split]))
        
    return DatasetDict(combined)


def tokenize_dataset(ds: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(_tokenize, batched=True, remove_columns=["text", "label_names"])


def compute_pos_weight(labels: Iterable[Iterable[float]], cap: float) -> torch.Tensor:
    label_array = np.asarray(list(labels), dtype=np.float32)
    positive = label_array.sum(axis=0)
    negative = label_array.shape[0] - positive
    positive = np.maximum(positive, 1.0)
    weights = negative / positive
    if cap > 0:
        weights = np.minimum(weights, cap)
    return torch.tensor(weights, dtype=torch.float32)


def build_model(args: argparse.Namespace):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "multi_label_classification"
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    if not args.lora:
        return model

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query_proj", "key_proj", "value_proj"],
        modules_to_save=["classifier", "pooler"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def make_trainer_class(pos_weight: torch.Tensor, label_smoothing: float):
    class WeightedMultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels").float()
            if label_smoothing > 0:
                labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            outputs = model(**inputs)
            
            # Handle potential NaNs from the model (especially on MPS/Apple Silicon)
            if not torch.isfinite(outputs.logits).all():
                log.warning("Non-finite logits detected! Skipping this batch to maintain stability.")
                zero_loss = torch.tensor(0.0, device=outputs.logits.device, requires_grad=True)
                return (zero_loss, outputs) if return_outputs else zero_loss

            # Clamp logits to prevent BCE overflow (a known issue on MPS)
            clamped_logits = outputs.logits.clamp(-50.0, 50.0)
            weight = pos_weight.to(outputs.logits.device)
            loss = F.binary_cross_entropy_with_logits(clamped_logits, labels, pos_weight=weight)
            
            # Print if weights are NaN
            if not torch.isfinite(next(model.parameters())).all():
                raise RuntimeError("Weights became NaN!")
                
            return (loss, outputs) if return_outputs else loss

    return WeightedMultiLabelTrainer


def multilabel_metrics_from_probs(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, float]:
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    labels = labels.astype(np.int32)
    primary = np.argmax(probs, axis=1)
    primary_accuracy = float(np.mean(labels[np.arange(len(labels)), primary] == 1))
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(labels, preds, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(labels, preds)),
        "primary_accuracy": primary_accuracy,
        "hamming_loss": float(hamming_loss(labels, preds)),
    }


def compute_training_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = sigmoid(logits)
    return multilabel_metrics_from_probs(labels, probs, np.full(len(LABELS), 0.5))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 80) -> float:
    """Fit one scalar temperature on validation logits using BCE."""
    # Clean logits: replace NaNs/Infs with reasonable values to prevent LBFGS crash
    logits = np.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.1, 10.0)
        loss = F.binary_cross_entropy_with_logits(logits_t / temperature, labels_t)
        if torch.isfinite(loss):
            loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as exc:
        log.warning("Temperature fitting failed: %s; using default T=1.0", exc)
        return 1.0

    return float(torch.exp(log_temperature).detach().clamp(0.1, 10.0).item())


def calibrate_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    candidates: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Choose one threshold per emotion to maximize validation F1."""
    if candidates is None:
        candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)

    thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    details: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(LABELS):
        y_true = labels[:, idx].astype(np.int32)
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in candidates:
            y_pred = (probs[:, idx] >= threshold).astype(np.int32)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        thresholds[idx] = best_threshold
        details[label] = {
            "threshold": best_threshold,
            "validation_f1": best_f1,
            "support": int(y_true.sum()),
        }
    return thresholds, details


def prediction_arrays(trainer: Trainer, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    output = trainer.predict(dataset)
    return np.asarray(output.predictions), np.asarray(output.label_ids)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_training_args(args: argparse.Namespace, device: str) -> TrainingArguments:
    kwargs = {
        "output_dir": str(Path(args.output_dir) / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch,
        "per_device_eval_batch_size": args.eval_batch or args.batch * 2,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": args.logging_steps,
        "save_total_limit": 2,
        "max_grad_norm": args.max_grad_norm,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": device == "cuda",
        "report_to": "none",
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["evaluation_strategy" if "evaluation_strategy" in signature else "eval_strategy"] = "epoch"
    if "bf16" in signature:
        kwargs["bf16"] = device == "cuda" and torch.cuda.is_bf16_supported()
    if "fp16" in signature:
        kwargs["fp16"] = device == "cuda" and not torch.cuda.is_bf16_supported()
    if "optim" in signature:
        kwargs["optim"] = "adamw_torch_fused" if device == "cuda" else "adamw_torch"
    if "gradient_checkpointing" in signature:
        kwargs["gradient_checkpointing"] = args.gradient_checkpointing
    return TrainingArguments(**kwargs)


def train(args: argparse.Namespace) -> Dict[str, object]:
    configure_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        batch=args.batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        use_gpu=args.gpu,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        label_smoothing=args.label_smoothing,
        pos_weight_cap=args.pos_weight_cap,
        max_grad_norm=args.max_grad_norm,
        limit=args.limit,
    )
    save_json(output_dir / "training_config.json", asdict(config))

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    raw_ds = load_combined_dataset(limit=args.limit)
    tokenized = tokenize_dataset(raw_ds, tokenizer, args.max_length)

    pos_weight = compute_pos_weight(raw_ds["train"]["labels"], args.pos_weight_cap)
    log.info("Positive class weights:")
    for label, weight in zip(LABELS, pos_weight.tolist()):
        log.info("  %-14s %.3f", label, weight)

    model = build_model(args)
    training_args = build_training_args(args, device)
    collator = MultiLabelCollator(
        tokenizer,
        pad_to_multiple_of=8 if device == "cuda" else None,
    )

    trainer_class = make_trainer_class(pos_weight, args.label_smoothing)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": collator,
        "compute_metrics": compute_training_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.patience)],
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trainer_class(**trainer_kwargs)
    log.info("Starting DeBERTa multi-label training")
    trainer.train()

    model_dir = output_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    log.info("Saved model to %s", model_dir)

    validation_logits, validation_labels = prediction_arrays(trainer, tokenized["validation"])
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_probs = sigmoid(validation_logits / temperature)
    thresholds, threshold_details = calibrate_thresholds(validation_labels, validation_probs)

    test_logits, test_labels = prediction_arrays(trainer, tokenized["test"])
    test_probs = sigmoid(test_logits / temperature)
    test_metrics = multilabel_metrics_from_probs(test_labels, test_probs, thresholds)
    test_preds = (test_probs >= thresholds.reshape(1, -1)).astype(np.int32)
    report = classification_report(
        test_labels.astype(np.int32),
        test_preds,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    calibration = {
        "labels": LABELS,
        "temperature": temperature,
        "thresholds": {label: float(thresholds[idx]) for idx, label in enumerate(LABELS)},
        "threshold_details": threshold_details,
        "calibrated_on": "validation",
        "threshold_metric": "per_label_f1",
    }
    metrics = {
        "model_name": args.model_name,
        "output_model": str(model_dir),
        "test_set_size": int(len(test_labels)),
        "metrics": test_metrics,
        "per_class": report,
        "calibration": calibration,
    }
    save_json(output_dir / "calibration.json", calibration)
    save_json(output_dir / "metrics.json", metrics)

    log.info("Final calibrated test metrics:")
    for key, value in test_metrics.items():
        log.info("  %-18s %.4f", key, value)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 for SentiFlow multi-label emotion detection")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--output-dir", default="./deberta_multilabel")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--lora", action="store_true", default=False, help="Use LoRA instead of full fine-tuning")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--pos-weight-cap", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per split")
    return parser.parse_args()

    
if __name__ == "__main__":
    train(parse_args())
