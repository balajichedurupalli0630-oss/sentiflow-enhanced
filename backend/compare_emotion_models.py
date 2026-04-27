"""
Compare calibrated SentiFlow DeBERTa against SamLowe GoEmotions ONNX INT8.

This script evaluates both accuracy and batch-1/batched inference speed on the
same GoEmotions split after mapping labels into SentiFlow's 8 emotions.

Example:
    python compare_emotion_models.py \
      --deberta-model ./deberta_multilabel/model \
      --deberta-calibration ./deberta_multilabel/calibration.json \
      --limit 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiflow_labels import (
    GO_TO_SENTIFLOW,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    go_row_to_multihot,
    sentiflow_scores_from_goemotions,
    sigmoid,
)

log = logging.getLogger("sentiflow.compare_models")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def load_split(split: str, limit: Optional[int]) -> Tuple[List[str], np.ndarray]:
    raw = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    int2str = raw.features["labels"].feature.int2str
    texts: List[str] = []
    labels: List[List[float]] = []
    for row in raw:
        converted = go_row_to_multihot(row, int2str)
        if converted is None:
            continue
        multi_hot, _ = converted
        texts.append(row["text"])
        labels.append(multi_hot)
        if limit is not None and len(texts) >= limit:
            break
    return texts, np.asarray(labels, dtype=np.float32)


def load_calibration(path: Optional[str]) -> Tuple[float, np.ndarray]:
    if not path:
        return 1.0, np.full(len(LABELS), 0.5, dtype=np.float32)
    payload = json.loads(Path(path).read_text())
    temperature = float(payload.get("temperature", 1.0))
    raw_thresholds = payload.get("thresholds", {})
    thresholds = np.asarray(
        [float(raw_thresholds.get(label, 0.5)) for label in LABELS],
        dtype=np.float32,
    )
    return temperature, thresholds


def calibrate_thresholds(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    for idx in range(len(LABELS)):
        best_f1 = -1.0
        best_threshold = 0.5
        y_true = labels[:, idx].astype(np.int32)
        for threshold in candidates:
            y_pred = (probs[:, idx] >= threshold).astype(np.int32)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        thresholds[idx] = best_threshold
    return thresholds


def metrics_for_probs(labels: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> Dict[str, object]:
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    labels_int = labels.astype(np.int32)
    primary = np.argmax(probs, axis=1)
    metrics = {
        "macro_f1": float(f1_score(labels_int, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels_int, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels_int, preds, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(labels_int, preds, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(labels_int, preds)),
        "primary_accuracy": float(np.mean(labels_int[np.arange(len(labels_int)), primary] == 1)),
        "hamming_loss": float(hamming_loss(labels_int, preds)),
        "per_class": classification_report(
            labels_int,
            preds,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def run_deberta(
    model_path: str,
    calibration_path: Optional[str],
    texts: List[str],
    labels: np.ndarray,
    batch_size: int,
    max_length: int,
    device: str,
) -> Dict[str, object]:
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    temperature, thresholds = load_calibration(calibration_path)
    all_probs: List[np.ndarray] = []
    infer_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            all_probs.append(sigmoid(logits / temperature))
    inference_seconds = time.perf_counter() - infer_started
    probs = np.vstack(all_probs)

    return {
        "name": "deberta_multilabel",
        "model": model_path,
        "calibration": calibration_path,
        "thresholds": {label: float(thresholds[idx]) for idx, label in enumerate(LABELS)},
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "avg_ms_per_text": (inference_seconds / len(texts)) * 1000,
        "throughput_texts_per_second": len(texts) / inference_seconds,
        "metrics": metrics_for_probs(labels, probs, thresholds),
    }


def run_samlowe_onnx(
    texts: List[str],
    labels: np.ndarray,
    calibration_texts: List[str],
    calibration_labels: np.ndarray,
    batch_size: int,
    max_length: int,
    file_name: str,
) -> Dict[str, object]:
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError(
            "optimum[onnxruntime] is required for SamLowe ONNX INT8 benchmarking"
        ) from exc

    model_id = "SamLowe/roberta-base-go_emotions-onnx"
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name)
    id2label = {int(idx): label.lower() for idx, label in model.config.id2label.items()}
    load_seconds = time.perf_counter() - load_started

    def infer(batch_texts: List[str]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for start in range(0, len(batch_texts), batch_size):
            encoded = tokenizer(
                batch_texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            probs = sigmoid(logits.detach().cpu().numpy())
            for row in probs:
                go_scores = {
                    id2label[idx]: float(row[idx])
                    for idx in range(len(row))
                    if id2label.get(idx) in GO_TO_SENTIFLOW
                }
                collapsed = sentiflow_scores_from_goemotions(go_scores)
                rows.append(np.asarray([collapsed[label] for label in LABELS], dtype=np.float32))
        return np.vstack(rows)

    calibration_probs = infer(calibration_texts)
    thresholds = calibrate_thresholds(calibration_labels, calibration_probs)

    infer_started = time.perf_counter()
    probs = infer(texts)
    inference_seconds = time.perf_counter() - infer_started

    return {
        "name": "samlowe_goemotions_onnx_int8",
        "model": model_id,
        "onnx_file": file_name,
        "thresholds": {label: float(thresholds[idx]) for idx, label in enumerate(LABELS)},
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "avg_ms_per_text": (inference_seconds / len(texts)) * 1000,
        "throughput_texts_per_second": len(texts) / inference_seconds,
        "metrics": metrics_for_probs(labels, probs, thresholds),
    }


def compare(args: argparse.Namespace) -> Dict[str, object]:
    configure_logging()
    log.info("Loading calibration split: %s", args.calibration_split)
    calibration_texts, calibration_labels = load_split(args.calibration_split, args.calibration_limit)
    log.info("Loading evaluation split: %s", args.eval_split)
    texts, labels = load_split(args.eval_split, args.limit)
    log.info("Evaluation rows: %d", len(texts))

    results: Dict[str, object] = {
        "eval_split": args.eval_split,
        "calibration_split": args.calibration_split,
        "n_eval": len(texts),
        "n_calibration": len(calibration_texts),
        "models": [],
    }

    if args.deberta_model:
        log.info("Running local DeBERTa model: %s", args.deberta_model)
        results["models"].append(
            run_deberta(
                args.deberta_model,
                args.deberta_calibration,
                texts,
                labels,
                args.batch,
                args.max_length,
                args.device,
            )
        )

    if not args.skip_onnx:
        log.info("Running SamLowe ONNX INT8 benchmark")
        results["models"].append(
            run_samlowe_onnx(
                texts,
                labels,
                calibration_texts,
                calibration_labels,
                args.batch,
                args.max_length,
                args.onnx_file,
            )
        )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    log.info("Saved comparison to %s", output_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SentiFlow DeBERTa vs SamLowe ONNX INT8")
    parser.add_argument("--deberta-model", default="./deberta_multilabel/model")
    parser.add_argument("--deberta-calibration", default="./deberta_multilabel/calibration.json")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--onnx-file", default="onnx/model_quantized.onnx")
    parser.add_argument("--eval-split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--calibration-split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--calibration-limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output", default="./model_comparison.json")
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
