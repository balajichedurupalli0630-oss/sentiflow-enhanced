"""Shared SentiFlow emotion label utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LABELS: List[str] = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
]

LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

GO_TO_SENTIFLOW: Dict[str, str] = {
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "love": "joy",
    "pride": "joy",
    "relief": "joy",
    "optimism": "joy",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "apprehension": "fear",
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "awe": "surprise",
    "disgust": "disgust",
    "embarrassment": "disgust",
    "contempt": "disgust",
    "admiration": "trust",
    "approval": "trust",
    "caring": "trust",
    "gratitude": "trust",
    "desire": "anticipation",
    "curiosity": "anticipation",
    "interest": "anticipation",
}

EMOTION_EMOJI: Dict[str, str] = {
    "joy": ":)",
    "sadness": ":(",
    "anger": "!",
    "fear": "?",
    "surprise": "*",
    "disgust": "x",
    "trust": "+",
    "anticipation": ">",
}

EMOTION_COLOR: Dict[str, str] = {
    "joy": "#d97706",
    "sadness": "#3b82f6",
    "anger": "#ef4444",
    "fear": "#a855f7",
    "surprise": "#db2777",
    "disgust": "#84cc16",
    "trust": "#22c55e",
    "anticipation": "#f97316",
}


def go_labels_to_sentiflow_ids(go_labels: Iterable[str]) -> List[int]:
    """Map GoEmotions label names to unique SentiFlow label ids."""
    ids = {
        LABEL2ID[mapped]
        for label in go_labels
        for mapped in [GO_TO_SENTIFLOW.get(label)]
        if mapped is not None
    }
    return sorted(ids)


def ids_to_multihot(ids: Sequence[int]) -> List[float]:
    """Convert SentiFlow label ids to an 8-label multi-hot vector."""
    vec = np.zeros(len(LABELS), dtype=np.float32)
    for idx in ids:
        vec[int(idx)] = 1.0
    return vec.tolist()


def go_row_to_multihot(row: dict, int2str) -> Optional[Tuple[List[float], List[str]]]:
    """Return a multi-hot SentiFlow vector for one GoEmotions dataset row."""
    go_labels = [int2str(label_id) for label_id in row.get("labels", [])]
    sentiflow_ids = go_labels_to_sentiflow_ids(go_labels)
    if not sentiflow_ids:
        return None
    sentiflow_labels = [ID2LABEL[idx] for idx in sentiflow_ids]
    return ids_to_multihot(sentiflow_ids), sentiflow_labels


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for NumPy arrays."""
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_x = np.exp(logits[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def sentiflow_scores_from_goemotions(
    go_scores: Dict[str, float],
    *,
    label_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Collapse 28 GoEmotions probabilities into SentiFlow's 8-label space."""
    scores = {label: 0.0 for label in LABELS}
    label_weights = label_weights or {}
    for go_label, raw_score in go_scores.items():
        target = GO_TO_SENTIFLOW.get(go_label)
        if target is None:
            continue
        weighted_score = float(raw_score) * float(label_weights.get(go_label, 1.0))
        scores[target] = max(scores[target], weighted_score)
    return scores
