"""
train_blend_weights.py — Learn Optimal Blending Weights via Regression
=======================================================================
Currently emotion_analyzer.py uses fixed hardcoded weights:
    joy/sadness/anger/fear/surprise/disgust  → 60% Hartmann + 40% GoEmotions
    trust/anticipation                       → 30% Hartmann + 70% GoEmotions

This script:
  1. Runs both models on GoEmotions test split
  2. Collects (hartmann_scores, go_scores, true_label) for every sample
  3. Trains a per-emotion Logistic Regression to learn the optimal weight
     for combining the two model outputs
  4. Saves learned weights to blend_weights.json
  5. Prints a comparison: fixed weights vs learned weights vs oracle

Usage:
    python train_blend_weights.py                          # CPU
    python train_blend_weights.py --gpu                    # MPS/CUDA
    python train_blend_weights.py --adapter adapter/adapter  # with fine-tuned adapter

Output:
    blend_weights.json   ← paste these into emotion_analyzer.py
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentiflow.blend")

# ── Constants ──────────────────────────────────────────────────────────────────

HARTMANN_MODEL = "j-hartmann/emotion-english-distilroberta-base"
GO_MODEL       = "SamLowe/roberta-base-go_emotions"

LABELS   = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# Hartmann's 7 labels → our 8
HARTMANN_MAP = {
    "joy":      "joy",
    "sadness":  "sadness",
    "anger":    "anger",
    "fear":     "fear",
    "surprise": "surprise",
    "disgust":  "disgust",
    "love":     "joy",
    "neutral":  "surprise",
}

# GoEmotions 28 labels → our 8
GO_MAP = {
    "joy":"joy","amusement":"joy","excitement":"joy","love":"joy",
    "pride":"joy","relief":"joy","optimism":"joy",
    "sadness":"sadness","grief":"sadness","disappointment":"sadness","remorse":"sadness",
    "anger":"anger","annoyance":"anger","disapproval":"anger",
    "fear":"fear","nervousness":"fear","apprehension":"fear",
    "surprise":"surprise","realization":"surprise","confusion":"surprise","awe":"surprise",
    "disgust":"disgust","embarrassment":"disgust","contempt":"disgust",
    "admiration":"trust","approval":"trust","caring":"trust","gratitude":"trust",
    "desire":"anticipation","curiosity":"anticipation","interest":"anticipation",
}

# ── Device ─────────────────────────────────────────────────────────────────────

def get_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "cpu"
    if torch.cuda.is_available():
        log.info("Using CUDA")
        return "cuda"
    if torch.backends.mps.is_available():
        log.info("Using Apple MPS")
        return "mps"
    log.warning("No GPU — using CPU")
    return "cpu"

# ── Model loading ──────────────────────────────────────────────────────────────

def load_hartmann(adapter_path: Optional[str], device: str):
    hf_device = 0 if device == "cuda" else -1
    if adapter_path and Path(adapter_path).exists():
        log.info("Loading fine-tuned Hartmann adapter from %s", adapter_path)
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(
            HARTMANN_MODEL, num_labels=8, ignore_mismatched_sizes=True
        )
        model     = PeftModel.from_pretrained(base, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if device in ("cuda", "mps"):
            model = model.to(device)
        pipe = pipeline(
            "text-classification", model=model, tokenizer=tokenizer,
            device=hf_device if device == "cuda" else -1,
            top_k=None, truncation=True, max_length=512,
        )
        if device == "mps":
            pipe.model = pipe.model.to("mps")
        return pipe, "hartmann_finetuned"
    else:
        log.info("Loading base Hartmann model")
        return pipeline(
            "text-classification", model=HARTMANN_MODEL,
            device=hf_device, top_k=None, truncation=True, max_length=512,
        ), "hartmann_base"

def load_go(device: str):
    hf_device = 0 if device == "cuda" else -1
    log.info("Loading GoEmotions model")
    return pipeline(
        "text-classification", model=GO_MODEL,
        device=hf_device, top_k=None, truncation=True, max_length=512,
    )

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(split: str = "test", limit: Optional[int] = None) -> Tuple[List[str], List[int]]:
    log.info("Loading GoEmotions %s split…", split)
    raw     = load_dataset("google-research-datasets/go_emotions", "simplified")
    int2str = raw[split].features["labels"].feature.int2str
    texts, labels = [], []
    for row in raw[split]:
        seen = set()
        for lid in row["labels"]:
            sf = GO_MAP.get(int2str(lid))
            if sf and sf not in seen:
                texts.append(row["text"])
                labels.append(LABEL2ID[sf])
                seen.add(sf)
    if limit:
        texts, labels = texts[:limit], labels[:limit]
    log.info("  %d samples loaded", len(texts))
    return texts, labels

# ── Inference → score matrices ─────────────────────────────────────────────────

def run_pipe_batch(pipe, texts: List[str], batch_size: int = 32) -> List[dict]:
    """Returns list of {label: score} dicts, one per text."""
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        outs  = pipe(batch)
        for out in outs:
            items = out if isinstance(out, list) else [out]
            all_scores.append({x["label"].lower(): x["score"] for x in items})
        if i % 320 == 0:
            log.info("  inference %d/%d", min(i+batch_size, len(texts)), len(texts))
    return all_scores

def hartmann_to_8class(scores: dict) -> np.ndarray:
    """Convert Hartmann's raw output dict to an 8-class probability vector."""
    vec = np.zeros(8)
    for label, score in scores.items():
        target = HARTMANN_MAP.get(label)
        if target:
            idx = LABEL2ID[target]
            vec[idx] = max(vec[idx], score)
    # Normalise so it sums to 1
    s = vec.sum()
    return vec / s if s > 0 else vec

def go_to_8class(scores: dict) -> np.ndarray:
    """Convert GoEmotions raw output dict to an 8-class probability vector."""
    vec = np.zeros(8)
    for label, score in scores.items():
        target = GO_MAP.get(label)
        if target:
            idx = LABEL2ID[target]
            vec[idx] = max(vec[idx], score)
    s = vec.sum()
    return vec / s if s > 0 else vec

def build_score_matrices(
    hartmann_pipe, go_pipe,
    texts: List[str], batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        H — (N, 8) Hartmann scores
        G — (N, 8) GoEmotions scores
    """
    log.info("Running Hartmann inference…")
    h_raw = run_pipe_batch(hartmann_pipe, texts, batch_size)
    log.info("Running GoEmotions inference…")
    g_raw = run_pipe_batch(go_pipe, texts, batch_size)

    H = np.array([hartmann_to_8class(s) for s in h_raw])
    G = np.array([go_to_8class(s)       for s in g_raw])
    return H, G


def load_or_build_score_matrices(
    *,
    cache_path: Optional[str],
    hartmann_pipe,
    go_pipe,
    texts: List[str],
    labels: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path:
        cache = Path(cache_path)
        if cache.exists():
            log.info("Loading score cache from %s", cache)
            arr = np.load(cache, allow_pickle=False)
            H = arr["H"]
            G = arr["G"]
            y = arr["y"]
            if len(H) == len(texts) == len(G) == len(y):
                return H, G
            log.warning(
                "Cache size mismatch (cache=%d, current=%d) — rebuilding cache",
                len(H), len(texts),
            )

    H, G = build_score_matrices(hartmann_pipe, go_pipe, texts)
    if cache_path:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, H=H, G=G, y=np.array(labels, dtype=np.int64))
        log.info("Saved score cache to %s", cache)
    return H, G

# ── Blend evaluation helpers ───────────────────────────────────────────────────

def blend_fixed(H: np.ndarray, G: np.ndarray,
                h_weight: float = 0.60, g_weight: float = 0.40) -> np.ndarray:
    """Apply fixed weights globally — current behavior in emotion_analyzer.py."""
    return H * h_weight + G * g_weight

def blend_per_emotion(H: np.ndarray, G: np.ndarray,
                      weights: Dict[str, float]) -> np.ndarray:
    """
    Apply per-emotion learned weights.
    weights[emotion] = hartmann_weight  (go_weight = 1 - hartmann_weight)
    """
    blended = np.zeros_like(H)
    for i, emotion in enumerate(LABELS):
        hw = weights.get(emotion, 0.60)
        gw = 1.0 - hw
        blended[:, i] = H[:, i] * hw + G[:, i] * gw
    return blended

def evaluate_blend(blended: np.ndarray, true_labels: List[int], name: str) -> dict:
    preds    = np.argmax(blended, axis=1)
    macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    acc      = accuracy_score(true_labels, preds)
    per_em   = f1_score(true_labels, preds, average=None, labels=list(range(8)), zero_division=0)
    log.info("%-30s  Macro F1=%.4f  Acc=%.4f", name, macro_f1, acc)
    return {
        "name":      name,
        "macro_f1":  round(macro_f1, 4),
        "accuracy":  round(acc, 4),
        "per_emotion_f1": {LABELS[i]: round(float(per_em[i]), 4) for i in range(8)},
    }

# ── Regression weight learning ─────────────────────────────────────────────────

def learn_weights_regression(
    H: np.ndarray,           # (N, 8) Hartmann scores
    G: np.ndarray,           # (N, 8) GoEmotions scores
    true_labels: List[int],  # (N,) ground truth class indices
) -> Dict[str, float]:
    """
    For each emotion class i, learn the optimal alpha such that:
        score_i = alpha * H_i + (1 - alpha) * G_i
    maximises classification accuracy on that emotion (one-vs-rest).

    Method: Grid search over alpha ∈ [0.0, 1.0] with 0.05 step,
    picking the alpha with best binary F1 for that emotion.
    Then fits a Logistic Regression on [H_i, G_i] features as validation.

    Returns dict: {emotion: hartmann_weight}
    """
    log.info("Learning per-emotion blend weights via regression…")
    weights = {}
    y = np.array(true_labels)

    print(f"\n{'─'*65}")
    print(f"  {'Emotion':<14} {'Best α (H)':<12} {'Fixed F1':<12} {'Learned F1':<12} {'Gain'}")
    print(f"{'─'*65}")

    alphas = np.arange(0.0, 1.05, 0.05)

    for i, emotion in enumerate(LABELS):
        # Binary: is this sample this emotion?
        y_binary = (y == i).astype(int)

        # Fixed weight baseline
        fixed_h = 0.30 if emotion in ("trust", "anticipation") else 0.60
        fixed_blend = H[:, i] * fixed_h + G[:, i] * (1 - fixed_h)
        fixed_preds  = (fixed_blend >= np.median(fixed_blend)).astype(int)
        fixed_f1     = f1_score(y_binary, fixed_preds, zero_division=0)

        # Grid search over alpha
        best_alpha, best_f1 = fixed_h, fixed_f1
        for alpha in alphas:
            blended_i = H[:, i] * alpha + G[:, i] * (1 - alpha)
            preds_i   = (blended_i >= np.median(blended_i)).astype(int)
            f1_i      = f1_score(y_binary, preds_i, zero_division=0)
            if f1_i > best_f1:
                best_f1   = f1_i
                best_alpha = alpha

        # Validate with Logistic Regression on [H_i, G_i] features
        X_feat = np.column_stack([H[:, i], G[:, i]])
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_feat)

        # Only train LR if we have enough positive samples
        if y_binary.sum() >= 10:
            lr = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
            cv_f1 = cross_val_score(lr, X_sc, y_binary, cv=5,
                                    scoring="f1", error_score=0.0).mean()
            lr.fit(X_sc, y_binary)
            # Extract implied weight: coef[0]=H, coef[1]=G
            coef = lr.coef_[0]
            if coef.sum() != 0:
                lr_alpha = max(0.0, min(1.0, coef[0] / (coef[0] + coef[1] + 1e-9)))
                # Blend: use LR weight if it validates better, else grid-search winner
                final_alpha = lr_alpha if cv_f1 > best_f1 else best_alpha
            else:
                final_alpha = best_alpha
        else:
            final_alpha = best_alpha
            cv_f1       = best_f1

        gain = best_f1 - fixed_f1
        sign = "+" if gain >= 0 else ""
        print(f"  {emotion:<14} α={final_alpha:.2f}        {fixed_f1:.3f}       {best_f1:.3f}       {sign}{gain:.3f}")

        weights[emotion] = round(float(final_alpha), 3)

    print(f"{'─'*65}\n")
    return weights

# ── Stacked meta-learner (advanced) ───────────────────────────────────────────

def learn_weights_stacking(
    H: np.ndarray,
    G: np.ndarray,
    true_labels: List[int],
) -> Dict[str, float]:
    """
    Advanced: trains a Logistic Regression meta-learner on the full
    16-feature input [H_1..H_8, G_1..G_8] → 8-class output.
    Extracts implied per-emotion blend weights from the model coefficients.

    This captures cross-emotion interactions (e.g. high G_trust being
    a signal for anticipation too).
    """
    log.info("Training stacked meta-learner on full 16-feature input…")
    X = np.hstack([H, G])         # (N, 16)
    y = np.array(true_labels)     # (N,)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    meta = LogisticRegression(
        max_iter=1000,
        C=0.5,
        solver="lbfgs",
        class_weight="balanced",
    )
    cv_scores = cross_val_score(meta, X_sc, y, cv=5, scoring="f1_macro")
    log.info("Meta-learner 5-fold CV macro F1: %.4f ± %.4f",
             cv_scores.mean(), cv_scores.std())

    meta.fit(X_sc, y)

    # Extract implied blend weights per emotion class
    # coef shape: (8 classes, 16 features) — first 8 = H, last 8 = G
    weights = {}
    print(f"\n{'─'*55}")
    print(f"  {'Emotion':<14} {'H coef':>8} {'G coef':>8} {'Implied α':>10}")
    print(f"{'─'*55}")
    for i, emotion in enumerate(LABELS):
        h_coef = float(meta.coef_[i, i])          # H_i weight for class i
        g_coef = float(meta.coef_[i, i + 8])      # G_i weight for class i
        total  = abs(h_coef) + abs(g_coef) + 1e-9
        alpha  = max(0.0, min(1.0, (h_coef + total/2) / total))
        weights[emotion] = round(alpha, 3)
        print(f"  {emotion:<14} {h_coef:>8.3f} {g_coef:>8.3f} {alpha:>10.3f}")
    print(f"{'─'*55}\n")

    return weights, meta, scaler, cv_scores.mean()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",  type=str, default=None,
                        help="Path to fine-tuned adapter (e.g. adapter/adapter)")
    parser.add_argument("--gpu",      action="store_true")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Limit samples for quick testing (e.g. 500)")
    parser.add_argument("--split",    type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use for learning weights")
    parser.add_argument("--method",   type=str, default="both",
                        choices=["grid", "stacking", "both"],
                        help="Weight learning method")
    parser.add_argument("--cache",    type=str, default=None,
                        help="Optional .npz cache for model score matrices")
    parser.add_argument("--output",   type=str, default="blend_weights.json")
    args = parser.parse_args()

    device = get_device(args.gpu)

    # ── Load models ───────────────────────────────────────────────────────────
    hartmann_pipe, hartmann_name = load_hartmann(args.adapter, device)
    go_pipe = load_go(device)

    # ── Load data ─────────────────────────────────────────────────────────────
    texts, true_labels = load_data(args.split, args.limit)

    # ── Run both models → score matrices ──────────────────────────────────────
    t0 = time.time()
    H, G = load_or_build_score_matrices(
        cache_path=args.cache,
        hartmann_pipe=hartmann_pipe,
        go_pipe=go_pipe,
        texts=texts,
        labels=true_labels,
    )
    log.info("Inference done in %.1fs", time.time() - t0)

    # ── Baseline: fixed weights (current behavior) ────────────────────────────
    print("\n" + "═"*65)
    print("  BASELINE — FIXED WEIGHTS (current emotion_analyzer.py)")
    print("═"*65)

    # Current system uses different weights for trust/anticipation
    fixed_blend = np.zeros_like(H)
    for i, em in enumerate(LABELS):
        hw = 0.30 if em in ("trust", "anticipation") else 0.60
        fixed_blend[:, i] = H[:, i] * hw + G[:, i] * (1 - hw)
    baseline = evaluate_blend(fixed_blend, true_labels, "fixed_weights (current)")

    # ── Oracle: best possible with any fixed global weight ────────────────────
    print("\n" + "═"*65)
    print("  ORACLE — BEST SINGLE GLOBAL WEIGHT")
    print("═"*65)
    best_oracle_f1, best_oracle_alpha = 0, 0.60
    for alpha in np.arange(0.0, 1.05, 0.05):
        blend = H * alpha + G * (1 - alpha)
        f1    = f1_score(true_labels, np.argmax(blend, axis=1),
                         average="macro", zero_division=0)
        if f1 > best_oracle_f1:
            best_oracle_f1, best_oracle_alpha = f1, alpha
    log.info("Best global alpha=%.2f  F1=%.4f", best_oracle_alpha, best_oracle_f1)
    oracle_blend  = H * best_oracle_alpha + G * (1 - best_oracle_alpha)
    oracle_result = evaluate_blend(oracle_blend, true_labels,
                                   f"oracle_global (α={best_oracle_alpha:.2f})")

    # ── Learn weights ─────────────────────────────────────────────────────────
    learned_weights = {}
    stacking_result = None

    if args.method in ("grid", "both"):
        print("\n" + "═"*65)
        print("  METHOD 1 — PER-EMOTION GRID SEARCH + LOGISTIC REGRESSION")
        print("═"*65)
        grid_weights = learn_weights_regression(H, G, true_labels)
        grid_blend   = blend_per_emotion(H, G, grid_weights)
        grid_result  = evaluate_blend(grid_blend, true_labels, "learned_per_emotion")
        learned_weights["grid"] = grid_weights

    if args.method in ("stacking", "both"):
        print("\n" + "═"*65)
        print("  METHOD 2 — STACKED META-LEARNER (16-feature LogReg)")
        print("═"*65)
        stack_weights, meta_model, scaler, cv_f1 = learn_weights_stacking(
            H, G, true_labels
        )
        stack_blend  = blend_per_emotion(H, G, stack_weights)
        stacking_result = evaluate_blend(stack_blend, true_labels, "stacking_meta")
        learned_weights["stacking"] = stack_weights

    # ── Pick best method ──────────────────────────────────────────────────────
    if args.method == "both":
        if grid_result["macro_f1"] >= stacking_result["macro_f1"]:
            best_weights = grid_weights
            best_method  = "grid"
            best_result  = grid_result
        else:
            best_weights = stack_weights
            best_method  = "stacking"
            best_result  = stacking_result
    elif args.method == "grid":
        best_weights = grid_weights
        best_method  = "grid"
        best_result  = grid_result
    else:
        best_weights = stack_weights
        best_method  = "stacking"
        best_result  = stacking_result

    # ── Final comparison ──────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  FINAL COMPARISON")
    print("═"*65)
    print(f"  Fixed weights (current) : Macro F1={baseline['macro_f1']:.4f}  Acc={baseline['accuracy']:.4f}")
    print(f"  Oracle global weight    : Macro F1={oracle_result['macro_f1']:.4f}  Acc={oracle_result['accuracy']:.4f}")
    if args.method in ("grid","both"):
        print(f"  Grid per-emotion        : Macro F1={grid_result['macro_f1']:.4f}  Acc={grid_result['accuracy']:.4f}")
    if args.method in ("stacking","both"):
        print(f"  Stacking meta-learner   : Macro F1={stacking_result['macro_f1']:.4f}  Acc={stacking_result['accuracy']:.4f}")

    delta = best_result["macro_f1"] - baseline["macro_f1"]
    sign  = "+" if delta >= 0 else ""
    print(f"\n  Best method: {best_method}  →  {sign}{delta:.4f} vs fixed weights")

    print(f"\n{'─'*65}")
    print("  PER-EMOTION F1 COMPARISON")
    print(f"  {'Emotion':<14} {'Fixed':>8} {'Learned':>9} {'Gain':>8}")
    print(f"{'─'*65}")
    for em in LABELS:
        f_fixed   = baseline["per_emotion_f1"][em]
        f_learned = best_result["per_emotion_f1"][em]
        gain      = f_learned - f_fixed
        sign      = "+" if gain >= 0 else ""
        flag      = " ✓" if gain > 0.01 else (" ✗" if gain < -0.01 else "")
        print(f"  {em:<14} {f_fixed:>8.3f} {f_learned:>9.3f} {sign}{gain:>7.3f}{flag}")
    print(f"{'─'*65}\n")

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "description": "Per-emotion Hartmann blend weights. go_weight = 1 - hartmann_weight.",
        "method":      best_method,
        "hartmann_model": hartmann_name,
        "dataset_split":  args.split,
        "n_samples":      len(texts),
        "metrics": {
            "fixed_macro_f1":   baseline["macro_f1"],
            "learned_macro_f1": best_result["macro_f1"],
            "improvement":      round(best_result["macro_f1"] - baseline["macro_f1"], 4),
        },
        "weights": best_weights,   # {emotion: hartmann_weight}
        "all_weights": learned_weights,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Weights saved → %s", args.output)

    # ── Print paste-ready code ────────────────────────────────────────────────
    print("=" * 65)
    print("  PASTE THIS INTO emotion_analyzer.py → _blend_emotion_scores()")
    print("=" * 65)
    print("""
# Replace the hardcoded weights with these learned values:
LEARNED_BLEND_WEIGHTS = {""")
    for em, w in best_weights.items():
        gw = round(1.0 - w, 3)
        print(f'    "{em}": {{"hartmann": {w}, "go": {gw}}},')
    print("}")
    print("""
# Then update _blend_emotion_scores():
def _blend_emotion_scores(self, hartmann, go_em, **_):
    blended = {}
    for emotion in set(hartmann) | set(go_em):
        w = LEARNED_BLEND_WEIGHTS.get(emotion, {"hartmann": 0.60, "go": 0.40})
        h = hartmann.get(emotion, 0.0)
        g = go_em.get(emotion, 0.0)
        blended[emotion] = h * w["hartmann"] + g * w["go"]
    return blended
""")
    print("=" * 65)


if __name__ == "__main__":
    main()
