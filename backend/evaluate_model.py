"""
evaluate_model.py - Formal evaluation on GoEmotions test set
Uses get_analyzer() to access the singleton instance
"""

import asyncio
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from emotion_analyzer import _init_analyzer, get_analyzer  # Import both
from tqdm import tqdm
import json
import time

# Constants
LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# Map GoEmotions labels to your 8 emotions
GO_MAP = {
    "joy": "joy", "amusement": "joy", "excitement": "joy", "love": "joy",
    "pride": "joy", "relief": "joy", "optimism": "joy",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness", 
    "remorse": "sadness",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "fear": "fear", "nervousness": "fear", "apprehension": "fear",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise", 
    "awe": "surprise",
    "disgust": "disgust", "embarrassment": "disgust", "contempt": "disgust",
    "admiration": "trust", "approval": "trust", "caring": "trust", "gratitude": "trust",
    "desire": "anticipation", "curiosity": "anticipation", "interest": "anticipation"
}

async def evaluate_on_goemotions(limit=None, use_gpu=False):
    """
    Evaluate model on GoEmotions test set using the singleton analyzer
    """
    print("=" * 70)
    print("FORMAL EVALUATION ON GOEMOTIONS TEST SET")
    print("=" * 70)
    
    # Load dataset
    print("\n📚 Loading GoEmotions test set...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    int2str = dataset.features["labels"].feature.int2str
    
    # Prepare test data
    test_texts = []
    test_labels = []
    label_distribution = Counter()
    
    for row in dataset:
        if limit and len(test_texts) >= limit:
            break
        
        if row["labels"]:
            lid = row["labels"][0]
            go_label = int2str(lid)
            if go_label in GO_MAP:
                mapped_label = GO_MAP[go_label]
                test_texts.append(row["text"])
                test_labels.append(LABEL2ID[mapped_label])
                label_distribution[mapped_label] += 1
    
    print(f"\n📊 Test set size: {len(test_texts)} samples")
    print("\nLabel distribution in test set:")
    for label in LABELS:
        count = label_distribution.get(label, 0)
        percentage = (count / len(test_texts)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {label:12}: {count:5} samples ({percentage:5.1f}%) {bar}")
    
    # Initialize the analyzer (creates singleton)
    print("\n🚀 Initializing emotion analyzer...")
    start_time = time.time()
    await _init_analyzer(use_gpu=use_gpu)
    load_time = time.time() - start_time
    print(f"✅ Analyzer initialized in {load_time:.2f} seconds")
    
    # Get the already-initialized instance
    analyzer = get_analyzer()
    print(f"✅ Retrieved analyzer instance (singleton)")
    
    # Run predictions
    print("\n🔍 Running predictions...")
    predictions = []
    confidence_scores = []
    
    start_time = time.time()
    for text in tqdm(test_texts, desc="Analyzing", unit="text"):
        result = await analyzer.analyze(text)
        pred_label = result["primary_emotion"]
        confidence = result["emotion_score"]
        
        predictions.append(LABEL2ID[pred_label])
        confidence_scores.append(confidence)
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    
    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(test_labels, predictions, average="micro", zero_division=0)
    weighted_f1 = f1_score(test_labels, predictions, average="weighted", zero_division=0)
    
    per_class_f1 = f1_score(test_labels, predictions, average=None, zero_division=0)
    
    class_report = classification_report(
        test_labels, predictions, 
        target_names=LABELS, 
        zero_division=0,
        output_dict=True
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Overall Performance:")
    print(f"  Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1:      {macro_f1:.4f}")
    print(f"  Micro F1:      {micro_f1:.4f}")
    print(f"  Weighted F1:   {weighted_f1:.4f}")
    
    print(f"\n🎯 Per-Class Performance:")
    print(f"  {'Emotion':12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    
    for i, label in enumerate(LABELS):
        prec = class_report[label]['precision']
        rec = class_report[label]['recall']
        f1 = class_report[label]['f1-score']
        sup = class_report[label]['support']
        print(f"  {label:12} {prec:10.4f} {rec:10.4f} {f1:10.4f} {sup:8.0f}")
    
    # Confusion matrix
    print(f"\n🔍 Common Confusions:")
    cm = confusion_matrix(test_labels, predictions)
    for i in range(len(LABELS)):
        row = cm[i]
        if row.sum() > 0:
            # Find most common wrong prediction
            wrong_predictions = [(j, row[j]) for j in range(len(LABELS)) if j != i and row[j] > 0]
            if wrong_predictions:
                most_confused, conf_count = max(wrong_predictions, key=lambda x: x[1])
                total = row.sum()
                percentage = (conf_count / total) * 100
                if percentage > 5:
                    print(f"  {LABELS[i]:12} → often as {LABELS[most_confused]:12} ({percentage:.1f}% of {total} cases)")
    
    # Performance metrics
    avg_confidence = np.mean(confidence_scores)
    avg_inference_time = (inference_time / len(test_texts)) * 1000
    
    print(f"\n⚡ Performance Metrics:")
    print(f"  Total inference time: {inference_time:.2f} seconds")
    print(f"  Avg per text:         {avg_inference_time:.1f} ms")
    print(f"  Avg confidence:       {avg_confidence:.1f}%")
    print(f"  Throughput:           {len(test_texts)/inference_time:.1f} texts/second")
    
    # Compare with baseline
    baseline_f1 = 0.3709
    improvement = macro_f1 - baseline_f1
    
    print(f"\n📈 Comparison with Baseline:")
    print(f"  Fixed weights baseline:  {baseline_f1:.4f}")
    print(f"  Your optimized model:    {macro_f1:.4f}")
    print(f"  Improvement:             {improvement:+.4f} ({improvement*100:+.1f}%)")
    
    # Store results
    results = {
        "model_version": "4.0.0",
        "test_set_size": len(test_texts),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "avg_confidence": float(avg_confidence),
        "avg_inference_time_ms": float(avg_inference_time),
        "throughput": float(len(test_texts)/inference_time),
        "per_class": {
            label: {
                "precision": float(class_report[label]['precision']),
                "recall": float(class_report[label]['recall']),
                "f1": float(class_report[label]['f1-score']),
                "support": int(class_report[label]['support'])
            }
            for label in LABELS
        },
        "improvement_over_baseline": float(improvement)
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to evaluation_results.json")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    
    return results

async def quick_evaluation():
    """Quick evaluation with 500 samples"""
    print("\n🚀 Running QUICK evaluation (500 samples)...")
    results = await evaluate_on_goemotions(limit=500, use_gpu=False)
    return results

async def full_evaluation():
    """Full evaluation on complete test set"""
    print("\n🚀 Running FULL evaluation (all samples)...")
    results = await evaluate_on_goemotions(limit=None, use_gpu=True)
    return results

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("SENTIFLOW MODEL EVALUATION SUITE")
    print("=" * 70)
    print("\nChoose evaluation mode:")
    print("  1. Quick evaluation (500 samples, ~2-3 minutes)")
    print("  2. Full evaluation (all samples, ~15-20 minutes)")
    print("  3. Custom limit")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        asyncio.run(quick_evaluation())
    elif choice == "2":
        confirm = input("Full evaluation will take 15-20 minutes. Continue? (y/n): ")
        if confirm.lower() == 'y':
            asyncio.run(full_evaluation())
        else:
            print("Cancelled.")
    elif choice == "3":
        limit = int(input("Enter number of samples to test: "))
        asyncio.run(evaluate_on_goemotions(limit=limit, use_gpu=False))
    else:
        print("Invalid choice. Running quick evaluation...")
        asyncio.run(quick_evaluation())