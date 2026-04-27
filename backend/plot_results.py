import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_comparison(json_path="./model_comparison.json", output_dir="./plots"):
    """Generate professional charts for PPT from model comparison data."""
    path = Path(json_path)
    if not path.exists():
        print(f"Error: {json_path} not found. Run compare_emotion_models.py first.")
        return

    data = json.loads(path.read_text())
    models = data["models"]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # --- 1. Macro F1 Score Comparison ---
    plt.figure(figsize=(10, 6))
    names = [m["name"].replace("_", " ").title() for m in models]
    f1_scores = [m["metrics"]["macro_f1"] for m in models]
    
    colors = ['#3b82f6', '#10b981'] # Blue and Green
    bars = plt.bar(names, f1_scores, color=colors, width=0.5)
    plt.title("Model Accuracy Comparison (Macro F1)", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Macro F1 Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_comparison.png", dpi=300)
    print(f"Saved: {output_path}/accuracy_comparison.png")

    # --- 2. Latency Comparison (ms per text) ---
    plt.figure(figsize=(10, 6))
    latencies = [m["avg_ms_per_text"] for m in models]
    
    bars = plt.bar(names, latencies, color=['#f59e0b', '#ef4444'], width=0.5)
    plt.title("Inference Latency (Lower is Better)", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Average ms per Text", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}ms', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / "latency_comparison.png", dpi=300)
    print(f"Saved: {output_path}/latency_comparison.png")

    # --- 3. Per-Emotion F1 (Heatmap style or Grouped Bar) ---
    plt.figure(figsize=(12, 7))
    emotions = list(models[0]["metrics"]["per_class"].keys())
    # Filter out average keys
    emotions = [e for e in emotions if e not in ["micro avg", "macro avg", "weighted avg", "samples avg"]]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    for i, m in enumerate(models):
        scores = [m["metrics"]["per_class"][e]["f1-score"] for e in emotions]
        plt.bar(x + (i * width) - (width/2), scores, width, label=m["name"].replace("_", " ").title())

    plt.title("F1 Score per Emotion Category", fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, [e.title() for e in emotions])
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path / "per_emotion_comparison.png", dpi=300)
    print(f"Saved: {output_path}/per_emotion_comparison.png")

if __name__ == "__main__":
    plot_comparison()
