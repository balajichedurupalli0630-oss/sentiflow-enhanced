import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_variants():
    """Plot comparison of the 3 completed model variants."""
    f1_variants = {
        "Default DeBERTa": 0.6158,
        "Corporate Tuned": 0.6340,
        "Massive Augmentation": 0.6338
    }
    
    acc_variants = {
        "Default DeBERTa": 0.7404,
        "Corporate Tuned": 0.7373,
        "Massive Augmentation": 0.7416
    }
    
    names = list(f1_variants.keys())
    f1_scores = list(f1_variants.values())
    acc_scores = list(acc_variants.values())
    
    output_path = Path("./plots")
    output_path.mkdir(exist_ok=True)

    # --- 1. Macro F1 Bar Chart (The Nuance Metric) ---
    plt.figure(figsize=(10, 6))
    colors = ['#94a3b8', '#3b82f6', '#10b981'] # Grey, Blue, Green
    bars = plt.bar(names, f1_scores, color=colors, width=0.6)
    
    plt.title("SentiFlow Model Variants: Macro F1 (Nuance Detection)", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Macro F1 Score", fontsize=12)
    plt.ylim(0.55, 0.65)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path / "variant_macro_f1.png", dpi=300)
    print(f"Saved: {output_path}/variant_macro_f1.png")

    # --- 2. Primary Accuracy Bar Chart (The Top-Level Success Rate) ---
    plt.figure(figsize=(10, 6))
    colors = ['#94a3b8', '#3b82f6', '#10b981']
    bars = plt.bar(names, acc_scores, color=colors, width=0.6)
    
    plt.title("SentiFlow Model Variants: Primary Accuracy (Top Emotion)", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Accuracy Rate (0.0 - 1.0)", fontsize=12)
    plt.ylim(0.70, 0.76)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path / "variant_primary_accuracy.png", dpi=300)
    print(f"Saved: {output_path}/variant_primary_accuracy.png")

if __name__ == "__main__":
    plot_variants()
