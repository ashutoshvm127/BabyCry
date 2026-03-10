#!/usr/bin/env python3
"""
Class Distribution Analyzer

Run this to see the class imbalance in your training data.
Shows which classes are over/under-represented and suggests weights.

Usage:
    python analyze_class_balance.py
"""

from pathlib import Path
from collections import Counter
import os

def analyze_cry_data():
    """Analyze baby cry class distribution"""
    print("\n" + "=" * 70)
    print("  BABY CRY CLASSIFICATION - Class Analysis")
    print("=" * 70)
    
    data_dir = Path("data_baby_respiratory")
    if not data_dir.exists():
        print("  [!] data_baby_respiratory directory not found")
        return
    
    class_counts = {}
    
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            wav_files = list(class_dir.glob("*.wav"))
            mp3_files = list(class_dir.glob("*.mp3"))
            total = len(wav_files) + len(mp3_files)
            class_counts[class_dir.name] = total
    
    if not class_counts:
        print("  [!] No audio files found")
        return
    
    total_samples = sum(class_counts.values())
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    print(f"\n  Total samples: {total_samples}")
    print(f"  Number of classes: {len(class_counts)}")
    print(f"  Imbalance ratio: {max_count / min_count:.1f}x")
    
    print("\n  Class Distribution:")
    print("  " + "-" * 55)
    print(f"  {'Class':<25} {'Count':<8} {'Percentage':<12} {'Status'}")
    print("  " + "-" * 55)
    
    avg_count = total_samples / len(class_counts)
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_samples) * 100
        
        # Status indicator
        if count > avg_count * 1.5:
            status = "⚠️  OVER-REPRESENTED"
        elif count < avg_count * 0.5:
            status = "⚠️  UNDER-REPRESENTED"
        else:
            status = "✓"
        
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {class_name:<25} {count:<8} {pct:>6.1f}%      {status}")
    
    print("  " + "-" * 55)
    
    # Calculate suggested weights
    print("\n  Suggested Class Weights (inverse frequency):")
    print("  " + "-" * 40)
    
    for class_name, count in sorted(class_counts.items()):
        weight = total_samples / (len(class_counts) * count)
        print(f"    {class_name:<25}: {weight:.3f}")
    
    # Identify problem classes
    overrep = [c for c, n in class_counts.items() if n > avg_count * 1.5]
    underrep = [c for c, n in class_counts.items() if n < avg_count * 0.5]
    
    if overrep:
        print(f"\n  ⚠️  Over-represented classes (may dominate training):")
        for c in overrep:
            print(f"      - {c}: {class_counts[c]} samples ({class_counts[c]/avg_count:.1f}x average)")
    
    if underrep:
        print(f"\n  ⚠️  Under-represented classes (may be poorly learned):")
        for c in underrep:
            print(f"      - {c}: {class_counts[c]} samples ({class_counts[c]/avg_count:.1f}x average)")
    
    return class_counts


def analyze_pulmonary_data():
    """Analyze pulmonary class distribution"""
    print("\n" + "=" * 70)
    print("  PULMONARY/RESPIRATORY - Class Analysis")
    print("=" * 70)
    
    class_counts = {}
    
    for data_subdir in ["data_baby_pulmonary", "data_adult_respiratory"]:
        data_dir = Path(data_subdir)
        if data_dir.exists():
            for class_dir in sorted(data_dir.iterdir()):
                if class_dir.is_dir():
                    wav_files = list(class_dir.glob("*.wav"))
                    mp3_files = list(class_dir.glob("*.mp3"))
                    total = len(wav_files) + len(mp3_files)
                    
                    class_name = class_dir.name
                    if class_name in class_counts:
                        class_counts[class_name] += total
                    else:
                        class_counts[class_name] = total
    
    if not class_counts:
        print("  [!] No audio files found")
        return
    
    total_samples = sum(class_counts.values())
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    print(f"\n  Total samples: {total_samples}")
    print(f"  Number of classes: {len(class_counts)}")
    print(f"  Imbalance ratio: {max_count / min_count:.1f}x")
    
    print("\n  Class Distribution:")
    print("  " + "-" * 55)
    print(f"  {'Class':<25} {'Count':<8} {'Percentage':<12} {'Status'}")
    print("  " + "-" * 55)
    
    avg_count = total_samples / len(class_counts)
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_samples) * 100
        
        if count > avg_count * 1.5:
            status = "⚠️  OVER-REPRESENTED"
        elif count < avg_count * 0.5:
            status = "⚠️  UNDER-REPRESENTED"
        else:
            status = "✓"
        
        print(f"  {class_name:<25} {count:<8} {pct:>6.1f}%      {status}")
    
    print("  " + "-" * 55)
    
    return class_counts


def main():
    print("\n" + "=" * 70)
    print("  CLASS BALANCE ANALYZER")
    print("  Identifies imbalanced classes in training data")
    print("=" * 70)
    
    cry_counts = analyze_cry_data()
    pulm_counts = analyze_pulmonary_data()
    
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)
    print("""
  To fix class imbalance, use the balanced training script:
  
    python train_balanced_ensemble.py --task cry --epochs 100
    python train_balanced_ensemble.py --task pulmonary --epochs 100
    
  This script uses:
    ✓ WeightedRandomSampler - balanced batch sampling
    ✓ Focal Loss - focuses on hard examples
    ✓ Effective Number of Samples - advanced class weighting
    ✓ Oversampling - duplicates minority class samples
    ✓ Data Augmentation - varies minority samples
    
  Options:
    --no-oversample    Disable oversampling
    --weight-method    inverse_freq, inverse_sqrt, effective_num, equal
    --epochs           Number of training epochs (default: 100)
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
