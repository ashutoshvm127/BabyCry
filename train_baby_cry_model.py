#!/usr/bin/env python3
"""
Train Baby Cry Classification Model
====================================
Trains a model on your baby cry dataset: data_baby_respiratory/
Output: ast_baby_cry_optimized/

Uses:
- Wav2Vec2 for audio feature extraction
- Simple classifier head for 8 cry types
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import audio libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[!] librosa not found - installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'librosa', '-q'])
    import librosa
    HAS_LIBROSA = True

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

try:
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    HAS_TRANSFORMERS = True
except ImportError:
    print("[!] transformers not found - installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers', '-q'])
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    HAS_TRANSFORMERS = True

# ======================== CONFIG ========================
DATA_DIR = Path("D:/projects/cry analysuis/data_baby_respiratory")
OUTPUT_DIR = Path("D:/projects/cry analysuis/ast_baby_cry_optimized")
SAMPLE_RATE = 16000
MAX_DURATION = 5.0  # seconds
BATCH_SIZE = 8
EPOCHS = 15  # More epochs for better accuracy
LEARNING_RATE = 1e-4

# Force CUDA - will fail if not available
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available! Please check your GPU drivers.")
    print("        Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)
DEVICE = torch.device("cuda")
print(f"[*] Using GPU: {torch.cuda.get_device_name(0)}")

# Cry type labels
LABELS = [
    "cold_cry", "discomfort_cry", "distress_cry", "hungry_cry",
    "normal_cry", "pain_cry", "sleepy_cry", "tired_cry"
]

label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for idx, label in enumerate(LABELS)}


# ======================== DATASET ========================
class BabyCryDataset(Dataset):
    def __init__(self, audio_paths, labels, feature_extractor, max_length=80000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            if HAS_LIBROSA:
                audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            elif HAS_SOUNDFILE and sf is not None:
                audio, sr = sf.read(audio_path)
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            else:
                raise RuntimeError("No audio library available")
                
            # Truncate or pad
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            elif len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
                
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
                
        except Exception as e:
            print(f"[!] Error loading {audio_path}: {e}")
            audio = np.zeros(self.max_length)
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt",
            padding=True
        )
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ======================== MODEL ========================
class BabyCryClassifier(nn.Module):
    def __init__(self, num_labels, pretrained_model="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        hidden_size = self.wav2vec2.config.hidden_size  # 768 for base
        
        # Freeze early layers to speed up training
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
            
        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # Pool over time dimension
        pooled = hidden_states.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ======================== TRAINING ========================
def load_dataset():
    """Load all audio files from data_baby_respiratory/"""
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    audio_paths = []
    labels = []
    
    for label_name in LABELS:
        label_dir = DATA_DIR / label_name
        if not label_dir.exists():
            print(f"[!] Directory not found: {label_dir}")
            continue
            
        files = list(label_dir.glob("*.wav")) + list(label_dir.glob("*.mp3")) + list(label_dir.glob("*.ogg"))
        
        for f in files:
            audio_paths.append(str(f))
            labels.append(label2id[label_name])
        
        print(f"  {label_name}: {len(files)} files")
    
    print(f"\nTotal: {len(audio_paths)} audio files")
    return audio_paths, labels


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    """Save confusion matrix as image"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Baby Cry Classification - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy info
    accuracy = np.trace(cm) / np.sum(cm) * 100
    plt.figtext(0.5, 0.01, f'Overall Accuracy: {accuracy:.2f}%', 
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved: {output_path}")


def save_training_metrics_graph(history, output_path):
    """Save training metrics graph (loss, accuracy, precision, recall, f1)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Training vs Validation Accuracy
    axes[0, 0].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', markersize=4)
    axes[0, 0].plot(epochs, history['val_acc'], 'r-o', label='Val Accuracy', markersize=4)
    axes[0, 0].set_title('Accuracy Over Epochs', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Precision
    axes[0, 1].plot(epochs, [p*100 for p in history['precision']], 'g-o', label='Precision', markersize=4)
    axes[0, 1].plot(epochs, [r*100 for r in history['recall']], 'm-o', label='Recall', markersize=4)
    axes[0, 1].set_title('Precision & Recall Over Epochs', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, [f*100 for f in history['f1']], 'c-o', label='F1 Score', markersize=4, linewidth=2)
    axes[1, 0].set_title('F1 Score Over Epochs', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].fill_between(epochs, [f*100 for f in history['f1']], alpha=0.3, color='cyan')
    
    # Loss
    axes[1, 1].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[1, 1].set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Baby Cry Model - Training Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training metrics graph saved: {output_path}")


def save_per_class_metrics_graph(y_true, y_pred, labels, output_path):
    """Save per-class precision, recall, f1 bar chart"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Baby Cry Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim((0, 1.1))
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Per-class metrics graph saved: {output_path}")


def train():
    """Train the model"""
    print("\n" + "=" * 60)
    print("BABY CRY CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    audio_paths, labels = load_dataset()
    
    if len(audio_paths) == 0:
        print("[ERROR] No audio files found!")
        return
    
    # Split into train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths, labels, test_size=0.15, stratify=labels, random_state=42
    )
    
    print(f"\nTrain: {len(train_paths)} samples")
    print(f"Validation: {len(val_paths)} samples")
    
    # Load feature extractor
    print("\n[*] Loading Wav2Vec2 feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # Create datasets
    train_dataset = BabyCryDataset(train_paths, train_labels, feature_extractor)
    val_dataset = BabyCryDataset(val_paths, val_labels, feature_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("[*] Creating model...")
    model = BabyCryClassifier(num_labels=len(LABELS)).to(DEVICE)
    
    # Calculate class weights to handle imbalanced data
    # This gives higher weight to minority classes (pain, distress, etc.)
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = []
    print("\n[*] Class weights (to balance imbalanced data):")
    for i in range(len(LABELS)):
        count = class_counts.get(i, 1)
        # Inverse frequency weighting
        weight = total_samples / (len(LABELS) * count)
        class_weights.append(weight)
        print(f"    {LABELS[i]}: {count} samples -> weight {weight:.2f}")
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    # Optimizer and loss with class weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    all_preds = []  # For final classification report
    all_labels = []
    
    # Track training history for graphs
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            input_values = batch["input_values"].to(DEVICE)
            labels_batch = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*train_correct/train_total:.1f}%"
            })
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(DEVICE)
                labels_batch = batch["label"].to(DEVICE)
                
                logits = model(input_values)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += labels_batch.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        
        # Calculate precision, recall, F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        print(f"           Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Track history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss / len(train_loader))
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {"precision": precision, "recall": recall, "f1": f1}
            print(f"  → Saving best model (val_acc={val_acc:.1f}%)")
            
            # Save model
            torch.save(model.state_dict(), OUTPUT_DIR / "pytorch_model.bin")
            
            # Save config
            config = {
                "num_labels": len(LABELS),
                "hidden_size": 768,
                "pretrained_model": "facebook/wav2vec2-base",
                "best_val_acc": best_val_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "epoch": epoch + 1,
            }
            with open(OUTPUT_DIR / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Save label mapping
            label_mapping = {
                "label2id": label2id,
                "id2label": id2label,
            }
            with open(OUTPUT_DIR / "label_mappings.json", "w") as f:
                json.dump(label_mapping, f, indent=2)
            
            # Save feature extractor config
            feature_extractor.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Precision: {best_metrics['precision']:.3f}")
    print(f"Recall: {best_metrics['recall']:.3f}")
    print(f"F1 Score: {best_metrics['f1']:.3f}")
    print(f"Model saved to: {OUTPUT_DIR}")
    
    # Print final per-class classification report
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS (from last epoch)")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0))
    
    # Generate and save graphs
    print("\n[*] Generating visualization graphs...")
    graph_dir = Path("output_graphs")
    graph_dir.mkdir(exist_ok=True)
    
    # Save confusion matrix
    save_confusion_matrix(all_labels, all_preds, LABELS, 
                         graph_dir / "confusion_matrix_baby_cry.png")
    
    # Save training metrics graph
    save_training_metrics_graph(history, graph_dir / "training_curves_baby_cry.png")
    
    # Save per-class metrics graph
    save_per_class_metrics_graph(all_labels, all_preds, LABELS, 
                                 graph_dir / "per_class_metrics_baby_cry.png")
    
    print(f"[OK] Graphs saved to: {graph_dir}")
    
    # Save training summary
    summary = {
        "trained_at": datetime.now().isoformat(),
        "device": str(DEVICE),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_val_acc": best_val_acc,
        "precision": best_metrics['precision'],
        "recall": best_metrics['recall'],
        "f1_score": best_metrics['f1'],
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "labels": LABELS,
    }
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n[OK] Training complete. Restart the server to use the new model.")


if __name__ == "__main__":
    train()
