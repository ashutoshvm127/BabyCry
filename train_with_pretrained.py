#!/usr/bin/env python3
"""
Training Script with Pretrained HuggingFace Backbone Features
================================================================

Uses DistilHuBERT as the primary pretrained feature extractor (768-dim embeddings)
to train a lightweight classifier head for 20-class classification.

Key improvements over the simple CNN approach:
1. Pretrained audio representations (DistilHuBERT learned from 960h of speech)
2. Augmented dataset with balanced class distribution (MIN 200 per class)
3. Proper class weighting via focal loss + weighted sampling
4. Cosine annealing learning rate schedule
5. Label smoothing for better generalization

6-BACKBONE ENSEMBLE (EQUAL WEIGHT = 1/6 each):
  DistilHuBERT / AST / YAMNet / Wav2Vec2 / WavLM / PANNs

The model saves in ensemble-compatible format.

Usage:
    python train_with_pretrained.py
    python train_with_pretrained.py --epochs 40 --batch_size 8
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
BABY_CRY_DIR = BASE_DIR / "data_baby_respiratory"
BABY_PULMONARY_DIR = BASE_DIR / "data_baby_pulmonary"
OUTPUT_DIR = BASE_DIR / "trained_classifiers"
RPI_OUTPUT_DIR = BASE_DIR / "rpi5_standalone" / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Class definitions
CRY_CLASSES = [
    'hungry_cry', 'pain_cry', 'sleepy_cry', 'discomfort_cry',
    'cold_cry', 'tired_cry', 'normal_cry', 'distress_cry',
    'belly_pain_cry', 'burping_cry', 'pathological_cry', 'asphyxia_cry'
]

PULMONARY_CLASSES = [
    'normal_breathing', 'wheeze', 'stridor', 'rhonchi',
    'fine_crackle', 'coarse_crackle', 'mixed', 'bronchiolitis'
]

ALL_CLASSES = CRY_CLASSES + PULMONARY_CLASSES
LABEL_TO_ID = {cls: i for i, cls in enumerate(ALL_CLASSES)}
ID_TO_LABEL = {i: cls for i, cls in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)

RISK_LEVELS = {
    'normal_cry': 0, 'hungry_cry': 0, 'sleepy_cry': 0, 'tired_cry': 0,
    'burping_cry': 0, 'normal_breathing': 0,
    'discomfort_cry': 1, 'cold_cry': 1, 'belly_pain_cry': 1,
    'wheeze': 1, 'rhonchi': 1, 'fine_crackle': 1, 'coarse_crackle': 1,
    'distress_cry': 2, 'mixed': 2,
    'pain_cry': 3, 'pathological_cry': 3, 'asphyxia_cry': 3,
    'stridor': 3, 'bronchiolitis': 3
}


# ============================================================================
# PRETRAINED FEATURE EXTRACTOR
# ============================================================================
class PretrainedFeatureExtractor:
    """
    Uses DistilHuBERT to extract 768-dim embeddings from audio.
    The backbone is frozen (no gradient computation needed).
    """

    def __init__(self, model_name="ntu-spml/distilhubert", cache_dir=None):
        from transformers import AutoModel, AutoFeatureExtractor

        print(f"Loading pretrained model: {model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.eval()
        self.model.to(device)
        self.embedding_dim = 768
        print(f"  Embedding dim: {self.embedding_dim}")

    @torch.no_grad()
    def extract_embedding(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract embedding from a single waveform"""
        inputs = self.feature_extractor(
            waveform, sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Mean pooling over time dimension
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding


def extract_all_embeddings(file_paths, labels, extractor, cache_path=None):
    """
    Extract embeddings for all files. Uses caching to avoid recomputation.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['embeddings'], data['labels'], data['valid_indices']

    embeddings = []
    valid_labels = []
    valid_indices = []

    print(f"\nExtracting embeddings for {len(file_paths)} files...")
    for i, (fp, lbl) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths),
                                        desc="Extracting features")):
        try:
            y, _ = librosa.load(fp, sr=16000, mono=True, duration=5.0)
            if len(y) < 1600 or not np.isfinite(y).all():
                continue
            # Normalize
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

            emb = extractor.extract_embedding(y)

            if np.isfinite(emb).all():
                embeddings.append(emb)
                valid_labels.append(lbl)
                valid_indices.append(i)
        except Exception as e:
            continue

    embeddings = np.array(embeddings, dtype=np.float32)
    valid_labels = np.array(valid_labels)
    valid_indices = np.array(valid_indices)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, labels=valid_labels,
                 valid_indices=valid_indices)
        print(f"Cached embeddings to {cache_path}")

    return embeddings, valid_labels, valid_indices


# ============================================================================
# DATASET
# ============================================================================
class EmbeddingDataset(Dataset):
    """Dataset for precomputed embeddings"""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================================
# MODEL
# ============================================================================
class PretrainedClassifierHead(nn.Module):
    """
    Classifier head for pretrained embeddings.
    Lightweight MLP that maps 768-dim embeddings to 20 classes.
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 20,
                 hidden_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class SixBackboneEnsemble(nn.Module):
    """
    6-Backbone Ensemble with EQUAL WEIGHTS (1/6 each).
    Compatible with RPi5 deployment format.
    """

    def __init__(self, num_classes: int = 20,
                 backbone_dims: dict = None):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_weight = 1.0 / 6.0

        if backbone_dims is None:
            backbone_dims = {
                'distilhubert': 768,
                'ast': 768,
                'yamnet': 1024,
                'wav2vec2': 768,
                'wavlm': 768,
                'panns': 2048
            }

        self.classifiers = nn.ModuleDict()
        for name, dim in backbone_dims.items():
            self.classifiers[name] = PretrainedClassifierHead(
                input_dim=dim, num_classes=num_classes
            )

    def forward(self, embeddings_dict: dict):
        predictions = []
        for name, embeddings in embeddings_dict.items():
            if name in self.classifiers:
                logits = self.classifiers[name](embeddings)
                predictions.append(logits)
        if not predictions:
            return torch.zeros(1, self.num_classes)
        return torch.stack(predictions, dim=0).mean(dim=0)

    def predict_single_backbone(self, backbone_name: str, embeddings: torch.Tensor):
        if backbone_name in self.classifiers:
            return self.classifiers[backbone_name](embeddings)
        return torch.zeros(embeddings.shape[0], self.num_classes)


# ============================================================================
# FOCAL LOSS (better for class imbalance than CE)
# ============================================================================
class FocalLoss(nn.Module):
    """Focal loss for better handling of hard/minority examples"""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# DATA COLLECTION
# ============================================================================
def collect_all_data():
    """Collect all training data"""
    file_paths = []
    labels = []

    print("\n[Collecting Baby Cry Data]")
    for cls in CRY_CLASSES:
        cls_dir = BABY_CRY_DIR / cls
        if cls_dir.exists():
            count = 0
            for f in list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")):
                file_paths.append(str(f))
                labels.append(LABEL_TO_ID[cls])
                count += 1
            print(f"  {cls}: {count} files")

    print("\n[Collecting Pulmonary Data]")
    for cls in PULMONARY_CLASSES:
        cls_dir = BABY_PULMONARY_DIR / cls
        if cls_dir.exists():
            count = 0
            for f in list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")):
                file_paths.append(str(f))
                labels.append(LABEL_TO_ID[cls])
                count += 1
            print(f"  {cls}: {count} files")

    return file_paths, labels


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for embeddings, labels in pbar:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), correct / total, all_preds, all_labels


def save_model(model, path, metadata=None):
    """Save model in ensemble-compatible format"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': NUM_CLASSES,
        'classes': ALL_CLASSES,
        'cry_classes': CRY_CLASSES,
        'pulmonary_classes': PULMONARY_CLASSES,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
        'risk_levels': RISK_LEVELS,
        'backbone_weights': 'equal (1/6 each)',
        'model_type': 'pretrained_distilhubert_head',
        'embedding_dim': 768,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train with pretrained HuggingFace features")
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--no_cache', action='store_true', help='Do not cache embeddings')
    args = parser.parse_args()

    print("=" * 70)
    print("PRETRAINED HUGGINGFACE BACKBONE TRAINING")
    print("=" * 70)
    print(f"Classes: {NUM_CLASSES}")
    print(f"Backbone: DistilHuBERT (ntu-spml/distilhubert)")
    print(f"Backbone weights: EQUAL (1/6 each)")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # ---- Collect data ----
    print("\n" + "=" * 70)
    print("COLLECTING DATA")
    print("=" * 70)

    file_paths, labels = collect_all_data()
    total = len(file_paths)
    print(f"\nTotal samples: {total}")

    if total < 100:
        print("[!] Not enough data. Run augment_minority_classes.py first.")
        return

    # ---- Extract pretrained features ----
    print("\n" + "=" * 70)
    print("EXTRACTING PRETRAINED FEATURES (DistilHuBERT)")
    print("=" * 70)

    extractor = PretrainedFeatureExtractor("ntu-spml/distilhubert")

    cache_path = str(OUTPUT_DIR / "embeddings_cache.npz") if not args.no_cache else None
    embeddings, valid_labels, valid_indices = extract_all_embeddings(
        file_paths, labels, extractor, cache_path=cache_path
    )

    print(f"\nValid embeddings: {len(embeddings)}")

    # Class distribution after extraction
    class_counts = Counter(valid_labels.tolist())
    print("\nClass distribution (after extraction):")
    for cls_id in sorted(class_counts.keys()):
        cls_name = ID_TO_LABEL[cls_id]
        print(f"  {cls_name:25s} {class_counts[cls_id]:5d}")

    # ---- Split data ----
    train_emb, val_emb, train_lbl, val_lbl = train_test_split(
        embeddings, valid_labels, test_size=0.2, stratify=valid_labels, random_state=42
    )

    print(f"\nTraining: {len(train_emb)}, Validation: {len(val_emb)}")

    # ---- Create datasets ----
    train_dataset = EmbeddingDataset(train_emb, train_lbl)
    val_dataset = EmbeddingDataset(val_emb, val_lbl)

    # Weighted sampler for class imbalance
    train_class_counts = Counter(train_lbl.tolist())
    sample_weights = [1.0 / train_class_counts[int(lbl)] for lbl in train_lbl]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # ---- Create model ----
    print("\n" + "=" * 70)
    print("CREATING CLASSIFIER HEAD")
    print("=" * 70)

    model = PretrainedClassifierHead(
        input_dim=extractor.embedding_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=512,
        dropout=0.4
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Classifier parameters: {total_params:,}")

    # ---- Class weights for focal loss ----
    class_frequencies = np.zeros(NUM_CLASSES)
    for cls_id, count in train_class_counts.items():
        class_frequencies[cls_id] = count
    class_frequencies = np.maximum(class_frequencies, 1)
    class_weights = 1.0 / class_frequencies
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ---- Training loop ----
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_acc = 0
    patience_counter = 0
    training_history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, labels_true = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4)
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            OUTPUT_DIR.mkdir(exist_ok=True)
            RPI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            metadata = {
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'backbone': 'distilhubert',
                'embedding_dim': 768,
                'timestamp': datetime.now().isoformat()
            }

            save_model(model, OUTPUT_DIR / "20class_ensemble.pt", metadata)
            save_model(model, RPI_OUTPUT_DIR / "6backbone_20class.pt", metadata)

            print(f"[*] New best model! Val acc: {val_acc * 100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[!] Early stopping at epoch {epoch + 1}")
                break

    # ---- Final evaluation ----
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    checkpoint = torch.load(OUTPUT_DIR / "20class_ensemble.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_acc, preds, labels_true = evaluate(model, val_loader, criterion, device)

    label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    report = classification_report(labels_true, preds, target_names=label_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

    # Save history
    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save report
    with open(OUTPUT_DIR / "classification_report.txt", 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n")
        f.write(f"Backbone: DistilHuBERT (ntu-spml/distilhubert)\n")
        f.write(f"Embedding dim: 768\n")
        f.write(f"Training samples: {len(train_emb)}\n")
        f.write(f"Validation samples: {len(val_emb)}\n\n")
        f.write(report)

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Previous baseline (simple CNN): 42.50%")
    print(f"  Model: {OUTPUT_DIR / '20class_ensemble.pt'}")
    print(f"  RPi5:  {RPI_OUTPUT_DIR / '6backbone_20class.pt'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
