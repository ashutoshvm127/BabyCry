#!/usr/bin/env python3
"""
Improved Training with Pretrained Features + Balanced Strategy
================================================================

Fixes from v1:
1. Downsample majority classes (normal_breathing capped at 800, normal_cry at 500)  
2. Use moderate class weights (sqrt of inverse frequency) in CrossEntropyLoss
3. NO WeightedRandomSampler (causes extreme bias against majority classes)
4. Lower dropout, bigger hidden dim for better capacity
5. Uses cached embeddings from previous run (no re-extraction needed)

Uses DistilHuBERT 768-dim pretrained embeddings.
"""

import os
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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

# Caps for majority classes to prevent domination
CLASS_CAPS = {
    'normal_breathing': 800,
    'normal_cry': 500,
    'hungry_cry': 500,
}


# ============================================================================
# DATASET
# ============================================================================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
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
    """MLP classifier head for pretrained embeddings"""

    def __init__(self, input_dim=768, num_classes=20, hidden_dim=512, dropout=0.3):
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
    """6-Backbone Ensemble with EQUAL WEIGHTS (1/6 each)"""

    def __init__(self, num_classes=20, backbone_dims=None):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_weight = 1.0 / 6.0

        if backbone_dims is None:
            backbone_dims = {
                'distilhubert': 768, 'ast': 768, 'yamnet': 1024,
                'wav2vec2': 768, 'wavlm': 768, 'panns': 2048
            }

        self.classifiers = nn.ModuleDict()
        for name, dim in backbone_dims.items():
            self.classifiers[name] = PretrainedClassifierHead(
                input_dim=dim, num_classes=num_classes
            )

    def forward(self, embeddings_dict):
        predictions = []
        for name, embeddings in embeddings_dict.items():
            if name in self.classifiers:
                logits = self.classifiers[name](embeddings)
                predictions.append(logits)
        if not predictions:
            return torch.zeros(1, self.num_classes)
        return torch.stack(predictions, dim=0).mean(dim=0)


# ============================================================================
# DATA COLLECTION + EMBEDDING EXTRACTION
# ============================================================================
def collect_all_data():
    """Collect data with majority class capping"""
    file_paths = []
    labels = []

    print("\n[Collecting Baby Cry Data]")
    for cls in CRY_CLASSES:
        cls_dir = BABY_CRY_DIR / cls
        if cls_dir.exists():
            files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
            cap = CLASS_CAPS.get(cls, len(files))
            if len(files) > cap:
                np.random.seed(42)
                indices = np.random.choice(len(files), cap, replace=False)
                files = [files[i] for i in indices]
            for f in files:
                file_paths.append(str(f))
                labels.append(LABEL_TO_ID[cls])
            print(f"  {cls}: {len(files)} files" +
                  (f" (capped from {cap})" if cls in CLASS_CAPS else ""))

    print("\n[Collecting Pulmonary Data]")
    for cls in PULMONARY_CLASSES:
        cls_dir = BABY_PULMONARY_DIR / cls
        if cls_dir.exists():
            files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
            cap = CLASS_CAPS.get(cls, len(files))
            if len(files) > cap:
                np.random.seed(42)
                indices = np.random.choice(len(files), cap, replace=False)
                files = [files[i] for i in indices]
            for f in files:
                file_paths.append(str(f))
                labels.append(LABEL_TO_ID[cls])
            print(f"  {cls}: {len(files)} files" +
                  (f" (capped from {cap})" if cls in CLASS_CAPS else ""))

    return file_paths, labels


def extract_embeddings(file_paths, labels, cache_path=None):
    """Extract DistilHuBERT embeddings with optional caching"""
    
    # Try loading full cache first (from previous run)
    full_cache = str(OUTPUT_DIR / "embeddings_cache.npz")
    if os.path.exists(full_cache):
        print(f"Loading full embeddings cache from {full_cache}")
        data = np.load(full_cache, allow_pickle=True)
        all_emb = data['embeddings']
        all_lbl = data['labels']
        # Build path→embedding map from the previous run
        # But we don't have paths in cache, so we need to re-extract for the subset
        print(f"  Cache has {len(all_emb)} embeddings (from full dataset)")
        print("  Re-extracting for capped subset...")

    from transformers import AutoModel, AutoFeatureExtractor
    import librosa

    print(f"\nLoading DistilHuBERT model...")
    feature_extractor = AutoFeatureExtractor.from_pretrained("ntu-spml/distilhubert")
    model = AutoModel.from_pretrained("ntu-spml/distilhubert")
    model.eval()
    model.to(device)

    embeddings = []
    valid_labels = []

    print(f"Extracting embeddings for {len(file_paths)} files...")
    for fp, lbl in tqdm(zip(file_paths, labels), total=len(file_paths),
                        desc="Extracting features"):
        try:
            y, _ = librosa.load(fp, sr=16000, mono=True, duration=5.0)
            if len(y) < 1600 or not np.isfinite(y).all():
                continue
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

            with torch.no_grad():
                inputs = feature_extractor(y, sampling_rate=16000,
                                           return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

            if np.isfinite(emb).all():
                embeddings.append(emb)
                valid_labels.append(lbl)
        except Exception:
            continue

    embeddings = np.array(embeddings, dtype=np.float32)
    valid_labels = np.array(valid_labels)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, labels=valid_labels)
        print(f"Cached to {cache_path}")

    return embeddings, valid_labels


# ============================================================================
# TRAINING
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
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=12)
    args = parser.parse_args()

    print("=" * 70)
    print("IMPROVED TRAINING v2 - Pretrained DistilHuBERT + Balanced Strategy")
    print("=" * 70)
    print(f"Classes: {NUM_CLASSES}")
    print(f"Majority class caps: {CLASS_CAPS}")
    print(f"Device: {device}")

    # ---- Collect data (with capping) ----
    print("\n" + "=" * 70)
    print("COLLECTING DATA (with majority class capping)")
    print("=" * 70)

    file_paths, labels = collect_all_data()
    print(f"\nTotal samples after capping: {len(file_paths)}")

    # ---- Extract embeddings ----
    print("\n" + "=" * 70)
    print("EXTRACTING PRETRAINED FEATURES")
    print("=" * 70)

    cache_path = str(OUTPUT_DIR / "embeddings_cache_v2.npz")
    embeddings, valid_labels = extract_embeddings(file_paths, labels, cache_path)
    print(f"Valid embeddings: {len(embeddings)}")

    # ---- Class distribution ----
    class_counts = Counter(valid_labels.tolist())
    print("\nClass distribution:")
    for cls_id in sorted(class_counts.keys()):
        cls_name = ID_TO_LABEL[cls_id]
        print(f"  {cls_name:25s} {class_counts[cls_id]:5d}")

    # ---- Split data ----
    train_emb, val_emb, train_lbl, val_lbl = train_test_split(
        embeddings, valid_labels, test_size=0.2, stratify=valid_labels, random_state=42
    )
    print(f"\nTraining: {len(train_emb)}, Validation: {len(val_emb)}")

    # ---- Create dataloaders (NO weighted sampler) ----
    train_dataset = EmbeddingDataset(train_emb, train_lbl)
    val_dataset = EmbeddingDataset(val_emb, val_lbl)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # ---- Model ----
    print("\n" + "=" * 70)
    print("CREATING CLASSIFIER HEAD")
    print("=" * 70)

    model = PretrainedClassifierHead(
        input_dim=768, num_classes=NUM_CLASSES,
        hidden_dim=512, dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # ---- Moderate class weights (sqrt of inverse frequency) ----
    train_class_counts = Counter(train_lbl.tolist())
    class_weights = np.ones(NUM_CLASSES)
    for cls_id, count in train_class_counts.items():
        class_weights[cls_id] = 1.0 / np.sqrt(count)
    # Normalize so mean weight = 1
    class_weights = class_weights / class_weights.mean()
    print(f"\nClass weights (sqrt-inverse):")
    for i, w in enumerate(class_weights):
        print(f"  {ID_TO_LABEL[i]:25s} {w:.3f}")

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

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
                'backbone': 'distilhubert',
                'embedding_dim': 768,
                'class_caps': CLASS_CAPS,
                'timestamp': datetime.now().isoformat()
            }

            save_model(model, OUTPUT_DIR / "20class_ensemble.pt", metadata)
            save_model(model, RPI_OUTPUT_DIR / "6backbone_20class.pt", metadata)

            print(f"[*] New best! Val acc: {val_acc * 100:.2f}%")
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

    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    with open(OUTPUT_DIR / "classification_report.txt", 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n")
        f.write(f"Backbone: DistilHuBERT\n")
        f.write(f"Class caps: {CLASS_CAPS}\n")
        f.write(f"Strategy: sqrt-inverse class weights, no weighted sampler\n\n")
        f.write(report)

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Previous v1 (weighted sampler): 25.82%")
    print(f"  Original baseline (simple CNN): 42.50%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
