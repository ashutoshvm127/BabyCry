#!/usr/bin/env python3
"""
Training v3: Multi-Feature Fusion + Mixup Augmentation
========================================================

Combines DistilHuBERT embeddings (768-dim) with hand-crafted audio features
(MFCCs + spectral features, ~85-dim) for a richer 853-dim representation.

Improvements over v2 (52.57%):
1. Multi-feature fusion: pretrained + hand-crafted features 
2. Mixup augmentation on embeddings for regularization
3. Deeper classifier with residual connection
4. OneCycleLR schedule for better convergence
5. Uses cached DistilHuBERT embeddings (no re-extraction needed)
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
from torch.optim.lr_scheduler import OneCycleLR
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
BABY_CRY_DIR = BASE_DIR / "data_baby_respiratory"
BABY_PULMONARY_DIR = BASE_DIR / "data_baby_pulmonary"
OUTPUT_DIR = BASE_DIR / "trained_classifiers"
RPI_OUTPUT_DIR = BASE_DIR / "rpi5_standalone" / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

CLASS_CAPS = {
    'normal_breathing': 800,
    'normal_cry': 500,
    'hungry_cry': 500,
}


# ============================================================================
# HAND-CRAFTED FEATURE EXTRACTION
# ============================================================================
def extract_handcrafted_features(y, sr=16000):
    """Extract MFCC + spectral features from audio waveform"""
    try:
        # MFCCs (40 mean + 40 std = 80 features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Spectral features (5 features)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, rms]
        ])

        if not np.isfinite(features).all():
            return np.zeros(85, dtype=np.float32)
        return features.astype(np.float32)
    except Exception:
        return np.zeros(85, dtype=np.float32)


# ============================================================================
# DATA COLLECTION + FEATURE EXTRACTION
# ============================================================================
def collect_all_data():
    file_paths = []
    labels = []

    for cls in CRY_CLASSES:
        cls_dir = BABY_CRY_DIR / cls
        if cls_dir.exists():
            files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
            cap = CLASS_CAPS.get(cls, len(files))
            if len(files) > cap:
                np.random.seed(42)
                idx = np.random.choice(len(files), cap, replace=False)
                files = [files[i] for i in idx]
            for f in files:
                file_paths.append(str(f))
                labels.append(LABEL_TO_ID[cls])

    for cls in PULMONARY_CLASSES:
        cls_dir = BABY_PULMONARY_DIR / cls
        if cls_dir.exists():
            files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
            cap = CLASS_CAPS.get(cls, len(files))
            if len(files) > cap:
                np.random.seed(42)
                idx = np.random.choice(len(files), cap, replace=False)
                files = [files[i] for i in idx]
            for f in files:
                file_paths.append(str(f))
                labels.append(LABEL_TO_ID[cls])

    return file_paths, labels


def extract_all_features(file_paths, labels):
    """Extract both DistilHuBERT embeddings and hand-crafted features"""
    from transformers import AutoModel, AutoFeatureExtractor

    # Load DistilHuBERT
    print("Loading DistilHuBERT...")
    fe = AutoFeatureExtractor.from_pretrained("ntu-spml/distilhubert")
    backbone = AutoModel.from_pretrained("ntu-spml/distilhubert")
    backbone.eval()
    backbone.to(device)

    hubert_embeddings = []
    handcrafted_features = []
    valid_labels = []

    print(f"Extracting features for {len(file_paths)} files...")
    for fp, lbl in tqdm(zip(file_paths, labels), total=len(file_paths),
                        desc="Feature extraction"):
        try:
            y, _ = librosa.load(fp, sr=16000, mono=True, duration=5.0)
            if len(y) < 1600 or not np.isfinite(y).all():
                continue
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

            # DistilHuBERT embedding
            with torch.no_grad():
                inputs = fe(y, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = backbone(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

            if not np.isfinite(emb).all():
                continue

            # Hand-crafted features
            hf = extract_handcrafted_features(y)

            hubert_embeddings.append(emb)
            handcrafted_features.append(hf)
            valid_labels.append(lbl)

        except Exception:
            continue

    hubert_embeddings = np.array(hubert_embeddings, dtype=np.float32)
    handcrafted_features = np.array(handcrafted_features, dtype=np.float32)
    valid_labels = np.array(valid_labels)

    return hubert_embeddings, handcrafted_features, valid_labels


# ============================================================================
# DATASET WITH MIXUP
# ============================================================================
class FusionDataset(Dataset):
    def __init__(self, hubert_emb, handcrafted_feat, labels, mixup_alpha=0.0):
        self.hubert = torch.tensor(hubert_emb, dtype=torch.float32)
        self.handcrafted = torch.tensor(handcrafted_feat, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        h = self.hubert[idx]
        hc = self.handcrafted[idx]
        label = self.labels[idx]

        if self.mixup_alpha > 0 and self.training_mode:
            # Mixup: blend with a random sample
            mix_idx = np.random.randint(len(self))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            h = lam * h + (1 - lam) * self.hubert[mix_idx]
            hc = lam * hc + (1 - lam) * self.handcrafted[mix_idx]
            # Return both labels for mixup loss
            return h, hc, label, self.labels[mix_idx], torch.tensor(lam)

        return h, hc, label, label, torch.tensor(1.0)

    @property
    def training_mode(self):
        return self._training_mode if hasattr(self, '_training_mode') else False

    @training_mode.setter
    def training_mode(self, val):
        self._training_mode = val


# ============================================================================
# MODEL: Fusion Classifier with Residual
# ============================================================================
class FusionClassifier(nn.Module):
    """
    Multi-feature fusion classifier.
    Combines DistilHuBERT (768) + hand-crafted (85) = 853 input dims.
    """

    def __init__(self, hubert_dim=768, handcrafted_dim=85, num_classes=20,
                 hidden_dim=512, dropout=0.3):
        super().__init__()
        total_dim = hubert_dim + handcrafted_dim

        # Normalization for each feature type
        self.hubert_norm = nn.LayerNorm(hubert_dim)
        self.handcrafted_norm = nn.BatchNorm1d(handcrafted_dim)

        # Main classifier with residual
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.dropout_light = nn.Dropout(dropout * 0.5)

    def forward(self, hubert_emb, handcrafted_feat):
        # Normalize each input type
        h = self.hubert_norm(hubert_emb)
        hc = self.handcrafted_norm(handcrafted_feat)

        # Concatenate
        x = torch.cat([h, hc], dim=-1)

        # Block 1
        out = F.gelu(self.bn1(self.fc1(x)))
        out = self.dropout(out)

        # Block 2 with residual
        residual = out
        out = F.gelu(self.bn2(self.fc2(out)))
        out = self.dropout_light(out)
        out = out + residual  # Residual connection

        # Block 3
        out = F.gelu(self.bn3(self.fc3(out)))
        out = self.dropout_light(out)

        return self.fc_out(out)


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                use_mixup=False):
    model.train()
    dataloader.dataset.training_mode = True
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for h_emb, hc_feat, labels_a, labels_b, lam in pbar:
        h_emb = h_emb.to(device)
        hc_feat = hc_feat.to(device)
        labels_a = labels_a.to(device)
        labels_b = labels_b.to(device)
        lam = lam.to(device)

        optimizer.zero_grad()
        outputs = model(h_emb, hc_feat)

        if use_mixup:
            loss = (lam * criterion(outputs, labels_a) +
                    (1 - lam) * criterion(outputs, labels_b)).mean()
        else:
            loss = criterion(outputs, labels_a)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels_a).sum().item()
        total += labels_a.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    dataloader.dataset.training_mode = False
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for h_emb, hc_feat, labels, _, _ in tqdm(dataloader, desc="Evaluating"):
            h_emb = h_emb.to(device)
            hc_feat = hc_feat.to(device)
            labels = labels.to(device)

            outputs = model(h_emb, hc_feat)
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
        'model_type': 'fusion_distilhubert_handcrafted',
        'hubert_dim': 768,
        'handcrafted_dim': 85,
        'backbone_name': 'ntu-spml/distilhubert',
        'embedding_dim': 768,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--no_cache', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING v3: MULTI-FEATURE FUSION + MIXUP")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Mixup alpha: {args.mixup_alpha}")

    # Collect data
    file_paths, labels = collect_all_data()
    print(f"Total samples: {len(file_paths)}")

    # Extract features (or load from cache)
    cache_path = str(OUTPUT_DIR / "fusion_features_cache.npz")
    if not args.no_cache and os.path.exists(cache_path):
        print(f"\nLoading cached features from {cache_path}")
        data = np.load(cache_path)
        hubert_emb = data['hubert']
        handcrafted = data['handcrafted']
        valid_labels = data['labels']
    else:
        hubert_emb, handcrafted, valid_labels = extract_all_features(
            file_paths, labels
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez(cache_path, hubert=hubert_emb, handcrafted=handcrafted,
                 labels=valid_labels)
        print(f"Cached to {cache_path}")

    print(f"Features: {len(hubert_emb)} samples")
    print(f"  DistilHuBERT dim: {hubert_emb.shape[1]}")
    print(f"  Hand-crafted dim: {handcrafted.shape[1]}")
    print(f"  Total input dim:  {hubert_emb.shape[1] + handcrafted.shape[1]}")

    # Normalize hand-crafted features globally
    hc_mean = handcrafted.mean(axis=0)
    hc_std = handcrafted.std(axis=0) + 1e-8
    handcrafted = (handcrafted - hc_mean) / hc_std

    # Class distribution
    class_counts = Counter(valid_labels.tolist())
    print("\nClass distribution:")
    for cid in sorted(class_counts.keys()):
        print(f"  {ID_TO_LABEL[cid]:25s} {class_counts[cid]:5d}")

    # Split
    indices = np.arange(len(valid_labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=valid_labels, random_state=42
    )

    train_h = hubert_emb[train_idx]
    val_h = hubert_emb[val_idx]
    train_hc = handcrafted[train_idx]
    val_hc = handcrafted[val_idx]
    train_lbl = valid_labels[train_idx]
    val_lbl = valid_labels[val_idx]

    print(f"\nTrain: {len(train_idx)}, Val: {len(val_idx)}")

    # Datasets
    train_dataset = FusionDataset(train_h, train_hc, train_lbl,
                                  mixup_alpha=args.mixup_alpha)
    val_dataset = FusionDataset(val_h, val_hc, val_lbl, mixup_alpha=0.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = FusionClassifier(
        hubert_dim=768, handcrafted_dim=85,
        num_classes=NUM_CLASSES, hidden_dim=512, dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Moderate class weights
    train_counts = Counter(train_lbl.tolist())
    cw = np.ones(NUM_CLASSES)
    for cid, cnt in train_counts.items():
        cw[cid] = 1.0 / np.sqrt(cnt)
    cw = cw / cw.mean()
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.05,
                                    reduction='none' if args.mixup_alpha > 0 else 'mean')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                           pct_start=0.1, anneal_strategy='cos')

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_acc = 0
    patience_counter = 0
    history = []

    # Need non-mixup criterion for eval
    eval_criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.05)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            use_mixup=(args.mixup_alpha > 0)
        )
        val_loss, val_acc, preds, true_labels = evaluate(
            model, val_loader, eval_criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        history.append({
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
                'backbone': 'distilhubert + handcrafted',
                'mixup_alpha': args.mixup_alpha,
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

    # Final eval
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    ckpt = torch.load(OUTPUT_DIR / "20class_ensemble.pt", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    _, final_acc, preds, true_labels = evaluate(
        model, val_loader, eval_criterion, device
    )

    label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    report = classification_report(true_labels, preds, target_names=label_names,
                                   zero_division=0)
    print("\nClassification Report:")
    print(report)

    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    with open(OUTPUT_DIR / "classification_report.txt", 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n")
        f.write(f"Backbone: DistilHuBERT + hand-crafted features\n")
        f.write(f"Mixup alpha: {args.mixup_alpha}\n")
        f.write(f"Strategy: fusion classifier, sqrt-inverse weights, OneCycleLR\n")
        f.write(f"Feature dims: 768 (HuBERT) + 85 (MFCC+spectral) = 853\n\n")
        f.write(report)

    # Save normalization params for inference
    np.savez(str(OUTPUT_DIR / "feature_normalization.npz"),
             hc_mean=hc_mean, hc_std=hc_std)

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val accuracy: {best_val_acc * 100:.2f}%")
    print(f"  v2 (frozen HuBERT only): 52.57%")
    print(f"  Original baseline (CNN): 42.50%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
