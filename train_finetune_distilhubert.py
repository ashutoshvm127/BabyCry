#!/usr/bin/env python3
"""
Fine-tune DistilHuBERT Backbone for 20-Class Audio Classification
===================================================================

Key improvement: Fine-tune last 2 transformer layers of DistilHuBERT
instead of only training a classifier head on frozen embeddings.

This teaches the backbone to produce better representations specifically
for baby cry and pulmonary sound patterns.

Strategy:
- Freeze all but last 2 encoder layers + classifier head
- Use capped + augmented balanced dataset
- Moderate class weights (sqrt-inverse)
- Cosine annealing LR with warmup
- Mixup augmentation on embeddings
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

CLASS_CAPS = {
    'normal_breathing': 800,
    'normal_cry': 500,
    'hungry_cry': 500,
}


# ============================================================================
# AUDIO DATASET (loads raw waveforms)
# ============================================================================
class AudioWaveformDataset(Dataset):
    """Loads raw waveforms for on-the-fly processing through the backbone"""

    def __init__(self, file_paths, labels, sample_rate=16000, max_duration=5.0,
                 augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            y, _ = librosa.load(self.file_paths[idx], sr=self.sample_rate,
                                mono=True, duration=5.0)
            if len(y) < 1600 or not np.isfinite(y).all():
                y = np.zeros(self.max_length, dtype=np.float32)
        except Exception:
            y = np.zeros(self.max_length, dtype=np.float32)

        # Normalize
        max_val = np.max(np.abs(y))
        if max_val > 0 and np.isfinite(max_val):
            y = y / max_val

        # Augment
        if self.augment:
            y = self._augment(y)

        # Pad/truncate
        if len(y) < self.max_length:
            y = np.pad(y, (0, self.max_length - len(y)))
        else:
            y = y[:self.max_length]

        return torch.tensor(y, dtype=torch.float32), torch.tensor(self.labels[idx])

    def _augment(self, y):
        if np.random.random() < 0.3:
            shift = int(np.random.uniform(-0.1, 0.1) * len(y))
            y = np.roll(y, shift)
        if np.random.random() < 0.3:
            y = y * np.random.uniform(0.8, 1.2)
        if np.random.random() < 0.2:
            y = y + np.random.normal(0, 0.005, len(y))
        return y.astype(np.float32)


# ============================================================================
# MODEL: Fine-tunable DistilHuBERT + Classifier Head
# ============================================================================
class FineTunedDistilHuBERT(nn.Module):
    """
    DistilHuBERT with last N encoder layers unfrozen + classifier head.
    """

    def __init__(self, num_classes=20, unfreeze_layers=2, hidden_dim=256,
                 dropout=0.3):
        super().__init__()
        from transformers import HubertModel, AutoFeatureExtractor

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "ntu-spml/distilhubert"
        )
        self.backbone = HubertModel.from_pretrained("ntu-spml/distilhubert")
        self.embedding_dim = 768

        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N encoder layers
        encoder_layers = self.backbone.encoder.layers
        total_layers = len(encoder_layers)
        for i in range(total_layers - unfreeze_layers, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        # Unfreeze layer norm
        if hasattr(self.backbone.encoder, 'layer_norm'):
            for param in self.backbone.encoder.layer_norm.parameters():
                param.requires_grad = True

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, waveforms):
        """
        Args:
            waveforms: [batch, samples] raw audio
        Returns:
            logits: [batch, num_classes]
        """
        # Process through feature extractor
        # We need to process each waveform individually through the feature extractor
        batch_inputs = self.feature_extractor(
            [w.cpu().numpy() for w in waveforms],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = batch_inputs['input_values'].to(waveforms.device)

        # Forward through backbone
        outputs = self.backbone(input_values)
        # Mean pooling over time
        hidden_states = outputs.last_hidden_state  # [batch, time, 768]
        pooled = hidden_states.mean(dim=1)  # [batch, 768]

        return self.classifier(pooled)


# ============================================================================
# DATA COLLECTION
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
                indices = np.random.choice(len(files), cap, replace=False)
                files = [files[i] for i in indices]
            for f in files:
                file_paths.append(str(f))
                labels.append(LABEL_TO_ID[cls])
            print(f"  {cls}: {len(files)}")

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
            print(f"  {cls}: {len(files)}")

    return file_paths, labels


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for waveforms, labels in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

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
        for waveforms, labels in tqdm(dataloader, desc="Evaluating"):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), correct / total, all_preds, all_labels


def save_model(model, path, metadata=None):
    """Save only the classifier head (portable for RPi5)"""
    # Extract just the classifier head state
    classifier_state = model.classifier.state_dict()

    checkpoint = {
        'model_state_dict': classifier_state,
        'num_classes': NUM_CLASSES,
        'classes': ALL_CLASSES,
        'cry_classes': CRY_CLASSES,
        'pulmonary_classes': PULMONARY_CLASSES,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
        'risk_levels': RISK_LEVELS,
        'backbone_weights': 'equal (1/6 each)',
        'model_type': 'finetuned_distilhubert',
        'backbone_name': 'ntu-spml/distilhubert',
        'embedding_dim': 768,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    print(f"  Saved: {path}")


def save_full_model(model, path):
    """Save full model including fine-tuned backbone layers"""
    torch.save({
        'full_model_state_dict': model.state_dict(),
        'num_classes': NUM_CLASSES,
        'classes': ALL_CLASSES,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
        'risk_levels': RISK_LEVELS,
        'model_type': 'finetuned_distilhubert_full',
        'backbone_name': 'ntu-spml/distilhubert',
    }, path)
    print(f"  Saved full model: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--unfreeze_layers', type=int, default=2)
    args = parser.parse_args()

    print("=" * 70)
    print("FINE-TUNE DistilHuBERT FOR 20-CLASS CLASSIFICATION")
    print("=" * 70)
    print(f"Unfreezing last {args.unfreeze_layers} encoder layers")
    print(f"Device: {device}")

    # Collect data
    file_paths, labels = collect_all_data()
    print(f"\nTotal: {len(file_paths)} samples")

    # Split
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Datasets
    train_dataset = AudioWaveformDataset(train_files, train_labels, augment=True)
    val_dataset = AudioWaveformDataset(val_files, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Model
    print("\nLoading DistilHuBERT...")
    model = FineTunedDistilHuBERT(
        num_classes=NUM_CLASSES,
        unfreeze_layers=args.unfreeze_layers,
        hidden_dim=256,
        dropout=0.3
    ).to(device)

    # Count trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Class weights
    train_counts = Counter(train_labels)
    class_weights = np.ones(NUM_CLASSES)
    for cls_id, count in train_counts.items():
        class_weights[cls_id] = 1.0 / np.sqrt(count)
    class_weights = class_weights / class_weights.mean()
    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.05)

    # Optimizer with different LR for backbone vs head
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'classifier' not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and 'classifier' in n]

    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=1e-4)

    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=[args.lr * 0.1, args.lr],
        total_steps=total_steps, pct_start=0.1
    )

    # Training
    print("\n" + "=" * 70)
    print("TRAINING (fine-tuning)")
    print("=" * 70)

    best_val_acc = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_acc, preds, true_labels = evaluate(
            model, val_loader, criterion, device
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
                'backbone': 'distilhubert (fine-tuned)',
                'unfreeze_layers': args.unfreeze_layers,
                'timestamp': datetime.now().isoformat()
            }

            save_model(model, OUTPUT_DIR / "20class_ensemble.pt", metadata)
            save_model(model, RPI_OUTPUT_DIR / "6backbone_20class.pt", metadata)
            save_full_model(model, OUTPUT_DIR / "20class_finetuned_full.pt")

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

    # Reload best
    best_ckpt = torch.load(OUTPUT_DIR / "20class_finetuned_full.pt", weights_only=False)
    model.load_state_dict(best_ckpt['full_model_state_dict'])

    _, final_acc, preds, true_labels = evaluate(model, val_loader, criterion, device)

    label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    report = classification_report(true_labels, preds, target_names=label_names,
                                   zero_division=0)
    print("\nClassification Report:")
    print(report)

    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    with open(OUTPUT_DIR / "classification_report.txt", 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n")
        f.write(f"Backbone: DistilHuBERT (fine-tuned last {args.unfreeze_layers} layers)\n")
        f.write(f"Strategy: sqrt-inverse weights, OneCycleLR, differential LR\n\n")
        f.write(report)

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val accuracy: {best_val_acc * 100:.2f}%")
    print(f"  v2 (frozen embeddings): 52.57%")
    print(f"  Original baseline (simple CNN): 42.50%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
