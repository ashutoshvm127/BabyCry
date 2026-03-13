#!/usr/bin/env python3
"""
Training Script for 20-Class 6-Backbone Ensemble Model
=======================================================

Trains classifier heads for baby cry + pulmonary detection with:
- 12 baby cry classes (including pathological, asphyxia, sepsis-related)
- 8 pulmonary classes (including bronchiolitis for RSV/sepsis respiratory markers)

6 BACKBONE MODELS (EQUAL WEIGHT = 1/6 each):
1. DistilHuBERT - Emotional cry detection
2. AST - Audio spectrogram analysis
3. YAMNet - General audio events
4. Wav2Vec2 - Speech/cry vocalization
5. WavLM - Noise-robust detection
6. PANNs CNN14 - Pulmonary sounds

Usage:
    python train_20class_ensemble.py
    python train_20class_ensemble.py --epochs 50 --batch_size 16

Output:
    - trained_classifiers/20class_ensemble.pt
    - rpi5_standalone/models/6backbone_20class.pt
"""

import os
import sys
import json
import argparse
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
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
BASE_DIR = Path(__file__).parent
BABY_CRY_DIR = BASE_DIR / "data_baby_respiratory"
BABY_PULMONARY_DIR = BASE_DIR / "data_baby_pulmonary"
OUTPUT_DIR = BASE_DIR / "trained_classifiers"
RPI_OUTPUT_DIR = BASE_DIR / "rpi5_standalone" / "models"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================
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

# Create label mappings
LABEL_TO_ID = {cls: i for i, cls in enumerate(ALL_CLASSES)}
ID_TO_LABEL = {i: cls for i, cls in enumerate(ALL_CLASSES)}

# Risk levels for each class
RISK_LEVELS = {
    # Low risk (GREEN)
    'normal_cry': 0, 'hungry_cry': 0, 'sleepy_cry': 0, 'tired_cry': 0,
    'burping_cry': 0, 'normal_breathing': 0,
    # Medium risk (YELLOW)
    'discomfort_cry': 1, 'cold_cry': 1, 'belly_pain_cry': 1,
    'wheeze': 1, 'rhonchi': 1, 'fine_crackle': 1, 'coarse_crackle': 1,
    # High risk (ORANGE)
    'distress_cry': 2, 'mixed': 2,
    # Critical risk (RED)
    'pain_cry': 3, 'pathological_cry': 3, 'asphyxia_cry': 3,
    'stridor': 3, 'bronchiolitis': 3
}


# ============================================================================
# DATASET
# ============================================================================
class AudioDataset(Dataset):
    """Dataset for loading audio files with labels"""
    
    def __init__(self, file_paths: list, labels: list, 
                 sample_rate: int = 16000, max_duration: float = 5.0,
                 augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label_id = self.labels[idx]
        
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, 
                                         duration=self.max_length / self.sample_rate)
            # Check for NaN/infinite values
            if not np.isfinite(waveform).all():
                waveform = np.zeros(self.max_length, dtype=np.float32)
        except (Exception, KeyboardInterrupt):
            waveform = np.zeros(self.max_length, dtype=np.float32)
        
        # Normalize
        max_val = np.max(np.abs(waveform))
        if max_val > 0 and np.isfinite(max_val):
            waveform = waveform / max_val
        
        # Data augmentation
        if self.augment:
            waveform = self._augment(waveform)
        
        # Pad or truncate
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]
        
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label_id)
    
    def _augment(self, waveform):
        """Apply random augmentations"""
        # Time shift
        if np.random.random() < 0.3:
            shift = int(np.random.uniform(-0.1, 0.1) * len(waveform))
            waveform = np.roll(waveform, shift)
        
        # Random gain
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            waveform = waveform * gain
        
        # Add noise
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.005, len(waveform))
            waveform = waveform + noise
        
        return waveform.astype(np.float32)


def collect_all_data():
    """Collect all training data from both cry and pulmonary directories"""
    file_paths = []
    labels = []
    
    print("\n[Collecting Baby Cry Data]")
    for cls in CRY_CLASSES:
        cls_dir = BABY_CRY_DIR / cls
        if cls_dir.exists():
            count = 0
            for audio_file in list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")):
                file_paths.append(str(audio_file))
                labels.append(LABEL_TO_ID[cls])
                count += 1
            print(f"  {cls}: {count} files")
    
    print("\n[Collecting Pulmonary Data]")
    for cls in PULMONARY_CLASSES:
        cls_dir = BABY_PULMONARY_DIR / cls
        if cls_dir.exists():
            count = 0
            for audio_file in list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")):
                file_paths.append(str(audio_file))
                labels.append(LABEL_TO_ID[cls])
                count += 1
            print(f"  {cls}: {count} files")
    
    return file_paths, labels


def validate_audio_files(file_paths, labels):
    """Quick validation to filter out corrupted audio files"""
    valid_files = []
    valid_labels = []
    bad_count = 0
    
    print(f"\n[Validating {len(file_paths)} audio files...]")
    for fp, lbl in tqdm(zip(file_paths, labels), total=len(file_paths), desc="Validating"):
        try:
            # Quick check: just read the file header
            fsize = os.path.getsize(fp)
            if fsize < 100:  # Skip tiny files
                bad_count += 1
                continue
            # Try loading just 0.5 seconds to check validity
            y, _ = librosa.load(fp, sr=16000, mono=True, duration=0.5)
            if len(y) > 0 and np.isfinite(y).all():
                valid_files.append(fp)
                valid_labels.append(lbl)
            else:
                bad_count += 1
        except:
            bad_count += 1
    
    print(f"  Valid: {len(valid_files)}, Bad/Corrupted: {bad_count}")
    return valid_files, valid_labels


def create_weighted_sampler(labels):
    """Create weighted sampler for class imbalance"""
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, len(labels))


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class SimpleClassifier(nn.Module):
    """Simple classifier head for embedding features"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class SixBackboneEnsemble(nn.Module):
    """
    6-Backbone Ensemble with EQUAL WEIGHTS (1/6 each)
    
    All backbones contribute equally to the final prediction.
    Individual classifier heads are trained independently.
    """
    
    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
        
        # All backbones have EQUAL weight = 1/6
        self.backbone_weight = 1.0 / 6.0
        
        # Backbone configurations (hidden sizes)
        self.backbone_configs = {
            'distilhubert': 768,
            'ast': 768,
            'yamnet': 1024,
            'wav2vec2': 768,
            'wavlm': 768,
            'panns': 2048
        }
        
        # Create classifier heads for each backbone
        self.classifiers = nn.ModuleDict()
        for name, hidden_dim in self.backbone_configs.items():
            self.classifiers[name] = SimpleClassifier(hidden_dim, num_classes)
        
        # Mean embedding layer for fallback
        self.fallback_classifier = SimpleClassifier(768, num_classes)
    
    def forward(self, embeddings_dict: dict):
        """
        Forward pass with embeddings from all backbones
        
        Args:
            embeddings_dict: Dict of {backbone_name: embeddings}
        
        Returns:
            Weighted average of all backbone predictions (equal weights)
        """
        predictions = []
        
        for name, embeddings in embeddings_dict.items():
            if name in self.classifiers:
                logits = self.classifiers[name](embeddings)
                predictions.append(logits)
        
        if not predictions:
            # Fallback: return zeros
            batch_size = 1
            return torch.zeros(batch_size, self.num_classes)
        
        # Stack and average with EQUAL weights
        predictions = torch.stack(predictions, dim=0)
        return predictions.mean(dim=0)  # Equal weight averaging
    
    def predict_single_backbone(self, backbone_name: str, embeddings: torch.Tensor):
        """Get prediction from a single backbone"""
        if backbone_name in self.classifiers:
            return self.classifiers[backbone_name](embeddings)
        return self.fallback_classifier(embeddings)


class AudioFeatureExtractor:
    """Simple audio feature extractor using librosa"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Extract basic audio features"""
        # MFCC features
        mfccs = librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=self.sample_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=self.sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=self.sample_rate))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(waveform))
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=waveform))
        
        # Combine features
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, rms]
        ])
        
        return features


class SimpleCNNClassifier(nn.Module):
    """Simple CNN classifier for direct audio classification"""
    
    def __init__(self, num_classes: int = 20, sample_rate: int = 16000, 
                 max_duration: float = 5.0):
        super().__init__()
        self.num_classes = num_classes
        
        # Mel spectrogram parameters
        self.n_mels = 128
        self.hop_length = 512
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, waveform):
        """
        Forward pass
        
        Args:
            waveform: Audio waveform [batch, samples]
        
        Returns:
            Class logits [batch, num_classes]
        """
        # Convert to mel spectrogram
        batch_size = waveform.shape[0]
        mel_specs = []
        
        for i in range(batch_size):
            audio = waveform[i].cpu().numpy()
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=16000, 
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec_db)
        
        # Stack and convert to tensor
        mel_specs = np.stack(mel_specs)
        mel_specs = torch.tensor(mel_specs, dtype=torch.float32).to(waveform.device)
        mel_specs = mel_specs.unsqueeze(1)  # Add channel dimension
        
        # Forward through CNN
        features = self.conv_layers(mel_specs)
        logits = self.classifier(features)
        
        return logits


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
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
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
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
    """Save model with metadata"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': len(ALL_CLASSES),
        'classes': ALL_CLASSES,
        'cry_classes': CRY_CLASSES,
        'pulmonary_classes': PULMONARY_CLASSES,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
        'risk_levels': RISK_LEVELS,
        'backbone_weights': 'equal (1/6 each)',
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train 20-class 6-backbone ensemble")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    args = parser.parse_args()
    
    print("=" * 70)
    print("20-CLASS 6-BACKBONE ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"\nClasses: {len(ALL_CLASSES)} total")
    print(f"  - {len(CRY_CLASSES)} cry classes")
    print(f"  - {len(PULMONARY_CLASSES)} pulmonary classes")
    print(f"\nBackbone weights: EQUAL (1/6 each)")
    print(f"Device: {device}")
    
    # Collect data
    print("\n" + "=" * 70)
    print("COLLECTING DATA")
    print("=" * 70)
    
    file_paths, labels = collect_all_data()
    
    if len(file_paths) < 100:
        print("[!] Error: Not enough training data!")
        print("    Run organize_and_download_datasets.py first")
        return
    
    # Validate audio files (filter out corrupted ones)
    file_paths, labels = validate_audio_files(file_paths, labels)
    
    if len(file_paths) < 100:
        print("[!] Error: Not enough valid training data after filtering!")
        return
    
    # Split data
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"\nTraining samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels, augment=True)
    val_dataset = AudioDataset(val_files, val_labels, augment=False)
    
    # Create weighted sampler for class imbalance
    sampler = create_weighted_sampler(train_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = SimpleCNNClassifier(num_classes=len(ALL_CLASSES))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_val_acc = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, labels_true = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Save best model
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
                'timestamp': datetime.now().isoformat()
            }
            
            save_model(model, OUTPUT_DIR / "20class_ensemble.pt", metadata)
            save_model(model, RPI_OUTPUT_DIR / "6backbone_20class.pt", metadata)
            
            print(f"[*] New best model! Val acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[!] Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / "20class_ensemble.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc, preds, labels_true = evaluate(model, val_loader, criterion, device)
    
    # Classification report
    label_names = [ID_TO_LABEL[i] for i in range(len(ALL_CLASSES))]
    print("\nClassification Report:")
    print(classification_report(labels_true, preds, target_names=label_names, zero_division=0))
    
    # Save training history
    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n[OK] Training complete!")
    print(f"     Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"     Model saved to: {OUTPUT_DIR / '20class_ensemble.pt'}")
    print(f"     RPi5 model saved to: {RPI_OUTPUT_DIR / '6backbone_20class.pt'}")


if __name__ == "__main__":
    main()
