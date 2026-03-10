#!/usr/bin/env python3
"""
Balanced Training Script for 6-Backbone Ensemble Model

Fixes class imbalance issues by:
1. Computing inverse frequency class weights
2. Using WeightedRandomSampler for balanced batches
3. Using Focal Loss (better for imbalanced data)
4. Oversampling minority classes
5. Data augmentation for minority samples

Usage:
    python train_balanced_ensemble.py --task cry
    python train_balanced_ensemble.py --task pulmonary
    python train_balanced_ensemble.py --task both --epochs 100

This ensures "normal" cries don't dominate the training!
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from collections import Counter
import random

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

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "baby_cry_diagnostic"))

from baby_cry_diagnostic.backend.models.ensemble import EnsembleModel


# ==============================================================================
# Focal Loss - Better than CrossEntropy for Imbalanced Data
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights well-classified examples, focuses on hard examples.
    From paper: "Focal Loss for Dense Object Detection"
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class balancing weight (can be per-class or scalar)
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss - Prevents overconfidence.
    Softens labels: [0, 1, 0] -> [0.05, 0.9, 0.05]
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        
        one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), confidence)
        one_hot += smooth_value
        one_hot[torch.arange(len(targets)), targets] = confidence
        
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(one_hot * log_probs).sum(dim=1).mean()
        
        return loss


# ==============================================================================
# Audio Dataset with Augmentation
# ==============================================================================

class BalancedAudioDataset(Dataset):
    """
    Dataset with audio augmentation for minority classes.
    
    Augmentations:
    - Time stretch (±10%)
    - Pitch shift (±2 semitones)
    - Add noise
    - Time mask
    """
    
    def __init__(self, file_paths: list, labels: list, label_map: dict,
                 sample_rate: int = 16000, max_duration: float = 5.0,
                 augment: bool = False, oversample_minority: bool = False):
        
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
        self.label_map = label_map
        self.augment = augment
        
        # Handle oversampling
        if oversample_minority:
            file_paths, labels = self._oversample(file_paths, labels)
        
        self.file_paths = file_paths
        self.labels = labels
    
    def _oversample(self, file_paths, labels):
        """Oversample minority classes to match majority class count"""
        counter = Counter(labels)
        max_count = max(counter.values())
        
        print("\n  Class distribution before oversampling:")
        for label, count in sorted(counter.items()):
            print(f"    {label}: {count}")
        
        new_paths = []
        new_labels = []
        
        # Group files by label
        label_to_files = {}
        for path, label in zip(file_paths, labels):
            if label not in label_to_files:
                label_to_files[label] = []
            label_to_files[label].append(path)
        
        # Oversample each class to max_count
        for label, files in label_to_files.items():
            current_count = len(files)
            needed = max_count - current_count
            
            # Add original files
            new_paths.extend(files)
            new_labels.extend([label] * len(files))
            
            # Add duplicates (will be augmented differently each time)
            if needed > 0:
                oversampled = random.choices(files, k=needed)
                new_paths.extend(oversampled)
                new_labels.extend([label] * needed)
        
        counter_after = Counter(new_labels)
        print("\n  Class distribution after oversampling:")
        for label, count in sorted(counter_after.items()):
            print(f"    {label}: {count}")
        
        return new_paths, new_labels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform = np.zeros(self.max_length, dtype=np.float32)
        
        # Apply augmentation
        if self.augment:
            waveform = self._augment(waveform)
        
        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        # Pad or truncate
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]
        
        label = self.label_map[self.labels[idx]]
        
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label)
    
    def _augment(self, waveform):
        """Apply random audio augmentations"""
        augmentations = [
            self._time_stretch,
            self._pitch_shift,
            self._add_noise,
            self._time_mask,
        ]
        
        # Apply 1-2 random augmentations
        n_augs = random.randint(1, 2)
        selected = random.sample(augmentations, n_augs)
        
        for aug in selected:
            try:
                waveform = aug(waveform)
            except:
                pass
        
        return waveform
    
    def _time_stretch(self, waveform):
        """Stretch time by ±10%"""
        rate = random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(waveform, rate=rate)
    
    def _pitch_shift(self, waveform):
        """Shift pitch by ±2 semitones"""
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(waveform, sr=self.sample_rate, n_steps=n_steps)
    
    def _add_noise(self, waveform):
        """Add random Gaussian noise"""
        noise_level = random.uniform(0.001, 0.01)
        noise = np.random.randn(len(waveform)) * noise_level
        return waveform + noise
    
    def _time_mask(self, waveform):
        """Mask random time segment"""
        mask_length = int(len(waveform) * random.uniform(0.05, 0.15))
        start = random.randint(0, len(waveform) - mask_length)
        waveform[start:start + mask_length] = 0
        return waveform


# ==============================================================================
# Class Weight Calculation
# ==============================================================================

def compute_class_weights(labels, method='inverse_freq'):
    """
    Compute class weights to balance training.
    
    Methods:
    - 'inverse_freq': Weight = 1 / class_frequency (standard)
    - 'inverse_sqrt': Weight = 1 / sqrt(class_frequency) (smoother)
    - 'effective_num': Effective number of samples (from paper)
    - 'equal': All weights = 1 (baseline)
    """
    counter = Counter(labels)
    n_samples = len(labels)
    n_classes = len(counter)
    
    if method == 'equal':
        return {label: 1.0 for label in counter.keys()}
    
    elif method == 'inverse_freq':
        # Weight inversely proportional to frequency
        weights = {}
        for label, count in counter.items():
            weights[label] = n_samples / (n_classes * count)
        return weights
    
    elif method == 'inverse_sqrt':
        # Smoother version
        weights = {}
        for label, count in counter.items():
            weights[label] = np.sqrt(n_samples / (n_classes * count))
        return weights
    
    elif method == 'effective_num':
        # Effective number of samples method
        # From: "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        weights = {}
        for label, count in counter.items():
            effective_num = 1.0 - np.power(beta, count)
            weights[label] = (1.0 - beta) / effective_num
        # Normalize
        total = sum(weights.values())
        weights = {k: v * n_classes / total for k, v in weights.items()}
        return weights
    
    else:
        raise ValueError(f"Unknown method: {method}")


def get_sample_weights(labels, class_weights):
    """Convert class weights to per-sample weights for WeightedRandomSampler"""
    return [class_weights[label] for label in labels]


# ==============================================================================
# Data Collection
# ==============================================================================

def collect_cry_data(data_dir: Path):
    """Collect baby cry training data"""
    folder_to_label = {
        "hungry_cry": "hungry",
        "pain_cry": "pain", 
        "sleepy_cry": "sleepy",
        "discomfort_cry": "discomfort",
        "cold_cry": "cold_hot",
        "distress_cry": "pathological",
        "tired_cry": "tired",
        "normal_cry": "normal",
        "belly_pain": "belly_pain",
        "burping": "burping",
        "scared": "scared",
        "lonely": "lonely",
    }
    
    file_paths = []
    labels = []
    
    baby_resp_dir = data_dir / "data_baby_respiratory"
    if baby_resp_dir.exists():
        for class_dir in baby_resp_dir.iterdir():
            if class_dir.is_dir():
                folder_name = class_dir.name.lower()
                mapped_label = folder_to_label.get(folder_name)
                
                if mapped_label is None:
                    for folder_pattern, label in folder_to_label.items():
                        if folder_pattern in folder_name or folder_name in folder_pattern:
                            mapped_label = label
                            break
                
                if mapped_label:
                    for audio_file in list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3")):
                        file_paths.append(str(audio_file))
                        labels.append(mapped_label)
    
    print(f"Collected {len(file_paths)} cry audio samples")
    return file_paths, labels


def collect_pulmonary_data(data_dir: Path):
    """Collect pulmonary/respiratory training data"""
    pulmonary_classes = {
        "normal": ["normal", "normal_breathing"],
        "wheeze": ["wheeze"],
        "crackle": ["crackle", "coarse_crackle", "fine_crackle"],
        "stridor": ["stridor"],
        "rhonchi": ["rhonchi"],
        "bronchiolitis": ["bronchiolitis"],
        "pneumonia": ["pneumonia"],
        "asthma": ["asthma"]
    }
    
    file_paths = []
    labels = []
    
    for data_subdir in ["data_baby_pulmonary", "data_adult_respiratory"]:
        pulm_dir = data_dir / data_subdir
        if pulm_dir.exists():
            for class_dir in pulm_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name.lower()
                    mapped_label = None
                    
                    for pulm_class, patterns in pulmonary_classes.items():
                        for pattern in patterns:
                            if pattern in class_name:
                                mapped_label = pulm_class
                                break
                        if mapped_label:
                            break
                    
                    if mapped_label:
                        for audio_file in list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3")):
                            file_paths.append(str(audio_file))
                            labels.append(mapped_label)
    
    print(f"Collected {len(file_paths)} pulmonary audio samples")
    return file_paths, labels


# ==============================================================================
# Embedding Extraction
# ==============================================================================

async def extract_embeddings(model: EnsembleModel, dataloader: DataLoader, 
                            model_name: str, device: torch.device):
    """Extract embeddings from a backbone model"""
    embeddings_list = []
    labels_list = []
    
    backbone = model.models.get(model_name)
    processor = model.processors.get(model_name)
    
    if backbone is None:
        return None, None
    
    backbone.eval()
    
    for batch_waveforms, batch_labels in tqdm(dataloader, desc=f"Extracting {model_name}"):
        with torch.no_grad():
            for waveform, label in zip(batch_waveforms, batch_labels):
                waveform_np = waveform.numpy()
                
                try:
                    if model_name == "yamnet":
                        scores, emb, _ = backbone(waveform_np)
                        embedding = emb.numpy().mean(axis=0)
                    elif model_name == "panns":
                        mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=16000, n_mels=128, fmax=8000)
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                        embedding = backbone(mel_tensor).cpu().numpy()[0]
                    elif model_name == "ast":
                        inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = backbone(**inputs, output_hidden_states=True)
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]
                        else:
                            embedding = outputs.logits.cpu().numpy()[0]
                    else:
                        inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = backbone(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    
                    embeddings_list.append(embedding)
                    labels_list.append(label.item())
                    
                except Exception as e:
                    print(f"  Error extracting from {model_name}: {e}")
    
    if len(embeddings_list) == 0:
        return None, None
    
    return np.array(embeddings_list), np.array(labels_list)


# ==============================================================================
# Balanced Classifier Training
# ==============================================================================

def train_classifier_balanced(embeddings: np.ndarray, labels: np.ndarray, 
                              classifier: nn.Module, device: torch.device,
                              num_classes: int, epochs: int = 100, 
                              lr: float = 1e-3, batch_size: int = 32,
                              weight_method: str = 'effective_num'):
    """
    Train a classifier head with balanced sampling and focal loss.
    """
    
    # Split data with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Compute class weights
    unique_labels = np.unique(y_train)
    label_counts = Counter(y_train)
    
    print(f"\n  Training set class distribution:")
    for label in sorted(unique_labels):
        print(f"    Class {label}: {label_counts[label]} samples")
    
    # Compute weights using effective number method
    beta = 0.9999
    class_weights = []
    for i in range(num_classes):
        if i in label_counts:
            count = label_counts[i]
            effective_num = 1.0 - np.power(beta, count)
            weight = (1.0 - beta) / effective_num
        else:
            weight = 1.0
        class_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(class_weights)
    class_weights = [w * num_classes / total_weight for w in class_weights]
    
    print(f"\n  Computed class weights (method={weight_method}):")
    for i, w in enumerate(class_weights):
        print(f"    Class {i}: {w:.4f}")
    
    # Create tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # Create sample weights for WeightedRandomSampler
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create simple dataset for sampler
    train_dataset = torch.utils.data.TensorDataset(X_train_t.cpu(), y_train_t.cpu())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    # Optimizer with weight decay
    optimizer = AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Use Focal Loss with class weights
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    
    # Also track with label smoothing loss for comparison
    # criterion_smooth = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    
    best_val_acc = 0.0
    best_state = None
    patience = 20
    no_improve_count = 0
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)
        
        train_acc = correct / total
        
        # Validation
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(X_val_t)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
            
            # Per-class accuracy
            per_class_correct = {}
            per_class_total = {}
            for pred, true in zip(val_preds.cpu().numpy(), y_val_t.cpu().numpy()):
                if true not in per_class_total:
                    per_class_total[true] = 0
                    per_class_correct[true] = 0
                per_class_total[true] += 1
                if pred == true:
                    per_class_correct[true] += 1
        
        scheduler.step()
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {train_acc:.4f}, Val: {val_acc:.4f} (best: {best_val_acc:.4f})")
            
            # Show per-class accuracy at key epochs
            if (epoch + 1) % 25 == 0:
                print("    Per-class val accuracy:")
                for cls in sorted(per_class_total.keys()):
                    cls_acc = per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0
                    print(f"      Class {cls}: {cls_acc:.4f} ({per_class_correct[cls]}/{per_class_total[cls]})")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best state
    if best_state:
        classifier.load_state_dict(best_state)
    
    # Final validation metrics
    classifier.eval()
    with torch.no_grad():
        val_outputs = classifier(X_val_t)
        val_preds = val_outputs.argmax(dim=1).cpu().numpy()
        val_true = y_val_t.cpu().numpy()
    
    print(f"\n  Final Classification Report:")
    print(classification_report(val_true, val_preds, zero_division=0))
    
    return best_val_acc


# ==============================================================================
# Main Training Loop
# ==============================================================================

async def train_task(ensemble: EnsembleModel, task: str, data_dir: Path, 
                     save_dir: Path, epochs: int = 100, 
                     oversample: bool = True, weight_method: str = 'effective_num'):
    """Train all classifier heads for a specific task with balancing"""
    
    print(f"\n{'='*70}")
    print(f"BALANCED TRAINING: 6-Backbone Ensemble for {task.upper()}")
    print(f"{'='*70}")
    print(f"  Oversampling: {oversample}")
    print(f"  Weight method: {weight_method}")
    print(f"  Epochs: {epochs}")
    
    # Collect data
    if task == "cry":
        file_paths, labels = collect_cry_data(data_dir)
        classes = ensemble.cry_classes
    else:
        file_paths, labels = collect_pulmonary_data(data_dir)
        classes = ensemble.pulmonary_classes
    
    if len(file_paths) == 0:
        print(f"No training data found for {task}!")
        return
    
    # Show original class distribution
    print("\n  Original class distribution:")
    counter = Counter(labels)
    for label, count in sorted(counter.items()):
        print(f"    {label}: {count}")
    
    # Check for severe imbalance
    max_count = max(counter.values())
    min_count = min(counter.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n  Imbalance ratio (max/min): {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 5:
        print("  [WARNING] Severe class imbalance detected! Using aggressive balancing.")
    
    # Create label map
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_map.items()}
    print(f"\n  Label mapping: {label_map}")
    
    # Create balanced dataset
    dataset = BalancedAudioDataset(
        file_paths, labels, label_map,
        augment=True,
        oversample_minority=oversample
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Train each backbone
    device = ensemble.device
    num_classes = len(unique_labels)
    
    models_to_train = ["distilhubert", "wav2vec2", "wavlm", "ast", "panns"]
    
    for model_name in models_to_train:
        if ensemble.models.get(model_name) is None:
            print(f"\n  Skipping {model_name} (not loaded)")
            continue
        
        print(f"\n  {'='*50}")
        print(f"  Training {model_name}_{task} classifier")
        print(f"  {'='*50}")
        
        # Extract embeddings
        embeddings, embedding_labels = await extract_embeddings(
            ensemble, dataloader, model_name, device
        )
        
        if embeddings is None:
            print(f"  Failed to extract embeddings from {model_name}")
            continue
        
        print(f"  Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
        
        # Get classifier
        classifier_name = f"{model_name}_{task}"
        classifier = ensemble.classifiers.get(classifier_name)
        
        if classifier is None:
            print(f"  No classifier found for {classifier_name}")
            continue
        
        # Move to device
        classifier = classifier.to(device)
        
        # Train with balanced approach
        best_acc = train_classifier_balanced(
            embeddings, embedding_labels, classifier, device,
            num_classes=num_classes, epochs=epochs,
            weight_method=weight_method
        )
        
        print(f"  {model_name}_{task}: Best validation accuracy = {best_acc:.4f}")
        
        # Save classifier weights
        save_path = save_dir / task
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), save_path / f"{classifier_name}.pt")
        print(f"  Saved: {save_path / f'{classifier_name}.pt'}")
    
    print(f"\n  Training complete for {task}!")


async def main():
    parser = argparse.ArgumentParser(description="Balanced 6-Backbone Ensemble Training")
    parser.add_argument("--task", choices=["cry", "pulmonary", "both"], default="cry",
                        help="Training task")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--no-oversample", action="store_true", 
                        help="Disable oversampling minority classes")
    parser.add_argument("--weight-method", choices=["inverse_freq", "inverse_sqrt", "effective_num", "equal"],
                        default="effective_num", help="Class weighting method")
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument("--save-dir", type=str, default="trained_classifiers", 
                        help="Output directory for trained weights")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    
    print("=" * 70)
    print("  BALANCED 6-BACKBONE ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"  Task: {args.task}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Oversampling: {not args.no_oversample}")
    print(f"  Weight method: {args.weight_method}")
    print(f"  Data dir: {data_dir}")
    print(f"  Save dir: {save_dir}")
    print("=" * 70)
    
    # Initialize ensemble
    print("\nInitializing 6-Backbone Ensemble...")
    ensemble = EnsembleModel()
    await ensemble.initialize()
    
    # Train tasks
    if args.task in ["cry", "both"]:
        await train_task(
            ensemble, "cry", data_dir, save_dir, 
            epochs=args.epochs,
            oversample=not args.no_oversample,
            weight_method=args.weight_method
        )
    
    if args.task in ["pulmonary", "both"]:
        await train_task(
            ensemble, "pulmonary", data_dir, save_dir,
            epochs=args.epochs,
            oversample=not args.no_oversample,
            weight_method=args.weight_method
        )
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Trained classifiers saved to: {save_dir}")
    print("\n  To use these weights:")
    print(f"    1. Copy {save_dir}/* to cloud_deployment/models/trained_weights/")
    print("    2. Redeploy to Render")
    print("=" * 70)
    
    # Cleanup
    await ensemble.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
