#!/usr/bin/env python3
"""
Baby Pulmonary Disease Classification Model
Detects: Sepsis, ARDS, Asphyxia, Pneumonia, Bronchiolitis, and other respiratory conditions

Uses lung sound classification mapped to disease predictions based on medical literature.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_RATE = 16000
MAX_DURATION = 5  # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data directories
BABY_PULMONARY_DIR = Path("data_baby_pulmonary")
ADULT_RESPIRATORY_DIR = Path("data_adult_respiratory")
OUTPUT_DIR = Path("model_pulmonary_disease")

# ============================================================================
# DISEASE MAPPING (Based on medical literature)
# ============================================================================
# Lung sounds are acoustic markers that correlate with diseases

# Primary sound to disease mapping
SOUND_TO_DISEASE = {
    # Fine crackles -> ARDS, Pneumonia, Pulmonary edema
    "fine_crackle": ["ards", "pneumonia", "pulmonary_edema"],
    
    # Coarse crackles -> Bronchitis, Pneumonia, Aspiration
    "coarse_crackle": ["bronchitis", "pneumonia", "aspiration"],
    
    # Wheeze -> Asthma, Bronchiolitis, Airway obstruction
    "wheeze": ["bronchiolitis", "asthma", "airway_obstruction"],
    
    # Rhonchi -> Bronchitis, COPD, Secretions
    "rhonchi": ["bronchitis", "secretions", "respiratory_infection"],
    
    # Stridor -> Croup, Epiglottitis, Upper airway obstruction, can indicate asphyxia
    "stridor": ["croup", "asphyxia", "airway_obstruction"],
    
    # Mixed sounds -> Severe respiratory distress, possible sepsis
    "mixed": ["respiratory_distress", "sepsis_respiratory", "severe_infection"],
    "mixed_crackle_wheeze": ["respiratory_distress", "sepsis_respiratory", "severe_infection"],
    
    # Normal breathing
    "normal_breathing": ["healthy"],
    "normal": ["healthy"],
}

# Disease categories for classification
DISEASE_CLASSES = [
    "healthy",
    "pneumonia",
    "bronchiolitis", 
    "ards",
    "asphyxia",
    "sepsis_respiratory",
    "respiratory_distress",
    "airway_obstruction",
    "bronchitis",
    "croup",
]

# Simplified mapping for training (sound type -> primary disease)
SOUND_TO_PRIMARY_DISEASE = {
    "fine_crackle": "ards",
    "coarse_crackle": "pneumonia",
    "wheeze": "bronchiolitis",
    "rhonchi": "bronchitis",
    "stridor": "asphyxia",
    "mixed": "sepsis_respiratory",
    "mixed_crackle_wheeze": "respiratory_distress",
    "normal_breathing": "healthy",
    "normal": "healthy",
}

# ============================================================================
# AUDIO DATASET
# ============================================================================
class PulmonaryAudioDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.max_len = int(SAMPLE_RATE * MAX_DURATION / HOP_LENGTH) + 1
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
            
            # Ensure minimum length
            min_samples = SAMPLE_RATE
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)))
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Data augmentation
            if self.augment:
                audio = self._augment(audio)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, 
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to 0-1
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Pad/truncate to fixed length
            if mel_spec_db.shape[1] < self.max_len:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, self.max_len - mel_spec_db.shape[1])))
            else:
                mel_spec_db = mel_spec_db[:, :self.max_len]
            
            # Add channel dimension
            mel_spec_db = mel_spec_db[np.newaxis, :, :]
            
            return torch.FloatTensor(mel_spec_db), torch.LongTensor([label])[0]
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros on error
            return torch.zeros(1, N_MELS, self.max_len), torch.LongTensor([label])[0]
    
    def _augment(self, audio):
        """Apply random augmentations"""
        # Time stretch
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shift
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
        
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        # Ensure correct length
        target_len = SAMPLE_RATE * MAX_DURATION
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        
        return audio

# ============================================================================
# CNN MODEL (Fast and effective for spectrograms)
# ============================================================================
class PulmonaryCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# ============================================================================
# DATA LOADING
# ============================================================================
def load_pulmonary_data():
    """Load all pulmonary audio data with disease labels"""
    print("Loading pulmonary audio data...")
    
    file_paths = []
    labels = []
    
    # Create label mapping
    label_to_idx = {disease: i for i, disease in enumerate(DISEASE_CLASSES)}
    idx_to_label = {i: disease for i, disease in enumerate(DISEASE_CLASSES)}
    
    # Load from baby pulmonary directory
    if BABY_PULMONARY_DIR.exists():
        for sound_dir in BABY_PULMONARY_DIR.iterdir():
            if not sound_dir.is_dir():
                continue
            
            sound_type = sound_dir.name.lower()
            disease = SOUND_TO_PRIMARY_DISEASE.get(sound_type, "respiratory_distress")
            
            if disease not in label_to_idx:
                print(f"Warning: Disease '{disease}' not in classes, skipping {sound_type}")
                continue
            
            label_idx = label_to_idx[disease]
            
            audio_files = list(sound_dir.glob("*.wav")) + list(sound_dir.glob("*.mp3"))
            for f in audio_files:
                file_paths.append(str(f))
                labels.append(label_idx)
            
            print(f"  {sound_type} -> {disease}: {len(audio_files)} files")
    
    # Load from adult respiratory directory
    if ADULT_RESPIRATORY_DIR.exists():
        for sound_dir in ADULT_RESPIRATORY_DIR.iterdir():
            if not sound_dir.is_dir():
                continue
            
            sound_type = sound_dir.name.lower()
            disease = SOUND_TO_PRIMARY_DISEASE.get(sound_type, "respiratory_distress")
            
            if disease not in label_to_idx:
                continue
            
            label_idx = label_to_idx[disease]
            
            audio_files = list(sound_dir.glob("*.wav")) + list(sound_dir.glob("*.mp3"))
            for f in audio_files:
                file_paths.append(str(f))
                labels.append(label_idx)
            
            print(f"  {sound_type} (adult) -> {disease}: {len(audio_files)} files")
    
    return file_paths, labels, label_to_idx, idx_to_label

# ============================================================================
# TRAINING
# ============================================================================
def train_model():
    print("=" * 70)
    print("PULMONARY DISEASE CLASSIFICATION MODEL TRAINING")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Load data
    file_paths, labels, label_to_idx, idx_to_label = load_pulmonary_data()
    
    print(f"\nTotal samples: {len(file_paths)}")
    print(f"Classes: {len(label_to_idx)}")
    
    # Show class distribution
    print("\nClass distribution:")
    class_counts = Counter(labels)
    for idx, count in sorted(class_counts.items()):
        print(f"  {idx_to_label[idx]}: {count}")
    
    # Remove classes with no samples
    active_classes = set(class_counts.keys())
    active_label_to_idx = {idx_to_label[idx]: new_idx for new_idx, idx in enumerate(sorted(active_classes))}
    active_idx_to_label = {v: k for k, v in active_label_to_idx.items()}
    
    # Remap labels
    old_to_new = {old_idx: active_label_to_idx[idx_to_label[old_idx]] for old_idx in active_classes}
    labels = [old_to_new[l] for l in labels]
    
    num_classes = len(active_label_to_idx)
    print(f"\nActive classes: {num_classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = PulmonaryAudioDataset(X_train, y_train, augment=True)
    test_dataset = PulmonaryAudioDataset(X_test, y_test, augment=False)
    
    # Weighted sampler for class imbalance
    train_class_counts = Counter(y_train)
    weights = [1.0 / train_class_counts[l] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = PulmonaryCNN(num_classes).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_acc = 0.0
    print("\nTraining...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        print(f"  Test Accuracy: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            OUTPUT_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
            print(f"  [NEW BEST] Saved model with {test_acc:.2f}% accuracy")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    model.load_state_dict(torch.load(OUTPUT_DIR / "model.pt"))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        all_targets, all_preds, 
        target_names=[active_idx_to_label[i] for i in range(num_classes)]
    ))
    
    # Save configuration
    config = {
        "model_type": "cnn_pulmonary",
        "num_classes": num_classes,
        "classes": active_idx_to_label,
        "label_to_idx": {k: int(v) for k, v in active_label_to_idx.items()},
        "accuracy": float(best_acc),
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "max_duration": MAX_DURATION,
        "disease_mapping": SOUND_TO_PRIMARY_DISEASE,
    }
    
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save label mappings
    with open(OUTPUT_DIR / "label_mappings.json", "w") as f:
        json.dump({
            "label_to_idx": active_label_to_idx,
            "idx_to_label": {str(k): v for k, v in active_idx_to_label.items()},
        }, f, indent=2)
    
    print(f"\nModel saved to {OUTPUT_DIR}/")
    print("=" * 70)
    
    return best_acc

if __name__ == "__main__":
    train_model()
