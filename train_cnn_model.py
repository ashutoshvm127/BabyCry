"""
Best Model Training Script
Uses CNN on Mel Spectrograms with heavy augmentation.
Combines all available cry datasets for maximum accuracy.
"""

import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, filtfilt
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(r"D:\projects\cry analysuis\data_baby_respiratory")
ESC50_DIR = Path(r"D:\projects\cry analysuis\downloads\esc50\ESC-50-master\audio")
ESC50_META = Path(r"D:\projects\cry analysuis\downloads\esc50\ESC-50-master\meta\esc50.csv")
OUTPUT_DIR = Path(r"D:\projects\cry analysuis\cnn_baby_cry_model")

# Audio config
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Training config
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ============= AUGMENTATION =============

def add_noise(audio, factor=0.01):
    return audio + factor * np.random.randn(len(audio))

def lowpass(audio, sr, cutoff=4000):
    nyq = 0.5 * sr
    b, a = butter(4, min(cutoff/nyq, 0.99), btype='low')
    return filtfilt(b, a, audio)

def time_shift(audio, shift_max=0.2):
    shift = int(len(audio) * np.random.uniform(-shift_max, shift_max))
    return np.roll(audio, shift)

def change_speed(audio, sr, factor=None):
    if factor is None:
        factor = np.random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(audio, rate=factor)

def change_pitch(audio, sr, n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def speaker_mic_sim(audio, sr):
    """Simulate speaker -> microphone recording"""
    audio = lowpass(audio, sr, np.random.uniform(3000, 5000))
    audio = add_noise(audio, np.random.uniform(0.005, 0.02))
    # Simple reverb
    delay = int(sr * np.random.uniform(0.02, 0.05))
    out = np.zeros(len(audio) + delay)
    out[:len(audio)] = audio
    out[delay:delay+len(audio)] += audio * np.random.uniform(0.1, 0.3)
    return out[:len(audio)]

def augment_audio(audio, sr, aug_type='random'):
    """Apply random augmentation"""
    if aug_type == 'random':
        aug_type = np.random.choice(['noise', 'shift', 'speed', 'pitch', 'speaker_mic', 'combined'])
    
    if aug_type == 'noise':
        return add_noise(audio, np.random.uniform(0.005, 0.02))
    elif aug_type == 'shift':
        return time_shift(audio, 0.2)
    elif aug_type == 'speed':
        return change_speed(audio, sr)
    elif aug_type == 'pitch':
        return change_pitch(audio, sr)
    elif aug_type == 'speaker_mic':
        return speaker_mic_sim(audio, sr)
    elif aug_type == 'combined':
        audio = speaker_mic_sim(audio, sr)
        audio = time_shift(audio, 0.1)
        return audio
    return audio

# ============= FEATURE EXTRACTION =============

def audio_to_melspec(audio, sr=SAMPLE_RATE):
    """Convert audio to mel spectrogram"""
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Ensure correct length
    target_len = sr * DURATION
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    return mel_db.astype(np.float32)

# ============= DATASET =============

class CryDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_paths[idx], sr=SAMPLE_RATE, duration=DURATION)
        
        # Augment if training
        if self.augment and np.random.random() > 0.5:
            try:
                audio = augment_audio(audio, sr, 'random')
            except:
                pass
        
        # Convert to mel spectrogram
        mel = audio_to_melspec(audio, sr)
        
        # Add channel dimension
        mel = np.expand_dims(mel, 0)
        
        return torch.FloatTensor(mel), self.labels[idx]

# ============= MODEL =============

class CryCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============= DATA LOADING =============

def load_all_data():
    """Load data from all available sources"""
    file_paths = []
    labels = []
    
    # 1. Current dataset
    print("Loading current dataset...")
    for cls_dir in DATA_DIR.iterdir():
        if cls_dir.is_dir():
            label = cls_dir.name
            for f in cls_dir.glob('*'):
                if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                    file_paths.append(str(f))
                    labels.append(label)
    
    print(f"  Loaded {len(file_paths)} files from current dataset")
    
    # 2. ESC-50 crying_baby (add to distress_cry category)
    if ESC50_META.exists():
        print("Loading ESC-50 crying_baby...")
        import csv
        with open(ESC50_META) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('category') == 'crying_baby':
                    audio_file = ESC50_DIR / row['filename']
                    if audio_file.exists():
                        file_paths.append(str(audio_file))
                        labels.append('distress_cry')  # Map to distress
    
    print(f"  Total files: {len(file_paths)}")
    
    # Count per class
    from collections import Counter
    counts = Counter(labels)
    print("\nClass distribution:")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count}")
    
    return file_paths, labels

# ============= TRAINING =============

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), correct / total

def main():
    print("=" * 60)
    print("CNN BABY CRY CLASSIFIER - BEST MODEL")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    file_paths, labels = load_all_data()
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    
    print(f"\nClasses ({num_classes}): {list(le.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Compute class weights for balanced sampling
    from collections import Counter
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Create datasets
    train_dataset = CryDataset(X_train, y_train, augment=True)
    test_dataset = CryDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("\n[2/4] Creating model...")
    model = CryCNN(num_classes).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with class weights
    class_weights = torch.FloatTensor([1.0 / class_counts[i] for i in range(num_classes)])
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training
    print("\n[3/4] Training...")
    best_acc = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f} Acc={train_acc:.1%} | Test Loss={test_loss:.4f} Acc={test_acc:.1%}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
            # Save best model
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    
    # Final evaluation per class
    print("\n[4/4] Final Evaluation...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    print("\nPer-class accuracy:")
    for i, cls in enumerate(le.classes_):
        mask = np.array(all_targets) == i
        if mask.sum() > 0:
            acc = (np.array(all_preds)[mask] == np.array(all_targets)[mask]).mean()
            print(f"  {cls}: {acc:.1%} ({mask.sum()} samples)")
    
    # Save model and config
    print("\nSaving model...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save for inference
    model_data = {
        'model_state_dict': model.state_dict(),
        'label_encoder': le,
        'classes': list(le.classes_),
        'num_classes': num_classes,
        'config': {
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION,
            'n_mels': N_MELS,
            'hop_length': HOP_LENGTH,
            'n_fft': N_FFT
        }
    }
    torch.save(model_data, OUTPUT_DIR / "model.pt")
    
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump({
            'model_type': 'cnn_melspec',
            'accuracy': float(best_acc),
            'num_classes': num_classes,
            'classes': list(le.classes_),
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Best Test Accuracy: {best_acc:.1%}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
