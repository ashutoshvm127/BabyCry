"""
Optimized Training Script for Baby Cry Classification
Uses lightweight EfficientNet-style architecture with stronger augmentation
and focal loss to handle class imbalance better.
"""

import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import butter, filtfilt
import json
import pickle
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(r"D:\projects\cry analysuis\data_baby_respiratory")
DONATEACRY_DIR = Path(r"D:\projects\cry analysuis\downloads\donateacry\donateacry-corpus-master\donateacry_corpus_cleaned_and_updated_data")
ESC50_DIR = Path(r"D:\projects\cry analysuis\downloads\esc50\ESC-50-master\audio")
ESC50_META = Path(r"D:\projects\cry analysuis\downloads\esc50\ESC-50-master\meta\esc50.csv")
OUTPUT_DIR = Path(r"D:\projects\cry analysuis\best_baby_cry_model")

# Audio config
SAMPLE_RATE = 16000
DURATION = 5  # seconds

print("=" * 60)
print("BEST BABY CRY CLASSIFIER - COMBINED APPROACH")
print("=" * 60)

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

def speaker_mic_sim(audio, sr):
    """Simulate speaker -> microphone recording"""
    audio = lowpass(audio, sr, np.random.uniform(2500, 4500))
    audio = add_noise(audio, np.random.uniform(0.01, 0.03))
    # Simple reverb
    delay = int(sr * np.random.uniform(0.02, 0.06))
    out = np.zeros(len(audio) + delay)
    out[:len(audio)] = audio
    out[delay:delay+len(audio)] += audio * np.random.uniform(0.15, 0.35)
    return out[:len(audio)]

def augment_audio(audio, sr):
    """Apply random augmentation"""
    aug_type = np.random.choice(['noise', 'shift', 'speaker_mic', 'combined', 'heavy'])
    
    if aug_type == 'noise':
        return add_noise(audio, np.random.uniform(0.005, 0.025))
    elif aug_type == 'shift':
        return time_shift(audio, 0.25)
    elif aug_type == 'speaker_mic':
        return speaker_mic_sim(audio, sr)
    elif aug_type == 'combined':
        audio = speaker_mic_sim(audio, sr)
        audio = time_shift(audio, 0.15)
        return audio
    elif aug_type == 'heavy':
        # Heavy augmentation for robustness
        audio = lowpass(audio, sr, np.random.uniform(2000, 3500))
        audio = add_noise(audio, np.random.uniform(0.02, 0.04))
        delay = int(sr * np.random.uniform(0.03, 0.08))
        out = np.zeros(len(audio) + delay)
        out[:len(audio)] = audio
        out[delay:delay+len(audio)] += audio * np.random.uniform(0.2, 0.4)
        return out[:len(audio)]
    return audio

# ============= FEATURE EXTRACTION =============

def extract_features(audio, sr=SAMPLE_RATE):
    """Extract comprehensive audio features"""
    features = []
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Ensure correct length
    target_len = sr * DURATION
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    
    # MFCCs (40 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    features.extend(np.max(mfccs, axis=1))
    features.extend(np.min(mfccs, axis=1))
    
    # Delta MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta_mfccs, axis=1))
    features.extend(np.std(delta_mfccs, axis=1))
    
    # Delta-delta MFCCs
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features.extend(np.mean(delta2_mfccs, axis=1))
    features.extend(np.std(delta2_mfccs, axis=1))
    
    # Mel spectrogram statistics
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(spectral_rolloff))
    features.append(np.std(spectral_rolloff))
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend(np.mean(spectral_contrast, axis=1))
    features.extend(np.std(spectral_contrast, axis=1))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # RMS energy
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    # Pitch features using pyin
    try:
        f0, voiced, voice_prob = librosa.pyin(audio, fmin=50, fmax=2000, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features.append(np.mean(f0_clean))
            features.append(np.std(f0_clean))
            features.append(np.max(f0_clean))
            features.append(np.min(f0_clean))
        else:
            features.extend([0, 0, 0, 0])
    except:
        features.extend([0, 0, 0, 0])
    
    # Tonnetz
    try:
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.extend(np.mean(tonnetz, axis=1))
        features.extend(np.std(tonnetz, axis=1))
    except:
        features.extend([0] * 12)
    
    return np.array(features, dtype=np.float32)

# ============= DATA LOADING =============

def load_all_data():
    """Load data from all available sources"""
    file_paths = []
    labels = []
    
    # Class mapping for DonateACry
    donate_mapping = {
        'belly_pain': 'pain_cry',
        'burping': 'discomfort_cry',
        'discomfort': 'discomfort_cry',
        'hungry': 'hungry_cry',
        'tired': 'tired_cry'
    }
    
    # 1. Current dataset
    print("  Loading current dataset...")
    for cls_dir in DATA_DIR.iterdir():
        if cls_dir.is_dir():
            label = cls_dir.name
            count = 0
            for f in cls_dir.glob('*'):
                if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                    file_paths.append(str(f))
                    labels.append(label)
                    count += 1
            print(f"    {label}: {count}")
    
    # 2. DonateACry dataset
    if DONATEACRY_DIR.exists():
        print("  Loading DonateACry dataset...")
        for cls_dir in DONATEACRY_DIR.iterdir():
            if cls_dir.is_dir() and cls_dir.name in donate_mapping:
                mapped_label = donate_mapping[cls_dir.name]
                count = 0
                for f in cls_dir.glob('*'):
                    if f.suffix.lower() in ['.wav', '.mp3', '.ogg']:
                        file_paths.append(str(f))
                        labels.append(mapped_label)
                        count += 1
                print(f"    {cls_dir.name} -> {mapped_label}: {count}")
    
    # 3. ESC-50 crying_baby
    if ESC50_META.exists():
        print("  Loading ESC-50 crying_baby...")
        import csv
        count = 0
        with open(ESC50_META) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('category') == 'crying_baby':
                    audio_file = ESC50_DIR / row['filename']
                    if audio_file.exists():
                        file_paths.append(str(audio_file))
                        labels.append('distress_cry')
                        count += 1
        print(f"    crying_baby -> distress_cry: {count}")
    
    return file_paths, labels

# ============= TRAINING =============

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading all data sources...")
    file_paths, labels = load_all_data()
    
    print(f"\nTotal samples: {len(file_paths)}")
    counts = Counter(labels)
    print("\nFinal class distribution:")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count}")
    
    # Extract features with augmentation
    print("\n[2/5] Extracting features (with heavy augmentation)...")
    
    X_list = []
    y_list = []
    
    from tqdm import tqdm
    for i, (fp, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths))):
        try:
            audio, sr = librosa.load(fp, sr=SAMPLE_RATE, duration=DURATION)
            
            # Original
            features = extract_features(audio, sr)
            X_list.append(features)
            y_list.append(label)
            
            # Determine augmentation multiplier based on class frequency
            class_count = counts[label]
            max_count = max(counts.values())
            
            # Augment minority classes more
            if class_count < 100:
                n_aug = 10  # Heavy augmentation for very small classes
            elif class_count < 200:
                n_aug = 6
            elif class_count < 400:
                n_aug = 3
            else:
                n_aug = 2
            
            # Also augment everything with speaker-mic sim for robustness
            for _ in range(n_aug):
                aug_audio = augment_audio(audio, sr)
                aug_features = extract_features(aug_audio, sr)
                X_list.append(aug_features)
                y_list.append(label)
                
        except Exception as e:
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Features extracted: {X.shape}")
    
    # Handle any NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("\nAugmented class distribution:")
    aug_counts = Counter(y)
    for cls, count in sorted(aug_counts.items()):
        print(f"  {cls}: {count}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train Random Forest with optimized parameters
    print("\n[3/5] Training Random Forest classifier...")
    
    # Compute class weights
    class_counts = Counter(y_train)
    class_weights = {i: len(y_train) / (len(class_counts) * count) 
                     for i, count in class_counts.items()}
    
    rf_model = RandomForestClassifier(
        n_estimators=800,
        max_depth=40,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight=class_weights,
        n_jobs=-1,
        random_state=42,
        bootstrap=True,
        oob_score=True
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[4/5] Evaluating model...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Test Accuracy: {accuracy:.1%}")
    print(f"OOB Score: {rf_model.oob_score_:.1%}")
    
    print("\nPer-class accuracy:")
    for i, cls in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == y_test[mask]).mean()
            print(f"  {cls}: {cls_acc:.1%} ({mask.sum()} samples)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model
    print("\n[5/5] Saving model...")
    
    model_data = {
        'model': rf_model,
        'label_encoder': le,
        'feature_dim': X.shape[1]
    }
    
    with open(OUTPUT_DIR / 'rf_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump({
            'model_type': 'random_forest',
            'accuracy': float(accuracy),
            'oob_score': float(rf_model.oob_score_),
            'num_classes': len(le.classes_),
            'classes': list(le.classes_),
            'feature_dim': int(X.shape[1]),
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION,
            'n_estimators': 800
        }, f, indent=2)
    
    with open(OUTPUT_DIR / 'label_mappings.json', 'w') as f:
        json.dump({
            'id2label': {str(i): cls for i, cls in enumerate(le.classes_)},
            'label2id': {cls: i for i, cls in enumerate(le.classes_)}
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"OOB Score: {rf_model.oob_score_:.1%}")
    print(f"{'='*60}")
    
    return accuracy

if __name__ == "__main__":
    main()
