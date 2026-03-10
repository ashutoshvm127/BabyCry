#!/usr/bin/env python3
"""
Train a balanced Random Forest model for baby cry classification.
Uses class weights and balanced sampling to handle imbalanced data.
"""

import os
import json
import pickle
import numpy as np
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data_baby_respiratory")
OUTPUT_DIR = Path("rf_baby_cry_model")
SAMPLE_RATE = 16000
MAX_DURATION = 5  # seconds

def extract_features(audio_path: str) -> np.ndarray:
    """Extract comprehensive audio features for classification."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        # Ensure minimum length
        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)))
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        features = []
        
        # MFCCs (40 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1))
        features.extend(np.min(mfcc, axis=1))
        
        # Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))
        
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel, ref=np.max)
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
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Pitch/F0 estimation
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            features.append(pitch_mean)
            features.append(pitch_std)
        except:
            features.extend([0, 0])
        
        return np.array(features)
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_dataset():
    """Load all audio files and extract features."""
    print("Loading dataset...")
    
    X = []
    y = []
    labels = []
    
    # Get all class directories
    class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    label_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    idx_to_label = {i: d.name for i, d in enumerate(class_dirs)}
    
    print(f"Classes: {list(label_to_idx.keys())}")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = label_to_idx[class_name]
        
        audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3")) + list(class_dir.glob("*.ogg"))
        print(f"  {class_name}: {len(audio_files)} files")
        
        for audio_path in tqdm(audio_files, desc=f"  {class_name}", leave=False):
            features = extract_features(str(audio_path))
            if features is not None:
                X.append(features)
                y.append(class_idx)
                labels.append(class_name)
    
    return np.array(X), np.array(y), label_to_idx, idx_to_label

def main():
    print("=" * 60)
    print("BALANCED BABY CRY CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Load data
    X, y, label_to_idx, idx_to_label = load_dataset()
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Show class distribution
    print("\nClass distribution:")
    class_counts = Counter(y)
    for idx, count in sorted(class_counts.items()):
        print(f"  {idx_to_label[idx]}: {count}")
    
    # Calculate class weights for balancing
    total = len(y)
    n_classes = len(label_to_idx)
    class_weights = {}
    for idx, count in class_counts.items():
        # Inverse frequency weighting
        class_weights[idx] = total / (n_classes * count)
    
    print("\nClass weights (for balancing):")
    for idx, weight in sorted(class_weights.items()):
        print(f"  {idx_to_label[idx]}: {weight:.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with class balancing
    print("\nTraining balanced Random Forest...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # KEY: balanced class weights
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[idx_to_label[i] for i in range(n_classes)]))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(n_classes):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_test[class_mask]).mean()
            print(f"  {idx_to_label[i]}: {class_acc*100:.1f}%")
    
    # Save model
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save as dict with model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'idx_to_label': idx_to_label,
        'label_to_idx': label_to_idx,
    }
    
    with open(OUTPUT_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # Save label mappings
    with open(OUTPUT_DIR / "label_mappings.json", "w") as f:
        json.dump({
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_label.items()}
        }, f, indent=2)
    
    # Save config
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump({
            "model_type": "rf_balanced",
            "accuracy": float(accuracy),
            "num_features": X.shape[1],
            "num_classes": n_classes,
            "class_balanced": True,
            "n_estimators": 500,
        }, f, indent=2)
    
    print(f"\nModel saved to {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
