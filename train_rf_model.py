#!/usr/bin/env python3
"""
Train a simpler audio classifier using MFCCs + Random Forest.
Better for imbalanced data with small datasets.
"""

import os
import json
import pickle
import numpy as np
import librosa
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# Config
DATA_DIR = Path("D:/projects/cry analysuis/data_baby_respiratory")
OUTPUT_DIR = Path("D:/projects/cry analysuis/rf_baby_cry_model")
SAMPLE_RATE = 16000
N_MFCC = 40

LABELS = [
    "cold_cry", "discomfort_cry", "distress_cry", "hungry_cry",
    "normal_cry", "pain_cry", "sleepy_cry", "tired_cry"
]
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for idx, label in enumerate(LABELS)}


def extract_features(audio_path: str) -> np.ndarray:
    """Extract MFCC features from audio file."""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Ensure minimum length
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Compute statistics across time
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.min(mfccs, axis=1),
            np.max(mfccs, axis=1),
        ])
        
        # Add other features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms = np.mean(librosa.feature.rms(y=audio))
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0
            pitch_std = np.std(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0
        else:
            pitch_mean, pitch_std = 0, 0
        
        features = np.concatenate([
            features,
            [spectral_centroid, spectral_bandwidth, spectral_rolloff, 
             zero_crossing_rate, rms, pitch_mean, pitch_std]
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def load_dataset():
    """Load all audio files and extract features."""
    print("\n" + "=" * 60)
    print("LOADING AND EXTRACTING FEATURES")
    print("=" * 60)
    
    X = []
    y = []
    
    for label_name in LABELS:
        label_dir = DATA_DIR / label_name
        if not label_dir.exists():
            continue
            
        files = list(label_dir.glob("*.wav")) + list(label_dir.glob("*.mp3"))
        print(f"\n{label_name}: {len(files)} files")
        
        for f in tqdm(files, desc=f"  {label_name}"):
            features = extract_features(str(f))
            if features is not None:
                X.append(features)
                y.append(label2id[label_name])
    
    return np.array(X), np.array(y)


def train():
    """Train the Random Forest classifier."""
    print("\n" + "=" * 60)
    print("BABY CRY CLASSIFICATION - RANDOM FOREST")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    X, y = load_dataset()
    
    print(f"\n\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Show class distribution
    print("\nClass distribution:")
    for label, count in sorted(Counter(y).items()):
        print(f"  {id2label[label]}: {count}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")
    
    # Apply SMOTE to oversample minority classes
    print("\n[*] Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {len(X_train_balanced)} samples")
    print("Class distribution after SMOTE:")
    for label, count in sorted(Counter(y_train_balanced).items()):
        print(f"  {id2label[label]}: {count}")
    
    # Train Random Forest
    print("\n[*] Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    y_pred = clf.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=LABELS))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_val)
    print(f"\n\nValidation Accuracy: {accuracy*100:.1f}%")
    
    # Save model
    print("\n[*] Saving model...")
    with open(OUTPUT_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    
    # Save label mapping
    label_mapping = {"label2id": label2id, "id2label": id2label}
    with open(OUTPUT_DIR / "label_mappings.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save feature info
    config = {
        "n_mfcc": N_MFCC,
        "sample_rate": SAMPLE_RATE,
        "feature_dim": X.shape[1],
        "accuracy": accuracy,
        "model_type": "random_forest"
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] Model saved to {OUTPUT_DIR}")
    print("Restart the server to use the new model.")


if __name__ == "__main__":
    train()
