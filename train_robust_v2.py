"""
Train a robust baby cry classifier with data augmentation.
Simpler version that runs faster.
"""

import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\projects\cry analysuis\data_baby_respiratory"
OUTPUT_DIR = r"D:\projects\cry analysuis\rf_baby_cry_model"

def add_noise(audio, noise_factor=0.005):
    """Add random noise"""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def extract_features(audio, sr):
    """Extract robust audio features"""
    features = []
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # MFCCs
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

def load_data_with_augmentation():
    """Load data with augmentation for minority classes"""
    X = []
    y = []
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    label_to_idx = {label: idx for idx, label in enumerate(sorted(classes))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"Classes: {list(label_to_idx.keys())}")
    
    # Count samples
    class_counts = {}
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        class_counts[cls] = len(files)
    
    print(f"Original distribution: {class_counts}")
    
    # Calculate augmentation factor
    max_count = max(class_counts.values())
    
    total = 0
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        # Augmentation factor for this class
        aug_factor = min(8, max(1, int(max_count / class_counts[cls])))
        
        cls_samples = 0
        for file in files:
            file_path = os.path.join(cls_dir, file)
            try:
                audio, sr = librosa.load(file_path, sr=16000, duration=5)
                
                if len(audio) < sr:
                    continue
                
                # Original
                features = extract_features(audio, sr)
                if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                    X.append(features)
                    y.append(label_to_idx[cls])
                    cls_samples += 1
                
                # Augmented versions for minority classes
                for i in range(aug_factor - 1):
                    aug_audio = add_noise(audio, 0.005 * (i + 1))
                    features = extract_features(aug_audio, sr)
                    if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                        X.append(features)
                        y.append(label_to_idx[cls])
                        cls_samples += 1
                        
            except Exception as e:
                continue
        
        print(f"  {cls}: {cls_samples} samples")
        total += cls_samples
    
    print(f"Total: {total} samples")
    return np.array(X), np.array(y), label_to_idx, idx_to_label

def train_model(X, y, label_to_idx, idx_to_label):
    """Train model"""
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {len(X_train_balanced)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    train_acc = rf.score(X_train_scaled, y_train)
    test_acc = rf.score(X_test_scaled, y_test)
    
    print(f"\nTrain accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")
    
    # Per-class accuracy
    print("\nPer-class accuracy on test set:")
    y_pred = rf.predict(X_test_scaled)
    for idx, label in idx_to_label.items():
        mask = y_test == idx
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            print(f"  {label}: {class_acc:.1%} ({mask.sum()} samples)")
    
    return rf, scaler, test_acc

def main():
    print("=" * 60)
    print("ROBUST BABY CRY CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading data with augmentation...")
    X, y, label_to_idx, idx_to_label = load_data_with_augmentation()
    
    # Train
    print("\n[2/3] Training model...")
    model, scaler, accuracy = train_model(X, y, label_to_idx, idx_to_label)
    
    # Save
    print("\n[3/3] Saving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f)
    
    with open(os.path.join(OUTPUT_DIR, "label_mappings.json"), 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {str(k): v for k, v in idx_to_label.items()}
        }, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "config.json"), 'w') as f:
        json.dump({
            'model_type': 'random_forest_v2',
            'accuracy': float(accuracy),
            'num_features': X.shape[1],
            'num_classes': len(label_to_idx)
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
