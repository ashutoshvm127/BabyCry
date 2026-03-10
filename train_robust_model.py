"""
Train a robust baby cry classifier with heavy data augmentation
to handle real-world microphone recordings.
"""

import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\projects\cry analysuis\data_baby_respiratory"
OUTPUT_DIR = r"D:\projects\cry analysuis\rf_baby_cry_model"

def add_noise(audio, noise_factor=0.005):
    """Add random noise to simulate microphone recording"""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_stretch(audio, rate=1.0):
    """Time stretch the audio"""
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps=0):
    """Pitch shift the audio"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_reverb(audio, decay=0.3, delay_samples=1000):
    """Simple reverb simulation"""
    reverb = np.zeros(len(audio) + delay_samples)
    reverb[:len(audio)] = audio
    reverb[delay_samples:delay_samples+len(audio)] += audio * decay
    return reverb[:len(audio)]

def lowpass_filter(audio, sr, cutoff=4000):
    """Simulate speaker/microphone frequency response"""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, audio)

def extract_robust_features(audio, sr):
    """Extract robust features that work with degraded audio"""
    features = []
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # 1. MFCCs with more coefficients
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.max(mfcc, axis=1))
    features.extend(np.min(mfcc, axis=1))
    
    # 2. Delta MFCCs (temporal changes)
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))
    
    # 3. Mel spectrogram statistics
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))
    
    # 4. Spectral features
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
    
    # 5. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # 6. RMS energy
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    # 7. Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    # 8. Tonnetz (tonal centroid features)
    try:
        harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        features.extend(np.mean(tonnetz, axis=1))
    except:
        features.extend([0] * 6)
    
    # 9. Tempo and beat features
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(float(tempo))
    except:
        features.append(0)
    
    return np.array(features)

def augment_audio(audio, sr, augmentation_type):
    """Apply a specific augmentation"""
    if augmentation_type == 'noise_low':
        return add_noise(audio, 0.002)
    elif augmentation_type == 'noise_med':
        return add_noise(audio, 0.01)
    elif augmentation_type == 'noise_high':
        return add_noise(audio, 0.02)
    elif augmentation_type == 'pitch_up':
        return pitch_shift(audio, sr, 2)
    elif augmentation_type == 'pitch_down':
        return pitch_shift(audio, sr, -2)
    elif augmentation_type == 'slow':
        stretched = time_stretch(audio, 0.9)
        # Ensure same length
        if len(stretched) > len(audio):
            return stretched[:len(audio)]
        else:
            return np.pad(stretched, (0, len(audio) - len(stretched)))
    elif augmentation_type == 'fast':
        stretched = time_stretch(audio, 1.1)
        if len(stretched) > len(audio):
            return stretched[:len(audio)]
        else:
            return np.pad(stretched, (0, len(audio) - len(stretched)))
    elif augmentation_type == 'reverb':
        return add_reverb(audio, 0.3) 
    elif augmentation_type == 'lowpass':
        return lowpass_filter(audio, sr, 3000)
    elif augmentation_type == 'combined':
        # Simulate speaker -> microphone recording
        audio = lowpass_filter(audio, sr, 3500)
        audio = add_noise(audio, 0.015)
        audio = add_reverb(audio, 0.2)
        return audio
    return audio

def load_and_augment_data():
    """Load data with augmentation"""
    X = []
    y = []
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    label_to_idx = {label: idx for idx, label in enumerate(sorted(classes))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"Classes: {classes}")
    print(f"Label mapping: {label_to_idx}")
    
    # Count samples per class
    class_counts = {}
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        class_counts[cls] = len(files)
    
    print(f"\nOriginal class distribution: {class_counts}")
    
    # Determine augmentation multipliers based on class size
    max_count = max(class_counts.values())
    augment_per_class = {}
    for cls, count in class_counts.items():
        # More augmentation for minority classes
        augment_per_class[cls] = max(1, int(max_count / count))
    
    print(f"Augmentation multipliers: {augment_per_class}")
    
    # Define augmentations - heavy augmentation for robustness
    augmentations = [
        'original',  # Always include original
        'noise_low',
        'noise_med', 
        'noise_high',
        'pitch_up',
        'pitch_down',
        'lowpass',
        'combined',  # Simulates microphone recording
    ]
    
    total_processed = 0
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        cls_samples = 0
        for file in files:
            file_path = os.path.join(cls_dir, file)
            try:
                audio, sr = librosa.load(file_path, sr=16000, duration=5)
                
                if len(audio) < sr:  # Skip very short files
                    continue
                
                # Apply augmentations based on how many we need for this class
                num_augs = min(augment_per_class[cls], len(augmentations))
                selected_augs = augmentations[:num_augs]
                
                for aug in selected_augs:
                    if aug == 'original':
                        aug_audio = audio
                    else:
                        try:
                            aug_audio = augment_audio(audio, sr, aug)
                        except:
                            continue
                    
                    features = extract_robust_features(aug_audio, sr)
                    if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                        X.append(features)
                        y.append(label_to_idx[cls])
                        cls_samples += 1
                
            except Exception as e:
                continue
        
        print(f"  {cls}: {cls_samples} samples (after augmentation)")
        total_processed += cls_samples
    
    print(f"\nTotal samples: {total_processed}")
    return np.array(X), np.array(y), label_to_idx, idx_to_label

def train_ensemble_model(X, y, label_to_idx, idx_to_label):
    """Train an ensemble model for better robustness"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE-Tomek for better class balance
    print("\nApplying SMOTE-Tomek...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE-Tomek: {len(X_train_balanced)} samples")
    
    # Train ensemble of classifiers
    print("\nTraining ensemble model...")
    
    # Random Forest with more trees
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # SVM with probability
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('svm', svm)
        ],
        voting='soft',  # Use probabilities
        weights=[2, 1, 1]  # Give RF more weight
    )
    
    # Train ensemble
    ensemble.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    train_acc = ensemble.score(X_train_scaled, y_train)
    test_acc = ensemble.score(X_test_scaled, y_test)
    
    print(f"\nTrain accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    y_pred = ensemble.predict(X_test_scaled)
    for idx, label in idx_to_label.items():
        mask = y_test == idx
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            print(f"  {label}: {class_acc:.1%} ({mask.sum()} samples)")
    
    # Cross-validation
    print("\nCross-validation...")
    cv_scores = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
    
    return ensemble, scaler, test_acc

def main():
    print("=" * 60)
    print("ROBUST BABY CRY CLASSIFIER TRAINING")
    print("With heavy data augmentation for real-world robustness")
    print("=" * 60)
    
    # Load and augment data
    print("\n[1/3] Loading and augmenting data...")
    X, y, label_to_idx, idx_to_label = load_and_augment_data()
    
    # Train model
    print("\n[2/3] Training ensemble model...")
    model, scaler, accuracy = train_ensemble_model(X, y, label_to_idx, idx_to_label)
    
    # Save model
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
            'model_type': 'ensemble_robust',
            'accuracy': float(accuracy),
            'num_features': X.shape[1],
            'num_classes': len(label_to_idx),
            'augmentations': ['noise', 'pitch_shift', 'lowpass', 'combined'],
            'ensemble': ['RandomForest', 'GradientBoosting', 'SVM']
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
