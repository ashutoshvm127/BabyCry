#!/usr/bin/env python3
"""
Combined Baby Diagnostic Model Training
- Cry Type Classification (8 classes)
- Pulmonary Sound Classification (7 classes)
- Disease Mapping based on findings
"""

import os
import json
import pickle
import numpy as np
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
CRY_DATA_DIR = Path("data_baby_respiratory")
PULMONARY_DATA_DIR = Path("data_baby_pulmonary")
OUTPUT_DIR = Path("combined_diagnostic_model")
SAMPLE_RATE = 16000
MAX_DURATION = 5

# Disease mapping based on lung sounds
PULMONARY_DISEASE_MAP = {
    "fine_crackle": {
        "diseases": ["Pneumonia", "ARDS", "Pulmonary Edema", "Early Heart Failure"],
        "severity": "high",
        "action": "Immediate medical evaluation recommended"
    },
    "coarse_crackle": {
        "diseases": ["Bronchitis", "Aspiration", "Mucus accumulation"],
        "severity": "medium",
        "action": "Monitor closely, consider pediatric consultation"
    },
    "wheeze": {
        "diseases": ["Bronchiolitis", "Asthma", "Airway obstruction", "Foreign body"],
        "severity": "high",
        "action": "Check airway, consider bronchodilator"
    },
    "rhonchi": {
        "diseases": ["Bronchitis", "Secretion buildup", "Upper respiratory infection"],
        "severity": "medium",
        "action": "Suction if needed, monitor breathing"
    },
    "stridor": {
        "diseases": ["Croup", "Epiglottitis", "Upper airway obstruction", "Laryngomalacia"],
        "severity": "critical",
        "action": "URGENT: Evaluate airway immediately"
    },
    "mixed": {
        "diseases": ["Multiple pathology", "Severe infection", "Complex respiratory condition"],
        "severity": "high",
        "action": "Comprehensive evaluation needed"
    },
    "normal_breathing": {
        "diseases": [],
        "severity": "normal",
        "action": "No respiratory concerns detected"
    }
}

# Cry type clinical mapping
CRY_CLINICAL_MAP = {
    "pain_cry": {
        "indicators": ["Possible sepsis", "NEC", "Acute illness", "Injury"],
        "severity": "high",
        "action": "Investigate source of pain, vital signs check"
    },
    "distress_cry": {
        "indicators": ["Respiratory distress", "Hypoxia", "Acute discomfort"],
        "severity": "high",
        "action": "Check oxygen saturation, breathing pattern"
    },
    "hungry_cry": {
        "indicators": ["Normal feeding cue"],
        "severity": "normal",
        "action": "Feed baby"
    },
    "tired_cry": {
        "indicators": ["Normal fatigue"],
        "severity": "normal",
        "action": "Rest and comfort"
    },
    "sleepy_cry": {
        "indicators": ["Normal sleep cue"],
        "severity": "normal",
        "action": "Help baby sleep"
    },
    "discomfort_cry": {
        "indicators": ["Wet diaper", "Temperature", "Position"],
        "severity": "low",
        "action": "Check comfort factors"
    },
    "cold_cry": {
        "indicators": ["Temperature regulation issue"],
        "severity": "low",
        "action": "Adjust environment temperature"
    },
    "normal_cry": {
        "indicators": ["Normal vocalization"],
        "severity": "normal",
        "action": "Normal baby sounds"
    }
}


def extract_features(audio_path: str) -> np.ndarray:
    """Extract comprehensive audio features."""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        if len(audio) < sr // 2:
            audio = np.pad(audio, (0, sr // 2 - len(audio)))
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        features = []
        
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
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spec_cent))
        features.append(np.std(spec_cent))
        
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend(np.mean(spec_contrast, axis=1))
        
        # Temporal features
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Pitch
        try:
            pitches, _ = librosa.piptrack(y=audio, sr=sr)
            pitch_vals = pitches[pitches > 0]
            features.append(np.mean(pitch_vals) if len(pitch_vals) > 0 else 0)
            features.append(np.std(pitch_vals) if len(pitch_vals) > 0 else 0)
        except:
            features.extend([0, 0])
        
        return np.array(features)
    except Exception as e:
        return None


def load_dataset(data_dir: Path, task_name: str):
    """Load dataset from directory."""
    print(f"\nLoading {task_name} dataset from {data_dir}...")
    
    X, y = [], []
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    label_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    idx_to_label = {i: d.name for i, d in enumerate(class_dirs)}
    
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
    
    return np.array(X), np.array(y), label_to_idx, idx_to_label


def train_model(X, y, task_name, idx_to_label):
    """Train a balanced classifier."""
    print(f"\nTraining {task_name} model...")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    
    # Show distribution
    for idx, count in sorted(Counter(y).items()):
        print(f"    {idx_to_label[idx]}: {count}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest with balanced class weights
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  {task_name} Test Accuracy: {accuracy*100:.1f}%")
    print(classification_report(y_test, y_pred, 
          target_names=[idx_to_label[i] for i in range(len(idx_to_label))]))
    
    return model, scaler, accuracy


def main():
    print("=" * 70)
    print("COMBINED BABY DIAGNOSTIC MODEL TRAINING")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ===== Train Cry Classifier =====
    X_cry, y_cry, cry_l2i, cry_i2l = load_dataset(CRY_DATA_DIR, "Baby Cry")
    cry_model, cry_scaler, cry_acc = train_model(X_cry, y_cry, "Cry", cry_i2l)
    
    # ===== Train Pulmonary Classifier =====
    X_pulm, y_pulm, pulm_l2i, pulm_i2l = load_dataset(PULMONARY_DATA_DIR, "Pulmonary")
    pulm_model, pulm_scaler, pulm_acc = train_model(X_pulm, y_pulm, "Pulmonary", pulm_i2l)
    
    # ===== Save Models =====
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    
    # Save cry model
    cry_data = {
        'model': cry_model,
        'scaler': cry_scaler,
        'idx_to_label': cry_i2l,
        'label_to_idx': cry_l2i,
        'clinical_map': CRY_CLINICAL_MAP,
        'accuracy': cry_acc
    }
    with open(OUTPUT_DIR / "cry_model.pkl", "wb") as f:
        pickle.dump(cry_data, f)
    
    # Save pulmonary model
    pulm_data = {
        'model': pulm_model,
        'scaler': pulm_scaler,
        'idx_to_label': pulm_i2l,
        'label_to_idx': pulm_l2i,
        'disease_map': PULMONARY_DISEASE_MAP,
        'accuracy': pulm_acc
    }
    with open(OUTPUT_DIR / "pulmonary_model.pkl", "wb") as f:
        pickle.dump(pulm_data, f)
    
    # Save combined config
    config = {
        "version": "2.0",
        "cry_model": {
            "accuracy": float(cry_acc),
            "classes": list(cry_l2i.keys()),
            "num_classes": len(cry_l2i)
        },
        "pulmonary_model": {
            "accuracy": float(pulm_acc),
            "classes": list(pulm_l2i.keys()),
            "num_classes": len(pulm_l2i),
            "disease_mapping": True
        },
        "diseases_detected": [
            "Pneumonia", "ARDS", "Bronchiolitis", "Asthma",
            "Croup", "Bronchitis", "Aspiration", "Sepsis indicators"
        ]
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save label mappings
    with open(OUTPUT_DIR / "label_mappings.json", "w") as f:
        json.dump({
            "cry": {
                "label_to_idx": cry_l2i,
                "idx_to_label": {str(k): v for k, v in cry_i2l.items()}
            },
            "pulmonary": {
                "label_to_idx": pulm_l2i,
                "idx_to_label": {str(k): v for k, v in pulm_i2l.items()}
            }
        }, f, indent=2)
    
    print(f"\nModels saved to {OUTPUT_DIR}/")
    print(f"  - cry_model.pkl (Accuracy: {cry_acc*100:.1f}%)")
    print(f"  - pulmonary_model.pkl (Accuracy: {pulm_acc*100:.1f}%)")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nThe system can now detect:")
    print("  [CRY] pain, distress, hungry, tired, sleepy, discomfort, cold, normal")
    print("  [LUNG] crackles, wheeze, rhonchi, stridor → mapped to diseases")
    print("\nDisease mapping includes:")
    for lung_sound, info in PULMONARY_DISEASE_MAP.items():
        if info["diseases"]:
            print(f"  {lung_sound}: {', '.join(info['diseases'][:3])}")


if __name__ == "__main__":
    main()
