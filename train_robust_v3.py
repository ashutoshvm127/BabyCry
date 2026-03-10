"""
Train robust baby cry classifier with speaker-to-microphone degradation simulation.
"""

import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, filtfilt
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\projects\cry analysuis\data_baby_respiratory"
OUTPUT_DIR = r"D:\projects\cry analysuis\rf_baby_cry_model"

def add_noise(audio, factor=0.01):
    return audio + factor * np.random.randn(len(audio))

def lowpass(audio, sr, cutoff=3500):
    nyq = 0.5 * sr
    b, a = butter(4, min(cutoff/nyq, 0.99), btype='low')
    return filtfilt(b, a, audio)

def speaker_mic_sim(audio, sr):
    """Simulate speaker -> microphone recording"""
    audio = lowpass(audio, sr, 4000)
    audio = add_noise(audio, 0.015)
    # Simple reverb
    delay = int(sr * 0.03)
    out = np.zeros(len(audio) + delay)
    out[:len(audio)] = audio
    out[delay:delay+len(audio)] += audio * 0.25
    return out[:len(audio)]

def extract_features(audio, sr):
    features = []
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.max(mfcc, axis=1))
    features.extend(np.min(mfcc, axis=1))
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.extend([np.mean(sc), np.std(sc)])
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.extend([np.mean(sb), np.std(sb)])
    sr_f = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.extend([np.mean(sr_f), np.std(sr_f)])
    scon = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend(np.mean(scon, axis=1))
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.extend([np.mean(zcr), np.std(zcr)])
    rms = librosa.feature.rms(y=audio)
    features.extend([np.mean(rms), np.std(rms)])
    try:
        pitches, mags = librosa.piptrack(y=audio, sr=sr)
        p = pitches[pitches > 0]
        features.extend([np.mean(p) if len(p) else 0, np.std(p) if len(p) else 0])
    except:
        features.extend([0, 0])
    return np.array(features)

def load_data():
    X, y = [], []
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    label_to_idx = {l: i for i, l in enumerate(classes)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    
    counts = {c: len([f for f in os.listdir(os.path.join(DATA_DIR, c)) if f.endswith(('.wav','.mp3'))]) for c in classes}
    print(f"Classes: {counts}")
    max_c = max(counts.values())
    
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.wav','.mp3','.flac'))]
        n_aug = min(8, max(2, int(max_c / counts[cls])))
        n = 0
        for f in files:
            try:
                audio, sr = librosa.load(os.path.join(cls_dir, f), sr=16000, duration=5)
                if len(audio) < sr: continue
                
                # Original
                feat = extract_features(audio, sr)
                if not np.any(np.isnan(feat)):
                    X.append(feat); y.append(label_to_idx[cls]); n += 1
                
                # Augmentations for minority classes
                for i in range(n_aug - 1):
                    if i % 2 == 0:
                        aug = speaker_mic_sim(audio, sr)
                    else:
                        aug = add_noise(audio, 0.01 * (i+1))
                    feat = extract_features(aug, sr)
                    if not np.any(np.isnan(feat)):
                        X.append(feat); y.append(label_to_idx[cls]); n += 1
            except: pass
        print(f"  {cls}: {n}")
    
    return np.array(X), np.array(y), label_to_idx, idx_to_label

def train(X, y, l2i, i2l):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    print("\nSMOTE...")
    X_tr_b, y_tr_b = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)
    print(f"Samples: {len(X_tr_b)}")
    
    print("\nTraining...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=30, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    rf.fit(X_tr_b, y_tr_b)
    
    acc = rf.score(X_te_s, y_te)
    print(f"\nTest accuracy: {acc:.1%}")
    
    y_pred = rf.predict(X_te_s)
    for idx, label in i2l.items():
        mask = y_te == idx
        if mask.sum() > 0:
            print(f"  {label}: {(y_pred[mask]==y_te[mask]).mean():.1%}")
    
    return rf, scaler, acc

print("="*50)
print("TRAINING WITH SPEAKER-MIC SIMULATION")
print("="*50)

X, y, l2i, i2l = load_data()
model, scaler, acc = train(X, y, l2i, i2l)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'label_to_idx': l2i, 'idx_to_label': i2l}, f)
with open(os.path.join(OUTPUT_DIR, "label_mappings.json"), 'w') as f:
    json.dump({'label_to_idx': l2i, 'idx_to_label': {str(k):v for k,v in i2l.items()}}, f)
with open(os.path.join(OUTPUT_DIR, "config.json"), 'w') as f:
    json.dump({'model_type': 'rf_robust_v3', 'accuracy': float(acc), 'num_features': X.shape[1], 'num_classes': len(l2i)}, f)

print(f"\nSaved! Accuracy: {acc:.1%}")
