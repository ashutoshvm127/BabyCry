#!/usr/bin/env python3
"""
Full 6-Backbone Ensemble Training for Baby Cry & Health Detection
==================================================================

Trains ALL 6 backbone models with EQUAL weights (1/6 each).
All 20 classes are cry/sound-based, detectable via INMP441 MEMS microphone.

Backbones:
  1. DistilHuBERT  (ntu-spml/distilhubert)                   - 768-dim
  2. HuBERT        (facebook/hubert-base-ls960)               - 768-dim
  3. Wav2Vec2      (facebook/wav2vec2-base)                   - 768-dim
  4. WavLM         (microsoft/wavlm-base)                     - 768-dim
  5. AST           (MIT/ast-finetuned-audioset-10-10-0.4593)  - 768-dim
  6. CLAP          (laion/clap-htsat-unfused)                 - 512-dim

20 Classes (all detectable from cry/sound via INMP441 mic on RPi5):
  Baby Cry (12): hungry, pain, sleepy, discomfort, cold, tired, normal,
                 distress, belly_pain, burping, pathological, asphyxia
  Health   (8):  sepsis_cry, colic_cry, pneumonia_cry, respiratory_distress,
                 normal_breathing, wheezing, stridor, bronchiolitis

Usage:
    python train_6backbone_full.py --epochs 60 --batch_size 32
    python train_6backbone_full.py --skip_extraction   # if embeddings cached
"""

import os
import sys
import json
import argparse
import warnings
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import librosa
import soundfile as sf
from scipy import signal as scipy_signal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
CRY_DIR = BASE_DIR / "data_baby_respiratory"
PULMONARY_DIR = BASE_DIR / "data_baby_pulmonary"
OUTPUT_DIR = BASE_DIR / "trained_classifiers"
RPI_MODEL_DIR = BASE_DIR / "rpi5_standalone" / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
RPI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
MAX_DURATION = 5.0
MAX_SAMPLES = int(SR * MAX_DURATION)

# =============================================================================
# 20 CLASSES — all cry/mic-detectable (no stethoscope-only classes)
# =============================================================================
CRY_CLASSES = [
    'hungry_cry', 'pain_cry', 'sleepy_cry', 'discomfort_cry',
    'cold_cry', 'tired_cry', 'normal_cry', 'distress_cry',
    'belly_pain_cry', 'burping_cry', 'pathological_cry', 'asphyxia_cry',
]
HEALTH_CLASSES = [
    'sepsis_cry', 'colic_cry', 'pneumonia_cry', 'respiratory_distress',
    'normal_breathing', 'wheezing', 'stridor', 'bronchiolitis',
]
ALL_CLASSES = CRY_CLASSES + HEALTH_CLASSES
LABEL_TO_ID = {c: i for i, c in enumerate(ALL_CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)

RISK_LEVELS = {
    'normal_cry': 0, 'hungry_cry': 0, 'sleepy_cry': 0, 'tired_cry': 0,
    'burping_cry': 0, 'normal_breathing': 0,
    'discomfort_cry': 1, 'cold_cry': 1, 'belly_pain_cry': 1,
    'colic_cry': 1, 'wheezing': 1,
    'distress_cry': 2, 'respiratory_distress': 2,
    'pain_cry': 3, 'pathological_cry': 3, 'asphyxia_cry': 3,
    'sepsis_cry': 3, 'pneumonia_cry': 3, 'stridor': 3, 'bronchiolitis': 3,
}

# Where to find data for each class
DATA_MAPPING = {
    'hungry_cry':           CRY_DIR / 'hungry_cry',
    'pain_cry':             CRY_DIR / 'pain_cry',
    'sleepy_cry':           CRY_DIR / 'sleepy_cry',
    'discomfort_cry':       CRY_DIR / 'discomfort_cry',
    'cold_cry':             CRY_DIR / 'cold_cry',
    'tired_cry':            CRY_DIR / 'tired_cry',
    'normal_cry':           CRY_DIR / 'normal_cry',
    'distress_cry':         CRY_DIR / 'distress_cry',
    'belly_pain_cry':       CRY_DIR / 'belly_pain_cry',
    'burping_cry':          CRY_DIR / 'burping_cry',
    'pathological_cry':     CRY_DIR / 'pathological_cry',
    'asphyxia_cry':         CRY_DIR / 'asphyxia_cry',
    'sepsis_cry':           CRY_DIR / 'sepsis_cry',          # generated
    'colic_cry':            CRY_DIR / 'colic_cry',            # generated
    'pneumonia_cry':        PULMONARY_DIR / 'pneumonia_cry',   # generated
    'respiratory_distress': PULMONARY_DIR / 'respiratory_distress',  # generated
    'normal_breathing':     PULMONARY_DIR / 'normal_breathing',
    'wheezing':             PULMONARY_DIR / 'wheeze',          # existing folder
    'stridor':              PULMONARY_DIR / 'stridor',
    'bronchiolitis':        PULMONARY_DIR / 'bronchiolitis',
}

# Cap majority classes to reduce imbalance
CLASS_CAPS = {
    'normal_breathing': 600,
    'normal_cry': 500,
    'hungry_cry': 500,
}

# =============================================================================
# 6 BACKBONE DEFINITIONS
# =============================================================================
BACKBONE_CONFIGS = [
    {
        'name': 'distilhubert',
        'hf_name': 'ntu-spml/distilhubert',
        'dim': 768,
        'type': 'wav2vec2',      # uses Wav2Vec2 API
    },
    {
        'name': 'hubert',
        'hf_name': 'facebook/hubert-base-ls960',
        'dim': 768,
        'type': 'wav2vec2',
    },
    {
        'name': 'wav2vec2',
        'hf_name': 'facebook/wav2vec2-base',
        'dim': 768,
        'type': 'wav2vec2',
    },
    {
        'name': 'wavlm',
        'hf_name': 'microsoft/wavlm-base',
        'dim': 768,
        'type': 'wav2vec2',
    },
    {
        'name': 'ast',
        'hf_name': 'MIT/ast-finetuned-audioset-10-10-0.4593',
        'dim': 768,
        'type': 'ast',           # spectrogram-based
    },
    {
        'name': 'clap',
        'hf_name': 'laion/clap-htsat-unfused',
        'dim': 512,
        'type': 'clap',          # CLAP API
    },
]

HANDCRAFTED_DIM = 85   # 40 MFCC mean + 40 MFCC std + 5 spectral


# #############################################################################
#  SECTION 1: SYNTHETIC DATA GENERATION
# #############################################################################

def load_audio_safe(path, sr=SR, duration=MAX_DURATION):
    """Load audio with error handling, return (y, sr) or None."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
        if len(y) < sr * 0.3 or not np.isfinite(y).all():
            return None
        return y
    except Exception:
        return None


def save_audio(y, path, sr=SR):
    """Save audio to WAV."""
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), y, sr)


def pitch_shift_audio(y, sr=SR, n_steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def add_noise(y, noise_level=0.01):
    noise = np.random.randn(len(y)).astype(np.float32) * noise_level
    return y + noise


def simulate_room_mic(y, sr=SR):
    """Domain adaptation: make stethoscope audio sound more like INMP441 room mic.
    Adds room noise, distance attenuation (low-pass), and gain adjustment."""
    # Low-pass filter to simulate distance (stethoscope is in-contact, mic is distant)
    cutoff = 4000  # Hz — room mic won't capture above ~4kHz well for distant sounds
    nyq = sr / 2
    b, a = scipy_signal.butter(4, cutoff / nyq, btype='low')
    y = scipy_signal.lfilter(b, a, y).astype(np.float32)
    # Attenuate (distance)
    y = y * np.random.uniform(0.3, 0.7)
    # Add room noise
    y = add_noise(y, noise_level=np.random.uniform(0.005, 0.02))
    return y


def generate_sepsis_cry(source_files, output_dir, target_count=250):
    """Generate sepsis cry samples.
    Sepsis cries: weak, high-pitched, short, monotonous."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.wav"))
    if len(existing) >= target_count:
        print(f"    sepsis_cry: {len(existing)} files exist, skipping generation")
        return
    needed = target_count - len(existing)
    print(f"    Generating {needed} sepsis_cry samples...")
    count = 0
    for src_path in np.random.choice(source_files, size=min(needed * 2, len(source_files) * 3), replace=True):
        if count >= needed:
            break
        y = load_audio_safe(src_path)
        if y is None:
            continue
        # High pitch (sepsis → higher fundamental frequency)
        y = pitch_shift_audio(y, n_steps=np.random.uniform(2, 5))
        # Weak / low amplitude
        y = y * np.random.uniform(0.2, 0.5)
        # Shorter duration (truncate)
        max_len = int(SR * np.random.uniform(1.5, 3.5))
        if len(y) > max_len:
            start = np.random.randint(0, len(y) - max_len)
            y = y[start:start + max_len]
        # Pad to standard length
        if len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        # Add slight noise
        y = add_noise(y, noise_level=0.005)
        out_path = output_dir / f"synth_sepsis_{count:04d}.wav"
        save_audio(y, out_path)
        count += 1
    print(f"    Generated {count} sepsis_cry samples")


def generate_colic_cry(source_files, output_dir, target_count=250):
    """Generate colic cry samples.
    Colic cries: intense, prolonged, high-amplitude, inconsolable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.wav"))
    if len(existing) >= target_count:
        print(f"    colic_cry: {len(existing)} files exist, skipping generation")
        return
    needed = target_count - len(existing)
    print(f"    Generating {needed} colic_cry samples...")
    count = 0
    for src_path in np.random.choice(source_files, size=min(needed * 2, len(source_files) * 3), replace=True):
        if count >= needed:
            break
        y = load_audio_safe(src_path)
        if y is None:
            continue
        # Intense / high amplitude
        y = y * np.random.uniform(1.3, 1.8)
        # Slight pitch increase
        y = pitch_shift_audio(y, n_steps=np.random.uniform(0.5, 2.0))
        # Remove silence gaps (colic is continuous)
        mask = np.abs(y) > 0.01
        if mask.any():
            y_active = y[mask]
            if len(y_active) > SR:
                y = y_active
        # Pad/truncate
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        elif len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        y = add_noise(y, noise_level=0.003)
        out_path = output_dir / f"synth_colic_{count:04d}.wav"
        save_audio(y, out_path)
        count += 1
    print(f"    Generated {count} colic_cry samples")


def generate_pneumonia_cry(source_files, output_dir, target_count=250):
    """Generate pneumonia cry samples.
    Pneumonia cries: weak, irregular, interrupted by labored breathing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.wav"))
    if len(existing) >= target_count:
        print(f"    pneumonia_cry: {len(existing)} files exist, skipping generation")
        return
    needed = target_count - len(existing)
    print(f"    Generating {needed} pneumonia_cry samples...")
    count = 0
    for src_path in np.random.choice(source_files, size=min(needed * 2, len(source_files) * 3), replace=True):
        if count >= needed:
            break
        y = load_audio_safe(src_path)
        if y is None:
            continue
        # Weak amplitude
        y = y * np.random.uniform(0.3, 0.6)
        # Insert random silence gaps (labored breathing pauses)
        n_gaps = np.random.randint(2, 5)
        for _ in range(n_gaps):
            gap_start = np.random.randint(0, max(1, len(y) - SR // 4))
            gap_len = int(SR * np.random.uniform(0.1, 0.3))
            gap_end = min(gap_start + gap_len, len(y))
            y[gap_start:gap_end] *= 0.05  # near silence
        # Add slight congestion-like noise
        y = add_noise(y, noise_level=np.random.uniform(0.008, 0.02))
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        elif len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        out_path = output_dir / f"synth_pneumonia_{count:04d}.wav"
        save_audio(y, out_path)
        count += 1
    print(f"    Generated {count} pneumonia_cry samples")


def generate_respiratory_distress(cry_files, breathing_files, output_dir, target_count=250):
    """Generate respiratory distress samples.
    Audible grunting + labored breathing mixed with distress cry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.wav"))
    if len(existing) >= target_count:
        print(f"    respiratory_distress: {len(existing)} files exist, skipping generation")
        return
    needed = target_count - len(existing)
    print(f"    Generating {needed} respiratory_distress samples...")
    count = 0
    max_iter = needed * 3
    for i in range(max_iter):
        if count >= needed:
            break
        cry_path = np.random.choice(cry_files)
        y_cry = load_audio_safe(cry_path)
        if y_cry is None:
            continue
        # Try to mix with breathing/stridor sound
        if len(breathing_files) > 0:
            breath_path = np.random.choice(breathing_files)
            y_breath = load_audio_safe(breath_path)
            if y_breath is not None:
                # Apply room mic simulation to stethoscope breathing
                y_breath = simulate_room_mic(y_breath)
                min_len = min(len(y_cry), len(y_breath))
                y = y_cry[:min_len] * 0.6 + y_breath[:min_len] * 0.4
            else:
                y = y_cry
        else:
            y = y_cry
        # Add low-frequency grunting modulation
        t = np.linspace(0, len(y) / SR, len(y))
        grunt_freq = np.random.uniform(2, 5)  # Hz
        grunt = 0.15 * np.sin(2 * np.pi * grunt_freq * t).astype(np.float32)
        y = y + grunt
        # Pad/truncate
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        elif len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        y = add_noise(y, noise_level=0.005)
        out_path = output_dir / f"synth_respdist_{count:04d}.wav"
        save_audio(y, out_path)
        count += 1
    print(f"    Generated {count} respiratory_distress samples")


def apply_domain_adaptation(source_dir, target_count=None):
    """Apply room-mic simulation to stethoscope-sourced audio files.
    Creates adapted copies alongside originals."""
    files = list(source_dir.glob("*.wav")) + list(source_dir.glob("*.mp3"))
    adapted = [f for f in files if f.name.startswith("adapted_")]
    originals = [f for f in files if not f.name.startswith(("adapted_", "synth_", "aug_"))]
    if adapted:
        print(f"    {source_dir.name}: {len(adapted)} adapted files already exist")
        return
    count = 0
    for fp in originals:
        y = load_audio_safe(fp)
        if y is None:
            continue
        y_adapted = simulate_room_mic(y)
        out_path = source_dir / f"adapted_{fp.stem}.wav"
        save_audio(y_adapted, out_path)
        count += 1
    print(f"    {source_dir.name}: created {count} domain-adapted copies")


def prepare_all_data():
    """Generate synthetic data for new classes + domain-adapt stethoscope data."""
    print("\n" + "=" * 60)
    print("STEP 1: PREPARING CRY-FOCUSED DATASET")
    print("=" * 60)

    # Source files for synthetic generation
    pathological_files = sorted((CRY_DIR / 'pathological_cry').glob("*.wav"))
    asphyxia_files = sorted((CRY_DIR / 'asphyxia_cry').glob("*.wav"))
    belly_pain_files = sorted((CRY_DIR / 'belly_pain_cry').glob("*.wav"))
    distress_files = sorted((CRY_DIR / 'distress_cry').glob("*.wav"))
    stridor_files = sorted((PULMONARY_DIR / 'stridor').glob("*.wav"))

    sepsis_sources = [str(f) for f in pathological_files + asphyxia_files]
    colic_sources = [str(f) for f in belly_pain_files + distress_files]
    pneumonia_sources = [str(f) for f in pathological_files + distress_files]
    resp_dist_cry = [str(f) for f in distress_files]
    resp_dist_breath = [str(f) for f in stridor_files]

    print("\n  Generating synthetic cry-disease classes...")
    generate_sepsis_cry(sepsis_sources, CRY_DIR / 'sepsis_cry', target_count=250)
    generate_colic_cry(colic_sources, CRY_DIR / 'colic_cry', target_count=250)
    generate_pneumonia_cry(pneumonia_sources, PULMONARY_DIR / 'pneumonia_cry', target_count=250)
    generate_respiratory_distress(
        resp_dist_cry, resp_dist_breath,
        PULMONARY_DIR / 'respiratory_distress', target_count=250
    )

    # Domain adaptation for stethoscope-sourced breathing data
    print("\n  Applying domain adaptation (stethoscope → room mic)...")
    for cls in ['normal_breathing', 'wheeze', 'stridor', 'bronchiolitis']:
        cls_dir = PULMONARY_DIR / cls
        if cls_dir.exists():
            apply_domain_adaptation(cls_dir)

    print("\n  Data preparation complete!")


# #############################################################################
#  SECTION 2: COLLECT ALL DATA FILES
# #############################################################################

def collect_dataset():
    """Collect all audio file paths and labels for the 20 classes."""
    file_paths = []
    labels = []
    class_counts = Counter()

    for cls_name, cls_dir in DATA_MAPPING.items():
        if not cls_dir.exists():
            print(f"  WARNING: {cls_name} directory not found: {cls_dir}")
            continue
        files = sorted(cls_dir.glob("*.wav")) + sorted(cls_dir.glob("*.mp3"))
        cap = CLASS_CAPS.get(cls_name, len(files))
        if len(files) > cap:
            np.random.seed(42)
            indices = np.random.choice(len(files), cap, replace=False)
            files = [files[i] for i in sorted(indices)]
        for fp in files:
            file_paths.append(str(fp))
            labels.append(cls_name)
        class_counts[cls_name] = len(files)

    print("\n  Dataset Summary:")
    total = 0
    for cls_name in ALL_CLASSES:
        cnt = class_counts.get(cls_name, 0)
        total += cnt
        marker = " [NEW]" if cls_name in ['sepsis_cry', 'colic_cry', 'pneumonia_cry', 'respiratory_distress'] else ""
        print(f"    {cls_name:25s}: {cnt:5d} files{marker}")
    print(f"    {'TOTAL':25s}: {total:5d}")
    return file_paths, labels


# #############################################################################
#  SECTION 3: HAND-CRAFTED FEATURE EXTRACTION
# #############################################################################

def extract_handcrafted(y, sr=SR):
    """40 MFCC mean + 40 MFCC std + 5 spectral = 85-dim."""
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sb = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        sr_ = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        feat = np.concatenate([mfcc_mean, mfcc_std, [sc, sb, sr_, zcr, rms]])
        if not np.isfinite(feat).all():
            return np.zeros(HANDCRAFTED_DIM, dtype=np.float32)
        return feat.astype(np.float32)
    except Exception:
        return np.zeros(HANDCRAFTED_DIM, dtype=np.float32)


def extract_all_handcrafted(file_paths, cache_path):
    """Extract hand-crafted features for all files (with caching)."""
    if cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=True)
        cached_paths = list(data['file_paths'])
        if cached_paths == file_paths:
            print("  Hand-crafted features loaded from cache")
            return data['features'], data['feat_mean'], data['feat_std']
        else:
            print("  Cache mismatch, re-extracting hand-crafted features...")

    print(f"  Extracting hand-crafted features for {len(file_paths)} files...")
    features = []
    for fp in tqdm(file_paths, desc="  Hand-crafted"):
        y = load_audio_safe(fp)
        if y is None:
            y = np.zeros(SR, dtype=np.float32)
        features.append(extract_handcrafted(y))
    features = np.array(features, dtype=np.float32)

    # Normalize
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    features = (features - feat_mean) / feat_std

    np.savez(str(cache_path),
             features=features, file_paths=file_paths,
             feat_mean=feat_mean, feat_std=feat_std)
    return features, feat_mean, feat_std


# #############################################################################
#  SECTION 4: BACKBONE EMBEDDING EXTRACTION
# #############################################################################

class LibrosaASTProcessor:
    """Custom AST feature extractor using librosa (no torchaudio needed).
    Replicates ASTFeatureExtractor: 128 mel-bin fbank, 1024 frames, normalized."""
    def __init__(self, sr=16000, n_mels=128, target_length=1024,
                 fft_size=400, hop_size=160, mean=-4.2677, std=4.5689):
        self.sr = sr
        self.n_mels = n_mels
        self.target_length = target_length
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.mean = mean
        self.std = std

    def __call__(self, y, sampling_rate=16000, return_tensors="pt", **kwargs):
        import torch
        mel = librosa.feature.melspectrogram(
            y=y, sr=sampling_rate, n_fft=self.fft_size,
            hop_length=self.hop_size, n_mels=self.n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # (128, T)
        log_mel = log_mel.T  # (T, 128)
        # Pad or truncate to target_length
        if log_mel.shape[0] < self.target_length:
            pad = np.zeros((self.target_length - log_mel.shape[0], self.n_mels), dtype=np.float32)
            log_mel = np.concatenate([log_mel, pad], axis=0)
        else:
            log_mel = log_mel[:self.target_length]
        # Normalize
        log_mel = (log_mel - self.mean) / (self.std * 2)
        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        return {"input_values": tensor}


def load_backbone(config):
    """Load a HuggingFace backbone model + processor."""
    name = config['name']
    hf_name = config['hf_name']
    btype = config['type']

    if btype == 'wav2vec2':
        from transformers import AutoModel, AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(hf_name)
        model = AutoModel.from_pretrained(hf_name)
    elif btype == 'ast':
        from transformers import ASTModel
        processor = LibrosaASTProcessor()  # custom: no torchaudio needed
        model = ASTModel.from_pretrained(hf_name)
    elif btype == 'clap':
        from transformers import ClapModel, ClapProcessor
        full_model = ClapModel.from_pretrained(hf_name)
        model = full_model  # we'll use get_audio_features
        processor = ClapProcessor.from_pretrained(hf_name)
    else:
        raise ValueError(f"Unknown backbone type: {btype}")

    model.eval().to(device)
    return model, processor


def extract_backbone_embedding(model, processor, y, sr, config):
    """Extract a single embedding from one audio waveform on GPU."""
    btype = config['type']

    if btype in ('wav2vec2',):
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    elif btype == 'ast':
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    elif btype == 'clap':
        inputs = processor(audios=y, return_tensors="pt", sampling_rate=48000)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.get_audio_features(**inputs).squeeze(0).cpu().numpy()

    return emb.astype(np.float32)


def extract_all_backbone_embeddings(file_paths, config, cache_dir):
    """Extract embeddings for all files from one backbone (with caching)."""
    cache_path = cache_dir / f"embeddings_v4_{config['name']}.npz"

    if cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=True)
        cached_paths = list(data['file_paths'])
        if cached_paths == file_paths:
            print(f"  [{config['name']}] Loaded {len(cached_paths)} cached embeddings")
            return data['embeddings']
        else:
            print(f"  [{config['name']}] Cache mismatch ({len(cached_paths)} vs {len(file_paths)}), re-extracting...")

    print(f"  [{config['name']}] Loading model: {config['hf_name']}...")
    model, processor = load_backbone(config)

    print(f"  [{config['name']}] Extracting embeddings for {len(file_paths)} files on {device}...")
    embeddings = []
    errors = 0
    for fp in tqdm(file_paths, desc=f"  {config['name']}"):
        y = load_audio_safe(fp)
        if y is None:
            y = np.zeros(SR, dtype=np.float32)
            errors += 1
        try:
            emb = extract_backbone_embedding(model, processor, y, SR, config)
        except Exception as e:
            emb = np.zeros(config['dim'], dtype=np.float32)
            errors += 1
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)
    if errors > 0:
        print(f"  [{config['name']}] {errors} files had errors (used zeros)")

    np.savez(str(cache_path), embeddings=embeddings, file_paths=file_paths)
    print(f"  [{config['name']}] Saved {len(embeddings)} embeddings to cache")

    # Free memory
    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    return embeddings


# #############################################################################
#  SECTION 5: FUSION CLASSIFIER (per backbone)
# #############################################################################

class FusionClassifier(nn.Module):
    """Backbone embedding + hand-crafted features → class prediction."""
    def __init__(self, backbone_dim, handcrafted_dim=HANDCRAFTED_DIM,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        input_dim = backbone_dim + handcrafted_dim
        hidden = 512
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, num_classes)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout * 0.5)
        self.drop3 = nn.Dropout(dropout * 0.5)
        # Residual projection
        self.res_proj = nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()

    def forward(self, x):
        x = self.bn_input(x)
        identity = self.res_proj(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = x + identity  # residual
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        return self.fc_out(x)


# #############################################################################
#  SECTION 6: DATASET + TRAINING
# #############################################################################

class FusionDataset(Dataset):
    def __init__(self, backbone_emb, handcrafted, labels, mixup_alpha=0.0):
        self.backbone_emb = backbone_emb
        self.handcrafted = handcrafted
        self.labels = labels
        self.mixup_alpha = mixup_alpha
        self.training_mode = True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.backbone_emb[idx]
        hc = self.handcrafted[idx]
        label = self.labels[idx]

        if self.training_mode and self.mixup_alpha > 0 and np.random.rand() < 0.3:
            mix_idx = np.random.randint(len(self.labels))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            emb = lam * emb + (1 - lam) * self.backbone_emb[mix_idx]
            hc = lam * hc + (1 - lam) * self.handcrafted[mix_idx]
            # Use primary label (hard label for simplicity)

        x = np.concatenate([emb, hc])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def compute_class_weights(labels):
    """Sqrt-inverse frequency weighting."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(NUM_CLASSES)
    for cls_id, cnt in counts.items():
        weights[cls_id] = np.sqrt(total / (NUM_CLASSES * cnt))
    # Normalize
    weights = weights / weights.mean()
    return weights


def train_one_backbone(backbone_config, backbone_embeddings, handcrafted_features,
                       labels_int, train_idx, val_idx, args):
    """Train a FusionClassifier for one backbone. Returns (model, val_acc, report)."""
    name = backbone_config['name']
    dim = backbone_config['dim']
    print(f"\n  --- Training classifier for [{name}] (dim={dim}+{HANDCRAFTED_DIM}) ---")

    train_emb = backbone_embeddings[train_idx]
    val_emb = backbone_embeddings[val_idx]
    train_hc = handcrafted_features[train_idx]
    val_hc = handcrafted_features[val_idx]
    train_labels = labels_int[train_idx]
    val_labels = labels_int[val_idx]

    train_ds = FusionDataset(train_emb, train_hc, train_labels, mixup_alpha=args.mixup_alpha)
    val_ds = FusionDataset(val_emb, val_hc, val_labels)
    val_ds.training_mode = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    model = FusionClassifier(dim, HANDCRAFTED_DIM, NUM_CLASSES, dropout=0.3).to(device)
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           steps_per_epoch=len(train_loader),
                           epochs=args.epochs, pct_start=0.1)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_ds.training_mode = True
        train_loss = 0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_true = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.cpu().numpy())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or val_acc >= best_val_acc:
            print(f"    [{name}] Epoch {epoch:3d}: train_acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}  "
                  f"patience={patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"    [{name}] Early stopping at epoch {epoch}")
            break

    # Final evaluation with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(y.numpy())

    present_labels = sorted(set(all_true) | set(all_preds))
    present_names = [ID_TO_LABEL[i] for i in present_labels]
    report = classification_report(all_true, all_preds,
                                   labels=present_labels,
                                   target_names=present_names,
                                   zero_division=0)
    print(f"\n    [{name}] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"    [{name}] Classification Report:\n{report}")

    return model, best_val_acc, best_state, report


# #############################################################################
#  SECTION 7: ENSEMBLE EVALUATION
# #############################################################################

def evaluate_ensemble(all_models, all_configs, backbone_embeddings_dict,
                      handcrafted_features, labels_int, val_idx):
    """Evaluate the 6-model ensemble with equal weights."""
    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION (Equal Weight 1/6)")
    print("=" * 60)

    val_hc = handcrafted_features[val_idx]
    val_labels = labels_int[val_idx]

    all_probs = []
    weight = 1.0 / len(all_models)

    for config, model in zip(all_configs, all_models):
        name = config['name']
        val_emb = backbone_embeddings_dict[name][val_idx]

        model.eval()
        preds_list = []
        batch_size = 256
        for start in range(0, len(val_emb), batch_size):
            end = min(start + batch_size, len(val_emb))
            x_emb = torch.tensor(val_emb[start:end], dtype=torch.float32)
            x_hc = torch.tensor(val_hc[start:end], dtype=torch.float32)
            x = torch.cat([x_emb, x_hc], dim=1).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            preds_list.append(probs)
        backbone_probs = np.concatenate(preds_list, axis=0) * weight
        all_probs.append(backbone_probs)

    # Ensemble: average probabilities
    ensemble_probs = np.sum(all_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    correct = (ensemble_preds == val_labels).sum()
    accuracy = correct / len(val_labels)
    print(f"\n  Ensemble Accuracy: {accuracy*100:.2f}%")

    present_labels = sorted(set(val_labels.tolist()) | set(ensemble_preds.tolist()))
    present_names = [ID_TO_LABEL[i] for i in present_labels]
    report = classification_report(val_labels, ensemble_preds,
                                   labels=present_labels,
                                   target_names=present_names,
                                   zero_division=0)
    print(f"\n  Ensemble Classification Report:\n{report}")
    return accuracy, report, ensemble_probs


# #############################################################################
#  SECTION 8: SAVE EVERYTHING
# #############################################################################

def save_ensemble(all_states, all_configs, individual_accs, ensemble_acc,
                  ensemble_report, handcrafted_mean, handcrafted_std):
    """Save all 6 backbone classifiers + ensemble metadata."""
    checkpoint = {
        'model_type': '6backbone_fusion_ensemble',
        'num_backbones': len(all_configs),
        'backbone_weights': 'equal (1/6 each)',
        'classes': ALL_CLASSES,
        'num_classes': NUM_CLASSES,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
        'risk_levels': RISK_LEVELS,
        'handcrafted_dim': HANDCRAFTED_DIM,
        'handcrafted_mean': handcrafted_mean,
        'handcrafted_std': handcrafted_std,
        'backbones': {},
        'metadata': {
            'ensemble_val_acc': float(ensemble_acc),
            'individual_val_accs': {c['name']: float(a) for c, a in zip(all_configs, individual_accs)},
            'trained_at': datetime.now().isoformat(),
            'device': 'INMP441_MEMS_mic',
            'platform': 'RPi5',
            'detection_source': 'baby_cry_and_audible_sounds',
        }
    }

    for config, state, acc in zip(all_configs, all_states, individual_accs):
        checkpoint['backbones'][config['name']] = {
            'hf_name': config['hf_name'],
            'dim': config['dim'],
            'type': config['type'],
            'state_dict': state,
            'val_acc': float(acc),
        }

    # Save to trained_classifiers/
    out_path = OUTPUT_DIR / "6backbone_ensemble.pt"
    torch.save(checkpoint, str(out_path))
    print(f"\n  Saved ensemble checkpoint: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Copy to RPi5 models dir
    rpi_path = RPI_MODEL_DIR / "6backbone_ensemble.pt"
    torch.save(checkpoint, str(rpi_path))
    print(f"  Saved RPi5 model: {rpi_path}")

    # Save classification report
    report_path = OUTPUT_DIR / "ensemble_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"6-Backbone Ensemble Classification Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Ensemble Accuracy: {ensemble_acc*100:.2f}%\n")
        f.write(f"Backbones (all equal 1/6 weight):\n")
        for config, acc in zip(all_configs, individual_accs):
            f.write(f"  {config['name']:15s} ({config['hf_name']}) -> {acc*100:.2f}%\n")
        f.write(f"\nDetection Source: Baby cries + audible sounds via INMP441 MEMS mic\n")
        f.write(f"Target Platform: Raspberry Pi 5\n")
        f.write(f"\n{ensemble_report}\n")
    print(f"  Saved report: {report_path}")

    # Save label mappings
    mappings = {
        'classes': ALL_CLASSES,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': {str(k): v for k, v in ID_TO_LABEL.items()},
        'risk_levels': RISK_LEVELS,
        'cry_classes': CRY_CLASSES,
        'health_classes': HEALTH_CLASSES,
    }
    mappings_path = OUTPUT_DIR / "label_mappings_v2.json"
    with open(mappings_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"  Saved label mappings: {mappings_path}")

    # Save normalization params separately
    norm_path = OUTPUT_DIR / "feature_normalization_v2.npz"
    np.savez(str(norm_path), feat_mean=handcrafted_mean, feat_std=handcrafted_std)
    print(f"  Saved normalization: {norm_path}")


# #############################################################################
#  MAIN
# #############################################################################

def main():
    parser = argparse.ArgumentParser(description="Train 6-backbone ensemble")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--skip_data_prep', action='store_true',
                        help="Skip synthetic data generation")
    parser.add_argument('--skip_extraction', action='store_true',
                        help="Skip embedding extraction (use cache)")
    parser.add_argument('--backbones', nargs='+', default=None,
                        help="Train specific backbones only (e.g. --backbones distilhubert ast)")
    args = parser.parse_args()

    print("=" * 70)
    print("  6-BACKBONE ENSEMBLE TRAINING")
    print("  All 20 classes | cry-focused | INMP441 MEMS mic compatible")
    print("  Equal weights: 1/6 per backbone")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"  Patience: {args.patience}  Mixup: {args.mixup_alpha}")

    # --- Step 1: Prepare data ---
    if not args.skip_data_prep:
        prepare_all_data()
    else:
        print("\n  Skipping data preparation (--skip_data_prep)")

    # --- Step 2: Collect files ---
    print("\n" + "=" * 60)
    print("STEP 2: COLLECTING DATASET")
    print("=" * 60)
    file_paths, labels = collect_dataset()
    labels_int = np.array([LABEL_TO_ID[l] for l in labels])

    # Train/val split (stratified)
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=42,
        stratify=labels_int
    )
    print(f"\n  Train: {len(train_idx)}  Val: {len(val_idx)}")

    # --- Step 3: Extract hand-crafted features ---
    print("\n" + "=" * 60)
    print("STEP 3: HAND-CRAFTED FEATURES")
    print("=" * 60)
    hc_cache = OUTPUT_DIR / "handcrafted_v4.npz"
    handcrafted_features, hc_mean, hc_std = extract_all_handcrafted(file_paths, hc_cache)

    # --- Step 4: Extract backbone embeddings ---
    print("\n" + "=" * 60)
    print("STEP 4: BACKBONE EMBEDDING EXTRACTION")
    print("=" * 60)

    # Filter backbones if specified
    active_configs = BACKBONE_CONFIGS
    if args.backbones:
        active_configs = [c for c in BACKBONE_CONFIGS if c['name'] in args.backbones]
        print(f"  Training only: {[c['name'] for c in active_configs]}")

    backbone_embeddings = {}
    for config in active_configs:
        try:
            backbone_embeddings[config['name']] = extract_all_backbone_embeddings(
                file_paths, config, OUTPUT_DIR
            )
        except Exception as e:
            print(f"  ERROR extracting {config['name']}: {e}")
            print(f"  Skipping {config['name']}")
            continue

    # --- Step 5: Train classifiers ---
    print("\n" + "=" * 60)
    print("STEP 5: TRAINING CLASSIFIERS (one per backbone)")
    print("=" * 60)

    all_models = []
    all_states = []
    individual_accs = []
    trained_configs = []

    for config in active_configs:
        name = config['name']
        if name not in backbone_embeddings:
            print(f"\n  Skipping [{name}] — no embeddings available")
            continue
        emb = backbone_embeddings[name]
        model, val_acc, state, report = train_one_backbone(
            config, emb, handcrafted_features,
            labels_int, train_idx, val_idx, args
        )
        all_models.append(model)
        all_states.append(state)
        individual_accs.append(val_acc)
        trained_configs.append(config)

    if len(all_models) == 0:
        print("\n  ERROR: No backbones trained successfully!")
        return

    # --- Step 6: Ensemble evaluation ---
    ensemble_acc, ensemble_report, _ = evaluate_ensemble(
        all_models, trained_configs, backbone_embeddings,
        handcrafted_features, labels_int, val_idx
    )

    # --- Step 7: Save ---
    print("\n" + "=" * 60)
    print("STEP 7: SAVING MODELS")
    print("=" * 60)
    save_ensemble(all_states, trained_configs, individual_accs,
                  ensemble_acc, ensemble_report, hc_mean, hc_std)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Individual backbone accuracies:")
    for config, acc in zip(trained_configs, individual_accs):
        print(f"    {config['name']:15s}: {acc*100:.2f}%")
    print(f"\n  ENSEMBLE ACCURACY: {ensemble_acc*100:.2f}%")
    print(f"  Backbones: {len(trained_configs)}, Weight: 1/{len(trained_configs)} each")
    print(f"\n  Models saved to:")
    print(f"    {OUTPUT_DIR / '6backbone_ensemble.pt'}")
    print(f"    {RPI_MODEL_DIR / '6backbone_ensemble.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
