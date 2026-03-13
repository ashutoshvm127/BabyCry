#!/usr/bin/env python3
"""
Audio Data Augmentation for Underrepresented Classes
=====================================================
Augments minority classes to at least MIN_TARGET samples using:
- Pitch shifting
- Time stretching
- Adding background noise
- Speed perturbation
- Combined augmentations

Classes to augment (current → target):
  burping_cry:      8  → 200
  belly_pain_cry:  16  → 200
  asphyxia_cry:    20  → 200
  stridor:         29  → 200
  pain_cry:        48  → 200
  pathological_cry:50  → 200
  distress_cry:    80  → 200
  bronchiolitis:  100  → 200
  discomfort_cry: 105  → 200
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
BABY_CRY_DIR = BASE_DIR / "data_baby_respiratory"
BABY_PULMONARY_DIR = BASE_DIR / "data_baby_pulmonary"

MIN_TARGET = 200  # Minimum samples per class
SR = 16000

# Classes that need augmentation and their directories
CLASS_DIRS = {}
for cls_dir in list(BABY_CRY_DIR.iterdir()) + list(BABY_PULMONARY_DIR.iterdir()):
    if cls_dir.is_dir():
        CLASS_DIRS[cls_dir.name] = cls_dir


def load_audio(path, sr=SR):
    """Load audio file safely"""
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True, duration=5.0)
        if len(y) < sr * 0.1 or not np.isfinite(y).all():
            return None
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y
    except Exception:
        return None


def pitch_shift(y, sr=SR):
    """Shift pitch by -2 to +2 semitones"""
    n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y):
    """Time stretch by 0.8x to 1.2x"""
    rate = np.random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(y=y, rate=rate)
    # Ensure same length
    if len(stretched) > len(y):
        stretched = stretched[:len(y)]
    else:
        stretched = np.pad(stretched, (0, max(0, len(y) - len(stretched))))
    return stretched


def add_noise(y):
    """Add Gaussian noise"""
    noise_level = np.random.uniform(0.002, 0.01)
    noise = np.random.normal(0, noise_level, len(y))
    return (y + noise).astype(np.float32)


def speed_perturb(y, sr=SR):
    """Change speed without pitch correction"""
    speed = np.random.uniform(0.9, 1.1)
    y_fast = librosa.resample(y, orig_sr=sr, target_sr=int(sr * speed))
    if len(y_fast) > len(y):
        y_fast = y_fast[:len(y)]
    else:
        y_fast = np.pad(y_fast, (0, max(0, len(y) - len(y_fast))))
    return y_fast


def random_gain(y):
    """Random volume change"""
    gain = np.random.uniform(0.7, 1.3)
    return y * gain


def augment_audio(y, sr=SR):
    """Apply a random combination of augmentations"""
    augmentations = [
        lambda x: pitch_shift(x, sr),
        time_stretch,
        add_noise,
        lambda x: speed_perturb(x, sr),
        random_gain,
    ]
    
    # Apply 1-3 random augmentations
    n_augs = np.random.randint(1, 4)
    chosen = np.random.choice(len(augmentations), size=n_augs, replace=False)
    
    result = y.copy()
    for idx in chosen:
        try:
            result = augmentations[idx](result)
        except Exception:
            continue
    
    # Normalize
    max_val = np.max(np.abs(result))
    if max_val > 0 and np.isfinite(max_val):
        result = result / max_val
    
    return result.astype(np.float32)


def augment_class(cls_name, cls_dir, target_count):
    """Augment a single class to reach target_count"""
    # Get existing audio files
    audio_files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
    current_count = len(audio_files)
    
    if current_count >= target_count:
        print(f"  {cls_name}: {current_count} files (already >= {target_count}, skipping)")
        return 0
    
    needed = target_count - current_count
    print(f"  {cls_name}: {current_count} → {target_count} (generating {needed} augmented files)")
    
    # Load all valid audio
    valid_audio = []
    for f in audio_files:
        y = load_audio(f)
        if y is not None:
            valid_audio.append(y)
    
    if not valid_audio:
        print(f"    [!] No valid audio files found for {cls_name}")
        return 0
    
    generated = 0
    for i in range(needed):
        # Pick a random source
        source = valid_audio[i % len(valid_audio)]
        
        # Apply augmentation
        augmented = augment_audio(source)
        
        # Save
        out_path = cls_dir / f"aug_{i:04d}_{cls_name}.wav"
        sf.write(str(out_path), augmented, SR)
        generated += 1
    
    return generated


def main():
    print("=" * 60)
    print("AUGMENTING MINORITY CLASSES")
    print("=" * 60)
    
    total_generated = 0
    
    for cls_name, cls_dir in sorted(CLASS_DIRS.items()):
        audio_files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
        count = len(audio_files)
        
        if count < MIN_TARGET:
            generated = augment_class(cls_name, cls_dir, MIN_TARGET)
            total_generated += generated
    
    print(f"\n{'=' * 60}")
    print(f"DONE: Generated {total_generated} augmented files")
    print(f"{'=' * 60}")
    
    # Print final class distribution
    print("\nFinal class distribution:")
    for cls_name, cls_dir in sorted(CLASS_DIRS.items()):
        audio_files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
        print(f"  {cls_name:25s} {len(audio_files):5d}")


if __name__ == "__main__":
    main()
