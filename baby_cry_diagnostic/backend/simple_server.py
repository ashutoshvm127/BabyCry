#!/usr/bin/env python3
"""
Baby Cry Diagnostic System - Simple Backend Server
Works without heavy dependencies - uses librosa for analysis
"""

import os
import io
import sys
import json
import uuid
import base64
import asyncio
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Try to import optional dependencies
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[!] librosa not installed - using basic analysis")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("[!] soundfile not installed")

try:
    import torch
    from transformers import AutoModel, AutoFeatureExtractor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[!] PyTorch/transformers not installed - using fallback mode")

# Check for pydub (handles many audio formats)
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("[!] pydub not installed - webm conversion may fail")


def load_audio_any_format(content: bytes, filename: str = "audio.webm") -> tuple:
    """
    Load audio from any format using multiple fallback methods.
    Returns (audio_data, sample_rate)
    """
    # Try different methods in order of preference
    errors = []
    
    # Method 1: Try soundfile directly (works for wav, flac, ogg)
    if HAS_SOUNDFILE:
        try:
            audio_data, sr = sf.read(io.BytesIO(content))
            return audio_data.astype(np.float32), sr
        except Exception as e:
            errors.append(f"soundfile: {e}")
    
    # Method 2: Try librosa directly
    if HAS_LIBROSA:
        try:
            audio_data, sr = librosa.load(io.BytesIO(content), sr=16000)
            return audio_data.astype(np.float32), sr
        except Exception as e:
            errors.append(f"librosa: {e}")
    
    # Method 3: Try pydub (handles webm, mp3, etc - needs ffmpeg)
    if HAS_PYDUB:
        try:
            # Determine format from filename
            ext = filename.split('.')[-1].lower() if '.' in filename else 'webm'
            audio_segment = AudioSegment.from_file(io.BytesIO(content), format=ext)
            # Convert to mono 16kHz
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
            # Get raw samples
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            # Normalize
            samples = samples / (2 ** (audio_segment.sample_width * 8 - 1))
            return samples, 16000
        except Exception as e:
            errors.append(f"pydub: {e}")
    
    # Method 4: Use ffmpeg directly via subprocess
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_in:
            tmp_in.write(content)
            tmp_in_path = tmp_in.name
        
        tmp_out_path = tmp_in_path.replace('.webm', '.wav')
        
        # Run ffmpeg
        result = subprocess.run([
            'ffmpeg', '-y', '-i', tmp_in_path,
            '-ar', '16000', '-ac', '1', '-f', 'wav', tmp_out_path
        ], capture_output=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(tmp_out_path):
            audio_data, sr = sf.read(tmp_out_path)
            # Cleanup
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
            return audio_data.astype(np.float32), sr
        else:
            errors.append(f"ffmpeg: {result.stderr.decode()[:200]}")
    except FileNotFoundError:
        errors.append("ffmpeg: not found in PATH")
    except Exception as e:
        errors.append(f"ffmpeg: {e}")
    finally:
        # Cleanup temp files
        try:
            if 'tmp_in_path' in locals() and os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)
            if 'tmp_out_path' in locals() and os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)
        except:
            pass
    
    # All methods failed
    raise ValueError(f"Could not load audio. Tried: {'; '.join(errors)}")


# ===================== CONFIG =====================
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
BABY_CRY_MODEL_PATH = WORKSPACE_ROOT / "ast_baby_cry_optimized"
RESPIRATORY_MODEL_PATH = WORKSPACE_ROOT / "ast_respiratory_optimized"

# Baby cry classes
BABY_CRY_CLASSES = [
    "cold_cry", "discomfort_cry", "distress_cry", "hungry_cry",
    "normal_cry", "pain_cry", "sleepy_cry", "tired_cry"
]

# Risk level mappings
RISK_LEVELS = {
    "pain_cry": {"level": "high", "color": "#FF0000", "action": "Seek medical attention"},
    "distress_cry": {"level": "high", "color": "#FF0000", "action": "Check immediately"},
    "cold_cry": {"level": "medium", "color": "#FFA500", "action": "Adjust temperature"},
    "discomfort_cry": {"level": "medium", "color": "#FFA500", "action": "Check diaper/position"},
    "hungry_cry": {"level": "low", "color": "#00FF00", "action": "Feed baby"},
    "sleepy_cry": {"level": "low", "color": "#00FF00", "action": "Put to sleep"},
    "tired_cry": {"level": "low", "color": "#00FF00", "action": "Rest time"},
    "normal_cry": {"level": "normal", "color": "#00FF00", "action": "Normal - no action needed"},
    "no_cry_detected": {"level": "normal", "color": "#888888", "action": "No baby cry detected in audio"},
    # Respiratory indicators
    "wheeze_detected": {"level": "high", "color": "#FF0000", "action": "Possible respiratory issue - consult pediatrician"},
    "cough_detected": {"level": "medium", "color": "#FFA500", "action": "Monitor for persistent cough"},
    "breathing_abnormal": {"level": "medium", "color": "#FFA500", "action": "Observe breathing pattern"},
}

# Map AudioSet labels to our diagnostic categories
# The AudioSet model detects SOUND TYPE, we then use biomarkers to classify cry TYPE
AUDIOSET_CRY_LABELS = {
    "Baby cry, infant cry",
    "Crying, sobbing",
    "Whimper",
    "Wail, moan",
    "Screaming",
}

AUDIOSET_RESPIRATORY_LABELS = {
    "Wheeze": {"diagnosis": "wheeze_detected", "severity": 3},
    "Cough": {"diagnosis": "cough_detected", "severity": 2},
    "Breathing": {"diagnosis": "breathing_detected", "severity": 1},
    "Gasp": {"diagnosis": "respiratory_distress", "severity": 3},
    "Snoring": {"diagnosis": "snoring_detected", "severity": 1},
    "Pant": {"diagnosis": "rapid_breathing", "severity": 2},
}

# Cry type classification based on acoustic features (research-based)
# Reference: Alaie & Naghsh-Nilchi (2016), LaGasse et al. (2005)
CRY_ACOUSTIC_PROFILES = {
    "pain_cry": {
        "f0_min": 550, "f0_max": 1000,  # Very high pitch
        "energy_min": 0.08,              # High energy
        "hnr_max": 8,                    # Some turbulence
        "description": "High-pitched, intense, sudden onset"
    },
    "hungry_cry": {
        "f0_min": 350, "f0_max": 500,   # Medium-high pitch
        "energy_min": 0.05,              # Moderate-high energy
        "pattern": "rhythmic",           # Rhythmic pattern
        "description": "Rhythmic, building intensity, moderate pitch"
    },
    "sleepy_cry": {
        "f0_min": 200, "f0_max": 350,   # Lower pitch
        "energy_max": 0.06,              # Lower energy
        "description": "Whiny, lower intensity, irregular"
    },
    "discomfort_cry": {
        "f0_min": 400, "f0_max": 550,   # Medium pitch
        "energy_min": 0.04,
        "description": "Fussy, intermittent, moderate pitch"
    },
    "distress_cry": {
        "f0_min": 500, "f0_max": 700,   # High pitch
        "energy_min": 0.07,              # High energy
        "f0_std_min": 80,                # High variability
        "description": "Urgent, high variability, high energy"
    },
}

# AudioSet label to baby diagnosis mapping (for MIT/ast-finetuned-audioset model)
AUDIOSET_TO_DIAGNOSIS = {
    # Baby cry specific
    "Baby cry, infant cry": {"diagnosis": "baby_cry_detected", "category": "cry", "severity": 2},
    "Crying, sobbing": {"diagnosis": "crying_detected", "category": "cry", "severity": 2},
    "Whimper": {"diagnosis": "whimper_cry", "category": "cry", "severity": 1},
    "Wail, moan": {"diagnosis": "distress_cry", "category": "cry", "severity": 3},
    "Screaming": {"diagnosis": "pain_cry", "category": "cry", "severity": 4},
    
    # Respiratory indicators (PULMONARY HEALTH)
    "Wheeze": {"diagnosis": "wheeze_detected", "category": "respiratory", "severity": 4},
    "Cough": {"diagnosis": "cough_detected", "category": "respiratory", "severity": 3},
    "Sneeze": {"diagnosis": "sneeze_detected", "category": "respiratory", "severity": 1},
    "Sniff": {"diagnosis": "congestion_possible", "category": "respiratory", "severity": 2},
    "Breathing": {"diagnosis": "breathing_detected", "category": "respiratory", "severity": 1},
    "Gasp": {"diagnosis": "breathing_difficulty", "category": "respiratory", "severity": 4},
    "Snoring": {"diagnosis": "airway_obstruction_possible", "category": "respiratory", "severity": 3},
    "Throat clearing": {"diagnosis": "throat_irritation", "category": "respiratory", "severity": 2},
    
    # Baby positive sounds
    "Baby laughter": {"diagnosis": "happy_baby", "category": "positive", "severity": 0},
    "Giggle": {"diagnosis": "content_baby", "category": "positive", "severity": 0},
    
    # Speech (possibly communication)
    "Babbling": {"diagnosis": "vocalizing", "category": "communication", "severity": 0},
    "Child speech, kid speaking": {"diagnosis": "vocalizing", "category": "communication", "severity": 0},
    
    # Environment
    "Silence": {"diagnosis": "no_cry_detected", "category": "none", "severity": 0},
}


# ===================== SIMPLE ANALYZER =====================
class SimpleCryAnalyzer:
    """
    Baby cry analyzer with AI model support.
    Uses HuggingFace Audio Spectrogram Transformer (AST) for classification,
    with librosa for acoustic biomarker extraction.
    Also supports pulmonary disease detection.
    """
    
    def __init__(self):
        self.model = None
        self.rf_model = None  # Random Forest model
        self.pulmonary_model = None  # Pulmonary disease CNN model
        self.pulmonary_config = None
        self.processor = None
        self.device = None
        self.label_mapping = None
        self.model_type = "feature-based"  # Will change to "ast" if model loads
        self._load_model()
        self._load_pulmonary_model()
    
    def _load_rf_model(self, model_dir: Path):
        """Load Random Forest model for cry classification"""
        import pickle
        
        # Load label mapping
        label_path = model_dir / "label_mappings.json"
        if label_path.exists():
            with open(label_path) as f:
                self.label_mapping = json.load(f)
        
        # Load config
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.rf_config = json.load(f)
        
        # Load model (may be dict with model+scaler or just model)
        model_path = model_dir / "rf_model.pkl"
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        
        # Handle both old format (just model) and new format (dict with model+scaler)
        if isinstance(loaded, dict) and 'model' in loaded:
            self.rf_model = loaded['model']
            self.rf_scaler = loaded.get('scaler', None)
            # Update label mapping from model file if available
            if 'idx_to_label' in loaded:
                self.label_mapping['idx_to_label'] = loaded['idx_to_label']
        else:
            self.rf_model = loaded
            self.rf_scaler = None
        
        print(f"[+] Random Forest model loaded successfully")
        print(f"    Accuracy: {self.rf_config.get('accuracy', 'unknown')*100:.1f}%")
    
    def _load_pulmonary_model(self):
        """Load CNN model for pulmonary disease classification"""
        pulmonary_path = Path(__file__).parent.parent.parent / "model_pulmonary_disease"
        
        if not pulmonary_path.exists() or not (pulmonary_path / "model.pt").exists():
            print("[!] Pulmonary disease model not found")
            return
        
        try:
            import torch
            import torch.nn as nn
            
            # Load config
            config_path = pulmonary_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self.pulmonary_config = json.load(f)
            else:
                # Default config if not saved yet
                self.pulmonary_config = {
                    "num_classes": 8,
                    "classes": {0: "healthy", 1: "pneumonia", 2: "bronchiolitis", 3: "ards", 
                               4: "asphyxia", 5: "sepsis_respiratory", 6: "respiratory_distress", 7: "bronchitis"},
                    "sample_rate": 16000,
                    "n_mels": 128,
                    "max_duration": 5
                }
            
            # Define CNN architecture (must match training)
            class PulmonaryCNN(nn.Module):
                def __init__(self, num_classes=8):
                    super().__init__()
                    # Convolutional layers
                    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm2d(32)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm2d(64)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.bn3 = nn.BatchNorm2d(128)
                    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                    self.bn4 = nn.BatchNorm2d(256)
                    
                    self.pool = nn.MaxPool2d(2, 2)
                    self.dropout = nn.Dropout(0.5)
                    
                    # Global average pooling (matches training script)
                    self.gap = nn.AdaptiveAvgPool2d(1)
                    
                    # Classifier (fc1: 256->128, fc2: 128->num_classes)
                    self.fc1 = nn.Linear(256, 128)
                    self.fc2 = nn.Linear(128, num_classes)
                    
                def forward(self, x):
                    # Conv blocks
                    import torch.nn.functional as F
                    x = self.pool(F.relu(self.bn1(self.conv1(x))))
                    x = self.pool(F.relu(self.bn2(self.conv2(x))))
                    x = self.pool(F.relu(self.bn3(self.conv3(x))))
                    x = self.pool(F.relu(self.bn4(self.conv4(x))))
                    
                    # Global average pooling
                    x = self.gap(x)
                    x = x.view(x.size(0), -1)
                    
                    # Classifier
                    x = self.dropout(F.relu(self.fc1(x)))
                    x = self.fc2(x)
                    return x
            
            # Create model
            num_classes = self.pulmonary_config.get("num_classes", 8)
            self.pulmonary_model = PulmonaryCNN(num_classes)
            
            # Load weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pulmonary_model.load_state_dict(
                torch.load(pulmonary_path / "model.pt", map_location=device)
            )
            self.pulmonary_model.to(device)
            self.pulmonary_model.eval()
            
            print(f"[+] Pulmonary disease model loaded successfully")
            print(f"    Classes: {list(self.pulmonary_config.get('classes', {}).values())}")
            
        except Exception as e:
            print(f"[!] Failed to load pulmonary model: {e}")
            import traceback
            traceback.print_exc()
    
    def _classify_pulmonary(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Classify pulmonary diseases using CNN model."""
        import torch
        import torch.nn.functional as F
        
        if self.pulmonary_model is None:
            return None
        
        # Prepare audio
        MAX_DURATION = self.pulmonary_config.get("max_duration", 5)
        N_MELS = self.pulmonary_config.get("n_mels", 128)
        SAMPLE_RATE = self.pulmonary_config.get("sample_rate", 16000)
        
        max_samples = SAMPLE_RATE * MAX_DURATION
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize spectrogram
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Convert to tensor
        device = next(self.pulmonary_model.parameters()).device
        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Classify
        with torch.no_grad():
            outputs = self.pulmonary_model(mel_tensor)
            probs = F.softmax(outputs, dim=-1).cpu().numpy()[0]
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        # Get label
        classes = self.pulmonary_config.get("classes", {})
        disease = classes.get(str(pred_idx), classes.get(pred_idx, f"class_{pred_idx}"))
        
        # Get all probabilities
        all_probs = {}
        for i, p in enumerate(probs):
            label = classes.get(str(i), classes.get(i, f"class_{i}"))
            all_probs[label] = float(p)
        
        return {
            "disease": disease,
            "confidence": confidence,
            "all_predictions": all_probs,
            "is_healthy": disease == "healthy",
            "requires_attention": disease not in ["healthy"] and confidence > 0.5
        }

    def _extract_rf_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract features for Random Forest model (must match training)"""
        features = []
        
        # CRITICAL: Limit to 5 seconds like training does
        max_samples = sr * 5
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Ensure minimum length  
        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)))
        
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
    
    def _load_model(self):
        """Load pre-trained baby cry classification model from HuggingFace"""
        if not HAS_TORCH:
            print("[!] PyTorch not installed - using feature-based analysis")
            return
        
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Device: {self.device}")
        
        # Try to load Random Forest model first (better for cry type classification)
        rf_model_path = Path(__file__).parent.parent.parent / "rf_baby_cry_model"
        
        if rf_model_path.exists() and (rf_model_path / "rf_model.pkl").exists():
            print(f"[*] Loading Random Forest model from {rf_model_path}")
            try:
                self._load_rf_model(rf_model_path)
                if self.rf_model is not None:
                    self.model_type = "random-forest"
                    self.model = "rf"  # Set model to non-None to indicate model is loaded
                    return
            except Exception as e:
                print(f"[!] Failed to load RF model: {e}")
        
        # Try to load CUSTOM TRAINED wav2vec2 model
        custom_model_path = Path(__file__).parent.parent.parent / "ast_baby_cry_optimized"
        
        if custom_model_path.exists() and (custom_model_path / "pytorch_model.bin").exists():
            print(f"[*] Loading custom trained model from {custom_model_path}")
            try:
                self._load_custom_model(custom_model_path)
                if self.model is not None:
                    self.model_type = "custom-trained"
                    return
            except Exception as e:
                print(f"[!] Failed to load custom model: {e}")
        
        # Fallback: Try to load local model from models_pretrained
        local_model_path = Path(__file__).parent.parent / "models_pretrained" / "baby_cry_ast"
        
        if local_model_path.exists():
            try:
                from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                self.model = AutoModelForAudioClassification.from_pretrained(str(local_model_path))
                self.processor = AutoFeatureExtractor.from_pretrained(str(local_model_path))
                self.model.to(self.device)
                self.model.eval()
                self.label_mapping = self.model.config.id2label
                self.model_type = "ast-local"
                print(f"[+] Loaded local AST model from {local_model_path}")
                print(f"    Labels: {list(self.label_mapping.values())}")
                return
            except Exception as e:
                print(f"[!] Failed to load local model: {e}")
        
        # Try multiple HuggingFace models in order of preference
        models_to_try = [
            ("Hemant333/Baby-crying-audio-detection", "Binary: crying/not crying"),
            ("atasoglu/wav2vec2-baby-cry-classification", "Multi-class baby cry"),
            ("MIT/ast-finetuned-audioset-10-10-0.4593", "General audio (includes baby cry)"),
        ]
        
        for model_name, description in models_to_try:
            try:
                from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
                print(f"[*] Trying model: {model_name} ({description})")
                
                # Try standard loading
                try:
                    self.model = AutoModelForAudioClassification.from_pretrained(model_name)
                    self.processor = AutoFeatureExtractor.from_pretrained(model_name)
                except:
                    # Try wav2vec2 specific loading
                    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                    self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                
                self.model.to(self.device)
                self.model.eval()
                self.label_mapping = self.model.config.id2label
                self.model_type = f"hf-{model_name.split('/')[-1]}"
                print(f"[+] Loaded model: {model_name}")
                print(f"    Labels: {list(self.label_mapping.values())}")
                
                # Save locally for faster startup next time
                try:
                    local_model_path.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(str(local_model_path))
                    self.processor.save_pretrained(str(local_model_path))
                    print(f"[+] Cached to {local_model_path}")
                except Exception as e:
                    print(f"[!] Could not cache: {e}")
                
                return
                
            except Exception as e:
                print(f"[!] Failed: {model_name} - {e}")
                continue
        
        # All models failed - use feature-based
        print("[!] No HuggingFace models available - using feature-based analysis")
        print("[*] Feature-based analysis uses acoustic biomarkers for classification")
        self.model = None
    
    def _load_custom_model(self, model_dir: Path):
        """Load custom trained model"""
        import torch
        import torch.nn as nn
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load label mapping
        label_path = model_dir / "label_mappings.json"
        if label_path.exists():
            with open(label_path) as f:
                self.label_mapping = json.load(f)
        
        # Load pretrained wav2vec2 BASE model (matching the trained checkpoint)
        pretrained_name = "facebook/wav2vec2-base"
        try:
            wav2vec2 = AutoModel.from_pretrained(pretrained_name)
            hidden_size = wav2vec2.config.hidden_size  # Should be 768 for base
            
            # Create classifier wrapper (same as training)
            class AudioClassifier(nn.Module):
                def __init__(self, encoder, num_labels, hidden_size):
                    super().__init__()
                    self.wav2vec2 = encoder
                    self.dropout = nn.Dropout(0.3)
                    self.classifier = nn.Linear(hidden_size, num_labels)
                
                def forward(self, input_values, attention_mask=None):
                    outputs = self.wav2vec2(input_values)
                    hidden_states = outputs.last_hidden_state
                    pooled = hidden_states.mean(dim=1)
                    pooled = self.dropout(pooled)
                    logits = self.classifier(pooled)
                    return logits
            
            num_labels = len(self.label_mapping["label2id"]) if self.label_mapping else 8
            self.model = AudioClassifier(wav2vec2, num_labels, hidden_size)
            
            # Load trained weights
            checkpoint_path = model_dir / "pytorch_model.bin"
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Load processor using Wav2Vec2FeatureExtractor
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_name)
            print(f"[+] Custom model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[!] Model loading error: {e}")
            self.model = None
    
    def analyze(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyze audio and return classification results with pulmonary health indicators.
        """
        # Ensure proper format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / 32768.0
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95
        
        # Extract biomarkers FIRST (needed for both classification methods)
        biomarkers = self._extract_biomarkers(audio_data, sample_rate)
        
        # Run classification - use Random Forest if available
        if self.rf_model is not None:
            classification = self._classify_with_rf(audio_data, sample_rate, biomarkers)
        elif self.model is not None:
            classification = self._classify_with_model(audio_data, sample_rate, biomarkers)
        else:
            classification = self._classify_with_features(biomarkers)
        
        # Run pulmonary disease classification if model is available
        pulmonary_disease = None
        if self.pulmonary_model is not None:
            try:
                pulmonary_disease = self._classify_pulmonary(audio_data, sample_rate)
            except Exception as e:
                print(f"[!] Pulmonary classification error: {e}")
        
        # Get risk level
        label = classification["label"]
        risk_info = RISK_LEVELS.get(label, RISK_LEVELS["normal_cry"])
        
        # Calculate pulmonary health indicators based on acoustic features
        pulmonary_indicators = self._assess_pulmonary_health(biomarkers)
        
        # If we have AI pulmonary disease classification, add it to indicators
        if pulmonary_disease:
            pulmonary_indicators["ai_disease_detection"] = pulmonary_disease
            if pulmonary_disease.get("requires_attention"):
                pulmonary_indicators["findings"].append({
                    "indicator": "AI Disease Detection",
                    "value": f"{pulmonary_disease['disease']} ({pulmonary_disease['confidence']*100:.1f}%)",
                    "significance": "Deep learning model detected potential pulmonary condition",
                    "requires_attention": True
                })
                pulmonary_indicators["recommendations"].append(
                    f"AI detected possible {pulmonary_disease['disease']}. Consult a pediatrician."
                )
        
        return {
            "classification": classification,
            "biomarkers": biomarkers,
            "pulmonary_health": pulmonary_indicators,
            "pulmonary_disease": pulmonary_disease,
            "risk_level": risk_info["level"],
            "risk_color": risk_info["color"],
            "recommended_action": risk_info["action"],
            "timestamp": datetime.now().isoformat(),
            "audio_duration": len(audio_data) / sample_rate,
            "model_used": self.model_type,
            "pulmonary_model_loaded": self.pulmonary_model is not None,
        }
    
    def _assess_pulmonary_health(self, biomarkers: Dict) -> Dict[str, Any]:
        """
        Assess potential pulmonary health indicators from cry acoustics.
        
        Medical Basis (peer-reviewed research):
        - LaGasse et al. (2005): High f0 correlates with respiratory distress
        - Várallyay (2007): Cry melody patterns indicate neurological/respiratory health
        - Alaie & Naghsh-Nilchi (2016): Spectral features correlate with pathology
        
        NOTE: This is a SCREENING tool, not a diagnostic device.
        """
        f0 = biomarkers.get("f0_mean", 0)
        hnr = biomarkers.get("hnr", 0)
        energy = biomarkers.get("energy_rms", 0)
        spec_cent = biomarkers.get("spectral_centroid", 0)
        f0_std = biomarkers.get("f0_std", 0)
        
        indicators = {
            "respiratory_distress_risk": "low",
            "airway_congestion_risk": "low",
            "breathing_effort": "normal",
            "cry_strength": "normal",
            "findings": [],
            "recommendations": [],
            "medical_note": "This is an AI screening tool. Always consult a pediatrician for medical concerns."
        }
        
        # High f0 (> 600 Hz) - potential respiratory distress
        if f0 > 600:
            indicators["respiratory_distress_risk"] = "high"
            indicators["findings"].append({
                "indicator": "High Fundamental Frequency",
                "value": f"{f0:.1f} Hz",
                "threshold": "> 600 Hz",
                "significance": "May indicate respiratory distress, laryngeal stress, or pain",
                "research": "LaGasse et al. (2005) - High pitch correlates with neurodevelopmental stress"
            })
            indicators["recommendations"].append("Monitor breathing rate and pattern closely")
        elif f0 > 500:
            indicators["respiratory_distress_risk"] = "moderate"
            indicators["findings"].append({
                "indicator": "Elevated Fundamental Frequency",
                "value": f"{f0:.1f} Hz",
                "threshold": "500-600 Hz",
                "significance": "Slightly elevated - may indicate mild distress or discomfort"
            })
        
        # Low HNR (< 5 dB) - turbulent airflow suggesting congestion
        if hnr < 5 and hnr != 0:
            indicators["airway_congestion_risk"] = "moderate"
            indicators["findings"].append({
                "indicator": "Low Harmonic-to-Noise Ratio",
                "value": f"{hnr:.1f} dB",
                "threshold": "< 5 dB",
                "significance": "Turbulent airflow - may indicate nasal congestion or airway obstruction",
                "research": "Turbulent phonation correlates with upper respiratory issues"
            })
            indicators["recommendations"].append("Check for nasal congestion, ensure clear airways")
        
        # Low energy - weak cry
        if energy < 0.02 and energy > 0:
            indicators["cry_strength"] = "weak"
            indicators["breathing_effort"] = "reduced"
            indicators["findings"].append({
                "indicator": "Weak Cry Energy",
                "value": f"{energy:.4f} RMS",
                "threshold": "< 0.02",
                "significance": "Low cry energy may indicate fatigue, respiratory weakness, or illness"
            })
            indicators["recommendations"].append("Monitor feeding, check temperature, observe energy levels")
        elif energy > 0.15:
            indicators["cry_strength"] = "strong"
        
        # High f0 variability - irregular breathing
        if f0_std > 100:
            indicators["breathing_effort"] = "irregular"
            indicators["findings"].append({
                "indicator": "High Pitch Variability",
                "value": f"{f0_std:.1f} Hz std",
                "threshold": "> 100 Hz",
                "significance": "Irregular pitch may indicate breathing difficulty or distress"
            })
        
        # Overall risk assessment
        risks = {
            "high": 3,
            "moderate": 2, 
            "low": 1
        }
        max_risk = max(
            risks.get(indicators["respiratory_distress_risk"], 1),
            risks.get(indicators["airway_congestion_risk"], 1)
        )
        
        if max_risk == 3:
            indicators["overall_pulmonary_status"] = "ATTENTION NEEDED"
            indicators["status_color"] = "#FF6B6B"
        elif max_risk == 2:
            indicators["overall_pulmonary_status"] = "MONITOR"
            indicators["status_color"] = "#FFD93D"
        else:
            indicators["overall_pulmonary_status"] = "NORMAL"
            indicators["status_color"] = "#6BCB77"
        
        return indicators
    
    def _extract_biomarkers(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract acoustic biomarkers from audio"""
        biomarkers = {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "spectral_centroid": 0.0,
            "hnr": 0.0,
            "energy_rms": 0.0,
            "zcr": 0.0,
        }
        
        if HAS_LIBROSA:
            try:
                # Fundamental frequency (f0)
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio, fmin=75, fmax=1000, sr=sr
                )
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    biomarkers["f0_mean"] = float(np.mean(f0_valid))
                    biomarkers["f0_std"] = float(np.std(f0_valid))
                
                # Spectral centroid
                spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
                biomarkers["spectral_centroid"] = float(np.mean(spec_cent))
                
                # RMS energy
                rms = librosa.feature.rms(y=audio)
                biomarkers["energy_rms"] = float(np.mean(rms))
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio)
                biomarkers["zcr"] = float(np.mean(zcr))
                
                # Harmonic-to-noise ratio (approximation)
                harmonic, percussive = librosa.effects.hpss(audio)
                h_energy = np.sum(harmonic ** 2)
                p_energy = np.sum(percussive ** 2)
                if p_energy > 0:
                    biomarkers["hnr"] = float(10 * np.log10(h_energy / p_energy + 1e-10))
                
            except Exception as e:
                print(f"[!] Biomarker extraction error: {e}")
        else:
            # Basic numpy fallback
            biomarkers["energy_rms"] = float(np.sqrt(np.mean(audio ** 2)))
            biomarkers["zcr"] = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
        
        return biomarkers
    
    def _classify_with_rf(self, audio: np.ndarray, sr: int, biomarkers: Dict) -> Dict[str, Any]:
        """Classification using Random Forest model."""
        # Debug biomarkers
        f0 = biomarkers.get("f0_mean", 0)
        f0_std = biomarkers.get("f0_std", 0)
        energy = biomarkers.get("energy_rms", 0)
        hnr = biomarkers.get("hnr", 0)
        print(f"[DEBUG] Biomarkers: f0={f0:.1f}Hz, energy={energy:.4f}, hnr={hnr:.1f}, f0_std={f0_std:.1f}")
        
        # Extract features for classification
        features = self._extract_rf_features(audio, sr)
        
        # Scale features if scaler is available
        if hasattr(self, 'rf_scaler') and self.rf_scaler is not None:
            features = self.rf_scaler.transform([features])[0]
        
        # Predict
        probs = self.rf_model.predict_proba([features])[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        # Get label - try idx_to_label first, then id2label
        id2label = self.label_mapping.get("idx_to_label", self.label_mapping.get("id2label", {}))
        cry_type = id2label.get(str(pred_idx), id2label.get(pred_idx, f"class_{pred_idx}"))
        
        # Get all probabilities
        all_probs = {}
        for i, p in enumerate(probs):
            label = id2label.get(str(i), id2label.get(i, f"class_{i}"))
            all_probs[label] = float(p)
        
        print(f"[DEBUG] RF prediction: {cry_type} ({confidence*100:.1f}%)")
        print(f"[DEBUG] All probs: {sorted(all_probs.items(), key=lambda x: -x[1])[:5]}")
        
        return {
            "label": cry_type,
            "confidence": confidence,
            "model": "random-forest",
            "cry_detected": True,
            "ai_cry_confidence": confidence,
            "respiratory_indicators": [],
            "top_ai_predictions": all_probs,
            "classification_method": "random_forest"
        }
    
    def _classify_with_model(self, audio: np.ndarray, sr: int, biomarkers: Dict) -> Dict[str, Any]:
        """
        Classification using trained model.
        
        For custom-trained models: Direct cry type classification (8 classes)
        For AudioSet models: Two-stage (detect cry + classify type from biomarkers)
        """
        import torch
        import torch.nn.functional as F
        
        # === PREPROCESSING (must match training) ===
        MAX_LENGTH = 80000  # 5 seconds at 16kHz
        
        # Truncate or pad to fixed length
        if len(audio) > MAX_LENGTH:
            audio = audio[:MAX_LENGTH]
        elif len(audio) < MAX_LENGTH:
            audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
        
        # Normalize (same as training)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        with torch.no_grad():
            # Process audio
            inputs = self.processor(
                audio, sampling_rate=sr, return_tensors="pt", padding=True
            )
            
            # Move to device
            if "input_values" in inputs:
                model_inputs = {"input_values": inputs.input_values.to(self.device)}
            elif "input_features" in inputs:
                model_inputs = {"input_features": inputs.input_features.to(self.device)}
            else:
                model_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = self.model(**model_inputs)
            # Handle both HuggingFace models (with .logits) and custom models (direct tensor)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs  # Custom model returns logits directly
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # ===== CUSTOM TRAINED MODEL: Direct cry type classification =====
            if self.model_type == "custom-trained" and self.label_mapping:
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                
                # Get label from mapping
                id2label = self.label_mapping.get("id2label", {})
                cry_type = id2label.get(str(top_idx), id2label.get(top_idx, f"class_{top_idx}"))
                
                # Get all class probabilities
                all_probs = {}
                for i, p in enumerate(probs):
                    label = id2label.get(str(i), id2label.get(i, f"class_{i}"))
                    all_probs[label] = float(p)
                
                print(f"[DEBUG] Custom model prediction: {cry_type} ({confidence*100:.1f}%)")
                print(f"[DEBUG] All probs: {sorted(all_probs.items(), key=lambda x: -x[1])[:5]}")
                
                return {
                    "label": cry_type,
                    "confidence": confidence,
                    "model": "custom-trained-wav2vec2",
                    "cry_detected": True,
                    "ai_cry_confidence": confidence,
                    "respiratory_indicators": [],
                    "top_ai_predictions": all_probs,
                    "classification_method": "custom_model_direct"
                }
            
            # ===== AUDIOSET MODELS: Two-stage classification =====
            # Get top predictions
            top_k = 15
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            # ===== STAGE 1: DETECT sound types using AI model =====
            cry_confidence = 0.0
            respiratory_detections = []
            top_predictions = {}
            
            # Extended cry-related keywords for broader detection
            CRY_KEYWORDS = {"cry", "crying", "sob", "wail", "scream", "whimper", "baby", "infant", "child"}
            
            for idx in top_indices:
                if self.label_mapping:
                    if isinstance(self.label_mapping, dict):
                        label = self.label_mapping.get(idx, self.label_mapping.get(str(idx), f"class_{idx}"))
                    else:
                        label = self.label_mapping[idx] if idx < len(self.label_mapping) else f"class_{idx}"
                else:
                    label = f"class_{idx}"
                
                prob = float(probs[idx])
                top_predictions[label] = prob
                
                # Check for cry-related sounds - exact match
                if label in AUDIOSET_CRY_LABELS:
                    cry_confidence = max(cry_confidence, prob)
                
                # Also check for keywords in label (broader matching)
                label_lower = label.lower()
                if any(kw in label_lower for kw in CRY_KEYWORDS):
                    cry_confidence = max(cry_confidence, prob * 0.9)  # Slightly lower for keyword matches
                
                # Check for respiratory sounds
                if label in AUDIOSET_RESPIRATORY_LABELS:
                    respiratory_detections.append({
                        "type": AUDIOSET_RESPIRATORY_LABELS[label]["diagnosis"],
                        "audioset_label": label,
                        "confidence": prob,
                        "severity": AUDIOSET_RESPIRATORY_LABELS[label]["severity"]
                    })
            
            # Print debug info
            print(f"[DEBUG] Top AI predictions: {list(top_predictions.items())[:5]}")
            print(f"[DEBUG] Cry confidence from AI: {cry_confidence:.3f}")
            print(f"[DEBUG] Biomarkers: f0={biomarkers.get('f0_mean', 0):.1f}Hz, energy={biomarkers.get('energy_rms', 0):.4f}")
            
            # ===== STAGE 2: CLASSIFY cry type using acoustic biomarkers =====
            # More lenient threshold - if AI detects ANY cry-related sound above 5%
            cry_detected = cry_confidence > 0.05
            
            if cry_detected:
                # Use biomarkers to determine specific cry type
                cry_type, cry_type_confidence = self._classify_cry_type_from_biomarkers(biomarkers)
                
                # Combine AI confidence with acoustic classification
                combined_confidence = (cry_confidence * 0.3) + (cry_type_confidence * 0.7)
                print(f"[DEBUG] AI detected cry -> Type: {cry_type}, Combined conf: {combined_confidence:.2f}")
            else:
                # Fallback: Use acoustic features if they indicate voice/cry
                # Baby cries typically have f0 > 250 Hz and some energy
                f0 = biomarkers.get("f0_mean", 0)
                energy = biomarkers.get("energy_rms", 0)
                
                print(f"[DEBUG] AI didn't detect cry, checking acoustics: f0={f0:.1f}, energy={energy:.4f}")
                
                if f0 > 150 and energy > 0.01:
                    # Audio has voice-like characteristics - classify based on acoustics alone
                    cry_type, cry_type_confidence = self._classify_cry_type_from_biomarkers(biomarkers)
                    combined_confidence = cry_type_confidence * 0.6  # Lower confidence without AI confirmation
                    cry_detected = True
                    print(f"[DEBUG] Acoustic fallback -> Type: {cry_type}, Conf: {combined_confidence:.2f}")
                else:
                    cry_type = "no_cry_detected"
                    combined_confidence = 0.0
                    print(f"[DEBUG] No cry detected (low f0 or energy)")
            
            return {
                "label": cry_type,
                "confidence": combined_confidence,
                "model": f"{self.model_type} + acoustic_features",
                "cry_detected": cry_detected,
                "ai_cry_confidence": cry_confidence,
                "respiratory_indicators": sorted(respiratory_detections, key=lambda x: x["severity"], reverse=True),
                "top_ai_predictions": dict(list(top_predictions.items())[:5]),
                "classification_method": "ai_detection + acoustic_classification"
            }
    
    def _classify_cry_type_from_biomarkers(self, biomarkers: Dict) -> tuple:
        """
        Classify cry type based on acoustic biomarkers.
        
        Based on research:
        - Pain cries: High f0 (>550 Hz), high energy, sudden onset
        - Hunger cries: Medium-high f0 (350-500 Hz), rhythmic, building intensity
        - Sleepy/tired cries: Lower f0 (<350 Hz), lower energy, irregular
        - Discomfort cries: Medium f0, intermittent
        - Distress cries: High f0 with high variability
        
        References:
        - LaGasse et al. (2005): "Acoustic cry analysis and neurodevelopmental outcome"
        - Várallyay (2007): "The melody of crying"
        - Alaie & Naghsh-Nilchi (2016): Cry classification research
        """
        f0 = biomarkers.get("f0_mean", 0)
        f0_std = biomarkers.get("f0_std", 0)
        energy = biomarkers.get("energy_rms", 0)
        hnr = biomarkers.get("hnr", 0)
        spec_cent = biomarkers.get("spectral_centroid", 0)
        
        # Calculate scores for each cry type
        scores = {}
        
        # Pain cry: Very high pitch, high energy
        if f0 > 550:
            pain_score = min(1.0, (f0 - 550) / 250)  # Scale 550-800 Hz to 0-1
            pain_score *= min(1.0, energy / 0.1)     # Boost with energy
            scores["pain_cry"] = pain_score
        else:
            scores["pain_cry"] = 0.0
        
        # Distress cry: High pitch with high variability
        if f0 > 480 and f0_std > 60:
            distress_score = min(1.0, (f0 - 480) / 220) * min(1.0, f0_std / 100)
            scores["distress_cry"] = distress_score
        else:
            scores["distress_cry"] = 0.0
        
        # Hungry cry: Medium-high pitch, moderate-high energy
        if 320 <= f0 <= 520:
            hungry_score = 1.0 - abs(f0 - 420) / 150  # Peak around 420 Hz
            hungry_score *= min(1.0, energy / 0.06)   # Needs some energy
            scores["hungry_cry"] = max(0, hungry_score)
        else:
            scores["hungry_cry"] = 0.0
        
        # Discomfort cry: Medium pitch, moderate energy
        if 350 <= f0 <= 500:
            discomfort_score = 1.0 - abs(f0 - 425) / 125
            discomfort_score *= 0.8  # Lower base confidence
            scores["discomfort_cry"] = max(0, discomfort_score)
        else:
            scores["discomfort_cry"] = 0.0
        
        # Sleepy/tired cry: Lower pitch, lower energy
        if f0 < 380:
            sleepy_score = 1.0 - f0 / 380  # Lower f0 = higher score
            if energy < 0.07:
                sleepy_score *= 1.2  # Boost for low energy
            scores["sleepy_cry"] = min(1.0, sleepy_score)
            scores["tired_cry"] = scores["sleepy_cry"] * 0.9
        else:
            scores["sleepy_cry"] = 0.0
            scores["tired_cry"] = 0.0
        
        # Cold cry: Similar to discomfort but with specific spectral characteristics
        if 380 <= f0 <= 480:
            scores["cold_cry"] = 0.4  # Base score
        else:
            scores["cold_cry"] = 0.0
        
        # Normal cry: Default if nothing else scores high
        scores["normal_cry"] = 0.3  # Base score
        
        # Find highest scoring cry type
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            # If best score is too low, default to normal cry
            if best_score < 0.25:
                return "normal_cry", 0.4
            
            # Normalize confidence to reasonable range (0.5-0.95)
            confidence = 0.5 + (best_score * 0.45)
            return best_type, min(0.95, confidence)
        
        return "normal_cry", 0.4
    
    def _classify_with_features(self, biomarkers: Dict) -> Dict[str, Any]:
        """Simple rule-based classification using biomarkers"""
        f0 = biomarkers.get("f0_mean", 0)
        energy = biomarkers.get("energy_rms", 0)
        spec_cent = biomarkers.get("spectral_centroid", 0)
        
        # Simple heuristic classification
        if f0 > 600:
            label = "pain_cry"
            confidence = min(0.9, f0 / 800)
        elif f0 > 500:
            label = "distress_cry"
            confidence = 0.75
        elif energy > 0.1 and f0 > 400:
            label = "hungry_cry"
            confidence = 0.7
        elif f0 > 350:
            label = "discomfort_cry"
            confidence = 0.65
        elif energy < 0.05:
            label = "sleepy_cry"
            confidence = 0.6
        elif f0 < 300:
            label = "tired_cry"
            confidence = 0.55
        else:
            label = "normal_cry"
            confidence = 0.5
        
        return {
            "label": label,
            "confidence": confidence,
            "model": "feature-based",
            "all_scores": {label: confidence}
        }


# ===================== FASTAPI APP =====================
app = FastAPI(
    title="Baby Cry Diagnostic API",
    description="AI-powered baby cry analysis system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer
analyzer = SimpleCryAnalyzer()

# Store analysis results
analysis_results: Dict[str, Any] = {}


# ===================== MODELS =====================
class AnalysisResult(BaseModel):
    id: str
    classification: Dict[str, Any]
    biomarkers: Dict[str, float]
    risk_level: str
    risk_color: str
    recommended_action: str
    timestamp: str
    audio_duration: float


# ===================== ROUTES =====================
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Baby Cry Diagnostic API",
        "version": "1.0.0",
        "model_loaded": analyzer.model is not None,
        "pulmonary_model_loaded": analyzer.pulmonary_model is not None,
        "librosa_available": HAS_LIBROSA,
        "torch_available": HAS_TORCH,
    }


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio file"""
    try:
        # Read audio file
        content = await file.read()
        filename = file.filename or "audio.webm"
        
        # Load audio using robust loader
        audio_data, sr = load_audio_any_format(content, filename)
        
        # Convert stereo to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if HAS_LIBROSA and sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Analyze
        result = analyzer.analyze(audio_data, sr)
        
        # Generate ID and store
        result_id = str(uuid.uuid4())
        result["id"] = result_id
        analysis_results[result_id] = result
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze-base64")
async def analyze_base64(data: Dict[str, Any]):
    """Analyze base64 encoded audio"""
    try:
        audio_b64 = data.get("audio")
        if not audio_b64:
            raise HTTPException(400, "Missing 'audio' field")
        
        # Decode base64
        audio_bytes = base64.b64decode(audio_b64)
        
        # Load audio
        if HAS_SOUNDFILE:
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        elif HAS_LIBROSA:
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        else:
            raise HTTPException(500, "No audio library available")
        
        # Convert stereo to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz
        if HAS_LIBROSA and sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Analyze
        result = analyzer.analyze(audio_data, sr)
        
        # Generate ID and store
        result_id = str(uuid.uuid4())
        result["id"] = result_id
        analysis_results[result_id] = result
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/api/v1/result/{result_id}")
async def get_result(result_id: str):
    """Get analysis result by ID"""
    if result_id not in analysis_results:
        raise HTTPException(404, "Result not found")
    return analysis_results[result_id]


@app.get("/api/v1/config")
async def get_config():
    """Get system configuration"""
    return {
        "model_loaded": analyzer.model is not None,
        "model_type": "wav2vec2" if analyzer.model else "feature-based",
        "device": str(analyzer.device) if analyzer.device else "cpu",
        "classes": BABY_CRY_CLASSES,
        "risk_levels": RISK_LEVELS,
    }


# WebSocket for real-time streaming
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming"""
    await websocket.accept()
    
    audio_buffer = []
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Convert to numpy array (assuming 16-bit PCM)
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.extend(chunk)
            
            # Analyze every 3 seconds of audio
            if len(audio_buffer) >= 48000:  # 3 seconds at 16kHz
                audio_data = np.array(audio_buffer[:48000])
                audio_buffer = audio_buffer[48000:]
                
                # Analyze
                result = analyzer.analyze(audio_data, 16000)
                
                # Send result
                await websocket.send_json(result)
                
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 60)
    print("BABY CRY DIAGNOSTIC - SIMPLE SERVER")
    print("=" * 60)
    print(f"  librosa: {'OK' if HAS_LIBROSA else 'NOT INSTALLED'}")
    print(f"  soundfile: {'OK' if HAS_SOUNDFILE else 'NOT INSTALLED'}")
    print(f"  PyTorch: {'OK' if HAS_TORCH else 'NOT INSTALLED'}")
    print(f"  Model: {'LOADED' if analyzer.model else 'FEATURE-BASED (no trained model)'}")
    print("=" * 60)
    print("\n  Starting server at http://localhost:8001\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
