#!/usr/bin/env python3
"""
Baby Cry Diagnostic System - RPi5 Standalone with Full 6-Backbone Ensemble

Complete self-contained system running entirely on Raspberry Pi 5.
Includes web dashboard accessible from any device on your network.

Features:
- 6-backbone AI ensemble (DistilHuBERT, Wav2Vec2, WavLM, HuBERT, AST, PANNs)
- All respiratory classifications (16 classes)
- Web dashboard with real-time updates
- INMP441 I2S microphone support
- Optional OLED display and LED indicators
- Systemd service for 24/7 operation
"""

import os
import io
import sys
import json
import wave
import time
import asyncio
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/babycry.log') if os.path.exists('/var/log') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_duration": 5.0,
        "device_name": "plughw:1,0",  # I2S device
        "use_sounddevice": True
    },
    "model": {
        "type": "lightweight",  # "lightweight" or "full"
        "use_quantization": True,
        "batch_size": 1
    },
    "web": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080
    },
    "display": {
        "oled_enabled": False,
        "oled_address": 0x3C,
        "led_enabled": False,
        "led_green": 17,
        "led_yellow": 27,
        "led_red": 22
    },
    "history": {
        "max_entries": 100
    }
}


def load_config() -> dict:
    """Load configuration from file"""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                user_config = json.load(f)
                # Deep merge
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    return config


# ==============================================================================
# All Classification Classes (20 total) - cry/sound-based, INMP441 mic compatible
# ==============================================================================

# Baby Cry Classes (12)
CRY_CLASSES = [
    "hungry_cry", "pain_cry", "sleepy_cry", "discomfort_cry",
    "cold_cry", "tired_cry", "normal_cry", "distress_cry",
    "belly_pain_cry", "burping_cry", "pathological_cry", "asphyxia_cry"
]

# Health/Respiratory Classes (8) - all detectable from cry/audible sounds
HEALTH_CLASSES = [
    "sepsis_cry", "colic_cry", "pneumonia_cry", "respiratory_distress",
    "normal_breathing", "wheezing", "stridor", "bronchiolitis"
]

ALL_CLASSES = CRY_CLASSES + HEALTH_CLASSES

# Risk Mapping for all 20 classes
RISK_MAP = {
    # Baby Cry - GREEN (low risk)
    "normal_cry": "GREEN", "hungry_cry": "GREEN",
    "sleepy_cry": "GREEN", "tired_cry": "GREEN",
    "burping_cry": "GREEN",
    # Baby Cry - YELLOW (moderate risk)
    "cold_cry": "YELLOW", "discomfort_cry": "YELLOW",
    "belly_pain_cry": "YELLOW", "colic_cry": "YELLOW",
    # Baby Cry - ORANGE (elevated risk)
    "distress_cry": "ORANGE", "respiratory_distress": "ORANGE",
    "wheezing": "ORANGE",
    # Baby Cry - RED (critical, urgent care needed)
    "pain_cry": "RED", "pathological_cry": "RED", "asphyxia_cry": "RED",
    "sepsis_cry": "RED", "pneumonia_cry": "RED",
    "stridor": "RED", "bronchiolitis": "RED",
    # Respiratory - GREEN
    "normal_breathing": "GREEN",
}

# Human-readable labels for display
DISPLAY_LABELS = {
    # Baby Cry
    "normal_cry": "Normal Cry", "hungry_cry": "Hungry",
    "sleepy_cry": "Sleepy", "tired_cry": "Tired",
    "burping_cry": "Needs Burping", "cold_cry": "Cold/Hot",
    "discomfort_cry": "Discomfort", "belly_pain_cry": "Belly Pain",
    "distress_cry": "Distress", "pain_cry": "Pain",
    "pathological_cry": "\u26a0\ufe0f PATHOLOGICAL",
    "asphyxia_cry": "\ud83d\udea8 ASPHYXIA - EMERGENCY",
    # Health indicators from cry
    "sepsis_cry": "\ud83d\udea8 SEPSIS - EMERGENCY",
    "colic_cry": "Colic",
    "pneumonia_cry": "\ud83d\udea8 PNEUMONIA INDICATORS",
    "respiratory_distress": "\u26a0\ufe0f Respiratory Distress",
    # Audible respiratory
    "normal_breathing": "Normal Breathing",
    "wheezing": "Wheezing",
    "stridor": "\u26a0\ufe0f STRIDOR - URGENT",
    "bronchiolitis": "\ud83d\udea8 Bronchiolitis/RSV",
}

# Recommendations for each class
RECOMMENDATIONS = {
    # Low risk
    "normal_cry": "Baby is fine, just expressing",
    "hungry_cry": "Try feeding the baby",
    "sleepy_cry": "Baby may be tired, try soothing",
    "tired_cry": "Baby is tired, needs rest",
    "burping_cry": "Try gentle burping",
    "normal_breathing": "Breathing sounds normal",
    # Moderate risk
    "cold_cry": "Check temperature, adjust clothing",
    "discomfort_cry": "Check for wet diaper, gas, or discomfort",
    "belly_pain_cry": "Possible gas pain. Try gentle massage",
    "colic_cry": "Colic episode. Swaddle, white noise, gentle rocking",
    # Elevated risk
    "distress_cry": "Baby is distressed. Check all basics first",
    "respiratory_distress": "\u26a0\ufe0f Labored breathing. Position upright. Seek evaluation",
    "wheezing": "\u26a0\ufe0f Monitor breathing closely. Consult doctor if persistent",
    # Critical
    "pain_cry": "Baby may be in pain. Examine and seek help",
    "pathological_cry": "UNUSUAL CRY PATTERN - Seek immediate evaluation",
    "asphyxia_cry": "\ud83d\udea8 EMERGENCY: Signs of oxygen deprivation. Call 911/999",
    "sepsis_cry": "\ud83d\udea8 EMERGENCY: Possible neonatal sepsis. Seek IMMEDIATE medical care",
    "pneumonia_cry": "\ud83d\udea8 Pneumonia indicators detected. Seek immediate pediatric care",
    "stridor": "\u26a0\ufe0f URGENT: Upper airway issue - Seek immediate care",
    "bronchiolitis": "\ud83d\udea8 RSV/Bronchiolitis detected - Seek immediate pediatric care",
}

# ==============================================================================
# 6-Backbone Ensemble Classifier
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class LibrosaASTProcessor:
    """Custom AST feature extractor using librosa (no torchaudio needed)."""
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
        import librosa
        mel = librosa.feature.melspectrogram(
            y=y, sr=sampling_rate, n_fft=self.fft_size,
            hop_length=self.hop_size, n_mels=self.n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).T
        if log_mel.shape[0] < self.target_length:
            pad = np.zeros((self.target_length - log_mel.shape[0], self.n_mels), dtype=np.float32)
            log_mel = np.concatenate([log_mel, pad], axis=0)
        else:
            log_mel = log_mel[:self.target_length]
        log_mel = (log_mel - self.mean) / (self.std * 2)
        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        return {"input_values": tensor}


class FusionClassifier(nn.Module):
    """Backbone embedding + hand-crafted features -> class prediction.
    Must match the architecture used during training."""
    def __init__(self, backbone_dim, handcrafted_dim=85, num_classes=20, dropout=0.3):
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
        self.res_proj = nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()

    def forward(self, x):
        x = self.bn_input(x)
        identity = self.res_proj(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = x + identity
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        return self.fc_out(x)


class SixBackboneEnsemble:
    """
    Full 6-backbone ensemble with EQUAL weights (1/6 each).
    Loads trained FusionClassifier heads from checkpoint.

    Backbones:
    1. DistilHuBERT  (ntu-spml/distilhubert)                  - 768-dim
    2. HuBERT        (facebook/hubert-base-ls960)              - 768-dim
    3. Wav2Vec2      (facebook/wav2vec2-base)                  - 768-dim
    4. WavLM         (microsoft/wavlm-base)                    - 768-dim
    5. AST           (MIT/ast-finetuned-audioset-10-10-0.4593) - 768-dim
    6. CLAP          (laion/clap-htsat-unfused)                - 512-dim

    All disease detection is from baby cries + audible sounds via INMP441 mic.
    """

    def __init__(self, use_quantization: bool = True):
        self.use_quantization = use_quantization
        self.device = "cpu"
        self.is_initialized = False

        self.backbones = {}
        self.processors = {}
        self.classifiers = {}
        self.weights = {}
        self.hc_mean = None
        self.hc_std = None

        self.classes = ALL_CLASSES
        self.num_classes = len(ALL_CLASSES)

    async def initialize(self):
        """Initialize ensemble: load all 6 backbones + trained classifier heads."""
        if self.is_initialized:
            return

        logger.info("=" * 50)
        logger.info("LOADING 6-BACKBONE ENSEMBLE")
        logger.info("All models have EQUAL weight (1/6)")
        logger.info("=" * 50)

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            # Load trained checkpoint
            model_path = Path(__file__).parent / "models" / "6backbone_ensemble.pt"
            checkpoint = None
            if model_path.exists():
                checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
                logger.info(f"  Loaded checkpoint: {model_path.name}")
                logger.info(f"  Ensemble accuracy: {checkpoint.get('metadata', {}).get('ensemble_val_acc', 0)*100:.1f}%")
                self.hc_mean = checkpoint.get('handcrafted_mean')
                self.hc_std = checkpoint.get('handcrafted_std')

            # Read backbone configs from checkpoint (matches training)
            if checkpoint and 'backbones' in checkpoint:
                backbone_configs = [
                    (name, info['hf_name'], info['dim'], info['type'])
                    for name, info in checkpoint['backbones'].items()
                ]
            else:
                backbone_configs = [
                    ("distilhubert", "ntu-spml/distilhubert", 768, "wav2vec2"),
                    ("hubert", "facebook/hubert-base-ls960", 768, "wav2vec2"),
                    ("wav2vec2", "facebook/wav2vec2-base", 768, "wav2vec2"),
                    ("wavlm", "microsoft/wavlm-base", 768, "wav2vec2"),
                    ("ast", "MIT/ast-finetuned-audioset-10-10-0.4593", 768, "ast"),
                    ("clap", "laion/clap-htsat-unfused", 512, "clap"),
                ]

            for name, hf_name, dim, btype in backbone_configs:
                logger.info(f"  Loading {name} ({hf_name})...")
                try:
                    await self._load_backbone(name, hf_name, dim, btype, checkpoint)
                    logger.info(f"    [OK] {name} loaded")
                except Exception as e:
                    logger.warning(f"    [SKIP] {name}: {e}")

            # Equal weights for loaded backbones
            if len(self.backbones) > 0:
                equal_weight = 1.0 / len(self.backbones)
                for name in self.backbones:
                    self.weights[name] = equal_weight

            self.is_initialized = True
            logger.info("=" * 50)
            logger.info(f"Ensemble ready: {len(self.backbones)} backbones")
            for name, weight in self.weights.items():
                logger.info(f"  {name}: {weight:.4f}")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {e}")
            self.is_initialized = True  # Use fallback

    async def _load_backbone(self, name, hf_name, dim, btype, checkpoint):
        """Load one backbone feature extractor + trained classifier head."""
        import torch
        import torch.nn as nn

        # Load feature extractor
        if btype == "wav2vec2":
            from transformers import AutoModel, AutoFeatureExtractor
            self.processors[name] = AutoFeatureExtractor.from_pretrained(hf_name)
            backbone = AutoModel.from_pretrained(hf_name)
        elif btype == "ast":
            from transformers import ASTModel
            # Custom processor using librosa (no torchaudio dependency)
            self.processors[name] = LibrosaASTProcessor()
            backbone = ASTModel.from_pretrained(hf_name)
        elif btype == "clap":
            from transformers import ClapModel, ClapProcessor
            self.processors[name] = ClapProcessor.from_pretrained(hf_name)
            backbone = ClapModel.from_pretrained(hf_name)
        else:
            raise ValueError(f"Unknown backbone type: {btype}")

        if self.use_quantization and btype != "clap":
            backbone = torch.quantization.quantize_dynamic(
                backbone, {torch.nn.Linear}, dtype=torch.qint8
            )
        backbone.eval()
        self.backbones[name] = backbone

        # Load trained classifier from checkpoint
        classifier = FusionClassifier(dim, 85, self.num_classes)
        if checkpoint and name in checkpoint.get('backbones', {}):
            state_dict = checkpoint['backbones'][name]['state_dict']
            classifier.load_state_dict(state_dict)
            logger.info(f"    Loaded trained classifier for {name}")
        classifier.eval()
        self.classifiers[name] = classifier
    
    def _extract_handcrafted(self, waveform, sr=16000):
        """Extract 85-dim hand-crafted features for fusion."""
        import librosa
        try:
            mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            sc = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr))
            sb = np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sr))
            sr_ = np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(waveform))
            rms = np.mean(librosa.feature.rms(y=waveform))
            feat = np.concatenate([mfcc_mean, mfcc_std, [sc, sb, sr_, zcr, rms]])
            if not np.isfinite(feat).all():
                return np.zeros(85, dtype=np.float32)
            # Normalize using training stats
            feat = feat.astype(np.float32)
            if self.hc_mean is not None and self.hc_std is not None:
                feat = (feat - self.hc_mean) / (self.hc_std + 1e-8)
            return feat
        except Exception:
            return np.zeros(85, dtype=np.float32)

    def _get_backbone_embedding(self, name, backbone, processor, waveform, sr):
        """Extract embedding from one backbone."""
        import torch
        btype = "clap" if name == "clap" else ("ast" if name == "ast" else "wav2vec2")

        if btype == "wav2vec2":
            inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = backbone(**inputs)
            return out.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        elif btype == "ast":
            inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = backbone(**inputs)
            return out.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        elif btype == "clap":
            inputs = processor(audios=waveform, return_tensors="pt", sampling_rate=48000)
            with torch.no_grad():
                return backbone.get_audio_features(**inputs).squeeze(0).numpy()

    async def classify(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Classify audio using the 6-backbone ensemble.
        Each backbone: extract embedding + hand-crafted -> FusionClassifier -> softmax.
        Final: weighted average of all backbone predictions (equal 1/6).
        """
        if not self.is_initialized:
            await self.initialize()

        import torch
        import torch.nn.functional as F

        # Normalize audio
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=0)
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        if len(self.backbones) == 0:
            return self._fallback_classify(waveform, sample_rate)

        # Extract hand-crafted features (shared across all backbones)
        hc_feat = self._extract_handcrafted(waveform, sample_rate)

        all_probs = []
        backbone_results = {}

        for name, backbone in self.backbones.items():
            try:
                processor = self.processors[name]
                classifier = self.classifiers[name]

                # Get backbone embedding
                emb = self._get_backbone_embedding(name, backbone, processor, waveform, sample_rate)

                # Concatenate with hand-crafted features
                x = np.concatenate([emb, hc_feat])
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

                # Run through trained FusionClassifier
                with torch.no_grad():
                    logits = classifier(x_tensor)
                    probs = F.softmax(logits, dim=-1).numpy()[0]

                weighted_probs = probs * self.weights[name]
                all_probs.append(weighted_probs)

                pred_idx = np.argmax(probs)
                backbone_results[name] = {
                    "class": self.classes[pred_idx],
                    "confidence": float(probs[pred_idx])
                }
            except Exception as e:
                logger.warning(f"Backbone {name} failed: {e}")

        if len(all_probs) == 0:
            return self._fallback_classify(waveform, sample_rate)

        ensemble_probs = np.sum(all_probs, axis=0)
        pred_idx = np.argmax(ensemble_probs)
        pred_class = self.classes[pred_idx]
        confidence = float(ensemble_probs[pred_idx])
        risk_level = RISK_MAP.get(pred_class, "YELLOW")
        display_label = DISPLAY_LABELS.get(pred_class, pred_class)
        task = "cry" if pred_class in CRY_CLASSES else "health"

        return {
            "classification": pred_class,
            "display_label": display_label,
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_score": self._compute_risk_score(pred_class, confidence, risk_level),
            "task": task,
            "all_probs": {c: float(p) for c, p in zip(self.classes, ensemble_probs)},
            "backbone_results": backbone_results,
            "recommendation": RECOMMENDATIONS.get(pred_class, ""),
            "model": f"ensemble-{len(self.backbones)}backbone"
        }
    
    def _fallback_classify(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Rule-based fallback when models not available"""
        energy = np.sqrt(np.mean(waveform**2))
        zcr = np.sum(np.abs(np.diff(np.sign(waveform))) > 0) / len(waveform)
        
        # Simple rules
        if energy < 0.05:
            pred_class = "normal_cry"
            confidence = 0.6
        elif energy > 0.4 and zcr > 0.1:
            pred_class = "pain_cry"
            confidence = 0.5
        elif energy > 0.3:
            pred_class = "hungry_cry"
            confidence = 0.5
        else:
            pred_class = "discomfort_cry"
            confidence = 0.45
        
        risk_level = RISK_MAP.get(pred_class, "YELLOW")
        
        return {
            "classification": pred_class,
            "display_label": DISPLAY_LABELS.get(pred_class, pred_class),
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_score": self._compute_risk_score(pred_class, confidence, risk_level),
            "task": "cry",
            "all_probs": {},
            "backbone_results": {},
            "model": "rule-based-fallback"
        }
    
    def _compute_risk_score(self, classification: str, confidence: float, 
                           risk_level: str) -> float:
        """Compute numeric risk score (0-100)"""
        base_scores = {"GREEN": 20, "YELLOW": 50, "RED": 80}
        base = base_scores.get(risk_level, 50)
        
        if risk_level == "RED":
            score = base + (confidence * 20)
        elif risk_level == "YELLOW":
            score = base + (confidence * 15)
        else:
            score = base - ((1 - confidence) * 10)
        
        return min(100, max(0, score))


# Alias for backwards compatibility
LightweightCryClassifier = SixBackboneEnsemble


# ==============================================================================
# Audio Capture (I2S Microphone)
# ==============================================================================

class I2SAudioCapture:
    """
    Audio capture from INMP441 I2S microphone.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_duration: float = 5.0, device_name: str = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device_name = device_name
        self.audio_lib = None
    
    def initialize(self) -> bool:
        """Initialize audio capture"""
        # Try sounddevice first (easier)
        try:
            import sounddevice as sd
            self.audio_lib = "sounddevice"
            
            devices = sd.query_devices()
            logger.info("Audio devices available:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    logger.info(f"  [{i}] {dev['name']}")
            
            return True
        except ImportError:
            pass
        
        # Try PyAudio as fallback
        try:
            import pyaudio
            self.audio_lib = "pyaudio"
            self.pa = pyaudio.PyAudio()
            logger.info("Using PyAudio for audio capture")
            return True
        except ImportError:
            pass
        
        # Try direct ALSA
        try:
            import alsaaudio
            self.audio_lib = "alsaaudio"
            logger.info("Using ALSA for audio capture")
            return True
        except ImportError:
            pass
        
        logger.error("No audio library available!")
        return False
    
    def record(self) -> np.ndarray:
        """Record audio chunk"""
        if self.audio_lib == "sounddevice":
            return self._record_sounddevice()
        elif self.audio_lib == "pyaudio":
            return self._record_pyaudio()
        elif self.audio_lib == "alsaaudio":
            return self._record_alsa()
        else:
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def _record_sounddevice(self) -> np.ndarray:
        """Record using sounddevice"""
        import sounddevice as sd
        
        try:
            audio = sd.rec(
                self.chunk_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def _record_pyaudio(self) -> np.ndarray:
        """Record using PyAudio"""
        import pyaudio
        
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        
        try:
            stream = self.pa.open(
                format=FORMAT,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            num_chunks = int(self.sample_rate / CHUNK * self.chunk_duration)
            
            for _ in range(num_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            
            stream.stop_stream()
            stream.close()
            
            return np.concatenate(frames)
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def _record_alsa(self) -> np.ndarray:
        """Record using ALSA directly"""
        import alsaaudio
        
        try:
            device = self.device_name or 'plughw:1,0'
            inp = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device=device
            )
            inp.setchannels(self.channels)
            inp.setrate(self.sample_rate)
            inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            inp.setperiodsize(1024)
            
            frames = []
            total_frames = 0
            
            while total_frames < self.chunk_samples:
                length, data = inp.read()
                if length > 0:
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    frames.append(audio)
                    total_frames += len(audio)
            
            return np.concatenate(frames)[:self.chunk_samples]
        except Exception as e:
            logger.error(f"ALSA recording failed: {e}")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'pa'):
            self.pa.terminate()


# ==============================================================================
# Display Output (OLED + LEDs)
# ==============================================================================

class DisplayOutput:
    """
    Output to OLED display and LEDs
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.oled = None
        self.gpio_available = False
        
        if config.get("oled_enabled"):
            self._init_oled()
        if config.get("led_enabled"):
            self._init_leds()
    
    def _init_oled(self):
        """Initialize OLED display"""
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            
            serial = i2c(port=1, address=self.config.get("oled_address", 0x3C))
            self.oled = ssd1306(serial)
            logger.info("OLED display initialized")
        except Exception as e:
            logger.warning(f"OLED not available: {e}")
    
    def _init_leds(self):
        """Initialize LED indicators"""
        try:
            import RPi.GPIO as GPIO
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            self.LED_GREEN = self.config.get("led_green", 17)
            self.LED_YELLOW = self.config.get("led_yellow", 27)
            self.LED_RED = self.config.get("led_red", 22)
            
            GPIO.setup([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.OUT)
            GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.LOW)
            
            self.gpio_available = True
            logger.info("LED indicators initialized")
        except Exception as e:
            logger.warning(f"GPIO not available: {e}")
    
    def show(self, result: Dict[str, Any]):
        """Display classification result"""
        classification = result.get("classification", "unknown")
        confidence = result.get("confidence", 0) * 100
        risk_level = result.get("risk_level", "YELLOW")
        
        # Console output
        print(f"\n{'='*40}")
        print(f"  {classification.upper()}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Risk: {risk_level}")
        print(f"{'='*40}\n")
        
        # OLED
        if self.oled:
            self._update_oled(classification, confidence, risk_level)
        
        # LEDs
        if self.gpio_available:
            self._update_leds(risk_level)
    
    def _update_oled(self, classification: str, confidence: float, risk_level: str):
        """Update OLED display"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            image = Image.new('1', (self.oled.width, self.oled.height))
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()
                font_small = font
            
            draw.text((0, 0), "Baby Cry Monitor", font=font_small, fill=255)
            draw.text((0, 16), classification.upper(), font=font, fill=255)
            draw.text((0, 32), f"Conf: {confidence:.0f}%", font=font_small, fill=255)
            draw.text((0, 48), f"Risk: {risk_level}", font=font_small, fill=255)
            
            self.oled.display(image)
        except Exception as e:
            logger.warning(f"OLED update failed: {e}")
    
    def _update_leds(self, risk_level: str):
        """Update LED indicators"""
        try:
            import RPi.GPIO as GPIO
            
            GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.LOW)
            
            if risk_level == "GREEN":
                GPIO.output(self.LED_GREEN, GPIO.HIGH)
            elif risk_level == "YELLOW":
                GPIO.output(self.LED_YELLOW, GPIO.HIGH)
            elif risk_level == "RED":
                GPIO.output(self.LED_RED, GPIO.HIGH)
        except:
            pass
    
    def cleanup(self):
        """Cleanup GPIO"""
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except:
            pass


# ==============================================================================
# Web Dashboard
# ==============================================================================

class WebDashboard:
    """
    Simple web dashboard for viewing results
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = None
        self.history = deque(maxlen=100)
        self.latest_result = None
    
    def setup(self, classifier: LightweightCryClassifier):
        """Setup FastAPI app"""
        try:
            from fastapi import FastAPI, WebSocket
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles
            import uvicorn
            
            self.app = FastAPI(title="Baby Cry Monitor")
            self.classifier = classifier
            
            @self.app.get("/", response_class=HTMLResponse)
            async def index():
                return self._get_html()
            
            @self.app.get("/api/latest")
            async def get_latest():
                return self.latest_result or {"status": "waiting"}
            
            @self.app.get("/api/history")
            async def get_history():
                return list(self.history)
            
            @self.app.get("/health")
            async def health():
                return {
                    "status": "healthy",
                    "model_loaded": classifier.is_initialized,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"Web dashboard ready at http://{self.host}:{self.port}")
            return True
            
        except ImportError as e:
            logger.warning(f"Web dashboard not available: {e}")
            return False
    
    def update(self, result: Dict[str, Any]):
        """Update with new result"""
        result["timestamp"] = datetime.now().isoformat()
        self.latest_result = result
        self.history.append(result)
    
    def run(self):
        """Run web server in background thread"""
        if self.app is None:
            return
        
        import uvicorn
        
        def run_server():
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning"
            )
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info(f"Web server started on port {self.port}")
    
    def _get_html(self) -> str:
        """Get dashboard HTML with all 16 classifications"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Baby Cry & Respiratory Monitor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f2027 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; font-size: 2.2em; }
        .subtitle { text-align: center; opacity: 0.7; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card-full { grid-column: 1 / -1; }
        .status-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        }
        .main-status {
            text-align: center;
            padding: 30px 0;
        }
        .status-label {
            font-size: 4em;
            font-weight: 800;
            margin: 10px 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .status-type {
            font-size: 1.2em;
            opacity: 0.8;
            margin-bottom: 10px;
        }
        .GREEN { color: #4ade80; text-shadow: 0 0 30px rgba(74, 222, 128, 0.5); }
        .YELLOW { color: #fbbf24; text-shadow: 0 0 30px rgba(251, 191, 36, 0.5); }
        .RED { color: #f87171; text-shadow: 0 0 30px rgba(248, 113, 113, 0.5); animation: pulse-red 1s infinite; }
        @keyframes pulse-red {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
        @media (max-width: 600px) { .metrics { grid-template-columns: repeat(2, 1fr); } }
        .metric {
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        .metric-value { font-size: 1.8em; font-weight: bold; }
        .metric-label { font-size: 0.85em; opacity: 0.7; margin-top: 5px; }
        .section-title { font-size: 1.3em; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        .section-title span { font-size: 1.2em; }
        .class-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        @media (max-width: 900px) { .class-grid { grid-template-columns: repeat(2, 1fr); } }
        .class-item {
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            font-size: 0.9em;
            background: rgba(0,0,0,0.2);
            transition: all 0.3s;
        }
        .class-item.active { transform: scale(1.05); }
        .class-item.active.GREEN { background: rgba(74, 222, 128, 0.3); border: 2px solid #4ade80; }
        .class-item.active.YELLOW { background: rgba(251, 191, 36, 0.3); border: 2px solid #fbbf24; }
        .class-item.active.RED { background: rgba(248, 113, 113, 0.3); border: 2px solid #f87171; }
        .history-list { max-height: 250px; overflow-y: auto; }
        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            background: rgba(0,0,0,0.2);
        }
        .history-class { font-weight: 600; }
        .history-time { opacity: 0.6; font-size: 0.9em; }
        .history-conf { font-size: 0.85em; opacity: 0.8; }
        .waiting { animation: pulse 2s infinite; opacity: 0.5; }
        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        .backbone-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
        .backbone-pill {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            background: rgba(99, 102, 241, 0.3);
            border: 1px solid rgba(99, 102, 241, 0.5);
        }
        .risk-bar {
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #4ade80 0%, #fbbf24 50%, #f87171 100%);
            margin-top: 15px;
            position: relative;
        }
        .risk-indicator {
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            position: absolute;
            top: -4px;
            transform: translateX(-50%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: left 0.5s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>👶 Baby Cry & Respiratory Monitor</h1>
        <p class="subtitle">6-Backbone AI Ensemble | 20 Classifications | Cry-Based Disease Detection | INMP441 Mic</p>
        
        <div class="grid">
            <!-- Main Status Card -->
            <div class="card card-full status-card">
                <div class="main-status">
                    <div class="status-type" id="taskType">Waiting for audio...</div>
                    <div class="status-label waiting" id="mainStatus">LISTENING</div>
                    <div id="confidence-display" style="font-size: 1.5em; opacity: 0.8;">--</div>
                </div>
                <div class="risk-bar">
                    <div class="risk-indicator" id="riskIndicator" style="left: 20%;"></div>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="riskLevel">--</div>
                        <div class="metric-label">Risk Level</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="riskScore">--</div>
                        <div class="metric-label">Risk Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="modelCount">--</div>
                        <div class="metric-label">Backbones</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="totalScans">0</div>
                        <div class="metric-label">Total Scans</div>
                    </div>
                </div>
            </div>
            
            <!-- Baby Cry Classes -->
            <div class="card">
                <div class="section-title"><span>👶</span> Baby Cry Classifications</div>
                <div class="class-grid" id="cryClasses">
                    <div class="class-item" data-class="hungry_cry">Hungry</div>
                    <div class="class-item" data-class="pain_cry">Pain</div>
                    <div class="class-item" data-class="sleepy_cry">Sleepy</div>
                    <div class="class-item" data-class="discomfort_cry">Discomfort</div>
                    <div class="class-item" data-class="cold_cry">Cold/Hot</div>
                    <div class="class-item" data-class="tired_cry">Tired</div>
                    <div class="class-item" data-class="normal_cry">Normal</div>
                    <div class="class-item" data-class="distress_cry">Distress</div>
                    <div class="class-item" data-class="belly_pain_cry">Belly Pain</div>
                    <div class="class-item" data-class="burping_cry">Burping</div>
                    <div class="class-item" data-class="pathological_cry">⚠️ Pathological</div>
                    <div class="class-item" data-class="asphyxia_cry">🚨 Asphyxia</div>
                </div>
            </div>

            <!-- Health/Disease Detection from Cry -->
            <div class="card">
                <div class="section-title"><span>🩺</span> Health Detection (from Cry & Sound)</div>
                <div class="class-grid" id="healthClasses">
                    <div class="class-item" data-class="sepsis_cry">🚨 Sepsis</div>
                    <div class="class-item" data-class="colic_cry">Colic</div>
                    <div class="class-item" data-class="pneumonia_cry">🚨 Pneumonia</div>
                    <div class="class-item" data-class="respiratory_distress">⚠️ Resp Distress</div>
                    <div class="class-item" data-class="normal_breathing">Normal Breathing</div>
                    <div class="class-item" data-class="wheezing">Wheezing</div>
                    <div class="class-item" data-class="stridor">⚠️ Stridor</div>
                    <div class="class-item" data-class="bronchiolitis">🚨 Bronchiolitis</div>
                </div>
            </div>
            
            <!-- History -->
            <div class="card card-full">
                <div class="section-title"><span>📊</span> Detection History</div>
                <div class="history-list" id="history">
                    <p style="opacity: 0.5; text-align: center; padding: 20px;">Waiting for first detection...</p>
                </div>
            </div>
            
            <!-- Model Info -->
            <div class="card card-full">
                <div class="section-title"><span>🤖</span> Active AI Models (Equal Weights)</div>
                <div class="backbone-pills" id="backbones">
                    <span class="backbone-pill">Loading models...</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const riskColors = { GREEN: '#4ade80', YELLOW: '#fbbf24', RED: '#f87171' };
        let totalScans = 0;
        
        async function update() {
            try {
                const resp = await fetch('/api/latest');
                const data = await resp.json();
                
                if (data.classification) {
                    totalScans++;
                    
                    // Update main status
                    const label = data.display_label || data.classification;
                    document.getElementById('mainStatus').textContent = label;
                    document.getElementById('mainStatus').className = 'status-label ' + data.risk_level;
                    
                    // Task type
                    const taskType = data.task === 'cry' ? '👶 Baby Cry Detected' : '🫁 Respiratory Sound Detected';
                    document.getElementById('taskType').textContent = taskType;
                    
                    // Confidence
                    document.getElementById('confidence-display').textContent = 
                        'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
                    
                    // Metrics
                    document.getElementById('riskLevel').textContent = data.risk_level;
                    document.getElementById('riskLevel').className = 'metric-value ' + data.risk_level;
                    document.getElementById('riskScore').textContent = data.risk_score.toFixed(0);
                    document.getElementById('totalScans').textContent = totalScans;
                    
                    // Risk indicator
                    const riskPos = Math.min(100, Math.max(0, data.risk_score));
                    document.getElementById('riskIndicator').style.left = riskPos + '%';
                    
                    // Model count
                    if (data.backbone_results) {
                        const count = Object.keys(data.backbone_results).length;
                        document.getElementById('modelCount').textContent = count;
                        
                        // Update backbone pills
                        const pills = Object.entries(data.backbone_results).map(([name, res]) => 
                            `<span class="backbone-pill">${name}: ${(res.confidence * 100).toFixed(0)}%</span>`
                        ).join('');
                        document.getElementById('backbones').innerHTML = pills || '<span class="backbone-pill">Ensemble Active</span>';
                    }
                    
                    // Highlight active class
                    document.querySelectorAll('.class-item').forEach(el => {
                        el.classList.remove('active', 'GREEN', 'YELLOW', 'RED');
                        if (el.dataset.class === data.classification) {
                            el.classList.add('active', data.risk_level);
                        }
                    });
                }
                
                // Update history
                const histResp = await fetch('/api/history');
                const history = await histResp.json();
                
                if (history.length > 0) {
                    const historyHtml = history.slice(-8).reverse().map(item => {
                        const label = item.display_label || item.classification;
                        const icon = item.task === 'cry' ? '👶' : '🫁';
                        return `
                            <div class="history-item">
                                <span class="history-class ${item.risk_level}">${icon} ${label}</span>
                                <span class="history-conf">${(item.confidence * 100).toFixed(0)}%</span>
                                <span class="history-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
                            </div>
                        `;
                    }).join('');
                    document.getElementById('history').innerHTML = historyHtml;
                }
            } catch (e) {
                console.error('Update failed:', e);
            }
        }
        
        setInterval(update, 2000);
        update();
    </script>
</body>
</html>
'''


# ==============================================================================
# Main Application
# ==============================================================================

async def main(config: dict):
    """Main application loop"""
    logger.info("=" * 50)
    logger.info("BABY CRY MONITOR - RPi5 Standalone")
    logger.info("=" * 50)
    
    # Initialize components
    classifier = SixBackboneEnsemble(
        use_quantization=config["model"].get("use_quantization", True)
    )
    await classifier.initialize()
    
    audio = I2SAudioCapture(
        sample_rate=config["audio"]["sample_rate"],
        channels=config["audio"]["channels"],
        chunk_duration=config["audio"]["chunk_duration"],
        device_name=config["audio"].get("device_name")
    )
    
    if not audio.initialize():
        logger.error("Failed to initialize audio capture")
        return
    
    display = DisplayOutput(config["display"])
    
    # Setup web dashboard
    web = None
    if config["web"].get("enabled", True):
        web = WebDashboard(
            host=config["web"].get("host", "0.0.0.0"),
            port=config["web"].get("port", 8080)
        )
        if web.setup(classifier):
            web.run()
    
    logger.info("")
    logger.info("System ready. Monitoring for baby cries...")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    try:
        while True:
            # Record audio
            logger.info("Recording...")
            waveform = audio.record()
            
            # Check for actual audio (not silence)
            energy = np.sqrt(np.mean(waveform**2))
            if energy < 0.01:
                logger.info("  [Silence - skipping]")
                await asyncio.sleep(1)
                continue
            
            # Classify
            logger.info("Classifying...")
            result = await classifier.classify(waveform, config["audio"]["sample_rate"])
            
            # Display
            display.show(result)
            
            # Update web dashboard
            if web:
                web.update(result)
            
            # Brief pause
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        audio.cleanup()
        display.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baby Cry Monitor - RPi5 Standalone")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--port", type=int, help="Web server port (default: 8080)")
    parser.add_argument("--no-web", action="store_true", help="Disable web dashboard")
    args = parser.parse_args()
    
    config = load_config()
    
    if args.port:
        config["web"]["port"] = args.port
    if args.no_web:
        config["web"]["enabled"] = False
    
    asyncio.run(main(config))
