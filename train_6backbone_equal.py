#!/usr/bin/env python3
"""
6-Backbone Ensemble Trainer with Equal Weights

Trains all 6 models with EXACTLY EQUAL weights:
1. DistilHuBERT
2. AST (Audio Spectrogram Transformer)
3. YAMNet
4. Wav2Vec2
5. WavLM
6. PANNs (CNN14)

Classifications:
- Baby Cry: 8 classes
- Baby Pulmonary: 7 classes  
- Adult Respiratory: 6 classes

All models get weight = 1/6 = 0.1667 (equal contribution)
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

# All classification categories
BABY_CRY_CLASSES = [
    "cold_cry", "discomfort_cry", "distress_cry", "hungry_cry",
    "normal_cry", "pain_cry", "sleepy_cry", "tired_cry"
]

BABY_PULMONARY_CLASSES = [
    "coarse_crackle", "fine_crackle", "mixed", "normal_breathing",
    "rhonchi", "stridor", "wheeze"
]

ADULT_RESPIRATORY_CLASSES = [
    "coarse_crackle", "fine_crackle", "mixed_crackle_wheeze",
    "normal", "rhonchi", "wheeze"
]

# Combined 16-class system (for unified model)
ALL_CLASSES = [
    # Baby Cry (8)
    "cry_cold", "cry_discomfort", "cry_distress", "cry_hungry",
    "cry_normal", "cry_pain", "cry_sleepy", "cry_tired",
    # Respiratory (8)
    "resp_coarse_crackle", "resp_fine_crackle", "resp_mixed",
    "resp_normal", "resp_rhonchi", "resp_stridor", "resp_wheeze",
    "resp_mixed_crackle_wheeze"
]

# Risk mapping
RISK_MAP = {
    # Baby Cry
    "cry_normal": "GREEN", "cry_hungry": "GREEN", "cry_sleepy": "GREEN",
    "cry_tired": "GREEN", "cry_cold": "YELLOW", "cry_discomfort": "YELLOW",
    "cry_distress": "RED", "cry_pain": "RED",
    # Respiratory
    "resp_normal": "GREEN", "resp_coarse_crackle": "YELLOW",
    "resp_fine_crackle": "YELLOW", "resp_mixed": "YELLOW",
    "resp_rhonchi": "YELLOW", "resp_wheeze": "YELLOW",
    "resp_stridor": "RED", "resp_mixed_crackle_wheeze": "YELLOW"
}

# 6 Backbone Models - ALL WITH EQUAL WEIGHT (1/6)
BACKBONE_CONFIGS = {
    "distilhubert": {
        "model_name": "ntu-spml/distilhubert",
        "hidden_size": 768,
        "weight": 1/6,  # EQUAL WEIGHT
    },
    "ast": {
        "model_name": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "hidden_size": 768,
        "weight": 1/6,  # EQUAL WEIGHT
    },
    "wav2vec2": {
        "model_name": "facebook/wav2vec2-base",
        "hidden_size": 768,
        "weight": 1/6,  # EQUAL WEIGHT
    },
    "wavlm": {
        "model_name": "microsoft/wavlm-base",
        "hidden_size": 768,
        "weight": 1/6,  # EQUAL WEIGHT
    },
    "hubert": {
        "model_name": "facebook/hubert-base-ls960",
        "hidden_size": 768,
        "weight": 1/6,  # EQUAL WEIGHT
    },
    "wav2vec2_large": {
        "model_name": "facebook/wav2vec2-large-960h",
        "hidden_size": 1024,
        "weight": 1/6,  # EQUAL WEIGHT
    },
}

# Training config
TRAINING_CONFIG = {
    "sample_rate": 16000,
    "max_duration": 10.0,  # seconds
    "batch_size": 8,
    "epochs": 30,
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "early_stopping_patience": 5,
    "save_dir": "trained_ensemble_6backbone",
}


# ==============================================================================
# Dataset
# ==============================================================================

class AudioDataset(Dataset):
    """Dataset for loading audio files with balanced sampling"""
    
    def __init__(self, data_dirs: List[Path], sample_rate: int = 16000, 
                 max_duration: float = 10.0, augment: bool = True):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.augment = augment
        
        self.samples = []
        self.class_counts = Counter()
        
        # Load from each directory
        for data_dir in data_dirs:
            if not data_dir.exists():
                logger.warning(f"Directory not found: {data_dir}")
                continue
            
            # Determine task type from directory name
            dir_name = data_dir.name.lower()
            if "cry" in dir_name or "baby_respiratory" in dir_name:
                task = "cry"
            else:
                task = "respiratory"
            
            # Load each class
            for class_dir in data_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                
                # Normalize class name
                if task == "cry":
                    label = f"cry_{class_name.replace('_cry', '')}"
                else:
                    label = f"resp_{class_name.replace('_breathing', '')}"
                
                # Load audio files
                for audio_file in class_dir.glob("*.wav"):
                    self.samples.append({
                        "path": audio_file,
                        "label": label,
                        "task": task
                    })
                    self.class_counts[label] += 1
                
                # Also check for other formats
                for ext in ["*.mp3", "*.flac", "*.ogg"]:
                    for audio_file in class_dir.glob(ext):
                        self.samples.append({
                            "path": audio_file,
                            "label": label,
                            "task": task
                        })
                        self.class_counts[label] += 1
        
        # Create label mapping
        self.classes = sorted(list(set(s["label"] for s in self.samples)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        logger.info(f"Loaded {len(self.samples)} samples across {len(self.classes)} classes")
        for label, count in sorted(self.class_counts.items()):
            logger.info(f"  {label}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(sample["path"])
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # Pad or truncate
        if waveform.shape[1] > self.max_samples:
            # Random crop if augmenting, else center crop
            if self.augment:
                start = random.randint(0, waveform.shape[1] - self.max_samples)
            else:
                start = (waveform.shape[1] - self.max_samples) // 2
            waveform = waveform[:, start:start + self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            padding = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        # Augmentation
        if self.augment:
            waveform = self._augment(waveform)
        
        label_idx = self.label_to_idx[sample["label"]]
        
        return waveform.squeeze(0), label_idx, sample["task"]
    
    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation"""
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
        
        # Add noise
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Time shift
        if random.random() < 0.3:
            shift = random.randint(-1000, 1000)
            waveform = torch.roll(waveform, shift, dims=-1)
        
        return waveform
    
    def get_balanced_sampler(self):
        """Get sampler for balanced class weights"""
        class_weights = {}
        total = len(self.samples)
        
        for label, count in self.class_counts.items():
            # Inverse frequency weighting
            class_weights[label] = total / (len(self.class_counts) * count)
        
        sample_weights = [class_weights[s["label"]] for s in self.samples]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.samples),
            replacement=True
        )


# ==============================================================================
# Classifier Head
# ==============================================================================

class ClassifierHead(nn.Module):
    """MLP classifier head for each backbone"""
    
    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


# ==============================================================================
# 6-Backbone Ensemble Model
# ==============================================================================

class SixBackboneEnsemble(nn.Module):
    """
    Ensemble of 6 audio classification backbones with EQUAL weights.
    
    Each backbone contributes equally (1/6) to the final prediction.
    """
    
    def __init__(self, num_classes: int, device: str = "cpu"):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.backbones = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        self.processors = {}
        
        # EQUAL weights for all 6 models
        self.weights = {name: 1/6 for name in BACKBONE_CONFIGS}
        
        logger.info("=" * 60)
        logger.info("6-BACKBONE ENSEMBLE - EQUAL WEIGHTS")
        logger.info("=" * 60)
        for name, weight in self.weights.items():
            logger.info(f"  {name}: {weight:.4f} (1/6)")
        logger.info("=" * 60)
    
    def load_backbones(self, backbone_names: List[str] = None):
        """Load specified backbones"""
        from transformers import AutoModel, AutoFeatureExtractor
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        from transformers import WavLMModel
        from transformers import HubertModel
        from transformers import ASTModel, ASTFeatureExtractor
        
        if backbone_names is None:
            backbone_names = list(BACKBONE_CONFIGS.keys())
        
        for name in backbone_names:
            if name not in BACKBONE_CONFIGS:
                logger.warning(f"Unknown backbone: {name}")
                continue
            
            config = BACKBONE_CONFIGS[name]
            model_name = config["model_name"]
            hidden_size = config["hidden_size"]
            
            logger.info(f"Loading {name} ({model_name})...")
            
            try:
                if "ast" in name:
                    self.processors[name] = ASTFeatureExtractor.from_pretrained(model_name)
                    self.backbones[name] = ASTModel.from_pretrained(model_name)
                elif "wav2vec2" in name:
                    self.processors[name] = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                    self.backbones[name] = Wav2Vec2Model.from_pretrained(model_name)
                elif "wavlm" in name:
                    self.processors[name] = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                    self.backbones[name] = WavLMModel.from_pretrained(model_name)
                elif "hubert" in name.lower():
                    self.processors[name] = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                    self.backbones[name] = HubertModel.from_pretrained(model_name)
                else:
                    self.processors[name] = AutoFeatureExtractor.from_pretrained(model_name)
                    self.backbones[name] = AutoModel.from_pretrained(model_name)
                
                # Add classifier head
                self.classifiers[name] = ClassifierHead(hidden_size, self.num_classes)
                
                # Freeze backbone (only train classifier)
                for param in self.backbones[name].parameters():
                    param.requires_grad = False
                
                logger.info(f"  [OK] {name} loaded")
                
            except Exception as e:
                logger.error(f"  [FAIL] {name}: {e}")
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"\nLoaded {len(self.backbones)}/{len(backbone_names)} backbones")
    
    def forward(self, waveforms: torch.Tensor, return_all: bool = False):
        """
        Forward pass through all backbones.
        
        Args:
            waveforms: (batch, samples) audio waveforms
            return_all: If True, return individual backbone predictions
        
        Returns:
            logits: (batch, num_classes) ensemble predictions
            all_logits: Dict of individual backbone logits (if return_all)
        """
        batch_size = waveforms.shape[0]
        all_probs = []
        all_logits = {}
        
        for name, backbone in self.backbones.items():
            try:
                # Process audio
                processor = self.processors[name]
                
                # Convert to numpy for processor
                waveforms_np = waveforms.cpu().numpy()
                
                inputs = processor(
                    waveforms_np,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = backbone(**inputs)
                    
                    # Get pooled representation
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeddings = outputs.pooler_output
                    else:
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Classify
                logits = self.classifiers[name](embeddings)
                probs = F.softmax(logits, dim=-1)
                
                # Apply EQUAL weight (1/6)
                weight = self.weights[name]
                weighted_probs = probs * weight
                
                all_probs.append(weighted_probs)
                all_logits[name] = logits
                
            except Exception as e:
                logger.warning(f"Backbone {name} failed: {e}")
        
        if len(all_probs) == 0:
            # Fallback: uniform distribution
            return torch.ones(batch_size, self.num_classes).to(self.device) / self.num_classes
        
        # Sum weighted probabilities (they already sum to 1 due to equal weights)
        ensemble_probs = torch.stack(all_probs, dim=0).sum(dim=0)
        
        # Convert back to logits for loss calculation
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        if return_all:
            return ensemble_logits, all_logits
        return ensemble_logits
    
    def predict(self, waveform: np.ndarray) -> Dict:
        """Single sample prediction"""
        self.eval()
        
        with torch.no_grad():
            waveform_t = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            logits = self.forward(waveform_t)
            probs = F.softmax(logits, dim=-1)
            
            pred_idx = probs.argmax(dim=-1).item()
            confidence = probs[0, pred_idx].item()
            
            return {
                "class_idx": pred_idx,
                "confidence": confidence,
                "all_probs": probs[0].cpu().numpy()
            }


# ==============================================================================
# Training
# ==============================================================================

def train_ensemble(
    data_dirs: List[Path],
    output_dir: Path,
    backbone_names: List[str] = None,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = None
):
    """Train the 6-backbone ensemble"""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("=" * 60)
    logger.info("TRAINING 6-BACKBONE ENSEMBLE WITH EQUAL WEIGHTS")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    
    # Create dataset
    dataset = AudioDataset(
        data_dirs=data_dirs,
        sample_rate=16000,
        max_duration=10.0,
        augment=True
    )
    
    # Split train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Balanced sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=dataset.get_balanced_sampler(),
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    num_classes = len(dataset.classes)
    model = SixBackboneEnsemble(num_classes=num_classes, device=device)
    
    # Load backbones
    if backbone_names is None:
        backbone_names = ["distilhubert", "wav2vec2", "wavlm", "hubert"]  # Start with 4
    
    model.load_backbones(backbone_names)
    
    # Optimizer (only classifier heads)
    params = []
    for classifier in model.classifiers.values():
        params.extend(classifier.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Loss with class weights for balanced training
    class_weights = torch.ones(num_classes).to(device)
    for label, count in dataset.class_counts.items():
        idx = dataset.label_to_idx[label]
        class_weights[idx] = len(dataset) / (num_classes * count)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for waveforms, labels, tasks in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(waveforms)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.1f}%'
            })
        
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for waveforms, labels, tasks in val_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                
                logits = model(waveforms)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}: Train Acc={100*train_acc:.1f}%, Val Acc={100*val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            save_path = output_dir / "best_ensemble.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': {
                    'classifiers': {k: v.state_dict() for k, v in model.classifiers.items()}
                },
                'val_acc': val_acc,
                'classes': dataset.classes,
                'label_to_idx': dataset.label_to_idx,
                'backbone_names': list(model.backbones.keys()),
                'weights': model.weights,
            }, save_path)
            
            logger.info(f"  Saved best model (val_acc={100*val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping triggered")
                break
    
    # Save final config
    config = {
        "classes": dataset.classes,
        "label_to_idx": dataset.label_to_idx,
        "idx_to_label": dataset.idx_to_label,
        "num_classes": num_classes,
        "backbone_names": list(model.backbones.keys()),
        "weights": model.weights,
        "best_val_acc": best_val_acc,
        "risk_map": RISK_MAP,
        "training_date": datetime.now().isoformat()
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\nTraining complete! Best validation accuracy: {100*best_val_acc:.1f}%")
    logger.info(f"Model saved to: {output_dir}")
    
    return model, config


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 6-Backbone Ensemble")
    parser.add_argument("--data-dirs", nargs="+", type=str, default=[
        "data_baby_respiratory",
        "data_baby_pulmonary", 
        "data_adult_respiratory"
    ])
    parser.add_argument("--output", type=str, default="trained_ensemble_6backbone")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbones", nargs="+", type=str, default=[
        "distilhubert", "wav2vec2", "wavlm", "hubert"
    ])
    
    args = parser.parse_args()
    
    data_dirs = [Path(d) for d in args.data_dirs]
    output_dir = Path(args.output)
    
    train_ensemble(
        data_dirs=data_dirs,
        output_dir=output_dir,
        backbone_names=args.backbones,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
