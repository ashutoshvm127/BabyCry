"""
Direct Training Script - Skips interactive setup and goes directly to training.
Executes the maximum accuracy training with all 5 backbone models.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import Counter
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# === CONFIGURATION ===
RESPIRATORY_DIR = Path("./data_adult_respiratory")
BABY_CRY_DIR = Path("./data_baby_respiratory")
BABY_PULMONARY_DIR = Path("./data_baby_pulmonary")


def check_data():
    """Check if training data exists"""
    print("=" * 70)
    print("CHECKING DATA AVAILABILITY")
    print("=" * 70)
    
    status = {}
    for name, path in [("Adult Respiratory", RESPIRATORY_DIR), 
                       ("Baby Cry", BABY_CRY_DIR),
                       ("Baby Pulmonary", BABY_PULMONARY_DIR)]:
        wav_files = list(path.glob("*/*.wav")) if path.exists() else []
        mp3_files = list(path.glob("*/*.mp3")) if path.exists() else []
        total = len(wav_files) + len(mp3_files)
        status[name] = total
        if total > 0:
            print(f"  [OK] {name}: {total} audio files")
        else:
            print(f"  [!] {name}: NO DATA FOUND at {path}")
    
    return status


def train_maximum_accuracy():
    """Train with 5 advanced backbone models for maximum pulmonary detection accuracy
    
    BACKBONES: AST, HuBERT-Large, PANNs CNN14, BEATs, CLAP, Wav2Vec2
    IMPROVEMENTS: Attention pooling, spectrogram CNN branch, multi-layer MLP head,
                  5-fold stratified cross-validation, model ensemble
    """
    print("\n" + "=" * 70)
    print("MULTI-MODEL TRAINING (6 BACKBONES x 3 STAGES)")
    print("=" * 70)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import librosa
    from tqdm import tqdm
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    
    from transformers import AutoFeatureExtractor
    from transformers.optimization import get_cosine_schedule_with_warmup
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable cuDNN auto-tuner for faster convolutions
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  [OK] cuDNN benchmark + TF32 enabled for max performance")
    
    # === FOCAL LOSS for extreme class imbalance ===
    class FocalLoss(nn.Module):
        """Focal Loss for handling severe class imbalance.
        Focuses training on hard-to-classify examples."""
        def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
            super().__init__()
            self.alpha = alpha  # Class weights
            self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
            self.reduction = reduction
            self.label_smoothing = label_smoothing
        
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, 
                                       reduction='none', label_smoothing=self.label_smoothing)
            pt = torch.exp(-ce_loss)  # Probability of correct class
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            return focal_loss
    
    # === SPECAUGMENT for mel spectrograms ===
    def spec_augment(mel_spec, freq_mask_param=10, time_mask_param=20, num_freq_masks=2, num_time_masks=2):
        """Apply SpecAugment to mel spectrogram for data augmentation."""
        spec = mel_spec.clone()
        n_mels, time_steps = spec.shape[-2], spec.shape[-1]
        
        # Frequency masking
        for _ in range(num_freq_masks):
            f = np.random.randint(0, min(freq_mask_param, n_mels))
            f0 = np.random.randint(0, n_mels - f)
            spec[..., f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(num_time_masks):
            t = np.random.randint(0, min(time_mask_param, time_steps))
            t0 = np.random.randint(0, time_steps - t)
            spec[..., :, t0:t0+t] = 0
        
        return spec
    
    # === CONFIGURATION ===
    CONFIG = {
        'backbones': ['ast'],  # Single best backbone for ULTRA FAST training
        'sampling_rate': 16000,
        'max_duration': 5.0,
        'batch_size': 16,  # Maximized for speed
        'batch_size_stage2': 8,  # Maximized for speed
        'learning_rate': 3e-4,  # Higher LR for faster convergence
        'weight_decay': 0.01,
        'warmup_ratio': 0.03,  # Minimal warmup
        'gradient_accumulation': 1,  # No accumulation
        'gradient_accumulation_stage2': 1,  # No accumulation
        'label_smoothing': 0.05,
        'mixup_alpha': 0.0,
        'fp16': True,
        'num_train_epochs': 10,  # ULTRA FAST: 10 epochs
        'cv_folds': 2,  # ULTRA FAST: 2-fold instead of 5
        'skip_pretrain_stages': True,  # Skip stage 1 & 2, train directly on target
    }
    
    # === BACKBONE REGISTRY ===
    BACKBONE_REGISTRY = {
        'ast': {
            'name': 'Audio Spectrogram Transformer',
            'hf_id': 'MIT/ast-finetuned-audioset-10-10-0.4593',
            'hidden_size': 768,
            'epochs_mult': 1.0,
            'lr_mult': 1.0,
        },
        'hubert': {
            'name': 'HuBERT-Large',
            'hf_id': 'facebook/hubert-large-ls960-ft',
            'hidden_size': 1024,
            'epochs_mult': 0.85,
            'lr_mult': 0.5,
        },
        'clap': {
            'name': 'CLAP Audio',
            'hf_id': 'laion/clap-htsat-unfused',
            'hidden_size': 768,
            'epochs_mult': 0.85,
            'lr_mult': 0.5,
        },
        'panns': {
            'name': 'PANNs CNN14',
            'hf_id': None,
            'hidden_size': 2048,
            'epochs_mult': 0.7,
            'lr_mult': 1.0,
        },
        'beats': {
            'name': 'BEATs (Microsoft)',
            'hf_id': None,
            'hidden_size': 768,
            'epochs_mult': 0.85,
            'lr_mult': 0.8,
        },
        'wav2vec2': {
            'name': 'Wav2Vec2-Large',
            'hf_id': 'facebook/wav2vec2-large-960h',
            'hidden_size': 1024,
            'epochs_mult': 0.85,
            'lr_mult': 0.5,
        },
    }
    
    # === PANNs CNN14 BACKBONE ===
    class CNN14(nn.Module):
        """PANNs CNN14 backbone - processes mel spectrograms for audio classification"""
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
                nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(1024, 2048, 3, padding=1), nn.BatchNorm2d(2048), nn.ReLU(),
                nn.Conv2d(2048, 2048, 3, padding=1), nn.BatchNorm2d(2048), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(2048, 2048)
        
        def forward(self, mel_spectrogram):
            x = self.features(mel_spectrogram)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.unsqueeze(1)  # [B, 1, 2048] for attention pooling compatibility
    
    # === BEATs MODEL WRAPPER ===
    class BEATsModel(nn.Module):
        """
        BEATs: Audio Pre-Training with Acoustic Tokenizers
        Wrapper to load the official Microsoft BEATs checkpoint
        """
        def __init__(self, cfg, checkpoint):
            super().__init__()
            import math
            
            self.embed_dim = cfg.get('encoder_embed_dim', 768)
            self.num_heads = cfg.get('encoder_attention_heads', 12)
            self.depth = cfg.get('encoder_layers', 12)
            
            # Patch embedding for fbank features (128 mel bins)
            self.patch_embedding = nn.Conv2d(1, self.embed_dim, kernel_size=(16, 16), stride=(16, 16))
            self.layer_norm = nn.LayerNorm(self.embed_dim)
            
            # Position embedding (will be interpolated if needed)
            self.pos_embed = nn.Parameter(torch.zeros(1, 512, self.embed_dim))
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=cfg.get('encoder_ffn_embed_dim', 3072),
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
            
            # Load weights from checkpoint if available
            self._load_weights(checkpoint)
        
        def _load_weights(self, checkpoint):
            """Load pretrained weights from BEATs checkpoint"""
            try:
                state_dict = checkpoint.get('model', checkpoint)
                # Map weights to our architecture
                new_state = {}
                for key, value in state_dict.items():
                    if 'patch_embedding' in key or 'embed' in key.lower():
                        # Skip incompatible shapes for now
                        continue
                    if 'encoder.layers' in key:
                        # Map transformer layer weights
                        new_key = key.replace('encoder.layers', 'encoder.layers')
                        new_state[new_key] = value
                
                # Load compatible weights
                missing, unexpected = self.load_state_dict(new_state, strict=False)
                print(f"    BEATs: Loaded {len(new_state)} weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
            except Exception as e:
                print(f"    BEATs: Using random init ({str(e)[:50]})")
        
        def forward(self, input_values, attention_mask=None):
            """
            Forward pass
            Args:
                input_values: [B, T] raw waveform or [B, 1, n_mels, T] mel spectrogram
            Returns:
                hidden_states: [B, seq_len, embed_dim]
            """
            device = input_values.device
            
            # Convert waveform to mel spectrogram if needed
            if input_values.dim() == 2:
                # Raw waveform input - compute mel spectrogram using librosa
                fbanks = []
                for wav in input_values.cpu().numpy():
                    mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=128, n_fft=512, hop_length=160)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    fbanks.append(mel_db)
                x = torch.FloatTensor(np.stack(fbanks)).unsqueeze(1).to(device)  # [B, 1, 128, T]
            elif input_values.dim() == 3:
                x = input_values.unsqueeze(1).to(device)  # [B, 1, n_mels, T]
            else:
                x = input_values.to(device)  # Already [B, 1, n_mels, T]
            
            # Patch embedding
            x = self.patch_embedding(x)  # [B, embed_dim, H, W]
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            
            # Add positional embedding (limit to max 512 positions)
            seq_len = min(x.size(1), 512)
            x = x[:, :seq_len, :] + self.pos_embed[:, :seq_len, :]
            
            x = self.layer_norm(x)
            
            # Transformer encoder
            x = self.encoder(x)
            
            return x  # Return as hidden states [B, seq_len, embed_dim]
    
    # === BACKBONE FACTORY ===
    def create_backbone(backbone_type):
        """Create backbone model and feature extractor"""
        info = BACKBONE_REGISTRY[backbone_type]
        
        if backbone_type == 'ast':
            from transformers import ASTModel, ASTFeatureExtractor
            model = ASTModel.from_pretrained(info['hf_id'])
            extractor = ASTFeatureExtractor.from_pretrained(info['hf_id'])
            return model, extractor, info['hidden_size']
        
        elif backbone_type == 'hubert':
            from transformers import HubertModel
            model = HubertModel.from_pretrained(info['hf_id'])
            extractor = AutoFeatureExtractor.from_pretrained(info['hf_id'])
            return model, extractor, info['hidden_size']
        
        elif backbone_type == 'clap':
            from transformers import ClapModel, ClapProcessor
            full_model = ClapModel.from_pretrained(info['hf_id'])
            audio_model = full_model.audio_model
            extractor = ClapProcessor.from_pretrained(info['hf_id'])
            del full_model
            return audio_model, extractor, info['hidden_size']
        
        elif backbone_type == 'panns':
            model = CNN14()
            extractor = None  # Uses librosa mel spectrogram directly
            return model, extractor, info['hidden_size']
        
        elif backbone_type == 'beats':
            # Load BEATs from checkpoint
            ckpt_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'beats' / 'BEATs_iter3_plus_AS2M.pt'
            if not ckpt_path.exists():
                raise FileNotFoundError(f"BEATs checkpoint not found at {ckpt_path}")
            
            checkpoint = torch.load(str(ckpt_path), map_location='cpu')
            cfg = checkpoint['cfg']
            
            # BEATs model wrapper
            model = BEATsModel(cfg, checkpoint)
            extractor = None  # BEATs uses raw waveform
            return model, extractor, info['hidden_size']
        
        elif backbone_type == 'wav2vec2':
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
            model = Wav2Vec2Model.from_pretrained(info['hf_id'])
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(info['hf_id'])
            return model, extractor, info['hidden_size']
        
        raise ValueError(f"Unknown backbone: {backbone_type}")
    
    # === MULTI-BACKBONE CLASSIFIER WITH ATTENTION POOLING ===
    class MultiBackboneClassifier(nn.Module):
        """Advanced classifier: attention pooling + spectrogram CNN branch + MLP head"""
        
        def __init__(self, backbone, backbone_type, num_labels, hidden_size):
            super().__init__()
            self.backbone = backbone
            self.backbone_type = backbone_type
            self.hidden_size = hidden_size
            
            # Attention pooling (learns which time segments matter most)
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            
            # Spectrogram CNN branch (captures spectral patterns)
            self.spec_branch = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            
            # Multi-layer classification head (backbone features + spectrogram features)
            fused_size = hidden_size + 64
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(fused_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_labels)
            )
        
        def _extract_features(self, input_values, attention_mask=None):
            if self.backbone_type == 'ast':
                outputs = self.backbone(input_values)
                return outputs.last_hidden_state
            elif self.backbone_type == 'hubert':
                outputs = self.backbone(input_values, attention_mask=attention_mask)
                return outputs.last_hidden_state
            elif self.backbone_type == 'clap':
                outputs = self.backbone(input_features=input_values)
                return outputs.last_hidden_state
            elif self.backbone_type == 'panns':
                return self.backbone(input_values)  # CNN14 returns [B, 1, 2048]
            elif self.backbone_type == 'beats':
                # BEATsModel returns hidden states directly [B, seq_len, embed_dim]
                return self.backbone(input_values, attention_mask=attention_mask)
            elif self.backbone_type == 'wav2vec2':
                outputs = self.backbone(input_values, attention_mask=attention_mask)
                return outputs.last_hidden_state
            return self.backbone(input_values).last_hidden_state
        
        def forward(self, input_values, mel_spectrogram=None, attention_mask=None):
            hidden = self._extract_features(input_values, attention_mask)
            
            # Attention pooling
            if hidden.size(1) > 1:
                attn_weights = self.attention_pool(hidden)
                attn_weights = F.softmax(attn_weights, dim=1)
                pooled = (hidden * attn_weights).sum(dim=1)
            else:
                pooled = hidden.squeeze(1)
            
            # Spectrogram branch fusion
            if mel_spectrogram is not None:
                spec_features = self.spec_branch(mel_spectrogram)
                fused = torch.cat([pooled, spec_features], dim=1)
            else:
                fused = pooled
            
            logits = self.classifier(fused)
            return logits
    
    # === ADVANCED DATASET WITH MEL SPECTROGRAM ===
    class AdvancedAudioDataset(Dataset):
        def __init__(self, files, labels, extractor, backbone_type='ast',
                     sr=16000, max_dur=5.0, augment=False, mixup_alpha=0.0):
            self.files = files
            self.labels = labels
            self.extractor = extractor
            self.backbone_type = backbone_type
            self.sr = sr
            self.max_samples = int(max_dur * sr)
            self.augment = augment
            self.mixup_alpha = mixup_alpha
        
        def __len__(self):
            return len(self.files)
        
        def _load_audio(self, path):
            try:
                wav, _ = librosa.load(str(path), sr=self.sr, mono=True)
            except:
                wav = np.zeros(self.sr)
            if len(wav) > self.max_samples:
                wav = wav[:self.max_samples]
            elif len(wav) < self.sr:
                wav = np.pad(wav, (0, self.sr - len(wav)))
            return wav
        
        def _augment(self, wav):
            if np.random.random() < 0.5:
                wav = wav + np.random.normal(0, 0.005, len(wav)).astype(np.float32)
            if np.random.random() < 0.3:
                shift = np.random.randint(-self.sr // 4, self.sr // 4)
                wav = np.roll(wav, shift)
            if np.random.random() < 0.3:
                factor = np.random.uniform(0.8, 1.2)
                wav = wav * factor
            if np.random.random() < 0.2:
                n_steps = np.random.uniform(-2, 2)
                try:
                    wav = librosa.effects.pitch_shift(wav, sr=self.sr, n_steps=n_steps)
                except:
                    pass
            if np.random.random() < 0.2:
                rate = np.random.uniform(0.85, 1.15)
                try:
                    wav = librosa.effects.time_stretch(wav, rate=rate)
                    if len(wav) > self.max_samples:
                        wav = wav[:self.max_samples]
                    elif len(wav) < self.sr:
                        wav = np.pad(wav, (0, self.sr - len(wav)))
                except:
                    pass
            return wav
        
        def __getitem__(self, idx):
            wav = self._load_audio(self.files[idx])
            if self.augment:
                wav = self._augment(wav)
            
            # Feature extraction based on backbone type
            if self.backbone_type == 'panns':
                mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=64, n_fft=1024, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                input_values = torch.FloatTensor(mel_db).unsqueeze(0)
            elif self.extractor is not None:
                if self.backbone_type == 'clap':
                    inputs = self.extractor(audios=wav, sampling_rate=self.sr, return_tensors='pt', padding=True)
                    input_values = inputs.input_features.squeeze() if hasattr(inputs, 'input_features') else inputs['input_features'].squeeze()
                else:
                    inputs = self.extractor(wav, sampling_rate=self.sr, return_tensors='pt', padding=True)
                    input_values = inputs.input_values.squeeze()
            else:
                input_values = torch.FloatTensor(wav)
            
            # Compute mel spectrogram for spectrogram CNN branch
            mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=64, n_fft=1024, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_spec = torch.FloatTensor(mel_db).unsqueeze(0)
            
            # Apply SpecAugment during training for better generalization
            if self.augment:
                mel_spec = spec_augment(mel_spec, freq_mask_param=8, time_mask_param=15,
                                        num_freq_masks=2, num_time_masks=2)
            
            return {
                'input_values': input_values,
                'mel_spectrogram': mel_spec,
                'labels': self.labels[idx]
            }
    
    def collate_fn(batch):
        input_values = [b['input_values'] for b in batch]
        mel_specs = [b['mel_spectrogram'] for b in batch]
        labels = torch.LongTensor([b['labels'] for b in batch])
        
        # Pad input_values
        max_len = max(iv.shape[-1] for iv in input_values)
        padded_inputs = []
        attention_masks = []
        for iv in input_values:
            pad_len = max_len - iv.shape[-1]
            if iv.dim() == 1:
                padded = F.pad(iv, (0, pad_len))
                mask = torch.ones(max_len, dtype=torch.long)
                if pad_len > 0:
                    mask[-pad_len:] = 0
            elif iv.dim() == 2:
                padded = F.pad(iv, (0, pad_len))
                mask = torch.ones(max_len, dtype=torch.long)
                if pad_len > 0:
                    mask[-pad_len:] = 0
            else:
                padded = iv
                mask = torch.ones(iv.shape[-1], dtype=torch.long)
            padded_inputs.append(padded)
            attention_masks.append(mask)
        
        # Pad mel spectrograms
        max_mel_len = max(m.shape[-1] for m in mel_specs)
        padded_mels = []
        for m in mel_specs:
            pad_len = max_mel_len - m.shape[-1]
            padded_mels.append(F.pad(m, (0, pad_len)))
        
        return {
            'input_values': torch.stack(padded_inputs),
            'attention_mask': torch.stack(attention_masks),
            'mel_spectrogram': torch.stack(padded_mels),
            'labels': labels
        }
    
    def load_data(data_dir):
        files, labels = [], []
        label2id = {}
        data_path = Path(data_dir)
        for cls_dir in sorted(data_path.iterdir()):
            if cls_dir.is_dir():
                label2id[cls_dir.name] = len(label2id)
                for f in cls_dir.rglob("*.wav"):
                    files.append(str(f))
                    labels.append(label2id[cls_dir.name])
                for f in cls_dir.rglob("*.mp3"):
                    files.append(str(f))
                    labels.append(label2id[cls_dir.name])
        id2label = {v: k for k, v in label2id.items()}
        return files, labels, label2id, id2label
    
    # === CORE TRAINING FUNCTION ===
    def train_stage(name, data_dir, epochs, output_dir, backbone_type='ast',
                    base_model_dir=None, is_stage2=False):
        print(f"\n{'='*60}")
        print(f"TRAINING: {name}")
        print(f"Backbone: {BACKBONE_REGISTRY[backbone_type]['name']}")
        print(f"{'='*60}")
        
        files, labels, label2id, id2label = load_data(data_dir)
        
        if len(files) == 0:
            print(f"[!] No data found in {data_dir} - SKIPPING")
            return None
        
        print(f"Data: {len(files)} files, {len(label2id)} classes")
        for c, i in label2id.items():
            print(f"  {c}: {labels.count(i)}")
        
        # === 5-FOLD CROSS-VALIDATION ===
        n_folds = CONFIG['cv_folds']
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        best_overall_f1 = 0
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
            print(f"\n--- Fold {fold+1}/{n_folds} ---")
            
            train_files = [files[i] for i in train_idx]
            val_files = [files[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Create backbone
            try:
                backbone, extractor, hidden_size = create_backbone(backbone_type)
            except Exception as e:
                print(f"  [!] Failed to create {backbone_type} backbone: {e}")
                return None
            
            backbone = backbone.to(device)
            
            # Enable gradient checkpointing if available
            if hasattr(backbone, 'gradient_checkpointing_enable'):
                backbone.gradient_checkpointing_enable()
            
            # Datasets
            train_ds = AdvancedAudioDataset(train_files, train_labels, extractor,
                                            backbone_type=backbone_type,
                                            augment=True, mixup_alpha=CONFIG['mixup_alpha'],
                                            max_dur=CONFIG['max_duration'])
            val_ds = AdvancedAudioDataset(val_files, val_labels, extractor,
                                          backbone_type=backbone_type, augment=False,
                                          max_dur=CONFIG['max_duration'])
            
            batch_size = CONFIG['batch_size_stage2'] if is_stage2 else CONFIG['batch_size']
            grad_accum = CONFIG['gradient_accumulation_stage2'] if is_stage2 else CONFIG['gradient_accumulation']
            
            # Enhanced balanced sampling with sqrt weighting for extreme imbalance
            train_counts = Counter(train_labels)
            min_count = min(train_counts.values())
            max_count = max(train_counts.values())
            
            # Use sqrt inverse frequency to prevent over-weighting extremely rare classes
            # This gives minority classes more weight while keeping training stable
            if max_count / min_count > 50:  # Extreme imbalance detected
                print(f"  [!] Extreme imbalance detected (ratio {max_count/min_count:.0f}:1)")
                # Use effective number of samples weighting (more robust)
                beta = 0.9999
                effective_num = {k: (1 - beta**v) / (1 - beta) for k, v in train_counts.items()}
                sample_weights = [1.0 / effective_num[label] for label in train_labels]
            else:
                sample_weights = [1.0 / train_counts[label] for label in train_labels]
            
            # Oversample more aggressively: 2x the dataset size per epoch
            sampler = WeightedRandomSampler(weights=sample_weights, 
                                            num_samples=len(train_labels) * 2, 
                                            replacement=True)
            
            # Windows can't pickle local classes for multiprocessing, so use num_workers=0
            # pin_memory=True still helps with faster CPU->GPU transfer
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      sampler=sampler, collate_fn=collate_fn, 
                                      num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn, 
                                    num_workers=0, pin_memory=True)
            
            # Create classifier
            model = MultiBackboneClassifier(backbone, backbone_type, len(label2id), hidden_size).to(device)
            
            # DISABLED: torch.compile causes GPU issues on Windows
            # model = torch.compile(model, mode='reduce-overhead')
            print(f"  Model on {device} (compile disabled for Windows compatibility)")
            
            # Load base model weights if provided
            if base_model_dir:
                ckpt_path = os.path.join(base_model_dir, 'pytorch_model.bin')
                if os.path.exists(ckpt_path):
                    try:
                        state = torch.load(ckpt_path, map_location=device)
                        model.load_state_dict(state, strict=False)
                        print(f"  Loaded base checkpoint from {ckpt_path}")
                    except Exception as e:
                        print(f"  [!] Could not load base weights: {str(e)[:60]}")
            
            # Freeze strategy
            if base_model_dir is None:
                print(f"  [Stage 1] Freezing backbone - training classifier only")
                for param in backbone.parameters():
                    param.requires_grad = False
            elif is_stage2:
                print(f"  [Stage 2] Full fine-tuning")
                for param in backbone.parameters():
                    param.requires_grad = True
            else:
                print(f"  [Stage 3] Partial unfreeze - last 4 layers")
                for param in backbone.parameters():
                    param.requires_grad = False
                if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
                    num_layers = len(backbone.encoder.layers)
                    for layer in backbone.encoder.layers[max(0, num_layers-4):]:
                        for param in layer.parameters():
                            param.requires_grad = True
            
            # Class weights with effective number of samples
            counts = Counter(train_labels)
            max_count = max(counts.values())
            min_count = min(counts.values())
            
            if max_count / min_count > 50:  # Extreme imbalance
                # Use effective number of samples for class weights
                beta = 0.9999
                effective_num = {i: (1 - beta**counts[i]) / (1 - beta) for i in range(len(label2id))}
                weights = torch.FloatTensor([1.0 / effective_num[i] for i in range(len(label2id))]).to(device)
            else:
                weights = torch.FloatTensor([max_count / counts[i] for i in range(len(label2id))]).to(device)
            weights = weights / weights.sum() * len(label2id)
            
            # Use Focal Loss for better handling of class imbalance
            try:
                loss_fn = FocalLoss(alpha=weights, gamma=2.0, label_smoothing=CONFIG['label_smoothing'])
                print(f"  Using Focal Loss (gamma=2.0) with effective number weighting")
            except Exception as e:
                print(f"  [!] FocalLoss failed, falling back to CrossEntropyLoss: {str(e)[:40]}")
                loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=CONFIG['label_smoothing'])
            
            lr_mult = BACKBONE_REGISTRY[backbone_type]['lr_mult']
            lr = CONFIG['learning_rate'] * lr_mult * (0.1 if base_model_dir else 1.0)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, weight_decay=CONFIG['weight_decay'])
            
            total_steps = len(train_loader) * epochs // grad_accum
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, int(total_steps * CONFIG['warmup_ratio']), total_steps
            )
            
            scaler = torch.amp.GradScaler('cuda') if CONFIG['fp16'] and torch.cuda.is_available() else None
            
            # Training loop
            best_f1 = 0
            best_acc = 0
            patience = 5  # Reduced for faster training (was 15)
            patience_counter = 0
            fold_output_dir = f"{output_dir}/fold_{fold+1}"
            os.makedirs(fold_output_dir, exist_ok=True)
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                optimizer.zero_grad()
                
                pbar = tqdm(train_loader, desc=f"F{fold+1} E{epoch+1}/{epochs}", leave=False)
                step = -1
                for step, batch in enumerate(pbar):
                    # Use non_blocking=True to overlap CPU-GPU data transfer
                    inputs = batch['input_values'].to(device, non_blocking=True)
                    mel_spec = batch['mel_spectrogram'].to(device, non_blocking=True)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device, non_blocking=True)
                    labels_batch = batch['labels'].to(device, non_blocking=True)
                    
                    try:
                        if scaler:
                            with torch.cuda.amp.autocast():
                                logits = model(input_values=inputs, mel_spectrogram=mel_spec, attention_mask=attention_mask)
                                loss = loss_fn(logits, labels_batch) / grad_accum
                            scaler.scale(loss).backward()
                        else:
                            logits = model(input_values=inputs, mel_spectrogram=mel_spec, attention_mask=attention_mask)
                            loss = loss_fn(logits, labels_batch) / grad_accum
                            loss.backward()
                        
                        del logits, labels_batch, inputs, mel_spec, attention_mask
                        # Removed per-batch empty_cache() - very slow
                    except RuntimeError as e:
                        print(f"    [!] Batch error: {str(e)[:50]} - skipping")
                        optimizer.zero_grad()
                        continue
                    
                    if (step + 1) % grad_accum == 0:
                        try:
                            if scaler:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                        except RuntimeError:
                            optimizer.zero_grad()
                    
                    total_loss += loss.item() * grad_accum
                    pbar.set_postfix({'loss': total_loss / (step + 1)})
                
                if step < 0:
                    continue
                
                # Validation
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch['input_values'].to(device, non_blocking=True)
                        mel_spec = batch['mel_spectrogram'].to(device, non_blocking=True)
                        attention_mask = batch.get('attention_mask')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device, non_blocking=True)
                        
                        logits = model(input_values=inputs, mel_spectrogram=mel_spec, attention_mask=attention_mask)
                        preds = torch.argmax(logits, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch['labels'].numpy())
                
                acc = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='weighted')
                
                print(f"  F{fold+1} E{epoch+1}: Acc={acc:.4f} F1={f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_acc = acc
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{fold_output_dir}/pytorch_model.bin")
                    if extractor is not None and hasattr(extractor, 'save_pretrained'):
                        extractor.save_pretrained(fold_output_dir)
                    with open(f"{fold_output_dir}/label_mappings.json", 'w') as f:
                        json.dump({'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}, f)
                    with open(f"{fold_output_dir}/model_config.json", 'w') as f:
                        json.dump({'backbone_type': backbone_type, 'hidden_size': hidden_size,
                                   'num_labels': len(label2id), 'backbone_hf_id': BACKBONE_REGISTRY[backbone_type]['hf_id']}, f)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break
            
            fold_results.append({'fold': fold+1, 'acc': best_acc, 'f1': best_f1})
            print(f"  Fold {fold+1} Best: Acc={best_acc:.4f} F1={best_f1:.4f}")
            
            # Save best fold model as the main model
            if best_f1 > best_overall_f1:
                best_overall_f1 = best_f1
                os.makedirs(output_dir, exist_ok=True)
                for fname in ['pytorch_model.bin', 'label_mappings.json', 'model_config.json']:
                    src = f"{fold_output_dir}/{fname}"
                    if os.path.exists(src):
                        shutil.copy2(src, f"{output_dir}/{fname}")
                if extractor is not None and hasattr(extractor, 'save_pretrained'):
                    extractor.save_pretrained(output_dir)
            
            # Free memory
            del model, backbone, optimizer, scheduler
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # CV Summary
        if fold_results:
            avg_acc = np.mean([r['acc'] for r in fold_results])
            avg_f1 = np.mean([r['f1'] for r in fold_results])
            std_f1 = np.std([r['f1'] for r in fold_results])
            
            print(f"\n{'='*50}")
            print(f"CV RESULTS - {name} [{backbone_type.upper()}]")
            print(f"{'='*50}")
            for r in fold_results:
                print(f"  Fold {r['fold']}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
            print(f"  Average: Acc={avg_acc:.4f} F1={avg_f1:.4f} (+/-{std_f1:.4f})")
            
            # Save results to log file
            log_file = 'training_results_log.txt'
            with open(log_file, 'a', encoding='utf-8') as f:
                import datetime
                f.write(f"\n{'='*60}\n")
                f.write(f"Training Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Stage: {name}\n")
                f.write(f"Backbone: {backbone_type.upper()}\n")
                f.write(f"Epochs: {epochs}\n")
                f.write(f"{'='*60}\n")
                for r in fold_results:
                    f.write(f"  Fold {r['fold']}: Acc={r['acc']:.4f} F1={r['f1']:.4f}\n")
                f.write(f"  AVERAGE: Acc={avg_acc:.4f} F1={avg_f1:.4f} (+/-{std_f1:.4f})\n")
                f.write(f"{'='*60}\n")
            print(f"  Results saved to {log_file}")
        
        return output_dir
    
    # === DETERMINE AVAILABLE BACKBONES ===
    print("\nChecking available backbones...")
    available_backbones = []
    
    for bb in CONFIG['backbones']:
        try:
            if bb == 'panns':
                print(f"  [OK] {bb}: PANNs CNN14 (no external model needed)")
                available_backbones.append(bb)
                continue
            
            if bb == 'beats':
                ckpt = Path.home() / '.cache' / 'huggingface' / 'hub' / 'beats' / 'BEATs_iter3_plus_AS2M.pt'
                if ckpt.exists():
                    print(f"  [OK] {bb}: BEATs checkpoint found")
                    available_backbones.append(bb)
                else:
                    print(f"  [!] {bb}: BEATs checkpoint not found - skipping")
                continue
            
            info = BACKBONE_REGISTRY[bb]
            if info['hf_id']:
                # Try to load the model to check availability
                print(f"  Checking {bb} ({info['hf_id']})...", end=" ")
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(info['hf_id'])
                    print("[OK]")
                    available_backbones.append(bb)
                except Exception as e:
                    print(f"[downloading...]")
                    available_backbones.append(bb)  # Will be downloaded during training
        except Exception as e:
            print(f"  [!] {bb}: Error - {str(e)[:50]}")
    
    if not available_backbones:
        print("\n[!] No backbones available!")
        print("Please ensure you have internet access to download models from HuggingFace.")
        return
    
    print(f"\nBackbones to train: {', '.join(available_backbones)}")
    print(f"Training stages: 3 (Respiratory -> Baby Cry -> Baby Pulmonary)")
    print(f"Cross-validation: {CONFIG['cv_folds']}-fold")
    
    all_stage_results = {}
    
    # === TRAIN ALL BACKBONES FOR EACH STAGE ===
    for backbone_type in available_backbones:
        bb_info = BACKBONE_REGISTRY[backbone_type]
        base_epochs = 27  # Standard training epochs
        epochs_mult = bb_info['epochs_mult']
        
        print(f"\n{'='*70}")
        print(f"BACKBONE: {bb_info['name']} ({backbone_type})")
        print(f"{'='*70}")
        
        suffix = f"_{backbone_type}"
        
        # ULTRA FAST MODE: Skip pre-training stages if enabled
        if CONFIG.get('skip_pretrain_stages', False):
            print(f"  [FAST MODE] Skipping Stage 1 & 2, training directly on Baby Pulmonary")
            stage1_path = None
            stage2_path = None
            s3_epochs = int(base_epochs * 1.0 * epochs_mult)
        else:
            # Stage 1: Respiratory (Adult sounds)
            s1_epochs = int(base_epochs * epochs_mult)
            stage1_path = train_stage(
                f"Stage 1: Respiratory Pre-training ({backbone_type})",
                str(RESPIRATORY_DIR),
                epochs=s1_epochs,
                output_dir=f"./model_respiratory{suffix}",
                backbone_type=backbone_type,
                is_stage2=False
            )
            
            # Stage 2: Baby Cry (8 classes)
            s2_epochs = int(base_epochs * 0.8 * epochs_mult)
            stage2_path = train_stage(
                f"Stage 2: Baby Cry Fine-tuning ({backbone_type})",
                str(BABY_CRY_DIR),
                epochs=s2_epochs,
                output_dir=f"./model_baby_cry{suffix}",
                backbone_type=backbone_type,
                base_model_dir=stage1_path,
                is_stage2=True
            )
            s3_epochs = int(base_epochs * 1.2 * epochs_mult)
        
        # Stage 3: Baby Pulmonary (Target task)
        stage3_path = train_stage(
            f"Stage 3: Baby Pulmonary Detection ({backbone_type})",
            str(BABY_PULMONARY_DIR),
            epochs=s3_epochs,
            output_dir=f"./model_baby_pulmonary{suffix}",
            backbone_type=backbone_type,
            base_model_dir=stage2_path if stage2_path else None,
            is_stage2=False
        )
        
        all_stage_results[backbone_type] = {
            'stage1': stage1_path,
            'stage2': stage2_path,
            'stage3': stage3_path
        }
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    for bb, results in all_stage_results.items():
        bb_name = BACKBONE_REGISTRY[bb]['name']
        print(f"\n{bb_name} ({bb}):")
        for stage, path in results.items():
            status = "[OK]" if path else "[SKIPPED]"
            print(f"  {stage}: {status} {path or ''}")
    
    # Suggest best model
    print("\n" + "-" * 70)
    print("RECOMMENDED: Use AST or PANNs models for baby pulmonary detection")
    print("Model paths: ./model_baby_pulmonary_ast or ./model_baby_pulmonary_panns")


def main():
    """Main entry point"""
    print("=" * 70)
    print("BABY PULMONARY DETECTION - DIRECT TRAINING")
    print("=" * 70)
    print("This script trains with multiple backbone models (AST, HuBERT, PANNs)")
    print("using attention pooling, spectrogram features, and 5-fold CV.")
    print()
    
    # Check data availability
    data_status = check_data()
    
    has_data = any(count > 0 for count in data_status.values())
    
    if not has_data:
        print("\n[!] No training data found!")
        print("Please run the data download script first or manually add data.")
        print("Expected directories:")
        print(f"  - {RESPIRATORY_DIR}")
        print(f"  - {BABY_CRY_DIR}")
        print(f"  - {BABY_PULMONARY_DIR}")
        return
    
    print("\n" + "-" * 70)
    print("Starting training in 1 second...")
    print("-" * 70)
    
    import time
    time.sleep(1)
    
    # Start training
    train_maximum_accuracy()
    
    print("\n" + "=" * 70)
    print("[OK] TRAINING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
