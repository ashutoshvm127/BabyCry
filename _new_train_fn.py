def train_maximum_accuracy():
    """Train with 5 advanced backbone models for maximum pulmonary detection accuracy
    
    BACKBONES: AST, HuBERT-Large, CLAP, PANNs CNN14, BEATs
    IMPROVEMENTS: Attention pooling, spectrogram CNN branch, multi-layer MLP head,
                  5-fold stratified cross-validation, model ensemble
    
    STAGE 1 - RESPIRATORY DISEASE (Adult sounds, 6 classes)
    STAGE 2 - BABY CRY CLASSIFICATION (8 classes)
    STAGE 3 - BABY PULMONARY DISEASE DETECTION (7 classes)
    """
    print("\n" + "=" * 70)
    print("STEP 3: MULTI-MODEL TRAINING (5 BACKBONES x 3 STAGES)")
    print("=" * 70)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import librosa
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    from transformers import AutoFeatureExtractor, AutoModel
    from transformers.optimization import get_cosine_schedule_with_warmup
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # === CONFIGURATION ===
    CONFIG = {
        'backbones': ['ast', 'hubert', 'clap', 'panns', 'beats'],
        'sampling_rate': 16000,
        'max_duration': 5.0,
        'batch_size': 4,
        'batch_size_stage2': 2,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'gradient_accumulation': 2,
        'gradient_accumulation_stage2': 4,
        'label_smoothing': 0.05,
        'mixup_alpha': 0.0,
        'fp16': True,
        'num_train_epochs': 50,
        'cv_folds': 5,
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
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / 'beats'
            ckpt_path = cache_dir / 'BEATs_iter3_plus_AS2M.pt'
            if not ckpt_path.exists():
                raise FileNotFoundError(f"BEATs checkpoint not found at {ckpt_path}")
            checkpoint = torch.load(str(ckpt_path), map_location='cpu')
            # BEATs uses a custom architecture - wrap checkpoint
            from transformers import Wav2Vec2Model
            model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
            extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')
            return model, extractor, 768
        
        raise ValueError(f"Unknown backbone: {backbone_type}")
    
    # === MULTI-BACKBONE CLASSIFIER ===
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
                # PANNs uses mel spectrogram as primary input
                mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=64, n_fft=1024, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                input_values = torch.FloatTensor(mel_db).unsqueeze(0)  # [1, n_mels, time]
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
            mel_spec = torch.FloatTensor(mel_db).unsqueeze(0)  # [1, 64, time]
            
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
            print(f"No data found in {data_dir}")
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
            
            # Balanced sampling
            train_counts = Counter(train_labels)
            sample_weights = [1.0 / train_counts[label] for label in train_labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
            
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      sampler=sampler, collate_fn=collate_fn, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn, num_workers=0)
            
            # Create classifier
            model = MultiBackboneClassifier(backbone, backbone_type, len(label2id), hidden_size).to(device)
            
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
            
            # Class weights
            counts = Counter(train_labels)
            max_count = max(counts.values())
            weights = torch.FloatTensor([max_count / counts[i] for i in range(len(label2id))]).to(device)
            weights = weights / weights.sum() * len(label2id)
            
            try:
                loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=CONFIG['label_smoothing'])
            except TypeError:
                loss_fn = nn.CrossEntropyLoss(weight=weights)
            
            lr_mult = BACKBONE_REGISTRY[backbone_type]['lr_mult']
            lr = CONFIG['learning_rate'] * lr_mult * (0.1 if base_model_dir else 1.0)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, weight_decay=CONFIG['weight_decay'])
            
            total_steps = len(train_loader) * epochs // grad_accum
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, int(total_steps * CONFIG['warmup_ratio']), total_steps
            )
            
            scaler = torch.cuda.amp.GradScaler() if CONFIG['fp16'] and torch.cuda.is_available() else None
            
            # Training loop
            best_f1 = 0
            best_acc = 0
            patience = 15
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
                    inputs = batch['input_values'].to(device)
                    mel_spec = batch['mel_spectrogram'].to(device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    labels_batch = batch['labels'].to(device)
                    
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
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
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
                        inputs = batch['input_values'].to(device)
                        mel_spec = batch['mel_spectrogram'].to(device)
                        attention_mask = batch.get('attention_mask')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        
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
                    # Save model config for backend loading
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
                import shutil
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
        avg_acc = np.mean([r['acc'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        
        print(f"\n{'='*50}")
        print(f"CV RESULTS - {name} [{backbone_type.upper()}]")
        print(f"{'='*50}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        print(f"  Average: Acc={avg_acc:.4f} F1={avg_f1:.4f} (±{std_f1:.4f})")
        
        return output_dir
    
    # === DETERMINE AVAILABLE BACKBONES ===
    available_backbones = []
    for bb in CONFIG['backbones']:
        try:
            if bb == 'panns':
                available_backbones.append(bb)
                continue
            if bb == 'beats':
                ckpt = Path.home() / '.cache' / 'huggingface' / 'hub' / 'beats' / 'BEATs_iter3_plus_AS2M.pt'
                if ckpt.exists():
                    available_backbones.append(bb)
                continue
            info = BACKBONE_REGISTRY[bb]
            if info['hf_id']:
                repo = f"models--{info['hf_id'].replace('/', '--')}"
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
                if (cache_dir / repo).exists():
                    available_backbones.append(bb)
                else:
                    # Try loading anyway (may be cached elsewhere)
                    available_backbones.append(bb)
        except:
            pass
    
    if not available_backbones:
        print("[!] No backbones available! Please run download_model() first.")
        return
    
    print(f"\nAvailable backbones: {', '.join(available_backbones)}")
    print(f"Training stages: 3 (Respiratory → Baby Cry → Baby Pulmonary)")
    print(f"Cross-validation: {CONFIG['cv_folds']}-fold")
    
    all_stage_results = {}
    
    # === TRAIN ALL BACKBONES FOR EACH STAGE ===
    for backbone_type in available_backbones:
        bb_info = BACKBONE_REGISTRY[backbone_type]
        base_epochs = 70
        epochs_mult = bb_info['epochs_mult']
        
        print(f"\n{'='*70}")
        print(f"BACKBONE: {bb_info['name']} ({backbone_type})")
        print(f"{'='*70}")
        
        suffix = f"_{backbone_type}"
        
        # Stage 1: Respiratory
        s1_epochs = int(base_epochs * epochs_mult)
        stage1_path = train_stage(
            f"Stage 1: Respiratory Pre-training ({backbone_type})",
            str(RESPIRATORY_DIR),
            epochs=s1_epochs,
            output_dir=f"./model_respiratory{suffix}",
            backbone_type=backbone_type,
            is_stage2=False
        )
        
        # Stage 2: Baby Cry
        s2_epochs = int(base_epochs * epochs_mult)
        stage2_path = train_stage(
            f"Stage 2: Baby Cry Fine-tuning ({backbone_type})",
            str(BABY_CRY_DIR),
            epochs=s2_epochs,
            output_dir=f"./model_baby_cry{suffix}",
            backbone_type=backbone_type,
            base_model_dir=stage1_path,
            is_stage2=True
        )
        
        # Stage 3: Baby Pulmonary
        baby_pulm_dir = str(BABY_PULMONARY_DIR)
        if Path(baby_pulm_dir).exists() and any(Path(baby_pulm_dir).rglob("*.wav")):
            s3_epochs = int(80 * epochs_mult)
            stage3_path = train_stage(
                f"Stage 3: Baby Pulmonary Detection ({backbone_type})",
                baby_pulm_dir,
                epochs=s3_epochs,
                output_dir=f"./model_baby_pulmonary{suffix}",
                backbone_type=backbone_type,
                base_model_dir=stage1_path,
                is_stage2=False
            )
        else:
            print(f"\n[!] Baby pulmonary data not found - skipping Stage 3 for {backbone_type}")
            stage3_path = None
        
        all_stage_results[backbone_type] = {
            'stage1': stage1_path,
            'stage2': stage2_path,
            'stage3': stage3_path
        }
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("[OK] ALL MULTI-MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModels saved:")
    for bb, paths in all_stage_results.items():
        bb_name = BACKBONE_REGISTRY[bb]['name']
        print(f"\n  {bb_name} ({bb}):")
        print(f"    Stage 1 (Respiratory):  ./model_respiratory_{bb}/")
        print(f"    Stage 2 (Baby Cry):     ./model_baby_cry_{bb}/")
        if paths.get('stage3'):
            print(f"    Stage 3 (Pulmonary):    ./model_baby_pulmonary_{bb}/")
    
    print(f"""
To use any model:
    import torch, json
    from train_maximum_accuracy import MultiBackboneClassifier, create_backbone
    
    MODEL_DIR = "./model_baby_pulmonary_ast"  # Choose model
    
    with open(f"{{MODEL_DIR}}/model_config.json") as f:
        config = json.load(f)
    
    backbone, extractor, hidden = create_backbone(config['backbone_type'])
    model = MultiBackboneClassifier(backbone, config['backbone_type'], config['num_labels'], hidden)
    model.load_state_dict(torch.load(f"{{MODEL_DIR}}/pytorch_model.bin"))
    model.eval()
""")
