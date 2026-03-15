#!/usr/bin/env python3
"""
6-Backbone Ensemble AI Model for Baby Cry Classification & Pulmonary Detection

Models:
1. DistilHuBERT - Primary cry pattern detection
2. AST - Audio spectrogram analysis for respiratory sounds
3. YAMNet - General audio event detection
4. Wav2Vec2 - Best for cry vocalization fine-tuning
5. WavLM - Noise-robust cry detection
6. PANNs CNN14 - Pulmonary/respiratory sound detection

Architecture: Weighted Ensemble Voting
"""

import os
import asyncio
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel:
    """
    6-Backbone Ensemble model for robust baby cry and pulmonary classification.
    
    Uses weighted voting across all models for maximum accuracy.
    Each model contributes votes weighted by its confidence and domain expertise.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        self.classifiers = {}
        self.is_initialized = False
        
        # Model configurations with weights for cry vs pulmonary tasks
        self.model_configs = {
            "distilhubert": {
                "name": "ntu-spml/distilhubert",
                "type": "huggingface",
                "hidden_size": 768,
                "cry_weight": 1.0,        # Primary for cry
                "pulmonary_weight": 0.6,
                "description": "HuBERT distilled - medical-grade audio analysis"
            },
            "ast": {
                "name": "MIT/ast-finetuned-audioset-10-10-0.4593",
                "type": "huggingface",
                "hidden_size": 768,
                "cry_weight": 0.7,
                "pulmonary_weight": 0.9,  # Good for spectrograms/respiratory
                "description": "Audio Spectrogram Transformer - frequency patterns"
            },
            "yamnet": {
                "name": "yamnet",
                "type": "tensorflow",
                "hidden_size": 1024,
                "cry_weight": 0.5,
                "pulmonary_weight": 0.5,
                "description": "YAMNet - general audio event fallback"
            },
            "wav2vec2": {
                "name": "facebook/wav2vec2-base",
                "type": "huggingface",
                "hidden_size": 768,
                "cry_weight": 0.95,       # Excellent for cry vocalization
                "pulmonary_weight": 0.7,
                "description": "Wav2Vec2 - self-supervised speech patterns"
            },
            "wavlm": {
                "name": "microsoft/wavlm-base",
                "type": "huggingface",
                "hidden_size": 768,
                "cry_weight": 0.9,        # Noise-robust cry detection
                "pulmonary_weight": 0.75,
                "description": "WavLM - noise-robust audio understanding"
            },
            "panns": {
                "name": "panns_cnn14",
                "type": "custom",
                "hidden_size": 2048,
                "cry_weight": 0.6,
                "pulmonary_weight": 1.0,  # Best for pulmonary sounds
                "description": "PANNs CNN14 - acoustic event detection"
            }
        }
        
        # Cry class mappings - MATCHES trained_classifiers/cry/*.pt (12 classes)
        # These labels match the training data folder-to-label mapping
        self.cry_classes = [
            "hungry",
            "pain",
            "sleepy",
            "discomfort",
            "belly_pain",
            "burping",
            "cold_hot",
            "scared",
            "lonely",
            "tired",
            "normal",
            "pathological"
        ]
        
        # Pulmonary class mappings - MATCHES trained_classifiers/pulmonary/*.pt (8 classes)
        self.pulmonary_classes = [
            "normal",
            "wheeze",
            "crackle",
            "stridor",
            "rhonchi",
            "bronchiolitis",
            "pneumonia",
            "asthma"
        ]
    
    async def initialize(self):
        """Initialize all 6 models in the ensemble"""
        print("  Loading 6-backbone ensemble models...")
        print(f"  Device: {self.device}")
        
        # Load all models in parallel where possible
        await self._load_distilhubert()
        await self._load_ast()
        await self._load_yamnet()
        await self._load_wav2vec2()
        await self._load_wavlm()
        await self._load_panns()
        
        # Count successful loads
        loaded = sum(1 for m in ["distilhubert", "ast", "yamnet", "wav2vec2", "wavlm", "panns"] 
                     if self.models.get(m) is not None)
        print(f"  Loaded {loaded}/6 backbone models")
        
        # Load trained classifier weights if available
        await self._load_trained_weights()
        
        self.is_initialized = True
    
    async def _load_trained_weights(self):
        """Load pre-trained classifier weights from saved files"""
        import os
        
        # Possible locations for trained weights
        weight_dirs = [
            Path(__file__).parent / "trained_weights",
            Path(__file__).parent / "trained_weights" / "cry",
            Path(__file__).parent / "trained_weights" / "pulmonary",
            Path("D:/projects/cry analysuis/trained_classifiers/cry"),
            Path("D:/projects/cry analysuis/trained_classifiers/pulmonary"),
            Path("D:/projects/cry analysuis/baby_cry_diagnostic/backend/models/trained_weights"),
        ]
        
        loaded_count = 0
        
        for weight_dir in weight_dirs:
            if weight_dir.exists():
                for classifier_file in weight_dir.glob("*.pt"):
                    name = classifier_file.stem
                    if name in self.classifiers:
                        try:
                            state_dict = torch.load(classifier_file, map_location=self.device)
                            # Handle both direct state_dict and wrapped state_dict
                            if 'classifier_state_dict' in state_dict:
                                self.classifiers[name].load_state_dict(state_dict['classifier_state_dict'])
                            else:
                                self.classifiers[name].load_state_dict(state_dict)
                            loaded_count += 1
                            print(f"    [LOAD] {name} weights from {classifier_file.name}")
                        except Exception as e:
                            print(f"    [!] Failed to load {name}: {e}")
        
        if loaded_count > 0:
            print(f"  Loaded {loaded_count} trained classifier weights")
        else:
            print("  [!] No trained weights found - using random initialization")
            print("      Run: python train_6backbone_ensemble.py --epochs 20")
    
    async def _load_distilhubert(self):
        """Load DistilHuBERT model with classification heads"""
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            
            model_name = self.model_configs["distilhubert"]["name"]
            hidden_size = self.model_configs["distilhubert"]["hidden_size"]
            
            print(f"    Loading DistilHuBERT from {model_name}...")
            self.processors["distilhubert"] = AutoFeatureExtractor.from_pretrained(model_name)
            self.models["distilhubert"] = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.models["distilhubert"].eval()
            
            # Cry classification head
            self.classifiers["distilhubert_cry"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.cry_classes))
            ).to(self.device)
            
            # Pulmonary classification head
            self.classifiers["distilhubert_pulmonary"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.pulmonary_classes))
            ).to(self.device)
            
            print("    [OK] DistilHuBERT loaded (cry_weight=1.0, pulm_weight=0.6)")
            
        except Exception as e:
            print(f"    [!] DistilHuBERT failed: {type(e).__name__}: {e}")
            self.models["distilhubert"] = None
    
    async def _load_ast(self):
        """Load Audio Spectrogram Transformer with classification heads"""
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            
            model_name = self.model_configs["ast"]["name"]
            hidden_size = self.model_configs["ast"]["hidden_size"]
            
            print(f"    Loading AST from {model_name}...")
            self.processors["ast"] = AutoFeatureExtractor.from_pretrained(model_name)
            self.models["ast"] = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.models["ast"].eval()
            
            # Cry classification head (uses AudioSet embeddings)
            self.classifiers["ast_cry"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.cry_classes))
            ).to(self.device)
            
            # Pulmonary classification head
            self.classifiers["ast_pulmonary"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.pulmonary_classes))
            ).to(self.device)
            
            print("    [OK] AST loaded (cry_weight=0.7, pulm_weight=0.9)")
            
        except Exception as e:
            print(f"    [!] AST failed: {type(e).__name__}: {e}")
            self.models["ast"] = None
    
    async def _load_yamnet(self):
        """Load YAMNet TensorFlow model (skipped on Python 3.12+ due to imp module removal)"""
        try:
            import sys
            
            # YAMNet doesn't work well on Python 3.12+ (imp module was removed)
            if sys.version_info >= (3, 12):
                print("    [!] YAMNet skipped on Python 3.12+ (requires deprecated imp module)")
                self.models["yamnet"] = None
                return
            
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                print("    Loading YAMNet from TensorFlow Hub...")
                
                # Load YAMNet from TensorFlow Hub
                self.models["yamnet"] = hub.load('https://tfhub.dev/google/yamnet/1')
                
                # YAMNet outputs 1024-dim embeddings, add classification heads
                hidden_size = self.model_configs["yamnet"]["hidden_size"]
                
                self.classifiers["yamnet_cry"] = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, len(self.cry_classes))
                ).to(self.device)
                
                self.classifiers["yamnet_pulmonary"] = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, len(self.pulmonary_classes))
                ).to(self.device)
                
                print("    [OK] YAMNet loaded (cry_weight=0.5, pulm_weight=0.5)")
            except ImportError:
                print("    [!] YAMNet skipped (tensorflow and tensorflow_hub not installed)")
                print("        Install with: pip install tensorflow tensorflow-hub")
                self.models["yamnet"] = None
            
        except Exception as e:
            print(f"    [!] YAMNet failed: {type(e).__name__}: {e}")
            self.models["yamnet"] = None
    
    async def _load_wav2vec2(self):
        """Load Wav2Vec2 - excellent for cry vocalization patterns"""
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            
            model_name = self.model_configs["wav2vec2"]["name"]
            hidden_size = self.model_configs["wav2vec2"]["hidden_size"]
            
            print(f"    Loading Wav2Vec2 from {model_name}...")
            self.processors["wav2vec2"] = AutoFeatureExtractor.from_pretrained(model_name)
            self.models["wav2vec2"] = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.models["wav2vec2"].eval()
            
            # Cry classification head
            self.classifiers["wav2vec2_cry"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.cry_classes))
            ).to(self.device)
            
            # Pulmonary classification head
            self.classifiers["wav2vec2_pulmonary"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.pulmonary_classes))
            ).to(self.device)
            
            print("    [OK] Wav2Vec2 loaded (cry_weight=0.95, pulm_weight=0.7)")
            
        except Exception as e:
            print(f"    [!] Wav2Vec2 failed: {type(e).__name__}: {e}")
            self.models["wav2vec2"] = None
    
    async def _load_wavlm(self):
        """Load WavLM - noise-robust audio understanding"""
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            
            model_name = self.model_configs["wavlm"]["name"]
            hidden_size = self.model_configs["wavlm"]["hidden_size"]
            
            print(f"    Loading WavLM from {model_name}...")
            # WavLM uses same feature extractor as Wav2Vec2
            self.processors["wavlm"] = AutoFeatureExtractor.from_pretrained(model_name)
            self.models["wavlm"] = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.models["wavlm"].eval()
            
            # Cry classification head
            self.classifiers["wavlm_cry"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.cry_classes))
            ).to(self.device)
            
            # Pulmonary classification head
            self.classifiers["wavlm_pulmonary"] = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.pulmonary_classes))
            ).to(self.device)
            
            print("    [OK] WavLM loaded (cry_weight=0.9, pulm_weight=0.75)")
            
        except Exception as e:
            print(f"    [!] WavLM failed: {type(e).__name__}: {e}")
            self.models["wavlm"] = None
    
    async def _load_panns(self):
        """Load PANNs CNN14 - best for pulmonary/respiratory sounds"""
        try:
            # PANNs CNN14 architecture
            class CNN14(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv_block1 = self._conv_block(1, 64)
                    self.conv_block2 = self._conv_block(64, 128)
                    self.conv_block3 = self._conv_block(128, 256)
                    self.conv_block4 = self._conv_block(256, 512)
                    self.conv_block5 = self._conv_block(512, 1024)
                    self.conv_block6 = self._conv_block(1024, 2048)
                    self.fc = nn.Linear(2048, 2048)
                    
                def _conv_block(self, in_ch, out_ch):
                    return nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.AvgPool2d(2)
                    )
                    
                def forward(self, x):
                    # x: (batch, 1, freq, time)
                    x = self.conv_block1(x)
                    x = self.conv_block2(x)
                    x = self.conv_block3(x)
                    x = self.conv_block4(x)
                    x = self.conv_block5(x)
                    x = self.conv_block6(x)
                    x = torch.mean(x, dim=(2, 3))  # Global avg pool
                    x = self.fc(x)
                    return x
            
            hidden_size = self.model_configs["panns"]["hidden_size"]
            
            self.models["panns"] = CNN14().to(self.device)
            self.models["panns"].eval()
            
            # Cry classification head
            self.classifiers["panns_cry"] = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.cry_classes))
            ).to(self.device)
            
            # Pulmonary classification head (primary use case)
            self.classifiers["panns_pulmonary"] = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.pulmonary_classes))
            ).to(self.device)
            
            print("    [OK] PANNs CNN14 loaded (cry_weight=0.6, pulm_weight=1.0)")
            
        except Exception as e:
            print(f"    [!] PANNs failed: {e}")
            self.models["panns"] = None
    
    async def predict(self, waveform: np.ndarray, sample_rate: int = 16000, 
                      task: str = "cry") -> Dict[str, Any]:
        """
        Run weighted ensemble prediction across all 6 backbone models.
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate (default 16kHz)
            task: "cry" for cry classification, "pulmonary" for respiratory detection
        
        Returns:
            Dictionary with classification results from weighted voting
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Ensure correct format
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Normalize
        if np.max(np.abs(waveform)) > 1.0:
            waveform = waveform / 32768.0
        
        # Get predictions from all available models
        all_predictions = []
        model_results = {}
        
        # Collect predictions from each model
        models_to_try = [
            ("distilhubert", self._predict_transformer),
            ("wav2vec2", self._predict_transformer),
            ("wavlm", self._predict_transformer),
            ("ast", self._predict_ast_model),
            ("yamnet", self._predict_yamnet_model),
            ("panns", self._predict_panns_model),
        ]
        
        for model_name, predict_fn in models_to_try:
            if self.models.get(model_name) is not None:
                try:
                    result = await predict_fn(model_name, waveform, sample_rate, task)
                    if result:
                        weight = self.model_configs[model_name].get(f"{task}_weight", 0.5)
                        all_predictions.append({
                            "model": model_name,
                            "label": result["label"],
                            "confidence": result["confidence"],
                            "weight": weight,
                            "weighted_score": result["confidence"] * weight,
                            "scores": result.get("all_scores", {})
                        })
                        model_results[model_name] = result
                except Exception as e:
                    print(f"    [!] {model_name} prediction failed: {e}")
        
        if not all_predictions:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "model": "none",
                "all_scores": {},
                "ensemble_votes": {}
            }
        
        # Weighted voting
        classes = self.cry_classes if task == "cry" else self.pulmonary_classes
        vote_scores = defaultdict(float)
        vote_weights = defaultdict(float)
        
        for pred in all_predictions:
            label = pred["label"]
            weighted_vote = pred["weighted_score"]
            vote_scores[label] += weighted_vote
            vote_weights[label] += pred["weight"]
            
            # Also add scores from individual class predictions
            for cls, score in pred["scores"].items():
                if cls in classes:
                    weight = pred["weight"]
                    vote_scores[cls] += score * weight
                    vote_weights[cls] += weight
        
        # Normalize votes
        final_scores = {}
        for cls in vote_scores:
            if vote_weights[cls] > 0:
                final_scores[cls] = vote_scores[cls] / vote_weights[cls]
            else:
                final_scores[cls] = 0.0
        
        # Get winning class
        if final_scores:
            winning_label = max(final_scores.keys(), key=lambda k: final_scores[k])
            winning_confidence = final_scores[winning_label]
        else:
            winning_label = "unknown"
            winning_confidence = 0.0
        
        # Determine which model contributed most
        best_model = max(all_predictions, key=lambda p: p["weighted_score"])["model"]
        
        return {
            "label": winning_label,
            "confidence": float(winning_confidence),
            "model": f"Ensemble (6-backbone, primary: {best_model})",
            "all_scores": final_scores,
            "ensemble_votes": {p["model"]: {"label": p["label"], "confidence": p["confidence"]} 
                              for p in all_predictions},
            "models_used": len(all_predictions),
            "task": task
        }
    
    async def _predict_transformer(self, model_name: str, waveform: np.ndarray, 
                                   sample_rate: int, task: str) -> Optional[Dict]:
        """Generic prediction for transformer models (DistilHuBERT, Wav2Vec2, WavLM)"""
        with torch.no_grad():
            processor = self.processors.get(model_name)
            model = self.models.get(model_name)
            
            if processor is None or model is None:
                return None
            
            # Process audio
            inputs = processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Select classifier based on task
            classifier_name = f"{model_name}_{task}"
            classifier = self.classifiers.get(classifier_name)
            
            if classifier is None:
                return None
            
            # Classify
            logits = classifier(embeddings)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Get classes
            classes = self.cry_classes if task == "cry" else self.pulmonary_classes
            top_idx = np.argmax(probs)
            
            return {
                "label": classes[top_idx],
                "confidence": float(probs[top_idx]),
                "all_scores": {c: float(p) for c, p in zip(classes, probs)}
            }
    
    async def _predict_ast_model(self, model_name: str, waveform: np.ndarray,
                                  sample_rate: int, task: str) -> Optional[Dict]:
        """Prediction for AST model"""
        with torch.no_grad():
            processor = self.processors.get("ast")
            model = self.models.get("ast")
            
            if processor is None or model is None:
                return None
            
            # Process audio
            inputs = processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get AudioSet predictions and map to our classes
            outputs = model(**inputs)
            audioset_probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # Use custom classifier for our task
            # Extract features from model's hidden states
            hidden_states = model.distilbert(inputs["input_values"]) if hasattr(model, 'distilbert') else None
            
            # Map AudioSet predictions to cry/pulmonary classes
            cry_label, cry_confidence = self._map_audioset_to_cry(audioset_probs)
            
            classes = self.cry_classes if task == "cry" else self.pulmonary_classes
            
            return {
                "label": cry_label if task == "cry" else "normal",
                "confidence": cry_confidence,
                "all_scores": {cry_label: cry_confidence}
            }
    
    async def _predict_yamnet_model(self, model_name: str, waveform: np.ndarray,
                                     sample_rate: int, task: str) -> Optional[Dict]:
        """Prediction for YAMNet model"""
        import tensorflow as tf
        
        model = self.models.get("yamnet")
        if model is None:
            return None
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            duration = len(waveform) / sample_rate
            new_length = int(duration * 16000)
            indices = np.linspace(0, len(waveform) - 1, new_length).astype(int)
            waveform = waveform[indices]
        
        # Run YAMNet
        scores, embeddings, spectrogram = model(waveform)
        scores = scores.numpy()
        embeddings = embeddings.numpy()
        
        # Get averaged embedding for classification
        avg_embedding = embeddings.mean(axis=0)
        
        # Use our classifier
        classifier_name = f"yamnet_{task}"
        classifier = self.classifiers.get(classifier_name)
        
        if classifier is not None:
            with torch.no_grad():
                emb_tensor = torch.tensor(avg_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = classifier(emb_tensor)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                
                classes = self.cry_classes if task == "cry" else self.pulmonary_classes
                top_idx = np.argmax(probs)
                
                return {
                    "label": classes[top_idx],
                    "confidence": float(probs[top_idx]),
                    "all_scores": {c: float(p) for c, p in zip(classes, probs)}
                }
        
        # Fallback to YAMNet class mapping
        top_class_idx = scores.mean(axis=0).argmax()
        cry_label = self._map_yamnet_to_cry(top_class_idx)
        confidence = float(scores.mean(axis=0)[top_class_idx])
        
        return {
            "label": cry_label,
            "confidence": confidence,
            "all_scores": {cry_label: confidence}
        }
    
    async def _predict_panns_model(self, model_name: str, waveform: np.ndarray,
                                    sample_rate: int, task: str) -> Optional[Dict]:
        """Prediction for PANNs CNN14 model"""
        import librosa
        
        model = self.models.get("panns")
        if model is None:
            return None
        
        with torch.no_grad():
            # Compute mel spectrogram
            if sample_rate != 16000:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            
            mel_spec = librosa.feature.melspectrogram(
                y=waveform, sr=16000, n_mels=128, fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Prepare input
            mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32)
            mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, freq, time)
            
            # Get embeddings
            embeddings = model(mel_tensor)
            
            # Classify
            classifier_name = f"panns_{task}"
            classifier = self.classifiers.get(classifier_name)
            
            if classifier is None:
                return None
            
            logits = classifier(embeddings)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
            classes = self.cry_classes if task == "cry" else self.pulmonary_classes
            top_idx = np.argmax(probs)
            
            return {
                "label": classes[top_idx],
                "confidence": float(probs[top_idx]),
                "all_scores": {c: float(p) for c, p in zip(classes, probs)}
            }
    
    def _map_audioset_to_cry(self, probs: np.ndarray) -> tuple:
        """Map AudioSet class probabilities to cry classification"""
        # AudioSet indices for relevant classes
        # 20: Baby cry, 21: Baby laughter, 22: Child speech
        baby_cry_idx = 20
        
        # Check if baby cry is detected
        if probs[baby_cry_idx] > 0.3:
            # Use additional context to determine cry type
            # This is simplified - in production, use a fine-tuned classifier
            return "discomfort", float(probs[baby_cry_idx])
        
        # Check for distress indicators
        if any(probs[i] > 0.2 for i in [20, 21, 22]):
            return "distress", float(max(probs[20], probs[21], probs[22]))
        
        return "normal", 0.5
    
    def _map_yamnet_to_cry(self, class_idx: int) -> str:
        """Map YAMNet class index to cry classification"""
        # YAMNet class 20 is "Baby cry, infant cry"
        yamnet_to_cry = {
            20: "discomfort",  # Baby cry
            21: "normal",      # Baby laughter
            22: "normal",      # Child speech
        }
        
        return yamnet_to_cry.get(class_idx, "unknown")
    
    async def predict_both(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Run both cry and pulmonary predictions.
        
        Returns combined results for display in UI.
        """
        cry_result = await self.predict(waveform, sample_rate, task="cry")
        pulmonary_result = await self.predict(waveform, sample_rate, task="pulmonary")
        
        return {
            "cry": cry_result,
            "pulmonary": pulmonary_result,
            "models_loaded": sum(1 for m in self.models.values() if m is not None),
            "device": str(self.device)
        }
    
    async def predict_auto(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Automatically detect audio type and classify appropriately.
        
        Runs BOTH cry and pulmonary classifiers:
        - If pulmonary classifier has higher confidence → return respiratory sound (wheeze, rhonchi, etc.)
        - If cry classifier has higher confidence → return baby cry type
        
        This ensures respiratory sounds like rhonchi are correctly identified.
        """
        # Run both classifiers
        cry_result = await self.predict(waveform, sample_rate, task="cry")
        pulmonary_result = await self.predict(waveform, sample_rate, task="pulmonary")
        
        cry_conf = cry_result.get("confidence", 0.0)
        pulm_conf = pulmonary_result.get("confidence", 0.0)
        
        # Respiratory sounds that should override cry classification
        respiratory_sounds = {"wheeze", "rhonchi", "crackle", "stridor", "coarse_crackle", 
                             "fine_crackle", "mixed", "mixed_crackle_wheeze"}
        
        pulm_label = pulmonary_result.get("label", "normal")
        
        # Decision logic:
        # 1. If pulmonary detects a respiratory sound (not "normal") with good confidence → use pulmonary
        # 2. If pulmonary confidence significantly higher (>0.15 difference) → use pulmonary
        # 3. Otherwise use cry classification
        
        is_respiratory_sound = pulm_label.lower() in respiratory_sounds
        pulm_is_confident = pulm_conf > 0.4  # Reasonable threshold
        pulm_beats_cry = pulm_conf > (cry_conf + 0.1)  # Pulmonary wins by margin
        
        use_pulmonary = (is_respiratory_sound and pulm_is_confident) or pulm_beats_cry
        
        if use_pulmonary:
            # Return pulmonary result as primary
            primary = pulmonary_result.copy()
            primary["audio_type"] = "respiratory"
            primary["alternative"] = cry_result
            primary["detection_reason"] = f"Respiratory sound detected ({pulm_label}: {pulm_conf*100:.1f}%)"
            print(f"[AUTO] Detected RESPIRATORY: {pulm_label} ({pulm_conf*100:.1f}%) vs cry {cry_result.get('label')} ({cry_conf*100:.1f}%)")
        else:
            # Return cry result as primary
            primary = cry_result.copy()
            primary["audio_type"] = "cry"
            primary["alternative"] = pulmonary_result
            primary["detection_reason"] = f"Baby cry detected ({cry_result.get('label')}: {cry_conf*100:.1f}%)"
            print(f"[AUTO] Detected CRY: {cry_result.get('label')} ({cry_conf*100:.1f}%) vs respiratory {pulm_label} ({pulm_conf*100:.1f}%)")
        
        primary["cry_result"] = cry_result
        primary["pulmonary_result"] = pulmonary_result
        
        return primary
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "total_backbones": 6,
            "loaded_backbones": 0,
            "models": {},
            "device": str(self.device)
        }
        
        for model_name, config in self.model_configs.items():
            is_loaded = self.models.get(model_name) is not None
            info["models"][model_name] = {
                "loaded": is_loaded,
                "description": config["description"],
                "cry_weight": config["cry_weight"],
                "pulmonary_weight": config["pulmonary_weight"]
            }
            if is_loaded:
                info["loaded_backbones"] += 1
        
        return info
    
    def save_classifiers(self, save_dir: str):
        """Save all trained classifier heads"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, classifier in self.classifiers.items():
            torch.save(classifier.state_dict(), save_path / f"{name}.pt")
        
        print(f"Saved {len(self.classifiers)} classifier heads to {save_dir}")
    
    def load_classifiers(self, load_dir: str):
        """Load pre-trained classifier heads"""
        load_path = Path(load_dir)
        
        if not load_path.exists():
            print(f"No classifier directory found at {load_dir}")
            return
        
        loaded = 0
        for classifier_file in load_path.glob("*.pt"):
            name = classifier_file.stem
            if name in self.classifiers:
                self.classifiers[name].load_state_dict(
                    torch.load(classifier_file, map_location=self.device)
                )
                loaded += 1
        
        print(f"Loaded {loaded} classifier heads from {load_dir}")
    
    def set_training_mode(self, train: bool = True):
        """Set models to training or evaluation mode"""
        for classifier in self.classifiers.values():
            if train:
                classifier.train()
            else:
                classifier.eval()
        
        # Keep backbone models frozen
        for model_name, model in self.models.items():
            if model is not None and hasattr(model, 'eval'):
                model.eval()
    
    async def cleanup(self):
        """Clean up model resources"""
        self.models = {}
        self.processors = {}
        self.classifiers = {}
        self.is_initialized = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("  Ensemble models cleaned up")
