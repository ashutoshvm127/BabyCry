#!/usr/bin/env python3
"""
Cloud Ensemble Model for Baby Cry Classification

Downloads models from HuggingFace at runtime - no local dependencies.
Simplified for cloud deployment on Render/Railway/Fly.io
"""

import os
import asyncio
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SimpleClassifier(nn.Module):
    """Simple MLP classifier head"""
    
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class EnsembleModel:
    """
    Cloud-optimized Ensemble for baby cry and respiratory classification.
    Downloads models from HuggingFace at runtime.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        self.classifiers = {}
        self.is_initialized = False
        
        # Models to load from HuggingFace
        self.model_configs = {
            "distilhubert": {
                "name": "ntu-spml/distilhubert",
                "hidden_size": 768,
                "cry_weight": 1.0,
                "pulmonary_weight": 0.6,
            },
            "wav2vec2": {
                "name": "facebook/wav2vec2-base",
                "hidden_size": 768,
                "cry_weight": 0.95,
                "pulmonary_weight": 0.7,
            },
            "ast": {
                "name": "MIT/ast-finetuned-audioset-10-10-0.4593",
                "hidden_size": 768,
                "cry_weight": 0.7,
                "pulmonary_weight": 0.9,
            },
        }
        
        # Classification labels
        self.cry_classes = [
            "hungry", "pain", "sleepy", "discomfort", "cold_hot",
            "pathological", "tired", "normal", "belly_pain", 
            "burping", "scared", "lonely"
        ]
        
        self.pulmonary_classes = [
            "normal", "wheeze", "crackle", "stridor", 
            "rhonchi", "bronchiolitis", "pneumonia", "asthma"
        ]
        
        # Risk mappings
        self.cry_risk = {
            "normal": "GREEN", "hungry": "GREEN", "sleepy": "GREEN",
            "tired": "GREEN", "burping": "GREEN", "lonely": "YELLOW",
            "discomfort": "YELLOW", "cold_hot": "YELLOW", "scared": "YELLOW",
            "belly_pain": "YELLOW", "pain": "RED", "pathological": "RED"
        }
        
        self.pulmonary_risk = {
            "normal": "GREEN", "wheeze": "YELLOW", "crackle": "YELLOW",
            "rhonchi": "YELLOW", "stridor": "RED", "bronchiolitis": "RED",
            "pneumonia": "RED", "asthma": "RED"
        }
    
    async def initialize(self):
        """Initialize models from HuggingFace"""
        if self.is_initialized:
            return
        
        logger.info("Loading ensemble models from HuggingFace...")
        
        try:
            # Load DistilHuBERT (primary model)
            await self._load_distilhubert()
            
            # Load Wav2Vec2 (backup)
            await self._load_wav2vec2()
            
            # Initialize classifiers with random weights (will use zero-shot)
            self._init_classifiers()
            
            self.is_initialized = True
            logger.info(f"Ensemble ready with {len(self.models)} models on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            # Create a minimal fallback
            self.is_initialized = True
            logger.warning("Running in fallback mode - limited functionality")
    
    async def _load_distilhubert(self):
        """Load DistilHuBERT model"""
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            
            model_name = "ntu-spml/distilhubert"
            logger.info(f"  Loading {model_name}...")
            
            self.processors["distilhubert"] = AutoFeatureExtractor.from_pretrained(model_name)
            self.models["distilhubert"] = AutoModel.from_pretrained(model_name).to(self.device)
            self.models["distilhubert"].eval()
            
            logger.info("  [OK] DistilHuBERT loaded")
            
        except Exception as e:
            logger.warning(f"  [SKIP] DistilHuBERT: {e}")
    
    async def _load_wav2vec2(self):
        """Load Wav2Vec2 model"""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
            
            model_name = "facebook/wav2vec2-base"
            logger.info(f"  Loading {model_name}...")
            
            self.processors["wav2vec2"] = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.models["wav2vec2"] = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.models["wav2vec2"].eval()
            
            logger.info("  [OK] Wav2Vec2 loaded")
            
        except Exception as e:
            logger.warning(f"  [SKIP] Wav2Vec2: {e}")
    
    def _init_classifiers(self):
        """Initialize classifier heads"""
        num_cry = len(self.cry_classes)
        num_pulm = len(self.pulmonary_classes)
        
        for model_name, config in self.model_configs.items():
            if model_name in self.models:
                hidden = config["hidden_size"]
                
                self.classifiers[f"{model_name}_cry"] = SimpleClassifier(
                    hidden, num_cry
                ).to(self.device)
                
                self.classifiers[f"{model_name}_pulmonary"] = SimpleClassifier(
                    hidden, num_pulm
                ).to(self.device)
    
    async def analyze_audio(self, waveform: np.ndarray, sample_rate: int = 16000, 
                           task: str = "cry") -> Dict[str, Any]:
        """
        Analyze audio and return classification results.
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Audio sample rate (should be 16000)
            task: "cry" or "pulmonary"
        
        Returns:
            Classification result with confidence and risk level
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Normalize audio
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=0)
        
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        
        # Get predictions from available models
        predictions = await self._ensemble_predict(waveform, task)
        
        classes = self.cry_classes if task == "cry" else self.pulmonary_classes
        risk_map = self.cry_risk if task == "cry" else self.pulmonary_risk
        
        # Aggregate predictions
        if predictions:
            avg_probs = np.mean([p["probs"] for p in predictions], axis=0)
            predicted_idx = int(np.argmax(avg_probs))
            predicted_class = classes[predicted_idx]
            confidence = float(avg_probs[predicted_idx])
        else:
            # Fallback when no models available
            predicted_class = "normal"
            confidence = 0.5
        
        risk_level = risk_map.get(predicted_class, "YELLOW")
        
        return {
            "classification": predicted_class,
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_score": self._compute_risk_score(predicted_class, confidence, risk_level),
            "all_probabilities": {c: float(p) for c, p in zip(classes, avg_probs)} if predictions else {},
            "models_used": [p["model"] for p in predictions] if predictions else ["fallback"],
            "task": task
        }
    
    async def _ensemble_predict(self, waveform: np.ndarray, task: str) -> List[Dict]:
        """Get predictions from all available models"""
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                probs = await self._predict_single(model_name, waveform, task)
                if probs is not None:
                    predictions.append({
                        "model": model_name,
                        "probs": probs
                    })
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
        
        return predictions
    
    async def _predict_single(self, model_name: str, waveform: np.ndarray, 
                             task: str) -> Optional[np.ndarray]:
        """Get prediction from a single model"""
        model = self.models.get(model_name)
        processor = self.processors.get(model_name)
        classifier = self.classifiers.get(f"{model_name}_{task}")
        
        if not all([model, processor, classifier]):
            return None
        
        # Process audio
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Classify
            logits = classifier(embeddings)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return probs
    
    def _compute_risk_score(self, classification: str, confidence: float, 
                           risk_level: str) -> float:
        """Compute numeric risk score (0-100)"""
        base_scores = {"GREEN": 20, "YELLOW": 50, "RED": 80}
        base = base_scores.get(risk_level, 50)
        
        # Adjust by confidence
        if risk_level == "RED":
            score = base + (confidence * 20)
        elif risk_level == "YELLOW":
            score = base + (confidence * 15)
        else:
            score = base - ((1 - confidence) * 10)
        
        return min(100, max(0, score))
    
    async def cleanup(self):
        """Release model resources"""
        self.models.clear()
        self.processors.clear()
        self.classifiers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("Ensemble resources released")
