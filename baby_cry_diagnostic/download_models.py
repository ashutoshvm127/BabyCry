#!/usr/bin/env python3
"""
Baby Cry Analysis - Medical AI Model Loader

Downloads and configures pre-trained models for:
1. Baby cry classification (hungry, pain, sleepy, etc.)
2. Respiratory health indicators from cry acoustics

Medical Basis:
- Fundamental Frequency (f0 > 600 Hz): High pitch correlates with respiratory distress
- Harmonic-to-Noise Ratio (HNR < 5 dB): Indicates turbulent airflow, possible congestion
- Cry Duration Patterns: Short fragmented cries may indicate breathing difficulty
- Spectral Features: Changes in spectral envelope correlate with laryngeal/pulmonary conditions

References:
- LaGasse et al. (2005): "Acoustic cry analysis and neurodevelopmental outcome"
- Várallyay (2007): "The melody of crying"
- Alaie & Naghsh-Nilchi (2016): "Automatic infant cry classification"
"""

import os
import sys
import json
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent
MODELS_DIR = OUTPUT_DIR / "models_pretrained"
MODELS_DIR.mkdir(exist_ok=True)

def download_baby_cry_model():
    """Download foduucom/baby-cry-classification from HuggingFace"""
    print("=" * 60)
    print("DOWNLOADING BABY CRY CLASSIFICATION MODEL")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        import torch
        
        model_name = "foduucom/baby-cry-classification"
        save_path = MODELS_DIR / "baby_cry_ast"
        
        print(f"\n[1] Downloading model: {model_name}")
        print("    This is an Audio Spectrogram Transformer (AST) model")
        print("    Fine-tuned for baby cry classification")
        
        # Download model and processor
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        processor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Save locally
        print(f"\n[2] Saving to: {save_path}")
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        
        # Save label mapping
        labels = model.config.id2label
        with open(save_path / "labels.json", "w") as f:
            json.dump(labels, f, indent=2)
        
        print(f"\n[3] Model labels: {list(labels.values())}")
        print("\n[OK] Baby cry model downloaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        return False


def download_hubert_model():
    """Download DistilHuBERT for audio embeddings"""
    print("\n" + "=" * 60)
    print("DOWNLOADING DISTILHUBERT FOR BIOMARKER ANALYSIS")
    print("=" * 60)
    
    try:
        from transformers import AutoModel, AutoFeatureExtractor
        
        model_name = "ntu-spml/distilhubert"
        save_path = MODELS_DIR / "distilhubert"
        
        print(f"\n[1] Downloading: {model_name}")
        
        model = AutoModel.from_pretrained(model_name)
        processor = AutoFeatureExtractor.from_pretrained(model_name)
        
        print(f"\n[2] Saving to: {save_path}")
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        
        print("\n[OK] DistilHuBERT downloaded!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        return False


def create_model_config():
    """Create configuration file for the server to use these models"""
    config = {
        "baby_cry_model": {
            "path": str(MODELS_DIR / "baby_cry_ast"),
            "type": "ast",
            "description": "AST model fine-tuned on baby cry sounds",
            "source": "foduucom/baby-cry-classification"
        },
        "embeddings_model": {
            "path": str(MODELS_DIR / "distilhubert"),
            "type": "hubert",
            "description": "DistilHuBERT for acoustic feature extraction"
        },
        "medical_thresholds": {
            "f0_high_risk": 600,
            "f0_very_high_risk": 800,
            "hnr_low_risk": 5,
            "spectral_centroid_high": 2000,
            "description": "Thresholds for pulmonary health indicators"
        },
        "pulmonary_indicators": {
            "high_f0": {
                "threshold": ">600 Hz",
                "indication": "Respiratory distress, possible laryngomalacia",
                "action": "Monitor breathing pattern, consult pediatrician if persistent"
            },
            "low_hnr": {
                "threshold": "<5 dB",
                "indication": "Turbulent airflow, possible nasal/airway congestion",
                "action": "Check for nasal congestion, ensure clear airways"
            },
            "irregular_cry": {
                "threshold": "Fragmented, short bursts",
                "indication": "Breathing difficulty, possible respiratory infection",
                "action": "Observe breathing rate, seek medical attention if distressed"
            },
            "weak_cry": {
                "threshold": "Low energy, RMS < 0.02",
                "indication": "Fatigue or respiratory weakness",
                "action": "Monitor feeding, check temperature, consult if persists"
            }
        }
    }
    
    config_path = OUTPUT_DIR / "backend" / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] Model config saved to: {config_path}")
    return config


def main():
    print("\n" + "=" * 70)
    print("BABY CRY DIAGNOSTIC - AI MODEL SETUP")
    print("=" * 70)
    print("""
This script downloads pre-trained AI models for baby cry analysis:

1. Baby Cry Classification Model (foduucom/baby-cry-classification)
   - Classifies cry type: hungry, pain, sleepy, discomfort, etc.
   - Based on Audio Spectrogram Transformer (AST)
   
2. DistilHuBERT for Acoustic Analysis
   - Extracts deep acoustic features
   - Used for medical biomarker correlation

MEDICAL BASIS FOR PULMONARY ASSESSMENT:
---------------------------------------
Baby cries contain acoustic signatures that correlate with respiratory health:

• High Fundamental Frequency (f0 > 600 Hz)
  → May indicate respiratory distress, laryngeal inflammation
  
• Low Harmonic-to-Noise Ratio (HNR < 5 dB)
  → Turbulent airflow suggests nasal/airway congestion
  
• Cry Melody Patterns
  → Falling melody = normal; Rising/flat = potential distress
  
• Spectral Energy Distribution
  → Abnormal patterns may indicate laryngomalacia or bronchiolitis

NOTE: This is a SCREENING tool, not a diagnostic device.
      Always consult a pediatrician for medical concerns.
""")
    
    # Check for required packages
    try:
        import torch
        import transformers
        print(f"[OK] PyTorch {torch.__version__}")
        print(f"[OK] Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("Please run: pip install torch transformers")
        return False
    
    # Download models
    success = True
    success = download_baby_cry_model() and success
    success = download_hubert_model() and success
    
    if success:
        create_model_config()
        print("\n" + "=" * 70)
        print("SUCCESS! Models downloaded and configured.")
        print("=" * 70)
        print("\nRestart the server to use the AI models:")
        print("  cd baby_cry_diagnostic/backend")
        print("  python simple_server.py")
    else:
        print("\n[!] Some models failed to download.")
        print("    The system will use feature-based analysis instead.")
    
    return success


if __name__ == "__main__":
    main()
