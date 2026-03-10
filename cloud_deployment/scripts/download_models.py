#!/usr/bin/env python3
"""
Model Download Script for Cloud Deployment

Downloads and caches all required AI models:
- DistilHuBERT
- AST (Audio Spectrogram Transformer)
- Wav2Vec2
- WavLM
- YAMNet (TensorFlow)
- PANNs

Also copies trained classifier weights to the appropriate locations.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_huggingface_models():
    """Download HuggingFace models"""
    try:
        from transformers import AutoModel, AutoFeatureExtractor, ASTForAudioClassification, ASTFeatureExtractor
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        
        models_to_download = [
            ("ntu-spml/distilhubert", "DistilHuBERT"),
            ("MIT/ast-finetuned-audioset-10-10-0.4593", "AST"),
            ("facebook/wav2vec2-base", "Wav2Vec2"),
            ("microsoft/wavlm-base", "WavLM"),
        ]
        
        for model_name, display_name in models_to_download:
            logger.info(f"Downloading {display_name}...")
            try:
                if "ast" in model_name.lower():
                    ASTFeatureExtractor.from_pretrained(model_name)
                    ASTForAudioClassification.from_pretrained(model_name)
                else:
                    AutoFeatureExtractor.from_pretrained(model_name)
                    AutoModel.from_pretrained(model_name)
                logger.info(f"  [OK] {display_name} downloaded")
            except Exception as e:
                logger.error(f"  [ERROR] {display_name}: {e}")
        
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")


def download_yamnet():
    """Download YAMNet from TensorFlow Hub"""
    try:
        import tensorflow_hub as hub
        
        logger.info("Downloading YAMNet...")
        hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("  [OK] YAMNet downloaded")
        
    except ImportError:
        logger.error("tensorflow_hub not installed. Run: pip install tensorflow tensorflow_hub")
    except Exception as e:
        logger.error(f"  [ERROR] YAMNet: {e}")


def copy_trained_weights():
    """Copy trained classifier weights to cloud deployment"""
    # Paths
    project_root = Path(__file__).parent.parent
    trained_classifiers = project_root / "trained_classifiers"
    cloud_models = project_root / "cloud_deployment" / "models" / "trained_weights"
    
    if not trained_classifiers.exists():
        logger.warning(f"Trained classifiers not found at {trained_classifiers}")
        return
    
    # Create target directories
    cry_target = cloud_models / "cry"
    pulmonary_target = cloud_models / "pulmonary"
    cry_target.mkdir(parents=True, exist_ok=True)
    pulmonary_target.mkdir(parents=True, exist_ok=True)
    
    # Copy cry classifiers
    cry_source = trained_classifiers / "cry"
    if cry_source.exists():
        for pt_file in cry_source.glob("*.pt"):
            target = cry_target / pt_file.name
            shutil.copy2(pt_file, target)
            logger.info(f"  Copied: {pt_file.name} -> cry/")
    
    # Copy pulmonary classifiers
    pulmonary_source = trained_classifiers / "pulmonary"
    if pulmonary_source.exists():
        for pt_file in pulmonary_source.glob("*.pt"):
            target = pulmonary_target / pt_file.name
            shutil.copy2(pt_file, target)
            logger.info(f"  Copied: {pt_file.name} -> pulmonary/")
    
    logger.info("[OK] Trained weights copied")


def copy_ast_models():
    """Copy AST fine-tuned models"""
    project_root = Path(__file__).parent.parent
    cloud_root = project_root / "cloud_deployment"
    
    # Copy AST baby cry model
    ast_cry_source = project_root / "ast_baby_cry_optimized"
    ast_cry_target = cloud_root / "ast_baby_cry_optimized"
    if ast_cry_source.exists():
        shutil.copytree(ast_cry_source, ast_cry_target, dirs_exist_ok=True)
        logger.info("[OK] AST baby cry model copied")
    
    # Copy AST respiratory model
    ast_resp_source = project_root / "ast_respiratory_optimized"
    ast_resp_target = cloud_root / "ast_respiratory_optimized"
    if ast_resp_source.exists():
        shutil.copytree(ast_resp_source, ast_resp_target, dirs_exist_ok=True)
        logger.info("[OK] AST respiratory model copied")


def verify_models():
    """Verify all models are available"""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    cloud_models = project_root / "cloud_deployment" / "models" / "trained_weights"
    
    required_files = [
        cloud_models / "cry" / "distilhubert_cry.pt",
        cloud_models / "cry" / "ast_cry.pt",
        cloud_models / "pulmonary" / "distilhubert_pulmonary.pt",
        cloud_models / "pulmonary" / "ast_pulmonary.pt",
    ]
    
    logger.info("\nVerifying models...")
    all_ok = True
    for path in required_files:
        if path.exists():
            logger.info(f"  [OK] {path.name}")
        else:
            logger.warning(f"  [MISSING] {path.name}")
            all_ok = False
    
    return all_ok


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for cloud deployment")
    parser.add_argument("--force", action="store_true", help="Force re-download of all models")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace models")
    parser.add_argument("--skip-yamnet", action="store_true", help="Skip YAMNet")
    parser.add_argument("--skip-copy", action="store_true", help="Skip copying trained weights")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MODEL DOWNLOAD SCRIPT FOR CLOUD DEPLOYMENT")
    logger.info("=" * 60)
    
    if not args.skip_hf:
        logger.info("\n[1/4] Downloading HuggingFace models...")
        download_huggingface_models()
    
    if not args.skip_yamnet:
        logger.info("\n[2/4] Downloading YAMNet...")
        download_yamnet()
    
    if not args.skip_copy:
        logger.info("\n[3/4] Copying trained classifier weights...")
        copy_trained_weights()
        
        logger.info("\n[4/4] Copying AST fine-tuned models...")
        copy_ast_models()
    
    logger.info("\n" + "=" * 60)
    if verify_models():
        logger.info("ALL MODELS READY FOR DEPLOYMENT")
    else:
        logger.warning("SOME MODELS MISSING - Check above")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
