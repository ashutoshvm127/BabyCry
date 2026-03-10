#!/usr/bin/env python3
"""
Training Script for 6-Backbone Ensemble Model

Trains classifier heads for baby cry and pulmonary detection:
1. DistilHuBERT - Primary cry detection
2. AST - Audio spectrogram analysis
3. YAMNet - General audio events
4. Wav2Vec2 - Cry vocalization patterns
5. WavLM - Noise-robust detection
6. PANNs CNN14 - Pulmonary sounds

Usage:
    python train_6backbone_ensemble.py --task cry
    python train_6backbone_ensemble.py --task pulmonary
    python train_6backbone_ensemble.py --task both
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "baby_cry_diagnostic"))

from baby_cry_diagnostic.backend.models.ensemble import EnsembleModel


class AudioDataset(Dataset):
    """Dataset for loading audio files with labels"""
    
    def __init__(self, file_paths: list, labels: list, label_map: dict,
                 sample_rate: int = 16000, max_duration: float = 5.0):
        self.file_paths = file_paths
        self.labels = labels
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.file_paths[idx]
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform = np.zeros(self.max_length, dtype=np.float32)
        
        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        # Pad or truncate
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]
        
        label = self.label_map[self.labels[idx]]
        
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label)


def collect_cry_data(data_dir: Path):
    """Collect baby cry training data"""
    # Direct folder name to label mapping
    folder_to_label = {
        "hungry_cry": "hungry",
        "pain_cry": "pain", 
        "sleepy_cry": "sleepy",
        "discomfort_cry": "discomfort",  # This is key - correctly map discomfort
        "cold_cry": "cold_hot",
        "distress_cry": "pathological",
        "tired_cry": "tired",
        "normal_cry": "normal",
        # Additional mappings
        "belly_pain": "belly_pain",
        "burping": "burping",
        "scared": "scared",
        "lonely": "lonely",
    }
    
    file_paths = []
    labels = []
    
    # Check data_baby_respiratory directory
    baby_resp_dir = data_dir / "data_baby_respiratory"
    if baby_resp_dir.exists():
        for class_dir in baby_resp_dir.iterdir():
            if class_dir.is_dir():
                folder_name = class_dir.name.lower()
                # Direct mapping from folder name
                mapped_label = folder_to_label.get(folder_name)
                
                # If no direct match, try partial matching
                if mapped_label is None:
                    for folder_pattern, label in folder_to_label.items():
                        if folder_pattern in folder_name or folder_name in folder_pattern:
                            mapped_label = label
                            break
                
                if mapped_label:
                    audio_count = 0
                    for audio_file in class_dir.glob("*.wav"):
                        file_paths.append(str(audio_file))
                        labels.append(mapped_label)
                        audio_count += 1
                    for audio_file in class_dir.glob("*.mp3"):
                        file_paths.append(str(audio_file))
                        labels.append(mapped_label)
                        audio_count += 1
                    print(f"  {folder_name} -> {mapped_label}: {audio_count} files")
    
    print(f"Collected {len(file_paths)} cry audio samples")
    return file_paths, labels


def collect_pulmonary_data(data_dir: Path):
    """Collect pulmonary/respiratory training data"""
    pulmonary_classes = {
        "normal": ["normal", "normal_breathing"],
        "wheeze": ["wheeze"],
        "crackle": ["crackle", "coarse_crackle", "fine_crackle"],
        "stridor": ["stridor"],
        "rhonchi": ["rhonchi"],
        "bronchiolitis": ["bronchiolitis"],
        "pneumonia": ["pneumonia"],
        "asthma": ["asthma"]
    }
    
    file_paths = []
    labels = []
    
    # Check data_baby_pulmonary directory
    baby_pulm_dir = data_dir / "data_baby_pulmonary"
    if baby_pulm_dir.exists():
        for class_dir in baby_pulm_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                mapped_label = None
                
                for pulm_class, patterns in pulmonary_classes.items():
                    for pattern in patterns:
                        if pattern in class_name:
                            mapped_label = pulm_class
                            break
                    if mapped_label:
                        break
                
                if mapped_label:
                    for audio_file in class_dir.glob("*.wav"):
                        file_paths.append(str(audio_file))
                        labels.append(mapped_label)
                    for audio_file in class_dir.glob("*.mp3"):
                        file_paths.append(str(audio_file))
                        labels.append(mapped_label)
    
    # Also check data_adult_respiratory
    adult_resp_dir = data_dir / "data_adult_respiratory"
    if adult_resp_dir.exists():
        for class_dir in adult_resp_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                mapped_label = None
                
                for pulm_class, patterns in pulmonary_classes.items():
                    for pattern in patterns:
                        if pattern in class_name:
                            mapped_label = pulm_class
                            break
                    if mapped_label:
                        break
                
                if mapped_label:
                    for audio_file in class_dir.glob("*.wav"):
                        file_paths.append(str(audio_file))
                        labels.append(mapped_label)
    
    print(f"Collected {len(file_paths)} pulmonary audio samples")
    return file_paths, labels


async def extract_embeddings(model: EnsembleModel, dataloader: DataLoader, 
                            model_name: str, device: torch.device):
    """Extract embeddings from a backbone model"""
    embeddings_list = []
    labels_list = []
    
    backbone = model.models.get(model_name)
    processor = model.processors.get(model_name)
    
    if backbone is None:
        return None, None
    
    backbone.eval()
    
    for batch_waveforms, batch_labels in tqdm(dataloader, desc=f"Extracting {model_name}"):
        with torch.no_grad():
            for waveform, label in zip(batch_waveforms, batch_labels):
                waveform_np = waveform.numpy()
                
                if model_name == "yamnet":
                    # YAMNet extraction
                    scores, emb, _ = backbone(waveform_np)
                    embedding = emb.numpy().mean(axis=0)
                elif model_name == "panns":
                    # PANNs extraction
                    mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=16000, n_mels=128, fmax=8000)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    embedding = backbone(mel_tensor).cpu().numpy()[0]
                elif model_name == "ast":
                    # AST extraction - use base model for hidden states
                    inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    # Get hidden states from ASTForAudioClassification
                    outputs = backbone(**inputs, output_hidden_states=True)
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]
                    else:
                        # Fallback: use logits as features
                        embedding = outputs.logits.cpu().numpy()[0]
                else:
                    # Transformer models
                    inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = backbone(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                
                embeddings_list.append(embedding)
                labels_list.append(label.item())
    
    return np.array(embeddings_list), np.array(labels_list)


def train_classifier(embeddings: np.ndarray, labels: np.ndarray, 
                    classifier: nn.Module, device: torch.device,
                    epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
    """Train a classifier head on extracted embeddings"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    
    classifier.train()
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        indices = torch.randperm(len(X_train_t))
        total_loss = 0.0
        correct = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            outputs = classifier(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()
        
        train_acc = correct / len(X_train_t)
        
        # Validation
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_acc = (val_outputs.argmax(dim=1) == y_val_t).float().mean().item()
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best state
    if best_state:
        classifier.load_state_dict(best_state)
    
    return best_val_acc


async def train_task(ensemble: EnsembleModel, task: str, data_dir: Path, 
                     save_dir: Path, epochs: int = 50):
    """Train all classifier heads for a specific task"""
    
    print(f"\n{'='*60}")
    print(f"Training 6-Backbone Ensemble for {task.upper()} classification")
    print(f"{'='*60}")
    
    # Collect data
    if task == "cry":
        file_paths, labels = collect_cry_data(data_dir)
        classes = ensemble.cry_classes
    else:
        file_paths, labels = collect_pulmonary_data(data_dir)
        classes = ensemble.pulmonary_classes
    
    if len(file_paths) == 0:
        print(f"No training data found for {task}!")
        return
    
    # Create label map
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Classes found: {unique_labels}")
    print(f"Samples per class: {[(l, labels.count(l)) for l in unique_labels]}")
    
    # Create dataset
    dataset = AudioDataset(file_paths, labels, label_map)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Train each backbone's classifier
    results = {}
    device = ensemble.device
    
    for model_name in ["distilhubert", "wav2vec2", "wavlm", "ast", "yamnet", "panns"]:
        if ensemble.models.get(model_name) is None:
            print(f"\n[SKIP] {model_name} - not loaded")
            continue
        
        print(f"\n[TRAIN] {model_name} classifier for {task}")
        
        # Extract embeddings
        embeddings, embed_labels = await extract_embeddings(
            ensemble, dataloader, model_name, device
        )
        
        if embeddings is None or embed_labels is None:
            print(f"  Failed to extract embeddings from {model_name}")
            continue
        
        print(f"  Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
        
        # Get classifier
        classifier_name = f"{model_name}_{task}"
        classifier = ensemble.classifiers.get(classifier_name)
        
        if classifier is None:
            print(f"  No classifier found for {classifier_name}")
            continue
        
        # Train
        val_acc = train_classifier(
            embeddings, embed_labels, classifier, device,
            epochs=epochs, lr=1e-3
        )
        
        results[model_name] = val_acc
        print(f"  Best validation accuracy: {val_acc:.4f}")
        
        # Save classifier immediately after training
        task_save_dir = save_dir / task
        task_save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            classifier.state_dict(),
            task_save_dir / f"{classifier_name}.pt"
        )
        print(f"  Saved {classifier_name}.pt")
    
    # Save classifiers
    task_save_dir = save_dir / task
    task_save_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in results.keys():
        classifier_name = f"{model_name}_{task}"
        classifier = ensemble.classifiers.get(classifier_name)
        if classifier:
            torch.save(
                classifier.state_dict(),
                task_save_dir / f"{classifier_name}.pt"
            )
    
    print(f"\nClassifiers saved to {task_save_dir}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Training Results for {task.upper()}")
    print(f"{'='*60}")
    for model_name, acc in sorted(results.items(), key=lambda x: -x[1]):
        weight = ensemble.model_configs[model_name][f"{task}_weight"]
        print(f"  {model_name:15} - Accuracy: {acc:.4f}, Weight: {weight:.2f}")


async def main():
    parser = argparse.ArgumentParser(description="Train 6-Backbone Ensemble")
    parser.add_argument("--task", choices=["cry", "pulmonary", "both"], default="both",
                       help="Which task to train")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per backbone")
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument("--save-dir", type=str, default="trained_classifiers", 
                       help="Directory to save trained classifiers")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir).resolve()
    
    print("="*60)
    print("6-Backbone Ensemble Training")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Task: {args.task}")
    print(f"Epochs: {args.epochs}")
    print("="*60)
    
    # Initialize ensemble
    print("\nInitializing 6-backbone ensemble...")
    ensemble = EnsembleModel()
    await ensemble.initialize()
    
    # Print model info
    info = ensemble.get_model_info()
    print(f"\nModels loaded: {info['loaded_backbones']}/{info['total_backbones']}")
    for name, model_info in info["models"].items():
        status = "✓" if model_info["loaded"] else "✗"
        print(f"  [{status}] {name}: {model_info['description']}")
    
    # Train
    if args.task in ["cry", "both"]:
        await train_task(ensemble, "cry", data_dir, save_dir, args.epochs)
    
    if args.task in ["pulmonary", "both"]:
        await train_task(ensemble, "pulmonary", data_dir, save_dir, args.epochs)
    
    # Cleanup
    await ensemble.cleanup()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
