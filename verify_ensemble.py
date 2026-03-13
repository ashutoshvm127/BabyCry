#!/usr/bin/env python3
"""Verify the 6-backbone ensemble checkpoint."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ckpt = torch.load('trained_classifiers/6backbone_ensemble.pt', map_location='cpu', weights_only=False)
print('=== 6-BACKBONE ENSEMBLE VERIFICATION ===')
print('Model type:', ckpt['model_type'])
print('Backbones:', list(ckpt['backbones'].keys()))
print('Num classes:', ckpt['num_classes'])
print('Classes:', ckpt['classes'])
print()

class FusionClassifier(nn.Module):
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

print('Loading and testing each backbone classifier:')
for name, info in ckpt['backbones'].items():
    dim = info['dim']
    model = FusionClassifier(dim, 85, 20)
    model.load_state_dict(info['state_dict'])
    model.eval()
    x = torch.randn(1, dim + 85)
    out = model(x)
    pred = out.argmax(1).item()
    print(f'  {name:15s}: dim={dim}, hf={info["hf_name"]}, loaded OK, test pred={ckpt["classes"][pred]}')

print()
print('Individual accuracies:')
for k, v in ckpt['metadata']['individual_val_accs'].items():
    print(f'  {k:15s}: {v*100:.2f}%')
print(f'\nENSEMBLE ACCURACY: {ckpt["metadata"]["ensemble_val_acc"]*100:.2f}%')

print()
print('Risk levels:')
for cls, risk in sorted(ckpt['risk_levels'].items(), key=lambda x: x[1]):
    level = ['GREEN', 'YELLOW', 'ORANGE', 'RED'][risk]
    print(f'  {cls:25s}: {level} ({risk})')

print('\n=== ALL CHECKS PASSED ===')
