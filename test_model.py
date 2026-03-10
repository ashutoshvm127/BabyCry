#!/usr/bin/env python3
"""Quick test script for the baby cry model."""

import requests
import os

# Test different classes
classes_to_test = ['hungry_cry', 'pain_cry', 'discomfort_cry', 'cold_cry', 'sleepy_cry']
results = {}

for class_name in classes_to_test:
    folder = f'data_baby_respiratory/{class_name}'
    files = [f for f in os.listdir(folder) if f.endswith('.wav')][:3]
    
    correct = 0
    print(f"\n=== {class_name.upper()} ===")
    for fname in files:
        path = f'{folder}/{fname}'
        with open(path, 'rb') as f:
            audio_bytes = f.read()
        
        files_req = {'file': ('test.wav', audio_bytes, 'audio/wav')}
        response = requests.post('http://localhost:8001/api/v1/analyze', files=files_req)
        result = response.json()
        predicted = result['classification']['label']
        probs = result['classification']['top_ai_predictions']
        
        if predicted == class_name:
            correct += 1
            marker = "✓"
        else:
            marker = "✗"
        print(f"  {marker} Predicted: {predicted:15s} | {class_name}={probs.get(class_name,0):.1%}")
    
    results[class_name] = f"{correct}/{len(files)}"

print("\n=== SUMMARY ===")
for cls, acc in results.items():
    print(f"  {cls}: {acc}")
