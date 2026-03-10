"""Test the new model on pain cry samples"""
import os
import requests

DATA_DIR = r"D:\projects\cry analysuis\data_baby_respiratory"

def test_cry(category, num_samples=3):
    print(f"\n=== {category.upper()} ===")
    cls_dir = os.path.join(DATA_DIR, category)
    files = [f for f in os.listdir(cls_dir) if f.endswith(('.wav', '.mp3'))][:num_samples]
    
    correct = 0
    for f in files:
        file_path = os.path.join(cls_dir, f)
        with open(file_path, 'rb') as audio_file:
            response = requests.post(
                "http://localhost:8001/api/v1/analyze",
                files={"file": (f, audio_file, "audio/wav")}
            )
            if response.status_code == 200:
                result = response.json()
                classification = result.get('classification', {})
                predicted = classification.get('label', 'unknown')
                confidence = classification.get('confidence', 0) * 100
                
                # Check if prediction matches category
                is_correct = category.replace('_cry', '') in predicted.lower() or predicted == category
                mark = "✓" if is_correct else "✗"
                if is_correct:
                    correct += 1
                print(f"  {mark} {f[:35]:35} -> {predicted} ({confidence:.0f}%)")
            else:
                print(f"  ✗ Error: {response.status_code}")
    
    print(f"  Accuracy: {correct}/{len(files)}")

# Test all cry types
for category in ['pain_cry', 'discomfort_cry', 'distress_cry', 'hungry_cry', 'normal_cry']:
    test_cry(category)
