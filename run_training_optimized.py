
import os
import sys
import subprocess

# Set PyTorch memory optimization BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("=" * 70)
print("OPTIMIZED TRAINING FOR RTX 4050 (6GB VRAM)")
print("=" * 70)
print()
print("Memory Optimization Settings:")
print("  ✓ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
print("  ✓ Stage 1: batch_size=4, grad_accum=2 (effective batch=8)")
print("  ✓ Stage 2: batch_size=2, grad_accum=4 (effective batch=8)")
print("  ✓ Max audio duration: 5 seconds")
print("  ✓ Gradient checkpointing enabled on encoder")
print()
print("Expected Memory Usage:")
print("  - Stage 1 (encoder frozen): ~5.5 GB")
print("  - Stage 2 (encoder unfrozen): ~5.8 GB")
print()
print("=" * 70)
print()

# Run the training script
try:
    result = subprocess.run(
        [sys.executable, 'train_maximum_accuracy.py'],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\n[!] Training interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Failed to run training: {e}")
    sys.exit(1)
