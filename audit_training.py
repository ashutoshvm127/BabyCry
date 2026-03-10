import os
from pathlib import Path

# Check training data directories
dirs = {
    "data_adult_respiratory": Path("d:/projects/cry analysuis/data_adult_respiratory"),
    "data_baby_respiratory": Path("d:/projects/cry analysuis/data_baby_respiratory"),
}

with open("d:/projects/cry analysuis/training_data_audit.txt", "w", encoding="utf-8") as f:
    for name, d in dirs.items():
        f.write(f"\n{'='*60}\n{name}\n{'='*60}\n")
        if not d.exists():
            f.write("  DOES NOT EXIST\n")
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                wav = len(list(sub.rglob("*.wav")))
                f.write(f"  {sub.name:30s} {wav:6d} wav files\n")
        total = len(list(d.rglob("*.wav")))
        f.write(f"  {'TOTAL':30s} {total:6d} wav files\n")
    
    # Also check coswara for actual audio content
    f.write(f"\n{'='*60}\nCOSWARA DATA CHECK\n{'='*60}\n")
    coswara = Path("d:/projects/cry analysuis/downloads/coswara/Coswara-Data-master")
    if coswara.exists():
        sample_dirs = list(coswara.iterdir())[:5]
        for sd in sample_dirs:
            f.write(f"  {sd.name}/\n")
            if sd.is_dir():
                for item in list(sd.iterdir())[:10]:
                    f.write(f"    {item.name} ({item.stat().st_size if item.is_file() else 'dir'})\n")
        f.write(f"  Total subdirectories: {len(list(coswara.iterdir()))}\n")
        wav_count = len(list(coswara.rglob("*.wav")))
        f.write(f"  WAV files in coswara: {wav_count}\n")
    
    # Check zip files in downloads
    f.write(f"\n{'='*60}\nZIP FILES IN DOWNLOADS (potential unextracted data)\n{'='*60}\n")
    dl = Path("d:/projects/cry analysuis/downloads")
    for zf in sorted(dl.rglob("*.zip")):
        size_mb = zf.stat().st_size / (1024*1024)
        f.write(f"  {str(zf.relative_to(dl)):50s} {size_mb:8.1f} MB\n")

print("Done - check training_data_audit.txt")
