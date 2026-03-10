import os, io, sys
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = Path("d:/projects/cry analysuis")

dirs = {
    "data_adult_respiratory": BASE_DIR / "data_adult_respiratory",
    "data_baby_respiratory": BASE_DIR / "data_baby_respiratory",
    "data_baby_pulmonary": BASE_DIR / "data_baby_pulmonary",
}

with open(str(BASE_DIR / "final_audit.txt"), "w", encoding="utf-8") as f:
    grand_total = 0
    
    for name, d in dirs.items():
        f.write(f"\n{'='*60}\n{name}\n{'='*60}\n")
        if not d.exists():
            f.write("  DOES NOT EXIST\n")
            continue
        total = 0
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                count = len(list(sub.glob("*.wav")))
                f.write(f"  {sub.name:30s} {count:6d} wav files\n")
                total += count
        f.write(f"  {'TOTAL':30s} {total:6d} wav files\n")
        grand_total += total
    
    f.write(f"\n{'='*60}\nGRAND TOTAL: {grand_total} training files\n{'='*60}\n")
    
    # Downloads folder status
    dl = BASE_DIR / "downloads"
    f.write(f"\n{'='*60}\nDOWNLOADS FOLDER STATUS\n{'='*60}\n")
    folder_count = 0
    empty_count = 0
    for folder in sorted(dl.iterdir()):
        if folder.is_dir():
            folder_count += 1
            wav = len(list(folder.rglob("*.wav")))
            mp3 = len(list(folder.rglob("*.mp3")))
            total = wav + mp3
            status = f"{total} audio" if total > 0 else "EMPTY"
            f.write(f"  {folder.name:40s} {status}\n")
            if total == 0:
                empty_count += 1
    f.write(f"\n  Total folders: {folder_count}, Empty: {empty_count}\n")

print("Done - check final_audit.txt")
