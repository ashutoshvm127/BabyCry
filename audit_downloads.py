import os
from pathlib import Path

downloads = Path("d:/projects/cry analysuis/downloads")

print(f"{'Folder':<40} {'WAV':>6} {'MP3':>6} {'FLAC':>6} {'Total':>7} {'Status'}")
print("-" * 85)

empty_folders = []
data_folders = []
total_audio = 0

for folder in sorted(downloads.iterdir()):
    if not folder.is_dir():
        continue
    wav = len(list(folder.rglob("*.wav")))
    mp3 = len(list(folder.rglob("*.mp3")))
    flac = len(list(folder.rglob("*.flac")))
    total = wav + mp3 + flac
    total_audio += total
    status = "OK" if total > 0 else "EMPTY"
    if total == 0:
        empty_folders.append(folder.name)
    else:
        data_folders.append((folder.name, total))
    print(f"{folder.name:<40} {wav:>6} {mp3:>6} {flac:>6} {total:>7} {status}")

print("-" * 85)
print(f"{'TOTAL':<40} {'':>6} {'':>6} {'':>6} {total_audio:>7}")
print(f"\nFolders WITH data: {len(data_folders)}")
print(f"EMPTY folders:     {len(empty_folders)}")
print(f"\nEmpty folder names:")
for f in empty_folders:
    print(f"  - {f}")
