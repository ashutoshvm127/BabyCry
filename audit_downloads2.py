import os, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

downloads = Path("d:/projects/cry analysuis/downloads")

results = []
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
    status = "HAS_DATA" if total > 0 else "EMPTY"
    if total == 0:
        empty_folders.append(folder.name)
    else:
        data_folders.append((folder.name, total))
    results.append(f"{folder.name:40s} wav={wav:5d} mp3={mp3:5d} flac={flac:5d} total={total:6d} [{status}]")

# Write to UTF-8 file
with open("d:/projects/cry analysuis/audit_output.txt", "w", encoding="utf-8") as f:
    f.write("DOWNLOAD FOLDER AUDIT\n")
    f.write("=" * 90 + "\n\n")
    for r in results:
        f.write(r + "\n")
    f.write("\n" + "=" * 90 + "\n")
    f.write(f"TOTAL AUDIO FILES: {total_audio}\n")
    f.write(f"Folders WITH data: {len(data_folders)}\n")
    f.write(f"EMPTY folders: {len(empty_folders)}\n\n")
    f.write("FOLDERS WITH DATA:\n")
    for name, count in sorted(data_folders, key=lambda x: -x[1]):
        f.write(f"  {name:40s} {count:6d} files\n")
    f.write(f"\nEMPTY FOLDERS ({len(empty_folders)}):\n")
    for name in empty_folders:
        f.write(f"  - {name}\n")

print("Done - check audit_output.txt")
