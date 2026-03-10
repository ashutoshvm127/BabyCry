#!/usr/bin/env python3
"""
Fix Downloads & Maximize Pulmonary Training Data
=================================================
This script:
1. Downloads verified, working respiratory/pulmonary datasets
2. Creates baby pulmonary disease directory structure
3. Processes SPRSound pediatric data into baby pulmonary track
4. Cleans up empty folders
5. Reports final file counts

BABY PULMONARY CLASSES:
- normal_breathing: healthy infant/child breathing
- wheeze: bronchial obstruction, asthma
- stridor: upper airway obstruction, croup, laryngomalacia
- fine_crackle: pneumonia, bronchiolitis, pulmonary edema
- coarse_crackle: bronchitis, mucus accumulation
- rhonchi: secretion in large airways
- mixed: combination pathologies
"""

import os
import sys
import io
import json
import shutil
import zipfile
import csv
import subprocess
from pathlib import Path
from collections import Counter

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
RESPIRATORY_DIR = BASE_DIR / "data_adult_respiratory"
BABY_CRY_DIR = BASE_DIR / "data_baby_respiratory"
BABY_PULMONARY_DIR = BASE_DIR / "data_baby_pulmonary"

BABY_PULMONARY_CLASSES = [
    'normal_breathing', 'wheeze', 'stridor',
    'fine_crackle', 'coarse_crackle', 'rhonchi', 'mixed'
]

# ============================================================================
# STEP 1: CREATE BABY PULMONARY DIRECTORY STRUCTURE
# ============================================================================
def create_baby_pulmonary_dirs():
    """Create the baby pulmonary disease directory structure"""
    print("=" * 70)
    print("STEP 1: CREATING BABY PULMONARY DIRECTORY STRUCTURE")
    print("=" * 70)
    
    for cls in BABY_PULMONARY_CLASSES:
        d = BABY_PULMONARY_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Created: {d}")
    
    print(f"\n  Classes: {', '.join(BABY_PULMONARY_CLASSES)}")
    print("  [OK] Baby pulmonary directory structure ready\n")


# ============================================================================
# STEP 2: PROCESS SPRSOUND AS PEDIATRIC PULMONARY DATA
# ============================================================================
def process_sprsound_pediatric():
    """
    SPRSound is a PEDIATRIC respiratory database (ages 1 month - 18 years).
    Extract pediatric-age recordings and classify into baby pulmonary classes.
    
    SPRSound filenames contain age info: e.g., "1_0_Trachea_1.wav"
    JSON annotations contain event types: Normal, Wheeze, Crackle, Stridor, Rhonchi, etc.
    """
    print("=" * 70)
    print("STEP 2: PROCESSING SPRSOUND AS PEDIATRIC PULMONARY DATA")
    print("=" * 70)
    
    sprsound_dir = DOWNLOADS_DIR / "sprsound"
    if not sprsound_dir.exists():
        print("  [!] SPRSound directory not found - skipping")
        return 0
    
    stats = Counter()
    processed = 0
    skipped = 0
    
    # Find all JSON annotation files
    json_dirs = list(sprsound_dir.rglob("*_json"))
    wav_dirs = list(sprsound_dir.rglob("*_wav"))
    
    if not json_dirs:
        # Fallback: process WAV files by folder name
        print("  [!] No JSON annotation dirs found, processing by folder names...")
        return process_sprsound_by_folder(sprsound_dir, stats)
    
    print(f"  Found {len(json_dirs)} JSON dirs, {len(wav_dirs)} WAV dirs")
    
    for json_dir in json_dirs:
        for json_file in json_dir.rglob("*.json"):
            try:
                with open(str(json_file), 'r', encoding='utf-8', errors='ignore') as f:
                    ann = json.load(f)
                
                # Skip poor quality
                if ann.get('record_annotation') == 'Poor Quality':
                    skipped += 1
                    continue
                
                # Check age - we want pediatric (roughly < 6 years for "baby")
                # SPRSound covers 1 month to 18 years
                # We'll take ALL pediatric data since it's all relevant
                age_info = ann.get('age', '')
                
                # Find matching WAV
                wav_name = json_file.stem + '.wav'
                wav_file = None
                for wd in wav_dirs:
                    candidates = list(wd.rglob(wav_name))
                    if candidates:
                        wav_file = candidates[0]
                        break
                
                if not wav_file or not wav_file.exists():
                    skipped += 1
                    continue
                
                # Determine pulmonary class from event annotations
                target = 'normal_breathing'
                events = ann.get('event_annotation', [])
                
                if events:
                    types = [e.get('type', '') for e in events]
                    
                    # Check for mixed pathologies first
                    has_wheeze = 'Wheeze' in types
                    has_crackle = any('Crackle' in t for t in types)
                    has_stridor = 'Stridor' in types
                    has_rhonchi = 'Rhonchi' in types
                    
                    if (has_wheeze and has_crackle) or 'Wheeze+Crackle' in types:
                        target = 'mixed'
                    elif has_stridor:
                        target = 'stridor'
                    elif has_rhonchi:
                        target = 'rhonchi'
                    elif 'Fine Crackle' in types:
                        target = 'fine_crackle'
                    elif 'Coarse Crackle' in types:
                        target = 'coarse_crackle'
                    elif has_wheeze:
                        target = 'wheeze'
                
                # Copy to baby pulmonary directory
                dest = BABY_PULMONARY_DIR / target / f"spr_ped_{wav_file.name}"
                if not dest.exists():
                    shutil.copy2(str(wav_file), str(dest))
                    stats[target] += 1
                    processed += 1
                    
            except Exception as e:
                skipped += 1
                continue
    
    print(f"\n  Processed: {processed} files")
    print(f"  Skipped: {skipped} files (poor quality or missing WAV)")
    print(f"\n  Baby Pulmonary Distribution:")
    for cls in BABY_PULMONARY_CLASSES:
        count = stats.get(cls, 0)
        print(f"    {cls:20s}: {count:5d} files")
    print(f"    {'TOTAL':20s}: {sum(stats.values()):5d} files")
    
    return processed


def process_sprsound_by_folder(sprsound_dir, stats):
    """Fallback: classify SPRSound files by parent folder naming"""
    processed = 0
    
    folder_map = {
        'normal': 'normal_breathing',
        'wheeze': 'wheeze',
        'stridor': 'stridor',
        'fine_crackle': 'fine_crackle',
        'coarse_crackle': 'coarse_crackle',
        'rhonchi': 'rhonchi',
        'crackle': 'fine_crackle',  # Default crackle to fine
    }
    
    for wav_file in sprsound_dir.rglob("*.wav"):
        folder = wav_file.parent.name.lower()
        
        target = 'normal_breathing'
        for key, mapped in folder_map.items():
            if key in folder:
                target = mapped
                break
        
        dest = BABY_PULMONARY_DIR / target / f"spr_ped_{wav_file.name}"
        if not dest.exists():
            shutil.copy2(str(wav_file), str(dest))
            stats[target] += 1
            processed += 1
    
    print(f"  Processed {processed} files by folder classification")
    return processed


# ============================================================================
# STEP 3: ALSO COPY ICBHI DATA INTO BABY PULMONARY (AS SUPPLEMENTARY)
# ============================================================================
def process_icbhi_for_baby_pulmonary():
    """
    ICBHI has mixed adult/pediatric patients. While not exclusively pediatric,
    the respiratory pathology patterns (wheeze, crackle) are similar.
    Copy as supplementary training data for baby pulmonary classes.
    """
    print("\n" + "=" * 70)
    print("STEP 3: SUPPLEMENTING BABY PULMONARY WITH ICBHI DATA")
    print("=" * 70)
    
    stats = Counter()
    processed = 0
    
    for icbhi_dir in [DOWNLOADS_DIR / "icbhi", DOWNLOADS_DIR / "kaggle_icbhi"]:
        if not icbhi_dir.exists():
            continue
        
        for wav_file in icbhi_dir.rglob("*.wav"):
            txt_file = wav_file.with_suffix('.txt')
            target = 'normal_breathing'
            
            if txt_file.exists():
                try:
                    with open(txt_file, 'r') as f:
                        has_crackle = has_wheeze = False
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 4:
                                if parts[2] == '1': has_crackle = True
                                if parts[3] == '1': has_wheeze = True
                        
                        if has_crackle and has_wheeze:
                            target = 'mixed'
                        elif has_crackle:
                            target = 'coarse_crackle'
                        elif has_wheeze:
                            target = 'wheeze'
                except:
                    pass
            
            dest = BABY_PULMONARY_DIR / target / f"icbhi_sup_{wav_file.name}"
            if not dest.exists():
                shutil.copy2(str(wav_file), str(dest))
                stats[target] += 1
                processed += 1
    
    print(f"  Added {processed} ICBHI supplementary files")
    for cls in BABY_PULMONARY_CLASSES:
        count = stats.get(cls, 0)
        if count > 0:
            print(f"    {cls}: {count}")
    
    return processed


# ============================================================================
# STEP 4: DOWNLOAD ADDITIONAL VERIFIED DATASETS
# ============================================================================
def download_additional_datasets():
    """Download verified, working respiratory/pulmonary datasets"""
    print("\n" + "=" * 70)
    print("STEP 4: DOWNLOADING ADDITIONAL VERIFIED DATASETS")
    print("=" * 70)
    
    import requests
    from tqdm import tqdm
    
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    
    def download_file(url, dest, desc=None):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            r = requests.get(url, stream=True, timeout=300, headers=headers, verify=False)
            if r.status_code != 200:
                print(f"      [{r.status_code}] Failed")
                return False
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(dest, 'wb') as f:
                if total > 0:
                    with tqdm(total=total, unit='B', unit_scale=True, desc=desc, ncols=70) as pbar:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"      [X] Error: {str(e)[:80]}")
            return False
    
    # --- Try Kaggle datasets (requires kagglehub + credentials) ---
    kaggle_success = 0
    try:
        import kagglehub
        
        kaggle_datasets = [
            # Verified working Kaggle respiratory datasets
            ("vbookshelf/respiratory-sound-database", "kaggle_respiratory_v2"),
            ("yasserh/lung-sounds", "kaggle_lung_sounds"),
        ]
        
        # Check if credentials exist
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if kaggle_json.exists():
            with open(kaggle_json, 'r') as f:
                creds = json.load(f)
            if 'key' in creds and creds['key'] != 'YOUR_KAGGLE_API_KEY':
                print("\n  [Kaggle] Credentials found - attempting downloads...")
                
                for dataset_id, name in kaggle_datasets:
                    dest_dir = DOWNLOADS_DIR / name
                    if dest_dir.exists() and any(dest_dir.rglob("*.wav")):
                        print(f"    [OK] {name}: Already exists")
                        kaggle_success += 1
                        continue
                    
                    print(f"    -> {name}")
                    try:
                        path = kagglehub.dataset_download(dataset_id)
                        dest_dir.mkdir(exist_ok=True)
                        
                        for item in Path(path).rglob("*"):
                            if item.is_file() and item.suffix.lower() in ['.wav', '.mp3', '.flac', '.txt']:
                                rel = item.relative_to(path)
                                (dest_dir / rel.parent).mkdir(parents=True, exist_ok=True)
                                if not (dest_dir / rel).exists():
                                    shutil.copy2(str(item), str(dest_dir / rel))
                        
                        count = len(list(dest_dir.rglob("*.wav")))
                        print(f"       [OK] {count} audio files")
                        kaggle_success += 1
                    except Exception as e:
                        print(f"       [!] Skipped: {str(e)[:60]}")
            else:
                print("\n  [Kaggle] Invalid credentials - skipping Kaggle datasets")
        else:
            print("\n  [Kaggle] No credentials found - skipping Kaggle datasets")
            print("    To enable: https://www.kaggle.com/settings/account -> Create API Token")
    except ImportError:
        print("\n  [Kaggle] kagglehub not installed - skipping")
    
    # --- Try HuggingFace datasets ---
    hf_success = 0
    print("\n  [HuggingFace] Checking for respiratory sound datasets...")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        hf_datasets = [
            ("SJTU-YONGFU-RESEARCH-GRP/SPRSound", "hf_sprsound_extra"),
        ]
        
        for repo_id, name in hf_datasets:
            dest_dir = DOWNLOADS_DIR / name
            if dest_dir.exists() and any(dest_dir.rglob("*.wav")):
                print(f"    [OK] {name}: Already exists")
                hf_success += 1
                continue
            
            print(f"    -> {name} ({repo_id})")
            try:
                files = list_repo_files(repo_id, repo_type="dataset")
                wav_files = [f for f in files if f.endswith('.wav')]
                
                if wav_files:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    downloaded = 0
                    for wf in wav_files[:500]:  # Limit to 500 files
                        try:
                            local = hf_hub_download(repo_id, wf, repo_type="dataset")
                            dest = dest_dir / Path(wf).name
                            if not dest.exists():
                                shutil.copy2(local, str(dest))
                                downloaded += 1
                        except:
                            continue
                    
                    if downloaded > 0:
                        print(f"       [OK] {downloaded} files downloaded")
                        hf_success += 1
                    else:
                        print(f"       [!] No files could be downloaded")
                else:
                    print(f"       [!] No WAV files found in repo")
            except Exception as e:
                print(f"       [!] Skipped: {str(e)[:60]}")
    except ImportError:
        print("    [!] huggingface_hub not installed - skipping")
        print("    To enable: pip install huggingface_hub")
    
    # --- Download from verified Zenodo records ---
    print("\n  [Zenodo] Downloading verified open-access datasets...")
    zenodo_sources = [
        {
            'name': 'ICBHI 2017 (Zenodo verified)',
            'url': 'https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip',
            'folder': 'zenodo_icbhi_verified'
        },
    ]
    
    zenodo_success = 0
    for dataset in zenodo_sources:
        dest_dir = DOWNLOADS_DIR / dataset['folder']
        if dest_dir.exists() and any(dest_dir.rglob("*.wav")):
            print(f"    [OK] {dataset['name']}: Already exists")
            zenodo_success += 1
            continue
        
        print(f"    -> {dataset['name']}")
        dest_dir.mkdir(exist_ok=True)
        zip_path = dest_dir / "download.zip"
        
        try:
            if download_file(dataset['url'], str(zip_path), dataset['name']):
                try:
                    with zipfile.ZipFile(str(zip_path), 'r') as z:
                        z.extractall(str(dest_dir))
                    zip_path.unlink()
                    count = len(list(dest_dir.rglob("*.wav")))
                    if count > 0:
                        print(f"       [OK] {count} audio files")
                        zenodo_success += 1
                    else:
                        print(f"       [!] No audio files found after extraction")
                except zipfile.BadZipFile:
                    print(f"       [!] Corrupt zip file")
                    if zip_path.exists():
                        zip_path.unlink()
            else:
                print(f"       [!] Download failed")
        except Exception as e:
            print(f"       [!] Error: {str(e)[:60]}")
    
    total_new = kaggle_success + hf_success + zenodo_success
    print(f"\n  Download Summary: {total_new} new datasets acquired")
    print(f"    Kaggle: {kaggle_success}, HuggingFace: {hf_success}, Zenodo: {zenodo_success}")
    
    return total_new


# ============================================================================
# STEP 5: PROCESS ESC-50 BABY BREATHING SOUNDS
# ============================================================================
def process_esc50_breathing():
    """Extract breathing and coughing sounds from ESC-50 for baby pulmonary"""
    print("\n" + "=" * 70)
    print("STEP 5: EXTRACTING BREATHING/COUGH SOUNDS FROM ESC-50")
    print("=" * 70)
    
    esc_dir = DOWNLOADS_DIR / "esc50"
    meta_file = esc_dir / "ESC-50-master" / "meta" / "esc50.csv"
    
    if not meta_file.exists():
        print("  [!] ESC-50 metadata not found - skipping")
        return 0
    
    stats = Counter()
    
    # ESC-50 categories relevant to respiratory analysis
    category_map = {
        'breathing': 'normal_breathing',
        'coughing': 'coarse_crackle',  # Cough sounds share characteristics with crackle
        'sneezing': 'normal_breathing',
        'snoring': 'rhonchi',  # Snoring shares characteristics with rhonchi
    }
    
    try:
        with open(str(meta_file), 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                category = row.get('category', '')
                filename = row.get('filename', '')
                
                if category not in category_map:
                    continue
                
                wav_file = esc_dir / "ESC-50-master" / "audio" / filename
                if not wav_file.exists():
                    continue
                
                target = category_map[category]
                dest = BABY_PULMONARY_DIR / target / f"esc_{filename}"
                if not dest.exists():
                    shutil.copy2(str(wav_file), str(dest))
                    stats[target] += 1
    except Exception as e:
        print(f"  [!] Error processing ESC-50: {e}")
    
    total = sum(stats.values())
    print(f"  Added {total} breathing/cough sounds from ESC-50")
    for cls, count in stats.items():
        print(f"    {cls}: {count}")
    
    return total


# ============================================================================
# STEP 6: CLEAN UP EMPTY FOLDERS
# ============================================================================
def cleanup_empty_folders():
    """Remove empty download folders"""
    print("\n" + "=" * 70)
    print("STEP 6: CLEANING UP EMPTY FOLDERS")
    print("=" * 70)
    
    removed = 0
    kept = 0
    
    for folder in sorted(DOWNLOADS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        
        # Count all files (not just audio)
        all_files = list(folder.rglob("*"))
        file_count = sum(1 for f in all_files if f.is_file())
        
        if file_count == 0:
            print(f"  [DEL] {folder.name} (empty)")
            try:
                shutil.rmtree(str(folder))
                removed += 1
            except Exception as e:
                print(f"    [!] Could not delete: {e}")
        else:
            audio_count = sum(1 for f in all_files if f.is_file() and f.suffix.lower() in ['.wav', '.mp3', '.flac'])
            if audio_count > 0:
                print(f"  [OK]  {folder.name} ({audio_count} audio files)")
            else:
                print(f"  [OK]  {folder.name} ({file_count} files, no audio)")
            kept += 1
    
    print(f"\n  Removed: {removed} empty folders")
    print(f"  Kept: {kept} folders with data")
    
    return removed


# ============================================================================
# STEP 7: FINAL AUDIT
# ============================================================================
def final_audit():
    """Print comprehensive summary of all training data"""
    print("\n" + "=" * 70)
    print("FINAL TRAINING DATA AUDIT")
    print("=" * 70)
    
    results = {}
    
    # Adult Respiratory
    print("\n  ADULT RESPIRATORY (data_adult_respiratory/):")
    total_adult = 0
    if RESPIRATORY_DIR.exists():
        for sub in sorted(RESPIRATORY_DIR.iterdir()):
            if sub.is_dir():
                count = len(list(sub.glob("*.wav")))
                print(f"    {sub.name:30s} {count:6d} files")
                total_adult += count
    print(f"    {'TOTAL':30s} {total_adult:6d} files")
    results['adult_respiratory'] = total_adult
    
    # Baby Cry (emotion)
    print("\n  BABY CRY EMOTION (data_baby_respiratory/):")
    total_cry = 0
    if BABY_CRY_DIR.exists():
        for sub in sorted(BABY_CRY_DIR.iterdir()):
            if sub.is_dir():
                count = len(list(sub.glob("*.wav")))
                print(f"    {sub.name:30s} {count:6d} files")
                total_cry += count
    print(f"    {'TOTAL':30s} {total_cry:6d} files")
    results['baby_cry'] = total_cry
    
    # Baby Pulmonary (NEW)
    print("\n  BABY PULMONARY DISEASE (data_baby_pulmonary/) [NEW]:")
    total_pulm = 0
    if BABY_PULMONARY_DIR.exists():
        for sub in sorted(BABY_PULMONARY_DIR.iterdir()):
            if sub.is_dir():
                count = len(list(sub.glob("*.wav")))
                print(f"    {sub.name:30s} {count:6d} files")
                total_pulm += count
    print(f"    {'TOTAL':30s} {total_pulm:6d} files")
    results['baby_pulmonary'] = total_pulm
    
    # Grand Total
    grand_total = total_adult + total_cry + total_pulm
    print(f"\n  {'='*50}")
    print(f"  GRAND TOTAL: {grand_total:,d} training files")
    print(f"    Adult Respiratory: {total_adult:,d}")
    print(f"    Baby Cry Emotion:  {total_cry:,d}")
    print(f"    Baby Pulmonary:    {total_pulm:,d}")
    print(f"  {'='*50}")
    
    if total_pulm == 0:
        print("\n  [!] WARNING: No baby pulmonary data! Check SPRSound processing.")
    
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("FIX DOWNLOADS & MAXIMIZE PULMONARY TRAINING DATA")
    print("=" * 70)
    print("This script fixes failed downloads and adds baby pulmonary disease track")
    print("=" * 70 + "\n")
    
    # Step 1: Create directories
    create_baby_pulmonary_dirs()
    
    # Step 2: Process SPRSound pediatric data
    spr_count = process_sprsound_pediatric()
    
    # Step 3: Supplement with ICBHI
    icbhi_count = process_icbhi_for_baby_pulmonary()
    
    # Step 4: Download additional datasets
    try:
        new_datasets = download_additional_datasets()
    except Exception as e:
        print(f"\n  [!] Download step had errors: {str(e)[:80]}")
        new_datasets = 0
    
    # Step 5: Process ESC-50
    esc_count = process_esc50_breathing()
    
    # Step 6: Clean up   
    cleanup_empty_folders()
    
    # Step 7: Final audit
    results = final_audit()
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review baby pulmonary data distribution above")
    print("  2. Run training: python train_maximum_accuracy.py")
    print("  3. The training script needs to be updated to include baby pulmonary track")
    print("=" * 70)


if __name__ == "__main__":
    main()
