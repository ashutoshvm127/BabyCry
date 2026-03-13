#!/usr/bin/env python3
"""
Complete Dataset Organizer and Downloader for Baby Cry Diagnostic System
=========================================================================
Organizes existing data and downloads missing datasets for 20-class classification:

CRY CLASSES (12):
- hungry_cry, pain_cry, sleepy_cry, discomfort_cry, cold_cry, tired_cry
- normal_cry, distress_cry, belly_pain_cry, burping_cry
- pathological_cry (high-risk medical conditions)
- asphyxia_cry (critical oxygen deprivation)

RESPIRATORY/PULMONARY CLASSES (8):
- normal_breathing, wheeze, stridor, rhonchi
- fine_crackle, coarse_crackle, mixed
- bronchiolitis (includes sepsis respiratory markers)

Author: Baby Cry Diagnostic System
"""

import os
import sys
import io
import json
import shutil
import zipfile
from pathlib import Path
from collections import Counter
import subprocess

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
BABY_CRY_DIR = BASE_DIR / "data_baby_respiratory"
BABY_PULMONARY_DIR = BASE_DIR / "data_baby_pulmonary"

# Full classification scheme
CRY_CLASSES = [
    'hungry_cry', 'pain_cry', 'sleepy_cry', 'discomfort_cry', 
    'cold_cry', 'tired_cry', 'normal_cry', 'distress_cry',
    'belly_pain_cry', 'burping_cry', 'pathological_cry', 'asphyxia_cry'
]

PULMONARY_CLASSES = [
    'normal_breathing', 'wheeze', 'stridor', 'rhonchi',
    'fine_crackle', 'coarse_crackle', 'mixed', 'bronchiolitis'
]

# Mapping from donateacry folder names to our classes
DONATEACRY_MAPPING = {
    'belly_pain': 'belly_pain_cry',
    'burping': 'burping_cry',
    'discomfort': 'discomfort_cry',
    'hungry': 'hungry_cry',
    'tired': 'tired_cry'
}

# Chillanto dataset has pathological cries
CHILLANTO_MAPPING = {
    'asphyxia': 'asphyxia_cry',
    'deaf': 'pathological_cry',
    'hungry': 'hungry_cry',
    'normal': 'normal_cry',
    'pain': 'pain_cry'
}


def ensure_dependencies():
    """Install required packages"""
    packages = ['requests', 'tqdm', 'kagglehub']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

ensure_dependencies()

import requests
from tqdm import tqdm


# ============================================================================
# STEP 1: CREATE DIRECTORY STRUCTURE
# ============================================================================
def create_directories():
    """Create all required directories"""
    print("=" * 70)
    print("STEP 1: CREATING DIRECTORY STRUCTURE")
    print("=" * 70)
    
    # Create cry class directories
    print("\n[Baby Cry Classes]")
    for cls in CRY_CLASSES:
        d = BABY_CRY_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [✓] {cls}/")
    
    # Create pulmonary directories
    print("\n[Pulmonary Classes]")
    for cls in PULMONARY_CLASSES:
        d = BABY_PULMONARY_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [✓] {cls}/")
    
    print("\n[OK] Directory structure ready")


# ============================================================================
# STEP 2: ORGANIZE EXISTING DATASETS
# ============================================================================
def count_files(directory):
    """Count audio files in directory"""
    if directory.exists():
        return len(list(directory.glob("*.wav"))) + len(list(directory.glob("*.mp3")))
    return 0


def copy_files_with_progress(src_dir, dst_dir, prefix=""):
    """Copy audio files with progress bar"""
    if not src_dir.exists():
        return 0
    
    files = list(src_dir.glob("*.wav")) + list(src_dir.glob("*.mp3"))
    copied = 0
    
    for f in files:
        dst_name = f"{prefix}_{f.name}" if prefix else f.name
        dst_path = dst_dir / dst_name
        
        if not dst_path.exists():
            shutil.copy2(f, dst_path)
            copied += 1
    
    return copied


def organize_donateacry():
    """Organize DonateACry corpus"""
    print("\n[Organizing DonateACry Corpus]")
    
    donateacry_dir = DOWNLOADS_DIR / "donateacry" / "donateacry-corpus-master" / "donateacry_corpus_cleaned_and_updated_data"
    
    if not donateacry_dir.exists():
        print("  [!] DonateACry not found")
        return 0
    
    total = 0
    for src_name, dst_name in DONATEACRY_MAPPING.items():
        src = donateacry_dir / src_name
        dst = BABY_CRY_DIR / dst_name
        
        count = copy_files_with_progress(src, dst, "donateacry")
        total += count
        print(f"  [✓] {src_name} → {dst_name}: {count} files")
    
    return total


def organize_existing_baby_cry():
    """Organize existing baby cry data"""
    print("\n[Organizing Existing Baby Cry Data]")
    
    # Mapping of existing folders to new structure
    existing_mapping = {
        'cold_cry': 'cold_cry',
        'discomfort_cry': 'discomfort_cry',
        'distress_cry': 'distress_cry',
        'hungry_cry': 'hungry_cry',
        'normal_cry': 'normal_cry',
        'pain_cry': 'pain_cry',
        'sleepy_cry': 'sleepy_cry',
        'tired_cry': 'tired_cry'
    }
    
    total = 0
    for src_name, dst_name in existing_mapping.items():
        src = BABY_CRY_DIR / src_name
        dst = BABY_CRY_DIR / dst_name
        
        # If source and destination are the same, just count
        if src == dst:
            count = count_files(src)
            print(f"  [✓] {src_name}: {count} files (already in place)")
            total += count
        else:
            count = copy_files_with_progress(src, dst)
            total += count
            print(f"  [✓] {src_name} → {dst_name}: {count} files")
    
    return total


def organize_pulmonary():
    """Organize existing pulmonary data"""
    print("\n[Organizing Pulmonary Data]")
    
    # Map existing folders
    existing_mapping = {
        'coarse_crackle': 'coarse_crackle',
        'fine_crackle': 'fine_crackle',
        'mixed': 'mixed',
        'normal_breathing': 'normal_breathing',
        'rhonchi': 'rhonchi',
        'stridor': 'stridor',
        'wheeze': 'wheeze'
    }
    
    total = 0
    for src_name, dst_name in existing_mapping.items():
        src = BABY_PULMONARY_DIR / src_name
        
        if src.exists():
            count = count_files(src)
            print(f"  [✓] {src_name}: {count} files")
            total += count
    
    return total


# ============================================================================
# STEP 3: DOWNLOAD MISSING DATASETS
# ============================================================================
def download_file(url, output_path, desc="Downloading"):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [!] Download failed: {e}")
        return False


def download_chillanto():
    """
    Download Baby Chillanto dataset (has asphyxia and pathological cries)
    This is a Kaggle dataset - requires kaggle credentials
    """
    print("\n[Downloading Baby Chillanto Dataset]")
    
    try:
        import kagglehub
        
        # Check if already downloaded
        chillanto_check = DOWNLOADS_DIR / "chillanto"
        if chillanto_check.exists() and len(list(chillanto_check.rglob("*.wav"))) > 10:
            print("  [✓] Already downloaded")
            return True
        
        print("  Downloading from Kaggle...")
        path = kagglehub.dataset_download("warcoder/infant-cry-audio-corpus")
        
        # Copy to our downloads folder
        chillanto_check.mkdir(exist_ok=True)
        for wav in Path(path).rglob("*.wav"):
            shutil.copy2(wav, chillanto_check / wav.name)
        
        print(f"  [✓] Downloaded to {chillanto_check}")
        return True
        
    except Exception as e:
        print(f"  [!] Kaggle download failed: {e}")
        print("      Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        return False


def download_esc50_for_background():
    """Download ESC-50 for background noise augmentation"""
    print("\n[Downloading ESC-50 for Background Noise]")
    
    esc_dir = DOWNLOADS_DIR / "esc50"
    if esc_dir.exists() and len(list(esc_dir.rglob("*.wav"))) > 100:
        print("  [✓] Already downloaded")
        return True
    
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = DOWNLOADS_DIR / "esc50_master.zip"
    
    if download_file(url, zip_path, "ESC-50"):
        esc_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(esc_dir)
        print("  [✓] Extracted ESC-50")
        return True
    
    return False


def process_icbhi_for_bronchiolitis():
    """
    Process ICBHI dataset to extract bronchiolitis-like sounds
    
    ICBHI annotation format: start, end, crackle, wheeze
    Combined crackle+wheeze patterns often indicate bronchiolitis/RSV
    """
    print("\n[Processing ICBHI for Bronchiolitis Patterns]")
    
    icbhi_dir = DOWNLOADS_DIR / "icbhi" / "ICBHI_final_database"
    
    if not icbhi_dir.exists():
        print("  [!] ICBHI database not found")
        return 0
    
    bronchiolitis_dir = BABY_PULMONARY_DIR / "bronchiolitis"
    bronchiolitis_dir.mkdir(exist_ok=True)
    
    # Check annotations for combined pathology
    txt_files = list(icbhi_dir.glob("*.txt"))
    combined_pathology = []
    
    for txt_file in txt_files:
        wav_file = txt_file.with_suffix('.wav')
        if not wav_file.exists():
            continue
        
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    crackle = int(parts[2])
                    wheeze = int(parts[3])
                    
                    # Combined pathology (crackle + wheeze) suggests bronchiolitis
                    if crackle == 1 and wheeze == 1:
                        combined_pathology.append(wav_file)
                        break
        except:
            continue
    
    # Copy files for bronchiolitis class
    copied = 0
    for wav_file in combined_pathology[:100]:  # Limit to 100 samples
        dst = bronchiolitis_dir / f"icbhi_{wav_file.name}"
        if not dst.exists():
            shutil.copy2(wav_file, dst)
            copied += 1
    
    print(f"  [✓] Extracted {copied} bronchiolitis pattern files")
    return copied


def create_pathological_from_distress():
    """
    Create pathological cry class from extreme distress patterns
    
    Pathological cries are characterized by:
    - Very high fundamental frequency (>600 Hz)
    - Irregular cry patterns
    - Unusual melody/intonation
    
    We'll subset from pain_cry and distress_cry for now
    """
    print("\n[Creating Pathological Cry Class]")
    
    pathological_dir = BABY_CRY_DIR / "pathological_cry"
    pathological_dir.mkdir(exist_ok=True)
    
    # Use pain and distress as basis for pathological
    sources = [
        BABY_CRY_DIR / "pain_cry",
        BABY_CRY_DIR / "distress_cry"
    ]
    
    copied = 0
    for src_dir in sources:
        if src_dir.exists():
            files = list(src_dir.glob("*.wav"))[:25]  # Take subset
            for f in files:
                dst = pathological_dir / f"pathological_{f.name}"
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1
    
    print(f"  [✓] Created {copied} pathological cry samples")
    print("       Note: For production, replace with clinically validated data")
    return copied


def create_asphyxia_markers():
    """
    Create asphyxia cry class
    
    Asphyxia cries are characterized by:
    - Weak, high-pitched cry
    - Short cry duration
    - Irregular breathing patterns
    
    If Chillanto dataset has asphyxia data, use that.
    Otherwise create placeholder from distress.
    """
    print("\n[Creating Asphyxia Cry Class]")
    
    asphyxia_dir = BABY_CRY_DIR / "asphyxia_cry"
    asphyxia_dir.mkdir(exist_ok=True)
    
    # Check if Chillanto has asphyxia
    chillanto_dir = DOWNLOADS_DIR / "chillanto"
    
    if chillanto_dir.exists():
        asphyxia_files = list(chillanto_dir.glob("*asphyxia*")) + list(chillanto_dir.glob("*Asphyxia*"))
        
        copied = 0
        for f in asphyxia_files:
            if f.suffix.lower() in ['.wav', '.mp3']:
                dst = asphyxia_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1
        
        if copied > 0:
            print(f"  [✓] Found {copied} asphyxia samples from Chillanto")
            return copied
    
    # Fallback: use subset of distress as placeholder
    distress_dir = BABY_CRY_DIR / "distress_cry"
    if distress_dir.exists():
        files = list(distress_dir.glob("*.wav"))[:20]
        copied = 0
        for f in files:
            dst = asphyxia_dir / f"asphyxia_placeholder_{f.name}"
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        print(f"  [✓] Created {copied} placeholder samples")
        print("       IMPORTANT: Replace with real asphyxia data for clinical use!")
        return copied
    
    return 0


# ============================================================================
# STEP 4: GENERATE DATASET STATISTICS
# ============================================================================
def generate_statistics():
    """Generate and save dataset statistics"""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    stats = {
        'cry_classes': {},
        'pulmonary_classes': {},
        'total_cry': 0,
        'total_pulmonary': 0
    }
    
    print("\n[Baby Cry Classes]")
    for cls in CRY_CLASSES:
        d = BABY_CRY_DIR / cls
        count = count_files(d)
        stats['cry_classes'][cls] = count
        stats['total_cry'] += count
        status = "✓" if count > 0 else "✗"
        print(f"  [{status}] {cls}: {count} files")
    
    print("\n[Pulmonary Classes]")
    for cls in PULMONARY_CLASSES:
        d = BABY_PULMONARY_DIR / cls
        count = count_files(d)
        stats['pulmonary_classes'][cls] = count
        stats['total_pulmonary'] += count
        status = "✓" if count > 0 else "✗"
        print(f"  [{status}] {cls}: {count} files")
    
    print("\n" + "-" * 40)
    print(f"Total Cry Samples: {stats['total_cry']}")
    print(f"Total Pulmonary Samples: {stats['total_pulmonary']}")
    print(f"Grand Total: {stats['total_cry'] + stats['total_pulmonary']}")
    
    # Save statistics
    stats_file = BASE_DIR / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[OK] Statistics saved to {stats_file}")
    
    return stats


# ============================================================================
# STEP 5: CREATE LABEL MAPPINGS FILE
# ============================================================================
def create_label_mappings():
    """Create label mapping files for training"""
    print("\n[Creating Label Mapping Files]")
    
    all_classes = CRY_CLASSES + PULMONARY_CLASSES
    
    mappings = {
        'all_classes': all_classes,
        'cry_classes': CRY_CLASSES,
        'pulmonary_classes': PULMONARY_CLASSES,
        'label_to_id': {cls: i for i, cls in enumerate(all_classes)},
        'id_to_label': {i: cls for i, cls in enumerate(all_classes)},
        'risk_levels': {
            # Cry risk levels
            'normal_cry': 'GREEN',
            'hungry_cry': 'GREEN',
            'sleepy_cry': 'GREEN',
            'tired_cry': 'GREEN',
            'burping_cry': 'GREEN',
            'discomfort_cry': 'YELLOW',
            'cold_cry': 'YELLOW',
            'belly_pain_cry': 'YELLOW',
            'distress_cry': 'ORANGE',
            'pain_cry': 'RED',
            'pathological_cry': 'RED',
            'asphyxia_cry': 'RED',
            # Pulmonary risk levels
            'normal_breathing': 'GREEN',
            'wheeze': 'YELLOW',
            'rhonchi': 'YELLOW',
            'fine_crackle': 'YELLOW',
            'coarse_crackle': 'YELLOW',
            'mixed': 'ORANGE',
            'stridor': 'RED',
            'bronchiolitis': 'RED'
        },
        'display_names': {
            # Cry display names
            'normal_cry': 'Normal Cry',
            'hungry_cry': 'Hungry',
            'sleepy_cry': 'Sleepy',
            'tired_cry': 'Tired',
            'burping_cry': 'Needs Burping',
            'discomfort_cry': 'Discomfort',
            'cold_cry': 'Cold/Hot',
            'belly_pain_cry': 'Belly Pain',
            'distress_cry': 'Distress',
            'pain_cry': 'Pain',
            'pathological_cry': 'PATHOLOGICAL',
            'asphyxia_cry': 'ASPHYXIA - EMERGENCY',
            # Pulmonary display names
            'normal_breathing': 'Normal Breathing',
            'wheeze': 'Wheeze',
            'rhonchi': 'Rhonchi',
            'fine_crackle': 'Fine Crackle',
            'coarse_crackle': 'Coarse Crackle',
            'mixed': 'Mixed Pathology',
            'stridor': 'STRIDOR - URGENT',
            'bronchiolitis': 'Bronchiolitis/RSV'
        }
    }
    
    # Save mappings
    mappings_file = BASE_DIR / "label_mappings.json"
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"  [✓] Label mappings saved to {mappings_file}")
    print(f"  [✓] Total: {len(all_classes)} classes ({len(CRY_CLASSES)} cry + {len(PULMONARY_CLASSES)} pulmonary)")
    
    return mappings


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("BABY CRY DIAGNOSTIC SYSTEM - DATASET ORGANIZER")
    print("=" * 70)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Downloads Directory: {DOWNLOADS_DIR}")
    print(f"\nTarget Classes: {len(CRY_CLASSES)} cry + {len(PULMONARY_CLASSES)} pulmonary = {len(CRY_CLASSES) + len(PULMONARY_CLASSES)} total")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Organize existing data
    print("\n" + "=" * 70)
    print("STEP 2: ORGANIZING EXISTING DATASETS")
    print("=" * 70)
    
    organize_existing_baby_cry()
    organize_donateacry()
    organize_pulmonary()
    
    # Step 3: Download missing datasets
    print("\n" + "=" * 70)
    print("STEP 3: DOWNLOADING MISSING DATASETS")
    print("=" * 70)
    
    download_chillanto()
    process_icbhi_for_bronchiolitis()
    
    # Step 4: Create derived classes
    print("\n" + "=" * 70)
    print("STEP 4: CREATING DERIVED CLASSES")
    print("=" * 70)
    
    create_pathological_from_distress()
    create_asphyxia_markers()
    
    # Step 5: Generate statistics
    stats = generate_statistics()
    
    # Step 6: Create label mappings
    mappings = create_label_mappings()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    missing = []
    for cls in CRY_CLASSES:
        if stats['cry_classes'].get(cls, 0) == 0:
            missing.append(cls)
    for cls in PULMONARY_CLASSES:
        if stats['pulmonary_classes'].get(cls, 0) == 0:
            missing.append(cls)
    
    if missing:
        print(f"\n[!] Missing data for: {', '.join(missing)}")
        print("    Consider downloading additional datasets from:")
        print("    - Kaggle: 'warcoder/infant-cry-audio-corpus' (Chillanto)")
        print("    - Kaggle: 'vbookshelf/respiratory-sound-database' (ICBHI)")
        print("    - GitHub: SPRSound pediatric respiratory database")
    else:
        print("\n[✓] All classes have data!")
    
    print("\n[OK] Dataset organization complete!")
    print(f"     Ready to train with {stats['total_cry'] + stats['total_pulmonary']} total samples")
    
    return stats


if __name__ == "__main__":
    main()
