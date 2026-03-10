# ==============================================================================
# Baby Cry Diagnostic System - Deployment Preparation Script (Windows)
# ==============================================================================
# This script prepares the codebase for cloud deployment and RPi5 installation
# Run: .\deploy.ps1
# ==============================================================================

param(
    [switch]$CloudOnly,
    [switch]$RpiOnly,
    [switch]$SkipModels,
    [string]$OutputDir = ".\deployment_package"
)

$ErrorActionPreference = "Stop"

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  BABY CRY DIAGNOSTIC - DEPLOYMENT PREPARATION" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $ProjectRoot)) {
    $ProjectRoot = Get-Location
}

Write-Host "[INFO] Project Root: $ProjectRoot" -ForegroundColor Gray

# Create output directory
$OutputPath = Join-Path $ProjectRoot $OutputDir
if (Test-Path $OutputPath) {
    Write-Host "[WARN] Output directory exists, cleaning..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $OutputPath
}
New-Item -ItemType Directory -Path $OutputPath | Out-Null

# ==============================================================================
# Prepare Cloud Deployment Package
# ==============================================================================

if (-not $RpiOnly) {
    Write-Host ""
    Write-Host "[1/3] Preparing Cloud Deployment Package..." -ForegroundColor Green
    
    $CloudDir = Join-Path $OutputPath "cloud"
    New-Item -ItemType Directory -Path $CloudDir | Out-Null
    
    # Copy cloud deployment files
    Copy-Item -Path (Join-Path $ProjectRoot "cloud_deployment\*") -Destination $CloudDir -Recurse
    
    # Copy models and services from original backend
    Copy-Item -Path (Join-Path $ProjectRoot "baby_cry_diagnostic\backend\models\*") -Destination (Join-Path $CloudDir "models") -Recurse -Force
    Copy-Item -Path (Join-Path $ProjectRoot "baby_cry_diagnostic\backend\services\*") -Destination (Join-Path $CloudDir "services") -Recurse -Force
    
    if (-not $SkipModels) {
        # Copy trained model weights
        $TrainedClassifiers = Join-Path $ProjectRoot "trained_classifiers"
        if (Test-Path $TrainedClassifiers) {
            $ModelsTarget = Join-Path $CloudDir "models\trained_weights"
            New-Item -ItemType Directory -Path $ModelsTarget -Force | Out-Null
            Copy-Item -Path "$TrainedClassifiers\*" -Destination $ModelsTarget -Recurse
            Write-Host "  [OK] Trained classifiers copied" -ForegroundColor Gray
        }
        
        # Copy AST fine-tuned models
        $AstCry = Join-Path $ProjectRoot "ast_baby_cry_optimized"
        if (Test-Path $AstCry) {
            Copy-Item -Path $AstCry -Destination $CloudDir -Recurse
            Write-Host "  [OK] AST baby cry model copied" -ForegroundColor Gray
        }
        
        $AstResp = Join-Path $ProjectRoot "ast_respiratory_optimized"
        if (Test-Path $AstResp) {
            Copy-Item -Path $AstResp -Destination $CloudDir -Recurse
            Write-Host "  [OK] AST respiratory model copied" -ForegroundColor Gray
        }
    }
    
    Write-Host "  [OK] Cloud package ready: $CloudDir" -ForegroundColor Green
}

# ==============================================================================
# Prepare RPi5 Client Package
# ==============================================================================

if (-not $CloudOnly) {
    Write-Host ""
    Write-Host "[2/3] Preparing RPi5 Client Package..." -ForegroundColor Green
    
    $RpiDir = Join-Path $OutputPath "rpi5_client"
    New-Item -ItemType Directory -Path $RpiDir | Out-Null
    
    # Copy RPi5 client files
    Copy-Item -Path (Join-Path $ProjectRoot "rpi5_client\*") -Destination $RpiDir -Recurse
    
    Write-Host "  [OK] RPi5 package ready: $RpiDir" -ForegroundColor Green
}

# ==============================================================================
# Create Deployment Instructions
# ==============================================================================

Write-Host ""
Write-Host "[3/3] Creating deployment instructions..." -ForegroundColor Green

$Instructions = @"
# Baby Cry Diagnostic System - Deployment Package

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Contents

- cloud/           - Cloud API server for Render/Railway/Fly.io
- rpi5_client/     - Raspberry Pi 5 audio capture client

## Deployment Steps

### 1. Deploy Cloud Backend (Render)

1. Push 'cloud/' folder to a Git repository
2. Create new Web Service on Render
3. Connect repository and select Docker runtime
4. Set environment variables:
   - HUGGINGFACE_TOKEN=your_token
   - PORT=10000
5. Deploy!

Alternatively, use render.yaml for Blueprint deployment.

### 2. Setup Raspberry Pi 5

1. Copy 'rpi5_client/' to your Raspberry Pi
2. Run: chmod +x setup.sh && sudo ./setup.sh
3. Edit config.json with your cloud URL
4. Reboot: sudo reboot
5. Start: python main.py

### 3. Enable Auto-Start on RPi5

```bash
sudo systemctl enable baby-cry-monitor
sudo systemctl start baby-cry-monitor
```

## Files Included

### Cloud Package
- Dockerfile          # Docker build configuration
- render.yaml         # Render deployment config
- main.py             # FastAPI server
- requirements.txt    # Python dependencies
- models/            # AI ensemble models
- services/          # Audio processing, PDF generation

### RPi5 Package
- main.py            # Audio capture client
- config.json        # Configuration file
- setup.sh           # Automated setup script
- requirements.txt   # Python dependencies
"@

$Instructions | Out-File -FilePath (Join-Path $OutputPath "DEPLOYMENT_GUIDE.md") -Encoding UTF8

# ==============================================================================
# Summary
# ==============================================================================

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  DEPLOYMENT PACKAGE READY" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "  Output: $OutputPath" -ForegroundColor White
Write-Host ""
Write-Host "  Next Steps:" -ForegroundColor White
Write-Host "  1. Push cloud/ to GitHub" -ForegroundColor Gray
Write-Host "  2. Deploy to Render/Railway" -ForegroundColor Gray
Write-Host "  3. Copy rpi5_client/ to Raspberry Pi" -ForegroundColor Gray
Write-Host "  4. Run setup.sh on Pi" -ForegroundColor Gray
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
