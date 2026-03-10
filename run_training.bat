@echo off
REM ============================================================================
REM BABY LUNG DISEASE DETECTION - TRAINING LAUNCHER
REM ============================================================================
REM This script automatically sets up and runs the training pipeline

echo.
echo ============================================================================
echo BABY LUNG DISEASE DETECTION - AUTOMATIC TRAINING LAUNCHER
echo ============================================================================
echo.
echo This script will:
echo   1. Verify your system is ready
echo   2. Install any missing dependencies
echo   3. Download datasets from 40+ sources
echo   4. Download the Wav2Vec2 model from HuggingFace
echo   5. Prepare and organize training data
echo   6. Train the model with maximum accuracy settings
echo.
echo Expected duration: 1-4 hours (with GPU), 8-12 hours (CPU)
echo Disk space required: 20GB+
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check disk space (approximately)
for /f "tokens=3" %%A in ('dir /-C %TEMP%') do (
    set "free_space=%%A"
)
echo Current disk space available: %free_space% bytes

REM Run verification script first
echo.
echo [Verification Phase]
echo Checking environment...
python verify_setup.py
if errorlevel 1 (
    echo.
    echo ERROR: Environment verification failed!
    pause
    exit /b 1
)

REM Run the main training script
echo.
echo [Training Phase]
echo Starting training pipeline...
echo This window will remain open for the duration of training.
echo.

python train_maximum_accuracy.py

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    echo Check the error messages above for details.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo TRAINING COMPLETE!
echo ============================================================================
echo.
echo Your trained model is saved in: ./ast_baby_cry_optimized/
echo.
pause
