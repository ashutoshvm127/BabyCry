@echo off
REM ==============================================================================
REM Baby Cry Diagnostic System - Mode Toggle Script
REM ==============================================================================
REM Usage:
REM   toggle_mode.bat          - Show current mode
REM   toggle_mode.bat rpi5     - Switch to RPi5 mode
REM   toggle_mode.bat desktop  - Switch to Desktop mode
REM ==============================================================================

cd /d "%~dp0"

if "%1"=="" (
    echo Current configuration:
    python config.py --mode show
    goto :end
)

if "%1"=="rpi5" (
    echo Switching to RPi5 mode...
    python config.py --mode rpi5
    goto :end
)

if "%1"=="desktop" (
    echo Switching to Desktop mode...
    python config.py --mode desktop
    goto :end
)

echo Usage:
echo   toggle_mode.bat          - Show current mode
echo   toggle_mode.bat rpi5     - Switch to RPi5 mode (INMP441 I2S)
echo   toggle_mode.bat desktop  - Switch to Desktop mode (Standard mic)

:end
