@echo off
REM ==============================================================================
REM Baby Cry Diagnostic System - Windows Startup Script
REM ==============================================================================

echo ======================================================================
echo BABY CRY DIAGNOSTIC SYSTEM
echo AI-Powered Infant Health Monitoring
echo ======================================================================
echo.

cd /d "%~dp0"

REM Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [1] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [1] Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

REM Install dependencies
echo [2] Checking dependencies...
pip install -r backend\requirements.txt -q

REM Start server
echo.
echo [3] Starting server...
echo     API Docs: http://localhost:8000/docs
echo     Dashboard: http://localhost:3000
echo.
echo Press Ctrl+C to stop
echo ----------------------------------------------------------------------
echo.

python start_server.py

pause
