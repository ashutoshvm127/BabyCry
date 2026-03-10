#!/usr/bin/env python3
"""
Baby Cry Diagnostic System - Startup Script
Launches the FastAPI backend server
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 70)
    print("BABY CRY DIAGNOSTIC SYSTEM")
    print("AI-Powered Infant Health Monitoring")
    print("=" * 70)
    print()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Check for .env file
    env_file = project_dir / ".env"
    env_template = project_dir / ".env.template"
    
    if not env_file.exists():
        print("[!] No .env file found.")
        if env_template.exists():
            print("    Copying from .env.template...")
            import shutil
            shutil.copy(env_template, env_file)
            print("    [OK] Created .env - please edit with your credentials")
        else:
            print("    Please create a .env file from .env.template")
    
    # Install dependencies
    print("\n[1] Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import torch
        print("    [OK] Core dependencies installed")
    except ImportError:
        print("    Installing dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            str(project_dir / "backend" / "requirements.txt")
        ])
    
    # Start server
    print("\n[2] Starting FastAPI server...")
    print("    API Docs: http://localhost:8000/docs")
    print("    Health:   http://localhost:8000/health")
    print()
    print("-" * 70)
    print("Press Ctrl+C to stop the server")
    print("-" * 70)
    print()
    
    # Set Python path
    os.environ["PYTHONPATH"] = str(project_dir)
    
    # Run uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


if __name__ == "__main__":
    main()
