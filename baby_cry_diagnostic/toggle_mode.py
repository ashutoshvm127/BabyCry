#!/usr/bin/env python3
"""
Baby Cry Diagnostic System - Quick Mode Toggle
Run this script to easily switch between RPi5 and Desktop modes.
"""

import sys
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "system_config.json"


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"is_rpi5_mode": False}


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def show_status():
    config = load_config()
    is_rpi5 = config.get("is_rpi5_mode", False)
    
    print()
    print("=" * 50)
    print("BABY CRY DIAGNOSTIC - SYSTEM MODE")
    print("=" * 50)
    print()
    
    if is_rpi5:
        print("  Current Mode: [*] RPi5 (INMP441 I2S)")
        print("                [ ] Desktop (Standard Mic)")
    else:
        print("  Current Mode: [ ] RPi5 (INMP441 I2S)")
        print("                [*] Desktop (Standard Mic)")
    
    print()
    print("  Config file:", CONFIG_FILE)
    print()
    print("-" * 50)
    print("  To change mode:")
    print("    python toggle_mode.py rpi5    - Raspberry Pi 5")
    print("    python toggle_mode.py desktop - Windows/Desktop")
    print("-" * 50)
    print()


def set_mode(mode: str):
    config = load_config()
    
    if mode == "rpi5":
        config["is_rpi5_mode"] = True
        print("[OK] Switched to RPi5 mode (INMP441 I2S, 24-bit, 44100 Hz)")
    elif mode == "desktop":
        config["is_rpi5_mode"] = False
        print("[OK] Switched to Desktop mode (Standard mic, 16-bit, 16000 Hz)")
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        print("        Valid modes: rpi5, desktop")
        return
    
    save_config(config)
    print(f"    Config saved to: {CONFIG_FILE}")


def main():
    if len(sys.argv) < 2:
        show_status()
        return
    
    mode = sys.argv[1].lower()
    
    if mode in ["rpi5", "rpi", "raspberry", "pi"]:
        set_mode("rpi5")
    elif mode in ["desktop", "windows", "win", "pc"]:
        set_mode("desktop")
    elif mode in ["show", "status", "?"]:
        show_status()
    else:
        print(f"Unknown command: {mode}")
        print("Usage: python toggle_mode.py [rpi5|desktop|show]")


if __name__ == "__main__":
    main()
