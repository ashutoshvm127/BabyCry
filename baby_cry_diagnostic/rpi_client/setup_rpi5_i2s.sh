#!/bin/bash
# ==============================================================================
# Raspberry Pi 5 - INMP441 I2S Microphone Setup Script
# ==============================================================================
# Run this script once to configure I2S audio input on your RPi5
# ==============================================================================

echo "========================================================================"
echo "RASPBERRY PI 5 - INMP441 I2S MICROPHONE SETUP"
echo "========================================================================"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "[!] Please run as root: sudo ./setup_rpi5_i2s.sh"
    exit 1
fi

echo "[1] Updating system..."
apt-get update -qq

echo "[2] Installing audio packages..."
apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    python3-pyaudio \
    python3-alsaaudio \
    alsa-utils \
    pulseaudio

echo "[3] Configuring I2S overlay..."

# Backup config.txt
cp /boot/firmware/config.txt /boot/firmware/config.txt.backup 2>/dev/null || \
cp /boot/config.txt /boot/config.txt.backup 2>/dev/null

# Add I2S overlay (RPi5 uses different config location)
CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

# Check if already configured
if grep -q "dtoverlay=i2s-mmap" "$CONFIG_FILE"; then
    echo "    I2S overlay already configured"
else
    echo "    Adding I2S overlay to config..."
    cat >> "$CONFIG_FILE" << EOF

# I2S Audio for INMP441 Microphone
dtparam=i2s=on
dtoverlay=i2s-mmap
dtoverlay=googlevoicehat-soundcard
EOF
fi

echo "[4] Creating ALSA configuration..."
cat > /etc/asound.conf << EOF
# ALSA configuration for INMP441 I2S Microphone

pcm.!default {
    type asym
    playback.pcm "plughw:0,0"
    capture.pcm "i2s_capture"
}

pcm.i2s_capture {
    type plug
    slave {
        pcm "hw:1,0"
        rate 44100
        format S24_LE
        channels 1
    }
}

ctl.!default {
    type hw
    card 0
}
EOF

echo "[5] Setting up Python environment..."
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install pyaudio websockets numpy sounddevice -q

echo "[6] Creating systemd service..."
cat > /etc/systemd/system/baby-cry-client.service << EOF
[Unit]
Description=Baby Cry Diagnostic - Audio Capture Client
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/.venv/bin:/usr/bin
ExecStart=$(pwd)/.venv/bin/python capture_client.py --server ws://YOUR_SERVER_IP:8000/ws/stream
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "[7] Testing audio..."
echo "    Running arecord test (5 seconds)..."
arecord -D plughw:1,0 -f S24_LE -r 44100 -c 1 -d 5 /tmp/test_audio.wav 2>/dev/null && {
    echo "    [OK] Audio capture working"
    rm /tmp/test_audio.wav
} || {
    echo "    [!] Audio capture test failed - check wiring"
}

echo
echo "========================================================================"
echo "SETUP COMPLETE"
echo "========================================================================"
echo
echo "Next steps:"
echo "1. Reboot: sudo reboot"
echo "2. Edit server URL in: /etc/systemd/system/baby-cry-client.service"
echo "3. Enable service: sudo systemctl enable baby-cry-client"
echo "4. Start service: sudo systemctl start baby-cry-client"
echo
echo "Manual run: python capture_client.py --server ws://SERVER_IP:8000/ws/stream"
echo
echo "Wiring reminder:"
echo "  INMP441 VDD  -> 3.3V (Pin 1)"
echo "  INMP441 GND  -> GND (Pin 6)"
echo "  INMP441 WS   -> GPIO 19 (Pin 35)"
echo "  INMP441 SCK  -> GPIO 18 (Pin 12)"
echo "  INMP441 SD   -> GPIO 20 (Pin 38)"
echo "  INMP441 L/R  -> GND (Left channel)"
echo
