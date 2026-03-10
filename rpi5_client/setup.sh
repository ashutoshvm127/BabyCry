#!/bin/bash
# ==============================================================================
# Raspberry Pi 5 - Baby Cry Monitor Setup Script
# ==============================================================================
# Run: chmod +x setup.sh && sudo ./setup.sh
# ==============================================================================

set -e

echo "========================================================================"
echo "  RASPBERRY PI 5 - BABY CRY MONITOR SETUP"
echo "========================================================================"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] Please run as root: sudo ./setup.sh"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get the user who ran sudo
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

echo "[1/8] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

echo "[2/8] Installing system dependencies..."
apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-rpi.gpio \
    python3-smbus \
    python3-alsaaudio \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1 \
    libsndfile1-dev \
    alsa-utils \
    git \
    i2c-tools \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev

echo "[3/8] Enabling I2C and I2S interfaces..."

# Enable I2C
if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null && \
   ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null; then
    echo "dtparam=i2c_arm=on" >> /boot/firmware/config.txt 2>/dev/null || \
    echo "dtparam=i2c_arm=on" >> /boot/config.txt
fi

# Enable I2S
CONFIG_FILE="/boot/firmware/config.txt"
[ ! -f "$CONFIG_FILE" ] && CONFIG_FILE="/boot/config.txt"

if ! grep -q "dtoverlay=i2s-mmap" "$CONFIG_FILE"; then
    cat >> "$CONFIG_FILE" << 'EOF'

# I2S Audio for INMP441 Microphone
dtparam=i2s=on
dtoverlay=i2s-mmap
dtoverlay=googlevoicehat-soundcard
EOF
    echo "    I2S overlay configured"
fi

# Enable SPI (for some displays)
if ! grep -q "^dtparam=spi=on" "$CONFIG_FILE"; then
    echo "dtparam=spi=on" >> "$CONFIG_FILE"
fi

echo "[4/8] Configuring ALSA for INMP441..."
cat > /etc/asound.conf << 'EOF'
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

echo "[5/8] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    chown -R "$REAL_USER:$REAL_USER" .venv
fi

# Activate venv and install packages
source .venv/bin/activate

echo "[6/8] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -q \
    pyaudio \
    sounddevice \
    websockets \
    numpy \
    scipy \
    pillow \
    luma.oled \
    luma.core \
    RPi.GPIO \
    smbus2 \
    aiofiles

echo "[7/8] Creating systemd service..."
cat > /etc/systemd/system/baby-cry-monitor.service << EOF
[Unit]
Description=Baby Cry Monitor - Audio Capture & Cloud Streaming
After=network-online.target sound.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$SCRIPT_DIR/.venv/bin/python main.py --config config.json
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/baby-cry-monitor.log
StandardError=append:/var/log/baby-cry-monitor.log

[Install]
WantedBy=multi-user.target
EOF

# Create log file
touch /var/log/baby-cry-monitor.log
chown "$REAL_USER:$REAL_USER" /var/log/baby-cry-monitor.log

# Reload systemd
systemctl daemon-reload

echo "[8/8] Setting permissions..."
chown -R "$REAL_USER:$REAL_USER" "$SCRIPT_DIR"
usermod -a -G audio,i2c,gpio,spi "$REAL_USER"

echo
echo "========================================================================"
echo "  SETUP COMPLETE!"
echo "========================================================================"
echo
echo "  Hardware Wiring (INMP441 → RPi5):"
echo "  ─────────────────────────────────"
echo "    VDD  → 3.3V (Pin 1)"
echo "    GND  → GND  (Pin 6)"
echo "    WS   → GPIO 19 (Pin 35) - I2S Frame Sync"
echo "    SCK  → GPIO 18 (Pin 12) - I2S Clock"
echo "    SD   → GPIO 20 (Pin 38) - I2S Data In"
echo "    L/R  → GND (Left channel)"
echo
echo "  Optional Hardware:"
echo "  ───────────────────"
echo "    OLED SSD1306 (I2C):"
echo "      VCC → 3.3V (Pin 1)"
echo "      GND → GND  (Pin 9)"
echo "      SDA → GPIO 2 (Pin 3)"
echo "      SCL → GPIO 3 (Pin 5)"
echo
echo "    Status LEDs:"
echo "      Green  → GPIO 22 (Pin 15)"
echo "      Yellow → GPIO 23 (Pin 16)"
echo "      Red    → GPIO 24 (Pin 18)"
echo
echo "    Buzzer:"
echo "      Signal → GPIO 17 (Pin 11)"
echo
echo "  Next Steps:"
echo "  ────────────"
echo "  1. Edit config.json with your cloud URL"
echo "  2. Reboot: sudo reboot"
echo "  3. Test audio: arecord -l"
echo "  4. Run manually: source .venv/bin/activate && python main.py"
echo "  5. Enable service: sudo systemctl enable baby-cry-monitor"
echo "  6. Start service: sudo systemctl start baby-cry-monitor"
echo "  7. View logs: tail -f /var/log/baby-cry-monitor.log"
echo
echo "========================================================================"
