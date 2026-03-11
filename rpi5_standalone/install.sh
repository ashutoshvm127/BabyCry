#!/bin/bash
# ==============================================================================
# Baby Cry Monitor - RPi5 Complete Installer
# ==============================================================================
# This script sets up everything needed to run the baby cry monitor on RPi5
#
# Run with: sudo ./install.sh
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "=============================================="
echo "  Baby Cry Monitor - RPi5 Installer"
echo "=============================================="
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root: sudo ./install.sh${NC}"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER=${SUDO_USER:-$USER}
INSTALL_DIR=$(pwd)

echo -e "${YELLOW}[1/7] Updating system packages...${NC}"
apt update
apt upgrade -y

echo -e "${YELLOW}[2/7] Installing system dependencies...${NC}"
apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libportaudio2 \
    portaudio19-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libasound2-dev \
    git \
    i2c-tools

echo -e "${YELLOW}[3/7] Enabling I2S audio interface...${NC}"

# Backup config
cp /boot/firmware/config.txt /boot/firmware/config.txt.backup 2>/dev/null || \
cp /boot/config.txt /boot/config.txt.backup 2>/dev/null || true

# Add I2S configuration
CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

# Check if already configured
if ! grep -q "dtparam=i2s=on" "$CONFIG_FILE"; then
    echo "" >> "$CONFIG_FILE"
    echo "# Baby Cry Monitor - I2S Audio" >> "$CONFIG_FILE"
    echo "dtparam=i2s=on" >> "$CONFIG_FILE"
    echo "dtoverlay=i2s-mmap" >> "$CONFIG_FILE"
    echo -e "${GREEN}  I2S enabled in $CONFIG_FILE${NC}"
else
    echo -e "${GREEN}  I2S already enabled${NC}"
fi

# Enable I2C for OLED (optional)
if ! grep -q "dtparam=i2c_arm=on" "$CONFIG_FILE"; then
    echo "dtparam=i2c_arm=on" >> "$CONFIG_FILE"
fi

echo -e "${YELLOW}[4/7] Creating Python virtual environment...${NC}"
python3 -m venv venv
chown -R $ACTUAL_USER:$ACTUAL_USER venv

echo -e "${YELLOW}[5/7] Installing Python packages...${NC}"
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch for ARM64
echo "  Installing PyTorch (this may take a while)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt

# Make sure user owns everything
chown -R $ACTUAL_USER:$ACTUAL_USER .

echo -e "${YELLOW}[6/7] Creating systemd service...${NC}"

# Create service file
cat > /etc/systemd/system/babycry.service << EOF
[Unit]
Description=Baby Cry Monitor - RPi5
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$INSTALL_DIR/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Watchdog - restart if not responding
WatchdogSec=300

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

echo -e "${YELLOW}[7/7] Setting up log rotation...${NC}"

cat > /etc/logrotate.d/babycry << EOF
/var/log/babycry.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 $ACTUAL_USER $ACTUAL_USER
}
EOF

# Create log file
touch /var/log/babycry.log
chown $ACTUAL_USER:$ACTUAL_USER /var/log/babycry.log

echo ""
echo -e "${GREEN}=============================================="
echo "  Installation Complete!"
echo "==============================================${NC}"
echo ""
echo -e "To start the service:"
echo -e "  ${YELLOW}sudo systemctl start babycry${NC}"
echo ""
echo -e "To enable auto-start on boot:"
echo -e "  ${YELLOW}sudo systemctl enable babycry${NC}"
echo ""
echo -e "To view logs:"
echo -e "  ${YELLOW}sudo journalctl -u babycry -f${NC}"
echo ""
echo -e "Web dashboard will be at:"
echo -e "  ${YELLOW}http://$(hostname -I | awk '{print $1}'):8080${NC}"
echo ""
echo -e "${RED}IMPORTANT: Reboot required for I2S audio!${NC}"
echo -e "  ${YELLOW}sudo reboot${NC}"
echo ""
