#!/bin/bash
# ==============================================================================
# Install systemd service only (manual step)
# ==============================================================================
# Use this if you already installed dependencies manually
#
# Run with: sudo ./install_service.sh
# ==============================================================================

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./install_service.sh"
    exit 1
fi

ACTUAL_USER=${SUDO_USER:-$USER}
INSTALL_DIR=$(pwd)

echo "Creating systemd service for Baby Cry Monitor..."

cat > /etc/systemd/system/babycry.service << EOF
[Unit]
Description=Baby Cry Monitor - RPi5 Standalone
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

# Auto-restart on failure
WatchdogSec=300

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo ""
echo "Service installed!"
echo ""
echo "Commands:"
echo "  sudo systemctl start babycry    # Start now"
echo "  sudo systemctl enable babycry   # Enable on boot"
echo "  sudo systemctl status babycry   # Check status"
echo "  sudo journalctl -u babycry -f   # View logs"
echo ""
