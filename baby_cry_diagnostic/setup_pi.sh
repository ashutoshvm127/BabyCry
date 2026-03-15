#!/bin/bash
# ==============================================================================
# Baby Cry Diagnostic - Raspberry Pi 5 Full Setup Script
# Tested on: Raspberry Pi 5, Debian Trixie, Python 3.13
#
# Usage:
#   cd ~/Desktop/baby_cry_diagnostic
#   chmod +x setup_pi.sh
#   bash setup_pi.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "============================================================"
echo "  Baby Cry Diagnostic - Raspberry Pi 5 Setup"
echo "  Working directory: $SCRIPT_DIR"
echo "============================================================"

# ── Step 1: Fix apt sources and update ───────────────────────────────────────
echo ""
echo "[1/8] Updating apt and installing system packages..."

sudo tee /etc/apt/sources.list.d/debian.sources > /dev/null <<'EOF'
Types: deb
URIs: http://deb.debian.org/debian
Suites: trixie trixie-updates
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: http://deb.debian.org/debian-security
Suites: trixie-security
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF

echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.com/debian/ trixie main" \
    | sudo tee /etc/apt/sources.list.d/raspi.list > /dev/null

sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get update -qq

# Pre-built scientific libs (avoids gfortran/Fortran build on Python 3.13)
sudo apt-get install -y \
    python3-full python3-venv python3-dev python3-pip \
    python3-numpy python3-scipy \
    libportaudio2 portaudio19-dev \
    libsndfile1 libsndfile1-dev \
    ffmpeg \
    build-essential pkg-config \
    ca-certificates curl gnupg

echo "  [OK] System packages installed"

# ── Step 2: Install Node.js + nginx ──────────────────────────────────────────
echo ""
echo "[2/8] Installing Node.js and nginx..."

if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

sudo apt-get install -y nginx
echo "  [OK] Node $(node -v), npm $(npm -v), nginx $(nginx -v 2>&1 | head -1)"

# ── Step 3: Enable I2S for INMP441 microphone ─────────────────────────────────
echo ""
echo "[3/8] Configuring I2S for INMP441 microphone..."

BOOT_CONFIG="/boot/firmware/config.txt"
if ! grep -q "dtparam=i2s=on" "$BOOT_CONFIG" 2>/dev/null; then
    echo "" | sudo tee -a "$BOOT_CONFIG" > /dev/null
    echo "# INMP441 I2S Microphone" | sudo tee -a "$BOOT_CONFIG" > /dev/null
    echo "dtparam=i2s=on" | sudo tee -a "$BOOT_CONFIG" > /dev/null
    echo "dtoverlay=googlevoicehat-soundcard" | sudo tee -a "$BOOT_CONFIG" > /dev/null
    echo "  [OK] I2S added to $BOOT_CONFIG (reboot required)"
else
    echo "  [OK] I2S already configured"
fi

# ── Step 4: Set is_rpi5_mode = true in system_config.json ────────────────────
echo ""
echo "[4/8] Enabling RPi5 mode in system_config.json..."

CONFIG_FILE="$SCRIPT_DIR/system_config.json"
if [ -f "$CONFIG_FILE" ]; then
    python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
cfg['is_rpi5_mode'] = True
with open('$CONFIG_FILE', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  [OK] is_rpi5_mode = true')
"
else
    echo "  [!] system_config.json not found at $CONFIG_FILE"
fi

# ── Step 5: Create backend Python venv ───────────────────────────────────────
echo ""
echo "[5/8] Setting up Python backend venv..."

cd "$BACKEND_DIR"
rm -rf venv

# --system-site-packages lets numpy/scipy from apt be visible inside the venv
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Confirm numpy/scipy already available via system packages
python -c "import numpy, scipy; print('  [OK] numpy', numpy.__version__, '/ scipy', scipy.__version__)"

python -m pip install --upgrade pip setuptools wheel --quiet

# ── Step 6: Install Python backend dependencies ───────────────────────────────
echo ""
echo "[6/8] Installing backend Python packages (this may take 10-20 min)..."

# Try piwheels first (fast Pi-specific wheels), fallback to PyPI
pip install \
    --extra-index-url https://www.piwheels.org/simple \
    --no-cache-dir \
    --retries 10 \
    --timeout 120 \
    -r requirements.pi.txt

echo "  [OK] Backend packages installed"
deactivate

# ── Step 7: Build React frontend ──────────────────────────────────────────────
echo ""
echo "[7/8] Building React frontend..."

PI_IP=$(hostname -I | awk '{print $1}')
cd "$FRONTEND_DIR"
npm install --silent
REACT_APP_API_URL="http://${PI_IP}:8000" npm run build
echo "  [OK] Frontend built (API URL: http://${PI_IP}:8000)"

# ── Step 8: Configure nginx + systemd ────────────────────────────────────────
echo ""
echo "[8/8] Configuring nginx and systemd..."

USERNAME="$(id -un)"

# nginx config
sudo tee /etc/nginx/sites-available/babycry > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    root $FRONTEND_DIR/build;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 120s;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/babycry /etc/nginx/sites-enabled/babycry
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t

# systemd service
sudo tee /etc/systemd/system/babycry-backend.service > /dev/null <<EOF
[Unit]
Description=Baby Cry Diagnostic Backend
After=network.target sound.target

[Service]
User=$USERNAME
WorkingDirectory=$BACKEND_DIR
ExecStart=$BACKEND_DIR/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable babycry-backend nginx
sudo systemctl restart nginx
sudo systemctl start babycry-backend

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Website:  http://${PI_IP}"
echo "  API docs: http://${PI_IP}:8000/docs"
echo ""
echo "  Check backend: sudo systemctl status babycry-backend"
echo "  View logs:     sudo journalctl -u babycry-backend -f"
echo ""
echo "  NOTE: Reboot once for I2S microphone to activate:"
echo "        sudo reboot"
echo "============================================================"
