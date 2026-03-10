also for this there is weight difference that has been an issue for the trining is there any way to remove the # 🍼 Baby Cry Diagnostic System - Complete Deployment Guide

## 📋 Overview

This guide walks you through deploying the Baby Cry Diagnostic System with:
1. **Cloud Backend** - AI inference server hosted on Render
2. **Raspberry Pi 5 Client** - Local audio capture and streaming

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────┘

     RASPBERRY PI 5                      RENDER CLOUD
    ┌─────────────────┐                ┌────────────────────────┐
    │   INMP441 Mic   │    WebSocket   │    FastAPI Backend     │
    │       ↓         │═══════════════▶│                        │
    │  Audio Capture  │      WSS       │   6-Backbone Ensemble  │
    │       ↓         │◀═══════════════│   • DistilHuBERT       │
    │   OLED Display  │    Results     │   • AST                │
    │   LED Status    │                │   • YAMNet             │
    │   Speaker Alert │                │   • Wav2Vec2           │
    └─────────────────┘                │   • WavLM              │
                                       │   • PANNs CNN14        │
                                       │                        │
                                       │   Trained Classifiers  │
                                       │   • Cry (12 classes)   │
                                       │   • Pulmonary (8 cls)  │
                                       └────────────────────────┘
```

---

## 🚀 Part 1: Deploy Cloud Backend to Render

### Prerequisites
- GitHub account
- Render account (free tier available)
- HuggingFace account (for model downloads)

### Step 1: Prepare Repository

```powershell
# Clone or navigate to your project
cd "d:\projects\cry analysuis - Copy"

# Initialize git (if not already)
git init

# Create .gitignore if needed
@"
__pycache__/
*.pyc
.env
.venv/
*.log
model_cache/
reports/
"@ | Out-File -FilePath .gitignore -Encoding UTF8

# Add and commit
git add .
git commit -m "Initial deployment"
```

### Step 2: Push to GitHub

```powershell
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/baby-cry-diagnostic.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to [render.com](https://render.com) and sign up/login
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `baby-cry-api`
   - **Root Directory**: `cloud_deployment`
   - **Environment**: `Docker`
   - **Plan**: Starter ($7/month) or Free (limited)

5. Add Environment Variables:
   | Key | Value |
   |-----|-------|
   | `HUGGINGFACE_TOKEN` | `hf_xxxxxxxxxxxxx` |
   | `LOG_LEVEL` | `INFO` |
   | `PORT` | `10000` |

6. Click **Deploy**

### Step 4: Verify Deployment

```bash
# Check health endpoint
curl https://baby-cry-api.onrender.com/health

# Should return:
# {"status":"healthy","models":{"ensemble":true,...}}
```

### Alternative: Deploy with render.yaml

1. There's a `render.yaml` in the `cloud_deployment` folder
2. On Render, go to **New +** → **Blueprint**
3. Connect repo and Render auto-detects the config

---

## 🔧 Part 2: Setup Raspberry Pi 5

### Hardware Required
- Raspberry Pi 5 (4GB or 8GB)
- INMP441 I2S MEMS Microphone ($3-5)
- MicroSD Card (32GB+)
- 5V/5A USB-C Power Supply
- Optional: SSD1306 OLED, LEDs, Buzzer

### Step 1: Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Choose: **Raspberry Pi OS Lite (64-bit)**
3. Click ⚙️ (Advanced Options):
   - Enable SSH
   - Set username/password
   - Configure WiFi
4. Flash to SD card

### Step 2: Wire the Hardware

#### INMP441 Microphone Wiring
```
INMP441 Pin    RPi5 GPIO
───────────    ─────────
VDD         →  3.3V   (Pin 1)
GND         →  GND    (Pin 6)
WS          →  GPIO19 (Pin 35)
SCK         →  GPIO18 (Pin 12)
SD          →  GPIO20 (Pin 38)
L/R         →  GND    (for left channel)
```

#### SSD1306 OLED (Optional)
```
OLED Pin       RPi5 GPIO
────────       ─────────
VCC         →  3.3V   (Pin 1)
GND         →  GND    (Pin 9)
SDA         →  GPIO2  (Pin 3)
SCL         →  GPIO3  (Pin 5)
```

### Step 3: Initial Pi Setup

```bash
# SSH into your Pi
ssh pi@raspberrypi.local

# Update system
sudo apt update && sudo apt upgrade -y

# Clone the repository
git clone https://github.com/YOUR_USERNAME/baby-cry-diagnostic.git
cd baby-cry-diagnostic/rpi5_client

# Make scripts executable
chmod +x setup.sh

# Run setup (installs everything)
sudo ./setup.sh
```

### Step 4: Configure Cloud Connection

Edit `config.json`:

```bash
nano config.json
```

Update the cloud URLs:

```json
{
  "cloud": {
    "api_url": "https://baby-cry-api.onrender.com",
    "ws_url": "wss://baby-cry-api.onrender.com/ws/stream"
  }
}
```

### Step 5: Test Audio

```bash
# Reboot to apply I2S changes
sudo reboot

# After reboot, test microphone
arecord -l

# Record a test clip
arecord -D hw:1,0 -f S24_LE -r 44100 -c 1 -d 5 test.wav

# List audio devices with our client
source .venv/bin/activate
python main.py --list-devices
```

### Step 6: Run the Client

```bash
# Activate virtual environment
source .venv/bin/activate

# Start client
python main.py

# You should see:
# [INFO] Connecting to wss://baby-cry-api.onrender.com/ws/stream...
# [INFO] Connected to cloud API
```

### Step 7: Enable Auto-Start

```bash
# Enable the systemd service
sudo systemctl enable baby-cry-monitor

# Start now
sudo systemctl start baby-cry-monitor

# Check status
sudo systemctl status baby-cry-monitor

# View logs
sudo journalctl -u baby-cry-monitor -f
```

---

## 📊 Part 3: Testing & Verification

### Test 1: API Health Check

```bash
curl https://baby-cry-api.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "models": {
    "ensemble": true,
    "biomarkers": true,
    "audio_processor": true,
    "pdf_generator": true
  }
}
```

### Test 2: Audio Analysis

```bash
# Upload a test audio file
curl -X POST \
  -F "audio_file=@test_cry.wav" \
  https://baby-cry-api.onrender.com/api/v1/analyze
```

Expected response:
```json
{
  "success": true,
  "diagnosis": {
    "primary_classification": "hungry_cry",
    "confidence": 0.85,
    "risk_level": "GREEN",
    "biomarkers": {...}
  }
}
```

### Test 3: WebSocket Stream

```bash
# Using wscat (npm install -g wscat)
wscat -c wss://baby-cry-api.onrender.com/ws/stream

# Send some test data or wait for response
> {"type": "ping"}
< {"type": "pong"}
```

---

## 🔍 Troubleshooting

### Cloud Issues

| Problem | Solution |
|---------|----------|
| Models not loading | Check HUGGINGFACE_TOKEN env var |
| Timeout on startup | First deploy is slow (model downloads) |
| 5xx errors | Check Render logs for exceptions |
| Memory issues | Upgrade to Standard plan ($25/mo) |

### Raspberry Pi Issues

| Problem | Solution |
|---------|----------|
| No audio devices | Check I2S wiring, run `arecord -l` |
| Cannot connect | Verify WiFi, check cloud URL |
| Service not starting | Check logs: `journalctl -u baby-cry-monitor` |
| Display blank | Run `sudo i2cdetect -y 1`, check wiring |

---

## 📁 Project Structure

```
baby-cry-diagnostic/
├── cloud_deployment/           # Cloud backend
│   ├── Dockerfile
│   ├── render.yaml
│   ├── main.py                 # FastAPI server
│   ├── requirements.txt
│   ├── models/
│   │   ├── ensemble.py         # 6-backbone AI
│   │   ├── biomarkers.py
│   │   └── trained_weights/    # Pre-trained classifiers
│   └── services/
│       ├── audio_processor.py
│       └── pdf_generator.py
│
├── rpi5_client/               # Raspberry Pi client
│   ├── main.py                # Audio capture & streaming
│   ├── config.json            # Configuration
│   ├── setup.sh               # Automated setup
│   └── requirements.txt
│
├── trained_classifiers/       # Trained model weights
│   ├── cry/*.pt
│   └── pulmonary/*.pt
│
├── ast_baby_cry_optimized/    # Fine-tuned AST models
└── ast_respiratory_optimized/
```

---

## 💰 Cost Estimation

### Cloud (Render)
- **Free Tier**: 750 hours/month (sleeps after 15min inactivity)
- **Starter**: $7/month (always-on, faster)
- **Standard**: $25/month (more RAM for models)

### Hardware (One-Time)
| Item | Cost |
|------|------|
| Raspberry Pi 5 (4GB) | $60 |
| MicroSD 32GB | $10 |
| INMP441 Microphone | $5 |
| Power Supply | $15 |
| SSD1306 OLED | $8 |
| LEDs + Resistors | $3 |
| Buzzer | $2 |
| **Total** | **~$103** |

---

## 🔐 Security Notes

1. **API Keys**: Never commit tokens to Git
2. **HTTPS**: Render provides free SSL
3. **WebSocket**: Uses secure WSS protocol
4. **Updates**: Regularly update Pi OS and packages

---

## 📞 Support

- **GitHub Issues**: Report bugs
- **Documentation**: See /docs folder
- **Model Info**: GET /api/v1/models

---

## ⚠️ Disclaimer

This is a **monitoring tool**, NOT a medical device. Always:
- Supervise infants directly
- Consult pediatricians for health concerns
- Don't rely solely on automated alerts
