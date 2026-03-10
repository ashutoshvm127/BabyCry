# Baby Cry Diagnostic System - Cloud Deployment Guide

## 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEPLOYMENT ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐           ┌──────────────────────────────────────────┐
│  RASPBERRY PI 5 │           │              RENDER CLOUD                │
│                 │           │                                          │
│  ┌───────────┐  │   HTTPS   │  ┌────────────────────────────────────┐  │
│  │ INMP441   │  │ WebSocket │  │        FastAPI Backend             │  │
│  │ I2S Mic   │──┼───────────┼──│  - AI Ensemble (6 backbones)       │  │
│  └───────────┘  │           │  │  - DistilHuBERT / AST / YAMNet     │  │
│                 │           │  │  - Wav2Vec2 / WavLM / PANNs        │  │
│  ┌───────────┐  │           │  │  - Biomarker Analysis              │  │
│  │ LCD/LED   │  │           │  │  - PDF Report Generation           │  │
│  │ Display   │◀─┼───────────┼──│  - Real-time WebSocket Stream      │  │
│  └───────────┘  │  Results  │  └────────────────────────────────────┘  │
│                 │           │                    │                      │
│  ┌───────────┐  │           │  ┌─────────────────▼────────────────┐    │
│  │ Speaker   │  │           │  │         Trained Models            │    │
│  │ (Alert)   │  │           │  │  - cry/*.pt (5 classifiers)       │    │
│  └───────────┘  │           │  │  - pulmonary/*.pt (5 classifiers) │    │
└─────────────────┘           │  │  - AST fine-tuned weights         │    │
                              │  └──────────────────────────────────┘    │
                              │                                          │
                              │  ┌──────────────────────────────────┐    │
                              │  │      React Dashboard (optional)  │    │
                              │  │      - Real-time monitoring      │    │
                              │  │      - Historical analysis       │    │
                              │  └──────────────────────────────────┘    │
                              └──────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Deploy to Render

#### Option A: One-Click Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

#### Option B: Manual Deploy

```bash
# 1. Push code to GitHub
git init
git add .
git commit -m "Initial cloud deployment"
git remote add origin https://github.com/YOUR_USERNAME/baby-cry-diagnostic.git
git push -u origin main

# 2. Create new Web Service on Render
# - Connect your GitHub repo
# - Select "cloud_deployment" as root directory
# - Environment: Docker
# - Plan: Starter ($7/mo) or higher for GPU support
```

### 2. Set Environment Variables on Render

```env
# Required
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx

# Optional
LOG_LEVEL=INFO
MODEL_CACHE_DIR=/tmp/model_cache
CORS_ORIGINS=*
USE_GPU=false
```

### 3. Deploy Models (First Time)

Models will auto-download on first startup. To pre-load:

```bash
# SSH into Render shell
cd /app
python -m scripts.download_models
```

### 4. Setup Raspberry Pi 5

```bash
# Clone repo on RPi
git clone https://github.com/YOUR_USERNAME/baby-cry-diagnostic.git
cd baby-cry-diagnostic/rpi5_client

# Run setup script
chmod +x setup.sh
sudo ./setup.sh

# Configure cloud endpoint
nano config.json  # Set your Render URL

# Start client
python main.py
```

## 📁 Project Structure

```
cloud_deployment/
├── Dockerfile              # Production Docker image
├── render.yaml             # Render deployment config
├── requirements.txt        # Python dependencies
├── main.py                 # FastAPI application
├── models/                 # AI models
│   ├── ensemble.py         # 6-backbone ensemble
│   ├── biomarkers.py       # Medical biomarker extraction
│   └── trained_weights/    # Pre-trained classifier weights
├── services/               # Backend services
│   ├── audio_processor.py
│   └── pdf_generator.py
└── scripts/
    └── download_models.py  # Model download script

rpi5_client/
├── setup.sh                # Automated RPi5 setup
├── main.py                 # Audio capture client
├── config.json             # Cloud connection config
├── hardware/               # Hardware abstractions
│   ├── microphone.py       # INMP441 I2S driver
│   ├── display.py          # LCD/OLED display
│   └── speaker.py          # Audio alerts
└── systemd/
    └── baby-cry.service    # Systemd service file
```

## 🔧 Hardware Requirements

### Raspberry Pi 5
- Raspberry Pi 5 (4GB or 8GB recommended)
- MicroSD card (32GB+)
- Power supply (5V 5A USB-C)
- INMP441 I2S MEMS Microphone
- Optional: SSD1306 OLED Display (128x64)
- Optional: Speaker/Buzzer for alerts

### Cloud (Render)
- Starter plan ($7/mo) for CPU inference
- Standard plan ($25/mo) for faster inference
- GPU plan (contact Render) for real-time processing

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info & health |
| `/health` | GET | Health check |
| `/api/v1/analyze` | POST | Analyze audio file |
| `/ws/stream` | WS | Real-time audio stream |
| `/api/v1/report/{id}` | GET | Download PDF report |

## 🔒 Security Considerations

1. **API Keys**: Store in Render environment variables
2. **HTTPS**: Render provides free SSL certificates
3. **WebSocket**: Uses secure WSS protocol
4. **Rate Limiting**: Built-in rate limiting (100 req/min)

## 📈 Monitoring

- Render Dashboard: View logs, metrics, and alerts
- Health endpoint: `/health` returns model status
- WebSocket ping/pong for connection monitoring

## 🐛 Troubleshooting

### Models not loading
```bash
# Check model cache
ls -la /tmp/model_cache

# Re-download models
python -m scripts.download_models --force
```

### WebSocket disconnects
- Check Render logs for memory issues
- Increase plan tier if needed
- Verify RPi network connectivity

### Audio quality issues
- Check INMP441 wiring
- Verify I2S configuration
- Test with `arecord -l` on RPi

## 📞 Support

- GitHub Issues: Report bugs and feature requests
- Documentation: See /docs folder
- Community: Join our Discord server
