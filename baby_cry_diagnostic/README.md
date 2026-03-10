# Baby Cry Diagnostic System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

AI-powered infant cry analysis system for respiratory health monitoring. Uses an ensemble of deep learning models to classify baby cries and extract medical biomarkers.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────┐     ┌─────────────────┐
│  Raspberry Pi 5 │     │           Cloud Backend              │     │  React Dashboard│
│  + INMP441 Mic  │────▶│  FastAPI + AI Ensemble               │────▶│  Real-time UI   │
│                 │ WS  │  DistilHuBERT→AST→YAMNet            │ API │                 │
└─────────────────┘     └──────────────────────────────────────┘     └─────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   PDF Reports   │
                              │ Medical Grade   │
                              └─────────────────┘
```

## 🚀 Features

- **AI Ensemble Model**: 3-tier fallback system for robust classification
  - Primary: DistilHuBERT for medical biomarker detection
  - Fallback 1: AST (Audio Spectrogram Transformer) for cry classification
  - Fallback 2: YAMNet for general audio event detection

- **Medical Biomarkers**: Extracts clinical-grade acoustic features
  - Fundamental Frequency (f₀): >600 Hz indicates respiratory distress
  - Spectral Centroid: Measures cry "sharpness" (lung congestion indicator)
  - HNR (Harmonic-to-Noise Ratio): Low HNR indicates turbulent breath sounds

- **Real-time Streaming**: WebSocket connection from RPi5 to cloud
- **PDF Medical Reports**: Professional reports with waveform visualization
- **Risk Assessment**: GREEN/YELLOW/RED classification with recommendations

## 📁 Project Structure

```
baby_cry_diagnostic/
├── backend/                    # FastAPI backend
│   ├── main.py                 # API endpoints
│   ├── models/
│   │   ├── ensemble.py         # AI ensemble model
│   │   └── biomarkers.py       # Acoustic biomarker analyzer
│   ├── services/
│   │   ├── audio_processor.py  # Audio I/O & preprocessing
│   │   └── pdf_generator.py    # Medical PDF reports
│   └── requirements.txt
├── frontend/                   # React dashboard
│   ├── src/
│   │   ├── App.tsx             # Main dashboard component
│   │   └── index.tsx           # Entry point
│   ├── public/
│   └── package.json
├── data_ingestion/             # Dataset download scripts
│   └── download_datasets.py
├── rpi_client/                 # Raspberry Pi capture client
│   └── capture_client.py
├── docker-compose.yml          # Container orchestration
├── Dockerfile.backend          # Backend container
├── Dockerfile.frontend         # Frontend container
└── .env.template               # Environment config template
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (optional)
- CUDA-capable GPU (optional, for faster inference)

### 1. Clone & Configure

```bash
# Copy environment template
cp .env.template .env

# Edit with your credentials
# - KAGGLE_USERNAME
# - KAGGLE_KEY
# - HUGGINGFACE_TOKEN (optional)
```

### 2. Download Datasets

```bash
python data_ingestion/download_datasets.py
```

### 3. Run with Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Access:
# - Dashboard: http://localhost:3000
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### 4. Run Manually (Development)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

### 5. Raspberry Pi Client

```bash
# On your Raspberry Pi 5
cd rpi_client

# Install dependencies
pip install pyaudio websockets numpy

# Run capture client
python capture_client.py --server ws://YOUR_SERVER_IP:8000/ws/stream

# List available audio devices
python capture_client.py --list-devices
```

## 🔌 Raspberry Pi 5 + INMP441 Wiring

```
INMP441        RPi5 GPIO
────────       ─────────
VDD     ────▶  3.3V (Pin 1)
GND     ────▶  GND (Pin 6)
WS      ────▶  GPIO 19 (Pin 35) - I2S Frame Sync
SCK     ────▶  GPIO 18 (Pin 12) - I2S Clock
SD      ────▶  GPIO 20 (Pin 38) - I2S Data In
L/R     ────▶  GND (Left channel)
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/api/v1/analyze` | POST | Upload audio for analysis |
| `/api/v1/report/{id}` | GET | Download PDF medical report |
| `/ws/stream` | WS | Real-time audio streaming |

### Example: Upload Audio

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "audio_file=@baby_cry.wav"
```

### Response

```json
{
  "success": true,
  "diagnosis": {
    "id": "uuid-here",
    "timestamp": "2024-01-15T10:30:00",
    "primary_classification": "pain",
    "confidence": 0.87,
    "risk_level": "YELLOW",
    "risk_score": 45.5,
    "biomarkers": {
      "fundamental_frequency": 520.3,
      "hnr": 12.5,
      "spectral_centroid": 2100.0,
      "health_score": 75.0
    },
    "recommendations": [
      "MODERATE RISK: Continue monitoring",
      "Check for physical discomfort"
    ]
  }
}
```

## 📑 Medical PDF Report

The generated PDF includes:

1. **Session Information**: Timestamp, duration, sample rate
2. **Audio Waveform**: Visual representation of the cry
3. **AI Diagnosis**: Classification with confidence score
4. **Risk Assessment**: Color-coded risk level (GREEN/YELLOW/RED)
5. **Biomarkers Table**:
   - Fundamental Frequency (f₀) - respiratory indicator
   - Spectral Centroid - lung congestion marker  
   - HNR - breath quality metric
   - Jitter/Shimmer - vocal stability
6. **Health Score**: Overall assessment (0-100)
7. **Recommendations**: Clinical action items

## 🧠 AI Models

### Primary: DistilHuBERT
- Architecture: Distilled HuBERT (HuggingFace)
- Purpose: Extract audio embeddings for medical biomarker detection
- Paper: [DistilHuBERT](https://arxiv.org/abs/2110.01900)

### Fallback 1: AST
- Architecture: Audio Spectrogram Transformer
- Purpose: Cry classification using spectrogram analysis
- Paper: [AST](https://arxiv.org/abs/2104.01778)

### Fallback 2: YAMNet
- Architecture: MobileNet-based audio classifier
- Purpose: General audio event detection (527 classes)
- Source: [TensorFlow Hub](https://tfhub.dev/google/yamnet/1)

## ⚙️ Configuration

Key environment variables:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Model Settings
USE_GPU=true
MODEL_CACHE_DIR=./model_cache
PRIMARY_MODEL=ntu-spml/distilhubert

# RPi Client
RPI_SERVER_URL=ws://server:8000/ws/stream
RPI_SAMPLE_RATE=44100
```

## 🔒 Security Notes

- **This is a diagnostic aid, not a medical device**
- Always consult healthcare professionals for medical decisions
- API endpoints should be secured in production (add authentication)
- Use HTTPS in production deployments

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Support

For issues and questions, please open a GitHub issue.
