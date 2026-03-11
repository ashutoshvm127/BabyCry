#!/usr/bin/env python3
"""
Baby Cry Diagnostic System - RPi5 Standalone

Complete self-contained system running entirely on Raspberry Pi 5.
No cloud, no external server - everything runs locally.

Features:
- Audio capture from INMP441 I2S microphone
- Lightweight AI model optimized for ARM
- Web dashboard accessible on local network
- Optional OLED display and LED indicators
- Systemd service for 24/7 operation
"""

import os
import io
import sys
import json
import wave
import time
import asyncio
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/babycry.log') if os.path.exists('/var/log') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_duration": 5.0,
        "device_name": "plughw:1,0",  # I2S device
        "use_sounddevice": True
    },
    "model": {
        "type": "lightweight",  # "lightweight" or "full"
        "use_quantization": True,
        "batch_size": 1
    },
    "web": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080
    },
    "display": {
        "oled_enabled": False,
        "oled_address": 0x3C,
        "led_enabled": False,
        "led_green": 17,
        "led_yellow": 27,
        "led_red": 22
    },
    "history": {
        "max_entries": 100
    }
}


def load_config() -> dict:
    """Load configuration from file"""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                user_config = json.load(f)
                # Deep merge
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    return config


# ==============================================================================
# Lightweight Model for RPi5
# ==============================================================================

class LightweightCryClassifier:
    """
    Optimized classifier for Raspberry Pi 5.
    Uses smaller models and optional quantization for speed.
    """
    
    def __init__(self, use_quantization: bool = True):
        self.use_quantization = use_quantization
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.is_initialized = False
        
        # Classification labels
        self.cry_classes = [
            "hungry", "pain", "sleepy", "discomfort", 
            "tired", "normal", "belly_pain", "scared"
        ]
        
        # Risk mapping
        self.risk_map = {
            "normal": "GREEN",
            "hungry": "GREEN", 
            "sleepy": "GREEN",
            "tired": "GREEN",
            "discomfort": "YELLOW",
            "belly_pain": "YELLOW",
            "scared": "YELLOW",
            "pain": "RED"
        }
    
    async def initialize(self):
        """Initialize the lightweight model"""
        if self.is_initialized:
            return
        
        logger.info("Loading lightweight model for RPi5...")
        
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
            
            # Use DistilHuBERT - smallest and fastest
            model_name = "ntu-spml/distilhubert"
            
            logger.info(f"  Loading {model_name}...")
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Apply dynamic quantization for faster inference on CPU
            if self.use_quantization:
                logger.info("  Applying INT8 quantization...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            self.model.eval()
            self.is_initialized = True
            logger.info("  [OK] Model ready")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("  Using fallback rule-based classifier")
            self.is_initialized = True  # Use fallback
    
    async def classify(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Classify baby cry audio.
        
        Returns:
            Dict with classification, confidence, risk_level
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Normalize audio
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=0)
        
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # If model loaded, use it
        if self.model is not None:
            return await self._model_classify(waveform, sample_rate)
        else:
            return self._fallback_classify(waveform, sample_rate)
    
    async def _model_classify(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Classify using the loaded model"""
        import torch
        import torch.nn.functional as F
        
        try:
            # Process audio
            inputs = self.processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Simple classification based on embedding statistics
            # (In production, you'd have a trained classifier head)
            emb = embeddings.numpy()[0]
            
            # Use embedding features to estimate class
            # This is a simplified approach - actual trained classifier would be better
            energy = np.mean(np.abs(waveform))
            pitch_proxy = np.mean(emb[:256])
            intensity_proxy = np.std(emb)
            
            # Rule-based classification using embeddings
            if energy < 0.1:
                pred_class = "normal"
                confidence = 0.7
            elif intensity_proxy > 0.5:
                pred_class = "pain"
                confidence = 0.6 + intensity_proxy * 0.3
            elif pitch_proxy > 0:
                pred_class = "hungry"
                confidence = 0.65
            else:
                pred_class = "discomfort"
                confidence = 0.55
            
            confidence = min(0.95, confidence)
            
        except Exception as e:
            logger.warning(f"Model inference failed: {e}")
            return self._fallback_classify(waveform, sample_rate)
        
        risk_level = self.risk_map.get(pred_class, "YELLOW")
        
        return {
            "classification": pred_class,
            "confidence": float(confidence),
            "risk_level": risk_level,
            "risk_score": self._compute_risk_score(pred_class, confidence, risk_level),
            "model": "distilhubert-quantized" if self.use_quantization else "distilhubert"
        }
    
    def _fallback_classify(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Rule-based fallback classifier when model not available"""
        # Extract simple features
        energy = np.sqrt(np.mean(waveform**2))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(waveform))) > 0)
        zcr = zero_crossings / len(waveform)
        
        # Simple rule-based classification
        if energy < 0.05:
            pred_class = "normal"
            confidence = 0.7
        elif energy > 0.4 and zcr > 0.1:
            pred_class = "pain"
            confidence = 0.6
        elif energy > 0.3:
            pred_class = "hungry"
            confidence = 0.55
        elif zcr > 0.08:
            pred_class = "discomfort"
            confidence = 0.5
        else:
            pred_class = "sleepy"
            confidence = 0.45
        
        risk_level = self.risk_map.get(pred_class, "YELLOW")
        
        return {
            "classification": pred_class,
            "confidence": float(confidence),
            "risk_level": risk_level,
            "risk_score": self._compute_risk_score(pred_class, confidence, risk_level),
            "model": "rule-based-fallback"
        }
    
    def _compute_risk_score(self, classification: str, confidence: float, risk_level: str) -> float:
        """Compute numeric risk score (0-100)"""
        base_scores = {"GREEN": 20, "YELLOW": 50, "RED": 80}
        base = base_scores.get(risk_level, 50)
        
        if risk_level == "RED":
            score = base + (confidence * 20)
        elif risk_level == "YELLOW":
            score = base + (confidence * 15)
        else:
            score = base - ((1 - confidence) * 10)
        
        return min(100, max(0, score))


# ==============================================================================
# Audio Capture (I2S Microphone)
# ==============================================================================

class I2SAudioCapture:
    """
    Audio capture from INMP441 I2S microphone.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_duration: float = 5.0, device_name: str = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device_name = device_name
        self.audio_lib = None
    
    def initialize(self) -> bool:
        """Initialize audio capture"""
        # Try sounddevice first (easier)
        try:
            import sounddevice as sd
            self.audio_lib = "sounddevice"
            
            devices = sd.query_devices()
            logger.info("Audio devices available:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    logger.info(f"  [{i}] {dev['name']}")
            
            return True
        except ImportError:
            pass
        
        # Try PyAudio as fallback
        try:
            import pyaudio
            self.audio_lib = "pyaudio"
            self.pa = pyaudio.PyAudio()
            logger.info("Using PyAudio for audio capture")
            return True
        except ImportError:
            pass
        
        # Try direct ALSA
        try:
            import alsaaudio
            self.audio_lib = "alsaaudio"
            logger.info("Using ALSA for audio capture")
            return True
        except ImportError:
            pass
        
        logger.error("No audio library available!")
        return False
    
    def record(self) -> np.ndarray:
        """Record audio chunk"""
        if self.audio_lib == "sounddevice":
            return self._record_sounddevice()
        elif self.audio_lib == "pyaudio":
            return self._record_pyaudio()
        elif self.audio_lib == "alsaaudio":
            return self._record_alsa()
        else:
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def _record_sounddevice(self) -> np.ndarray:
        """Record using sounddevice"""
        import sounddevice as sd
        
        try:
            audio = sd.rec(
                self.chunk_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def _record_pyaudio(self) -> np.ndarray:
        """Record using PyAudio"""
        import pyaudio
        
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        
        try:
            stream = self.pa.open(
                format=FORMAT,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            num_chunks = int(self.sample_rate / CHUNK * self.chunk_duration)
            
            for _ in range(num_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            
            stream.stop_stream()
            stream.close()
            
            return np.concatenate(frames)
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def _record_alsa(self) -> np.ndarray:
        """Record using ALSA directly"""
        import alsaaudio
        
        try:
            device = self.device_name or 'plughw:1,0'
            inp = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device=device
            )
            inp.setchannels(self.channels)
            inp.setrate(self.sample_rate)
            inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            inp.setperiodsize(1024)
            
            frames = []
            total_frames = 0
            
            while total_frames < self.chunk_samples:
                length, data = inp.read()
                if length > 0:
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    frames.append(audio)
                    total_frames += len(audio)
            
            return np.concatenate(frames)[:self.chunk_samples]
        except Exception as e:
            logger.error(f"ALSA recording failed: {e}")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'pa'):
            self.pa.terminate()


# ==============================================================================
# Display Output (OLED + LEDs)
# ==============================================================================

class DisplayOutput:
    """
    Output to OLED display and LEDs
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.oled = None
        self.gpio_available = False
        
        if config.get("oled_enabled"):
            self._init_oled()
        if config.get("led_enabled"):
            self._init_leds()
    
    def _init_oled(self):
        """Initialize OLED display"""
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            
            serial = i2c(port=1, address=self.config.get("oled_address", 0x3C))
            self.oled = ssd1306(serial)
            logger.info("OLED display initialized")
        except Exception as e:
            logger.warning(f"OLED not available: {e}")
    
    def _init_leds(self):
        """Initialize LED indicators"""
        try:
            import RPi.GPIO as GPIO
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            self.LED_GREEN = self.config.get("led_green", 17)
            self.LED_YELLOW = self.config.get("led_yellow", 27)
            self.LED_RED = self.config.get("led_red", 22)
            
            GPIO.setup([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.OUT)
            GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.LOW)
            
            self.gpio_available = True
            logger.info("LED indicators initialized")
        except Exception as e:
            logger.warning(f"GPIO not available: {e}")
    
    def show(self, result: Dict[str, Any]):
        """Display classification result"""
        classification = result.get("classification", "unknown")
        confidence = result.get("confidence", 0) * 100
        risk_level = result.get("risk_level", "YELLOW")
        
        # Console output
        print(f"\n{'='*40}")
        print(f"  {classification.upper()}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Risk: {risk_level}")
        print(f"{'='*40}\n")
        
        # OLED
        if self.oled:
            self._update_oled(classification, confidence, risk_level)
        
        # LEDs
        if self.gpio_available:
            self._update_leds(risk_level)
    
    def _update_oled(self, classification: str, confidence: float, risk_level: str):
        """Update OLED display"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            image = Image.new('1', (self.oled.width, self.oled.height))
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()
                font_small = font
            
            draw.text((0, 0), "Baby Cry Monitor", font=font_small, fill=255)
            draw.text((0, 16), classification.upper(), font=font, fill=255)
            draw.text((0, 32), f"Conf: {confidence:.0f}%", font=font_small, fill=255)
            draw.text((0, 48), f"Risk: {risk_level}", font=font_small, fill=255)
            
            self.oled.display(image)
        except Exception as e:
            logger.warning(f"OLED update failed: {e}")
    
    def _update_leds(self, risk_level: str):
        """Update LED indicators"""
        try:
            import RPi.GPIO as GPIO
            
            GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.LOW)
            
            if risk_level == "GREEN":
                GPIO.output(self.LED_GREEN, GPIO.HIGH)
            elif risk_level == "YELLOW":
                GPIO.output(self.LED_YELLOW, GPIO.HIGH)
            elif risk_level == "RED":
                GPIO.output(self.LED_RED, GPIO.HIGH)
        except:
            pass
    
    def cleanup(self):
        """Cleanup GPIO"""
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except:
            pass


# ==============================================================================
# Web Dashboard
# ==============================================================================

class WebDashboard:
    """
    Simple web dashboard for viewing results
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = None
        self.history = deque(maxlen=100)
        self.latest_result = None
    
    def setup(self, classifier: LightweightCryClassifier):
        """Setup FastAPI app"""
        try:
            from fastapi import FastAPI, WebSocket
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles
            import uvicorn
            
            self.app = FastAPI(title="Baby Cry Monitor")
            self.classifier = classifier
            
            @self.app.get("/", response_class=HTMLResponse)
            async def index():
                return self._get_html()
            
            @self.app.get("/api/latest")
            async def get_latest():
                return self.latest_result or {"status": "waiting"}
            
            @self.app.get("/api/history")
            async def get_history():
                return list(self.history)
            
            @self.app.get("/health")
            async def health():
                return {
                    "status": "healthy",
                    "model_loaded": classifier.is_initialized,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"Web dashboard ready at http://{self.host}:{self.port}")
            return True
            
        except ImportError as e:
            logger.warning(f"Web dashboard not available: {e}")
            return False
    
    def update(self, result: Dict[str, Any]):
        """Update with new result"""
        result["timestamp"] = datetime.now().isoformat()
        self.latest_result = result
        self.history.append(result)
    
    def run(self):
        """Run web server in background thread"""
        if self.app is None:
            return
        
        import uvicorn
        
        def run_server():
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning"
            )
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info(f"Web server started on port {self.port}")
    
    def _get_html(self) -> str:
        """Get dashboard HTML"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Baby Cry Monitor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 30px; font-size: 2em; }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .status {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            margin: 20px 0;
        }
        .GREEN { color: #4ade80; }
        .YELLOW { color: #facc15; }
        .RED { color: #f87171; }
        .info { display: flex; justify-content: space-around; margin-top: 20px; }
        .info-item { text-align: center; }
        .info-label { font-size: 0.9em; opacity: 0.7; }
        .info-value { font-size: 1.5em; font-weight: bold; }
        .history { max-height: 300px; overflow-y: auto; }
        .history-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .waiting { animation: pulse 2s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>👶 Baby Cry Monitor</h1>
        
        <div class="card">
            <div id="status" class="status waiting">Listening...</div>
            <div class="info">
                <div class="info-item">
                    <div class="info-label">Classification</div>
                    <div class="info-value" id="classification">-</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Confidence</div>
                    <div class="info-value" id="confidence">-</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Risk Level</div>
                    <div class="info-value" id="risk">-</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 15px;">History</h2>
            <div class="history" id="history">
                <p style="opacity: 0.5; text-align: center;">No detections yet</p>
            </div>
        </div>
    </div>
    
    <script>
        async function update() {
            try {
                const resp = await fetch('/api/latest');
                const data = await resp.json();
                
                if (data.classification) {
                    document.getElementById('status').className = 'status ' + data.risk_level;
                    document.getElementById('status').textContent = data.classification.toUpperCase();
                    document.getElementById('classification').textContent = data.classification;
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(0) + '%';
                    document.getElementById('risk').textContent = data.risk_level;
                    document.getElementById('risk').className = 'info-value ' + data.risk_level;
                }
                
                const histResp = await fetch('/api/history');
                const history = await histResp.json();
                
                if (history.length > 0) {
                    const historyHtml = history.slice(-10).reverse().map(item => `
                        <div class="history-item">
                            <span class="${item.risk_level}">${item.classification}</span>
                            <span>${new Date(item.timestamp).toLocaleTimeString()}</span>
                        </div>
                    `).join('');
                    document.getElementById('history').innerHTML = historyHtml;
                }
            } catch (e) {
                console.error('Update failed:', e);
            }
        }
        
        setInterval(update, 2000);
        update();
    </script>
</body>
</html>
'''


# ==============================================================================
# Main Application
# ==============================================================================

async def main(config: dict):
    """Main application loop"""
    logger.info("=" * 50)
    logger.info("BABY CRY MONITOR - RPi5 Standalone")
    logger.info("=" * 50)
    
    # Initialize components
    classifier = LightweightCryClassifier(
        use_quantization=config["model"].get("use_quantization", True)
    )
    await classifier.initialize()
    
    audio = I2SAudioCapture(
        sample_rate=config["audio"]["sample_rate"],
        channels=config["audio"]["channels"],
        chunk_duration=config["audio"]["chunk_duration"],
        device_name=config["audio"].get("device_name")
    )
    
    if not audio.initialize():
        logger.error("Failed to initialize audio capture")
        return
    
    display = DisplayOutput(config["display"])
    
    # Setup web dashboard
    web = None
    if config["web"].get("enabled", True):
        web = WebDashboard(
            host=config["web"].get("host", "0.0.0.0"),
            port=config["web"].get("port", 8080)
        )
        if web.setup(classifier):
            web.run()
    
    logger.info("")
    logger.info("System ready. Monitoring for baby cries...")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    try:
        while True:
            # Record audio
            logger.info("Recording...")
            waveform = audio.record()
            
            # Check for actual audio (not silence)
            energy = np.sqrt(np.mean(waveform**2))
            if energy < 0.01:
                logger.info("  [Silence - skipping]")
                await asyncio.sleep(1)
                continue
            
            # Classify
            logger.info("Classifying...")
            result = await classifier.classify(waveform, config["audio"]["sample_rate"])
            
            # Display
            display.show(result)
            
            # Update web dashboard
            if web:
                web.update(result)
            
            # Brief pause
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        audio.cleanup()
        display.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baby Cry Monitor - RPi5 Standalone")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--port", type=int, help="Web server port (default: 8080)")
    parser.add_argument("--no-web", action="store_true", help="Disable web dashboard")
    args = parser.parse_args()
    
    config = load_config()
    
    if args.port:
        config["web"]["port"] = args.port
    if args.no_web:
        config["web"]["enabled"] = False
    
    asyncio.run(main(config))
