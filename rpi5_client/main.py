#!/usr/bin/env python3
"""
RPi5 Audio Capture Client - Lightweight Version

This client ONLY handles:
- Audio recording from INMP441 I2S microphone
- Basic waveform preprocessing (normalize, trim)
- Sending audio to Windows laptop for AI processing
- Displaying results

ALL AI processing happens on the laptop.
"""

import os
import io
import json
import time
import wave
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_CONFIG = {
    "server_url": "http://192.168.1.100:8080",  # Your laptop IP
    "api_key": "test-key",
    "sample_rate": 16000,
    "channels": 1,
    "chunk_duration": 5.0,  # seconds per chunk
    "recording_device": None,  # Auto-detect
    "save_recordings": True,
    "recordings_dir": "recordings",
    "display_enabled": False,  # Set True if you have OLED display
    "led_enabled": False,  # Set True if you have LEDs connected
}


class AudioCapture:
    """
    Handles audio recording from microphone.
    Works with INMP441 I2S mic or any ALSA device.
    """
    
    def __init__(self, sample_rate=16000, channels=1, chunk_duration=5.0, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device = device
        self.stream = None
        self.audio_interface = None
        
    def initialize(self):
        """Initialize audio capture"""
        try:
            import sounddevice as sd
            self.audio_interface = "sounddevice"
            
            # List available devices
            devices = sd.query_devices()
            logger.info(f"Available audio devices:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    logger.info(f"  [{i}] {dev['name']} (inputs: {dev['max_input_channels']})")
            
            # Use specified device or default
            if self.device is not None:
                sd.default.device = self.device
            
            logger.info(f"Audio capture ready: {self.sample_rate}Hz, {self.channels}ch")
            return True
            
        except ImportError:
            logger.warning("sounddevice not available, trying pyaudio...")
            try:
                import pyaudio
                self.audio_interface = "pyaudio"
                self.pa = pyaudio.PyAudio()
                logger.info("Using PyAudio for capture")
                return True
            except ImportError:
                logger.error("No audio library available! Install sounddevice or pyaudio")
                return False
    
    def record_chunk(self) -> np.ndarray:
        """Record a chunk of audio"""
        if self.audio_interface == "sounddevice":
            return self._record_sounddevice()
        elif self.audio_interface == "pyaudio":
            return self._record_pyaudio()
        else:
            # Fallback: generate silence (for testing)
            logger.warning("No audio interface - returning silence")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def _record_sounddevice(self) -> np.ndarray:
        """Record using sounddevice"""
        import sounddevice as sd
        
        logger.info(f"Recording {self.chunk_duration}s of audio...")
        try:
            audio = sd.rec(
                self.chunk_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
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
            
            logger.info(f"Recording {self.chunk_duration}s of audio...")
            for _ in range(num_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            
            stream.stop_stream()
            stream.close()
            
            return np.concatenate(frames)
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return np.zeros(self.chunk_samples, dtype=np.float32)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'pa'):
            self.pa.terminate()


class AudioPreprocessor:
    """
    Basic audio preprocessing on RPi5.
    Keeps it lightweight - heavy processing on laptop.
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def preprocess(self, waveform: np.ndarray) -> np.ndarray:
        """
        Basic preprocessing:
        - Normalize amplitude
        - Remove DC offset
        - Trim silence (basic)
        """
        # Remove DC offset
        waveform = waveform - np.mean(waveform)
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # Basic silence trimming (remove leading/trailing near-silence)
        threshold = 0.01
        non_silent = np.where(np.abs(waveform) > threshold)[0]
        
        if len(non_silent) > 0:
            start = max(0, non_silent[0] - int(0.1 * self.sample_rate))
            end = min(len(waveform), non_silent[-1] + int(0.1 * self.sample_rate))
            waveform = waveform[start:end]
        
        return waveform.astype(np.float32)
    
    def to_wav_bytes(self, waveform: np.ndarray) -> bytes:
        """Convert waveform to WAV file bytes for sending"""
        # Scale to int16
        audio_int16 = (waveform * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def get_basic_features(self, waveform: np.ndarray) -> dict:
        """Extract basic features on RPi5 (lightweight)"""
        return {
            "duration_seconds": len(waveform) / self.sample_rate,
            "rms_energy": float(np.sqrt(np.mean(waveform**2))),
            "max_amplitude": float(np.max(np.abs(waveform))),
            "zero_crossings": int(np.sum(np.abs(np.diff(np.sign(waveform))) > 0)),
        }


class LaptopClient:
    """
    Sends audio to laptop for AI processing.
    """
    
    def __init__(self, server_url: str, api_key: str = ""):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP client"""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession()
            
            # Test connection
            async with self.session.get(f"{self.server_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Connected to laptop server: {data.get('status')}")
                    return True
                else:
                    logger.error(f"Server returned status {resp.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to laptop: {e}")
            logger.info(f"Make sure server is running on {self.server_url}")
            return False
    
    async def analyze_audio(self, wav_bytes: bytes, basic_features: dict) -> dict:
        """
        Send audio to laptop for AI analysis.
        
        Args:
            wav_bytes: WAV file as bytes
            basic_features: Basic features computed on RPi5
        
        Returns:
            Analysis result from laptop
        """
        if self.session is None:
            return {"error": "Not connected to server"}
        
        try:
            import aiohttp
            
            # Create form data with audio file
            form = aiohttp.FormData()
            form.add_field(
                'audio',
                wav_bytes,
                filename='recording.wav',
                content_type='audio/wav'
            )
            
            # Add headers
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            
            # Send to laptop
            url = f"{self.server_url}/api/v1/analyze"
            async with self.session.post(url, data=form, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result
                else:
                    error_text = await resp.text()
                    logger.error(f"Analysis failed: {resp.status} - {error_text}")
                    return {"error": f"Server error: {resp.status}"}
                    
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        if self.session:
            await self.session.close()


class ResultDisplay:
    """
    Display results on console (and optionally OLED/LEDs)
    """
    
    def __init__(self, use_display=False, use_leds=False):
        self.use_display = use_display
        self.use_leds = use_leds
        self.oled = None
        
        if use_display:
            self._init_oled()
        if use_leds:
            self._init_leds()
    
    def _init_oled(self):
        """Initialize OLED display"""
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            
            serial = i2c(port=1, address=0x3C)
            self.oled = ssd1306(serial)
            logger.info("OLED display initialized")
        except Exception as e:
            logger.warning(f"OLED not available: {e}")
            self.use_display = False
    
    def _init_leds(self):
        """Initialize LED indicators"""
        try:
            import RPi.GPIO as GPIO
            
            self.LED_GREEN = 17
            self.LED_YELLOW = 27
            self.LED_RED = 22
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.OUT)
            GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.LOW)
            
            logger.info("LED indicators initialized")
        except Exception as e:
            logger.warning(f"LEDs not available: {e}")
            self.use_leds = False
    
    def show_result(self, result: dict):
        """Display analysis result"""
        if "error" in result:
            self._show_error(result["error"])
            return
        
        # Extract info
        diagnosis = result.get("diagnosis", {})
        classification = diagnosis.get("primary_classification", "Unknown")
        confidence = diagnosis.get("confidence", 0) * 100
        risk_level = diagnosis.get("risk_level", "YELLOW")
        
        # Console output
        print("\n" + "=" * 50)
        print(f"  BABY CRY ANALYSIS RESULT")
        print("=" * 50)
        print(f"  Classification: {classification.upper()}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Risk Level: {risk_level}")
        print("=" * 50)
        
        # Recommendations
        recommendations = diagnosis.get("recommendations", [])
        if recommendations:
            print("\n  Recommendations:")
            for rec in recommendations[:3]:
                print(f"    - {rec}")
        print()
        
        # OLED display
        if self.use_display and self.oled:
            self._update_oled(classification, confidence, risk_level)
        
        # LED indicators
        if self.use_leds:
            self._update_leds(risk_level)
    
    def _show_error(self, error: str):
        """Display error"""
        print(f"\n[ERROR] {error}\n")
        if self.use_leds:
            self._blink_all_leds()
    
    def _update_oled(self, classification: str, confidence: float, risk_level: str):
        """Update OLED display"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            image = Image.new('1', (self.oled.width, self.oled.height))
            draw = ImageDraw.Draw(image)
            
            # Try to load font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()
                font_small = font
            
            draw.text((0, 0), "Baby Cry Monitor", font=font_small, fill=255)
            draw.text((0, 16), f"{classification.upper()}", font=font, fill=255)
            draw.text((0, 32), f"Conf: {confidence:.0f}%", font=font_small, fill=255)
            draw.text((0, 48), f"Risk: {risk_level}", font=font_small, fill=255)
            
            self.oled.display(image)
        except Exception as e:
            logger.warning(f"OLED update failed: {e}")
    
    def _update_leds(self, risk_level: str):
        """Update LED indicators based on risk"""
        try:
            import RPi.GPIO as GPIO
            
            # Turn all off
            GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.LOW)
            
            # Turn on appropriate LED
            if risk_level == "GREEN":
                GPIO.output(self.LED_GREEN, GPIO.HIGH)
            elif risk_level == "YELLOW":
                GPIO.output(self.LED_YELLOW, GPIO.HIGH)
            elif risk_level == "RED":
                GPIO.output(self.LED_RED, GPIO.HIGH)
        except:
            pass
    
    def _blink_all_leds(self):
        """Blink all LEDs for error indication"""
        try:
            import RPi.GPIO as GPIO
            for _ in range(3):
                GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.HIGH)
                time.sleep(0.2)
                GPIO.output([self.LED_GREEN, self.LED_YELLOW, self.LED_RED], GPIO.LOW)
                time.sleep(0.2)
        except:
            pass
    
    def show_recording(self):
        """Indicate recording in progress"""
        print("  [Recording...]", end='\r')
    
    def show_processing(self):
        """Indicate processing"""
        print("  [Sending to laptop for AI analysis...]", end='\r')


async def main_loop(config: dict):
    """Main capture and analysis loop"""
    
    # Initialize components
    audio = AudioCapture(
        sample_rate=config["sample_rate"],
        channels=config["channels"],
        chunk_duration=config["chunk_duration"],
        device=config.get("recording_device")
    )
    
    if not audio.initialize():
        logger.error("Failed to initialize audio capture")
        return
    
    preprocessor = AudioPreprocessor(sample_rate=config["sample_rate"])
    
    client = LaptopClient(
        server_url=config["server_url"],
        api_key=config.get("api_key", "")
    )
    
    if not await client.initialize():
        logger.error("Failed to connect to laptop server")
        logger.info(f"Check that server is running on {config['server_url']}")
        return
    
    display = ResultDisplay(
        use_display=config.get("display_enabled", False),
        use_leds=config.get("led_enabled", False)
    )
    
    # Create recordings directory
    if config.get("save_recordings"):
        recordings_dir = Path(config.get("recordings_dir", "recordings"))
        recordings_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("Baby Cry Monitor - RPi5 Client")
    logger.info(f"Sending audio to: {config['server_url']}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 50)
    
    try:
        while True:
            # Record audio chunk
            display.show_recording()
            waveform = audio.record_chunk()
            
            # Basic preprocessing on RPi5
            waveform = preprocessor.preprocess(waveform)
            
            # Get basic features (computed locally)
            basic_features = preprocessor.get_basic_features(waveform)
            
            # Check if there's actual audio (not silence)
            if basic_features["rms_energy"] < 0.01:
                logger.info("  [Silence detected - skipping]")
                await asyncio.sleep(1)
                continue
            
            # Convert to WAV bytes
            wav_bytes = preprocessor.to_wav_bytes(waveform)
            
            # Save recording if enabled
            if config.get("save_recordings"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = recordings_dir / f"recording_{timestamp}.wav"
                with open(save_path, 'wb') as f:
                    f.write(wav_bytes)
            
            # Send to laptop for AI analysis
            display.show_processing()
            result = await client.analyze_audio(wav_bytes, basic_features)
            
            # Display result
            display.show_result(result)
            
            # Brief pause before next recording
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("\nStopping...")
    finally:
        audio.cleanup()
        await client.cleanup()


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                user_config = json.load(f)
                config.update(user_config)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baby Cry Monitor - RPi5 Client")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--server", type=str, help="Laptop server URL (overrides config)")
    parser.add_argument("--duration", type=float, help="Recording chunk duration in seconds")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line args
    if args.server:
        config["server_url"] = args.server
    if args.duration:
        config["chunk_duration"] = args.duration
    
    # Run main loop
    asyncio.run(main_loop(config))
