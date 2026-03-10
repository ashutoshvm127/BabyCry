#!/usr/bin/env python3
"""
Raspberry Pi 5 - Baby Cry Monitor Client
Captures audio from INMP441 I2S microphone and streams to cloud API

Features:
- Real-time audio capture from INMP441 I2S microphone
- WebSocket streaming to cloud API
- Local LCD/OLED display for status
- Audio alerts via speaker/buzzer
- Offline buffering with retry
- Auto-reconnection to cloud
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from queue import Queue, Empty
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/baby_cry_client.log")
    ]
)
logger = logging.getLogger("BabyCryClient")


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class CloudConfig:
    """Cloud connection settings"""
    api_url: str = "https://baby-cry-api.onrender.com"
    ws_url: str = "wss://baby-cry-api.onrender.com/ws/stream"
    api_key: str = ""
    reconnect_interval: int = 5
    ping_interval: int = 30
    connection_timeout: int = 30


@dataclass
class AudioConfig:
    """Audio capture settings"""
    sample_rate: int = 16000  # Will resample from 44100
    bit_depth: int = 16
    channels: int = 1
    buffer_duration: float = 3.0
    device_index: Optional[int] = None
    
    # INMP441 native settings
    inmp441_sample_rate: int = 44100
    inmp441_bit_depth: int = 24


@dataclass
class HardwareConfig:
    """Hardware settings"""
    # Display (SSD1306 OLED)
    display_enabled: bool = True
    display_type: str = "ssd1306"  # ssd1306, st7735, none
    display_width: int = 128
    display_height: int = 64
    display_i2c_address: int = 0x3C
    
    # Speaker/Buzzer
    speaker_enabled: bool = True
    speaker_gpio: int = 17
    
    # LED Status
    led_enabled: bool = True
    led_gpio_green: int = 22
    led_gpio_yellow: int = 23
    led_gpio_red: int = 24


@dataclass
class ClientConfig:
    """Complete client configuration"""
    cloud: CloudConfig = field(default_factory=CloudConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Runtime settings
    offline_buffer_size: int = 100  # Number of audio chunks to buffer offline
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, path: str = "config.json") -> "ClientConfig":
        """Load configuration from JSON file"""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            cloud=CloudConfig(**data.get("cloud", {})),
            audio=AudioConfig(**data.get("audio", {})),
            hardware=HardwareConfig(**data.get("hardware", {})),
            offline_buffer_size=data.get("offline_buffer_size", 100),
            log_level=data.get("log_level", "INFO")
        )
    
    def save(self, path: str = "config.json"):
        """Save configuration to JSON file"""
        data = {
            "cloud": {
                "api_url": self.cloud.api_url,
                "ws_url": self.cloud.ws_url,
                "api_key": self.cloud.api_key,
                "reconnect_interval": self.cloud.reconnect_interval,
                "ping_interval": self.cloud.ping_interval,
                "connection_timeout": self.cloud.connection_timeout
            },
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "bit_depth": self.audio.bit_depth,
                "channels": self.audio.channels,
                "buffer_duration": self.audio.buffer_duration,
                "device_index": self.audio.device_index,
                "inmp441_sample_rate": self.audio.inmp441_sample_rate,
                "inmp441_bit_depth": self.audio.inmp441_bit_depth
            },
            "hardware": {
                "display_enabled": self.hardware.display_enabled,
                "display_type": self.hardware.display_type,
                "display_width": self.hardware.display_width,
                "display_height": self.hardware.display_height,
                "display_i2c_address": self.hardware.display_i2c_address,
                "speaker_enabled": self.hardware.speaker_enabled,
                "speaker_gpio": self.hardware.speaker_gpio,
                "led_enabled": self.hardware.led_enabled,
                "led_gpio_green": self.hardware.led_gpio_green,
                "led_gpio_yellow": self.hardware.led_gpio_yellow,
                "led_gpio_red": self.hardware.led_gpio_red
            },
            "offline_buffer_size": self.offline_buffer_size,
            "log_level": self.log_level
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ==============================================================================
# Audio Capture
# ==============================================================================

class AudioCapture:
    """
    Audio capture from INMP441 I2S microphone
    
    Wiring:
    - VDD → 3.3V
    - GND → GND
    - WS  → GPIO 19 (I2S Frame Sync)
    - SCK → GPIO 18 (I2S Clock)
    - SD  → GPIO 20 (I2S Data In)
    """
    
    def __init__(self, config: AudioConfig, callback: Callable[[bytes], None]):
        self.config = config
        self.callback = callback
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._backend = None
        
        self._init_backend()
    
    def _init_backend(self):
        """Initialize audio capture backend"""
        # Try PyAudio first
        try:
            import pyaudio
            self._backend = "pyaudio"
            self._pa = pyaudio.PyAudio()
            logger.info("Using PyAudio backend")
            return
        except ImportError:
            pass
        
        # Try sounddevice
        try:
            import sounddevice
            self._backend = "sounddevice"
            logger.info("Using SoundDevice backend")
            return
        except ImportError:
            pass
        
        # Try alsaaudio (Linux/RPi)
        try:
            import alsaaudio
            self._backend = "alsaaudio"
            logger.info("Using ALSA backend")
            return
        except ImportError:
            pass
        
        logger.error("No audio backend available!")
        logger.error("Install with: pip install pyaudio sounddevice")
        logger.error("Or on RPi: sudo apt install python3-alsaaudio")
    
    def list_devices(self) -> list:
        """List available audio input devices"""
        devices = []
        
        if self._backend == "pyaudio":
            for i in range(self._pa.get_device_count()):
                info = self._pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
        
        elif self._backend == "sounddevice":
            import sounddevice as sd
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': int(device['default_samplerate'])
                    })
        
        elif self._backend == "alsaaudio":
            import alsaaudio
            for i, pcm in enumerate(alsaaudio.pcms(alsaaudio.PCM_CAPTURE)):
                devices.append({
                    'index': i,
                    'name': pcm,
                    'channels': 1,
                    'sample_rate': 44100
                })
        
        return devices
    
    def start(self):
        """Start audio capture"""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Audio capture started")
    
    def stop(self):
        """Stop audio capture"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Audio capture stopped")
    
    def _capture_loop(self):
        """Main capture loop"""
        import numpy as np
        
        # Calculate buffer size
        chunk_samples = int(self.config.sample_rate * 0.1)  # 100ms chunks
        
        if self._backend == "pyaudio":
            self._capture_pyaudio(chunk_samples)
        elif self._backend == "sounddevice":
            self._capture_sounddevice(chunk_samples)
        elif self._backend == "alsaaudio":
            self._capture_alsa(chunk_samples)
    
    def _capture_pyaudio(self, chunk_samples: int):
        """Capture using PyAudio"""
        import pyaudio
        import numpy as np
        
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=chunk_samples,
            input_device_index=self.config.device_index
        )
        
        try:
            while self.is_running:
                data = stream.read(chunk_samples, exception_on_overflow=False)
                self.callback(data)
        finally:
            stream.stop_stream()
            stream.close()
    
    def _capture_sounddevice(self, chunk_samples: int):
        """Capture using SoundDevice"""
        import sounddevice as sd
        import numpy as np
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            # Convert to int16 bytes
            audio_int16 = (indata * 32767).astype(np.int16)
            self.callback(audio_int16.tobytes())
        
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype='float32',
            blocksize=chunk_samples,
            device=self.config.device_index,
            callback=audio_callback
        ):
            while self.is_running:
                time.sleep(0.1)
    
    def _capture_alsa(self, chunk_samples: int):
        """Capture using ALSA (Raspberry Pi)"""
        import alsaaudio
        import numpy as np
        
        # For INMP441, use the native 44100 sample rate
        inp = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device='default'
        )
        inp.setchannels(self.config.channels)
        inp.setrate(self.config.inmp441_sample_rate)
        inp.setformat(alsaaudio.PCM_FORMAT_S24_LE)  # INMP441 is 24-bit
        inp.setperiodsize(chunk_samples)
        
        # Resample from 44100 to 16000
        resample_ratio = self.config.sample_rate / self.config.inmp441_sample_rate
        
        while self.is_running:
            length, data = inp.read()
            if length > 0:
                # Convert 24-bit to 16-bit
                audio_24 = np.frombuffer(data, dtype=np.int32)
                audio_16 = (audio_24 >> 8).astype(np.int16)
                
                # Resample
                if resample_ratio != 1.0:
                    from scipy.signal import resample
                    audio_16 = resample(audio_16, int(len(audio_16) * resample_ratio)).astype(np.int16)
                
                self.callback(audio_16.tobytes())


# ==============================================================================
# Display Manager
# ==============================================================================

class DisplayManager:
    """Manages OLED/LCD display for status information"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self._display = None
        self._draw = None
        self._font = None
        
        if config.display_enabled:
            self._init_display()
    
    def _init_display(self):
        """Initialize display hardware"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            if self.config.display_type == "ssd1306":
                from luma.core.interface.serial import i2c
                from luma.oled.device import ssd1306
                
                serial = i2c(port=1, address=self.config.display_i2c_address)
                self._display = ssd1306(serial, width=self.config.display_width, height=self.config.display_height)
            
            # Load font
            try:
                self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                self._font = ImageFont.load_default()
            
            logger.info(f"Display initialized: {self.config.display_type}")
            
        except ImportError as e:
            logger.warning(f"Display not available: {e}")
            logger.warning("Install with: pip install luma.oled pillow")
        except Exception as e:
            logger.error(f"Display init error: {e}")
    
    def show_status(self, status: str, risk_level: str = "GREEN", confidence: float = 0.0):
        """Display current status on screen"""
        if not self._display:
            return
        
        try:
            from PIL import Image, ImageDraw
            
            img = Image.new('1', (self.config.display_width, self.config.display_height), 0)
            draw = ImageDraw.Draw(img)
            
            # Title
            draw.text((0, 0), "Baby Monitor", font=self._font, fill=1)
            draw.line([(0, 12), (self.config.display_width, 12)], fill=1)
            
            # Status
            draw.text((0, 16), f"Status: {status}", font=self._font, fill=1)
            
            # Risk level with indicator
            risk_color = {"GREEN": "OK", "YELLOW": "WARN", "RED": "ALERT"}.get(risk_level, "?")
            draw.text((0, 28), f"Risk: {risk_color}", font=self._font, fill=1)
            
            # Confidence
            draw.text((0, 40), f"Conf: {confidence:.1%}", font=self._font, fill=1)
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            draw.text((0, 52), timestamp, font=self._font, fill=1)
            
            self._display.display(img)
            
        except Exception as e:
            logger.error(f"Display update error: {e}")
    
    def show_connecting(self):
        """Show connecting status"""
        self.show_status("Connecting...", "GREEN", 0.0)
    
    def show_error(self, message: str):
        """Show error message"""
        if not self._display:
            return
        
        try:
            from PIL import Image, ImageDraw
            
            img = Image.new('1', (self.config.display_width, self.config.display_height), 0)
            draw = ImageDraw.Draw(img)
            
            draw.text((0, 0), "ERROR", font=self._font, fill=1)
            draw.line([(0, 12), (self.config.display_width, 12)], fill=1)
            draw.text((0, 20), message[:20], font=self._font, fill=1)
            
            self._display.display(img)
            
        except Exception as e:
            logger.error(f"Display error: {e}")
    
    def clear(self):
        """Clear display"""
        if self._display:
            try:
                self._display.clear()
            except Exception as e:
                logger.error(f"Display clear error: {e}")


# ==============================================================================
# LED Status
# ==============================================================================

class LEDStatus:
    """Manages status LEDs (Green/Yellow/Red)"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self._gpio_available = False
        
        if config.led_enabled:
            self._init_gpio()
    
    def _init_gpio(self):
        """Initialize GPIO for LEDs"""
        try:
            import RPi.GPIO as GPIO
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            GPIO.setup(self.config.led_gpio_green, GPIO.OUT)
            GPIO.setup(self.config.led_gpio_yellow, GPIO.OUT)
            GPIO.setup(self.config.led_gpio_red, GPIO.OUT)
            
            self._gpio_available = True
            self.set_status("GREEN")
            
            logger.info("GPIO LEDs initialized")
            
        except ImportError:
            logger.warning("RPi.GPIO not available (not running on Pi?)")
        except Exception as e:
            logger.error(f"GPIO init error: {e}")
    
    def set_status(self, level: str):
        """Set LED status (GREEN, YELLOW, RED)"""
        if not self._gpio_available:
            return
        
        try:
            import RPi.GPIO as GPIO
            
            # Turn off all LEDs first
            GPIO.output(self.config.led_gpio_green, GPIO.LOW)
            GPIO.output(self.config.led_gpio_yellow, GPIO.LOW)
            GPIO.output(self.config.led_gpio_red, GPIO.LOW)
            
            # Turn on appropriate LED
            if level == "GREEN":
                GPIO.output(self.config.led_gpio_green, GPIO.HIGH)
            elif level == "YELLOW":
                GPIO.output(self.config.led_gpio_yellow, GPIO.HIGH)
            elif level == "RED":
                GPIO.output(self.config.led_gpio_red, GPIO.HIGH)
                
        except Exception as e:
            logger.error(f"LED set error: {e}")
    
    def cleanup(self):
        """Cleanup GPIO"""
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
            except:
                pass


# ==============================================================================
# Speaker/Buzzer
# ==============================================================================

class SpeakerAlert:
    """Manages audio alerts via speaker/buzzer"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self._gpio_available = False
        self._pwm = None
        
        if config.speaker_enabled:
            self._init_gpio()
    
    def _init_gpio(self):
        """Initialize GPIO for speaker/buzzer"""
        try:
            import RPi.GPIO as GPIO
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.speaker_gpio, GPIO.OUT)
            self._pwm = GPIO.PWM(self.config.speaker_gpio, 1000)
            self._gpio_available = True
            
            logger.info("Speaker GPIO initialized")
            
        except ImportError:
            logger.warning("RPi.GPIO not available")
        except Exception as e:
            logger.error(f"Speaker init error: {e}")
    
    def alert(self, level: str, duration: float = 0.5):
        """Play alert based on risk level"""
        if not self._gpio_available or not self._pwm:
            return
        
        try:
            if level == "RED":
                # Urgent: 3 short beeps
                for _ in range(3):
                    self._pwm.start(50)
                    self._pwm.ChangeFrequency(1500)
                    time.sleep(0.15)
                    self._pwm.stop()
                    time.sleep(0.1)
            
            elif level == "YELLOW":
                # Warning: 2 medium beeps
                for _ in range(2):
                    self._pwm.start(50)
                    self._pwm.ChangeFrequency(1000)
                    time.sleep(0.2)
                    self._pwm.stop()
                    time.sleep(0.15)
            
            elif level == "GREEN":
                # OK: 1 short low beep
                self._pwm.start(30)
                self._pwm.ChangeFrequency(500)
                time.sleep(0.1)
                self._pwm.stop()
                
        except Exception as e:
            logger.error(f"Speaker alert error: {e}")
    
    def cleanup(self):
        """Cleanup PWM"""
        if self._pwm:
            try:
                self._pwm.stop()
            except:
                pass


# ==============================================================================
# Cloud Client
# ==============================================================================

class CloudClient:
    """WebSocket client for cloud API communication"""
    
    def __init__(self, config: CloudConfig, on_result: Callable[[Dict], None]):
        self.config = config
        self.on_result = on_result
        self._ws = None
        self._connected = False
        self._reconnect_task = None
        self._ping_task = None
        self._message_queue: Queue = Queue()
        self._offline_buffer: deque = deque(maxlen=100)
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self):
        """Connect to cloud WebSocket"""
        import websockets
        
        while True:
            try:
                logger.info(f"Connecting to {self.config.ws_url}...")
                
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.config.ws_url,
                        ping_interval=self.config.ping_interval,
                        ping_timeout=10
                    ),
                    timeout=self.config.connection_timeout
                )
                
                self._connected = True
                logger.info("Connected to cloud API")
                
                # Send any buffered messages
                await self._flush_offline_buffer()
                
                # Start receiver
                await self._receive_loop()
                
            except asyncio.TimeoutError:
                logger.warning("Connection timeout, retrying...")
            except Exception as e:
                logger.error(f"Connection error: {e}")
            
            self._connected = False
            logger.info(f"Reconnecting in {self.config.reconnect_interval}s...")
            await asyncio.sleep(self.config.reconnect_interval)
    
    async def _receive_loop(self):
        """Receive messages from cloud"""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    self.on_result(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"Receive error: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to cloud"""
        if self._connected and self._ws:
            try:
                await self._ws.send(audio_data)
            except Exception as e:
                logger.error(f"Send error: {e}")
                self._connected = False
                self._offline_buffer.append(audio_data)
        else:
            # Buffer for later
            self._offline_buffer.append(audio_data)
    
    async def _flush_offline_buffer(self):
        """Send buffered messages after reconnection"""
        while self._offline_buffer and self._connected:
            try:
                data = self._offline_buffer.popleft()
                await self._ws.send(data)
                await asyncio.sleep(0.05)  # Small delay between sends
            except Exception as e:
                logger.error(f"Buffer flush error: {e}")
                break
    
    async def disconnect(self):
        """Disconnect from cloud"""
        self._connected = False
        if self._ws:
            await self._ws.close()


# ==============================================================================
# Main Application
# ==============================================================================

class BabyCryMonitor:
    """Main application orchestrator"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        
        # Audio queue for sending to cloud
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        
        # Initialize components
        self.display = DisplayManager(config.hardware)
        self.leds = LEDStatus(config.hardware)
        self.speaker = SpeakerAlert(config.hardware)
        self.cloud = CloudClient(config.cloud, self._on_result)
        self.audio = AudioCapture(config.audio, self._on_audio)
        
        # State
        self._running = False
        self._last_result: Optional[Dict] = None
        self._last_alert_time = 0
    
    def _on_audio(self, audio_data: bytes):
        """Callback for audio data from capture"""
        try:
            self._audio_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            pass  # Drop oldest data
    
    def _on_result(self, result: Dict):
        """Callback for results from cloud"""
        try:
            if result.get("type") == "analysis":
                classification = result.get("classification", "unknown")
                confidence = result.get("confidence", 0.0)
                risk_level = result.get("risk_level", "GREEN")
                
                logger.info(f"Result: {classification} ({risk_level}) {confidence:.1%}")
                
                # Update display
                self.display.show_status(classification, risk_level, confidence)
                
                # Update LEDs
                self.leds.set_status(risk_level)
                
                # Play alert (with rate limiting)
                current_time = time.time()
                if risk_level in ["RED", "YELLOW"] and current_time - self._last_alert_time > 10:
                    self.speaker.alert(risk_level)
                    self._last_alert_time = current_time
                
                self._last_result = result
            
            elif result.get("type") == "pong":
                pass  # Ping response
            
            elif result.get("type") == "error":
                logger.error(f"Cloud error: {result.get('message')}")
                self.display.show_error(result.get("message", "Unknown")[:20])
                
        except Exception as e:
            logger.error(f"Result handling error: {e}")
    
    async def _audio_sender(self):
        """Send audio data to cloud"""
        buffer = b""
        buffer_size = int(self.config.audio.sample_rate * self.config.audio.buffer_duration * 2)  # int16 = 2 bytes
        
        while self._running:
            try:
                # Get audio chunk with timeout
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                buffer += chunk
                
                # Send when buffer is full
                if len(buffer) >= buffer_size:
                    await self.cloud.send_audio(buffer[:buffer_size])
                    buffer = buffer[buffer_size:]
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio sender error: {e}")
    
    async def run(self):
        """Run the monitor"""
        logger.info("=" * 60)
        logger.info("BABY CRY MONITOR - RPi5 Client")
        logger.info(f"Cloud API: {self.config.cloud.ws_url}")
        logger.info("=" * 60)
        
        self._running = True
        
        # Show connecting status
        self.display.show_connecting()
        self.leds.set_status("YELLOW")
        
        # Start audio capture
        self.audio.start()
        
        try:
            # Run cloud connection and audio sender concurrently
            await asyncio.gather(
                self.cloud.connect(),
                self._audio_sender()
            )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self._running = False
            self.audio.stop()
            await self.cloud.disconnect()
            self.display.clear()
            self.leds.cleanup()
            self.speaker.cleanup()
    
    def run_sync(self):
        """Synchronous entry point"""
        asyncio.run(self.run())


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baby Cry Monitor - RPi5 Client")
    parser.add_argument("--config", "-c", default="config.json", help="Config file path")
    parser.add_argument("--server", "-s", help="Cloud WebSocket URL")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--device", "-d", type=int, help="Audio device index")
    parser.add_argument("--generate-config", action="store_true", help="Generate default config file")
    args = parser.parse_args()
    
    # Generate config
    if args.generate_config:
        config = ClientConfig()
        config.save(args.config)
        print(f"Config saved to {args.config}")
        return
    
    # Load config
    config = ClientConfig.from_file(args.config)
    
    # Override from CLI
    if args.server:
        config.cloud.ws_url = args.server
    if args.device is not None:
        config.audio.device_index = args.device
    
    # List devices
    if args.list_devices:
        audio = AudioCapture(config.audio, lambda x: None)
        devices = audio.list_devices()
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        for d in devices:
            print(f"  [{d['index']}] {d['name']}")
            print(f"      Channels: {d['channels']}, Rate: {d['sample_rate']}")
        return
    
    # Run monitor
    monitor = BabyCryMonitor(config)
    monitor.run_sync()


if __name__ == "__main__":
    main()
