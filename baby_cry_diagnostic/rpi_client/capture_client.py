#!/usr/bin/env python3
"""
Raspberry Pi 5 Audio Capture Client
Streams audio from INMP441 I2S microphone to cloud server

Supports both RPi5 (I2S) and Desktop (standard mic) modes via config file.
"""

import os
import sys
import time
import json
import struct
import asyncio
import argparse
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import config
try:
    from config import get_config, SystemConfig
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Configuration defaults (used if config file not available)
DEFAULT_SERVER_URL = "ws://localhost:8000/ws/stream"
DEFAULT_SAMPLE_RATE = 16000  # Desktop default
DEFAULT_SAMPLE_RATE_RPI5 = 44100  # INMP441 native rate
DEFAULT_BIT_DEPTH = 16  # Desktop default
DEFAULT_BIT_DEPTH_RPI5 = 24  # INMP441 native
DEFAULT_BUFFER_DURATION = 3.0  # seconds


def load_system_config() -> Dict[str, Any]:
    """Load configuration from system_config.json"""
    if HAS_CONFIG:
        config = get_config()
        return {
            "is_rpi5_mode": config.is_rpi5_mode,
            "server_url": config.server.ws_url,
            **config.get_effective_audio_config()
        }
    
    # Fallback: try to load JSON directly
    config_path = Path(__file__).parent.parent / "system_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        is_rpi5 = data.get("is_rpi5_mode", False)
        server = data.get("server", {})
        
        ws_protocol = "wss" if server.get("use_ssl", False) else "ws"
        ws_url = f"{ws_protocol}://{server.get('host', 'localhost')}:{server.get('port', 8000)}{server.get('websocket_path', '/ws/stream')}"
        
        if is_rpi5:
            rpi5 = data.get("rpi5", {})
            return {
                "is_rpi5_mode": True,
                "server_url": ws_url,
                "sample_rate": rpi5.get("inmp441_sample_rate", 44100),
                "bit_depth": rpi5.get("inmp441_bit_depth", 24),
                "channels": data.get("audio", {}).get("channels", 1),
                "buffer_duration": data.get("audio", {}).get("buffer_duration", 3.0),
                "device_index": data.get("audio", {}).get("device_index"),
                "backend": "alsaaudio",
                "is_i2s": True
            }
        else:
            audio = data.get("audio", {})
            return {
                "is_rpi5_mode": False,
                "server_url": ws_url,
                "sample_rate": audio.get("sample_rate", 16000),
                "bit_depth": audio.get("bit_depth", 16),
                "channels": audio.get("channels", 1),
                "buffer_duration": audio.get("buffer_duration", 3.0),
                "device_index": audio.get("device_index"),
                "backend": "pyaudio",
                "is_i2s": False
            }
    
    # Default config (Desktop mode)
    return {
        "is_rpi5_mode": False,
        "server_url": DEFAULT_SERVER_URL,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "bit_depth": DEFAULT_BIT_DEPTH,
        "channels": 1,
        "buffer_duration": DEFAULT_BUFFER_DURATION,
        "device_index": None,
        "backend": "pyaudio",
        "is_i2s": False
    }


class INMP441Microphone:
    """
    INMP441 I2S MEMS Microphone Controller for Raspberry Pi 5
    
    Wiring (INMP441 → RPi5):
    - VDD → 3.3V
    - GND → GND
    - WS  → GPIO 19 (I2S Frame Sync)
    - SCK → GPIO 18 (I2S Clock)
    - SD  → GPIO 20 (I2S Data In)
    - L/R → GND (Left channel) or VDD (Right channel)
    """
    
    def __init__(
        self, 
        sample_rate: int = 44100,
        bit_depth: int = 24,
        channels: int = 1
    ):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channels = channels
        self.is_running = False
        self._audio_buffer = []
        self._lock = threading.Lock()
        
        # Try to import audio libraries
        self._init_audio_backend()
    
    def _init_audio_backend(self):
        """Initialize audio capture backend"""
        self.backend = None
        
        # Try PyAudio first
        try:
            import pyaudio
            self.backend = "pyaudio"
            self.pa = pyaudio.PyAudio()
            print("[OK] PyAudio backend initialized")
            return
        except ImportError:
            pass
        
        # Try sounddevice
        try:
            import sounddevice
            self.backend = "sounddevice"
            print("[OK] SoundDevice backend initialized")
            return
        except ImportError:
            pass
        
        # Try alsaaudio for Raspberry Pi
        try:
            import alsaaudio
            self.backend = "alsaaudio"
            print("[OK] ALSA backend initialized")
            return
        except ImportError:
            pass
        
        print("[!] No audio backend available")
        print("    Install with: pip install pyaudio sounddevice")
        print("    Or for RPi: sudo apt install python3-alsaaudio")
    
    def list_devices(self):
        """List available audio input devices"""
        devices = []
        
        if self.backend == "pyaudio":
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
        
        elif self.backend == "sounddevice":
            import sounddevice as sd
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': int(device['default_samplerate'])
                    })
        
        elif self.backend == "alsaaudio":
            import alsaaudio
            pcms = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
            for i, pcm in enumerate(pcms):
                devices.append({
                    'index': i,
                    'name': pcm,
                    'channels': 1,
                    'sample_rate': self.sample_rate
                })
        
        return devices
    
    def start_capture(self, device_index: Optional[int] = None, callback=None):
        """Start audio capture"""
        if self.backend is None:
            raise RuntimeError("No audio backend available")
        
        self.is_running = True
        
        if self.backend == "pyaudio":
            self._start_pyaudio_capture(device_index, callback)
        elif self.backend == "sounddevice":
            self._start_sounddevice_capture(device_index, callback)
        elif self.backend == "alsaaudio":
            self._start_alsa_capture(device_index, callback)
    
    def _start_pyaudio_capture(self, device_index: Optional[int], callback):
        """Start capture with PyAudio"""
        import pyaudio
        
        # Determine format
        if self.bit_depth == 24:
            format_type = pyaudio.paInt24
        elif self.bit_depth == 16:
            format_type = pyaudio.paInt16
        else:
            format_type = pyaudio.paFloat32
        
        def audio_callback(in_data, frame_count, time_info, status):
            if callback:
                callback(in_data)
            else:
                with self._lock:
                    self._audio_buffer.append(in_data)
            return (None, pyaudio.paContinue)
        
        self.stream = self.pa.open(
            format=format_type,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
            stream_callback=audio_callback
        )
        
        self.stream.start_stream()
    
    def _start_sounddevice_capture(self, device_index: Optional[int], callback):
        """Start capture with sounddevice"""
        import sounddevice as sd
        import numpy as np
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"[!] Audio status: {status}")
            
            # Convert to bytes
            if self.bit_depth == 16:
                audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = indata.tobytes()
            
            if callback:
                callback(audio_bytes)
            else:
                with self._lock:
                    self._audio_buffer.append(audio_bytes)
        
        self.stream = sd.InputStream(
            device=device_index,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype='float32',
            callback=audio_callback
        )
        
        self.stream.start()
    
    def _start_alsa_capture(self, device_index: Optional[int], callback):
        """Start capture with ALSA (Raspberry Pi)"""
        import alsaaudio
        
        # Find I2S device
        pcms = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
        device = 'default'
        
        for pcm in pcms:
            if 'i2s' in pcm.lower() or 'inmp441' in pcm.lower():
                device = pcm
                break
        
        if device_index is not None and device_index < len(pcms):
            device = pcms[device_index]
        
        self.pcm = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device=device
        )
        
        # Configure
        self.pcm.setchannels(self.channels)
        self.pcm.setrate(self.sample_rate)
        
        if self.bit_depth == 24:
            self.pcm.setformat(alsaaudio.PCM_FORMAT_S24_LE)
        elif self.bit_depth == 16:
            self.pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        else:
            self.pcm.setformat(alsaaudio.PCM_FORMAT_FLOAT_LE)
        
        self.pcm.setperiodsize(1024)
        
        # Start capture thread
        def capture_thread():
            while self.is_running:
                length, data = self.pcm.read()
                if length > 0:
                    if callback:
                        callback(data)
                    else:
                        with self._lock:
                            self._audio_buffer.append(data)
        
        self._capture_thread = threading.Thread(target=capture_thread)
        self._capture_thread.start()
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if hasattr(self, 'pcm'):
            self.pcm.close()
        
        if hasattr(self, '_capture_thread'):
            self._capture_thread.join(timeout=2.0)
    
    def get_buffer(self) -> bytes:
        """Get and clear audio buffer"""
        with self._lock:
            data = b''.join(self._audio_buffer)
            self._audio_buffer.clear()
        return data
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_capture()
        
        if self.backend == "pyaudio" and hasattr(self, 'pa'):
            self.pa.terminate()


class CloudStreamingClient:
    """
    WebSocket client for streaming audio to cloud server
    """
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0
    
    async def connect(self):
        """Connect to cloud server"""
        import websockets
        
        while True:
            try:
                print(f"[*] Connecting to {self.server_url}...")
                self.websocket = await websockets.connect(self.server_url)
                self.is_connected = True
                self.reconnect_delay = 1.0
                print("[OK] Connected to cloud server")
                return
            except Exception as e:
                print(f"[!] Connection failed: {e}")
                print(f"    Retrying in {self.reconnect_delay:.1f}s...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to server"""
        if not self.is_connected or self.websocket is None:
            return
        
        try:
            await self.websocket.send(audio_data)
        except Exception as e:
            print(f"[!] Send failed: {e}")
            self.is_connected = False
            await self.connect()
    
    async def receive_results(self):
        """Receive analysis results from server"""
        if not self.is_connected or self.websocket is None:
            return None
        
        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
            return json.loads(message)
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None
    
    async def close(self):
        """Close connection"""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False


async def main():
    """Main entry point"""
    # Load system configuration
    sys_config = load_system_config()
    
    parser = argparse.ArgumentParser(description="Baby Cry Capture Client (RPi5 / Desktop)")
    parser.add_argument(
        "--server", "-s",
        default=sys_config["server_url"],
        help=f"Cloud server WebSocket URL (default from config: {sys_config['server_url']})"
    )
    parser.add_argument(
        "--sample-rate", "-r",
        type=int, default=sys_config["sample_rate"],
        help=f"Sample rate in Hz (default from config: {sys_config['sample_rate']})"
    )
    parser.add_argument(
        "--bit-depth",
        type=int, default=sys_config["bit_depth"],
        help=f"Bit depth (default from config: {sys_config['bit_depth']})"
    )
    parser.add_argument(
        "--buffer", "-b",
        type=float, default=sys_config["buffer_duration"],
        help=f"Buffer duration in seconds (default from config: {sys_config['buffer_duration']})"
    )
    parser.add_argument(
        "--device", "-d",
        type=int, default=sys_config["device_index"],
        help="Audio input device index (default: system default)"
    )
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--rpi5", 
        action="store_true",
        default=sys_config["is_rpi5_mode"],
        help="Force RPi5 mode (24-bit I2S)"
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Force Desktop mode (16-bit standard)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    is_rpi5_mode = args.rpi5 and not args.desktop
    
    print("=" * 60)
    print("BABY CRY DIAGNOSTIC - CAPTURE CLIENT")
    if is_rpi5_mode:
        print("Mode: Raspberry Pi 5 + INMP441 I2S Microphone")
    else:
        print("Mode: Desktop/Windows + Standard Microphone")
    print("=" * 60)
    print()
    
    # Adjust settings for mode
    if is_rpi5_mode and not args.desktop:
        sample_rate = args.sample_rate if args.sample_rate != sys_config["sample_rate"] else DEFAULT_SAMPLE_RATE_RPI5
        bit_depth = args.bit_depth if args.bit_depth != sys_config["bit_depth"] else DEFAULT_BIT_DEPTH_RPI5
    else:
        sample_rate = args.sample_rate
        bit_depth = args.bit_depth
    
    # Initialize microphone
    mic = INMP441Microphone(sample_rate=sample_rate, bit_depth=bit_depth)
    
    # List devices if requested
    if args.list_devices:
        print("Available audio input devices:")
        print("-" * 40)
        devices = mic.list_devices()
        for device in devices:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['channels']}, Rate: {device['sample_rate']} Hz")
        print()
        return
    
    # Check backend
    if mic.backend is None:
        print("[ERROR] No audio backend available. Install required packages.")
        sys.exit(1)
    
    print(f"Configuration:")
    print(f"  Mode:        {'RPi5 (I2S)' if is_rpi5_mode else 'Desktop'}")
    print(f"  Server:      {args.server}")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Bit Depth:   {bit_depth}-bit")
    print(f"  Buffer:      {args.buffer} seconds")
    print(f"  Device:      {args.device or 'default'}")
    print()
    
    # Connect to cloud server
    client = CloudStreamingClient(args.server)
    await client.connect()
    
    # Audio buffer
    audio_buffer = []
    samples_needed = int(args.buffer * args.sample_rate)
    
    def audio_callback(data: bytes):
        audio_buffer.append(data)
    
    # Start capture
    print("[*] Starting audio capture...")
    mic.start_capture(device_index=args.device, callback=audio_callback)
    print("[OK] Capturing audio. Press Ctrl+C to stop.")
    print()
    
    try:
        while True:
            # Check buffer
            total_data = b''.join(audio_buffer)
            bytes_per_sample = mic.bit_depth // 8
            current_samples = len(total_data) // bytes_per_sample
            
            if current_samples >= samples_needed:
                # Send to server
                await client.send_audio(total_data[:samples_needed * bytes_per_sample])
                
                # Keep remainder
                audio_buffer.clear()
                remainder = total_data[samples_needed * bytes_per_sample:]
                if remainder:
                    audio_buffer.append(remainder)
                
                # Check for results
                result = await client.receive_results()
                if result and result.get('type') == 'analysis':
                    timestamp = result.get('timestamp', 'N/A')
                    classification = result.get('classification', 'Unknown')
                    confidence = result.get('confidence', 0) * 100
                    risk_level = result.get('risk_level', 'GREEN')
                    
                    # Color output based on risk
                    if risk_level == 'RED':
                        color = '\033[91m'  # Red
                    elif risk_level == 'YELLOW':
                        color = '\033[93m'  # Yellow
                    else:
                        color = '\033[92m'  # Green
                    
                    reset = '\033[0m'
                    
                    print(f"[{timestamp}] {color}{classification.upper()}{reset} "
                          f"({confidence:.1f}%) | Risk: {color}{risk_level}{reset}")
            
            await asyncio.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n[*] Stopping...")
    
    finally:
        mic.cleanup()
        await client.close()
        print("[OK] Client stopped")


if __name__ == "__main__":
    # Install websockets if not available
    try:
        import websockets
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'websockets'])
    
    asyncio.run(main())
