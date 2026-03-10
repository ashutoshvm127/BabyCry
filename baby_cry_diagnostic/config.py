# ==============================================================================
# Baby Cry Diagnostic System - System Configuration
# ==============================================================================
# This file controls whether the system runs in RPi5 mode or Desktop mode
# ==============================================================================

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Configuration file path
CONFIG_FILE = Path(__file__).parent / "system_config.json"


@dataclass
class AudioConfig:
    """Audio capture configuration"""
    sample_rate: int = 16000
    bit_depth: int = 16
    channels: int = 1
    buffer_duration: float = 3.0
    device_index: Optional[int] = None


@dataclass
class RPi5Config:
    """Raspberry Pi 5 specific configuration"""
    i2s_enabled: bool = True
    gpio_ws: int = 19      # I2S Frame Sync
    gpio_sck: int = 18     # I2S Clock
    gpio_sd: int = 20      # I2S Data In
    inmp441_bit_depth: int = 24
    inmp441_sample_rate: int = 44100


@dataclass
class DesktopConfig:
    """Desktop/Windows specific configuration"""
    use_default_mic: bool = True
    preferred_device_name: Optional[str] = None


@dataclass
class ServerConfig:
    """Server connection configuration"""
    host: str = "localhost"
    port: int = 8000
    use_ssl: bool = False
    websocket_path: str = "/ws/stream"
    
    @property
    def http_url(self) -> str:
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"
    
    @property
    def ws_url(self) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}{self.websocket_path}"


@dataclass
class SystemConfig:
    """
    Main system configuration
    
    Set is_rpi5_mode = True for Raspberry Pi 5 with INMP441 microphone
    Set is_rpi5_mode = False for Desktop/Windows with standard microphone
    """
    # =========================================================================
    # MAIN TOGGLE - Set this to switch between RPi5 and Desktop mode
    # =========================================================================
    is_rpi5_mode: bool = False  # True = RPi5, False = Desktop/Windows
    
    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    rpi5: RPi5Config = field(default_factory=RPi5Config)
    desktop: DesktopConfig = field(default_factory=DesktopConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    # Paths
    model_cache_dir: str = "./model_cache"
    reports_dir: str = "./reports"
    logs_dir: str = "./logs"
    
    # Debug/Development
    debug: bool = False
    log_level: str = "INFO"
    
    def get_effective_audio_config(self) -> Dict[str, Any]:
        """Get audio config based on current mode"""
        if self.is_rpi5_mode:
            return {
                "sample_rate": self.rpi5.inmp441_sample_rate,
                "bit_depth": self.rpi5.inmp441_bit_depth,
                "channels": self.audio.channels,
                "buffer_duration": self.audio.buffer_duration,
                "device_index": self.audio.device_index,
                "backend": "alsaaudio",
                "is_i2s": True
            }
        else:
            return {
                "sample_rate": self.audio.sample_rate,
                "bit_depth": self.audio.bit_depth,
                "channels": self.audio.channels,
                "buffer_duration": self.audio.buffer_duration,
                "device_index": self.audio.device_index,
                "backend": "pyaudio",
                "is_i2s": False
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "is_rpi5_mode": self.is_rpi5_mode,
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "bit_depth": self.audio.bit_depth,
                "channels": self.audio.channels,
                "buffer_duration": self.audio.buffer_duration,
                "device_index": self.audio.device_index
            },
            "rpi5": {
                "i2s_enabled": self.rpi5.i2s_enabled,
                "gpio_ws": self.rpi5.gpio_ws,
                "gpio_sck": self.rpi5.gpio_sck,
                "gpio_sd": self.rpi5.gpio_sd,
                "inmp441_bit_depth": self.rpi5.inmp441_bit_depth,
                "inmp441_sample_rate": self.rpi5.inmp441_sample_rate
            },
            "desktop": {
                "use_default_mic": self.desktop.use_default_mic,
                "preferred_device_name": self.desktop.preferred_device_name
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "use_ssl": self.server.use_ssl,
                "websocket_path": self.server.websocket_path
            },
            "model_cache_dir": self.model_cache_dir,
            "reports_dir": self.reports_dir,
            "logs_dir": self.logs_dir,
            "debug": self.debug,
            "log_level": self.log_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        """Create config from dictionary"""
        config = cls()
        
        config.is_rpi5_mode = data.get("is_rpi5_mode", False)
        
        if "audio" in data:
            audio = data["audio"]
            config.audio = AudioConfig(
                sample_rate=audio.get("sample_rate", 16000),
                bit_depth=audio.get("bit_depth", 16),
                channels=audio.get("channels", 1),
                buffer_duration=audio.get("buffer_duration", 3.0),
                device_index=audio.get("device_index")
            )
        
        if "rpi5" in data:
            rpi5 = data["rpi5"]
            config.rpi5 = RPi5Config(
                i2s_enabled=rpi5.get("i2s_enabled", True),
                gpio_ws=rpi5.get("gpio_ws", 19),
                gpio_sck=rpi5.get("gpio_sck", 18),
                gpio_sd=rpi5.get("gpio_sd", 20),
                inmp441_bit_depth=rpi5.get("inmp441_bit_depth", 24),
                inmp441_sample_rate=rpi5.get("inmp441_sample_rate", 44100)
            )
        
        if "desktop" in data:
            desktop = data["desktop"]
            config.desktop = DesktopConfig(
                use_default_mic=desktop.get("use_default_mic", True),
                preferred_device_name=desktop.get("preferred_device_name")
            )
        
        if "server" in data:
            server = data["server"]
            config.server = ServerConfig(
                host=server.get("host", "localhost"),
                port=server.get("port", 8000),
                use_ssl=server.get("use_ssl", False),
                websocket_path=server.get("websocket_path", "/ws/stream")
            )
        
        config.model_cache_dir = data.get("model_cache_dir", "./model_cache")
        config.reports_dir = data.get("reports_dir", "./reports")
        config.logs_dir = data.get("logs_dir", "./logs")
        config.debug = data.get("debug", False)
        config.log_level = data.get("log_level", "INFO")
        
        return config
    
    def save(self, path: Optional[Path] = None):
        """Save config to JSON file"""
        path = path or CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"[OK] Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SystemConfig":
        """Load config from JSON file"""
        path = path or CONFIG_FILE
        
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            # Return default config
            return cls()


# ==============================================================================
# Global config instance
# ==============================================================================
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = SystemConfig.load()
    return _config


def set_rpi5_mode(enabled: bool):
    """Quick toggle for RPi5 mode"""
    config = get_config()
    config.is_rpi5_mode = enabled
    config.save()
    print(f"[OK] System mode set to: {'RPi5' if enabled else 'Desktop/Windows'}")


def reload_config():
    """Reload configuration from file"""
    global _config
    _config = SystemConfig.load()
    return _config


# ==============================================================================
# CLI for config management
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baby Cry Diagnostic - System Configuration")
    parser.add_argument("--mode", choices=["rpi5", "desktop", "show", "init"], 
                       default="show", help="Set system mode or show current config")
    parser.add_argument("--server", type=str, help="Set server host:port")
    
    args = parser.parse_args()
    
    if args.mode == "init":
        # Create default config file
        config = SystemConfig()
        config.save()
        print("\n[OK] Created default configuration file")
        print(f"    File: {CONFIG_FILE}")
        print(f"    Mode: {'RPi5' if config.is_rpi5_mode else 'Desktop/Windows'}")
    
    elif args.mode == "rpi5":
        set_rpi5_mode(True)
    
    elif args.mode == "desktop":
        set_rpi5_mode(False)
    
    elif args.mode == "show":
        config = get_config()
        print("\n" + "=" * 60)
        print("BABY CRY DIAGNOSTIC - SYSTEM CONFIGURATION")
        print("=" * 60)
        print(f"\nMode: {'RPi5 (INMP441 I2S)' if config.is_rpi5_mode else 'Desktop/Windows'}")
        print(f"\nServer: {config.server.http_url}")
        print(f"WebSocket: {config.server.ws_url}")
        print(f"\nAudio Settings:")
        audio_cfg = config.get_effective_audio_config()
        for key, value in audio_cfg.items():
            print(f"  {key}: {value}")
        print(f"\nConfig file: {CONFIG_FILE}")
        print()
    
    if args.server:
        config = get_config()
        if ":" in args.server:
            host, port = args.server.rsplit(":", 1)
            config.server.host = host
            config.server.port = int(port)
        else:
            config.server.host = args.server
        config.save()
        print(f"[OK] Server updated: {config.server.http_url}")
