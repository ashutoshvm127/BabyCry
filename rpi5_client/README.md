# Baby Cry Monitor - Raspberry Pi 5 Client

Real-time audio capture from INMP441 I2S microphone with cloud AI analysis.

## Features

- **Audio Capture**: INMP441 I2S MEMS microphone support
- **Cloud Streaming**: WebSocket connection to cloud AI backend
- **Status Display**: SSD1306 OLED display (128x64)
- **LED Indicators**: Green/Yellow/Red risk level LEDs
- **Audio Alerts**: Buzzer for high-risk notifications
- **Offline Buffering**: Queue audio when cloud disconnected
- **Auto-Reconnect**: Automatic cloud reconnection

## Hardware Requirements

### Required
- Raspberry Pi 5 (4GB or 8GB)
- INMP441 I2S MEMS Microphone
- MicroSD card (32GB+)
- 5V 5A USB-C Power Supply

### Optional
- SSD1306 OLED Display (128x64, I2C)
- 3x LEDs (Green, Yellow, Red) + resistors
- Passive buzzer or speaker

## Wiring Diagram

### INMP441 Microphone → RPi5

```
INMP441        RPi5 GPIO Header
────────       ─────────────────
VDD     ────▶  3.3V (Pin 1)
GND     ────▶  GND  (Pin 6)
WS      ────▶  GPIO 19 (Pin 35) - I2S Frame Sync
SCK     ────▶  GPIO 18 (Pin 12) - I2S Clock
SD      ────▶  GPIO 20 (Pin 38) - I2S Data In
L/R     ────▶  GND (Left channel)
```

### SSD1306 OLED Display → RPi5 (I2C)

```
SSD1306        RPi5 GPIO Header
────────       ─────────────────
VCC     ────▶  3.3V (Pin 1)
GND     ────▶  GND  (Pin 9)
SDA     ────▶  GPIO 2 (Pin 3) - I2C Data
SCL     ────▶  GPIO 3 (Pin 5) - I2C Clock
```

### Status LEDs → RPi5

```
LED            RPi5 GPIO Header
────────       ─────────────────
Green   ────▶  GPIO 22 (Pin 15) via 330Ω resistor
Yellow  ────▶  GPIO 23 (Pin 16) via 330Ω resistor
Red     ────▶  GPIO 24 (Pin 18) via 330Ω resistor
Common  ────▶  GND
```

### Buzzer → RPi5

```
Buzzer         RPi5 GPIO Header
────────       ─────────────────
Signal  ────▶  GPIO 17 (Pin 11)
GND     ────▶  GND
```

## Quick Start

### 1. Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Select Raspberry Pi OS Lite (64-bit)
3. Configure WiFi and SSH in advanced options
4. Flash to SD card

### 2. Initial Setup

```bash
# SSH into your Pi
ssh pi@raspberrypi.local

# Clone the repository
git clone https://github.com/YOUR_USERNAME/baby-cry-diagnostic.git
cd baby-cry-diagnostic/rpi5_client

# Run setup script
chmod +x setup.sh
sudo ./setup.sh

# Reboot to apply I2S configuration
sudo reboot
```

### 3. Configure Cloud Connection

Edit `config.json` with your cloud API URL:

```json
{
  "cloud": {
    "ws_url": "wss://YOUR-APP-NAME.onrender.com/ws/stream"
  }
}
```

### 4. Test Audio

```bash
# Check if microphone is detected
arecord -l

# Record a test sample
arecord -D hw:1,0 -f S24_LE -r 44100 -c 1 -d 5 test.wav

# Play back (if you have speakers)
aplay test.wav
```

### 5. Run Client

```bash
# Activate virtual environment
source .venv/bin/activate

# Run client
python main.py

# Or run in background
nohup python main.py > /tmp/baby_cry.log 2>&1 &
```

### 6. Enable Auto-Start

```bash
# Enable service
sudo systemctl enable baby-cry-monitor

# Start service
sudo systemctl start baby-cry-monitor

# Check status
sudo systemctl status baby-cry-monitor

# View logs
sudo journalctl -u baby-cry-monitor -f
```

## Configuration

### config.json

```json
{
  "cloud": {
    "api_url": "https://baby-cry-api.onrender.com",
    "ws_url": "wss://baby-cry-api.onrender.com/ws/stream",
    "reconnect_interval": 5,
    "ping_interval": 30
  },
  "audio": {
    "sample_rate": 16000,
    "buffer_duration": 3.0,
    "device_index": null
  },
  "hardware": {
    "display_enabled": true,
    "display_type": "ssd1306",
    "speaker_enabled": true,
    "led_enabled": true
  }
}
```

## Command Line Options

```bash
# List audio devices
python main.py --list-devices

# Use specific device
python main.py --device 1

# Use custom config
python main.py --config /path/to/config.json

# Generate default config
python main.py --generate-config
```

## LED Behavior

| LED | Condition |
|-----|-----------|
| 🟢 Green | Normal / Healthy |
| 🟡 Yellow | Warning / Monitor closely |
| 🔴 Red | Alert / Action needed |
| Blinking Yellow | Connecting to cloud |
| All Off | System error |

## Troubleshooting

### No audio input

```bash
# Check ALSA configuration
cat /proc/asound/cards

# Check I2S overlay
vcgencmd get_config dtparam

# Verify microphone
arecord -l
```

### Display not working

```bash
# Check I2C devices
sudo i2cdetect -y 1

# Should show 3C (SSD1306 address)
```

### Cannot connect to cloud

```bash
# Check network
ping google.com

# Test WebSocket
wscat -c wss://your-app.onrender.com/ws/stream

# Check logs
tail -f /var/log/baby-cry-monitor.log
```

### High CPU usage

```bash
# Check if multiple instances running
ps aux | grep python

# Kill duplicates
pkill -f "python main.py"
```

## Performance Optimization

For minimal power consumption:

1. Use WiFi instead of Ethernet when possible
2. Disable HDMI: `sudo tvservice -o`
3. Reduce GPU memory: `gpu_mem=16` in `/boot/config.txt`
4. Use lightweight OS (Raspian Lite)

## Safety Notes

⚠️ **This is a monitoring tool, NOT a medical device**

- Always supervise your baby directly
- Seek medical attention for any concerning symptoms
- Do not rely solely on automated alerts
- Follow your pediatrician's guidance

## License

MIT License - See LICENSE file
