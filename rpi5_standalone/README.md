# Baby Cry Diagnostic System - Raspberry Pi 5 Standalone

**Complete self-contained system running entirely on RPi5. No cloud, no external server.**

## Hardware Requirements

- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- INMP441 I2S MEMS Microphone
- MicroSD Card (32GB+ recommended)
- Power supply (5V 5A USB-C)
- Optional: SSD via USB3 for better performance
- Optional: SSD1306 OLED display (128x64)
- Optional: LEDs for status indication

---

## INMP441 I2S Microphone Wiring

```
INMP441 Pin    →    Raspberry Pi 5 GPIO
─────────────────────────────────────────
VDD            →    3.3V (Pin 1)
GND            →    GND (Pin 6)
SD (Data)      →    GPIO 21 / PCM_DIN (Pin 40)
SCK (Clock)    →    GPIO 18 / PCM_CLK (Pin 12)
WS (Word Sel)  →    GPIO 19 / PCM_FS (Pin 35)
L/R            →    GND (for LEFT channel) or 3.3V (for RIGHT)
─────────────────────────────────────────
```

### Pin Diagram

```
                    Raspberry Pi 5 GPIO Header
                    ┌─────────────────────────┐
           3.3V [1] │ ●  ●                    │ [2] 5V
      I2C SDA2  [3] │ ●  ●                    │ [4] 5V
      I2C SCL2  [5] │ ●  ●                    │ [6] GND ◄─── INMP441 GND
                [7] │ ●  ●                    │ [8]
            GND [9] │ ●  ●                    │ [10]
               [11] │ ●  ●                    │ [12] GPIO18 (PCM_CLK) ◄─── INMP441 SCK
               [13] │ ●  ●                    │ [14] GND
               [15] │ ●  ●                    │ [16]
          3.3V [17] │ ●  ●                    │ [18]
               [19] │ ●  ●                    │ [20] GND
               [21] │ ●  ●                    │ [22]
               [23] │ ●  ●                    │ [24]
           GND [25] │ ●  ●                    │ [26]
               [27] │ ●  ●                    │ [28]
               [29] │ ●  ●                    │ [30] GND
               [31] │ ●  ●                    │ [32]
               [33] │ ●  ●                    │ [34] GND
   GPIO19 (FS) [35] │ ●  ●                    │ [36] ◄─── INMP441 WS
               [37] │ ●  ●                    │ [38]
           GND [39] │ ●  ●                    │ [40] GPIO21 (PCM_DIN) ◄─── INMP441 SD
                    └─────────────────────────┘
```

### INMP441 L/R Pin Configuration

| L/R Connected To | Result |
|------------------|--------|
| GND              | Left channel (mono) - **Recommended** |
| 3.3V             | Right channel |

---

## Optional: OLED Display Wiring (SSD1306)

```
SSD1306 Pin    →    Raspberry Pi 5 GPIO
─────────────────────────────────────────
VCC            →    3.3V (Pin 1)
GND            →    GND (Pin 9)
SCL            →    GPIO 3 / I2C SCL (Pin 5)
SDA            →    GPIO 2 / I2C SDA (Pin 3)
```

## Optional: LED Indicators

```
LED            →    Raspberry Pi 5 GPIO
─────────────────────────────────────────
Green LED (+)  →    GPIO 17 (Pin 11) via 330Ω resistor
Yellow LED (+) →    GPIO 27 (Pin 13) via 330Ω resistor
Red LED (+)    →    GPIO 22 (Pin 15) via 330Ω resistor
All LED (-)    →    GND (Pin 14)
```

---

## Quick Install

```bash
# Clone repository
git clone https://github.com/ashutoshvm127/BabyCry.git
cd BabyCry/rpi5_standalone

# Run installer (does everything)
chmod +x install.sh
sudo ./install.sh
```

## Manual Installation

### 1. Enable I2S Audio

```bash
# Edit boot config
sudo nano /boot/firmware/config.txt

# Add these lines:
dtparam=i2s=on
dtoverlay=i2s-mmap
dtoverlay=googlevoicehat-soundcard

# Reboot
sudo reboot
```

### 2. Install Dependencies

```bash
# System packages
sudo apt update
sudo apt install -y python3-pip python3-venv libportaudio2 portaudio19-dev \
    libatlas-base-dev libopenblas-dev libasound2-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Test Microphone

```bash
# List audio devices
arecord -l

# Test recording
arecord -D plughw:1,0 -f S32_LE -r 16000 -c 1 -d 5 test.wav
aplay test.wav
```

### 4. Run the System

```bash
# Activate venv
source venv/bin/activate

# Run (manual)
python main.py

# Or enable as service (auto-start on boot)
sudo ./install_service.sh
```

---

## Systemd Service (24/7 Auto-Restart)

The service automatically:
- Starts on boot
- Restarts if it crashes
- Logs to journalctl

```bash
# Check status
sudo systemctl status babycry

# View logs
sudo journalctl -u babycry -f

# Stop/Start/Restart
sudo systemctl stop babycry
sudo systemctl start babycry
sudo systemctl restart babycry

# Disable auto-start
sudo systemctl disable babycry
```

---

## Web Interface

Access the web dashboard from any device on your network:

```
http://<rpi5-ip>:8080
```

Features:
- Real-time cry classification
- History of detections
- Audio waveform visualization
- Risk level indicators

---

## Cost Analysis

| Solution | Monthly Cost | Latency | Reliability |
|----------|--------------|---------|-------------|
| **RPi5 Standalone** | **$0** | <1s | 24/7 on your network |
| Render Cloud | $7-25+ | 2-5s | Depends on tier |
| AWS/GCP | $20-50+ | 1-3s | High |

**RPi5 Standalone = Zero ongoing costs!**

---

## Troubleshooting

### No audio input
```bash
# Check if microphone is detected
arecord -l

# Should show something like:
# card 1: googlevoicehat [googlevoicehat-soundcard], device 0: ...
```

### I2S not working
```bash
# Verify overlays loaded
sudo vcdbg log msg | grep -i i2s

# Check config
cat /boot/firmware/config.txt | grep -i i2s
```

### Model too slow
- Use RPi5 8GB version
- Add USB3 SSD for swap
- Use quantized model (INT8)

### Service not starting
```bash
# Check logs
sudo journalctl -u babycry -n 50 --no-pager

# Check file permissions
ls -la /home/pi/BabyCry/rpi5_standalone/
```
