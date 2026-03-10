#!/usr/bin/env python3
"""
Audio Processing Service for Baby Cry Diagnostic System
Handles audio I/O, resampling, and preprocessing
"""

import io
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np


class AudioProcessor:
    """
    Handles audio processing for the diagnostic system.
    
    Supports:
    - WAV, MP3, FLAC, OGG formats
    - Raw PCM I2S data (24-bit from INMP441)
    - Resampling to 16kHz (model requirement)
    - Normalization and preprocessing
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self._ensure_dependencies()
    
    def _ensure_dependencies(self):
        """Ensure required audio libraries are available"""
        try:
            import soundfile
            import librosa
            self.has_soundfile = True
            self.has_librosa = True
        except ImportError:
            self.has_soundfile = False
            self.has_librosa = False
            print("[!] Audio processing libraries not fully available")
    
    async def process_upload(
        self, 
        audio_bytes: bytes, 
        filename: Optional[str] = None
    ) -> Dict:
        """
        Process uploaded audio file.
        
        Args:
            audio_bytes: Raw bytes of audio file
            filename: Original filename for format detection
        
        Returns:
            Dictionary with waveform, sample_rate, and metadata
        """
        # Detect format from filename or magic bytes
        file_format = self._detect_format(audio_bytes, filename)
        
        # Load audio based on format
        if file_format == "raw_pcm":
            waveform, sample_rate = self._load_raw_pcm(audio_bytes)
        else:
            waveform, sample_rate = await self._load_audio_file(audio_bytes, file_format)
        
        # Preprocess
        waveform = self._preprocess(waveform, sample_rate)
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            waveform = self._resample(waveform, sample_rate, self.target_sample_rate)
        
        return {
            "waveform": waveform,
            "sample_rate": self.target_sample_rate,
            "original_sample_rate": sample_rate,
            "duration_seconds": len(waveform) / self.target_sample_rate,
            "format": file_format,
            "num_samples": len(waveform)
        }
    
    def process_i2s_chunk(
        self, 
        raw_bytes: bytes, 
        bit_depth: int = 24,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Process raw I2S data from INMP441 microphone.
        
        The INMP441 outputs 24-bit I2S data, MSB first.
        
        Args:
            raw_bytes: Raw I2S byte data
            bit_depth: Bit depth (24 for INMP441)
            sample_rate: Original sample rate
        
        Returns:
            Normalized float32 waveform
        """
        if bit_depth == 24:
            # Convert 24-bit I2S to float32
            waveform = self._convert_24bit_i2s(raw_bytes)
        elif bit_depth == 16:
            waveform = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            waveform = waveform / 32768.0
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Resample to target
        if sample_rate != self.target_sample_rate:
            waveform = self._resample(waveform, sample_rate, self.target_sample_rate)
        
        return waveform
    
    def _detect_format(self, audio_bytes: bytes, filename: Optional[str]) -> str:
        """Detect audio format from magic bytes or filename"""
        # Check magic bytes
        if audio_bytes[:4] == b'RIFF':
            return "wav"
        elif audio_bytes[:4] == b'fLaC':
            return "flac"
        elif audio_bytes[:4] == b'OggS':
            return "ogg"
        elif audio_bytes[:2] == b'\xff\xfb' or audio_bytes[:3] == b'ID3':
            return "mp3"
        
        # Fallback to filename extension
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in ['.wav', '.wave']:
                return "wav"
            elif ext == '.mp3':
                return "mp3"
            elif ext == '.flac':
                return "flac"
            elif ext == '.ogg':
                return "ogg"
            elif ext in ['.pcm', '.raw']:
                return "raw_pcm"
        
        # Default to raw PCM
        return "raw_pcm"
    
    async def _load_audio_file(
        self, 
        audio_bytes: bytes, 
        file_format: str
    ) -> Tuple[np.ndarray, int]:
        """Load audio from various file formats"""
        try:
            import soundfile as sf
            
            # Load using soundfile
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = sf.read(audio_buffer)
            
            # Convert stereo to mono
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            return waveform.astype(np.float32), sample_rate
            
        except Exception as e:
            # Fallback to librosa
            try:
                import librosa
                
                audio_buffer = io.BytesIO(audio_bytes)
                waveform, sample_rate = librosa.load(
                    audio_buffer, 
                    sr=None,  # Keep original sample rate
                    mono=True
                )
                
                return waveform.astype(np.float32), sample_rate
                
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load audio: soundfile error: {e}, librosa error: {e2}"
                )
    
    def _load_raw_pcm(
        self, 
        audio_bytes: bytes, 
        bit_depth: int = 16,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """Load raw PCM audio data"""
        if bit_depth == 16:
            waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            waveform = waveform / 32768.0
        elif bit_depth == 24:
            waveform = self._convert_24bit_i2s(audio_bytes)
        elif bit_depth == 32:
            waveform = np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        return waveform, sample_rate
    
    def _convert_24bit_i2s(self, raw_bytes: bytes) -> np.ndarray:
        """
        Convert 24-bit I2S data to float32.
        
        INMP441 outputs 24-bit signed integers, MSB first.
        Data is left-justified in 32-bit words.
        """
        # Ensure length is multiple of 3
        n_bytes = len(raw_bytes)
        n_samples = n_bytes // 3
        
        waveform = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            idx = i * 3
            # MSB first (big-endian 24-bit)
            sample = (raw_bytes[idx] << 16) | (raw_bytes[idx+1] << 8) | raw_bytes[idx+2]
            
            # Handle sign extension
            if sample & 0x800000:
                sample = sample - 0x1000000
            
            # Normalize to [-1, 1]
            waveform[i] = sample / 8388608.0  # 2^23
        
        return waveform
    
    def _preprocess(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply preprocessing to audio waveform"""
        # Ensure float32
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Remove DC offset
        waveform = waveform - np.mean(waveform)
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # Apply pre-emphasis filter (optional, helps with high-freq content)
        # pre_emphasis = 0.97
        # waveform = np.append(waveform[0], waveform[1:] - pre_emphasis * waveform[:-1])
        
        return waveform
    
    def _resample(
        self, 
        waveform: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return waveform
        
        try:
            import librosa
            return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation fallback
            duration = len(waveform) / orig_sr
            new_length = int(duration * target_sr)
            indices = np.linspace(0, len(waveform) - 1, new_length)
            return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)
    
    def generate_waveform_image(
        self, 
        waveform: np.ndarray, 
        sample_rate: int,
        figsize: Tuple[int, int] = (10, 3)
    ) -> bytes:
        """
        Generate waveform visualization as PNG image bytes.
        
        Returns:
            PNG image as bytes
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Time axis
        time = np.arange(len(waveform)) / sample_rate
        
        # Plot waveform
        ax.plot(time, waveform, color='#2196F3', linewidth=0.5)
        ax.fill_between(time, waveform, alpha=0.3, color='#2196F3')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time[-1] if len(time) > 0 else 1)
        ax.set_ylim(-1.1, 1.1)
        
        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf.read()
    
    def generate_spectrogram_image(
        self, 
        waveform: np.ndarray, 
        sample_rate: int,
        figsize: Tuple[int, int] = (10, 4)
    ) -> bytes:
        """
        Generate spectrogram visualization as PNG image bytes.
        
        Returns:
            PNG image as bytes
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute spectrogram
        n_fft = 2048
        hop_length = 512
        
        # STFT
        n_frames = 1 + (len(waveform) - n_fft) // hop_length
        if n_frames < 1:
            n_frames = 1
        
        spectrogram = np.zeros((n_fft // 2 + 1, n_frames))
        
        for i in range(n_frames):
            start = i * hop_length
            frame = waveform[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            
            # Apply Hann window
            window = np.hanning(n_fft)
            frame = frame * window
            
            # FFT
            fft = np.fft.rfft(frame)
            spectrogram[:, i] = np.abs(fft)
        
        # Convert to dB
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        # Plot
        im = ax.imshow(
            spectrogram_db,
            aspect='auto',
            origin='lower',
            extent=[0, len(waveform) / sample_rate, 0, sample_rate / 2],
            cmap='magma'
        )
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        plt.colorbar(im, ax=ax, label='dB')
        
        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf.read()
