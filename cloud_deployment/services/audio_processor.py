#!/usr/bin/env python3
"""
Audio Processing Service for Cloud Deployment

Handles audio file loading, preprocessing, and format conversion.
"""

import io
import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processing utilities for the cloud API.
    Handles file loading, resampling, and preprocessing.
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
    
    def load_audio(self, audio_bytes: bytes, 
                   original_format: str = "wav") -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes.
        
        Args:
            audio_bytes: Raw audio file bytes
            original_format: File format (wav, mp3, etc.)
        
        Returns:
            Tuple of (waveform as numpy array, sample rate)
        """
        try:
            import soundfile as sf
            
            # Load from bytes buffer
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = sf.read(audio_buffer)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                waveform = self._resample(waveform, sample_rate, self.target_sample_rate)
                sample_rate = self.target_sample_rate
            
            return waveform.astype(np.float32), sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise ValueError(f"Could not process audio file: {e}")
    
    def load_audio_from_base64(self, base64_data: str) -> Tuple[np.ndarray, int]:
        """Load audio from base64 encoded string"""
        import base64
        
        audio_bytes = base64.b64decode(base64_data)
        return self.load_audio(audio_bytes)
    
    def _resample(self, waveform: np.ndarray, orig_sr: int, 
                  target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            import librosa
            return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback using scipy
            from scipy import signal
            duration = len(waveform) / orig_sr
            target_length = int(duration * target_sr)
            return signal.resample(waveform, target_length)
    
    def preprocess(self, waveform: np.ndarray, 
                   normalize: bool = True,
                   trim_silence: bool = True,
                   min_duration: float = 0.5,
                   max_duration: float = 10.0) -> np.ndarray:
        """
        Preprocess audio waveform.
        
        Args:
            waveform: Input waveform
            normalize: Whether to normalize amplitude
            trim_silence: Whether to trim leading/trailing silence
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
        
        Returns:
            Preprocessed waveform
        """
        sr = self.target_sample_rate
        
        # Trim silence
        if trim_silence:
            try:
                import librosa
                waveform, _ = librosa.effects.trim(waveform, top_db=30)
            except:
                pass
        
        # Enforce minimum duration
        min_samples = int(min_duration * sr)
        if len(waveform) < min_samples:
            waveform = np.pad(waveform, (0, min_samples - len(waveform)))
        
        # Enforce maximum duration
        max_samples = int(max_duration * sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        # Normalize
        if normalize:
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
        
        return waveform
    
    def get_audio_info(self, waveform: np.ndarray, 
                       sample_rate: int) -> dict:
        """Get basic audio information"""
        duration = len(waveform) / sample_rate
        
        return {
            "duration_seconds": round(duration, 3),
            "sample_rate": sample_rate,
            "num_samples": len(waveform),
            "channels": 1,
            "dtype": str(waveform.dtype),
            "min_amplitude": float(np.min(waveform)),
            "max_amplitude": float(np.max(waveform)),
            "rms_energy": float(np.sqrt(np.mean(waveform**2)))
        }
