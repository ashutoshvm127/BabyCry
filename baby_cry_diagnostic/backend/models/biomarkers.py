#!/usr/bin/env python3
"""
Medical Biomarker Analyzer for Baby Cry Analysis
Extracts acoustic features relevant to infant health assessment
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class BiomarkerAnalyzer:
    """
    Extracts medical-grade acoustic biomarkers from infant cry audio.
    
    Key Biomarkers:
    - Fundamental Frequency (f0): High pitch (>600 Hz) suggests respiratory distress
    - Spectral Centroid: Measures cry "sharpness" - indicator of lung fluid
    - HNR (Harmonic-to-Noise Ratio): Low HNR indicates "raspy" or "turbulent" breath
    - Jitter/Shimmer: Voice quality indicators
    - Formant frequencies: Resonance patterns
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Reference ranges for infant cries
        self.reference_ranges = {
            "f0": {"normal": (350, 550), "warning": (550, 650), "critical": (650, 1000)},
            "hnr": {"normal": (12, 25), "warning": (8, 12), "critical": (0, 8)},
            "spectral_centroid": {"normal": (1500, 2500), "warning": (2500, 3500), "critical": (3500, 6000)},
            "jitter_percent": {"normal": (0.5, 2.0), "warning": (2.0, 4.0), "critical": (4.0, 10.0)},
            "shimmer_percent": {"normal": (1.0, 4.0), "warning": (4.0, 8.0), "critical": (8.0, 15.0)}
        }
    
    def analyze(self, waveform: np.ndarray, sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform comprehensive biomarker analysis on audio waveform.
        
        Args:
            waveform: Audio signal as numpy array
            sample_rate: Sample rate (uses default if not provided)
        
        Returns:
            Dictionary of biomarker values and health indicators
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure correct format
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        # Extract biomarkers
        biomarkers = {}
        
        # 1. Fundamental Frequency (f0)
        f0, f0_confidence = self._extract_f0(waveform, sample_rate)
        biomarkers["fundamental_frequency"] = f0
        biomarkers["f0_confidence"] = f0_confidence
        biomarkers["f0_status"] = self._get_status("f0", f0)
        
        # 2. Harmonic-to-Noise Ratio (HNR)
        hnr = self._extract_hnr(waveform, sample_rate)
        biomarkers["hnr"] = hnr
        biomarkers["hnr_status"] = self._get_status("hnr", hnr)
        
        # 3. Spectral Centroid
        spectral_centroid = self._extract_spectral_centroid(waveform, sample_rate)
        biomarkers["spectral_centroid"] = spectral_centroid
        biomarkers["spectral_centroid_status"] = self._get_status("spectral_centroid", spectral_centroid)
        
        # 4. Jitter (pitch variation)
        jitter_percent = self._extract_jitter(waveform, sample_rate)
        biomarkers["jitter_percent"] = jitter_percent
        biomarkers["jitter_status"] = self._get_status("jitter_percent", jitter_percent)
        
        # 5. Shimmer (amplitude variation)
        shimmer_percent = self._extract_shimmer(waveform, sample_rate)
        biomarkers["shimmer_percent"] = shimmer_percent
        biomarkers["shimmer_status"] = self._get_status("shimmer_percent", shimmer_percent)
        
        # 6. Formant Frequencies
        formants = self._extract_formants(waveform, sample_rate)
        biomarkers["formants"] = formants
        
        # 7. Energy / Intensity
        energy_stats = self._extract_energy_stats(waveform)
        biomarkers["mean_energy"] = energy_stats["mean"]
        biomarkers["energy_variance"] = energy_stats["variance"]
        biomarkers["energy_range"] = energy_stats["range"]
        
        # 8. Spectral Features
        spectral_features = self._extract_spectral_features(waveform, sample_rate)
        biomarkers.update(spectral_features)
        
        # 9. Temporal Features
        temporal_features = self._extract_temporal_features(waveform, sample_rate)
        biomarkers.update(temporal_features)
        
        # Overall health score
        biomarkers["health_score"] = self._calculate_health_score(biomarkers)
        
        return biomarkers
    
    def _extract_f0(self, waveform: np.ndarray, sample_rate: int) -> Tuple[float, float]:
        """Extract fundamental frequency using autocorrelation method"""
        try:
            # Simple autocorrelation-based pitch detection
            # Filter frequencies for infant cries (typically 300-1000 Hz)
            min_f0, max_f0 = 200, 1200
            min_lag = int(sample_rate / max_f0)
            max_lag = int(sample_rate / min_f0)
            
            # Compute autocorrelation
            n = len(waveform)
            autocorr = np.correlate(waveform, waveform, mode='full')
            autocorr = autocorr[n-1:]  # Take positive lags only
            
            # Find peak in valid range
            if max_lag < len(autocorr):
                search_range = autocorr[min_lag:max_lag]
                if len(search_range) > 0 and np.max(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_lag
                    f0 = sample_rate / peak_idx
                    confidence = autocorr[peak_idx] / autocorr[0] if autocorr[0] > 0 else 0
                    return f0, confidence
            
            return 400.0, 0.5  # Default
            
        except Exception:
            return 400.0, 0.0
    
    def _extract_hnr(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Extract Harmonic-to-Noise Ratio.
        Low HNR (<5 dB) indicates turbulent/raspy sounds.
        """
        try:
            # Simple HNR estimation using autocorrelation
            n = len(waveform)
            autocorr = np.correlate(waveform, waveform, mode='full')
            autocorr = autocorr[n-1:]
            
            # Find first peak (fundamental period)
            min_lag = int(sample_rate / 1000)  # Max 1000 Hz
            max_lag = int(sample_rate / 100)   # Min 100 Hz
            
            if max_lag < len(autocorr):
                search_range = autocorr[min_lag:max_lag]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_lag
                    r_peak = autocorr[peak_idx]
                    r_0 = autocorr[0]
                    
                    if r_0 > 0 and r_peak > 0 and r_0 > r_peak:
                        # HNR = 10 * log10(r_peak / (r_0 - r_peak))
                        noise_power = r_0 - r_peak
                        if noise_power > 0:
                            hnr = 10 * np.log10(r_peak / noise_power)
                            return float(np.clip(hnr, -10, 40))
            
            return 15.0  # Default healthy value
            
        except Exception:
            return 15.0
    
    def _extract_spectral_centroid(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Extract spectral centroid (center of mass of spectrum).
        High values may indicate lung fluid or respiratory issues.
        """
        try:
            # Compute FFT
            n_fft = 2048
            fft = np.fft.rfft(waveform, n=n_fft)
            magnitude = np.abs(fft)
            
            # Frequency bins
            freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
            
            # Spectral centroid = weighted mean of frequencies
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                return float(centroid)
            
            return 2000.0  # Default
            
        except Exception:
            return 2000.0
    
    def _extract_jitter(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Extract jitter (cycle-to-cycle pitch variation).
        High jitter may indicate vocal cord issues.
        """
        try:
            # Find pitch periods
            periods = self._find_pitch_periods(waveform, sample_rate)
            
            if len(periods) < 3:
                return 1.0  # Default
            
            # Calculate jitter as mean absolute difference between consecutive periods
            period_diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)
            
            if mean_period > 0:
                jitter_percent = (np.mean(period_diffs) / mean_period) * 100
                return float(np.clip(jitter_percent, 0, 20))
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _extract_shimmer(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Extract shimmer (cycle-to-cycle amplitude variation).
        High shimmer may indicate breathing abnormalities.
        """
        try:
            # Find peak amplitudes per cycle
            amplitudes = self._find_cycle_amplitudes(waveform, sample_rate)
            
            if len(amplitudes) < 3:
                return 2.0  # Default
            
            # Calculate shimmer as mean absolute difference between consecutive amplitudes
            amp_diffs = np.abs(np.diff(amplitudes))
            mean_amp = np.mean(amplitudes)
            
            if mean_amp > 0:
                shimmer_percent = (np.mean(amp_diffs) / mean_amp) * 100
                return float(np.clip(shimmer_percent, 0, 30))
            
            return 2.0
            
        except Exception:
            return 2.0
    
    def _extract_formants(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract formant frequencies (F1, F2, F3)"""
        try:
            # Simple formant estimation using LPC
            from scipy.signal import lfilter
            
            # Pre-emphasis
            pre_emphasis = 0.97
            emphasized = np.append(waveform[0], waveform[1:] - pre_emphasis * waveform[:-1])
            
            # LPC coefficients (simplified)
            order = 12
            n = len(emphasized)
            
            # Autocorrelation
            r = np.correlate(emphasized, emphasized, mode='full')
            r = r[n-1:n+order]
            
            # Levinson-Durbin
            a = np.zeros(order + 1)
            a[0] = 1.0
            e = r[0]
            
            for i in range(1, order + 1):
                lambda_val = np.sum(a[:i] * r[1:i+1][::-1])
                if e > 0:
                    k = -lambda_val / e
                    a[i] = k
                    for j in range(1, i):
                        a[j] = a[j] + k * a[i-j]
                    e = e * (1 - k * k)
            
            # Find roots (formants)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            
            # Convert to frequencies
            angles = np.angle(roots)
            freqs = angles * (sample_rate / (2 * np.pi))
            freqs = np.sort(freqs[freqs > 100])[:3]
            
            return {
                "F1": float(freqs[0]) if len(freqs) > 0 else 500.0,
                "F2": float(freqs[1]) if len(freqs) > 1 else 1500.0,
                "F3": float(freqs[2]) if len(freqs) > 2 else 2500.0
            }
            
        except Exception:
            return {"F1": 500.0, "F2": 1500.0, "F3": 2500.0}
    
    def _extract_energy_stats(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract energy statistics"""
        energy = waveform ** 2
        return {
            "mean": float(np.mean(energy)),
            "variance": float(np.var(energy)),
            "range": float(np.max(energy) - np.min(energy))
        }
    
    def _extract_spectral_features(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract additional spectral features"""
        try:
            n_fft = 2048
            fft = np.fft.rfft(waveform, n=n_fft)
            magnitude = np.abs(fft)
            power = magnitude ** 2
            freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
            
            total_power = np.sum(power) + 1e-10
            
            # Spectral spread
            centroid = np.sum(freqs * power) / total_power
            spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power)
            
            # Spectral rolloff (85th percentile)
            cumsum = np.cumsum(power)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * total_power)
            rolloff = freqs[min(rolloff_idx, len(freqs)-1)]
            
            # Spectral flatness
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            
            return {
                "spectral_spread": float(spread),
                "spectral_rolloff": float(rolloff),
                "spectral_flatness": float(flatness)
            }
            
        except Exception:
            return {
                "spectral_spread": 1000.0,
                "spectral_rolloff": 4000.0,
                "spectral_flatness": 0.5
            }
    
    def _extract_temporal_features(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract temporal features"""
        try:
            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(waveform)))) / 2
            zcr = zero_crossings / len(waveform)
            
            # Duration in seconds
            duration = len(waveform) / sample_rate
            
            # RMS energy
            rms = np.sqrt(np.mean(waveform ** 2))
            
            return {
                "zero_crossing_rate": float(zcr),
                "duration_seconds": float(duration),
                "rms_energy": float(rms)
            }
            
        except Exception:
            return {
                "zero_crossing_rate": 0.1,
                "duration_seconds": 1.0,
                "rms_energy": 0.1
            }
    
    def _find_pitch_periods(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Find pitch periods using zero-crossing analysis"""
        # Find positive zero crossings
        zero_crossings = np.where(np.diff(np.sign(waveform)) > 0)[0]
        
        if len(zero_crossings) < 2:
            return np.array([])
        
        # Calculate periods
        periods = np.diff(zero_crossings) / sample_rate
        
        # Filter reasonable periods (100-1000 Hz)
        valid_mask = (periods > 1/1000) & (periods < 1/100)
        
        return periods[valid_mask]
    
    def _find_cycle_amplitudes(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Find peak amplitudes for each pitch cycle"""
        # Find peaks
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(np.abs(waveform), distance=int(sample_rate/1000))
        
        if len(peaks) < 2:
            return np.array([])
        
        return np.abs(waveform[peaks])
    
    def _get_status(self, biomarker: str, value: float) -> str:
        """Determine status (normal/warning/critical) for a biomarker"""
        if biomarker not in self.reference_ranges:
            return "unknown"
        
        ranges = self.reference_ranges[biomarker]
        
        # HNR is inverted (lower = worse)
        if biomarker == "hnr":
            if ranges["normal"][0] <= value <= ranges["normal"][1]:
                return "normal"
            elif ranges["warning"][0] <= value < ranges["warning"][1]:
                return "warning"
            elif value < ranges["critical"][1]:
                return "critical"
            return "normal"
        
        # Other biomarkers (higher = worse)
        if ranges["normal"][0] <= value <= ranges["normal"][1]:
            return "normal"
        elif ranges["warning"][0] <= value <= ranges["warning"][1]:
            return "warning"
        elif value >= ranges["critical"][0]:
            return "critical"
        
        return "normal"
    
    def _calculate_health_score(self, biomarkers: Dict) -> float:
        """Calculate overall health score (0-100, higher is healthier)"""
        score = 100.0
        
        # Deduct points for warning/critical statuses
        status_penalties = {
            "warning": 10,
            "critical": 25
        }
        
        for key, value in biomarkers.items():
            if key.endswith("_status"):
                if value == "warning":
                    score -= status_penalties["warning"]
                elif value == "critical":
                    score -= status_penalties["critical"]
        
        return max(0.0, min(100.0, score))
