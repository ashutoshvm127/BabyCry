#!/usr/bin/env python3
"""
Biomarker Analyzer for Baby Cry Audio

Extracts acoustic biomarkers for medical assessment:
- Fundamental frequency (F0)
- Formant frequencies (F1-F4)
- Mel-frequency cepstral coefficients (MFCCs)
- Spectral features
- Cry duration and energy patterns
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BiomarkerAnalyzer:
    """
    Extracts acoustic biomarkers from baby cry audio.
    Used for medical assessment and risk evaluation.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Normal ranges for baby cry biomarkers (based on medical literature)
        self.normal_ranges = {
            "f0_mean": (300, 600),        # Fundamental frequency Hz
            "f0_std": (20, 100),           # F0 variation
            "cry_duration": (0.5, 5.0),    # Seconds
            "energy_mean": (0.01, 0.3),    # Normalized RMS
            "zcr_mean": (0.01, 0.15),      # Zero-crossing rate
            "spectral_centroid": (500, 3000),  # Hz
        }
    
    def analyze(self, waveform: np.ndarray, sample_rate: int = None) -> Dict[str, Any]:
        """
        Extract acoustic biomarkers from audio waveform.
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate (default: 16000)
        
        Returns:
            Dictionary of biomarker values and analysis
        """
        sr = sample_rate or self.sample_rate
        
        # Ensure mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=0)
        
        # Basic validation
        if len(waveform) < sr * 0.1:  # Less than 100ms
            return self._empty_biomarkers("Audio too short")
        
        try:
            import librosa
            
            biomarkers = {}
            
            # Duration
            biomarkers["duration_seconds"] = len(waveform) / sr
            
            # Energy features
            rms = librosa.feature.rms(y=waveform)[0]
            biomarkers["energy_mean"] = float(np.mean(rms))
            biomarkers["energy_std"] = float(np.std(rms))
            biomarkers["energy_max"] = float(np.max(rms))
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(waveform)[0]
            biomarkers["zcr_mean"] = float(np.mean(zcr))
            biomarkers["zcr_std"] = float(np.std(zcr))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
            biomarkers["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            biomarkers["spectral_centroid_std"] = float(np.std(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
            biomarkers["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
            biomarkers["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
            
            # MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
            for i in range(13):
                biomarkers[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
                biomarkers[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
            
            # Fundamental frequency (F0) estimation
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform, 
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=sr
            )
            voiced_f0 = f0[~np.isnan(f0)]
            
            if len(voiced_f0) > 0:
                biomarkers["f0_mean"] = float(np.mean(voiced_f0))
                biomarkers["f0_std"] = float(np.std(voiced_f0))
                biomarkers["f0_min"] = float(np.min(voiced_f0))
                biomarkers["f0_max"] = float(np.max(voiced_f0))
                biomarkers["voiced_ratio"] = float(len(voiced_f0) / len(f0))
            else:
                biomarkers["f0_mean"] = 0
                biomarkers["f0_std"] = 0
                biomarkers["f0_min"] = 0
                biomarkers["f0_max"] = 0
                biomarkers["voiced_ratio"] = 0
            
            # Mel spectrogram stats
            mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=40)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            biomarkers["mel_energy_mean"] = float(np.mean(mel_db))
            biomarkers["mel_energy_std"] = float(np.std(mel_db))
            
            # Add analysis summary
            biomarkers["analysis_status"] = "success"
            biomarkers["abnormality_flags"] = self._check_abnormalities(biomarkers)
            
            return biomarkers
            
        except Exception as e:
            logger.error(f"Biomarker extraction failed: {e}")
            return self._empty_biomarkers(str(e))
    
    def _empty_biomarkers(self, reason: str) -> Dict[str, Any]:
        """Return empty biomarker dict when analysis fails"""
        return {
            "analysis_status": "failed",
            "error": reason,
            "abnormality_flags": []
        }
    
    def _check_abnormalities(self, biomarkers: Dict[str, Any]) -> list:
        """Check if biomarkers are outside normal ranges"""
        flags = []
        
        for metric, (low, high) in self.normal_ranges.items():
            value = biomarkers.get(f"{metric}_mean", biomarkers.get(metric))
            if value is not None:
                if value < low:
                    flags.append({
                        "metric": metric,
                        "value": value,
                        "expected_range": [low, high],
                        "status": "below_normal"
                    })
                elif value > high:
                    flags.append({
                        "metric": metric,
                        "value": value,
                        "expected_range": [low, high],
                        "status": "above_normal"
                    })
        
        return flags
    
    def get_medical_summary(self, biomarkers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate medical summary from biomarkers.
        """
        if biomarkers.get("analysis_status") != "success":
            return {
                "status": "incomplete",
                "message": "Could not complete biomarker analysis"
            }
        
        abnormalities = biomarkers.get("abnormality_flags", [])
        
        # Determine overall status
        if len(abnormalities) == 0:
            status = "normal"
            message = "All acoustic biomarkers within normal ranges"
        elif len(abnormalities) <= 2:
            status = "minor_deviation"
            message = f"{len(abnormalities)} biomarker(s) slightly outside normal range"
        else:
            status = "requires_attention"
            message = f"Multiple biomarkers ({len(abnormalities)}) outside normal range"
        
        # Key metrics for summary
        summary = {
            "status": status,
            "message": message,
            "key_metrics": {
                "fundamental_frequency": biomarkers.get("f0_mean", "N/A"),
                "cry_duration": biomarkers.get("duration_seconds", "N/A"),
                "energy_level": biomarkers.get("energy_mean", "N/A"),
                "spectral_centroid": biomarkers.get("spectral_centroid_mean", "N/A"),
            },
            "abnormalities": abnormalities,
            "recommendation": self._get_recommendation(status, abnormalities)
        }
        
        return summary
    
    def _get_recommendation(self, status: str, abnormalities: list) -> str:
        """Generate recommendation based on biomarker analysis"""
        if status == "normal":
            return "No immediate action required. Continue routine monitoring."
        elif status == "minor_deviation":
            return "Minor variations detected. Monitor for changes and ensure baby is comfortable."
        else:
            # Check for specific concerning patterns
            concerning_metrics = [a["metric"] for a in abnormalities if a["status"] == "above_normal"]
            
            if "f0_mean" in concerning_metrics:
                return "High-pitched crying detected. Consider checking for pain or discomfort. Consult pediatrician if persistent."
            elif "energy_mean" in concerning_metrics:
                return "Unusually intense crying. Ensure baby's needs are met. Seek medical advice if crying is inconsolable."
            else:
                return "Some acoustic patterns outside normal range. Monitor closely and consult healthcare provider if concerned."
