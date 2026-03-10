#!/usr/bin/env python3
"""
Baby Cry Diagnostic System - FastAPI Backend
Ensemble AI Pipeline: DistilHuBERT → AST → YAMNet

Supports both RPi5 (I2S audio) and Desktop modes via system_config.json
"""

import os
import io
import sys
import json
import uuid
import base64
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import AI models and utilities
from models.ensemble import EnsembleModel
from models.biomarkers import BiomarkerAnalyzer
from services.pdf_generator import MedicalReportGenerator
from services.audio_processor import AudioProcessor

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Try to load system config
try:
    from config import get_config
    SYSTEM_CONFIG = get_config()
    IS_RPI5_MODE = SYSTEM_CONFIG.is_rpi5_mode
except ImportError:
    # Fallback: load directly from JSON
    config_path = Path(__file__).parent.parent / "system_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            _cfg = json.load(f)
        IS_RPI5_MODE = _cfg.get("is_rpi5_mode", False)
    else:
        IS_RPI5_MODE = False
    SYSTEM_CONFIG = None

# Global instances
ensemble_model: Optional[EnsembleModel] = None
biomarker_analyzer: Optional[BiomarkerAnalyzer] = None
audio_processor: Optional[AudioProcessor] = None
pdf_generator: Optional[MedicalReportGenerator] = None

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - initialize models on startup"""
    global ensemble_model, biomarker_analyzer, audio_processor, pdf_generator
    
    print("=" * 70)
    print("BABY CRY DIAGNOSTIC SYSTEM - INITIALIZING")
    print(f"Mode: {'RPi5 (I2S)' if IS_RPI5_MODE else 'Desktop/Windows'}")
    print("=" * 70)
    
    # Initialize components
    print("[1/4] Loading Ensemble AI Model...")
    ensemble_model = EnsembleModel()
    await ensemble_model.initialize()
    print("  [OK] Ensemble model ready")
    
    print("[2/4] Loading Biomarker Analyzer...")
    biomarker_analyzer = BiomarkerAnalyzer()
    print("  [OK] Biomarker analyzer ready")
    
    print("[3/4] Initializing Audio Processor...")
    audio_processor = AudioProcessor()
    print("  [OK] Audio processor ready")
    
    print("[4/4] Initializing PDF Generator...")
    pdf_generator = MedicalReportGenerator()
    print("  [OK] PDF generator ready")
    
    print("\n[OK] All systems initialized. Server ready.\n")
    
    yield
    
    # Cleanup on shutdown
    print("\n[!] Shutting down...")
    if ensemble_model:
        await ensemble_model.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Baby Cry Diagnostic API",
    description="Cloud-based AI system for infant cry analysis and medical biomarker detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class DiagnosisResult(BaseModel):
    """Diagnosis result from AI analysis"""
    id: str = Field(description="Unique diagnosis ID")
    timestamp: str = Field(description="ISO format timestamp")
    
    # Classification results
    primary_classification: str = Field(description="Primary cry classification")
    confidence: float = Field(description="Confidence score (0-1)")
    model_used: str = Field(description="Which model in ensemble was used")
    
    # Risk assessment
    risk_level: str = Field(description="GREEN/YELLOW/RED")
    risk_score: float = Field(description="Risk score (0-100)")
    
    # Medical biomarkers
    biomarkers: Dict[str, Any] = Field(description="Acoustic biomarkers")
    
    # Recommendations
    recommendations: List[str] = Field(description="Medical recommendations")


class AudioUploadResponse(BaseModel):
    """Response for audio upload"""
    success: bool
    diagnosis: Optional[DiagnosisResult] = None
    error: Optional[str] = None


class StreamStatus(BaseModel):
    """Real-time stream status"""
    device_id: str
    is_active: bool
    last_audio_timestamp: Optional[str] = None
    current_risk_level: str = "GREEN"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Baby Cry Diagnostic API",
        "version": "1.0.0",
        "status": "online",
        "mode": "RPi5" if IS_RPI5_MODE else "Desktop",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "stream": "/ws/stream",
            "report": "/api/v1/report/{diagnosis_id}",
            "config": "/api/v1/config",
            "health": "/health"
        }
    }


@app.get("/api/v1/config")
async def get_system_config():
    """Get current system configuration"""
    config_path = Path(__file__).parent.parent / "system_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"is_rpi5_mode": False}
    
    return {
        "is_rpi5_mode": config.get("is_rpi5_mode", False),
        "mode_name": "RPi5 (INMP441 I2S)" if config.get("is_rpi5_mode") else "Desktop (Standard Mic)",
        "config": config
    }


@app.post("/api/v1/config/mode")
async def set_system_mode(rpi5_mode: bool):
    """Set system mode (RPi5 or Desktop)"""
    config_path = Path(__file__).parent.parent / "system_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    config["is_rpi5_mode"] = rpi5_mode
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return {
        "success": True,
        "is_rpi5_mode": rpi5_mode,
        "mode_name": "RPi5 (INMP441 I2S)" if rpi5_mode else "Desktop (Standard Mic)",
        "message": "Configuration updated. Restart server for changes to take effect."
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "ensemble": ensemble_model is not None,
            "biomarkers": biomarker_analyzer is not None,
            "audio": audio_processor is not None,
            "pdf": pdf_generator is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/analyze", response_model=AudioUploadResponse)
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, or raw PCM)")
):
    """
    Analyze uploaded audio for cry classification and biomarkers.
    
    Returns diagnosis with:
    - Cry classification (hungry, pain, sleepy, etc.)
    - Risk level (GREEN/YELLOW/RED)
    - Medical biomarkers (f0, spectral centroid, HNR)
    - Recommendations
    """
    try:
        # Read audio data
        audio_bytes = await audio_file.read()
        
        # Process audio
        audio_data = await audio_processor.process_upload(
            audio_bytes, 
            audio_file.filename
        )
        
        # Run ensemble classification with AUTO-DETECTION
        # This runs both cry and pulmonary classifiers and picks the best match
        classification = await ensemble_model.predict_auto(audio_data["waveform"])
        
        # Extract biomarkers
        biomarkers = biomarker_analyzer.analyze(
            audio_data["waveform"],
            audio_data["sample_rate"]
        )
        
        # Calculate risk level
        risk_level, risk_score = calculate_risk_level(
            classification, 
            biomarkers
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(
            classification,
            biomarkers,
            risk_level
        )
        
        # Convert numpy types to native Python types for JSON serialization
        classification = convert_numpy_types(classification)
        biomarkers = convert_numpy_types(biomarkers)
        risk_score = convert_numpy_types(risk_score)
        
        # Create diagnosis result
        diagnosis_id = str(uuid.uuid4())
        diagnosis = DiagnosisResult(
            id=diagnosis_id,
            timestamp=datetime.now().isoformat(),
            primary_classification=classification["label"],
            confidence=classification["confidence"],
            model_used=classification["model"],
            risk_level=risk_level,
            risk_score=risk_score,
            biomarkers=biomarkers,
            recommendations=recommendations
        )
        
        # Store for PDF generation (in production, use database)
        _store_diagnosis(diagnosis_id, diagnosis, audio_data)
        
        return AudioUploadResponse(success=True, diagnosis=diagnosis)
        
    except Exception as e:
        return AudioUploadResponse(success=False, error=str(e))


@app.get("/api/v1/report/{diagnosis_id}")
async def get_medical_report(diagnosis_id: str):
    """
    Generate and download medical PDF report for a diagnosis.
    
    Report includes:
    - Patient/session information
    - Audio waveform visualization
    - AI diagnosis results
    - Medical biomarkers
    - Recommendations
    """
    try:
        # Retrieve stored diagnosis
        diagnosis_data = _get_diagnosis(diagnosis_id)
        if not diagnosis_data:
            raise HTTPException(status_code=404, detail="Diagnosis not found")
        
        # Generate PDF
        pdf_path = await pdf_generator.generate_report(
            diagnosis_data["diagnosis"],
            diagnosis_data["audio_data"]
        )
        
        return FileResponse(
            path=pdf_path,
            filename=f"infant_health_report_{diagnosis_id[:8]}.pdf",
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming from Raspberry Pi.
    
    Accepts:
    - Raw PCM audio chunks (24-bit I2S from INMP441)
    - JSON control messages
    
    Sends:
    - Real-time classification results
    - Risk level updates
    """
    await websocket.accept()
    active_connections.append(websocket)
    device_id = str(uuid.uuid4())[:8]
    
    print(f"[WS] Client connected: {device_id}")
    
    try:
        # Audio buffer for accumulating samples
        audio_buffer = []
        buffer_duration = 3.0  # seconds
        sample_rate = 16000
        samples_needed = int(buffer_duration * sample_rate)
        
        while True:
            # Receive data
            data = await websocket.receive()
            
            if "bytes" in data:
                # Raw audio data
                audio_chunk = np.frombuffer(data["bytes"], dtype=np.int16)
                audio_buffer.extend(audio_chunk.tolist())
                
                # Process when buffer is full
                if len(audio_buffer) >= samples_needed:
                    # Convert to numpy
                    waveform = np.array(audio_buffer[:samples_needed], dtype=np.float32)
                    waveform = waveform / 32768.0  # Normalize
                    
                    # Clear processed samples
                    audio_buffer = audio_buffer[samples_needed:]
                    
                    # Run analysis with AUTO-DETECTION
                    classification = await ensemble_model.predict_auto(waveform)
                    biomarkers = biomarker_analyzer.analyze(waveform, sample_rate)
                    risk_level, risk_score = calculate_risk_level(classification, biomarkers)
                    
                    # Send result
                    await websocket.send_json({
                        "type": "analysis",
                        "device_id": device_id,
                        "timestamp": datetime.now().isoformat(),
                        "classification": classification["label"],
                        "confidence": classification["confidence"],
                        "risk_level": risk_level,
                        "risk_score": risk_score,
                        "biomarkers": biomarkers
                    })
            
            elif "text" in data:
                # JSON control message
                msg = json.loads(data["text"])
                
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif msg.get("type") == "config":
                    # Update configuration
                    if "buffer_duration" in msg:
                        buffer_duration = msg["buffer_duration"]
                        samples_needed = int(buffer_duration * sample_rate)
                    
                    await websocket.send_json({
                        "type": "config_ack",
                        "buffer_duration": buffer_duration
                    })
    
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {device_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_risk_level(classification: Dict, biomarkers: Dict) -> tuple:
    """
    Calculate risk level based on classification and biomarkers.
    
    Risk Factors:
    - High f0 (>600 Hz) = respiratory distress indicator
    - Low HNR (<5 dB) = turbulent/raspy breath sounds
    - High spectral centroid = lung fluid indicator
    - Pain/distress classification = immediate concern
    - Respiratory sounds (wheeze, rhonchi, crackle) = HIGH risk
    """
    risk_score = 0.0
    
    # Classification-based risk - matches trained class names
    classification_risks = {
        # Baby cry types (12 classes from training)
        "pain": 40,
        "pathological": 50,  # distress_cry maps to this
        "hungry": 10,
        "sleepy": 5,
        "discomfort": 20,
        "cold_hot": 15,
        "tired": 10,
        "normal": 0,
        "belly_pain": 35,
        "burping": 5,
        "scared": 25,
        "lonely": 10,
        # RESPIRATORY SOUNDS - HIGH RISK (8 classes from training)
        "wheeze": 70,           # Airway obstruction
        "rhonchi": 65,          # Secretions in airways
        "crackle": 60,          # Fluid in lungs
        "stridor": 80,          # Upper airway obstruction - URGENT
        "bronchiolitis": 65,    # RSV infection
        "pneumonia": 75,        # Lung infection
        "asthma": 60,           # Chronic airway disease
    }
    
    label = classification["label"].lower()
    
    # Try exact match first
    if label in classification_risks:
        risk_score += classification_risks[label] * classification["confidence"]
    else:
        # Fallback to partial match
        for key, value in classification_risks.items():
            if key in label or label in key:
                risk_score += value * classification["confidence"]
                break
    
    # Biomarker-based risk
    f0 = biomarkers.get("fundamental_frequency", 400)
    hnr = biomarkers.get("hnr", 15)
    spectral_centroid = biomarkers.get("spectral_centroid", 2000)
    
    # High pitch indicates respiratory distress
    if f0 > 600:
        risk_score += 25 * min((f0 - 600) / 200, 1.0)
    elif f0 > 500:
        risk_score += 10
    
    # Low HNR indicates turbulent breath
    if hnr < 5:
        risk_score += 20
    elif hnr < 10:
        risk_score += 10
    
    # High spectral centroid indicates potential lung issues
    if spectral_centroid > 4000:
        risk_score += 15
    elif spectral_centroid > 3000:
        risk_score += 8
    
    # Normalize
    risk_score = min(100, max(0, risk_score))
    
    # Determine level
    if risk_score >= 60:
        risk_level = "RED"
    elif risk_score >= 30:
        risk_level = "YELLOW"
    else:
        risk_level = "GREEN"
    
    return risk_level, risk_score


def generate_recommendations(
    classification: Dict, 
    biomarkers: Dict, 
    risk_level: str
) -> List[str]:
    """Generate medical recommendations based on analysis"""
    recommendations = []
    
    label = classification["label"].lower()
    
    # Classification-based recommendations for 12 cry classes + 8 pulmonary classes
    if "pain" in label and "belly" not in label:
        recommendations.extend([
            "Check for signs of discomfort or illness",
            "Monitor for fever or unusual symptoms",
            "If crying persists, consult a pediatrician"
        ])
    elif "belly_pain" in label:
        recommendations.extend([
            "Possible colic or gas discomfort",
            "Try gentle tummy massage or bicycle legs",
            "Check for bloating, consider simethicone drops",
            "Consult pediatrician if persists >3 hours/day"
        ])
    elif "hungry" in label:
        recommendations.extend([
            "Consider feeding the infant",
            "Track feeding schedule for patterns"
        ])
    elif "sleepy" in label:
        recommendations.extend([
            "Create a calm, quiet environment",
            "Check room temperature and lighting"
        ])
    elif "tired" in label:
        recommendations.extend([
            "Baby may be overtired - help settle to sleep",
            "Reduce stimulation, darken room",
            "Gentle rocking or swaddling may help"
        ])
    elif "pathological" in label or "distress" in label:
        recommendations.extend([
            "URGENT: Possible medical issue detected",
            "Check for physical discomfort or illness",
            "Monitor breathing patterns and skin color",
            "Seek immediate pediatric consultation"
        ])
    elif "scared" in label:
        recommendations.extend([
            "Comfort baby with gentle reassurance",
            "Hold close and speak softly",
            "Remove any startling stimuli if present"
        ])
    elif "lonely" in label:
        recommendations.extend([
            "Baby needs comfort and attention",
            "Hold and engage with baby",
            "Singing or talking may help soothe"
        ])
    elif "cold_hot" in label:
        recommendations.extend([
            "Check baby's temperature comfort",
            "Adjust clothing or room temperature",
            "Feel back of neck - should be warm not sweaty"
        ])
    elif "burping" in label:
        recommendations.extend([
            "Baby may need to burp",
            "Try gentle back patting or upright position",
            "Common after feeding"
        ])
    elif "discomfort" in label:
        recommendations.extend([
            "Check for common discomforts",
            "Diaper change, clothing adjustment",
            "Position change may help"
        ])
    # RESPIRATORY SOUND RECOMMENDATIONS
    elif "wheeze" in label:
        recommendations.extend([
            "Wheezing detected - possible airway obstruction",
            "Monitor for breathing difficulty",
            "Seek pediatric evaluation for potential asthma or bronchiolitis"
        ])
    elif "rhonchi" in label:
        recommendations.extend([
            "Rhonchi detected - secretions in airways",
            "Monitor hydration and clear nasal passages",
            "Consult pediatrician if accompanied by fever or labored breathing"
        ])
    elif "stridor" in label:
        recommendations.extend([
            "URGENT: Stridor detected - upper airway obstruction",
            "Keep child calm and upright",
            "Seek immediate medical attention"
        ])
    elif "crackle" in label or "fine_crackle" in label or "coarse_crackle" in label:
        recommendations.extend([
            "Crackles detected - possible fluid in lungs",
            "Monitor for signs of respiratory distress",
            "Pediatric evaluation recommended to rule out pneumonia"
        ])
    elif "bronchiolitis" in label:
        recommendations.extend([
            "Bronchiolitis detected - viral respiratory infection",
            "Monitor breathing rate and effort",
            "Keep baby hydrated with frequent feeds",
            "Seek medical care for breathing difficulty or dehydration"
        ])
    elif "pneumonia" in label:
        recommendations.extend([
            "URGENT: Possible pneumonia detected",
            "Monitor breathing rate, temperature, and oxygen",
            "Seek immediate medical evaluation",
            "May require antibiotics or hospitalization"
        ])
    elif "asthma" in label:
        recommendations.extend([
            "Asthma symptoms detected",
            "Monitor for wheezing and breathing difficulty",
            "Follow prescribed treatment plan if available",
            "Consult pediatric pulmonologist for assessment"
        ])
    elif "mixed" in label:
        recommendations.extend([
            "Mixed respiratory sounds detected",
            "Multiple abnormal lung sounds present",
            "Comprehensive pulmonary evaluation recommended"
        ])
    elif "normal" in label or "normal_breathing" in label:
        recommendations.extend([
            "Normal respiratory sounds detected",
            "Continue routine monitoring"
        ])
    
    # Biomarker-based recommendations
    f0 = biomarkers.get("fundamental_frequency", 400)
    hnr = biomarkers.get("hnr", 15)
    
    if f0 > 600:
        recommendations.append(
            f"Elevated cry pitch ({f0:.0f} Hz) detected - monitor respiratory health"
        )
    
    if hnr < 5:
        recommendations.append(
            f"Low harmonic-to-noise ratio ({hnr:.1f} dB) - may indicate respiratory turbulence"
        )
    
    # Risk-level recommendations
    if risk_level == "RED":
        recommendations.insert(0, "HIGH RISK: Recommend immediate pediatric consultation")
    elif risk_level == "YELLOW":
        recommendations.insert(0, "MODERATE RISK: Continue monitoring and consult if symptoms persist")
    
    return recommendations


# Simple in-memory storage (use database in production)
_diagnosis_store: Dict[str, Dict] = {}


def _store_diagnosis(diagnosis_id: str, diagnosis: DiagnosisResult, audio_data: Dict):
    """Store diagnosis for later PDF generation"""
    _diagnosis_store[diagnosis_id] = {
        "diagnosis": diagnosis.model_dump(),
        "audio_data": audio_data
    }


def _get_diagnosis(diagnosis_id: str) -> Optional[Dict]:
    """Retrieve stored diagnosis"""
    return _diagnosis_store.get(diagnosis_id)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "baby_cry_diagnostic.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
