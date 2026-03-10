#!/usr/bin/env python3
"""
Baby Cry Diagnostic System - Cloud API Server
Optimized for Render, Railway, Fly.io deployment

Ensemble AI Pipeline: DistilHuBERT → AST → YAMNet → Wav2Vec2 → WavLM → PANNs
"""

import os
import io
import sys
import json
import uuid
import base64
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

class Settings:
    """Application settings from environment variables"""
    MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/model_cache"))
    REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "/app/reports"))
    LOGS_DIR = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
    PORT = int(os.environ.get("PORT", 10000))
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    
    # Trained model paths
    TRAINED_CLASSIFIERS_DIR = Path(__file__).parent.parent / "trained_classifiers"
    AST_CRY_MODEL_DIR = Path(__file__).parent.parent / "ast_baby_cry_optimized"
    AST_RESPIRATORY_MODEL_DIR = Path(__file__).parent.parent / "ast_respiratory_optimized"

settings = Settings()

# Ensure directories exist
settings.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)


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


# ==============================================================================
# Global Model Instances
# ==============================================================================

ensemble_model = None
biomarker_analyzer = None
audio_processor = None
pdf_generator = None

# Diagnosis storage (in-memory - use Redis/DB in production)
diagnosis_store: Dict[str, Dict] = {}


# ==============================================================================
# Startup / Shutdown
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ensemble_model, biomarker_analyzer, audio_processor, pdf_generator
    
    logger.info("=" * 70)
    logger.info("BABY CRY DIAGNOSTIC SYSTEM - CLOUD API")
    logger.info(f"Device: {'GPU' if settings.USE_GPU else 'CPU'}")
    logger.info(f"Model Cache: {settings.MODEL_CACHE_DIR}")
    logger.info("=" * 70)
    
    try:
        # Import and initialize models
        from models.ensemble import EnsembleModel
        from models.biomarkers import BiomarkerAnalyzer
        from services.audio_processor import AudioProcessor
        from services.pdf_generator import MedicalReportGenerator
        
        logger.info("[1/4] Loading Ensemble AI Model...")
        ensemble_model = EnsembleModel()
        await ensemble_model.initialize()
        logger.info("       [OK] Ensemble model ready")
        
        logger.info("[2/4] Loading Biomarker Analyzer...")
        biomarker_analyzer = BiomarkerAnalyzer()
        logger.info("       [OK] Biomarker analyzer ready")
        
        logger.info("[3/4] Initializing Audio Processor...")
        audio_processor = AudioProcessor()
        logger.info("       [OK] Audio processor ready")
        
        logger.info("[4/4] Initializing PDF Generator...")
        pdf_generator = MedicalReportGenerator()
        logger.info("       [OK] PDF generator ready")
        
        logger.info("")
        logger.info("[OK] All systems initialized. Server ready.")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Continue even if models fail - health check will report status
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if ensemble_model:
        await ensemble_model.cleanup()


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Baby Cry Diagnostic API",
    description="Cloud AI system for infant cry analysis and medical biomarker detection",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic Models
# ==============================================================================

class DiagnosisResult(BaseModel):
    """Diagnosis result from AI analysis"""
    id: str = Field(description="Unique diagnosis ID")
    timestamp: str = Field(description="ISO format timestamp")
    primary_classification: str = Field(description="Primary cry classification")
    confidence: float = Field(description="Confidence score (0-1)")
    model_used: str = Field(description="Which model in ensemble was used")
    risk_level: str = Field(description="GREEN/YELLOW/RED")
    risk_score: float = Field(description="Risk score (0-100)")
    biomarkers: Dict[str, Any] = Field(description="Acoustic biomarkers")
    recommendations: List[str] = Field(description="Medical recommendations")


class AudioUploadResponse(BaseModel):
    """Response for audio upload"""
    success: bool
    diagnosis: Optional[DiagnosisResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models: Dict[str, bool]
    timestamp: str
    version: str
    environment: str


# ==============================================================================
# Helper Functions
# ==============================================================================

def calculate_risk_level(classification: Dict, biomarkers: Dict) -> tuple:
    """Calculate risk level based on classification and biomarkers"""
    risk_score = 0.0
    
    # Classification-based risk
    high_risk_labels = ["pain", "distress", "stridor", "wheeze", "crackle", "pneumonia", "asthma"]
    medium_risk_labels = ["discomfort", "belly_pain", "rhonchi", "bronchiolitis"]
    
    label = classification.get("label", "").lower()
    confidence = classification.get("confidence", 0.5)
    
    if any(risk in label for risk in high_risk_labels):
        risk_score += 60 * confidence
    elif any(risk in label for risk in medium_risk_labels):
        risk_score += 40 * confidence
    else:
        risk_score += 10 * confidence
    
    # Biomarker-based risk adjustments
    if biomarkers:
        f0 = biomarkers.get("fundamental_frequency", 0)
        hnr = biomarkers.get("harmonic_to_noise_ratio", 0)
        spectral_centroid = biomarkers.get("spectral_centroid", 0)
        
        # High f0 (>600 Hz) indicates respiratory distress
        if f0 > 600:
            risk_score += 20
        elif f0 > 450:
            risk_score += 10
        
        # Low HNR indicates turbulent breath sounds
        if hnr < 5:
            risk_score += 15
        elif hnr < 10:
            risk_score += 5
        
        # High spectral centroid indicates congestion
        if spectral_centroid > 2000:
            risk_score += 10
    
    # Normalize to 0-100
    risk_score = min(100, max(0, risk_score))
    
    # Determine risk level
    if risk_score >= 70:
        return "RED", risk_score
    elif risk_score >= 40:
        return "YELLOW", risk_score
    else:
        return "GREEN", risk_score


def generate_recommendations(classification: Dict, biomarkers: Dict, risk_level: str) -> List[str]:
    """Generate medical recommendations based on analysis"""
    recommendations = []
    
    label = classification.get("label", "").lower()
    confidence = classification.get("confidence", 0.5)
    
    if risk_level == "RED":
        recommendations.append("⚠️ URGENT: Seek immediate medical attention")
        recommendations.append("Monitor breathing rate and oxygen saturation")
    
    if "pain" in label or "distress" in label:
        recommendations.append("Check for signs of physical discomfort")
        recommendations.append("Assess for fever or illness symptoms")
    
    if "hungry" in label:
        recommendations.append("Recent feeding time may need review")
        recommendations.append("Monitor feeding patterns")
    
    if "wheeze" in label or "stridor" in label:
        recommendations.append("Monitor for respiratory distress signs")
        recommendations.append("Consult pediatric pulmonologist if persistent")
    
    if "crackle" in label:
        recommendations.append("Listen for chest sounds with stethoscope")
        recommendations.append("May indicate lung fluid - consult physician")
    
    if biomarkers:
        f0 = biomarkers.get("fundamental_frequency", 0)
        if f0 > 600:
            recommendations.append(f"High cry frequency ({f0:.0f} Hz) may indicate respiratory distress")
    
    if not recommendations:
        recommendations.append("Continue regular monitoring")
        recommendations.append("No immediate concerns detected")
    
    return recommendations


def store_diagnosis(diagnosis_id: str, diagnosis: DiagnosisResult, audio_data: Dict):
    """Store diagnosis for later retrieval"""
    diagnosis_store[diagnosis_id] = {
        "diagnosis": diagnosis.model_dump(),
        "audio_data": audio_data,
        "created_at": datetime.now().isoformat()
    }
    
    # Cleanup old entries (keep last 100)
    if len(diagnosis_store) > 100:
        oldest_key = min(diagnosis_store.keys(), key=lambda k: diagnosis_store[k]["created_at"])
        del diagnosis_store[oldest_key]


def get_diagnosis(diagnosis_id: str) -> Optional[Dict]:
    """Retrieve stored diagnosis"""
    return diagnosis_store.get(diagnosis_id)


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "service": "Baby Cry Diagnostic API",
        "version": "2.0.0",
        "status": "online",
        "environment": "cloud",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "stream": "WS /ws/stream",
            "report": "GET /api/v1/report/{id}",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers"""
    return HealthResponse(
        status="healthy" if ensemble_model else "degraded",
        models={
            "ensemble": ensemble_model is not None,
            "biomarkers": biomarker_analyzer is not None,
            "audio_processor": audio_processor is not None,
            "pdf_generator": pdf_generator is not None
        },
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        environment="cloud"
    )


@app.post("/api/v1/analyze", response_model=AudioUploadResponse, tags=["Analysis"])
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, or raw PCM)")
):
    """
    Analyze uploaded audio for cry classification and biomarkers.
    
    Returns:
    - Cry classification (hungry, pain, sleepy, etc.)
    - Risk level (GREEN/YELLOW/RED)
    - Medical biomarkers (f0, spectral centroid, HNR)
    - Recommendations
    """
    if not ensemble_model or not biomarker_analyzer:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Read audio data
        audio_bytes = await audio_file.read()
        logger.info(f"Received audio: {audio_file.filename}, {len(audio_bytes)} bytes")
        
        # Process audio
        audio_data = await audio_processor.process_upload(audio_bytes, audio_file.filename)
        
        # Run ensemble classification
        classification = await ensemble_model.predict_auto(audio_data["waveform"])
        
        # Extract biomarkers
        biomarkers = biomarker_analyzer.analyze(
            audio_data["waveform"],
            audio_data["sample_rate"]
        )
        
        # Calculate risk level
        risk_level, risk_score = calculate_risk_level(classification, biomarkers)
        
        # Generate recommendations
        recommendations = generate_recommendations(classification, biomarkers, risk_level)
        
        # Convert numpy types
        classification = convert_numpy_types(classification)
        biomarkers = convert_numpy_types(biomarkers)
        risk_score = convert_numpy_types(risk_score)
        
        # Create diagnosis
        diagnosis_id = str(uuid.uuid4())
        diagnosis = DiagnosisResult(
            id=diagnosis_id,
            timestamp=datetime.now().isoformat(),
            primary_classification=classification["label"],
            confidence=classification["confidence"],
            model_used=classification.get("model", "ensemble"),
            risk_level=risk_level,
            risk_score=risk_score,
            biomarkers=biomarkers,
            recommendations=recommendations
        )
        
        # Store for PDF generation
        store_diagnosis(diagnosis_id, diagnosis, audio_data)
        
        logger.info(f"Analysis complete: {classification['label']} ({risk_level})")
        
        return AudioUploadResponse(success=True, diagnosis=diagnosis)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return AudioUploadResponse(success=False, error=str(e))


@app.get("/api/v1/report/{diagnosis_id}", tags=["Reports"])
async def get_medical_report(diagnosis_id: str):
    """Generate and download medical PDF report for a diagnosis"""
    if not pdf_generator:
        raise HTTPException(status_code=503, detail="PDF generator not initialized")
    
    try:
        diagnosis_data = get_diagnosis(diagnosis_id)
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
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming from Raspberry Pi.
    
    Protocol:
    - Send raw PCM audio chunks (16-bit, 16kHz mono)
    - Receive JSON classification results
    """
    await websocket.accept()
    device_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[WS] Client connected: {device_id}")
    
    try:
        # Audio buffer configuration
        audio_buffer = []
        buffer_duration = 3.0  # seconds
        sample_rate = 16000
        samples_needed = int(buffer_duration * sample_rate)
        
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                # Raw audio data
                if not ensemble_model:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Models not initialized"
                    })
                    continue
                
                audio_chunk = np.frombuffer(data["bytes"], dtype=np.int16)
                audio_buffer.extend(audio_chunk.tolist())
                
                # Process when buffer is full
                if len(audio_buffer) >= samples_needed:
                    waveform = np.array(audio_buffer[:samples_needed], dtype=np.float32)
                    waveform = waveform / 32768.0  # Normalize
                    audio_buffer = audio_buffer[samples_needed:]
                    
                    # Run analysis
                    classification = await ensemble_model.predict_auto(waveform)
                    biomarkers = biomarker_analyzer.analyze(waveform, sample_rate)
                    risk_level, risk_score = calculate_risk_level(classification, biomarkers)
                    
                    # Convert and send result
                    result = convert_numpy_types({
                        "type": "analysis",
                        "device_id": device_id,
                        "timestamp": datetime.now().isoformat(),
                        "classification": classification["label"],
                        "confidence": classification["confidence"],
                        "risk_level": risk_level,
                        "risk_score": risk_score,
                        "biomarkers": biomarkers
                    })
                    
                    await websocket.send_json(result)
                    logger.info(f"[WS] {device_id}: {classification['label']} ({risk_level})")
            
            elif "text" in data:
                msg = json.loads(data["text"])
                
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                
                elif msg.get("type") == "config":
                    if "buffer_duration" in msg:
                        buffer_duration = msg["buffer_duration"]
                        samples_needed = int(buffer_duration * sample_rate)
                    
                    await websocket.send_json({
                        "type": "config_ack",
                        "buffer_duration": buffer_duration,
                        "sample_rate": sample_rate
                    })
    
    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {device_id}")
    except Exception as e:
        logger.error(f"[WS] Error for {device_id}: {e}")


@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """List available AI models and their status"""
    if not ensemble_model:
        return {"models": [], "status": "not_initialized"}
    
    return {
        "models": [
            {
                "name": "DistilHuBERT",
                "loaded": ensemble_model.models.get("distilhubert") is not None,
                "cry_weight": 1.0,
                "pulmonary_weight": 0.6
            },
            {
                "name": "AST",
                "loaded": ensemble_model.models.get("ast") is not None,
                "cry_weight": 0.7,
                "pulmonary_weight": 0.9
            },
            {
                "name": "YAMNet",
                "loaded": ensemble_model.models.get("yamnet") is not None,
                "cry_weight": 0.5,
                "pulmonary_weight": 0.5
            },
            {
                "name": "Wav2Vec2",
                "loaded": ensemble_model.models.get("wav2vec2") is not None,
                "cry_weight": 0.95,
                "pulmonary_weight": 0.7
            },
            {
                "name": "WavLM",
                "loaded": ensemble_model.models.get("wavlm") is not None,
                "cry_weight": 0.9,
                "pulmonary_weight": 0.75
            },
            {
                "name": "PANNs CNN14",
                "loaded": ensemble_model.models.get("panns") is not None,
                "cry_weight": 0.6,
                "pulmonary_weight": 1.0
            }
        ],
        "cry_classes": ensemble_model.cry_classes,
        "pulmonary_classes": ensemble_model.pulmonary_classes,
        "status": "ready"
    }


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=False,
        workers=1,
        log_level="info"
    )
