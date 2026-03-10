#!/usr/bin/env python3
"""
Medical Report Generator for Cloud Deployment

Generates PDF reports (placeholder - actual PDF library optional)
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MedicalReportGenerator:
    """
    Generates medical reports from diagnosis results.
    PDF generation is optional - returns data structure for frontend rendering.
    """
    
    def __init__(self):
        self.template_version = "1.0"
    
    def generate_report(self, diagnosis: Dict[str, Any], 
                       biomarkers: Dict[str, Any],
                       patient_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a medical report structure.
        
        Args:
            diagnosis: Classification results from AI
            biomarkers: Extracted acoustic biomarkers
            patient_info: Optional patient information
        
        Returns:
            Report data structure (can be rendered to PDF or displayed)
        """
        timestamp = datetime.now().isoformat()
        
        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": timestamp,
            "template_version": self.template_version,
            
            "header": {
                "title": "Baby Cry Diagnostic Report",
                "subtitle": "AI-Assisted Acoustic Analysis",
                "institution": "Baby Cry Diagnostic System",
                "disclaimer": "This report is for informational purposes only and should not replace professional medical advice."
            },
            
            "patient": patient_info or {
                "name": "N/A",
                "age": "N/A",
                "id": "N/A"
            },
            
            "diagnosis_summary": {
                "primary_classification": diagnosis.get("classification", "Unknown"),
                "confidence": round(diagnosis.get("confidence", 0) * 100, 1),
                "risk_level": diagnosis.get("risk_level", "YELLOW"),
                "risk_score": round(diagnosis.get("risk_score", 50), 1),
                "task_type": diagnosis.get("task", "cry")
            },
            
            "analysis_details": {
                "models_used": diagnosis.get("models_used", []),
                "all_probabilities": diagnosis.get("all_probabilities", {}),
                "processing_time_ms": diagnosis.get("processing_time_ms", 0)
            },
            
            "biomarker_analysis": self._format_biomarkers(biomarkers),
            
            "recommendations": self._generate_recommendations(diagnosis, biomarkers),
            
            "footer": {
                "generated_by": "Baby Cry Diagnostic AI System v2.0",
                "timestamp": timestamp,
                "note": "For medical emergencies, please contact healthcare services immediately."
            }
        }
        
        return report
    
    def _format_biomarkers(self, biomarkers: Dict[str, Any]) -> Dict[str, Any]:
        """Format biomarkers for report display"""
        if biomarkers.get("analysis_status") != "success":
            return {
                "status": "incomplete",
                "message": biomarkers.get("error", "Analysis not available")
            }
        
        key_metrics = {
            "Fundamental Frequency (F0)": f"{biomarkers.get('f0_mean', 0):.1f} Hz",
            "Audio Duration": f"{biomarkers.get('duration_seconds', 0):.2f} seconds",
            "Energy Level": f"{biomarkers.get('energy_mean', 0):.4f}",
            "Spectral Centroid": f"{biomarkers.get('spectral_centroid_mean', 0):.1f} Hz",
            "Voiced Ratio": f"{biomarkers.get('voiced_ratio', 0):.1%}",
        }
        
        return {
            "status": "complete",
            "key_metrics": key_metrics,
            "abnormalities": biomarkers.get("abnormality_flags", []),
            "mfcc_summary": {
                "mfcc_1": biomarkers.get("mfcc_1_mean", 0),
                "mfcc_2": biomarkers.get("mfcc_2_mean", 0),
                "mfcc_3": biomarkers.get("mfcc_3_mean", 0),
            }
        }
    
    def _generate_recommendations(self, diagnosis: Dict[str, Any],
                                   biomarkers: Dict[str, Any]) -> list:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        classification = diagnosis.get("classification", "")
        risk_level = diagnosis.get("risk_level", "YELLOW")
        confidence = diagnosis.get("confidence", 0)
        
        # Risk-based recommendations
        if risk_level == "RED":
            recommendations.append({
                "priority": "HIGH",
                "type": "medical",
                "message": "Immediate attention recommended. Consider consulting a pediatrician."
            })
        
        # Classification-specific recommendations
        if classification == "hungry":
            recommendations.append({
                "priority": "NORMAL",
                "type": "care",
                "message": "Baby may be hungry. Consider feeding if appropriate time."
            })
        elif classification == "pain":
            recommendations.append({
                "priority": "HIGH",
                "type": "medical",
                "message": "Pain indicators detected. Check for obvious causes of discomfort."
            })
        elif classification == "sleepy":
            recommendations.append({
                "priority": "LOW",
                "type": "care",
                "message": "Baby appears tired. Consider creating a calm sleep environment."
            })
        elif classification == "discomfort":
            recommendations.append({
                "priority": "NORMAL",
                "type": "care",
                "message": "Check diaper, temperature, and clothing comfort."
            })
        
        # Pulmonary-specific
        if classification in ["wheeze", "stridor", "crackle"]:
            recommendations.append({
                "priority": "HIGH",
                "type": "medical",
                "message": "Respiratory sounds detected. Monitor breathing and consult healthcare provider."
            })
        
        # Low confidence warning
        if confidence < 0.6:
            recommendations.append({
                "priority": "INFO",
                "type": "system",
                "message": f"Classification confidence is moderate ({confidence:.0%}). Consider additional recording."
            })
        
        # Default if no specific recommendations
        if not recommendations:
            recommendations.append({
                "priority": "LOW",
                "type": "info",
                "message": "No specific concerns detected. Continue routine monitoring."
            })
        
        return recommendations
    
    def generate_pdf_bytes(self, report: Dict[str, Any]) -> Optional[bytes]:
        """
        Generate PDF bytes from report (optional).
        Returns None if PDF library not available.
        """
        try:
            # This would use reportlab or similar
            # For now, return None - frontend can render the data
            logger.info("PDF generation not implemented - use report JSON data")
            return None
        except ImportError:
            return None
