#!/usr/bin/env python3
"""
Medical PDF Report Generator for Baby Cry Diagnostic System
Generates professional "Infant Pulmonary Health Reports"
"""

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure reportlab is available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Fallback to FPDF if reportlab not available
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


class MedicalReportGenerator:
    """
    Generates professional medical PDF reports for infant cry analysis.
    
    Report includes:
    - Header with timestamp and session info
    - Audio waveform visualization
    - AI diagnosis results
    - Medical biomarkers (f0, spectral centroid, HNR)
    - Risk assessment
    - Recommendations
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "baby_cry_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_REPORTLAB:
            self.generator = "reportlab"
            self._setup_styles()
        elif HAS_FPDF:
            self.generator = "fpdf"
        else:
            raise ImportError(
                "No PDF library available. Install with: pip install reportlab fpdf"
            )
    
    def _setup_styles(self):
        """Setup ReportLab styles"""
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a237e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#303f9f'),
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyJustified',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskRed',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#c62828'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskYellow',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#f9a825'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskGreen',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#2e7d32'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
    
    async def generate_report(
        self, 
        diagnosis: Dict[str, Any],
        audio_data: Dict[str, Any]
    ) -> str:
        """
        Generate medical PDF report.
        
        Args:
            diagnosis: Diagnosis result dictionary
            audio_data: Audio data with waveform
        
        Returns:
            Path to generated PDF file
        """
        # Generate unique filename
        report_id = diagnosis.get("id", datetime.now().strftime("%Y%m%d%H%M%S"))
        filename = f"infant_health_report_{report_id[:8]}.pdf"
        pdf_path = self.output_dir / filename
        
        if self.generator == "reportlab":
            await self._generate_with_reportlab(pdf_path, diagnosis, audio_data)
        else:
            await self._generate_with_fpdf(pdf_path, diagnosis, audio_data)
        
        return str(pdf_path)
    
    async def _generate_with_reportlab(
        self,
        pdf_path: Path,
        diagnosis: Dict[str, Any],
        audio_data: Dict[str, Any]
    ):
        """Generate PDF using ReportLab"""
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build content
        story = []
        
        # === Header ===
        story.append(Paragraph(
            "Infant Pulmonary Health Report",
            self.styles['ReportTitle']
        ))
        
        # Subtitle
        timestamp = diagnosis.get("timestamp", datetime.now().isoformat())
        story.append(Paragraph(
            f"Generated: {timestamp}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # === Session Information ===
        story.append(Paragraph("Session Information", self.styles['SectionHeader']))
        
        session_data = [
            ["Report ID:", diagnosis.get("id", "N/A")[:16]],
            ["Analysis Date:", self._format_date(timestamp)],
            ["Audio Duration:", f"{audio_data.get('duration_seconds', 0):.2f} seconds"],
            ["Sample Rate:", f"{audio_data.get('sample_rate', 16000)} Hz"],
        ]
        
        session_table = Table(session_data, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(session_table)
        story.append(Spacer(1, 0.2*inch))
        
        # === Audio Waveform ===
        story.append(Paragraph("Audio Waveform Analysis", self.styles['SectionHeader']))
        
        # Generate waveform image
        waveform_image = await self._generate_waveform_image(audio_data)
        if waveform_image:
            story.append(Image(waveform_image, width=6*inch, height=1.5*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # === AI Diagnosis ===
        story.append(Paragraph("AI Diagnosis Results", self.styles['SectionHeader']))
        
        # Risk level box
        risk_level = diagnosis.get("risk_level", "GREEN")
        risk_score = diagnosis.get("risk_score", 0)
        risk_style = f'Risk{risk_level.capitalize()}'
        
        story.append(Paragraph(
            f"Risk Level: {risk_level} ({risk_score:.1f}/100)",
            self.styles.get(risk_style, self.styles['Normal'])
        ))
        story.append(Spacer(1, 0.1*inch))
        
        # Classification
        classification = diagnosis.get("primary_classification", "Unknown")
        confidence = diagnosis.get("confidence", 0) * 100
        model_used = diagnosis.get("model_used", "Ensemble")
        
        diag_data = [
            ["Primary Classification:", classification.replace("_", " ").title()],
            ["Confidence Score:", f"{confidence:.1f}%"],
            ["Model Used:", model_used],
        ]
        
        diag_table = Table(diag_data, colWidths=[2.5*inch, 3.5*inch])
        diag_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#e3f2fd')),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 0.2*inch))
        
        # === Medical Biomarkers ===
        story.append(Paragraph("Acoustic Biomarkers", self.styles['SectionHeader']))
        story.append(Paragraph(
            "The following acoustic biomarkers are extracted from the infant's cry "
            "and may indicate respiratory health status:",
            self.styles['BodyJustified']
        ))
        
        biomarkers = diagnosis.get("biomarkers", {})
        
        # Biomarker table with explanations
        biomarker_data = [
            ["Biomarker", "Value", "Status", "Clinical Significance"],
            [
                "Fundamental Frequency (f₀)",
                f"{biomarkers.get('fundamental_frequency', 'N/A'):.1f} Hz" if isinstance(biomarkers.get('fundamental_frequency'), (int, float)) else "N/A",
                biomarkers.get('f0_status', 'normal').upper(),
                "High pitch (>600 Hz) may indicate respiratory distress"
            ],
            [
                "Spectral Centroid",
                f"{biomarkers.get('spectral_centroid', 'N/A'):.1f} Hz" if isinstance(biomarkers.get('spectral_centroid'), (int, float)) else "N/A",
                biomarkers.get('spectral_centroid_status', 'normal').upper(),
                "Measures cry \"sharpness\" - indicator of lung congestion"
            ],
            [
                "Harmonic-to-Noise Ratio",
                f"{biomarkers.get('hnr', 'N/A'):.1f} dB" if isinstance(biomarkers.get('hnr'), (int, float)) else "N/A",
                biomarkers.get('hnr_status', 'normal').upper(),
                "Low HNR (<5 dB) indicates raspy/turbulent breath"
            ],
            [
                "Jitter",
                f"{biomarkers.get('jitter_percent', 'N/A'):.2f}%" if isinstance(biomarkers.get('jitter_percent'), (int, float)) else "N/A",
                biomarkers.get('jitter_status', 'normal').upper(),
                "Pitch instability - high values may indicate vocal issues"
            ],
            [
                "Shimmer",
                f"{biomarkers.get('shimmer_percent', 'N/A'):.2f}%" if isinstance(biomarkers.get('shimmer_percent'), (int, float)) else "N/A",
                biomarkers.get('shimmer_status', 'normal').upper(),
                "Amplitude variation - elevated in respiratory conditions"
            ],
        ]
        
        biomarker_table = Table(
            biomarker_data, 
            colWidths=[1.5*inch, 1*inch, 0.8*inch, 2.7*inch]
        )
        biomarker_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#303f9f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Body
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        
        # Color status cells
        for i, row in enumerate(biomarker_data[1:], start=1):
            status = row[2]
            if status == "CRITICAL":
                biomarker_table.setStyle(TableStyle([
                    ('BACKGROUND', (2, i), (2, i), colors.HexColor('#ffcdd2')),
                    ('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#c62828')),
                ]))
            elif status == "WARNING":
                biomarker_table.setStyle(TableStyle([
                    ('BACKGROUND', (2, i), (2, i), colors.HexColor('#fff9c4')),
                    ('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#f9a825')),
                ]))
            else:
                biomarker_table.setStyle(TableStyle([
                    ('BACKGROUND', (2, i), (2, i), colors.HexColor('#c8e6c9')),
                    ('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#2e7d32')),
                ]))
        
        story.append(biomarker_table)
        story.append(Spacer(1, 0.2*inch))
        
        # === Health Score ===
        health_score = biomarkers.get('health_score', 100)
        story.append(Paragraph(
            f"Overall Health Score: {health_score:.0f}/100",
            self.styles['SectionHeader']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # === Recommendations ===
        story.append(Paragraph("Clinical Recommendations", self.styles['SectionHeader']))
        
        recommendations = diagnosis.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(
                    f"<b>{i}.</b> {rec}",
                    self.styles['BodyJustified']
                ))
        else:
            story.append(Paragraph(
                "No specific recommendations at this time. Continue routine monitoring.",
                self.styles['BodyJustified']
            ))
        
        story.append(Spacer(1, 0.3*inch))
        
        # === Disclaimer ===
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by an AI-assisted diagnostic system "
            "and is intended for informational purposes only. It does not constitute medical advice, "
            "diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns. "
            "The system's classifications and recommendations should be verified by trained medical personnel.",
            ParagraphStyle(
                'Disclaimer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_JUSTIFY
            )
        ))
        
        # Build PDF
        doc.build(story)
    
    async def _generate_with_fpdf(
        self,
        pdf_path: Path,
        diagnosis: Dict[str, Any],
        audio_data: Dict[str, Any]
    ):
        """Generate PDF using FPDF (fallback)"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title
        pdf.set_font('Arial', 'B', 24)
        pdf.set_text_color(26, 35, 126)  # Dark blue
        pdf.cell(0, 20, 'Infant Pulmonary Health Report', ln=True, align='C')
        
        # Timestamp
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(100, 100, 100)
        timestamp = diagnosis.get("timestamp", datetime.now().isoformat())
        pdf.cell(0, 10, f'Generated: {timestamp}', ln=True, align='C')
        pdf.ln(10)
        
        # Risk Level
        risk_level = diagnosis.get("risk_level", "GREEN")
        risk_score = diagnosis.get("risk_score", 0)
        
        pdf.set_font('Arial', 'B', 16)
        if risk_level == "RED":
            pdf.set_text_color(198, 40, 40)
        elif risk_level == "YELLOW":
            pdf.set_text_color(249, 168, 37)
        else:
            pdf.set_text_color(46, 125, 50)
        
        pdf.cell(0, 12, f'Risk Level: {risk_level} ({risk_score:.1f}/100)', ln=True, align='C')
        pdf.ln(10)
        
        # Classification
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'AI Diagnosis Results', ln=True)
        pdf.set_font('Arial', '', 10)
        
        classification = diagnosis.get("primary_classification", "Unknown")
        confidence = diagnosis.get("confidence", 0) * 100
        
        pdf.cell(60, 8, 'Classification:')
        pdf.cell(0, 8, classification.replace("_", " ").title(), ln=True)
        pdf.cell(60, 8, 'Confidence:')
        pdf.cell(0, 8, f'{confidence:.1f}%', ln=True)
        pdf.ln(10)
        
        # Biomarkers
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Acoustic Biomarkers', ln=True)
        pdf.set_font('Arial', '', 9)
        
        biomarkers = diagnosis.get("biomarkers", {})
        
        f0 = biomarkers.get('fundamental_frequency', 'N/A')
        hnr = biomarkers.get('hnr', 'N/A')
        sc = biomarkers.get('spectral_centroid', 'N/A')
        
        pdf.cell(60, 7, 'Fundamental Frequency (f0):')
        pdf.cell(0, 7, f'{f0:.1f} Hz' if isinstance(f0, (int, float)) else 'N/A', ln=True)
        
        pdf.cell(60, 7, 'Harmonic-to-Noise Ratio:')
        pdf.cell(0, 7, f'{hnr:.1f} dB' if isinstance(hnr, (int, float)) else 'N/A', ln=True)
        
        pdf.cell(60, 7, 'Spectral Centroid:')
        pdf.cell(0, 7, f'{sc:.1f} Hz' if isinstance(sc, (int, float)) else 'N/A', ln=True)
        
        pdf.ln(10)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Recommendations', ln=True)
        pdf.set_font('Arial', '', 10)
        
        recommendations = diagnosis.get("recommendations", [])
        for rec in recommendations:
            pdf.multi_cell(0, 6, f'• {rec}')
        
        pdf.ln(10)
        
        # Disclaimer
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 5, 
            'DISCLAIMER: This report is generated by an AI-assisted diagnostic system '
            'and is intended for informational purposes only. Always consult a qualified '
            'healthcare provider for medical concerns.'
        )
        
        # Save
        pdf.output(str(pdf_path))
    
    async def _generate_waveform_image(self, audio_data: Dict) -> Optional[io.BytesIO]:
        """Generate waveform image from audio data"""
        try:
            from .audio_processor import AudioProcessor
            
            waveform = audio_data.get("waveform")
            sample_rate = audio_data.get("sample_rate", 16000)
            
            if waveform is None:
                return None
            
            processor = AudioProcessor()
            image_bytes = processor.generate_waveform_image(waveform, sample_rate)
            
            # Save to temp file for ReportLab
            temp_path = self.output_dir / "temp_waveform.png"
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            return str(temp_path)
            
        except Exception as e:
            print(f"[!] Failed to generate waveform image: {e}")
            return None
    
    def _format_date(self, iso_timestamp: str) -> str:
        """Format ISO timestamp to readable date"""
        try:
            dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
            return dt.strftime("%B %d, %Y at %I:%M %p")
        except:
            return iso_timestamp
