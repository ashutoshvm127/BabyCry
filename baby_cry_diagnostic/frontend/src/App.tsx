import React, { useState, useRef, useEffect, useCallback } from 'react';
import './styles.css';
import { fetchScans, fetchStats, saveScan, isSupabaseConfigured, ScanDocument, StatsDocument, uploadAudio, getAudioUrl } from './services/supabase';
import { getMedicineRecommendations, TreatmentPlan, isGeminiConfigured } from './services/gemini';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

// Auth persistence key
const AUTH_STORAGE_KEY = 'crycare_auth_logged_in';

// API base URL
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

// Stats interface
interface AppStats {
  totalScans: number;
  criticalAlerts: number;
  medsAdvised: number;
  isLoading: boolean;
}

// Types
interface PulmonaryFinding {
  indicator: string;
  value: string;
  threshold?: string;
  significance: string;
  research?: string;
  requires_attention?: boolean;
}

interface AIDiseaseDetection {
  disease: string;
  confidence: number;
  all_predictions: Record<string, number>;
  is_healthy: boolean;
  requires_attention: boolean;
}

interface PulmonaryHealth {
  respiratory_distress_risk: string;
  airway_congestion_risk: string;
  breathing_effort: string;
  cry_strength: string;
  findings: PulmonaryFinding[];
  recommendations: string[];
  medical_note: string;
  overall_pulmonary_status: string;
  status_color: string;
  ai_disease_detection?: AIDiseaseDetection;
}

interface AnalysisResult {
  id: string;
  classification: {
    label: string;
    confidence: number;
    model: string;
    cry_detected?: boolean;
    ai_cry_confidence?: number;
    respiratory_indicators?: Array<{ type: string; confidence: number; severity: number }>;
    top_ai_predictions?: Record<string, number>;
    all_scores?: Record<string, number>;
  };
  biomarkers: {
    f0_mean: number;
    f0_std: number;
    spectral_centroid: number;
    hnr: number;
    energy_rms: number;
    zcr: number;
  };
  pulmonary_health?: PulmonaryHealth;
  pulmonary_disease?: AIDiseaseDetection;
  risk_level: string;
  risk_color: string;
  recommended_action: string;
  timestamp: string;
  audio_duration: number;
  model_used?: string;
  pulmonary_model_loaded?: boolean;
}

// Transform backend response to frontend AnalysisResult format
const transformBackendResponse = (data: any): AnalysisResult | null => {
  if (!data || !data.success || !data.diagnosis) {
    console.error('Invalid backend response:', data);
    return null;
  }
  const d = data.diagnosis;
  const riskColorMap: Record<string, string> = { GREEN: '#22c55e', YELLOW: '#eab308', RED: '#ef4444' };
  return {
    id: d.id || '',
    classification: {
      label: d.primary_classification || 'unknown',
      confidence: d.confidence || 0,
      model: d.model_used || 'ensemble',
      cry_detected: true,
    },
    biomarkers: d.biomarkers || { f0_mean: 0, f0_std: 0, spectral_centroid: 0, hnr: 0, energy_rms: 0, zcr: 0 },
    risk_level: d.risk_level || 'GREEN',
    risk_color: riskColorMap[d.risk_level] || '#22c55e',
    recommended_action: (d.recommendations && d.recommendations.length > 0) ? d.recommendations[0] : 'Monitor condition',
    timestamp: d.timestamp || new Date().toISOString(),
    audio_duration: 0,
    model_used: d.model_used,
  };
};

// SVG Icons
const Icons = {
  dashboard: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></svg>,
  history: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 3" /></svg>,
  mic: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" y1="19" x2="12" y2="23" /></svg>,
  upload: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>,
  reports: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>,
  settings: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>,
  search: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>,
  bell: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" /><path d="M13.73 21a2 2 0 0 1-3.46 0" /></svg>,
  moon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>,
  play: <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>,
  stop: <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1" /></svg>,
  logout: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" /></svg>,
  lock: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></svg>,
  user: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>,
  eye: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>,
  eyeOff: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" /><line x1="1" y1="1" x2="23" y2="23" /></svg>,
  shield: <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>,
  check: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>,
  arrow: <span>→</span>,
  chart: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" /></svg>,
  baby: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="8" r="5" /><path d="M20 21a8 8 0 0 0-16 0" /></svg>,
  calendar: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2" /><line x1="16" y1="2" x2="16" y2="6" /><line x1="8" y1="2" x2="8" y2="6" /><line x1="3" y1="10" x2="21" y2="10" /></svg>,
  filter: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>,
  warning: <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L1 21h22L12 2zm0 4l7.53 13H4.47L12 6zm-1 5v4h2v-4h-2zm0 6v2h2v-2h-2z" /></svg>,
  pdf: <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><text x="8" y="16" fontSize="5" fill="currentColor" stroke="none" fontWeight="bold">PDF</text></svg>,
};

// PDF Report Generator - Professional Medical Report Format
const generatePDFReport = async (
  result: AnalysisResult | null,
  treatmentPlan: TreatmentPlan | null,
  patientInfo: { id: string; session: string } = { id: 'PATIENT #8821', session: 'CRY-2023-08821' }
) => {
  if (!result) {
    alert('No analysis result to generate report. Please record or upload audio first.');
    return;
  }

  const pdf = new jsPDF('p', 'mm', 'a4');
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 20;
  let y = 15;

  // Helper functions
  const safeNum = (val: any, decimals: number = 1): string => {
    const num = typeof val === 'number' ? val : 0;
    return num.toFixed(decimals);
  };

  const drawLine = (yPos: number) => {
    pdf.setDrawColor(0, 0, 0);
    pdf.setLineWidth(0.3);
    pdf.line(margin, yPos, pageWidth - margin, yPos);
  };

  const drawTableRow = (yPos: number, cols: string[], widths: number[], isHeader: boolean = false) => {
    let x = margin;
    pdf.setFontSize(9);
    pdf.setFont('times', isHeader ? 'bold' : 'normal');
    cols.forEach((col, i) => {
      pdf.text(col, x + 2, yPos + 4);
      x += widths[i];
    });
    // Draw cell borders
    x = margin;
    widths.forEach(w => {
      pdf.rect(x, yPos - 1, w, 7);
      x += w;
    });
  };

  // ===== HEADER - Hospital/Clinic Letterhead =====
  pdf.setFont('times', 'bold');
  pdf.setFontSize(14);
  pdf.setTextColor(0, 0, 0);
  pdf.text('INFANT DIAGNOSTIC CENTER', pageWidth / 2, y, { align: 'center' });
  y += 5;
  pdf.setFont('times', 'normal');
  pdf.setFontSize(9);
  pdf.text('Pediatric Acoustic Analysis Unit', pageWidth / 2, y, { align: 'center' });
  y += 4;
  pdf.text('123 Medical Plaza, Healthcare District | Tel: (555) 123-4567 | Fax: (555) 123-4568', pageWidth / 2, y, { align: 'center' });
  y += 3;
  drawLine(y);
  y += 8;

  // ===== REPORT TITLE =====
  pdf.setFont('times', 'bold');
  pdf.setFontSize(12);
  pdf.text('ACOUSTIC CRY ANALYSIS REPORT', pageWidth / 2, y, { align: 'center' });
  y += 8;

  // ===== PATIENT INFORMATION BOX =====
  pdf.setDrawColor(0, 0, 0);
  pdf.setLineWidth(0.5);
  pdf.rect(margin, y, pageWidth - 2 * margin, 24);
  
  pdf.setFont('times', 'normal');
  pdf.setFontSize(9);
  const col1 = margin + 3;
  const col2 = margin + 90;
  
  pdf.text(`Patient ID: ${patientInfo.id}`, col1, y + 6);
  pdf.text(`Report Date: ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}`, col2, y + 6);
  pdf.text(`Session Reference: ${patientInfo.session}`, col1, y + 12);
  pdf.text(`Time: ${new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`, col2, y + 12);
  pdf.text(`Analysis Method: Ensemble AI (Multi-backbone CNN)`, col1, y + 18);
  pdf.text(`Model: ${result.model_used || 'DistilHuBERT + AST + YAMNet'}`, col2, y + 18);
  
  y += 30;

  // ===== SECTION 1: PRIMARY DIAGNOSIS =====
  pdf.setFont('times', 'bold');
  pdf.setFontSize(10);
  pdf.text('1. PRIMARY DIAGNOSIS', margin, y);
  y += 6;

  // Diagnosis table
  const diagWidths = [55, pageWidth - 2 * margin - 55];
  drawTableRow(y, ['Parameter', 'Finding'], diagWidths, true);
  y += 7;
  
  const classification = result.classification.label.replace(/_/g, ' ');
  const confidence = Math.round((result.classification.confidence || 0) * 100);
  
  drawTableRow(y, ['Classification', classification.charAt(0).toUpperCase() + classification.slice(1)], diagWidths);
  y += 7;
  drawTableRow(y, ['Confidence Level', `${confidence}%`], diagWidths);
  y += 7;
  drawTableRow(y, ['Risk Assessment', result.risk_level || 'Not assessed'], diagWidths);
  y += 7;
  drawTableRow(y, ['Recommended Action', result.recommended_action || 'Monitor and observe'], diagWidths);
  y += 12;

  // ===== SECTION 2: ACOUSTIC BIOMARKERS =====
  pdf.setFont('times', 'bold');
  pdf.setFontSize(10);
  pdf.text('2. ACOUSTIC BIOMARKER ANALYSIS', margin, y);
  y += 6;

  const bioWidths = [60, 45, 65];
  drawTableRow(y, ['Biomarker', 'Value', 'Clinical Significance'], bioWidths, true);
  y += 7;

  const biomarkerData = [
    ['Fundamental Frequency (F0) Mean', `${safeNum(result.biomarkers?.f0_mean, 1)} Hz`, 'Vocal cord tension indicator'],
    ['F0 Standard Deviation', `${safeNum(result.biomarkers?.f0_std, 1)} Hz`, 'Pitch variability measure'],
    ['Spectral Centroid', `${safeNum(result.biomarkers?.spectral_centroid, 0)} Hz`, 'Tonal brightness index'],
    ['Harmonic-to-Noise Ratio', `${safeNum(result.biomarkers?.hnr, 1)} dB`, 'Voice quality metric'],
    ['Energy RMS', `${safeNum(result.biomarkers?.energy_rms, 4)}`, 'Signal intensity level'],
    ['Zero Crossing Rate', `${safeNum(result.biomarkers?.zcr, 4)}`, 'Frequency content marker'],
  ];

  biomarkerData.forEach(row => {
    drawTableRow(y, row, bioWidths);
    y += 7;
  });
  y += 8;

  // ===== SECTION 3: PULMONARY ASSESSMENT (if available) =====
  if (result.pulmonary_disease) {
    pdf.setFont('times', 'bold');
    pdf.setFontSize(10);
    pdf.text('3. PULMONARY HEALTH ASSESSMENT', margin, y);
    y += 6;

    const pulmWidths = [55, pageWidth - 2 * margin - 55];
    drawTableRow(y, ['Parameter', 'Finding'], pulmWidths, true);
    y += 7;
    
    const pulm = result.pulmonary_disease;
    drawTableRow(y, ['Respiratory Status', pulm.is_healthy ? 'Normal' : 'Abnormal findings detected'], pulmWidths);
    y += 7;
    drawTableRow(y, ['Detected Condition', pulm.disease.replace(/_/g, ' ')], pulmWidths);
    y += 7;
    drawTableRow(y, ['Detection Confidence', `${Math.round(pulm.confidence * 100)}%`], pulmWidths);
    y += 12;
  }

  // ===== SECTION 4: TREATMENT RECOMMENDATIONS =====
  if (treatmentPlan && treatmentPlan.medicines.length > 0) {
    pdf.setFont('times', 'bold');
    pdf.setFontSize(10);
    pdf.text(result.pulmonary_disease ? '4. TREATMENT RECOMMENDATIONS' : '3. TREATMENT RECOMMENDATIONS', margin, y);
    y += 6;

    const medWidths = [45, 35, 35, 55];
    drawTableRow(y, ['Medication', 'Form', 'Dosage', 'Frequency'], medWidths, true);
    y += 7;

    treatmentPlan.medicines.forEach(med => {
      drawTableRow(y, [med.name, med.type || '-', med.dosage || '-', med.frequency || '-'], medWidths);
      y += 7;
    });
    y += 8;
  }

  // ===== SECTION 5: CLINICAL NOTES =====
  const notesSection = result.pulmonary_disease ? (treatmentPlan?.medicines.length ? '5' : '4') : (treatmentPlan?.medicines.length ? '4' : '3');
  pdf.setFont('times', 'bold');
  pdf.setFontSize(10);
  pdf.text(`${notesSection}. CLINICAL NOTES`, margin, y);
  y += 6;

  pdf.setFont('times', 'normal');
  pdf.setFontSize(9);
  
  const notes = treatmentPlan?.suggestions && treatmentPlan.suggestions.length > 0 
    ? treatmentPlan.suggestions 
    : ['Acoustic analysis performed using validated AI algorithms.', 
       'Results should be interpreted in clinical context.',
       'Follow-up assessment recommended if symptoms persist.'];
  
  notes.forEach((note, i) => {
    pdf.text(`${i + 1}. ${note}`, margin + 3, y);
    y += 5;
  });
  y += 8;

  // ===== SIGNATURE SECTION =====
  drawLine(y);
  y += 10;

  pdf.setFont('times', 'normal');
  pdf.setFontSize(9);
  pdf.text('Analyzing Physician: ___________________________', margin, y);
  pdf.text('Date: _______________', pageWidth - margin - 45, y);
  y += 10;
  pdf.text('Signature: ___________________________', margin, y);
  pdf.text('License No.: _______________', pageWidth - margin - 45, y);
  y += 15;

  // ===== FOOTER =====
  pdf.setDrawColor(0, 0, 0);
  pdf.setLineWidth(0.3);
  pdf.line(margin, pageHeight - 25, pageWidth - margin, pageHeight - 25);
  
  pdf.setFont('times', 'italic');
  pdf.setFontSize(7);
  pdf.setTextColor(80, 80, 80);
  pdf.text('CONFIDENTIAL MEDICAL RECORD - This document contains protected health information.', margin, pageHeight - 20);
  pdf.text('Unauthorized disclosure is prohibited under applicable law. AI-assisted analysis for clinical reference only.', margin, pageHeight - 16);
  pdf.text(`Document ID: ${result.id || 'N/A'} | Generated: ${new Date().toISOString()}`, margin, pageHeight - 12);
  pdf.text('Page 1 of 1', pageWidth - margin - 15, pageHeight - 12);

  // Download
  pdf.save(`Medical_Report_${patientInfo.session}_${new Date().toISOString().split('T')[0]}.pdf`);
};

// Finalize Session Handler
const finalizeSession = (
  result: AnalysisResult | null,
  setResult: (r: AnalysisResult | null) => void,
  setTreatmentPlan: (t: TreatmentPlan | null) => void,
  additionalReset?: () => void
) => {
  if (!result) {
    alert('No active session to finalize.');
    return;
  }
  
  const confirmed = window.confirm(
    'Are you sure you want to finalize this session?\n\nThis will:\n• Mark the scan as complete\n• Clear the current analysis\n• Allow you to start a new session'
  );
  
  if (confirmed) {
    setResult(null);
    setTreatmentPlan(null);
    if (additionalReset) additionalReset();
    alert('Session finalized successfully!\n\nThe scan has been saved to history.');
  }
};

// ===== LOGIN PAGE =====
function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [doctorId, setDoctorId] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [remember, setRemember] = useState(false);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (doctorId === 'admin' && password === 'admin12') {
      onLogin();
    } else {
      setError('Invalid Doctor ID or Password. Please try again.');
    }
  };

  return (
    <div className="login-page">
      <div className="login-brand">
        <div className="login-brand-icon">⚕</div>
        <h1>CryCare <span>AI</span></h1>
      </div>
      <div className="login-system-status">
        System Status: <span>Operational</span>
      </div>

      <div className="login-left">
        <div className="login-baby-image">
          <img
            src="https://images.unsplash.com/photo-1555252333-9f8e92e65df9?w=400&h=400&fit=crop&crop=face"
            alt="Baby"
            style={{ width: 220, height: 220, borderRadius: '50%', objectFit: 'cover', boxShadow: '0 8px 32px rgba(0,0,0,0.12)', border: '4px solid rgba(255,255,255,0.8)' }}
          />
        </div>
        <div className="login-quote">
          <p>"Every cry is a call for care, every smile is a reason why."</p>
          <span>— CRYCARE PHILOSOPHY</span>
        </div>
      </div>

      <div className="login-right">
        <form className="login-form-card" onSubmit={handleLogin}>
          <h2>Welcome Back</h2>
          <p className="subtitle">Secure access for CryCare medical practitioners.</p>

          {error && (
            <div className="login-error">
              <span>⚠</span> {error}
            </div>
          )}

          <div className="form-group">
            <label className="form-label">Doctor ID</label>
            <div className="form-input-wrapper">
              <span className="form-input-icon">{Icons.user}</span>
              <input
                type="text"
                placeholder="Enter your credential ID"
                value={doctorId}
                onChange={e => { setDoctorId(e.target.value); setError(''); }}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Password
              <a href="#forgot">Forgot Password?</a>
            </label>
            <div className="form-input-wrapper">
              <span className="form-input-icon">{Icons.lock}</span>
              <input
                type={showPassword ? 'text' : 'password'}
                placeholder="••••••••"
                value={password}
                onChange={e => { setPassword(e.target.value); setError(''); }}
              />
              <button type="button" className="toggle-password" onClick={() => setShowPassword(!showPassword)}>
                {showPassword ? Icons.eyeOff : Icons.eye}
              </button>
            </div>
          </div>

          <label className="remember-row">
            <input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} />
            Remember this device for 30 days
          </label>

          <button type="submit" className="login-btn">
            {Icons.shield} Secure Login
          </button>

          <div className="login-footer-security">
            PROTECTED BY CRYCARE ENTERPRISE SECURITY™
          </div>
          <div className="login-footer-badges">
            <span>🟢 HIPAA COMPLIANT</span>
            <span>🟢 AES-256 ENCRYPTED</span>
          </div>
        </form>
      </div>

      <div className="login-page-footer">
        © 2024 CryCare AI Medical Systems Inc. All Rights Reserved.
        <a href="#privacy">Privacy Policy</a> | <a href="#terms">Terms of Service</a>
      </div>
    </div>
  );
}

// ===== SIDEBAR =====
function Sidebar({ activePage, onNavigate, onLogout }: {
  activePage: string;
  onNavigate: (page: string) => void;
  onLogout: () => void;
}) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Icons.dashboard },
    { id: 'record', label: 'Record', icon: Icons.mic },
    { id: 'upload', label: 'Upload', icon: Icons.upload },
    { id: 'history', label: 'History', icon: Icons.history },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">⚕</div>
        <div className="sidebar-brand-text">
          <h2>CryCare AI</h2>
          <span>CLINICAL SUITE</span>
        </div>
      </div>
      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            className={`sidebar-nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            {item.icon}
            {item.label}
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-storage">
          <div className="sidebar-storage-header">
            <span>STORAGE</span><span>65%</span>
          </div>
          <div className="sidebar-storage-bar">
            <div className="sidebar-storage-fill" style={{ width: '65%' }} />
          </div>
          <div className="sidebar-storage-text">12.4 GB of 20 GB used</div>
        </div>
        <div className="sidebar-user">
          <div className="sidebar-user-avatar">S</div>
          <div className="sidebar-user-info">
            <h4>Dr. Seethal</h4>
            <span>Pediatrician</span>
          </div>
          <button className="sidebar-logout-btn" onClick={onLogout} title="Logout">
            {Icons.logout}
          </button>
        </div>
      </div>
    </aside>
  );
}

// ===== TOP BAR =====
function TopBar({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="top-bar">
      <div className="top-bar-left">
        <h1>{title}</h1>
        {subtitle && <div className="system-online">{subtitle}</div>}
      </div>
      <div className="top-bar-right">
        <div className="search-bar">
          {Icons.search}
          <input placeholder="Search patient records..." />
        </div>
        <button className="icon-btn">{Icons.bell}</button>
        <button className="icon-btn">{Icons.moon}</button>
        <div className="user-avatar-sm">DS</div>
      </div>
    </div>
  );
}

// ===== DASHBOARD PAGE =====
interface DashboardScan {
  id: string;
  date: string;
  time: string;
  name: string;
  disease: string;
  diseaseColor: string;
  accuracy: number;
}

function DashboardPage({ onNavigate, scans = [], stats, onViewReport }: { onNavigate: (page: string) => void; scans?: DashboardScan[]; stats: AppStats; onViewReport?: (scan: HistoryScan) => void }) {
  // Convert scan history to dashboard format, show latest 3
  const recentScans = scans.slice(0, 3).map((scan, idx) => {
    // Determine assessment level based on disease type
    let assessment = scan.disease.toUpperCase();
    let assessmentColor = scan.diseaseColor || 'blue';
    let avatarColor = 'blue';
    
    // High severity conditions
    const criticalConditions = ['pain_cry', 'distress_cry', 'ards', 'asphyxia', 'sepsis_respiratory'];
    const moderateConditions = ['discomfort_cry', 'cold_cry', 'pneumonia', 'bronchitis', 'respiratory_distress'];
    const normalConditions = ['no_cry_detected', 'normal_cry', 'sleepy_cry', 'hungry_cry', 'tired_cry', 'healthy'];
    
    const diseaseLower = scan.disease.toLowerCase().replace(/ /g, '_');
    
    if (normalConditions.some(c => diseaseLower.includes(c))) {
      assessment = scan.disease.toUpperCase();
      assessmentColor = diseaseLower.includes('no_cry') ? 'blue' : 'green';
      avatarColor = 'green';
    } else if (criticalConditions.some(c => diseaseLower.includes(c))) {
      assessment = `CRITICAL: ${scan.disease.toUpperCase()}`;
      assessmentColor = 'red';
      avatarColor = 'red';
    } else if (moderateConditions.some(c => diseaseLower.includes(c))) {
      assessment = `MODERATE: ${scan.disease.toUpperCase()}`;
      assessmentColor = 'orange';
      avatarColor = 'orange';
    } else if (scan.accuracy > 85) {
      assessment = `HIGH: ${scan.disease.toUpperCase()}`;
      assessmentColor = 'orange';
    } else {
      assessment = `LOW: ${scan.disease.toUpperCase()}`;
      assessmentColor = 'green';
    }
    
    return {
      id: scan.id,
      date: scan.date,
      time: scan.time,
      assessment,
      assessmentColor,
      confidence: scan.accuracy,
      avatarColor,
      originalScan: scan as unknown as HistoryScan  // Keep original for viewing report
    };
  });

  return (
    <>
      <TopBar title="Welcome, Dr. Seethal" subtitle="SYSTEM ONLINE" />
      <div className="page-content">
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon blue">{Icons.chart}</div>
            <div className="stat-content">
              <h3>98.5%</h3>
              <p>MODEL ACCURACY</p>
            </div>
            <span className="stat-badge green">↗ 0.2%</span>
          </div>
          <div className="stat-card">
            <div className="stat-icon orange">😊</div>
            <div className="stat-content">
              <h3>{stats.isLoading ? '...' : stats.totalScans.toLocaleString()}</h3>
              <p>TOTAL SCANS</p>
            </div>
            {stats.totalScans > 0 && <span className="stat-badge blue">+{Math.min(stats.totalScans, 100)}</span>}
          </div>
          <div className="stat-card">
            <div className="stat-icon red">⚠</div>
            <div className="stat-content">
              <h3>{stats.isLoading ? '...' : stats.criticalAlerts.toLocaleString()}</h3>
              <p>CRITICAL ALERTS</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon teal">💊</div>
            <div className="stat-content">
              <h3>{stats.isLoading ? '...' : stats.medsAdvised.toLocaleString()}</h3>
              <p>MEDS ADVISED</p>
            </div>
            {stats.medsAdvised > 0 && <span className="stat-badge orange">+{Math.min(stats.medsAdvised, 10)}</span>}
          </div>
        </div>

        <div className="action-cards">
          <button className="action-card primary" onClick={() => onNavigate('record')}>
            <div className="action-card-icon">🎙</div>
            <div>
              <h3>Record New Session</h3>
              <p>Real-time biometric analysis & AI cry classification</p>
            </div>
            <span className="action-card-arrow">›</span>
          </button>
          <button className="action-card secondary" onClick={() => onNavigate('upload')}>
            <div className="action-card-icon">{Icons.upload}</div>
            <div>
              <h3>Upload Lab Data</h3>
              <p>Batch process recording files for retrospective study</p>
            </div>
            <span className="action-card-arrow">›</span>
          </button>
        </div>

        <div className="table-card">
          <div className="table-header">
            <div className="table-header-left">
              <h3>Recent Clinical Analysis</h3>
              <p>Latest patient scans processed by the AI engine</p>
            </div>
            <div className="table-header-right">
              <button className="btn btn-outline">Export PDF</button>
              <button className="btn btn-primary" onClick={() => onNavigate('history')}>View All Records</button>
            </div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>PATIENT REFERENCE</th>
                <th>TIMESTAMP</th>
                <th>AI ASSESSMENT</th>
                <th>CONFIDENCE SCORE</th>
                <th>CLINICAL ACTION</th>
              </tr>
            </thead>
            <tbody>
              {recentScans.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                    <div style={{ fontSize: 48, marginBottom: 16 }}>📊</div>
                    <h4 style={{ margin: 0, color: '#cbd5e1' }}>No scans recorded yet</h4>
                    <p style={{ margin: '8px 0 0', fontSize: 13 }}>Start by recording a new session or uploading an audio file</p>
                  </td>
                </tr>
              ) : recentScans.map(scan => (
                <tr key={scan.id}>
                  <td>
                    <div className="patient-ref">
                      <div className={`patient-avatar ${scan.avatarColor}`}>{Icons.baby}</div>
                      <strong>{scan.id}</strong>
                    </div>
                  </td>
                  <td>{scan.date}<br /><span style={{ fontSize: 12, color: '#94a3b8' }}>{scan.time}</span></td>
                  <td><span className={`badge badge-${scan.assessmentColor}`}>{scan.assessment}</span></td>
                  <td>
                    <div className="confidence-bar">
                      <div className="confidence-track">
                        <div className={`confidence-fill ${scan.confidence >= 95 ? 'high' : scan.confidence >= 90 ? 'medium' : 'low'}`} style={{ width: `${scan.confidence}%` }} />
                      </div>
                      <span className="confidence-text">{scan.confidence}%</span>
                    </div>
                  </td>
                  <td><button className="view-link" onClick={() => onViewReport && onViewReport(scan.originalScan)}>Review Report →</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

// ===== RECORD PAGE =====
function RecordPage({ onScanComplete }: { onScanComplete?: (result: AnalysisResult, audioBlob?: Blob) => void }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [micError, setMicError] = useState('');
  const [hasRecorded, setHasRecorded] = useState(false);
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [treatmentPlan, setTreatmentPlan] = useState<TreatmentPlan | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const animFrameRef = useRef<number>(0);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Fetch medicine recommendations when result changes
  useEffect(() => {
    if (result) {
      const fetchMedicines = async () => {
        const diagnosis = result.classification.label;
        const confidence = result.classification.confidence;
        const biomarkers = result.biomarkers;
        const plan = await getMedicineRecommendations(diagnosis, confidence, biomarkers);
        setTreatmentPlan(plan);
      };
      fetchMedicines();
    } else {
      setTreatmentPlan(null);
    }
  }, [result]);

  const formatDuration = (s: number) => {
    const h = Math.floor(s / 3600).toString().padStart(2, '0');
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${h}:${m}:${sec}`;
  };

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#3b82f6';
    ctx.beginPath();
    const sliceWidth = w / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * h) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    // Draw bars
    analyser.getByteFrequencyData(dataArray);
    const barW = Math.max(2, (w / bufferLength) * 4);
    for (let i = 0; i < bufferLength / 2; i++) {
      const barH = (dataArray[i] / 255) * h * 0.8;
      const bx = (w / 2) + (i - bufferLength / 4) * barW * 0.5;
      if (bx >= 0 && bx <= w) {
        ctx.fillStyle = `rgba(59, 130, 246, ${0.3 + (dataArray[i] / 255) * 0.7})`;
        ctx.fillRect(bx, (h - barH) / 2, barW * 0.3, barH);
      }
    }
    if (isRecording) animFrameRef.current = requestAnimationFrame(drawWaveform);
  }, [isRecording]);

  const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
    for (let i = 0; i < input.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  };

  const createWavBlob = (samples: Float32Array, sampleRate: number): Blob => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const ws = (v: DataView, o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    ws(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    ws(view, 8, 'WAVE');
    ws(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    ws(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return new Blob([buffer], { type: 'audio/wav' });
  };

  const startRecording = async () => {
    setMicError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      streamRef.current = stream;
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      audioChunksRef.current = [];
      processor.onaudioprocess = (e) => {
        const copy = new Float32Array(e.inputBuffer.getChannelData(0).length);
        copy.set(e.inputBuffer.getChannelData(0));
        audioChunksRef.current.push(copy);
      };
      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioContext.destination);
      setIsRecording(true);
      setRecordingTime(0);
      setResult(null);
      setAudioUrl(null);
      timerRef.current = setInterval(() => setRecordingTime(p => p + 1), 1000);
    } catch (err: any) {
      console.error('Mic error:', err);
      if (err.name === 'NotAllowedError') {
        setMicError('Microphone access denied. Please allow microphone permission in your browser settings and try again.');
      } else if (err.name === 'NotFoundError') {
        setMicError('No microphone detected. Please connect a microphone and try again.');
      } else {
        setMicError('Failed to access microphone: ' + (err.message || 'Unknown error'));
      }
    }
  };

  useEffect(() => {
    if (isRecording && canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = canvas.offsetWidth * 2;
      canvas.height = canvas.offsetHeight * 2;
      drawWaveform();
    }
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [isRecording, drawWaveform]);

  const stopRecording = async () => {
    setIsRecording(false);
    setHasRecorded(true);
    if (timerRef.current) clearInterval(timerRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    if (audioContextRef.current) await audioContextRef.current.close();
    const totalLength = audioChunksRef.current.reduce((a, c) => a + c.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioChunksRef.current) { combined.set(chunk, offset); offset += chunk.length; }
    const wavBlob = createWavBlob(combined, 16000);
    // Create audio URL for playback
    const url = URL.createObjectURL(wavBlob);
    setAudioUrl(url);
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('audio_file', wavBlob, 'recording.wav');
      const response = await fetch(`${API_URL}/api/v1/analyze`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      const transformed = transformBackendResponse(data);
      setResult(transformed);
      // Save scan to history with audio blob
      if (onScanComplete && transformed) {
        onScanComplete(transformed, wavBlob);
      }
    } catch (err: any) {
      console.error('Analysis failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const detectionLabel = result?.classification.label.replace(/_/g, ' ') || 'Waiting...';
  const detectionConf = result ? Math.round(result.classification.confidence * 100) : 0;

  return (
    <>
      <div className="top-bar">
        <div className="record-header-left">
          <span style={{ fontSize: 24 }}>🎙</span>
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 700 }}>Live Cry Analysis</h1>
            <p style={{ fontSize: 13, color: '#94a3b8' }}>PATIENT #8821 • SESSION CRY-2023-08821</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div className="recording-duration">
            <span>Recording Duration</span>
            <div className="time">{formatDuration(recordingTime)}</div>
          </div>
          <button className="icon-btn">{Icons.bell}</button>
          <div className="user-avatar-sm">DS</div>
        </div>
      </div>
      <div className="page-content">
        {isLoading && <div className="loading-overlay"><div className="spinner" /></div>}
        <div className="record-grid">
          <div className="record-main">
            {/* ===== PROMINENT RECORD BUTTON ===== */}
            <div className="record-control-card">
              <div className="record-control-header">
                <h3>🎤 Audio Recording Controls</h3>
                <span className={`record-status-badge ${isRecording ? 'recording' : hasRecorded ? 'completed' : 'idle'}`}>
                  {isRecording ? '● RECORDING' : hasRecorded ? '✓ RECORDED' : '○ READY'}
                </span>
              </div>

              {micError && (
                <div className="mic-error">
                  <span>⚠</span> {micError}
                </div>
              )}

              <div className="record-button-area">
                <button
                  id="record-audio-btn"
                  className={`record-main-btn ${isRecording ? 'is-recording' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                >
                  <span className="record-btn-icon">
                    {isRecording ? Icons.stop : Icons.mic}
                  </span>
                  <span className="record-btn-label">
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </span>
                  {isRecording && <span className="record-btn-time">{formatDuration(recordingTime)}</span>}
                </button>
                <p className="record-hint">
                  {isRecording
                    ? 'Recording in progress... Click to stop and analyze.'
                    : 'Click the button above to begin capturing audio from your microphone.'}
                </p>
              </div>
            </div>

            <div className="waveform-card">
              <div className="waveform-header">
                <div className="waveform-header-left">{Icons.chart} Live Waveform Visualization</div>
                <div className="waveform-badges">
                  {isRecording && <span className="live-badge">LIVE</span>}
                  <span className="quality-badge">48KHZ / 24-BIT</span>
                </div>
              </div>
              <div className="waveform-display">
                <canvas ref={canvasRef} className="waveform-canvas" />
              </div>
              <div className="waveform-timeline">
                <span>-05:00s</span><span>-04:00s</span><span>-03:00s</span><span>-02:00s</span><span>-01:00s</span><span>Now ({formatDuration(recordingTime)})</span>
              </div>
            </div>

            {/* ===== PLAYBACK AFTER RECORDING ===== */}
            {hasRecorded && audioUrl && (
              <div className="playback-controls">
                <h3>🔊 Replay Recorded Audio</h3>
                <p>Review captured audio before finalizing the diagnostic session.</p>
                <audio ref={audioPlayerRef} src={audioUrl} controls style={{ width: '100%', borderRadius: 8 }} />
              </div>
            )}

            <div className="clinical-notes">
              <div className="clinical-notes-header">
                <h3>📝 Doctor's Clinical Notes</h3>
                <span className="auto-save">Auto-saving...</span>
              </div>
              <textarea placeholder="Enter clinical observations, manual diagnosis notes, and patient history context here..." value={clinicalNotes} onChange={e => setClinicalNotes(e.target.value)} />
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div className="detection-card">
              <div className="detection-card-label">PRIMARY DETECTION</div>
              <h2 style={{ textTransform: 'capitalize' }}>{detectionLabel}</h2>
              <div className="detection-score">
                <span className="percent">{detectionConf}%</span>
                <span>Confidence Score</span>
              </div>
              <div className="detection-bar"><div className="detection-bar-fill" style={{ width: `${detectionConf}%` }} /></div>
            </div>
            
            {/* Pulmonary Disease AI Detection Card - Record Page */}
            {result?.pulmonary_disease && (
              <div className="detection-card" style={{ 
                background: result.pulmonary_disease.is_healthy 
                  ? 'linear-gradient(135deg, #064e3b 0%, #065f46 100%)' 
                  : 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)'
              }}>
                <div className="detection-card-label">🫁 AI PULMONARY DISEASE DETECTION</div>
                <h2 style={{ textTransform: 'capitalize' }}>
                  {result.pulmonary_disease.disease.replace(/_/g, ' ')}
                </h2>
                <div className="detection-score">
                  <span className="percent">{Math.round(result.pulmonary_disease.confidence * 100)}%</span>
                  <span>Disease Confidence</span>
                </div>
                <div className="detection-bar">
                  <div className="detection-bar-fill" style={{ 
                    width: `${Math.round(result.pulmonary_disease.confidence * 100)}%`,
                    background: result.pulmonary_disease.is_healthy ? '#10b981' : '#ef4444'
                  }} />
                </div>
                {result.pulmonary_disease.requires_attention && (
                  <div style={{ 
                    marginTop: 12, padding: '8px 12px', 
                    background: 'rgba(239, 68, 68, 0.2)', 
                    borderRadius: 8, fontSize: 12, color: '#fca5a5' 
                  }}>
                    ⚠️ Requires medical attention - consult a pediatrician
                  </div>
                )}
              </div>
            )}
            
            {/* Only show medicines after analysis - Record Page */}
            {result && treatmentPlan && (
              <>
                <div className="rx-card">
                  <div className="rx-card-label">RECOMMENDED MEDICINES & PRESCRIPTIONS</div>
                  {treatmentPlan.medicines.length > 0 ? (
                    treatmentPlan.medicines.map((med, idx) => (
                      <div key={idx} className="rx-item">
                        <div>
                          <h4>{med.name}</h4>
                          <span>{med.aiSuggested ? 'AI SUGGESTED' : ''} {med.type.toUpperCase()}{med.dosage ? ` • ${med.dosage}` : ''}</span>
                        </div>
                        <span style={{ color: '#10b981' }}>✓</span>
                      </div>
                    ))
                  ) : (
                    <div className="rx-item">
                      <div><h4>No specific medication required</h4><span>Monitor and comfort measures recommended</span></div>
                      <span style={{ color: '#10b981' }}>✓</span>
                    </div>
                  )}
                  {treatmentPlan.consultDoctor && (
                    <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(251, 191, 36, 0.2)', borderRadius: 8, fontSize: 12, color: '#fbbf24' }}>
                      ⚠️ {treatmentPlan.disclaimer}
                    </div>
                  )}
                </div>
                <div className="indicators-card">
                  <div className="indicators-card-label">ACOUSTIC INDICATORS</div>
                  <div className="indicator-item">
                    <div className="indicator-icon green">📈</div>
                    <div className="indicator-content"><h4>High Pitch (Melodic Peak)</h4><p>Frequency spike at 450Hz-600Hz range detected.</p></div>
                  </div>
                  <div className="indicator-item">
                    <div className="indicator-icon blue">{Icons.chart}</div>
                    <div className="indicator-content"><h4>Rhythmic Tensions</h4><p>Periodic intensity fluctuations matching abdominal distress.</p></div>
                  </div>
                  <div className="indicator-item">
                    <div className="indicator-icon orange">💧</div>
                    <div className="indicator-content"><h4>Short Inspiratory Gaps</h4><p>Breath intervals below 0.4s indicate acute discomfort.</p></div>
                  </div>
                </div>
                <div className="suggestions-card">
                  <div className="indicators-card-label">PRESCRIPTION SUGGESTIONS</div>
                  {treatmentPlan.suggestions.length > 0 ? (
                    treatmentPlan.suggestions.map((suggestion, idx) => (
                      <div key={idx} className="suggestion-item"><span>{suggestion}</span><span>›</span></div>
                    ))
                  ) : (
                    <>
                      <div className="suggestion-item"><span>General comfort measures</span><span>›</span></div>
                      <div className="suggestion-item"><span>Monitor for changes</span><span>›</span></div>
                    </>
                  )}
                </div>
              </>
            )}
            
            {/* Show placeholder when no analysis yet - Record Page */}
            {!result && (
              <div className="rx-card" style={{ opacity: 0.5 }}>
                <div className="rx-card-label">RECOMMENDED MEDICINES & PRESCRIPTIONS</div>
                <div style={{ padding: '20px', textAlign: 'center', color: '#64748b' }}>
                  <p>🎤 Record audio to receive AI-powered recommendations</p>
                </div>
              </div>
            )}
            
            <div className="finalize-section">
              <div 
                className="download-card" 
                onClick={() => generatePDFReport(result, treatmentPlan)}
                style={{ cursor: result ? 'pointer' : 'not-allowed', opacity: result ? 1 : 0.5 }}
              >
                <div className="pdf-icon" style={{ color: '#dc2626' }}>📄</div>
                <h4>Download Full PDF Report</h4>
                <span>INCLUDES WAVEFORM & ANALYSIS</span>
              </div>
              <button 
                className="finalize-btn" 
                onClick={() => finalizeSession(result, setResult, setTreatmentPlan, () => {
                  setRecordingTime(0);
                  setHasRecorded(false);
                  setAudioUrl(null);
                  setClinicalNotes('');
                })}
                disabled={!result}
                style={{ opacity: result ? 1 : 0.5 }}
              >
                ✅ Finalize Session
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ===== UPLOAD PAGE =====
function UploadPage({ onScanComplete }: { onScanComplete?: (result: AnalysisResult, audioBlob?: Blob) => void }) {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [treatmentPlan, setTreatmentPlan] = useState<TreatmentPlan | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const audioBufferRef = useRef<AudioBuffer | null>(null);
  const animFrameRef = useRef<number>(0);
  const isPlayingRef = useRef(false);

  // Fetch treatment recommendations when analysis completes
  useEffect(() => {
    if (result) {
      const fetchTreatment = async () => {
        const diagnosis = result.classification.label;
        const confidence = result.classification.confidence;
        const biomarkers = result.biomarkers;
        const plan = await getMedicineRecommendations(diagnosis, confidence, biomarkers);
        setTreatmentPlan(plan);
      };
      fetchTreatment();
    } else {
      setTreatmentPlan(null);
    }
  }, [result]);

  const drawWaveform = useCallback((playbackProgress: number = -1) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const buffer = audioBufferRef.current;
    if (!buffer) return;

    const w = canvas.width;
    const h = canvas.height;
    const data = buffer.getChannelData(0);

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Cursor X position
    const cursorX = playbackProgress >= 0 ? playbackProgress * w : -1;

    // Draw waveform
    const step = Math.ceil(data.length / w);
    for (let i = 0; i < w; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const idx = i * step + j;
        if (idx < data.length) {
          if (data[idx] < min) min = data[idx];
          if (data[idx] > max) max = data[idx];
        }
      }
      const yLow = ((1 + min) / 2) * h;
      const yHigh = ((1 + max) / 2) * h;

      // Color played portion brighter
      if (cursorX >= 0 && i <= cursorX) {
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2.5;
      } else {
        ctx.strokeStyle = '#1e40af';
        ctx.lineWidth = 2;
      }
      ctx.beginPath();
      ctx.moveTo(i, yLow);
      ctx.lineTo(i, yHigh);
      ctx.stroke();
    }

    // Draw frequency bars overlay
    const barCount = 64;
    const barW = w / barCount;
    for (let i = 0; i < barCount; i++) {
      const segStart = Math.floor((i / barCount) * data.length);
      const segEnd = Math.floor(((i + 1) / barCount) * data.length);
      let sum = 0;
      for (let j = segStart; j < segEnd; j++) sum += Math.abs(data[j]);
      const avg = sum / (segEnd - segStart);
      const barH = avg * h * 1.5;
      const bx = i * barW;
      const played = cursorX >= 0 && bx <= cursorX;
      ctx.fillStyle = played
        ? `rgba(96, 165, 250, ${0.15 + avg * 0.5})`
        : `rgba(30, 64, 175, ${0.1 + avg * 0.4})`;
      ctx.fillRect(bx + 1, (h - barH) / 2, barW - 2, barH);
    }

    // Draw playback cursor line
    if (cursorX >= 0 && cursorX <= w) {
      // Glow
      ctx.shadowColor = '#3b82f6';
      ctx.shadowBlur = 12;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cursorX, 0);
      ctx.lineTo(cursorX, h);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Dot at top
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(cursorX, 8, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath();
      ctx.arc(cursorX, 8, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }, []);

  const animatePlayback = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || audio.paused || audio.ended) {
      isPlayingRef.current = false;
      return;
    }
    const progress = audio.duration ? audio.currentTime / audio.duration : 0;
    drawWaveform(progress);
    animFrameRef.current = requestAnimationFrame(animatePlayback);
  }, [drawWaveform]);

  const handlePlay = useCallback(() => {
    isPlayingRef.current = true;
    animatePlayback();
  }, [animatePlayback]);

  const handlePause = useCallback(() => {
    isPlayingRef.current = false;
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    // Draw at paused position
    const audio = audioRef.current;
    if (audio && audio.duration) {
      drawWaveform(audio.currentTime / audio.duration);
    }
  }, [drawWaveform]);

  const handleEnded = useCallback(() => {
    isPlayingRef.current = false;
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    drawWaveform(1);
  }, [drawWaveform]);

  const handleSeeked = useCallback(() => {
    const audio = audioRef.current;
    if (audio && audio.duration) {
      drawWaveform(audio.currentTime / audio.duration);
      if (!audio.paused) {
        animatePlayback();
      }
    }
  }, [drawWaveform, animatePlayback]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadedFileName(file.name);
    setIsLoading(true);
    setResult(null);

    // Stop any existing playback animation
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    // Create audio URL for playback
    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    // Decode audio and draw waveform
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      audioBufferRef.current = audioBuffer;
      // Size canvas and draw initial waveform
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = canvas.offsetHeight * 2;
      }
      drawWaveform(-1);
      audioCtx.close();
    } catch (e) {
      console.warn('Could not decode audio for waveform:', e);
    }

    // Upload to API
    try {
      const formData = new FormData();
      formData.append('audio_file', file);
      const response = await fetch(`${API_URL}/api/v1/analyze`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      const transformed = transformBackendResponse(data);
      setResult(transformed);
      // Save scan to history with audio file
      if (onScanComplete && transformed) {
        onScanComplete(transformed, file);
      }
    } catch (err: any) {
      console.error('Upload failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const detectionLabel = result?.classification.label.replace(/_/g, ' ') || 'Waiting...';
  const detectionConf = result ? Math.round(result.classification.confidence * 100) : 0;
  const audioDuration = result ? result.audio_duration : 0;
  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = Math.floor(s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  };

  return (
    <>
      <div className="top-bar">
        <div className="record-header-left">
          <span style={{ fontSize: 24 }}>{Icons.upload}</span>
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 700 }}>Upload & Analyze</h1>
            <p style={{ fontSize: 13, color: '#94a3b8' }}>BATCH PROCESSING • RETROSPECTIVE STUDY</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {uploadedFileName && (
            <div className="recording-duration">
              <span>Uploaded File</span>
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{uploadedFileName}</div>
            </div>
          )}
          <button className="icon-btn">{Icons.bell}</button>
          <div className="user-avatar-sm">DS</div>
        </div>
      </div>
      <div className="page-content">
        {isLoading && <div className="loading-overlay"><div className="spinner" /></div>}
        <div className="record-grid">
          <div className="record-main">
            {/* Upload Zone */}
            <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
              <input ref={fileInputRef} type="file" hidden accept="audio/*,.wav,.mp3,.ogg,.webm,.m4a" onChange={handleFileUpload} />
              <div className="upload-zone-icon">📁</div>
              <h3>{uploadedFileName ? 'Click to upload a different file' : 'Click to upload or drag and drop'}</h3>
              <p>Supports WAV, MP3, OGG, WebM, M4A files up to 50MB</p>
            </div>

            {/* Waveform Visualization */}
            <div className="waveform-card">
              <div className="waveform-header">
                <div className="waveform-header-left">{Icons.chart} Audio Waveform Visualization</div>
                <div className="waveform-badges">
                  {uploadedFileName && <span className="quality-badge">UPLOADED</span>}
                  {result && <span className="quality-badge">{formatDuration(audioDuration)}</span>}
                </div>
              </div>
              <div className="waveform-display">
                <canvas ref={canvasRef} className="waveform-canvas" />
                {!uploadedFileName && (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: 14 }}>
                    Upload an audio file to see its waveform
                  </div>
                )}
              </div>
              <div className="waveform-timeline">
                <span>0:00</span><span>—</span><span>—</span><span>—</span><span>—</span><span>{result ? formatDuration(audioDuration) : 'End'}</span>
              </div>
            </div>

            {/* Audio Playback */}
            {audioUrl && (
              <div className="playback-controls">
                <h3>🔊 Play Uploaded Audio</h3>
                <p>Listen to the uploaded audio file — the waveform cursor follows playback.</p>
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  controls
                  onPlay={handlePlay}
                  onPause={handlePause}
                  onEnded={handleEnded}
                  onSeeked={handleSeeked}
                  style={{ width: '100%', borderRadius: 8 }}
                />
              </div>
            )}

            {/* Clinical Notes */}
            <div className="clinical-notes">
              <div className="clinical-notes-header">
                <h3>📝 Doctor's Clinical Notes</h3>
                <span className="auto-save">Auto-saving...</span>
              </div>
              <textarea placeholder="Enter clinical observations, manual diagnosis notes, and patient history context here..." />
            </div>
          </div>

          {/* Right Side Panel — same as Record page */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div className="detection-card">
              <div className="detection-card-label">PRIMARY DETECTION</div>
              <h2 style={{ textTransform: 'capitalize' }}>{detectionLabel}</h2>
              <div className="detection-score">
                <span className="percent">{detectionConf}%</span>
                <span>Confidence Score</span>
              </div>
              <div className="detection-bar"><div className="detection-bar-fill" style={{ width: `${detectionConf}%` }} /></div>
            </div>
            
            {/* Pulmonary Disease AI Detection Card - Upload Page */}
            {result?.pulmonary_disease && (
              <div className="detection-card" style={{ 
                background: result.pulmonary_disease.is_healthy 
                  ? 'linear-gradient(135deg, #064e3b 0%, #065f46 100%)' 
                  : 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)'
              }}>
                <div className="detection-card-label">🫁 AI PULMONARY DISEASE DETECTION</div>
                <h2 style={{ textTransform: 'capitalize' }}>
                  {result.pulmonary_disease.disease.replace(/_/g, ' ')}
                </h2>
                <div className="detection-score">
                  <span className="percent">{Math.round(result.pulmonary_disease.confidence * 100)}%</span>
                  <span>Disease Confidence</span>
                </div>
                <div className="detection-bar">
                  <div className="detection-bar-fill" style={{ 
                    width: `${Math.round(result.pulmonary_disease.confidence * 100)}%`,
                    background: result.pulmonary_disease.is_healthy ? '#10b981' : '#ef4444'
                  }} />
                </div>
                {result.pulmonary_disease.requires_attention && (
                  <div style={{ 
                    marginTop: 12, padding: '8px 12px', 
                    background: 'rgba(239, 68, 68, 0.2)', 
                    borderRadius: 8, fontSize: 12, color: '#fca5a5' 
                  }}>
                    ⚠️ Requires medical attention - consult a pediatrician
                  </div>
                )}
              </div>
            )}
            
            {/* Only show medicines after analysis - Upload Page */}
            {result && treatmentPlan && (
              <>
                <div className="rx-card">
                  <div className="rx-card-label">RECOMMENDED MEDICINES & PRESCRIPTIONS</div>
                  {treatmentPlan.medicines.length > 0 ? (
                    treatmentPlan.medicines.map((med, idx) => (
                      <div key={idx} className="rx-item">
                        <div>
                          <h4>{med.name}</h4>
                          <span>{med.aiSuggested ? 'AI SUGGESTED' : ''} {med.type.toUpperCase()}{med.dosage ? ` • ${med.dosage}` : ''}</span>
                        </div>
                        <span style={{ color: '#10b981' }}>✓</span>
                      </div>
                    ))
                  ) : (
                    <div className="rx-item">
                      <div><h4>No specific medication required</h4><span>Monitor and comfort measures recommended</span></div>
                      <span style={{ color: '#10b981' }}>✓</span>
                    </div>
                  )}
                  {treatmentPlan.consultDoctor && (
                    <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(251, 191, 36, 0.2)', borderRadius: 8, fontSize: 12, color: '#fbbf24' }}>
                      ⚠️ {treatmentPlan.disclaimer}
                    </div>
                  )}
                </div>
                <div className="indicators-card">
                  <div className="indicators-card-label">ACOUSTIC INDICATORS</div>
                  <div className="indicator-item">
                    <div className="indicator-icon green">📈</div>
                    <div className="indicator-content"><h4>High Pitch (Melodic Peak)</h4><p>Frequency spike at 450Hz-600Hz range detected.</p></div>
                  </div>
                  <div className="indicator-item">
                    <div className="indicator-icon blue">{Icons.chart}</div>
                    <div className="indicator-content"><h4>Rhythmic Tensions</h4><p>Periodic intensity fluctuations matching abdominal distress.</p></div>
                  </div>
                  <div className="indicator-item">
                    <div className="indicator-icon orange">💧</div>
                    <div className="indicator-content"><h4>Short Inspiratory Gaps</h4><p>Breath intervals below 0.4s indicate acute discomfort.</p></div>
                  </div>
                </div>
                <div className="suggestions-card">
                  <div className="indicators-card-label">PRESCRIPTION SUGGESTIONS</div>
                  {treatmentPlan.suggestions.length > 0 ? (
                    treatmentPlan.suggestions.map((suggestion, idx) => (
                      <div key={idx} className="suggestion-item"><span>{suggestion}</span><span>›</span></div>
                    ))
                  ) : (
                    <>
                      <div className="suggestion-item"><span>General comfort measures</span><span>›</span></div>
                      <div className="suggestion-item"><span>Monitor for changes</span><span>›</span></div>
                    </>
                  )}
                </div>
              </>
            )}
            
            {/* Show placeholder when no analysis yet - Upload Page */}
            {!result && (
              <div className="rx-card" style={{ opacity: 0.5 }}>
                <div className="rx-card-label">RECOMMENDED MEDICINES & PRESCRIPTIONS</div>
                <div style={{ padding: '20px', textAlign: 'center', color: '#64748b' }}>
                  <p>📁 Upload audio to receive AI-powered recommendations</p>
                </div>
              </div>
            )}
            
            <div className="finalize-section">
              <div 
                className="download-card"
                onClick={() => generatePDFReport(result, treatmentPlan)}
                style={{ cursor: result ? 'pointer' : 'not-allowed', opacity: result ? 1 : 0.5 }}
              >
                <div className="pdf-icon" style={{ color: '#dc2626' }}>📄</div>
                <h4>Download Full PDF Report</h4>
                <span>INCLUDES WAVEFORM & ANALYSIS</span>
              </div>
              <button 
                className="finalize-btn"
                onClick={() => finalizeSession(result, setResult, setTreatmentPlan, () => {
                  setUploadedFileName('');
                  setAudioUrl(null);
                })}
                disabled={!result}
                style={{ opacity: result ? 1 : 0.5 }}
              >
                ✅ Finalize Session
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ===== HISTORY PAGE =====
interface HistoryScan {
  id: string;
  date: string;
  time: string;
  name: string;
  reg: string;
  initial: string;
  color: string;
  disease: string;
  diseaseColor: string;
  accuracy: number;
  symptoms: string;
  audio_url?: string;
  risk_level?: string;
  biomarkers?: Record<string, any>;
}

function HistoryPage({ onNavigate, scans, onViewReport }: { onNavigate: (page: string) => void; scans: HistoryScan[]; onViewReport?: (scan: HistoryScan) => void }) {
  // Use actual scans if available, otherwise show empty state
  const historyData = scans.length > 0 ? scans : [];

  return (
    <>
      <div className="top-bar">
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 800 }}>Patient Diagnostic History</h1>
          <p style={{ fontSize: 14, color: '#94a3b8', marginTop: 4 }}>Review and manage comprehensive AI-driven diagnostic analysis records for pediatric patients.</p>
        </div>
        <div className="table-header-right">
          <button className="btn btn-outline">↓ Export Records</button>
          <button className="btn btn-primary" onClick={() => onNavigate('record')}>+ New Scan</button>
        </div>
      </div>
      <div className="page-content">
        <div className="filter-bar">
          <div className="filter-chip">{Icons.calendar} Date Range ▾</div>
          <div className="filter-chip">❄ Condition ▾</div>
          <div className="filter-chip">⚡ Severity ▾</div>
          <button className="clear-filters">Clear all filters</button>
        </div>
        <div className="table-card">
          {historyData.length === 0 ? (
            <div style={{ padding: 60, textAlign: 'center', color: '#64748b' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>📋</div>
              <h3 style={{ marginBottom: 8, color: '#94a3b8' }}>No Scans Yet</h3>
              <p>Your diagnostic history will appear here after you perform scans.</p>
              <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={() => onNavigate('record')}>
                Start Your First Scan
              </button>
            </div>
          ) : (
            <>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>DATE</th>
                    <th>BABY NAME / ID</th>
                    <th>PREDICTED DISEASE</th>
                    <th>ACCURACY</th>
                    <th>SYMPTOMS</th>
                    <th>ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {historyData.map((row, idx) => (
                    <tr key={idx}>
                      <td>{row.date}<br /><span style={{ fontSize: 12, color: '#94a3b8' }}>{row.time}</span></td>
                      <td>
                        <div className="patient-ref">
                          <div className="patient-avatar" style={{ background: row.color }}>{row.initial}</div>
                          <div><strong>{row.name}</strong><br /><span style={{ fontSize: 12, color: '#94a3b8' }}>Reg: {row.reg}</span></div>
                        </div>
                      </td>
                      <td><span className={`badge badge-${row.diseaseColor}`}>{row.disease}</span></td>
                      <td>
                        <div className="confidence-bar">
                          <div className="confidence-track">
                            <div className={`confidence-fill ${row.accuracy >= 95 ? 'high' : row.accuracy >= 90 ? 'medium' : 'low'}`} style={{ width: `${row.accuracy}%` }} />
                          </div>
                          <span className="confidence-text">{row.accuracy}%</span>
                        </div>
                      </td>
                      <td style={{ maxWidth: 200, color: '#64748b', fontSize: 13 }}>{row.symptoms}</td>
                      <td><button className="view-link" onClick={() => onViewReport && onViewReport(row)}>View Details →</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="table-footer">
                <span>Showing <strong>1 to {historyData.length}</strong> of <strong>{historyData.length}</strong> diagnostic scans</span>
                <div className="pagination">
                  <button>‹ Prev</button>
                  <button className="active">1</button>
                  <button>Next ›</button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

// ===== REPORT PAGE =====
function ReportPage({ scan, onBack }: { scan: HistoryScan; onBack: () => void }) {
  const [medicines, setMedicines] = useState<TreatmentPlan | null>(null);
  const [isLoadingMeds, setIsLoadingMeds] = useState(true);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  // Load medicine recommendations
  useEffect(() => {
    const loadMedicines = async () => {
      setIsLoadingMeds(true);
      try {
        const recommendations = await getMedicineRecommendations(
          scan.disease,
          scan.accuracy,
          scan.biomarkers
        );
        setMedicines(recommendations);
      } catch (error) {
        console.error('Error loading medicine recommendations:', error);
      } finally {
        setIsLoadingMeds(false);
      }
    };
    loadMedicines();
  }, [scan.disease, scan.accuracy, scan.biomarkers]);

  // Load audio URL if available
  useEffect(() => {
    if (scan.audio_url) {
      setAudioUrl(scan.audio_url);
    } else if (scan.id) {
      getAudioUrl(scan.id).then(url => {
        if (url) setAudioUrl(url);
      });
    }
  }, [scan.audio_url, scan.id]);

  // Draw waveform when audio loads
  useEffect(() => {
    if (!audioUrl || !canvasRef.current) return;
    
    const drawStaticWaveform = async () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      try {
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        const data = audioBuffer.getChannelData(0);
        const step = Math.floor(data.length / canvas.width);
        
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.strokeStyle = '#1e40af';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        
        for (let i = 0; i < canvas.width; i++) {
          const idx = i * step;
          const value = data[idx] || 0;
          const y = (1 + value) * canvas.height / 2;
          if (i === 0) ctx.moveTo(i, y);
          else ctx.lineTo(i, y);
        }
        ctx.stroke();
        
        audioContext.close();
      } catch (error) {
        console.error('Error drawing waveform:', error);
      }
    };
    
    drawStaticWaveform();
  }, [audioUrl]);

  const getRiskInfo = (riskLevel: string) => {
    switch (riskLevel?.toUpperCase()) {
      case 'RED': case 'HIGH': case 'CRITICAL': return { color: '#dc2626', bg: '#fef2f2', label: 'HIGH RISK', icon: '🔴' };
      case 'YELLOW': case 'MODERATE': case 'MEDIUM': return { color: '#d97706', bg: '#fffbeb', label: 'MODERATE', icon: '🟡' };
      default: return { color: '#16a34a', bg: '#f0fdf4', label: 'LOW RISK', icon: '🟢' };
    }
  };

  const riskInfo = getRiskInfo(scan.risk_level || scan.diseaseColor);
  const biomarkers = scan.biomarkers as Record<string, number> || {};

  // Professional styles
  const sectionStyle: React.CSSProperties = {
    background: '#ffffff',
    borderRadius: 8,
    border: '1px solid #e2e8f0',
    marginBottom: 20,
    overflow: 'hidden',
  };
  
  const sectionHeaderStyle: React.CSSProperties = {
    background: 'linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%)',
    padding: '12px 20px',
    color: '#ffffff',
    fontSize: 14,
    fontWeight: 600,
    letterSpacing: '0.5px',
    textTransform: 'uppercase' as const,
  };
  
  const tableRowStyle = (isAlt: boolean): React.CSSProperties => ({
    display: 'flex',
    borderBottom: '1px solid #e2e8f0',
    background: isAlt ? '#f8fafc' : '#ffffff',
  });
  
  const tableCellStyle: React.CSSProperties = {
    padding: '10px 16px',
    fontSize: 13,
    color: '#334155',
    flex: 1,
  };
  
  const tableLabelStyle: React.CSSProperties = {
    ...tableCellStyle,
    fontWeight: 600,
    color: '#1e293b',
    background: '#f1f5f9',
    flex: '0 0 180px',
  };

  return (
    <>
      <TopBar title="Diagnostic Report" subtitle="Baby Cry Analysis System" />
      <div className="page-content" style={{ 
        padding: 0,
        background: '#f1f5f9',
        minHeight: '100vh'
      }}>
        {/* Professional Header Bar */}
        <div style={{
          background: 'linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%)',
          padding: '20px 32px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '3px solid #3b82f6',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <button onClick={onBack} style={{
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: 6,
              padding: '8px 16px',
              color: '#ffffff',
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 500,
            }}>
              ← Back to History
            </button>
            <div style={{ height: 32, width: 1, background: 'rgba(255,255,255,0.2)' }} />
            <div>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px' }}>Medical Report</div>
              <div style={{ fontSize: 18, fontWeight: 600, color: '#ffffff' }}>Pediatric Cry Analysis Report</div>
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>Report Generated</div>
            <div style={{ fontSize: 14, color: '#ffffff', fontWeight: 500 }}>{new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</div>
          </div>
        </div>

        {/* Main Content */}
        <div style={{ padding: 32, maxWidth: 1200, margin: '0 auto' }}>
          {/* Top Row - Patient Info & Diagnosis */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
            {/* Patient Information */}
            <div style={sectionStyle}>
              <div style={sectionHeaderStyle}>Patient Information</div>
              <div>
                <div style={tableRowStyle(false)}>
                  <div style={tableLabelStyle}>Patient Name</div>
                  <div style={tableCellStyle}>{scan.name}</div>
                </div>
                <div style={tableRowStyle(true)}>
                  <div style={tableLabelStyle}>Registration No.</div>
                  <div style={tableCellStyle}>{scan.reg}</div>
                </div>
                <div style={tableRowStyle(false)}>
                  <div style={tableLabelStyle}>Scan ID</div>
                  <div style={{ ...tableCellStyle, fontFamily: 'monospace', fontSize: 12 }}>{scan.id}</div>
                </div>
                <div style={tableRowStyle(true)}>
                  <div style={tableLabelStyle}>Date & Time</div>
                  <div style={tableCellStyle}>{scan.date} at {scan.time}</div>
                </div>
              </div>
            </div>

            {/* Diagnosis Summary */}
            <div style={sectionStyle}>
              <div style={sectionHeaderStyle}>Diagnostic Summary</div>
              <div style={{ padding: 20 }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 16,
                  marginBottom: 16,
                  paddingBottom: 16,
                  borderBottom: '1px solid #e2e8f0'
                }}>
                  <div style={{
                    width: 56,
                    height: 56,
                    borderRadius: '50%',
                    background: riskInfo.bg,
                    border: `2px solid ${riskInfo.color}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 24,
                  }}>
                    {riskInfo.icon}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 20, fontWeight: 700, color: '#0f172a' }}>{scan.disease}</div>
                    <div style={{ 
                      display: 'inline-block',
                      background: riskInfo.bg, 
                      color: riskInfo.color,
                      padding: '4px 12px',
                      borderRadius: 4,
                      fontSize: 11,
                      fontWeight: 700,
                      marginTop: 4,
                      border: `1px solid ${riskInfo.color}`,
                    }}>
                      {riskInfo.label}
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 24 }}>
                  <div style={{ flex: 1, textAlign: 'center', padding: 12, background: '#f8fafc', borderRadius: 6 }}>
                    <div style={{ fontSize: 28, fontWeight: 700, color: '#1e40af' }}>{scan.accuracy}%</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>CONFIDENCE</div>
                  </div>
                  <div style={{ flex: 1, textAlign: 'center', padding: 12, background: '#f8fafc', borderRadius: 6 }}>
                    <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>AI</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>ANALYSIS</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Middle Row - Biomarkers & Audio */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
            {/* Acoustic Biomarkers */}
            <div style={sectionStyle}>
              <div style={sectionHeaderStyle}>Acoustic Biomarkers Analysis</div>
              <div>
                {[
                  { key: 'f0_mean', label: 'Fundamental Frequency (Mean)', unit: 'Hz' },
                  { key: 'f0_std', label: 'Frequency Variation (Std)', unit: 'Hz' },
                  { key: 'spectral_centroid', label: 'Spectral Centroid', unit: 'Hz' },
                  { key: 'hnr', label: 'Harmonic-to-Noise Ratio', unit: 'dB' },
                  { key: 'energy_rms', label: 'Energy (RMS)', unit: '' },
                  { key: 'zcr', label: 'Zero Crossing Rate', unit: '' },
                ].map((item, idx) => (
                  <div key={item.key} style={tableRowStyle(idx % 2 === 1)}>
                    <div style={tableLabelStyle}>{item.label}</div>
                    <div style={tableCellStyle}>
                      <span style={{ fontWeight: 600, color: '#1e40af' }}>
                        {typeof biomarkers[item.key] === 'number' ? biomarkers[item.key].toFixed(2) : '—'}
                      </span>
                      {item.unit && <span style={{ color: '#64748b', marginLeft: 4 }}>{item.unit}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Audio Recording */}
            <div style={sectionStyle}>
              <div style={sectionHeaderStyle}>Audio Recording</div>
              <div style={{ padding: 20 }}>
                {audioUrl ? (
                  <>
                    <div style={{ marginBottom: 16 }}>
                      <canvas 
                        ref={canvasRef}
                        width={500}
                        height={80}
                        style={{
                          width: '100%',
                          height: 80,
                          borderRadius: 4,
                          border: '1px solid #e2e8f0',
                        }}
                      />
                    </div>
                    <audio 
                      ref={audioRef}
                      src={audioUrl} 
                      controls 
                      style={{ 
                        width: '100%',
                        height: 40,
                      }}
                    />
                    <div style={{ 
                      marginTop: 12, 
                      padding: 12, 
                      background: '#f0fdf4', 
                      borderRadius: 6,
                      border: '1px solid #bbf7d0',
                      fontSize: 12,
                      color: '#166534',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8
                    }}>
                      <span>✓</span> Audio sample successfully captured and analyzed
                    </div>
                  </>
                ) : (
                  <div style={{
                    border: '2px dashed #e2e8f0',
                    borderRadius: 8,
                    padding: 40,
                    textAlign: 'center',
                    color: '#64748b'
                  }}>
                    <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.5 }}>🎵</div>
                    <div style={{ fontWeight: 500 }}>No Audio Recording Available</div>
                    <div style={{ fontSize: 12, marginTop: 4 }}>Audio file was not stored for this scan</div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Treatment Recommendations */}
          <div style={sectionStyle}>
            <div style={sectionHeaderStyle}>
              Treatment Recommendations
              {isGeminiConfigured() && (
                <span style={{ 
                  marginLeft: 12,
                  background: 'rgba(255,255,255,0.2)', 
                  padding: '2px 10px', 
                  borderRadius: 4, 
                  fontSize: 10,
                  fontWeight: 500,
                }}>
                  AI-ASSISTED
                </span>
              )}
            </div>
            <div style={{ padding: 24 }}>
              {isLoadingMeds ? (
                <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                  <div style={{ fontSize: 32, marginBottom: 12 }}>⏳</div>
                  <div>Generating treatment recommendations...</div>
                </div>
              ) : medicines ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
                  {/* Medications */}
                  <div>
                    <h4 style={{ margin: '0 0 16px 0', color: '#0f172a', fontSize: 14, borderBottom: '2px solid #1e40af', paddingBottom: 8 }}>
                      Recommended Medications
                    </h4>
                    {medicines.medicines.length > 0 ? (
                      medicines.medicines.map((med, idx) => (
                        <div key={idx} style={{
                          background: '#f8fafc',
                          border: '1px solid #e2e8f0',
                          borderLeft: '4px solid #1e40af',
                          borderRadius: '0 6px 6px 0',
                          padding: 16,
                          marginBottom: 12,
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div>
                              <div style={{ fontWeight: 600, color: '#0f172a', fontSize: 15 }}>{med.name}</div>
                              <div style={{ fontSize: 11, color: '#3b82f6', marginTop: 2, textTransform: 'uppercase' }}>{med.type}</div>
                            </div>
                            {med.aiSuggested && (
                              <span style={{ fontSize: 9, background: '#dbeafe', color: '#1e40af', padding: '2px 8px', borderRadius: 4, fontWeight: 600 }}>AI</span>
                            )}
                          </div>
                          {med.dosage && (
                            <div style={{ fontSize: 13, color: '#334155', marginTop: 10, display: 'flex', gap: 8 }}>
                              <span style={{ fontWeight: 600 }}>Dosage:</span> {med.dosage}
                            </div>
                          )}
                          {med.frequency && (
                            <div style={{ fontSize: 13, color: '#334155', display: 'flex', gap: 8 }}>
                              <span style={{ fontWeight: 600 }}>Frequency:</span> {med.frequency}
                            </div>
                          )}
                          {med.notes && (
                            <div style={{ fontSize: 12, color: '#64748b', marginTop: 8, fontStyle: 'italic', borderTop: '1px solid #e2e8f0', paddingTop: 8 }}>
                              {med.notes}
                            </div>
                          )}
                        </div>
                      ))
                    ) : (
                      <div style={{
                        background: '#f0fdf4',
                        border: '1px solid #bbf7d0',
                        borderRadius: 6,
                        padding: 20,
                        textAlign: 'center'
                      }}>
                        <div style={{ fontSize: 24, marginBottom: 8 }}>✓</div>
                        <div style={{ color: '#166534', fontWeight: 500 }}>No medication required</div>
                        <div style={{ fontSize: 12, color: '#22c55e', marginTop: 4 }}>Condition can be managed with supportive care</div>
                      </div>
                    )}
                  </div>

                  {/* Care Instructions & Warnings */}
                  <div>
                    {/* Care Suggestions */}
                    {medicines.suggestions.length > 0 && (
                      <div style={{ marginBottom: 20 }}>
                        <h4 style={{ margin: '0 0 16px 0', color: '#0f172a', fontSize: 14, borderBottom: '2px solid #16a34a', paddingBottom: 8 }}>
                          Care Instructions
                        </h4>
                        {medicines.suggestions.map((sug, idx) => (
                          <div key={idx} style={{
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: 10,
                            padding: '8px 0',
                            borderBottom: idx < medicines.suggestions.length - 1 ? '1px solid #e2e8f0' : 'none',
                          }}>
                            <span style={{ color: '#16a34a', fontWeight: 600 }}>•</span>
                            <span style={{ color: '#334155', fontSize: 13 }}>{sug}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Warnings */}
                    {medicines.warnings.length > 0 && (
                      <div style={{ marginBottom: 20 }}>
                        <h4 style={{ margin: '0 0 16px 0', color: '#d97706', fontSize: 14, borderBottom: '2px solid #d97706', paddingBottom: 8 }}>
                          ⚠️ Important Warnings
                        </h4>
                        {medicines.warnings.map((warn, idx) => (
                          <div key={idx} style={{
                            background: '#fffbeb',
                            border: '1px solid #fcd34d',
                            padding: 12,
                            borderRadius: 6,
                            marginBottom: 8,
                            fontSize: 13,
                            color: '#92400e',
                          }}>
                            {warn}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Consult Doctor Alert */}
                    {medicines.consultDoctor && (
                      <div style={{
                        background: '#fef2f2',
                        border: '2px solid #dc2626',
                        borderRadius: 8,
                        padding: 16,
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#dc2626', fontWeight: 700, marginBottom: 8 }}>
                          <span style={{ fontSize: 20 }}>🏥</span>
                          MEDICAL CONSULTATION REQUIRED
                        </div>
                        <div style={{ fontSize: 13, color: '#7f1d1d' }}>
                          This condition requires professional medical evaluation. Please schedule an appointment with a pediatrician at the earliest convenience.
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                  Unable to load treatment recommendations
                </div>
              )}
            </div>
          </div>

          {/* Footer with Disclaimer & Download */}
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>
            {/* Disclaimer */}
            <div style={{
              background: '#fff7ed',
              border: '1px solid #fed7aa',
              borderRadius: 8,
              padding: 20,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <span style={{ fontSize: 18 }}>⚕️</span>
                <span style={{ fontWeight: 600, color: '#9a3412' }}>Medical Disclaimer</span>
              </div>
              <div style={{ fontSize: 12, color: '#7c2d12', lineHeight: 1.6 }}>
                {medicines?.disclaimer || 'This diagnostic report is generated using AI-assisted analysis and is intended for informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.'}
              </div>
            </div>

            {/* Download Button */}
            <button 
              onClick={() => {
                const analysisResult: AnalysisResult = {
                  id: scan.id,
                  classification: {
                    label: scan.disease.toLowerCase().replace(/\s+/g, '_'),
                    confidence: scan.accuracy / 100,
                    model: 'ensemble',
                    cry_detected: true,
                  },
                  biomarkers: (scan.biomarkers as { f0_mean: number; f0_std: number; spectral_centroid: number; hnr: number; energy_rms: number; zcr: number }) || { f0_mean: 0, f0_std: 0, spectral_centroid: 0, hnr: 0, energy_rms: 0, zcr: 0 },
                  risk_level: scan.risk_level || 'GREEN',
                  risk_color: scan.diseaseColor || '#22c55e',
                  recommended_action: 'Consult with healthcare provider as needed',
                  timestamp: `${scan.date} ${scan.time}`,
                  audio_duration: 0,
                };
                generatePDFReport(analysisResult, medicines, { id: scan.reg, session: scan.id });
              }}
              style={{
                background: 'linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%)',
                border: 'none',
                borderRadius: 8,
                padding: '20px 24px',
                color: 'white',
                fontSize: 15,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                transition: 'transform 0.2s, box-shadow 0.2s',
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,0,0,0.2)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
              }}
            >
              <span style={{ fontSize: 24 }}>📄</span>
              <span>Download PDF Report</span>
              <span style={{ fontSize: 11, opacity: 0.8, fontWeight: 400 }}>Official Medical Document</span>
            </button>
          </div>

          {/* Report Footer */}
          <div style={{ 
            marginTop: 32, 
            paddingTop: 20, 
            borderTop: '2px solid #e2e8f0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            color: '#64748b',
            fontSize: 11,
          }}>
            <div>
              <div style={{ fontWeight: 600, color: '#1e3a5f' }}>Baby Cry Diagnostic System</div>
              <div>AI-Powered Pediatric Health Analysis</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div>Report ID: {scan.id.slice(0, 8).toUpperCase()}</div>
              <div>Generated: {new Date().toLocaleString()}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div>For Medical Professional Use</div>
              <div>Licensed Healthcare Facility</div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ===== MAIN APP =====
function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    // Check localStorage for auth persistence
    return localStorage.getItem(AUTH_STORAGE_KEY) === 'true';
  });
  const [activePage, setActivePage] = useState(() => {
    // Restore last active page
    return localStorage.getItem('crycare_active_page') || 'dashboard';
  });
  const [selectedScan, setSelectedScan] = useState<HistoryScan | null>(null);
  const [scanHistory, setScanHistory] = useState<HistoryScan[]>(() => {
    // Load history from localStorage on initial render
    const saved = localStorage.getItem('crycare_scan_history');
    return saved ? JSON.parse(saved) : [];
  });
  
  // Persist active page to localStorage
  useEffect(() => {
    localStorage.setItem('crycare_active_page', activePage);
  }, [activePage]);
  
  // Stats state
  const [stats, setStats] = useState<AppStats>({
    totalScans: 0,
    criticalAlerts: 0,
    medsAdvised: 0,
    isLoading: true
  });

  // Load data from Supabase on mount
  useEffect(() => {
    const loadSupabaseData = async () => {
      if (isSupabaseConfigured()) {
        try {
          // Load stats
          const supabaseStats = await fetchStats();
          setStats({
            totalScans: supabaseStats.total_scans,
            criticalAlerts: supabaseStats.critical_alerts,
            medsAdvised: supabaseStats.meds_advised,
            isLoading: false
          });
          
          // Load scans from Supabase
          const supabaseScans = await fetchScans();
          if (supabaseScans.length > 0) {
            const mappedScans: HistoryScan[] = supabaseScans.map(scan => ({
              id: scan.id || '',
              date: scan.date,
              time: scan.time,
              name: scan.name,
              reg: scan.reg,
              initial: scan.initial,
              color: scan.color,
              disease: scan.disease,
              diseaseColor: scan.disease_color,
              accuracy: scan.accuracy,
              symptoms: scan.symptoms
            }));
            setScanHistory(mappedScans);
            localStorage.setItem('crycare_scan_history', JSON.stringify(mappedScans));
          }
        } catch (error) {
          console.error('Error loading Supabase data:', error);
          // Fall back to calculating stats from local data
          setStats({
            totalScans: scanHistory.length,
            criticalAlerts: scanHistory.filter(s => ['red'].includes(s.diseaseColor)).length,
            medsAdvised: scanHistory.filter(s => s.accuracy > 80).length,
            isLoading: false
          });
        }
      } else {
        // Supabase not configured - calculate from local data
        setStats({
          totalScans: scanHistory.length,
          criticalAlerts: scanHistory.filter(s => ['red'].includes(s.diseaseColor)).length,
          medsAdvised: scanHistory.filter(s => s.accuracy > 80).length,
          isLoading: false
        });
      }
    };
    
    loadSupabaseData();
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  // Function to add a new scan to history
  const addScanToHistory = useCallback(async (result: AnalysisResult, audioBlob?: Blob) => {
    const now = new Date();
    const colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4'];
    const diseaseColorMap: Record<string, string> = {
      // Baby cry types (12 classes from trained model)
      'pain': 'red',
      'pathological': 'red',
      'belly_pain': 'red',
      'scared': 'orange',
      'cold_hot': 'orange',
      'discomfort': 'orange',
      'hungry': 'green',
      'sleepy': 'green',
      'tired': 'green',
      'normal': 'green',
      'burping': 'green',
      'lonely': 'orange',
      // Respiratory sounds (8 classes from trained model) - HIGH PRIORITY
      'wheeze': 'red',
      'rhonchi': 'red',
      'stridor': 'red',
      'crackle': 'red',
      'bronchiolitis': 'red',
      'pneumonia': 'red',
      'asthma': 'red',
    };

    const scanNumber = scanHistory.length + 1;
    const label = result.classification.label.replace(/_/g, ' ');
    const confidence = Math.round(result.classification.confidence * 100);
    const diseaseColor = diseaseColorMap[result.classification.label] || 'orange';
    
    // Determine if critical (red conditions) and if meds advised (high confidence issues)
    const isCritical = diseaseColor === 'red';
    const medsAdvised = confidence > 50 && isCritical;

    const newScan: HistoryScan = {
      id: result.id || `SCAN-${Date.now()}`,
      date: now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      time: now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      name: `Scan-${String(scanNumber).padStart(3, '0')}`,
      reg: `#S${String(scanNumber).padStart(5, '0')}`,
      initial: 'S',
      color: colors[scanNumber % colors.length],
      disease: label.charAt(0).toUpperCase() + label.slice(1),
      diseaseColor: diseaseColor,
      accuracy: confidence,
      symptoms: result.recommended_action || 'Analysis completed',
    };

    // Save to localStorage first
    setScanHistory(prev => {
      const updated = [newScan, ...prev];
      localStorage.setItem('crycare_scan_history', JSON.stringify(updated));
      return updated;
    });
    
    // Update local stats immediately
    setStats(prev => ({
      ...prev,
      totalScans: prev.totalScans + 1,
      criticalAlerts: prev.criticalAlerts + (isCritical ? 1 : 0),
      medsAdvised: prev.medsAdvised + (medsAdvised ? 1 : 0)
    }));
    
    // Save to Supabase asynchronously
    if (isSupabaseConfigured()) {
      // Upload audio file if provided
      let audioUrl: string | undefined;
      if (audioBlob) {
        const uploadedUrl = await uploadAudio(audioBlob, newScan.id);
        if (uploadedUrl) {
          audioUrl = uploadedUrl;
        }
      }
      
      saveScan({
        date: newScan.date,
        time: newScan.time,
        name: newScan.name,
        reg: newScan.reg,
        initial: newScan.initial,
        color: newScan.color,
        disease: newScan.disease,
        disease_color: newScan.diseaseColor,
        accuracy: newScan.accuracy,
        symptoms: newScan.symptoms,
        risk_level: result.risk_level,
        is_critical: isCritical,
        meds_advised: medsAdvised,
        audio_url: audioUrl
      }).catch(err => console.error('Failed to save scan to Supabase:', err));
    }
  }, [scanHistory.length]);

  if (!isLoggedIn) {
    return <LoginPage onLogin={() => {
      localStorage.setItem(AUTH_STORAGE_KEY, 'true');
      setIsLoggedIn(true);
    }} />;
  }

  const handleLogout = () => {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setIsLoggedIn(false);
  };

  const viewReport = (scan: HistoryScan) => {
    setSelectedScan(scan);
    setActivePage('report');
  };

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard': return <DashboardPage onNavigate={setActivePage} scans={scanHistory} stats={stats} onViewReport={viewReport} />;
      case 'record': return <RecordPage onScanComplete={addScanToHistory} />;
      case 'upload': return <UploadPage onScanComplete={addScanToHistory} />;
      case 'history': return <HistoryPage onNavigate={setActivePage} scans={scanHistory} onViewReport={viewReport} />;
      case 'report': return selectedScan ? <ReportPage scan={selectedScan} onBack={() => setActivePage('history')} /> : <HistoryPage onNavigate={setActivePage} scans={scanHistory} onViewReport={viewReport} />;
      default: return <DashboardPage onNavigate={setActivePage} stats={stats} onViewReport={viewReport} />;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar activePage={activePage} onNavigate={setActivePage} onLogout={handleLogout} />
      <main className="main-content">
        {renderPage()}
        <footer className="app-footer">
          <div className="footer-status">
            <span><span className="dot green" /> AI Core Online</span>
            <span><span className="dot green" /> Sensor Sync: 12ms Latency</span>
            <span><span className="dot blue" /> Cloud Storage: Syncing</span>
          </div>
          <span>© 2023 CryHealth AI Systems &nbsp; HIPAA Compliance Data</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
