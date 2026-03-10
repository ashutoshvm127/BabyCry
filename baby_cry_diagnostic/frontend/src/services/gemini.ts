/**
 * Gemini API Service for Medicine Recommendations
 * Uses Google's Gemini AI to provide dynamic medicine suggestions based on diagnosis
 */

const GEMINI_API_KEY = process.env.REACT_APP_GEMINI_API_KEY || '';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent';

export interface MedicineRecommendation {
  name: string;
  type: string;
  dosage?: string;
  frequency?: string;
  notes?: string;
  aiSuggested: boolean;
}

export interface TreatmentPlan {
  medicines: MedicineRecommendation[];
  suggestions: string[];
  warnings: string[];
  consultDoctor: boolean;
  disclaimer: string;
}

// Fallback recommendations for when API is unavailable
// Keys match model output class names (12 cry classes + 8 pulmonary classes)
const FALLBACK_RECOMMENDATIONS: Record<string, TreatmentPlan> = {
  // CRY CLASSES (12 classes from trained model)
  'pain': {
    medicines: [
      { name: 'Infant Acetaminophen (Tylenol)', type: 'Analgesic', dosage: 'Per pediatrician guidance', notes: 'For pain relief', aiSuggested: false },
      { name: 'Simethicone Drops', type: 'Anti-gas', dosage: '0.3mL before feeds', notes: 'If colic suspected', aiSuggested: false }
    ],
    suggestions: ['Check for signs of ear infection', 'Monitor temperature', 'Gentle rocking motion'],
    warnings: ['Seek immediate care if pain persists >30 minutes', 'Watch for fever >100.4°F'],
    consultDoctor: true,
    disclaimer: 'Consult a pediatrician before administering any medication'
  },
  'pathological': {
    medicines: [
      { name: 'Gripe Water', type: 'Digestive Aid', dosage: 'Per package directions', notes: 'For stomach discomfort', aiSuggested: false }
    ],
    suggestions: ['Swaddling technique', 'White noise', 'Check for hunger/diaper needs', 'Skin-to-skin contact'],
    warnings: ['URGENT: May indicate serious condition', 'Monitor breathing rate', 'Seek immediate medical care'],
    consultDoctor: true,
    disclaimer: 'URGENT: Consult a pediatrician immediately'
  },
  'belly_pain': {
    medicines: [
      { name: 'Simethicone Drops', type: 'Anti-gas', dosage: '0.3mL', notes: 'Relieves gas', aiSuggested: false },
      { name: 'Gripe Water', type: 'Digestive Aid', dosage: 'Per package', notes: 'For colic', aiSuggested: false }
    ],
    suggestions: ['Bicycle legs exercise', 'Gentle tummy massage', 'Warm compress on belly', 'Upright position after feeds'],
    warnings: ['Seek care if lasts >3 hours/day for 3+ days', 'Watch for blood in stool'],
    consultDoctor: true,
    disclaimer: 'Consult a pediatrician before administering any medication'
  },
  'cold_hot': {
    medicines: [],
    suggestions: ['Check baby temperature', 'Adjust clothing layers', 'Check room temperature (68-72°F ideal)', 'Feel back of neck for warmth'],
    warnings: ['Overheating increases SIDS risk', 'Hypothermia signs: blue lips, lethargy'],
    consultDoctor: false,
    disclaimer: 'Temperature regulation is critical for infants'
  },
  'hungry': {
    medicines: [],
    suggestions: ['Feed immediately', 'Check latch if breastfeeding', 'Ensure proper formula preparation', 'Follow demand feeding'],
    warnings: ['Track feeding times and amounts'],
    consultDoctor: false,
    disclaimer: 'No medication typically needed - ensure adequate nutrition'
  },
  'sleepy': {
    medicines: [],
    suggestions: ['Create calm environment', 'Dim lights', 'Gentle swaying', 'White noise machine'],
    warnings: [],
    consultDoctor: false,
    disclaimer: 'No medication needed - establish sleep routine'
  },
  'tired': {
    medicines: [],
    suggestions: ['Reduce stimulation', 'Darken room', 'Swaddling', 'Gentle rocking', 'White noise'],
    warnings: ['Overtired babies may have difficulty settling'],
    consultDoctor: false,
    disclaimer: 'Help baby settle to sleep with consistent routine'
  },
  'scared': {
    medicines: [],
    suggestions: ['Comfort with gentle voice', 'Hold close', 'Remove startling stimuli', 'Skin-to-skin contact'],
    warnings: [],
    consultDoctor: false,
    disclaimer: 'Reassurance and comfort typically sufficient'
  },
  'lonely': {
    medicines: [],
    suggestions: ['Pick up and hold baby', 'Talk or sing to baby', 'Engage with eye contact', 'Gentle play'],
    warnings: [],
    consultDoctor: false,
    disclaimer: 'Baby needs social interaction and comfort'
  },
  'burping': {
    medicines: [],
    suggestions: ['Pat back gently', 'Hold upright over shoulder', 'Sit baby up and support chin', 'Gentle tummy massage'],
    warnings: ['Frequent spitting may indicate reflux'],
    consultDoctor: false,
    disclaimer: 'Normal after feeding - help baby release gas'
  },
  'discomfort': {
    medicines: [
      { name: 'Simethicone Drops', type: 'Anti-gas', dosage: '0.3mL', notes: 'For gas/colic', aiSuggested: false },
      { name: 'Diaper Rash Cream', type: 'Topical', dosage: 'Apply thin layer', notes: 'If rash present', aiSuggested: false }
    ],
    suggestions: ['Check diaper', 'Adjust clothing temperature', 'Bicycle legs for gas', 'Tummy massage'],
    warnings: [],
    consultDoctor: false,
    disclaimer: 'Consult a pediatrician before administering any medication'
  },
  'normal': {
    medicines: [],
    suggestions: ['Continue normal care routine', 'Regular feeding and sleep schedule', 'Monitor for changes'],
    warnings: [],
    consultDoctor: false,
    disclaimer: 'Normal sounds detected - continue routine monitoring'
  },
  // PULMONARY CLASSES (8 classes from trained model)
  'wheeze': {
    medicines: [
      { name: 'Saline Nebulizer', type: 'Respiratory', dosage: 'As prescribed', notes: 'Opens airways', aiSuggested: false },
      { name: 'Albuterol (if prescribed)', type: 'Bronchodilator', dosage: 'Per prescription only', notes: 'Requires prescription', aiSuggested: false }
    ],
    suggestions: ['Keep baby upright', 'Use humidifier', 'Clear nasal passages', 'Monitor oxygen levels if possible'],
    warnings: ['Seek immediate care for severe wheezing', 'Watch for retractions (chest pulling in)', 'Blue lips = Emergency'],
    consultDoctor: true,
    disclaimer: 'RESPIRATORY SYMPTOMS REQUIRE MEDICAL EVALUATION'
  },
  'pneumonia': {
    medicines: [
      { name: 'Prescribed Antibiotics', type: 'Antibiotic', dosage: 'Per prescription', notes: 'If bacterial', aiSuggested: false },
      { name: 'Acetaminophen', type: 'Fever Reducer', dosage: 'Weight-based', notes: 'For fever management', aiSuggested: false }
    ],
    suggestions: ['Keep baby hydrated', 'Rest', 'Monitor breathing', 'Follow-up appointments'],
    warnings: ['REQUIRES IMMEDIATE MEDICAL CARE', 'Complete full antibiotic course', 'Monitor oxygen saturation'],
    consultDoctor: true,
    disclaimer: 'PNEUMONIA REQUIRES PROFESSIONAL MEDICAL TREATMENT - SEE DOCTOR IMMEDIATELY'
  },
  'bronchiolitis': {
    medicines: [
      { name: 'Saline Drops', type: 'Decongestant', dosage: '2-3 drops', notes: 'Clears mucus', aiSuggested: false },
      { name: 'Acetaminophen', type: 'Fever Reducer', dosage: 'If needed', notes: 'For comfort', aiSuggested: false }
    ],
    suggestions: ['Humidified air', 'Frequent small feeds', 'Nasal suctioning', 'Elevate sleeping position'],
    warnings: ['Watch for dehydration', 'Monitor breathing effort', 'ER if breathing rate >60/min'],
    consultDoctor: true,
    disclaimer: 'Bronchiolitis requires monitoring - consult pediatrician'
  },
  'asthma': {
    medicines: [
      { name: 'Inhaled Corticosteroids (if prescribed)', type: 'Anti-inflammatory', dosage: 'Per prescription', notes: 'Controller medication', aiSuggested: false },
      { name: 'Albuterol Inhaler (if prescribed)', type: 'Bronchodilator', dosage: 'Per prescription', notes: 'Rescue medication', aiSuggested: false }
    ],
    suggestions: ['Remove allergen triggers', 'Use air purifier', 'Keep humidity controlled', 'Follow asthma action plan'],
    warnings: ['Severe attacks require ER', 'Monitor for breathing difficulty', 'Blue lips = Emergency'],
    consultDoctor: true,
    disclaimer: 'ASTHMA REQUIRES ONGOING MEDICAL MANAGEMENT'
  },
  'rhonchi': {
    medicines: [
      { name: 'Saline Drops', type: 'Decongestant', dosage: '2-3 drops per nostril', notes: 'Clears mucus', aiSuggested: false },
      { name: 'Chest Physiotherapy', type: 'Physical Therapy', dosage: 'Gentle percussion', notes: 'Helps clear secretions', aiSuggested: false }
    ],
    suggestions: ['Keep airway clear', 'Use humidifier', 'Ensure adequate hydration', 'Position baby slightly elevated'],
    warnings: ['Monitor for breathing difficulty', 'Watch for fever', 'Seek care if accompanied by rapid breathing'],
    consultDoctor: true,
    disclaimer: 'RHONCHI indicates airway secretions - Consult pediatrician'
  },
  'stridor': {
    medicines: [
      { name: 'Cool Mist', type: 'Respiratory Support', dosage: 'As needed', notes: 'Reduces swelling', aiSuggested: false },
      { name: 'Dexamethasone (if prescribed)', type: 'Corticosteroid', dosage: 'Per prescription only', notes: 'For croup - prescription required', aiSuggested: false }
    ],
    suggestions: ['Keep child calm', 'Cool night air may help', 'Sit upright', 'Do NOT examine throat'],
    warnings: ['STRIDOR IS URGENT - Upper airway obstruction', 'Call 911 if severe distress', 'Drooling + stridor = ER immediately'],
    consultDoctor: true,
    disclaimer: 'STRIDOR REQUIRES IMMEDIATE MEDICAL ATTENTION - Go to ER'
  },
  'crackle': {
    medicines: [
      { name: 'Antibiotics (if prescribed)', type: 'Antibiotic', dosage: 'Per prescription', notes: 'If bacterial infection suspected', aiSuggested: false },
      { name: 'Acetaminophen', type: 'Fever/Pain', dosage: 'Weight-based dosing', notes: 'For comfort', aiSuggested: false }
    ],
    suggestions: ['Monitor breathing closely', 'Keep baby hydrated', 'Rest and comfort', 'Follow up with pediatrician'],
    warnings: ['Crackles may indicate fluid in lungs', 'Watch for worsening symptoms', 'Fever + crackles needs evaluation'],
    consultDoctor: true,
    disclaimer: 'LUNG CRACKLES NEED EVALUATION - Consult pediatrician promptly'
  },
  'fine_crackle': {
    medicines: [
      { name: 'Monitor only', type: 'Observation', dosage: 'N/A', notes: 'May resolve with treatment of underlying cause', aiSuggested: false }
    ],
    suggestions: ['Monitor breathing', 'Keep follow-up appointments', 'Ensure adequate hydration'],
    warnings: ['Fine crackles may indicate early lung involvement', 'Monitor for worsening'],
    consultDoctor: true,
    disclaimer: 'Fine crackles require medical evaluation'
  },
  'coarse_crackle': {
    medicines: [
      { name: 'Prescribed treatment', type: 'As directed', dosage: 'Per prescription', notes: 'Based on underlying cause', aiSuggested: false },
      { name: 'Chest Physiotherapy', type: 'Physical Therapy', dosage: 'As directed', notes: 'Helps clear secretions', aiSuggested: false }
    ],
    suggestions: ['Airway clearance techniques', 'Adequate hydration', 'Rest', 'Position changes'],
    warnings: ['Coarse crackles indicate significant secretions', 'Monitor for respiratory distress'],
    consultDoctor: true,
    disclaimer: 'COARSE CRACKLES NEED PROMPT MEDICAL ATTENTION'
  },
  'mixed': {
    medicines: [
      { name: 'As prescribed by physician', type: 'Multiple', dosage: 'Per prescription', notes: 'Combination therapy may be needed', aiSuggested: false }
    ],
    suggestions: ['Comprehensive evaluation needed', 'Monitor all respiratory signs', 'Keep detailed symptom diary'],
    warnings: ['Mixed sounds indicate multiple respiratory issues', 'Requires thorough evaluation'],
    consultDoctor: true,
    disclaimer: 'MIXED RESPIRATORY SOUNDS REQUIRE COMPREHENSIVE EVALUATION'
  },
  'normal_breathing': {
    medicines: [],
    suggestions: ['Continue routine care', 'Maintain good sleep hygiene', 'Regular feeding schedule'],
    warnings: [],
    consultDoctor: false,
    disclaimer: 'Normal respiratory sounds detected - continue routine monitoring'
  },
  'default': {
    medicines: [],
    suggestions: ['Monitor baby closely', 'Ensure comfort', 'Check basic needs (feed, diaper, sleep)'],
    warnings: ['Seek medical attention if symptoms worsen'],
    consultDoctor: false,
    disclaimer: 'Always consult a healthcare provider for medical concerns'
  }
};

/**
 * Get medicine recommendations using Gemini API
 */
export const getMedicineRecommendations = async (
  diagnosis: string,
  confidence: number,
  biomarkers?: Record<string, any>
): Promise<TreatmentPlan> => {
  // Check if API key is configured
  if (!GEMINI_API_KEY) {
    console.warn('Gemini API key not configured, using fallback recommendations');
    return getFallbackRecommendations(diagnosis);
  }

  const prompt = `You are a pediatric medical assistant AI. Based on the following baby cry analysis diagnosis, provide medicine recommendations.

DIAGNOSIS: ${diagnosis}
CONFIDENCE: ${confidence}%
${biomarkers ? `BIOMARKERS: ${JSON.stringify(biomarkers)}` : ''}

IMPORTANT: This is for INFANT (0-12 months) care only.

Respond ONLY with a valid JSON object in this exact format:
{
  "medicines": [
    {"name": "Medicine Name", "type": "Type", "dosage": "Dosage", "frequency": "How often", "notes": "Special notes"}
  ],
  "suggestions": ["Non-medical suggestion 1", "suggestion 2"],
  "warnings": ["Warning 1", "Warning 2"],
  "consultDoctor": true/false,
  "disclaimer": "Medical disclaimer text"
}

Rules:
1. Only recommend OTC infant-safe medicines
2. Always include dosage guidance
3. Set consultDoctor to true for serious conditions
4. Include appropriate warnings
5. Keep it concise - max 3 medicines, 4 suggestions, 3 warnings`;

  try {
    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: prompt }]
        }],
        generationConfig: {
          temperature: 0.3,
          maxOutputTokens: 1024,
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.status}`);
    }

    const data = await response.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    
    // Extract JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No JSON found in response');
    }

    const parsed = JSON.parse(jsonMatch[0]);
    
    // Add aiSuggested flag to medicines
    const medicines = (parsed.medicines || []).map((m: any) => ({
      ...m,
      aiSuggested: true
    }));

    return {
      medicines,
      suggestions: parsed.suggestions || [],
      warnings: parsed.warnings || [],
      consultDoctor: parsed.consultDoctor ?? true,
      disclaimer: parsed.disclaimer || 'AI-generated recommendations. Consult a healthcare provider.'
    };
  } catch (error) {
    console.error('Gemini API error:', error);
    return getFallbackRecommendations(diagnosis);
  }
};

/**
 * Get fallback recommendations when API is unavailable
 */
const getFallbackRecommendations = (diagnosis: string): TreatmentPlan => {
  const normalizedDiagnosis = diagnosis.toLowerCase().replace(/ /g, '_');
  
  // Find matching recommendation
  for (const key of Object.keys(FALLBACK_RECOMMENDATIONS)) {
    if (normalizedDiagnosis.includes(key) || key.includes(normalizedDiagnosis)) {
      return FALLBACK_RECOMMENDATIONS[key];
    }
  }
  
  return FALLBACK_RECOMMENDATIONS['default'];
};

/**
 * Check if Gemini API is configured
 */
export const isGeminiConfigured = (): boolean => {
  return !!GEMINI_API_KEY;
};
