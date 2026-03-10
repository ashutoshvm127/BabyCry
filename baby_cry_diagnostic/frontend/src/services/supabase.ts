/**
 * Supabase Database Service
 * Handles scan history storage and statistics tracking
 * Falls back to localStorage when Supabase is unavailable
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Supabase Configuration from environment variables
const SUPABASE_URL = process.env.REACT_APP_SUPABASE_URL || '';
const SUPABASE_ANON_KEY = process.env.REACT_APP_SUPABASE_ANON_KEY || '';

// Local Storage Keys
const LOCAL_SCANS_KEY = 'baby_cry_scans';
const LOCAL_STATS_KEY = 'baby_cry_stats';

// Track if Supabase is actually reachable (will be set to false on timeout)
let supabaseAvailable = true;

// Initialize Supabase Client (singleton)
let supabase: SupabaseClient | null = null;

if (SUPABASE_URL && SUPABASE_ANON_KEY && !supabase) {
  supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
}

// Types
export interface ScanDocument {
  id?: string;
  created_at?: string;
  date: string;
  time: string;
  name: string;
  reg: string;
  initial: string;
  color: string;
  disease: string;
  disease_color: string;
  accuracy: number;
  symptoms: string;
  risk_level?: string;
  is_critical: boolean;
  meds_advised: boolean;
  audio_url?: string;  // URL to audio file in Supabase Storage
}

export interface StatsDocument {
  id?: string;
  total_scans: number;
  critical_alerts: number;
  meds_advised: number;
  last_updated: string;
}

// Check if Supabase is configured
export const isSupabaseConfigured = (): boolean => {
  return !!(SUPABASE_URL && SUPABASE_ANON_KEY) && supabaseAvailable;
};

// ===== LOCAL STORAGE HELPERS =====

const getLocalScans = (): ScanDocument[] => {
  try {
    const data = localStorage.getItem(LOCAL_SCANS_KEY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

const saveLocalScans = (scans: ScanDocument[]): void => {
  try {
    localStorage.setItem(LOCAL_SCANS_KEY, JSON.stringify(scans));
  } catch (e) {
    console.warn('Failed to save to localStorage:', e);
  }
};

const getLocalStats = (): StatsDocument => {
  try {
    const data = localStorage.getItem(LOCAL_STATS_KEY);
    if (data) return JSON.parse(data);
  } catch {}
  return {
    total_scans: 0,
    critical_alerts: 0,
    meds_advised: 0,
    last_updated: new Date().toISOString()
  };
};

const saveLocalStats = (stats: StatsDocument): void => {
  try {
    localStorage.setItem(LOCAL_STATS_KEY, JSON.stringify(stats));
  } catch (e) {
    console.warn('Failed to save stats to localStorage:', e);
  }
};

// ===== SCAN OPERATIONS =====

/**
 * Save a new scan to Supabase database (with localStorage fallback)
 */
export const saveScan = async (scan: Omit<ScanDocument, 'id' | 'created_at'>): Promise<ScanDocument | null> => {
  const newScan: ScanDocument = {
    ...scan,
    id: crypto.randomUUID(),
    created_at: new Date().toISOString()
  };

  // Always save to localStorage as backup
  const localScans = getLocalScans();
  localScans.unshift(newScan);
  saveLocalScans(localScans.slice(0, 500)); // Keep last 500

  // Update local stats
  const localStats = getLocalStats();
  localStats.total_scans++;
  if (scan.is_critical) localStats.critical_alerts++;
  if (scan.meds_advised) localStats.meds_advised++;
  localStats.last_updated = new Date().toISOString();
  saveLocalStats(localStats);

  if (!supabase || !supabaseAvailable) {
    console.info('Using localStorage for scan storage.');
    return newScan;
  }

  try {
    const { data, error } = await supabase
      .from('scans')
      .insert([scan])
      .select()
      .single();

    if (error) throw error;
    
    // Update stats after saving scan
    await incrementStats(scan.is_critical, scan.meds_advised);
    
    return data as ScanDocument;
  } catch (error) {
    console.error('Error saving scan to Supabase:', error);
    return null;
  }
};

/**
 * Fetch all scans from Supabase database (with localStorage fallback)
 */
export const fetchScans = async (limit: number = 100): Promise<ScanDocument[]> => {
  // If Supabase is unavailable, use localStorage
  if (!supabase || !supabaseAvailable) {
    console.info('Using localStorage for scan history.');
    return getLocalScans().slice(0, limit);
  }

  try {
    const { data, error } = await supabase
      .from('scans')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) throw error;
    
    return (data || []) as ScanDocument[];
  } catch (error: any) {
    console.warn('Supabase unavailable, falling back to localStorage:', error?.message || error);
    supabaseAvailable = false; // Mark as unavailable to prevent repeated timeouts
    return getLocalScans().slice(0, limit);
  }
};

/**
 * Delete a scan from database (with localStorage fallback)
 */
export const deleteScan = async (scanId: string): Promise<boolean> => {
  // Always remove from localStorage
  const localScans = getLocalScans();
  const filteredScans = localScans.filter(s => s.id !== scanId);
  saveLocalScans(filteredScans);

  if (!supabase || !supabaseAvailable) {
    return true;
  }

  try {
    const { error } = await supabase
      .from('scans')
      .delete()
      .eq('id', scanId);

    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Error deleting scan from Supabase:', error);
    return true; // Still return true since we deleted from localStorage
  }
};

// ===== STATS OPERATIONS =====

const STATS_ROW_ID = 1; // Single row for global stats

/**
 * Get current statistics (with localStorage fallback)
 */
export const fetchStats = async (): Promise<StatsDocument> => {
  // If Supabase unavailable, use localStorage
  if (!supabase || !supabaseAvailable) {
    console.info('Using localStorage for statistics.');
    return getLocalStats();
  }

  try {
    const { data, error } = await supabase
      .from('stats')
      .select('*')
      .eq('id', STATS_ROW_ID)
      .single();

    if (error) {
      // If row doesn't exist, return local stats
      if (error.code === 'PGRST116') {
        return getLocalStats();
      }
      throw error;
    }
    
    return data as StatsDocument;
  } catch (error: any) {
    console.warn('Supabase unavailable for stats, using localStorage:', error?.message || error);
    supabaseAvailable = false;
    return getLocalStats();
  }
};

/**
 * Increment statistics after a new scan
 */
export const incrementStats = async (isCritical: boolean, medsAdvised: boolean): Promise<void> => {
  if (!supabase || !supabaseAvailable) {
    // Already handled in saveScan via localStorage
    return;
  }

  try {
    const currentStats = await fetchStats();
    
    const { error } = await supabase
      .from('stats')
      .update({
        total_scans: currentStats.total_scans + 1,
        critical_alerts: currentStats.critical_alerts + (isCritical ? 1 : 0),
        meds_advised: currentStats.meds_advised + (medsAdvised ? 1 : 0),
        last_updated: new Date().toISOString()
      })
      .eq('id', STATS_ROW_ID);

    if (error) throw error;
  } catch (error) {
    console.error('Error updating stats:', error);
  }
};

/**
 * Recalculate stats from all scans (useful for data recovery)
 */
export const recalculateStats = async (): Promise<StatsDocument> => {
  const defaultStats: StatsDocument = {
    total_scans: 0,
    critical_alerts: 0,
    meds_advised: 0,
    last_updated: new Date().toISOString()
  };

  if (!supabase) {
    return defaultStats;
  }

  try {
    const scans = await fetchScans(10000);
    
    const stats: StatsDocument = {
      total_scans: scans.length,
      critical_alerts: scans.filter(s => s.is_critical).length,
      meds_advised: scans.filter(s => s.meds_advised).length,
      last_updated: new Date().toISOString()
    };

    const { error } = await supabase
      .from('stats')
      .update(stats)
      .eq('id', STATS_ROW_ID);

    if (error) throw error;

    return stats;
  } catch (error) {
    console.error('Error recalculating stats:', error);
    return defaultStats;
  }
};

export { supabase };

// ===== AUDIO STORAGE OPERATIONS =====

const AUDIO_BUCKET = 'voices';

/**
 * Upload audio file to Supabase Storage
 * @param audioBlob - The audio file as a Blob
 * @param scanId - Unique identifier for the scan
 * @returns Public URL of the uploaded file, or null on failure
 */
export const uploadAudio = async (audioBlob: Blob, scanId: string): Promise<string | null> => {
  if (!supabase) {
    console.warn('Supabase not configured. Audio not uploaded.');
    return null;
  }

  try {
    const fileName = `${scanId}.wav`;
    const filePath = `scans/${fileName}`;

    const { data, error } = await supabase.storage
      .from(AUDIO_BUCKET)
      .upload(filePath, audioBlob, {
        contentType: 'audio/wav',
        upsert: true
      });

    if (error) throw error;

    // Get public URL
    const { data: urlData } = supabase.storage
      .from(AUDIO_BUCKET)
      .getPublicUrl(filePath);

    return urlData?.publicUrl || null;
  } catch (error) {
    console.error('Error uploading audio to Supabase:', error);
    return null;
  }
};

/**
 * Delete audio file from Supabase Storage
 * @param scanId - The scan ID (used as filename)
 */
export const deleteAudio = async (scanId: string): Promise<boolean> => {
  if (!supabase) {
    return false;
  }

  try {
    const filePath = `scans/${scanId}.wav`;
    const { error } = await supabase.storage
      .from(AUDIO_BUCKET)
      .remove([filePath]);

    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Error deleting audio:', error);
    return false;
  }
};

/**
 * Get audio download URL for a scan
 * @param scanId - The scan ID
 * @returns Signed URL valid for 1 hour
 */
export const getAudioUrl = async (scanId: string): Promise<string | null> => {
  if (!supabase) {
    return null;
  }

  try {
    const filePath = `scans/${scanId}.wav`;
    const { data, error } = await supabase.storage
      .from(AUDIO_BUCKET)
      .createSignedUrl(filePath, 3600); // 1 hour expiry

    if (error) throw error;
    return data?.signedUrl || null;
  } catch (error) {
    console.error('Error getting audio URL:', error);
    return null;
  }
};
