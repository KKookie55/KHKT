/**
 * v9.4 JS inference engine — pure JavaScript, no Python needed.
 * Works on Netlify (serverless Node.js).
 *
 * Loads:
 * - model_forest_fire_compact.json (XGBoost trees)
 * - iso_forest_fire.json (isotonic calibration)
 * - district_lookup.json (static features per district)
 * - district_clim.json (monthly climatology per district)
 * - district_ndvi.json (real MODIS NDVI per district × doy)
 * - district_cfclim.json (cell_fire_rate climatology per district × month)
 * - features_forest_fire.json (feature names in order)
 */

import modelData from "../../public/model/v94/model_forest_fire_compact.json";
import isoData from "../../public/model/v94/iso_forest_fire.json";
import districtLookup from "../../public/model/v94/district_lookup.json";
import districtClim from "../../public/model/v94/district_clim.json";
import districtNdvi from "../../public/model/v94/district_ndvi.json";
import districtCfclim from "../../public/model/v94/district_cfclim.json";
import featuresData from "../../public/model/v94/features_forest_fire.json";

const TREES = modelData as any[];
const ISO = isoData as any;
const DISTRICTS = districtLookup as any[];
const CLIM = districtClim as Record<number, Record<number, any>>;
const NDVI = districtNdvi as Record<number, Record<number, number>>;
const CFCLIM = districtCfclim as Record<number, Record<number, any>>;
const FEATURE_NAMES: string[] = featuresData.features;

// Build feature name → index map
const FEATURE_INDEX: Record<string, number> = {};
FEATURE_NAMES.forEach((name, i) => { FEATURE_INDEX[name] = i; });
// Also support f0, f1, ... format
FEATURE_NAMES.forEach((_, i) => { FEATURE_INDEX[`f${i}`] = i; });

// Build district lookup by ID
const DISTRICT_MAP: Record<number, any> = {};
DISTRICTS.forEach((d: any) => { DISTRICT_MAP[d.district_id] = d; });

/**
 * Traverse a single XGBoost tree to get the leaf value.
 * Tree structure: {nodeid, split, split_condition, yes, no, missing, children, leaf}
 */
function traverseTree(node: any, features: number[]): number {
  if (node.leaf !== undefined) {
    return node.leaf;
  }
  const splitName = node.split;
  if (!splitName) return 0;
  const featureIdx = FEATURE_INDEX[splitName];
  if (featureIdx === undefined) {
    // Unknown feature — go to missing branch
    if (node.children && node.missing !== undefined) {
      const missingChild = node.children.find((c: any) => c.nodeid === node.missing);
      if (missingChild) return traverseTree(missingChild, features);
    }
    return 0;
  }
  const featureValue = features[featureIdx];
  const threshold = node.split_condition ?? 0;
  let childId: number;
  if (isNaN(featureValue)) {
    childId = node.missing ?? node.yes ?? 0;
  } else if (featureValue < threshold) {
    childId = node.yes ?? 0;
  } else {
    childId = node.no ?? 0;
  }
  if (node.children) {
    const child = node.children.find((c: any) => c.nodeid === childId);
    if (child) return traverseTree(child, features);
  }
  return 0;
}

/**
 * Apply isotonic calibration using piecewise linear interpolation.
 */
function isotonicCalibrate(rawProb: number): number {
  const xs = ISO.x_thresholds;
  const ys = ISO.y_thresholds;
  if (!xs || !ys || xs.length === 0) return rawProb;
  const x = Math.max(ISO.x_min, Math.min(ISO.x_max, rawProb));
  let lo = 0, hi = xs.length - 1;
  while (lo < hi - 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const x0 = xs[lo], x1 = xs[hi];
  const y0 = ys[lo], y1 = ys[hi];
  if (x1 === x0) return y0;
  const t = (x - x0) / (x1 - x0);
  return y0 + t * (y1 - y0);
}

/**
 * Get climatology for a district × doy (monthly lookup).
 */
function getClim(districtId: number, doy: number) {
  const month = Math.min(12, Math.floor((doy - 1) / 30.5) + 1);
  const d = CLIM[districtId];
  if (!d) return { t: 25, r: 75, p: 3, w: 3 };
  const m = d[month] || d[Object.keys(d)[0]];
  return m || { t: 25, r: 75, p: 3, w: 3 };
}

/**
 * Get NDVI for a district × doy.
 */
function getNdvi(districtId: number, doy: number): number {
  const d = NDVI[districtId];
  if (!d) return 0.5;
  // Find nearest doy
  const doys = Object.keys(d).map(Number);
  let nearest = doys[0];
  let minDist = Math.abs(doy - nearest);
  for (const dy of doys) {
    const dist = Math.abs(doy - dy);
    if (dist < minDist) { minDist = dist; nearest = dy; }
  }
  return d[nearest] || 0.5;
}

/**
 * Get cell_fire_rate climatology for a district × doy (monthly lookup).
 */
function getCfclim(districtId: number, doy: number) {
  const month = Math.min(12, Math.floor((doy - 1) / 30.5) + 1);
  const d = CFCLIM[districtId];
  if (!d) return { cfr: 0.05, cfc: 1.5, dry: 0, cfd: 0.05 };
  const m = d[month] || d[Object.keys(d)[0]];
  return m || { cfr: 0.05, cfc: 1.5, dry: 0, cfd: 0.05 };
}

/**
 * Build feature vector for a district-day.
 */
function buildFeatures(districtId: number, doy: number, temp: number, humidity: number, wind: number, rain: number, daysSinceRain: number): number[] {
  const static_ = DISTRICT_MAP[districtId] || {};
  const clim = getClim(districtId, doy);
  const ndvi = getNdvi(districtId, doy);
  const cfclim = getCfclim(districtId, doy);

  const t2m = temp ?? clim.t;
  const rh = humidity ?? clim.r;
  const wind_ = wind ?? clim.w;
  const precip = rain ?? clim.p;
  const dsr = daysSinceRain ?? 5;

  // Lags (climatology-based, with override propagation)
  const useOverride = Math.abs(t2m - clim.t) > 1.5 || Math.abs(rh - clim.r) > 1.5 ||
                       Math.abs(wind_ - clim.w) > 1.5 || Math.abs(precip - clim.p) > 1.5;
  const lag = useOverride ? t2m : clim.t;
  const rlag = useOverride ? rh : clim.r;
  const wlag = useOverride ? wind_ : clim.w;
  const plag = useOverride ? precip : clim.p;

  const roll = useOverride ? t2m : clim.t;
  const rroll = useOverride ? rh : clim.r;
  const p7 = (useOverride ? precip : clim.p) * 7;
  const p14 = (useOverride ? precip : clim.p) * 14;
  const p30 = (useOverride ? precip : clim.p) * 30;

  // Anomalies
  const t_anom = t2m - clim.t;
  const r_anom = rh - clim.r;
  const p_anom = precip - clim.p;
  const w_anom = wind_ - clim.w;
  const dry_anom = t_anom * 0.5 - r_anom * 0.3 - p_anom * 0.2;

  // FWI (simplified)
  const fwi = Math.max(0, Math.min(50, (t2m - 15) * 0.5 + (30 - rh) * 0.3 + wind_ * 2 - precip * 2));
  const fwi_ffmc = Math.max(0, Math.min(100, 50 + (t2m - 25) * 2 - (rh - 50) * 0.5));
  const fwi_dmc = Math.max(0, Math.min(100, 30 + (t2m - 25) * 1.5 - precip * 1.5));
  const fwi_dc = Math.max(0, Math.min(800, 200 + (t2m - 25) * 10 - precip * 5));
  const fwi_isi = Math.max(0, Math.min(50, 5 + wind_ * 3 - rh * 0.05));
  const fwi_bui = Math.max(0, Math.min(200, 50 + (t2m - 25) * 2));
  const fwi_comp = Math.max(0, Math.min(1.5, (t2m - 20) * 0.03 + (30 - rh) * 0.015 + wind_ * 0.04 - precip * 0.05));
  const heat_index = Math.max(-10, Math.min(40, t2m - 35 + (rh - 50) * 0.1));
  const wind_chill = Math.max(-20, Math.min(30, t2m - wind_ * 0.5));

  // Interactions
  const wind_temp = wind_ * t2m;
  const humidity_temp = rh * t2m;
  const dry_wind = dry_anom * wind_;
  const heat_drought = t_anom * dsr;
  const ndvi_anom = ndvi - 0.499;

  // Build feature array in order matching FEATURE_NAMES
  const featureMap: Record<string, number> = {
    't2m': t2m, 'rh': rh, 'wind': wind_, 'precip': precip, 'doy': doy,
    't2m_lag1': lag, 't2m_lag3': lag, 't2m_lag7': lag, 't2m_lag14': lag,
    'rh_lag1': rlag, 'rh_lag3': rlag, 'rh_lag7': rlag, 'rh_lag14': rlag,
    'wind_lag1': wlag, 'wind_lag3': wlag, 'wind_lag7': wlag, 'wind_lag14': wlag,
    'precip_lag1': plag, 'precip_lag3': plag, 'precip_lag7': plag, 'precip_lag14': plag,
    't2m_roll7': roll, 't2m_roll14': roll, 'rh_roll7': rroll, 'rh_roll14': rroll,
    'precip_roll7_sum': p7, 'precip_roll14_sum': p14, 'precip_roll30_sum': p30,
    'days_since_rain': dsr, 't2m_diff3': t_anom, 'rh_diff3': r_anom, 'wind_diff3': w_anom,
    'fwi_compound': fwi_comp, 'heat_index': heat_index, 'wind_chill': wind_chill,
    't2m_anom_doy': t_anom, 'rh_anom_doy': r_anom, 'precip_anom_doy': p_anom, 'wind_anom_doy': w_anom,
    't2m_anom_roll7': t_anom, 'rh_anom_roll7': r_anom, 'precip_anom_roll7': p_anom * 7 - clim.p * 7,
    'dry_anomaly_composite': dry_anom,
    'fwi': fwi, 'fwi_ffmc': fwi_ffmc, 'fwi_dmc': fwi_dmc, 'fwi_dc': fwi_dc, 'fwi_isi': fwi_isi, 'fwi_bui': fwi_bui,
    'elevation_mean': static_.elevation_mean ?? 300,
    'elevation_std': static_.elevation_std ?? 90,
    'slope_mean': static_.slope_mean ?? 9,
    'koppen_Cwa': static_.koppen_Cwa ?? 0,
    'koppen_Am': static_.koppen_Am ?? 0,
    'koppen_Aw': static_.koppen_Aw ?? 1,
    'latitude_band': static_.latitude_band ?? 2,
    'coastal_distance': static_.coastal_distance ?? 2,
    'population_density_proxy': static_.population_density_proxy ?? 50,
    'ndvi': ndvi,
    'cell_fire_rate_exp': cfclim.cfr,
    'cell_fire_count_30d': cfclim.cfc,
    'is_dry': cfclim.dry,
    'cell_fire_rate_dry_exp': cfclim.cfd,
    'centroid_lat': static_.centroid_lat ?? 16,
    'centroid_lon': static_.centroid_lon ?? 106,
    'wind_temp': wind_temp,
    'humidity_temp': humidity_temp,
    'dry_wind': dry_wind,
    'heat_drought': heat_drought,
    'ndvi_anom': ndvi_anom,
    'neighbor_fire': 0, // placeholder — no real-time FIRMS
    'forest_prop_real': static_.forest_prop ?? 0.5,
    'agri_prop_real': static_.agriculture_prop ?? 0.3,
    'urban_prop_real': static_.urban_prop ?? 0.05,
    'water_prop_real': static_.water_prop ?? 0.05,
  };

  // Build array in feature order
  return FEATURE_NAMES.map(f => featureMap[f] ?? 0);
}

export interface PredictResult {
  district_id: number;
  probability: number;
  raw_probability: number;
  risk_level: string;
  risk_color: string;
  risk_score: number;
  model_version: string;
  mode: string;
  n_features: number;
  district_name: string;
  province: string;
}

export function predictBatchV94JS(inputs: any[]): PredictResult[] {
  const results: PredictResult[] = [];
  for (const inp of inputs) {
    const districtId = parseInt(inp.district_id);
    const doy = inp.doy ?? Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000);
    const temp = parseFloat(inp.temp) || 25;
    const humidity = parseFloat(inp.humidity) || 75;
    const wind = parseFloat(inp.wind) || 3;
    const rain = parseFloat(inp.rain) || 0;
    const dsr = parseFloat(inp.daysSinceRain) || 5;

    const features = buildFeatures(districtId, doy, temp, humidity, wind, rain, dsr);

    // XGBoost predict: sum of all tree leaf values
    let rawScore = 0;
    for (const tree of TREES) {
      rawScore += traverseTree(tree, features);
    }
    const p_raw = 1 / (1 + Math.exp(-rawScore));
    const p_cal = isotonicCalibrate(p_raw);
    const p = Math.max(0.001, Math.min(0.999, p_cal));

    let risk_level = "Thấp", risk_color = "#22c55e";
    if (p >= 0.30) { risk_level = "Nguy hiểm"; risk_color = "#4c1d95"; }
    else if (p >= 0.20) { risk_level = "Rất cao"; risk_color = "#7f1d1d"; }
    else if (p >= 0.10) { risk_level = "Cao"; risk_color = "#dc2626"; }
    else if (p >= 0.05) { risk_level = "Trung bình"; risk_color = "#f59e0b"; }

    const static_ = DISTRICT_MAP[districtId] || {};
    results.push({
      district_id: districtId,
      probability: p,
      raw_probability: p_raw,
      risk_level,
      risk_color,
      risk_score: Math.round(p * 100),
      model_version: "v9.4 JS (75 features, Optuna + spatial lag)",
      mode: inp.mode || "forest",
      n_features: FEATURE_NAMES.length,
      district_name: static_.district_name || static_.name || "?",
      province: static_.province || "?",
    });
  }
  return results;
}
