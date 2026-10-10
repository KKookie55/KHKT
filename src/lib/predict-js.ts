/**
 * JS inference module for XGBoost v8.6 wildfire model.
 * Loads the compact JSON model and performs prediction without Python.
 *
 * This is used as a fallback when the Python subprocess is unavailable
 * (e.g., on Netlify serverless functions).
 */

import modelData from "../../public/model/v86_compact.json";

export interface PredictInput {
  temp: number;
  humidity: number;
  wind: number;
  rain: number;
  daysSinceRain: number;
  region: "north" | "central" | "south";
  doy?: number;
}

export interface PredictResult {
  probability: number;
  raw_probability: number;
  calibrated_probability: number;
  risk_level: string;
  risk_color: string;
  risk_score: number;
  confidence: string;
  out_of_domain: boolean;
  warning: string | null;
  model_version: string;
  model_auroc_iid: number;
  model_ece_iid: number;
  n_features: number;
  region_penalty: number;
}

interface TreeNode {
  id: number;
  split?: string; // e.g. "f5"
  threshold?: number;
  yes?: number;
  no?: number;
  missing?: number;
  leaf?: number;
  children?: TreeNode[];
}

interface CompactModel {
  version: string;
  n_features: number;
  feature_names: string[];
  drought_idx: number;
  drought_mean: number;
  drought_std: number;
  num_trees: number;
  trees: TreeNode[];
  iso_calibrator: {
    x_thresholds: number[];
    y_thresholds: number[];
    x_min: number;
    x_max: number;
  };
  imputer: {
    statistics: number[];
    strategy: string;
  };
  performance: {
    AUROC_raw: number;
    AUROC_iso: number;
    ECE_raw: number;
    ECE_iso: number;
  };
  region_penalty: Record<string, number>;
}

const MODEL = modelData as unknown as CompactModel;

// Feature names in order (must match predict_v86.py)
const FEATURE_NAMES = [
  "t2m", "rh", "wind", "precip", "doy",
  "t2m_lag1", "t2m_lag3", "t2m_lag7", "t2m_lag14",
  "rh_lag1", "rh_lag3", "rh_lag7", "rh_lag14",
  "wind_lag1", "wind_lag3", "wind_lag7", "wind_lag14",
  "precip_lag1", "precip_lag3", "precip_lag7", "precip_lag14",
  "t2m_roll7", "t2m_roll14", "rh_roll7", "rh_roll14",
  "precip_roll7_sum", "precip_roll14_sum", "precip_roll30_sum",
  "days_since_rain", "t2m_diff3", "rh_diff3", "wind_diff3",
  "fwi_compound", "heat_index", "wind_chill",
];

// Map feature name to index
const FEATURE_INDEX: Record<string, number> = {};
FEATURE_NAMES.forEach((name, i) => { FEATURE_INDEX[name] = i; });
// Also support "f0", "f1", ... format from XGBoost dump
FEATURE_NAMES.forEach((_, i) => { FEATURE_INDEX[`f${i}`] = i; });

/**
 * Build 35-feature vector from 5 raw inputs (mirrors predict_v86.py engineer_features).
 */
function engineerFeatures(input: PredictInput): number[] {
  const t2m = input.temp;
  const rh = input.humidity;
  const wind = input.wind;
  const precip = input.rain;
  const daysSinceRain = input.daysSinceRain;
  const doy = input.doy ?? 60;

  // Lag features (steady-state assumption)
  const t2m_lag1 = t2m, t2m_lag3 = t2m, t2m_lag7 = t2m, t2m_lag14 = t2m;
  const rh_lag1 = rh, rh_lag3 = rh, rh_lag7 = rh, rh_lag14 = rh;
  const wind_lag1 = wind, wind_lag3 = wind, wind_lag7 = wind, wind_lag14 = wind;
  const precip_lag1 = precip, precip_lag3 = precip, precip_lag7 = precip, precip_lag14 = precip;

  // Rolling means
  const t2m_roll7 = t2m, t2m_roll14 = t2m;
  const rh_roll7 = rh, rh_roll14 = rh;

  // Rolling precip sums
  const precip_roll7_sum = precip * 7;
  const precip_roll14_sum = precip * 14;
  const precip_roll30_sum = precip * 30;

  // Diffs (steady state = 0)
  const t2m_diff3 = 0.0;
  const rh_diff3 = 0.0;
  const wind_diff3 = 0.0;

  // Derived composites
  const fwi_compound = Math.max(0, Math.min(1.5, (
    (t2m - 20) * 0.03 + (30 - rh) * 0.015 + wind * 0.04 - precip * 0.05
  )));
  const heat_index = Math.max(-10, Math.min(40, (t2m - 35 + (rh - 50) * 0.1)));
  const wind_chill = Math.max(-20, Math.min(30, t2m - wind * 0.5));

  return [
    t2m, rh, wind, precip, doy,
    t2m_lag1, t2m_lag3, t2m_lag7, t2m_lag14,
    rh_lag1, rh_lag3, rh_lag7, rh_lag14,
    wind_lag1, wind_lag3, wind_lag7, wind_lag14,
    precip_lag1, precip_lag3, precip_lag7, precip_lag14,
    t2m_roll7, t2m_roll14, rh_roll7, rh_roll14,
    precip_roll7_sum, precip_roll14_sum, precip_roll30_sum,
    daysSinceRain, t2m_diff3, rh_diff3, wind_diff3,
    fwi_compound, heat_index, wind_chill,
  ];
}

/**
 * Traverse a single tree to get the leaf value.
 */
function traverseTree(node: TreeNode, features: number[]): number {
  // If this is a leaf, return its value
  if (node.leaf !== undefined) {
    return node.leaf;
  }

  // Get the split feature index
  const splitName = node.split;
  if (!splitName) {
    // No split and no leaf — shouldn't happen, but return 0
    return 0;
  }

  const featureIdx = FEATURE_INDEX[splitName];
  if (featureIdx === undefined) {
    // Unknown feature — go to missing branch
    if (node.children && node.missing !== undefined) {
      const missingChild = node.children.find(c => c.id === node.missing);
      if (missingChild) return traverseTree(missingChild, features);
    }
    return 0;
  }

  const featureValue = features[featureIdx];
  const threshold = node.threshold ?? 0;

  // Check for NaN (missing value)
  let childId: number;
  if (isNaN(featureValue)) {
    childId = node.missing ?? node.yes ?? 0;
  } else if (featureValue < threshold) {
    childId = node.yes ?? 0;
  } else {
    childId = node.no ?? 0;
  }

  // Find the child node
  if (node.children) {
    const child = node.children.find(c => c.id === childId);
    if (child) return traverseTree(child, features);
  }

  return 0;
}

/**
 * Apply isotonic calibration using piecewise linear interpolation.
 */
function isotonicCalibrate(rawProb: number): number {
  const iso = MODEL.iso_calibrator;
  if (!iso.x_thresholds || !iso.y_thresholds || iso.x_thresholds.length === 0) {
    return rawProb; // fallback
  }

  // Clamp to range
  const x = Math.max(iso.x_min, Math.min(iso.x_max, rawProb));

  // Find the interval in x_thresholds
  const xs = iso.x_thresholds;
  const ys = iso.y_thresholds;

  // Binary search for the interval
  let lo = 0, hi = xs.length - 1;
  while (lo < hi - 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (xs[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  // Linear interpolation between (xs[lo], ys[lo]) and (xs[hi], ys[hi])
  const x0 = xs[lo], x1 = xs[hi];
  const y0 = ys[lo], y1 = ys[hi];
  if (x1 === x0) return y0;
  const t = (x - x0) / (x1 - x0);
  return y0 + t * (y1 - y0);
}

/**
 * Main prediction function — mirrors predict_v86.py predict().
 */
export function predictJS(input: PredictInput): PredictResult {
  // 1. Feature engineering
  const features = engineerFeatures(input);

  // 2. Normalize drought_idx (feature 27 = days_since_rain index in feature list)
  // Actually drought_idx in meta is 27, which is precip_roll30_sum (index 26 in 0-based)
  // Wait, let me check: meta['drought_idx'] = 27, features[27] = days_since_rain
  // So feature index 27 = days_since_rain — normalize it
  const diIdx = MODEL.drought_idx;
  if (diIdx >= 0 && diIdx < features.length) {
    features[diIdx] = (features[diIdx] - MODEL.drought_mean) / MODEL.drought_std;
  }

  // 3. Apply imputer (median fill for any NaN)
  // Since we don't have NaN in our engineered features (all are computed from valid inputs),
  // the imputer is a no-op. But we apply the statistics just in case.
  // (The Python imputer.strategy is "median", which replaces NaN with the median.)
  // Since we have no NaN, we skip this step.

  // 4. XGBoost prediction — traverse all 200 trees and sum leaf values
  let rawScore = 0;
  for (const tree of MODEL.trees) {
    rawScore += traverseTree(tree, features);
  }

  // 5. Apply sigmoid to get raw probability
  const p_raw = 1 / (1 + Math.exp(-rawScore));

  // 6. Apply isotonic calibration
  const p_calibrated = isotonicCalibrate(p_raw);

  // 7. Apply region penalty
  const region = input.region ?? "central";
  const penalty = MODEL.region_penalty[region] ?? 0.9;
  let finalProb = 0.5 + (p_calibrated - 0.5) * penalty;
  finalProb = Math.max(0.01, Math.min(0.99, finalProb));

  // 8. Risk level
  let risk_level: string, risk_color: string;
  if (finalProb >= 0.7) {
    risk_level = "Rất cao";
    risk_color = "#991b1b";
  } else if (finalProb >= 0.5) {
    risk_level = "Cao";
    risk_color = "#dc2626";
  } else if (finalProb >= 0.3) {
    risk_level = "Trung bình";
    risk_color = "#f59e0b";
  } else {
    risk_level = "Thấp";
    risk_color = "#16a34a";
  }

  // 9. OOD check
  const out_of_domain = (
    input.temp < 10 || input.temp > 38 ||
    input.humidity < 40 || input.humidity > 100 ||
    input.wind > 8 ||
    input.daysSinceRain > 15
  );

  let confidence: string;
  if (out_of_domain) {
    confidence = "Thấp";
  } else if (finalProb > 0.85 || finalProb < 0.05) {
    confidence = "Cao";
  } else {
    confidence = "Trung bình";
  }

  let warning: string | null = null;
  if (out_of_domain) {
    warning = "Đầu vào ngoài miền huấn luyện, dự báo có thể không đáng tin";
  } else if (region === "south") {
    warning = "Vùng Nam: model suy giảm AUROC (0,946 → 0,584 trên VIIRS)";
  }

  return {
    probability: Math.round(finalProb * 10000) / 10000,
    raw_probability: Math.round(p_raw * 10000) / 10000,
    calibrated_probability: Math.round(p_calibrated * 10000) / 10000,
    risk_level,
    risk_color,
    risk_score: Math.round(finalProb * 100) / 1,
    confidence,
    out_of_domain,
    warning,
    model_version: "v8.6_canonical_model",
    model_auroc_iid: MODEL.performance.AUROC_iso,
    model_ece_iid: MODEL.performance.ECE_iso,
    n_features: MODEL.n_features,
    region_penalty: penalty,
  };
}

/**
 * Batch prediction for multiple inputs.
 */
export function predictBatchJS(inputs: PredictInput[]): PredictResult[] {
  return inputs.map(input => {
    try {
      return predictJS(input);
    } catch (e) {
      return {
        probability: 0,
        raw_probability: 0,
        calibrated_probability: 0,
        risk_level: "Lỗi",
        risk_color: "#94a3b8",
        risk_score: 0,
        confidence: "Thấp",
        out_of_domain: true,
        warning: `Prediction error: ${e instanceof Error ? e.message : String(e)}`,
        model_version: "v8.6_canonical_model",
        model_auroc_iid: MODEL.performance.AUROC_iso,
        model_ece_iid: MODEL.performance.ECE_iso,
        n_features: MODEL.n_features,
        region_penalty: 0.9,
      };
    }
  });
}
