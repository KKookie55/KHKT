#!/usr/bin/env python3
"""Inference example: load saved model, predict fire risk from a feature row."""
import json, pickle
import numpy as np
import xgboost as xgb

# Load components
print("Loading model...")
with open("model_meta.json") as f: meta = json.load(f)
features = meta["features"]
m = xgb.XGBClassifier(); m.load_model("xgb_modis_iid.json")
with open("iso_calibrator.pkl", "rb") as f: iso = pickle.load(f)
with open("imputer.pkl", "rb") as f: imp = pickle.load(f)
print("  ✓ Model loaded")

def predict_fire_risk(feature_row):
    """Predict fire risk for a single feature row (dict with feature keys)."""
    x = np.array([[feature_row.get(f, 0.0) for f in features]], dtype=np.float32)
    di_idx = meta["drought_idx"]
    x[0, di_idx] = (x[0, di_idx] - meta["drought_mean"]) / meta["drought_std"]
    x = imp.transform(x)
    p_raw = m.predict_proba(x)[0, 1]
    p_iso = float(iso.predict([p_raw])[0])
    return {
        "raw_probability": float(p_raw),
        "calibrated_probability": p_iso,
        "interpretation": ("LOW risk (<0.1)" if p_iso < 0.1 else
                            "MEDIUM risk (0.1-0.5)" if p_iso < 0.5 else
                            "HIGH risk (>=0.5)"),
    }

if __name__ == "__main__":
    # Sample: hot, dry, windy day
    sample = {"t2m":28.5,"rh":65.0,"wind":4.5,"precip":0.0,"doy":60,
              "t2m_lag1":28.0,"t2m_lag3":27.5,"t2m_lag7":26.0,"t2m_lag14":25.0,
              "rh_lag1":67.0,"rh_lag3":68.0,"rh_lag7":70.0,"rh_lag14":72.0,
              "wind_lag1":4.0,"wind_lag3":5.0,"wind_lag7":3.5,"wind_lag14":3.0,
              "precip_lag1":0.0,"precip_lag3":0.0,"precip_lag7":0.0,"precip_lag14":5.0,
              "t2m_roll7":27.0,"t2m_roll14":26.5,"rh_roll7":68.0,"rh_roll14":70.0,
              "precip_roll7_sum":0.0,"precip_roll14_sum":5.0,"precip_roll30_sum":50.0,
              "days_since_rain":7,"t2m_diff3":0.5,"rh_diff3":-2.0,"wind_diff3":0.5,
              "fwi_compound":0.65,"heat_index":5.5,"wind_chill":13.5}
    r = predict_fire_risk(sample)
    print(f"\n=== PREDICTION ===")
    print(f"  Raw probability:     {r['raw_probability']:.4f}")
    print(f"  Calibrated (Iso):    {r['calibrated_probability']:.4f}")
    print(f"  Interpretation:      {r['interpretation']}")
