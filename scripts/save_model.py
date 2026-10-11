#!/usr/bin/env python3
"""
Save the canonical XGBoost model + Isotonic calibrator as pickle files
for download + deployment.

Trained on: MODIS labels, IID group-blocked split, seed 42, t+1 horizon.
This is the model that achieved AUROC=0.947, ECE Iso=0.009 (MODIS IID).
"""
import os, sys, json, pickle, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, f1_score
warnings.filterwarnings("ignore")

OUT = Path("/home/z/my-project/download/real_data_v85")
MODEL_DIR = Path("/home/z/my-project/download/model_v86")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*60); print("SAVING CANONICAL MODEL FOR DOWNLOAD"); print("="*60)

# Load panel + splits
panel = pd.read_parquet(OUT / "panel_v85.parquet")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
splits_npz = np.load(OUT / "splits_v85.npz")
features = json.loads((OUT/"panel_v85_meta.json").read_text())["features"]
print(f"  Panel: {len(panel):,} rows, {len(features)} features")

# IID split (the working case)
tr_idx = splits_npz["IID_groupblocked_train"]
va_idx = splits_npz["IID_groupblocked_val"]
te_idx = splits_npz["IID_groupblocked_test"]
tr = panel.iloc[tr_idx].dropna(subset=["label_h1"])
va = panel.iloc[va_idx].dropna(subset=["label_h1"])
te = panel.iloc[te_idx].dropna(subset=["label_h1"])
print(f"  Train: {len(tr):,} rows, Val: {len(va):,} rows, Test: {len(te):,} rows")

# Apply drought normalization (TRAIN-only)
di_idx = features.index("precip_roll30_sum")
di_mean = np.nanmean(tr[features[di_idx]].values)
di_std = np.nanstd(tr[features[di_idx]].values) + 1e-6

X_tr = tr[features].values.astype(np.float32).copy()
X_tr[:, di_idx] = np.nan_to_num((X_tr[:, di_idx] - di_mean) / di_std, nan=0.0)
y_tr = tr["label_h1"].values.astype(np.float32)
X_val = va[features].values.astype(np.float32).copy()
X_val[:, di_idx] = np.nan_to_num((X_val[:, di_idx] - di_mean) / di_std, nan=0.0)
y_val = va["label_h1"].values.astype(np.float32)
X_test = te[features].values.astype(np.float32).copy()
X_test[:, di_idx] = np.nan_to_num((X_test[:, di_idx] - di_mean) / di_std, nan=0.0)
y_test = te["label_h1"].values.astype(np.float32)

# Imputer
imp = SimpleImputer(strategy="median")
X_tr = imp.fit_transform(X_tr)
X_val = imp.transform(X_val)
X_test = imp.transform(X_test)

# Train XGB
pos = max(int(y_tr.sum()), 1); neg = max(int((1-y_tr).sum()), 1)
spw = max(math.sqrt(neg/pos), 1.0)
print(f"\n  Training XGBoost (n_estimators=200, max_depth=6, scale_pos_weight={spw:.2f})...")
m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
p_val = m.predict_proba(X_val)[:, 1]
p_test = m.predict_proba(X_test)[:, 1]
auroc_raw = roc_auc_score(y_test, p_test)
print(f"  ✓ XGB trained. Test AUROC (raw) = {auroc_raw:.4f}")

# Train Isotonic calibrator on val
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
iso.fit(p_val, y_val)
p_test_iso = iso.predict(p_test)
auroc_iso = roc_auc_score(y_test, p_test_iso)

# ECE
def ece(y, p):
    bins = np.linspace(0, 1, 11); e = 0; n = len(y)
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        ib = ((p >= lo) & (p <= hi)) if i == 9 else ((p >= lo) & (p < hi))
        if ib.sum() == 0: continue
        e += (ib.sum() / n) * abs(y[ib].mean() - p[ib].mean())
    return e
ece_raw = ece(y_test, p_test)
ece_iso = ece(y_test, p_test_iso)
print(f"  ✓ Iso calibrator trained. Test AUROC (Iso) = {auroc_iso:.4f}, ECE = {ece_iso:.4f}")
print(f"  (raw ECE = {ece_raw:.4f}, Iso improvement = -{(ece_iso-ece_raw)*1000:.1f} per mille)")

# Save model + components
print(f"\n  Saving model components to {MODEL_DIR}...")

# 1. XGBoost model (binary)
m.save_model(str(MODEL_DIR / "xgb_modis_iid.json"))
print(f"  ✓ XGB model: xgb_modis_iid.json ({(MODEL_DIR/'xgb_modis_iid.json').stat().st_size//1024} KB)")

# 2. Isotonic calibrator
with open(MODEL_DIR / "iso_calibrator.pkl", "wb") as f:
    pickle.dump(iso, f)
print(f"  ✓ Iso calibrator: iso_calibrator.pkl ({(MODEL_DIR/'iso_calibrator.pkl').stat().st_size//1024} KB)")

# 3. Imputer
with open(MODEL_DIR / "imputer.pkl", "wb") as f:
    pickle.dump(imp, f)
print(f"  ✓ Imputer: imputer.pkl ({(MODEL_DIR/'imputer.pkl').stat().st_size//1024} KB)")

# 4. Metadata (features, normalization stats)
meta = {
    "version": "v8.6_canonical_model",
    "model": "XGBoost + Isotonic",
    "trained_on": "MODIS labels, IID group-blocked, seed 42, t+1 horizon",
    "features": features,
    "n_features": len(features),
    "drought_idx": di_idx,
    "drought_mean": float(di_mean),
    "drought_std": float(di_std),
    "performance": {
        "AUROC_raw": float(auroc_raw),
        "AUROC_iso": float(auroc_iso),
        "ECE_raw": float(ece_raw),
        "ECE_iso": float(ece_iso),
    },
    "xgb_hyperparameters": {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.85, "colsample_bytree": 0.7, "min_child_weight": 20,
        "reg_lambda": 3.0, "reg_alpha": 0.5, "gamma": 0.1,
        "scale_pos_weight": float(spw),
        "tree_method": "hist", "random_state": 42,
    },
    "iso_hyperparameters": {
        "out_of_bounds": "clip", "y_min": 0, "y_max": 1,
    },
    "data": {
        "panel_size": len(panel),
        "train_size": len(tr), "val_size": len(va), "test_size": len(te),
        "fire_prevalence_train": float(y_tr.mean()),
        "fire_prevalence_test": float(y_test.mean()),
    },
    "input_format": "numpy array of shape (n_samples, 35) with float32 dtype",
    "output_format": "fire probability in [0, 1] (after Iso calibration)",
    "usage": "see predict_example.py for end-to-end inference",
}
with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  ✓ Metadata: model_meta.json")

# 5. Inference example script
predict_script = '''#!/usr/bin/env python3
"""Inference example: load saved model, predict fire risk from a feature row.

Usage:
  python predict_example.py

This script demonstrates how to use the saved v8.6 model to predict
fire risk for a new weather feature vector.
"""
import json, pickle
import numpy as np
import xgboost as xgb

MODEL_DIR = "."

# Load components
print("Loading model...")
with open(f"{MODEL_DIR}/model_meta.json") as f:
    meta = json.load(f)
features = meta["features"]
print(f"  Features: {len(features)}")

m = xgb.XGBClassifier()
m.load_model(f"{MODEL_DIR}/xgb_modis_iid.json")
print(f"  XGBoost loaded")

with open(f"{MODEL_DIR}/iso_calibrator.pkl", "rb") as f:
    iso = pickle.load(f)
print(f"  Isotonic calibrator loaded")

with open(f"{MODEL_DIR}/imputer.pkl", "rb") as f:
    imp = pickle.load(f)
print(f"  Imputer loaded")

def predict_fire_risk(feature_row):
    """Predict fire risk for a single feature row.
    
    Args:
        feature_row: dict with keys matching `features` list
                     (e.g., {"t2m": 28.5, "rh": 65, "wind": 4.5, ...})
    Returns:
        dict with raw and calibrated probabilities
    """
    # Convert to array in feature order
    x = np.array([[feature_row.get(f, 0.0) for f in features]], dtype=np.float32)
    
    # Apply drought normalization
    di_idx = meta["drought_idx"]
    di_mean = meta["drought_mean"]
    di_std = meta["drought_std"]
    x[0, di_idx] = (x[0, di_idx] - di_mean) / di_std
    
    # Impute NaN
    x = imp.transform(x)
    
    # XGBoost predict
    p_raw = m.predict_proba(x)[0, 1]
    
    # Isotonic calibration
    p_iso = float(iso.predict([p_raw])[0])
    
    return {
        "raw_probability": float(p_raw),
        "calibrated_probability": p_iso,
        "interpretation": (
            "LOW risk (<0.1)" if p_iso < 0.1 else
            "MEDIUM risk (0.1-0.5)" if p_iso < 0.5 else
            "HIGH risk (>=0.5)"
        ),
        "model_version": meta["version"],
        "performance_at_train": meta["performance"],
    }

# Example usage
if __name__ == "__main__":
    # Sample feature row (replace with real values from NASA POWER)
    sample_row = {
        "t2m": 28.5, "rh": 65.0, "wind": 4.5, "precip": 0.0,
        "doy": 60, "t2m_lag1": 28.0, "t2m_lag3": 27.5, "t2m_lag7": 26.0, "t2m_lag14": 25.0,
        "rh_lag1": 67.0, "rh_lag3": 68.0, "rh_lag7": 70.0, "rh_lag14": 72.0,
        "wind_lag1": 4.0, "wind_lag3": 5.0, "wind_lag7": 3.5, "wind_lag14": 3.0,
        "precip_lag1": 0.0, "precip_lag3": 0.0, "precip_lag7": 0.0, "precip_lag14": 5.0,
        "t2m_roll7": 27.0, "t2m_roll14": 26.5, "rh_roll7": 68.0, "rh_roll14": 70.0,
        "precip_roll7_sum": 0.0, "precip_roll14_sum": 5.0, "precip_roll30_sum": 50.0,
        "days_since_rain": 7, "t2m_diff3": 0.5, "rh_diff3": -2.0, "wind_diff3": 0.5,
        "fwi_compound": 0.65, "heat_index": 5.5, "wind_chill": 13.5,
    }
    
    print("\\n=== PREDICTION EXAMPLE ===")
    print(f"  Sample weather: T={sample_row['t2m']}°C, RH={sample_row['rh']}%, W={sample_row['wind']}m/s, precip={sample_row['precip']}mm")
    result = predict_fire_risk(sample_row)
    print(f"  Raw probability:     {result['raw_probability']:.4f}")
    print(f"  Calibrated (Iso):    {result['calibrated_probability']:.4f}")
    print(f"  Interpretation:      {result['interpretation']}")
    print(f"  Model version:       {result['model_version']}")
    print(f"  Train performance:   AUROC={result['performance_at_train']['AUROC_iso']:.4f}, ECE={result['performance_at_train']['ECE_iso']:.4f}")
    print(f"\\n  Note: This is the MODIS-trained model.")
    print(f"  Reliability-aware product: when forecasting South Vietnam or other OOD regions,")
    print(f"  reliability is LOW (see Phase A diagnosis in report).")
'''
(MODEL_DIR / "predict_example.py").write_text(predict_script)
print(f"  ✓ Inference script: predict_example.py")

# 6. Sample input CSV (3 example rows)
sample_rows = [
    {"t2m": 28.5, "rh": 65.0, "wind": 4.5, "precip": 0.0, "doy": 60,
     "t2m_lag1": 28.0, "t2m_lag3": 27.5, "t2m_lag7": 26.0, "t2m_lag14": 25.0,
     "rh_lag1": 67.0, "rh_lag3": 68.0, "rh_lag7": 70.0, "rh_lag14": 72.0,
     "wind_lag1": 4.0, "wind_lag3": 5.0, "wind_lag7": 3.5, "wind_lag14": 3.0,
     "precip_lag1": 0.0, "precip_lag3": 0.0, "precip_lag7": 0.0, "precip_lag14": 5.0,
     "t2m_roll7": 27.0, "t2m_roll14": 26.5, "rh_roll7": 68.0, "rh_roll14": 70.0,
     "precip_roll7_sum": 0.0, "precip_roll14_sum": 5.0, "precip_roll30_sum": 50.0,
     "days_since_rain": 7, "t2m_diff3": 0.5, "rh_diff3": -2.0, "wind_diff3": 0.5,
     "fwi_compound": 0.65, "heat_index": 5.5, "wind_chill": 13.5},
    {"t2m": 18.0, "rh": 90.0, "wind": 1.5, "precip": 25.0, "doy": 200,
     "t2m_lag1": 18.5, "t2m_lag3": 19.0, "t2m_lag7": 20.0, "t2m_lag14": 22.0,
     "rh_lag1": 88.0, "rh_lag3": 85.0, "rh_lag7": 80.0, "rh_lag14": 75.0,
     "wind_lag1": 2.0, "wind_lag3": 2.5, "wind_lag7": 3.0, "wind_lag14": 3.5,
     "precip_lag1": 20.0, "precip_lag3": 30.0, "precip_lag7": 15.0, "precip_lag14": 60.0,
     "t2m_roll7": 19.5, "t2m_roll14": 21.0, "rh_roll7": 82.0, "rh_roll14": 78.0,
     "precip_roll7_sum": 80.0, "precip_roll14_sum": 150.0, "precip_roll30_sum": 300.0,
     "days_since_rain": 0, "t2m_diff3": -1.0, "rh_diff3": 3.0, "wind_diff3": -1.0,
     "fwi_compound": 0.05, "heat_index": -1.6, "wind_chill": 4.5},
    {"t2m": 32.0, "rh": 35.0, "wind": 8.0, "precip": 0.0, "doy": 90,
     "t2m_lag1": 31.5, "t2m_lag3": 30.5, "t2m_lag7": 29.0, "t2m_lag14": 28.0,
     "rh_lag1": 38.0, "rh_lag3": 40.0, "rh_lag7": 45.0, "rh_lag14": 50.0,
     "wind_lag1": 7.5, "wind_lag3": 7.0, "wind_lag7": 6.5, "wind_lag14": 5.0,
     "precip_lag1": 0.0, "precip_lag3": 0.0, "precip_lag7": 0.0, "precip_lag14": 0.0,
     "t2m_roll7": 30.0, "t2m_roll14": 29.0, "rh_roll7": 42.0, "rh_roll14": 48.0,
     "precip_roll7_sum": 0.0, "precip_roll14_sum": 0.0, "precip_roll30_sum": 0.0,
     "days_since_rain": 30, "t2m_diff3": 1.5, "rh_diff3": -5.0, "wind_diff3": 1.0,
     "fwi_compound": 0.95, "heat_index": 18.0, "wind_chill": 28.0},
]
import pandas as pd
df_sample = pd.DataFrame(sample_rows)
df_sample.to_csv(MODEL_DIR / "sample_input.csv", index=False)
print(f"  ✓ Sample input: sample_input.csv (3 example rows)")

# 7. README
(MODEL_DIR / "README.md").write_text("""# v8.6 Wildfire Model Package

## What's inside

| File | Size | Description |
|------|------|-------------|
| `xgb_modis_iid.json` | ~150 KB | XGBoost model (35 features → fire probability) |
| `iso_calibrator.pkl` | ~50 KB | Isotonic regression calibrator (fit on val) |
| `imputer.pkl` | ~5 KB | Median imputer (fit on train) |
| `model_meta.json` | ~2 KB | Features list, normalization stats, hyperparameters |
| `predict_example.py` | ~3 KB | End-to-end inference example |
| `sample_input.csv` | ~1 KB | 3 example weather rows for testing |

## Performance (test set, IID, MODIS labels)

| Metric | Raw | After Iso |
|--------|-----|-----------|
| AUROC | {auroc_raw:.4f} | {auroc_iso:.4f} |
| ECE | {ece_raw:.4f} | {ece_iso:.4f} |

## How to use

```bash
# 1. Install dependencies
pip install xgboost scikit-learn numpy

# 2. Run prediction example
python predict_example.py
```

Or load in your own code:

```python
import json, pickle, numpy as np, xgboost as xgb

# Load components
with open("model_meta.json") as f: meta = json.load(f)
m = xgb.XGBClassifier(); m.load_model("xgb_modis_iid.json")
with open("iso_calibrator.pkl", "rb") as f: iso = pickle.load(f)
with open("imputer.pkl", "rb") as f: imp = pickle.load(f)

# Predict for a single feature row (35 features, see meta["features"] for order)
x = np.array([[28.5, 65, 4.5, ...]])  # shape (1, 35)
# Apply drought normalization
x[0, meta["drought_idx"]] = (x[0, meta["drought_idx"]] - meta["drought_mean"]) / meta["drought_std"]
x = imp.transform(x)
p_raw = m.predict_proba(x)[0, 1]
p_calibrated = float(iso.predict([p_raw])[0])
print(f"Fire risk: {{p_calibrated:.4f}}")
```

## Important caveats

1. **Model trained on MODIS labels** — VIIRS cross-validation showed MODIS evaluation overestimates by 0.31 AUROC on spatial OOD. Use with caution outside the training region (North Vietnam, 2018-2021).
2. **No spatial features** — model uses only weather features (T, RH, wind, precip + lags + derived). Adding raw lat/lon recovers +0.23 AUROC on South, but this is geography memorization.
3. **Reliability-aware use**: when forecasting for a region outside training distribution, treat model output as LOW reliability. See `phaseA_diagnosis/` for details.
4. **Forecast horizon**: trained for t+1 (next-day forecast). For t+3, t+7, performance degrades ~0.02 AUROC.

## For production deployment

The "Reliability-Aware Wildfire Forecasting System" proposed in this project
wraps the saved model with a reliability layer that detects:
- Spatial OOD (region outside training bbox)
- Covariate shift (weather distribution differs)
- Concept shift (fire-weather relationship differs)

See `product/reliability_dashboard.html` for the UI mockup.
""".format(auroc_raw=auroc_raw, auroc_iso=auroc_iso, ece_raw=ece_raw, ece_iso=ece_iso))
print(f"  ✓ README: README.md")

# Final summary
print("\n" + "="*60)
print("✓ MODEL PACKAGE COMPLETE")
print("="*60)
print(f"  Directory: {MODEL_DIR}")
print(f"  Total size: {sum(os.path.getsize(MODEL_DIR / f) for f in os.listdir(MODEL_DIR))//1024} KB")
print(f"\n  Files:")
for f in sorted(os.listdir(MODEL_DIR)):
    sz = os.path.getsize(MODEL_DIR / f) // 1024
    print(f"    {f} ({sz} KB)")

# ZIP the model package
import zipfile
MODEL_ZIP = Path("/home/z/my-project/download/wildfire_v86_model.zip")
with zipfile.ZipFile(MODEL_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(MODEL_DIR):
        zf.write(MODEL_DIR / f, f"wildfire_v86_model/{f}")
print(f"\n  Model ZIP: {MODEL_ZIP}")
print(f"  ZIP size: {MODEL_ZIP.stat().st_size//1024} KB")

# Also add to v86_final.zip
print("\n  Adding model package to v86_final.zip...")
v86_zip = Path("/home/z/my-project/download/WILDFIRE_PROJECT_v86_final.zip")
with zipfile.ZipFile(v86_zip, "a", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(MODEL_DIR):
        zf.write(MODEL_DIR / f, f"WILDFIRE_PROJECT_v86/model/{f}")
print(f"  ✓ Added to {v86_zip.name}")
import os as _os
print(f"  ZIP size: {_os.path.getsize(v86_zip)/1e6:.1f} MB")
