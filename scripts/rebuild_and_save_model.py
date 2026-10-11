#!/usr/bin/env python3
"""Quick rebuild: preprocess panel + train XGB + save model — all in one script."""
import os, json, math, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, f1_score
warnings.filterwarnings("ignore")

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/real_data_v85")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("/home/z/my-project/download/model_v86")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GRID_RES=0.5; LAT_MIN,LAT_MAX=11.0,21.0; LON_MIN,LON_MAX=104.0,109.5
YEARS=list(range(2018,2024)); MASTER_SEED=42

print("="*60); print("REBUILD v8.5 + SAVE MODEL"); print("="*60)

# 1. Load weather
print("[1/5] Loading weather...")
wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv")
wx["date"] = pd.to_datetime(wx["date"])
print(f"  Weather: {len(wx):,} rows, {wx['lat'].nunique()}x{wx['lon'].nunique()} grid")

# 2. Load MODIS fires
print("[2/5] Loading MODIS fires...")
fires = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in YEARS], ignore_index=True)
fires["lat"] = np.round((fires["latitude"]-LAT_MIN)/GRID_RES)*GRID_RES+LAT_MIN
fires["lon"] = np.round((fires["longitude"]-LON_MIN)/GRID_RES)*GRID_RES+LON_MIN
fires["date"] = pd.to_datetime(fires["acq_date"])
fires = fires[(fires["lat"]>=LAT_MIN)&(fires["lat"]<=LAT_MAX)&(fires["lon"]>=LON_MIN)&(fires["lon"]<=LON_MAX)]
fcd = fires[["lat","lon","date"]].drop_duplicates(); fcd["fire"]=1
print(f"  Unique fire cell-days: {len(fcd):,}")

# 3. Build panel
print("[3/5] Building panel...")
cells = wx[["lat","lon"]].drop_duplicates().sort_values(["lat","lon"]).reset_index(drop=True)
all_dates = pd.date_range(wx["date"].min(), wx["date"].max(), freq="D")
cells["_k"]=1; ad=pd.DataFrame({"date":all_dates}); ad["_k"]=1
panel = cells.merge(ad, on="_k").drop("_k", axis=1)
panel = panel.merge(wx, on=["lat","lon","date"], how="left")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
for c in ["t2m","rh","wind","precip"]:
    panel[c] = panel.groupby(["lat","lon"])[c].transform(lambda s: s.ffill().bfill())
panel = panel.merge(fcd, on=["lat","lon","date"], how="left")
panel["fire"] = panel["fire"].fillna(0).astype(int)
panel["doy"] = panel["date"].dt.dayofyear
panel["year"] = panel["date"].dt.year

# Features
for v in ["t2m","rh","wind","precip"]:
    for lag in [1,3,7,14]:
        panel[f"{v}_lag{lag}"] = panel.groupby(["lat","lon"])[v].shift(lag)
for v in ["t2m","rh"]:
    panel[f"{v}_roll7"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(7,min_periods=3).mean())
    panel[f"{v}_roll14"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(14,min_periods=5).mean())
panel["precip_roll7_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(7,min_periods=3).sum())
panel["precip_roll14_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(14,min_periods=5).sum())
panel["precip_roll30_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(30,min_periods=10).sum())
def days_since_rain(s, t=0.5):
    is_dry = (s<t).astype(int); g = (is_dry==0).cumsum()
    return g.groupby(g).cumcount()*is_dry
panel["days_since_rain"] = panel.groupby(["lat","lon"])["precip"].transform(days_since_rain)
for v in ["t2m","rh","wind"]:
    panel[f"{v}_diff3"] = panel.groupby(["lat","lon"])[v].diff(3)
fwi = (0.4*(panel["t2m"]-10) - 0.62*panel["rh"] + 0.4*panel["wind"] - 0.3*panel["precip_lag7"])
panel["fwi_compound"] = 1/(1+np.exp(-(fwi-30)/5))
panel["heat_index"] = (panel["t2m"]-20)*(50-panel["rh"])/50.0
panel["wind_chill"] = (panel["t2m"]-15)*panel["wind"]/30.0
# Forecast horizon t+1
panel["label_h1"] = panel.groupby(["lat","lon"])["fire"].shift(-1)

NON_FEATURE = {"lat","lon","date","fire","year","label_h1"}
FEATURES = [c for c in panel.columns if c not in NON_FEATURE]
print(f"  Panel: {len(panel):,} rows, {len(FEATURES)} features")

# Save panel + meta
panel.to_parquet(OUT/"panel_v85.parquet", index=False)
with open(OUT/"panel_v85_meta.json","w") as f:
    json.dump({"version":"v8.5","features":FEATURES,"horizons":[1]},f,indent=2,default=str)
print(f"  Saved: panel_v85.parquet ({(OUT/'panel_v85.parquet').stat().st_size//1024//1024} MB)")

# 4. Build IID split (group-blocked)
print("[4/5] Building IID group-blocked split...")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
panel["row_idx"] = np.arange(len(panel))
cells_df = panel[["lat","lon"]].drop_duplicates().reset_index(drop=True)
cells_df["cell_id"] = np.arange(len(cells_df))
panel = panel.merge(cells_df, on=["lat","lon"], how="left")
cells_df["lat_band"] = pd.cut(cells_df["lat"], bins=8, labels=False)
rng = np.random.RandomState(MASTER_SEED)
tr_ids,va_ids,te_ids = [],[],[]
for band, grp in cells_df.groupby("lat_band"):
    idx = grp["cell_id"].values; rng.shuffle(idx); n=len(idx)
    n_tr=int(n*0.6); n_va=int(n*0.2)
    tr_ids.extend(idx[:n_tr]); va_ids.extend(idx[n_tr:n_tr+n_va]); te_ids.extend(idx[n_tr+n_va:])
tr_idx = panel[panel["cell_id"].isin(tr_ids)]["row_idx"].values
va_idx = panel[panel["cell_id"].isin(va_ids)]["row_idx"].values
te_idx = panel[panel["cell_id"].isin(te_ids)]["row_idx"].values
np.savez(OUT/"splits_v85.npz",
         IID_groupblocked_train=tr_idx, IID_groupblocked_val=va_idx, IID_groupblocked_test=te_idx)
print(f"  IID: train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,}")

# 5. Train XGB on MODIS IID
print("[5/5] Training XGB on MODIS IID (seed 42)...")
tr = panel.iloc[tr_idx].dropna(subset=["label_h1"])
va = panel.iloc[va_idx].dropna(subset=["label_h1"])
te = panel.iloc[te_idx].dropna(subset=["label_h1"])

di_idx = FEATURES.index("precip_roll30_sum")
di_mean = np.nanmean(tr[FEATURES[di_idx]].values)
di_std = np.nanstd(tr[FEATURES[di_idx]].values) + 1e-6

X_tr = tr[FEATURES].values.astype(np.float32).copy()
X_tr[:, di_idx] = np.nan_to_num((X_tr[:, di_idx] - di_mean) / di_std, nan=0.0)
y_tr = tr["label_h1"].values.astype(np.float32)
X_val = va[FEATURES].values.astype(np.float32).copy()
X_val[:, di_idx] = np.nan_to_num((X_val[:, di_idx] - di_mean) / di_std, nan=0.0)
y_val = va["label_h1"].values.astype(np.float32)
X_test = te[FEATURES].values.astype(np.float32).copy()
X_test[:, di_idx] = np.nan_to_num((X_test[:, di_idx] - di_mean) / di_std, nan=0.0)
y_test = te["label_h1"].values.astype(np.float32)

imp = SimpleImputer(strategy="median")
X_tr = imp.fit_transform(X_tr); X_val = imp.transform(X_val); X_test = imp.transform(X_test)

pos = max(int(y_tr.sum()), 1); neg = max(int((1-y_tr).sum()), 1)
spw = max(math.sqrt(neg/pos), 1.0)
m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
p_val = m.predict_proba(X_val)[:, 1]
p_test = m.predict_proba(X_test)[:, 1]
auroc_raw = roc_auc_score(y_test, p_test)

# Iso calibration
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
print(f"  Test AUROC raw={auroc_raw:.4f}, Iso={auroc_iso:.4f}")
print(f"  Test ECE   raw={ece_raw:.4f}, Iso={ece_iso:.4f}")

# Save model files
print(f"\n  Saving model files to {MODEL_DIR}...")
m.save_model(str(MODEL_DIR / "xgb_modis_iid.json"))
with open(MODEL_DIR / "iso_calibrator.pkl", "wb") as f: pickle.dump(iso, f)
with open(MODEL_DIR / "imputer.pkl", "wb") as f: pickle.dump(imp, f)
meta = {
    "version": "v8.6_canonical_model",
    "model": "XGBoost + Isotonic",
    "trained_on": "MODIS labels, IID group-blocked, seed 42, t+1 horizon",
    "features": FEATURES, "n_features": len(FEATURES),
    "drought_idx": di_idx, "drought_mean": float(di_mean), "drought_std": float(di_std),
    "performance": {"AUROC_raw": float(auroc_raw), "AUROC_iso": float(auroc_iso),
                     "ECE_raw": float(ece_raw), "ECE_iso": float(ece_iso)},
    "xgb_hyperparameters": {"n_estimators":200,"max_depth":6,"learning_rate":0.1,
        "subsample":0.85,"colsample_bytree":0.7,"min_child_weight":20,
        "reg_lambda":3.0,"reg_alpha":0.5,"gamma":0.1,"scale_pos_weight":float(spw),
        "tree_method":"hist","random_state":42},
    "iso_hyperparameters": {"out_of_bounds":"clip","y_min":0,"y_max":1},
    "data": {"panel_size":len(panel),"train_size":len(tr),"val_size":len(va),"test_size":len(te),
             "fire_prevalence_train":float(y_tr.mean()),"fire_prevalence_test":float(y_test.mean())},
    "input_format": "numpy array of shape (n_samples, 35) with float32 dtype",
    "output_format": "fire probability in [0, 1] (after Iso calibration)",
}
with open(MODEL_DIR / "model_meta.json", "w") as f: json.dump(meta, f, indent=2, default=str)

# Sample input CSV
sample_rows = [
    {"t2m":28.5,"rh":65.0,"wind":4.5,"precip":0.0,"doy":60,
     "t2m_lag1":28.0,"t2m_lag3":27.5,"t2m_lag7":26.0,"t2m_lag14":25.0,
     "rh_lag1":67.0,"rh_lag3":68.0,"rh_lag7":70.0,"rh_lag14":72.0,
     "wind_lag1":4.0,"wind_lag3":5.0,"wind_lag7":3.5,"wind_lag14":3.0,
     "precip_lag1":0.0,"precip_lag3":0.0,"precip_lag7":0.0,"precip_lag14":5.0,
     "t2m_roll7":27.0,"t2m_roll14":26.5,"rh_roll7":68.0,"rh_roll14":70.0,
     "precip_roll7_sum":0.0,"precip_roll14_sum":5.0,"precip_roll30_sum":50.0,
     "days_since_rain":7,"t2m_diff3":0.5,"rh_diff3":-2.0,"wind_diff3":0.5,
     "fwi_compound":0.65,"heat_index":5.5,"wind_chill":13.5},
    {"t2m":18.0,"rh":90.0,"wind":1.5,"precip":25.0,"doy":200,
     "t2m_lag1":18.5,"t2m_lag3":19.0,"t2m_lag7":20.0,"t2m_lag14":22.0,
     "rh_lag1":88.0,"rh_lag3":85.0,"rh_lag7":80.0,"rh_lag14":75.0,
     "wind_lag1":2.0,"wind_lag3":2.5,"wind_lag7":3.0,"wind_lag14":3.5,
     "precip_lag1":20.0,"precip_lag3":30.0,"precip_lag7":15.0,"precip_lag14":60.0,
     "t2m_roll7":19.5,"t2m_roll14":21.0,"rh_roll7":82.0,"rh_roll14":78.0,
     "precip_roll7_sum":80.0,"precip_roll14_sum":150.0,"precip_roll30_sum":300.0,
     "days_since_rain":0,"t2m_diff3":-1.0,"rh_diff3":3.0,"wind_diff3":-1.0,
     "fwi_compound":0.05,"heat_index":-1.6,"wind_chill":4.5},
    {"t2m":32.0,"rh":35.0,"wind":8.0,"precip":0.0,"doy":90,
     "t2m_lag1":31.5,"t2m_lag3":30.5,"t2m_lag7":29.0,"t2m_lag14":28.0,
     "rh_lag1":38.0,"rh_lag3":40.0,"rh_lag7":45.0,"rh_lag14":50.0,
     "wind_lag1":7.5,"wind_lag3":7.0,"wind_lag7":6.5,"wind_lag14":5.0,
     "precip_lag1":0.0,"precip_lag3":0.0,"precip_lag7":0.0,"precip_lag14":0.0,
     "t2m_roll7":30.0,"t2m_roll14":29.0,"rh_roll7":42.0,"rh_roll14":48.0,
     "precip_roll7_sum":0.0,"precip_roll14_sum":0.0,"precip_roll30_sum":0.0,
     "days_since_rain":30,"t2m_diff3":1.5,"rh_diff3":-5.0,"wind_diff3":1.0,
     "fwi_compound":0.95,"heat_index":18.0,"wind_chill":28.0},
]
pd.DataFrame(sample_rows).to_csv(MODEL_DIR / "sample_input.csv", index=False)

# Predict example
predict_script = '''#!/usr/bin/env python3
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
    print(f"\\n=== PREDICTION ===")
    print(f"  Raw probability:     {r['raw_probability']:.4f}")
    print(f"  Calibrated (Iso):    {r['calibrated_probability']:.4f}")
    print(f"  Interpretation:      {r['interpretation']}")
'''
(MODEL_DIR / "predict_example.py").write_text(predict_script)

# README
(MODEL_DIR / "README.md").write_text(f"""# v8.6 Wildfire Model Package

## Files

| File | Description |
|------|-------------|
| `xgb_modis_iid.json` | XGBoost model (35 features → fire probability) |
| `iso_calibrator.pkl` | Isotonic regression calibrator |
| `imputer.pkl` | Median imputer |
| `model_meta.json` | Features, normalization stats, hyperparameters |
| `predict_example.py` | End-to-end inference example |
| `sample_input.csv` | 3 example weather rows for testing |

## Performance (test set, IID, MODIS labels)

| Metric | Raw | After Iso |
|--------|-----|-----------|
| AUROC | {auroc_raw:.4f} | {auroc_iso:.4f} |
| ECE | {ece_raw:.4f} | {ece_iso:.4f} |

## How to use

```bash
pip install xgboost scikit-learn numpy
python predict_example.py
```

Or load in your own code:

```python
import json, pickle, numpy as np, xgboost as xgb
with open("model_meta.json") as f: meta = json.load(f)
m = xgb.XGBClassifier(); m.load_model("xgb_modis_iid.json")
with open("iso_calibrator.pkl", "rb") as f: iso = pickle.load(f)
with open("imputer.pkl", "rb") as f: imp = pickle.load(f)

# Predict for a single feature row
x = np.array([[28.5, 65, 4.5, ...]])  # shape (1, 35), see meta["features"] for order
di_idx = meta["drought_idx"]
x[0, di_idx] = (x[0, di_idx] - meta["drought_mean"]) / meta["drought_std"]
x = imp.transform(x)
p_raw = m.predict_proba(x)[0, 1]
p_calibrated = float(iso.predict([p_raw])[0])
print(f"Fire risk: {{p_calibrated:.4f}}")
```

## Important caveats

1. **Model trained on MODIS labels** — VIIRS cross-validation showed MODIS evaluation overestimates by 0.31 AUROC on spatial OOD. Use with caution outside training region.
2. **No spatial features** — model uses only weather features. Adding raw lat/lon recovers +0.23 AUROC on South, but this is geography memorization.
3. **Reliability-aware use**: when forecasting for a region outside training distribution, treat model output as LOW reliability.
4. **Forecast horizon**: trained for t+1 (next-day forecast). For t+3, t+7, performance degrades ~0.02 AUROC.
""")

# Build standalone model ZIP
import zipfile
MODEL_ZIP = Path("/home/z/my-project/download/wildfire_v86_model.zip")
with zipfile.ZipFile(MODEL_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(MODEL_DIR):
        zf.write(MODEL_DIR / f, f"wildfire_v86_model/{f}")

print(f"\n{'='*60}")
print("✓ MODEL PACKAGE COMPLETE")
print(f"{'='*60}")
print(f"  Model dir: {MODEL_DIR}")
print(f"  Files:")
for f in sorted(os.listdir(MODEL_DIR)):
    print(f"    {f} ({os.path.getsize(MODEL_DIR/f)//1024} KB)")
print(f"\n  Standalone ZIP: {MODEL_ZIP}")
print(f"  ZIP size: {MODEL_ZIP.stat().st_size//1024} KB")
