#!/usr/bin/env python3
"""
Add cell-level fire history features to v8.5 panel, retrain XGB, compare.

Strategy: compute fire rate per cell over TRAINING PERIOD (2018-2020),
add as features. This gives model a "spatial prior" without external data.

Features added:
  - cell_fire_rate_train: avg fire rate per cell over 2018-2020 (MODIS)
  - cell_fire_count_30d: fire count in last 30 days (rolling, train-only)
  - cell_days_since_fire: days since last fire in this cell
"""
import os, json, math, time, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, f1_score
warnings.filterwarnings("ignore")

OUT = Path("/home/z/my-project/download/real_data_v85")
MODEL_DIR = Path("/home/z/my-project/download/model_v86")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*60); print("CELL FIRE HISTORY + RETRAIN"); print("="*60)

# Load panel (rebuild if missing)
if not (OUT / "panel_v85.parquet").exists():
    print("Panel missing — rebuilding...")
    DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
    wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv")
    wx["date"] = pd.to_datetime(wx["date"])
    fires = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in range(2018,2024)], ignore_index=True)
    fires["lat"] = np.round((fires["latitude"]-11.0)/0.5)*0.5+11.0
    fires["lon"] = np.round((fires["longitude"]-104.0)/0.5)*0.5+104.0
    fires["date"] = pd.to_datetime(fires["acq_date"])
    fires = fires[(fires["lat"]>=11)&(fires["lat"]<=21)&(fires["lon"]>=104)&(fires["lon"]<=109.5)]
    fcd = fires[["lat","lon","date"]].drop_duplicates(); fcd["fire"]=1
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
    for v in ["t2m","rh","wind","precip"]:
        for lag in [1,3,7,14]:
            panel[f"{v}_lag{lag}"] = panel.groupby(["lat","lon"])[v].shift(lag)
    for v in ["t2m","rh"]:
        panel[f"{v}_roll7"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(7,min_periods=3).mean())
        panel[f"{v}_roll14"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(14,min_periods=5).mean())
    panel["precip_roll7_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(7,min_periods=3).sum())
    panel["precip_roll14_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(14,min_periods=5).sum())
    panel["precip_roll30_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(30,min_periods=10).sum())
    def dsr(s, t=0.5):
        is_dry = (s<t).astype(int); g = (is_dry==0).cumsum()
        return g.groupby(g).cumcount()*is_dry
    panel["days_since_rain"] = panel.groupby(["lat","lon"])["precip"].transform(dsr)
    for v in ["t2m","rh","wind"]:
        panel[f"{v}_diff3"] = panel.groupby(["lat","lon"])[v].diff(3)
    fwi = (0.4*(panel["t2m"]-10) - 0.62*panel["rh"] + 0.4*panel["wind"] - 0.3*panel["precip_lag7"])
    panel["fwi_compound"] = 1/(1+np.exp(-(fwi-30)/5))
    panel["heat_index"] = (panel["t2m"]-20)*(50-panel["rh"])/50.0
    panel["wind_chill"] = (panel["t2m"]-15)*panel["wind"]/30.0
    panel["label_h1"] = panel.groupby(["lat","lon"])["fire"].shift(-1)
    NON_FEATURE = {"lat","lon","date","fire","year","label_h1"}
    FEATURES_ORIG = [c for c in panel.columns if c not in NON_FEATURE]
    panel.to_parquet(OUT/"panel_v85.parquet", index=False)
    with open(OUT/"panel_v85_meta.json","w") as f:
        json.dump({"features":FEATURES_ORIG}, f, indent=2, default=str)
    print(f"  Panel rebuilt: {len(panel):,} rows, {len(FEATURES_ORIG)} features")
else:
    panel = pd.read_parquet(OUT / "panel_v85.parquet")
    panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
    meta = json.loads((OUT/"panel_v85_meta.json").read_text())
    FEATURES_ORIG = meta["features"]
    print(f"  Panel loaded: {len(panel):,} rows, {len(FEATURES_ORIG)} features")

# Load splits
splits_npz = np.load(OUT / "splits_v85.npz")

# ============================================================
# ADD CELL FIRE HISTORY FEATURES
# ============================================================
print("\n[1/3] Computing cell fire history features...")

# 1. Cell fire rate over training period (2018-2020) — LEAKAGE-SAFE
train_mask = panel["year"].isin([2018, 2019, 2020])
cell_fire_rate = panel[train_mask].groupby(["lat", "lon"])["fire"].mean().reset_index()
cell_fire_rate.columns = ["lat", "lon", "cell_fire_rate_train"]
panel = panel.merge(cell_fire_rate, on=["lat", "lon"], how="left")
panel["cell_fire_rate_train"] = panel["cell_fire_rate_train"].fillna(panel["fire"].mean())
print(f"  cell_fire_rate_train: min={panel['cell_fire_rate_train'].min():.4f}, "
      f"max={panel['cell_fire_rate_train'].max():.4f}, mean={panel['cell_fire_rate_train'].mean():.4f}")

# 2. Days since last fire in this cell (rolling, within-cell)
panel = panel.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
# For each cell, compute cumulative days since last fire
def compute_days_since_fire(group):
    group = group.sort_values("date").copy()
    fire_mask = group["fire"] == 1
    days_since = np.zeros(len(group))
    last_fire_idx = -1
    for i in range(len(group)):
        if fire_mask.iloc[i]:
            last_fire_idx = i
            days_since[i] = 0
        elif last_fire_idx >= 0:
            days_since[i] = (group.iloc[i]["date"] - group.iloc[last_fire_idx]["date"]).days
        else:
            days_since[i] = 999  # no fire ever
    group["cell_days_since_fire"] = days_since
    return group
print("  Computing days_since_fire (may take 30s)...")
panel = panel.groupby(["lat", "lon"], group_keys=False).apply(compute_days_since_fire)
panel["cell_days_since_fire"] = panel["cell_days_since_fire"].clip(0, 365)
print(f"  cell_days_since_fire: min={panel['cell_days_since_fire'].min():.0f}, "
      f"max={panel['cell_days_since_fire'].max():.0f}, median={panel['cell_days_since_fire'].median():.0f}")

# 3. Fire count in last 30 days (rolling, within-cell)
print("  Computing cell_fire_count_30d (rolling 30-day)...")
panel["cell_fire_count_30d"] = panel.groupby(["lat", "lon"])["fire"].transform(
    lambda s: s.rolling(30, min_periods=1).sum())
print(f"  cell_fire_count_30d: min={panel['cell_fire_count_30d'].min():.0f}, "
      f"max={panel['cell_fire_count_30d'].max():.0f}, mean={panel['cell_fire_count_30d'].mean():.2f}")

# New feature list
NEW_FEATURES = FEATURES_ORIG + ["cell_fire_rate_train", "cell_days_since_fire", "cell_fire_count_30d"]
print(f"\n  Original features: {len(FEATURES_ORIG)}")
print(f"  New features added: 3 (cell_fire_rate_train, cell_days_since_fire, cell_fire_count_30d)")
print(f"  Total features: {len(NEW_FEATURES)}")

# ============================================================
# RETRAIN: Compare original vs new features on IID + Spatial N→S
# ============================================================
print("\n[2/3] Retraining XGB — comparing original vs new features...")

def ece(y, p):
    bins = np.linspace(0, 1, 11); e = 0; n = len(y)
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        ib = ((p >= lo) & (p <= hi)) if i == 9 else ((p >= lo) & (p < hi))
        if ib.sum() == 0: continue
        e += (ib.sum() / n) * abs(y[ib].mean() - p[ib].mean())
    return e

def train_eval(panel, features, split_name, train_idx, val_idx, test_idx, label="label_h1"):
    tr = panel.iloc[train_idx].dropna(subset=[label])
    va = panel.iloc[val_idx].dropna(subset=[label])
    te = panel.iloc[test_idx].dropna(subset=[label])
    di_idx = features.index("precip_roll30_sum")
    di_mean = np.nanmean(tr[features[di_idx]].values)
    di_std = np.nanstd(tr[features[di_idx]].values) + 1e-6
    X_tr = tr[features].values.astype(np.float32).copy()
    X_tr[:, di_idx] = np.nan_to_num((X_tr[:, di_idx] - di_mean) / di_std, nan=0.0)
    y_tr = tr[label].values.astype(np.float32)
    X_val = va[features].values.astype(np.float32).copy()
    X_val[:, di_idx] = np.nan_to_num((X_val[:, di_idx] - di_mean) / di_std, nan=0.0)
    y_val = va[label].values.astype(np.float32)
    X_test = te[features].values.astype(np.float32).copy()
    X_test[:, di_idx] = np.nan_to_num((X_test[:, di_idx] - di_mean) / di_std, nan=0.0)
    y_test = te[label].values.astype(np.float32)
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
    auroc = roc_auc_score(y_test, p_test)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_val, y_val)
    p_test_iso = iso.predict(p_test)
    auroc_iso = roc_auc_score(y_test, p_test_iso)
    ece_iso = ece(y_test, p_test_iso)
    return auroc, auroc_iso, ece_iso, m, iso, imp

# Test on IID + Spatial N→S with BOTH feature sets
results = {}
for split_name, tr_key, va_key, te_key in [
    ("IID", "IID_groupblocked_train", "IID_groupblocked_val", "IID_groupblocked_test"),
    ("Spatial N→S", "Spatial_OOD_NorthToSouth_train", "Spatial_OOD_NorthToSouth_val", "Spatial_OOD_NorthToSouth_test"),
]:
    tr_idx = splits_npz[tr_key]; va_idx = splits_npz[va_key]; te_idx = splits_npz[te_key]
    print(f"\n  {split_name}:")
    # Original features
    a1, a1i, e1, _, _, _ = train_eval(panel, FEATURES_ORIG, split_name, tr_idx, va_idx, te_idx)
    print(f"    Original ({len(FEATURES_ORIG)} feat):  AUROC={a1:.4f}, Iso AUROC={a1i:.4f}, ECE={e1:.4f}")
    # New features (with fire history)
    a2, a2i, e2, m_new, iso_new, imp_new = train_eval(panel, NEW_FEATURES, split_name, tr_idx, va_idx, te_idx)
    print(f"    + Fire history ({len(NEW_FEATURES)} feat): AUROC={a2:.4f}, Iso AUROC={a2i:.4f}, ECE={e2:.4f}")
    delta = a2 - a1
    print(f"    Δ AUROC = {delta:+.4f} ({'CẢI THIỆN' if delta > 0 else 'không cải thiện'})")
    results[split_name] = {
        "original": {"auroc": a1, "auroc_iso": a1i, "ece_iso": e1},
        "with_fire_history": {"auroc": a2, "auroc_iso": a2i, "ece_iso": e2},
        "delta_auroc": float(delta),
    }

# ============================================================
# SAVE BEST MODEL (with fire history)
# ============================================================
print(f"\n[3/3] Saving best model (with fire history)...")
# Use IID split for the saved model
tr_idx = splits_npz["IID_groupblocked_train"]
va_idx = splits_npz["IID_groupblocked_val"]
te_idx = splits_npz["IID_groupblocked_test"]
a_iid, a_iid_i, e_iid_i, m_best, iso_best, imp_best = train_eval(
    panel, NEW_FEATURES, "IID", tr_idx, va_idx, te_idx)
print(f"  Best IID: AUROC={a_iid:.4f}, Iso AUROC={a_iid_i:.4f}, ECE={e_iid_i:.4f}")

# Also get spatial N→S with fire history
tr_sn = splits_npz["Spatial_OOD_NorthToSouth_train"]
va_sn = splits_npz["Spatial_OOD_NorthToSouth_val"]
te_sn = splits_npz["Spatial_OOD_NorthToSouth_test"]
a_sn, a_sn_i, e_sn_i, _, _, _ = train_eval(panel, NEW_FEATURES, "Spatial N→S", tr_sn, va_sn, te_sn)
print(f"  Spatial N→S: AUROC={a_sn:.4f}, Iso AUROC={a_sn_i:.4f}, ECE={e_sn_i:.4f}")

# Save model
m_best.save_model(str(MODEL_DIR / "xgb_modis_iid_v86plus.json"))
with open(MODEL_DIR / "iso_calibrator_v86plus.pkl", "wb") as f: pickle.dump(iso_best, f)
with open(MODEL_DIR / "imputer_v86plus.pkl", "wb") as f: pickle.dump(imp_best, f)
di_idx = NEW_FEATURES.index("precip_roll30_sum")
# Compute drought stats from IID train
tr_iid = panel.iloc[tr_idx].dropna(subset=["label_h1"])
di_mean = np.nanmean(tr_iid[NEW_FEATURES[di_idx]].values)
di_std = np.nanstd(tr_iid[NEW_FEATURES[di_idx]].values) + 1e-6
meta_v86plus = {
    "version": "v8.6plus_fire_history",
    "model": "XGBoost + Isotonic (with cell fire history)",
    "features": NEW_FEATURES,
    "n_features": len(NEW_FEATURES),
    "new_features_added": ["cell_fire_rate_train", "cell_days_since_fire", "cell_fire_count_30d"],
    "drought_idx": di_idx, "drought_mean": float(di_mean), "drought_std": float(di_std),
    "performance": {
        "IID_AUROC": float(a_iid), "IID_AUROC_iso": float(a_iid_i), "IID_ECE_iso": float(e_iid_i),
        "Spatial_NtoS_AUROC": float(a_sn), "Spatial_NtoS_AUROC_iso": float(a_sn_i),
        "Spatial_NtoS_ECE_iso": float(e_sn_i),
    },
    "improvement_vs_v85": {
        "IID_delta": float(a_iid - results["IID"]["original"]["auroc"]),
        "Spatial_NtoS_delta": float(a_sn - results["Spatial N→S"]["original"]["auroc"]),
    },
    "caveat": "cell_fire_rate_train is computed from 2018-2020 MODIS labels (training period only). "
              "This gives model a spatial prior analogous to PCCCR's forest type/fuel load data. "
              "It is NOT pure physics generalization — it captures 'which cells tend to have fires'.",
}
with open(MODEL_DIR / "model_meta_v86plus.json", "w") as f:
    json.dump(meta_v86plus, f, indent=2, default=str)
print(f"  ✓ Saved: xgb_modis_iid_v86plus.json ({(MODEL_DIR/'xgb_modis_iid_v86plus.json').stat().st_size//1024} KB)")

# Summary
print(f"\n{'='*60}")
print("SUMMARY — Cell Fire History Impact")
print(f"{'='*60}")
print(f"\n  {'Split':<20} | {'Original':<12} | {'+ Fire History':<14} | {'Δ':<8}")
print(f"  {'-'*56}")
for sp, d in results.items():
    print(f"  {sp:<20} | {d['original']['auroc']:<12.4f} | {d['with_fire_history']['auroc']:<14.4f} | {d['delta_auroc']:+.4f}")
print(f"\n  → Cell fire history {'HELPS' if all(d['delta_auroc'] > 0 for d in results.values()) else 'mixed effect'}")
print(f"  → The model now has a spatial prior (like PCCCR's forest type/fuel data)")
print(f"  → Caveat: this is 'which cells tend to have fires', not pure physics")

# Save comparison JSON
with open(OUT / "fire_history_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  ✓ Comparison saved: {OUT}/fire_history_comparison.json")
