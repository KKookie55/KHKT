#!/usr/bin/env python3
"""
SC-PMH E2: Weather anomalies (relative climatology) as alternative to FWI
=========================================================================
Hypothesis: rather than adding FWI (which is information-redundant with raw
weather), compute anomalies — i.e., how UNUSUAL is today's weather compared
to the local seasonal norm at the same cell + same day-of-year.

Anomalies computed using ONLY training-period data (2018-2021) to avoid any
temporal leakage.

Features computed:
  t2m_anom_doy   = t2m - cell_doy_clim(t2m)     # temp anomaly vs seasonal norm
  rh_anom_doy    = rh - cell_doy_clim(rh)        # RH anomaly
  precip_anom_doy = precip - cell_doy_clim(precip)  # precipitation anomaly
  wind_anom_doy  = wind - cell_doy_clim(wind)   # wind anomaly

  # Plus smoothed versions using 7-day rolling mean of the anomaly
  t2m_anom_roll7
  rh_anom_roll7
  precip_anom_roll7

  # Composite dry anomaly: z-score of (high temp, low RH, low precip) — purely from train stats
  dry_anomaly_composite = z(t2m) - z(rh) - z(precip_roll30_sum)
  
This tests whether RELATIVE climatology provides signal that FWI couldn't.

Setup: cold-start mode (no fire-history features), spatial OOD.
  M1c_cold: 35 features (current cold-start, no anomalies)
  M1c_anom: 35 + 8 anomaly features = 43 features
  M1c_anom_composite: 35 + 1 composite = 36 features

5 seeds, paired t-test.
"""
from __future__ import annotations
import json, os, sys, time, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'e2_weather_anomalies'
EX.mkdir(parents=True, exist_ok=True)
PANEL_P = SC / 'panel_with_fwi.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

# === Load panel ===
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel['doy'] = panel['date'].dt.dayofyear
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f"[info] panel shape={panel.shape}")

# === Step 1: Compute per-cell day-of-year climatology from TRAIN period (2018-2021) ===
print("\n[1/4] computing per-cell day-of-year climatology from 2018-2021 only...")
train_only = panel[panel.year <= 2021].copy()
# Smooth doy: use 15-day rolling window centered on doy to reduce noise
# But cells are at 0.5° grid, 6 years of data per cell -> 6 obs per doy per cell
# Strategy: compute mean per (cell, doy), then rolling-mean over doy with window=15 (circular)
print("  computing mean per (cell, doy) from train period...")
cell_doy_clim = train_only.groupby(['lat', 'lon', 'doy']).agg(
    t2m_clim=('t2m', 'mean'),
    rh_clim=('rh', 'mean'),
    precip_clim=('precip', 'mean'),
    wind_clim=('wind', 'mean'),
).reset_index()
print(f"  cell_doy_clim shape: {cell_doy_clim.shape}")

# Smooth across doy (circular 15-day window)
def smooth_doy(grp, value_col, window=15):
    """For a single cell, smooth doy climatology with circular rolling window."""
    grp = grp.sort_values('doy')
    vals = grp[value_col].values
    n = len(vals)  # should be 365 or 366
    if n == 0:
        return grp
    # Circular padding
    pad = window // 2
    padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
    smoothed = pd.Series(padded).rolling(window=window, center=True, min_periods=1).mean().values[pad:pad+n]
    grp[value_col] = smoothed
    return grp

print("  applying 15-day circular smoothing per cell...")
cell_doy_clim_smoothed = []
for (lat, lon), g in cell_doy_clim.groupby(['lat', 'lon']):
    g = smooth_doy(g, 't2m_clim')
    g = smooth_doy(g, 'rh_clim')
    g = smooth_doy(g, 'precip_clim')
    g = smooth_doy(g, 'wind_clim')
    cell_doy_clim_smoothed.append(g)
cell_doy_clim = pd.concat(cell_doy_clim_smoothed, ignore_index=True)
print(f"  smoothed climatology: {cell_doy_clim.shape}")

# For doy=366 (leap year), we have very few samples. Forward-fill from doy=365
# Actually 2018-2021 includes 2020 (leap), so we have 1 sample for doy=366
# Let's just keep them all — model will handle it

# === Step 2: Merge climatology into panel, compute anomalies ===
print("\n[2/4] merging climatology and computing anomalies...")
# Left join: every row in panel gets the climatology for its (cell, doy)
# For cells/doy not in training (unlikely since training is 4 years), fallback to global mean
panel = panel.merge(cell_doy_clim, on=['lat', 'lon', 'doy'], how='left')
# Fallback for any NaN (e.g., leap day doy=366 for cells without 2020 data)
global_doy_clim = train_only.groupby('doy').agg(
    t2m_clim_g=('t2m', 'mean'), rh_clim_g=('rh', 'mean'),
    precip_clim_g=('precip', 'mean'), wind_clim_g=('wind', 'mean')
).reset_index()
panel = panel.merge(global_doy_clim, on='doy', how='left')
for col, gcol in [('t2m_clim', 't2m_clim_g'), ('rh_clim', 'rh_clim_g'),
                  ('precip_clim', 'precip_clim_g'), ('wind_clim', 'wind_clim_g')]:
    panel[col] = panel[col].fillna(panel[gcol])
panel = panel.drop(columns=['t2m_clim_g', 'rh_clim_g', 'precip_clim_g', 'wind_clim_g'])

# Compute anomalies
panel['t2m_anom_doy'] = panel['t2m'] - panel['t2m_clim']
panel['rh_anom_doy'] = panel['rh'] - panel['rh_clim']
panel['precip_anom_doy'] = panel['precip'] - panel['precip_clim']
panel['wind_anom_doy'] = panel['wind'] - panel['wind_clim']

# Rolling 7-day mean of anomaly (per cell)
for col in ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy']:
    roll_col = col.replace('_doy', '_roll7')
    panel[roll_col] = panel.groupby(['lat', 'lon'])[col].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

# Composite dry anomaly: standardize each component using TRAIN stats
print("  computing composite dry anomaly (z-scored from train)...")
scaler_t2m = StandardScaler().fit(train_only[['t2m']])
scaler_rh = StandardScaler().fit(train_only[['rh']])
scaler_precip30 = StandardScaler().fit(train_only[['precip_roll30_sum']])
panel['t2m_z'] = scaler_t2m.transform(panel[['t2m']])
panel['rh_z'] = scaler_rh.transform(panel[['rh']])
panel['precip30_z'] = scaler_precip30.transform(panel[['precip_roll30_sum']])
# Composite: high temp + low RH + low precip => high dry anomaly
panel['dry_anomaly_composite'] = panel['t2m_z'] - panel['rh_z'] - panel['precip30_z']

# Drop intermediate cols
panel = panel.drop(columns=['t2m_z', 'rh_z', 'precip30_z', 't2m_clim', 'rh_clim', 'precip_clim', 'wind_clim'])

ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
COMPOSITE_FEAT = ['dry_anomaly_composite']
print(f"\n  anomaly features: {ANOM_FEATS}")
print(f"  composite: {COMPOSITE_FEAT}")

# Stats
print("\n  anomaly stats:")
print(panel[ANOM_FEATS + COMPOSITE_FEAT].describe().round(3).to_string())

# Save panel
panel.to_parquet(EX / 'panel_with_anomalies.parquet', index=False)
print(f"\n  saved panel_with_anomalies.parquet shape={panel.shape}")

# === Step 3: Define feature sets ===
FEATS_M1c = COLD_FEATS  # 35 features (cold-start baseline, no anomalies)
FEATS_M1c_anom = COLD_FEATS + ANOM_FEATS  # 35 + 7 = 42
FEATS_M1c_anom_comp = COLD_FEATS + ANOM_FEATS + COMPOSITE_FEAT  # 43

# === Step 4: Spatial OOD train + eval ===
print("\n[3/4] training 3 feature-sets × 5 seeds × spatial OOD (cold-start)...")
panel_strict = panel  # already has anomalies
train = panel_strict[(panel_strict.region == 'North') & (panel_strict.year <= 2021)].dropna(subset=[LABEL_COL]).copy()
val   = panel_strict[(panel_strict.region == 'North') & (panel_strict.year == 2022)].dropna(subset=[LABEL_COL]).copy()
test  = panel_strict[(panel_strict.region == 'South') & (panel_strict.year == 2023)].dropna(subset=[LABEL_COL]).copy()
y_train = train[LABEL_COL].astype(int).values
y_val = val[LABEL_COL].astype(int).values
y_test = test[LABEL_COL].astype(int).values
print(f"  train={len(train)} val={len(val)} test={len(test)} pos_test={y_test.mean():.4f}")

# For South test, V3-style zero-out of fire-history — but wait, we already use COLD_FEATS (no fire-hist).
# So we're already in cold-start mode. No need for zero-out.
# In spatial OOD cold-start, the model has NO fire-history features for any cell.

def _ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)

def metrics(p, y):
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
    return dict(AUROC=float(roc_auc_score(y, p)), PR_AUC=float(average_precision_score(y, p)),
                Brier=float(brier_score_loss(y, p)), ECE=_ece(p, y))

rows = []
for variant_name, feats in [
    ('M1c_cold_baseline', FEATS_M1c),
    ('M1c_anom', FEATS_M1c_anom),
    ('M1c_anom_composite', FEATS_M1c_anom_comp),
]:
    print(f"\n  --- {variant_name} ({len(feats)} feats) ---")
    for seed in SEEDS:
        params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, reg_lambda=1.0, seed=seed, verbosity=0)
        dtr = gbm.DMatrix(train[feats], label=y_train)
        dv  = gbm.DMatrix(val[feats], label=y_val)
        dte = gbm.DMatrix(test[feats], label=y_test)
        m = gbm.train(params, dtr, num_boost_round=500, evals=[(dtr, 'tr'), (dv, 'val')],
                      early_stopping_rounds=30, verbose_eval=False)
        p_va = m.predict(dv); p_te = m.predict(dte)
        iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_val)
        p_te_iso = iso.predict(p_te)
        rows.append(dict(variant=variant_name, seed=seed, calibrated=False, **metrics(p_te, y_test)))
        rows.append(dict(variant=variant_name, seed=seed, calibrated=True, **metrics(p_te_iso, y_test)))
        # also save feature importances for the anomaly variant
        if variant_name != 'M1c_cold_baseline' and seed == 42:
            imps = m.get_score(importance_type='gain')
            sorted_imps = sorted(imps.items(), key=lambda x: -x[1])[:15]
            print(f"    seed=42 top-15 features:")
            for fname, score in sorted_imps:
                tag = " <== ANOMALY" if fname in ANOM_FEATS + COMPOSITE_FEAT else ""
                print(f"      {fname:35s} {score:.2f}{tag}")

# === Step 5: Summary ===
print("\n[4/4] summary...")
df_results = pd.DataFrame(rows)
df_results.to_csv(EX / 'e2_results.csv', index=False)

agg = df_results.groupby(['variant', 'calibrated']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
).reset_index()
agg.to_csv(EX / 'e2_summary.csv', index=False)

print("\n=== Spatial OOD cold-start, South test 2023 (mean ± std, n=5 seeds) ===")
for cal in [False, True]:
    print(f"\n  calibrated={cal}:")
    sub = agg[agg.calibrated == cal].sort_values('AUROC_mean', ascending=False)
    print(sub.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# Paired delta vs M1c_cold_baseline (uncalibrated, since cold-start calibration matters less)
print("\n=== Paired Δ (variant − M1c_cold_baseline) on South OOD, n=5 seeds ===")
for cal in [False, True]:
    print(f"\n  calibrated={cal}:")
    per_seed = {}
    for r in rows:
        if r['calibrated'] != cal: continue
        per_seed.setdefault(r['seed'], {})[r['variant']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
    print(f"  {'variant':<25} {'ΔAUROC':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22} {'ΔECE':>22}")
    print("  " + "-" * 100)
    for vs in ['M1c_anom', 'M1c_anom_composite']:
        line = f"  {vs:<25} "
        for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
            diff = np.array([per_seed[s][vs][metric] - per_seed[s]['M1c_cold_baseline'][metric] for s in SEEDS])
            mean, std = float(diff.mean()), float(diff.std())
            n = len(SEEDS)
            tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
            sig = " *" if abs(tstat) > 2.776 else ""
            line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
        print(line)

# Save summary
with open(EX / 'e2_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'variants': ['M1c_cold_baseline', 'M1c_anom', 'M1c_anom_composite'],
        'anomaly_features': ANOM_FEATS,
        'composite_feature': COMPOSITE_FEAT,
        'climatology_source': '2018-2021 training period only',
        'smoothing': '15-day circular rolling mean per (cell, doy)',
        'summary': agg.to_dict('records'),
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/e2_summary.csv, e2_summary.json, panel_with_anomalies.parquet")
