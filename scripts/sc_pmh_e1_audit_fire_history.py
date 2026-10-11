#!/usr/bin/env python3
"""
SC-PMH E1: Audit & ablation of cell_fire_rate_exp
=================================================
Goal: determine how much of XGBoost's predictive power comes from the
per-cell fire-history feature, and whether the current computation has
material label leakage.

5 variants of the fire-history feature:
  V1: ORIGINAL (v8.7.1): expanding mean of fire_prev, fillna with global panel mean
                          — small leakage in fillna (uses future labels for day 1)
  V2: STRICT: same expanding mean, fillna with TRAINING-period global mean only
              — no future leakage whatsoever
  V3: ZERO-OUT-ON-TEST-REGION: for spatial OOD, set cell_fire_rate_exp = 0 on
                                South test cells (simulate unseen-region cold-start)
  V4: LATITUDE-BAND PROXY: for South test cells, replace cell_fire_rate_exp with
                           the mean value of North training cells at the closest
                           latitude band (warm-start via geographic transfer)
  V5: DROP: remove cell_fire_rate_exp, cell_fire_rate_dry_exp, cell_fire_count_30d
            entirely (full cold-start, 35 features)

Each variant is evaluated on:
  - IID temporal (train 2018-21 all Vietnam, val 2022 all Vietnam, test 2023 all Vietnam)
  - Spatial OOD (train North 2018-21, val North 2022, test South 2023)

Outputs: /home/z/my-project/download/sc_pmh/e1_fire_history_audit/
"""
from __future__ import annotations
import json, os, sys, time, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'e1_fire_history_audit'
EX.mkdir(parents=True, exist_ok=True)
PANEL_P = SC / 'panel_with_fwi.parquet'  # has all 38 v8.7.1 features + 6 FWI
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']  # 38 features
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

# --- Load panel ---
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')

# === Step 1: Compute STRICT version of fire-history features ===
print("[1/6] computing STRICT fire-history features (no fillna leakage)...")
# Compute train-period global means (used for fillna instead of full-panel means)
train_global_mean_fire = panel[(panel.year <= 2021)].fire_modis.mean()
print(f"  train_global_mean_fire = {train_global_mean_fire:.6f}")
print(f"  full_panel_mean_fire   = {panel.fire_modis.mean():.6f}  (used in v8.7.1 fillna)")

panel_strict = panel.copy()
# Compute fire_prev (already done in panel? Actually panel has fire_modis, not fire_prev)
# The v8.7.1 panel HAS cell_fire_rate_exp already (computed with fillna=panel.mean)
# We need to RECOMPUTE it with strict fillna
panel_strict['fire_prev'] = panel_strict.groupby(['lat', 'lon'])['fire_modis'].shift(1)
panel_strict['cell_fire_rate_exp_strict'] = panel_strict.groupby(['lat', 'lon'])['fire_prev'].transform(
    lambda s: s.expanding(min_periods=1).mean()
).fillna(train_global_mean_fire)

panel_strict['cell_fire_count_30d_strict'] = panel_strict.groupby(['lat', 'lon'])['fire_prev'].transform(
    lambda s: s.rolling(30, min_periods=1).sum()
)
# Dry-season version: similar but only counts dry-season days
panel_strict['is_dry'] = panel_strict['date'].dt.month.apply(lambda m: 1 if m in [1,2,3,4,11,12] else 0)
panel_strict['is_dry_prev'] = panel_strict.groupby(['lat', 'lon'])['is_dry'].shift(1)
# For cell_fire_rate_dry_exp, use cumulative sum of (fire_prev * is_dry_prev) / cumulative sum of is_dry_prev
def dry_rate_strict(g):
    num = (g['fire_prev'] * g['is_dry_prev']).expanding(min_periods=1).sum()
    den = g['is_dry_prev'].expanding(min_periods=1).sum() + 1e-6
    return num / den
panel_strict['cell_fire_rate_dry_exp_strict'] = panel_strict.groupby(['lat', 'lon']).apply(
    dry_rate_strict
).reset_index(level=[0, 1], drop=True).fillna(train_global_mean_fire)
panel_strict = panel_strict.drop(columns=['fire_prev', 'is_dry_prev'])

# Sanity check: compare original vs strict
print(f"\n  Original cell_fire_rate_exp stats: mean={panel['cell_fire_rate_exp'].mean():.4f} std={panel['cell_fire_rate_exp'].std():.4f}")
print(f"  Strict  cell_fire_rate_exp stats: mean={panel_strict['cell_fire_rate_exp_strict'].mean():.4f} std={panel_strict['cell_fire_rate_exp_strict'].std():.4f}")
diff = (panel['cell_fire_rate_exp'].values - panel_strict['cell_fire_rate_exp_strict'].values)
print(f"  Diff: mean={np.nanmean(diff):.6f}  max_abs={np.nanmax(np.abs(diff)):.6f}  nonzero_rows={np.sum(np.abs(diff) > 1e-9)}")

# === Step 2: Compute LATITUDE-BAND PROXY for South cells ===
print("\n[2/6] computing LATITUDE-BAND PROXY fire-history for South cells...")
# For South cells, replace cell_fire_rate_exp with the mean of North training cells at the closest latitude
# Build latitude-band lookup from North training data (per-latitude mean of the feature)
north_train = panel_strict[(panel_strict.region == 'North') & (panel_strict.year <= 2021)].copy()
lat_band_lookup = north_train.groupby(['lat', 'date'])[['cell_fire_rate_exp_strict', 'cell_fire_rate_dry_exp_strict', 'cell_fire_count_30d_strict']].mean().reset_index()
# For South cells: match by DATE and closest available North latitude (the closest one >=14°)
# Build a per-date lookup: for each date, lat_band_lookup has values per North latitude
south = panel_strict[panel_strict.region == 'South'].copy()
# Match each (date, lat) South row to the closest North lat available on that date
# Strategy: for each South row, find the closest North lat (>=14) and join
north_lats_sorted = np.array(sorted(lat_band_lookup.lat.unique()))
def find_closest_north_lat(south_lat):
    # South lats are 11.0-13.5, North lats are 14.0-21.0; closest is always 14.0
    # but let's be general
    valid = north_lats_sorted[north_lats_sorted >= 14]
    if len(valid) == 0:
        return None
    return valid[np.argmin(np.abs(valid - south_lat))]
closest_north = {lat: find_closest_north_lat(lat) for lat in south.lat.unique()}
print(f"  South lat → closest North lat mapping: {closest_north}")
south['closest_north_lat'] = south.lat.map(closest_north)
south_proxy = south.merge(
    lat_band_lookup.rename(columns={'lat': 'closest_north_lat',
                                     'cell_fire_rate_exp_strict': 'cell_fire_rate_exp_proxy',
                                     'cell_fire_rate_dry_exp_strict': 'cell_fire_rate_dry_exp_proxy',
                                     'cell_fire_count_30d_strict': 'cell_fire_count_30d_proxy'}),
    on=['closest_north_lat', 'date'],
    how='left'
)
# Now south_proxy has both _strict (real) and _proxy (transferred)
print(f"  South rows matched to proxy: {south_proxy['cell_fire_rate_exp_proxy'].notna().sum()}/{len(south_proxy)}")

# === Step 3: Build the 5 variant panels ===
print("\n[3/6] building 5 variant feature sets...")

def make_variant_features(panel_full, south_proxy_df, variant):
    """
    Returns a single DataFrame with columns: lat, lon, date, region, year, label, + features
    depending on variant:
      V1_original: original v8.7.1 features (38)
      V2_strict: same 38 but fire-hist columns replaced with _strict versions
      V3_zero_south: same as V2 but fire-hist cols = 0 for South cells
      V4_proxy_south: same as V2 but fire-hist cols replaced with _proxy for South cells
      V5_drop: 35 features (no fire-hist)
    """
    cols = ['lat', 'lon', 'date', 'region', 'year', LABEL_COL]
    feat_cols = list(BASE_FEATS)
    df = panel_full[cols + feat_cols].copy()
    
    if variant == 'V1_original':
        pass  # use as-is
    elif variant == 'V2_strict':
        # Replace fire-hist columns with strict versions
        for orig, strict in zip(FIRE_HIST, ['cell_fire_rate_exp_strict', 'cell_fire_rate_dry_exp_strict', 'cell_fire_count_30d_strict']):
            df[orig] = panel_full[strict].values
    elif variant == 'V3_zero_south':
        # Replace with strict, then zero-out for South cells
        for orig, strict in zip(FIRE_HIST, ['cell_fire_rate_exp_strict', 'cell_fire_rate_dry_exp_strict', 'cell_fire_count_30d_strict']):
            df[orig] = panel_full[strict].values
        south_mask = df.region == 'South'
        for orig in FIRE_HIST:
            df.loc[south_mask, orig] = 0.0
    elif variant == 'V4_proxy_south':
        # Replace with strict globally, then for South cells, use proxy values
        for orig, strict in zip(FIRE_HIST, ['cell_fire_rate_exp_strict', 'cell_fire_rate_dry_exp_strict', 'cell_fire_count_30d_strict']):
            df[orig] = panel_full[strict].values
        # Now overwrite South with proxy
        # Need to merge proxy back in
        proxy_cols = ['lat', 'lon', 'date', 'cell_fire_rate_exp_proxy', 'cell_fire_rate_dry_exp_proxy', 'cell_fire_count_30d_proxy']
        df = df.merge(south_proxy_df[proxy_cols], on=['lat', 'lon', 'date'], how='left')
        south_mask = df.region == 'South'
        df.loc[south_mask, 'cell_fire_rate_exp'] = df.loc[south_mask, 'cell_fire_rate_exp_proxy']
        df.loc[south_mask, 'cell_fire_rate_dry_exp'] = df.loc[south_mask, 'cell_fire_rate_dry_exp_proxy']
        df.loc[south_mask, 'cell_fire_count_30d'] = df.loc[south_mask, 'cell_fire_count_30d_proxy']
        df = df.drop(columns=[c for c in proxy_cols if c not in ['lat','lon','date']])
    elif variant == 'V5_drop':
        df = df.drop(columns=FIRE_HIST)
        feat_cols = COLD_FEATS
    else:
        raise ValueError(variant)
    return df, feat_cols

# === Step 4: Train + eval each variant on both IID and spatial OOD splits ===
print("\n[4/6] training 5 variants × 5 seeds × 2 splits...")

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

VARIANTS = ['V1_original', 'V2_strict', 'V3_zero_south', 'V4_proxy_south', 'V5_drop']
SPLITS = ['iid_temporal', 'spatial_ood']

rows = []
for variant in VARIANTS:
    df_v, feats_v = make_variant_features(panel_strict, south_proxy, variant)
    print(f"\n--- {variant} ({len(feats_v)} feats) ---")
    for split in SPLITS:
        if split == 'iid_temporal':
            train = df_v[df_v.year <= 2021].dropna(subset=[LABEL_COL]).copy()
            val   = df_v[df_v.year == 2022].dropna(subset=[LABEL_COL]).copy()
            test  = df_v[df_v.year == 2023].dropna(subset=[LABEL_COL]).copy()
        else:  # spatial_ood
            train = df_v[(df_v.region == 'North') & (df_v.year <= 2021)].dropna(subset=[LABEL_COL]).copy()
            val   = df_v[(df_v.region == 'North') & (df_v.year == 2022)].dropna(subset=[LABEL_COL]).copy()
            test  = df_v[(df_v.region == 'South') & (df_v.year == 2023)].dropna(subset=[LABEL_COL]).copy()
        y_tr = train[LABEL_COL].astype(int).values
        y_va = val[LABEL_COL].astype(int).values
        y_te = test[LABEL_COL].astype(int).values
        print(f"  {split}: train={len(train)} val={len(val)} test={len(test)} (pos_rate train={y_tr.mean():.4f} test={y_te.mean():.4f})")

        for seed in SEEDS:
            params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                          max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, reg_lambda=1.0, seed=seed, verbosity=0)
            dtr = gbm.DMatrix(train[feats_v], label=y_tr)
            dv  = gbm.DMatrix(val[feats_v], label=y_va)
            dte = gbm.DMatrix(test[feats_v], label=y_te)
            m = gbm.train(params, dtr, num_boost_round=500, evals=[(dtr, 'tr'), (dv, 'val')],
                          early_stopping_rounds=30, verbose_eval=False)
            p_va = m.predict(dv); p_te = m.predict(dte)
            iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
            p_te_iso = iso.predict(p_te)
            rows.append(dict(variant=variant, split=split, seed=seed, calibrated=False,
                             **metrics(p_te, y_te)))
            rows.append(dict(variant=variant, split=split, seed=seed, calibrated=True,
                             **metrics(p_te_iso, y_te)))

# === Step 5: Summary ===
print("\n[5/6] summary...")
df_results = pd.DataFrame(rows)
df_results.to_csv(EX / 'e1_results.csv', index=False)

# Aggregate
agg = df_results.groupby(['variant', 'split', 'calibrated']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
).reset_index()
agg.to_csv(EX / 'e1_summary.csv', index=False)

print("\n=== IID temporal split (train 2018-21 all VN, val 2022 all VN, test 2023 all VN) ===")
for cal in [False, True]:
    print(f"\n  calibrated={cal}:")
    sub = agg[(agg.split == 'iid_temporal') & (agg.calibrated == cal)].sort_values('AUROC_mean', ascending=False)
    print(sub.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

print("\n=== Spatial OOD split (train North 2018-21, val North 2022, test South 2023) ===")
for cal in [False, True]:
    print(f"\n  calibrated={cal}:")
    sub = agg[(agg.split == 'spatial_ood') & (agg.calibrated == cal)].sort_values('AUROC_mean', ascending=False)
    print(sub.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# === Step 6: Paired delta vs V1 on spatial_ood, uncalibrated ===
print("\n[6/6] paired Δ (variant − V1) on spatial_ood, n=5 seeds, uncalibrated:")
per_seed = {}
for r in rows:
    if r['split'] != 'spatial_ood' or r['calibrated']: continue
    per_seed.setdefault(r['seed'], {})[r['variant']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}

print(f"{'variant':<18} {'ΔAUROC':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22} {'ΔECE':>22}")
print("-" * 110)
for vs in ['V2_strict', 'V3_zero_south', 'V4_proxy_south', 'V5_drop']:
    line = f"{vs:<18} "
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([per_seed[s][vs][metric] - per_seed[s]['V1_original'][metric] for s in SEEDS])
        mean, std = float(diff.mean()), float(diff.std())
        n = len(SEEDS)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""
        line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
    print(line)

# Same for IID
print("\n=== paired Δ (variant − V1) on iid_temporal, n=5 seeds, uncalibrated ===")
per_seed_iid = {}
for r in rows:
    if r['split'] != 'iid_temporal' or r['calibrated']: continue
    per_seed_iid.setdefault(r['seed'], {})[r['variant']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}

print(f"{'variant':<18} {'ΔAUROC':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22} {'ΔECE':>22}")
print("-" * 110)
for vs in ['V2_strict', 'V3_zero_south', 'V4_proxy_south', 'V5_drop']:
    line = f"{vs:<18} "
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([per_seed_iid[s][vs][metric] - per_seed_iid[s]['V1_original'][metric] for s in SEEDS])
        mean, std = float(diff.mean()), float(diff.std())
        n = len(SEEDS)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""
        line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
    print(line)

# Save summary JSON
with open(EX / 'e1_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'variants': VARIANTS,
        'splits': SPLITS,
        'iid_summary': agg[agg.split == 'iid_temporal'].to_dict('records'),
        'spatial_ood_summary': agg[agg.split == 'spatial_ood'].to_dict('records'),
        'findings': {
            'fillna_leakage_magnitude': f"max_abs_diff = {np.nanmax(np.abs(diff)):.6f} across {np.sum(np.abs(diff) > 1e-9)} rows out of 598143",
            'closest_north_lat_mapping': {float(k): float(v) for k, v in closest_north.items()},
        },
    }, f, indent=2, default=str)

print(f"\n[done] saved → {EX}/e1_summary.csv, e1_summary.json")
