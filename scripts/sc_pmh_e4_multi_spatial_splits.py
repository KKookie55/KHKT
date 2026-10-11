#!/usr/bin/env python3
"""
SC-PMH E4: Multiple Spatial Split Boundaries
==============================================
Goal: prove anomaly improvement is NOT an artifact of one specific lat=14°
boundary. Test boundaries at lat ∈ {12, 13, 14, 15, 16}.

For each boundary:
  - Train = cells with lat >= boundary, years 2018-2021 (cold-start features only)
  - Val   = cells with lat >= boundary, year 2022
  - Test  = cells with lat <  boundary, year 2023
  - South cells in test get their anomaly features (computed from their own climatology,
    which is valid because anomaly climatology uses only 2018-2021 train-period data)

Variants compared:
  M1c_cold (35 feats)  vs  M1c_cold + 8 anomalies (43 feats)

5 seeds × 5 boundaries × 2 variants = 50 trainings per direction = 100 total runs.
Output: per-(boundary, variant, seed) metrics + paired Δ with bootstrap CI.
"""
from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'e4_multi_spatial_splits'
EX.mkdir(parents=True, exist_ok=True)
PANEL_P = SC / 'e2_weather_anomalies' / 'panel_with_anomalies.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
COMPOSITE_FEAT = ['dry_anomaly_composite']
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]
BOUNDARIES = [12.0, 13.0, 14.0, 15.0, 16.0]

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
print(f"[info] panel shape={panel.shape}, lats: {sorted(panel.lat.unique())}")

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

VARIANTS = {
    'cold_baseline': COLD_FEATS,
    'cold_anom_comp': COLD_FEATS + ANOM_FEATS + COMPOSITE_FEAT,  # 43 feats
}

rows = []
for boundary in BOUNDARIES:
    print(f"\n=== Boundary lat={boundary}° (train cells lat≥{boundary}, test cells lat<{boundary}) ===")
    train = panel[(panel.lat >= boundary) & (panel.year <= 2021)].dropna(subset=[LABEL_COL])
    val   = panel[(panel.lat >= boundary) & (panel.year == 2022)].dropna(subset=[LABEL_COL])
    test  = panel[(panel.lat < boundary) & (panel.year == 2023)].dropna(subset=[LABEL_COL])
    n_train_cells = train[['lat','lon']].drop_duplicates().shape[0]
    n_test_cells = test[['lat','lon']].drop_duplicates().shape[0]
    print(f"  train: {len(train):,} rows / {n_train_cells} cells / pos_rate={train[LABEL_COL].mean():.4f}")
    print(f"  val:   {len(val):,} rows / pos_rate={val[LABEL_COL].mean():.4f}")
    print(f"  test:  {len(test):,} rows / {n_test_cells} cells / pos_rate={test[LABEL_COL].mean():.4f}")
    if len(test) == 0 or test[LABEL_COL].sum() == 0:
        print(f"  SKIP — no test data or no positives")
        continue

    y_tr = train[LABEL_COL].astype(int).values
    y_va = val[LABEL_COL].astype(int).values
    y_te = test[LABEL_COL].astype(int).values

    for variant_name, feats in VARIANTS.items():
        print(f"  variant={variant_name} ({len(feats)} feats)")
        for seed in SEEDS:
            params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                          max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, reg_lambda=1.0, seed=seed, verbosity=0)
            dtr = gbm.DMatrix(train[feats], label=y_tr)
            dv  = gbm.DMatrix(val[feats], label=y_va)
            dte = gbm.DMatrix(test[feats], label=y_te)
            m = gbm.train(params, dtr, num_boost_round=500, evals=[(dtr, 'tr'), (dv, 'val')],
                          early_stopping_rounds=30, verbose_eval=False)
            p_va = m.predict(dv); p_te = m.predict(dte)
            iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
            p_te_iso = iso.predict(p_te)
            m_te = metrics(p_te, y_te)
            m_te_iso = metrics(p_te_iso, y_te)
            rows.append(dict(boundary=boundary, variant=variant_name, seed=seed, calibrated=False, n_test=len(y_te), n_pos=int(y_te.sum()), **m_te))
            rows.append(dict(boundary=boundary, variant=variant_name, seed=seed, calibrated=True, n_test=len(y_te), n_pos=int(y_te.sum()), **m_te_iso))

df = pd.DataFrame(rows)
df.to_csv(EX / 'e4_results.csv', index=False)

# Summary per boundary
agg = df.groupby(['boundary', 'variant', 'calibrated']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
    n_test=('n_test', 'first'), n_pos=('n_pos', 'first'),
).reset_index()
agg.to_csv(EX / 'e4_summary.csv', index=False)

print("\n=== Per boundary, uncalibrated ===")
sub = agg[~agg.calibrated].copy()
sub = sub.pivot(index='boundary', columns='variant', values=['AUROC_mean', 'PR_AUC_mean', 'Brier_mean', 'ECE_mean'])
print(sub.to_string(float_format=lambda v: f'{v:.4f}'))

# Paired delta per boundary
print("\n=== Paired Δ (cold_anom_comp − cold_baseline) per boundary ===")
print(f"{'boundary':>10} {'ΔAUROC':>20} {'ΔPR_AUC':>20} {'ΔBrier':>20} {'ΔECE':>20}")
print("-" * 95)
for boundary in BOUNDARIES:
    per_seed = {}
    for r in rows:
        if r['boundary'] != boundary or r['calibrated']: continue
        per_seed.setdefault(r['seed'], {})[r['variant']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
    if len(per_seed) < 2: continue
    line = f"{boundary:>10.1f} "
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([per_seed[s]['cold_anom_comp'][metric] - per_seed[s]['cold_baseline'][metric] for s in SEEDS if s in per_seed[s] if 'cold_anom_comp' in per_seed[s]])
        if len(diff) == 0:
            line += f"{'N/A':>20} "
            continue
        mean, std = float(diff.mean()), float(diff.std())
        n = len(diff)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""
        line += f" {mean:+.5f}±{std:.5f} (t={tstat:.2f}{sig})"
    print(line)

# Save summary
with open(EX / 'e4_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'boundaries': BOUNDARIES,
        'variants': list(VARIANTS.keys()),
        'summary': agg.to_dict('records'),
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/e4_summary.csv, e4_summary.json")
