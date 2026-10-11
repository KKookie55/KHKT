#!/usr/bin/env python3
"""
SC-PMH E3: Temporal OOD + Spatial OOD on the same pipeline
============================================================
Goal: test whether the anomaly features (from E2) generalize across BOTH
space (train North, test South) AND time (train 2018-2021, test 2023).
The user's plan called for "kiểm tra khả năng chuyển miền có điều kiện"
and "kiểm định thời gian riêng để đánh giá cả hai khả năng tổng quát hóa".

Setup (cold-start, no fire-history features, 35 base + anomaly features):

  Split A — Spatial OOD (warm-start):
    train North 2018-21 + fire history, val North 2022 + fire history, test South 2023 + fire history
    (the "warm" version with real fire history — current SC-PMH H1 setup)

  Split B — Spatial OOD (cold-start):
    train North 2018-21, val North 2022, test South 2023 — all with fire-history features DROPPED

  Split C — Temporal OOD (cold-start):
    train 2018-21 all Vietnam, val 2022 all Vietnam, test 2023 all Vietnam — fire-history DROPPED

  Split D — Spatial+Temporal OOD (cold-start):
    train North 2018-21, val North 2022, test South 2023 — fire-history DROPPED
    (same as B but interpreted as combined OOD)

Wait — Split B and D are identical. We use them as the spatial OOD cold-start.
So we have 3 meaningful splits: A (warm spatial), B (cold spatial), C (cold temporal).

Variants (per split):
  - Cold baseline (35 features)
  - Cold + 7 anomalies (42 features)
  - Cold + 7 anomalies + composite (43 features)

For Split A (warm-start), we use the FULL feature set (38 base + 7 anomalies + composite = 46).

Outputs: /home/z/my-project/download/sc_pmh/e3_ood_eval/
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
EX = SC / 'e3_ood_eval'
EX.mkdir(parents=True, exist_ok=True)
PANEL_P = SC / 'e2_weather_anomalies' / 'panel_with_anomalies.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']  # 38
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]  # 35
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
COMPOSITE_FEAT = ['dry_anomaly_composite']
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f"[info] panel shape={panel.shape}")

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

# Define splits
SPLITS = {
    'A_warm_spatial_ood': {
        'train': panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL]),
        'val':   panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL]),
        'test':  panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL]),
        'feats_variants': {
            'warm_baseline': BASE_FEATS,
            'warm_baseline_anom': BASE_FEATS + ANOM_FEATS,
            'warm_baseline_anom_comp': BASE_FEATS + ANOM_FEATS + COMPOSITE_FEAT,
        }
    },
    'B_cold_spatial_ood': {
        'train': panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL]),
        'val':   panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL]),
        'test':  panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL]),
        'feats_variants': {
            'cold_baseline': COLD_FEATS,
            'cold_baseline_anom': COLD_FEATS + ANOM_FEATS,
            'cold_baseline_anom_comp': COLD_FEATS + ANOM_FEATS + COMPOSITE_FEAT,
        }
    },
    'C_cold_temporal_ood': {
        'train': panel[panel.year <= 2021].dropna(subset=[LABEL_COL]),
        'val':   panel[panel.year == 2022].dropna(subset=[LABEL_COL]),
        'test':  panel[panel.year == 2023].dropna(subset=[LABEL_COL]),
        'feats_variants': {
            'cold_baseline': COLD_FEATS,
            'cold_baseline_anom': COLD_FEATS + ANOM_FEATS,
            'cold_baseline_anom_comp': COLD_FEATS + ANOM_FEATS + COMPOSITE_FEAT,
        }
    },
}

rows = []
for split_name, split_def in SPLITS.items():
    train_df = split_def['train']; val_df = split_def['val']; test_df = split_def['test']
    y_tr = train_df[LABEL_COL].astype(int).values
    y_va = val_df[LABEL_COL].astype(int).values
    y_te = test_df[LABEL_COL].astype(int).values
    print(f"\n=== Split {split_name} ===")
    print(f"  train={len(train_df)} (pos_rate={y_tr.mean():.4f})  val={len(val_df)}  test={len(test_df)} (pos_rate={y_te.mean():.4f})")
    
    for variant_name, feats in split_def['feats_variants'].items():
        print(f"\n  variant={variant_name} ({len(feats)} feats)")
        for seed in SEEDS:
            params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                          max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, reg_lambda=1.0, seed=seed, verbosity=0)
            dtr = gbm.DMatrix(train_df[feats], label=y_tr)
            dv  = gbm.DMatrix(val_df[feats], label=y_va)
            dte = gbm.DMatrix(test_df[feats], label=y_te)
            m = gbm.train(params, dtr, num_boost_round=500, evals=[(dtr, 'tr'), (dv, 'val')],
                          early_stopping_rounds=30, verbose_eval=False)
            p_va = m.predict(dv); p_te = m.predict(dte)
            iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
            p_te_iso = iso.predict(p_te)
            rows.append(dict(split=split_name, variant=variant_name, seed=seed, calibrated=False, **metrics(p_te, y_te)))
            rows.append(dict(split=split_name, variant=variant_name, seed=seed, calibrated=True, **metrics(p_te_iso, y_te)))

# Summary
df_results = pd.DataFrame(rows)
df_results.to_csv(EX / 'e3_results.csv', index=False)

agg = df_results.groupby(['split', 'variant', 'calibrated']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
).reset_index()
agg.to_csv(EX / 'e3_summary.csv', index=False)

# Print summary per split
for split in SPLITS.keys():
    print(f"\n=== Split {split} ===")
    for cal in [False, True]:
        print(f"\n  calibrated={cal}:")
        sub = agg[(agg.split == split) & (agg.calibrated == cal)].sort_values('AUROC_mean', ascending=False)
        print(sub.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# Paired delta within each split (cold_baseline vs cold_baseline_anom and _anom_comp)
print("\n=== Paired Δ (variant − baseline) per split, n=5 seeds, uncalibrated ===")
for split in SPLITS.keys():
    if split == 'A_warm_spatial_ood':
        baseline = 'warm_baseline'; vs_list = ['warm_baseline_anom', 'warm_baseline_anom_comp']
    else:
        baseline = 'cold_baseline'; vs_list = ['cold_baseline_anom', 'cold_baseline_anom_comp']
    print(f"\n  split={split}, baseline={baseline}:")
    per_seed = {}
    for r in rows:
        if r['split'] != split or r['calibrated']: continue
        per_seed.setdefault(r['seed'], {})[r['variant']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
    print(f"  {'variant':<30} {'ΔAUROC':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22} {'ΔECE':>22}")
    print("  " + "-" * 110)
    for vs in vs_list:
        line = f"  {vs:<30} "
        for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
            diff = np.array([per_seed[s][vs][metric] - per_seed[s][baseline][metric] for s in SEEDS])
            mean, std = float(diff.mean()), float(diff.std())
            n = len(SEEDS)
            tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
            sig = " *" if abs(tstat) > 2.776 else ""
            line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
        print(line)

# Save summary
with open(EX / 'e3_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'splits': list(SPLITS.keys()),
        'variants': ['baseline', 'baseline+anom', 'baseline+anom+composite'],
        'summary': agg.to_dict('records'),
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/e3_summary.csv, e3_summary.json")
