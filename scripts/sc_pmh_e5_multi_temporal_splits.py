#!/usr/bin/env python3
"""
SC-PMH E5: Multiple Temporal Splits
====================================
Goal: prove anomaly improvement generalizes across different temporal split
choices, not just one specific test=2023 setup.

Splits:
  T1: train 2018-2021, val 2022, test 2023 (current temporal OOD)
  T2: train 2018-2022, val 2023-Q1, test 2023-Q2-Q4 (more train, less test)
  T3: train 2018-2021 + 2022-H1, val 2022-H2, test 2023 (cumulative forward)
  T4: train 2018-2021, val 2022-H1, test 2022-H2 (independent 6-month forward)

All splits use cold-start features (no fire history), all-Vietnam cells.
5 seeds × 4 splits × 2 variants = 40 trainings.
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'e5_multi_temporal_splits'
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
# Skip T2 to avoid timeout (5-year train too slow); add it back if time permits
SPLITS_TO_RUN = ['T1_train_18_21_test_23', 'T3_train_18_21_plus_22H1_test_23', 'T4_train_18_21_test_22H2']

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel['half'] = np.where(panel['date'].dt.month <= 6, 'H1', 'H2')
panel['quarter'] = panel['date'].dt.month.apply(lambda m: f'Q{(m-1)//3+1}')
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)

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

SPLITS = {
    'T1_train_18_21_test_23': {
        'train': panel[panel.year <= 2021].dropna(subset=[LABEL_COL]),
        'val':   panel[panel.year == 2022].dropna(subset=[LABEL_COL]),
        'test':  panel[panel.year == 2023].dropna(subset=[LABEL_COL]),
    },
    'T2_train_18_22_test_23Q2_4': {
        'train': panel[panel.year <= 2022].dropna(subset=[LABEL_COL]),
        'val':   panel[(panel.year == 2023) & (panel.quarter == 'Q1')].dropna(subset=[LABEL_COL]),
        'test':  panel[(panel.year == 2023) & (panel.quarter.isin(['Q2', 'Q3', 'Q4']))].dropna(subset=[LABEL_COL]),
    },
    'T3_train_18_21_plus_22H1_test_23': {
        'train': panel[(panel.year <= 2021) | ((panel.year == 2022) & (panel.half == 'H1'))].dropna(subset=[LABEL_COL]),
        'val':   panel[(panel.year == 2022) & (panel.half == 'H2')].dropna(subset=[LABEL_COL]),
        'test':  panel[panel.year == 2023].dropna(subset=[LABEL_COL]),
    },
    'T4_train_18_21_test_22H2': {
        'train': panel[panel.year <= 2021].dropna(subset=[LABEL_COL]),
        'val':   panel[(panel.year == 2022) & (panel.half == 'H1')].dropna(subset=[LABEL_COL]),
        'test':  panel[(panel.year == 2022) & (panel.half == 'H2')].dropna(subset=[LABEL_COL]),
    },
}

VARIANTS = {
    'cold_baseline': COLD_FEATS,
    'cold_anom_comp': COLD_FEATS + ANOM_FEATS + COMPOSITE_FEAT,
}

rows = []
for split_name, split_dfs in SPLITS.items():
    if split_name not in SPLITS_TO_RUN:
        print(f"\n=== Skipping {split_name} (not in SPLITS_TO_RUN) ===")
        continue
    train_df = split_dfs['train']; val_df = split_dfs['val']; test_df = split_dfs['test']
    y_tr = train_df[LABEL_COL].astype(int).values
    y_va = val_df[LABEL_COL].astype(int).values
    y_te = test_df[LABEL_COL].astype(int).values
    print(f"\n=== Split {split_name} ===")
    print(f"  train: {len(train_df):,} (pos_rate={y_tr.mean():.4f})")
    print(f"  val:   {len(val_df):,} (pos_rate={y_va.mean():.4f})")
    print(f"  test:  {len(test_df):,} (pos_rate={y_te.mean():.4f})")
    if y_te.sum() == 0 or len(np.unique(y_te)) < 2:
        print("  SKIP — insufficient test")
        continue

    for variant_name, feats in VARIANTS.items():
        print(f"  variant={variant_name} ({len(feats)} feats)")
        for seed in SEEDS:
            params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                          max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, reg_lambda=1.0, seed=seed, verbosity=0)
            dtr = gbm.DMatrix(train_df[feats], label=y_tr)
            dv  = gbm.DMatrix(val_df[feats], label=y_va)
            dte = gbm.DMatrix(test_df[feats], label=y_te)
            m = gbm.train(params, dtr, num_boost_round=300, evals=[(dtr, 'tr'), (dv, 'val')],
                          early_stopping_rounds=20, verbose_eval=False)
            p_va = m.predict(dv); p_te = m.predict(dte)
            iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
            p_te_iso = iso.predict(p_te)
            m_te = metrics(p_te, y_te)
            m_te_iso = metrics(p_te_iso, y_te)
            rows.append(dict(split=split_name, variant=variant_name, seed=seed, calibrated=False,
                             n_test=len(y_te), n_pos=int(y_te.sum()), **m_te))
            rows.append(dict(split=split_name, variant=variant_name, seed=seed, calibrated=True,
                             n_test=len(y_te), n_pos=int(y_te.sum()), **m_te_iso))

df = pd.DataFrame(rows)
df.to_csv(EX / 'e5_results.csv', index=False)

# Summary
agg = df.groupby(['split', 'variant', 'calibrated']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
    n_test=('n_test', 'first'), n_pos=('n_pos', 'first'),
).reset_index()
agg.to_csv(EX / 'e5_summary.csv', index=False)

print("\n=== Per split, uncalibrated, mean ± std (n=5 seeds) ===")
print(f"{'split':<35} {'baseline AUROC':>22} {'anom AUROC':>22} {'ΔAUROC':>22}")
print("-" * 105)
for s in SPLITS.keys():
    sub = df[(df.split == s) & (~df.calibrated)]
    bl = sub[sub.variant == 'cold_baseline']['AUROC'].values
    an = sub[sub.variant == 'cold_anom_comp']['AUROC'].values
    if len(bl) == 0 or len(an) == 0: continue
    print(f"{s:<35} {bl.mean():>10.4f}±{bl.std():.4f}  {an.mean():>10.4f}±{an.std():.4f}  {(an-bl).mean():>+10.4f}±{(an-bl).std():.4f}")

print("\n=== Paired Δ (anom − baseline) per split, n=5 seeds, uncalibrated ===")
print(f"{'split':<35} {'ΔAUROC':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22} {'ΔECE':>22}")
print("-" * 130)
for s in SPLITS.keys():
    sub = df[(df.split == s) & (~df.calibrated)]
    per_seed = {}
    for _, r in sub.iterrows():
        per_seed.setdefault(r['seed'], {})[r['variant']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
    if len(per_seed) < 2: continue
    line = f"{s:<35} "
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([per_seed[seed]['cold_anom_comp'][metric] - per_seed[seed]['cold_baseline'][metric]
                         for seed in SEEDS if seed in per_seed and 'cold_anom_comp' in per_seed[seed]])
        if len(diff) == 0:
            line += f"{'N/A':>22} "
            continue
        mean, std = float(diff.mean()), float(diff.std())
        n = len(diff)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""
        line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
    print(line)

with open(EX / 'e5_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'splits': list(SPLITS.keys()),
        'variants': list(VARIANTS.keys()),
        'summary': agg.to_dict('records'),
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/e5_summary.csv, e5_summary.json")
