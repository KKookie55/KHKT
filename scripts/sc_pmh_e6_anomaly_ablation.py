#!/usr/bin/env python3
"""
SC-PMH E6: Anomaly Ablation — Leave-One-Out
=============================================
Goal: identify which anomaly features actually drive the improvement.
Drop one anomaly feature at a time and measure the loss.

Setup:
  - Split: cold spatial OOD (train North 2018-21, val North 2022, test South 2023)
  - Baseline: cold + ALL 8 anomaly features (43 feats)
  - For each of the 8 anomaly features, drop just that one (42 feats)
  - Also test "drop composite only", "drop 7 doy anomalies only", "drop 7 + composite"
  - Compare to: cold baseline (no anomalies, 35 feats) and cold + ALL anomalies (43)

5 seeds per variant. Output: which features matter most.
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
EX = SC / 'e6_anomaly_ablation'
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
ALL_ANOM = ANOM_FEATS + COMPOSITE_FEAT  # 8 features
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')

train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL])
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL])
test  = panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL])
y_tr = train[LABEL_COL].astype(int).values
y_va = val[LABEL_COL].astype(int).values
y_te = test[LABEL_COL].astype(int).values
print(f"train={len(train)} val={len(val)} test={len(test)} pos_test={y_te.mean():.4f}")

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

# Variants:
# - cold_baseline: 35 feats (no anomalies)
# - cold_all_anom: 35 + 8 = 43 (all anomalies)
# - drop_one_<feat>: 35 + 7 = 42 (drop one anomaly at a time)
# - drop_composite: 35 + 7 doy = 42 (drop only the composite)
# - drop_doy_only: 35 + 1 composite = 36 (drop all 7 doy anomalies, keep composite)
# - drop_rolling: 35 + 4 (1 doy + composite) = 39 wait actually 5 = drop the 3 rolling-anomaly feats
VARIANTS = {
    'cold_baseline': COLD_FEATS,
    'cold_all_anom': COLD_FEATS + ALL_ANOM,
}
# Leave-one-out
for feat in ALL_ANOM:
    drop_name = f'drop_{feat}'
    VARIANTS[drop_name] = COLD_FEATS + [f for f in ALL_ANOM if f != feat]
# Group drops
VARIANTS['drop_all_rolling_anom'] = COLD_FEATS + ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy', 'dry_anomaly_composite']  # drop the 3 _roll7 versions
VARIANTS['drop_all_doy_anom'] = COLD_FEATS + ['t2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7', 'dry_anomaly_composite']  # drop the 4 _doy versions
VARIANTS['drop_composite'] = COLD_FEATS + ANOM_FEATS  # drop only composite
VARIANTS['only_composite'] = COLD_FEATS + ['dry_anomaly_composite']  # only composite, no individual
VARIANTS['only_doy_no_composite'] = COLD_FEATS + ANOM_FEATS  # only doy anomalies, no composite

print(f"\n[info] {len(VARIANTS)} variants:")
for vname, feats in VARIANTS.items():
    print(f"  {vname}: {len(feats)} feats")

rows = []
for variant_name, feats in VARIANTS.items():
    print(f"\n  variant={variant_name} ({len(feats)} feats)")
    for seed in SEEDS:
        params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, reg_lambda=1.0, seed=seed, verbosity=0)
        dtr = gbm.DMatrix(train[feats], label=y_tr)
        dv  = gbm.DMatrix(val[feats], label=y_va)
        dte = gbm.DMatrix(test[feats], label=y_te)
        m = gbm.train(params, dtr, num_boost_round=300, evals=[(dtr, 'tr'), (dv, 'val')],
                      early_stopping_rounds=20, verbose_eval=False)
        p_va = m.predict(dv); p_te = m.predict(dte)
        iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
        p_te_iso = iso.predict(p_te)
        m_te = metrics(p_te, y_te)
        rows.append(dict(variant=variant_name, seed=seed, calibrated=False, **m_te))

df = pd.DataFrame(rows)
df.to_csv(EX / 'e6_results.csv', index=False)

# Aggregate
agg = df.groupby(['variant', 'calibrated']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
).reset_index()
agg.to_csv(EX / 'e6_summary.csv', index=False)

# Per-variant summary
print("\n=== Per variant (mean ± std, n=5 seeds, uncalibrated) ===")
print(f"{'variant':<35} {'AUROC':>15} {'PR_AUC':>15} {'Brier':>15} {'ECE':>15}")
print("-" * 100)
sub = agg[~agg.calibrated].sort_values('AUROC_mean', ascending=False)
for _, r in sub.iterrows():
    print(f"{r.variant:<35} {r.AUROC_mean:>10.4f}±{r.AUROC_std:.4f} {r.PR_AUC_mean:>10.4f}±{r.PR_AUC_std:.4f} {r.Brier_mean:>10.4f}±{r.Brier_std:.4f} {r.ECE_mean:>10.4f}±{r.ECE_std:.4f}")

# Paired delta vs cold_all_anom (the full anomaly set)
print("\n=== Paired Δ (variant − cold_all_anom) — lower Δ means dropping this feature HURTS ===")
all_anom_per_seed = {}
for r in rows:
    if r['variant'] != 'cold_all_anom' or r['calibrated']: continue
    all_anom_per_seed[r['seed']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}

print(f"{'variant':<35} {'ΔAUROC':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22} {'ΔECE':>22}")
print("-" * 130)
# Skip variants that are subsets
SKIP = ['cold_baseline', 'cold_all_anom', 'only_composite', 'only_doy_no_composite', 'drop_all_rolling_anom', 'drop_all_doy_anom', 'drop_composite']
for variant_name in [v for v in VARIANTS.keys() if v not in SKIP]:
    sub = df[df.variant == variant_name]
    per_seed_v = {}
    for _, r in sub.iterrows():
        if r['calibrated']: continue
        per_seed_v[r['seed']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
    line = f"{variant_name:<35} "
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([per_seed_v[s][metric] - all_anom_per_seed[s][metric] for s in SEEDS if s in per_seed_v])
        if len(diff) == 0:
            line += f"{'N/A':>22} "
            continue
        mean, std = float(diff.mean()), float(diff.std())
        n = len(diff)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""
        line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
    print(line)

# Group drops summary
print("\n=== Group drops (drop a group, measure loss) ===")
print(f"{'variant':<35} {'ΔAUROC vs cold_all_anom':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22}")
print("-" * 110)
for variant_name in ['cold_baseline', 'only_composite', 'only_doy_no_composite', 'drop_composite', 'drop_all_rolling_anom', 'drop_all_doy_anom']:
    sub = df[df.variant == variant_name]
    per_seed_v = {}
    for _, r in sub.iterrows():
        if r['calibrated']: continue
        per_seed_v[r['seed']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
    line = f"{variant_name:<35} "
    for metric in ['AUROC', 'PR_AUC', 'Brier']:
        diff = np.array([per_seed_v[s][metric] - all_anom_per_seed[s][metric] for s in SEEDS if s in per_seed_v])
        if len(diff) == 0:
            line += f"{'N/A':>22} "
            continue
        mean, std = float(diff.mean()), float(diff.std())
        n = len(diff)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""
        line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
    print(line)

with open(EX / 'e6_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'variants': list(VARIANTS.keys()),
        'variant_descriptions': {
            'cold_baseline': 'no anomaly features (35 feats)',
            'cold_all_anom': 'all 8 anomaly features (43 feats)',
            'drop_<feat>': 'leave-one-out: drop one anomaly feature',
            'only_composite': 'only dry_anomaly_composite (36 feats)',
            'only_doy_no_composite': '7 doy/rolling anomalies, no composite (42 feats)',
            'drop_composite': 'drop only the composite (42 feats)',
            'drop_all_rolling_anom': 'drop 3 _roll7 versions (38 feats)',
            'drop_all_doy_anom': 'drop 4 _doy versions, keep rolling + composite (39 feats)',
        },
        'summary': agg.to_dict('records'),
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/e6_summary.csv, e6_summary.json")
