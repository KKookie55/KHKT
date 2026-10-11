#!/usr/bin/env python3
"""
SC-PMH Experiment H1 — COLD-START ablation (no fire-history features)
====================================================================
Goal: test if FWI physical branch contributes useful information when the model
does NOT have access to per-cell fire history (the dominant feature in M1).

Setup: same spatial OOD split (train North 2018-21, val North 2022, test South 2023)
but using only "cold-start features" (no cell_fire_rate_exp, no cell_fire_count_30d,
no cell_fire_rate_dry_exp). This is analogous to the v8.7.1 B1_weather baseline.

Models:
  M1c: XGBoost cold-start (weather-only, ~35 features)
  M4c: M1c + FWI features (41 features)
  M6c: Logistic combiner σ(β0 + β1·logit(p_M1c) + β2·z_FWI)

Outputs: /home/z/my-project/download/sc_pmh/exp_h1_coldstart/
"""
from __future__ import annotations
import json, os, sys, time, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'exp_h1_coldstart'
EX.mkdir(parents=True, exist_ok=True)
PANEL_P = SC / 'panel_with_fwi.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FWI_FEATS = ['fwi', 'fwi_ffmc', 'fwi_dmc', 'fwi_dc', 'fwi_isi', 'fwi_bui']
# Cold-start = remove fire-history features
FIRE_HIST_FEATS = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST_FEATS]
COLD_FEAT_M4 = COLD_FEATS + FWI_FEATS
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

print(f"[info] cold-start features ({len(COLD_FEATS)}): {COLD_FEATS[:5]}...{COLD_FEATS[-3:]}")
print(f"[info] removed fire-history: {FIRE_HIST_FEATS}")

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')

train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL]).copy()
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL]).copy()
test  = panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL]).copy()
test_id = panel[(panel.region == 'North') & (panel.year == 2023)].dropna(subset=[LABEL_COL]).copy()
y_train, y_val, y_test, y_test_id = (
    train[LABEL_COL].astype(int).values,
    val[LABEL_COL].astype(int).values,
    test[LABEL_COL].astype(int).values,
    test_id[LABEL_COL].astype(int).values,
)
print(f"  train={len(train)} val={len(val)} test_south={len(test)} test_north_id={len(test_id)}")

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
for seed in SEEDS:
    print(f"\n[seed={seed}]")
    params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, seed=seed, verbosity=0)

    # M1c: cold-start XGB
    dtr = gbm.DMatrix(train[COLD_FEATS], label=y_train)
    dv  = gbm.DMatrix(val[COLD_FEATS], label=y_val)
    dte = gbm.DMatrix(test[COLD_FEATS], label=y_test)
    dtei = gbm.DMatrix(test_id[COLD_FEATS], label=y_test_id)
    m1c = gbm.train(params, dtr, num_boost_round=500, evals=[(dtr, 'tr'), (dv, 'val')],
                   early_stopping_rounds=30, verbose_eval=False)
    p1v, p1t, p1ti = m1c.predict(dv), m1c.predict(dte), m1c.predict(dtei)

    # M2c: M1c + isotonic
    iso1 = IsotonicRegression(out_of_bounds='clip').fit(p1v, y_val)
    p2v, p2t, p2ti = iso1.predict(p1v), iso1.predict(p1t), iso1.predict(p1ti)

    # M4c: cold-start XGB + FWI
    dtr4 = gbm.DMatrix(train[COLD_FEAT_M4], label=y_train)
    dv4  = gbm.DMatrix(val[COLD_FEAT_M4], label=y_val)
    dte4 = gbm.DMatrix(test[COLD_FEAT_M4], label=y_test)
    dtei4 = gbm.DMatrix(test_id[COLD_FEAT_M4], label=y_test_id)
    m4c = gbm.train(params, dtr4, num_boost_round=500, evals=[(dtr4, 'tr'), (dv4, 'val')],
                   early_stopping_rounds=30, verbose_eval=False)
    p4v, p4t, p4ti = m4c.predict(dv4), m4c.predict(dte4), m4c.predict(dtei4)

    # M5c: M4c + isotonic
    iso4 = IsotonicRegression(out_of_bounds='clip').fit(p4v, y_val)
    p5v, p5t, p5ti = iso4.predict(p4v), iso4.predict(p4t), iso4.predict(p4ti)

    # M6c: logistic combiner
    eps = 1e-6
    z_ml_v = np.log(np.clip(p1v, eps, 1-eps) / np.clip(1-p1v, eps, 1))
    z_ml_t = np.log(np.clip(p1t, eps, 1-eps) / np.clip(1-p1t, eps, 1))
    z_ml_ti = np.log(np.clip(p1ti, eps, 1-eps) / np.clip(1-p1ti, eps, 1))
    fwi_sc = StandardScaler().fit(train[['fwi']])
    z_fwi_v = fwi_sc.transform(val[['fwi']]).ravel()
    z_fwi_t = fwi_sc.transform(test[['fwi']]).ravel()
    z_fwi_ti = fwi_sc.transform(test_id[['fwi']]).ravel()
    m6c = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
        np.column_stack([z_ml_v, z_fwi_v]), y_val)
    p6v = m6c.predict_proba(np.column_stack([z_ml_v, z_fwi_v]))[:, 1]
    p6t = m6c.predict_proba(np.column_stack([z_ml_t, z_fwi_t]))[:, 1]
    p6ti = m6c.predict_proba(np.column_stack([z_ml_ti, z_fwi_ti]))[:, 1]
    print(f"  M6c coefs: β1={m6c.coef_[0,0]:.4f}  β2(z_fwi)={m6c.coef_[0,1]:.4f}  intercept={m6c.intercept_[0]:.4f}")

    for name, p_v, p_t, p_ti in [
        ('M1c_cold', p1v, p1t, p1ti), ('M2c_cold_iso', p2v, p2t, p2ti),
        ('M4c_cold_fwi', p4v, p4t, p4ti), ('M5c_cold_fwi_iso', p5v, p5t, p5ti),
        ('M6c_combiner', p6v, p6t, p6ti),
    ]:
        rows.append(dict(seed=seed, model=name, region='val_north', **metrics(p_v, y_val)))
        rows.append(dict(seed=seed, model=name, region='test_south', **metrics(p_t, y_test)))
        rows.append(dict(seed=seed, model=name, region='test_north_id', **metrics(p_ti, y_test_id)))

df = pd.DataFrame(rows)
df.to_csv(EX / 'coldstart_results.csv', index=False)

# Summary
agg = df.groupby(['model', 'region']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
).reset_index()
agg.to_csv(EX / 'coldstart_summary.csv', index=False)

print("\n=== Cold-start South test 2023 mean ± std (n=5 seeds) ===")
south = agg[agg.region == 'test_south'].sort_values('AUROC_mean', ascending=False)
print(south.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

print("\n=== Cold-start North ID test 2023 (reference) ===")
north_id = agg[agg.region == 'test_north_id'].sort_values('AUROC_mean', ascending=False)
print(north_id.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# Paired Δ vs M1c
per_seed = {}
for r in rows:
    if r['region'] != 'test_south': continue
    per_seed.setdefault(r['seed'], {})[r['model']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}

print("\n=== Paired Δ (cold-start) vs M1c on South test, n=5 seeds ===")
for vs in ['M2c_cold_iso', 'M4c_cold_fwi', 'M5c_cold_fwi_iso', 'M6c_combiner']:
    print(f"\n  {vs} - M1c:")
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([per_seed[s][vs][metric] - per_seed[s]['M1c_cold'][metric] for s in SEEDS])
        mean, std, med = float(diff.mean()), float(diff.std()), float(np.median(diff))
        n = len(SEEDS)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""  # t critical for df=4, two-tailed 0.05
        print(f"    Δ{metric:7s} = {mean:+.5f} ± {std:.5f}  (median {med:+.5f}, t={tstat:.2f}{sig})")

# Save
with open(EX / 'coldstart_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'removed_features': FIRE_HIST_FEATS,
        'cold_features_count': len(COLD_FEATS),
        'south_summary': south.to_dict('records'),
        'north_id_summary': north_id.to_dict('records'),
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/coldstart_summary.csv")
