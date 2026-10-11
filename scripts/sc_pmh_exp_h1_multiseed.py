#!/usr/bin/env python3
"""
SC-PMH Experiment H1 — Multi-seed stability check (5 seeds × 4 key models)
=========================================================================
Goal: determine if M4 (XGB+FWI) and M6 (logistic combiner) really improve over
M1 (XGB baseline) on South OOD test, or if observed differences are noise.

Models tested:
  M1 XGB (38 features)
  M4 XGB+FWI (44 features)
  M5 XGB+FWI+iso (M4 calibrated)
  M6 Logistic combiner σ(β0 + β1·logit(p_M1) + β2·z_FWI)

Outputs: /home/z/my-project/download/sc_pmh/exp_h1_multiseed/
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
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'exp_h1_multiseed'
EX.mkdir(parents=True, exist_ok=True)
PANEL_P = SC / 'panel_with_fwi.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FWI_FEATS = ['fwi', 'fwi_ffmc', 'fwi_dmc', 'fwi_dc', 'fwi_isi', 'fwi_bui']
FEATS_M4 = BASE_FEATS + FWI_FEATS
LABEL_COL = 'label_modis_h1'

print("[1/4] loading panel...")
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')

train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL]).copy()
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL]).copy()
test  = panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL]).copy()
test_id = panel[(panel.region == 'North') & (panel.year == 2023)].dropna(subset=[LABEL_COL]).copy()
y_train = train[LABEL_COL].astype(int).values
y_val = val[LABEL_COL].astype(int).values
y_test = test[LABEL_COL].astype(int).values
y_test_id = test_id[LABEL_COL].astype(int).values
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

SEEDS = [42, 123, 777, 2024, 31415]
rows = []

for seed in SEEDS:
    print(f"\n[2/4] seed={seed}")
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist',
        'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 5, 'reg_lambda': 1.0, 'seed': seed, 'verbosity': 0,
    }
    # --- M1 ---
    dtr = gbm.DMatrix(train[BASE_FEATS], label=y_train)
    dv  = gbm.DMatrix(val[BASE_FEATS], label=y_val)
    dte = gbm.DMatrix(test[BASE_FEATS], label=y_test)
    dtei = gbm.DMatrix(test_id[BASE_FEATS], label=y_test_id)
    m1 = gbm.train(params, dtr, num_boost_round=500, evals=[(dtr, 'tr'), (dv, 'val')],
                  early_stopping_rounds=30, verbose_eval=False)
    p1v, p1t, p1ti = m1.predict(dv), m1.predict(dte), m1.predict(dtei)

    # --- M2 (M1 + iso) ---
    iso1 = IsotonicRegression(out_of_bounds='clip').fit(p1v, y_val)
    p2v, p2t, p2ti = iso1.predict(p1v), iso1.predict(p1t), iso1.predict(p1ti)

    # --- M4 (XGB+FWI) ---
    dtr4 = gbm.DMatrix(train[FEATS_M4], label=y_train)
    dv4  = gbm.DMatrix(val[FEATS_M4], label=y_val)
    dte4 = gbm.DMatrix(test[FEATS_M4], label=y_test)
    dtei4 = gbm.DMatrix(test_id[FEATS_M4], label=y_test_id)
    m4 = gbm.train(params, dtr4, num_boost_round=500, evals=[(dtr4, 'tr'), (dv4, 'val')],
                  early_stopping_rounds=30, verbose_eval=False)
    p4v, p4t, p4ti = m4.predict(dv4), m4.predict(dte4), m4.predict(dtei4)

    # --- M5 (M4 + iso) ---
    iso4 = IsotonicRegression(out_of_bounds='clip').fit(p4v, y_val)
    p5v, p5t, p5ti = iso4.predict(p4v), iso4.predict(p4t), iso4.predict(p4ti)

    # --- M6 (logistic combiner) ---
    eps = 1e-6
    z_ml_v = np.log(np.clip(p1v, eps, 1-eps) / np.clip(1-p1v, eps, 1))
    z_ml_t = np.log(np.clip(p1t, eps, 1-eps) / np.clip(1-p1t, eps, 1))
    z_ml_ti = np.log(np.clip(p1ti, eps, 1-eps) / np.clip(1-p1ti, eps, 1))
    fwi_sc = StandardScaler().fit(train[['fwi']])
    z_fwi_v = fwi_sc.transform(val[['fwi']]).ravel()
    z_fwi_t = fwi_sc.transform(test[['fwi']]).ravel()
    z_fwi_ti = fwi_sc.transform(test_id[['fwi']]).ravel()
    m6 = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
        np.column_stack([z_ml_v, z_fwi_v]), y_val)
    p6v = m6.predict_proba(np.column_stack([z_ml_v, z_fwi_v]))[:, 1]
    p6t = m6.predict_proba(np.column_stack([z_ml_t, z_fwi_t]))[:, 1]
    p6ti = m6.predict_proba(np.column_stack([z_ml_ti, z_fwi_ti]))[:, 1]

    for name, p_v, p_t, p_ti in [
        ('M1', p1v, p1t, p1ti), ('M2', p2v, p2t, p2ti),
        ('M4', p4v, p4t, p4ti), ('M5', p5v, p5t, p5ti),
        ('M6', p6v, p6t, p6ti),
    ]:
        mv = metrics(p_v, y_val); mt = metrics(p_t, y_test); mti = metrics(p_ti, y_test_id)
        rows.append(dict(seed=seed, model=name, region='val_north', **mv))
        rows.append(dict(seed=seed, model=name, region='test_south', **mt))
        rows.append(dict(seed=seed, model=name, region='test_north_id', **mti))

df = pd.DataFrame(rows)
df.to_csv(EX / 'multiseed_results.csv', index=False)

# Summary table — mean ± std per (model, region)
agg = df.groupby(['model', 'region']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
).reset_index()
agg.to_csv(EX / 'multiseed_summary.csv', index=False)

print("\n[3/4] === South test 2023 mean ± std (n=5 seeds) ===")
south = agg[agg.region == 'test_south'].sort_values('AUROC_mean', ascending=False)
print(south.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

print("\n=== North ID test 2023 (reference) ===")
north_id = agg[agg.region == 'test_north_id'].sort_values('AUROC_mean', ascending=False)
print(north_id.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# --- Paired bootstrap test: M1 vs M4, M1 vs M6 on South test ---
# For each seed, compute (M_i - M1) per metric, then mean ± std
print("\n[4/4] paired Δ (model - M1) on South test, across 5 seeds:")
def deltas(m1_vals, m_vals):
    """m1_vals, m_vals: dict per seed -> metrics dict"""
    out = {}
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([m_vals[s][metric] - m1_vals[s][metric] for s in SEEDS])
        out[metric] = (float(diff.mean()), float(diff.std()), float(np.median(diff)))
    return out

# Build dicts: per-seed per-model metrics for south
per_seed = {}
for r in rows:
    if r['region'] != 'test_south': continue
    s = r['seed']; m = r['model']
    per_seed.setdefault(s, {})[m] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}

for vs in ['M4', 'M5', 'M6']:
    d = deltas({s: per_seed[s]['M1'] for s in SEEDS},
               {s: per_seed[s][vs] for s in SEEDS})
    print(f"\n  {vs} - M1:")
    for metric, (mean, std, med) in d.items():
        # paired t-stat: t = mean / (std/sqrt(n))
        n = len(SEEDS)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        print(f"    Δ{metric:7s} = {mean:+.5f} ± {std:.5f}  (median {med:+.5f}, t={tstat:.2f}, n={n})")

# Save summary
with open(EX / 'multiseed_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'south_summary': south.to_dict('records'),
        'north_id_summary': north_id.to_dict('records'),
        'paired_delta_vs_M1_on_south': {
            vs: {metric: {'mean': d[metric][0], 'std': d[metric][1], 'median': d[metric][2]}
                 for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
            for vs, d in [('M4', deltas({s: per_seed[s]['M1'] for s in SEEDS},
                                        {s: per_seed[s]['M4'] for s in SEEDS})),
                          ('M5', deltas({s: per_seed[s]['M1'] for s in SEEDS},
                                        {s: per_seed[s]['M5'] for s in SEEDS})),
                          ('M6', deltas({s: per_seed[s]['M1'] for s in SEEDS},
                                        {s: per_seed[s]['M6'] for s in SEEDS}))]
        },
    }, f, indent=2, default=str)

print(f"\n[done] saved → {EX}/multiseed_summary.csv, multiseed_summary.json")
