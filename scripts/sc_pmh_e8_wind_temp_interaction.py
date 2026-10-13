#!/usr/bin/env python3
"""
v8.7.3 — Add wind × temperature interaction features to address Phase 3 finding.

Phase 3 found: FN occurs when t2m=23.3°C (LOW) + wind_anom=+0.54 (HIGHER than TN's -0.03).
Model misses wind-driven fires in cool conditions because it over-weights temperature.

New features (added to 42-feature v8.7.2 set):
  - wind_t2m_anom_interaction = wind_anom_doy * max(0, -t2m_anom_doy)
    → captures "wind high when temp below normal" (wind-driven regime)
  - wind_per_t2m = wind / max(t2m, 5)
    → captures "wind strong relative to low temperature" (non-linear wind effect when cool)

Tests:
  - Re-train on cold spatial OOD (train North 2018-21, val North 2022, test South 2023)
  - Compare to v8.7.2 baseline (42 features, no interaction)
  - Multi-seed (5 seeds) for stability
  - Check if FN rate (fires missed at p≥0.20) decreases
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, confusion_matrix
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
PANEL_P = SC / 'e2_weather_anomalies' / 'panel_with_anomalies.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'
EX = SC / 'e8_wind_temp_interaction'
EX.mkdir(parents=True, exist_ok=True)

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
FEATS_OLD = COLD_FEATS + ANOM_FEATS  # 42 features (v8.7.2 production)
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')

# === Compute new interaction features ===
print("[1/4] computing wind × temperature interaction features...")
# wind_t2m_anom_interaction: high when wind is unusually high AND temp is unusually low
panel['wind_t2m_anom_interaction'] = panel['wind_anom_doy'] * np.maximum(0, -panel['t2m_anom_doy'])
# wind_per_t2m: wind / (max(t2m, 5)) — high wind relative to cool temperatures
panel['wind_per_t2m'] = panel['wind'] / np.maximum(panel['t2m'], 5.0)

NEW_FEATS = ['wind_t2m_anom_interaction', 'wind_per_t2m']
FEATS_NEW = FEATS_OLD + NEW_FEATS  # 44 features (v8.7.3)
print(f"  old: {len(FEATS_OLD)} feats, new: {len(FEATS_NEW)} feats (added 2 interaction)")
print(f"  wind_t2m_anom_interaction: mean={panel['wind_t2m_anom_interaction'].mean():.4f}, "
      f"std={panel['wind_t2m_anom_interaction'].std():.4f}, "
      f"nonzero={100*(panel['wind_t2m_anom_interaction'] != 0).mean():.1f}%")
print(f"  wind_per_t2m:              mean={panel['wind_per_t2m'].mean():.4f}, "
      f"std={panel['wind_per_t2m'].std():.4f}")

# Save updated panel
panel.to_parquet(SC / 'panel_with_interactions.parquet', index=False)
print(f"  saved panel_with_interactions.parquet ({panel.shape})")

# === Train + eval both versions on cold spatial OOD ===
print("\n[2/4] training v8.7.2 (42 feats) vs v8.7.3 (44 feats) on cold spatial OOD, 5 seeds...")

train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL])
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL])
test  = panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL])
y_tr = train[LABEL_COL].astype(int).values
y_va = val[LABEL_COL].astype(int).values
y_te = test[LABEL_COL].astype(int).values
print(f"  train={len(train)} val={len(val)} test={len(test)} pos_test={y_te.mean():.4f}")

def _ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)

def metrics_full(p, y, thr=0.20):
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
    m = dict(AUROC=float(roc_auc_score(y, p)),
             PR_AUC=float(average_precision_score(y, p)),
             Brier=float(brier_score_loss(y, p)),
             ECE=_ece(p, y),
             p_mean=float(p.mean()),
             p_q90=float(np.quantile(p, 0.9)),
             p_q95=float(np.quantile(p, 0.95)),
             p_q99=float(np.quantile(p, 0.99)),
             warn_p10=float((p > 0.10).mean()),
             warn_p20=float((p > 0.20).mean()),
             warn_p30=float((p > 0.30).mean()))
    # Classification at threshold
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m['TP'] = int(tp); m['FP'] = int(fp); m['FN'] = int(fn); m['TN'] = int(tn)
    m['precision'] = float(tp / max(tp + fp, 1))
    m['recall'] = float(tp / max(tp + fn, 1))
    # FN weather pattern
    fn_mask = (pred == 0) & (y == 1)
    if fn_mask.sum() > 0:
        m['fn_t2m_mean'] = float(test.iloc[np.where(fn_mask)[0]]['t2m'].mean())
        m['fn_wind_mean'] = float(test.iloc[np.where(fn_mask)[0]]['wind'].mean())
        m['fn_wind_anom_mean'] = float(test.iloc[np.where(fn_mask)[0]]['wind_anom_doy'].mean())
    return m

rows = []
for variant_name, feats in [('v872_old_42feats', FEATS_OLD), ('v873_new_44feats', FEATS_NEW)]:
    print(f"\n  --- {variant_name} ({len(feats)} feats) ---")
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
        m_test = metrics_full(p_te_iso, y_te, thr=0.20)
        m_test['variant'] = variant_name
        m_test['seed'] = seed
        rows.append(m_test)
        # Print FN/FP for inspection
        print(f"    seed={seed}: AUROC={m_test['AUROC']:.4f} FN={m_test['FN']:>3d} FP={m_test['FP']:>4d} "
              f"recall={m_test['recall']:.3f} prec={m_test['precision']:.3f} "
              f"fn_t2m={m_test.get('fn_t2m_mean', 0):.1f} fn_wind_anom={m_test.get('fn_wind_anom_mean', 0):.2f}")

df = pd.DataFrame(rows)
df.to_csv(EX / 'results.csv', index=False)

# Aggregate
agg = df.groupby('variant').agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
    FN_mean=('FN', 'mean'), FP_mean=('FP', 'mean'),
    TP_mean=('TP', 'mean'), TN_mean=('TN', 'mean'),
    recall_mean=('recall', 'mean'), precision_mean=('precision', 'mean'),
    fn_t2m_mean=('fn_t2m_mean', 'mean'), fn_wind_anom_mean=('fn_wind_anom_mean', 'mean'),
    p_mean=('p_mean', 'mean'), p_q95=('p_q95', 'mean'), p_q99=('p_q99', 'mean'),
    warn_p20=('warn_p20', 'mean'),
).reset_index()
agg.to_csv(EX / 'summary.csv', index=False)

print("\n=== Summary (mean over 5 seeds) ===")
print(agg.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# Paired delta
print("\n=== Paired Δ (v873 − v872) over 5 seeds ===")
per_seed = {}
for r in rows:
    per_seed.setdefault(r['seed'], {})[r['variant']] = r

print(f"{'metric':<15} {'delta':>22} {'t-stat':>10} {'sig?':>6}")
print("-" * 60)
for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE', 'FN', 'FP', 'TP', 'recall', 'precision', 'fn_t2m_mean', 'fn_wind_anom_mean', 'p_mean', 'p_q95', 'warn_p20']:
    diff = np.array([per_seed[s]['v873_new_44feats'][metric] - per_seed[s]['v872_old_42feats'][metric] for s in SEEDS])
    mean, std = float(diff.mean()), float(diff.std())
    n = len(diff)
    tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
    sig = "*" if abs(tstat) > 2.776 else ""
    print(f"{metric:<15} {mean:>+10.5f}±{std:.5f} {tstat:>10.2f} {sig:>6}")

# Save panel with interactions for production use
print(f"\n[3/4] panel saved to {SC}/panel_with_interactions.parquet — ready for production")

# Save summary JSON
with open(EX / 'summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'old_features_count': len(FEATS_OLD),
        'new_features_count': len(FEATS_NEW),
        'new_features_added': NEW_FEATS,
        'summary': agg.to_dict('records'),
    }, f, indent=2, default=str)
print(f"[4/4] saved → {EX}/summary.json")
