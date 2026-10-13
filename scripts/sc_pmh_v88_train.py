#!/usr/bin/env python3
"""
v8.8 — Train with 58 features (35 cold + 7 anomaly + 7 expanded LC + 9 geo)

Replaces v8.7.3-LC's 4 LC features with 7 expanded LC + 9 static geo features.
Compare to v8.7.3-LC baseline.

Output: /home/z/my-project/download/sc_pmh/v88_train/
"""
from __future__ import annotations
import json, os, sys, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, confusion_matrix
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
PANEL_P = SC / 'panel_v873_landcover.parquet'
LC_EXP_P = SC / 'landcover_expanded.parquet'
GEO_P = SC / 'geo_features_static.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'
EX = SC / 'v88_train'
EX.mkdir(parents=True, exist_ok=True)

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
LC_EXP_FEATS = ['forest_prop', 'shrubland_prop', 'grassland_prop', 'agriculture_prop',
                'urban_prop', 'bare_prop', 'water_prop']
GEO_FEATS = ['elevation_mean', 'elevation_std', 'slope_mean',
             'koppen_Cwa', 'koppen_Am', 'koppen_Aw',
             'latitude_band', 'coastal_distance', 'population_density_proxy']

# v8.7.3-LC baseline = 35 + 7 + 4 = 46 features (uses 4 LC: forest, agriculture, urban, water)
LC_OLD = ['forest_prop', 'agriculture_prop', 'urban_prop', 'water_prop']
FEATS_OLD = COLD_FEATS + ANOM_FEATS + LC_OLD  # 46

# v8.8 NEW = 35 + 7 + 7 + 9 = 58 features
FEATS_NEW = COLD_FEATS + ANOM_FEATS + LC_EXP_FEATS + GEO_FEATS  # 58

LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

print(f"[info] v873-LC baseline: {len(FEATS_OLD)} feats")
print(f"[info] v8.8 NEW:         {len(FEATS_NEW)} feats")

# Load panel + LC expanded + geo
print("\n[1/4] loading panel + features...")
panel = pd.read_parquet(PANEL_P)
# Drop the OLD 4 LC features (forest/agriculture/urban/water) — we'll add 7 expanded instead
panel = panel.drop(columns=[c for c in LC_OLD if c in panel.columns])
lc_exp = pd.read_parquet(LC_EXP_P)
geo = pd.read_parquet(GEO_P)
# Merge
panel = panel.merge(lc_exp, on=['lat', 'lon'], how='left')
panel = panel.merge(geo, on=['lat', 'lon'], how='left')
# Fill NaN
for col in LC_EXP_FEATS + GEO_FEATS:
    if col in panel.columns:
        panel[col] = panel[col].fillna(panel[col].median() if panel[col].notna().any() else 0)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f"  panel: {panel.shape}")

# Save v8.8 panel
panel.to_parquet(SC / 'panel_v88.parquet', index=False)
print(f"  saved panel_v88.parquet")

# Train + eval
train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL])
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL])
test  = panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL])
y_tr = train[LABEL_COL].astype(int).values
y_va = val[LABEL_COL].astype(int).values
y_te = test[LABEL_COL].astype(int).values
print(f"\n[2/4] train={len(train)} val={len(val)} test={len(test)}")

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
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m['TP'] = int(tp); m['FP'] = int(fp); m['FN'] = int(fn); m['TN'] = int(tn)
    m['precision'] = float(tp / max(tp + fp, 1))
    m['recall'] = float(tp / max(tp + fn, 1))
    return m

rows = []
feature_importances_v88 = {}
for variant_name, feats in [('v873_LC_baseline_46', FEATS_OLD), ('v88_new_58', FEATS_NEW)]:
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
        print(f"    seed={seed}: AUROC={m_test['AUROC']:.4f} PR-AUC={m_test['PR_AUC']:.4f} "
              f"Brier={m_test['Brier']:.4f} ECE={m_test['ECE']:.4f} "
              f"FP={m_test['FP']:>4d} p_q95={m_test['p_q95']:.3f} warn_p20={m_test['warn_p20']:.3f}")
        if seed == 42 and variant_name == 'v88_new_58':
            feature_importances_v88 = m.get_score(importance_type='gain')

df = pd.DataFrame(rows)
df.to_csv(EX / 'results.csv', index=False)

# Aggregate
agg = df.groupby('variant').agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
    FN_mean=('FN', 'mean'), FP_mean=('FP', 'mean'),
    recall_mean=('recall', 'mean'), precision_mean=('precision', 'mean'),
    p_mean=('p_mean', 'mean'), p_q95=('p_q95', 'mean'),
    warn_p10=('warn_p10', 'mean'), warn_p20=('warn_p20', 'mean'), warn_p30=('warn_p30', 'mean'),
).reset_index()
agg.to_csv(EX / 'summary.csv', index=False)

print("\n=== Summary (mean over 5 seeds) ===")
print(agg.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# Paired delta
print("\n=== Paired Δ (v88 − v873_LC) over 5 seeds ===")
per_seed = {}
for r in rows:
    per_seed.setdefault(r['seed'], {})[r['variant']] = r

print(f"{'metric':<15} {'delta':>22} {'t-stat':>10} {'sig?':>6}")
print("-" * 60)
for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE', 'FN', 'FP', 'TP', 'TN', 'recall', 'precision', 'p_mean', 'p_q95', 'warn_p10', 'warn_p20', 'warn_p30']:
    diff = np.array([per_seed[s]['v88_new_58'][metric] - per_seed[s]['v873_LC_baseline_46'][metric] for s in SEEDS])
    mean, std = float(diff.mean()), float(diff.std())
    n = len(diff)
    tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
    sig = "*" if abs(tstat) > 2.776 else ""
    print(f"{metric:<15} {mean:>+10.5f}±{std:.5f} {tstat:>10.2f} {sig:>6}")

# Top-25 feature importance
print("\n=== Top-25 features by gain (v8.8, seed=42) ===")
sorted_imps = sorted(feature_importances_v88.items(), key=lambda x: -x[1])
for fname, score in sorted_imps[:25]:
    tag = " <== LC" if fname in LC_EXP_FEATS else (" <== GEO" if fname in GEO_FEATS else (" <== ANOM" if fname in ANOM_FEATS else ""))
    print(f"  {fname:<35s} {score:.2f}{tag}")

with open(EX / 'summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'baseline_features_count': len(FEATS_OLD),
        'new_features_count': len(FEATS_NEW),
        'new_features_added': LC_EXP_FEATS + GEO_FEATS,
        'summary': agg.to_dict('records'),
        'top_10_features_v88': sorted_imps[:10],
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/summary.json")
