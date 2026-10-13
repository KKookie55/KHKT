#!/usr/bin/env python3
"""
v8.7.3-LC — Add 4 land-cover features (forest, agriculture, urban, water proportions)
from ESA WorldCover 2021 to v8.7.2 production model.

Test if land cover features help model:
1. Reduce over-prediction in South (agricultural burning cells should be down-weighted)
2. Improve AUROC, PR-AUC, Brier
3. Improve calibration
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
PANEL_P = SC / 'panel_with_interactions.parquet'
LC_P = SC / 'landcover_features.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'
EX = SC / 'e9_landcover'
EX.mkdir(parents=True, exist_ok=True)

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
FEATS_OLD = COLD_FEATS + ANOM_FEATS  # 42 features (v8.7.2)
LC_FEATS = ['forest_prop', 'agriculture_prop', 'urban_prop', 'water_prop']
FEATS_NEW = FEATS_OLD + LC_FEATS  # 46 features (v8.7.3 with land cover)
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]

# Load panel + land cover
panel = pd.read_parquet(PANEL_P)
lc = pd.read_parquet(LC_P)
panel = panel.merge(lc[['lat', 'lon', 'forest_prop', 'agriculture_prop', 'urban_prop', 'water_prop']],
                    on=['lat', 'lon'], how='left')
# Fill missing land cover with median (only 2 cells missing)
for col in LC_FEATS:
    panel[col] = panel[col].fillna(panel[col].median())
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f"[info] panel after LC merge: {panel.shape}, LC features non-null: {panel['forest_prop'].notna().sum()}")
print(f"[info] LC stats: forest_mean={panel['forest_prop'].mean():.3f}  agriculture_mean={panel['agriculture_prop'].mean():.3f}")

# Save updated panel
panel.to_parquet(SC / 'panel_v873_landcover.parquet', index=False)
print(f"[info] saved panel_v873_landcover.parquet")

# Train + eval
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
             warn_p05=float((p > 0.05).mean()),
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
feature_importances = {}
for variant_name, feats in [('v872_baseline_42', FEATS_OLD), ('v873_lc_46', FEATS_NEW)]:
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
              f"FN={m_test['FN']:>3d} FP={m_test['FP']:>4d} "
              f"p_q95={m_test['p_q95']:.3f} warn_p20={m_test['warn_p20']:.3f}")
        if seed == 42 and variant_name == 'v873_lc_46':
            imps = m.get_score(importance_type='gain')
            feature_importances = imps

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
    p_mean=('p_mean', 'mean'), p_q95=('p_q95', 'mean'), p_q99=('p_q99', 'mean'),
    warn_p10=('warn_p10', 'mean'), warn_p20=('warn_p20', 'mean'), warn_p30=('warn_p30', 'mean'),
).reset_index()
agg.to_csv(EX / 'summary.csv', index=False)

print("\n=== Summary (mean over 5 seeds) ===")
print(agg.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# Paired delta
print("\n=== Paired Δ (v873_lc − v872_baseline) over 5 seeds ===")
per_seed = {}
for r in rows:
    per_seed.setdefault(r['seed'], {})[r['variant']] = r

print(f"{'metric':<15} {'delta':>22} {'t-stat':>10} {'sig?':>6}")
print("-" * 60)
for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE', 'FN', 'FP', 'TP', 'TN', 'recall', 'precision', 'p_mean', 'p_q95', 'p_q99', 'warn_p10', 'warn_p20', 'warn_p30']:
    diff = np.array([per_seed[s]['v873_lc_46'][metric] - per_seed[s]['v872_baseline_42'][metric] for s in SEEDS])
    mean, std = float(diff.mean()), float(diff.std())
    n = len(diff)
    tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
    sig = "*" if abs(tstat) > 2.776 else ""
    print(f"{metric:<15} {mean:>+10.5f}±{std:.5f} {tstat:>10.2f} {sig:>6}")

# Feature importance top-20
print("\n=== Top-20 features by gain (v873_lc, seed=42) ===")
sorted_imps = sorted(feature_importances.items(), key=lambda x: -x[1])
for fname, score in sorted_imps[:20]:
    tag = " <== LC" if fname in LC_FEATS else ""
    print(f"  {fname:<35s} {score:.2f}{tag}")

# Save summary
with open(EX / 'summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'baseline_features': len(FEATS_OLD),
        'new_features_count': len(FEATS_NEW),
        'land_cover_features': LC_FEATS,
        'summary': agg.to_dict('records'),
        'top_5_features_by_gain': sorted_imps[:5],
    }, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/summary.json")
