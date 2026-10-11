#!/usr/bin/env python3
"""
SC-PMH E7: Recall at Same Warning Budget (Operational Value)
=============================================================
Goal: shift from "higher AUROC" to "at same warning budget, which model
finds more fires?" This is the operationally meaningful question per the
user's pillar B.

Setup: cold spatial OOD (train North 2018-21, val North 2022, test South 2023).
Compare:
  - cold_baseline (35 feats)
  - cold_anom (42 feats — drop composite per E6 finding)

For each model × seed:
  - For each day in test, sort cells by predicted probability (descending)
  - Flag top-K% of cells as "warning"
  - Compute recall = (# fires captured in flagged cells) / (# total fires that day)
  - Repeat for K ∈ {1%, 5%, 10%, 20%, 30%}

Then aggregate over days: mean recall ± std per (model, K, seed).
Also compute precision and "fires captured per cell flagged" (efficiency).
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
EX = SC / 'e7_warning_budget'
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
# Skip composite per E6 finding
LABEL_COL = 'label_modis_h1'
SEEDS = [42, 123, 777, 2024, 31415]
BUDGETS = [0.01, 0.05, 0.10, 0.20, 0.30]  # top-K% cells to flag

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

VARIANTS = {
    'cold_baseline': COLD_FEATS,
    'cold_anom_7feats': COLD_FEATS + ANOM_FEATS,
}

# Storage: per (variant, seed, day) — recall/precision at each K
all_per_day = []

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
        p_te = m.predict(dte)
        
        # Build test dataframe with date, label, prediction
        test_pred = test[['date']].copy()
        test_pred['y'] = y_te
        test_pred['p'] = p_te
        test_pred['variant'] = variant_name
        test_pred['seed'] = seed
        
        # For each day, compute recall at top-K
        for day, sub in test_pred.groupby('date'):
            n = len(sub)
            n_pos = int(sub['y'].sum())
            if n_pos == 0:
                continue  # skip days with no fires
            for K in BUDGETS:
                k_n = max(int(n * K), 1)
                sub_sorted = sub.sort_values('p', ascending=False)
                topk = sub_sorted.head(k_n)
                n_captured = int(topk['y'].sum())
                recall = n_captured / n_pos
                precision = n_captured / k_n if k_n > 0 else 0
                all_per_day.append({
                    'variant': variant_name, 'seed': seed, 'date': day, 'budget': K,
                    'n_cells': n, 'n_pos': n_pos, 'n_flagged': k_n,
                    'n_captured': n_captured, 'recall': recall, 'precision': precision,
                })

per_day_df = pd.DataFrame(all_per_day)
per_day_df.to_csv(EX / 'per_day_recall.csv', index=False)

# Aggregate per (variant, budget, seed) — mean over days
per_seed_df = per_day_df.groupby(['variant', 'budget', 'seed']).agg(
    recall_mean=('recall', 'mean'),
    recall_std=('recall', 'std'),
    precision_mean=('precision', 'mean'),
    n_days=('date', 'nunique'),
    n_fires_total=('n_pos', 'sum'),
    n_captured_total=('n_captured', 'sum'),
).reset_index()
per_seed_df.to_csv(EX / 'per_seed_summary.csv', index=False)

# Aggregate over seeds (mean ± std)
final = per_seed_df.groupby(['variant', 'budget']).agg(
    recall_mean=('recall_mean', 'mean'),
    recall_std=('recall_mean', 'std'),
    precision_mean=('precision_mean', 'mean'),
    precision_std=('precision_mean', 'std'),
).reset_index()
final.to_csv(EX / 'final_summary.csv', index=False)

print("\n=== Recall at same warning budget, South test 2023 ===")
print(f"{'budget':>10} {'variant':>20} {'recall':>15} {'precision':>15}")
print("-" * 65)
for K in BUDGETS:
    for v in ['cold_baseline', 'cold_anom_7feats']:
        sub = final[(final.budget == K) & (final.variant == v)]
        if len(sub) == 0: continue
        r = sub.iloc[0]
        print(f"{K:>10.2f} {v:>20} {r.recall_mean:>10.4f}±{r.recall_std:.4f} {r.precision_mean:>10.4f}±{r.precision_std:.4f}")
    print()

# Paired delta: anom − baseline per budget per seed
print("\n=== Paired Δ (anom − baseline) per budget, n=5 seeds ===")
print(f"{'budget':>10} {'Δrecall':>22} {'Δprecision':>22} {'# extra fires captured':>30}")
print("-" * 90)
for K in BUDGETS:
    per_seed_pivot = per_seed_df[per_seed_df.budget == K].pivot(index='seed', columns='variant', values='recall_mean')
    if 'cold_baseline' not in per_seed_pivot.columns or 'cold_anom_7feats' not in per_seed_pivot.columns:
        continue
    diff_recall = (per_seed_pivot['cold_anom_7feats'] - per_seed_pivot['cold_baseline']).values
    per_seed_pivot_prec = per_seed_df[per_seed_df.budget == K].pivot(index='seed', columns='variant', values='precision_mean')
    diff_prec = (per_seed_pivot_prec['cold_anom_7feats'] - per_seed_pivot_prec['cold_baseline']).values
    
    mean_r, std_r = float(np.mean(diff_recall)), float(np.std(diff_recall))
    mean_p, std_p = float(np.mean(diff_prec)), float(np.std(diff_prec))
    n = len(diff_recall)
    tstat_r = mean_r / (std_r / np.sqrt(n)) if std_r > 0 else float('inf')
    tstat_p = mean_p / (std_p / np.sqrt(n)) if std_p > 0 else float('inf')
    sig_r = " *" if abs(tstat_r) > 2.776 else ""
    sig_p = " *" if abs(tstat_p) > 2.776 else ""
    
    # Total fires captured difference (mean over seeds)
    fires_baseline = per_seed_df[(per_seed_df.budget == K) & (per_seed_df.variant == 'cold_baseline')]['n_captured_total'].mean()
    fires_anom = per_seed_df[(per_seed_df.budget == K) & (per_seed_df.variant == 'cold_anom_7feats')]['n_captured_total'].mean()
    extra = fires_anom - fires_baseline
    
    print(f"{K:>10.2f} {mean_r:>+10.5f}±{std_r:.5f}(t={tstat_r:.2f}{sig_r}) {mean_p:>+10.5f}±{std_p:.5f}(t={tstat_p:.2f}{sig_p}) {extra:>+15.1f} fires ({fires_baseline:.0f}→{fires_anom:.0f})")

# Total fires in South test 2023
total_fires = int(y_te.sum())
print(f"\n[context] total fires in South test 2023: {total_fires}")

with open(EX / 'e7_summary.json', 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'budgets_K': BUDGETS,
        'total_fires_south_test': total_fires,
        'n_test_days': int(test['date'].nunique()),
        'final_summary': final.to_dict('records'),
    }, f, indent=2, default=str)

# Plot: recall vs budget
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
colors = {'cold_baseline': '#1f77b4', 'cold_anom_7feats': '#d62728'}
labels = {'cold_baseline': 'Baseline (35 feats)', 'cold_anom_7feats': 'Baseline + 7 anomalies (42 feats)'}
for v in ['cold_baseline', 'cold_anom_7feats']:
    sub = final[final.variant == v].sort_values('budget')
    ax.errorbar(sub.budget * 100, sub.recall_mean, yerr=sub.recall_std,
                marker='o', label=labels[v], color=colors[v], capsize=4, linewidth=2, markersize=8)
ax.set_xlabel('Warning budget (% cells flagged per day)')
ax.set_ylabel('Recall (fraction of fires captured)')
ax.set_title(f'South test 2023 OOD — recall at same warning budget (n=5 seeds)\nTotal fires: {total_fires}, days: {test["date"].nunique()}')
ax.set_xscale('log')
ax.set_xticks([1, 5, 10, 20, 30])
ax.set_xticklabels(['1%', '5%', '10%', '20%', '30%'])
ax.grid(alpha=0.3, which='both')
ax.legend(loc='lower right')
fig.savefig(EX / 'recall_vs_budget.png', dpi=150)
plt.close()
print(f"\n[done] saved → {EX}/final_summary.csv, e7_summary.json, recall_vs_budget.png")
