#!/usr/bin/env python3
"""
SC-PMH Phase 3 — Case Analysis: TP/FP/FN/TN on specific days
================================================================
Goal: visualize where the model succeeds and fails. Find patterns in
false positives and false negatives.

Setup: 42-feature model (35 cold baseline + 7 anomalies, drop composite per E6)
on cold spatial OOD (train North 2018-21, val North 2022, test South 2023).

Outputs: /home/z/my-project/download/sc_pmh/phase3_case_analysis/
  - case_dates.json — list of selected dates with case type
  - per_day_maps/ — TP/FP/FN/TN visualization per day (PNG)
  - cell_failure_patterns.png — chronic FN/FP cells across South test
  - weather_at_fn.png — weather conditions when FNs happen
  - fn_fp_analysis.json — quantitative summary
"""
from __future__ import annotations
import json, os, sys, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'phase3_case_analysis'
EX.mkdir(parents=True, exist_ok=True)
(EX / 'per_day_maps').mkdir(exist_ok=True)
PANEL_P = SC / 'e2_weather_anomalies' / 'panel_with_anomalies.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
FEATS_FINAL = COLD_FEATS + ANOM_FEATS  # 42 features
LABEL_COL = 'label_modis_h1'
SEED = 42
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]

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
print(f"train={len(train)} val={len(val)} test={len(test)}")

# Train model
params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
              max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=1.0, seed=SEED, verbosity=0)
dtr = gbm.DMatrix(train[FEATS_FINAL], label=y_tr)
dv  = gbm.DMatrix(val[FEATS_FINAL], label=y_va)
dte = gbm.DMatrix(test[FEATS_FINAL], label=y_te)
m = gbm.train(params, dtr, num_boost_round=300, evals=[(dtr, 'tr'), (dv, 'val')],
              early_stopping_rounds=20, verbose_eval=False)
print(f"trained, best_iter={m.best_iteration}")
p_va = m.predict(dv); p_te = m.predict(dte)
iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
p_te_iso = iso.predict(p_te)

test = test.assign(p_fire=p_te_iso)
print(f"Test: AUROC={roc_auc_score(y_te, p_te_iso):.4f} PR-AUC={average_precision_score(y_te, p_te_iso):.4f} Brier={brier_score_loss(y_te, p_te_iso):.4f}")

# === Step 1: Pick representative dates ===
print("\n[1/5] selecting representative dates...")
test['date_str'] = test['date'].dt.strftime('%Y-%m-%d')
daily = test.groupby('date').agg(
    n_cells=('fire_modis', 'size'),
    n_fires=('fire_modis', 'sum'),
    p_mean=('p_fire', 'mean'),
    p_max=('p_fire', 'max'),
).reset_index()
daily['fire_rate'] = daily['n_fires'] / daily['n_cells']

# Categories:
# - Top fire days: most fires
# - High p, low fire (FP-prone day): high p_max but few fires
# - Low p, high fire (FN-prone day): low p_max but many fires
# - Zero fire day: no fires (for context)
# - Dry season day (Nov-Apr)
# - Wet season day (May-Oct)

top_fire_dates = daily.sort_values('n_fires', ascending=False).head(5)['date'].tolist()
fp_dates = daily[(daily.p_max > 0.5) & (daily.n_fires <= 2)].sort_values('p_max', ascending=False).head(3)['date'].tolist()
fn_dates = daily[(daily.p_max < 0.3) & (daily.n_fires >= 8)].sort_values('n_fires', ascending=False).head(3)['date'].tolist()
zero_fire_dates = daily[daily.n_fires == 0].head(2)['date'].tolist()
dry_season_dates = daily[daily.date.dt.month.isin([1, 2, 3, 12])].sort_values('n_fires', ascending=False).head(2)['date'].tolist()
wet_season_dates = daily[daily.date.dt.month.isin([6, 7, 8])].sort_values('n_fires', ascending=False).head(2)['date'].tolist()

cases = []
for d in top_fire_dates:
    cases.append(('top_fire', d, 'Ngày cháy nhiều'))
for d in fp_dates:
    cases.append(('fp_prone', d, 'Dự báo cao nhưng ít cháy thực (FP)'))
for d in fn_dates:
    cases.append(('fn_prone', d, 'Dự báo thấp nhưng nhiều cháy thực (FN)'))
for d in zero_fire_dates:
    cases.append(('zero_fire', d, 'Không có cháy'))
for d in dry_season_dates:
    cases.append(('dry_season', d, 'Mùa khô'))
for d in wet_season_dates:
    cases.append(('wet_season', d, 'Mùa mưa'))
# Deduplicate
seen = set()
cases_unique = []
for ct, d, label in cases:
    if d not in seen:
        cases_unique.append({'case_type': ct, 'date': d.strftime('%Y-%m-%d'), 'description': label,
                              'n_fires': int(daily[daily.date == d].iloc[0].n_fires),
                              'p_max': float(daily[daily.date == d].iloc[0].p_max)})
        seen.add(d)
print(f"  selected {len(cases_unique)} case dates:")
for c in cases_unique:
    print(f"    {c['case_type']:>12s} {c['date']}  fires={c['n_fires']:>3d}  p_max={c['p_max']:.3f}  {c['description']}")

with open(EX / 'case_dates.json', 'w') as f:
    json.dump(cases_unique, f, indent=2)

# === Step 2: Visualize TP/FP/FN/TN per case date ===
print("\n[2/5] visualizing TP/FP/FN/TN per case date...")
THR = 0.20  # use 20% threshold as default

def classify(p, y, thr):
    """Returns TP/FP/FN/TN codes."""
    if p >= thr and y == 1:
        return 'TP'
    if p >= thr and y == 0:
        return 'FP'
    if p < thr and y == 1:
        return 'FN'
    return 'TN'

for case in cases_unique:
    date_val = pd.to_datetime(case['date'])
    sub = test[test.date == date_val].copy()
    if len(sub) == 0:
        continue
    sub['class'] = sub.apply(lambda r: classify(r.p_fire, r.fire_modis, THR), axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Plot 1: predicted probability
    ax = axes[0]
    sc = ax.scatter(sub.lon, sub.lat, c=sub.p_fire, s=80, cmap='YlOrRd', vmin=0, vmax=1, edgecolor='k', linewidths=0.5)
    ax.set_title(f'Predicted probability — {case["date"]}\n(case: {case["case_type"]}, fires={case["n_fires"]})')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    plt.colorbar(sc, ax=ax, label='p_fire')
    ax.grid(alpha=0.3)
    
    # Plot 2: observed fires
    ax = axes[1]
    fires = sub[sub.fire_modis == 1]
    no_fires = sub[sub.fire_modis == 0]
    ax.scatter(no_fires.lon, no_fires.lat, c='lightgray', s=80, label=f'No fire ({len(no_fires)})', edgecolor='k', linewidths=0.5)
    ax.scatter(fires.lon, fires.lat, c='red', s=120, marker='X', label=f'Fire ({len(fires)})', edgecolor='k', linewidths=1)
    ax.set_title(f'Observed fires — {case["date"]}')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: classification
    ax = axes[2]
    colors_map = {'TP': '#2ca02c', 'FP': '#d62728', 'FN': '#ff7f0e', 'TN': '#cccccc'}
    for cls, color in colors_map.items():
        s = sub[sub['class'] == cls]
        if len(s) == 0: continue
        marker = 'o' if cls in ['TP', 'TN'] else 'X' if cls == 'FP' else '^'
        ax.scatter(s.lon, s.lat, c=color, s=120 if cls in ['TP', 'FP', 'FN'] else 60,
                   marker=marker, label=f'{cls} ({len(s)})', edgecolor='k', linewidths=0.5)
    ax.set_title(f'Classification @ p≥{THR:.2f}\nTP=captured, FP=false alarm, FN=missed, TN=correct null')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle(f'South Vietnam — {case["date"]} — case: {case["case_type"]}', fontsize=13, fontweight='bold')
    fname = f'{case["case_type"]}_{case["date"]}.png'
    fig.savefig(EX / 'per_day_maps' / fname, dpi=120, bbox_inches='tight')
    plt.close()

print(f"  saved {len(cases_unique)} case maps to {EX / 'per_day_maps'}")

# === Step 3: Cell-level failure patterns ===
print("\n[3/5] analyzing cell-level failure patterns across all South test days...")
test_all = test.copy()
test_all['class'] = test_all.apply(lambda r: classify(r.p_fire, r.fire_modis, THR), axis=1)
# Per cell, count occurrences of each class
cell_pattern = test_all.groupby(['lat', 'lon', 'class']).size().unstack(fill_value=0)
# Make sure all 4 classes are columns
for cls in ['TP', 'FP', 'FN', 'TN']:
    if cls not in cell_pattern.columns:
        cell_pattern[cls] = 0
cell_pattern = cell_pattern[['TP', 'FP', 'FN', 'TN']].reset_index()
n_days = test_all.date.nunique()
cell_pattern['fp_rate'] = cell_pattern['FP'] / n_days
cell_pattern['fn_rate'] = cell_pattern['FN'] / n_days
cell_pattern['tp_rate'] = cell_pattern['TP'] / n_days
cell_pattern['fire_rate'] = (cell_pattern['TP'] + cell_pattern['FN']) / n_days

# Find chronically FP cells (FP >= 30 days) and chronically FN cells (FN >= 5 days)
chronic_fp = cell_pattern[cell_pattern['FP'] >= 30].sort_values('FP', ascending=False)
chronic_fn = cell_pattern[cell_pattern['FN'] >= 5].sort_values('FN', ascending=False)
print(f"  chronic FP cells (>=30 days): {len(chronic_fp)} (top 5: {[tuple(r) for _, r in chronic_fp.head(5)[['lat','lon','FP']].iterrows()]} ...)")
print(f"  chronic FN cells (>=5 days):  {len(chronic_fn)}")

# Save to JSON for product integration
cell_pattern_json = []
for _, r in cell_pattern.iterrows():
    cell_pattern_json.append({
        'lat': float(r.lat), 'lon': float(r.lon),
        'tp': int(r.TP), 'fp': int(r.FP), 'fn': int(r.FN), 'tn': int(r.TN),
        'fp_rate': float(r.fp_rate), 'fn_rate': float(r.fn_rate), 'tp_rate': float(r.tp_rate),
        'fire_rate': float(r.fire_rate),
    })

# Plot cell failure pattern map
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
for ax, metric_col, metric_label, title, cmap in zip(axes,
                                       ['FP', 'FN', 'fire_rate'],
                                       ['fp', 'fn', 'fire_rate'],
                                       ['FP rate (days flagged but no fire)', 'FN rate (days missed fires)', 'Empirical fire rate (per day)'],
                                       ['Reds', 'Oranges', 'YlOrRd']):
    sc = ax.scatter(cell_pattern.lon, cell_pattern.lat, c=cell_pattern[metric_col], s=80, cmap=cmap, edgecolor='k', linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    plt.colorbar(sc, ax=ax)
    ax.grid(alpha=0.3)
fig.suptitle(f'Phase 3: Cell-level failure patterns across {n_days} South test days (2023)', fontsize=13, fontweight='bold')
fig.savefig(EX / 'cell_failure_patterns.png', dpi=150)
plt.close()
print(f"  saved cell_failure_patterns.png")

# === Step 4: Weather conditions when FN vs FP happen ===
print("\n[4/5] analyzing weather conditions for FN vs FP...")
# For each FN row (p < thr, y=1), record weather
# For each FP row (p >= thr, y=0), record weather
fn_weather = test_all[test_all['class'] == 'FN'][['t2m', 'rh', 'wind', 'precip', 'days_since_rain',
                                                    't2m_anom_doy', 'rh_anom_doy', 'wind_anom_doy',
                                                    'precip_anom_doy', 'fire_modis', 'p_fire']].copy()
fp_weather = test_all[test_all['class'] == 'FP'][['t2m', 'rh', 'wind', 'precip', 'days_since_rain',
                                                    't2m_anom_doy', 'rh_anom_doy', 'wind_anom_doy',
                                                    'precip_anom_doy', 'fire_modis', 'p_fire']].copy()
tn_weather = test_all[test_all['class'] == 'TN'][['t2m', 'rh', 'wind', 'precip', 'days_since_rain',
                                                    't2m_anom_doy', 'rh_anom_doy', 'wind_anom_doy',
                                                    'precip_anom_doy', 'fire_modis', 'p_fire']].copy()
tp_weather = test_all[test_all['class'] == 'TP'][['t2m', 'rh', 'wind', 'precip', 'days_since_rain',
                                                    't2m_anom_doy', 'rh_anom_doy', 'wind_anom_doy',
                                                    'precip_anom_doy', 'fire_modis', 'p_fire']].copy()

print(f"  FN: {len(fn_weather):,}  FP: {len(fp_weather):,}  TP: {len(tp_weather):,}  TN: {len(tn_weather):,}")

# Box plots comparing FN/FP/TN/TP
fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
for ax, feat, title in zip(axes.ravel(),
                            ['t2m', 'rh', 'wind', 'precip',
                             't2m_anom_doy', 'rh_anom_doy', 'wind_anom_doy', 'precip_anom_doy'],
                            ['Nhiệt độ (°C)', 'Độ ẩm (%)', 'Gió (m/s)', 'Mưa (mm)',
                             'Anomaly nhiệt', 'Anomaly độ ẩm', 'Anomaly gió', 'Anomaly mưa']):
    data = [tp_weather[feat].values, fp_weather[feat].values, fn_weather[feat].values, tn_weather[feat].values]
    labels = [f'TP\n(n={len(tp_weather)})', f'FP\n(n={len(fp_weather)})', f'FN\n(n={len(fn_weather)})', f'TN\n(n={len(tn_weather)})']
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#2ca02c', '#d62728', '#ff7f0e', '#cccccc']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
fig.suptitle('Phase 3: Weather conditions by prediction class\nTP=correct warning, FP=false alarm, FN=missed fire, TN=correct null', fontsize=13, fontweight='bold')
fig.savefig(EX / 'weather_by_class.png', dpi=150)
plt.close()
print(f"  saved weather_by_class.png")

# === Step 5: Quantitative summary ===
print("\n[5/5] quantitative summary...")

# Per-class weather stats
summary = {
    'threshold_p': THR,
    'n_test_days': int(n_days),
    'n_test_rows': int(len(test_all)),
    'class_counts': {
        'TP': int((test_all['class'] == 'TP').sum()),
        'FP': int((test_all['class'] == 'FP').sum()),
        'FN': int((test_all['class'] == 'FN').sum()),
        'TN': int((test_all['class'] == 'TN').sum()),
    },
    'class_rates': {
        'precision': float((test_all['class'] == 'TP').sum() / max((test_all['class'].isin(['TP', 'FP'])).sum(), 1)),
        'recall': float((test_all['class'] == 'TP').sum() / max((test_all['class'].isin(['TP', 'FN'])).sum(), 1)),
    },
    'mean_weather_by_class': {
        cls: {
            't2m': float(df.t2m.mean()),
            'rh': float(df.rh.mean()),
            'wind': float(df.wind.mean()),
            'precip': float(df.precip.mean()),
            'days_since_rain': float(df.days_since_rain.mean()),
            't2m_anom': float(df.t2m_anom_doy.mean()),
            'rh_anom': float(df.rh_anom_doy.mean()),
            'wind_anom': float(df.wind_anom_doy.mean()),
            'p_fire': float(df.p_fire.mean()),
        }
        for cls, df in [('TP', tp_weather), ('FP', fp_weather), ('FN', fn_weather), ('TN', tn_weather)]
    },
    'chronic_fp_cells': [{'lat': float(r.lat), 'lon': float(r.lon), 'fp_days': int(r.FP)} for _, r in chronic_fp.head(10).iterrows()],
    'chronic_fn_cells': [{'lat': float(r.lat), 'lon': float(r.lon), 'fn_days': int(r.FN)} for _, r in chronic_fn.head(10).iterrows()],
    'interpretation': {
        'FN_pattern': 'Fires missed by model tend to occur when [summary]',
        'FP_pattern': 'False alarms tend to occur when [summary]',
    }
}

# Auto-fill interpretation based on numbers
fn_high_wind = fn_weather.wind.mean() > tn_weather.wind.mean()
fn_low_anom = fn_weather.wind_anom_doy.mean() < tn_weather.wind_anom_doy.mean()
fp_high_p = fp_weather.p_fire.mean()
fp_high_temp_anom = fp_weather.t2m_anom_doy.mean() > tn_weather.t2m_anom_doy.mean()

fn_interp = f"FN (missed fires) occur with mean t2m={fn_weather.t2m.mean():.1f}°C, rh={fn_weather.rh.mean():.1f}%, wind={fn_weather.wind.mean():.1f}m/s. "
if fn_low_anom:
    fn_interp += f"Wind anomaly is LOWER than TN (mean {fn_weather.wind_anom_doy.mean():.2f} vs {tn_weather.wind_anom_doy.mean():.2f}) — model fails to flag fires when weather is unusual-but-not-extreme. "
else:
    fn_interp += f"Wind anomaly is HIGHER than TN (mean {fn_weather.wind_anom_doy.mean():.2f} vs {tn_weather.wind_anom_doy.mean():.2f}) — fires happen with unusual wind. "

fp_interp = f"FP (false alarms) occur with mean p_fire={fp_weather.p_fire.mean():.2f} (just above threshold), t2m={fp_weather.t2m.mean():.1f}°C, rh={fp_weather.rh.mean():.1f}%. "
if fp_high_temp_anom:
    fp_interp += f"Temp anomaly is HIGHER than TN (mean {fp_weather.t2m_anom_doy.mean():.2f} vs {tn_weather.t2m_anom_doy.mean():.2f}) — model over-predicts when temperature is anomalously high but other conditions don't support ignition."
else:
    fp_interp += f"Temp anomaly is similar to TN."
summary['interpretation']['FN_pattern'] = fn_interp
summary['interpretation']['FP_pattern'] = fp_interp

print("\n=== Phase 3 Summary ===")
print(f"Threshold: p_fire >= {THR}")
print(f"  TP={summary['class_counts']['TP']}  FP={summary['class_counts']['FP']}  FN={summary['class_counts']['FN']}  TN={summary['class_counts']['TN']}")
print(f"  Precision={summary['class_rates']['precision']:.4f}  Recall={summary['class_rates']['recall']:.4f}")
print(f"  Chronic FP cells (>=30 days): {len(chronic_fp)}")
print(f"  Chronic FN cells (>=5 days): {len(chronic_fn)}")
print(f"\n  Mean weather by class:")
for cls in ['TP', 'FP', 'FN', 'TN']:
    s = summary['mean_weather_by_class'][cls]
    print(f"    {cls}: t2m={s['t2m']:.1f}  rh={s['rh']:.1f}  wind={s['wind']:.2f}  wind_anom={s['wind_anom']:.2f}  p_fire={s['p_fire']:.3f}")
print(f"\n  FN interpretation: {fn_interp}")
print(f"  FP interpretation: {fp_interp}")

# Save cell pattern JSON for UI integration
with open(EX / 'cell_patterns.json', 'w') as f:
    json.dump(cell_pattern_json, f)
with open(EX / 'fn_fp_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n[done] saved → {EX}/")
print(f"  case_dates.json, cell_patterns.json, fn_fp_analysis.json")
print(f"  per_day_maps/ ({len(cases_unique)} PNGs)")
print(f"  cell_failure_patterns.png, weather_by_class.png")
