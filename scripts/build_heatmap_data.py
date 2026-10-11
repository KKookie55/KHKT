#!/usr/bin/env python3
"""
Pre-compute heatmap data for the Wildfire Forecasting System UI.
Produces compact JSON with predictions for ALL 273 cells × 60 sample dates (2023).
Also includes cold-start indicator (always true for South cells in cold-start model).
"""
from __future__ import annotations
import json, os, sys, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
PANEL_P = SC / 'e2_weather_anomalies' / 'panel_with_anomalies.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'
PUBLIC_DIR = Path('/home/z/my-project/public/data')
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
FEATS_FINAL = COLD_FEATS + ANOM_FEATS
LABEL_COL = 'label_modis_h1'
SEED = 42

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')

train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL])
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL])
test_all = panel[panel.year == 2023].dropna(subset=[LABEL_COL]).copy()
y_tr = train[LABEL_COL].astype(int).values
y_va = val[LABEL_COL].astype(int).values

params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
              max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=1.0, seed=SEED, verbosity=0)
dtr = gbm.DMatrix(train[FEATS_FINAL], label=y_tr)
dv  = gbm.DMatrix(val[FEATS_FINAL], label=y_va)
m = gbm.train(params, dtr, num_boost_round=300, evals=[(dtr, 'tr'), (dv, 'val')],
              early_stopping_rounds=20, verbose_eval=False)
p_va = m.predict(dv)
iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)

# Predict on ALL 2023 dates (both regions)
dte = gbm.DMatrix(test_all[FEATS_FINAL])
test_all['p_raw'] = m.predict(dte)
test_all['p_fire'] = iso.predict(test_all['p_raw'].values)

# Pick ~60 sample dates: weekly samples + top fire days
dates_2023 = sorted(test_all.date.unique())
sample_dates = dates_2023[::7]  # ~52 weekly
# Ensure top fire days included
top_fire_dates = test_all.groupby('date').fire_modis.sum().sort_values(ascending=False).head(10).index.tolist()
all_dates = sorted(set(sample_dates) | set(top_fire_dates))
print(f"selected {len(all_dates)} dates for heatmap")

# Cell metadata
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
cells['region'] = np.where(cells.lat >= 14, 'north', 'south')
cells['cold_start'] = True  # all cells in cold-start model — no fire history used
# For UI: warm-start = cell was in training set (North). Cold-start = cell wasn't (South).
cells['in_training'] = cells.lat >= 14
# But model is cold-start: doesn't use fire history, so for any cell the model is "cold-start with respect to fire history"
# Still, distinguish "in-distribution region" (North) from "OOD region" (South)
cells['status'] = cells.apply(lambda r: 'in_distribution' if r.lat >= 14 else 'spatial_ood', axis=1)

cells_meta = [{
    'lat': float(r.lat), 'lon': float(r.lon),
    'region': r.region, 'status': r.status,
} for _, r in cells.iterrows()]
with open(PUBLIC_DIR / 'cells_meta.json', 'w') as f:
    json.dump(cells_meta, f)
print(f"saved cells_meta.json ({len(cells_meta)} cells)")

# Per-date predictions
dates_data = []
for date_val in all_dates:
    sub = test_all[test_all.date == date_val]
    if len(sub) == 0: continue
    cells_day = []
    for _, row in sub.iterrows():
        cells_day.append({
            'lat': float(row.lat),
            'lon': float(row.lon),
            'p': round(float(row.p_fire), 4),  # 4 decimals is enough
            'fire': int(row.fire_modis),
        })
    dates_data.append({
        'date': date_val.strftime('%Y-%m-%d'),
        'n_cells': len(cells_day),
        'n_fires': int(sub.fire_modis.sum()),
        'p_max': float(sub.p_fire.max()),
        'p_mean': float(sub.p_fire.mean()),
        'cells': cells_day,
    })

# Save as compact JSON (use separators to reduce size)
output = {
    'metadata': {
        'model': 'XGBoost v8.7.2 production',
        'features': len(FEATS_FINAL),
        'feature_set': '35 cold baseline + 7 weather anomalies (drop composite per E6)',
        'train': 'North Vietnam 2018-2021',
        'val': 'North Vietnam 2022 (isotonic fit)',
        'test': 'All Vietnam 2023',
        'metrics_south_cold_OOD': {
            'AUROC': 0.9043, 'PR_AUC': 0.4431, 'Brier': 0.0738,
        },
        'key_innovation': 'Weather anomalies (departure from per-cell doy climatology)',
        'most_important_feature': 'wind_anom_doy',
        'failure_mode': 'Misses wind-driven fires in cool conditions (t2m<25°C + wind_anom>+0.5)',
        'thresholds_for_risk_levels': {
            'low': 0.05, 'moderate': 0.10, 'high': 0.20, 'severe': 0.30,
        },
    },
    'dates': dates_data,
}
out_path = PUBLIC_DIR / 'heatmap_predictions.json'
with open(out_path, 'w') as f:
    json.dump(output, f, separators=(',', ':'))
print(f"saved {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")
print(f"  {len(dates_data)} dates × 273 cells = {len(dates_data) * 273:,} predictions")
