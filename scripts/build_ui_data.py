#!/usr/bin/env python3
"""
Pre-compute predictions for Wildfire Forecasting System UI.
Produces a JSON file with:
  - List of available dates (test 2023)
  - For each date: cell-level predictions with:
      lat, lon, p_fire, region (North/South), warm_or_cold, top_features, observed_fire
  - Cell metadata (lat/lon grid)

Setup: cold-start spatial OOD model (42 features = 35 cold baseline + 7 anomalies)
trained on North 2018-2021, val North 2022, test South 2023.
Also compute predictions for North 2023 (in-distribution reference).

Output: /home/z/my-project/public/data/predictions.json (compact, ~5MB)
        /home/z/my-project/public/data/cell_meta.json (small)
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
OUT_DIR = Path('/home/z/my-project/public/data')
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f"[info] panel shape={panel.shape}")

# Spatial OOD setup
train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL])
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL])
# Test set: ALL of 2023 (both North and South) so UI can show both regions
test  = panel[(panel.year == 2023)].dropna(subset=[LABEL_COL])
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

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
y_te_south = test[test.region == 'South'][LABEL_COL].values
p_te_south = p_te_iso[test.region.values == 'South']
print(f"South test 2023 (calibrated): AUROC={roc_auc_score(y_te_south, p_te_south):.4f}  PR-AUC={average_precision_score(y_te_south, p_te_south):.4f}  Brier={brier_score_loss(y_te_south, p_te_south):.4f}")

# Add predictions back to test
test = test.assign(p_fire=p_te_iso)

# Also compute SHAP-like feature contributions using xgboost's predict_contributions
print("\n[info] computing feature contributions (gain-based approx)...")
# We use the simpler approach: per-feature importance × feature value, normalized
# Get global feature importance
imp = m.get_score(importance_type='gain')
imp_arr = np.array([imp.get(f, 0) for f in FEATS_FINAL])
# Normalize to sum=1
imp_arr = imp_arr / imp_arr.sum()
print(f"  top-5 features: {sorted(zip(FEATS_FINAL, imp_arr), key=lambda x: -x[1])[:5]}")

# For each test row, compute per-feature contribution score = imp * |value - mean(train)| (signed by direction)
# Sign approximation: positive feature value with positive imp = positive contribution
train_feat_means = train[FEATS_FINAL].mean()
print(f"  computing contributions for {len(test):,} rows...")

# For UI performance, we'll pick a subset of dates: every 7 days in 2023 (52 dates)
dates_2023 = sorted(test.date.unique())
sample_dates = dates_2023[::7]  # weekly samples
if dates_2023[-1] not in sample_dates:
    sample_dates = list(sample_dates) + [dates_2023[-1]]
print(f"  selected {len(sample_dates)} sample dates from 2023")

# Also pick some "interesting" days — days with most fires in South
south_fires_by_date = test[test.region == 'South'].groupby('date')[LABEL_COL].sum().sort_values(ascending=False)
top_fire_dates = south_fires_by_date.head(10).index.tolist()
# Merge and dedupe
all_dates = sorted(set(sample_dates) | set(top_fire_dates))
print(f"  final dates (weekly + top-fire): {len(all_dates)}")

# Filter test to selected dates
test_sel = test[test.date.isin(all_dates)].copy()
print(f"  selected rows: {len(test_sel)}")

# Build per-date JSON
dates_data = []
for date_val in all_dates:
    sub = test_sel[test_sel.date == date_val]
    if len(sub) == 0: continue
    cells = []
    for _, row in sub.iterrows():
        # Top-3 features by contribution magnitude for this cell
        # Compute contributions on the fly (simplified)
        feat_vals = np.array([row[f] for f in FEATS_FINAL])
        # contribution = imp_arr * (feat_vals - train_mean) — sign indicates direction
        train_diff = feat_vals - train_feat_means.values
        contribs = imp_arr * train_diff
        # Top-3 by absolute value
        top3_idx = np.argsort(np.abs(contribs))[-3:][::-1]
        top3 = []
        for idx in top3_idx:
            feat_name = FEATS_FINAL[idx]
            # Translate technical feature names to Vietnamese labels for UI
            label_map = {
                't2m': 'Nhiệt độ',
                'rh': 'Độ ẩm',
                'wind': 'Gió',
                'precip': 'Mưa',
                'doy': 'Ngày trong năm',
                'days_since_rain': 'Số ngày không mưa',
                'heat_index': 'Chỉ số nhiệt',
                'wind_chill': 'Làm lạnh gió',
                't2m_anom_doy': 'Bất thường nhiệt độ',
                'rh_anom_doy': 'Bất thường độ ẩm',
                'precip_anom_doy': 'Bất thường mưa',
                'wind_anom_doy': 'Bất thường gió',
                't2m_anom_roll7': 'Bất thường nhiệt (7 ngày)',
                'rh_anom_roll7': 'Bất thường độ ẩm (7 ngày)',
                'precip_anom_roll7': 'Bất thường mưa (7 ngày)',
                't2m_lag1': 'Nhiệt độ (hôm qua)',
                'rh_lag1': 'Độ ẩm (hôm qua)',
                'wind_lag1': 'Gió (hôm qua)',
                'precip_lag1': 'Mưa (hôm qua)',
                't2m_roll7': 'Nhiệt độ (7 ngày)',
                'rh_roll7': 'Độ ẩm (7 ngày)',
                'precip_roll7_sum': 'Mưa (7 ngày)',
                'precip_roll30_sum': 'Mưa (30 ngày)',
            }
            ui_label = label_map.get(feat_name, feat_name)
            top3.append({
                'name': feat_name,
                'label': ui_label,
                'value': float(feat_vals[idx]),
                'contrib': float(contribs[idx]),
            })
        cells.append({
            'lat': float(row.lat),
            'lon': float(row.lon),
            'region': 'south' if row.region == 'South' else 'north',
            'p_fire': float(row.p_fire),
            'fire_observed': int(row.fire_modis),
            'warm_start': False,  # cold-start model — always False (no fire history used)
            'top_features': top3,
        })
    dates_data.append({
        'date': date_val.strftime('%Y-%m-%d'),
        'n_cells': len(cells),
        'n_fires': int(sub.fire_modis.sum()),
        'cells': cells,
    })

# Save predictions
output = {
    'metadata': {
        'model': 'XGBoost v8.7.2 cold-start spatial OOD (42 feats = 35 cold baseline + 7 anomalies)',
        'train': 'North Vietnam 2018-2021',
        'val': 'North Vietnam 2022',
        'test': 'All Vietnam 2023',
        'features': FEATS_FINAL,
        'n_features': len(FEATS_FINAL),
        'seed': SEED,
        'metrics_south_test_2023_calibrated': {
            'AUROC': float(roc_auc_score(y_te_south, p_te_south)),
            'PR_AUC': float(average_precision_score(y_te_south, p_te_south)),
            'Brier': float(brier_score_loss(y_te_south, p_te_south)),
            'n_test_south': int(len(y_te_south)),
            'n_pos_south': int(y_te_south.sum()),
        },
    },
    'dates': dates_data,
}
with open(OUT_DIR / 'predictions.json', 'w') as f:
    json.dump(output, f)
print(f"\n[done] saved → {OUT_DIR / 'predictions.json'} ({os.path.getsize(OUT_DIR / 'predictions.json') / 1024:.1f} KB)")

# Save cell metadata (Vietnam grid info)
cells_meta = panel[['lat', 'lon', 'region']].drop_duplicates().sort_values(['lat', 'lon'])
cells_meta_dict = {
    'grid_resolution_deg': 0.5,
    'lat_range': [float(panel.lat.min()), float(panel.lat.max())],
    'lon_range': [float(panel.lon.min()), float(panel.lon.max())],
    'n_cells': len(cells_meta),
    'north_boundary_lat': 14.0,
    'cells': [
        {'lat': float(r.lat), 'lon': float(r.lon), 'region': 'south' if r.region == 'South' else 'north'}
        for _, r in cells_meta.iterrows()
    ],
}
with open(OUT_DIR / 'cell_meta.json', 'w') as f:
    json.dump(cells_meta_dict, f)
print(f"[done] saved → {OUT_DIR / 'cell_meta.json'} ({os.path.getsize(OUT_DIR / 'cell_meta.json') / 1024:.1f} KB)")
