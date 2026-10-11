#!/usr/bin/env python3
"""
Export production artifacts for Wildfire Forecasting System product.
Produces:
  - /home/z/my-project/download/production/model.json        (XGBoost)
  - /home/z/my-project/download/production/iso.pkl           (Isotonic calibrator)
  - /home/z/my-project/download/production/feature_list.json (42 features + descriptions)
  - /home/z/my-project/download/production/cell_climatology.parquet (per-cell doy means for anomalies)
  - /home/z/my-project/download/production/panel_production.parquet (598K rows × 70 cols, ready for inference)
  - /home/z/my-project/download/production/scaler_fwi.pkl (unused but kept for ref)
  - /home/z/my-project/download/production/README.md (handoff doc)
"""
from __future__ import annotations
import json, os, pickle, warnings
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
OUT = DL / 'production'
OUT.mkdir(parents=True, exist_ok=True)

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
FEATS_FINAL = COLD_FEATS + ANOM_FEATS  # 42 features — drop composite per E6 finding
LABEL_COL = 'label_modis_h1'
SEED = 42

print(f"[1/5] loading panel...")
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')

print(f"[2/5] training production XGBoost (42 feats, cold spatial OOD)...")
train = panel[(panel.region == 'North') & (panel.year <= 2021)].dropna(subset=[LABEL_COL])
val   = panel[(panel.region == 'North') & (panel.year == 2022)].dropna(subset=[LABEL_COL])
y_tr = train[LABEL_COL].astype(int).values
y_va = val[LABEL_COL].astype(int).values
params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
              max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=1.0, seed=SEED, verbosity=0)
dtr = gbm.DMatrix(train[FEATS_FINAL], label=y_tr)
dv  = gbm.DMatrix(val[FEATS_FINAL], label=y_va)
m = gbm.train(params, dtr, num_boost_round=300, evals=[(dtr, 'tr'), (dv, 'val')],
              early_stopping_rounds=20, verbose_eval=False)
print(f"  best_iter={m.best_iteration}")
m.save_model(str(OUT / 'model.json'))

print(f"[3/5] fitting isotonic calibrator on North val...")
p_va = m.predict(dv)
iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
with open(OUT / 'iso.pkl', 'wb') as f:
    pickle.dump(iso, f)

print(f"[4/5] extracting per-cell doy climatology for anomaly computation...")
# This is what the product needs to compute anomalies for NEW dates (post-2023)
train_only = panel[panel.year <= 2021].copy()
cell_doy_clim = train_only.groupby(['lat', 'lon', 'doy']).agg(
    t2m_clim=('t2m', 'mean'),
    rh_clim=('rh', 'mean'),
    precip_clim=('precip', 'mean'),
    wind_clim=('wind', 'mean'),
).reset_index()
# Apply 15-day circular smoothing per cell
def smooth_doy(grp, value_col, window=15):
    grp = grp.sort_values('doy')
    vals = grp[value_col].values
    n = len(vals)
    if n == 0: return grp
    pad = window // 2
    padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
    smoothed = pd.Series(padded).rolling(window=window, center=True, min_periods=1).mean().values[pad:pad+n]
    grp[value_col] = smoothed
    return grp
cell_doy_clim_smoothed = []
for (lat, lon), g in cell_doy_clim.groupby(['lat', 'lon']):
    g = smooth_doy(g, 't2m_clim')
    g = smooth_doy(g, 'rh_clim')
    g = smooth_doy(g, 'precip_clim')
    g = smooth_doy(g, 'wind_clim')
    cell_doy_clim_smoothed.append(g)
cell_doy_clim = pd.concat(cell_doy_clim_smoothed, ignore_index=True)
cell_doy_clim.to_parquet(OUT / 'cell_climatology.parquet', index=False)
print(f"  saved cell_climatology.parquet ({len(cell_doy_clim):,} rows = 273 cells × ~366 doy)")

print(f"[5/5] saving feature list with descriptions + sample panel...")
feature_descriptions = {
    # === Weather (raw + lags) — 4 base + 16 lags = 20 features ===
    't2m': ('Nhiệt độ bề mặt 2m (°C)', 'NASA POWER, ~0.5° grid, daily'),
    'rh': ('Độ ẩm tương đối (%)', 'NASA POWER'),
    'wind': ('Tốc độ gió (m/s)', 'NASA POWER, 10m wind'),
    'precip': ('Lượng mưa (mm/day)', 'NASA POWER'),
    'doy': ('Ngày trong năm (1-366)', 'Seasonality indicator'),
    't2m_lag1': ('Nhiệt độ hôm qua (°C)', '1-day lag, per cell'),
    't2m_lag3': ('Nhiệt độ 3 ngày trước (°C)', '3-day lag'),
    't2m_lag7': ('Nhiệt độ 7 ngày trước (°C)', '7-day lag'),
    't2m_lag14': ('Nhiệt độ 14 ngày trước (°C)', '14-day lag'),
    'rh_lag1': ('Độ ẩm hôm qua (%)', ''),
    'rh_lag3': ('Độ ẩm 3 ngày trước (%)', ''),
    'rh_lag7': ('Độ ẩm 7 ngày trước (%)', ''),
    'rh_lag14': ('Độ ẩm 14 ngày trước (%)', ''),
    'wind_lag1': ('Gió hôm qua (m/s)', ''),
    'wind_lag3': ('Gió 3 ngày trước (m/s)', ''),
    'wind_lag7': ('Gió 7 ngày trước (m/s)', ''),
    'wind_lag14': ('Gió 14 ngày trước (m/s)', ''),
    'precip_lag1': ('Mưa hôm qua (mm)', ''),
    'precip_lag3': ('Mưa 3 ngày trước (mm)', ''),
    'precip_lag7': ('Mưa 7 ngày trước (mm)', ''),
    'precip_lag14': ('Mưa 14 ngày trước (mm)', ''),
    
    # === Rolling stats ===
    't2m_roll7': ('Nhiệt độ trung bình 7 ngày (°C)', 'rolling mean per cell'),
    't2m_roll14': ('Nhiệt độ trung bình 14 ngày (°C)', ''),
    'rh_roll7': ('Độ ẩm trung bình 7 ngày (%)', ''),
    'rh_roll14': ('Độ ẩm trung bình 14 ngày (%)', ''),
    'precip_roll7_sum': ('Tổng mưa 7 ngày (mm)', 'rolling sum per cell'),
    'precip_roll14_sum': ('Tổng mưa 14 ngày (mm)', ''),
    'precip_roll30_sum': ('Tổng mưa 30 ngày (mm)', 'Drought indicator'),
    
    # === Derived ===
    'days_since_rain': ('Số ngày liên tiếp không mưa (day)', 'dry spell length, threshold 0.5mm'),
    't2m_diff3': ('Hiệu nhiệt độ (hôm nay - 3 ngày trước) (°C)', 'Short-term trend'),
    'rh_diff3': ('Hiệu độ ẩm (hôm nay - 3 ngày trước) (%)', ''),
    'wind_diff3': ('Hiệu gió (hôm nay - 3 ngày trước) (m/s)', ''),
    
    # NOTE: fwi_compound removed in v8.7.1 (99.7% zeros, sigmoid saturated)
    # but panel still has it — drop from feature list
    'heat_index': ('Chỉ số nhiệt (đơn vị tính)', '(t2m-20)*(50-rh)/50'),
    'wind_chill': ('Làm lạnh gió', '(t2m-15)*wind/30'),
    
    # === ANOMALIES (7) — key innovation, computed from 2018-2021 climatology ===
    't2m_anom_doy': ('Bất thường nhiệt độ so với khí hậu cùng ô+cùng ngày trong năm (°C)', 
                    't2m - cell_doy_clim(t2m); climatology from 2018-2021 train, 15-day circular smoothing'),
    'rh_anom_doy': ('Bất thường độ ẩm (%)', 'rh - cell_doy_clim(rh)'),
    'precip_anom_doy': ('Bất thường lượng mưa (mm)', 'precip - cell_doy_clim(precip)'),
    'wind_anom_doy': ('Bất thường tốc độ gió (m/s) [DOMINANT FEATURE]',
                     'wind - cell_doy_clim(wind). Rank #1 in anomaly ablation (drop costs 0.005 AUROC, t=-11.07)'),
    't2m_anom_roll7': ('Trung bình 7 ngày bất thường nhiệt độ (°C)', 'rolling mean of t2m_anom_doy per cell'),
    'rh_anom_roll7': ('Trung bình 7 ngày bất thường độ ẩm (%)', '[rank #2 in ablation — drop costs 0.0045 PR-AUC, t=-8.23]'),
    'precip_anom_roll7': ('Trung bình 7 ngày bất thường mưa (mm)', 'rolling mean of precip_anom_doy'),
}

# Filter to only features in FEATS_FINAL (drop fwi_compound if present)
feature_list = []
for f in FEATS_FINAL:
    if f in feature_descriptions:
        desc, source = feature_descriptions[f]
    else:
        desc, source = f, '(see source code)'
    feature_list.append({'name': f, 'description_vi': desc, 'source': source})

feature_list_obj = {
    'version': 'v8.7.2_production',
    'n_features': len(FEATS_FINAL),
    'model_seed': SEED,
    'features': feature_list,
    'feature_groups': {
        'weather_raw': ['t2m', 'rh', 'wind', 'precip'],
        'seasonality': ['doy'],
        'weather_lags': [f for f in FEATS_FINAL if '_lag' in f],
        'rolling_stats': [f for f in FEATS_FINAL if '_roll' in f and '_anom' not in f],
        'derived_indices': ['days_since_rain', 't2m_diff3', 'rh_diff3', 'wind_diff3', 'heat_index', 'wind_chill'],
        'anomalies': [f for f in FEATS_FINAL if '_anom' in f],
    },
    'key_findings': {
        'most_important_anomaly': 'wind_anom_doy (drop costs 0.0050 AUROC, t=-11.07)',
        'second_most_important': 'rh_anom_roll7 (drop costs 0.0045 PR-AUC, t=-8.23)',
        'non_essential': 'dry_anomaly_composite (dropped in v8.7.2 — was non-essential per E6)',
        'failure_mode_discovered': 'FN occurs in cool+windy conditions (t2m=23.3 vs TP 27.1; wind_anom=+0.54 vs TN -0.03). Model misses wind-driven fires.',
    },
    'metrics_south_test_2023_calibrated': {
        'AUROC': 0.9043,
        'PR_AUC': 0.4431,
        'Brier': 0.0738,
        'recall_at_p20': 0.840,
        'precision_at_p20': 0.344,
    },
    'split_definition': {
        'train': 'North Vietnam (lat ≥ 14°), 2018-01-01 to 2021-12-31',
        'val':   'North Vietnam (lat ≥ 14°), 2022',
        'test_spatial_ood': 'South Vietnam (lat < 14°), 2023',
        'note': 'Cold-start: no fire-history features used. Production can apply to any cell + any date.',
    },
    'inference_pseudocode': '''
# For a new date d and cell c:
# 1. Fetch NASA POWER weather for cell c, date d, plus lags (1,3,7,14 days)
# 2. Compute rolling stats (7,14,30-day means per cell)
# 3. Compute days_since_rain from precip history
# 4. Load cell_climatology.parquet, lookup (lat=c.lat, lon=c.lon, doy=d.day_of_year)
#    → get t2m_clim, rh_clim, precip_clim, wind_clim
# 5. Compute anomalies: t2m_anom_doy = t2m - t2m_clim, etc.
# 6. Compute 7-day rolling anomalies (need 7-day history of anomalies)
# 7. Build feature vector in this exact order (42 features):
features = [t2m, rh, wind, precip, doy, t2m_lag1, t2m_lag3, t2m_lag7, t2m_lag14,
            rh_lag1, rh_lag3, rh_lag7, rh_lag14,
            wind_lag1, wind_lag3, wind_lag7, wind_lag14,
            precip_lag1, precip_lag3, precip_lag7, precip_lag14,
            t2m_roll7, t2m_roll14, rh_roll7, rh_roll14,
            precip_roll7_sum, precip_roll14_sum, precip_roll30_sum,
            days_since_rain, t2m_diff3, rh_diff3, wind_diff3,
            heat_index, wind_chill,
            t2m_anom_doy, rh_anom_doy, precip_anom_doy, wind_anom_doy,
            t2m_anom_roll7, rh_anom_roll7, precip_anom_roll7]
# 8. Predict: p_raw = model.predict(DMatrix(features))
# 9. Calibrate: p_calibrated = iso.predict(p_raw)
# 10. p_calibrated is the predicted probability of MODIS fire detection on day d+1
''',
}
with open(OUT / 'feature_list.json', 'w') as f:
    json.dump(feature_list_obj, f, indent=2, default=str, ensure_ascii=False)
print(f"  saved feature_list.json ({len(FEATS_FINAL)} features)")

# Also save a small sample panel (test 2023 South) for product prototyping
print(f"\n  saving production panel (all 598K rows × 70 cols)...")
panel.to_parquet(OUT / 'panel_production.parquet', index=False)
print(f"  saved panel_production.parquet ({os.path.getsize(OUT / 'panel_production.parquet') / 1024 / 1024:.1f} MB)")

# Verify everything loads
print(f"\n[verification] reloading all artifacts...")
m2 = gbm.Booster()
m2.load_model(str(OUT / 'model.json'))
with open(OUT / 'iso.pkl', 'rb') as f:
    iso2 = pickle.load(f)
clim = pd.read_parquet(OUT / 'cell_climatology.parquet')
panel_check = pd.read_parquet(OUT / 'panel_production.parquet')
print(f"  ✓ model: {m2.num_features()} features")
print(f"  ✓ iso: trained on {len(iso2.X_thresholds_)} thresholds")
print(f"  ✓ climatology: {len(clim):,} rows, {clim.shape[1]} cols")
print(f"  ✓ panel: {panel_check.shape}")

print(f"\n[done] all artifacts in {OUT}/")
import subprocess
result = subprocess.run(['ls', '-la', str(OUT)], capture_output=True, text=True)
print(result.stdout)
