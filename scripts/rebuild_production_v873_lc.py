#!/usr/bin/env python3
"""
Rebuild production artifacts with v8.7.3-LC (46 features with land cover).

Outputs:
  - /home/z/my-project/download/production/model.json     (XGBoost v8.7.3-LC)
  - /home/z/my-project/download/production/iso.pkl         (Isotonic calibrator)
  - /home/z/my-project/download/production/feature_list.json (46 features + LC descriptions)
  - /home/z/my-project/download/production/cell_climatology.parquet (unchanged)
  - /home/z/my-project/download/production/landcover_lookup.parquet (per-cell LC proportions)
  - /home/z/my-project/public/data/heatmap_predictions.json (pre-computed for UI)
"""
from __future__ import annotations
import json, os, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
PANEL_P = SC / 'panel_v873_landcover.parquet'  # panel with anomalies + interactions + landcover
META_P = DL / 'v871' / 'model' / 'meta_v871.json'
PROD = DL / 'production'
PROD.mkdir(parents=True, exist_ok=True)
PUBLIC_DATA = Path('/home/z/my-project/public/data')
PUBLIC_DATA.mkdir(parents=True, exist_ok=True)

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']
FIRE_HIST = ['cell_fire_rate_exp', 'cell_fire_rate_dry_exp', 'cell_fire_count_30d']
COLD_FEATS = [f for f in BASE_FEATS if f not in FIRE_HIST]
ANOM_FEATS = ['t2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
              't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7']
LC_FEATS = ['forest_prop', 'agriculture_prop', 'urban_prop', 'water_prop']
FEATS_FINAL = COLD_FEATS + ANOM_FEATS + LC_FEATS  # 46 features (v8.7.3-LC production)
LABEL_COL = 'label_modis_h1'
SEED = 42

print(f"[1/6] loading panel...")
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f"  panel shape: {panel.shape}")

print(f"\n[2/6] training v8.7.3-LC production XGBoost (46 feats)...")
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
m.save_model(str(PROD / 'model.json'))

# Calibrator
p_va = m.predict(dv)
iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
with open(PROD / 'iso.pkl', 'wb') as f:
    pickle.dump(iso, f)

# Evaluate
test_south = panel[(panel.region == 'South') & (panel.year == 2023)].dropna(subset=[LABEL_COL])
y_te_south = test_south[LABEL_COL].astype(int).values
dte_south = gbm.DMatrix(test_south[FEATS_FINAL])
p_te_south = iso.predict(m.predict(dte_south))
metrics = {
    'AUROC': float(roc_auc_score(y_te_south, p_te_south)),
    'PR_AUC': float(average_precision_score(y_te_south, p_te_south)),
    'Brier': float(brier_score_loss(y_te_south, p_te_south)),
    'n_test_south': int(len(y_te_south)),
    'n_pos_south': int(y_te_south.sum()),
}
print(f"  South test 2023 metrics: {metrics}")

print(f"\n[3/6] saving feature_list.json with updated descriptions...")
feature_descriptions = {
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
    't2m_roll7': ('Nhiệt độ trung bình 7 ngày (°C)', 'rolling mean per cell'),
    't2m_roll14': ('Nhiệt độ trung bình 14 ngày (°C)', ''),
    'rh_roll7': ('Độ ẩm trung bình 7 ngày (%)', ''),
    'rh_roll14': ('Độ ẩm trung bình 14 ngày (%)', ''),
    'precip_roll7_sum': ('Tổng mưa 7 ngày (mm)', 'rolling sum per cell'),
    'precip_roll14_sum': ('Tổng mưa 14 ngày (mm)', ''),
    'precip_roll30_sum': ('Tổng mưa 30 ngày (mm)', 'Drought indicator'),
    'days_since_rain': ('Số ngày liên tiếp không mưa (day)', 'dry spell length, threshold 0.5mm'),
    't2m_diff3': ('Hiệu nhiệt độ (hôm nay - 3 ngày trước) (°C)', 'Short-term trend'),
    'rh_diff3': ('Hiệu độ ẩm (hôm nay - 3 ngày trước) (%)', ''),
    'wind_diff3': ('Hiệu gió (hôm nay - 3 ngày trước) (m/s)', ''),
    'heat_index': ('Chỉ số nhiệt (đơn vị tính)', '(t2m-20)*(50-rh)/50'),
    'wind_chill': ('Làm lạnh gió', '(t2m-15)*wind/30'),
    't2m_anom_doy': ('Bất thường nhiệt độ so với khí hậu cùng ô+cùng ngày trong năm (°C)',
                    't2m - cell_doy_clim(t2m); climatology from 2018-2021 train, 15-day circular smoothing'),
    'rh_anom_doy': ('Bất thường độ ẩm (%)', 'rh - cell_doy_clim(rh)'),
    'precip_anom_doy': ('Bất thường lượng mưa (mm)', 'precip - cell_doy_clim(precip)'),
    'wind_anom_doy': ('Bất thường tốc độ gió (m/s) [DOMINANT ANOMALY]',
                     'wind - cell_doy_clim(wind). Rank #1 in anomaly ablation (drop costs 0.005 AUROC, t=-11.07)'),
    't2m_anom_roll7': ('Trung bình 7 ngày bất thường nhiệt độ (°C)', 'rolling mean of t2m_anom_doy per cell'),
    'rh_anom_roll7': ('Trung bình 7 ngày bất thường độ ẩm (%)', '[rank #2 in anomaly ablation — drop costs 0.0045 PR-AUC, t=-8.23]'),
    'precip_anom_roll7': ('Trung bình 7 ngày bất thường mưa (mm)', 'rolling mean of precip_anom_doy'),
    # === LAND COVER (new in v8.7.3) ===
    'forest_prop': ('Tỷ lệ diện tích rừng trong ô (0-1) [TOP LC FEATURE, rank #3 in importance]',
                    'ESA WorldCover 2021 v200. Class 10 (tree cover) + 95 (mangroves). Aggregated to 0.5° grid.'),
    'agriculture_prop': ('Tỷ lệ diện tích nông nghiệp (0-1)',
                         'ESA WorldCover 2021. Class 40 (cropland). Helps distinguish agricultural burning from forest fires.'),
    'urban_prop': ('Tỷ lệ diện tích đô thị (0-1)',
                   'ESA WorldCover 2021. Class 50 (built-up). Low fire risk proxy.'),
    'water_prop': ('Tỷ lệ diện tích mặt nước (0-1)',
                  'ESA WorldCover 2021. Classes 80 (water) + 90 (wetland). No fire risk.'),
}

feature_list = []
for f in FEATS_FINAL:
    desc, source = feature_descriptions.get(f, (f, ''))
    feature_list.append({'name': f, 'description_vi': desc, 'source': source})

feature_list_obj = {
    'version': 'v8.7.3_LC_production',
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
        'land_cover': LC_FEATS,
    },
    'key_findings_v873_LC': {
        'vs_v872_baseline': {
            'AUROC': '+0.0099 (t=15.34, sig)',
            'PR_AUC': '+0.0425 (t=99.26, sig) — huge improvement',
            'Brier': '-0.0043 (t=-9.23, sig better)',
            'ECE': '-0.0067 (t=-5.87, sig better)',
            'FP_false_alarms': '-515 (t=-5.69, sig better)',
            'precision': '+0.021 (t=9.55, sig better)',
            'warn_p20': '-2.0% (t=-4.93, sig better) — fewer cells flagged, addresses South over-prediction',
            'tradeoff_FN': '+52.6 (t=2.06, NOT sig) — slight increase in missed fires, acceptable trade-off',
        },
        'top_land_cover_features_by_importance': {
            'forest_prop': 'rank #3 (gain=389) — dominant LC feature, captures real forest fire risk',
            'urban_prop': 'rank #6 (gain=181)',
            'agriculture_prop': 'rank #8 (gain=173) — distinguishes agricultural burning (down-weighted)',
            'water_prop': 'rank #19 (gain=78)',
        },
        'mechanism': 'Model learns: high forest_prop + dry conditions = real wildfire risk. High agriculture_prop = agricultural burning (down-weighted). Addresses user complaint about South over-prediction.',
    },
    'split_definition': {
        'train': 'North Vietnam (lat ≥ 14°), 2018-01-01 to 2021-12-31',
        'val':   'North Vietnam (lat ≥ 14°), 2022',
        'test_spatial_ood': 'South Vietnam (lat < 14°), 2023',
        'note': 'Cold-start: no fire-history features used. Production can apply to any cell + any date.',
    },
    'metrics_south_test_2023_calibrated': metrics,
    'inference_pseudocode': '''
# For a new date d and cell c:
# 1. Fetch NASA POWER weather for cell c, date d, plus lags (1,3,7,14 days)
# 2. Compute rolling stats (7,14,30-day means per cell)
# 3. Compute days_since_rain from precip history
# 4. Load cell_climatology.parquet, lookup (lat=c.lat, lon=c.lon, doy=d.day_of_year)
#    → get t2m_clim, rh_clim, precip_clim, wind_clim
# 5. Compute anomalies: t2m_anom_doy = t2m - t2m_clim, etc.
# 6. Compute 7-day rolling anomalies (need 7-day history of anomalies)
# 7. Look up land cover: load landcover_lookup.parquet, find (lat=c.lat, lon=c.lon)
#    → get forest_prop, agriculture_prop, urban_prop, water_prop
# 8. Build feature vector in this exact order (46 features)
# 9. Predict: p_raw = model.predict(DMatrix(features))
# 10. Calibrate: p_calibrated = iso.predict(p_raw)
''',
}
with open(PROD / 'feature_list.json', 'w') as f:
    json.dump(feature_list_obj, f, indent=2, default=str, ensure_ascii=False)
print(f"  saved feature_list.json ({len(FEATS_FINAL)} features)")

# Save landcover lookup
print(f"\n[4/6] saving landcover_lookup.parquet...")
lc_lookup = panel[['lat', 'lon'] + LC_FEATS].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
lc_lookup.to_parquet(PROD / 'landcover_lookup.parquet', index=False)
print(f"  saved ({len(lc_lookup)} cells)")

# === Rebuild heatmap predictions for UI ===
print(f"\n[5/6] rebuilding heatmap_predictions.json for UI...")
test_all = panel[panel.year == 2023].dropna(subset=[LABEL_COL]).copy()
dte_all = gbm.DMatrix(test_all[FEATS_FINAL])
test_all['p_fire'] = iso.predict(m.predict(dte_all))

dates_2023 = sorted(test_all.date.unique())
sample_dates = dates_2023[::7]
top_fire_dates = test_all.groupby('date').fire_modis.sum().sort_values(ascending=False).head(10).index.tolist()
all_dates = sorted(set(sample_dates) | set(top_fire_dates))
print(f"  selected {len(all_dates)} dates for heatmap")

dates_data = []
for date_val in all_dates:
    sub = test_all[test_all.date == date_val]
    if len(sub) == 0: continue
    cells_day = []
    for _, row in sub.iterrows():
        cells_day.append({
            'lat': float(row.lat),
            'lon': float(row.lon),
            'p': round(float(row.p_fire), 4),
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

output = {
    'metadata': {
        'model': 'XGBoost v8.7.3-LC production (with land cover)',
        'features': len(FEATS_FINAL),
        'feature_set': '35 cold baseline + 7 weather anomalies + 4 land cover (forest, agriculture, urban, water)',
        'train': 'North Vietnam 2018-2021',
        'val': 'North Vietnam 2022 (isotonic fit)',
        'test': 'All Vietnam 2023',
        'metrics_south_cold_OOD': metrics,
        'key_innovation_v873_LC': 'Added ESA WorldCover 2021 land cover proportions. Model learns to distinguish forest fires (high forest_prop + dry) from agricultural burning (high agriculture_prop). Reduces FP by 515 cells/year, precision +2.1pp.',
        'most_important_feature': 'doy (gain=628)',
        'most_important_land_cover_feature': 'forest_prop (rank #3, gain=389)',
        'failure_mode': 'Misses wind-driven fires in cool conditions (still applies but less severe)',
        'thresholds_for_risk_levels': {
            'low': 0.05, 'moderate': 0.10, 'high': 0.20, 'severe': 0.30,
        },
    },
    'dates': dates_data,
}
with open(PUBLIC_DATA / 'heatmap_predictions.json', 'w') as f:
    json.dump(output, f, separators=(',', ':'))
print(f"  saved heatmap_predictions.json ({os.path.getsize(PUBLIC_DATA / 'heatmap_predictions.json') / 1024:.1f} KB)")
print(f"  {len(dates_data)} dates × 273 cells = {len(dates_data) * 273:,} predictions")

# Save updated production panel
print(f"\n[6/6] saving panel_production.parquet...")
panel.to_parquet(PROD / 'panel_production.parquet', index=False)
print(f"  saved ({os.path.getsize(PROD / 'panel_production.parquet') / 1024 / 1024:.1f} MB)")

print(f"\n[done] all production artifacts updated to v8.7.3-LC")
print(f"  model: {PROD / 'model.json'}")
print(f"  iso:   {PROD / 'iso.pkl'}")
print(f"  features: {PROD / 'feature_list.json'}")
print(f"  panel: {PROD / 'panel_production.parquet'}")
print(f"  heatmap UI data: {PUBLIC_DATA / 'heatmap_predictions.json'}")
