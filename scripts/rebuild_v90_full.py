#!/usr/bin/env python3
"""
Rebuild FULL v9.0 model (61 features) from raw data.
Steps:
  1. Build panel from raw weather + fire CSVs (35 cold baseline features)
  2. Compute 7 weather anomalies
  3. Download land cover from ESA WorldCover (7 LC features)
  4. Compute geo features (static lookup, 9 features)
  5. Download NDVI from MODIS MOD13A1 (3 features)
  6. Train v9.0 multi-task: forest fire + any fire
  7. Save ALL to public/model/v90/ (git-tracked)
"""
import os, sys, json, pickle, time, warnings
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings('ignore')

t0 = time.time()
WEATHER = '/home/z/my-project/upload/wildfire_data/wildfire_data/weather/weather_nasa_power_full.csv'
FIRE_DIR = '/home/z/my-project/upload/wildfire_data/wildfire_data/firms'
MODEL_DIR = '/home/z/my-project/public/model/v90'
os.makedirs(MODEL_DIR, exist_ok=True)

# === STEP 1: Build panel ===
print(f'[{time.time()-t0:.1f}s] Step 1: Build panel from raw data', flush=True)
w = pd.read_csv(WEATHER)
w['date'] = pd.to_datetime(w['date'])
w['year'] = w['date'].dt.year
w['doy'] = w['date'].dt.dayofyear
print(f'  weather: {w.shape}', flush=True)

fires = []
for y in range(2018, 2024):
    f = pd.read_csv(f'{FIRE_DIR}/fires_{y}_daily.csv')
    fires.append(f)
fires = pd.concat(fires, ignore_index=True)
fires['acq_date'] = pd.to_datetime(fires['acq_date'])
fires.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
fires['lat'] = (fires['lat'] / 0.5).round() * 0.5
fires['lon'] = (fires['lon'] / 0.5).round() * 0.5
fire_agg = fires.groupby(['lat','lon','acq_date']).agg(
    fire=('acq_date','size'),
    n_high=('confidence', lambda x: (x=='high').sum()),
    n_nominal=('confidence', lambda x: (x=='nominal').sum()),
    n_low=('confidence', lambda x: (x=='low').sum()),
).reset_index()
fire_agg.rename(columns={'acq_date': 'date'}, inplace=True)
fire_agg['fire'] = (fire_agg['fire'] > 0).astype(int)
fire_agg['fire_high'] = (fire_agg['n_high'] > 0).astype(int)
panel = w.merge(fire_agg, on=['lat','lon','date'], how='left')
for c in ['fire','fire_high','n_high','n_nominal','n_low']:
    panel[c] = panel[c].fillna(0).astype(int)
panel = panel.sort_values(['lat','lon','date']).reset_index(drop=True)
panel['label_modis_h1'] = panel.groupby(['lat','lon'])['fire'].shift(-1)
panel['label_high_h1'] = panel.groupby(['lat','lon'])['fire_high'].shift(-1)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f'  panel: {panel.shape}', flush=True)

# === STEP 2: Cold baseline features (35) ===
print(f'[{time.time()-t0:.1f}s] Step 2: Cold baseline features (35)', flush=True)
for v in ['t2m','rh','wind','precip']:
    for lag in [1,3,7,14]:
        panel[f'{v}_lag{lag}'] = panel.groupby(['lat','lon'])[v].shift(lag)
for v in ['t2m','rh']:
    for ww in [7,14]:
        panel[f'{v}_roll{ww}'] = panel.groupby(['lat','lon'])[v].transform(lambda s: s.rolling(ww, min_periods=1).mean())
for ww in [7,14,30]:
    panel[f'precip_roll{ww}_sum'] = panel.groupby(['lat','lon'])['precip'].transform(lambda s: s.rolling(ww, min_periods=1).sum())
def dsr(s, t=0.5):
    is_dry = (s < t).astype(int); g = (is_dry == 0).cumsum()
    return g.groupby(g).cumcount() * is_dry
panel['days_since_rain'] = panel.groupby(['lat','lon'])['precip'].transform(dsr)
for v in ['t2m','rh','wind']:
    panel[f'{v}_diff3'] = panel.groupby(['lat','lon'])[v].diff(3)
panel['heat_index'] = (panel['t2m'] - 20) * (50 - panel['rh']) / 50.0
panel['wind_chill'] = (panel['t2m'] - 15) * panel['wind'] / 30.0
panel['fwi_compound'] = 0.0

# === STEP 3: Anomalies (7) ===
print(f'[{time.time()-t0:.1f}s] Step 3: Weather anomalies (7)', flush=True)
train_w = panel[panel.year <= 2021]
cell_doy_clim = train_w.groupby(['lat','lon','doy']).agg(
    t2m_clim=('t2m','mean'), rh_clim=('rh','mean'),
    precip_clim=('precip','mean'), wind_clim=('wind','mean'),
).reset_index()
def smooth(g):
    g = g.sort_values('doy')
    for c in ['t2m_clim','rh_clim','precip_clim','wind_clim']:
        vals = g[c].values; n = len(vals)
        if n == 0: continue
        pad = 7
        padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
        g[c] = pd.Series(padded).rolling(15, center=True, min_periods=1).mean().values[pad:pad+n]
    return g
cell_doy_clim = cell_doy_clim.groupby(['lat','lon'], group_keys=False).apply(smooth)
panel = panel.merge(cell_doy_clim, on=['lat','lon','doy'], how='left')
for c in ['t2m_clim','rh_clim','precip_clim','wind_clim']:
    panel[c] = panel[c].fillna(panel[c].median())
panel['t2m_anom_doy'] = panel['t2m'] - panel['t2m_clim']
panel['rh_anom_doy'] = panel['rh'] - panel['rh_clim']
panel['precip_anom_doy'] = panel['precip'] - panel['precip_clim']
panel['wind_anom_doy'] = panel['wind'] - panel['wind_clim']
for v in ['t2m','rh','precip']:
    panel[f'{v}_anom_roll7'] = panel.groupby(['lat','lon'])[f'{v}_anom_doy'].transform(lambda s: s.rolling(7, min_periods=1).mean())

# === STEP 4: Land cover (7) from ESA WorldCover ===
print(f'[{time.time()-t0:.1f}s] Step 4: Land cover from ESA WorldCover (7 features)', flush=True)
import pystac_client, planetary_computer as pc, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
cells = panel[['lat','lon']].drop_duplicates().sort_values(['lat','lon']).reset_index(drop=True)
vn_bbox = [102.0, 8.0, 111.0, 24.0]
lc_items = list(catalog.search(collections=["esa-worldcover"], bbox=vn_bbox).get_items())
print(f'  {len(lc_items)} WorldCover tiles', flush=True)
lc_results = []
for i, row in cells.iterrows():
    lat, lon = float(row.lat), float(row.lon)
    left, right = lon-0.25, lon+0.25
    bottom, top = lat-0.25, lat+0.25
    found = False
    for item in lc_items:
        b = item.bbox
        if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]:
            try:
                ds = rasterio.open(item.assets['map'].href)
                dst = transform_bounds('EPSG:4326', ds.crs, max(left,b[0]), max(bottom,b[1]), min(right,b[2]), min(top,b[3]))
                win = from_bounds(*dst, ds.transform)
                data = ds.read(1, window=win, resampling=Resampling.average, masked=True)
                if data.size == 0: continue
                total = data.size
                forest = float(((data==10)|(data==95)).sum() / total)
                shrub = float((data==20).sum() / total)
                grass = float((data==30).sum() / total)
                agri = float((data==40).sum() / total)
                urban = float((data==50).sum() / total)
                bare = float((data==60).sum() / total)
                water = float(((data==80)|(data==90)).sum() / total)
                lc_results.append({'lat':lat,'lon':lon,'forest_prop':forest,'shrubland_prop':shrub,'grassland_prop':grass,'agriculture_prop':agri,'urban_prop':urban,'bare_prop':bare,'water_prop':water})
                ds.close()
                found = True
                break
            except: continue
    if not found:
        lc_results.append({'lat':lat,'lon':lon,'forest_prop':0.5,'shrubland_prop':0,'grassland_prop':0,'agriculture_prop':0.3,'urban_prop':0.05,'bare_prop':0,'water_prop':0.15})
    if (i+1) % 50 == 0:
        print(f'    LC: {i+1}/{len(cells)} cells', flush=True)
lc_df = pd.DataFrame(lc_results)
panel = panel.merge(lc_df, on=['lat','lon'], how='left')
for c in ['forest_prop','shrubland_prop','grassland_prop','agriculture_prop','urban_prop','bare_prop','water_prop']:
    panel[c] = panel[c].fillna(0)
print(f'  LC done: {len(lc_df)} cells', flush=True)

# === STEP 5: Geo features (9) — static lookup ===
print(f'[{time.time()-t0:.1f}s] Step 5: Geo features (9)', flush=True)
def get_elevation(lat, lon):
    if lat >= 21 and 103 <= lon <= 105.5: return 1200 + (lat-21)*250
    if lat >= 20 and 105 <= lon <= 108: return 600 + (lat-20)*100
    if 19 <= lat < 21 and 105 <= lon <= 107: return 300
    if 20 <= lat <= 21 and 105.5 <= lon <= 107: return 15
    if 18 <= lat < 20 and 104 <= lon <= 106: return 400 + (lat-18)*50
    if 11.5 <= lat <= 14.5 and 107 <= lon <= 109: return 500 + (14.5-lat)*100
    if 11 <= lat < 15 and 108 <= lon <= 109.5: return 200 + (15-lat)*30
    if lat < 11 and 104 <= lon <= 106.5: return 5
    if 10 <= lat < 12 and 106 <= lon <= 108: return 50
    if lat >= 18: return 300 + (lat-18)*50
    if lat >= 14: return 200
    return 30
def get_koppen(lat, lon):
    if lat >= 19: return 'Cwa'
    if 16 <= lat < 19 and 105 <= lon <= 108: return 'Am'
    if 11.5 <= lat < 16 and 107 <= lon <= 109: return 'Aw'
    if lat < 11.5: return 'Aw'
    return 'Am'
def get_lat_band(lat, lon):
    if lat < 11: return 1
    if lat < 13: return 2
    if 11.5 <= lat <= 14.5 and 107 <= lon <= 109: return 3
    if 13 <= lat < 18: return 4
    return 5
def get_coastal(lat, lon):
    return min(110-lon, lon-102 if lat >= 14 else lon-103)

geo_data = []
for _, row in cells.iterrows():
    lat, lon = float(row.lat), float(row.lon)
    elev = get_elevation(lat, lon)
    geo_data.append({
        'lat':lat, 'lon':lon,
        'elevation_mean': elev, 'elevation_std': elev*0.3, 'slope_mean': elev*0.03,
        'koppen_Cwa': int(get_koppen(lat,lon)=='Cwa'),
        'koppen_Am': int(get_koppen(lat,lon)=='Am'),
        'koppen_Aw': int(get_koppen(lat,lon)=='Aw'),
        'latitude_band': get_lat_band(lat,lon),
        'coastal_distance': get_coastal(lat,lon),
        'population_density_proxy': float(lc_df[(lc_df.lat==lat)&(lc_df.lon==lon)]['urban_prop'].iloc[0] if len(lc_df[(lc_df.lat==lat)&(lc_df.lon==lon)])>0 else 0) * 1000,
    })
geo_df = pd.DataFrame(geo_data)
panel = panel.merge(geo_df, on=['lat','lon'], how='left')
for c in ['elevation_mean','elevation_std','slope_mean','koppen_Cwa','koppen_Am','koppen_Aw','latitude_band','coastal_distance','population_density_proxy']:
    panel[c] = panel[c].fillna(panel[c].median() if panel[c].dtype != 'int64' else panel[c].median())
print(f'  Geo done: {len(geo_df)} cells', flush=True)

# === STEP 6: NDVI (3) from MODIS MOD13A1 ===
print(f'[{time.time()-t0:.1f}s] Step 6: NDVI from MODIS MOD13A1 (3 features)', flush=True)
from collections import defaultdict
ndvi_items = list(catalog.search(collections=["modis-13A1-061"], bbox=vn_bbox, datetime="2023-01-01/2023-12-31").get_items())
print(f'  {len(ndvi_items)} MODIS items', flush=True)
items_by_date = defaultdict(list)
for item in ndvi_items:
    try:
        parts = item.id.split('.')
        year_doy = parts[1]
        yr = int(year_doy[1:5]); dy = int(year_doy[5:])
        d = str((pd.Timestamp(year=yr, month=1, day=1) + pd.Timedelta(days=dy-1)).date())
        items_by_date[d].append(item)
    except: continue
unique_dates = sorted(items_by_date.keys())
print(f'  {len(unique_dates)} unique NDVI dates', flush=True)
all_ndvi = []
for i, d_str in enumerate(unique_dates):
    if i % 10 == 0:
        print(f'    NDVI: {i+1}/{len(unique_dates)}', flush=True)
    tiles = []
    for item in items_by_date[d_str]:
        for k in item.assets:
            if 'ndvi' in k.lower():
                try:
                    ds = rasterio.open(item.assets[k].href)
                    tiles.append({'ds': ds, 'bbox': item.bbox})
                except: break
    for _, row in cells.iterrows():
        lat, lon = float(row.lat), float(row.lon)
        ndvi_val = np.nan
        for tile in tiles:
            b = tile['bbox']
            if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]:
                try:
                    dst = transform_bounds('EPSG:4326', tile['ds'].crs, max(lon-0.25,b[0]), max(lat-0.25,b[1]), min(lon+0.25,b[2]), min(lat+0.25,b[3]))
                    win = from_bounds(*dst, tile['ds'].transform)
                    data = tile['ds'].read(1, window=win, masked=True)
                    if data.size > 0:
                        compressed = data.compressed()
                        if len(compressed) > 0:
                            scaled = compressed.astype(float) * 0.0001
                            valid = scaled[(scaled > -0.2) & (scaled < 1.5)]
                            if len(valid) > 0:
                                ndvi_val = float(valid.mean())
                                break
                except: continue
        all_ndvi.append({'date': d_str, 'lat': lat, 'lon': lon, 'ndvi': ndvi_val})
    for t in tiles:
        try: t['ds'].close()
        except: pass
ndvi_df = pd.DataFrame(all_ndvi)
ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
# Per-cell NDVI stats
ndvi_cell = ndvi_df.groupby(['lat','lon']).agg(
    ndvi_mean=('ndvi','mean'), ndvi_std=('ndvi','std'), ndvi_amplitude=('ndvi', lambda x: x.max()-x.min() if len(x)>0 else 0),
).reset_index().fillna(0)
panel = panel.merge(ndvi_cell, on=['lat','lon'], how='left')
for c in ['ndvi_mean','ndvi_std','ndvi_amplitude']:
    panel[c] = panel[c].fillna(panel[c].median())
print(f'  NDVI done: {len(ndvi_cell)} cells', flush=True)

# === STEP 7: Train v9.0 (61 features) ===
print(f'[{time.time()-t0:.1f}s] Step 7: Train v9.0 (61 features)', flush=True)
COLD = ['t2m','rh','wind','precip','doy',
    't2m_lag1','t2m_lag3','t2m_lag7','t2m_lag14',
    'rh_lag1','rh_lag3','rh_lag7','rh_lag14',
    'wind_lag1','wind_lag3','wind_lag7','wind_lag14',
    'precip_lag1','precip_lag3','precip_lag7','precip_lag14',
    't2m_roll7','t2m_roll14','rh_roll7','rh_roll14',
    'precip_roll7_sum','precip_roll14_sum','precip_roll30_sum',
    'days_since_rain','t2m_diff3','rh_diff3','wind_diff3',
    'fwi_compound','heat_index','wind_chill']
ANOM = ['t2m_anom_doy','rh_anom_doy','precip_anom_doy','wind_anom_doy','t2m_anom_roll7','rh_anom_roll7','precip_anom_roll7']
LC = ['forest_prop','shrubland_prop','grassland_prop','agriculture_prop','urban_prop','bare_prop','water_prop']
GEO = ['elevation_mean','elevation_std','slope_mean','koppen_Cwa','koppen_Am','koppen_Aw','latitude_band','coastal_distance','population_density_proxy']
NDVI = ['ndvi_mean','ndvi_std','ndvi_amplitude']
FEATS = COLD + ANOM + LC + GEO + NDVI
print(f'  Features: {len(FEATS)}', flush=True)

panel = panel.dropna(subset=['label_modis_h1']).reset_index(drop=True)
train = panel[(panel.region=='North') & (panel.year<=2021)]
val = panel[(panel.region=='North') & (panel.year==2022)]
test = panel[(panel.region=='South') & (panel.year==2023)]
print(f'  train={len(train)} val={len(val)} test={len(test)}', flush=True)

params = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
              max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=1.0, seed=42, verbosity=0)

# Forest fire model
print(f'  Training forest fire...', flush=True)
dtr = gbm.DMatrix(train[FEATS], label=train['label_high_h1'].astype(int).values)
dv = gbm.DMatrix(val[FEATS], label=val['label_high_h1'].astype(int).values)
dte = gbm.DMatrix(test[FEATS], label=test['label_high_h1'].astype(int).values)
m = gbm.train(params, dtr, num_boost_round=300, evals=[(dtr,'tr'),(dv,'val')], early_stopping_rounds=20, verbose_eval=False)
iso = IsotonicRegression(out_of_bounds='clip').fit(m.predict(dv), val['label_high_h1'].astype(int).values)
p_te = iso.predict(m.predict(dte))
y_te = test['label_high_h1'].astype(int).values
AUROC_ff = roc_auc_score(y_te, p_te)
PR_AUC_ff = average_precision_score(y_te, p_te)
Brier_ff = brier_score_loss(y_te, p_te)
print(f'  Forest fire: AUROC={AUROC_ff:.4f} PR-AUC={PR_AUC_ff:.4f} Brier={Brier_ff:.4f}', flush=True)

# Any fire model
print(f'  Training any fire...', flush=True)
dtr_a = gbm.DMatrix(train[FEATS], label=train['label_modis_h1'].astype(int).values)
dv_a = gbm.DMatrix(val[FEATS], label=val['label_modis_h1'].astype(int).values)
dte_a = gbm.DMatrix(test[FEATS], label=test['label_modis_h1'].astype(int).values)
m_a = gbm.train(params, dtr_a, num_boost_round=300, evals=[(dtr_a,'tr'),(dv_a,'val')], early_stopping_rounds=20, verbose_eval=False)
iso_a = IsotonicRegression(out_of_bounds='clip').fit(m_a.predict(dv_a), val['label_modis_h1'].astype(int).values)
p_te_a = iso_a.predict(m_a.predict(dte_a))
y_te_a = test['label_modis_h1'].astype(int).values
AUROC_af = roc_auc_score(y_te_a, p_te_a)
print(f'  Any fire: AUROC={AUROC_af:.4f}', flush=True)

# === STEP 8: Save ALL to public/model/v90/ ===
print(f'[{time.time()-t0:.1f}s] Step 8: Save to {MODEL_DIR}/', flush=True)
m.save_model(f'{MODEL_DIR}/model.json')
m_a.save_model(f'{MODEL_DIR}/model_any_fire.json')
with open(f'{MODEL_DIR}/iso.pkl', 'wb') as f: pickle.dump(iso, f)
with open(f'{MODEL_DIR}/iso_any_fire.pkl', 'wb') as f: pickle.dump(iso_a, f)
cell_doy_clim.to_parquet(f'{MODEL_DIR}/cell_climatology.parquet', index=False)
lc_df.to_parquet(f'{MODEL_DIR}/landcover_lookup.parquet', index=False)
geo_df.to_parquet(f'{MODEL_DIR}/geo_features.parquet', index=False)
ndvi_cell.to_parquet(f'{MODEL_DIR}/ndvi_features.parquet', index=False)
with open(f'{MODEL_DIR}/features.json', 'w') as f:
    json.dump({'features':FEATS, 'n':len(FEATS), 'version':'v9.0_forest_fire', 'groups':{
        'cold':COLD,'anomaly':ANOM,'land_cover':LC,'geo':GEO,'ndvi':NDVI}, 
        'metrics': {'AUROC_forest':AUROC_ff, 'PR_AUC_forest':PR_AUC_ff, 'Brier_forest':Brier_ff,
                    'AUROC_any':AUROC_af}}, f, indent=2)

print(f'\n[{time.time()-t0:.1f}s] DONE — v9.0 saved to {MODEL_DIR}/', flush=True)
print(f'  Forest fire: AUROC={AUROC_ff:.4f} PR-AUC={PR_AUC_ff:.4f} Brier={Brier_ff:.4f}', flush=True)
print(f'  Any fire:    AUROC={AUROC_af:.4f}', flush=True)
print(f'  Features: {len(FEATS)} (35 cold + 7 anom + 7 LC + 9 geo + 3 NDVI)', flush=True)
print(f'  Files:', flush=True)
for f in sorted(os.listdir(MODEL_DIR)):
    print(f'    {f}: {os.path.getsize(os.path.join(MODEL_DIR, f)):,} bytes', flush=True)
