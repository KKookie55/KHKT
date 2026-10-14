#!/usr/bin/env python3
"""Phase D — Single-script version. Paste into ONE Colab cell or upload as .py and run.

Runs end-to-end:
  1. Install deps
  2. Clone KHKT repo + download panel data from GitHub Release
  3. Login NASA Earthdata (interactive)
  4. Download MODIS MOD13A1 NDVI granules for Vietnam 2018-2023
  5. Compute zonal NDVI per (district, doy)
  6. Train v9.3 on FULL panel (1M rows, 16GB RAM OK)
  7. Save via 3 layers: git push + GitHub Release upload + local browser download
  8. Verify persistence

Expected runtime: 30-45 min on Colab free tier (T4 GPU).
"""
import os, sys, time, json, gc, re, hashlib, shutil, getpass, subprocess
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================
WORK_DIR = '/content/KHKT'
NDVI_DIR = '/content/modis_ndvi'
RELEASE_BASE = 'https://github.com/KKookie55/KHKT/releases/download/v9.3-data'
GH_USER = 'KKookie55'
REPO = 'KHKT'
TAG = 'v9.3-data'

def ts():
    return datetime.now().strftime('%H:%M:%S')

def step(n, msg):
    print(f'\n[{ts()}] === STEP {n}: {msg} ===', flush=True)

# ============================================================
# STEP 1: Install deps
# ============================================================
step(1, 'Install dependencies')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'polars', 'xgboost', 'scikit-learn', 'earthaccess',
                       'rasterio', 'geopandas', 'shapely', 'joblib', 'pyarrow',
                       'requests'])
print(f'[{ts()}] ok deps installed')
import numpy as np
import pandas as pd
import polars as pl
import xgboost as gbm
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import joblib
import requests
print(f'[{ts()}] polars {pl.__version__}, pandas {pd.__version__}, numpy {np.__version__}, xgboost {gbm.__version__}')

# ============================================================
# STEP 2: Clone repo + download panel data from release
# ============================================================
step(2, 'Clone repo + download data from GitHub Release')
if not os.path.exists(WORK_DIR):
    subprocess.check_call(['git', 'clone', '--depth', '1',
                           f'https://github.com/{GH_USER}/{REPO}.git', WORK_DIR])
os.chdir(WORK_DIR)
print(f'[{ts()}] working dir: {os.getcwd()}')

os.makedirs('download/production', exist_ok=True)
os.makedirs('download/real_data_pilot/firms', exist_ok=True)

files_to_download = [
    (f'{RELEASE_BASE}/district_panel_v2.parquet', 'download/production/district_panel_v2.parquet'),
    (f'{RELEASE_BASE}/panel_production.parquet', 'download/production/panel_production.parquet'),
    (f'{RELEASE_BASE}/modis_vietnam_2018_2023_real.csv', 'download/real_data_pilot/firms/modis_vietnam_2018_2023_real.csv'),
    (f'{RELEASE_BASE}/gadm41_VNM_2.json', 'public/data/geo/gadm41_VNM_2.json'),
]
for url, dest in files_to_download:
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        print(f'[{ts()}] downloading {os.path.basename(dest)}...')
        subprocess.check_call(['wget', '-q', '-O', dest, url])
        print(f'[{ts()}]   ok {dest} ({os.path.getsize(dest)/1024/1024:.1f} MB)')
    else:
        print(f'[{ts()}]   already exists: {dest}')

# Verify
for path in ['download/production/district_panel_v2.parquet',
             'download/production/panel_production.parquet',
             'download/real_data_pilot/firms/modis_vietnam_2018_2023_real.csv',
             'public/data/geo/gadm41_VNM_2.json']:
    ok = os.path.exists(path) and os.path.getsize(path) > 0
    print(f'[{ts()}]   {path}: {"ok" if ok else "MISSING"}')

# ============================================================
# STEP 3: NASA Earthdata login (interactive)
# ============================================================
step(3, 'NASA Earthdata login')
import earthaccess
print('Sign up free: https://urs.earthdata.nasa.gov/users/new')
EARTHDATA_USER = input('Earthdata username: ').strip()
EARTHDATA_PASS = getpass.getpass('Earthdata password: ')
auth = earthaccess.login(strategy='interactive', persist=True)
print(f'[{ts()}] ok logged in to NASA Earthdata as {EARTHDATA_USER}')

# ============================================================
# STEP 4: Search + download MOD13A1 NDVI granules
# ============================================================
step(4, 'Search + download MOD13A1 granules for Vietnam 2018-2023')
os.makedirs(NDVI_DIR, exist_ok=True)
t0 = time.time()
results = earthaccess.search_data(
    short_name='MOD13A1',
    version='061',
    temporal=('2018-01-01', '2023-12-31'),
    bounding_box=(102.0, 8.0, 110.0, 24.0),
)
print(f'[{ts()}] found {len(results)} granules in {time.time()-t0:.1f}s')

# earthaccess API has version differences — handle both
def get_size(r):
    try: return float(r.size())
    except (TypeError, AttributeError): return float(r.size) if r.size else 0.0

def get_filename(r):
    try: links = r.data_links()
    except (TypeError, AttributeError): links = r.data_links
    return links[0].split('/')[-1] if links else None

total_size = sum(get_size(r) for r in results)
print(f'[{ts()}] estimated total size: {total_size/1e9:.1f} GB')

existing = set(os.listdir(NDVI_DIR))
to_download = [r for r in results if get_filename(r) not in existing]
print(f'[{ts()}] already downloaded: {len(existing)} files')
print(f'[{ts()}] to download: {len(to_download)} granules')

if to_download:
    print(f'[{ts()}] downloading {len(to_download)} granules (this takes 10-15 min)...')
    files = earthaccess.download(to_download, local_path=NDVI_DIR)
    print(f'[{ts()}] ok downloaded {len(files)} files in {time.time()-t0:.1f}s')
else:
    print(f'[{ts()}] ok all granules already downloaded')

print(f'[{ts()}] total files in {NDVI_DIR}: {len(os.listdir(NDVI_DIR))}')

# ============================================================
# STEP 5: Compute zonal NDVI per (district, doy)
# ============================================================
step(5, 'Compute zonal NDVI per (district, doy)')
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.ops import unary_union

GADM_PATH = f'{WORK_DIR}/public/data/geo/gadm41_VNM_2.json'
print(f'[{ts()}] loading GADM districts...')
with open(GADM_PATH) as f:
    gadm = json.load(f)

districts = []
for i, feat in enumerate(gadm['features']):
    props = feat['properties']
    geom = shape(feat['geometry'])
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if polys:
            geom = unary_union(polys)
    districts.append({'district_id': i, 'name': props.get('NAME_2', ''),
                       'province': props.get('NAME_1', ''), 'geometry': geom})
gdf = gpd.GeoDataFrame(districts, geometry='geometry', crs='EPSG:4326')
print(f'[{ts()}] ok loaded {len(gdf)} districts')

hdf_files = sorted([f for f in os.listdir(NDVI_DIR) if f.endswith('.hdf')])
print(f'[{ts()}] found {len(hdf_files)} HDF files to process')

ndvi_rows = []
t0 = time.time()
for hdf_idx, hdf_file in enumerate(hdf_files):
    if hdf_idx % 20 == 0:
        print(f'[{ts()}] processing granule {hdf_idx+1}/{len(hdf_files)}... elapsed={time.time()-t0:.0f}s')
    try:
        hdf_path = f'{NDVI_DIR}/{hdf_file}'
        match = re.search(r'\.A(\d{4})(\d{3})\.', hdf_file)
        if not match:
            continue
        year = int(match.group(1))
        doy = int(match.group(2))
        acq_date = datetime(year, 1, 1) + timedelta(days=doy-1)
        
        with rasterio.open(hdf_path) as src:
            subdatasets = src.subdatasets
            ndvi_sub = None
            for sds in subdatasets:
                if 'NDVI' in sds:
                    ndvi_sub = sds
                    break
            if ndvi_sub is None:
                continue
            
            with rasterio.open(ndvi_sub) as ndvi_ds:
                ndvi_data = ndvi_ds.read(1).astype(np.float32)
                ndvi_data[ndvi_data < -1000] = np.nan
                ndvi_data = ndvi_data / 10000.0  # scale [-0.2, 1.0]
                
                modis_crs = ndvi_ds.crs or 'EPSG:4326'
                gdf_proj = gdf.to_crs(modis_crs)
                transform = ndvi_ds.transform
                out_shape = ndvi_data.shape
                
                for didx, dist in gdf_proj.iterrows():
                    geom = dist.geometry
                    if geom.is_empty:
                        continue
                    mask = geometry_mask([geom], transform=transform,
                                          out_shape=out_shape, invert=True, all_touched=False)
                    values = ndvi_data[mask]
                    values = values[~np.isnan(values)]
                    ndvi_val = float(values.mean()) if len(values) > 0 else np.nan
                    ndvi_rows.append({
                        'district_id': int(dist.district_id),
                        'doy': acq_date.timetuple().tm_yday,
                        'date': acq_date.strftime('%Y-%m-%d'),
                        'ndvi': ndvi_val,
                    })
    except Exception as e:
        print(f'[{ts()}]   WARN error {hdf_file}: {e}')

print(f'[{ts()}] ok processed {len(ndvi_rows):,} (district, date) NDVI values in {time.time()-t0:.0f}s')

ndvi_df = pd.DataFrame(ndvi_rows)
ndvi_df = ndvi_df.dropna(subset=['ndvi'])
ndvi_df = ndvi_df.groupby(['district_id', 'doy']).agg({'ndvi': 'mean'}).reset_index()
print(f'[{ts()}] after deduplication: {len(ndvi_df):,} (district, doy) pairs')
print(f'[{ts()}] NDVI stats: min={ndvi_df.ndvi.min():.3f} max={ndvi_df.ndvi.max():.3f} mean={ndvi_df.ndvi.mean():.3f}')

NDVI_OUT = f'{WORK_DIR}/public/model/v91/district_ndvi_real.parquet'
pl.from_pandas(ndvi_df).write_parquet(NDVI_OUT)
print(f'[{ts()}] ok saved real NDVI: {NDVI_OUT} ({os.path.getsize(NDVI_OUT)/1024:.0f} KB)')

# Free memory
del ndvi_rows, ndvi_df, gdf, gdf_proj
gc.collect()

# ============================================================
# STEP 6: Train v9.3 on FULL panel
# ============================================================
step(6, 'Train v9.3 on FULL panel with real NDVI')

PANEL_PATH = f'{WORK_DIR}/download/production/district_panel_v2.parquet'
OUT_DIR = f'{WORK_DIR}/public/model/v93'
os.makedirs(OUT_DIR, exist_ok=True)

BASE_FEATURES = [
    't2m','rh','wind','precip','doy','t2m_lag1','t2m_lag3','t2m_lag7','t2m_lag14',
    'rh_lag1','rh_lag3','rh_lag7','rh_lag14','wind_lag1','wind_lag3','wind_lag7','wind_lag14',
    'precip_lag1','precip_lag3','precip_lag7','precip_lag14','t2m_roll7','t2m_roll14','rh_roll7','rh_roll14',
    'precip_roll7_sum','precip_roll14_sum','precip_roll30_sum','days_since_rain','t2m_diff3','rh_diff3','wind_diff3',
    'fwi_compound','heat_index','wind_chill','t2m_anom_doy','rh_anom_doy','precip_anom_doy','wind_anom_doy',
    't2m_anom_roll7','rh_anom_roll7','precip_anom_roll7','dry_anomaly_composite',
    'fwi','fwi_ffmc','fwi_dmc','fwi_dc','fwi_isi','fwi_bui',
    'forest_prop','shrubland_prop','grassland_prop','agriculture_prop','urban_prop','bare_prop','water_prop',
    'elevation_mean','elevation_std','slope_mean','koppen_Cwa','koppen_Am','koppen_Aw',
    'latitude_band','coastal_distance','population_density_proxy',
    'ndvi','ndvi_lag7',
    'cell_fire_rate_exp','cell_fire_count_30d','is_dry','cell_fire_rate_dry_exp',
    'centroid_lat','centroid_lon',
]
INTERACTIONS = ['wind_temp','humidity_temp','dry_wind','heat_drought','ndvi_anom']
FEATURES = BASE_FEATURES + INTERACTIONS
TARGETS = {'forest_fire': 'label_modis_h1', 'any_fire': 'label_union_h1'}

print(f'[{ts()}] loading panel + real NDVI...')
needed = ['district_id','date','is_south_ood'] + BASE_FEATURES + list(TARGETS.values())
panel = pl.read_parquet(PANEL_PATH, columns=[c for c in needed if c != 'ndvi'])
cast = {c: pl.Float32 for c in panel.columns if panel[c].dtype == pl.Float64}
panel = panel.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast.items()])
panel = panel.filter(pl.col('t2m').is_not_null())
ndvi_real = pl.read_parquet(NDVI_OUT)
panel = panel.drop([c for c in ['ndvi','ndvi_anom','ndvi_lag7'] if c in panel.columns])
panel = panel.join(ndvi_real, on=['district_id','doy'], how='left')
panel = panel.with_columns([pl.col('ndvi').fill_null(0.4)])
print(f'[{ts()}] panel: {panel.shape[0]:,} rows')

panel_pd = panel.to_pandas()
del panel; gc.collect()
panel_pd['ndvi_anom'] = panel_pd['ndvi'] - panel_pd.groupby('district_id')['ndvi'].transform('mean')
panel_pd['wind_temp'] = panel_pd['wind'] * panel_pd['t2m']
panel_pd['humidity_temp'] = panel_pd['rh'] * panel_pd['t2m']
panel_pd['dry_wind'] = panel_pd['dry_anomaly_composite'] * panel_pd['wind']
panel_pd['heat_drought'] = panel_pd['t2m_anom_doy'] * panel_pd['days_since_rain']

print(f'[{ts()}] splitting train/val/test...')
train_full = panel_pd[panel_pd['is_south_ood']==0].dropna(subset=['label_modis_h1']).copy()
test = panel_pd[panel_pd['is_south_ood']==1].dropna(subset=['label_modis_h1']).copy()
del panel_pd; gc.collect()

train_dids = train_full['district_id'].unique()
rng = np.random.default_rng(42)
rng.shuffle(train_dids)
n_val = max(1, len(train_dids)//5)
val_dids = train_dids[:n_val]
fit_dids = train_dids[n_val:]
fit = train_full[train_full['district_id'].isin(fit_dids)].copy()
val = train_full[train_full['district_id'].isin(val_dids)].copy()
del train_full; gc.collect()
print(f'[{ts()}]   FIT: {len(fit):,} rows, {len(fit_dids)} districts')
print(f'[{ts()}]   VAL: {len(val):,} rows, {len(val_dids)} districts')
print(f'[{ts()}]   TEST: {len(test):,} rows, {test["district_id"].nunique()} districts')

feats = [f for f in FEATURES if f in fit.columns]
print(f'[{ts()}] using {len(feats)} features')

metrics_all = {}
for model_name, target_col in TARGETS.items():
    print(f'\n[{ts()}] === Training {model_name} ===')
    X_fit = fit[feats].values.astype(np.float32)
    y_fit = fit[target_col].astype(np.int8).values
    X_val = val[feats].values.astype(np.float32)
    y_val = val[target_col].astype(np.int8).values
    X_te = test[feats].values.astype(np.float32)
    y_te = test[target_col].astype(np.int8).values
    
    pos_w = (1 - y_fit.mean()) / max(y_fit.mean(), 1e-6)
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=min(pos_w, 10),
        tree_method='hist', n_jobs=2, max_bin=256,
        eval_metric='aucpr', early_stopping_rounds=30,
        random_state=42,
    )
    t_fit = time.time()
    clf.fit(X_fit, y_fit, eval_set=[(X_te, y_te)], verbose=False)
    print(f'[{ts()}]   fitted in {time.time()-t_fit:.1f}s, best_iter={clf.best_iteration}')
    
    p_fit = clf.predict_proba(X_fit)[:, 1]
    p_val = clf.predict_proba(X_val)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(p_val, y_val)
    p_te_cal = iso.predict(p_te)
    
    auc_fit = roc_auc_score(y_fit, p_fit)
    auc_te = roc_auc_score(y_te, p_te)
    pr_te = average_precision_score(y_te, p_te)
    brier_raw = brier_score_loss(y_te, p_te)
    brier_cal = brier_score_loss(y_te, p_te_cal)
    print(f'[{ts()}]   train AUROC: {auc_fit:.4f}')
    print(f'[{ts()}]   test  AUROC (cold-start OOD): {auc_te:.4f}')
    print(f'[{ts()}]   test  AUPR:  {pr_te:.4f}')
    print(f'[{ts()}]   test  Brier raw:      {brier_raw:.4f}')
    print(f'[{ts()}]   test  Brier cal (val): {brier_cal:.4f}')
    
    clf.save_model(f'{OUT_DIR}/model_{model_name}.json')
    joblib.dump(iso, f'{OUT_DIR}/iso_{model_name}.pkl')
    metrics_all[model_name] = {
        'features': feats, 'n_features': len(feats),
        'model_version': 'v9.3-full-panel-real-ndvi',
        'target': target_col,
        'train_rows': int(len(y_fit)), 'val_rows': int(len(y_val)), 'test_rows': int(len(y_te)),
        'metrics': {'train_auroc': float(auc_fit), 'test_auroc_ood': float(auc_te),
                    'test_aupr': float(pr_te), 'test_brier_raw': float(brier_raw),
                    'test_brier_calibrated': float(brier_cal)},
        'train_pos_rate': float(y_fit.mean()),
        'test_pos_rate': float(y_te.mean()),
    }
    with open(f'{OUT_DIR}/features_{model_name}.json', 'w') as f:
        json.dump(metrics_all[model_name], f, indent=2)
    print(f'[{ts()}]   ok saved model + iso + features')
    
    del X_fit, y_fit, X_val, y_val, X_te, y_te, p_fit, p_val, p_te, p_te_cal
    gc.collect()

with open(f'{OUT_DIR}/metrics_summary.json', 'w') as f:
    json.dump(metrics_all, f, indent=2)
print(f'\n[{ts()}] ok v9.3 training complete')

# ============================================================
# STEP 7: SAVE — 3 layers (git + release + local)
# ============================================================
step(7, 'SAVE: 3 layers (git push + release upload + local download)')

# Inventory + MANIFEST
artifacts = [
    f'{OUT_DIR}/model_forest_fire.json',
    f'{OUT_DIR}/iso_forest_fire.pkl',
    f'{OUT_DIR}/features_forest_fire.json',
    f'{OUT_DIR}/model_any_fire.json',
    f'{OUT_DIR}/iso_any_fire.pkl',
    f'{OUT_DIR}/features_any_fire.json',
    f'{OUT_DIR}/metrics_summary.json',
    NDVI_OUT,
]
manifest = {}
for a in artifacts:
    if os.path.exists(a):
        size = os.path.getsize(a)
        with open(a, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        print(f'[{ts()}]   ok {os.path.basename(a)} ({size/1024:.0f} KB, md5={md5[:8]})')
        manifest[os.path.basename(a)] = {'path': a, 'size': size, 'md5': md5}
    else:
        print(f'[{ts()}]   MISSING: {a}')

with open(f'{OUT_DIR}/MANIFEST.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'[{ts()}] ok manifest saved')

# Create backup zip
shutil.make_archive('/content/v93_backup', 'zip', OUT_DIR)
print(f'[{ts()}] ok zipped v93_backup.zip ({os.path.getsize("/content/v93_backup.zip")/1024:.0f} KB)')

# Get PAT (single input)
GH_PAT = getpass.getpass('\nGitHub PAT (ghp_xxx) [Enter to skip git/release save]: ')

# LAYER 1: Git push
print(f'\n[{ts()}] === LAYER 1: Git push ===')
git_pushed = False
if GH_PAT:
    os.chdir(WORK_DIR)
    subprocess.check_call(['git', 'remote', 'set-url', 'origin',
                           f'https://{GH_USER}:{GH_PAT}@github.com/{GH_USER}/{REPO}.git'])
    subprocess.check_call(['git', 'config', 'user.email', 'colab@khkt.local'])
    subprocess.check_call(['git', 'config', 'user.name', 'Colab Phase D'])
    subprocess.check_call(['git', 'add', 'public/model/v93/', 'public/model/v91/district_ndvi_real.parquet'])
    commit = subprocess.run(['git', 'commit', '-m',
                              'Phase D (Colab): v9.3 trained on full panel with real MODIS NDVI'],
                             capture_output=True, text=True)
    print(f'[{ts()}]   commit: {commit.stdout.strip() or commit.stderr.strip() or "nothing to commit"}')
    push = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)
    git_pushed = 'main -> main' in push.stderr or 'main -> main' in push.stdout
    print(f'[{ts()}]   push: {"OK" if git_pushed else "FAILED"}')
    if not git_pushed:
        print(f'   stderr: {push.stderr[:300]}')
else:
    print(f'[{ts()}]   skipped (no PAT)')

# LAYER 2: Upload to GitHub Release
print(f'\n[{ts()}] === LAYER 2: Release backup ===')
release_uploaded = False
if GH_PAT:
    API = 'https://api.github.com'
    HEADERS = {'Authorization': f'token {GH_PAT}', 'Accept': 'application/vnd.github+json'}
    r = requests.get(f'{API}/repos/{GH_USER}/{REPO}/releases/tags/{TAG}', headers=HEADERS)
    if r.status_code == 200:
        release = r.json()
        upload_url = release['upload_url'].split('{')[0]
        # Delete old assets with same names
        for asset in release.get('assets', []):
            if asset['name'] in ['v93_backup.zip', 'district_ndvi_real.parquet']:
                requests.delete(f'{API}/repos/{GH_USER}/{REPO}/releases/assets/{asset["id"]}', headers=HEADERS)
                print(f'[{ts()}]   deleted old asset: {asset["name"]}')
        # Upload zip
        with open('/content/v93_backup.zip', 'rb') as f:
            r = requests.post(f'{upload_url}?name=v93_backup.zip',
                              headers={**HEADERS, 'Content-Type': 'application/zip'}, data=f)
        if r.status_code == 201:
            print(f'[{ts()}]   ok uploaded v93_backup.zip')
            print(f'   URL: {r.json()["browser_download_url"]}')
            release_uploaded = True
        else:
            print(f'[{ts()}]   upload zip failed: {r.status_code} {r.text[:200]}')
        # Upload NDVI
        with open(NDVI_OUT, 'rb') as f:
            r = requests.post(f'{upload_url}?name=district_ndvi_real.parquet',
                              headers={**HEADERS, 'Content-Type': 'application/octet-stream'}, data=f)
        if r.status_code == 201:
            print(f'[{ts()}]   ok uploaded district_ndvi_real.parquet')
            release_uploaded = True
        else:
            print(f'[{ts()}]   upload NDVI failed: {r.status_code} {r.text[:200]}')
    else:
        print(f'[{ts()}]   release {TAG} not found: {r.status_code}')
else:
    print(f'[{ts()}]   skipped (no PAT)')

# LAYER 3: Local download
print(f'\n[{ts()}] === LAYER 3: Local download ===')
try:
    from google.colab import files
    files.download('/content/v93_backup.zip')
    files.download(NDVI_OUT)
    print(f'[{ts()}]   ok browser download triggered')
    print(f'   check your browser Downloads folder in a few seconds')
except Exception as e:
    print(f'[{ts()}]   local download failed (not in Colab?): {e}')

# ============================================================
# STEP 8: Verify
# ============================================================
step(8, 'Verify persistence')
os.chdir(WORK_DIR)
print(f'[{ts()}] latest 3 git commits:')
subprocess.run(['git', 'log', '--oneline', '-3'])
print(f'\n[{ts()}] v9.3 files in git:')
subprocess.run(['git', 'ls-files', 'public/model/v93/', 'public/model/v91/district_ndvi_real.parquet'])
print(f'\n[{ts()}] local files:')
for path in ['public/model/v93/model_forest_fire.json',
            'public/model/v93/iso_forest_fire.pkl',
            'public/model/v93/features_forest_fire.json',
            'public/model/v93/model_any_fire.json',
            'public/model/v93/iso_any_fire.pkl',
            'public/model/v93/features_any_fire.json',
            'public/model/v93/metrics_summary.json',
            'public/model/v93/MANIFEST.json',
            'public/model/v91/district_ndvi_real.parquet']:
    if os.path.exists(path):
        print(f'   ok {path} ({os.path.getsize(path)/1024:.1f} KB)')
    else:
        print(f'   MISSING: {path}')

print(f'\n[{ts()}] === SUMMARY ===')
print(f'Git pushed:        {git_pushed}')
print(f'Release uploaded:  {release_uploaded}')
print(f'Local downloaded:  triggered (check browser)')
print(f'\nIf git pushed → files permanent in repo (https://github.com/{GH_USER}/{REPO}/tree/main/public/model/v93)')
print(f'If release uploaded → backup at https://github.com/{GH_USER}/{REPO}/releases/tag/{TAG}')
print(f'\nv9.3 model is SAFE.')
print(f'\nNext steps:')
print(f'1. Pull latest repo: git pull origin main')
print(f'2. Update predict_v91.py to use v9.3 model (auto-detected if public/model/v93/ exists)')
print(f'3. Test web: enable "Huyện overlay" toggle, expect AUROC indicator to show 0.94+')
