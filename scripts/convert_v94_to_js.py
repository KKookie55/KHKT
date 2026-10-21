"""Convert v9.4 model + lookup tables to JS-compatible JSON for Netlify.

Outputs:
  public/model/v94/model_forest_fire_compact.json  — XGBoost trees in JS-traversable format
  public/model/v94/model_any_fire_compact.json
  public/model/v94/iso_forest_fire.json             — isotonic regression x/y thresholds
  public/model/v94/iso_any_fire.json
  public/model/v94/district_lookup.json             — static features per district (710)
  public/model/v94/district_climatology.json        — per (district, doy) climatology
  public/model/v94/district_ndvi.json               — real MODIS NDVI per (district, doy)
  public/model/v94/district_cell_fire_clim.json      — cell_fire_rate climatology per (district, doy)
"""
import os, json, time
import xgboost as gbm
import joblib
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

MODEL_DIR = 'public/model/v94'
LOOKUP_DIR = 'public/model/v91'
OUT_DIR = 'public/model/v94'

def convert_xgboost_to_js(model_path, out_path):
    """Convert XGBoost Booster to JS-compatible JSON (tree array)."""
    model = gbm.Booster()
    model.load_model(model_path)
    
    # Get dump as JSON — get_dump returns list of strings (one per tree)
    dump_str = model.get_dump(dump_format='json')
    # Each element is a JSON string for one tree — parse all
    trees = [json.loads(t) for t in dump_str]
    
    # Compact format: each tree is a nested dict with {node, split, threshold, yes, no, missing, leaf, children}
    # Already in good format — just save it
    with open(out_path, 'w') as f:
        json.dump(trees, f, separators=(',', ':'))
    
    print(f'  ✓ {out_path} ({os.path.getsize(out_path)/1024:.0f} KB, {len(trees)} trees)')
    return trees

def convert_isotonic_to_js(iso_path, out_path):
    """Convert isotonic regression to JS-compatible JSON."""
    iso = joblib.load(iso_path)
    data = {
        'x_thresholds': iso.x_thresholds_.tolist() if hasattr(iso, 'x_thresholds_') else [],
        'y_thresholds': iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else [],
        'x_min': float(iso.x_min) if hasattr(iso, 'x_min') else 0,
        'x_max': float(iso.x_max) if hasattr(iso, 'x_max') else 1,
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f'  ✓ {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)')

def convert_parquet_to_json(parquet_path, out_path, max_rows=None):
    """Convert parquet to JSON. If max_rows, subsample."""
    df = pd.read_parquet(parquet_path)
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    # Convert to list of dicts
    records = df.to_dict('records')
    with open(out_path, 'w') as f:
        json.dump(records, f, separators=(',', ':'), default=str)
    print(f'  ✓ {out_path} ({os.path.getsize(out_path)/1024:.0f} KB, {len(records)} rows)')

def main():
    t0 = time.time()
    print('=== Convert v9.4 model + lookups to JS JSON ===\n')
    
    # 1. XGBoost models
    print('[1] Convert XGBoost models:')
    convert_xgboost_to_js(f'{MODEL_DIR}/model_forest_fire.json', f'{OUT_DIR}/model_forest_fire_compact.json')
    convert_xgboost_to_js(f'{MODEL_DIR}/model_any_fire.json', f'{OUT_DIR}/model_any_fire_compact.json')
    
    # 2. Isotonic calibrators
    print('\n[2] Convert isotonic calibrators:')
    convert_isotonic_to_js(f'{MODEL_DIR}/iso_forest_fire.pkl', f'{OUT_DIR}/iso_forest_fire.json')
    convert_isotonic_to_js(f'{MODEL_DIR}/iso_any_fire.pkl', f'{OUT_DIR}/iso_any_fire.json')
    
    # 3. Feature list
    print('\n[3] Feature list:')
    with open(f'{MODEL_DIR}/features_forest_fire.json') as f:
        feats = json.load(f)
    print(f'  forest_fire: {feats["n_features"]} features')
    with open(f'{MODEL_DIR}/features_any_fire.json') as f:
        feats_any = json.load(f)
    print(f'  any_fire: {feats_any["n_features"]} features')
    
    # 4. District features (static per district)
    print('\n[4] District static features:')
    convert_parquet_to_json(f'{LOOKUP_DIR}/district_features.parquet', f'{OUT_DIR}/district_lookup.json')
    
    # 5. District climatology (per district × doy) — might be large (259K rows)
    print('\n[5] District climatology:')
    clim = pd.read_parquet(f'{LOOKUP_DIR}/district_climatology.parquet')
    # Convert to dict: {district_id: {doy: {t2m_clim, rh_clim, precip_clim, wind_clim}}}
    clim_dict = {}
    for _, row in clim.iterrows():
        did = int(row['district_id'])
        doy = int(row['doy'])
        if did not in clim_dict:
            clim_dict[did] = {}
        clim_dict[did][doy] = {
            't2m_clim': round(float(row['t2m_clim']), 2),
            'rh_clim': round(float(row['rh_clim']), 2),
            'precip_clim': round(float(row['precip_clim']), 3),
            'wind_clim': round(float(row['wind_clim']), 2),
        }
    with open(f'{OUT_DIR}/district_climatology.json', 'w') as f:
        json.dump(clim_dict, f, separators=(',', ':'))
    print(f'  ✓ district_climatology.json ({os.path.getsize(f"{OUT_DIR}/district_climatology.json")/1024:.0f} KB, {len(clim_dict)} districts)')
    
    # 6. Real NDVI
    print('\n[6] Real MODIS NDVI:')
    ndvi_path = f'{LOOKUP_DIR}/district_ndvi_real.parquet'
    if os.path.exists(ndvi_path):
        ndvi = pd.read_parquet(ndvi_path)
        ndvi_dict = {}
        for _, row in ndvi.iterrows():
            did = int(row['district_id'])
            doy = int(row['doy'])
            if did not in ndvi_dict:
                ndvi_dict[did] = {}
            ndvi_dict[did][doy] = round(float(row['ndvi']), 4)
        with open(f'{OUT_DIR}/district_ndvi.json', 'w') as f:
            json.dump(ndvi_dict, f, separators=(',', ':'))
        print(f'  ✓ district_ndvi.json ({os.path.getsize(f"{OUT_DIR}/district_ndvi.json")/1024:.0f} KB, {len(ndvi_dict)} districts)')
    else:
        print('  ⚠ district_ndvi_real.parquet not found, skipping')
    
    # 7. Cell fire rate climatology
    print('\n[7] Cell fire rate climatology:')
    cfclim_path = f'{LOOKUP_DIR}/district_cell_fire_rate_clim.parquet'
    if os.path.exists(cfclim_path):
        cfc = pd.read_parquet(cfclim_path)
        cfc_dict = {}
        for _, row in cfc.iterrows():
            did = int(row['district_id'])
            doy = int(row['doy'])
            if did not in cfc_dict:
                cfc_dict[did] = {}
            cfc_dict[did][doy] = {
                'cell_fire_rate_exp': round(float(row['cell_fire_rate_exp']), 4),
                'cell_fire_count_30d': round(float(row['cell_fire_count_30d']), 2),
                'is_dry': int(row['is_dry']),
                'cell_fire_rate_dry_exp': round(float(row['cell_fire_rate_dry_exp']), 4),
            }
        with open(f'{OUT_DIR}/district_cell_fire_clim.json', 'w') as f:
            json.dump(cfc_dict, f, separators=(',', ':'))
        print(f'  ✓ district_cell_fire_clim.json ({os.path.getsize(f"{OUT_DIR}/district_cell_fire_clim.json")/1024:.0f} KB, {len(cfc_dict)} districts)')
    else:
        print('  ⚠ district_cell_fire_rate_clim.parquet not found, skipping')
    
    # Summary
    print(f'\n=== Summary ===')
    total_size = 0
    for f in os.listdir(OUT_DIR):
        if f.endswith('.json'):
            sz = os.path.getsize(f'{OUT_DIR}/{f}')
            total_size += sz
            print(f'  {f}: {sz/1024:.0f} KB')
    print(f'  Total: {total_size/1024:.0f} KB ({total_size/1024/1024:.1f} MB)')
    print(f'  Elapsed: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
