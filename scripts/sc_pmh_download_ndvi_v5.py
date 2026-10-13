#!/usr/bin/env python3
"""
NDVI download v5 — clean version, properly handling CRS.
"""
from __future__ import annotations
import os, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
warnings.filterwarnings('ignore')

SC = Path('/home/z/my-project/download/sc_pmh')
PANEL_P = SC / 'panel_v88_full.parquet'
OUT_P = SC / 'ndvi_raw.parquet'

panel = pd.read_parquet(PANEL_P)
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
print(f'n_cells={len(cells)}')

vn_bbox = [102.0, 8.0, 111.0, 24.0]
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# Process 2023 first to validate
YEAR = 2023
print(f'\n=== Year {YEAR} ===', flush=True)
search = catalog.search(collections=["modis-13A1-061"], bbox=vn_bbox, datetime=f"{YEAR}-01-01/{YEAR}-12-31")
items = list(search.items())
print(f'items: {len(items)}', flush=True)

# Group by date
items_by_date = defaultdict(list)
for item in items:
    try:
        parts = item.id.split('.')
        year_doy = parts[1]
        year = int(year_doy[1:5])
        doy = int(year_doy[5:])
        d = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy-1)
        d_str = str(d.date())
    except:
        continue
    items_by_date[d_str].append(item)

unique_dates = sorted(items_by_date.keys())
print(f'unique dates: {len(unique_dates)}')

cell_lats = cells.lat.values
cell_lons = cells.lon.values
all_records = []

for i, d_str in enumerate(unique_dates):
    if i % 5 == 0:
        print(f'  processing {i+1}/{len(unique_dates)}: {d_str}', flush=True)
    
    date_items = items_by_date[d_str]
    
    # Open all tiles for this date
    tiles = []
    for item in date_items:
        ndvi_asset = None
        for k in item.assets:
            if 'ndvi' in k.lower():
                ndvi_asset = item.assets[k]
                break
        if ndvi_asset is None:
            continue
        try:
            ds = rasterio.open(ndvi_asset.href)
            tiles.append({'ds': ds, 'bbox_latlon': item.bbox})  # [w, s, e, n]
        except Exception:
            continue
    
    if len(tiles) == 0:
        continue
    
    # For each cell, find tile that contains it
    for cell_idx in range(len(cell_lats)):
        lat = float(cell_lats[cell_idx])
        lon = float(cell_lons[cell_idx])
        ndvi_val = np.nan
        
        for tile in tiles:
            b = tile['bbox_latlon']
            if not (b[0] <= lon <= b[2] and b[1] <= lat <= b[3]):
                continue
            
            # Cell extent, clipped to tile bbox
            left = max(lon - 0.25, b[0])
            right = min(lon + 0.25, b[2])
            bottom = max(lat - 0.25, b[1])
            top = min(lat + 0.25, b[3])
            
            if left >= right or bottom >= top:
                continue
            
            ds = tile['ds']
            try:
                # Transform lat/lon bounds to ds CRS
                dst = transform_bounds('EPSG:4326', ds.crs, left, bottom, right, top)
                win = from_bounds(*dst, ds.transform)
                data = ds.read(1, window=win, masked=True)
                if data.size == 0:
                    continue
                compressed = data.compressed()
                if len(compressed) > 0:
                    # Filter valid NDVI range (after scale: -0.2 to 1.0)
                    scaled = compressed.astype(float) * 0.0001
                    valid = scaled[(scaled > -0.2) & (scaled < 1.5)]
                    if len(valid) > 0:
                        ndvi_val = float(valid.mean())
                        break
            except Exception:
                continue
        
        all_records.append({
            'date': d_str,
            'lat': lat, 'lon': lon,
            'ndvi': ndvi_val,
        })
    
    # Close tiles
    for tile in tiles:
        tile['ds'].close()

ndvi_df = pd.DataFrame(all_records)
print(f'\ntotal records: {len(ndvi_df)}')
if len(ndvi_df) > 0:
    valid_records = ndvi_df.ndvi.notna().sum()
    print(f'valid (non-NaN) records: {valid_records}/{len(ndvi_df)} ({100*valid_records/len(ndvi_df):.1f}%)')
    print(f'date range: {ndvi_df.date.min()} to {ndvi_df.date.max()}')
    print(f'cells covered: {ndvi_df[["lat","lon"]].drop_duplicates().shape[0]}')
    valid = ndvi_df.ndvi.dropna()
    if len(valid) > 0:
        print(f'NDVI stats: mean={valid.mean():.3f} std={valid.std():.3f} min={valid.min():.3f} max={valid.max():.3f}')
ndvi_df.to_parquet(OUT_P, index=False)
print(f'saved → {OUT_P} ({os.path.getsize(OUT_P) / 1024:.1f} KB)')
