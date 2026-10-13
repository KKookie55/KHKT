#!/usr/bin/env python3
"""
Download MODIS MaxFRP from MOD14A1 (Terra) + MYD14A1 (Aqua) for Vietnam 2018-2023.

Output: /home/z/my-project/download/sc_pmh/fire_with_frp.parquet
  Columns: lat, lon, date, frp_max (max FRP per cell-day, MW), n_detections
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
OUT_P = SC / 'fire_with_frp.parquet'

panel = pd.read_parquet(PANEL_P)
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
print(f'n_cells={len(cells)}')

vn_bbox = [102.0, 8.0, 111.0, 24.0]
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# Helper: extract date from item.id
def extract_date(item):
    if item.datetime is not None:
        return str(pd.to_datetime(item.datetime).date())
    try:
        parts = item.id.split('.')
        year_doy = parts[1]
        year = int(year_doy[1:5])
        doy = int(year_doy[5:])
        d = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy-1)
        return str(d.date())
    except:
        return None

# Get the per-cell window in tile CRS for each (cell, tile) pair
def get_cell_window(lat, lon, ds, tile_bbox):
    left = max(lon - 0.25, tile_bbox[0])
    right = min(lon + 0.25, tile_bbox[2])
    bottom = max(lat - 0.25, tile_bbox[1])
    top = min(lat + 0.25, tile_bbox[3])
    if left >= right or bottom >= top:
        return None
    try:
        dst = transform_bounds('EPSG:4326', ds.crs, left, bottom, right, top)
        win = from_bounds(*dst, ds.transform)
        return win
    except:
        return None

# Process one year at a time, both Terra (MOD14A1) + Aqua (MYD14A1)
all_records = []
for year in [2018, 2019, 2020, 2021, 2022, 2023]:
    print(f'\n=== Year {year} ===', flush=True)
    
    for collection in ['modis-14A1-061', 'modis-14A2-061']:
        print(f'  collection {collection}...', flush=True)
        search = catalog.search(collections=[collection], bbox=vn_bbox, datetime=f'{year}-01-01/{year}-12-31')
        items = list(search.items())
        print(f'    items: {len(items)}', flush=True)
        
        # Group by date — multiple tiles per date
        by_date = defaultdict(list)
        for item in items:
            d_str = extract_date(item)
            if d_str is None:
                continue
            by_date[d_str].append(item)
        
        unique_dates = sorted(by_date.keys())
        print(f'    unique dates: {len(unique_dates)}', flush=True)
        
        # Process each date
        for i, d_str in enumerate(unique_dates):
            if i % 30 == 0:
                print(f'    processing {i+1}/{len(unique_dates)}: {d_str} ({collection})', flush=True)
            
            date_items = by_date[d_str]
            
            # Open all tiles' MaxFRP assets
            tiles = []
            for item in date_items:
                if 'MaxFRP' not in item.assets:
                    continue
                try:
                    ds = rasterio.open(item.assets['MaxFRP'].href)
                    tiles.append({'ds': ds, 'bbox_latlon': item.bbox})
                except:
                    continue
            
            if len(tiles) == 0:
                continue
            
            # For each cell, find tile + read MaxFRP
            for _, cell in cells.iterrows():
                lat, lon = float(cell.lat), float(cell.lon)
                frp_max = 0.0  # 0 means no fire detected
                n_det = 0
                
                for tile in tiles:
                    b = tile['bbox_latlon']
                    if not (b[0] <= lon <= b[2] and b[1] <= lat <= b[3]):
                        continue
                    win = get_cell_window(lat, lon, tile['ds'], b)
                    if win is None:
                        continue
                    try:
                        data = tile['ds'].read(1, window=win, masked=True)
                        if data.size == 0:
                            continue
                        # MaxFRP values: 0 = no fire, >0 = max FRP in MW
                        compressed = data.compressed()
                        if len(compressed) > 0:
                            valid = compressed[compressed > 0]
                            if len(valid) > 0:
                                if valid.max() > frp_max:
                                    frp_max = float(valid.max())
                                n_det += len(valid)
                    except:
                        continue
                
                all_records.append({
                    'date': d_str,
                    'lat': lat, 'lon': lon,
                    'frp_max': frp_max,
                    'n_detections': n_det,
                })
            
            # Close tiles
            for tile in tiles:
                tile['ds'].close()

# Save
print(f'\nsaving {OUT_P}...')
df = pd.DataFrame(all_records)
print(f'total records: {len(df)}')
if len(df) > 0:
    print(f'date range: {df.date.min()} to {df.date.max()}')
    print(f'cells: {df[["lat","lon"]].drop_duplicates().shape[0]}')
    fires = df[df.frp_max > 0]
    print(f'fires (frp > 0): {len(fires)} ({100*len(fires)/len(df):.2f}%)')
    print(f'frp stats (fires only):')
    print(fires.frp_max.describe().round(2).to_string())
df.to_parquet(OUT_P, index=False)
print(f'saved → {OUT_P} ({os.path.getsize(OUT_P) / 1024 / 1024:.1f} MB)')
