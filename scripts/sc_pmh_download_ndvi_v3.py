#!/usr/bin/env python3
"""
EFFICIENT NDVI download — for each unique date, merge all tiles that cover Vietnam.
"""
from __future__ import annotations
import json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
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

# Process 2023 first (fastest validation)
YEAR = 2023
print(f'\n=== Year {YEAR} ===', flush=True)
search = catalog.search(
    collections=["modis-13A1-061"],
    bbox=vn_bbox,
    datetime=f"{YEAR}-01-01/{YEAR}-12-31",
)
items = list(search.items())
print(f'items: {len(items)}', flush=True)

# Group items by date (multiple tiles per date)
items_by_date = {}
for item in items:
    if item.datetime is None:
        try:
            parts = item.id.split('.')
            year_doy = parts[1]
            year = int(year_doy[1:5])
            doy = int(year_doy[5:])
            d = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy-1)
            d_str = str(d.date())
        except:
            continue
    else:
        d_str = str(pd.to_datetime(item.datetime).date())
    items_by_date.setdefault(d_str, []).append(item)

unique_dates = sorted(items_by_date.keys())
print(f'unique dates: {len(unique_dates)}')
print(f'  items per date: {len(items_by_date[unique_dates[0]])}')

# For each date, read NDVI from the appropriate tile for each cell
all_records = []
for i, d_str in enumerate(unique_dates):
    if i % 5 == 0:
        print(f'  processing {i+1}/{len(unique_dates)}: {d_str}', flush=True)
    
    date_items = items_by_date[d_str]
    
    # Open all tiles' NDVI assets for this date
    tile_datasets = []
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
            tile_datasets.append((item, ds))
        except Exception as e:
            continue
    
    if len(tile_datasets) == 0:
        continue
    
    # For each cell, find tile that covers it
    for _, cell in cells.iterrows():
        lat, lon = float(cell.lat), float(cell.lon)
        left, right = lon - 0.25, lon + 0.25
        bottom, top = lat - 0.25, lat + 0.25
        
        ndvi_val = np.nan
        for item, ds in tile_datasets:
            tile_b = ds.bounds
            # Check if cell is in this tile
            if not (tile_b.left <= lon <= tile_b.right and tile_b.bottom <= lat <= tile_b.top):
                continue
            
            wleft = max(left, tile_b.left)
            wright = min(right, tile_b.right)
            wbottom = max(bottom, tile_b.bottom)
            wtop = min(top, tile_b.top)
            
            if wleft >= wright or wbottom >= wtop:
                continue
            
            try:
                win = from_bounds(wleft, wbottom, wright, wtop, ds.transform)
                data = ds.read(1, window=win, resampling=Resampling.average, masked=True)
                data = data.filled(np.nan)
                valid = data[~np.isnan(data) & (data > -0.5)]
                if len(valid) > 0:
                    ndvi_val = float(valid.mean()) * 0.0001
                    break  # found, no need to check other tiles
            except Exception:
                continue
        
        all_records.append({
            'date': d_str,
            'lat': lat, 'lon': lon,
            'ndvi': ndvi_val,
        })
    
    # Close datasets
    for _, ds in tile_datasets:
        ds.close()

ndvi_df = pd.DataFrame(all_records)
print(f'\ntotal records: {len(ndvi_df)}')
if len(ndvi_df) > 0:
    print(f'date range: {ndvi_df.date.min()} to {ndvi_df.date.max()}')
    print(f'cells covered: {ndvi_df[["lat","lon"]].drop_duplicates().shape[0]}')
    print(f'NaN rate: {100*ndvi_df.ndvi.isna().mean():.1f}%')
    print(f'NDVI stats: mean={ndvi_df.ndvi.mean():.3f} std={ndvi_df.ndvi.std():.3f} min={ndvi_df.ndvi.min():.3f} max={ndvi_df.ndvi.max():.3f}')
ndvi_df.to_parquet(OUT_P, index=False)
print(f'saved → {OUT_P}')
