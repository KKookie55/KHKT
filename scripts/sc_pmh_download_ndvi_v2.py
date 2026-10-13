#!/usr/bin/env python3
"""
EFFICIENT NDVI download — read entire Vietnam bbox at once per item, not per cell.
"""
from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = Path('/home/z/my-project/download/sc_pmh')
PANEL_P = SC / 'panel_v88_full.parquet'
OUT_P = SC / 'ndvi_raw.parquet'

# Load cells
panel = pd.read_parquet(PANEL_P)
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
print(f'n_cells={len(cells)}')

# Vietnam bbox
vn_bbox = [102.0, 8.0, 111.0, 24.0]

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# Process only 2023 first (validation), then expand if time
YEAR = 2023
print(f'\n=== Year {YEAR} ===', flush=True)
search = catalog.search(
    collections=["modis-13A1-061"],
    bbox=vn_bbox,
    datetime=f"{YEAR}-01-01/{YEAR}-12-31",
)
items = list(search.items())
print(f'items: {len(items)}', flush=True)

# Dedupe by date — keep first item per date (avoid Terra+Aqua duplicates)
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
    if d_str not in items_by_date:
        items_by_date[d_str] = item
unique_dates = sorted(items_by_date.keys())
print(f'unique dates: {len(unique_dates)}')

# For each unique date, read NDVI for all 273 cells efficiently
all_records = []
for i, d_str in enumerate(unique_dates):
    item = items_by_date[d_str]
    if i % 10 == 0:
        print(f'  processing {i+1}/{len(unique_dates)}: {d_str}', flush=True)
    
    ndvi_asset = None
    for k in item.assets:
        if 'ndvi' in k.lower():
            ndvi_asset = item.assets[k]
            break
    if ndvi_asset is None:
        continue
    
    try:
        with rasterio.open(ndvi_asset.href) as ds:
            # For each cell, read its 0.5° window
            for _, cell in cells.iterrows():
                lat, lon = float(cell.lat), float(cell.lon)
                left, right = lon - 0.25, lon + 0.25
                bottom, top = lat - 0.25, lat + 0.25
                
                # Clip to tile extent
                tile_bounds = ds.bounds
                if right < tile_bounds.left or left > tile_bounds.right or top < tile_bounds.bottom or bottom > tile_bounds.top:
                    continue
                
                wleft = max(left, tile_bounds.left)
                wright = min(right, tile_bounds.right)
                wbottom = max(bottom, tile_bounds.bottom)
                wtop = min(top, tile_bounds.top)
                
                if wleft >= wright or wbottom >= wtop:
                    continue
                
                try:
                    win = from_bounds(wleft, wbottom, wright, wtop, ds.transform)
                    data = ds.read(1, window=win, resampling=Resampling.average, masked=True)
                    data = data.filled(np.nan)
                    valid = data[~np.isnan(data) & (data > -0.5)]
                    if len(valid) > 0:
                        ndvi_val = float(valid.mean()) * 0.0001
                    else:
                        ndvi_val = np.nan
                    all_records.append({
                        'date': d_str,
                        'lat': lat, 'lon': lon,
                        'ndvi': ndvi_val,
                    })
                except Exception:
                    pass
    except Exception as e:
        print(f'    ERROR: {e}')
        continue

ndvi_df = pd.DataFrame(all_records)
print(f'\ntotal records: {len(ndvi_df)}')
if len(ndvi_df) > 0:
    print(f'date range: {ndvi_df.date.min()} to {ndvi_df.date.max()}')
    print(f'cells covered: {ndvi_df[["lat","lon"]].drop_duplicates().shape[0]}')
    print(f'NaN rate: {100*ndvi_df.ndvi.isna().mean():.1f}%')
    print(f'NDVI stats: mean={ndvi_df.ndvi.mean():.3f} std={ndvi_df.ndvi.std():.3f} min={ndvi_df.ndvi.min():.3f} max={ndvi_df.ndvi.max():.3f}')
ndvi_df.to_parquet(OUT_P, index=False)
print(f'saved → {OUT_P}')
