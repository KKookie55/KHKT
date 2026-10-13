#!/usr/bin/env python3
"""
EFFICIENT NDVI download v4 — use item.bbox (lat/lon) for tile-cell matching.
For each date, open all tiles and for each cell pick the tile whose bbox contains it.
"""
from __future__ import annotations
import os, time, warnings
from pathlib import Path
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

YEAR = 2023
print(f'\n=== Year {YEAR} ===', flush=True)
search = catalog.search(
    collections=["modis-13A1-061"],
    bbox=vn_bbox,
    datetime=f"{YEAR}-01-01/{YEAR}-12-31",
)
items = list(search.items())
print(f'items: {len(items)}', flush=True)

# Group by date — multiple tiles per date (each tile covers different region)
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

# Pre-compute cell lat/lon arrays for speed
cell_lats = cells.lat.values
cell_lons = cells.lon.values

all_records = []
for i, d_str in enumerate(unique_dates):
    if i % 5 == 0:
        print(f'  processing {i+1}/{len(unique_dates)}: {d_str}', flush=True)
    
    date_items = items_by_date[d_str]
    
    # For each tile (item), open + cache its bbox and dataset
    tile_cache = []
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
            # item.bbox is [west, south, east, north] in lat/lon
            tile_bbox_latlon = item.bbox
            tile_cache.append({'ds': ds, 'bbox_latlon': tile_bbox_lon_latlon} if False else
                              {'ds': ds, 'bbox_latlon': tile_bbox_latlon})
        except Exception:
            continue
    
    if len(tile_cache) == 0:
        continue
    
    # For each cell, find tile that contains it
    for cell_idx, (lat, lon) in enumerate(zip(cell_lats, cell_lons)):
        ndvi_val = np.nan
        for tile in tile_cache:
            bbox = tile['bbox_latlon']  # [west, south, east, north]
            if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
                # This tile covers this cell. Read window.
                # Convert cell lat/lon bounds to tile CRS
                left, right = lon - 0.25, lon + 0.25
                bottom, top = lat - 0.25, lat + 0.25
                # Clip to tile bbox
                w_left = max(left, bbox[0])
                w_right = min(right, bbox[2])
                w_bottom = max(bottom, bbox[1])
                w_top = min(top, bbox[3])
                
                if w_left >= w_right or w_bottom >= w_top:
                    continue
                
                ds = tile['ds']
                # Transform lat/lon bounds to ds CRS
                try:
                    # transform_bounds(src_crs='EPSG:4326', dst_crs=ds.crs, ...)
                    dst_bounds = transform_bounds('EPSG:4326', ds.crs, w_left, w_bottom, w_right, w_top, densify_pts=21)
                    win = from_bounds(*dst_bounds, ds.transform)
                    data = ds.read(1, window=win, resampling=Resampling.average, masked=True)
                    if data.size == 0:
                        continue
                    data = data.filled(np.nan)
                    # Filter: NDVI nodata = -3000 (scaled = -0.3); valid range -0.2 to 1.0
                    valid = data[~np.isnan(data) & (data > -0.25) & (data < 1.5)]
                    if len(valid) > 0:
                        ndvi_val = float(valid.mean()) * 0.0001
                        break  # found
                except Exception as e:
                    continue
        
        all_records.append({
            'date': d_str,
            'lat': float(lat), 'lon': float(lon),
            'ndvi': ndvi_val,
        })
    
    # Close datasets
    for tile in tile_cache:
        tile['ds'].close()

ndvi_df = pd.DataFrame(all_records)
print(f'\ntotal records: {len(ndvi_df)}')
if len(ndvi_df) > 0:
    print(f'date range: {ndvi_df.date.min()} to {ndvi_df.date.max()}')
    print(f'cells covered: {ndvi_df[["lat","lon"]].drop_duplicates().shape[0]}')
    print(f'NaN rate: {100*ndvi_df.ndvi.isna().mean():.1f}%')
    valid = ndvi_df.ndvi.dropna()
    if len(valid) > 0:
        print(f'NDVI stats: mean={valid.mean():.3f} std={valid.std():.3f} min={valid.min():.3f} max={valid.max():.3f}')
ndvi_df.to_parquet(OUT_P, index=False)
print(f'saved → {OUT_P}')
