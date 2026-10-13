#!/usr/bin/env python3
"""
Download MODIS MOD13A1 NDVI 16-day composites for Vietnam 2018-2023.

MOD13A1 is 500m resolution, 16-day NDVI composite.
Available on Microsoft Planetary Computer as collection 'modis-13A1-061'.

Output: /home/z/my-project/download/sc_pmh/ndvi_raw.parquet
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
SC = DL / 'sc_pmh'
PANEL_P = SC / 'panel_v88_full.parquet'
OUT_P = SC / 'ndvi_raw.parquet'

print("[1/4] loading panel...")
panel = pd.read_parquet(PANEL_P)
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
print(f"  n_cells={len(cells)}")

# Vietnam bbox
vn_bbox = [102.0, 8.0, 111.0, 24.0]

print("\n[2/4] connecting to Planetary Computer...")
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# Test search
test_search = catalog.search(
    collections=["modis-13A1-061"],
    bbox=vn_bbox,
    datetime="2023-06-01/2023-06-30",
)
test_items = list(test_search.items())
print(f"  test (Jun 2023): {len(test_items)} items")
if test_items:
    print(f"  example item: {test_items[0].id}")
    print(f"  assets: {list(test_items[0].assets.keys())[:5]}")
    # Find NDVI asset
    for k in test_items[0].assets:
        print(f"    asset '{k}': {test_items[0].assets[k].title}")

# === Step 3: Iterate by year ===
print("\n[3/4] downloading NDVI per year...")
all_ndvi_data = []

for year in [2018, 2019, 2020, 2021, 2022, 2023]:
    print(f"\n  === Year {year} ===", flush=True)
    year_search = catalog.search(
        collections=["modis-13A1-061"],
        bbox=vn_bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(year_search.items())
    print(f"    items: {len(items)}", flush=True)
    
    if len(items) == 0:
        continue
    
    items.sort(key=lambda x: x.datetime or pd.Timestamp("1900-01-01"))
    
    for i, item in enumerate(items):
        if i % 20 == 0:
            print(f"    processing {i+1}/{len(items)}: {item.id[:50]}", flush=True)
        
        # Find NDVI asset
        ndvi_asset = None
        for k in item.assets:
            if 'ndvi' in k.lower():
                ndvi_asset = item.assets[k]
                break
        if ndvi_asset is None:
            continue
        
        # Use item ID to extract date if datetime is None
        if item.datetime is None:
            # MYD13A1.A2018361.h29v08.061... → 2018-361
            try:
                parts = item.id.split('.')
                year_doy = parts[1]  # e.g. A2018361
                year = int(year_doy[1:5])
                doy = int(year_doy[5:])
                item_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy-1)
                item_date = item_date.date()
            except Exception:
                continue
        else:
            item_date = pd.to_datetime(item.datetime).date()
        
        try:
            with rasterio.open(ndvi_asset.href) as ds:
                for _, cell in cells.iterrows():
                    lat, lon = float(cell.lat), float(cell.lon)
                    left, right = lon - 0.25, lon + 0.25
                    bottom, top = lat - 0.25, lat + 0.25
                    
                    try:
                        win = from_bounds(left, bottom, right, top, ds.transform)
                        data = ds.read(1, window=win, resampling=Resampling.average, masked=True)
                        data = data.filled(np.nan)
                        valid = data[~np.isnan(data) & (data > -0.5)]
                        if len(valid) == 0:
                            ndvi_val = np.nan
                        else:
                            ndvi_val = float(valid.mean()) * 0.0001
                        all_ndvi_data.append({
                            'date': str(item_date),
                            'lat': lat, 'lon': lon,
                            'ndvi': ndvi_val,
                        })
                    except Exception:
                        all_ndvi_data.append({
                            'date': str(item_date),
                            'lat': lat, 'lon': lon,
                            'ndvi': np.nan,
                        })
        except Exception as e:
            print(f"    ERROR opening {item.id}: {e}")
            continue

print(f"\n[4/4] saving NDVI data...")
ndvi_df = pd.DataFrame(all_ndvi_data)
print(f"  total records: {len(ndvi_df)}")
if len(ndvi_df) > 0:
    print(f"  date range: {ndvi_df.date.min()} to {ndvi_df.date.max()}")
    print(f"  cells covered: {ndvi_df[['lat','lon']].drop_duplicates().shape[0]}")
    print(f"  NaN rate: {100*ndvi_df.ndvi.isna().mean():.1f}%")
    print(f"  NDVI stats: mean={ndvi_df.ndvi.mean():.3f}, min={ndvi_df.ndvi.min():.3f}, max={ndvi_df.ndvi.max():.3f}")
ndvi_df.to_parquet(OUT_P, index=False)
print(f"  saved → {OUT_P} ({os.path.getsize(OUT_P) / 1024 / 1024:.1f} MB)")
