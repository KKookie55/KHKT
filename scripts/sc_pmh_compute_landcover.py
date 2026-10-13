#!/usr/bin/env python3
"""
Compute land-cover features for Vietnam 0.5° grid using ALL ESA WorldCover 2021 tiles.

For each cell:
  - Find the tile(s) that cover (lat, lon)
  - Open that tile, read window for cell extent (0.5° × 0.5° centered on lat/lon)
  - Compute class proportions: forest, agriculture, urban, water, other
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
PANEL_P = SC / 'panel_with_interactions.parquet'
OUT_P = SC / 'landcover_features.parquet'

print("[1/4] loading panel to get cell list...")
panel = pd.read_parquet(PANEL_P)
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
print(f"  n_cells={len(cells)} lat=[{cells.lat.min()}, {cells.lat.max()}] lon=[{cells.lon.min()}, {cells.lon.max()}]")

# === Step 2: Search ALL ESA WorldCover tiles covering Vietnam ===
print("\n[2/4] searching all ESA WorldCover tiles covering Vietnam...")
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
# Vietnam bbox with padding
vn_bbox = [102.0, 8.0, 111.0, 22.0]
search = catalog.search(collections=["esa-worldcover"], bbox=vn_bbox)
items = list(search.get_items())
print(f"  found {len(items)} tiles")

# Open each tile as a rasterio dataset, store in dict keyed by item id
print(f"\n  opening {len(items)} tiles...")
tiles = {}  # item_id -> (rasterio dataset, bbox)
for item in items:
    asset = item.assets['map']
    href = asset.href
    ds = rasterio.open(href)
    # Compute tile bbox from item's geometry
    geom = item.geometry  # [min_lon, min_lat, max_lon, max_lat]
    min_lon, min_lat, max_lon, max_lat = geom['coordinates'][0][0][0], geom['coordinates'][0][0][1], geom['coordinates'][0][2][0], geom['coordinates'][0][2][1]
    # Actually let's use bbox
    bbox_item = item.bbox  # [min_lon, min_lat, max_lon, max_lat]
    tiles[item.id] = {'ds': ds, 'bbox': bbox_item}
    print(f"    {item.id}: bbox={bbox_item}")

# === Step 3: For each cell, find covering tile + compute proportions ===
print("\n[3/4] computing land cover proportions for each cell...")

# ESA WorldCover 2021 v200 class codes:
# 10 = tree cover, 20 = shrubland, 30 = grassland, 40 = cropland,
# 50 = built-up, 60 = bare/sparse, 70 = snow/ice, 80 = water bodies,
# 90 = herbaceous wetland, 95 = mangroves, 100 = moss/lichen

results = []
for i, row in cells.iterrows():
    lat, lon = float(row.lat), float(row.lon)
    # Cell extent: 0.5° × 0.5° centered on (lat, lon)
    left, right = lon - 0.25, lon + 0.25
    bottom, top = lat - 0.25, lat + 0.25
    
    # Find tile that contains this cell
    cell_center_lon = lon
    cell_center_lat = lat
    matching_tile = None
    for tid, info in tiles.items():
        b = info['bbox']
        if (b[0] <= cell_center_lon <= b[2] and b[1] <= cell_center_lat <= b[3]):
            matching_tile = info
            break
    
    if matching_tile is None:
        # Try by extent overlap (cells may straddle tile boundary — pick any tile that overlaps)
        for tid, info in tiles.items():
            b = info['bbox']
            if (b[0] <= right and b[2] >= left and b[1] <= top and b[3] >= bottom):
                matching_tile = info
                break
    
    if matching_tile is None:
        print(f"  WARNING: no tile found for cell ({lat}, {lon})")
        results.append({'lat': lat, 'lon': lon,
                        'forest_prop': np.nan, 'agriculture_prop': np.nan,
                        'urban_prop': np.nan, 'water_prop': np.nan,
                        'other_prop': np.nan, 'n_pixels': 0, 'tile': 'none'})
        continue
    
    try:
        ds = matching_tile['ds']
        # Clamp window bounds to tile extent
        tile_b = matching_tile['bbox']
        win_left = max(left, tile_b[0])
        win_right = min(right, tile_b[2])
        win_bottom = max(bottom, tile_b[1])
        win_top = min(top, tile_b[3])
        
        if win_left >= win_right or win_bottom >= win_top:
            results.append({'lat': lat, 'lon': lon,
                            'forest_prop': np.nan, 'agriculture_prop': np.nan,
                            'urban_prop': np.nan, 'water_prop': np.nan,
                            'other_prop': np.nan, 'n_pixels': 0, 'tile': 'out_of_bounds'})
            continue
        
        win = from_bounds(win_left, win_bottom, win_right, win_top, ds.transform)
        data = ds.read(1, window=win, resampling=Resampling.nearest)
        
        if data.size == 0:
            results.append({'lat': lat, 'lon': lon,
                            'forest_prop': np.nan, 'agriculture_prop': np.nan,
                            'urban_prop': np.nan, 'water_prop': np.nan,
                            'other_prop': np.nan, 'n_pixels': 0, 'tile': 'empty'})
            continue
        
        total = data.size
        forest = ((data == 10) | (data == 95)).sum()
        agriculture = (data == 40).sum()
        urban = (data == 50).sum()
        water = ((data == 80) | (data == 90)).sum()
        other = total - forest - agriculture - urban - water
        
        results.append({
            'lat': lat, 'lon': lon,
            'forest_prop': float(forest / total),
            'agriculture_prop': float(agriculture / total),
            'urban_prop': float(urban / total),
            'water_prop': float(water / total),
            'other_prop': float(other / total),
            'n_pixels': int(total),
            'tile': 'ok',
        })
        
        if (i + 1) % 50 == 0:
            print(f"    processed {i+1}/{len(cells)} cells")
    except Exception as e:
        print(f"  ERROR cell ({lat}, {lon}): {e}")
        results.append({'lat': lat, 'lon': lon,
                        'forest_prop': np.nan, 'agriculture_prop': np.nan,
                        'urban_prop': np.nan, 'water_prop': np.nan,
                        'other_prop': np.nan, 'n_pixels': 0, 'tile': f'error: {e}'})

lc_df = pd.DataFrame(results)
print(f"\n  done — {len(lc_df)} cells processed")
print(f"  cells with valid data: {lc_df['forest_prop'].notna().sum()}")
print(f"  cells with NaN: {lc_df['forest_prop'].isna().sum()}")
print(f"  tile status:")
print(lc_df['tile'].value_counts().to_string())

# Close all datasets
for info in tiles.values():
    info['ds'].close()

# === Step 4: Save ===
print("\n[4/4] saving landcover_features.parquet...")
lc_df.to_parquet(OUT_P, index=False)
print(f"  saved → {OUT_P}")

# Summary
print("\n=== Land cover stats (cells with valid data only) ===")
print(lc_df[lc_df.forest_prop.notna()][['forest_prop', 'agriculture_prop', 'urban_prop', 'water_prop', 'other_prop']].describe().round(3).to_string())

# Top fire cells vs land cover
panel_test = panel[panel.year == 2023].copy()
panel_test['region'] = np.where(panel_test.lat >= 14, 'North', 'South')
cell_fire_rate = panel_test.groupby(['lat', 'lon']).fire_modis.mean().reset_index()
cell_fire_rate = cell_fire_rate.merge(lc_df, on=['lat', 'lon'])

print("\n=== Top-10 highest fire rate cells in South ===")
south_top = cell_fire_rate[cell_fire_rate.lat < 14].sort_values('fire_modis', ascending=False).head(10)
print(south_top[['lat', 'lon', 'fire_modis', 'forest_prop', 'agriculture_prop', 'urban_prop', 'water_prop']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

print("\n=== Top-10 highest fire rate cells in North ===")
north_top = cell_fire_rate[cell_fire_rate.lat >= 18].sort_values('fire_modis', ascending=False).head(10)
print(north_top[['lat', 'lon', 'fire_modis', 'forest_prop', 'agriculture_prop', 'urban_prop', 'water_prop']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
