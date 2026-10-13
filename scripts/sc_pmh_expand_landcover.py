#!/usr/bin/env python3
"""
Compute EXPANDED land-cover features (7 categories) for Vietnam 0.5° grid.

ESA WorldCover 2021 v200 classes:
  10 = tree cover (forest)
  20 = shrubland
  30 = grassland
  40 = cropland (agriculture)
  50 = built-up (urban)
  60 = bare/sparse vegetation
  80 = water bodies
  90 = herbaceous wetland
  95 = mangroves

Aggregate to 7 proportions per cell:
  forest_prop        = (10 + 95) — includes mangroves as coastal forest
  shrubland_prop     = 20
  grassland_prop     = 30
  agriculture_prop   = 40
  urban_prop         = 50
  bare_prop          = 60 — bare/sparse, important for fire risk (dry grass, no canopy)
  water_prop         = 80 + 90

Output: /home/z/my-project/download/sc_pmh/landcover_expanded.parquet
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
PANEL_P = SC / 'panel_v873_landcover.parquet'
OUT_P = SC / 'landcover_expanded.parquet'

print("[1/4] loading panel to get cell list...")
panel = pd.read_parquet(PANEL_P)
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
print(f"  n_cells={len(cells)}")

print("\n[2/4] searching all ESA WorldCover tiles covering Vietnam...")
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
vn_bbox = [102.0, 8.0, 111.0, 22.0]
search = catalog.search(collections=["esa-worldcover"], bbox=vn_bbox)
items = list(search.get_items())
print(f"  found {len(items)} tiles")

# Open all tiles
print(f"  opening {len(items)} tiles...")
tiles = {}
for item in items:
    asset = item.assets['map']
    ds = rasterio.open(asset.href)
    tiles[item.id] = {'ds': ds, 'bbox': item.bbox}

# Compute per-cell proportions for 7 categories
print("\n[3/4] computing expanded land cover proportions per cell...")
results = []
for i, row in cells.iterrows():
    lat, lon = float(row.lat), float(row.lon)
    left, right = lon - 0.25, lon + 0.25
    bottom, top = lat - 0.25, lat + 0.25
    
    # Find matching tile by cell center
    matching_tile = None
    for tid, info in tiles.items():
        b = info['bbox']
        if (b[0] <= lon <= b[2] and b[1] <= lat <= b[3]):
            matching_tile = info
            break
    
    # Try overlap if center not in any tile
    if matching_tile is None:
        for tid, info in tiles.items():
            b = info['bbox']
            if (b[0] <= right and b[2] >= left and b[1] <= top and b[3] >= bottom):
                matching_tile = info
                break
    
    if matching_tile is None:
        results.append({'lat': lat, 'lon': lon,
                        'forest_prop': np.nan, 'shrubland_prop': np.nan,
                        'grassland_prop': np.nan, 'agriculture_prop': np.nan,
                        'urban_prop': np.nan, 'bare_prop': np.nan, 'water_prop': np.nan,
                        'n_pixels': 0, 'tile': 'none'})
        continue
    
    try:
        ds = matching_tile['ds']
        tile_b = matching_tile['bbox']
        win_left = max(left, tile_b[0])
        win_right = min(right, tile_b[2])
        win_bottom = max(bottom, tile_b[1])
        win_top = min(top, tile_b[3])
        
        if win_left >= win_right or win_bottom >= win_top:
            results.append({'lat': lat, 'lon': lon,
                            'forest_prop': np.nan, 'shrubland_prop': np.nan,
                            'grassland_prop': np.nan, 'agriculture_prop': np.nan,
                            'urban_prop': np.nan, 'bare_prop': np.nan, 'water_prop': np.nan,
                            'n_pixels': 0, 'tile': 'out_of_bounds'})
            continue
        
        win = from_bounds(win_left, win_bottom, win_right, win_top, ds.transform)
        data = ds.read(1, window=win, resampling=Resampling.nearest)
        
        if data.size == 0:
            results.append({'lat': lat, 'lon': lon,
                            'forest_prop': np.nan, 'shrubland_prop': np.nan,
                            'grassland_prop': np.nan, 'agriculture_prop': np.nan,
                            'urban_prop': np.nan, 'bare_prop': np.nan, 'water_prop': np.nan,
                            'n_pixels': 0, 'tile': 'empty'})
            continue
        
        total = data.size
        forest = ((data == 10) | (data == 95)).sum()
        shrubland = (data == 20).sum()
        grassland = (data == 30).sum()
        agriculture = (data == 40).sum()
        urban = (data == 50).sum()
        bare = (data == 60).sum()
        water = ((data == 80) | (data == 90)).sum()
        
        results.append({
            'lat': lat, 'lon': lon,
            'forest_prop': float(forest / total),
            'shrubland_prop': float(shrubland / total),
            'grassland_prop': float(grassland / total),
            'agriculture_prop': float(agriculture / total),
            'urban_prop': float(urban / total),
            'bare_prop': float(bare / total),
            'water_prop': float(water / total),
            'n_pixels': int(total),
            'tile': 'ok',
        })
        
        if (i + 1) % 50 == 0:
            print(f"    processed {i+1}/{len(cells)} cells")
    except Exception as e:
        print(f"  ERROR cell ({lat}, {lon}): {e}")
        results.append({'lat': lat, 'lon': lon,
                        'forest_prop': np.nan, 'shrubland_prop': np.nan,
                        'grassland_prop': np.nan, 'agriculture_prop': np.nan,
                        'urban_prop': np.nan, 'bare_prop': np.nan, 'water_prop': np.nan,
                        'n_pixels': 0, 'tile': f'error: {e}'})

lc_df = pd.DataFrame(results)
print(f"\n  done — {len(lc_df)} cells, valid: {lc_df['forest_prop'].notna().sum()}")

for info in tiles.values():
    info['ds'].close()

# Save
print("\n[4/4] saving landcover_expanded.parquet...")
lc_df.to_parquet(OUT_P, index=False)
print(f"  saved → {OUT_P}")

# Summary stats
print("\n=== Land cover stats (7 categories) ===")
print(lc_df[['forest_prop', 'shrubland_prop', 'grassland_prop', 'agriculture_prop',
            'urban_prop', 'bare_prop', 'water_prop']].describe().round(3).to_string())

# Compare top fire cells vs low fire cells
panel_test = panel[panel.year == 2023].copy()
panel_test['region'] = np.where(panel_test.lat >= 14, 'North', 'South')
cell_fire_rate = panel_test.groupby(['lat', 'lon']).fire_modis.mean().reset_index()
cell_fire_rate = cell_fire_rate.merge(lc_df, on=['lat', 'lon'])

print("\n=== Top-10 highest fire rate cells (whole Vietnam) ===")
top10 = cell_fire_rate.sort_values('fire_modis', ascending=False).head(10)
print(top10[['lat', 'lon', 'fire_modis', 'forest_prop', 'shrubland_prop', 'grassland_prop',
             'agriculture_prop', 'urban_prop', 'bare_prop', 'water_prop']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

print("\n=== Bottom-10 fire rate cells (whole Vietnam) ===")
bot10 = cell_fire_rate.sort_values('fire_modis', ascending=True).head(10)
print(bot10[['lat', 'lon', 'fire_modis', 'forest_prop', 'shrubland_prop', 'grassland_prop',
             'agriculture_prop', 'urban_prop', 'bare_prop', 'water_prop']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# Regional breakdown
print("\n=== Land cover by region ===")
for region in ['North', 'Central', 'South']:
    if region == 'North':
        sub = lc_df[lc_df.lat >= 18]
    elif region == 'Central':
        sub = lc_df[(lc_df.lat >= 14) & (lc_df.lat < 18)]
    else:
        sub = lc_df[lc_df.lat < 14]
    print(f"\n  {region} (n={len(sub)} cells):")
    print(f"    forest={sub.forest_prop.mean():.3f}  shrubland={sub.shrubland_prop.mean():.3f}  grassland={sub.grassland_prop.mean():.3f}")
    print(f"    agriculture={sub.agriculture_prop.mean():.3f}  urban={sub.urban_prop.mean():.3f}  bare={sub.bare_prop.mean():.3f}  water={sub.water_prop.mean():.3f}")
