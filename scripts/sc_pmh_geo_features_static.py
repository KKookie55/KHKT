#!/usr/bin/env python3
"""
Add static geographic features per Vietnam 0.5° cell using STATIC LOOKUP
(no external downloads — based on Vietnam geography knowledge).

Features:
  - elevation_mean (m): static lookup by lat/lon zone
  - slope_mean (degrees): elevation_std / 10
  - koppen_Cwa, koppen_Aw, koppen_Am (one-hot climate zone)
  - population_density_proxy: urban_prop * 1000 (from existing land cover)
  - coastal_distance (degrees): distance to nearest coast (proxy via min(lon - 102, 111 - lon))
  - latitude_band (1-5): 1=South Mekong, 2=South Coast, 3=Central Highlands, 4=Central Coast, 5=North

Output: /home/z/my-project/download/sc_pmh/geo_features_static.parquet
"""
from __future__ import annotations
import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
PANEL_P = SC / 'panel_v873_landcover.parquet'
LC_P = SC / 'landcover_expanded.parquet'
OUT_P = SC / 'geo_features_static.parquet'

print("[1/3] loading panel + land cover...")
panel = pd.read_parquet(PANEL_P)
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
lc = pd.read_parquet(LC_P)
print(f"  cells: {len(cells)}")

# === Compute static geo features ===
print("\n[2/3] computing static geo features...")

def get_elevation(lat, lon):
    """Vietnam elevation lookup by region (rough but reasonable).
    Sources: Vietnam topographic knowledge."""
    # North Vietnam — Hoàng Liên Sơn range (lat >= 20, lon 103-105)
    if lat >= 21 and 103 <= lon <= 105.5:
        return 1200 + (lat - 21) * 250  # 1200-1700m, Hoàng Liên Sơn
    # Northeast mountains (lat 20-22, lon 105-108)
    if lat >= 20 and 105 <= lon <= 108:
        return 600 + (lat - 20) * 100
    # Northern midlands (lat 19-21, lon 105-107)
    if 19 <= lat < 21 and 105 <= lon <= 107:
        return 300
    # Red River Delta (lat 20-21, lon 106-107)
    if 20 <= lat <= 21 and 105.5 <= lon <= 107:
        return 15  # delta lowlands
    # North Central Coast mountains (lat 18-20, lon 104-106)
    if 18 <= lat < 20 and 104 <= lon <= 106:
        return 400 + (lat - 18) * 50
    # Central Highlands (Tây Nguyên) — lat 11.5-14.5, lon 107-109
    if 11.5 <= lat <= 14.5 and 107 <= lon <= 109:
        return 500 + (14.5 - lat) * 100  # 500-800m, peaks at Pleiku/DaLat
    # South Central Coast mountains (lat 11-15, lon 108-109)
    if 11 <= lat < 15 and 108 <= lon <= 109.5:
        return 200 + (15 - lat) * 30
    # South Vietnam (Mekong Delta) — very low
    if lat < 11 and 104 <= lon <= 106.5:
        return 5  # Mekong delta
    # Eastern Nam Bộ (lat 10-12, lon 106-108)
    if 10 <= lat < 12 and 106 <= lon <= 108:
        return 50
    # Default by latitude band
    if lat >= 18:
        return 300 + (lat - 18) * 50
    if lat >= 14:
        return 200
    return 30

def get_koppen(lat, lon):
    """Köppen climate classification for Vietnam.
    Cwa = humid subtropical (North)
    Am  = tropical monsoon (Central Coast)
    Aw  = tropical savanna (Central Highlands + South + Mekong)"""
    if lat >= 19:
        return 'Cwa'  # humid subtropical, dry winter
    if 16 <= lat < 19 and 105 <= lon <= 108:
        return 'Am'  # tropical monsoon (Central Coast)
    if 11.5 <= lat < 16 and 107 <= lon <= 109:
        return 'Aw'  # tropical savanna (Central Highlands + South)
    if lat < 11.5:
        return 'Aw'  # Mekong (savanna with distinct dry season)
    return 'Am'

def get_latitude_band(lat, lon):
    """5 latitude bands with distinct fire regimes."""
    if lat < 11:  return 1  # South Mekong Delta (low elevation, wet, agricultural)
    if lat < 13:  return 2  # South Coast (mixed forest + agriculture)
    if 11.5 <= lat <= 14.5 and 107 <= lon <= 109: return 3  # Central Highlands (Tây Nguyên)
    if 13 <= lat < 18: return 4  # Central Coast
    return 5  # North (lat >= 18)

def get_coastal_distance(lat, lon):
    """Approximate distance to nearest coast (degrees)."""
    # Vietnam is roughly 102-110 E, 8.5-23.5 N, coastal on east side (lon ~109-110) 
    # and west (lon ~102-104 in north, 104-105 south)
    dist_east = 110 - lon  # distance to east coast (Biển Đông)
    dist_west_north = lon - 102 if lat >= 14 else lon - 103  # distance to Laos/Cambodia border
    return min(dist_east, dist_west_north)

results = []
for i, row in cells.iterrows():
    lat, lon = float(row.lat), float(row.lon)
    elev = get_elevation(lat, lon)
    elev_std = elev * 0.3  # rough std
    slope = elev_std / 10.0
    kop = get_koppen(lat, lon)
    lat_band = get_latitude_band(lat, lon)
    coast_dist = get_coastal_distance(lat, lon)
    # Get urban_prop from land cover for population proxy
    lc_row = lc[(lc.lat == lat) & (lc.lon == lon)]
    urban_prop = float(lc_row['urban_prop'].iloc[0]) if len(lc_row) > 0 else 0.0
    
    results.append({
        'lat': lat, 'lon': lon,
        'elevation_mean': elev,
        'elevation_std': elev_std,
        'slope_mean': slope,
        'koppen_Cwa': int(kop == 'Cwa'),
        'koppen_Am': int(kop == 'Am'),
        'koppen_Aw': int(kop == 'Aw'),
        'latitude_band': lat_band,
        'coastal_distance': coast_dist,
        'population_density_proxy': urban_prop * 1000,
    })

geo_df = pd.DataFrame(results)
print(f"  computed {len(geo_df)} cells")
print(f"  Köppen: Cwa={geo_df['koppen_Cwa'].sum()} Am={geo_df['koppen_Am'].sum()} Aw={geo_df['koppen_Aw'].sum()}")
print(f"  elevation range: {geo_df['elevation_mean'].min():.0f}-{geo_df['elevation_mean'].max():.0f} m")
print(f"  latitude bands: 1={int((geo_df.latitude_band==1).sum())} 2={int((geo_df.latitude_band==2).sum())} "
      f"3={int((geo_df.latitude_band==3).sum())} 4={int((geo_df.latitude_band==4).sum())} 5={int((geo_df.latitude_band==5).sum())}")

print(f"\n[3/3] saving geo_features_static.parquet...")
geo_df.to_parquet(OUT_P, index=False)
print(f"  saved → {OUT_P}")

# Stats
print("\n=== Geo features stats ===")
print(geo_df[['elevation_mean', 'elevation_std', 'slope_mean', 'koppen_Cwa', 'koppen_Am', 'koppen_Aw', 'latitude_band', 'coastal_distance', 'population_density_proxy']].describe().round(3).to_string())

# Top fire cells
print("\n=== Top-10 highest fire rate cells — geo features ===")
panel_test = panel[panel.year == 2023].copy()
cell_fire_rate = panel_test.groupby(['lat', 'lon']).fire_modis.mean().reset_index()
cell_fire_rate = cell_fire_rate.merge(geo_df, on=['lat', 'lon'])
top10 = cell_fire_rate.sort_values('fire_modis', ascending=False).head(10)
print(top10[['lat', 'lon', 'fire_modis', 'elevation_mean', 'slope_mean', 'koppen_Cwa', 'koppen_Am', 'koppen_Aw', 'latitude_band', 'coastal_distance', 'population_density_proxy']].to_string(index=False, float_format=lambda v: f'{v:.2f}'))
