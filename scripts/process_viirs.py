#!/usr/bin/env python3
"""
Process uploaded VIIRS data:
1. Load each year's CSV
2. Filter to MODIS bbox (so apples-to-apples comparison)
3. Keep MODIS-compatible columns
4. Save as viirs_fires_{year}.csv in project data dir
"""
import os, sys, json, shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Source: extracted VIIRS CSVs
SRC_BASE = Path("/home/z/my-project/upload/viirs_extracted")
# Destination: project data dir (next to MODIS fires_*.csv)
DST = Path("/home/z/my-project/upload/wildfire_data/wildfire_data/viirs")
DST.mkdir(parents=True, exist_ok=True)

# Vietnam bbox (matches MODIS query in v8.5)
WEST, SOUTH, EAST, NORTH = 104.0, 11.0, 109.5, 21.0
YEARS = list(range(2018, 2024))

# MODIS-compatible columns to keep
KEEP_COLS = ["latitude", "longitude", "acq_date", "satellite", "confidence"]

print("=" * 70)
print("PROCESSING VIIRS DATA (filter to MODIS bbox + standardize format)")
print("=" * 70)
print(f"  Source: {SRC_BASE}")
print(f"  Dest:   {DST}")
print(f"  BBox:   {WEST},{SOUTH} → {EAST},{NORTH}")

total_raw = 0
total_filtered = 0
all_data = {}

for year in YEARS:
    print(f"\n=== {year} ===")
    # Find CSV in source folder
    src_csvs = list((SRC_BASE / str(year)).glob("fire_archive_SV-C2_*.csv"))
    if not src_csvs:
        print(f"  ✗ No VIIRS CSV found for {year}")
        continue
    src_csv = src_csvs[0]
    print(f"  Loading: {src_csv.name}")
    df = pd.read_csv(src_csv)
    print(f"  Raw rows: {len(df):,}")
    total_raw += len(df)

    # Check column names
    print(f"  Columns: {list(df.columns)}")
    
    # Filter to MODIS bbox
    mask = (df["longitude"] >= WEST) & (df["longitude"] <= EAST) & \
           (df["latitude"]  >= SOUTH) & (df["latitude"]  <= NORTH)
    df_vn = df[mask].copy()
    print(f"  After bbox filter: {len(df_vn):,}")
    total_filtered += len(df_vn)

    # Keep MODIS-compatible columns
    keep = [c for c in KEEP_COLS if c in df_vn.columns]
    df_vn = df_vn[keep].copy()
    
    # Save to project data dir
    out_file = DST / f"viirs_fires_{year}.csv"
    df_vn.to_csv(out_file, index=False)
    print(f"  ✓ Saved: {out_file} ({out_file.stat().st_size/1e6:.1f} MB)")
    all_data[year] = df_vn

# Combined file
if all_data:
    df_all = pd.concat([all_data[y] for y in sorted(all_data)], ignore_index=True)
    df_all.to_csv(DST / "viirs_fires_ALL.csv", index=False)
    print(f"\n✓ Combined: {DST}/viirs_fires_ALL.csv ({len(df_all):,} rows)")

# Comparison with MODIS
print(f"\n{'=' * 70}")
print("VIIRS vs MODIS DETECTION COUNT (after bbox filter)")
print("=" * 70)
print(f"\n  {'Year':<6} | {'MODIS':<10} | {'VIIRS':<10} | {'VIIRS/MODIS':<12}")
print(f"  {'-'*50}")
modis_dir = Path("/home/z/my-project/upload/wildfire_data/wildfire_data/firms")
for year in YEARS:
    modis_file = modis_dir / f"fires_{year}_daily.csv"
    viirs_file = DST / f"viirs_fires_{year}.csv"
    if modis_file.exists() and viirs_file.exists():
        n_modis = len(pd.read_csv(modis_file))
        n_viirs = len(pd.read_csv(viirs_file))
        ratio = n_viirs / max(n_modis, 1)
        print(f"  {year:<6} | {n_modis:<10,} | {n_viirs:<10,} | {ratio:<12.2f}")

print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"  Total raw VIIRS detections (2018-2023): {total_raw:,}")
print(f"  After MODIS bbox filter:                {total_filtered:,}")
print(f"  Reduction: {100*(1-total_filtered/total_raw):.1f}% (fires outside bbox removed)")
print(f"\n  Files saved in: {DST}")
print(f"  - viirs_fires_2018.csv through viirs_fires_2023.csv")
print(f"  - viirs_fires_ALL.csv (combined)")

print(f"\nNext: build VIIRS validation script (compare MODIS labels vs VIIRS ground truth)")
