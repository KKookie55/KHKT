#!/usr/bin/env python3
"""
Colab script: Download VIIRS active fire data for Vietnam, 2018-2023.

VIIRS (Suomi-NPP / NOAA-20) detects active fires at 375m resolution — 
finer than MODIS (1km). Used as INDEPENDENT VALIDATION for our MODIS model.

Source: NASA FIRMS (Fire Information for Resource Management System)
API: https://firms.modaps.eosdis.nasa.gov/api/

Setup (one-time, free):
  1. Register at https://firms.modaps.eosdis.nasa.gov/api/area/
     (you'll receive an API key by email immediately)
  2. Paste your API key below
  3. Run this script in Google Colab

Output: 6 CSV files (one per year) saved to Google Drive:
  /content/drive/MyDrive/wildfire_data/viirs/viirs_fires_{year}.csv

Each row: latitude, longitude, acq_date, satellite, confidence
(matches our MODIS CSV format exactly, so re-running preprocess_v85.py
with VIIRS validation enabled is trivial)

Run in Colab:
  !pip install -q requests tqdm pandas
  from google.colab import drive
  drive.mount('/content/drive')
  # paste this whole script into a cell
"""

import os, sys, time, json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# CONFIG — fill in your FIRMS API key (free)
# ============================================================
FIRMS_API_KEY = "YOUR_FIRMS_API_KEY_HERE"  # ← paste your key

# Vietnam bounding box (matches our MODIS query)
WEST, SOUTH, EAST, NORTH = 104.0, 11.0, 109.5, 21.0
YEARS = list(range(2018, 2024))

# VIIRS sources on FIRMS:
#   VIIRS_SNPP_NRT  — Suomi-NPP (2012-present, most historical)
#   VIIRS_NOAA20_NRT — NOAA-20 (2018-present)
#   VIIRS_ALL_NRT   — both satellites combined (best coverage)
SOURCE = "VIIRS_ALL_NRT"

# FIRMS API limits each request to 10 days max
DAYS_PER_REQUEST = 10

# Output
OUT_DIR = Path("/content/drive/MyDrive/wildfire_data/viirs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("VIIRS Active Fire Download (Vietnam, 2018-2023)")
print("=" * 70)
print(f"  BBox: {WEST},{SOUTH},{EAST},{NORTH}")
print(f"  Years: {YEARS}")
print(f"  Source: {SOURCE}")
print(f"  Output: {OUT_DIR}")

if FIRMS_API_KEY == "YOUR_FIRMS_API_KEY_HERE":
    print("\n❌ ERROR: You haven't set your FIRMS API key!")
    print("   1. Register at https://firms.modaps.eosdis.nasa.gov/api/area/")
    print("   2. Check email — you'll get an API key (MAP_KEY)")
    print("   3. Replace YOUR_FIRMS_API_KEY_HERE in this script")
    sys.exit(1)


def download_chunk(date_start, date_end, max_retries=3):
    """Download VIIRS detections for a date range. Returns DataFrame."""
    url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{}/{}/{},{},{},{}/{}".format(
        FIRMS_API_KEY,
        SOURCE,
        WEST, SOUTH, EAST, NORTH,
        DAYS_PER_REQUEST
    )
    # The FIRMS API uses the date the request is made + N days backward
    # To get historical data we need to use the time_range endpoint:
    # Actually, FIRMS "area" endpoint gives the last N days from today.
    # For historical data, use the "area" endpoint but step backward in time.
    # Alternative: use FIRMS archive download (no API key needed):
    #   https://firms.modaps.eosdis.nasa.gov/active_fire/viirs-c2/csv/{region}/{year}/viirs-c2_{region}_all_{year}.csv
    # Let's use the archive CSV approach instead — see below.
    pass


def download_year_archive(year, max_retries=3):
    """Download full year of VIIRS archive data via FIRMS CSV download.
    
    FIRMS provides annual archive CSVs at:
    https://firms.modaps.eosdis.nasa.gov/active_fire/viirs-c2/csv/{region}/{year}/viirs-c2_{region}_all_{year}.csv
    
    These are NOT bbox-filtered, so we'll need to filter post-download.
    Region options: "south_asia", "south_america", etc. Vietnam is in "south_asia".
    """
    # Vietnam falls under "south_asia" region in FIRMS archive
    region = "south_asia"
    url = f"https://firms.modaps.eosdis.nasa.gov/active_fire/viirs-c2/csv/{region}/{year}/viirs-c2_{region}_all_{year}.csv"
    
    print(f"  Downloading: {url}")
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.text) > 100:
                # Parse CSV
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                print(f"    Got {len(df):,} rows (raw)")
                # Filter to Vietnam bbox
                mask = (df["longitude"] >= WEST) & (df["longitude"] <= EAST) & \
                       (df["latitude"] >= SOUTH) & (df["latitude"] <= NORTH)
                df_vn = df[mask].copy()
                print(f"    Filtered to bbox: {len(df_vn):,} rows")
                return df_vn
            elif r.status_code == 404:
                print(f"    404 — archive not available for {year} (try NRT API instead)")
                return None
            else:
                print(f"    HTTP {r.status_code}, retrying ({attempt+1}/{max_retries})...")
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"    Error: {e}, retrying ({attempt+1}/{max_retries})...")
            time.sleep(5 * (attempt + 1))
    return None


def download_year_nrt(year, max_retries=3):
    """Fallback: use NRT (Near Real Time) API for years that don't have archive CSV.
    
    Note: FIRMS NRT only provides ~last 60 days, so this won't work for historical years.
    For historical data, use the archive approach above.
    """
    print(f"  NRT fallback not available for historical year {year}")
    return None


# ============================================================
# MAIN LOOP
# ============================================================
total_detections = 0
all_years_data = {}

for year in YEARS:
    print(f"\n[{year}] ===")
    out_file = OUT_DIR / f"viirs_fires_{year}.csv"
    if out_file.exists():
        print(f"  Already exists ({out_file.stat().st_size/1e6:.1f} MB) — skipping")
        df_existing = pd.read_csv(out_file)
        all_years_data[year] = df_existing
        total_detections += len(df_existing)
        continue

    # Try archive first
    df_year = download_year_archive(year)
    if df_year is None or len(df_year) == 0:
        # Try NRT fallback
        df_year = download_year_nrt(year)

    if df_year is None or len(df_year) == 0:
        print(f"  ⚠ No data for {year}")
        continue

    # Keep only relevant columns + standardize format to match MODIS
    keep_cols = ["latitude", "longitude", "acq_date", "satellite", "confidence"]
    for c in keep_cols:
        if c not in df_year.columns:
            print(f"  ⚠ Missing column {c} — keeping all columns")
            keep_cols = list(df_year.columns)
            break
    df_year = df_year[keep_cols].copy()
    df_year.to_csv(out_file, index=False)
    print(f"  ✓ Saved {len(df_year):,} rows to {out_file}")
    all_years_data[year] = df_year
    total_detections += len(df_year)

# Save combined file
combined_path = OUT_DIR / "viirs_fires_ALL.csv"
if all_years_data:
    df_all = pd.concat([all_years_data[y] for y in sorted(all_years_data.keys())], ignore_index=True)
    df_all.to_csv(combined_path, index=False)
    print(f"\n✓ Combined: {combined_path} ({len(df_all):,} rows)")

print(f"\n{'=' * 70}")
print(f"✓ VIIRS DOWNLOAD COMPLETE")
print(f"{'=' * 70}")
print(f"  Total detections: {total_detections:,}")
print(f"  Files saved in: {OUT_DIR}")
print(f"\nNext steps:")
print(f"  1. Copy the viirs/ folder to your local project at:")
print(f"     /home/z/my-project/upload/wildfire_data/wildfire_data/viirs/")
print(f"  2. Re-run preprocess_v85.py — it will auto-detect VIIRS and add")
print(f"     an independent validation label column `fire_viirs`.")
print(f"  3. Compare MODIS-based predictions against VIIRS ground truth.")
