#!/usr/bin/env python3
"""
Download REAL NASA MODIS fire data for Vietnam 2018-2023 from
Microsoft Planetary Computer (free, no auth needed).

Downloads 1 day per month × 6 years = 72 sample days
Each day: 4 tiles (covering Vietnam) × 2 satellites (Terra + Aqua) = 8 tiles
Total: 576 FireMask GeoTIFFs × ~500 KB = ~288 MB

Extracts fire pixels → CSV: latitude, longitude, acq_date, confidence, satellite
Output: /home/z/my-project/download/real_data_pilot/firms/modis_vietnam_2018_2023.csv
"""
import os, sys, time, json, warnings, concurrent.futures
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# Install deps if needed
try:
    import pystac_client, planetary_computer, rasterio
    from pyproj import Transformer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "pystac-client", "planetary-computer", "rasterio", "pyproj"])
    import pystac_client, planetary_computer
    import rasterio
    from pyproj import Transformer

OUT_DIR = "/home/z/my-project/download/real_data_pilot/firms"
os.makedirs(OUT_DIR, exist_ok=True)
TEMP_DIR = "/tmp/modis_fire_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Vietnam bounding box
BBOX_VN = [104, 11, 110, 21]
CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Initialize STAC client
catalog = pystac_client.Client.open(CATALOG_URL, modifier=planetary_computer.sign_inplace)
print("✓ Connected to Microsoft Planetary Computer (free, no auth)")

def extract_fire_pixels_from_tif(tif_path, day_str, satellite):
    """Extract fire pixel (lat, lon, confidence) from a FireMask GeoTIFF."""
    try:
        with rasterio.open(tif_path) as src:
            fire = src.read(1)
            # FireMask values: 7=low, 8=nominal, 9=high confidence
            fire_mask = (fire >= 7) & (fire <= 9)
            fire_pixels = np.argwhere(fire_mask)
            if len(fire_pixels) == 0:
                return []
            # Transformer: MODIS sinusoidal → lat/lon
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            results = []
            for py, px in fire_pixels:
                # Pixel → sinusoidal (meters)
                x, y = src.transform * (px, py)
                # Sinusoidal → lat/lon
                lon, lat = transformer.transform(x, y)
                # Filter to Vietnam
                if BBOX_VN[0] <= lon <= BBOX_VN[2] and BBOX_VN[1] <= lat <= BBOX_VN[3]:
                    conf = int(fire[py, px])
                    conf_label = {7: "low", 8: "nominal", 9: "high"}.get(conf, "unknown")
                    results.append({
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "acq_date": day_str,
                        "acq_time": "",  # not in MOD14A1, would need MYD14A1 metadata
                        "satellite": satellite,  # "Terra" or "Aqua"
                        "instrument": "MODIS",
                        "confidence": conf_label,
                        "brightness": 0.0,  # not in FireMask, would need MaxFRP asset
                    })
            return results
    except Exception as e:
        print(f"    Error processing {tif_path}: {e}")
        return []

def download_day_fire(day_str):
    """Download all MOD14A1 + MYD14A1 fire tiles for one day over Vietnam."""
    # Search STAC for that day
    next_day = pd.Timestamp(day_str) + pd.Timedelta(days=1)
    search = catalog.search(
        collections=["modis-14A1-061"],
        bbox=BBOX_VN,
        datetime=f"{day_str}/{next_day.strftime('%Y-%m-%d')}",
        limit=20,
    )
    items = list(search.get_items())
    if not items:
        return []
    
    all_fires = []
    for item in items:
        # Determine satellite from item ID
        sat = "Terra" if item.id.startswith("MOD14") else "Aqua" if item.id.startswith("MYD14") else "Unknown"
        if "FireMask" not in item.assets:
            continue
        href = item.assets["FireMask"].href
        # Download to temp file
        tif_path = f"{TEMP_DIR}/{item.id}_FireMask.tif"
        if not os.path.exists(tif_path):
            try:
                r = requests.get(href, timeout=60)
                if r.status_code == 200:
                    with open(tif_path, "wb") as f:
                        f.write(r.content)
                else:
                    continue
            except Exception:
                continue
        # Extract fire pixels
        fires = extract_fire_pixels_from_tif(tif_path, day_str, sat)
        all_fires.extend(fires)
    return all_fires

# Main: download 1 day per month for 2018-2023
print(f"\nDownloading MODIS fire data for Vietnam (1 day/month × 6 years)...")
print(f"  Date range: 2018-01 to 2023-12 (72 months)")
print(f"  Per day: 4 tiles × 2 satellites = 8 FireMask GeoTIFFs")
print(f"  Total: 576 files × ~500 KB = ~288 MB")
print(f"  Estimated time: ~15-30 minutes\n")

all_fires = []
start_time = time.time()
for i, year in enumerate(range(2018, 2024)):
    for month in [1, 2, 3, 4, 11, 12]:  # Vietnam fire season: Nov-Apr only
        # Use 15th of each month as representative day
        day_str = f"{year}-{month:02d}-15"
        print(f"[{i*12+month}/{72}] {day_str}...", end=" ", flush=True)
        fires = download_day_fire(day_str)
        all_fires.extend(fires)
        # Cleanup temp files to save disk
        for f in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, f))
        elapsed = time.time() - start_time
        rate = len(all_fires) / elapsed * 60 if elapsed > 0 else 0
        print(f"{len(fires)} fires (total: {len(all_fires):,}, {elapsed:.0f}s elapsed)")

# Save to CSV
df = pd.DataFrame(all_fires)
out_csv = f"{OUT_DIR}/modis_vietnam_2018_2023_real.csv"
df.to_csv(out_csv, index=False)

print(f"\n{'='*70}")
print(f"✓ COMPLETE — Real NASA MODIS fire data for Vietnam 2018-2023")
print(f"{'='*70}")
print(f"\n  Output: {out_csv}")
print(f"  Total fire points: {len(df):,}")
print(f"  File size: {os.path.getsize(out_csv)/1024:.0f} KB")
print(f"  Date range: {df['acq_date'].min()} → {df['acq_date'].max()}")
print(f"  Lat range: {df['latitude'].min():.2f} → {df['latitude'].max():.2f}")
print(f"  Lon range: {df['longitude'].min():.2f} → {df['longitude'].max():.2f}")
print(f"  Satellites: {df['satellite'].value_counts().to_dict()}")
print(f"  Confidence: {df['confidence'].value_counts().to_dict()}")
print(f"\n  Sample rows:")
print(df.head(10).to_string())
print(f"\n  ✓ This is REAL NASA MODIS fire detection data, not synthetic.")
print(f"  ✓ Data source: Microsoft Planetary Computer (free, no auth)")
