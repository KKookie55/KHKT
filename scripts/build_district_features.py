"""Phase A4: Build district_features.parquet + district_climatology.parquet.

district_features.parquet: static features per district_id
  - centroid_lat, centroid_lon, region, is_south_ood
  - district_name, province, district_type
  - landcover (7), geo (9), NDVI (3)
  - cell_fire_rate_exp, cell_fire_count_30d, is_dry, cell_fire_rate_dry_exp
    (cell-level legacy — averaged)

district_climatology.parquet: per (district_id, doy) climatology
  - t2m_clim, rh_clim, precip_clim, wind_clim
  - computed from training period 2018-2021
"""
import os, json, time
import polars as pl
import pandas as pd
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.ops import unary_union

PANEL_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"
GADM_PATH = "/home/z/my-project/public/data/geo/gadm41_VNM_2.json"
OUT_DIR = "/home/z/my-project/public/model/v91"
os.makedirs(OUT_DIR, exist_ok=True)

STATIC_COLS = [
    "forest_prop", "shrubland_prop", "grassland_prop", "agriculture_prop",
    "urban_prop", "bare_prop", "water_prop",
    "elevation_mean", "elevation_std", "slope_mean",
    "koppen_Cwa", "koppen_Am", "koppen_Aw",
    "latitude_band", "coastal_distance", "population_density_proxy",
    "ndvi", "ndvi_anom", "ndvi_lag7",
    "cell_fire_rate_exp", "cell_fire_count_30d", "is_dry", "cell_fire_rate_dry_exp",
    "centroid_lat", "centroid_lon", "is_south_ood", "region",
]


def main():
    t0 = time.time()
    print("=== Phase A4: Build district_features + climatology ===")

    # Load panel
    print("\n[1] Loading panel...")
    panel = pl.read_parquet(PANEL_PATH)
    print(f"  {panel.shape[0]:,} rows × {panel.shape[1]} cols")

    # Build static features per district (take first row per district — static features don't change)
    print("\n[2] Extracting static features per district...")
    static_cols_present = [c for c in STATIC_COLS if c in panel.columns]
    # Group by district_id, take first row of static features
    static = panel.group_by("district_id", maintain_order=True).agg([
        pl.col(c).first().alias(c) for c in static_cols_present
    ])
    # Also get doy from date
    print(f"  Static features for {static.shape[0]} districts")

    # Add district metadata (name, province, type) from GADM
    print("\n[3] Loading GADM for district names...")
    with open(GADM_PATH) as f:
        gadm = json.load(f)
    meta_rows = []
    for i, feat in enumerate(gadm["features"]):
        p = feat["properties"]
        meta_rows.append({
            "district_id": i,
            "district_name": p.get("NAME_2", ""),
            "province": p.get("NAME_1", ""),
            "district_type": p.get("TYPE_2", ""),
            "gid": p.get("GID_2", ""),
        })
    meta = pl.DataFrame(meta_rows)
    static = static.join(meta, on="district_id", how="left")
    print(f"  After GADM join: {static.shape[0]} districts")

    static_path = os.path.join(OUT_DIR, "district_features.parquet")
    static.write_parquet(static_path)
    print(f"  ✓ Saved: {static_path} ({os.path.getsize(static_path)/1024:.0f} KB)")

    # Build climatology per district × doy (mean over 2018-2021)
    print("\n[4] Building climatology per (district, doy)...")
    panel_pd = panel.to_pandas()
    panel_pd["date"] = pd.to_datetime(panel_pd["date"])
    train_pd = panel_pd[panel_pd["date"].dt.year <= 2021].copy()  # 2018-2021
    print(f"  Train rows (2018-2021): {len(train_pd):,}")

    clim = train_pd.groupby(["district_id", "doy"]).agg({
        "t2m": "mean",
        "rh": "mean",
        "precip": "mean",
        "wind": "mean",
    }).reset_index()
    clim.columns = ["district_id", "doy", "t2m_clim", "rh_clim", "precip_clim", "wind_clim"]
    print(f"  Climatology rows: {len(clim):,} ({clim['district_id'].nunique()} districts × {clim['doy'].nunique()} doys)")

    clim_path = os.path.join(OUT_DIR, "district_climatology.parquet")
    pl.from_pandas(clim).write_parquet(clim_path)
    print(f"  ✓ Saved: {clim_path} ({os.path.getsize(clim_path)/1024:.0f} KB)")

    print(f"\n=== Elapsed: {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
