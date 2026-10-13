"""Phase A5: Rebuild district_features + climatology using SAME mapping as cell_to_district.json.

This fixes the bug where build_district_panel.py used intersection-area while
build_district_lookup.py used point-in-polygon — giving different district sets.
"""
import os, json, time
import polars as pl
import pandas as pd
import numpy as np

PANEL_PATH = "/home/z/my-project/download/production/panel_production.parquet"
DCW_PATH = "/home/z/my-project/download/production/district_cell_weights.parquet"  # old (intersection-based)
CELL_TO_DISTRICT_PATH = "/home/z/my-project/public/data/geo/cell_to_district.json"
GADM_PATH = "/home/z/my-project/public/data/geo/gadm41_VNM_2.json"
LC_PATH = "/home/z/my-project/public/model/v90/landcover_lookup.parquet"
GEO_PATH = "/home/z/my-project/public/model/v90/geo_features.parquet"
OUT_DIR = "/home/z/my-project/public/model/v91"
os.makedirs(OUT_DIR, exist_ok=True)

STATIC_LANDCOVER = ["forest_prop", "shrubland_prop", "grassland_prop", "agriculture_prop",
                    "urban_prop", "bare_prop", "water_prop"]
STATIC_GEO = ["elevation_mean", "elevation_std", "slope_mean", "koppen_Cwa", "koppen_Am",
              "koppen_Aw", "latitude_band", "coastal_distance", "population_density_proxy"]


def main():
    t0 = time.time()
    print("=== Phase A5: Rebuild district_features (consistent mapping) ===")

    # Load cell_to_district (the authoritative mapping from Phase D)
    print("\n[1] Loading cell_to_district.json (Phase D mapping)...")
    with open(CELL_TO_DISTRICT_PATH) as f:
        c2d = json.load(f)
    # Build (cell_lat, cell_lon, district_id) rows
    cell_district_rows = []
    for cell_id_str, m in c2d.items():
        if not m:
            continue
        cell_id = int(cell_id_str)
        # cell_id indexing: cells are in row-major order from lat=8.5, lon=102.0, step 0.5
        # cell 0 = (9.0, 105.0), cell 1 = (9.0, 105.5), etc.
        # Need to recover (lat, lon) from cell_id — but we have it in VIETNAM_GRID
        # For now, leave it — we'll join with panel via lat/lon in next step
        cell_district_rows.append({"district_id": m["district_id"]})

    # Better: get (lat, lon, district_id) from VIETNAM_GRID generation
    # Mirror generateVietnamGrid() in src/lib/data.ts
    POLYGON_VN = [
        (22.85, 103.55), (22.55, 103.75), (22.10, 103.55), (21.55, 103.50),
        (21.10, 103.55), (20.65, 104.05), (20.10, 104.05), (19.55, 104.55),
        (19.05, 104.10), (18.55, 104.10), (18.05, 104.55), (17.55, 105.55),
        (17.05, 106.05), (16.55, 106.55), (16.05, 107.05), (15.55, 107.45),
        (15.05, 107.50), (14.55, 107.10), (14.05, 107.05), (13.55, 106.05),
        (13.05, 105.55), (12.05, 105.05), (11.05, 105.05), (10.55, 104.95),
        (10.55, 104.65), (10.45, 104.55), (10.30, 104.40), (10.05, 104.55),
        (9.55, 104.75), (9.05, 104.75), (8.65, 104.65), (8.55, 104.85), (8.95, 105.05),
        (9.35, 105.35), (9.85, 105.55),
        (10.15, 105.85), (10.45, 106.45), (10.75, 106.85), (10.85, 107.30),
        (10.95, 107.55), (10.85, 107.95),
        (11.20, 108.55), (11.55, 108.85), (11.95, 109.05), (12.35, 109.20),
        (12.75, 109.30), (13.15, 109.40), (13.55, 109.30), (13.95, 109.20),
        (14.45, 109.05), (14.95, 108.90), (15.45, 108.70), (15.95, 108.30),
        (16.45, 108.10), (16.95, 107.80), (17.45, 107.00), (17.95, 106.50),
        (18.45, 106.00), (18.95, 105.80), (19.45, 105.70), (19.95, 105.75),
        (20.45, 106.55), (20.75, 107.05), (20.95, 107.45), (21.10, 107.75),
        (21.30, 107.95), (21.55, 108.05),
        (21.75, 107.80), (22.05, 107.50), (22.25, 107.20), (22.45, 106.80),
        (22.65, 106.50), (22.85, 106.00), (22.95, 105.50), (22.95, 105.00),
        (22.95, 104.50), (22.75, 104.25), (22.55, 104.00), (22.75, 103.80),
        (22.85, 103.55),
    ]

    def point_in_polygon(lat, lon, poly):
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            yi, xi = poly[i]
            yj, xj = poly[j]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    cells = []
    lat = 8.5
    while lat <= 23.5:
        lon = 102.0
        while lon <= 110.0:
            if point_in_polygon(lat, lon, POLYGON_VN):
                cells.append({"id": len(cells), "lat": round(lat, 2), "lon": round(lon, 2)})
            lon = round(lon + 0.5, 2)
        lat = round(lat + 0.5, 2)
    print(f"  Generated {len(cells)} cells from VIETNAM_GRID polygon")

    # Build (district_id, lat, lon, weight=1/N) where N = number of cells in district
    district_cells = {}  # district_id → list of (lat, lon)
    for cell in cells:
        m = c2d.get(str(cell["id"]))
        if not m:
            continue
        did = m["district_id"]
        if did not in district_cells:
            district_cells[did] = []
        district_cells[did].append((cell["lat"], cell["lon"]))

    print(f"  {len(district_cells)} districts with at least 1 mapped cell")

    # Build new district_cell_weights table with weight=1/N per cell in district
    new_dcw_rows = []
    for did, cell_list in district_cells.items():
        n = len(cell_list)
        for lat, lon in cell_list:
            new_dcw_rows.append({
                "district_id": did,
                "lat": lat,
                "lon": lon,
                "weight": 1.0,
                "weight_norm": 1.0 / n,
            })
    new_dcw = pd.DataFrame(new_dcw_rows)
    new_dcw_path = "/home/z/my-project/download/production/district_cell_weights_v2.parquet"
    new_dcw.to_parquet(new_dcw_path, index=False)
    print(f"  Saved new weights: {new_dcw_path} ({len(new_dcw):,} pairs)")

    # Build district_features using new mapping
    print("\n[2] Building district_features (new mapping)...")
    lc = pd.read_parquet(LC_PATH)
    gf = pd.read_parquet(GEO_PATH)
    dcw_static = new_dcw.merge(lc, on=["lat", "lon"], how="left").merge(gf, on=["lat", "lon"], how="left")
    for f in STATIC_LANDCOVER + STATIC_GEO:
        if f in dcw_static.columns:
            dcw_static[f] = dcw_static[f] * dcw_static["weight_norm"]
    district_static = dcw_static.groupby("district_id", as_index=False)[STATIC_LANDCOVER + STATIC_GEO].sum()

    # Add NDVI (placeholder by region)
    # First need region per district — from GADM centroid
    from shapely.geometry import shape
    from shapely.validation import make_valid
    from shapely.ops import unary_union
    with open(GADM_PATH) as f:
        gadm = json.load(f)
    centroids = []
    for i, feat in enumerate(gadm["features"]):
        props = feat["properties"]
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.geom_type == "GeometryCollection":
            polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if polys:
                geom = unary_union(polys)
        cy = geom.centroid.y
        cx = geom.centroid.x
        region = "north" if cy >= 18.5 else ("central" if cy >= 13.5 else "south")
        is_south_ood = int(cy < 14)
        centroids.append({
            "district_id": i,
            "district_name": props.get("NAME_2", ""),
            "province": props.get("NAME_1", ""),
            "district_type": props.get("TYPE_2", ""),
            "centroid_lat": round(cy, 4),
            "centroid_lon": round(cx, 4),
            "region": region,
            "is_south_ood": is_south_ood,
        })
    cent_df = pd.DataFrame(centroids)

    # Filter to only districts in new_dcw (94 districts)
    valid_ids = set(new_dcw["district_id"].unique())
    cent_df = cent_df[cent_df["district_id"].isin(valid_ids)]
    print(f"  Centroids for {len(cent_df)} mapped districts")

    # Merge static + centroids
    static_full = district_static.merge(cent_df, on="district_id", how="right")
    static_full["ndvi"] = static_full["region"].map({"north": 0.6, "central": 0.1, "south": 0.4}).fillna(0.4)
    static_full["ndvi_anom"] = 0.0
    static_full["ndvi_lag7"] = static_full["ndvi"]

    # Cell-level legacy features (averaged across cells in district)
    # cell_fire_rate_exp, cell_fire_count_30d, is_dry, cell_fire_rate_dry_exp
    # These are time-varying in panel, but for inference we use a constant per cell.
    # Take mean across cells per district.
    panel = pd.read_parquet(PANEL_PATH, columns=["lat", "lon", "cell_fire_rate_exp", "cell_fire_count_30d", "is_dry", "cell_fire_rate_dry_exp"])
    panel_d = panel.merge(new_dcw[["district_id", "lat", "lon", "weight_norm"]], on=["lat", "lon"], how="inner")
    for c in ["cell_fire_rate_exp", "cell_fire_count_30d", "is_dry", "cell_fire_rate_dry_exp"]:
        panel_d[c] = panel_d[c] * panel_d["weight_norm"]
    cell_legacy = panel_d.groupby("district_id", as_index=False)[["cell_fire_rate_exp", "cell_fire_count_30d", "is_dry", "cell_fire_rate_dry_exp"]].mean()
    static_full = static_full.merge(cell_legacy, on="district_id", how="left")

    print(f"  Final static features: {static_full.shape}")

    # Save
    static_path = os.path.join(OUT_DIR, "district_features.parquet")
    pl.from_pandas(static_full).write_parquet(static_path)
    print(f"  ✓ Saved: {static_path} ({os.path.getsize(static_path)/1024:.0f} KB)")

    # Build district_climatology: per (district_id, doy), mean over 2018-2021
    print("\n[3] Building district_climatology...")
    panel2 = pd.read_parquet(PANEL_PATH, columns=["date", "lat", "lon", "doy", "t2m", "rh", "precip", "wind"])
    panel2["date"] = pd.to_datetime(panel2["date"])
    train2 = panel2[panel2["date"].dt.year <= 2021].copy()
    train2 = train2.merge(new_dcw[["district_id", "lat", "lon", "weight_norm"]], on=["lat", "lon"], how="inner")
    for c in ["t2m", "rh", "precip", "wind"]:
        train2[c] = train2[c] * train2["weight_norm"]
    clim = train2.groupby(["district_id", "doy"], as_index=False).agg({
        "t2m": "sum", "rh": "sum", "precip": "sum", "wind": "sum",
    })
    clim.columns = ["district_id", "doy", "t2m_clim", "rh_clim", "precip_clim", "wind_clim"]
    print(f"  Climatology: {len(clim):,} rows ({clim['district_id'].nunique()} districts × {clim['doy'].nunique()} doys)")
    clim_path = os.path.join(OUT_DIR, "district_climatology.parquet")
    pl.from_pandas(clim).write_parquet(clim_path)
    print(f"  ✓ Saved: {clim_path} ({os.path.getsize(clim_path)/1024:.0f} KB)")

    # Print summary
    print("\n=== Summary ===")
    print(f"  Districts with features: {static_full['district_id'].nunique()}")
    print(f"  Districts with climatology: {clim['district_id'].nunique()}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
