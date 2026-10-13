"""Phase A6: Rebuild district_features + climatology for ALL 710 districts.

For districts without overlapping cells (e.g. coastal/border small districts),
use NEAREST cell's features as proxy (by centroid distance).
"""
import os, json, time
import polars as pl
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
from shapely.validation import make_valid
from shapely.ops import unary_union

PANEL_PATH = "/home/z/my-project/download/production/panel_production.parquet"
GADM_PATH = "/home/z/my-project/public/data/geo/gadm41_VNM_2.json"
LC_PATH = "/home/z/my-project/public/model/v90/landcover_lookup.parquet"
GEO_PATH = "/home/z/my-project/public/model/v90/geo_features.parquet"
CLIM_CELL_PATH = "/home/z/my-project/public/model/v90/cell_climatology.parquet"
OUT_DIR = "/home/z/my-project/public/model/v91"
os.makedirs(OUT_DIR, exist_ok=True)

STATIC_LANDCOVER = ["forest_prop", "shrubland_prop", "grassland_prop", "agriculture_prop",
                    "urban_prop", "bare_prop", "water_prop"]
STATIC_GEO = ["elevation_mean", "elevation_std", "slope_mean", "koppen_Cwa", "koppen_Am",
              "koppen_Aw", "latitude_band", "coastal_distance", "population_density_proxy"]


def main():
    t0 = time.time()
    print("=== Phase A6: Rebuild for ALL 710 districts ===")

    # 1. Load GADM (all 710 districts)
    print("\n[1] Loading GADM (all 710 districts)...")
    with open(GADM_PATH) as f:
        gadm = json.load(f)
    districts_all = []
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
        districts_all.append({
            "district_id": i,
            "district_name": props.get("NAME_2", ""),
            "province": props.get("NAME_1", ""),
            "district_type": props.get("TYPE_2", ""),
            "gid": props.get("GID_2", ""),
            "centroid_lat": round(cy, 4),
            "centroid_lon": round(cx, 4),
            "region": region,
            "is_south_ood": int(cy < 14),
        })
    print(f"  {len(districts_all)} districts total")

    # 2. Load cell-static features
    print("\n[2] Loading cell-static features...")
    lc = pd.read_parquet(LC_PATH)
    geo = pd.read_parquet(GEO_PATH)
    cell_static = lc.merge(geo, on=["lat", "lon"], how="outer")
    print(f"  {len(cell_static)} cells with static features")

    # 3. Load cell climatology
    print("\n[3] Loading cell climatology...")
    cell_clim = pd.read_parquet(CLIM_CELL_PATH)
    print(f"  {len(cell_clim):,} (cell, doy) entries")

    # 4. For each district, find NEAREST cell (by centroid distance)
    print("\n[4] Finding nearest cell for each district...")
    cell_coords = cell_static[["lat", "lon"]].values
    cell_tree = []  # list of (lat, lon, idx)
    for i, row in cell_static.iterrows():
        cell_tree.append((row["lat"], row["lon"], i))
    
    def find_nearest_cell(lat, lon):
        best_idx = None
        best_dist = float("inf")
        for clat, clon, idx in cell_tree:
            d = (clat - lat) ** 2 + (clon - lon) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    # 5. Build district_features for all 710 districts
    print("\n[5] Building district_features for all 710 districts...")
    district_features_rows = []
    for d in districts_all:
        nearest_cell_idx = find_nearest_cell(d["centroid_lat"], d["centroid_lon"])
        if nearest_cell_idx is None:
            # No cells available — use defaults
            row = {col: 0.0 for col in STATIC_LANDCOVER + STATIC_GEO}
            row.update({
                "forest_prop": 0.5, "agriculture_prop": 0.3, "water_prop": 0.05,
                "elevation_mean": 100, "elevation_std": 50, "slope_mean": 5,
                "latitude_band": int((d["centroid_lat"] - 8) / 2),
                "coastal_distance": 0, "population_density_proxy": 50,
            })
        else:
            cell_row = cell_static.iloc[nearest_cell_idx]
            row = {col: float(cell_row[col]) for col in STATIC_LANDCOVER + STATIC_GEO if col in cell_row}
        row.update({
            "district_id": d["district_id"],
            "district_name": d["district_name"],
            "province": d["province"],
            "district_type": d["district_type"],
            "gid": d["gid"],
            "centroid_lat": d["centroid_lat"],
            "centroid_lon": d["centroid_lon"],
            "region": d["region"],
            "is_south_ood": d["is_south_ood"],
            "ndvi": 0.6 if d["region"] == "north" else (0.1 if d["region"] == "central" else 0.4),
            "ndvi_anom": 0.0,
            "ndvi_lag7": 0.6 if d["region"] == "north" else (0.1 if d["region"] == "central" else 0.4),
            "cell_fire_rate_exp": 0.05,
            "cell_fire_count_30d": 1.5,
            "is_dry": 0,
            "cell_fire_rate_dry_exp": 0.05,
        })
        district_features_rows.append(row)
    district_features = pd.DataFrame(district_features_rows)
    print(f"  Built: {len(district_features)} districts × {len(district_features.columns)} cols")
    
    feat_path = os.path.join(OUT_DIR, "district_features.parquet")
    pl.from_pandas(district_features).write_parquet(feat_path)
    print(f"  ✓ Saved: {feat_path} ({os.path.getsize(feat_path)/1024:.0f} KB)")

    # 6. Build district_climatology for all 710 districts (use nearest cell's climatology)
    print("\n[6] Building district_climatology for all 710 districts...")
    # For each district, get nearest cell, then copy that cell's climatology with district_id swapped
    clim_rows = []
    cell_clim_unique = cell_clim.drop_duplicates(subset=["lat", "lon", "doy"])
    cell_clim_by_cell = {f"{r['lat']}_{r['lon']}": [] for _, r in cell_static[["lat","lon"]].drop_duplicates().iterrows()}
    for _, r in cell_clim_unique.iterrows():
        key = f"{r['lat']}_{r['lon']}"
        if key in cell_clim_by_cell:
            cell_clim_by_cell[key].append(r)
    
    for d in districts_all:
        nearest_cell_idx = find_nearest_cell(d["centroid_lat"], d["centroid_lon"])
        if nearest_cell_idx is None:
            continue
        cell_row = cell_static.iloc[nearest_cell_idx]
        cell_lat, cell_lon = cell_row["lat"], cell_row["lon"]
        key = f"{cell_lat}_{cell_lon}"
        cell_clims = cell_clim_by_cell.get(key, [])
        for cc in cell_clims:
            clim_rows.append({
                "district_id": d["district_id"],
                "doy": int(cc["doy"]),
                "t2m_clim": float(cc["t2m_clim"]),
                "rh_clim": float(cc["rh_clim"]),
                "precip_clim": float(cc["precip_clim"]),
                "wind_clim": float(cc["wind_clim"]),
            })
    district_clim = pd.DataFrame(clim_rows)
    print(f"  Built: {len(district_clim):,} rows ({district_clim['district_id'].nunique()} districts × {district_clim['doy'].nunique()} doys)")
    
    clim_path = os.path.join(OUT_DIR, "district_climatology.parquet")
    pl.from_pandas(district_clim).write_parquet(clim_path)
    print(f"  ✓ Saved: {clim_path} ({os.path.getsize(clim_path)/1024:.0f} KB)")

    print("\n=== Summary ===")
    print(f"  Total districts: {len(districts_all)}")
    print(f"  With features: {len(district_features)}")
    print(f"  With climatology: {district_clim['district_id'].nunique()}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
