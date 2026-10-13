"""Phase A2: Enrich district panel with static features (landcover, geo, region, NDVI)."""
import time, os, json
import polars as pl
import pandas as pd
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.ops import unary_union

PANEL_PATH = "/home/z/my-project/download/production/district_panel.parquet"
DCW_PATH = "/home/z/my-project/download/production/district_cell_weights.parquet"
LC_PATH = "/home/z/my-project/public/model/v90/landcover_lookup.parquet"
GEO_PATH = "/home/z/my-project/public/model/v90/geo_features.parquet"
GADM_PATH = "/home/z/my-project/public/data/geo/gadm41_VNM_2.json"
OUT_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"

STATIC_LANDCOVER = ["forest_prop", "shrubland_prop", "grassland_prop", "agriculture_prop",
                    "urban_prop", "bare_prop", "water_prop"]
STATIC_GEO = ["elevation_mean", "elevation_std", "slope_mean", "koppen_Cwa", "koppen_Am",
              "koppen_Aw", "latitude_band", "coastal_distance", "population_density_proxy"]


def main():
    t0 = time.time()
    print("=== Phase A2: Enrich district panel ===")

    # Load panel
    print("\n[1] Loading district panel...")
    panel = pl.read_parquet(PANEL_PATH)
    # Drop duplicate cols from full join
    drop_cols = [c for c in panel.columns if c.endswith("_right")]
    panel = panel.drop(drop_cols)
    print(f"  {panel.shape[0]:,} rows × {panel.shape[1]} cols (after drop {len(drop_cols)} dups)")

    # Load DCW + LC + GEO + compute static per district
    print("\n[2] Computing static features per district...")
    dcw = pd.read_parquet(DCW_PATH)
    lc = pd.read_parquet(LC_PATH)
    gf = pd.read_parquet(GEO_PATH)

    # Merge dcw with lc + geo
    dcw_static = dcw.merge(lc, on=["lat", "lon"], how="left").merge(gf, on=["lat", "lon"], how="left")
    # Weighted avg of static features per district
    for f in STATIC_LANDCOVER + STATIC_GEO:
        if f in dcw_static.columns:
            dcw_static[f] = dcw_static[f] * dcw_static["weight_norm"]
    district_static = dcw_static.groupby("district_id", as_index=False)[STATIC_LANDCOVER + STATIC_GEO].sum()
    print(f"  Static features for {len(district_static)} districts")

    # Compute district centroid + name + region from GADM
    print("\n[3] Loading GADM for centroid + name + region...")
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
        is_south_ood = int(cy < 14)  # cold-start OOD mask
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
    print(f"  {len(cent_df)} districts (with centroid)")

    # Merge static features + centroids
    print("\n[4] Merging static + centroids into panel...")
    static_full = district_static.merge(cent_df, on="district_id", how="right")
    print(f"  Static full: {len(static_full)} districts (incl. those without weather cells)")

    # Add NDVI placeholder (0.6 / 0.1 / 0.4 by region — same as v9.0)
    static_full["ndvi"] = static_full["region"].map({"north": 0.6, "central": 0.1, "south": 0.4}).fillna(0.4)
    static_full["ndvi_anom"] = 0.0
    static_full["ndvi_lag7"] = static_full["ndvi"]

    # Convert to polars + join with panel
    static_pl = pl.from_pandas(static_full)
    final = panel.join(static_pl, on="district_id", how="left")
    print(f"  Final panel: {final.shape[0]:,} rows × {final.shape[1]} cols")

    # Stats
    print("\n=== Summary ===")
    df = final.to_pandas()
    print(f"  Districts with weather: {df['t2m'].notna().sum() / 2191} (avg per day)")
    print(f"  Districts total: {df['district_id'].nunique()}")
    print(f"  South OOD (lat<14): {df[df['is_south_ood']==1]['district_id'].nunique()} districts")
    print(f"  North+Central (train): {df[df['is_south_ood']==0]['district_id'].nunique()} districts")
    # Check label balance
    train_mask = df["is_south_ood"] == 0
    test_mask = df["is_south_ood"] == 1
    print(f"\n  TRAIN (lat≥14):")
    print(f"    Rows: {train_mask.sum():,}")
    print(f"    Label h1 forest fire: {df.loc[train_mask, 'label_modis_h1'].sum():.0f} ({100*df.loc[train_mask, 'label_modis_h1'].sum()/train_mask.sum():.2f}%)")
    print(f"    Label h1 union fire: {df.loc[train_mask, 'label_union_h1'].sum():.0f} ({100*df.loc[train_mask, 'label_union_h1'].sum()/train_mask.sum():.2f}%)")
    print(f"\n  TEST (lat<14, cold-start OOD):")
    print(f"    Rows: {test_mask.sum():,}")
    print(f"    Label h1 forest fire: {df.loc[test_mask, 'label_modis_h1'].sum():.0f} ({100*df.loc[test_mask, 'label_modis_h1'].sum()/test_mask.sum():.2f}%)")
    print(f"    Label h1 union fire: {df.loc[test_mask, 'label_union_h1'].sum():.0f} ({100*df.loc[test_mask, 'label_union_h1'].sum()/test_mask.sum():.2f}%)")

    print(f"\n[5] Saving...")
    final.write_parquet(OUT_PATH)
    print(f"  ✓ Saved: {OUT_PATH}")
    print(f"  Size: {os.path.getsize(OUT_PATH) / (1024*1024):.1f} MB")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
