"""Phase A: Build district-level training panel using POLARS for speed."""
import time, os, sys
import polars as pl
import pandas as pd

PANEL_PATH = "/home/z/my-project/download/production/panel_production.parquet"
DCW_PATH = "/home/z/my-project/download/production/district_cell_weights.parquet"
OUT_PATH = "/home/z/my-project/download/production/district_panel.parquet"

WEATHER_FEATURES = [
    "t2m", "rh", "wind", "precip",
    "t2m_lag1", "t2m_lag3", "t2m_lag7", "t2m_lag14",
    "rh_lag1", "rh_lag3", "rh_lag7", "rh_lag14",
    "wind_lag1", "wind_lag3", "wind_lag7", "wind_lag14",
    "precip_lag1", "precip_lag3", "precip_lag7", "precip_lag14",
    "t2m_roll7", "t2m_roll14", "rh_roll7", "rh_roll14",
    "precip_roll7_sum", "precip_roll14_sum", "precip_roll30_sum",
    "days_since_rain", "t2m_diff3", "rh_diff3", "wind_diff3",
    "fwi_compound", "heat_index", "wind_chill",
    "t2m_anom_doy", "rh_anom_doy", "precip_anom_doy", "wind_anom_doy",
    "t2m_anom_roll7", "rh_anom_roll7", "precip_anom_roll7",
    "dry_anomaly_composite",
    "fwi", "fwi_ffmc", "fwi_dmc", "fwi_dc", "fwi_isi", "fwi_bui",
    "cell_fire_rate_exp", "cell_fire_count_30d", "is_dry", "cell_fire_rate_dry_exp",
]
FIRE_COUNT_FEATURES = ["fire_modis", "fire_viirs", "fire_union"]
BINARY_LABELS = [
    "label_modis_h1", "label_viirs_h1", "label_union_h1",
    "label_modis_h3", "label_viirs_h3", "label_union_h3",
    "label_modis_h7", "label_viirs_h7", "label_union_h7",
]


def main():
    t0 = time.time()
    print("=== Phase A: District panel (Polars) ===")

    print("\n[1] Loading cell panel (polars)...")
    panel = pl.read_parquet(PANEL_PATH)
    print(f"  {panel.shape[0]:,} rows × {panel.shape[1]} cols")

    print("\n[2] Loading district-cell weights...")
    dcw = pl.read_parquet(DCW_PATH)
    # Keep only needed cols
    dcw = dcw.select(["district_id", "lat", "lon", "weight_norm"])
    print(f"  {dcw.shape[0]:,} pairs, {dcw['district_id'].n_unique()} districts")

    print("\n[3] Joining panel × weights...")
    merged = panel.join(dcw, on=["lat", "lon"], how="inner")
    print(f"  Merged: {merged.shape[0]:,} rows × {merged.shape[1]} cols")

    # Pre-multiply weather cols by weight_norm (vectorized in Rust)
    print("\n[4] Pre-multiplying weather by weight_norm...")
    weather_cols_present = [c for c in WEATHER_FEATURES if c in merged.columns]
    t_mul = time.time()
    for c in weather_cols_present:
        merged = merged.with_columns((pl.col(c) * pl.col("weight_norm")).alias(c))
    print(f"  Done in {time.time()-t_mul:.1f}s")

    # 2 groupbys: weather+fire (sum), labels (max)
    print("\n[5] Groupby weather+fire (sum)...")
    t_gb1 = time.time()
    sum_cols = weather_cols_present + FIRE_COUNT_FEATURES
    sum_agg = merged.group_by(["district_id", "date"]).agg([
        pl.col(c).sum().alias(c) for c in sum_cols if c in merged.columns
    ] + [
        pl.col("doy").first().alias("doy"),
        pl.col("year").first().alias("year"),
    ])
    print(f"  Sum agg: {sum_agg.shape[0]:,} rows in {time.time()-t_gb1:.1f}s")

    print("\n[6] Groupby labels (max)...")
    t_gb2 = time.time()
    label_cols_present = [c for c in BINARY_LABELS if c in merged.columns]
    max_agg = merged.group_by(["district_id", "date"]).agg([
        pl.col(c).max().alias(c) for c in label_cols_present
    ])
    print(f"  Max agg: {max_agg.shape[0]:,} rows in {time.time()-t_gb2:.1f}s")

    print("\n[7] Joining sum_agg × max_agg...")
    final = sum_agg.join(max_agg, on=["district_id", "date"], how="full")
    # Sort by district + date
    final = final.sort(["district_id", "date"])
    print(f"  Final: {final.shape[0]:,} rows × {final.shape[1]} cols")

    print(f"\n[8] Saving...")
    final.write_parquet(OUT_PATH)
    print(f"  ✓ Saved: {OUT_PATH}")
    print(f"  Size: {os.path.getsize(OUT_PATH) / (1024*1024):.1f} MB")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    print("\n=== Summary ===")
    print(f"  Districts: {final['district_id'].n_unique()}")
    print(f"  Days: {final['date'].n_unique()}")
    print(f"  Total rows: {final.shape[0]:,}")
    # Sample for verification
    df = final.to_pandas()
    print(f"  Districts with fire_modis>0: {df[df['fire_modis']>0]['district_id'].nunique()}")
    print(f"  Days with fire_modis>0: {(df['fire_modis']>0).sum()}")
    print(f"  label_modis_h1 sum: {df['label_modis_h1'].sum()}")
    print(f"  label_union_h1 sum: {df['label_union_h1'].sum()}")


if __name__ == "__main__":
    main()
