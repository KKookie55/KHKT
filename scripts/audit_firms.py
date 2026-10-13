"""Phase B3: FIRMS Audit — memory-efficient version using polars throughout."""
import os, time, json, gc
import polars as pl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely.strtree import STRtree
import xgboost as gbm
import joblib

FIRMS_PATH = "/home/z/my-project/download/real_data_pilot/firms/modis_vietnam_2018_2023_real.csv"
GADM_PATH = "/home/z/my-project/public/data/geo/gadm41_VNM_2.json"
PANEL_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"
NDVI_PATH = "/home/z/my-project/public/model/v91/district_ndvi.parquet"
MODEL_PATH = "/home/z/my-project/public/model/v91/model_forest_fire.json"
ISO_PATH = "/home/z/my-project/public/model/v91/iso_forest_fire.pkl"
OUT_DIR = "/home/z/my-project/download/audit"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "t2m", "rh", "wind", "precip", "doy",
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
    "forest_prop", "shrubland_prop", "grassland_prop", "agriculture_prop",
    "urban_prop", "bare_prop", "water_prop",
    "elevation_mean", "elevation_std", "slope_mean",
    "koppen_Cwa", "koppen_Am", "koppen_Aw",
    "latitude_band", "coastal_distance", "population_density_proxy",
    "ndvi", "ndvi_anom", "ndvi_lag7",
    "cell_fire_rate_exp", "cell_fire_count_30d", "is_dry", "cell_fire_rate_dry_exp",
    "centroid_lat", "centroid_lon",
]


def main():
    t0 = time.time()
    print("=== Phase B3: FIRMS Audit (memory-efficient) ===", flush=True)

    # 1. Load FIRMS + spatial join to districts
    print("\n[1] Loading FIRMS + spatial join...", flush=True)
    firms = pd.read_csv(FIRMS_PATH)
    firms["acq_date"] = pd.to_datetime(firms["acq_date"])
    print(f"  {len(firms):,} FIRMS records", flush=True)

    with open(GADM_PATH) as f:
        gadm = json.load(f)
    district_polys = []
    district_meta = {}
    for i, feat in enumerate(gadm["features"]):
        props = feat["properties"]
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.geom_type == "GeometryCollection":
            polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if polys:
                geom = unary_union(polys)
        district_polys.append(geom)
        district_meta[i] = {
            "name": props.get("NAME_2", ""),
            "province": props.get("NAME_1", ""),
            "centroid_lat": geom.centroid.y,
            "centroid_lon": geom.centroid.x,
            "is_south": geom.centroid.y < 14,
        }
    tree = STRtree(district_polys)
    print(f"  {len(district_polys)} districts", flush=True)

    # Spatial join
    firms_fire_dids = []
    for idx, row in firms.iterrows():
        if idx % 5000 == 0:
            print(f"  join {idx:,}/{len(firms):,}", flush=True)
        pt = Point(row["longitude"], row["latitude"])
        candidates = tree.query(pt)
        found = None
        for ci in candidates:
            geom = district_polys[int(ci)] if not isinstance(ci, int) else district_polys[ci]
            if geom.contains(pt):
                found = int(ci) if not isinstance(ci, int) else ci
                break
        if found is None:
            for ci in candidates:
                geom = district_polys[int(ci)] if not isinstance(ci, int) else district_polys[ci]
                if geom.touches(pt):
                    found = int(ci) if not isinstance(ci, int) else ci
                    break
        firms_fire_dids.append(found if found is not None else -1)
    firms["district_id"] = firms_fire_dids
    print(f"  Mapped: {(firms['district_id'] != -1).sum()}/{len(firms)}", flush=True)
    firms_mapped = firms[firms["district_id"] != -1].copy()
    firms_mapped["date"] = firms_mapped["acq_date"].dt.normalize()
    fire_counts = firms_mapped.groupby(["district_id", "date"]).size().reset_index(name="fire_count")
    print(f"  {len(fire_counts)} (district, date) fire events", flush=True)

    # 2. Load panel — POLARS, only needed columns, cast to Float32
    print("\n[2] Loading panel (polars, Float32)...", flush=True)
    needed_cols = ["district_id", "date", "is_south_ood", "label_modis_h1"] + FEATURES
    panel = pl.read_parquet(PANEL_PATH, columns=[c for c in needed_cols if c != "ndvi"])
    cast_map = {c: pl.Float32 for c in panel.columns if panel[c].dtype == pl.Float64}
    panel = panel.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast_map.items()])
    # Drop NaN t2m
    panel = panel.filter(pl.col("t2m").is_not_null())
    print(f"  Panel: {panel.shape[0]:,} rows, mem={panel.estimated_size('mb'):.0f} MB", flush=True)

    # 3. Join with NDVI (polars)
    print("\n[3] Joining with NDVI...", flush=True)
    ndvi = pl.read_parquet(NDVI_PATH)
    # NDVI has district_id, doy, ndvi, ndvi_anom, ndvi_lag7
    panel = panel.join(ndvi.select(["district_id", "doy", "ndvi", "ndvi_anom", "ndvi_lag7"]),
                       on=["district_id", "doy"], how="left")
    # Fill NaN NDVI with default
    panel = panel.with_columns([
        pl.col("ndvi").fill_null(0.4),
        pl.col("ndvi_anom").fill_null(0.0),
        pl.col("ndvi_lag7").fill_null(0.4),
    ])
    print(f"  After NDVI join: {panel.shape[0]:,} rows", flush=True)

    # 4. Join with FIRMS fire_counts (polars) — actual labels
    print("\n[4] Joining with FIRMS fire_counts...", flush=True)
    fire_counts_pl = pl.from_pandas(fire_counts)
    # Cast both sides to same date type (datetime[μs])
    fire_counts_pl = fire_counts_pl.with_columns(pl.col("date").cast(pl.Datetime("us")))
    panel = panel.with_columns(pl.col("date").cast(pl.Datetime("us")))
    panel = panel.join(fire_counts_pl, on=["district_id", "date"], how="left")
    panel = panel.with_columns([
        pl.col("fire_count").fill_null(0),
        (pl.col("fire_count") > 0).cast(pl.Int8).alias("fire_today"),
    ])
    # Sort + add fire_h1 (next day fire)
    panel = panel.sort(["district_id", "date"])
    # shift forward within group: fire_h1 = fire_today shifted back 1 day = today's fire indicates yesterday's h1 prediction would have caught it
    # Actually: fire_h1 = tomorrow's fire_today. For each row (district, date), fire_h1 = fire_today of next day.
    # In polars: shift(-1) = take next row's value
    panel = panel.with_columns([
        pl.col("fire_today").shift(-1).over("district_id").fill_null(0).cast(pl.Int8).alias("fire_h1"),
    ])
    print(f"  After FIRMS join: {panel.shape[0]:,} rows", flush=True)

    # 5. Filter to TEST (South, cold-start OOD)
    print("\n[5] Filter to TEST (South, cold-start OOD)...", flush=True)
    test = panel.filter(pl.col("is_south_ood") == 1)
    test = test.filter(pl.col("label_modis_h1").is_not_null())
    print(f"  TEST: {test.shape[0]:,} rows, {test['district_id'].n_unique()} districts", flush=True)
    print(f"  fire_today: {test['fire_today'].sum()}", flush=True)
    print(f"  fire_h1: {test['fire_h1'].sum()}", flush=True)
    print(f"  label_modis_h1 (training label): {test['label_modis_h1'].sum()}", flush=True)

    # 6. Predict with v9.1 model (in chunks)
    print("\n[6] Running v9.1 predictions on TEST...", flush=True)
    model = gbm.Booster()
    model.load_model(MODEL_PATH)
    iso = joblib.load(ISO_PATH)
    feats_present = [f for f in FEATURES if f in test.columns]
    print(f"  Using {len(feats_present)} features", flush=True)

    BATCH = 30000
    test_pd = test.to_pandas()
    print(f"  Converted to pandas: {len(test_pd):,} rows", flush=True)
    del test, panel
    gc.collect()

    all_p = []
    for i in range(0, len(test_pd), BATCH):
        if i % (BATCH * 5) == 0:
            print(f"  Batch {i//BATCH + 1}/{(len(test_pd)+BATCH-1)//BATCH}", flush=True)
        X = test_pd.iloc[i:i+BATCH][feats_present].values.astype(np.float32)
        dmat = gbm.DMatrix(X)
        p = model.predict(dmat)
        p_cal = iso.predict(p)
        all_p.extend(p_cal.tolist())
        del X, dmat, p
        gc.collect()
    test_pd["p_pred"] = all_p
    print(f"  Predictions done. p range: {test_pd['p_pred'].min():.4f} - {test_pd['p_pred'].max():.4f}, mean: {test_pd['p_pred'].mean():.4f}", flush=True)

    # 7. Compute metrics
    print("\n[7] Computing metrics...", flush=True)
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y_true = test_pd["fire_h1"].values
    p_pred = test_pd["p_pred"].values
    n_pos = int(y_true.sum())
    n = len(y_true)

    if n_pos > 0 and n_pos < n:
        auc = roc_auc_score(y_true, p_pred)
        aupr = average_precision_score(y_true, p_pred)
        brier = brier_score_loss(y_true, p_pred)
    else:
        auc = aupr = brier = float("nan")
    print(f"  TEST AUROC: {auc:.4f}", flush=True)
    print(f"  TEST AUPR:  {aupr:.4f}", flush=True)
    print(f"  TEST Brier: {brier:.4f}", flush=True)

    # Precision@K, Recall@K
    sorted_idx = np.argsort(-p_pred)
    print(f"\n  {'K%':<6} {'K':>8} {'P@K':>10} {'R@K':>10}", flush=True)
    pk_results = []
    for k_pct in [0.5, 1, 2, 5, 10, 20]:
        k = max(1, int(n * k_pct / 100))
        tp = int(y_true[sorted_idx[:k]].sum())
        prec = tp / k
        rec = tp / n_pos if n_pos > 0 else 0
        print(f"  {k_pct:<5} {k:>8} {prec:>10.4f} {rec:>10.4f}", flush=True)
        pk_results.append({"k_pct": k_pct, "k": k, "precision": prec, "recall": rec})

    # 8. Reliability diagram
    print("\n[8] Reliability diagram...", flush=True)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    bin_pred, bin_freq = [], []
    for i in range(n_bins):
        mask = (p_pred >= bin_edges[i]) & (p_pred < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_pred.append(float(p_pred[mask].mean()))
        bin_freq.append(float(y_true[mask].mean()))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.plot(bin_pred, bin_freq, "o-", color="#059669", label=f"v9.1 (Brier={brier:.4f})")
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed fire frequency")
    ax.set_title(f"Reliability diagram — TEST (South, cold-start OOD)\n{len(test_pd):,} district-days, {n_pos} actual fires")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{OUT_DIR}/reliability_diagram.png", dpi=120, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR}/reliability_diagram.png", flush=True)

    # 9. Top/bottom 10 district-days
    print("\n[9] Top 10 highest-p district-days:", flush=True)
    print(f"  {'Rank':<5} {'District':<25} {'Date':<12} {'p':>8} {'Fire h1':>10}", flush=True)
    top10 = []
    for rank, idx in enumerate(sorted_idx[:10], 1):
        row = test_pd.iloc[idx]
        did = int(row["district_id"])
        meta = district_meta.get(did, {"name": "?", "province": "?"})
        line = f"  {rank:<5} {(meta['name']+', '+meta['province'])[:25]:<25} {row['date'].strftime('%Y-%m-%d'):<12} {row['p_pred']:>8.4f} {int(row['fire_h1']):>10}"
        print(line, flush=True)
        top10.append({
            "rank": rank,
            "district": meta["name"],
            "province": meta["province"],
            "date": str(row["date"].strftime("%Y-%m-%d")),
            "p": float(row["p_pred"]),
            "fire_h1": int(row["fire_h1"]),
        })

    # 10. Save summary
    summary = {
        "audit_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": {
            "firms_records": int(len(firms)),
            "firms_mapped_to_districts": int((firms["district_id"] != -1).sum()),
            "districts_with_fires": int(fire_counts["district_id"].nunique()),
        },
        "model": "v9.1 forest fire (74 features, with realistic NDVI seasonal model)",
        "test_set": {
            "rows": int(len(test_pd)),
            "districts": int(test_pd["district_id"].nunique()),
            "actual_fires_h1": int(n_pos),
            "fire_rate": float(y_true.mean()),
            "label_modis_h1_count": int(test_pd["label_modis_h1"].sum()),
            "note": "fire_h1 = next-day actual fire (from FIRMS), compared with v9.1 predicted p",
        },
        "metrics_full_test": {
            "AUROC": float(auc),
            "AUPR": float(aupr),
            "Brier": float(brier),
        },
        "precision_recall_at_k": pk_results,
        "top_10_highest_p": top10,
        "calibration": {
            "bins": [{"pred_mean": p, "obs_freq": f} for p, f in zip(bin_pred, bin_freq)],
        },
    }
    with open(f"{OUT_DIR}/firms_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT_DIR}/firms_audit_summary.json", flush=True)
    print(f"\n=== Total elapsed: {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
