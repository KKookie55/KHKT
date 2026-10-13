"""Phase C2: Re-audit FIRMS with v9.2 model (fixed calibration + interaction features)."""
import os, time, json, gc
import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as gbm
import joblib

PANEL_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"
NDVI_PATH = "/home/z/my-project/public/model/v91/district_ndvi.parquet"
MODEL_PATH = "/home/z/my-project/public/model/v92/model_forest_fire.json"
ISO_PATH = "/home/z/my-project/public/model/v92/iso_forest_fire.pkl"
OUT_DIR = "/home/z/my-project/download/audit"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_FEATURES = [
    "t2m","rh","wind","precip","doy","t2m_lag1","t2m_lag3","t2m_lag7","t2m_lag14",
    "rh_lag1","rh_lag3","rh_lag7","rh_lag14","wind_lag1","wind_lag3","wind_lag7","wind_lag14",
    "precip_lag1","precip_lag3","precip_lag7","precip_lag14","t2m_roll7","t2m_roll14","rh_roll7","rh_roll14",
    "precip_roll7_sum","precip_roll14_sum","precip_roll30_sum","days_since_rain","t2m_diff3","rh_diff3","wind_diff3",
    "fwi_compound","heat_index","wind_chill","t2m_anom_doy","rh_anom_doy","precip_anom_doy","wind_anom_doy",
    "t2m_anom_roll7","rh_anom_roll7","precip_anom_roll7","dry_anomaly_composite",
    "fwi","fwi_ffmc","fwi_dmc","fwi_dc","fwi_isi","fwi_bui",
    "forest_prop","shrubland_prop","grassland_prop","agriculture_prop","urban_prop","bare_prop","water_prop",
    "elevation_mean","elevation_std","slope_mean","koppen_Cwa","koppen_Am","koppen_Aw",
    "latitude_band","coastal_distance","population_density_proxy",
    "ndvi","ndvi_lag7",
    "cell_fire_rate_exp","cell_fire_count_30d","is_dry","cell_fire_rate_dry_exp",
    "centroid_lat","centroid_lon",
]
INTERACTIONS = ["wind_temp", "humidity_temp", "dry_wind", "heat_drought", "ndvi_anom"]
FEATURES = BASE_FEATURES + INTERACTIONS


def add_features(df):
    df = df.copy()
    df["wind_temp"] = df["wind"] * df["t2m"]
    df["humidity_temp"] = df["rh"] * df["t2m"]
    df["dry_wind"] = df["dry_anomaly_composite"] * df["wind"]
    df["heat_drought"] = df["t2m_anom_doy"] * df["days_since_rain"]
    df["ndvi_anom"] = df["ndvi"] - df.groupby("district_id")["ndvi"].transform("mean")
    return df


def main():
    t0 = time.time()
    print("=== Phase C2: Re-audit FIRMS with v9.2 ===", flush=True)

    # Load panel + NDVI
    print("\n[1] Loading panel + NDVI...", flush=True)
    needed = ["district_id","date","is_south_ood","label_modis_h1","label_union_h1","fire_modis"] + BASE_FEATURES
    panel = pl.read_parquet(PANEL_PATH, columns=[c for c in needed if c != "ndvi"])
    cast = {c: pl.Float32 for c in panel.columns if panel[c].dtype == pl.Float64}
    panel = panel.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast.items()])
    panel = panel.filter(pl.col("t2m").is_not_null())
    ndvi = pl.read_parquet(NDVI_PATH)
    panel = panel.join(ndvi.select(["district_id","doy","ndvi","ndvi_lag7"]),
                       on=["district_id","doy"], how="left")
    panel = panel.with_columns([
        pl.col("ndvi").fill_null(0.4),
        pl.col("ndvi_lag7").fill_null(0.4),
    ])
    print(f"  Panel: {panel.shape[0]:,} rows", flush=True)

    # Filter to TEST (South, cold-start OOD)
    test = panel.filter(pl.col("is_south_ood") == 1).filter(pl.col("label_modis_h1").is_not_null())
    test_pd = test.to_pandas()
    del panel, test; gc.collect()
    print(f"  TEST: {len(test_pd):,} rows, {test_pd['district_id'].nunique()} districts", flush=True)

    # Add interaction features
    print("\n[2] Adding interaction features...", flush=True)
    test_pd = add_features(test_pd)
    feats = [f for f in FEATURES if f in test_pd.columns]
    print(f"  Using {len(feats)} features", flush=True)

    # Predict with v9.2
    print("\n[3] Predicting with v9.2...", flush=True)
    model = gbm.Booster(); model.load_model(MODEL_PATH)
    iso = joblib.load(ISO_PATH)
    all_p = []
    BATCH = 30000
    for i in range(0, len(test_pd), BATCH):
        if i % (BATCH * 4) == 0:
            print(f"  Batch {i//BATCH + 1}/{(len(test_pd)+BATCH-1)//BATCH}", flush=True)
        X = test_pd.iloc[i:i+BATCH][feats].values.astype(np.float32)
        dmat = gbm.DMatrix(X)
        p = model.predict(dmat)
        p_cal = iso.predict(p)
        all_p.extend(p_cal.tolist())
        del X, dmat, p; gc.collect()
    test_pd["p_pred"] = all_p
    print(f"  Predictions done. p mean={test_pd['p_pred'].mean():.4f}, max={test_pd['p_pred'].max():.4f}", flush=True)

    # Audit on AGGREGATED labels
    print(f"\n[4] Audit results (v9.2 vs v9.1 comparison):", flush=True)
    y_agg = test_pd["label_modis_h1"].astype(int).values
    p = test_pd["p_pred"].values
    n = len(y_agg); n_pos = int(y_agg.sum())

    auc_v92 = roc_auc_score(y_agg, p)
    aupr_v92 = average_precision_score(y_agg, p)
    brier_v92 = brier_score_loss(y_agg, p)
    print(f"  [AGGREGATED labels] n_pos={n_pos:,} ({100*n_pos/n:.2f}%)", flush=True)
    print(f"  v9.2 AUROC: {auc_v92:.4f}  (v9.1 was 0.9313)", flush=True)
    print(f"  v9.2 AUPR:  {aupr_v92:.4f}  (v9.1 was 0.607)", flush=True)
    print(f"  v9.2 Brier: {brier_v92:.4f}  (v9.1 was 0.113)", flush=True)

    # Precision@K
    sorted_idx = np.argsort(-p)
    print(f"\n  P@K / R@K (v9.2):", flush=True)
    pk_results = []
    for k_pct in [0.5, 1, 2, 5, 10, 20]:
        k = max(1, int(n * k_pct / 100))
        tp = int(y_agg[sorted_idx[:k]].sum())
        prec = tp / k
        rec = tp / n_pos if n_pos > 0 else 0
        print(f"    top {k_pct}% ({k}): P={prec:.4f}, R={rec:.4f}", flush=True)
        pk_results.append({"k_pct": k_pct, "k": k, "precision": prec, "recall": rec})

    # Reliability diagram
    print(f"\n[5] Reliability diagram (v9.2)...", flush=True)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    bin_pred, bin_freq = [], []
    for i in range(n_bins):
        mask = (p >= bin_edges[i]) & (p < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_pred.append(float(p[mask].mean()))
        bin_freq.append(float(y_agg[mask].mean()))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.plot(bin_pred, bin_freq, "o-", color="#059669", label=f"v9.2 (Brier={brier_v92:.4f})")
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed fire frequency")
    ax.set_title(f"Reliability diagram v9.2 — TEST (South, cold-start OOD)\n{len(test_pd):,} district-days, {n_pos:,} actual fires")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = f"{OUT_DIR}/reliability_diagram_v92.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Saved: {out_path}", flush=True)
    # Copy to public for web
    import shutil
    shutil.copy(out_path, "/home/z/my-project/public/reliability_diagram_v92.png")

    # Top 10
    print(f"\n[6] Top 10 highest-p district-days (v9.2):", flush=True)
    print(f"  {'Rank':<5} {'District':<25} {'Date':<12} {'p':>8} {'Fire h1':>10}", flush=True)
    top10 = []
    for rank, idx in enumerate(sorted_idx[:10], 1):
        row = test_pd.iloc[idx]
        did = int(row["district_id"])
        # Load district name from GADM
        import json
        with open("/home/z/my-project/public/data/geo/gadm41_VNM_2.json") as f:
            gadm = json.load(f)
        meta = gadm["features"][did]["properties"]
        name = meta.get("NAME_2", "?")
        prov = meta.get("NAME_1", "?")
        print(f"  {rank:<5} {(name+', '+prov)[:25]:<25} {str(row['date'])[:10]:<12} {row['p_pred']:>8.4f} {int(row['label_modis_h1']):>10}", flush=True)
        top10.append({"rank": rank, "district": name, "province": prov, "date": str(row['date'])[:10],
                      "p": float(row["p_pred"]), "fire_h1": int(row["label_modis_h1"])})

    # Save summary
    summary = {
        "audit_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "v9.2 (78 features, iso fit on validation, +interaction features)",
        "test_set": {"rows": int(n), "districts": int(test_pd["district_id"].nunique()),
                     "actual_fires": int(n_pos), "fire_rate": float(y_agg.mean())},
        "metrics": {"AUROC": float(auc_v92), "AUPR": float(aupr_v92), "Brier": float(brier_v92)},
        "vs_v91": {"AUROC": {"v91": 0.9313, "v92": float(auc_v92), "delta": float(auc_v92 - 0.9313)},
                   "AUPR":  {"v91": 0.607, "v92": float(aupr_v92), "delta": float(aupr_v92 - 0.607)},
                   "Brier": {"v91": 0.113, "v92": float(brier_v92), "delta": float(brier_v92 - 0.113)}},
        "precision_recall_at_k": pk_results,
        "top_10_highest_p": top10,
        "calibration": {"bins": [{"pred_mean": p, "obs_freq": f} for p, f in zip(bin_pred, bin_freq)]},
    }
    out_json = f"{OUT_DIR}/audit_v92.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_json}", flush=True)
    print(f"\n=== Elapsed: {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
