"""Phase B4: Compare audit metrics — strict (point-in-polygon) vs aggregated (cell-overlap) labels."""
import polars as pl
import pandas as pd
import numpy as np
import os, time, json
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as gbm
import joblib

PANEL_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"
NDVI_PATH = "/home/z/my-project/public/model/v91/district_ndvi.parquet"
MODEL_PATH = "/home/z/my-project/public/model/v91/model_forest_fire.json"
ISO_PATH = "/home/z/my-project/public/model/v91/iso_forest_fire.pkl"

FEATURES = [
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
    "ndvi","ndvi_anom","ndvi_lag7",
    "cell_fire_rate_exp","cell_fire_count_30d","is_dry","cell_fire_rate_dry_exp",
    "centroid_lat","centroid_lon",
]


def main():
    t0 = time.time()
    print("=== Phase B4: Audit comparison: strict vs aggregated labels ===", flush=True)

    # Load panel + NDVI
    needed = ["district_id","date","is_south_ood","label_modis_h1","label_union_h1","fire_modis"] + FEATURES
    panel = pl.read_parquet(PANEL_PATH, columns=[c for c in needed if c != "ndvi"])
    cast = {c: pl.Float32 for c in panel.columns if panel[c].dtype == pl.Float64}
    panel = panel.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast.items()])
    panel = panel.filter(pl.col("t2m").is_not_null())
    ndvi = pl.read_parquet(NDVI_PATH)
    panel = panel.join(ndvi.select(["district_id","doy","ndvi","ndvi_anom","ndvi_lag7"]),
                       on=["district_id","doy"], how="left")
    panel = panel.with_columns([
        pl.col("ndvi").fill_null(0.4),
        pl.col("ndvi_anom").fill_null(0.0),
        pl.col("ndvi_lag7").fill_null(0.4),
    ])
    print(f"  Panel: {panel.shape[0]:,} rows", flush=True)

    # Filter to TEST (South)
    test = panel.filter(pl.col("is_south_ood") == 1).filter(pl.col("label_modis_h1").is_not_null())
    print(f"  TEST: {test.shape[0]:,} rows, {test['district_id'].n_unique()} districts", flush=True)

    # Predict
    model = gbm.Booster(); model.load_model(MODEL_PATH)
    iso = joblib.load(ISO_PATH)
    feats = [f for f in FEATURES if f in test.columns]
    test_pd = test.to_pandas()
    print(f"  → pandas: {len(test_pd):,} rows", flush=True)
    del panel, test
    import gc; gc.collect()

    # Predict in batches
    all_p = []
    BATCH = 30000
    for i in range(0, len(test_pd), BATCH):
        X = test_pd.iloc[i:i+BATCH][feats].values.astype(np.float32)
        dmat = gbm.DMatrix(X)
        p = model.predict(dmat)
        p_cal = iso.predict(p)
        all_p.extend(p_cal.tolist())
        del X, dmat, p; gc.collect()
    test_pd["p_pred"] = all_p
    print(f"  Predictions done. p mean={test_pd['p_pred'].mean():.4f}", flush=True)

    # Compute metrics on BOTH label types
    print(f"\n=== AUDIT COMPARISON ===", flush=True)
    print(f"  Test rows: {len(test_pd):,}", flush=True)
    print(f"  Test districts: {test_pd['district_id'].nunique()}", flush=True)
    print(f"  Test date range: {test_pd['date'].min()} → {test_pd['date'].max()}", flush=True)

    # 1. Aggregated (training labels — fire in any cell overlapping district)
    y_agg = test_pd["label_modis_h1"].astype(int).values
    p = test_pd["p_pred"].values
    n_agg = int(y_agg.sum())
    print(f"\n  [AGGREGATED labels — training definition]", flush=True)
    print(f"  fire events: {n_agg} ({100*n_agg/len(y_agg):.2f}%)", flush=True)
    print(f"  AUROC: {roc_auc_score(y_agg, p):.4f}", flush=True)
    print(f"  AUPR:  {average_precision_score(y_agg, p):.4f}", flush=True)
    print(f"  Brier: {brier_score_loss(y_agg, p):.4f}", flush=True)

    # P@K on aggregated
    sorted_idx = np.argsort(-p)
    print(f"  P@K / R@K:", flush=True)
    for k_pct in [0.5, 1, 5, 10]:
        k = max(1, int(len(y_agg) * k_pct / 100))
        tp = int(y_agg[sorted_idx[:k]].sum())
        print(f"    top {k_pct}% ({k}): P={tp/k:.4f}, R={tp/n_agg:.4f}", flush=True)

    # 2. Union (any fire) labels
    y_union = test_pd["label_union_h1"].astype(int).values
    n_union = int(y_union.sum())
    print(f"\n  [UNION labels — any fire]", flush=True)
    print(f"  fire events: {n_union} ({100*n_union/len(y_union):.2f}%)", flush=True)
    print(f"  AUROC: {roc_auc_score(y_union, p):.4f}", flush=True)
    print(f"  AUPR:  {average_precision_score(y_union, p):.4f}", flush=True)
    print(f"  Brier: {brier_score_loss(y_union, p):.4f}", flush=True)

    # 3. Fire count > 0 (any MODIS detection today = "fire today")
    if "fire_modis" in test_pd.columns:
        y_today = (test_pd["fire_modis"] > 0).astype(int).values
        n_today = int(y_today.sum())
        if n_today > 0 and n_today < len(y_today):
            print(f"\n  [FIRE TODAY — any MODIS detection]", flush=True)
            print(f"  fire events: {n_today} ({100*n_today/len(y_today):.2f}%)", flush=True)
            print(f"  AUROC: {roc_auc_score(y_today, p):.4f}", flush=True)
            print(f"  AUPR:  {average_precision_score(y_today, p):.4f}", flush=True)

    # Summary
    summary = {
        "audit_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_set": {
            "rows": int(len(test_pd)),
            "districts": int(test_pd["district_id"].nunique()),
            "date_min": str(test_pd["date"].min()),
            "date_max": str(test_pd["date"].max()),
        },
        "model": "v9.1 forest fire (74 features + realistic NDVI)",
        "audit_results": [
            {
                "label_type": "AGGREGATED (label_modis_h1) — training definition",
                "definition": "fire in any cell overlapping district within next 24h",
                "n_positive": n_agg,
                "pos_rate": float(n_agg / len(y_agg)),
                "AUROC": float(roc_auc_score(y_agg, p)),
                "AUPR": float(average_precision_score(y_agg, p)),
                "Brier": float(brier_score_loss(y_agg, p)),
                "precision_at_k": [],  # computed below
            },
            {
                "label_type": "UNION (label_union_h1) — any fire",
                "definition": "any MODIS detection in district within next 24h",
                "n_positive": n_union,
                "pos_rate": float(n_union / len(y_union)),
                "AUROC": float(roc_auc_score(y_union, p)),
                "AUPR": float(average_precision_score(y_union, p)),
                "Brier": float(brier_score_loss(y_union, p)),
            },
        ],
    }
    # Compute precision_at_k properly
    pa = summary["audit_results"][0]["precision_at_k"]
    srt = np.argsort(-p)
    for k_pct in [0.5, 1, 5, 10]:
        k = max(1, int(len(y_agg) * k_pct / 100))
        tp = int(y_agg[srt[:k]].sum())
        pa.append({
            "k_pct": k_pct, "k": k,
            "precision": float(tp / k),
            "recall": float(tp / n_agg) if n_agg > 0 else 0.0,
        })
    out_path = "/home/z/my-project/download/audit/audit_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}", flush=True)
    print(f"\n=== Total elapsed: {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
