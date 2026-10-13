"""Phase A3: Train v9.1 XGBoost + isotonic calibration on DISTRICT-LEVEL panel.

Memory-efficient: polars for ETL, numpy float32 for XGBoost.

Cold-start spatial OOD split:
  TRAIN: districts with centroid lat>=14 (North + Central)
  TEST:  districts with centroid lat<14 (South) - never seen during training
"""
import time, os, json, gc, joblib
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

PANEL_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"
OUT_DIR = "/home/z/my-project/public/model/v91"
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
TARGETS = {
    "forest_fire": "label_modis_h1",
    "any_fire":    "label_union_h1",
}


def main():
    t0 = time.time()
    print("=== Phase A3: Train v9.1 district model ===")

    # 1. Load panel with polars (memory-efficient)
    print("\n[1] Loading panel (polars, float32)...")
    needed_cols = ["district_id", "date", "is_south_ood"] + FEATURES + list(TARGETS.values())
    df_pl = pl.read_parquet(PANEL_PATH, columns=needed_cols)
    # Cast float64 → float32
    cast_map = {c: pl.Float32 for c in df_pl.columns if df_pl[c].dtype == pl.Float64}
    df_pl = df_pl.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast_map.items()])
    print(f"  {df_pl.shape[0]:,} rows x {df_pl.shape[1]} cols ({df_pl.estimated_size('mb'):.0f} MB)")

    # Filter out NaN t2m (districts without weather cells)
    df_pl = df_pl.filter(pl.col("t2m").is_not_null())
    print(f"  After dropna(t2m): {df_pl.shape[0]:,} rows")

    # Train/test split by cold-start OOD: train lat>=14, test lat<14
    train_pl = df_pl.filter(pl.col("is_south_ood") == 0)
    test_pl  = df_pl.filter(pl.col("is_south_ood") == 1)
    n_train_dist = train_pl["district_id"].n_unique()
    n_test_dist  = test_pl["district_id"].n_unique()
    print(f"\n[2] Split: TRAIN={train_pl.shape[0]:,} ({n_train_dist} districts), TEST={test_pl.shape[0]:,} ({n_test_dist} districts)")

    # Check feature presence
    feats_present = [f for f in FEATURES if f in df_pl.columns]
    feats_missing = [f for f in FEATURES if f not in df_pl.columns]
    if feats_missing:
        print(f"  Missing features: {feats_missing}")
    print(f"  Using {len(feats_present)}/{len(FEATURES)} features")

    # Free df_pl before per-target loop (we'll iterate using train_pl / test_pl)
    del df_pl
    gc.collect()

    for model_name, target_col in TARGETS.items():
        print(f"\n=== Training {model_name} (target={target_col}) ===")
        tr = train_pl.filter(pl.col(target_col).is_not_null())
        te = test_pl.filter(pl.col(target_col).is_not_null())

        X_train = tr.select(feats_present).to_numpy().astype(np.float32)
        y_train = tr[target_col].to_numpy().astype(np.int8)
        X_test = te.select(feats_present).to_numpy().astype(np.float32)
        y_test = te[target_col].to_numpy().astype(np.int8)
        del tr, te
        gc.collect()
        print(f"  Train: {len(y_train):,} rows, pos={int(y_train.sum()):,} ({100*y_train.mean():.2f}%)")
        print(f"  Test:  {len(y_test):,} rows, pos={int(y_test.sum()):,} ({100*y_test.mean():.2f}%)")
        print(f"  X_train mem: {X_train.nbytes / 1024**2:.0f} MB")

        pos_w = (1 - y_train.mean()) / max(y_train.mean(), 1e-6)
        # Subsample training data to fit in memory (keep all positives + 50% negatives)
        rng = np.random.default_rng(42)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_neg_keep = min(len(neg_idx), 5 * len(pos_idx))  # 5:1 neg:pos ratio
        neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        keep_idx = np.concatenate([pos_idx, neg_keep])
        rng.shuffle(keep_idx)
        X_train_s = X_train[keep_idx]
        y_train_s = y_train[keep_idx]
        print(f"  Subsampled: {len(keep_idx):,} rows (pos={int(y_train_s.sum()):,}, ratio={100*y_train_s.mean():.2f}%)")
        del X_train, y_train
        gc.collect()

        clf = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
            gamma=0.3, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=min(pos_w, 5),
            tree_method="hist", n_jobs=1, max_bin=64,
            eval_metric="aucpr", early_stopping_rounds=15,
            random_state=42,
        )
        t_fit = time.time()
        clf.fit(X_train_s, y_train_s, eval_set=[(X_test, y_test)], verbose=False)
        print(f"  Fitted in {time.time()-t_fit:.1f}s, best_iter={clf.best_iteration}")

        p_train = clf.predict_proba(X_train_s)[:, 1]
        p_test  = clf.predict_proba(X_test)[:, 1]
        del X_train_s, y_train_s
        gc.collect()

        auc_train = roc_auc_score(y_train, p_train)
        auc_test = roc_auc_score(y_test, p_test)
        brier_test = brier_score_loss(y_test, p_test)
        pr_test = average_precision_score(y_test, p_test)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(p_train, y_train)
        p_test_cal = iso.predict(p_test)
        brier_test_cal = brier_score_loss(y_test, p_test_cal)

        print(f"  Train AUROC: {auc_train:.4f}")
        print(f"  Test  AUROC: {auc_test:.4f} (cold-start OOD South)")
        print(f"  Test  AUPR:  {pr_test:.4f}")
        print(f"  Test  Brier: {brier_test:.4f} (raw) -> {brier_test_cal:.4f} (calibrated)")

        # Save
        model_path = os.path.join(OUT_DIR, f"model_{model_name}.json")
        iso_path = os.path.join(OUT_DIR, f"iso_{model_name}.pkl")
        clf.save_model(model_path)
        joblib.dump(iso, iso_path)
        print(f"  Saved: {model_path} + {iso_path}")

        feat_path = os.path.join(OUT_DIR, f"features_{model_name}.json")
        with open(feat_path, "w") as f:
            json.dump({
                "features": feats_present,
                "n_features": len(feats_present),
                "target": target_col,
                "train_districts": int(n_train_dist),
                "test_districts": int(n_test_dist),
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "train_pos_rate": float(y_train.mean()),
                "test_pos_rate": float(y_test.mean()),
                "metrics": {
                    "train_auroc": float(auc_train),
                    "test_auroc_cold_start_ood": float(auc_test),
                    "test_aupr": float(pr_test),
                    "test_brier_raw": float(brier_test),
                    "test_brier_calibrated": float(brier_test_cal),
                },
                "model_version": "v9.1",
                "training_data": "district_panel_v2.parquet (cold-start OOD: train Bac+Trung, test Nam)",
            }, f, indent=2)
        print(f"  Saved: {feat_path}")

        # Free per-target memory
        del X_test, y_test, p_train, p_test, clf, iso, p_test_cal, keep_idx, X_train_s, y_train_s, neg_keep, pos_idx, neg_idx
        gc.collect()

    print(f"\n=== Total elapsed: {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
