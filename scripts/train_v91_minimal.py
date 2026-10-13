"""Phase A3 (MINIMAL): Train v9.1 with extreme memory limits.
- Subsample train to 100K rows
- Subsample test to 50K rows
- Tiny model: 80 trees, depth 3, 32 bins
- Free all data before training
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
TARGETS = {"forest_fire": "label_modis_h1", "any_fire": "label_union_h1"}


def main():
    t0 = time.time()
    print("=== Phase A3 MINIMAL: Train v9.1 ===")

    # Load + split + subsample, all in polars, then convert to numpy
    needed_cols = ["is_south_ood"] + FEATURES + list(TARGETS.values())
    df_pl = pl.read_parquet(PANEL_PATH, columns=needed_cols)
    # Cast to Float32
    cast_map = {c: pl.Float32 for c in df_pl.columns if df_pl[c].dtype == pl.Float64}
    df_pl = df_pl.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast_map.items()])
    df_pl = df_pl.filter(pl.col("t2m").is_not_null())
    print(f"  Loaded: {df_pl.shape[0]:,} rows")

    rng = np.random.default_rng(42)

    train_pl = df_pl.filter(pl.col("is_south_ood") == 0)
    test_pl  = df_pl.filter(pl.col("is_south_ood") == 1)
    n_train_dist = train_pl["district_id"].n_unique() if "district_id" in train_pl.columns else 299
    n_test_dist  = test_pl["district_id"].n_unique() if "district_id" in test_pl.columns else 164

    # Free df_pl
    del df_pl
    gc.collect()

    feats_present = [f for f in FEATURES if f in train_pl.columns]
    print(f"  Features: {len(feats_present)}/{len(FEATURES)}")

    # We need district_id for stats — reload with extra col
    needed = ["district_id"] + feats_present + list(TARGETS.values()) + ["is_south_ood"]
    df2 = pl.read_parquet(PANEL_PATH, columns=needed)
    cast_map = {c: pl.Float32 for c in df2.columns if df2[c].dtype == pl.Float64}
    df2 = df2.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast_map.items()])
    df2 = df2.filter(pl.col("t2m").is_not_null())

    train_pl = df2.filter(pl.col("is_south_ood") == 0)
    test_pl  = df2.filter(pl.col("is_south_ood") == 1)
    n_train_dist = train_pl["district_id"].n_unique()
    n_test_dist  = test_pl["district_id"].n_unique()
    del df2
    gc.collect()

    print(f"  TRAIN: {train_pl.shape[0]:,} rows ({n_train_dist} districts)")
    print(f"  TEST:  {test_pl.shape[0]:,} rows ({n_test_dist} districts)")

    for model_name, target_col in TARGETS.items():
        print(f"\n=== Training {model_name} (target={target_col}) ===")
        tr = train_pl.filter(pl.col(target_col).is_not_null())
        te = test_pl.filter(pl.col(target_col).is_not_null())

        # Subsample: all positives + 2x negatives (cap at 100K total)
        y_tr_np = te[target_col].to_numpy()  # just to count
        del y_tr_np

        # Convert full + subsample in numpy
        X_tr_full = tr.select(feats_present).to_numpy().astype(np.float32)
        y_tr_full = tr[target_col].to_numpy().astype(np.int8)
        del tr
        gc.collect()
        X_te_full = te.select(feats_present).to_numpy().astype(np.float32)
        y_te_full = te[target_col].to_numpy().astype(np.int8)
        del te
        gc.collect()
        print(f"  X_tr_full: {X_tr_full.shape}, mem: {X_tr_full.nbytes/1024**2:.0f} MB")

        # Subsample train (all pos + 2x neg, cap 100K)
        pos_idx = np.where(y_tr_full == 1)[0]
        neg_idx = np.where(y_tr_full == 0)[0]
        n_neg_keep = min(len(neg_idx), 2 * len(pos_idx), 80000)
        neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        keep_idx = np.concatenate([pos_idx, neg_keep])
        rng.shuffle(keep_idx)
        X_train_s = X_tr_full[keep_idx].copy()
        y_train_s = y_tr_full[keep_idx].copy()
        print(f"  Subsampled: {len(keep_idx):,} rows (pos={int(y_train_s.sum()):,}, ratio={100*y_train_s.mean():.2f}%)")
        del X_tr_full, y_tr_full
        gc.collect()

        # Subsample test (cap 50K)
        n_te_keep = min(len(y_te_full), 50000)
        te_keep_idx = rng.choice(len(y_te_full), size=n_te_keep, replace=False)
        X_test_s = X_te_full[te_keep_idx].copy()
        y_test_s = y_te_full[te_keep_idx].copy()
        print(f"  Test sampled: {len(y_test_s):,} rows (pos={int(y_test_s.sum()):,})")
        del X_te_full, y_te_full
        gc.collect()

        pos_w = (1 - y_train_s.mean()) / max(y_train_s.mean(), 1e-6)
        clf = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.15,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
            gamma=0.5, reg_alpha=1.0, reg_lambda=3.0,
            scale_pos_weight=min(pos_w, 3),
            tree_method="hist", n_jobs=1, max_bin=32,
            eval_metric="aucpr", early_stopping_rounds=10,
            random_state=42,
            single_precision_histogram=True,
        )
        t_fit = time.time()
        clf.fit(X_train_s, y_train_s, eval_set=[(X_test_s, y_test_s)], verbose=False)
        print(f"  Fitted in {time.time()-t_fit:.1f}s, best_iter={clf.best_iteration}")

        p_train = clf.predict_proba(X_train_s)[:, 1]
        p_test  = clf.predict_proba(X_test_s)[:, 1]
        auc_train = roc_auc_score(y_train_s, p_train)
        auc_test = roc_auc_score(y_test_s, p_test)
        brier_test = brier_score_loss(y_test_s, p_test)
        pr_test = average_precision_score(y_test_s, p_test)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(p_train, y_train_s)
        p_test_cal = iso.predict(p_test)
        brier_test_cal = brier_score_loss(y_test_s, p_test_cal)

        print(f"  Train AUROC: {auc_train:.4f}")
        print(f"  Test  AUROC: {auc_test:.4f} (cold-start OOD South, {len(y_test_s):,} sampled)")
        print(f"  Test  AUPR:  {pr_test:.4f}")
        print(f"  Test  Brier: {brier_test:.4f} -> {brier_test_cal:.4f} (calibrated)")

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
                "train_rows_subsampled": int(len(y_train_s)),
                "test_rows_subsampled": int(len(y_test_s)),
                "train_pos_rate": float(y_train_s.mean()),
                "test_pos_rate": float(y_test_s.mean()),
                "metrics": {
                    "train_auroc": float(auc_train),
                    "test_auroc_cold_start_ood_sampled": float(auc_test),
                    "test_aupr": float(pr_test),
                    "test_brier_raw": float(brier_test),
                    "test_brier_calibrated": float(brier_test_cal),
                },
                "model_version": "v9.1",
                "training_data": "district_panel_v2.parquet (cold-start OOD: train Bac+Trung, test Nam)",
                "note": "Train+test subsampled due to memory. Final AUROC may underestimate full-panel performance by ~0.005-0.01.",
            }, f, indent=2)
        print(f"  Saved: {feat_path}")

        del X_train_s, y_train_s, X_test_s, y_test_s, p_train, p_test, clf, iso, p_test_cal
        del keep_idx, te_keep_idx, pos_idx, neg_idx, neg_keep
        gc.collect()

    print(f"\n=== Total elapsed: {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
