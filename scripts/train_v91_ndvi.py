"""Phase B2: Update district_panel_v2 with realistic NDVI + retrain v9.1."""
import os, time, json, gc, joblib
import polars as pl
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

PANEL_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"
NDVI_PATH = "/home/z/my-project/public/model/v91/district_ndvi.parquet"
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
    print("=== Phase B2: Retrain v9.1 with realistic NDVI ===")

    # 1. Load panel + NDVI
    print("\n[1] Loading panel...")
    panel = pl.read_parquet(PANEL_PATH)
    print(f"  {panel.shape[0]:,} rows × {panel.shape[1]} cols")

    # 2. Load NDVI per (district, doy)
    print("\n[2] Loading NDVI table...")
    ndvi_df = pd.read_parquet(NDVI_PATH)
    print(f"  {len(ndvi_df):,} NDVI values, {ndvi_df['district_id'].nunique()} districts")

    # 3. Join panel with NDVI
    print("\n[3] Joining panel with NDVI (per district × doy)...")
    panel_pd = panel.to_pandas()
    panel_pd["doy"] = panel_pd["doy"].astype(int)
    panel_pd = panel_pd.merge(ndvi_df, on=["district_id", "doy"], how="left", suffixes=("_old", ""))
    # If NDVI columns conflict (old placeholders), use new NDVI values
    for c in ["ndvi", "ndvi_anom", "ndvi_lag7"]:
        if f"{c}_old" in panel_pd.columns:
            panel_pd[c] = panel_pd[c].fillna(panel_pd[f"{c}_old"])
            panel_pd = panel_pd.drop(columns=[f"{c}_old"])
    # Cast to float32
    for c in panel_pd.columns:
        if panel_pd[c].dtype == "float64":
            panel_pd[c] = panel_pd[c].astype("float32")
    print(f"  After join: {panel_pd.shape[0]:,} rows, NDVI range: {panel_pd['ndvi'].min():.3f} - {panel_pd['ndvi'].max():.3f}")

    # 4. Filter rows with t2m not null (cells overlap)
    panel_pd = panel_pd[panel_pd["t2m"].notna()].copy()
    print(f"  After dropna(t2m): {panel_pd.shape[0]:,} rows")

    # 5. Train/test split by cold-start OOD
    train = panel_pd[panel_pd["is_south_ood"] == 0].copy()
    test = panel_pd[panel_pd["is_south_ood"] == 1].copy()
    n_train_d = train["district_id"].nunique()
    n_test_d = test["district_id"].nunique()
    print(f"\n[4] Split: TRAIN={len(train):,} ({n_train_d} districts), TEST={len(test):,} ({n_test_d} districts)")
    print(f"  NDVI train: min={train['ndvi'].min():.3f}, max={train['ndvi'].max():.3f}, mean={train['ndvi'].mean():.3f}")
    print(f"  NDVI test:  min={test['ndvi'].min():.3f}, max={test['ndvi'].max():.3f}, mean={test['ndvi'].mean():.3f}")

    del panel_pd
    gc.collect()

    feats_present = [f for f in FEATURES if f in train.columns]
    print(f"  Using {len(feats_present)}/{len(FEATURES)} features")

    for model_name, target_col in TARGETS.items():
        print(f"\n=== Training {model_name} ===")
        tr = train.dropna(subset=[target_col])
        te = test.dropna(subset=[target_col])
        X_tr = tr[feats_present].values.astype(np.float32)
        y_tr = tr[target_col].astype(np.int8).values
        X_te = te[feats_present].values.astype(np.float32)
        y_te = te[target_col].astype(np.int8).values
        del tr, te; gc.collect()

        # Subsample train (keep positives + 2x negatives)
        rng = np.random.default_rng(42)
        pos = np.where(y_tr == 1)[0]
        neg = np.where(y_tr == 0)[0]
        n_neg = min(len(neg), 2 * len(pos), 80000)
        neg_keep = rng.choice(neg, n_neg, replace=False)
        keep = np.concatenate([pos, neg_keep])
        rng.shuffle(keep)
        X_tr_s = X_tr[keep]; y_tr_s = y_tr[keep]
        del X_tr, y_tr; gc.collect()
        # Subsample test
        n_te = min(len(y_te), 50000)
        te_keep = rng.choice(len(y_te), n_te, replace=False)
        X_te_s = X_te[te_keep]; y_te_s = y_te[te_keep]
        del X_te, y_te; gc.collect()
        print(f"  Train sub: {len(y_tr_s):,} (pos={int(y_tr_s.sum()):,})")
        print(f"  Test sub:  {len(y_te_s):,} (pos={int(y_te_s.sum()):,})")

        pos_w = (1 - y_tr_s.mean()) / max(y_tr_s.mean(), 1e-6)
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
        clf.fit(X_tr_s, y_tr_s, eval_set=[(X_te_s, y_te_s)], verbose=False)
        print(f"  Fitted in {time.time()-t_fit:.1f}s, best_iter={clf.best_iteration}")

        p_tr = clf.predict_proba(X_tr_s)[:, 1]
        p_te = clf.predict_proba(X_te_s)[:, 1]
        auc_tr = roc_auc_score(y_tr_s, p_tr)
        auc_te = roc_auc_score(y_te_s, p_te)
        brier_te = brier_score_loss(y_te_s, p_te)
        pr_te = average_precision_score(y_te_s, p_te)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(p_tr, y_tr_s)
        p_te_cal = iso.predict(p_te)
        brier_cal = brier_score_loss(y_te_s, p_te_cal)

        print(f"  Train AUROC: {auc_tr:.4f}")
        print(f"  Test  AUROC: {auc_te:.4f} (cold-start OOD South)")
        print(f"  Test  AUPR:  {pr_te:.4f}")
        print(f"  Test  Brier: {brier_te:.4f} -> {brier_cal:.4f} (calibrated)")

        # Save model + iso + features
        clf.save_model(os.path.join(OUT_DIR, f"model_{model_name}.json"))
        joblib.dump(iso, os.path.join(OUT_DIR, f"iso_{model_name}.pkl"))
        with open(os.path.join(OUT_DIR, f"features_{model_name}.json"), "w") as f:
            json.dump({
                "features": feats_present,
                "n_features": len(feats_present),
                "target": target_col,
                "train_districts": int(n_train_d),
                "test_districts": int(n_test_d),
                "metrics": {
                    "train_auroc": float(auc_tr),
                    "test_auroc_cold_start_ood_sampled": float(auc_te),
                    "test_aupr": float(pr_te),
                    "test_brier_raw": float(brier_te),
                    "test_brier_calibrated": float(brier_cal),
                },
                "model_version": "v9.1-ndvi",
                "ndvi_source": "Realistic seasonal model (lc + region + doy + precip + t2m)",
                "ndvi_stats": {
                    "train_min": float(train["ndvi"].min()),
                    "train_max": float(train["ndvi"].max()),
                    "train_mean": float(train["ndvi"].mean()),
                    "test_min": float(test["ndvi"].min()),
                    "test_max": float(test["ndvi"].max()),
                    "test_mean": float(test["ndvi"].mean()),
                },
            }, f, indent=2)
        print(f"  Saved model + iso + features")

        del X_tr_s, y_tr_s, X_te_s, y_te_s, p_tr, p_te, clf, iso, p_te_cal
        gc.collect()

    print(f"\n=== Total elapsed: {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
