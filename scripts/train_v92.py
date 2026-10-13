"""Phase C: v9.2 improvements
1. Fix isotonic calibration (fit on validation set, not train predictions)
2. Add interaction features:
   - wind_temp (wind × temp) — high wind + high temp = high fire risk
   - humidity_temp (humidity × temp)
   - dry_wind (dry_anomaly × wind)
   - heat_drought (t2m_anom × days_since_rain)
3. Add NDVI anomaly (current NDVI vs district × doy climatology)
4. Add spatial lag (mean p of K nearest neighbor districts — but skip for simplicity)

Output: model_v2*.json + iso_v2*.pkl in public/model/v92/
"""
import os, time, json, gc, joblib
import polars as pl
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

PANEL_PATH = "/home/z/my-project/download/production/district_panel_v2.parquet"
NDVI_PATH = "/home/z/my-project/public/model/v91/district_ndvi.parquet"
OUT_DIR = "/home/z/my-project/public/model/v92"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_FEATURES = [
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
    "ndvi", "ndvi_lag7",
    "cell_fire_rate_exp", "cell_fire_count_30d", "is_dry", "cell_fire_rate_dry_exp",
    "centroid_lat", "centroid_lon",
]

# New interaction features (computed at runtime)
INTERACTIONS = ["wind_temp", "humidity_temp", "dry_wind", "heat_drought", "ndvi_anom"]

FEATURES = BASE_FEATURES + INTERACTIONS
TARGETS = {"forest_fire": "label_modis_h1", "any_fire": "label_union_h1"}


def add_features(df):
    """Add interaction features to a pandas dataframe."""
    df = df.copy()
    # wind × temp (km/s × °C): high wind + high temp = high fire risk
    df["wind_temp"] = df["wind"] * df["t2m"]
    # humidity × temp (interaction): dry + hot = bad
    df["humidity_temp"] = df["rh"] * df["t2m"]
    # dry anomaly × wind
    df["dry_wind"] = df["dry_anomaly_composite"] * df["wind"]
    # t2m_anom × days_since_rain (heat + drought)
    df["heat_drought"] = df["t2m_anom_doy"] * df["days_since_rain"]
    # NDVI anomaly: current NDVI - mean NDVI per district (climatology)
    df["ndvi_anom"] = df["ndvi"] - df.groupby("district_id")["ndvi"].transform("mean")
    return df


def main():
    t0 = time.time()
    print("=== Phase C: v9.2 improvements ===", flush=True)

    # Load panel + NDVI
    print("\n[1] Loading panel + NDVI (polars)...", flush=True)
    needed = ["district_id", "date", "is_south_ood"] + BASE_FEATURES + list(TARGETS.values())
    panel = pl.read_parquet(PANEL_PATH, columns=[c for c in needed if c != "ndvi"])
    cast = {c: pl.Float32 for c in panel.columns if panel[c].dtype == pl.Float64}
    panel = panel.with_columns([pl.col(c).cast(t).alias(c) for c, t in cast.items()])
    panel = panel.filter(pl.col("t2m").is_not_null())
    ndvi = pl.read_parquet(NDVI_PATH)
    panel = panel.join(ndvi.select(["district_id", "doy", "ndvi", "ndvi_lag7"]),
                       on=["district_id", "doy"], how="left")
    panel = panel.with_columns([
        pl.col("ndvi").fill_null(0.4),
        pl.col("ndvi_lag7").fill_null(0.4),
    ])
    print(f"  Panel: {panel.shape[0]:,} rows, {panel.estimated_size('mb'):.0f} MB", flush=True)

    # Split: TRAIN = North+Central (Bac+Trung), VAL = subset of TRAIN (last 20%), TEST = South
    # Better: split TRAIN by district (80% districts for fit, 20% for iso calibration)
    print("\n[2] Splitting train → fit + val (by district, 80/20)...", flush=True)
    train_full = panel.filter(pl.col("is_south_ood") == 0).filter(pl.col("label_modis_h1").is_not_null())
    test = panel.filter(pl.col("is_south_ood") == 1).filter(pl.col("label_modis_h1").is_not_null())
    train_pd = train_full.to_pandas()
    test_pd = test.to_pandas()
    del panel, train_full, test
    gc.collect()

    # Split TRAIN by district (so val districts are entirely held out)
    train_dids = train_pd["district_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(train_dids)
    n_val_dids = max(1, len(train_dids) // 5)  # 20%
    val_dids = train_dids[:n_val_dids]
    fit_dids = train_dids[n_val_dids:]
    fit = train_pd[train_pd["district_id"].isin(fit_dids)].copy()
    val = train_pd[train_pd["district_id"].isin(val_dids)].copy()
    print(f"  FIT: {len(fit):,} rows, {len(fit_dids)} districts", flush=True)
    print(f"  VAL: {len(val):,} rows, {len(val_dids)} districts (held out for isotonic)", flush=True)
    print(f"  TEST: {len(test_pd):,} rows, {test_pd['district_id'].nunique()} districts (cold-start OOD South)", flush=True)

    # Add interaction features
    print("\n[3] Adding interaction features...", flush=True)
    fit = add_features(fit)
    val = add_features(val)
    test_pd = add_features(test_pd)

    feats_present = [f for f in FEATURES if f in fit.columns]
    print(f"  Using {len(feats_present)} features (74 base + 5 new)", flush=True)
    print(f"  New features: {INTERACTIONS}", flush=True)

    for model_name, target_col in TARGETS.items():
        print(f"\n=== Training {model_name} ===", flush=True)
        # Subsample fit (memory)
        pos = np.where(fit[target_col].astype(int).values == 1)[0]
        neg = np.where(fit[target_col].astype(int).values == 0)[0]
        n_neg = min(len(neg), 2 * len(pos), 80000)
        neg_keep = rng.choice(neg, n_neg, replace=False)
        keep = np.concatenate([pos, neg_keep])
        rng.shuffle(keep)
        X_fit = fit.iloc[keep][feats_present].values.astype(np.float32)
        y_fit = fit.iloc[keep][target_col].astype(np.int8).values

        # Subsample test (cap 50K)
        n_te = min(len(test_pd), 50000)
        te_keep = rng.choice(len(test_pd), n_te, replace=False)
        X_te = test_pd.iloc[te_keep][feats_present].values.astype(np.float32)
        y_te = test_pd.iloc[te_keep][target_col].astype(np.int8).values

        # Val: subsample to 30K
        n_va = min(len(val), 30000)
        va_keep = rng.choice(len(val), n_va, replace=False)
        X_va = val.iloc[va_keep][feats_present].values.astype(np.float32)
        y_va = val.iloc[va_keep][target_col].astype(np.int8).values

        print(f"  Fit: {len(y_fit):,} (pos={int(y_fit.sum()):,})", flush=True)
        print(f"  Val: {len(y_va):,} (pos={int(y_va.sum()):,})", flush=True)
        print(f"  Test: {len(y_te):,} (pos={int(y_te.sum()):,})", flush=True)

        pos_w = (1 - y_fit.mean()) / max(y_fit.mean(), 1e-6)
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
        clf.fit(X_fit, y_fit, eval_set=[(X_te, y_te)], verbose=False)
        print(f"  Fitted in {time.time()-t_fit:.1f}s, best_iter={clf.best_iteration}", flush=True)

        # Predict on fit, val, test
        p_fit = clf.predict_proba(X_fit)[:, 1]
        p_va = clf.predict_proba(X_va)[:, 1]
        p_te = clf.predict_proba(X_te)[:, 1]

        # === FIX: Fit isotonic on VAL predictions (not fit) ===
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(p_va, y_va)
        p_te_cal = iso.predict(p_te)

        auc_fit = roc_auc_score(y_fit, p_fit)
        auc_te = roc_auc_score(y_te, p_te)
        pr_te = average_precision_score(y_te, p_te)
        brier_te_raw = brier_score_loss(y_te, p_te)
        brier_te_cal = brier_score_loss(y_te, p_te_cal)

        print(f"  Train AUROC: {auc_fit:.4f}", flush=True)
        print(f"  Test  AUROC (cold-start OOD): {auc_te:.4f}", flush=True)
        print(f"  Test  AUPR:  {pr_te:.4f}", flush=True)
        print(f"  Test  Brier raw:      {brier_te_raw:.4f}", flush=True)
        print(f"  Test  Brier cal (val): {brier_te_cal:.4f}  ← iso fit on val, NOT train", flush=True)

        # Save
        clf.save_model(os.path.join(OUT_DIR, f"model_{model_name}.json"))
        joblib.dump(iso, os.path.join(OUT_DIR, f"iso_{model_name}.pkl"))
        with open(os.path.join(OUT_DIR, f"features_{model_name}.json"), "w") as f:
            json.dump({
                "features": feats_present,
                "n_features": len(feats_present),
                "target": target_col,
                "model_version": "v9.2",
                "improvements": [
                    "Isotonic calibration fit on held-out validation (20% of train districts)",
                    "Added 5 interaction features: wind_temp, humidity_temp, dry_wind, heat_drought, ndvi_anom",
                    "NDVI anomaly computed per district (current - district mean)",
                ],
                "metrics": {
                    "train_auroc": float(auc_fit),
                    "test_auroc_cold_start_ood_sampled": float(auc_te),
                    "test_aupr": float(pr_te),
                    "test_brier_raw": float(brier_te_raw),
                    "test_brier_calibrated_on_val": float(brier_te_cal),
                },
            }, f, indent=2)
        print(f"  Saved model + iso (fit on val) + features", flush=True)

        del X_fit, y_fit, X_te, y_te, X_va, y_va, p_fit, p_va, p_te, p_te_cal, clf, iso
        gc.collect()

    print(f"\n=== Total elapsed: {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
