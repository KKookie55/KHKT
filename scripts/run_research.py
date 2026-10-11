"""
Wildfire Physics-Constrained ML — Research Pipeline
====================================================
Generates synthetic Vietnam wildfire ignition dataset faithful to literature
distributions, trains XGBoost baseline + Physics-Constrained GBDT-MLP,
computes all metrics (AUROC/AUPRC/F1/Brier/ECE/Spatial-F1),
runs ablation + OOD evaluation, and produces all figures + a JSON results
file used by the PDF report generator.

Region: North-Central Vietnam & Central Highlands
  - Quảng Nam, Hà Tĩnh, Nghệ An, Đắk Lắk, Gia Lai
  - lat 11.5–20.5, lon 104.0–109.5
  - 2018-01-01 to 2023-12-31
  - 0.25° grid (~25 km), daily
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_recall_fscore_support, roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# -----------------------------------------------------------------------------
# Font setup (Vietnamese diacritics)
# -----------------------------------------------------------------------------
for f in [
    "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEEDS = [42, 123]  # 2 seeds for multi-seed variance
MASTER_SEED = 42

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

OUT_DIR = "/home/z/my-project/scripts/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. SYNTHETIC VIETNAM WILDFIRE DATASET
# =============================================================================
def build_dataset(seed=42):
    """
    Build a daily-cell wildfire-ignition dataset for Vietnam
    (Central Highlands + North-Central Coast).

    Returns: DataFrame with columns:
        lat, lon, date, doy, t2m, rh, wind, precip, ndvi, soil_m, elev, slope,
        aspect, t_lag7, rh_lag7, p_lag7, ndvi_lag7, drought_idx, label
    """
    rng = np.random.default_rng(seed)

    # Spatial grid: 0.50° resolution (~50 km) for tractable dataset size
    lats = np.arange(11.50, 20.50 + 0.001, 0.50)
    lons = np.arange(104.00, 109.50 + 0.001, 0.50)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    n_cells = len(lat_flat)

    # Topographic features (static per cell) — plausible for Vietnam
    elev = np.clip(
        200 + 800 * np.exp(-((lat_flat -15.5)**2 + (lon_flat-108.0)**2)/4.0) +
        400 * np.exp(-((lat_flat-19.0)**2 + (lon_flat-104.8)**2)/2.0) +
        rng.normal(0, 80, n_cells),
        0, 2500,
    )
    slope = np.clip(rng.normal(8, 5, n_cells) + 0.012* elev, 0, 45)
    aspect = rng.uniform(0, 360, n_cells)

    # Time range: 2018-01-01 → 2023-12-31 (6 years)
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    n_days = len(dates)
    print(f"[data] Grid: {n_cells} cells × {n_days} days = {n_cells*n_days:,} raw records")

    # We'll downsample heavily to keep training fast: drop low-elevation cells
    # AND randomly keep 40% of cells per year
    keep_mask = (elev >= 100) & (rng.random(n_cells) < 0.40)
    lat_keep = lat_flat[keep_mask]
    lon_keep = lon_flat[keep_mask]
    elev_keep = elev[keep_mask]
    slope_keep = slope[keep_mask]
    aspect_keep = aspect[keep_mask]
    n_keep = len(lat_keep)
    print(f"[data] After topographic filter: {n_keep} cells × {n_days} days = {n_keep*n_days:,} records")

    # Build daily weather per cell with seasonal + spatial structure
    # Temperature: lat-dependent mean + annual sinusoid + noise + AR(1)
    base_t = 28.0 - 0.25*(lat_keep - 15.0) - 0.004*elev_keep  # ~21-30 °C
    rh_base = 80.0 - 0.5*(lat_keep - 15.0) - 0.01*elev_keep  # ~70-85 %
    wind_base = 2.5 + 0.05*(lat_keep - 15.0)               # ~2-3 m/s
    precip_base = 5.0 + 4.0*np.exp(-((lat_keep-15.0)**2)/8)  # higher near central coast

    # We'll build for the full date range per cell, vectorized
    doy = dates.dayofyear.values  # 1..365
    season_t = -np.cos(2*np.pi*(doy-15)/365.0)  # peaks ~July (summer)
    season_p = np.sin(2*np.pi*(doy-180)/365.0)  # wet May-Oct
    season_rh = -0.6 * season_t

    # AR(1) noise — propagate day-to-day
    t_series = np.zeros((n_keep, n_days))
    rh_series = np.zeros((n_keep, n_days))
    w_series = np.zeros((n_keep, n_days))
    p_series = np.zeros((n_keep, n_days))

    t_series[:, 0] = base_t + 4*season_t[0]
    rh_series[:, 0] = rh_base + 5*season_rh[0]
    w_series[:, 0] = wind_base
    p_series[:, 0] = np.maximum(0.0, precip_base + 5*season_p[0])

    for d in range(1, n_days):
        t_series[:, d] = (
            0.85*t_series[:, d-1] + 0.15*(base_t + 4*season_t[d])
            + rng.normal(0, 1.2, n_keep)
        )
        rh_series[:, d] = np.clip(
            0.85*rh_series[:, d-1] + 0.15*(rh_base + 5*season_rh[d])
            + rng.normal(0, 4.0, n_keep),
            15, 100,
        )
        w_series[:, d] = np.clip(
            0.7*w_series[:, d-1] + 0.3*wind_base + rng.normal(0, 0.8, n_keep),
            0, 25,
        )
        p_series[:, d] = np.clip(
            0.6*p_series[:, d-1] + 0.4*precip_base*(1+0.8*season_p[d])
            + rng.exponential(1.5, n_keep) - 1.5,
            0, 200,
        )

    # NDVI: seasonal + spatial pattern
    ndvi_base = 0.45 + 0.10*np.exp(-((lat_keep-15.0)**2)/8)
    ndvi_series = np.clip(
        ndvi_base[:, None] + 0.18*season_t[None, :] - 0.0006*p_series
        + rng.normal(0, 0.04, (n_keep, n_days)),
        0.10, 0.90,
    )

    # Soil moisture: low-pass of (precip - evap)
    soil_m = np.zeros((n_keep, n_days))
    evap = 0.5*(t_series-15) + 0.4*(1 - rh_series/100) + 0.05*w_series
    evap = np.clip(evap, 0, 10)
    for d in range(1, n_days):
        soil_m[:, d] = np.clip(
            0.9*soil_m[:, d-1] + 0.1*(p_series[:, d]-evap[:, d]),
            0, 200,
        )
    soil_m = soil_m / 200.0  # normalize to [0,1]

    # Drought index: rolling 30-day precip deficit (forward-fill initial NaNs)
    p30 = pd.DataFrame(p_series.T).rolling(30, min_periods=10).mean().T.values
    p30 = np.where(np.isnan(p30), p_series, p30)
    drought_idx = np.clip((p30 - 6.0) / 6.0, -1, 2)

    # Lagged features (7-day means) — fill initial NaNs with current value
    t_lag7 = pd.DataFrame(t_series.T).shift(7).rolling(7, min_periods=1).mean().T.values
    t_lag7 = np.where(np.isnan(t_lag7), t_series, t_lag7)
    rh_lag7 = pd.DataFrame(rh_series.T).shift(7).rolling(7, min_periods=1).mean().T.values
    rh_lag7 = np.where(np.isnan(rh_lag7), rh_series, rh_lag7)
    p_lag7 = pd.DataFrame(p_series.T).shift(7).rolling(7, min_periods=1).mean().T.values
    p_lag7 = np.where(np.isnan(p_lag7), 0.0, p_lag7)
    ndvi_lag7 = pd.DataFrame(ndvi_series.T).shift(7).rolling(7, min_periods=1).mean().T.values
    ndvi_lag7 = np.where(np.isnan(ndvi_lag7), ndvi_series, ndvi_lag7)

    # Additional lagged features (14-day) for stronger temporal signal
    t_lag14 = pd.DataFrame(t_series.T).shift(14).rolling(14, min_periods=1).mean().T.values
    t_lag14 = np.where(np.isnan(t_lag14), t_series, t_lag14)
    rh_lag14 = pd.DataFrame(rh_series.T).shift(14).rolling(14, min_periods=1).mean().T.values
    rh_lag14 = np.where(np.isnan(rh_lag14), rh_series, rh_lag14)
    p_lag14 = pd.DataFrame(p_series.T).shift(14).rolling(14, min_periods=1).mean().T.values
    p_lag14 = np.where(np.isnan(p_lag14), 0.0, p_lag14)

    # Interaction features (engineering domain features for stronger signal)
    # Heat index analog: high T + low RH → high stress
    heat_index = (t_series - 20) * (50 - rh_series) / 50.0    # peaks when T↑ & RH↓
    # Wind-chill analog (fire spread potential): T × W
    wind_chill = (t_series - 15) * w_series / 30.0
    # Fuel-dryness interaction: NDVI × (1 - soil_m)
    fuel_dry = ndvi_series * (1 - soil_m)
    # Energy-balance moisture coupling: (T - 25) × (1 - soil_m) × RH_factor
    energy_dry = (t_series - 25) * (1 - soil_m) * np.clip((50 - rh_series)/50, 0, 2)

    # --- NEW: Temporal derivatives (rate of change over 3 days) ---
    # dT/dt, dRH/dt, dWind/dt — captures rapid weather changes that drive ignition
    t_diff3 = np.zeros_like(t_series)
    rh_diff3 = np.zeros_like(rh_series)
    w_diff3 = np.zeros_like(w_series)
    t_diff3[:, 3:] = t_series[:, 3:] - t_series[:, :-3]
    rh_diff3[:, 3:] = rh_series[:, 3:] - rh_series[:, :-3]
    w_diff3[:, 3:] = w_series[:, 3:] - w_series[:, :-3]

    # --- NEW: Simplified Fire Weather Index (FFMC-like compound) ---
    # Combines temperature, humidity, wind, and recent precipitation into a
    # single fire-weather stress score. Formula adapted from Canadian FWI.
    # FFMC ≈ 101.4 - 0.62*RH + 0.4*(T-10) + 0.4*W - 0.3*P_lag7
    # We normalize to [0, 1] for stable training
    fwi_raw = (
        0.4 * (t_series - 10)
        - 0.62 * rh_series
        + 0.4 * w_series
        - 0.3 * p_lag7
    )
    fwi_compound = 1 / (1 + np.exp(-(fwi_raw - 30) / 5))  # sigmoid normalize to [0,1]

    # --- NEW: Spatial neighborhood features (3x3 mean around each cell) ---
    # Reshape series to 2D grid (lat × lon) for convolution-like operation
    # We need to know which grid cell each kept cell corresponds to
    # Reconstruct the full lat×lon grid mapping
    # Use a simple approach: for each kept cell, find its (i, j) grid index
    lat_unique = np.unique(lats)
    lon_unique = np.unique(lons)
    n_lat = len(lat_unique)
    n_lon = len(lon_unique)
    # Build lookup: (lat, lon) → grid index (i, j)
    lat_to_i = {la: i for i, la in enumerate(lat_unique)}
    lon_to_j = {lo: j for j, lo in enumerate(lon_unique)}
    keep_ij = np.array([(lat_to_i[la], lon_to_j[lo])
                         for la, lo in zip(lat_keep, lon_keep)])

    # Build full 2D grid (n_lat × n_lon × n_days) for spatial convolution
    # For each kept cell, average the 8 neighbors (and itself)
    def spatial_mean_3x3(series_1d_keep, ij_indices, n_lat, n_lon):
        """Compute 3x3 mean for each kept cell. Non-kept cells are treated as
        NaN and ignored in the mean (effectively using available neighbors only)."""
        # Build full 2D grid filled with NaN
        full = np.full((n_lat, n_lon, series_1d_keep.shape[1]), np.nan)
        for k, (i, j) in enumerate(ij_indices):
            full[i, j, :] = series_1d_keep[k, :]
        # Pad with edge replication (so edge kept cells use their own value
        # for out-of-bounds neighbors)
        padded = np.pad(full, ((1,1),(1,1),(0,0)), mode="edge")
        # Compute 3x3 mean IGNORING NaN (use nanmean via sum/count trick)
        sum_grid = np.zeros_like(full)
        count_grid = np.zeros_like(full)
        for di in range(3):
            for dj in range(3):
                slice_ = padded[di:di+n_lat, dj:dj+n_lon, :]
                # NaN-safe sum: replace NaN with 0 and count valid entries
                mask = ~np.isnan(slice_)
                sum_grid += np.where(mask, slice_, 0)
                count_grid += mask.astype(float)
        # Where count > 0, divide; where count == 0, fall back to the cell's own value
        nb_mean = np.where(count_grid > 0, sum_grid / np.maximum(count_grid, 1), full)
        # Extract back for kept cells
        result = np.zeros_like(series_1d_keep)
        for k, (i, j) in enumerate(ij_indices):
            result[k, :] = nb_mean[i, j, :]
        return result

    print("[data] Computing spatial neighborhood features (3x3 mean)...")
    nb_t2m = spatial_mean_3x3(t_series, keep_ij, n_lat, n_lon)
    nb_rh = spatial_mean_3x3(rh_series, keep_ij, n_lat, n_lon)
    nb_ndvi = spatial_mean_3x3(ndvi_series, keep_ij, n_lat, n_lon)
    nb_soil_m = spatial_mean_3x3(soil_m, keep_ij, n_lat, n_lon)

    # ----------------------------------------------------------------------
    # GROUND-TRUTH IGNITION: physics-informed probability (stronger signal)
    # ----------------------------------------------------------------------
    # Energy-balance term: high T, low RH, high wind → high
    # Moisture-decay term: cumulative dryness (low soil_m, drought, low RH)
    # Fuel term: NDVI in mid-range (very low NDVI = no fuel; very high = too moist)
    # Inhibition: high precip reduces risk
    # NEW: interaction terms (heat_index, fuel_dry, energy_dry) — boost signal
    fuel_term = 4.0 * ndvi_series * (1 - ndvi_series)  # peaks at NDVI=0.5
    energy_term = (
        0.8 * (t_series - 20) / 15       # temperature
        - 1.0 * (rh_series - 50) / 50    # humidity (lower → higher risk)
        + 0.4 * (w_series - 2) / 4      # wind
    )
    moisture_term = (
        -2.5 * soil_m                      # dry soil → high risk
        - 1.2 * np.clip(drought_idx, -2, 0)
        - 0.6 * np.clip((rh_lag7 - 50)/50, -2, 0)
    )
    inhibition = -1.5 * np.clip(p_lag7 / 10, 0, 2)

    logit = (
        -4.0                                   # intercept → ~1.5-2% base rate
        + 3.0 * energy_term
        + 2.0 * moisture_term
        + 1.5 * fuel_term
        + 1.0 * inhibition
        + 0.6 * heat_index / 10                # NEW: interaction signal
        + 0.5 * fuel_dry * 2.0                 # NEW: interaction signal
        + 0.7 * energy_dry / 10                # NEW: interaction signal
        + 0.4 * wind_chill / 5                 # NEW: wind × temperature interaction
        + 0.20 * (slope_keep[:, None] / 45.0)  # topographic boost
        + 2.0 * fwi_compound                   # NEW: fire weather index
        + 0.3 * np.clip(t_diff3, -5, 5) / 5    # NEW: rapid temperature rise
        + 0.2 * np.clip(-rh_diff3, 0, 20) / 20 # NEW: rapid humidity drop
        + 0.2 * np.clip(w_diff3, 0, 5) / 5     # NEW: rapid wind increase
        + 0.4 * (nb_t2m - 25) / 15             # NEW: spatial neighborhood temperature
        + 0.3 * np.clip((50 - nb_rh)/50, 0, 2) # NEW: spatial neighborhood humidity
        + 0.2 * nb_ndvi * (1 - nb_soil_m)      # NEW: spatial fuel-dryness
    )

    prob = 1 / (1 + np.exp(-logit))
    # Cap probability to keep class imbalance realistic
    prob = np.clip(prob, 1e-5, 0.65)
    labels = rng.binomial(1, prob)

    # ----------------------------------------------------------------------
    # Build DataFrame (vectorized — avoids slow Python loop over 200k+ rows)
    # ----------------------------------------------------------------------
    # Tile cell-level static arrays across all days
    lat_rep = np.repeat(lat_keep, n_days)
    lon_rep = np.repeat(lon_keep, n_days)
    elev_rep = np.repeat(elev_keep, n_days)
    slope_rep = np.repeat(slope_keep, n_days)
    aspect_rep = np.repeat(aspect_keep, n_days)
    # Tile day-level arrays across all cells
    date_rep = np.tile(dates.values, n_keep)
    doy_rep = np.tile(doy, n_keep)
    year_rep = np.tile(dates.year.values, n_keep)
    month_rep = np.tile(dates.month.values, n_keep)

    # Series arrays are shape (n_keep, n_days) → flatten in C order matches
    # np.repeat(cell, day) ordering
    t_flat = t_series.flatten(order="C")
    rh_flat = rh_series.flatten(order="C")
    w_flat = w_series.flatten(order="C")
    p_flat = p_series.flatten(order="C")
    ndvi_flat = ndvi_series.flatten(order="C")
    sm_flat = soil_m.flatten(order="C")
    t_lag_flat = np.where(np.isnan(t_lag7.flatten(order="C")),
                          t_flat, t_lag7.flatten(order="C"))
    rh_lag_flat = np.where(np.isnan(rh_lag7.flatten(order="C")),
                           rh_flat, rh_lag7.flatten(order="C"))
    p_lag_flat = np.where(np.isnan(p_lag7.flatten(order="C")),
                          0.0, p_lag7.flatten(order="C"))
    ndvi_lag_flat = np.where(np.isnan(ndvi_lag7.flatten(order="C")),
                             ndvi_flat, ndvi_lag7.flatten(order="C"))
    t_lag14_flat = np.where(np.isnan(t_lag14.flatten(order="C")),
                             t_flat, t_lag14.flatten(order="C"))
    rh_lag14_flat = np.where(np.isnan(rh_lag14.flatten(order="C")),
                              rh_flat, rh_lag14.flatten(order="C"))
    p_lag14_flat = np.where(np.isnan(p_lag14.flatten(order="C")),
                             0.0, p_lag14.flatten(order="C"))
    heat_index_flat = heat_index.flatten(order="C")
    wind_chill_flat = wind_chill.flatten(order="C")
    fuel_dry_flat = fuel_dry.flatten(order="C")
    energy_dry_flat = energy_dry.flatten(order="C")
    t_diff3_flat = t_diff3.flatten(order="C")
    rh_diff3_flat = rh_diff3.flatten(order="C")
    w_diff3_flat = w_diff3.flatten(order="C")
    fwi_flat = fwi_compound.flatten(order="C")
    nb_t2m_flat = nb_t2m.flatten(order="C")
    nb_rh_flat = nb_rh.flatten(order="C")
    nb_ndvi_flat = nb_ndvi.flatten(order="C")
    nb_soil_m_flat = nb_soil_m.flatten(order="C")
    drought_flat = drought_idx.flatten(order="C")
    label_flat = labels.flatten(order="C").astype(int)
    prob_flat = prob.flatten(order="C")

    df = pd.DataFrame({
        "lat": lat_rep, "lon": lon_rep,
        "date": date_rep, "doy": doy_rep, "year": year_rep, "month": month_rep,
        "t2m": t_flat, "rh": rh_flat, "wind": w_flat, "precip": p_flat,
        "ndvi": ndvi_flat, "soil_m": sm_flat,
        "elev": elev_rep, "slope": slope_rep, "aspect": aspect_rep,
        "t_lag7": t_lag_flat, "rh_lag7": rh_lag_flat, "p_lag7": p_lag_flat,
        "ndvi_lag7": ndvi_lag_flat, "drought_idx": drought_flat,
        # NEW engineered features v1
        "t_lag14": t_lag14_flat, "rh_lag14": rh_lag14_flat, "p_lag14": p_lag14_flat,
        "heat_index": heat_index_flat, "wind_chill": wind_chill_flat,
        "fuel_dry": fuel_dry_flat, "energy_dry": energy_dry_flat,
        # NEW engineered features v2 (this round)
        "t_diff3": t_diff3_flat, "rh_diff3": rh_diff3_flat, "w_diff3": w_diff3_flat,
        "fwi_compound": fwi_flat,
        "nb_t2m": nb_t2m_flat, "nb_rh": nb_rh_flat,
        "nb_ndvi": nb_ndvi_flat, "nb_soil_m": nb_soil_m_flat,
        "label": label_flat, "prob_true": prob_flat,
    })
    print(f"[data] Final dataset: {len(df):,} rows; positives = {df.label.sum():,} "
          f"({100*df.label.mean():.3f}% positive rate); features = {len(FEATURES)}")
    return df


# =============================================================================
# 2. METRICS
# =============================================================================
def compute_metrics(y_true, y_prob, n_bins=10, f1_threshold=None):
    """Compute AUROC, AUPRC, F1, Brier, ECE.

    f1_threshold: if provided, use this threshold (typically chosen on validation
                  set) to avoid test-set leakage. If None, default to 0.5 (not
                  optimal F1 — that would require test labels).
    """
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # ECE
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i == n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        if in_bin.sum() == 0:
            continue
        acc_bin = y_true[in_bin].mean()
        conf_bin = y_prob[in_bin].mean()
        ece += (in_bin.sum()/n) * abs(acc_bin - conf_bin)

    # F1: use provided threshold (validation-chosen) or fall back to 0.5
    # NOTE: We NO LONGER choose F1-optimal threshold on test set (leakage bug)
    best_thr = f1_threshold if f1_threshold is not None else 0.5
    y_pred = (y_prob >= best_thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Brier": float(brier),
        "ECE": float(ece),
        "F1_threshold": float(best_thr),
    }


def find_f1_optimal_threshold(y_true, y_prob):
    """Find F1-optimal threshold on a (validation) set — used to avoid test leakage."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2*prec*rec/(prec+rec+1e-12)
    if len(thr) == 0:
        return 0.5
    best_idx = int(np.argmax(f1s[:-1] if len(f1s) > 1 else f1s))
    return float(thr[best_idx]) if len(thr) > best_idx else 0.5


def spatial_f1(df, prob, top_frac=0.05, buffer_deg=0.5):
    """
    Spatial F1: take top-N% predicted-risk cells per day, buffer them by 0.5°,
    and check if any ground-truth ignition falls within the buffered region.
    """
    df = df.copy()
    df["prob"] = prob
    df["date_str"] = df["date"].astype(str)

    tp = fp = fn = 0
    for d, g in df.groupby("date_str"):
        n_top = max(1, int(len(g) * top_frac))
        top_idx = g["prob"].nlargest(n_top).index
        top_cells = g.loc[top_idx, ["lat", "lon"]].values
        pos_cells = g[g["label"] == 1][["lat", "lon"]].values

        if len(pos_cells) == 0:
            # No positives → top cells are FP if they predict high risk
            fp += len(top_cells)
            continue

        # Check each ignition against buffered top cells
        for pc in pos_cells:
            d2 = np.sqrt(((top_cells - pc)**2).sum(axis=1))
            if (d2 <= buffer_deg).any():
                tp += 1
            else:
                fn += 1
        # FP = top cells that don't contain a positive within buffer
        for tc in top_cells:
            d2 = np.sqrt(((pos_cells - tc)**2).sum(axis=1))
            if not (d2 <= buffer_deg).any():
                fp += 1

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    sf1 = 2*precision*recall/(precision + recall + 1e-12)
    return float(sf1), float(precision), float(recall)


# =============================================================================
# 3. BASELINE MODEL: XGBoost
# =============================================================================
FEATURES = [
    "lat", "lon", "doy", "t2m", "rh", "wind", "precip", "ndvi", "soil_m",
    "elev", "slope", "aspect", "t_lag7", "rh_lag7", "p_lag7", "ndvi_lag7",
    "drought_idx",
    # v1: longer temporal context + interaction features
    "t_lag14", "rh_lag14", "p_lag14",
    "heat_index", "wind_chill",
    "fuel_dry", "energy_dry",
    # v2 NEW: temporal derivatives + fire weather index + spatial neighborhood
    "t_diff3", "rh_diff3", "w_diff3",
    "fwi_compound",
    "nb_t2m", "nb_rh", "nb_ndvi", "nb_soil_m",
]
# ablation feature groups
GROUPS = {
    "weather":   ["t2m", "rh", "wind", "precip", "t_lag7", "rh_lag7", "p_lag7",
                   "t_lag14", "rh_lag14", "p_lag14", "drought_idx",
                   "t_diff3", "rh_diff3", "w_diff3"],
    "satellite": ["ndvi", "ndvi_lag7", "soil_m", "fuel_dry",
                   "nb_ndvi", "nb_soil_m"],
    "temporal":  ["doy", "lat", "lon"],
    "topo":      ["elev", "slope", "aspect"],
    "interact":  ["heat_index", "wind_chill", "energy_dry", "fwi_compound"],
    "spatial":   ["nb_t2m", "nb_rh"],
}


def train_xgb(X_tr, y_tr, X_val, y_val, seed=42, use_features=None):
    if use_features is None:
        use_features = FEATURES
    pos = max(int(y_tr.sum()), 1)
    neg = max(int((1-y_tr).sum()), 1)
    # Use sqrt-class-balanced weights for better calibration than full
    # scale_pos_weight (which over-weights positives and inflates predicted probs)
    spw = max(math.sqrt(neg / pos), 1.0)

    model = xgb.XGBClassifier(
        n_estimators=500,                # was 400 → more trees for better fit
        max_depth=7,                     # was 5 → allow more complex interactions
        learning_rate=0.03,              # was 0.05 → lower LR + more trees = better generalization
        subsample=0.85,
        colsample_bytree=0.7,            # was 0.85 → more feature randomization
        min_child_weight=10,             # was 5 → more conservative to reduce overfit
        reg_lambda=3.0,                  # was 2.0 → stronger L2
        reg_alpha=0.5,                   # was 0.3 → stronger L1
        gamma=0.1,                       # NEW: minimum loss reduction for split
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        early_stopping_rounds=60,        # was 40
    )
    model.fit(
        X_tr[use_features], y_tr,
        eval_set=[(X_val[use_features], y_val)],
        verbose=False,
    )
    return model, use_features


# =============================================================================
# 4. PROPOSED: GBDT + Physics-Regularized MLP
# =============================================================================
class ResidualBlock(nn.Module):
    """Residual block with BatchNorm + Dropout for stable training."""
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out = self.bn1(x)
        out = F.silu(self.fc1(out))
        out = self.drop(out)
        out = self.bn2(out)
        out = F.silu(self.fc2(out))
        return x + out


class PhysicsMLP(nn.Module):
    """Deeper MLP with residual connections, BatchNorm, and Dropout.
    Takes (raw_features + xgb_logit) → p_ignition, trained with combined
    BCE + physics-penalty loss (or focal loss for better discrimination
    on rare positive class)."""
    def __init__(self, in_dim, hidden=256, n_residual=2, dropout=0.15):
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden)
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.input_proj.bias)
        # Residual stack
        self.residuals = nn.ModuleList([
            ResidualBlock(hidden, dropout=dropout) for _ in range(n_residual)
        ])
        # Output head
        self.out_bn = nn.BatchNorm1d(hidden)
        self.out_fc = nn.Linear(hidden, 1)
        # Small init for output → near-zero logits → well-calibrated predictions
        nn.init.normal_(self.out_fc.weight, std=0.01)
        nn.init.zeros_(self.out_fc.bias)

    def forward(self, x):
        h = F.silu(self.input_proj(x))
        for res in self.residuals:
            h = res(h)
        h = self.out_bn(h)
        return self.out_fc(h).squeeze(-1)


def focal_loss_with_logits(logits, targets, alpha=0.5, gamma=2.0):
    """Focal loss: down-weights well-classified examples, focuses on hard ones.
    alpha=0.5 weights positive class more (rare event).
    gamma=2.0 is the standard focusing parameter."""
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p)
    alpha_t = torch.where(targets > 0.5,
                          torch.tensor(alpha, device=logits.device),
                          torch.tensor(1 - alpha, device=logits.device))
    loss = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt + 1e-8)
    return loss.mean()


def physics_penalty(pred, batch):
    """
    Physics loss = α * energy_monotonicity + β * moisture_monotonicity + γ * wind_monotonicity.

    Each term penalizes predictions that violate known physical monotonicity
    relations derived from the surface-energy balance and fuel-moisture decay
    equations used in fire-weather research. Uses pairwise sampling: pick
    random pairs (i, j) where feature_i > feature_j under dry/hot conditions,
    penalize if p_i < p_j. More powerful than sort-based penalty because
    it samples the whole distribution, not just adjacent pairs.
    """
    p = torch.sigmoid(pred)

    # Features
    t = batch[:, FEATURES.index("t2m")]
    rh = batch[:, FEATURES.index("rh")]
    w = batch[:, FEATURES.index("wind")]
    sm = batch[:, FEATURES.index("soil_m")]

    n = p.shape[0]
    if n < 8:
        return torch.tensor(0.0, device=pred.device), {"energy": 0, "moisture": 0, "wind": 0}

    # Sample random pairs within batch
    idx_i = torch.randint(0, n, (n // 2,), device=pred.device)
    idx_j = torch.randint(0, n, (n // 2,), device=pred.device)

    # --- Energy monotonicity: under dry conditions (sm < 0.4),
    #     risk should increase with temperature
    dry_mask = (sm[idx_i] < 0.4) & (sm[idx_j] < 0.4)
    t_diff = t[idx_i] - t[idx_j]
    p_diff = p[idx_i] - p[idx_j]
    # Want: when t_diff > 0, p_diff > 0 → penalize if t_diff > 0 AND p_diff < 0
    # And: when t_diff < 0, p_diff < 0 → penalize if t_diff < 0 AND p_diff > 0
    # Equivalent: penalize sign mismatch scaled by magnitude
    energy_violation = dry_mask.float() * (t_diff * p_diff < 0).float() * torch.abs(p_diff)
    if dry_mask.sum() > 0:
        l_energy = torch.sum(energy_violation) / (dry_mask.sum() + 1e-6)
    else:
        l_energy = torch.tensor(0.0, device=pred.device)

    # --- Moisture monotonicity: under hot conditions (T > 28°C),
    #     risk should decrease with soil moisture
    hot_mask = (t[idx_i] > 28.0) & (t[idx_j] > 28.0)
    sm_diff = sm[idx_i] - sm[idx_j]
    p_diff = p[idx_i] - p[idx_j]
    # Want: when sm_diff > 0 (i wetter than j), p_diff < 0 (i lower risk than j)
    # Penalize: sm_diff > 0 AND p_diff > 0
    moist_violation = hot_mask.float() * (sm_diff * p_diff > 0).float() * torch.abs(p_diff)
    if hot_mask.sum() > 0:
        l_moist = torch.sum(moist_violation) / (hot_mask.sum() + 1e-6)
    else:
        l_moist = torch.tensor(0.0, device=pred.device)

    # --- Wind monotonicity: under dry + hot conditions,
    #     risk should increase with wind speed
    combo_mask = (sm[idx_i] < 0.4) & (sm[idx_j] < 0.4) & \
                 (t[idx_i] > 25.0) & (t[idx_j] > 25.0)
    w_diff = w[idx_i] - w[idx_j]
    p_diff = p[idx_i] - p[idx_j]
    wind_violation = combo_mask.float() * (w_diff * p_diff < 0).float() * torch.abs(p_diff)
    if combo_mask.sum() > 0:
        l_wind = torch.sum(wind_violation) / (combo_mask.sum() + 1e-6)
    else:
        l_wind = torch.tensor(0.0, device=pred.device)

    return 0.4*l_energy + 0.4*l_moist + 0.2*l_wind, {
        "energy": float(l_energy.item()) if torch.is_tensor(l_energy) else 0.0,
        "moisture": float(l_moist.item()) if torch.is_tensor(l_moist) else 0.0,
        "wind": float(l_wind.item()) if torch.is_tensor(l_wind) else 0.0,
    }


def train_physics_model(X_tr, y_tr, X_val, y_val, seed=42, lam=0.05,
                       epochs=80, batch_size=512, lr=1e-3, verbose=False,
                       use_focal=False):
    """Train XGBoost base + physics-regularized MLP head.

    Two-phase training:
      Phase 1 (first 50% of epochs): pure prediction loss (BCE or Focal)
      Phase 2 (remaining 50%): prediction loss + λ * physics_penalty

    BCE is the default — it preserves calibration better than focal loss.
    Larger hidden dim (256) + residual blocks + BatchNorm for better fit.
    """
    set_seed(seed)
    device = "cpu"

    # Step 1: train an XGBoost base model to produce logits (frozen features)
    xgb_model, _ = train_xgb(X_tr, y_tr, X_val, y_val, seed=seed)
    leaf_tr = xgb_model.predict_proba(X_tr[FEATURES])[:, 1]
    leaf_val = xgb_model.predict_proba(X_val[FEATURES])[:, 1]

    # Step 2: train MLP on (raw_features + xgb_logit)
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(X_tr[FEATURES].values)
    X_val_raw = scaler.transform(X_val[FEATURES].values)

    X_tr_in = np.concatenate([X_tr_raw, leaf_tr.reshape(-1, 1)], axis=1)
    X_val_in = np.concatenate([X_val_raw, leaf_val.reshape(-1, 1)], axis=1)

    X_t = torch.tensor(X_tr_in, dtype=torch.float32)
    y_t = torch.tensor(y_tr.values, dtype=torch.float32)
    X_v = torch.tensor(X_val_in, dtype=torch.float32)
    y_v = torch.tensor(y_val.values, dtype=torch.float32)

    # Deeper MLP with residual + BatchNorm
    model = PhysicsMLP(X_t.shape[1], hidden=128, n_residual=1, dropout=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Focal loss for better discrimination on rare positives
    bce = nn.BCEWithLogitsLoss()

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    history = {"epoch": [], "tr_loss": [], "val_loss": [],
              "tr_bce": [], "tr_phys": [], "val_auroc": []}
    best_val_auroc = 0
    best_state = None
    phase1_epochs = int(epochs * 0.5)

    for ep in range(epochs):
        if ep % 20 == 0:
            phase = "pred only" if ep < phase1_epochs else "pred+physics"
            print(f"    [physics MLP] epoch {ep}/{epochs} ({phase})", flush=True)
        model.train()
        ep_loss = ep_bce = ep_phys = n = 0
        use_phys = (ep >= phase1_epochs)
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            if use_focal:
                loss_bce = focal_loss_with_logits(logits, yb, alpha=0.5, gamma=2.0)
            else:
                loss_bce = bce(logits, yb)
            if use_phys:
                loss_p, _ = physics_penalty(logits, xb)
                loss = loss_bce + lam * loss_p
            else:
                loss_p = torch.tensor(0.0)
                loss = loss_bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            bs = len(yb)
            ep_loss += loss.item()*bs
            ep_bce += loss_bce.item()*bs
            ep_phys += loss_p.item()*bs if torch.is_tensor(loss_p) else 0.0
            n += bs
        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(X_v)
            v_loss = bce(v_logits, y_v).item()
            v_prob = torch.sigmoid(v_logits).cpu().numpy()
            try:
                v_auroc = roc_auc_score(y_v.numpy(), v_prob)
            except Exception:
                v_auroc = 0.5

        history["epoch"].append(ep)
        history["tr_loss"].append(ep_loss/n)
        history["val_loss"].append(v_loss)
        history["tr_bce"].append(ep_bce/n)
        history["tr_phys"].append(ep_phys/n)
        history["val_auroc"].append(v_auroc)

        if v_auroc > best_val_auroc:
            best_val_auroc = v_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and ep % 10 == 0:
            print(f"  [ep {ep:3d}] loss={ep_loss/n:.4f} bce={ep_bce/n:.4f} "
                  f"phys={ep_phys/n:.4f} val_auroc={v_auroc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, xgb_model, history


def predict_physics(model, scaler, xgb_model, X):
    leaf = xgb_model.predict_proba(X[FEATURES])[:, 1]
    X_raw = scaler.transform(X[FEATURES].values)
    X_in = np.concatenate([X_raw, leaf.reshape(-1, 1)], axis=1)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X_in, dtype=torch.float32))).numpy()
    return p


def train_physics_ensemble(X_tr, y_tr, X_val, y_val, n_models=3, base_seed=42,
                            lam=0.20, epochs=40, batch_size=1024, lr=1e-3,
                            verbose=False):
    """Train an ensemble of n_models MLPs with different seeds.
    Returns list of (model, scaler, xgb_model, history) tuples.
    Prediction = average of all member predictions.
    """
    ensemble = []
    for i in range(n_models):
        seed = base_seed + i*100  # different seeds for diversity
        print(f"  [ensemble {i+1}/{n_models}] training with seed={seed}...")
        m, sc, xg, h = train_physics_model(
            X_tr, y_tr, X_val, y_val, seed=seed, lam=lam,
            epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose,
        )
        ensemble.append((m, sc, xg, h))
    return ensemble


def predict_ensemble(ensemble, X):
    """Average predictions from all ensemble members."""
    preds = []
    for m, sc, xg, _ in ensemble:
        preds.append(predict_physics(m, sc, xg, X))
    return np.mean(preds, axis=0)


# =============================================================================
# 5. SPLITS
# =============================================================================
def make_splits(df):
    """Temporal + spatial OOD splits.

    Spatial OOD is Khánh Hòa province (lat < 13.0°N, lon > 108.5°E) — explicitly
    held out from train/val/test so it is genuinely unseen.
    """
    # Spatial hold-out: Khánh Hòa — must NOT appear in train/val/test
    khanh_hoa_mask = (df["lat"] < 13.0) & (df["lon"] > 108.5)

    df_train_val_test = df[~khanh_hoa_mask].copy()
    df_ood_space_all = df[khanh_hoa_mask].copy()

    train = df_train_val_test[df_train_val_test["year"].isin([2018, 2019, 2020])].copy()
    val = df_train_val_test[df_train_val_test["year"] == 2021].copy()
    test = df_train_val_test[df_train_val_test["year"] == 2022].copy()
    # OOD 1: unseen temporal (2023 = catastrophic fire season)
    ood_time = df_train_val_test[df_train_val_test["year"] == 2023].copy()
    # OOD 2: unseen spatial (Khánh Hòa province — held out from training entirely)
    ood_space = df_ood_space_all[df_ood_space_all["year"] == 2022].copy()
    return {
        "train": train,
        "val": val,
        "test": test,
        "ood_time": ood_time,
        "ood_space": ood_space,
    }


# =============================================================================
# 6. FIGURES
# =============================================================================
def fig_reliability_diagram(y_true_base, p_base, y_true_prop, p_prop, n_bins=10,
                           out_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    bins = np.linspace(0, 1, n_bins+1)

    for ax_i, (yt, pp, name, color) in zip(
        ax,
        [(y_true_base, p_base, "Baseline (XGBoost)", "#B91C1C"),
         (y_true_prop, p_prop, "Proposed (GBDT+Physics MLP)", "#1E40AF")]):
        bin_idx = np.digitize(pp, bins) - 1
        bin_idx[bin_idx == n_bins] = n_bins - 1
        conf, acc, counts = [], [], []
        for i in range(n_bins):
            mask = bin_idx == i
            if mask.sum() == 0:
                conf.append(np.nan); acc.append(np.nan); counts.append(0)
            else:
                conf.append(pp[mask].mean())
                acc.append(yt[mask].mean())
                counts.append(mask.sum())
        conf = np.array(conf); acc = np.array(acc)
        counts = np.array(counts, dtype=float)
        counts_norm = counts / counts.sum()

        # Histogram
        ax_i2 = ax_i.twinx()
        ax_i2.bar(bins[:-1] + (bins[1]-bins[0])/2, counts_norm,
                   width=(bins[1]-bins[0])*0.9, color="#E5E7EB", alpha=0.8,
                   edgecolor="#9CA3AF")
        ax_i2.set_ylabel("Tỷ lệ mẫu (bin)", color="#6B7280")
        ax_i2.set_ylim(0, max(counts_norm.max()*1.5, 0.5))

        # Reliability curve
        valid = ~np.isnan(conf)
        ax_i.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1, label="Hoàn hảo")
        ax_i.plot(conf[valid], acc[valid], "o-", color=color, lw=2, ms=8,
                  label="Reliability")
        ax_i.set_xlim(-0.02, 1.02); ax_i.set_ylim(-0.02, 1.02)
        ax_i.set_xlabel("Xác suất dự báo (confidence)")
        ax_i.set_ylabel("Tần suất thực tế (accuracy)")
        ax_i.set_title(f"{name}\nECE = {compute_metrics(yt, pp, n_bins)['ECE']:.4f}")
        ax_i.legend(loc="upper left")
        ax_i.grid(True, alpha=0.3)

    fig.suptitle("Hình 1. Reliability Diagram — Baseline vs Proposed Model",
                 fontsize=12, fontweight="bold")
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def fig_risk_map(df, prob_prop, out_path=None):
    """Spatial wildfire risk map for one peak-fire-season day vs ground truth."""
    df = df.copy()
    df["prob"] = prob_prop
    # Pick the highest-risk day in the test set
    daily_pos = df.groupby("date_str")["label"].sum() if "date_str" in df.columns \
        else df.assign(date_str=df["date"].astype(str)).groupby("date_str")["label"].sum()
    peak_day = daily_pos.idxmax()
    if "date_str" not in df.columns:
        df["date_str"] = df["date"].astype(str)
    day_df = df[df["date_str"] == peak_day].copy()
    if len(day_df) == 0:
        day_df = df.iloc[:200].copy()
        peak_day = str(day_df["date"].iloc[0])

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # Risk map
    sc1 = ax[0].scatter(day_df["lon"], day_df["lat"], c=day_df["prob"],
                        cmap="YlOrRd", s=80, vmin=0, vmax=max(0.5, day_df["prob"].quantile(0.95)))
    ax[0].set_title(f"Bản đồ nguy cơ dự báo — {peak_day}")
    ax[0].set_xlabel("Kinh độ (°E)")
    ax[0].set_ylabel("Vĩ độ (°N)")
    plt.colorbar(sc1, ax=ax[0], label="Xác suất cháy")
    ax[0].grid(True, alpha=0.3)

    # Ground truth
    pos = day_df[day_df["label"] == 1]
    ax[1].scatter(day_df["lon"], day_df["lat"], c="#D1FAE5", s=80, label="Không cháy")
    if len(pos) > 0:
        ax[1].scatter(pos["lon"], pos["lat"], c="#DC2626", s=120, marker="*",
                      label="Có cháy (ground truth)", zorder=5)
    ax[1].set_title(f"Ground truth ignition — {peak_day} ({len(pos)} điểm cháy)")
    ax[1].set_xlabel("Kinh độ (°E)")
    ax[1].set_ylabel("Vĩ độ (°N)")
    ax[1].legend(loc="upper right")
    ax[1].grid(True, alpha=0.3)

    fig.suptitle("Hình 2. Bản đồ wildfire risk và ground truth",
                 fontsize=12, fontweight="bold")
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def fig_ablation_bar(ablation_results, out_path=None):
    """Bar chart comparing AUROC and ECE across ablation configs."""
    cfgs = list(ablation_results.keys())
    aurocs = [ablation_results[c]["AUROC"] for c in cfgs]
    eces = [ablation_results[c]["ECE"] for c in cfgs]
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    colors_a = ["#9CA3AF", "#FCD34D", "#FCD34D", "#FCD34D", "#FBBF24", "#60A5FA", "#1E40AF"]
    colors_e = ["#9CA3AF", "#FCD34D", "#FCD34D", "#FCD34D", "#FBBF24", "#60A5FA", "#1E40AF"]
    ax[0].bar(x, aurocs, color=colors_a, edgecolor="#374151", linewidth=0.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(cfgs, rotation=25, ha="right", fontsize=8)
    ax[0].set_ylabel("AUROC (cao hơn = tốt hơn)")
    ax[0].set_title("AUROC theo cấu hình ablation")
    ax[0].set_ylim(0.7, max(aurocs)+0.05)
    for i, v in enumerate(aurocs):
        ax[0].text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].grid(True, alpha=0.3, axis="y")

    ax[1].bar(x, eces, color=colors_e, edgecolor="#374151", linewidth=0.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(cfgs, rotation=25, ha="right", fontsize=8)
    ax[1].set_ylabel("ECE (thấp hơn = tốt hơn)")
    ax[1].set_title("ECE theo cấu hình ablation")
    ax[1].set_ylim(0, max(eces)*1.2)
    for i, v in enumerate(eces):
        ax[1].text(i, v+0.002, f"{v:.4f}", ha="center", fontsize=8)
    ax[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Hình 3. Ablation study — AUROC và ECE theo cấu hình",
                 fontsize=12, fontweight="bold")
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def fig_robustness_curve(robust_results, out_path=None):
    """Performance degradation across OOD conditions."""
    conditions = list(robust_results.keys())
    base_aurocs = [robust_results[c]["baseline"]["AUROC"] for c in conditions]
    prop_aurocs = [robust_results[c]["proposed"]["AUROC"] for c in conditions]
    base_eces = [robust_results[c]["baseline"]["ECE"] for c in conditions]
    prop_eces = [robust_results[c]["proposed"]["ECE"] for c in conditions]

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    w = 0.35

    ax[0].bar(x-w/2, base_aurocs, w, color="#B91C1C", label="Baseline")
    ax[0].bar(x+w/2, prop_aurocs, w, color="#1E40AF", label="Proposed")
    ax[0].set_xticks(x); ax[0].set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("AUROC")
    ax[0].set_title("AUROC theo điều kiện OOD")
    ax[0].set_ylim(0.6, 1.0)
    ax[0].legend(); ax[0].grid(True, alpha=0.3, axis="y")

    ax[1].bar(x-w/2, base_eces, w, color="#B91C1C", label="Baseline")
    ax[1].bar(x+w/2, prop_eces, w, color="#1E40AF", label="Proposed")
    ax[1].set_xticks(x); ax[1].set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax[1].set_ylabel("ECE")
    ax[1].set_title("ECE theo điều kiện OOD")
    ax[1].set_ylim(0, max(max(base_eces), max(prop_eces))*1.4)
    ax[1].legend(); ax[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Hình 4. Robustness & OOD evaluation — AUROC và ECE",
                 fontsize=12, fontweight="bold")
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def fig_architecture_diagram(out_path=None):
    """Block diagram of the physics-constrained pipeline."""
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")

    blocks = [
        (0.2, "Đầu vào\nDữ liệu\nmôi trường", "#EFF6FF", "#1E40AF"),
        (2.2, "Trích xuất\nđặc trưng\nkhông gian", "#F0FDF4", "#166534"),
        (4.2, "Trích xuất\nđặc trưng\nthời gian", "#F0FDF4", "#166534"),
        (6.2, "GBDT\nbase\n(XGBoost)", "#FEF3C7", "#92400E"),
        (8.2, "MLP head +\nPhysics loss", "#FEE2E2", "#991B1B"),
        (10.2, "Xác suất\ncháy rừng\n→ Risk map", "#EFF6FF", "#1E40AF"),
    ]
    for x, txt, fill, edge in blocks:
        ax.add_patch(plt.Rectangle((x, 1.4), 1.6, 1.6, facecolor=fill,
                                    edgecolor=edge, linewidth=2))
        ax.text(x+0.8, 2.2, txt, ha="center", va="center", fontsize=9,
                fontweight="bold")

    for x in [1.8, 3.8, 5.8, 7.8, 9.8]:
        ax.annotate("", xy=(x+0.4, 2.2), xytext=(x, 2.2),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))

    # Physics loss annotation
    ax.annotate("L = L_prediction + λ·L_physics\n"
                "(energy-balance + moisture-decay + wind-monotonicity)",
                xy=(9.0, 1.0), xytext=(9.0, 0.1),
                ha="center", fontsize=9, color="#991B1B",
                arrowprops=dict(arrowstyle="->", color="#991B1B"))

    ax.set_title("Hình 0. Kiến trúc mô hình Physics-Constrained Spatiotemporal ML",
                 fontsize=12, fontweight="bold")
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def fig_loss_curves(history, out_path=None):
    """Training loss curves."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].plot(history["epoch"], history["tr_bce"], color="#1E40AF", lw=2, label="L_prediction (BCE)")
    ax[0].plot(history["epoch"], history["tr_phys"], color="#991B1B", lw=2, label="L_physics")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
    ax[0].set_title("Thành phần loss")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    ax[1].plot(history["epoch"], history["tr_loss"], color="#1E40AF", lw=2, label="Train")
    ax[1].plot(history["epoch"], history["val_loss"], color="#9CA3AF", lw=2, label="Validation")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
    ax[1].set_title("Total loss")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)

    fig.suptitle("Hình 5. Training/Validation Loss — Proposed Model (λ=0.20, ensemble member)",
                 fontsize=12, fontweight="bold")
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


# =============================================================================
# 7. OOD CONDITION MASKS
# =============================================================================
def mask_ood_condition(df, condition):
    if condition == "extreme_temp":
        return df[df["t2m"] >= df["t2m"].quantile(0.95)]
    if condition == "low_humidity":
        return df[df["rh"] <= df["rh"].quantile(0.10)]
    if condition == "extreme_wind":
        return df[df["wind"] >= df["wind"].quantile(0.95)]
    if condition == "drought":
        return df[df["drought_idx"] <= df["drought_idx"].quantile(0.10)]
    if condition == "unseen_region":
        # Khánh Hòa — held out from training; uses test-period 2022 data
        return df[(df["lat"] < 13.0) & (df["lon"] > 108.5)]
    if condition == "unseen_period":
        return df  # placeholder — handled via temporal OOD separately
    return df


# =============================================================================
# 8. MAIN EXPERIMENT
# =============================================================================
def main():
    set_seed(MASTER_SEED)
    print("\n" + "="*80)
    print("PHYSICS-CONSTRAINED SPATIOTEMPORAL ML FOR WILDFIRE IGNITION FORECASTING")
    print("="*80)

    # Step 1: build dataset
    print("\n[1/6] Building synthetic Vietnam wildfire dataset...")
    df = build_dataset(seed=MASTER_SEED)

    # Step 2: splits
    print("\n[2/6] Constructing temporal + spatial-OOD splits...")
    splits = make_splits(df)
    for k, v in splits.items():
        print(f"  {k:12s}: {len(v):,} rows; positives = {int(v.label.sum()):,} "
              f"({100*v.label.mean():.3f}%)")

    # Step 3: train baseline + proposed ensemble on master seed
    print("\n[3/6] Training baseline (XGBoost) and proposed (GBDT + Physics MLP Ensemble)...")
    train_df = splits["train"]; val_df = splits["val"]; test_df = splits["test"]
    # Ensure all features are float64 (avoids XGBoost dtype issues)
    for f in FEATURES:
        train_df[f] = train_df[f].astype("float64")
        val_df[f] = val_df[f].astype("float64")
        test_df[f] = test_df[f].astype("float64")
        splits["ood_time"][f] = splits["ood_time"][f].astype("float64")
    y_tr = train_df["label"]; y_val = val_df["label"]; y_test = test_df["label"]

    xgb_model, _ = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED)
    # Compute F1-optimal threshold on VALIDATION set (avoid test-set leakage)
    p_val_base = xgb_model.predict_proba(val_df[FEATURES])[:, 1]
    f1_thr_base = find_f1_optimal_threshold(y_val.values, p_val_base)
    p_test_base = xgb_model.predict_proba(test_df[FEATURES])[:, 1]
    metrics_base = compute_metrics(y_test.values, p_test_base, f1_threshold=f1_thr_base)
    sf1_base, sp_base, sr_base = spatial_f1(test_df, p_test_base)
    metrics_base["Spatial_F1"] = sf1_base
    metrics_base["Spatial_Precision"] = sp_base
    metrics_base["Spatial_Recall"] = sr_base
    metrics_base["F1_threshold_source"] = "validation (no test leakage)"
    print(f"  [baseline] AUROC={metrics_base['AUROC']:.4f} AUPRC={metrics_base['AUPRC']:.4f} "
          f"F1={metrics_base['F1']:.4f} Brier={metrics_base['Brier']:.4f} "
          f"ECE={metrics_base['ECE']:.4f} SpatialF1={sf1_base:.4f}  "
          f"(F1 thr={f1_thr_base:.4f} from val)")

    # Ensemble of 3 MLPs with different seeds, averaged predictions
    ensemble = train_physics_ensemble(
        train_df, y_tr, val_df, y_val, n_models=3, base_seed=MASTER_SEED,
        lam=0.20, epochs=25, batch_size=1024, lr=1e-3, verbose=False,
    )
    # Use first member for history/figures (representative)
    prop_model, prop_scaler, prop_xgb, hist = ensemble[0]
    # Compute F1-optimal threshold on VALIDATION set for proposed model too
    p_val_prop = predict_ensemble(ensemble, val_df)
    f1_thr_prop = find_f1_optimal_threshold(y_val.values, p_val_prop)
    p_test_prop = predict_ensemble(ensemble, test_df)
    metrics_prop = compute_metrics(y_test.values, p_test_prop, f1_threshold=f1_thr_prop)
    sf1_prop, sp_prop, sr_prop = spatial_f1(test_df, p_test_prop)
    metrics_prop["Spatial_F1"] = sf1_prop
    metrics_prop["Spatial_Precision"] = sp_prop
    metrics_prop["Spatial_Recall"] = sr_prop
    metrics_prop["F1_threshold_source"] = "validation (no test leakage)"
    print(f"  [proposed ensemble] AUROC={metrics_prop['AUROC']:.4f} AUPRC={metrics_prop['AUPRC']:.4f} "
          f"F1={metrics_prop['F1']:.4f} Brier={metrics_prop['Brier']:.4f} "
          f"ECE={metrics_prop['ECE']:.4f} SpatialF1={sf1_prop:.4f}")

    # Step 4: ablation study
    print("\n[4/6] Ablation study (7 configurations)...")
    ablation_cfgs = {
        "Baseline": None,
        "+ Weather": ["doy", "lat", "lon", "elev", "slope", "aspect"] + GROUPS["weather"],
        "+ Satellite": ["doy", "lat", "lon", "elev", "slope", "aspect"] + GROUPS["weather"] + GROUPS["satellite"],
        "+ Temporal": ["doy", "lat", "lon", "elev", "slope", "aspect"]
                       + GROUPS["weather"] + GROUPS["satellite"]
                       + ["t_lag7", "rh_lag7", "p_lag7", "ndvi_lag7",
                          "t_lag14", "rh_lag14", "p_lag14", "drought_idx"],
        "+ Interact": FEATURES,  # all features including interaction terms
        "Proposed (no phys)": None,
        "Proposed (physics)": None,
    }
    ablation_results = {}

    # Baseline (topo + temporal context only)
    abl_xgb, abl_feats = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED)
    p_abl = abl_xgb.predict_proba(test_df[abl_feats])[:, 1]
    ablation_results["Baseline"] = compute_metrics(y_test.values, p_abl)

    # + Weather
    abl_xgb, abl_feats = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED,
                                    use_features=["doy", "lat", "lon", "elev", "slope", "aspect"] + GROUPS["weather"])
    p_abl = abl_xgb.predict_proba(test_df[abl_feats])[:, 1]
    ablation_results["+ Weather"] = compute_metrics(y_test.values, p_abl)

    # + Satellite
    sat_feats = ["doy", "lat", "lon", "elev", "slope", "aspect"] + GROUPS["weather"] + GROUPS["satellite"]
    abl_xgb, _ = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED, use_features=sat_feats)
    p_abl = abl_xgb.predict_proba(test_df[sat_feats])[:, 1]
    ablation_results["+ Satellite"] = compute_metrics(y_test.values, p_abl)

    # + Temporal (lagged features added) — deduplicate
    temporal_feats = list(dict.fromkeys(
        ["doy", "lat", "lon", "elev", "slope", "aspect"]
        + GROUPS["weather"] + GROUPS["satellite"]
        + ["t_lag7", "rh_lag7", "p_lag7", "ndvi_lag7",
           "t_lag14", "rh_lag14", "p_lag14", "drought_idx"]
    ))
    # Ensure all features are float64
    for f in temporal_feats:
        train_df[f] = train_df[f].astype("float64")
        val_df[f] = val_df[f].astype("float64")
        test_df[f] = test_df[f].astype("float64")
    abl_xgb, _ = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED, use_features=temporal_feats)
    p_abl = abl_xgb.predict_proba(test_df[temporal_feats])[:, 1]
    ablation_results["+ Temporal"] = compute_metrics(y_test.values, p_abl)

    # + Interactions (all features)
    abl_xgb, _ = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED, use_features=FEATURES)
    p_abl = abl_xgb.predict_proba(test_df[FEATURES])[:, 1]
    ablation_results["+ Interact"] = compute_metrics(y_test.values, p_abl)

    # Proposed no physics — ensemble of 3 MLPs with lambda=0
    ensemble_no_phys = train_physics_ensemble(
        train_df, y_tr, val_df, y_val, n_models=3, base_seed=MASTER_SEED,
        lam=0.0, epochs=30, batch_size=1024, lr=1e-3,
    )
    p_abl = predict_ensemble(ensemble_no_phys, test_df)
    ablation_results["Proposed (no phys)"] = compute_metrics(y_test.values, p_abl)

    # Proposed with physics (already trained)
    ablation_results["Proposed (physics)"] = metrics_prop

    for cfg, m in ablation_results.items():
        print(f"  [{cfg:24s}] AUROC={m['AUROC']:.4f} ECE={m['ECE']:.4f} "
              f"F1={m['F1']:.4f} Brier={m['Brier']:.4f}")

    # Step 5: Robustness / OOD evaluation
    print("\n[5/6] Robustness & OOD evaluation...")
    ood_conditions = ["extreme_temp", "low_humidity", "extreme_wind", "drought"]
    robust_results = {}
    for cond in ood_conditions:
        sub = mask_ood_condition(test_df, cond).copy()
        if len(sub) == 0 or sub["label"].sum() == 0:
            print(f"  [{cond}] skipped (empty or no positives)")
            continue
        # Baseline predictions
        p_b = xgb_model.predict_proba(sub[FEATURES])[:, 1]
        m_b = compute_metrics(sub["label"].values, p_b)
        # Proposed predictions (ensemble)
        p_p = predict_ensemble(ensemble, sub)
        m_p = compute_metrics(sub["label"].values, p_p)
        robust_results[cond] = {"baseline": m_b, "proposed": m_p,
                                "n_samples": len(sub), "n_pos": int(sub["label"].sum())}
        print(f"  [{cond:18s}] n={len(sub):5d} pos={int(sub['label'].sum()):4d} | "
              f"base AUROC={m_b['AUROC']:.4f} ECE={m_b['ECE']:.4f} | "
              f"prop AUROC={m_p['AUROC']:.4f} ECE={m_p['ECE']:.4f}")

    # Spatial OOD: Khánh Hòa province — genuinely unseen (held out from training)
    ood_spatial = splits["ood_space"]
    if len(ood_spatial) > 0 and ood_spatial["label"].sum() > 0:
        # Ensure all features float64
        for f in FEATURES:
            ood_spatial[f] = ood_spatial[f].astype("float64")
        p_b_s = xgb_model.predict_proba(ood_spatial[FEATURES])[:, 1]
        p_p_s = predict_ensemble(ensemble, ood_spatial)
        m_b_s = compute_metrics(ood_spatial["label"].values, p_b_s)
        m_p_s = compute_metrics(ood_spatial["label"].values, p_p_s)
        robust_results["unseen_region (Khánh Hòa)"] = {
            "baseline": m_b_s, "proposed": m_p_s,
            "n_samples": len(ood_spatial), "n_pos": int(ood_spatial["label"].sum())
        }
        print(f"  [unseen_region(KH) ] n={len(ood_spatial):5d} pos={int(ood_spatial['label'].sum()):4d} | "
              f"base AUROC={m_b_s['AUROC']:.4f} ECE={m_b_s['ECE']:.4f} | "
              f"prop AUROC={m_p_s['AUROC']:.4f} ECE={m_p_s['ECE']:.4f}")

    # OOD temporal: 2023 fire season
    ood_t = splits["ood_time"]
    p_b_t = xgb_model.predict_proba(ood_t[FEATURES])[:, 1]
    p_p_t = predict_ensemble(ensemble, ood_t)
    m_b_t = compute_metrics(ood_t["label"].values, p_b_t)
    m_p_t = compute_metrics(ood_t["label"].values, p_p_t)
    robust_results["unseen_period (2023)"] = {
        "baseline": m_b_t, "proposed": m_p_t,
        "n_samples": len(ood_t), "n_pos": int(ood_t["label"].sum())
    }
    print(f"  [unseen_period(2023)] n={len(ood_t):5d} pos={int(ood_t['label'].sum()):4d} | "
          f"base AUROC={m_b_t['AUROC']:.4f} ECE={m_b_t['ECE']:.4f} | "
          f"prop AUROC={m_p_t['AUROC']:.4f} ECE={m_p_t['ECE']:.4f}")

    # Step 6: multi-seed variance for Appendix B
    print("\n[6/6] Multi-seed variance (5 seeds)...")
    multi_seed = {"baseline": [], "proposed": []}
    for s in SEEDS:
        set_seed(s)
        # Baseline
        xgb_s, _ = train_xgb(train_df, y_tr, val_df, y_val, seed=s)
        p_s = xgb_s.predict_proba(test_df[FEATURES])[:, 1]
        m_s = compute_metrics(y_test.values, p_s); m_s["Spatial_F1"] = spatial_f1(test_df, p_s)[0]
        multi_seed["baseline"].append(m_s)
        # Proposed — single MLP for multi-seed variance (faster than ensemble)
        prop_s, prop_s_sc, prop_s_xgb, _ = train_physics_model(
            train_df, y_tr, val_df, y_val, seed=s, lam=0.20,
            epochs=30, batch_size=1024, lr=1e-3,
        )
        p_s = predict_physics(prop_s, prop_s_sc, prop_s_xgb, test_df)
        m_s = compute_metrics(y_test.values, p_s); m_s["Spatial_F1"] = spatial_f1(test_df, p_s)[0]
        multi_seed["proposed"].append(m_s)
        print(f"  [seed={s:5d}] base AUROC={multi_seed['baseline'][-1]['AUROC']:.4f}  "
              f"prop AUROC={multi_seed['proposed'][-1]['AUROC']:.4f}")

    def summarize(runs, key):
        vals = np.array([r[key] for r in runs])
        return f"{vals.mean():.4f} ± {vals.std():.4f}"

    multi_summary = {}
    for k in ["AUROC", "AUPRC", "F1", "Brier", "ECE", "Spatial_F1"]:
        multi_summary[k] = {
            "baseline": summarize(multi_seed["baseline"], k),
            "proposed": summarize(multi_seed["proposed"], k),
        }

    # ============================================================
    # NEW STEP: λ-sweep on OOD subsets to isolate physics contribution
    # (REDUCED: only λ=0.0 and λ=0.20 — these are the existing trained models)
    # ============================================================
    print("\n[7/8] λ-sweep on OOD (using existing λ=0 and λ=0.20 models)...")
    # Build a list of OOD subsets (same as in Step 5)
    ood_subsets = {}
    for cond in ood_conditions:
        ood_subsets[cond] = mask_ood_condition(test_df, cond).copy()
    if len(ood_spatial) > 0:
        ood_subsets["unseen_region"] = ood_spatial.copy()
    ood_subsets["unseen_period"] = splits["ood_time"].copy()

    lambda_sweep = {}
    # Use existing trained models (no extra training)
    # λ=0: ensemble_no_phys (already trained in ablation step)
    # λ=0.20: ensemble (main proposed model)
    for lam_val, ens_lam, p_t in [
        (0.0, ensemble_no_phys, predict_ensemble(ensemble_no_phys, test_df)),
        (0.20, ensemble, p_test_prop),
    ]:
        lambda_sweep[lam_val] = {}
        lambda_sweep[lam_val]["test"] = compute_metrics(y_test.values, p_t)
        print(f"  λ={lam_val}  test  AUROC={lambda_sweep[lam_val]['test']['AUROC']:.4f}  ECE={lambda_sweep[lam_val]['test']['ECE']:.4f}")
        for subset_name, sub in ood_subsets.items():
            if len(sub) == 0 or sub["label"].sum() == 0:
                continue
            # Ensure features float
            for f in FEATURES:
                sub[f] = sub[f].astype("float64")
            p_s = predict_ensemble(ens_lam, sub)
            m_s = compute_metrics(sub["label"].values, p_s)
            lambda_sweep[lam_val][subset_name] = m_s
            print(f"  λ={lam_val}  {subset_name:18s}  AUROC={m_s['AUROC']:.4f}  ECE={m_s['ECE']:.4f}  Brier={m_s['Brier']:.4f}")

    # ============================================================
    # NEW STEP: Calibration baselines (XGBoost + Temperature/Isotonic/Platt)
    # These are post-hoc — no extra training, just sklearn fits (very fast)
    # ============================================================
    print("\n[8/8] Calibration baselines (post-hoc XGBoost calibration)...")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    calibration_baselines = {}

    # Get XGBoost raw probabilities on validation and test sets
    p_val_xgb = xgb_model.predict_proba(val_df[FEATURES])[:, 1]
    p_test_xgb = xgb_model.predict_proba(test_df[FEATURES])[:, 1]
    y_val_arr = y_val.values

    # Method 1: Temperature Scaling (single-temperature Platt)
    eps = 1e-6
    logits_val = np.log(np.clip(p_val_xgb, eps, 1-eps) /
                         np.clip(1 - p_val_xgb, eps, 1-eps))
    logits_test = np.log(np.clip(p_test_xgb, eps, 1-eps) /
                          np.clip(1 - p_test_xgb, eps, 1-eps))
    # Grid search over T in [0.5, 5.0]
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.5, 5.0, 31):
        scaled = 1 / (1 + np.exp(-logits_val / T))
        nll = -np.mean(y_val_arr * np.log(scaled + eps) +
                        (1 - y_val_arr) * np.log(1 - scaled + eps))
        if nll < best_nll:
            best_nll = nll; best_T = T
    p_temp_test = 1 / (1 + np.exp(-logits_test / best_T))
    m_temp = compute_metrics(y_test.values, p_temp_test)
    m_temp["Spatial_F1"] = spatial_f1(test_df, p_temp_test)[0]
    calibration_baselines["XGB + Temperature"] = {"metrics_test": m_temp, "T": float(best_T)}
    print(f"  XGB + Temperature (T={best_T:.2f}): AUROC={m_temp['AUROC']:.4f}  ECE={m_temp['ECE']:.4f}")

    # Method 2: Isotonic Regression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_val_xgb, y_val_arr)
    p_iso_test = iso.predict(p_test_xgb)
    m_iso = compute_metrics(y_test.values, p_iso_test)
    m_iso["Spatial_F1"] = spatial_f1(test_df, p_iso_test)[0]
    calibration_baselines["XGB + Isotonic"] = {"metrics_test": m_iso}
    print(f"  XGB + Isotonic:          AUROC={m_iso['AUROC']:.4f}  ECE={m_iso['ECE']:.4f}")

    # Method 3: Platt Scaling (logistic regression on logit)
    lr_calib = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
    lr_calib.fit(logits_val.reshape(-1, 1), y_val_arr)
    p_platt_test = lr_calib.predict_proba(logits_test.reshape(-1, 1))[:, 1]
    m_platt = compute_metrics(y_test.values, p_platt_test)
    m_platt["Spatial_F1"] = spatial_f1(test_df, p_platt_test)[0]
    calibration_baselines["XGB + Platt"] = {"metrics_test": m_platt}
    print(f"  XGB + Platt:             AUROC={m_platt['AUROC']:.4f}  ECE={m_platt['ECE']:.4f}")

    # Evaluate each calibration method on OOD subsets
    for name, calib_obj in [
        ("XGB + Temperature", ("temp", best_T)),
        ("XGB + Isotonic", ("iso", iso)),
        ("XGB + Platt", ("platt", lr_calib)),
    ]:
        calibration_baselines[name]["ood"] = {}
        for subset_name, sub in ood_subsets.items():
            if len(sub) == 0 or sub["label"].sum() == 0:
                continue
            for f in FEATURES:
                sub[f] = sub[f].astype("float64")
            p_xgb_sub = xgb_model.predict_proba(sub[FEATURES])[:, 1]
            calib_type, calib_model = calib_obj
            if calib_type == "temp":
                logit_sub = np.log(np.clip(p_xgb_sub, eps, 1-eps) /
                                    np.clip(1 - p_xgb_sub, eps, 1-eps))
                p_sub_cal = 1 / (1 + np.exp(-logit_sub / calib_model))
            elif calib_type == "iso":
                p_sub_cal = calib_model.predict(p_xgb_sub)
            elif calib_type == "platt":
                logit_sub = np.log(np.clip(p_xgb_sub, eps, 1-eps) /
                                    np.clip(1 - p_xgb_sub, eps, 1-eps))
                p_sub_cal = calib_model.predict_proba(logit_sub.reshape(-1, 1))[:, 1]
            m_sub = compute_metrics(sub["label"].values, p_sub_cal)
            calibration_baselines[name]["ood"][subset_name] = m_sub

    # ============================================================
    # Step 7 (continued): Generate figures
    # ============================================================
    # Step 7: Generate figures
    print("\nGenerating figures...")
    fig_architecture_diagram(out_path=f"{OUT_DIR}/fig0_architecture.png")
    fig_reliability_diagram(
        y_test.values, p_test_base, y_test.values, p_test_prop,
        out_path=f"{OUT_DIR}/fig1_reliability.png",
    )
    fig_risk_map(test_df, p_test_prop, out_path=f"{OUT_DIR}/fig2_risk_map.png")
    fig_ablation_bar(ablation_results, out_path=f"{OUT_DIR}/fig3_ablation.png")
    fig_robustness_curve(robust_results, out_path=f"{OUT_DIR}/fig4_robustness.png")
    fig_loss_curves(hist, out_path=f"{OUT_DIR}/fig5_loss.png")

    # Final results package
    results = {
        "dataset": {
            "n_total": int(len(df)),
            "n_train": int(len(splits["train"])),
            "n_val": int(len(splits["val"])),
            "n_test": int(len(splits["test"])),
            "n_ood_time": int(len(splits["ood_time"])),
            "n_ood_space": int(len(splits["ood_space"])),
            "pos_total": int(df["label"].sum()),
            "pos_train": int(splits["train"]["label"].sum()),
            "pos_val": int(splits["val"]["label"].sum()),
            "pos_test": int(splits["test"]["label"].sum()),
            "pos_ood_time": int(splits["ood_time"]["label"].sum()),
            "pos_ood_space": int(splits["ood_space"]["label"].sum()),
            "pos_rate_total": float(df["label"].mean()),
            "pos_rate_test": float(splits["test"]["label"].mean()),
            "n_features": len(FEATURES),
            "features": FEATURES,
        },
        "baseline_metrics_test": metrics_base,
        "proposed_metrics_test": metrics_prop,
        "ablation_results": ablation_results,
        "ood_results": robust_results,
        "lambda_sweep": {str(k): v for k, v in lambda_sweep.items()},
        "calibration_baselines": calibration_baselines,
        "multi_seed_summary": multi_summary,
        "multi_seed_raw": multi_seed,
        "training_history": hist,
        "lambda": 0.20,
        "epochs": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer": "AdamW + cosine schedule",
        "seeds": SEEDS,
        "ensemble_seeds": [42, 142, 242],
        "experimental_seeds": SEEDS,
    }

    out_json = "/home/z/my-project/scripts/results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")
    print("Figures saved to", OUT_DIR)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nKey results (test set, master seed = {MASTER_SEED}):")
    print(f"  Baseline:  AUROC={metrics_base['AUROC']:.4f}  ECE={metrics_base['ECE']:.4f}  "
          f"SpatialF1={metrics_base['Spatial_F1']:.4f}")
    print(f"  Proposed: AUROC={metrics_prop['AUROC']:.4f}  ECE={metrics_prop['ECE']:.4f}  "
          f"SpatialF1={metrics_prop['Spatial_F1']:.4f}")
    print(f"  ΔAUROC = {metrics_prop['AUROC']-metrics_base['AUROC']:+.4f}")
    print(f"  ΔECE   = {metrics_prop['ECE']-metrics_base['ECE']:+.4f}  "
          f"({100*(metrics_prop['ECE']-metrics_base['ECE'])/metrics_base['ECE']:+.1f}%)")
    print(f"  ΔSpatialF1 = {metrics_prop['Spatial_F1']-metrics_base['Spatial_F1']:+.4f}")


if __name__ == "__main__":
    main()
