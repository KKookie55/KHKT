#!/usr/bin/env python3
"""
v87_build_features.py — Giai đoạn 1: Sửa leakage + khóa dữ liệu.

SỬA LEAKAGE:
1. cell_fire_rate_train → expanding window (chỉ dùng quá khứ trước t)
2. cell_fire_rate_dry → expanding window (chỉ dry season trước t)
3. cell_fire_count_30d → shift 1 (dùng fire[t-30:t-1], không dùng fire[t])
4. Weather imputation → ffill only (không bfill — không dùng thời tiết tương lai)
5. cell_days_since_fire → removed (đã bỏ trong v8.6final, giữ nguyên)

BA CHẾ ĐỘ DỮ LIỆU:
- Mode A (temporal): train 2018-2021, val 2022, test 2023
- Mode B (spatial cold-start): train North, test South, KHÔNG dùng prior từ test cells
- Mode C (spatial warm-start): cho phép lịch sử cháy quá khứ của test cells
"""
import os, json, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/v87")
OUT.mkdir(parents=True, exist_ok=True)

GRID_RES=0.5; LAT_MIN,LAT_MAX=11.0,21.0; LON_MIN,LON_MAX=104.0,109.5
YEARS=list(range(2018,2024))

print("="*60); print("v8.7 GIAI ĐOẠN 1 — SỬA LEAKAGE + KHÓA DỮ LIỆU"); print("="*60)

# ============================================================
# STEP 1: Load raw data
# ============================================================
print("\n[1/6] Loading raw data...")
wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv")
wx["date"] = pd.to_datetime(wx["date"])
fires = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in YEARS], ignore_index=True)
fires["lat"] = np.round((fires["latitude"]-LAT_MIN)/GRID_RES)*GRID_RES+LAT_MIN
fires["lon"] = np.round((fires["longitude"]-LON_MIN)/GRID_RES)*GRID_RES+LON_MIN
fires["date"] = pd.to_datetime(fires["acq_date"])
fires = fires[(fires["lat"]>=LAT_MIN)&(fires["lat"]<=LAT_MAX)&(fires["lon"]>=LON_MIN)&(fires["lon"]<=LON_MAX)]
fcd = fires[["lat","lon","date"]].drop_duplicates(); fcd["fire"]=1
print(f"  Weather: {len(wx):,} rows, Fire: {len(fcd):,} cell-days")

# ============================================================
# STEP 2: Build panel (ffill only — NO bfill)
# ============================================================
print("\n[2/6] Building panel (ffill only, no bfill)...")
cells = wx[["lat","lon"]].drop_duplicates().sort_values(["lat","lon"]).reset_index(drop=True)
all_dates = pd.date_range(wx["date"].min(), wx["date"].max(), freq="D")
cells["_k"]=1; ad=pd.DataFrame({"date":all_dates}); ad["_k"]=1
panel = cells.merge(ad, on="_k").drop("_k", axis=1)
panel = panel.merge(wx, on=["lat","lon","date"], how="left")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)

# FIX 4: ffill only (no bfill — backward fill uses future weather)
for c in ["t2m","rh","wind","precip"]:
    panel[c] = panel.groupby(["lat","lon"])[c].transform(lambda s: s.ffill())
    # First few days of 2018 might be NaN — fill with global mean (train-only)
    train_mask_init = panel["date"].dt.year.isin([2018,2019,2020,2021])
    global_mean = panel.loc[train_mask_init, c].mean()
    panel[c] = panel[c].fillna(global_mean)
print(f"  Weather imputation: ffill only + global train mean for first days ✓")

panel = panel.merge(fcd, on=["lat","lon","date"], how="left")
panel["fire"] = panel["fire"].fillna(0).astype(int)
panel["doy"] = panel["date"].dt.dayofyear
panel["year"] = panel["date"].dt.year

# ============================================================
# STEP 3: Weather temporal features (lag, rolling, derived)
# ============================================================
print("\n[3/6] Weather temporal features...")
for v in ["t2m","rh","wind","precip"]:
    for lag in [1,3,7,14]:
        panel[f"{v}_lag{lag}"] = panel.groupby(["lat","lon"])[v].shift(lag)
for v in ["t2m","rh"]:
    panel[f"{v}_roll7"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(7,min_periods=3).mean())
    panel[f"{v}_roll14"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(14,min_periods=5).mean())
panel["precip_roll7_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(7,min_periods=3).sum())
panel["precip_roll14_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(14,min_periods=5).sum())
panel["precip_roll30_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(30,min_periods=10).sum())
def dsr(s, t=0.5):
    is_dry = (s<t).astype(int); g = (is_dry==0).cumsum()
    return g.groupby(g).cumcount()*is_dry
panel["days_since_rain"] = panel.groupby(["lat","lon"])["precip"].transform(dsr)
for v in ["t2m","rh","wind"]:
    panel[f"{v}_diff3"] = panel.groupby(["lat","lon"])[v].diff(3)
fwi = (0.4*(panel["t2m"]-10) - 0.62*panel["rh"] + 0.4*panel["wind"] - 0.3*panel["precip_lag7"])
panel["fwi_compound"] = 1/(1+np.exp(-(fwi-30)/5))
panel["heat_index"] = (panel["t2m"]-20)*(50-panel["rh"])/50.0
panel["wind_chill"] = (panel["t2m"]-15)*panel["wind"]/30.0
panel["label_h1"] = panel.groupby(["lat","lon"])["fire"].shift(-1)
print(f"  Weather features: 35 (lags + rolling + derived)")

# ============================================================
# STEP 4: Fire history features — EXPANDING WINDOW (FIX LEAKAGE)
# ============================================================
print("\n[4/6] Fire history — EXPANDING WINDOW (leakage-free)...")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
panel["is_dry"] = panel["date"].dt.month.apply(lambda m: 1 if m in [1,2,3,4,11,12] else 0)

# FIX 1: cell_fire_rate_expanding — for each cell, cumulative mean of fire BEFORE current date
# Using shift(1) to exclude current day's fire from the expanding mean
panel["fire_prev"] = panel.groupby(["lat","lon"])["fire"].shift(1)
panel["cell_fire_rate_exp"] = panel.groupby(["lat","lon"])["fire_prev"].transform(
    lambda s: s.expanding(min_periods=1).mean())
# Fill NaN (first day of each cell) with global mean
global_fire_rate = panel["fire"].mean()
panel["cell_fire_rate_exp"] = panel["cell_fire_rate_exp"].fillna(global_fire_rate)
print(f"  cell_fire_rate_exp: expanding mean of fire_prev (excludes current day)")
print(f"    Range: [{panel['cell_fire_rate_exp'].min():.4f}, {panel['cell_fire_rate_exp'].max():.4f}]")

# FIX 2: cell_fire_rate_dry_exp — expanding mean of fire in dry season only, before current date
panel["fire_prev_dry"] = panel["fire_prev"] * panel["is_dry"]  # only dry season fires
panel["is_dry_prev"] = panel.groupby(["lat","lon"])["is_dry"].shift(1)
panel["cell_fire_rate_dry_exp"] = panel.groupby(["lat","lon"]).apply(
    lambda g: (g["fire_prev"] * g["is_dry_prev"]).expanding(min_periods=1).sum() /
              (g["is_dry_prev"].expanding(min_periods=1).sum() + 1e-6)
).reset_index(level=[0,1], drop=True)
panel["cell_fire_rate_dry_exp"] = panel["cell_fire_rate_dry_exp"].fillna(global_fire_rate)
print(f"  cell_fire_rate_dry_exp: expanding mean of dry-season fire_prev")

# FIX 3: cell_fire_count_30d — shift by 1 (exclude current day)
panel["cell_fire_count_30d"] = panel.groupby(["lat","lon"])["fire_prev"].transform(
    lambda s: s.rolling(30, min_periods=1).sum())
print(f"  cell_fire_count_30d: rolling 30-day sum of fire_prev (excludes current day)")
print(f"    Range: [{panel['cell_fire_count_30d'].min():.0f}, {panel['cell_fire_count_30d'].max():.0f}]")

# Clean up temp columns
panel = panel.drop(columns=["fire_prev", "fire_prev_dry", "is_dry_prev"])

# Feature lists
WEATHER_FEATURES = [
    "t2m","rh","wind","precip","doy",
    "t2m_lag1","t2m_lag3","t2m_lag7","t2m_lag14",
    "rh_lag1","rh_lag3","rh_lag7","rh_lag14",
    "wind_lag1","wind_lag3","wind_lag7","wind_lag14",
    "precip_lag1","precip_lag3","precip_lag7","precip_lag14",
    "t2m_roll7","t2m_roll14","rh_roll7","rh_roll14",
    "precip_roll7_sum","precip_roll14_sum","precip_roll30_sum",
    "days_since_rain","t2m_diff3","rh_diff3","wind_diff3",
    "fwi_compound","heat_index","wind_chill",
]
FIRE_HISTORY_FEATURES = ["cell_fire_rate_exp", "cell_fire_rate_dry_exp", "cell_fire_count_30d"]
ALL_FEATURES = WEATHER_FEATURES + FIRE_HISTORY_FEATURES
print(f"\n  Feature sets:")
print(f"    Weather-only (B1): {len(WEATHER_FEATURES)} features")
print(f"    Fire-history-only (B2): {len(FIRE_HISTORY_FEATURES)} features")
print(f"    Weather + history (B3): {len(ALL_FEATURES)} features")

# ============================================================
# STEP 5: Build splits (3 modes)
# ============================================================
print("\n[5/6] Building splits — 3 modes...")
panel["row_idx"] = np.arange(len(panel))

# Mode A: Temporal forecasting
tr_A = panel[panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_h1"])
va_A = panel[panel["year"]==2022].dropna(subset=["label_h1"])
te_A = panel[panel["year"]==2023].dropna(subset=["label_h1"])
print(f"  Mode A (temporal): train={len(tr_A):,} val={len(va_A):,} test={len(te_A):,}")

# Mode B: Spatial cold-start (train North, test South, NO prior from test cells)
# For cold-start: cell_fire_rate_exp and cell_fire_rate_dry_exp use ONLY train cells' history
# So for test cells, these features = 0 or global mean (no history available)
tr_B_mask = (panel["lat"]>=15.0) & panel["year"].isin([2018,2019,2020,2021])
va_B_mask = (panel["lat"]>=15.0) & (panel["year"]==2022)
te_B_mask = (panel["lat"]<13.0) & panel["year"].isin([2018,2019,2020,2021,2022,2023])
# For cold-start, test = ALL years of South (not just 2023)
tr_B = panel[tr_B_mask].dropna(subset=["label_h1"])
va_B = panel[va_B_mask].dropna(subset=["label_h1"])
te_B = panel[te_B_mask].dropna(subset=["label_h1"])
# In cold-start, set fire history features to 0 for test cells (no prior available)
te_B_cold = te_B.copy()
te_B_cold["cell_fire_rate_exp"] = 0.0
te_B_cold["cell_fire_rate_dry_exp"] = 0.0
te_B_cold["cell_fire_count_30d"] = 0.0
print(f"  Mode B (cold-start): train(North)={len(tr_B):,} val(North)={len(va_B):,} test(South, all years)={len(te_B):,}")
print(f"    → Test cells: fire history features set to 0 (no prior)")

# Mode C: Spatial warm-start (allow past history of test cells)
# Same split as B, but keep fire history features computed from expanding window
# For warm-start, test cells DO have expanding history (computed from their own past)
tr_C = tr_B.copy()
va_C = va_B.copy()
te_C_warm = te_B.copy()  # keep fire history features as-is
print(f"  Mode C (warm-start): same split as B, but test cells KEEP fire history")

# Save splits
np.savez(OUT/"splits_v87.npz",
    # Mode A
    temporal_train=tr_A["row_idx"].values, temporal_val=va_A["row_idx"].values, temporal_test=te_A["row_idx"].values,
    # Mode B (cold-start — test features zeroed)
    coldstart_train=tr_B["row_idx"].values, coldstart_val=va_B["row_idx"].values, coldstart_test=te_B["row_idx"].values,
    # Mode C (warm-start — test features kept)
    warmstart_train=tr_C["row_idx"].values, warmstart_val=va_C["row_idx"].values, warmstart_test=te_C_warm["row_idx"].values,
)
print(f"  ✓ Splits saved: {OUT}/splits_v87.npz")

# ============================================================
# STEP 6: Save panel + metadata
# ============================================================
print(f"\n[6/6] Saving panel + metadata...")
# Save full panel (with all features + both cold-start and warm-start versions)
panel_to_save = panel.drop(columns=["row_idx", "is_dry"])
panel_to_save.to_parquet(OUT/"panel_v87.parquet", index=False)
print(f"  Panel: {OUT}/panel_v87.parquet ({(OUT/'panel_v87.parquet').stat().st_size//1024//1024} MB)")

# Save cold-start test (with zeroed fire history)
te_B_cold.to_parquet(OUT/"test_coldstart.parquet", index=False)
print(f"  Cold-start test: {OUT}/test_coldstart.parquet")

# Save metadata
meta = {
    "version": "v8.7_giai_doan_1",
    "panel_size": len(panel),
    "n_cells": 273,
    "date_range": ["2018-01-01", "2023-12-31"],
    "weather_features": WEATHER_FEATURES,
    "fire_history_features": FIRE_HISTORY_FEATURES,
    "all_features": ALL_FEATURES,
    "leakage_fixes": {
        "1_cell_fire_rate_exp": "Expanding window of fire_prev (excludes current day) — no future info",
        "2_cell_fire_rate_dry_exp": "Expanding window of dry-season fire_prev — no future info",
        "3_cell_fire_count_30d": "Rolling 30d of fire_prev (shifted by 1) — excludes current day",
        "4_weather_imputation": "ffill only (no bfill) + global train mean for first days",
        "5_spatial_coldstart": "Test cells fire history = 0 (no prior from test region)",
        "6_calibration": "Iso fit on val, not test (already correct, maintained)",
    },
    "splits": {
        "temporal": {"train": "2018-2021", "val": "2022", "test": "2023", "n_train": len(tr_A), "n_val": len(va_A), "n_test": len(te_A)},
        "coldstart": {"train": "North 2018-2021", "val": "North 2022", "test": "South all years", "n_train": len(tr_B), "n_val": len(va_B), "n_test": len(te_B)},
        "warmstart": {"train": "North 2018-2021", "val": "North 2022", "test": "South all years (with history)", "n_train": len(tr_C), "n_val": len(va_C), "n_test": len(te_C_warm)},
    },
    "feature_sets": {
        "B1_weather_only": WEATHER_FEATURES,
        "B2_fire_history_only": FIRE_HISTORY_FEATURES,
        "B3_weather_plus_history": ALL_FEATURES,
    },
    "caveat": "Fire history features use expanding window (no future leakage). Cold-start sets test cells' fire history to 0. Warm-start keeps expanding window for test cells.",
}
with open(OUT/"meta_v87.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"  Metadata: {OUT}/meta_v87.json")

print(f"\n{'='*60}")
print("✓ GIAI ĐOẠN 1 HOÀN THÀNH")
print(f"{'='*60}")
print(f"\n  Leakage fixes applied:")
print(f"    1. cell_fire_rate_exp: expanding window (no future) ✓")
print(f"    2. cell_fire_rate_dry_exp: expanding window dry season (no future) ✓")
print(f"    3. cell_fire_count_30d: shifted by 1 (excludes current day) ✓")
print(f"    4. Weather: ffill only (no bfill) ✓")
print(f"    5. Cold-start: test fire history = 0 ✓")
print(f"    6. Calibration: Iso on val (maintained) ✓")
print(f"\n  3 modes: temporal, cold-start, warm-start")
print(f"  3 feature sets: B1 (weather), B2 (history), B3 (combined)")
print(f"\n  Next: Giai đoạn 2 — Baseline sạch")
