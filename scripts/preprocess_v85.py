#!/usr/bin/env python3
"""v8.5 PREPROCESSING — calendar-aware, leakage-safe, forecast horizons."""
import os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/real_data_v85")
OUT.mkdir(parents=True, exist_ok=True)
GRID_RES = 0.5; LAT_MIN, LAT_MAX = 11.0, 21.0; LON_MIN, LON_MAX = 104.0, 109.5
YEARS = list(range(2018, 2024)); HORIZONS = [1, 3, 7]

print("="*70); print("v8.5 PREPROCESSING"); print("="*70)
print("[1/6] Loading weather...")
wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv")
wx["date"] = pd.to_datetime(wx["date"])
print(f"  Weather: {len(wx):,} rows, {wx['lat'].nunique()}x{wx['lon'].nunique()} grid")

print("[2/6] Loading MODIS fires → cell-day...")
fires = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in YEARS], ignore_index=True)
print(f"  Raw MODIS: {len(fires):,} detections")
fires["lat"] = np.round((fires["latitude"]-LAT_MIN)/GRID_RES)*GRID_RES+LAT_MIN
fires["lon"] = np.round((fires["longitude"]-LON_MIN)/GRID_RES)*GRID_RES+LON_MIN
fires["date"] = pd.to_datetime(fires["acq_date"])
fires = fires[(fires["lat"]>=LAT_MIN)&(fires["lat"]<=LAT_MAX)&(fires["lon"]>=LON_MIN)&(fires["lon"]<=LON_MAX)]
fire_cell_day = fires[["lat","lon","date"]].drop_duplicates()
fire_cell_day["fire"] = 1
print(f"  Unique fire cell-days: {len(fire_cell_day):,}")

print("[3/6] Building full daily panel...")
cells = wx[["lat","lon"]].drop_duplicates().sort_values(["lat","lon"]).reset_index(drop=True)
all_dates = pd.date_range(wx["date"].min(), wx["date"].max(), freq="D")
cells["_k"]=1; ad=pd.DataFrame({"date":all_dates}); ad["_k"]=1
panel = cells.merge(ad, on="_k").drop("_k", axis=1)
panel = panel.merge(wx, on=["lat","lon","date"], how="left")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
for c in ["t2m","rh","wind","precip"]:
    panel[c] = panel.groupby(["lat","lon"])[c].transform(lambda s: s.ffill().bfill())
panel = panel.merge(fire_cell_day, on=["lat","lon","date"], how="left")
panel["fire"] = panel["fire"].fillna(0).astype(int)
print(f"  Panel: {len(panel):,} rows, fire prevalence: {100*panel['fire'].mean():.2f}%")

print("[4/6] Calendar-aware temporal features...")
panel["doy"] = panel["date"].dt.dayofyear
panel["year"] = panel["date"].dt.year
for v in ["t2m","rh","wind","precip"]:
    for lag in [1,3,7,14]:
        panel[f"{v}_lag{lag}"] = panel.groupby(["lat","lon"])[v].shift(lag)
for v in ["t2m","rh"]:
    panel[f"{v}_roll7"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(7,min_periods=3).mean())
    panel[f"{v}_roll14"] = panel.groupby(["lat","lon"])[v].transform(lambda s: s.rolling(14,min_periods=5).mean())
panel["precip_roll7_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(7,min_periods=3).sum())
panel["precip_roll14_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(14,min_periods=5).sum())
panel["precip_roll30_sum"] = panel.groupby(["lat","lon"])["precip"].transform(lambda s: s.rolling(30,min_periods=10).sum())
def days_since_rain(s, t=0.5):
    is_dry = (s<t).astype(int); g = (is_dry==0).cumsum()
    return g.groupby(g).cumcount()*is_dry
panel["days_since_rain"] = panel.groupby(["lat","lon"])["precip"].transform(days_since_rain)
for v in ["t2m","rh","wind"]:
    panel[f"{v}_diff3"] = panel.groupby(["lat","lon"])[v].diff(3)
fwi = (0.4*(panel["t2m"]-10) - 0.62*panel["rh"] + 0.4*panel["wind"] - 0.3*panel["precip_lag7"])
panel["fwi_compound"] = 1/(1+np.exp(-(fwi-30)/5))
panel["heat_index"] = (panel["t2m"]-20)*(50-panel["rh"])/50.0
panel["wind_chill"] = (panel["t2m"]-15)*panel["wind"]/30.0

print("[5/6] Forecast horizon labels t+1, t+3, t+7...")
for h in HORIZONS:
    panel[f"label_h{h}"] = panel.groupby(["lat","lon"])["fire"].shift(-h)

print("[6/6] Saving...")
NON_FEATURE = {"lat","lon","date","fire","year","label_h1","label_h3","label_h7"}
FEATURES = [c for c in panel.columns if c not in NON_FEATURE]
panel.to_parquet(OUT/"panel_v85.parquet", index=False)
meta = {"version":"v8.5_preprocess","panel_size":len(panel),"features":FEATURES,
        "horizons":HORIZONS,"fire_prevalence":float(panel["fire"].mean())}
with open(OUT/"panel_v85_meta.json","w") as f: json.dump(meta,f,indent=2,default=str)
print(f"  Panel saved: {OUT}/panel_v85.parquet ({(OUT/'panel_v85.parquet').stat().st_size/1e6:.1f} MB)")
print(f"  Features: {len(FEATURES)}")
print("✓ DONE")
