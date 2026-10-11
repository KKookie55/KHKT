#!/usr/bin/env python3
"""Build VIIRS panel — same features as MODIS panel, but VIIRS fire labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
SRC = Path("/home/z/my-project/download/real_data_v85/panel_v85.parquet")
DST = Path("/home/z/my-project/download/real_data_v85/panel_v85_viirs.parquet")
GRID_RES=0.5; LAT_MIN,LON_MIN=11.0,104.0; HORIZONS=[1,3,7]; YEARS=list(range(2018,2024))

print("="*70); print("BUILD VIIRS-LABELED PANEL"); print("="*70)
panel = pd.read_parquet(SRC)
for c in ["fire","label_h1","label_h3","label_h7"]:
    if c in panel.columns: panel = panel.drop(columns=[c])
print(f"  Loaded MODIS panel: {len(panel):,} rows")

viirs = pd.concat([pd.read_csv(DATA/"viirs"/f"viirs_fires_{y}.csv") for y in YEARS], ignore_index=True)
viirs["lat"] = np.round((viirs["latitude"]-LAT_MIN)/GRID_RES)*GRID_RES+LAT_MIN
viirs["lon"] = np.round((viirs["longitude"]-LON_MIN)/GRID_RES)*GRID_RES+LON_MIN
viirs["date"] = pd.to_datetime(viirs["acq_date"])
vcd = viirs[["lat","lon","date"]].drop_duplicates()
vcd["fire"] = 1
print(f"  VIIRS unique cell-days: {len(vcd):,}")

panel["lat"]=panel["lat"].round(4); panel["lon"]=panel["lon"].round(4)
vcd["lat"]=vcd["lat"].round(4); vcd["lon"]=vcd["lon"].round(4)
panel = panel.merge(vcd, on=["lat","lon","date"], how="left")
panel["fire"] = panel["fire"].fillna(0).astype(int)
print(f"  VIIRS fire prevalence: {100*panel['fire'].mean():.2f}%")

for h in HORIZONS:
    panel[f"label_h{h}"] = panel.groupby(["lat","lon"])["fire"].shift(-h)
panel.to_parquet(DST, index=False)
print(f"  ✓ Saved: {DST} ({DST.stat().st_size/1e6:.1f} MB)")
print("DONE")
