#!/usr/bin/env python3
"""v8.5 SPLITS — IID group-blocked + Temporal OOD + 3 Spatial OOD."""
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = Path("/home/z/my-project/download/real_data_v85")
PANEL_PATH = OUT / "panel_v85.parquet"
MASTER_SEED = 42

print("="*70); print("v8.5 SPLITS"); print("="*70)
panel = pd.read_parquet(PANEL_PATH)
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
panel["row_idx"] = np.arange(len(panel))
cells = panel[["lat","lon"]].drop_duplicates().reset_index(drop=True)
cells["cell_id"] = np.arange(len(cells))
panel = panel.merge(cells, on=["lat","lon"], how="left")
panel["year"] = panel["date"].dt.year

def stratified_cell_split(cells_df, train_frac=0.6, val_frac=0.2, seed=MASTER_SEED):
    rng = np.random.RandomState(seed)
    cells_df = cells_df.copy()
    cells_df["lat_band"] = pd.cut(cells_df["lat"], bins=8, labels=False)
    tr,va,te = [],[],[]
    for band, grp in cells_df.groupby("lat_band"):
        idx = grp["cell_id"].values; rng.shuffle(idx); n=len(idx)
        n_tr=int(n*train_frac); n_va=int(n*val_frac)
        tr.extend(idx[:n_tr]); va.extend(idx[n_tr:n_tr+n_va]); te.extend(idx[n_tr+n_va:])
    return sorted(tr), sorted(va), sorted(te)

splits = {}
tr_c,va_c,te_c = stratified_cell_split(cells)
splits["IID_groupblocked"] = {"train_idx": panel[panel["cell_id"].isin(tr_c)]["row_idx"].values.tolist(),
                                "val_idx": panel[panel["cell_id"].isin(va_c)]["row_idx"].values.tolist(),
                                "test_idx": panel[panel["cell_id"].isin(te_c)]["row_idx"].values.tolist(),
                                "design":"60/20/20 stratified by lat-band","leakage":"none (cells blocked)"}
splits["Temporal_OOD"] = {"train_idx": np.where(panel["year"].isin([2018,2019,2020,2021]).values)[0].tolist(),
                          "val_idx": np.where((panel["year"]==2022).values)[0].tolist(),
                          "test_idx": np.where((panel["year"]==2023).values)[0].tolist(),
                          "design":"train=2018-2021,val=2022,test=2023","leakage":"none"}
# Spatial OOD #1: Train North (lat>=15), Test South (lat<13)
north = cells[cells["lat"]>=15.0]["cell_id"].values
rng = np.random.RandomState(MASTER_SEED); rng.shuffle(north)
n_va = int(0.2*len(north)); va_s1=north[:n_va]; tr_s1=north[n_va:]
mask_va = panel["cell_id"].isin(va_s1)
mask_tr = (panel["lat"]>=15.0) & (~mask_va)
mask_te = panel["lat"]<13.0
splits["Spatial_OOD_NorthToSouth"] = {"train_idx": np.where(mask_tr.values)[0].tolist(),
                                      "val_idx": np.where(mask_va.values)[0].tolist(),
                                      "test_idx": np.where(mask_te.values)[0].tolist(),
                                      "design":"train=lat>=15 (excl 20% val),test=lat<13","leakage":"none"}
# Spatial OOD #2: Train South (lat<14), Test Central (14<=lat<15.5)
south = cells[cells["lat"]<14.0]["cell_id"].values
rng = np.random.RandomState(MASTER_SEED); rng.shuffle(south)
n_va = int(0.2*len(south)); va_s2=south[:n_va]; tr_s2=south[n_va:]
mask_va = panel["cell_id"].isin(va_s2)
mask_tr = (panel["lat"]<14.0) & (~mask_va)
mask_te = (panel["lat"]>=14.0) & (panel["lat"]<15.5)
splits["Spatial_OOD_SouthToCentral"] = {"train_idx": np.where(mask_tr.values)[0].tolist(),
                                        "val_idx": np.where(mask_va.values)[0].tolist(),
                                        "test_idx": np.where(mask_te.values)[0].tolist(),
                                        "design":"train=lat<14 (excl 20% val),test=14<=lat<15.5","leakage":"none"}
# Spatial OOD #3: Train C+S (lat<15), Test North (lat>=15)
cs = cells[cells["lat"]<15.0]["cell_id"].values
rng = np.random.RandomState(MASTER_SEED); rng.shuffle(cs)
n_va = int(0.2*len(cs)); va_s3=cs[:n_va]; tr_s3=cs[n_va:]
mask_va = panel["cell_id"].isin(va_s3)
mask_tr = (panel["lat"]<15.0) & (~mask_va)
mask_te = panel["lat"]>=15.0
splits["Spatial_OOD_CentralSouthToNorth"] = {"train_idx": np.where(mask_tr.values)[0].tolist(),
                                              "val_idx": np.where(mask_va.values)[0].tolist(),
                                              "test_idx": np.where(mask_te.values)[0].tolist(),
                                              "design":"train=lat<15 (excl 20% val),test=lat>=15","leakage":"none"}

# Save as npz
arrays = {}
for k,v in splits.items():
    for k2 in ["train_idx","val_idx","test_idx"]:
        arrays[f"{k}_{k2.replace('_idx','')}"] = np.array(v[k2], dtype=np.int64)
np.savez(OUT/"splits_v85.npz", **arrays)
# Parquet for portability
sdf_rows = []
for k,v in splits.items():
    for kind in ["train","val","test"]:
        for idx in v[f"{kind}_idx"]:
            sdf_rows.append({"split":k, "kind":kind, "idx":idx})
pd.DataFrame(sdf_rows).to_parquet(OUT/"splits_v85.parquet", index=False)
# Meta
meta = {"version":"v8.5_splits","n_splits":len(splits),
        "splits_summary":{k:{"train_n":len(v["train_idx"]),"val_n":len(v["val_idx"]),"test_n":len(v["test_idx"]),
                             "design":v["design"],"leakage":v["leakage"]} for k,v in splits.items()},
        "master_seed":MASTER_SEED}
with open(OUT/"splits_v85_meta.json","w") as f: json.dump(meta,f,indent=2,default=str)

print("  Splits:")
for k,v in splits.items():
    print(f"    {k}: train={len(v['train_idx']):,} val={len(v['val_idx']):,} test={len(v['test_idx']):,}")
print("✓ DONE")
