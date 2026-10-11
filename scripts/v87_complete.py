#!/usr/bin/env python3
"""v8.7 complete rebuild: panel → splits → baselines → export → ZIP."""
import os, json, math, pickle, warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/v87")
MODEL_DIR = Path("/home/z/my-project/download/model_v87")
ZIP_PATH = Path("/home/z/my-project/download/wildfire_v87_complete.zip")
OUT.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)

GRID=0.5; LAT_MIN,LAT_MAX=11.0,21.0; LON_MIN,LON_MAX=104.0,109.5
YEARS=list(range(2018,2024))

print("="*60); print("v8.7 COMPLETE REBUILD"); print("="*60)

# 1. PANEL
print("\n[1/6] Panel (leakage-free)...")
wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv"); wx["date"]=pd.to_datetime(wx["date"])
fires = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in YEARS], ignore_index=True)
fires["lat"]=np.round((fires["latitude"]-LAT_MIN)/GRID)*GRID+LAT_MIN
fires["lon"]=np.round((fires["longitude"]-LON_MIN)/GRID)*GRID+LON_MIN
fires["date"]=pd.to_datetime(fires["acq_date"])
fires=fires[(fires["lat"]>=LAT_MIN)&(fires["lat"]<=LAT_MAX)&(fires["lon"]>=LON_MIN)&(fires["lon"]<=LON_MAX)]
fcd=fires[["lat","lon","date"]].drop_duplicates(); fcd["fire"]=1
cells=wx[["lat","lon"]].drop_duplicates().sort_values(["lat","lon"]).reset_index(drop=True)
ad=pd.DataFrame({"date":pd.date_range(wx["date"].min(),wx["date"].max(),freq="D")})
cells["_k"]=1;ad["_k"]=1
panel=cells.merge(ad,on="_k").drop("_k",axis=1)
panel=panel.merge(wx,on=["lat","lon","date"],how="left")
panel=panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
for c in ["t2m","rh","wind","precip"]:
    panel[c]=panel.groupby(["lat","lon"])[c].transform(lambda s:s.ffill())
    gm=panel.loc[panel["date"].dt.year.isin([2018,2019,2020,2021]),c].mean()
    panel[c]=panel[c].fillna(gm)
panel=panel.merge(fcd,on=["lat","lon","date"],how="left")
panel["fire"]=panel["fire"].fillna(0).astype(int)
panel["doy"]=panel["date"].dt.dayofyear; panel["year"]=panel["date"].dt.year
# Features
for v in ["t2m","rh","wind","precip"]:
    for lag in [1,3,7,14]: panel[f"{v}_lag{lag}"]=panel.groupby(["lat","lon"])[v].shift(lag)
for v in ["t2m","rh"]:
    panel[f"{v}_roll7"]=panel.groupby(["lat","lon"])[v].transform(lambda s:s.rolling(7,min_periods=3).mean())
    panel[f"{v}_roll14"]=panel.groupby(["lat","lon"])[v].transform(lambda s:s.rolling(14,min_periods=5).mean())
panel["precip_roll7_sum"]=panel.groupby(["lat","lon"])["precip"].transform(lambda s:s.rolling(7,min_periods=3).sum())
panel["precip_roll14_sum"]=panel.groupby(["lat","lon"])["precip"].transform(lambda s:s.rolling(14,min_periods=5).sum())
panel["precip_roll30_sum"]=panel.groupby(["lat","lon"])["precip"].transform(lambda s:s.rolling(30,min_periods=10).sum())
def dsr(s,t=0.5):
    is_dry=(s<t).astype(int);g=(is_dry==0).cumsum();return g.groupby(g).cumcount()*is_dry
panel["days_since_rain"]=panel.groupby(["lat","lon"])["precip"].transform(dsr)
for v in ["t2m","rh","wind"]: panel[f"{v}_diff3"]=panel.groupby(["lat","lon"])[v].diff(3)
fwi=0.4*(panel["t2m"]-10)-0.62*panel["rh"]+0.4*panel["wind"]-0.3*panel["precip_lag7"]
panel["fwi_compound"]=1/(1+np.exp(-(fwi-30)/5))
panel["heat_index"]=(panel["t2m"]-20)*(50-panel["rh"])/50.0
panel["wind_chill"]=(panel["t2m"]-15)*panel["wind"]/30.0
panel["label_h1"]=panel.groupby(["lat","lon"])["fire"].shift(-1)
# Fire history (expanding window — no future leakage)
panel["fire_prev"]=panel.groupby(["lat","lon"])["fire"].shift(1)
panel["cell_fire_rate_exp"]=panel.groupby(["lat","lon"])["fire_prev"].transform(lambda s:s.expanding(min_periods=1).mean()).fillna(panel["fire"].mean())
panel["cell_fire_count_30d"]=panel.groupby(["lat","lon"])["fire_prev"].transform(lambda s:s.rolling(30,min_periods=1).sum())
panel["is_dry"]=panel["date"].dt.month.apply(lambda m:1 if m in[1,2,3,4,11,12]else 0)
panel["is_dry_prev"]=panel.groupby(["lat","lon"])["is_dry"].shift(1)
panel["cell_fire_rate_dry_exp"]=panel.groupby(["lat","lon"]).apply(lambda g:(g["fire_prev"]*g["is_dry_prev"]).expanding(min_periods=1).sum()/(g["is_dry_prev"].expanding(min_periods=1).sum()+1e-6)).reset_index(level=[0,1],drop=True).fillna(panel["fire"].mean())
panel=panel.drop(columns=["fire_prev","is_dry_prev"])
WEATHER=[c for c in ["t2m","rh","wind","precip","doy","t2m_lag1","t2m_lag3","t2m_lag7","t2m_lag14","rh_lag1","rh_lag3","rh_lag7","rh_lag14","wind_lag1","wind_lag3","wind_lag7","wind_lag14","precip_lag1","precip_lag3","precip_lag7","precip_lag14","t2m_roll7","t2m_roll14","rh_roll7","rh_roll14","precip_roll7_sum","precip_roll14_sum","precip_roll30_sum","days_since_rain","t2m_diff3","rh_diff3","wind_diff3","fwi_compound","heat_index","wind_chill"]]
HISTORY=["cell_fire_rate_exp","cell_fire_rate_dry_exp","cell_fire_count_30d"]
COMBINED=WEATHER+HISTORY
panel.to_parquet(OUT/"panel_v87.parquet",index=False)
print(f"  Panel: {len(panel):,} rows, {len(COMBINED)} features")

# 2. SPLITS
print("\n[2/6] Splits...")
panel["row_idx"]=np.arange(len(panel))
cells_df=panel[["lat","lon"]].drop_duplicates().reset_index(drop=True)
cells_df["cell_id"]=np.arange(len(cells_df))
panel=panel.merge(cells_df,on=["lat","lon"],how="left")
# Temporal
tr_A=panel[panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_h1"])
va_A=panel[panel["year"]==2022].dropna(subset=["label_h1"])
te_A=panel[panel["year"]==2023].dropna(subset=["label_h1"])
# Cold-start
tr_B=panel[(panel["lat"]>=15.0)&panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_h1"])
va_B=panel[(panel["lat"]>=15.0)&(panel["year"]==2022)].dropna(subset=["label_h1"])
te_B=panel[(panel["lat"]<13.0)].dropna(subset=["label_h1"])
te_B_cold=te_B.copy()
te_B_cold["cell_fire_rate_exp"]=0.0;te_B_cold["cell_fire_rate_dry_exp"]=0.0;te_B_cold["cell_fire_count_30d"]=0.0
te_B_cold.to_parquet(OUT/"test_coldstart.parquet",index=False)
# Warm-start = same as cold but keep history
tr_C=tr_B.copy();va_C=va_B.copy();te_C=te_B.copy()
np.savez(OUT/"splits_v87.npz",
    temporal_train=tr_A["row_idx"].values,temporal_val=va_A["row_idx"].values,temporal_test=te_A["row_idx"].values,
    coldstart_train=tr_B["row_idx"].values,coldstart_val=va_B["row_idx"].values,coldstart_test=te_B["row_idx"].values,
    warmstart_train=tr_C["row_idx"].values,warmstart_val=va_C["row_idx"].values,warmstart_test=te_C["row_idx"].values)
print(f"  Temporal: tr={len(tr_A):,} va={len(va_A):,} te={len(te_A):,}")
print(f"  Cold: tr={len(tr_B):,} va={len(va_B):,} te={len(te_B):,}")
print(f"  Warm: tr={len(tr_C):,} va={len(va_C):,} te={len(te_C):,}")

# 3. BASELINES (B1/B2/B3 × 3 modes, seed 42)
print("\n[3/6] Baselines (9 configs)...")
di_idx=COMBINED.index("precip_roll30_sum")
def train_xgb(tr,va,te,feats,seed=42):
    dm=np.nanmean(tr[feats[di_idx]].values) if "precip_roll30_sum" in feats else 0
    ds=(np.nanstd(tr[feats[di_idx]].values)+1e-6) if "precip_roll30_sum" in feats else 1
    def prep(d):
        x=d[feats].values.astype(np.float32).copy()
        if "precip_roll30_sum" in feats: x[:,di_idx]=np.nan_to_num((x[:,di_idx]-dm)/ds,nan=0.0)
        return x
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(prep(tr));Xva=imp.transform(prep(va));Xte=imp.transform(prep(te))
    ytr=tr["label_h1"].values.astype(np.float32);yva=va["label_h1"].values.astype(np.float32);yte=te["label_h1"].values.astype(np.float32)
    pos=max(int(ytr.sum()),1);neg=max(int((1-ytr).sum()),1);spw=max(math.sqrt(neg/pos),1.0)
    m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=seed,n_jobs=4)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    pva=m.predict_proba(Xva)[:,1];pte=m.predict_proba(Xte)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1);iso.fit(pva,yva)
    pi=iso.predict(pte)
    return {"AUROC_iso":float(roc_auc_score(yte,pi)),"AUPRC":float(average_precision_score(yte,pi)),"Brier":float(brier_score_loss(yte,pi))},m,iso,imp

baselines={}
modes=[("temporal",tr_A,va_A,te_A),("coldstart",tr_B,va_B,te_B_cold),("warmstart",tr_C,va_C,te_C)]
for cname,feats in [("B1_weather",WEATHER),("B2_history",HISTORY),("B3_combined",COMBINED)]:
    for mname,tr,va,te in modes:
        r,_,_,_=train_xgb(tr,va,te,feats)
        baselines[f"{cname}_{mname}"]=r
        print(f"  {cname}_{mname}: AUROC={r['AUROC_iso']:.4f}")

# 4. CONCEPT SHIFT
print("\n[4/6] Concept shift (standardized)...")
r_cold,m_cold,iso_cold,imp_cold=train_xgb(tr_B,va_B,te_B_cold,COMBINED)
imp_w=SimpleImputer(strategy="median");scaler=StandardScaler()
X_nw=scaler.fit_transform(imp_w.fit_transform(tr_B[WEATHER].values.astype(np.float32)))
X_sw=scaler.transform(imp_w.transform(te_B[WEATHER].values.astype(np.float32)))
nn=NearestNeighbors(n_neighbors=1,metric="euclidean");nn.fit(X_nw)
distances,indices=nn.kneighbors(X_sw)
dm=np.nanmean(tr_B[COMBINED[di_idx]].values);ds=np.nanstd(tr_B[COMBINED[di_idx]].values)+1e-6
X_north=imp_cold.transform(tr_B[COMBINED].values.astype(np.float32).copy())
X_north[:,di_idx]=np.nan_to_num((X_north[:,di_idx]-dm)/ds,nan=0.0)
p_north=iso_cold.predict(m_cold.predict_proba(X_north)[:,1])
y_north=tr_B["label_h1"].values.astype(np.float32)
p_nm=p_north[indices.flatten()];y_nm=y_north[indices.flatten()]
X_south=imp_cold.transform(np.nan_to_num(te_B_cold[COMBINED].values.astype(np.float32).copy(),nan=0.0))
X_south[:,di_idx]=np.nan_to_num((X_south[:,di_idx]-dm)/ds,nan=0.0)
p_south=iso_cold.predict(m_cold.predict_proba(X_south)[:,1])
y_south=te_B_cold["label_h1"].values.astype(np.float32)
nc=((p_nm>=0.5).astype(int)==y_nm.astype(int))
sc=((p_south>=0.5).astype(int)==y_south.astype(int))
cs=nc&~sc;me=~nc&~sc
n=len(y_south)
print(f"  CS={cs.sum()} ({100*cs.sum()/n:.1f}%), ME={me.sum()} ({100*me.sum()/n:.1f}%), ratio={cs.sum()/max(me.sum(),1):.1f}:1")

# 5. EXPORT MODEL
print("\n[5/6] Export model...")
r_temp,m_temp,iso_temp,imp_temp=train_xgb(tr_A,va_A,te_A,COMBINED)
m_temp.save_model(str(MODEL_DIR/"xgb_v87_temporal.json"))
with open(MODEL_DIR/"iso_v87.pkl","wb")as f:pickle.dump(iso_temp,f)
with open(MODEL_DIR/"imp_v87.pkl","wb")as f:pickle.dump(imp_temp,f)
dm_t=np.nanmean(tr_A[COMBINED[di_idx]].values);ds_t=np.nanstd(tr_A[COMBINED[di_idx]].values)+1e-6
meta_v87={"version":"v8.7_leakage_free","features":COMBINED,"n_features":len(COMBINED),
    "trained_on":"2018-2021 temporal (expanding window)","val":"2022","test":"2023",
    "drought_idx":di_idx,"drought_mean":float(dm_t),"drought_std":float(ds_t),
    "test_AUROC_iso":float(r_temp["AUROC_iso"]),"product_status":"demo",
    "leakage_fixes":["expanding window fire history","ffill only weather","cold-start prior=0","Iso on val"],
    "caveats":["NASA POWER vs Open-Meteo bias chưa kiểm tra","Demo only, không phải operational"]}
with open(MODEL_DIR/"meta_v87.json","w")as f:json.dump(meta_v87,f,indent=2,default=str)
print(f"  AUROC={r_temp['AUROC_iso']:.4f}")

# Save results
results={"baselines":baselines,"concept_shift":{"both_correct":int((nc&sc).sum()),"concept_shift":int(cs.sum()),"model_error":int(me.sum()),"ratio":float(cs.sum()/max(me.sum(),1)),"weather_dist":float(distances.mean())},"model_export":{"AUROC_iso":r_temp["AUROC_iso"],"AUPRC":r_temp["AUPRC"],"Brier":r_temp["Brier"]}}
with open(OUT/"results_v87.json","w")as f:json.dump(results,f,indent=2,default=str)

# 6. ZIP
print(f"\n[6/6] ZIP...")
with zipfile.ZipFile(ZIP_PATH,"w",zipfile.ZIP_DEFLATED)as zf:
    for d in [OUT,MODEL_DIR]:
        for f in sorted(d.iterdir()):
            if f.is_file():
                zf.write(f,f"{d.name}/{f.name}")
    zf.write(__file__,"scripts/v87_complete.py")
    zf.writestr("README.md",f"""# Wildfire v8.7 — Leakage-Free Pipeline

## Results (seed 42)

| Mode | B1 Weather | B2 History | B3 Combined |
|------|-----------|------------|-------------|
| Temporal | {baselines['B1_weather_temporal']['AUROC_iso']:.4f} | {baselines['B2_history_temporal']['AUROC_iso']:.4f} | {baselines['B3_combined_temporal']['AUROC_iso']:.4f} |
| Cold-start | {baselines['B1_weather_coldstart']['AUROC_iso']:.4f} | {baselines['B2_history_coldstart']['AUROC_iso']:.4f} | {baselines['B3_combined_coldstart']['AUROC_iso']:.4f} |
| Warm-start | {baselines['B1_weather_warmstart']['AUROC_iso']:.4f} | {baselines['B2_history_warmstart']['AUROC_iso']:.4f} | {baselines['B3_combined_warmstart']['AUROC_iso']:.4f} |

Concept shift: CS={cs.sum()} ({100*cs.sum()/n:.1f}%), ME={me.sum()} ({100*me.sum()/n:.1f}%), ratio={cs.sum()/max(me.sum(),1):.1f}:1

Model: AUROC={r_temp['AUROC_iso']:.4f} (temporal test 2023)
Status: demo (chưa validate Open-Meteo)

## Leakage fixes
1. Fire history: expanding window (no future)
2. Weather: ffill only (no bfill)
3. Cold-start: test fire history = 0
4. Calibration: Iso fit on val

## Files
- v87/panel_v87.parquet — leakage-free panel (598K rows, 38 features)
- v87/test_coldstart.parquet — cold-start test (history=0)
- v87/splits_v87.npz — 3 modes (temporal, cold, warm)
- v87/results_v87.json — all results
- model_v87/xgb_v87_temporal.json — final model
- model_v87/iso_v87.pkl — calibrator
- model_v87/imp_v87.pkl — imputer
- model_v87/meta_v87.json — metadata
""")

print(f"\n{'='*60}")
print(f"✓ v8.7 ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size//1024//1024} MB)")
print(f"{'='*60}")
print(f"\nResults:")
print(f"  B3 Temporal:   {baselines['B3_combined_temporal']['AUROC_iso']:.4f}")
print(f"  B3 Cold-start: {baselines['B3_combined_coldstart']['AUROC_iso']:.4f}")
print(f"  B3 Warm-start: {baselines['B3_combined_warmstart']['AUROC_iso']:.4f}")
print(f"  Concept shift: {cs.sum()/max(me.sum(),1):.1f}:1")
print(f"  Model AUROC:   {r_temp['AUROC_iso']:.4f}")

# RESUMABILITY: if results exist, just re-ZIP
if (OUT/"results_v87.json").exists() and not ZIP_PATH.exists():
    print("\nResults exist — building ZIP only...")
    results = json.loads(open(OUT/"results_v87.json").read())
    baselines = results["baselines"]
    cs = results["concept_shift"]
    r_temp = results["model_export"]
    # Re-ZIP
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in [OUT, MODEL_DIR]:
            for f in sorted(d.iterdir()):
                if f.is_file():
                    zf.write(f, f"{d.name}/{f.name}")
    print(f"✓ ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size//1024//1024} MB)")
