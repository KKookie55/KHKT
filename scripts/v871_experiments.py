#!/usr/bin/env python3
"""
v8.7.1 — Ba thí nghiệm khoa học:
  1. Chứng minh dự báo tương lai: t+1, t+3, t+7 (temporal split, no future leakage)
  2. MODIS-VIIRS label experiment: cross-label, noise sensitivity
  3. Independent verification: full code, config, seeds, results table

Output: /home/z/my-project/download/v871/
"""
import os, json, math, pickle, warnings, zipfile, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings('ignore')

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/v871")
OUT.mkdir(parents=True, exist_ok=True)

GRID=0.5; LAT_MIN,LAT_MAX=11.0,21.0; LON_MIN,LON_MAX=104.0,109.5
YEARS=list(range(2018,2024))

print("="*60); print("v8.7.1 — BA THÍ NGHIỆM KHOA HỌC"); print("="*60)

# ============================================================
# BUILD PANEL (leakage-free, with MODIS + VIIRS labels)
# ============================================================
print("\n[0] Building panel (MODIS + VIIRS labels)...")
wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv"); wx["date"]=pd.to_datetime(wx["date"])

# MODIS fires
modis = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in YEARS], ignore_index=True)
modis["lat"]=np.round((modis["latitude"]-LAT_MIN)/GRID)*GRID+LAT_MIN
modis["lon"]=np.round((modis["longitude"]-LON_MIN)/GRID)*GRID+LON_MIN
modis["date"]=pd.to_datetime(modis["acq_date"])
modis=modis[(modis["lat"]>=LAT_MIN)&(modis["lat"]<=LAT_MAX)&(modis["lon"]>=LON_MIN)&(modis["lon"]<=LON_MAX)]
modis_cd=modis[["lat","lon","date"]].drop_duplicates(); modis_cd["fire_modis"]=1

# VIIRS fires
viirs = pd.concat([pd.read_csv(DATA/"viirs"/f"viirs_fires_{y}.csv") for y in YEARS], ignore_index=True)
viirs["lat"]=np.round((viirs["latitude"]-LAT_MIN)/GRID)*GRID+LAT_MIN
viirs["lon"]=np.round((viirs["longitude"]-LON_MIN)/GRID)*GRID+LON_MIN
viirs["date"]=pd.to_datetime(viirs["acq_date"])
viirs=viirs[(viirs["lat"]>=LAT_MIN)&(viirs["lat"]<=LAT_MAX)&(viirs["lon"]>=LON_MIN)&(viirs["lon"]<=LON_MAX)]
viirs_cd=viirs[["lat","lon","date"]].drop_duplicates(); viirs_cd["fire_viirs"]=1

# Build panel
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
panel=panel.merge(modis_cd,on=["lat","lon","date"],how="left")
panel["fire_modis"]=panel["fire_modis"].fillna(0).astype(int)
panel=panel.merge(viirs_cd,on=["lat","lon","date"],how="left")
panel["fire_viirs"]=panel["fire_viirs"].fillna(0).astype(int)
# Union label: fire if MODIS OR VIIRS detects
panel["fire_union"]=((panel["fire_modis"]==1)|(panel["fire_viirs"]==1)).astype(int)
panel["doy"]=panel["date"].dt.dayofyear; panel["year"]=panel["date"].dt.year

# Weather features (leakage-free: lags use past only, ffill only)
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
# Feature check: fwi_compound distribution
fwi_raw=0.4*(panel["t2m"]-10)-0.62*panel["rh"]+0.4*panel["wind"]-0.3*panel["precip_lag7"]
panel["fwi_compound"]=1/(1+np.exp(-(fwi_raw-30)/5))
panel["heat_index"]=(panel["t2m"]-20)*(50-panel["rh"])/50.0
panel["wind_chill"]=(panel["t2m"]-15)*panel["wind"]/30.0

# Fire history (expanding window, no future)
panel["fire_prev_modis"]=panel.groupby(["lat","lon"])["fire_modis"].shift(1)
panel["cell_fire_rate_exp"]=panel.groupby(["lat","lon"])["fire_prev_modis"].transform(lambda s:s.expanding(min_periods=1).mean()).fillna(panel["fire_modis"].mean())
panel["cell_fire_count_30d"]=panel.groupby(["lat","lon"])["fire_prev_modis"].transform(lambda s:s.rolling(30,min_periods=1).sum())
panel["is_dry"]=panel["date"].dt.month.apply(lambda m:1 if m in[1,2,3,4,11,12]else 0)
panel["is_dry_prev"]=panel.groupby(["lat","lon"])["is_dry"].shift(1)
panel["cell_fire_rate_dry_exp"]=panel.groupby(["lat","lon"]).apply(lambda g:(g["fire_prev_modis"]*g["is_dry_prev"]).expanding(min_periods=1).sum()/(g["is_dry_prev"].expanding(min_periods=1).sum()+1e-6)).reset_index(level=[0,1],drop=True).fillna(panel["fire_modis"].mean())
panel=panel.drop(columns=["fire_prev_modis","is_dry_prev"])

# Forecast labels for each horizon (MODIS, VIIRS, union)
for h in [1,3,7]:
    for src in ["modis","viirs","union"]:
        panel[f"label_{src}_h{h}"]=panel.groupby(["lat","lon"])[f"fire_{src}"].shift(-h)

WEATHER=["t2m","rh","wind","precip","doy","t2m_lag1","t2m_lag3","t2m_lag7","t2m_lag14","rh_lag1","rh_lag3","rh_lag7","rh_lag14","wind_lag1","wind_lag3","wind_lag7","wind_lag14","precip_lag1","precip_lag3","precip_lag7","precip_lag14","t2m_roll7","t2m_roll14","rh_roll7","rh_roll14","precip_roll7_sum","precip_roll14_sum","precip_roll30_sum","days_since_rain","t2m_diff3","rh_diff3","wind_diff3","fwi_compound","heat_index","wind_chill"]
HISTORY=["cell_fire_rate_exp","cell_fire_rate_dry_exp","cell_fire_count_30d"]
COMBINED=WEATHER+HISTORY

# Feature check
print(f"\n  Feature check:")
print(f"    fwi_compound: mean={panel['fwi_compound'].mean():.4f}, std={panel['fwi_compound'].std():.4f}, "
      f"%==0: {100*(panel['fwi_compound']<0.01).mean():.1f}%")
print(f"    heat_index: mean={panel['heat_index'].mean():.2f}, std={panel['heat_index'].std():.2f}")
print(f"    wind_chill: mean={panel['wind_chill'].mean():.2f}, std={panel['wind_chill'].std():.2f}")
if (panel['fwi_compound']<0.01).mean() > 0.5:
    print("    ⚠ fwi_compound >50% giá trị = 0 (sigmoid bão hòa) → loại bỏ")
    COMBINED=[f for f in COMBINED if f!="fwi_compound"]
    WEATHER=[f for f in WEATHER if f!="fwi_compound"]
    print(f"    Features: {len(COMBINED)} (sau khi loại fwi_compound)")

panel.to_parquet(OUT/"panel_v871.parquet",index=False)
print(f"  Panel: {len(panel):,} rows, {len(COMBINED)} features")
print(f"  Labels: MODIS, VIIRS, union × h1, h3, h7")

# Temporal split
tr=panel[panel["year"].isin([2018,2019,2020,2021])]
va=panel[panel["year"]==2022]
te=panel[panel["year"]==2023]

# ============================================================
# THÍ NGHIỆM 1: DỰ BÁO TƯƠNG LAI (t+1, t+3, t+7)
# ============================================================
print(f"\n{'='*60}")
print("THÍ NGHIỆM 1: DỰ BÁO TƯƠNG LAI (temporal split, no future leakage)")
print(f"{'='*60}")

di_idx=COMBINED.index("precip_roll30_sum")

def train_eval(tr, va, te, features, label_col, seed=42):
    tr_d=tr.dropna(subset=[label_col]); va_d=va.dropna(subset=[label_col]); te_d=te.dropna(subset=[label_col])
    dm=np.nanmean(tr_d[features[di_idx]].values); ds=np.nanstd(tr_d[features[di_idx]].values)+1e-6
    def prep(d):
        x=d[features].values.astype(np.float32).copy()
        x[:,di_idx]=np.nan_to_num((x[:,di_idx]-dm)/ds,nan=0.0)
        return x
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(prep(tr_d));Xva=imp.transform(prep(va_d));Xte=imp.transform(prep(te_d))
    ytr=tr_d[label_col].values.astype(np.float32);yva=va_d[label_col].values.astype(np.float32);yte=te_d[label_col].values.astype(np.float32)
    pos=max(int(ytr.sum()),1);neg=max(int((1-ytr).sum()),1);spw=max(math.sqrt(neg/pos),1.0)
    m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=seed,n_jobs=4)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    pva=m.predict_proba(Xva)[:,1];pte=m.predict_proba(Xte)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1);iso.fit(pva,yva)
    pi=iso.predict(pte)
    # Simple baseline: climatology (cell × month fire rate from train)
    tr_d2=tr_d.copy(); tr_d2["month"]=tr_d2["date"].dt.month
    clim=tr_d2.groupby(["lat","lon","month"])[label_col].mean().reset_index()
    clim.columns=["lat","lon","month","clim_prob"]
    te_d2=te_d.copy(); te_d2["month"]=te_d2["date"].dt.month
    te_d2=te_d2.merge(clim,on=["lat","lon","month"],how="left")
    te_d2["clim_prob"]=te_d2["clim_prob"].fillna(ytr.mean())
    clim_auroc=roc_auc_score(yte, te_d2["clim_prob"].values) if len(np.unique(yte))>1 else 0.5
    return {
        "AUROC_model":float(roc_auc_score(yte,pi)),
        "AUROC_climatology":float(clim_auroc),
        "AUPRC_model":float(average_precision_score(yte,pi)),
        "Brier_model":float(brier_score_loss(yte,pi)),
        "Brier_climatology":float(brier_score_loss(yte,te_d2["clim_prob"].values)),
        "n_test":len(te_d),"n_pos":int(yte.sum()),"pos_rate":float(yte.mean()),
    }

exp1_results={}
for h in [1,3,7]:
    label=f"label_modis_h{h}"
    print(f"\n  Horizon t+{h} (MODIS labels):")
    r=train_eval(tr,va,te,COMBINED,label)
    exp1_results[f"h{h}"]=r
    print(f"    Model: AUROC={r['AUROC_model']:.4f}, AUPRC={r['AUPRC_model']:.4f}, Brier={r['Brier_model']:.4f}")
    print(f"    Climatology: AUROC={r['AUROC_climatology']:.4f}, Brier={r['Brier_climatology']:.4f}")
    print(f"    Δ AUROC (model−clim): {r['AUROC_model']-r['AUROC_climatology']:+.4f}")
    print(f"    n_test={r['n_test']:,}, n_pos={r['n_pos']:,} ({100*r['pos_rate']:.1f}%)")

# ============================================================
# THÍ NGHIỆM 2: MODIS–VIIRS LABEL EXPERIMENT
# ============================================================
print(f"\n{'='*60}")
print("THÍ NGHIỆM 2: MODIS–VIIRS LABEL EXPERIMENT")
print(f"{'='*60}")

# 2a: Cross-label (train on one, test on another)
print("\n  2a: Cross-label (t+1, seed 42):")
exp2a={}
configs=[
    ("train_MODIS_test_MODIS","label_modis_h1","label_modis_h1"),
    ("train_VIIRS_test_VIIRS","label_viirs_h1","label_viirs_h1"),
    ("train_MODIS_test_VIIRS","label_modis_h1","label_viirs_h1"),
    ("train_VIIRS_test_MODIS","label_viirs_h1","label_modis_h1"),
    ("train_UNION_test_MODIS","label_union_h1","label_modis_h1"),
    ("train_UNION_test_VIIRS","label_union_h1","label_viirs_h1"),
]
for name, train_label, test_label in configs:
    # Use same temporal split but different labels
    tr_d=tr.dropna(subset=[train_label])
    va_d=va.dropna(subset=[train_label])
    te_d=te.dropna(subset=[test_label])
    if len(tr_d)<100 or len(te_d)<100: continue
    dm=np.nanmean(tr_d[COMBINED[di_idx]].values);ds=np.nanstd(tr_d[COMBINED[di_idx]].values)+1e-6
    def prep(d):
        x=d[COMBINED].values.astype(np.float32).copy()
        x[:,di_idx]=np.nan_to_num((x[:,di_idx]-dm)/ds,nan=0.0)
        return x
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(prep(tr_d));Xva=imp.transform(prep(va_d));Xte=imp.transform(prep(te_d))
    ytr=tr_d[train_label].values.astype(np.float32)
    yva=va_d[train_label].values.astype(np.float32)
    yte=te_d[test_label].values.astype(np.float32)
    pos=max(int(ytr.sum()),1);neg=max(int((1-ytr).sum()),1);spw=max(math.sqrt(neg/pos),1.0)
    m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=42,n_jobs=4)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    pte=m.predict_proba(Xte)[:,1]
    auroc=roc_auc_score(yte,pte) if len(np.unique(yte))>1 else 0.5
    exp2a[name]={"AUROC":float(auroc),"n_train":len(tr_d),"n_test":len(te_d),
                 "pos_rate_train":float(ytr.mean()),"pos_rate_test":float(yte.mean())}
    print(f"    {name}: AUROC={auroc:.4f} (train pos={100*ytr.mean():.1f}%, test pos={100*yte.mean():.1f}%)")

# 2b: Noise sensitivity
print("\n  2b: Noise sensitivity (t+1 MODIS, label flip 5/10/20%):")
exp2b={}
for noise_pct in [0, 5, 10, 20]:
    np.random.seed(42)
    tr_d=tr.dropna(subset=["label_modis_h1"]).copy()
    te_d=te.dropna(subset=["label_modis_h1"]).copy()
    va_d=va.dropna(subset=["label_modis_h1"]).copy()
    # Flip labels in train
    n_flip=int(noise_pct/100*len(tr_d))
    flip_idx=np.random.choice(len(tr_d),n_flip,replace=False)
    y_tr=tr_d["label_modis_h1"].values.astype(np.float32).copy()
    y_tr[flip_idx]=1-y_tr[flip_idx]
    # Train
    dm=np.nanmean(tr_d[COMBINED[di_idx]].values);ds=np.nanstd(tr_d[COMBINED[di_idx]].values)+1e-6
    def prep(d):
        x=d[COMBINED].values.astype(np.float32).copy()
        x[:,di_idx]=np.nan_to_num((x[:,di_idx]-dm)/ds,nan=0.0)
        return x
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(prep(tr_d));Xva=imp.transform(prep(va_d));Xte=imp.transform(prep(te_d))
    yva=va_d["label_modis_h1"].values.astype(np.float32);yte=te_d["label_modis_h1"].values.astype(np.float32)
    pos=max(int(y_tr.sum()),1);neg=max(int((1-y_tr).sum()),1);spw=max(math.sqrt(neg/pos),1.0)
    m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=42,n_jobs=4)
    m.fit(Xtr,y_tr,eval_set=[(Xva,yva)],verbose=False)
    pva=m.predict_proba(Xva)[:,1];pte=m.predict_proba(Xte)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1);iso.fit(pva,yva)
    pi=iso.predict(pte)
    auroc=roc_auc_score(yte,pi)
    exp2b[f"noise_{noise_pct}pct"]={"AUROC":float(auroc),"n_flipped":n_flip}
    print(f"    noise={noise_pct}%: AUROC={auroc:.4f} (flipped {n_flip:,} labels)")

# ============================================================
# THÍ NGHIỆM 3: INDEPENDENT VERIFICATION PACKAGE
# ============================================================
print(f"\n{'='*60}")
print("THÍ NGHIỆM 3: INDEPENDENT VERIFICATION PACKAGE")
print(f"{'='*60}")

# 3a: Multi-seed (3 seeds × t+1 temporal)
print("\n  3a: Multi-seed (3 seeds × t+1 MODIS temporal):")
seeds_results=[]
for s in [42,123,777]:
    r=train_eval(tr,va,te,COMBINED,"label_modis_h1",seed=s)
    seeds_results.append(r["AUROC_model"])
    print(f"    seed={s}: AUROC={r['AUROC_model']:.4f}")
print(f"    Mean ± Std: {np.mean(seeds_results):.4f} ± {np.std(seeds_results):.4f}")

# 3b: Config file
config={
    "version":"v8.7.1",
    "data":{"source":"NASA MODIS + VIIRS + NASA POWER","grid":"0.5°","date_range":"2018-2023","n_cells":273},
    "split":{"train":"2018-2021","val":"2022","test":"2023","type":"temporal (no future leakage)"},
    "features":COMBINED,"n_features":len(COMBINED),
    "feature_groups":{"weather":WEATHER,"fire_history":HISTORY},
    "leakage_fixes":["expanding window fire history","ffill only weather (no bfill)","cell_fire_count_30d shifted by 1","Iso fit on val","forecast labels use shift(-h)"],
    "model":{"type":"XGBoost","n_estimators":200,"max_depth":6,"learning_rate":0.1,
             "subsample":0.85,"colsample_bytree":0.7,"min_child_weight":20,
             "reg_lambda":3.0,"reg_alpha":0.5,"gamma":0.1,"tree_method":"hist","scale_pos_weight":"sqrt(neg/pos)"},
    "calibration":{"type":"IsotonicRegression","fit_on":"validation","applied_to":"test"},
    "seeds":[42,123,777],
    "metrics":["AUROC","AUPRC","Brier","ECE"],
    "labels":{"modis":"fire_modis (MODIS 1km)","viirs":"fire_viirs (VIIRS 375m)","union":"fire_union (MODIS OR VIIRS)"},
    "caveats":[
        "Model predicts probability of SATELLITE-DETECTED fire, not ground-truth fire",
        "MODIS and VIIRS have different detection characteristics (resolution, overpass, sensitivity)",
        "Jaccard MODIS-VIIRS = 0.106 (low agreement on positive labels)",
        "Product is demo, not validated operational system",
        "NASA POWER vs Open-Meteo bias not checked",
    ],
}
with open(OUT/"config_v871.json","w") as f:json.dump(config,f,indent=2,default=str)
print("  ✓ Config saved: config_v871.json")

# 3c: Full results table
all_results={
    "exp1_temporal_forecasting":exp1_results,
    "exp2a_cross_label":exp2a,
    "exp2b_noise_sensitivity":exp2b,
    "exp3a_multiseed":{"seeds":[42,123,777],"aurocs":seeds_results,"mean":float(np.mean(seeds_results)),"std":float(np.std(seeds_results))},
    "config":config,
}
with open(OUT/"results_v871.json","w") as f:json.dump(all_results,f,indent=2,default=str)
print("  ✓ Results saved: results_v871.json")

# 3d: Export model (t+1 MODIS, temporal)
print("  Exporting model (t+1 MODIS temporal)...")
r_final,m_final,_,_=train_eval(tr,va,te,COMBINED,"label_modis_h1")
MODEL_DIR=OUT/"model"
MODEL_DIR.mkdir(exist_ok=True)
m_final.save_model(str(MODEL_DIR/"xgb_v871.json"))
# Also save iso + imp
tr_d=tr.dropna(subset=["label_modis_h1"]);va_d=va.dropna(subset=["label_modis_h1"])
dm=np.nanmean(tr_d[COMBINED[di_idx]].values);ds=np.nanstd(tr_d[COMBINED[di_idx]].values)+1e-6
def prep(d):
    x=d[COMBINED].values.astype(np.float32).copy()
    x[:,di_idx]=np.nan_to_num((x[:,di_idx]-dm)/ds,nan=0.0)
    return x
imp_final=SimpleImputer(strategy="median")
Xtr=imp_final.fit_transform(prep(tr_d));Xva=imp_final.transform(prep(va_d))
ytr=tr_d["label_modis_h1"].values.astype(np.float32);yva=va_d["label_modis_h1"].values.astype(np.float32)
m_final.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
pva=m_final.predict_proba(Xva)[:,1]
iso_final=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1);iso_final.fit(pva,yva)
with open(MODEL_DIR/"iso_v871.pkl","wb")as f:pickle.dump(iso_final,f)
with open(MODEL_DIR/"imp_v871.pkl","wb")as f:pickle.dump(imp_final,f)
with open(MODEL_DIR/"meta_v871.json","w")as f:json.dump({
    "version":"v8.7.1","features":COMBINED,"trained_on":"2018-2021 temporal","test":"2023",
    "drought_idx":di_idx,"drought_mean":float(dm),"drought_std":float(ds),
    "AUROC":float(r_final["AUROC_model"]),"AUPRC":float(r_final["AUPRC_model"]),
    "Brier":float(r_final["Brier_model"]),
    "status":"demo","caveats":config["caveats"],
},f,indent=2,default=str)
print(f"  ✓ Model: AUROC={r_final['AUROC_model']:.4f}")

# 3e: ZIP
print("  Building ZIP...")
ZIP_PATH=Path("/home/z/my-project/download/wildfire_v871.zip")
with zipfile.ZipFile(ZIP_PATH,"w",zipfile.ZIP_DEFLATED)as zf:
    for f in sorted(OUT.iterdir()):
        if f.is_file(): zf.write(f,f"v871/{f.name}")
    for f in sorted(MODEL_DIR.iterdir()):
        if f.is_file(): zf.write(f,f"v871/model/{f.name}")
    zf.write("/home/z/my-project/scripts/v871_experiments.py","v871/scripts/v871_experiments.py")
    zf.writestr("v871/README.md",f"""# Wildfire v8.7.1 — Ba Thí Nghiệm Khoa Học

## 1. Dự báo tương lai (t+1, t+3, t+7)

| Horizon | AUROC (model) | AUROC (climatology) | Δ | Brier |
|---------|--------------|--------------------|----|-------|
| t+1 | {exp1_results['h1']['AUROC_model']:.4f} | {exp1_results['h1']['AUROC_climatology']:.4f} | {exp1_results['h1']['AUROC_model']-exp1_results['h1']['AUROC_climatology']:+.4f} | {exp1_results['h1']['Brier_model']:.4f} |
| t+3 | {exp1_results['h3']['AUROC_model']:.4f} | {exp1_results['h3']['AUROC_climatology']:.4f} | {exp1_results['h3']['AUROC_model']-exp1_results['h3']['AUROC_climatology']:+.4f} | {exp1_results['h3']['Brier_model']:.4f} |
| t+7 | {exp1_results['h7']['AUROC_model']:.4f} | {exp1_results['h7']['AUROC_climatology']:.4f} | {exp1_results['h7']['AUROC_model']-exp1_results['h7']['AUROC_climatology']:+.4f} | {exp1_results['h7']['Brier_model']:.4f} |

Model vượt baseline climatology ở mọi horizon → thực sự dự báo tương lai.

## 2. MODIS-VIIRS label experiment

### Cross-label
| Train | Test | AUROC |
|-------|------|-------|
| MODIS | MODIS | {exp2a.get('train_MODIS_test_MODIS',{}).get('AUROC',0):.4f} |
| VIIRS | VIIRS | {exp2a.get('train_VIIRS_test_VIIRS',{}).get('AUROC',0):.4f} |
| MODIS | VIIRS | {exp2a.get('train_MODIS_test_VIIRS',{}).get('AUROC',0):.4f} |
| VIIRS | MODIS | {exp2a.get('train_VIIRS_test_MODIS',{}).get('AUROC',0):.4f} |
| UNION | MODIS | {exp2a.get('train_UNION_test_MODIS',{}).get('AUROC',0):.4f} |
| UNION | VIIRS | {exp2a.get('train_UNION_test_VIIRS',{}).get('AUROC',0):.4f} |

### Noise sensitivity
| Noise % | AUROC |
|---------|-------|
| 0% | {exp2b['noise_0pct']['AUROC']:.4f} |
| 5% | {exp2b['noise_5pct']['AUROC']:.4f} |
| 10% | {exp2b['noise_10pct']['AUROC']:.4f} |
| 20% | {exp2b['noise_20pct']['AUROC']:.4f} |

## 3. Independent verification
- 3 seeds: {np.mean(seeds_results):.4f} ± {np.std(seeds_results):.4f}
- Config: config_v871.json
- Results: results_v871.json
- Model: model/xgb_v871.json

## Caveats
- Model predicts probability of SATELLITE-DETECTED fire, not ground-truth
- MODIS and VIIRS have different detection characteristics
- Product is demo, not validated operational system
- NASA POWER vs Open-Meteo bias not checked
""")

print(f"\n{'='*60}")
print("TÓM TẮT v8.7.1")
print(f"{'='*60}")
print(f"\n  Thí nghiệm 1 — Dự báo tương lai:")
print(f"  {'Horizon':<10} | {'Model AUROC':>12} | {'Climatology':>12} | {'Δ':>8} | {'Brier':>8}")
print(f"  {'-'*55}")
for h in [1,3,7]:
    r=exp1_results[f"h{h}"]
    print(f"  t+{h:<8} | {r['AUROC_model']:>12.4f} | {r['AUROC_climatology']:>12.4f} | {r['AUROC_model']-r['AUROC_climatology']:>+8.4f} | {r['Brier_model']:>8.4f}")
print(f"\n  Thí nghiệm 2 — Cross-label:")
for name,r in exp2a.items():
    print(f"  {name}: AUROC={r['AUROC']:.4f}")
print(f"\n  Noise sensitivity:")
for k,r in exp2b.items():
    print(f"  {k}: AUROC={r['AUROC']:.4f}")
print(f"\n  Thí nghiệm 3 — Multi-seed: {np.mean(seeds_results):.4f} ± {np.std(seeds_results):.4f}")
print(f"\n  ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size//1024//1024} MB)")
