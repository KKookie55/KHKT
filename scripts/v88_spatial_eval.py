#!/usr/bin/env python3
"""
v8.8 — Spatial Generalization Framework

Build 5 spatial folds (leave-one-region-out by lat band)
+ spatiotemporal folds (source regions 2018-21 → held-out region 2023)
+ B0-B3 baselines on each fold
+ Resumable: saves after each fold, skips completed

Phases:
  Phase 1: Build folds + run B0-B3 on temporal (1 fold, fast)
  Phase 2: Run B0-B3 on 5 spatial folds (resumable)
  Phase 3: Run B0-B3 on 5 spatiotemporal folds (resumable)
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
OUT = Path("/home/z/my-project/download/v88")
OUT.mkdir(parents=True, exist_ok=True)

# Config
GRID=0.5; LAT_MIN,LAT_MAX=11.0,21.0; LON_MIN,LON_MAX=104.0,109.5
YEARS=list(range(2018,2024))
SEEDS=[42, 123, 777]

# 5 spatial regions by latitude
REGIONS = [
    ("R1_North_Mtns",  19.0, 21.0),
    ("R2_North_Central", 17.0, 19.0),
    ("R3_Central_North", 15.0, 17.0),
    ("R4_Central_Highlands", 13.0, 15.0),
    ("R5_South",        11.0, 13.0),
]

print("="*60); print("v8.8 — SPATIAL GENERALIZATION FRAMEWORK"); print("="*60)

# ============================================================
# LOAD/REBUILD PANEL
# ============================================================
panel_path = OUT / "panel_v88.parquet"
if panel_path.exists():
    print("\n[0] Loading panel...")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
else:
    print("\n[0] Building panel...")
    wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv"); wx["date"]=pd.to_datetime(wx["date"])
    modis = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in YEARS], ignore_index=True)
    modis["lat"]=np.round((modis["latitude"]-LAT_MIN)/GRID)*GRID+LAT_MIN
    modis["lon"]=np.round((modis["longitude"]-LON_MIN)/GRID)*GRID+LON_MIN
    modis["date"]=pd.to_datetime(modis["acq_date"])
    modis=modis[(modis["lat"]>=LAT_MIN)&(modis["lat"]<=LAT_MAX)&(modis["lon"]>=LON_MIN)&(modis["lon"]<=LON_MAX)]
    fcd=modis[["lat","lon","date"]].drop_duplicates(); fcd["fire"]=1
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
    panel["heat_index"]=(panel["t2m"]-20)*(50-panel["rh"])/50.0
    panel["wind_chill"]=(panel["t2m"]-15)*panel["wind"]/30.0
    panel["fire_prev"]=panel.groupby(["lat","lon"])["fire"].shift(1)
    panel["cell_fire_rate_exp"]=panel.groupby(["lat","lon"])["fire_prev"].transform(lambda s:s.expanding(min_periods=1).mean()).fillna(panel["fire"].mean())
    panel["cell_fire_count_30d"]=panel.groupby(["lat","lon"])["fire_prev"].transform(lambda s:s.rolling(30,min_periods=1).sum())
    panel["is_dry"]=panel["date"].dt.month.apply(lambda m:1 if m in[1,2,3,4,11,12]else 0)
    panel["is_dry_prev"]=panel.groupby(["lat","lon"])["is_dry"].shift(1)
    panel["cell_fire_rate_dry_exp"]=panel.groupby(["lat","lon"]).apply(lambda g:(g["fire_prev"]*g["is_dry_prev"]).expanding(min_periods=1).sum()/(g["is_dry_prev"].expanding(min_periods=1).sum()+1e-6)).reset_index(level=[0,1],drop=True).fillna(panel["fire"].mean())
    panel=panel.drop(columns=["fire_prev","is_dry_prev"])
    panel["label_h1"]=panel.groupby(["lat","lon"])["fire"].shift(-1)
    # Region assignment
    panel["region"]="unknown"
    for name, lo, hi in REGIONS:
        panel.loc[(panel["lat"]>=lo)&(panel["lat"]<hi+0.001), "region"]=name
    panel.to_parquet(panel_path, index=False)

panel_hash = int(pd.util.hash_pandas_object(panel[["lat","lon","date","fire"]]).sum())
print(f"  Panel: {len(panel):,} rows, hash={panel_hash}")
print(f"  Regions: {panel['region'].value_counts().to_dict()}")

WEATHER=["t2m","rh","wind","precip","doy","t2m_lag1","t2m_lag3","t2m_lag7","t2m_lag14","rh_lag1","rh_lag3","rh_lag7","rh_lag14","wind_lag1","wind_lag3","wind_lag7","wind_lag14","precip_lag1","precip_lag3","precip_lag7","precip_lag14","t2m_roll7","t2m_roll14","rh_roll7","rh_roll14","precip_roll7_sum","precip_roll14_sum","precip_roll30_sum","days_since_rain","t2m_diff3","rh_diff3","wind_diff3","heat_index","wind_chill"]
HISTORY=["cell_fire_rate_exp","cell_fire_rate_dry_exp","cell_fire_count_30d"]
COMBINED=WEATHER+HISTORY
di_idx=COMBINED.index("precip_roll30_sum")

# ============================================================
# HELPERS
# ============================================================
def train_xgb(tr, va, te, features, seed=42):
    dm=np.nanmean(tr[features[di_idx]].values) if "precip_roll30_sum" in features else 0
    ds=(np.nanstd(tr[features[di_idx]].values)+1e-6) if "precip_roll30_sum" in features else 1
    def prep(d):
        x=d[features].values.astype(np.float32).copy()
        if "precip_roll30_sum" in features: x[:,di_idx]=np.nan_to_num((x[:,di_idx]-dm)/ds,nan=0.0)
        return x
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(prep(tr));Xva=imp.transform(prep(va));Xte=imp.transform(prep(te))
    ytr=tr["label_h1"].values.astype(np.float32);yva=va["label_h1"].values.astype(np.float32);yte=te["label_h1"].values.astype(np.float32)
    if len(np.unique(yte))<2 or yte.sum()<5: return None
    pos=max(int(ytr.sum()),1);neg=max(int((1-ytr).sum()),1);spw=max(math.sqrt(neg/pos),1.0)
    m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=seed,n_jobs=4)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    pva=m.predict_proba(Xva)[:,1];pte=m.predict_proba(Xte)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1);iso.fit(pva,yva)
    pi=iso.predict(pte)
    # ECE
    bins=np.linspace(0,1,11);ece=0;n=len(yte)
    for i in range(10):
        lo,hi=bins[i],bins[i+1]
        ib=((pi>=lo)&(pi<=hi)) if i==9 else ((pi>=lo)&(pi<hi))
        if ib.sum()==0: continue
        ece+=(ib.sum()/n)*abs(yte[ib].mean()-pi[ib].mean())
    return {"AUROC":float(roc_auc_score(yte,pi)),"AUPRC":float(average_precision_score(yte,pi)),
            "Brier":float(brier_score_loss(yte,pi)),"ECE":float(ece),
            "n_test":len(te),"n_pos":int(yte.sum()),"pos_rate":float(yte.mean())}

def climatology_baseline(tr, te):
    """B0: cell × month fire rate from train, applied to test."""
    tr2=tr.copy(); tr2["month"]=tr2["date"].dt.month
    clim=tr2.groupby(["lat","lon","month"])["label_h1"].mean().reset_index()
    clim.columns=["lat","lon","month","prob"]
    te2=te.copy(); te2["month"]=te2["date"].dt.month
    te2=te2.merge(clim, on=["lat","lon","month"], how="left")
    te2["prob"]=te2["prob"].fillna(tr["label_h1"].mean())
    yte=te["label_h1"].values.astype(np.float32)
    p=te2["prob"].values
    if len(np.unique(yte))<2 or yte.sum()<5: return None
    return {"AUROC":float(roc_auc_score(yte,p)),"AUPRC":float(average_precision_score(yte,p)),
            "Brier":float(brier_score_loss(yte,p)),"ECE":0.0,
            "n_test":len(te),"n_pos":int(yte.sum()),"pos_rate":float(yte.mean())}

# ============================================================
# RESUMABLE: load existing results
# ============================================================
results_path = OUT / "results_v88.json"
if results_path.exists():
    all_results = json.loads(results_path.read_text())
    print(f"\n  Loaded {len(all_results.get('folds',{}))} existing folds")
else:
    all_results = {"version":"v8.8","panel_hash":panel_hash,"folds":{}}

def save_results():
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  ✓ Saved: {results_path}")

# ============================================================
# PHASE 1: TEMPORAL (train 2018-21, val 2022, test 2023, all regions)
# ============================================================
fold_name = "temporal"
if fold_name not in all_results["folds"]:
    print(f"\n[Phase 1] {fold_name}...")
    tr=panel[panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_h1"])
    va=panel[panel["year"]==2022].dropna(subset=["label_h1"])
    te=panel[panel["year"]==2023].dropna(subset=["label_h1"])
    r_b0=climatology_baseline(tr, te)
    r_b1=train_xgb(tr,va,te,WEATHER)
    r_b2=train_xgb(tr,va,te,HISTORY)
    r_b3=train_xgb(tr,va,te,COMBINED)
    all_results["folds"][fold_name]={"B0":r_b0,"B1":r_b1,"B2":r_b2,"B3":r_b3}
    print(f"  B0={r_b0['AUROC']:.4f} B1={r_b1['AUROC']:.4f} B2={r_b2['AUROC']:.4f} B3={r_b3['AUROC']:.4f}")
    save_results()
else:
    print(f"\n[Phase 1] {fold_name} already done — skipping")

# ============================================================
# PHASE 2: SPATIAL LEAVE-ONE-REGION-OUT
# ============================================================
print(f"\n[Phase 2] Spatial leave-one-region-out (5 folds)...")
for region_name, lat_lo, lat_hi in REGIONS:
    fold_name = f"spatial_LORO_{region_name}"
    if fold_name in all_results["folds"]:
        print(f"  {fold_name}: already done — skipping")
        continue
    # Test = held-out region (all years)
    te = panel[(panel["region"]==region_name)].dropna(subset=["label_h1"])
    # Train = all other regions (2018-2021)
    tr = panel[(panel["region"]!=region_name)&panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_h1"])
    va = panel[(panel["region"]!=region_name)&(panel["year"]==2022)].dropna(subset=["label_h1"])
    if len(tr)<1000 or len(te)<100 or te["label_h1"].sum()<5:
        print(f"  {fold_name}: insufficient data (tr={len(tr)}, te={len(te)}, pos={int(te['label_h1'].sum())}) — skipping")
        continue
    r_b0=climatology_baseline(tr, te)
    r_b1=train_xgb(tr,va,te,WEATHER)
    r_b2=train_xgb(tr,va,te,HISTORY)
    r_b3=train_xgb(tr,va,te,COMBINED)
    all_results["folds"][fold_name]={"B0":r_b0,"B1":r_b1,"B2":r_b2,"B3":r_b3}
    print(f"  {fold_name}: B0={r_b0['AUROC']:.4f} B1={r_b1['AUROC']:.4f} B2={r_b2['AUROC']:.4f} B3={r_b3['AUROC']:.4f} (n={len(te):,})")
    save_results()

# ============================================================
# PHASE 3: SPATIOTEMPORAL (source regions 2018-21 → held-out region 2023)
# ============================================================
print(f"\n[Phase 3] Spatiotemporal (source 2018-21 → target 2023)...")
for region_name, lat_lo, lat_hi in REGIONS:
    fold_name = f"spatiotemporal_{region_name}"
    if fold_name in all_results["folds"]:
        print(f"  {fold_name}: already done — skipping")
        continue
    # Test = held-out region × 2023 only
    te = panel[(panel["region"]==region_name)&(panel["year"]==2023)].dropna(subset=["label_h1"])
    tr = panel[(panel["region"]!=region_name)&panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_h1"])
    va = panel[(panel["region"]!=region_name)&(panel["year"]==2022)].dropna(subset=["label_h1"])
    if len(tr)<1000 or len(te)<50 or te["label_h1"].sum()<3:
        print(f"  {fold_name}: insufficient data (tr={len(tr)}, te={len(te)}, pos={int(te['label_h1'].sum())}) — skipping")
        continue
    r_b0=climatology_baseline(tr, te)
    r_b1=train_xgb(tr,va,te,WEATHER)
    r_b2=train_xgb(tr,va,te,HISTORY)
    r_b3=train_xgb(tr,va,te,COMBINED)
    all_results["folds"][fold_name]={"B0":r_b0,"B1":r_b1,"B2":r_b2,"B3":r_b3}
    print(f"  {fold_name}: B0={r_b0['AUROC']:.4f} B1={r_b1['AUROC']:.4f} B2={r_b2['AUROC']:.4f} B3={r_b3['AUROC']:.4f} (n={len(te):,})")
    save_results()

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("TÓM TẮT v8.8 — SPATIAL GENERALIZATION")
print(f"{'='*60}")

print(f"\n  Temporal (all regions, 2018-21→2023):")
f = all_results["folds"].get("temporal", {})
for b in ["B0","B1","B2","B3"]:
    if b in f and f[b]: print(f"    {b}: AUROC={f[b]['AUROC']:.4f}, AUPRC={f[b]['AUPRC']:.4f}, Brier={f[b]['Brier']:.4f}")

print(f"\n  Spatial leave-one-region-out:")
print(f"  {'Fold':<30} | {'B0':>7} | {'B1':>7} | {'B2':>7} | {'B3':>7} | {'n_test':>7}")
print(f"  {'-'*70}")
spatial_aurocs = {"B0":[],"B1":[],"B2":[],"B3":[]}
for region_name, _, _ in REGIONS:
    fn = f"spatial_LORO_{region_name}"
    f = all_results["folds"].get(fn, {})
    if not f: continue
    vals = []
    for b in ["B0","B1","B2","B3"]:
        v = f.get(b, {})
        auroc = v.get("AUROC", 0) if v else 0
        vals.append(f"{auroc:.4f}")
        if v: spatial_aurocs[b].append(auroc)
    n = f.get("B1",{}).get("n_test",0) if f.get("B1") else 0
    print(f"  {fn:<30} | {vals[0]:>7} | {vals[1]:>7} | {vals[2]:>7} | {vals[3]:>7} | {n:>7,}")

print(f"\n  Macro-average (spatial):")
for b in ["B0","B1","B2","B3"]:
    if spatial_aurocs[b]:
        m = np.mean(spatial_aurocs[b]); s = np.std(spatial_aurocs[b])
        worst = np.min(spatial_aurocs[b])
        print(f"    {b}: mean={m:.4f}±{s:.4f}, worst={worst:.4f}")

print(f"\n  Spatiotemporal (source 2018-21 → target 2023):")
print(f"  {'Fold':<30} | {'B0':>7} | {'B1':>7} | {'B2':>7} | {'B3':>7} | {'n_test':>7}")
print(f"  {'-'*70}")
st_aurocs = {"B0":[],"B1":[],"B2":[],"B3":[]}
for region_name, _, _ in REGIONS:
    fn = f"spatiotemporal_{region_name}"
    f = all_results["folds"].get(fn, {})
    if not f: continue
    vals = []
    for b in ["B0","B1","B2","B3"]:
        v = f.get(b, {})
        auroc = v.get("AUROC", 0) if v else 0
        vals.append(f"{auroc:.4f}")
        if v: st_aurocs[b].append(auroc)
    n = f.get("B1",{}).get("n_test",0) if f.get("B1") else 0
    print(f"  {fn:<30} | {vals[0]:>7} | {vals[1]:>7} | {vals[2]:>7} | {vals[3]:>7} | {n:>7,}")

print(f"\n  Macro-average (spatiotemporal):")
for b in ["B0","B1","B2","B3"]:
    if st_aurocs[b]:
        m = np.mean(st_aurocs[b]); s = np.std(st_aurocs[b])
        worst = np.min(st_aurocs[b])
        print(f"    {b}: mean={m:.4f}±{s:.4f}, worst={worst:.4f}")

# ZIP
ZIP = Path("/home/z/my-project/download/wildfire_v88.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT.iterdir()):
        if f.is_file(): zf.write(f, f"v88/{f.name}")
    zf.write(__file__, "v88/scripts/v88_spatial_eval.py")
    zf.writestr("v88/README.md", f"""# Wildfire v8.8 — Spatial Generalization Framework

## Research Question
Can a wildfire risk model maintain accuracy, calibration, and reliability
when transferred to a region not seen in training?

## Experimental Design
- 3 evaluation modes: temporal, spatial (leave-one-region-out), spatiotemporal
- 4 baselines: B0 (climatology), B1 (weather-only), B2 (history-only), B3 (combined)
- 5 regions by latitude: North Mtns, North Central, Central North, Central Highlands, South
- Metrics: AUROC, AUPRC, Brier, ECE, n_test, pos_rate
- Panel hash: {panel_hash}

## Leakage fixes
1. Fire history: expanding window (no future)
2. Weather: ffill only (no bfill)
3. Calibration: Iso fit on val
4. Cold-start: test fire history from expanding window (no future labels)

## Caveats
- Model predicts satellite-detected fire (MODIS), not ground-truth
- 0.5° grid is coarse (~55km)
- NASA POWER vs Open-Meteo bias not checked
- Product is demo, not validated operational system
""")
print(f"\n  ZIP: {ZIP} ({ZIP.stat().st_size//1024} KB)")
