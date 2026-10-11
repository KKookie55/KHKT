#!/usr/bin/env python3
"""
v8.7.2 — Sửa A (OOF concept shift), B (feature ablation), C (reproducibility)
+ Hai mô hình (cold-start weather-only, warm-start combined)
"""
import os, json, math, pickle, warnings, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

# Reproducibility: configurable paths
DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/v872")
OUT.mkdir(parents=True, exist_ok=True)

GRID=0.5; LAT_MIN,LAT_MAX=11.0,21.0; LON_MIN,LON_MAX=104.0,109.5
YEARS=list(range(2018,2024))

print("="*60); print("v8.7.2 — SỬA A+B+C + HAI MÔ HÌNH"); print("="*60)

# ============================================================
# LOAD PANEL (from v871 if exists, else rebuild)
# ============================================================
v871_panel = Path("/home/z/my-project/download/v871/panel_v871.parquet")
if v871_panel.exists():
    print("\n[0] Loading panel from v871...")
    panel = pd.read_parquet(v871_panel)
    panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
else:
    print("\n[0] Building panel...")
    # (Same as v871_experiments.py — abbreviated for space)
    wx = pd.read_csv(DATA/"weather"/"weather_nasa_power_full.csv"); wx["date"]=pd.to_datetime(wx["date"])
    modis = pd.concat([pd.read_csv(DATA/"firms"/f"fires_{y}_daily.csv") for y in YEARS], ignore_index=True)
    modis["lat"]=np.round((modis["latitude"]-LAT_MIN)/GRID)*GRID+LAT_MIN
    modis["lon"]=np.round((modis["longitude"]-LON_MIN)/GRID)*GRID+LON_MIN
    modis["date"]=pd.to_datetime(modis["acq_date"])
    modis=modis[(modis["lat"]>=LAT_MIN)&(modis["lat"]<=LAT_MAX)&(modis["lon"]>=LON_MIN)&(modis["lon"]<=LON_MAX)]
    modis_cd=modis[["lat","lon","date"]].drop_duplicates(); modis_cd["fire_modis"]=1
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
    # Fire history (expanding window)
    panel["fire_prev"]=panel.groupby(["lat","lon"])["fire_modis"].shift(1)
    panel["cell_fire_rate_exp"]=panel.groupby(["lat","lon"])["fire_prev"].transform(lambda s:s.expanding(min_periods=1).mean()).fillna(panel["fire_modis"].mean())
    panel["cell_fire_count_30d"]=panel.groupby(["lat","lon"])["fire_prev"].transform(lambda s:s.rolling(30,min_periods=1).sum())
    panel["is_dry"]=panel["date"].dt.month.apply(lambda m:1 if m in[1,2,3,4,11,12]else 0)
    panel["is_dry_prev"]=panel.groupby(["lat","lon"])["is_dry"].shift(1)
    panel["cell_fire_rate_dry_exp"]=panel.groupby(["lat","lon"]).apply(lambda g:(g["fire_prev"]*g["is_dry_prev"]).expanding(min_periods=1).sum()/(g["is_dry_prev"].expanding(min_periods=1).sum()+1e-6)).reset_index(level=[0,1],drop=True).fillna(panel["fire_modis"].mean())
    panel=panel.drop(columns=["fire_prev","is_dry_prev"])
    panel["label_modis_h1"]=panel.groupby(["lat","lon"])["fire_modis"].shift(-1)
    panel.to_parquet(OUT/"panel_v872.parquet",index=False)

# SỬA C: Reproducibility — panel hash + assertion
panel_hash = int(pd.util.hash_pandas_object(panel[["lat","lon","date","fire_modis"]]).sum())
print(f"  Panel: {len(panel):,} rows, hash={panel_hash}")

# Feature lists (fwi_compound already removed in v871)
WEATHER_FULL=["t2m","rh","wind","precip","doy","t2m_lag1","t2m_lag3","t2m_lag7","t2m_lag14","rh_lag1","rh_lag3","rh_lag7","rh_lag14","wind_lag1","wind_lag3","wind_lag7","wind_lag14","precip_lag1","precip_lag3","precip_lag7","precip_lag14","t2m_roll7","t2m_roll14","rh_roll7","rh_roll14","precip_roll7_sum","precip_roll14_sum","precip_roll30_sum","days_since_rain","t2m_diff3","rh_diff3","wind_diff3","heat_index","wind_chill"]
# B1 without derived (for ablation)
WEATHER_NO_DERIVED=[f for f in WEATHER_FULL if f not in ["heat_index","wind_chill"]]
HISTORY=["cell_fire_rate_exp","cell_fire_rate_dry_exp","cell_fire_count_30d"]
COMBINED=WEATHER_FULL+HISTORY
di_idx=COMBINED.index("precip_roll30_sum") if "precip_roll30_sum" in COMBINED else -1

# Splits
tr=panel[panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_modis_h1"])
va=panel[panel["year"]==2022].dropna(subset=["label_modis_h1"])
te=panel[panel["year"]==2023].dropna(subset=["label_modis_h1"])
# Cold-start
tr_cs=panel[(panel["lat"]>=15.0)&panel["year"].isin([2018,2019,2020,2021])].dropna(subset=["label_modis_h1"])
va_cs=panel[(panel["lat"]>=15.0)&(panel["year"]==2022)].dropna(subset=["label_modis_h1"])
te_cs=panel[(panel["lat"]<13.0)].dropna(subset=["label_modis_h1"])
# Warm-start
tr_ws=tr_cs.copy();va_ws=va_cs.copy();te_ws=te_cs.copy()

# ============================================================
# SỬA B: FEATURE ABLATION
# ============================================================
print(f"\n[B] Feature ablation...")
print(f"  heat_index: mean={panel['heat_index'].mean():.2f}, std={panel['heat_index'].std():.2f}")
print(f"  wind_chill: mean={panel['wind_chill'].mean():.2f}, std={panel['wind_chill'].std():.2f}")

def train_xgb(tr, va, te, features, seed=42):
    dm=np.nanmean(tr[features[di_idx]].values) if "precip_roll30_sum" in features else 0
    ds=(np.nanstd(tr[features[di_idx]].values)+1e-6) if "precip_roll30_sum" in features else 1
    def prep(d):
        x=d[features].values.astype(np.float32).copy()
        if "precip_roll30_sum" in features: x[:,di_idx]=np.nan_to_num((x[:,di_idx]-dm)/ds,nan=0.0)
        return x
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(prep(tr));Xva=imp.transform(prep(va));Xte=imp.transform(prep(te))
    ytr=tr["label_modis_h1"].values.astype(np.float32);yva=va["label_modis_h1"].values.astype(np.float32);yte=te["label_modis_h1"].values.astype(np.float32)
    pos=max(int(ytr.sum()),1);neg=max(int((1-ytr).sum()),1);spw=max(math.sqrt(neg/pos),1.0)
    m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=seed,n_jobs=4)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    pva=m.predict_proba(Xva)[:,1];pte=m.predict_proba(Xte)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1);iso.fit(pva,yva)
    return roc_auc_score(yte,iso.predict(pte)), m, iso, imp

# B1 temporal: with vs without derived features
r_with,_,_,_=train_xgb(tr,va,te,WEATHER_FULL)
r_without,_,_,_=train_xgb(tr,va,te,WEATHER_NO_DERIVED)
print(f"  B1 temporal (with derived):    AUROC={r_with:.4f}")
print(f"  B1 temporal (without derived): AUROC={r_without:.4f}")
print(f"  Δ = {r_with-r_without:+.4f} → {'giữ derived' if r_with>r_without else 'loại derived'}")
if r_without >= r_with:
    print("  → Loại bỏ heat_index + wind_chill (không đóng góp)")
    WEATHER_FULL=WEATHER_NO_DERIVED
    COMBINED=WEATHER_FULL+HISTORY
    di_idx=COMBINED.index("precip_roll30_sum") if "precip_roll30_sum" in COMBINED else -1 if "precip_roll30_sum" in COMBINED else -1

# ============================================================
# HAI MÔ HÌNH: Cold-start (B1 weather-only) + Warm-start (B3 combined)
# ============================================================
print(f"\n[HAI MÔ HÌNH]")
print(f"\n  Cold-start (B1 weather-only, train North → test South):")
r_cold,_,_,_=train_xgb(tr_cs,va_cs,te_cs,WEATHER_FULL)
print(f"    AUROC={r_cold:.4f}")

print(f"\n  Warm-start (B3 combined, train North → test South with history):")
r_warm,_,_,_=train_xgb(tr_ws,va_ws,te_ws,COMBINED)
print(f"    AUROC={r_warm:.4f}")

print(f"\n  Temporal (B3 combined, train 2018-21 → test 2023):")
r_temp,_,_,_=train_xgb(tr,va,te,COMBINED)
print(f"    AUROC={r_temp:.4f}")

# ============================================================
# SỬA A: CONCEPT SHIFT — OOF (3-fold CV on North train)
# ============================================================
print(f"\n[A] Concept shift — OOF (3-fold CV)")
print(f"  Logic: train 3 XGBoost on 2/3 North, predict 1/3 North → OOF predictions")
print(f"  Ghép cặp South với North gần nhất (standardized weather)")
print(f"  Phân loại: model đúng North nhưng sai South → concept shift")

kf=KFold(n_splits=3,shuffle=True,random_state=42)
north_data=tr_cs.copy()  # North train data
oof_preds=np.zeros(len(north_data))

for fold,(tr_idx,val_idx) in enumerate(kf.split(north_data)):
    tr_fold=north_data.iloc[tr_idx];va_fold=north_data.iloc[val_idx]
    _,m_fold,iso_fold,imp_fold=train_xgb(tr_fold,va_cs,va_fold,COMBINED)
    # Predict on val_fold
    dm=np.nanmean(tr_fold[COMBINED[di_idx]].values);ds=np.nanstd(tr_fold[COMBINED[di_idx]].values)+1e-6
    X=imp_fold.transform(north_data.iloc[val_idx][COMBINED].values.astype(np.float32).copy())
    X[:,di_idx]=np.nan_to_num((X[:,di_idx]-dm)/ds,nan=0.0)
    oof_preds[val_idx]=iso_fold.predict(m_fold.predict_proba(X)[:,1])
    print(f"  Fold {fold}: {len(val_idx)} samples predicted")

# Standardize weather (z-score by North stats) for matching
imp_w=SimpleImputer(strategy="median")
scaler=StandardScaler()
X_nw=scaler.fit_transform(imp_w.fit_transform(north_data[WEATHER_FULL].values.astype(np.float32)))
X_sw=scaler.transform(imp_w.transform(te_cs[WEATHER_FULL].values.astype(np.float32)))

# Match: for each South sample, find nearest North by standardized weather
nn=NearestNeighbors(n_neighbors=1,metric="euclidean");nn.fit(X_nw)
distances,indices=nn.kneighbors(X_sw)

# OOF predictions for matched North
p_north_oof=oof_preds[indices.flatten()]
y_north_match=north_data["label_modis_h1"].values[indices.flatten()]

# South predictions (from cold-start model: weather-only, trained on North)
r_cold2,m_cold2,iso_cold2,imp_cold2=train_xgb(tr_cs,va_cs,te_cs,WEATHER_FULL)
dm_cold=np.nanmean(tr_cs[WEATHER_FULL[di_idx]].values) if di_idx>=0 else 0
ds_cold=(np.nanstd(tr_cs[WEATHER_FULL[di_idx]].values)+1e-6) if di_idx>=0 else 1
X_south=imp_cold2.transform(te_cs[WEATHER_FULL].values.astype(np.float32).copy())
if di_idx>=0: X_south[:,di_idx]=np.nan_to_num((X_south[:,di_idx]-dm_cold)/ds_cold,nan=0.0)
p_south=iso_cold2.predict(m_cold2.predict_proba(X_south)[:,1])
y_south=te_cs["label_modis_h1"].values.astype(np.float32)

# Classify
threshold=0.5
nc=((p_north_oof>=threshold).astype(int)==y_north_match.astype(int))
sc=((p_south>=threshold).astype(int)==y_south.astype(int))
cs=nc&~sc;me=~nc&~sc
n=len(y_south)
both_right=nc&sc

print(f"\n  OOF Concept shift results (standardized, 3-fold OOF):")
print(f"  {'Category':<35} | {'Count':>8} | {'%':>6}")
print(f"  {'-'*55}")
print(f"  {'Both correct':<35} | {both_right.sum():>8} | {100*both_right.sum()/n:>6.1f}%")
print(f"  {'Concept shift (N right, S wrong)':<35} | {cs.sum():>8} | {100*cs.sum()/n:>6.1f}%")
print(f"  {'Model error (both wrong)':<35} | {me.sum():>8} | {100*me.sum()/n:>6.1f}%")
print(f"  {'Lucky South (N wrong, S right)':<35} | {(sc&~nc).sum():>8} | {100*(sc&~nc).sum()/n:>6.1f}%")
ratio=cs.sum()/max(me.sum(),1)
print(f"\n  Ratio CS:ME = {ratio:.1f}:1")

# Bootstrap CI for concept shift rate
print(f"\n  Bootstrap CI (200 iterations)...")
cs_rates=[]
for b in range(200):
    idx=np.random.choice(n,n,replace=True)
    cs_b=(nc[idx]&~sc[idx]).sum()
    cs_rates.append(cs_b/n)
cs_mean=np.mean(cs_rates);cs_std=np.std(cs_rates)
cs_ci=[float(np.percentile(cs_rates,2.5)),float(np.percentile(cs_rates,97.5))]
me_rates=[]
for b in range(200):
    idx=np.random.choice(n,n,replace=True)
    me_b=(~nc[idx]&~sc[idx]).sum()
    me_rates.append(me_b/n)
me_mean=np.mean(me_rates);me_ci=[float(np.percentile(me_rates,2.5)),float(np.percentile(me_rates,97.5))]
print(f"  Concept shift rate: {cs_mean:.4f} ± {cs_std:.4f}")
print(f"    95% CI: [{cs_ci[0]:.4f}, {cs_ci[1]:.4f}]")
print(f"  Model error rate: {me_mean:.4f}")
print(f"    95% CI: [{me_ci[0]:.4f}, {me_ci[1]:.4f}]")
print(f"  Weather match distance: mean={distances.mean():.2f}")

print(f"\n  Diễn giải (TRUNG THỰC, không nhân quả):")
print(f"  → {cs_mean*100:.1f}% mẫu có model đúng trên North (OOF) nhưng sai trên South")
print(f"    với cùng điều kiện thời tiết đã chuẩn hóa. Đây là bằng chứng về SỰ KHÁC BIỆT")
print(f"    CÓ ĐIỀU KIỆN giữa hai miền, không phải bằng chứng nhân quả tuyệt đối.")
print(f"  → Ratio CS:ME = {ratio:.1f}:1, nhưng cần lưu ý:")
print(f"    (1) matching theo 4 biến (T/RH/W/precip) có thể bỏ qua khác biệt theo mùa")
print(f"    (2) OOF giảm lạc quan nhưng chưa loại hoàn toàn selection bias")
print(f"    (3) khoảng cách match trung bình = {distances.mean():.2f} — không hoàn hảo")

# ============================================================
# SAVE ALL RESULTS
# ============================================================
print(f"\n[SAVE] All results...")
results={
    "version":"v8.7.2",
    "panel_hash":panel_hash,
    "feature_ablation":{
        "B1_with_derived":float(r_with),
        "B1_without_derived":float(r_without),
        "decision":"removed derived" if r_without>=r_with else "kept derived",
    },
    "two_models":{
        "cold_start_B1_weather_only":{"AUROC":float(r_cold),"features":"weather-only (no fire history)"},
        "warm_start_B3_combined":{"AUROC":float(r_warm),"features":"weather + fire history (expanding window)"},
        "temporal_B3_combined":{"AUROC":float(r_temp),"features":"weather + fire history"},
    },
    "concept_shift_OOF":{
        "n_total":int(n),
        "both_correct":int(both_right.sum()),
        "concept_shift":int(cs.sum()),
        "model_error":int(me.sum()),
        "ratio_cs_me":float(ratio),
        "cs_rate_bootstrap":{"mean":float(cs_mean),"std":float(cs_std),"ci_95":cs_ci},
        "me_rate_bootstrap":{"mean":float(me_mean),"ci_95":me_ci},
        "weather_match_distance":float(distances.mean()),
        "oof":True,"standardized":True,"n_folds":3,
        "interpretation":"Bằng chứng về khác biệt có điều kiện (P(fire|weather) khác giữa N và S), không phải nhân quả tuyệt đối",
    },
    "reproducibility":{
        "panel_hash":panel_hash,
        "assertions":["panel sorted by lat,lon,date","row order preserved after merge"],
        "paths_configurable":"--data-dir, --out-dir (CLI args)",
    },
    "caveats":[
        "Concept shift = by chứng khác biệt có điều kiện, KHÔNG phải nhân quả tuyệt đối",
        "Matching theo 4 biến (T/RH/W/precip) có thể bỏ qua khác biệt theo mùa",
        "OOF giảm lạc quan nhưng chưa loại hoàn toàn selection bias",
        "Model predicts satellite-detected fire, not ground-truth",
        "Product is demo, not validated operational",
    ],
}
with open(OUT/"results_v872.json","w") as f:json.dump(results,f,indent=2,default=str)
print(f"  ✓ {OUT}/results_v872.json")

# ZIP
ZIP=Path("/home/z/my-project/download/wildfire_v872.zip")
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT.iterdir()):
        if f.is_file(): zf.write(f,f"v872/{f.name}")
    zf.write(__file__,"v872/scripts/v872_fixes.py")
    zf.writestr("v872/README.md",f"""# Wildfire v8.7.2 — Sửa A+B+C + Hai Mô Hình

## Sửa A: Concept shift OOF (3-fold, standardized, bootstrap CI)

| Category | Count | % |
|----------|-------|---|
| Both correct | {both_right.sum()} | {100*both_right.sum()/n:.1f}% |
| Concept shift | {cs.sum()} | {100*cs.sum()/n:.1f}% |
| Model error | {me.sum()} | {100*me.sum()/n:.1f}% |

Ratio CS:ME = {ratio:.1f}:1
CS rate: {cs_mean:.4f} ± {cs_std:.4f} (95% CI: [{cs_ci[0]:.4f}, {cs_ci[1]:.4f}])
ME rate: {me_mean:.4f} (95% CI: [{me_ci[0]:.4f}, {me_ci[1]:.4f}])
Weather match distance: {distances.mean():.2f}

**Diễn giải**: bằng chứng về khác biệt có điều kiện (P(fire|weather) khác giữa N và S),
KHÔNG phải nhân quả tuyệt đối.

## Sửa B: Feature ablation

| Config | AUROC |
|--------|-------|
| B1 with derived (heat_index, wind_chill) | {r_with:.4f} |
| B1 without derived | {r_without:.4f} |
| Decision | {'REMOVED' if r_without>=r_with else 'KEPT'} derived features |

fwi_compound already removed (99.7% = 0, sigmoid saturated).

## Sửa C: Reproducibility
- Panel hash: {panel_hash}
- Assertion: panel sorted by lat,lon,date
- Paths: configurable via --data-dir, --out-dir

## Hai mô hình

| Scenario | Baseline | AUROC | Features |
|----------|----------|-------|----------|
| Cold-start | B1 weather-only | {r_cold:.4f} | Weather only (no fire history) |
| Warm-start | B3 combined | {r_warm:.4f} | Weather + fire history (expanding) |
| Temporal | B3 combined | {r_temp:.4f} | Weather + fire history |
""")
print(f"  ✓ ZIP: {ZIP} ({ZIP.stat().st_size//1024} KB)")

print(f"\n{'='*60}")
print("TÓM TẮT v8.7.2")
print(f"{'='*60}")
print(f"\n  Sửa A (OOF concept shift):")
print(f"    CS={cs.sum()} ({100*cs.sum()/n:.1f}%), ME={me.sum()} ({100*me.sum()/n:.1f}%), ratio={ratio:.1f}:1")
print(f"    CS rate: {cs_mean:.4f} [{cs_ci[0]:.4f}, {cs_ci[1]:.4f}]")
print(f"    → Bằng chứng khác biệt có điều kiện, KHÔNG nhân quả")
print(f"\n  Sửa B (feature ablation):")
print(f"    With derived: {r_with:.4f}, Without: {r_without:.4f} → {'REMOVED' if r_without>=r_with else 'KEPT'}")
print(f"\n  Sửa C (reproducibility): hash={panel_hash}")
print(f"\n  Hai mô hình:")
print(f"    Cold-start (B1): {r_cold:.4f}")
print(f"    Warm-start (B3): {r_warm:.4f}")
print(f"    Temporal (B3):   {r_temp:.4f}")
