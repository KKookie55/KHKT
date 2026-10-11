#!/usr/bin/env python3
"""v87_train_baselines.py — Giai đoạn 2: Baseline sạch."""
from pathlib import Path
import json, math, pickle, warnings, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings('ignore')

OUT = Path("/home/z/my-project/download/v87")
panel = pd.read_parquet(OUT/"panel_v87.parquet")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
splits = np.load(OUT/"splits_v87.npz")
meta = json.loads(open(OUT/"meta_v87.json").read())
cold_test = pd.read_parquet(OUT/"test_coldstart.parquet")
WEATHER = meta["feature_sets"]["B1_weather_only"]
HISTORY = meta["feature_sets"]["B2_fire_history_only"]
COMBINED = meta["feature_sets"]["B3_weather_plus_history"]

print("="*60); print("v8.7 GIAI ĐOẠN 2 — BASELINE SẠCH"); print("="*60)

def train_xgb(tr, va, te, features, seed=42):
    di_idx = features.index("precip_roll30_sum") if "precip_roll30_sum" in features else -1
    di_mean = np.nanmean(tr[features[di_idx]].values) if di_idx >= 0 else 0
    di_std = (np.nanstd(tr[features[di_idx]].values) + 1e-6) if di_idx >= 0 else 1
    def prep(df):
        X = df[features].values.astype(np.float32).copy()
        if di_idx >= 0: X[:, di_idx] = np.nan_to_num((X[:, di_idx]-di_mean)/di_std, nan=0.0)
        return X
    imp = SimpleImputer(strategy="median")
    X_tr=imp.fit_transform(prep(tr)); X_val=imp.transform(prep(va)); X_test=imp.transform(prep(te))
    y_tr=tr["label_h1"].values.astype(np.float32); y_val=va["label_h1"].values.astype(np.float32); y_test=te["label_h1"].values.astype(np.float32)
    pos=max(int(y_tr.sum()),1); neg=max(int((1-y_tr).sum()),1); spw=max(math.sqrt(neg/pos),1.0)
    m = xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric="aucpr",tree_method="hist",random_state=seed,n_jobs=4)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    p_val=m.predict_proba(X_val)[:,1]; p_test=m.predict_proba(X_test)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1); iso.fit(p_val,y_val)
    p_test_iso=iso.predict(p_test)
    auroc=roc_auc_score(y_test,p_test); auroc_iso=roc_auc_score(y_test,p_test_iso)
    auprc=average_precision_score(y_test,p_test_iso); brier=brier_score_loss(y_test,p_test_iso)
    bins=np.linspace(0,1,11); ece=0; n=len(y_test)
    for i in range(10):
        lo,hi=bins[i],bins[i+1]
        ib=((p_test_iso>=lo)&(p_test_iso<=hi)) if i==9 else ((p_test_iso>=lo)&(p_test_iso<hi))
        if ib.sum()==0: continue
        ece+=(ib.sum()/n)*abs(y_test[ib].mean()-p_test_iso[ib].mean())
    return {"AUROC":float(auroc),"AUROC_iso":float(auroc_iso),"AUPRC":float(auprc),"Brier":float(brier),"ECE_iso":float(ece),"n_test":len(te),"n_pos_test":int(y_test.sum()),"pos_rate_test":float(y_test.mean())}

tr_A = panel.iloc[splits["temporal_train"]].dropna(subset=["label_h1"])
va_A = panel.iloc[splits["temporal_val"]].dropna(subset=["label_h1"])
te_A = panel.iloc[splits["temporal_test"]].dropna(subset=["label_h1"])
tr_B = panel.iloc[splits["coldstart_train"]].dropna(subset=["label_h1"])
va_B = panel.iloc[splits["coldstart_val"]].dropna(subset=["label_h1"])
te_B = cold_test.dropna(subset=["label_h1"])
tr_C = panel.iloc[splits["warmstart_train"]].dropna(subset=["label_h1"])
va_C = panel.iloc[splits["warmstart_val"]].dropna(subset=["label_h1"])
te_C = panel.iloc[splits["warmstart_test"]].dropna(subset=["label_h1"])

print(f"\n  Mode A (temporal): tr={len(tr_A):,} va={len(va_A):,} te={len(te_A):,}")
print(f"  Mode B (cold-start): tr={len(tr_B):,} va={len(va_B):,} te={len(te_B):,}")
print(f"  Mode C (warm-start): tr={len(tr_C):,} va={len(va_C):,} te={len(te_C):,}")

results = {}
configs = [("B1_weather", WEATHER), ("B2_history", HISTORY), ("B3_combined", COMBINED)]
modes = [("A_temporal", tr_A, va_A, te_A), ("B_coldstart", tr_B, va_B, te_B), ("C_warmstart", tr_C, va_C, te_C)]

print(f"\n{'Config':<15} | {'Mode':<15} | {'AUROC':>7} | {'AUROC_iso':>10} | {'AUPRC':>7} | {'Brier':>7} | {'ECE':>7} | {'n_pos':>6}")
print(f"{'-'*85}")
for cfg_name, feats in configs:
    for mode_name, tr, va, te in modes:
        key = f"{cfg_name}_{mode_name}"
        r = train_xgb(tr, va, te, feats)
        results[key] = r
        print(f"{cfg_name:<15} | {mode_name:<15} | {r['AUROC']:>7.4f} | {r['AUROC_iso']:>10.4f} | {r['AUPRC']:>7.4f} | {r['Brier']:>7.4f} | {r['ECE_iso']:>7.4f} | {r['n_pos_test']:>6}")

with open(OUT/"baselines_v87.json", "w") as f: json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print("TÓM TẮT BASELINE")
print(f"{'='*60}")
print(f"\n  {'Config':<15} | {'Temporal':>10} | {'Cold-start':>10} | {'Warm-start':>10}")
print(f"  {'-'*50}")
for cfg_name, _ in configs:
    t = results[f"{cfg_name}_A_temporal"]["AUROC_iso"]
    c = results[f"{cfg_name}_B_coldstart"]["AUROC_iso"]
    w = results[f"{cfg_name}_C_warmstart"]["AUROC_iso"]
    print(f"  {cfg_name:<15} | {t:>10.4f} | {c:>10.4f} | {w:>10.4f}")

print(f"\n  Câu trả lời:")
print(f"  1. AUROC sau sửa leakage (B3 temporal): {results['B3_combined_A_temporal']['AUROC_iso']:.4f}")
print(f"  2. Bỏ fire-history: B1={results['B1_weather_A_temporal']['AUROC_iso']:.4f} vs B3={results['B3_combined_A_temporal']['AUROC_iso']:.4f} Δ={results['B3_combined_A_temporal']['AUROC_iso']-results['B1_weather_A_temporal']['AUROC_iso']:+.4f}")
print(f"  3. B1 weather-only: cold={results['B1_weather_B_coldstart']['AUROC_iso']:.4f} warm={results['B1_weather_C_warmstart']['AUROC_iso']:.4f}")
print(f"  4. B3: temporal={results['B3_combined_A_temporal']['AUROC_iso']:.4f} cold={results['B3_combined_B_coldstart']['AUROC_iso']:.4f} warm={results['B3_combined_C_warmstart']['AUROC_iso']:.4f}")
print(f"\n✓ Saved: {OUT}/baselines_v87.json")
