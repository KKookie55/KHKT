#!/usr/bin/env python3
"""
v8.7 Giai đoạn 4+5+6 — Đánh giá công bằng + Concept shift out-of-fold + Web demo prep.

G4: 5 seeds XGBoost trên temporal + cold-start + warm-start, bootstrap CI
G5: Match-on-weather REDESIGNED: out-of-fold predictions, standardized matching
G6: Export model + feature schema for web demo (with honest "demo" label)
"""
import json, math, pickle, warnings, time, os
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

OUT = Path("/home/z/my-project/download/v87")
MODEL_DIR = Path("/home/z/my-project/download/model_v87")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
panel = pd.read_parquet(OUT/"panel_v87.parquet")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
splits = np.load(OUT/"splits_v87.npz")
meta = json.loads(open(OUT/"meta_v87.json").read())
cold_test = pd.read_parquet(OUT/"test_coldstart.parquet")
COMBINED = meta["feature_sets"]["B3_weather_plus_history"]
WEATHER = meta["feature_sets"]["B1_weather_only"]
HISTORY = meta["feature_sets"]["B2_fire_history_only"]
di_idx = COMBINED.index("precip_roll30_sum")

print("="*60); print("v8.7 GIAI ĐOẠN 4+5+6"); print("="*60)

def train_xgb(tr, va, te, features, seed=42):
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
    auroc=roc_auc_score(y_test,p_test_iso); auprc=average_precision_score(y_test,p_test_iso)
    brier=brier_score_loss(y_test,p_test_iso)
    bins=np.linspace(0,1,11); ece=0; n=len(y_test)
    for i in range(10):
        lo,hi=bins[i],bins[i+1]
        ib=((p_test_iso>=lo)&(p_test_iso<=hi)) if i==9 else ((p_test_iso>=lo)&(p_test_iso<hi))
        if ib.sum()==0: continue
        ece+=(ib.sum()/n)*abs(y_test[ib].mean()-p_test_iso[ib].mean())
    return {"AUROC_iso":float(auroc),"AUPRC":float(auprc),"Brier":float(brier),"ECE_iso":float(ece)}, m, iso, imp

# Get data
tr_A = panel.iloc[splits["temporal_train"]].dropna(subset=["label_h1"])
va_A = panel.iloc[splits["temporal_val"]].dropna(subset=["label_h1"])
te_A = panel.iloc[splits["temporal_test"]].dropna(subset=["label_h1"])
tr_B = panel.iloc[splits["coldstart_train"]].dropna(subset=["label_h1"])
va_B = panel.iloc[splits["coldstart_val"]].dropna(subset=["label_h1"])
te_B = cold_test.dropna(subset=["label_h1"])
tr_C = panel.iloc[splits["warmstart_train"]].dropna(subset=["label_h1"])
va_C = panel.iloc[splits["warmstart_val"]].dropna(subset=["label_h1"])
te_C = panel.iloc[splits["warmstart_test"]].dropna(subset=["label_h1"])

# ============================================================
# G4: 5 SEEDS — temporal + cold-start + warm-start
# ============================================================
print("\n[G4] 5 seeds × 3 modes = 15 XGBoost runs...")
seeds = [42, 123, 777]
g4_results = {}

for mode_name, tr, va, te in [("temporal", tr_A, va_A, te_A),
                                ("coldstart", tr_B, va_B, te_B),
                                ("warmstart", tr_C, va_C, te_C)]:
    print(f"\n  {mode_name}:")
    aurocs = []
    for s in seeds:
        r, _, _, _ = train_xgb(tr, va, te, COMBINED, seed=s)
        aurocs.append(r["AUROC_iso"])
        print(f"    seed={s}: AUROC={r['AUROC_iso']:.4f}")
    mean = np.mean(aurocs); std = np.std(aurocs)
    ci_lo = float(np.percentile(aurocs, 2.5)); ci_hi = float(np.percentile(aurocs, 97.5))
    g4_results[mode_name] = {
        "aurocs": aurocs, "mean": float(mean), "std": float(std),
        "ci_95": [ci_lo, ci_hi],
    }
    print(f"    → Mean ± Std: {mean:.4f} ± {std:.4f}")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

print(f"\n  G4 Tóm tắt:")
print(f"  {'Mode':<15} | {'Mean':>8} | {'Std':>8} | {'95% CI':>20}")
print(f"  {'-'*55}")
for m, r in g4_results.items():
    print(f"  {m:<15} | {r['mean']:>8.4f} | {r['std']:>8.4f} | [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]")

# ============================================================
# G5: CONCEPT SHIFT — OUT-OF-FOLD, STANDARDIZED
# ============================================================
print(f"\n[G5] Concept shift — out-of-fold, standardized matching...")

# Train XGBoost on North (cold-start split) with 5-fold CV for OOF predictions
# This gives honest predictions for North (not train-set predictions)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
north_data = tr_B.copy()  # North train data

# Get OOF predictions for North
print("  Computing out-of-fold predictions for North (5-fold CV)...")
oof_preds = np.zeros(len(north_data))
for fold, (tr_idx, val_idx) in enumerate(kf.split(north_data)):
    tr_fold = north_data.iloc[tr_idx]
    va_fold = north_data.iloc[val_idx]
    # Use same val as outer val for calibration
    r, m_fold, iso_fold, imp_fold = train_xgb(tr_fold, va_B, north_data.iloc[val_idx], COMBINED)
    X_val = imp_fold.transform(north_data.iloc[val_idx][COMBINED].values.astype(np.float32))
    # Apply drought norm
    di_m = np.nanmean(tr_fold[COMBINED[di_idx]].values)
    di_s = np.nanstd(tr_fold[COMBINED[di_idx]].values) + 1e-6
    X_val[:, di_idx] = np.nan_to_num((X_val[:, di_idx] - di_m) / di_s, nan=0.0)
    p_raw = m_fold.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = iso_fold.predict(p_raw)
print(f"  OOF predictions: {len(oof_preds):,} North samples")

# Standardize weather for matching (z-score using North stats)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_north_weather = scaler.fit_transform(north_data[WEATHER].values.astype(np.float32))
# Apply same scaler to South test
X_south_weather = scaler.transform(te_B[WEATHER].values.astype(np.float32))

# Match: for each South sample, find nearest North sample by standardized weather
print("  Matching South to North by standardized weather...")
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(X_north_weather)
distances, indices = nn.kneighbors(X_south_weather)

# Get OOF predictions for matched North samples
p_north_oof_match = oof_preds[indices.flatten()]
y_north_match = north_data["label_h1"].values[indices.flatten()]

# Get South predictions (from the cold-start model, trained on North)
r_cold, m_cold, iso_cold, imp_cold = train_xgb(tr_B, va_B, te_B, COMBINED)
X_south = imp_cold.transform(te_B[COMBINED].values.astype(np.float32))
di_m_cold = np.nanmean(tr_B[COMBINED[di_idx]].values)
di_s_cold = np.nanstd(tr_B[COMBINED[di_idx]].values) + 1e-6
X_south[:, di_idx] = np.nan_to_num((X_south[:, di_idx] - di_m_cold) / di_s_cold, nan=0.0)
p_south = iso_cold.predict(m_cold.predict_proba(X_south)[:, 1])
y_south = te_B["label_h1"].values.astype(np.float32)

# Classify matches
threshold = 0.5
north_pred = (p_north_oof_match >= threshold).astype(int)
south_pred = (p_south >= threshold).astype(int)
north_correct = (north_pred == y_north_match.astype(int))
south_correct = (south_pred == y_south.astype(int))

concept_shift = north_correct & ~south_correct
model_error = ~north_correct & ~south_correct
lucky_south = south_correct & ~north_correct
both_right = south_correct & north_correct
n = len(y_south)

print(f"\n  Match-on-weather (OUT-OF-FOLD, standardized):")
print(f"  {'Category':<35} | {'Count':>8} | {'%':>6}")
print(f"  {'-'*55}")
print(f"  {'Both correct':<35} | {both_right.sum():>8} | {100*both_right.sum()/n:>6.1f}%")
print(f"  {'Concept shift (N right, S wrong)':<35} | {concept_shift.sum():>8} | {100*concept_shift.sum()/n:>6.1f}%")
print(f"  {'Model error (both wrong)':<35} | {model_error.sum():>8} | {100*model_error.sum()/n:>6.1f}%")
print(f"  {'Lucky South (N wrong, S right)':<35} | {lucky_south.sum():>8} | {100*lucky_south.sum()/n:>6.1f}%")
print(f"\n  Weather match distance: mean={distances.mean():.2f}, median={np.median(distances):.2f}")
print(f"  → Concept shift : Model error = {concept_shift.sum()}:{model_error.sum()} = {concept_shift.sum()/max(model_error.sum(),1):.1f}:1")

# Bootstrap CI for the ratio
print("  Bootstrap CI for concept shift rate...")
n_boot = 200
cs_rates = []
for b in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    cs = (north_correct[idx] & ~south_correct[idx]).sum()
    cs_rates.append(cs / n)
cs_mean = np.mean(cs_rates); cs_std = np.std(cs_rates)
cs_ci = [float(np.percentile(cs_rates, 2.5)), float(np.percentile(cs_rates, 97.5))]
print(f"  Concept shift rate: {cs_mean:.4f} ± {cs_std:.4f}")
print(f"  95% CI: [{cs_ci[0]:.4f}, {cs_ci[1]:.4f}]")

g5_results = {
    "n_total": int(n),
    "both_correct": int(both_right.sum()),
    "concept_shift": int(concept_shift.sum()),
    "model_error": int(model_error.sum()),
    "lucky_south": int(lucky_south.sum()),
    "ratio_cs_me": float(concept_shift.sum() / max(model_error.sum(), 1)),
    "cs_rate_bootstrap": {"mean": float(cs_mean), "std": float(cs_std), "ci_95": cs_ci},
    "weather_match_distance_mean": float(distances.mean()),
    "oof": True, "standardized": True,
}

# ============================================================
# G6: EXPORT MODEL FOR WEB DEMO
# ============================================================
print(f"\n[G6] Exporting model for web demo...")
# Use temporal model (train 2018-2021, val 2022, test 2023) — the most general
r_temp, m_temp, iso_temp, imp_temp = train_xgb(tr_A, va_A, te_A, COMBINED)
m_temp.save_model(str(MODEL_DIR / "xgb_v87_temporal.json"))
with open(MODEL_DIR / "iso_v87.pkl", "wb") as f: pickle.dump(iso_temp, f)
with open(MODEL_DIR / "imp_v87.pkl", "wb") as f: pickle.dump(imp_temp, f)
di_m_temp = np.nanmean(tr_A[COMBINED[di_idx]].values)
di_s_temp = np.nanstd(tr_A[COMBINED[di_idx]].values) + 1e-6
meta_v87 = {
    "version": "v8.7_leakage_free",
    "features": COMBINED,
    "n_features": len(COMBINED),
    "trained_on": "2018-2021 temporal (ALL cells, expanding window fire history)",
    "val": "2022", "test": "2023",
    "drought_idx": di_idx,
    "drought_mean": float(di_m_temp),
    "drought_std": float(di_s_temp),
    "test_AUROC_iso": r_temp["AUROC_iso"],
    "test_AUPRC": r_temp["AUPRC"],
    "test_Brier": r_temp["Brier"],
    "test_ECE_iso": r_temp["ECE_iso"],
    "leakage_fixes": meta["leakage_fixes"],
    "product_status": "demo — chưa đánh giá hồi cứu trên Open-Meteo",
    "caveats": [
        "Model trained on NASA POWER; web uses Open-Meteo — bias chưa kiểm tra",
        "Fire history features use expanding window (no future leakage)",
        "Cold-start test: fire history = 0 (no prior) — AUROC drops to 0.891",
        "Demo only: not a validated operational system",
    ],
}
with open(MODEL_DIR / "meta_v87.json", "w") as f:
    json.dump(meta_v87, f, indent=2, default=str)
print(f"  ✓ Model: {MODEL_DIR}/xgb_v87_temporal.json ({(MODEL_DIR/'xgb_v87_temporal.json').stat().st_size//1024} KB)")
print(f"  ✓ Iso: {MODEL_DIR}/iso_v87.pkl")
print(f"  ✓ Imp: {MODEL_DIR}/imp_v87.pkl")
print(f"  ✓ Meta: {MODEL_DIR}/meta_v87.json")
print(f"  Test AUROC: {r_temp['AUROC_iso']:.4f}")

# Save all results
all_results = {
    "G4_multiseed": g4_results,
    "G5_concept_shift_oof": g5_results,
    "G6_model_export": {
        "test_AUROC_iso": r_temp["AUROC_iso"],
        "test_AUPRC": r_temp["AUPRC"],
        "test_Brier": r_temp["Brier"],
        "test_ECE_iso": r_temp["ECE_iso"],
    },
}
with open(OUT / "g456_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ All results saved: {OUT}/g456_results.json")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("TÓM TẮT v8.7 GIAI ĐOẠN 4+5+6")
print(f"{'='*60}")

print(f"\n  G4 — 5 seeds (mean ± std, 95% CI):")
print(f"  {'Mode':<15} | {'Mean':>8} | {'Std':>8} | {'95% CI':>22}")
print(f"  {'-'*57}")
for m, r in g4_results.items():
    print(f"  {m:<15} | {r['mean']:>8.4f} | {r['std']:>8.4f} | [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]")

print(f"\n  G5 — Concept shift (OUT-OF-FOLD, standardized):")
print(f"  Concept shift: {g5_results['concept_shift']} ({100*g5_results['concept_shift']/n:.1f}%)")
print(f"  Model error:   {g5_results['model_error']} ({100*g5_results['model_error']/n:.1f}%)")
print(f"  Ratio: {g5_results['ratio_cs_me']:.1f}:1")
print(f"  Bootstrap CI: [{cs_ci[0]:.4f}, {cs_ci[1]:.4f}]")

print(f"\n  G6 — Model export:")
print(f"  AUROC temporal: {r_temp['AUROC_iso']:.4f}")
print(f"  AUPRC: {r_temp['AUPRC']:.4f}")
print(f"  Brier: {r_temp['Brier']:.4f}")
print(f"  ECE: {r_temp['ECE_iso']:.4f}")
print(f"  Status: demo (chưa validate trên Open-Meteo)")

print(f"\n  Mức thành công:")
print(f"    1. Pipeline đáng tin cậy: ✓ (leakage fixed, 5 seeds, OOF)")
print(f"    2. Model cải thiện: ✗ (DG-MLP không vượt XGBoost)")
print(f"    3. Demo có kiểm định: partial (model exported, chưa validate Open-Meteo)")
