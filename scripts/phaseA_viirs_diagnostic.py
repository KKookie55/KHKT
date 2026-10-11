#!/usr/bin/env python3
"""PHASE A v2 — Use VIIRS labels (the FAILURE case) for A2 + A3."""
import os, sys, json, time, warnings, random, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except: pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import xgboost as xgb

OUT = Path("/home/z/my-project/download/real_data_v85")
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
splits_npz = np.load(OUT / "splits_v85.npz")
features = json.loads((OUT/"panel_v85_meta.json").read_text())["features"]

print("="*70); print("PHASE A v2 — VIIRS LABELS (FAILURE CASE)"); print("="*70)

# Load VIIRS panel
panel = pd.read_parquet(OUT / "panel_v85_viirs.parquet")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
print(f"  Panel: {len(panel):,} rows (VIIRS labels)")

# Get splits
north_train_idx = splits_npz["Spatial_OOD_NorthToSouth_train"]
north_val_idx = splits_npz["Spatial_OOD_NorthToSouth_val"]
south_test_idx = splits_npz["Spatial_OOD_NorthToSouth_test"]

# Subsets
tr = panel.iloc[north_train_idx].dropna(subset=["label_h1"])
va = panel.iloc[north_val_idx].dropna(subset=["label_h1"])
te = panel.iloc[south_test_idx].dropna(subset=["label_h1"])
print(f"  North train: {len(tr):,} rows, pos rate={tr['label_h1'].mean():.4f}")
print(f"  South test:  {len(te):,} rows, pos rate={te['label_h1'].mean():.4f}")

# Apply drought normalization
di_idx = features.index("precip_roll30_sum")
di_mean = np.nanmean(tr[features[di_idx]].values)
di_std = np.nanstd(tr[features[di_idx]].values) + 1e-6

def prep_X(df, di_idx, di_mean, di_std, add_features=None):
    X = df[features].values.astype(np.float32).copy()
    X[:, di_idx] = np.nan_to_num((X[:, di_idx] - di_mean) / di_std, nan=0.0)
    if add_features is not None:
        X = np.hstack([X, add_features(df).astype(np.float32).values.reshape(-1, 1)])
    return X

X_tr_v1 = prep_X(tr, di_idx, di_mean, di_std)
X_val_v1 = prep_X(va, di_idx, di_mean, di_std)
X_test_v1 = prep_X(te, di_idx, di_mean, di_std)
y_tr = tr["label_h1"].values.astype(np.float32)
y_val = va["label_h1"].values.astype(np.float32)
y_test = te["label_h1"].values.astype(np.float32)

imp = SimpleImputer(strategy="median")
X_tr_v1 = imp.fit_transform(X_tr_v1); X_val_v1 = imp.transform(X_val_v1); X_test_v1 = imp.transform(X_test_v1)

# Train XGB v1 (weather only) on North VIIRS labels
pos = max(int(y_tr.sum()), 1); neg = max(int((1-y_tr).sum()), 1)
spw = max(math.sqrt(neg/pos), 1.0)
m_v1 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m_v1.fit(X_tr_v1, y_tr, eval_set=[(X_val_v1, y_val)], verbose=False)
p_v1 = m_v1.predict_proba(X_test_v1)[:, 1]
auroc_v1 = roc_auc_score(y_test, p_v1)
print(f"\n  v1 (weather only) South VIIRS AUROC = {auroc_v1:.4f}")

# ============================================================
# A2: PREVALENCE-CONTROLLED N→S (VIIRS)
# ============================================================
print("\n[A2] Prevalence-balanced bootstrap (VIIRS labels)...")
north_prevalence = y_tr.mean()
n_pos_test = int(y_test.sum())
n_neg_test = int((1-y_test).sum())
# Target negatives: keep all positives + sample negatives to match North prevalence
# South pos = 0.089, North pos = 0.052 → to balance to 0.052, sample 0.052/(1-0.052) ≈ 0.0548 × n_pos = neg count
target_n_neg = int(n_pos_test * (1 - north_prevalence) / north_prevalence)
target_n_neg = min(target_n_neg, n_neg_test)  # can't sample more than available
print(f"  N prevalence={north_prevalence:.4f}, S pos={n_pos_test}, target neg={target_n_neg}, S neg available={n_neg_test}")

# Note: if target_n_neg = n_neg_test, prevalence balancing can't help (we already use all negatives)
# In that case, we can OVERSAMPLE North positives instead
# Actually, since S pos rate (0.089) > N pos rate (0.052), and we use ALL S positives,
# the balanced South prevalence will still be higher than 0.052 if target_n_neg < n_neg_test
# Let me also test: subsample South POSITIVES instead

# Method 1: Subsample South negatives to balance
auroc_balanced_neg = []
n_bootstrap = 200
neg_idx = np.where(y_test == 0)[0]
pos_idx = np.where(y_test == 1)[0]
for b in range(n_bootstrap):
    rng = np.random.RandomState(b)
    if target_n_neg < len(neg_idx):
        sampled_neg = rng.choice(neg_idx, target_n_neg, replace=False)
    else:
        sampled_neg = neg_idx
    keep_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep_idx)
    y_b = y_test[keep_idx]; p_b = p_v1[keep_idx]
    try: auroc_balanced_neg.append(roc_auc_score(y_b, p_b))
    except: pass
auroc_balanced_neg = np.array(auroc_balanced_neg)
print(f"  Method 1 (subsample negatives): AUROC={auroc_balanced_neg.mean():.4f}±{auroc_balanced_neg.std():.4f}")
print(f"    95% CI: [{np.percentile(auroc_balanced_neg, 2.5):.4f}, {np.percentile(auroc_balanced_neg, 97.5):.4f}]")

# Method 2: Compute AUROC within South positive cells only (case-control)
# This is the proper "prevalence-controlled" test — measure discrimination within positives' neighborhoods
print("\n  Method 2: stratified sampling (equal pos/neg) — case-control style")
auroc_balanced_cc = []
target_neg_cc = n_pos_test  # 1:1 ratio
for b in range(n_bootstrap):
    rng = np.random.RandomState(b)
    if target_neg_cc < len(neg_idx):
        sampled_neg = rng.choice(neg_idx, target_neg_cc, replace=False)
    else:
        sampled_neg = rng.choice(neg_idx, target_neg_cc, replace=True)
    keep_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep_idx)
    y_b = y_test[keep_idx]; p_b = p_v1[keep_idx]
    try: auroc_balanced_cc.append(roc_auc_score(y_b, p_b))
    except: pass
auroc_balanced_cc = np.array(auroc_balanced_cc)
print(f"  Method 2 (1:1 case-control): AUROC={auroc_balanced_cc.mean():.4f}±{auroc_balanced_cc.std():.4f}")
print(f"    95% CI: [{np.percentile(auroc_balanced_cc, 2.5):.4f}, {np.percentile(auroc_balanced_cc, 97.5):.4f}]")

# ============================================================
# A3: SPATIAL PROXY ABLATION (VIIRS)
# ============================================================
print("\n[A3] Spatial proxy ablation (VIIRS labels)...")
# v2: weather + raw lat/lon
X_tr_v2 = np.hstack([X_tr_v1, tr[["lat"]].values.astype(np.float32), tr[["lon"]].values.astype(np.float32)])
X_val_v2 = np.hstack([X_val_v1, va[["lat"]].values.astype(np.float32), va[["lon"]].values.astype(np.float32)])
X_test_v2 = np.hstack([X_test_v1, te[["lat"]].values.astype(np.float32), te[["lon"]].values.astype(np.float32)])
m_v2 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m_v2.fit(X_tr_v2, y_tr, eval_set=[(X_val_v2, y_val)], verbose=False)
p_v2 = m_v2.predict_proba(X_test_v2)[:, 1]
auroc_v2 = roc_auc_score(y_test, p_v2)
print(f"  v2 (weather + lat/lon) AUROC = {auroc_v2:.4f} (Δ={auroc_v2-auroc_v1:+.4f})")

# v3: weather + coarse lat_band (10 bands)
tr_lb = pd.cut(tr["lat"], bins=10, labels=False).fillna(5).astype(np.float32).values.reshape(-1, 1)
va_lb = pd.cut(va["lat"], bins=10, labels=False).fillna(5).astype(np.float32).values.reshape(-1, 1)
te_lb = pd.cut(te["lat"], bins=10, labels=False).fillna(5).astype(np.float32).values.reshape(-1, 1)
X_tr_v3 = np.hstack([X_tr_v1, tr_lb])
X_val_v3 = np.hstack([X_val_v1, va_lb])
X_test_v3 = np.hstack([X_test_v1, te_lb])
m_v3 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m_v3.fit(X_tr_v3, y_tr, eval_set=[(X_val_v3, y_val)], verbose=False)
p_v3 = m_v3.predict_proba(X_test_v3)[:, 1]
auroc_v3 = roc_auc_score(y_test, p_v3)
print(f"  v3 (weather + lat_band) AUROC = {auroc_v3:.4f} (Δ={auroc_v3-auroc_v1:+.4f})")

# v4: weather + lat_band + lon_band (more spatial context)
tr_lb2 = pd.cut(tr["lat"], bins=5, labels=False).fillna(2).astype(np.float32).values.reshape(-1, 1)
tr_lbo2 = pd.cut(tr["lon"], bins=5, labels=False).fillna(2).astype(np.float32).values.reshape(-1, 1)
va_lb2 = pd.cut(va["lat"], bins=5, labels=False).fillna(2).astype(np.float32).values.reshape(-1, 1)
va_lbo2 = pd.cut(va["lon"], bins=5, labels=False).fillna(2).astype(np.float32).values.reshape(-1, 1)
te_lb2 = pd.cut(te["lat"], bins=5, labels=False).fillna(2).astype(np.float32).values.reshape(-1, 1)
te_lbo2 = pd.cut(te["lon"], bins=5, labels=False).fillna(2).astype(np.float32).values.reshape(-1, 1)
X_tr_v4 = np.hstack([X_tr_v1, tr_lb2, tr_lbo2])
X_val_v4 = np.hstack([X_val_v1, va_lb2, va_lbo2])
X_test_v4 = np.hstack([X_test_v1, te_lb2, te_lbo2])
m_v4 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m_v4.fit(X_tr_v4, y_tr, eval_set=[(X_val_v4, y_val)], verbose=False)
p_v4 = m_v4.predict_proba(X_test_v4)[:, 1]
auroc_v4 = roc_auc_score(y_test, p_v4)
print(f"  v4 (weather + lat_band5 + lon_band5) AUROC = {auroc_v4:.4f} (Δ={auroc_v4-auroc_v1:+.4f})")

# Save results
results = {
    "label_source": "viirs",
    "A2_prevalence_controlled": {
        "north_prevalence": float(north_prevalence),
        "south_prevalence": float(y_test.mean()),
        "n_pos_test": n_pos_test, "n_neg_test": n_neg_test,
        "auroc_full_test": float(auroc_v1),
        "method1_subsample_neg": {
            "auroc_mean": float(auroc_balanced_neg.mean()),
            "auroc_std": float(auroc_balanced_neg.std()),
            "ci_95": [float(np.percentile(auroc_balanced_neg, 2.5)),
                      float(np.percentile(auroc_balanced_neg, 97.5))],
        },
        "method2_case_control_1to1": {
            "auroc_mean": float(auroc_balanced_cc.mean()),
            "auroc_std": float(auroc_balanced_cc.std()),
            "ci_95": [float(np.percentile(auroc_balanced_cc, 2.5)),
                      float(np.percentile(auroc_balanced_cc, 97.5))],
        },
    },
    "A3_spatial_proxy_ablation": {
        "v1_weather_only": {"auroc": float(auroc_v1)},
        "v2_weather_lat_lon": {"auroc": float(auroc_v2), "delta": float(auroc_v2-auroc_v1)},
        "v3_weather_lat_band_10": {"auroc": float(auroc_v3), "delta": float(auroc_v3-auroc_v1)},
        "v4_weather_lat_lon_band_5": {"auroc": float(auroc_v4), "delta": float(auroc_v4-auroc_v1)},
    },
}
with open(OUT / "phaseA_viirs_diagnosis.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Saved: {OUT}/phaseA_viirs_diagnosis.json")

# Final figure: combined A2 + A3
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# A2 panel
ax = axes[0]
methods = ["Original\nSouth", "Subsample\nneg (M1)", "1:1 case-\ncontrol (M2)"]
aurocs_a2 = [auroc_v1, auroc_balanced_neg.mean(), auroc_balanced_cc.mean()]
errs_a2 = [0, auroc_balanced_neg.std(), auroc_balanced_cc.std()]
bars = ax.bar(methods, aurocs_a2, yerr=errs_a2, capsize=5,
              color=["#e7298a","#7570b3","#1b9e77"], width=0.5)
for b, a in zip(bars, aurocs_a2):
    ax.text(b.get_x()+b.get_width()/2, a + 0.005, f"{a:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("AUROC on South VIIRS test")
ax.set_title("A2: Prevalence-controlled N→S (VIIRS labels)\nDoes balancing base rate recover performance?")
ax.set_ylim(0.5, 0.7); ax.grid(axis="y", alpha=0.3)
ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label="random")
ax.legend()

# A3 panel
ax = axes[1]
labels_a3 = ["v1: weather\nonly", "v2: +lat/lon", "v3: +lat_band\n(10)", "v4: +lat+lon\nbands (5)"]
aurocs_a3 = [auroc_v1, auroc_v2, auroc_v3, auroc_v4]
bars = ax.bar(labels_a3, aurocs_a3, color=["#e7298a","#7570b3","#1b9e77","#d95f02"], width=0.5)
for b, a in zip(bars, aurocs_a3):
    ax.text(b.get_x()+b.get_width()/2, a + 0.005, f"{a:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("AUROC on South VIIRS test")
ax.set_title("A3: Spatial proxy ablation (VIIRS labels)\nDoes adding spatial features reduce N→S gap?")
ax.set_ylim(0.5, 0.7); ax.grid(axis="y", alpha=0.3)
ax.axhline(auroc_v1, color='red', linestyle='--', alpha=0.5, label=f"baseline {auroc_v1:.4f}")
ax.legend()

plt.suptitle("Phase A v2 — VIIRS labels (the failure case)", fontsize=12, y=1.02)
plt.savefig(FIG/"fig_phaseA_viirs_combined.png", dpi=110); plt.close()
print(f"✓ Figure: {FIG}/fig_phaseA_viirs_combined.png")

# Final verdict
print("\n" + "="*70)
print("PHASE A v2 — VERDICT (VIIRS labels, the failure case)")
print("="*70)
print(f"\nA2: Prevalence-controlled N→S evaluation (VIIRS)")
print(f"  Original South AUROC:                  {auroc_v1:.4f}")
print(f"  Subsample negatives (M1):             {auroc_balanced_neg.mean():.4f} ± {auroc_balanced_neg.std():.4f}")
print(f"  Case-control 1:1 (M2):                 {auroc_balanced_cc.mean():.4f} ± {auroc_balanced_cc.std():.4f}")
delta_m1 = auroc_balanced_neg.mean() - auroc_v1
delta_m2 = auroc_balanced_cc.mean() - auroc_v1
if abs(delta_m2) < 0.02:
    print(f"  → Prevalence balancing does NOT help (Δ M2 = {delta_m2:+.4f})")
    print(f"  → The N→S failure is NOT primarily a base-rate mismatch problem")
    print(f"  → The MODEL itself fails to generalize, regardless of label balance")
else:
    print(f"  → Prevalence balancing {'helps' if delta_m2 > 0 else 'hurts'} by {delta_m2:+.4f}")
    print(f"  → Some of the gap is base-rate related")

print(f"\nA3: Spatial proxy ablation (VIIRS)")
print(f"  v1 (weather only):                     AUROC = {auroc_v1:.4f}")
print(f"  v2 (weather + lat/lon):                AUROC = {auroc_v2:.4f}  (Δ = {auroc_v2-auroc_v1:+.4f})")
print(f"  v3 (weather + lat_band 10):             AUROC = {auroc_v3:.4f}  (Δ = {auroc_v3-auroc_v1:+.4f})")
print(f"  v4 (weather + lat_band5 + lon_band5):   AUROC = {auroc_v4:.4f}  (Δ = {auroc_v4-auroc_v1:+.4f})")
max_delta = max(auroc_v2-auroc_v1, auroc_v3-auroc_v1, auroc_v4-auroc_v1)
if max_delta > 0.05:
    print(f"  → Spatial proxy HELPS (+{max_delta:.4f}) → hidden spatial structure IS recoverable from lat/lon")
    print(f"  → But the gap from baseline {auroc_v1:.4f} → IID performance 0.84 is huge, need real spatial features (Phase B)")
elif max_delta > 0.01:
    print(f"  → Spatial proxy slightly helps (+{max_delta:.4f})")
else:
    print(f"  → Spatial proxy does NOT help (max Δ = {max_delta:.4f})")
    print(f"  → Hidden structure is NOT captured by lat/lon alone — Phase B (real spatial data) is essential")
