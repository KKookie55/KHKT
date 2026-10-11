#!/usr/bin/env python3
"""
PHASE A — Spatial diagnostic without downloading new data.

A1. Spatial autocorrelation:
   - Moran's I (global)
   - Distance-decay curve
   - Fire-conditioned neighbor probability (P(neighbor fire | this cell fire) vs random P(fire))

A2. Prevalence-controlled N→S:
   - Subsample South negatives to match North fire prevalence
   - Re-evaluate MODIS-trained model on balanced South
   - Bootstrap CI

A3. Spatial proxy ablation:
   - v1: weather only (current, no lat/lon)
   - v2: weather + raw lat/lon
   - v3: weather + coarse lat_band (binned)
   - Train XGB on North, evaluate on South
   - Compare AUROC — does adding spatial proxy reduce the gap?
"""
import os, sys, json, time, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
warnings.filterwarnings("ignore")
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except: pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import xgboost as xgb
from sklearn.impute import SimpleImputer
import math

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/real_data_v85")
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)

print("="*70); print("PHASE A — SPATIAL DIAGNOSTIC (no new data)"); print("="*70)

# Load MODIS panel
panel = pd.read_parquet(OUT / "panel_v85.parquet")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
splits_npz = np.load(OUT / "splits_v85.npz")

# Get unique cells
cells = panel[["lat","lon"]].drop_duplicates().reset_index(drop=True)
cells["cell_id"] = np.arange(len(cells))
panel["cell_id"] = panel.merge(cells, on=["lat","lon"], how="left")["cell_id"]
print(f"  Panel: {len(panel):,} rows, {len(cells)} cells")

# Build cell coordinate array
cell_lats = cells["lat"].values
cell_lons = cells["lon"].values

# Compute pairwise distances (in km, approximate)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Compute distance matrix (273 cells × 273 cells = 74529 pairs, fast)
print("\n[A1] Computing cell-to-cell distances...")
dist_matrix = np.zeros((len(cells), len(cells)))
for i in range(len(cells)):
    dist_matrix[i] = haversine_km(cell_lats[i], cell_lons[i], cell_lats, cell_lons)
print(f"  Distance matrix: {dist_matrix.shape}, max={dist_matrix.max():.0f} km, median={np.median(dist_matrix[dist_matrix>0]):.0f} km")

results = {}

# ============================================================
# A1: SPATIAL AUTOCORRELATION
# ============================================================
print("\n[A1] SPATIAL AUTOCORRELATION")

# A1a: Moran's I (per-date, then average) using k-nearest neighbors
print("  Computing Moran's I (k=8 neighbors, per-date)...")
K_NEIGHBORS = 8
# Find k nearest neighbors for each cell (excluding self)
nn_indices = np.argsort(dist_matrix, axis=1)[:, 1:K_NEIGHBORS+1]  # exclude self (index 0)

# Sample dates (full compute would be 2191 dates × 273 cells = too slow for O(n²))
sample_dates = panel["date"].drop_duplicates().sample(n=50, random_state=42)
morans_I_per_date = []
for d in sample_dates:
    day_data = panel[panel["date"] == d][["cell_id","fire"]].set_index("cell_id")["fire"]
    if len(day_data) < 50: continue
    x = day_data.reindex(range(len(cells))).fillna(0).values
    x_mean = x.mean()
    # Spatial weights: 1/k for each of k nearest neighbors
    W_total = K_NEIGHBORS * len(cells)
    # Moran's I numerator: sum_i sum_j w_ij (x_i - x_mean)(x_j - x_mean)
    num = 0.0
    for i in range(len(cells)):
        for j in nn_indices[i]:
            num += (x[i] - x_mean) * (x[j] - x_mean)
    num = (len(cells) / W_total) * num
    # Denominator: sum (x_i - x_mean)^2
    den = np.sum((x - x_mean)**2)
    if den == 0:
        morans_I_per_date.append(0)
    else:
        morans_I_per_date.append(num / den)

morans_I_mean = float(np.mean(morans_I_per_date))
morans_I_std = float(np.std(morans_I_per_date))
print(f"  Moran's I: {morans_I_mean:.4f} ± {morans_I_std:.4f} (50 random dates)")
print(f"  → I=0 random, I=1 perfect clustering, I=-1 perfect dispersion")
print(f"  → {'STRONG spatial clustering' if morans_I_mean > 0.3 else 'Moderate' if morans_I_mean > 0.1 else 'Weak'} clustering")

# A1b: Distance-decay curve
print("\n  Computing distance-decay curve (label agreement vs distance)...")
dist_bins = np.linspace(0, 800, 17)  # 0 to 800 km, 50 km bins
bin_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
agreement_per_bin = []
both_fire_per_bin = []
both_nofire_per_bin = []
n_pairs_per_bin = []
for d in sample_dates:
    day_data = panel[panel["date"] == d][["cell_id","fire"]].set_index("cell_id")["fire"]
    if len(day_data) < 50: continue
    x = day_data.reindex(range(len(cells))).fillna(0).values
    for i in range(len(cells)):
        for j in range(i+1, len(cells)):
            d_ij = dist_matrix[i,j]
            for b in range(len(dist_bins)-1):
                if dist_bins[b] <= d_ij < dist_bins[b+1]:
                    n_pairs_per_bin.append(b)
                    agreement_per_bin.append(int(x[i] == x[j]))
                    both_fire_per_bin.append(int(x[i] == 1 and x[j] == 1))
                    both_nofire_per_bin.append(int(x[i] == 0 and x[j] == 0))
                    break

n_pairs_per_bin = np.array(n_pairs_per_bin)
agreement_per_bin = np.array(agreement_per_bin)
both_fire_per_bin = np.array(both_fire_per_bin)
mean_agreement = []
mean_both_fire = []
for b in range(len(dist_bins)-1):
    mask = n_pairs_per_bin == b
    if mask.sum() > 0:
        mean_agreement.append(agreement_per_bin[mask].mean())
        mean_both_fire.append(both_fire_per_bin[mask].mean())
    else:
        mean_agreement.append(np.nan); mean_both_fire.append(np.nan)
mean_agreement = np.array(mean_agreement); mean_both_fire = np.array(mean_both_fire)

# A1c: Fire-conditioned neighbor probability
print("\n  Computing fire-conditioned neighbor probability...")
dist_bins_short = np.linspace(0, 200, 11)  # 0 to 200 km, 20 km bins
bin_centers_short = 0.5 * (dist_bins_short[:-1] + dist_bins_short[1:])
neighbor_fire_given_fire = []
random_fire_rate = []
n_pairs_short = []
for d in sample_dates:
    day_data = panel[panel["date"] == d][["cell_id","fire"]].set_index("cell_id")["fire"]
    if len(day_data) < 50: continue
    x = day_data.reindex(range(len(cells))).fillna(0).values
    fire_rate = x.mean()
    for i in range(len(cells)):
        if x[i] != 1: continue  # only consider fire cells
        for j in range(len(cells)):
            if i == j: continue
            d_ij = dist_matrix[i,j]
            for b in range(len(dist_bins_short)-1):
                if dist_bins_short[b] <= d_ij < dist_bins_short[b+1]:
                    n_pairs_short.append(b)
                    neighbor_fire_given_fire.append(x[j])
                    random_fire_rate.append(fire_rate)
                    break

n_pairs_short = np.array(n_pairs_short)
neighbor_fire_given_fire = np.array(neighbor_fire_given_fire)
random_fire_rate = np.array(random_fire_rate)
mean_conditional = []
mean_random = []
for b in range(len(dist_bins_short)-1):
    mask = n_pairs_short == b
    if mask.sum() > 0:
        mean_conditional.append(neighbor_fire_given_fire[mask].mean())
        mean_random.append(random_fire_rate[mask].mean())
    else:
        mean_conditional.append(np.nan); mean_random.append(np.nan)
mean_conditional = np.array(mean_conditional)
mean_random = np.array(mean_random)

results["A1_spatial_autocorr"] = {
    "morans_I_mean": morans_I_mean, "morans_I_std": morans_I_std,
    "distance_bins_km": bin_centers.tolist(),
    "label_agreement_per_bin": [float(x) for x in mean_agreement if not np.isnan(x)],
    "conditional_bins_km": bin_centers_short.tolist(),
    "neighbor_fire_given_fire": [float(x) for x in mean_conditional if not np.isnan(x)],
    "random_fire_rate": [float(x) for x in mean_random if not np.isnan(x)],
}

# Figure: 3-panel spatial autocorrelation
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
ax = axes[0]
ax.hist(morans_I_per_date, bins=20, color="#1b9e77", alpha=0.7)
ax.axvline(morans_I_mean, color='red', linestyle='--', label=f"mean={morans_I_mean:.3f}")
ax.set_xlabel("Moran's I (per date)"); ax.set_ylabel("Frequency")
ax.set_title("A1a: Moran's I distribution\n(>0.3 = strong clustering)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
valid = ~np.isnan(mean_agreement)
ax.plot(bin_centers[valid], mean_agreement[valid], 'o-', color="#7570b3", linewidth=2, markersize=8)
ax.set_xlabel("Spatial distance (km)"); ax.set_ylabel("Label agreement rate")
ax.set_title("A1b: Distance-decay curve\n(label agreement vs spatial distance)")
ax.grid(alpha=0.3)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

ax = axes[2]
valid = ~np.isnan(mean_conditional)
ax.plot(bin_centers_short[valid], mean_conditional[valid], 'o-', color="#e7298a", linewidth=2, markersize=8, label="P(neighbor fire | this cell fire)")
ax.plot(bin_centers_short[valid], mean_random[valid], 's--', color="gray", linewidth=2, markersize=8, label="Random P(fire)")
ax.set_xlabel("Distance to neighbor (km)"); ax.set_ylabel("Probability")
ax.set_title("A1c: Fire-conditioned neighbor probability\n(how much more likely neighbor has fire)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Phase A1: Spatial autocorrelation analysis", fontsize=12, y=1.02)
plt.savefig(FIG/"fig_phaseA1_spatial_autocorr.png", dpi=110); plt.close()
print("  ✓ fig_phaseA1_spatial_autocorr.png saved")

# ============================================================
# A2: PREVALENCE-CONTROLLED N→S EVALUATION
# ============================================================
print("\n[A2] PREVALENCE-CONTROLLED N→S EVALUATION")
# Load MODIS-trained XGB model from runs (seed 42, h1, IID train → predict South test)
# We need to re-train XGB on North train, evaluate on South test (matching Spatial_OOD_NorthToSouth split)
print("  Re-training XGB on North (lat≥15) train, evaluating on South (lat<13)...")

# Get split indices
north_train_idx = splits_npz["Spatial_OOD_NorthToSouth_train"]
north_val_idx = splits_npz["Spatial_OOD_NorthToSouth_val"]
south_test_idx = splits_npz["Spatial_OOD_NorthToSouth_test"]

# Get train/val/test subsets (drop NaN labels)
features = json.loads((OUT/"panel_v85_meta.json").read_text())["features"]
tr = panel.iloc[north_train_idx].dropna(subset=["label_h1"])
va = panel.iloc[north_val_idx].dropna(subset=["label_h1"])
te = panel.iloc[south_test_idx].dropna(subset=["label_h1"])

print(f"  Train (North): {len(tr):,} rows, pos rate={tr['label_h1'].mean():.4f}")
print(f"  Val (North 20%): {len(va):,} rows, pos rate={va['label_h1'].mean():.4f}")
print(f"  Test (South): {len(te):,} rows, pos rate={te['label_h1'].mean():.4f}")

# Apply drought normalization
di_idx = features.index("precip_roll30_sum")
di_mean = np.nanmean(tr[features[di_idx]].values)
di_std = np.nanstd(tr[features[di_idx]].values) + 1e-6

X_tr = tr[features].values.astype(np.float32)
X_tr[:, di_idx] = np.nan_to_num((X_tr[:, di_idx] - di_mean) / di_std, nan=0.0)
y_tr = tr["label_h1"].values.astype(np.float32)
X_val = va[features].values.astype(np.float32)
X_val[:, di_idx] = np.nan_to_num((X_val[:, di_idx] - di_mean) / di_std, nan=0.0)
y_val = va["label_h1"].values.astype(np.float32)
X_test = te[features].values.astype(np.float32)
X_test[:, di_idx] = np.nan_to_num((X_test[:, di_idx] - di_mean) / di_std, nan=0.0)
y_test = te["label_h1"].values.astype(np.float32)

# Impute
imp = SimpleImputer(strategy="median")
X_tr = imp.fit_transform(X_tr)
X_val = imp.transform(X_val)
X_test = imp.transform(X_test)

# Train XGB
pos = max(int(y_tr.sum()), 1); neg = max(int((1-y_tr).sum()), 1)
spw = max(math.sqrt(neg/pos), 1.0)
m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
p_test = m.predict_proba(X_test)[:, 1]

# Original (unbalanced) test
auroc_full = roc_auc_score(y_test, p_test)
print(f"\n  Original South test (full): AUROC={auroc_full:.4f}, pos rate={y_test.mean():.4f}")

# A2: Prevalence-balanced — subsample South negatives to match North prevalence
print("\n  Prevalence-balanced bootstrap (1000 iterations)...")
north_prevalence = y_tr.mean()
n_pos_test = int(y_test.sum())
n_neg_test = int((1-y_test).sum())
# To balance: keep all positives, sample negatives so prevalence matches
target_n_neg = int(n_pos_test * (1 - north_prevalence) / north_prevalence)
print(f"  North prevalence: {north_prevalence:.4f}, South pos: {n_pos_test}, target neg: {target_n_neg}")

n_bootstrap = 200
auroc_balanced = []
brier_balanced = []
for b in range(n_bootstrap):
    rng = np.random.RandomState(b)
    neg_idx = np.where(y_test == 0)[0]
    if len(neg_idx) > target_n_neg:
        sampled_neg = rng.choice(neg_idx, target_n_neg, replace=False)
    else:
        sampled_neg = neg_idx
    pos_idx = np.where(y_test == 1)[0]
    keep_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep_idx)
    y_b = y_test[keep_idx]
    p_b = p_test[keep_idx]
    try:
        auroc_balanced.append(roc_auc_score(y_b, p_b))
        brier_balanced.append(brier_score_loss(y_b, p_b))
    except: pass

auroc_balanced = np.array(auroc_balanced)
brier_balanced = np.array(brier_balanced)
auroc_mean = float(auroc_balanced.mean())
auroc_std = float(auroc_balanced.std())
auroc_ci_lo = float(np.percentile(auroc_balanced, 2.5))
auroc_ci_hi = float(np.percentile(auroc_balanced, 97.5))
print(f"  Prevalence-balanced South: AUROC={auroc_mean:.4f} ± {auroc_std:.4f}")
print(f"  95% CI: [{auroc_ci_lo:.4f}, {auroc_ci_hi:.4f}]")
print(f"  → Δ from full test = {auroc_mean - auroc_full:+.4f}")

results["A2_prevalence_controlled"] = {
    "north_prevalence": float(north_prevalence),
    "south_prevalence": float(y_test.mean()),
    "n_pos_test": n_pos_test, "n_neg_test": n_neg_test, "target_n_neg": target_n_neg,
    "auroc_full_test": float(auroc_full),
    "auroc_balanced_mean": auroc_mean, "auroc_balanced_std": auroc_std,
    "auroc_balanced_ci_95": [auroc_ci_lo, auroc_ci_hi],
    "brier_balanced_mean": float(brier_balanced.mean()),
    "n_bootstrap": n_bootstrap,
}

# Figure
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
ax.hist(auroc_balanced, bins=30, color="#1b9e77", alpha=0.7)
ax.axvline(auroc_full, color='red', linestyle='--', linewidth=2, label=f"Original South (full): {auroc_full:.4f}")
ax.axvline(auroc_mean, color='blue', linestyle='-', linewidth=2, label=f"Balanced: {auroc_mean:.4f}±{auroc_std:.4f}")
ax.axvspan(auroc_ci_lo, auroc_ci_hi, alpha=0.2, color='blue', label=f"95% CI [{auroc_ci_lo:.4f}, {auroc_ci_hi:.4f}]")
ax.set_xlabel("AUROC"); ax.set_ylabel("Frequency (bootstrap)")
ax.set_title("A2: Prevalence-controlled N→S evaluation\n(Does AUROC recover when South prevalence matches North?)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.savefig(FIG/"fig_phaseA2_prevalence_balanced.png", dpi=110); plt.close()
print("  ✓ fig_phaseA2_prevalence_balanced.png saved")

# ============================================================
# A3: SPATIAL PROXY ABLATION
# ============================================================
print("\n[A3] SPATIAL PROXY ABLATION")
print("  Train XGB on North with 3 feature sets, evaluate on South:")
print("    v1: weather only (current, no lat/lon)")
print("    v2: weather + raw lat/lon")
print("    v3: weather + coarse lat_band (10 bands)")

# v1: weather only — already trained above (m)
auroc_v1 = roc_auc_score(y_test, p_test)
print(f"  v1 (weather only): AUROC={auroc_v1:.4f}")

# v2: weather + raw lat/lon
features_v2 = features + ["lat", "lon"]
X_tr_v2 = tr[features_v2].values.astype(np.float32)
X_tr_v2[:, di_idx] = np.nan_to_num((X_tr_v2[:, di_idx] - di_mean) / di_std, nan=0.0)
X_val_v2 = va[features_v2].values.astype(np.float32)
X_val_v2[:, di_idx] = np.nan_to_num((X_val_v2[:, di_idx] - di_mean) / di_std, nan=0.0)
X_test_v2 = te[features_v2].values.astype(np.float32)
X_test_v2[:, di_idx] = np.nan_to_num((X_test_v2[:, di_idx] - di_mean) / di_std, nan=0.0)
imp_v2 = SimpleImputer(strategy="median")
X_tr_v2 = imp_v2.fit_transform(X_tr_v2); X_val_v2 = imp_v2.transform(X_val_v2); X_test_v2 = imp_v2.transform(X_test_v2)
m_v2 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m_v2.fit(X_tr_v2, y_tr, eval_set=[(X_val_v2, y_val)], verbose=False)
p_test_v2 = m_v2.predict_proba(X_test_v2)[:, 1]
auroc_v2 = roc_auc_score(y_test, p_test_v2)
print(f"  v2 (weather + lat/lon): AUROC={auroc_v2:.4f}")

# v3: weather + coarse lat_band (10 bands, encoded as numeric)
lat_bands_10 = pd.cut(te["lat"], bins=10, labels=False)
# Add lat_band to features — but need to compute for train/val too
tr_lat_band = pd.cut(tr["lat"], bins=10, labels=False).fillna(5).astype(np.float32).values.reshape(-1, 1)
va_lat_band = pd.cut(va["lat"], bins=10, labels=False).fillna(5).astype(np.float32).values.reshape(-1, 1)
te_lat_band = pd.cut(te["lat"], bins=10, labels=False).fillna(5).astype(np.float32).values.reshape(-1, 1)
X_tr_v3 = np.hstack([X_tr, tr_lat_band])
X_val_v3 = np.hstack([X_val, va_lat_band])
X_test_v3 = np.hstack([X_test, te_lat_band])
m_v3 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.85,
    colsample_bytree=0.7, min_child_weight=20, reg_lambda=3.0, reg_alpha=0.5, gamma=0.1,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=4)
m_v3.fit(X_tr_v3, y_tr, eval_set=[(X_val_v3, y_val)], verbose=False)
p_test_v3 = m_v3.predict_proba(X_test_v3)[:, 1]
auroc_v3 = roc_auc_score(y_test, p_test_v3)
print(f"  v3 (weather + lat_band): AUROC={auroc_v3:.4f}")

results["A3_spatial_proxy_ablation"] = {
    "v1_weather_only": {"features_n": len(features), "auroc": float(auroc_v1)},
    "v2_weather_lat_lon": {"features_n": len(features_v2), "auroc": float(auroc_v2)},
    "v3_weather_lat_band": {"features_n": X_tr_v3.shape[1], "auroc": float(auroc_v3)},
    "delta_v2_v1": float(auroc_v2 - auroc_v1),
    "delta_v3_v1": float(auroc_v3 - auroc_v1),
}

# Figure
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
x_pos = [0, 1, 2]
aurocs = [auroc_v1, auroc_v2, auroc_v3]
labels = ["v1: weather only\n(current)", "v2: + lat/lon\n(raw)", "v3: + lat_band\n(coarse)"]
bars = ax.bar(x_pos, aurocs, color=["#e7298a","#7570b3","#1b9e77"], width=0.5)
for b, a in zip(bars, aurocs):
    ax.text(b.get_x() + b.get_width()/2, a + 0.005, f"{a:.4f}",
            ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(labels)
ax.set_ylabel("AUROC on South test (Spatial OOD N→S)")
ax.set_title("A3: Spatial proxy ablation\n(Does adding spatial features reduce N→S gap?)")
ax.set_ylim(0.5, 1.0)
ax.axhline(auroc_v1, color='red', linestyle='--', alpha=0.5)
ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG/"fig_phaseA3_spatial_proxy_ablation.png", dpi=110); plt.close()
print("  ✓ fig_phaseA3_spatial_proxy_ablation.png saved")

# Save all results
with open(OUT / "phaseA_diagnosis.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Results saved to {OUT}/phaseA_diagnosis.json")

# Final verdict
print("\n" + "="*70)
print("PHASE A — FINAL VERDICT")
print("="*70)
print(f"\nA1 (Spatial autocorrelation):")
print(f"  Moran's I = {morans_I_mean:.4f} ± {morans_I_std:.4f}")
if morans_I_mean > 0.3:
    print(f"  → STRONG spatial clustering of fire (IID is biased easy)")
elif morans_I_mean > 0.1:
    print(f"  → Moderate spatial clustering")
else:
    print(f"  → Weak spatial clustering")

print(f"\nA2 (Prevalence-controlled N→S):")
print(f"  Original South AUROC: {auroc_full:.4f}")
print(f"  Prevalence-balanced AUROC: {auroc_mean:.4f} ± {auroc_std:.4f} (95% CI: [{auroc_ci_lo:.4f}, {auroc_ci_hi:.4f}])")
delta = auroc_mean - auroc_full
if abs(delta) < 0.02:
    print(f"  → Prevalence balancing does NOT recover performance (Δ={delta:+.4f})")
    print(f"  → The failure is NOT primarily a base-rate mismatch problem")
elif delta > 0:
    print(f"  → Prevalence balancing RECOVERS {delta:+.4f} AUROC")
    print(f"  → Some of the gap is base-rate mismatch, but other factors also contribute")
else:
    print(f"  → Prevalence balancing HURTS {delta:+.4f} AUROC")

print(f"\nA3 (Spatial proxy ablation):")
print(f"  v1 (weather only):         AUROC = {auroc_v1:.4f}")
print(f"  v2 (weather + lat/lon):     AUROC = {auroc_v2:.4f}  (Δ = {auroc_v2-auroc_v1:+.4f})")
print(f"  v3 (weather + lat_band):    AUROC = {auroc_v3:.4f}  (Δ = {auroc_v3-auroc_v1:+.4f})")
if auroc_v3 > auroc_v1 + 0.05:
    print(f"  → Adding spatial proxy HELPS (+{(auroc_v3-auroc_v1):.4f})")
    print(f"  → Hidden spatial structure is real and recoverable from lat/lon alone")
    print(f"  → Need real spatial data (NDVI/land cover/elevation) in Phase B to fully address")
elif auroc_v3 > auroc_v1 + 0.01:
    print(f"  → Adding spatial proxy slightly helps (+{(auroc_v3-auroc_v1):.4f})")
else:
    print(f"  → Adding spatial proxy does NOT help (Δ < 0.01)")
    print(f"  → Hidden structure is NOT captured by lat/lon alone — need real spatial features")
