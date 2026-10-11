#!/usr/bin/env python3
"""
Run training pipeline with REAL NASA MODIS fire labels + synthetic weather.
This removes the double-dipping issue (labels no longer generated from
the same physics formula as features).
"""
import os, sys, time, json, math, warnings, random
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/scripts")
warnings.filterwarnings("ignore")

from run_research import (
    build_dataset, make_splits, train_xgb, train_physics_model, predict_physics,
    compute_metrics, find_f1_optimal_threshold, spatial_f1,
    FEATURES, MASTER_SEED, set_seed, ResidualBlock, PhysicsMLP,
    physics_penalty,
)
import xgboost as xgb
import torch
import torch.nn as nn

REAL_FIRE_CSV = "/home/z/my-project/download/real_data_pilot/firms/modis_vietnam_2018_2023_real.csv"

set_seed(MASTER_SEED)
print("="*70)
print("REAL-DATA PILOT: Real NASA MODIS fire labels + synthetic weather")
print("="*70)

# Step 1: Generate synthetic weather features (same as before)
print("\n[1/4] Generating synthetic weather features...")
df = build_dataset(seed=MASTER_SEED)
print(f"  Synthetic features: {len(df):,} rows, {len(FEATURES)} features")

# Step 2: Load REAL fire labels and replace synthetic labels
print("\n[2/4] Loading REAL NASA MODIS fire labels...")
firms = pd.read_csv(REAL_FIRE_CSV)
print(f"  Real fire points: {len(firms):,}")
print(f"  Date range: {firms['acq_date'].min()} → {firms['acq_date'].max()}")

# Snap fire lat/lon to 0.50° grid (matching synthetic data)
firms["lat_grid"] = np.round((firms["latitude"] - 11) / 0.5) * 0.5 + 11
firms["lon_grid"] = np.round((firms["longitude"] - 104) / 0.5) * 0.5 + 104
firms["date"] = pd.to_datetime(firms["acq_date"])

# Create fire set: (lat_grid, lon_grid, date) → label=1
fire_set = firms[["lat_grid", "lon_grid", "date"]].drop_duplicates()
fire_set["label"] = 1
print(f"  Fire events after grid snapping: {len(fire_set):,}")

# Merge: for each (lat, lon, date) in synthetic data, check if fire exists
print("\n  Merging real labels with synthetic weather...")
df["date"] = pd.to_datetime(df["date"])
df["lat_grid"] = df["lat"]
df["lon_grid"] = df["lon"]

# Drop synthetic label and prob_true
df = df.drop(columns=["label", "prob_true"])

# Merge with real fire labels
df = df.merge(fire_set, on=["lat_grid", "lon_grid", "date"], how="left")
df["label"] = df["label"].fillna(0).astype(int)

# Clean up
df = df.drop(columns=["lat_grid", "lon_grid"])

n_fire = df["label"].sum()
n_total = len(df)
pos_rate = 100 * n_fire / n_total
print(f"  ✓ Real labels merged: {n_fire:,} fire events out of {n_total:,} ({pos_rate:.2f}%)")
print(f"  Double-dipping REMOVED: labels now come from real NASA MODIS, not synthetic logit")

# Step 3: Train + Evaluate
print("\n[3/4] Training models...")
splits = make_splits(df)
train_df = splits["train"]; val_df = splits["val"]; test_df = splits["test"]
ood_time_df = splits["ood_time"]; ood_space_df = splits["ood_space"]

for f in FEATURES:
    train_df[f] = train_df[f].astype("float64")
    val_df[f] = val_df[f].astype("float64")
    test_df[f] = test_df[f].astype("float64")
    if len(ood_time_df) > 0: ood_time_df[f] = ood_time_df[f].astype("float64")
    if len(ood_space_df) > 0: ood_space_df[f] = ood_space_df[f].astype("float64")

y_tr = train_df["label"]; y_val = val_df["label"]; y_test = test_df["label"]

# XGBoost baseline
print("  Training XGBoost baseline...")
xgb_m, _ = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED)
p_val_b = xgb_m.predict_proba(val_df[FEATURES])[:, 1]
f1_thr_b = find_f1_optimal_threshold(y_val.values, p_val_b)
p_test_b = xgb_m.predict_proba(test_df[FEATURES])[:, 1]
m_b = compute_metrics(y_test.values, p_test_b, f1_threshold=f1_thr_b)
m_b["Spatial_F1"] = spatial_f1(test_df, p_test_b)
print(f"  Baseline:  AUROC={m_b['AUROC']:.4f} ECE={m_b['ECE']:.4f} Brier={m_b['Brier']:.4f} F1={m_b['F1']:.4f}")

# Physics MLP (λ=0.20)
print("  Training Physics MLP (λ=0.20, 25 epochs)...")
m_p1, sc_p1, xg_p1, _ = train_physics_model(train_df, y_tr, val_df, y_val,
    seed=MASTER_SEED, lam=0.20, epochs=25, verbose=False)
p_val_p = predict_physics(m_p1, sc_p1, xg_p1, val_df)
f1_thr_p = find_f1_optimal_threshold(y_val.values, p_val_p)
p_test_p = predict_physics(m_p1, sc_p1, xg_p1, test_df)
m_p = compute_metrics(y_test.values, p_test_p, f1_threshold=f1_thr_p)
m_p["Spatial_F1"] = spatial_f1(test_df, p_test_p)
print(f"  Proposed:  AUROC={m_p['AUROC']:.4f} ECE={m_p['ECE']:.4f} Brier={m_p['Brier']:.4f} F1={m_p['F1']:.4f}")

# Step 4: λ-sweep (most important!)
print("\n[4/4] λ-sweep — does physics help on REAL labels?")
print("  Training MLP with λ=0 (no physics)...")
m_np, sc_np, xg_np, _ = train_physics_model(train_df, y_tr, val_df, y_val,
    seed=MASTER_SEED, lam=0.0, epochs=25, verbose=False)
p_test_np = predict_physics(m_np, sc_np, xg_np, test_df)
m_np_test = compute_metrics(y_test.values, p_test_np)

print(f"\n  {'='*60}")
print(f"  λ-SWEEP RESULTS (REAL fire labels)")
print(f"  {'='*60}")
print(f"  λ=0   (no physics): AUROC={m_np_test['AUROC']:.4f}  ECE={m_np_test['ECE']:.4f}  Brier={m_np_test['Brier']:.4f}")
print(f"  λ=0.2 (physics):    AUROC={m_p['AUROC']:.4f}  ECE={m_p['ECE']:.4f}  Brier={m_p['Brier']:.4f}")
print(f"  Δ AUROC: {m_p['AUROC']-m_np_test['AUROC']:+.4f}")
print(f"  Δ ECE:   {m_p['ECE']-m_np_test['ECE']:+.4f}")
print(f"  Δ Brier: {m_p['Brier']-m_np_test['Brier']:+.4f}")
print()

# Comparison with synthetic
print(f"  {'='*60}")
print(f"  COMPARISON: Synthetic vs Real labels")
print(f"  {'='*60}")
# Load synthetic results
with open("/home/z/my-project/scripts/results.json") as f:
    synth = json.load(f)
bm_s = synth["baseline_metrics_test"]
pm_s = synth["proposed_metrics_test"]
print(f"  {'Metric':<14} | {'Synth base':<12} | {'Synth prop':<12} | {'Real base':<12} | {'Real prop':<12}")
print(f"  {'-'*60}")
for metric in ["AUROC", "AUPRC", "F1", "Brier", "ECE"]:
    print(f"  {metric:<14} | {bm_s[metric]:<12.4f} | {pm_s[metric]:<12.4f} | {m_b[metric]:<12.4f} | {m_p[metric]:<12.4f}")

# Conclusion
print(f"\n  {'='*60}")
print(f"  CONCLUSION")
print(f"  {'='*60}")
delta_ece_real = m_p['ECE'] - m_np_test['ECE']
delta_ece_synth = pm_s['ECE'] - bm_s['ECE']
print(f"  Synthetic Δ ECE (prop-base): {delta_ece_synth:+.4f}")
print(f"  Real Δ ECE (λ=0.2 vs λ=0):   {delta_ece_real:+.4f}")
print()
if abs(delta_ece_real) < 0.005:
    print("  → Physics STILL doesn't help on real data (Δ ECE < 0.005)")
    print("  → Paper should PIVOT contribution to 'intrinsically calibrated architecture'")
elif delta_ece_real < 0:
    print("  → Physics HELPS on real data! (ECE reduced by physics)")
    print("  → Synthetic result was misleading — physics contribution is real!")
else:
    print("  → Physics HURTS on real data — surprising result")
    print("  → Worth investigating further")

# OOD evaluation
print(f"\n  {'='*60}")
print(f"  OOD Evaluation (Real labels)")
print(f"  {'='*60}")
ood_conds = [
    ("extreme_temp", test_df[test_df["t2m"] >= test_df["t2m"].quantile(0.95)]),
    ("low_humidity", test_df[test_df["rh"] <= test_df["rh"].quantile(0.10)]),
    ("extreme_wind", test_df[test_df["wind"] >= test_df["wind"].quantile(0.95)]),
    ("drought", test_df[test_df["drought_idx"] <= test_df["drought_idx"].quantile(0.10)]),
]
if len(ood_space_df) > 0 and ood_space_df["label"].sum() > 0:
    ood_conds.append(("unseen_region_KH", ood_space_df))
if len(ood_time_df) > 0 and ood_time_df["label"].sum() > 0:
    ood_conds.append(("unseen_period_2023", ood_time_df))

for name, sub in ood_conds:
    if len(sub) == 0 or sub["label"].sum() == 0:
        print(f"  {name:25s} skipped (no positives)")
        continue
    p_b = xgb_m.predict_proba(sub[FEATURES])[:, 1]
    p_p = predict_physics(m_p1, sc_p1, xg_p1, sub)
    m_b_ood = compute_metrics(sub["label"].values, p_b, f1_threshold=f1_thr_b)
    m_p_ood = compute_metrics(sub["label"].values, p_p, f1_threshold=f1_thr_p)
    print(f"  {name:25s} n={len(sub):5d} pos={int(sub['label'].sum()):4d} | "
          f"base AUROC={m_b_ood['AUROC']:.4f} ECE={m_b_ood['ECE']:.4f} | "
          f"prop AUROC={m_p_ood['AUROC']:.4f} ECE={m_p_ood['ECE']:.4f}")

# Save results
results = {
    "data_source": "real_labels_synthetic_weather",
    "description": "Real NASA MODIS fire labels (17,780 points, 2018-2023) + synthetic weather features",
    "double_dipping_resolved": True,
    "baseline_metrics_test": m_b,
    "proposed_metrics_test": m_p,
    "lambda_sweep": {
        "0.0": {"test": m_np_test},
        "0.2": {"test": m_p},
    },
    "delta_ece_synth": delta_ece_synth,
    "delta_ece_real": delta_ece_real,
}
out_path = "/home/z/my-project/download/real_data_pilot/results_real_labels.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  ✓ Results saved to {out_path}")
print(f"\n{'='*70}")
print("✓ REAL-DATA PILOT COMPLETE")
print("="*70)
