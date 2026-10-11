#!/usr/bin/env python3
"""v8.5 COUNTERFACTUAL physics consistency test."""
import os, sys, json, time, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/z/my-project/scripts")
import train_v85

OUT = Path("/home/z/my-project/download/real_data_v85")
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
print("="*70); print("v8.5 COUNTERFACTUAL PHYSICS CONSISTENCY"); print("="*70)

panel = pd.read_parquet(OUT/"panel_v85.parquet")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
splits_npz = np.load(OUT/"splits_v85.npz")
splits = {"IID_groupblocked": {"train_idx": splits_npz["IID_groupblocked_train"].tolist(),
                                "val_idx": splits_npz["IID_groupblocked_val"].tolist(),
                                "test_idx": splits_npz["IID_groupblocked_test"].tolist()}}
meta = json.loads((OUT/"panel_v85_meta.json").read_text())
features = meta["features"]

sp = splits["IID_groupblocked"]
tr = panel.iloc[np.array(sp["train_idx"])].dropna(subset=["label_h1"])
va = panel.iloc[np.array(sp["val_idx"])].dropna(subset=["label_h1"])
X_tr = tr[features].values.astype(np.float32); y_tr = tr["label_h1"].values.astype(np.float32)
X_val = va[features].values.astype(np.float32); y_val = va["label_h1"].values.astype(np.float32)
X_raw_tr = tr[train_v85.RAW_PHYS_COLS].values.astype(np.float32)
di_idx = features.index("precip_roll30_sum")
di_mean = np.nanmean(X_tr[:, di_idx]); di_std = np.nanstd(X_tr[:, di_idx]) + 1e-6
X_tr[:, di_idx] = np.nan_to_num((X_tr[:, di_idx] - di_mean) / di_std, nan=0.0)
X_val[:, di_idx] = np.nan_to_num((X_val[:, di_idx] - di_mean) / di_std, nan=0.0)
print(f"  Train: {len(tr):,} rows")

SEED = 42
print("Training XGB, MLP, PhysMLP...")
_, pxgb = train_v85.train_xgb(X_tr, y_tr, X_val, y_val, seed=SEED)
_, pmlp = train_v85.train_mlp(X_tr, y_tr, X_val, y_val, X_raw_tr, seed=SEED, lam=0.0, epochs=8)
_, pphys = train_v85.train_mlp(X_tr, y_tr, X_val, y_val, X_raw_tr, seed=SEED, lam=0.10, epochs=8)

base = X_tr.mean(axis=0)
N_STEPS = 50
def cf_curve(feature, predict_fn, base, X_train):
    idx = features.index(feature)
    p_lo, p_hi = np.percentile(X_train[:, idx], [5, 95])
    sv = np.linspace(p_lo, p_hi, N_STEPS)
    rows = np.tile(base, (N_STEPS, 1)); rows[:, idx] = sv
    return sv, predict_fn(rows)

def viol_rate(x, p, direction):
    n = len(x); v = 0; t = 0
    for i in range(n):
        for j in range(i+1, n):
            if x[i] < x[j]:
                if direction == "positive" and p[i] > p[j]: v += 1
                elif direction == "negative" and p[i] < p[j]: v += 1
                t += 1
    return v / max(t, 1)

tests = [("t2m","positive","T↑ → risk↑"), ("rh","negative","RH↑ → risk↓"),
         ("wind","positive","W↑ → risk↑"), ("precip_roll30_sum","negative","precip↑ → risk↓")]
results = {}
for mn, pf in [("XGB",pxgb),("MLP",pmlp),("PhysMLP",pphys)]:
    print(f"  {mn}:")
    results[mn] = {}
    for f,d,desc in tests:
        x, p = cf_curve(f, pf, base, X_tr)
        r = viol_rate(x, p, d)
        c = "✓" if r<0.05 else ("~" if r<0.20 else "✗")
        print(f"    {f:25s} ({desc:20s}): violation={r:.1%}  {c}")
        results[mn][f] = {"direction":d,"violation_rate":float(r),"consistent":r<0.05,
                          "x_range":[float(x.min()),float(x.max())],
                          "p_range":[float(p.min()),float(p.max())]}

out = {"version":"v8.5_counterfactual","n_steps":N_STEPS,
       "summary":{m:{f:{"violation_rate":r["violation_rate"],"consistent":r["consistent"]} for f,r in res.items()} for m,res in results.items()},
       "details":{m:{f:{"violation_rate":r["violation_rate"],"consistent":r["consistent"],
                        "x_range":r["x_range"],"p_range":r["p_range"]} for f,r in res.items()} for m,res in results.items()}}
with open(OUT/"counterfactual_v85.json","w") as f: json.dump(out,f,indent=2,default=str)
print(f"✓ JSON saved")

# Figures
COLORS = {"XGB":"#e7298a","MLP":"#d95f02","PhysMLP":"#1b9e77"}
fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
axes = axes.flatten()
for i, (f,d,desc) in enumerate(tests):
    ax = axes[i]
    for mn in ["XGB","MLP","PhysMLP"]:
        x = np.linspace(*[r for r in results[mn][f]["x_range"]], N_STEPS) if False else None
        # Actually re-run curve (saved only ranges). Simpler: re-compute
        x_vals, p_vals = cf_curve(f, {"XGB":pxgb,"MLP":pmlp,"PhysMLP":pphys}[mn], base, X_tr)
        ax.plot(x_vals, p_vals, marker='o', linewidth=2, markersize=4,
                label=f"{mn} (viol={results[mn][f]['violation_rate']:.1%})",
                color=COLORS[mn])
    ax.set_xlabel(f); ax.set_ylabel("Predicted fire risk")
    ax.set_title(f"{desc}\n(direction: {d})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Counterfactual physics consistency (v8.5, IID, seed 42)", y=1.02)
plt.savefig(FIG/"fig7_counterfactual_curves_v85.png", dpi=120); plt.close()
print(f"✓ fig7 saved")

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
x_pos = np.arange(len(tests)); w = 0.27
for i, mn in enumerate(["XGB","MLP","PhysMLP"]):
    rates = [results[mn][f]["violation_rate"] for f,_,_ in tests]
    bars = ax.bar(x_pos + (i-1)*w, rates, w, label=mn, color=COLORS[mn])
    for b, r in zip(bars, rates): ax.text(b.get_x()+b.get_width()/2, r+0.01, f"{r:.1%}", ha="center", fontsize=8)
ax.set_xticks(x_pos); ax.set_xticklabels([f"{f}\n({d})" for f,d,_ in tests], fontsize=9)
ax.set_ylabel("Monotonicity violation rate (lower = better)")
ax.set_title("Physical consistency: violation rate by model & feature (v8.5)")
ax.axhline(0.05, color='green', linestyle='--', alpha=0.5, label="5% threshold")
ax.axhline(0.20, color='orange', linestyle='--', alpha=0.5, label="20% threshold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG/"fig8_counterfactual_summary_v85.png", dpi=120); plt.close()
print(f"✓ fig8 saved")
print(f"\n{'='*70}\n✓ COUNTERFACTUAL COMPLETE\n{'='*70}")
