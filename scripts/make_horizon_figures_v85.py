#!/usr/bin/env python3
"""
Generate horizon comparison figures from v8.5 results.
Compares t+1, t+3, t+7 across splits and models.
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception: pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = Path("/home/z/my-project/download/real_data_v85")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# Load all runs
runs = []
for f in (OUT / "runs").glob("*.json"):
    try: runs.append(json.loads(f.read_text()))
    except: pass
print(f"Loaded {len(runs)} runs")

rows = []
for r in runs:
    for cal, m in r["metrics"].items():
        rows.append({"split": r["split"], "horizon": r["horizon"], "seed": r["seed"],
                     "model": r["model"], "calibrator": cal, **m})
df = pd.DataFrame(rows)
print(f"DataFrame: {len(df)} rows")
print(f"Horizons: {sorted(df['horizon'].unique())}")
print(f"Seeds: {sorted(df['seed'].unique())}")
print(f"Splits: {sorted(df['split'].unique())}")

COLORS = {
    "LogReg": "#999999", "RF": "#66a61e", "XGB": "#e7298a",
    "MLP": "#d95f02", "PhysMLP": "#1b9e77",
}
HORIZON_LABELS = {1: "t+1 (next day)", 3: "t+3 (3-day)", 7: "t+7 (week)"}

# ============================================================
# FIG 9: AUROC vs horizon, IID + Temporal OOD, key models
# ============================================================
print("\n[FIG 9] AUROC vs horizon...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
splits_to_plot = ["IID_groupblocked", "Temporal_OOD", "Spatial_OOD_NorthToSouth"]
split_labels = {"IID_groupblocked": "IID", "Temporal_OOD": "Temporal OOD",
                "Spatial_OOD_NorthToSouth": "Spatial N→S"}
horizons = sorted(df["horizon"].unique())
# For each (split, model), plot AUROC vs horizon
for sp in splits_to_plot:
    for model in ["XGB", "MLP", "PhysMLP"]:
        means = []; stds = []
        for h in horizons:
            sub = df[(df["split"] == sp) & (df["model"] == model) &
                     (df["horizon"] == h) & (df["calibrator"] == "Iso")]
            if len(sub):
                means.append(sub["AUROC"].mean())
                stds.append(sub["AUROC"].std())
            else:
                means.append(None); stds.append(None)
        # Filter None
        valid = [(h, m, s) for h, m, s in zip(horizons, means, stds) if m is not None]
        if not valid: continue
        xs, ys, errs = zip(*valid)
        # Use offset based on split to avoid overlap
        offset = {"IID_groupblocked": -0.15, "Temporal_OOD": 0,
                  "Spatial_OOD_NorthToSouth": 0.15}[sp]
        ax.errorbar([x + offset for x in xs], ys, yerr=errs,
                    marker='o', linewidth=2, markersize=10, capsize=5,
                    label=f"{model} ({split_labels[sp]})",
                    color=COLORS[model], linestyle='--' if sp != 'IID_groupblocked' else '-')

ax.set_xlabel("Forecast horizon")
ax.set_ylabel("AUROC (Iso calibration)")
ax.set_xticks(horizons)
ax.set_xticklabels([HORIZON_LABELS.get(h, f"t+{h}") for h in horizons])
ax.set_title("Discrimination degradation by forecast horizon (v8.5)\n"
             "t+1 has 5 seeds; t+3 and t+7 have 1 seed (so far)")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
plt.savefig(FIG / "fig9_auroc_vs_horizon.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig9_auroc_vs_horizon.png")

# ============================================================
# FIG 10: ECE vs horizon, IID + Temporal OOD + Spatial N→S
# ============================================================
print("\n[FIG 10] ECE vs horizon...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
for sp in splits_to_plot:
    for model in ["XGB", "MLP", "PhysMLP"]:
        means = []; stds = []
        for h in horizons:
            sub = df[(df["split"] == sp) & (df["model"] == model) &
                     (df["horizon"] == h) & (df["calibrator"] == "Iso")]
            if len(sub):
                means.append(sub["ECE"].mean())
                stds.append(sub["ECE"].std())
            else:
                means.append(None); stds.append(None)
        valid = [(h, m, s) for h, m, s in zip(horizons, means, stds) if m is not None]
        if not valid: continue
        xs, ys, errs = zip(*valid)
        offset = {"IID_groupblocked": -0.15, "Temporal_OOD": 0,
                  "Spatial_OOD_NorthToSouth": 0.15}[sp]
        ax.errorbar([x + offset for x in xs], ys, yerr=errs,
                    marker='o', linewidth=2, markersize=10, capsize=5,
                    label=f"{model} ({split_labels[sp]})",
                    color=COLORS[model], linestyle='--' if sp != 'IID_groupblocked' else '-')

ax.set_xlabel("Forecast horizon")
ax.set_ylabel("ECE (Iso, lower = better)")
ax.set_xticks(horizons)
ax.set_xticklabels([HORIZON_LABELS.get(h, f"t+{h}") for h in horizons])
ax.set_title("Calibration degradation by forecast horizon (v8.5)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
plt.savefig(FIG / "fig10_ece_vs_horizon.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig10_ece_vs_horizon.png")

# ============================================================
# FIG 11: Comprehensive results table as image
# ============================================================
print("\n[FIG 11] Comprehensive results table...")
fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
ax.axis("off")
rows = []
for sp in ["IID_groupblocked", "Temporal_OOD",
           "Spatial_OOD_NorthToSouth", "Spatial_OOD_SouthToCentral",
           "Spatial_OOD_CentralSouthToNorth"]:
    sp_short = {"IID_groupblocked": "IID",
                "Temporal_OOD": "Temp OOD",
                "Spatial_OOD_NorthToSouth": "Spatial N→S",
                "Spatial_OOD_SouthToCentral": "Spatial S→C",
                "Spatial_OOD_CentralSouthToNorth": "Spatial C+S→N"}[sp]
    for h in horizons:
        for model in ["LogReg", "RF", "XGB", "MLP", "PhysMLP"]:
            sub = df[(df["split"] == sp) & (df["model"] == model) &
                     (df["horizon"] == h) & (df["calibrator"] == "Iso")]
            if len(sub):
                r = sub.iloc[0]
                n_seeds = len(sub)
                rows.append([
                    sp_short,
                    f"t+{h}",
                    model,
                    f"{r['AUROC']:.4f}",
                    f"{r['ECE']:.4f}",
                    f"{r['Brier']:.4f}",
                    f"{r['F1']:.4f}",
                    f"{n_seeds}"
                ])

tbl = ax.table(cellText=rows,
               colLabels=["Split", "Horizon", "Model", "AUROC", "ECE(Iso)", "Brier", "F1", "n_seeds"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.4)
ax.set_title("v8.5 Comprehensive Results (Iso calibration, all splits × horizons × models)\n"
             "n_seeds=5 for t+1; n_seeds=1 for t+3, t+7 (seed 42)",
             fontsize=12, pad=20)
plt.savefig(FIG / "fig11_comprehensive_table.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig11_comprehensive_table.png")

print(f"\n✓ All horizon comparison figures saved to {FIG}")
