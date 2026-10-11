#!/usr/bin/env python3
"""
Generate figures from v8.5 results (smoke test single seed, will refresh after multi-seed).
Outputs PNGs to /home/z/my-project/download/real_data_v85/figures/
"""
import json, os, sys
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

# Load all run JSONs
runs = []
for f in (OUT / "runs").glob("*.json"):
    try:
        runs.append(json.loads(f.read_text()))
    except Exception: pass
print(f"Loaded {len(runs)} runs")

# Aggregate to DataFrame
rows = []
for r in runs:
    for cal, m in r["metrics"].items():
        rows.append({
            "split": r["split"], "horizon": r["horizon"], "seed": r["seed"],
            "model": r["model"], "calibrator": cal,
            **m
        })
df = pd.DataFrame(rows)
print(f"DataFrame: {len(df)} rows")
print(df.head())

# Filter to t+1 horizon, all splits, all models, Iso + Raw calibrators
df_focus = df[(df["horizon"] == 1) & (df["calibrator"].isin(["Raw", "Iso"]))]
print(f"Focus (t+1, Raw+Iso): {len(df_focus)} rows")

# Color palette
COLORS = {
    "LogReg":   "#999999",
    "RF":       "#66a61e",
    "XGB":      "#e7298a",
    "MLP":      "#d95f02",
    "PhysMLP":  "#1b9e77",
}

# ============================================================
# FIG 1: ΔECE degradation (IID → Temporal OOD → 3 Spatial OODs)
# The headline finding
# ============================================================
print("\n[FIG 1] ΔECE degradation across splits...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

splits_order = ["IID_groupblocked", "Temporal_OOD",
                "Spatial_OOD_NorthToSouth", "Spatial_OOD_SouthToCentral",
                "Spatial_OOD_CentralSouthToNorth"]
split_labels = ["IID", "Temporal OOD", "Spatial N→S", "Spatial S→C", "Spatial C+S→N"]

# Plot Iso ECE for each model across splits
for model in ["LogReg", "RF", "XGB", "MLP", "PhysMLP"]:
    means = []
    stds = []
    for sp in splits_order:
        sub = df_focus[(df_focus["model"] == model) & (df_focus["split"] == sp) &
                       (df_focus["calibrator"] == "Iso")]
        if len(sub):
            means.append(sub["ECE"].mean())
            stds.append(sub["ECE"].std())
        else:
            means.append(0); stds.append(0)
    ax.errorbar(range(len(splits_order)), means, yerr=stds,
                marker='o', linewidth=2, markersize=10, capsize=5,
                label=model, color=COLORS[model])

ax.set_xticks(range(len(splits_order)))
ax.set_xticklabels(split_labels, rotation=15, ha="right")
ax.set_ylabel("ECE (Iso calibration, lower = better)")
ax.set_title("Calibration degradation across distribution shifts (v8.5 smoke test, 1 seed, t+1)\n"
             "XGB+Iso stays robust; MLP degrades 5× from IID to spatial OOD")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(FIG / "fig1_ece_degradation_v85.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig1_ece_degradation_v85.png")

# ============================================================
# FIG 2: AUROC across splits
# ============================================================
print("\n[FIG 2] AUROC across splits...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
for model in ["LogReg", "RF", "XGB", "MLP", "PhysMLP"]:
    means = []; stds = []
    for sp in splits_order:
        sub = df_focus[(df_focus["model"] == model) & (df_focus["split"] == sp)]
        means.append(sub["AUROC"].mean())
        stds.append(sub["AUROC"].std())
    ax.errorbar(range(len(splits_order)), means, yerr=stds,
                marker='s', linewidth=2, markersize=10, capsize=5,
                label=model, color=COLORS[model])

ax.set_xticks(range(len(splits_order)))
ax.set_xticklabels(split_labels, rotation=15, ha="right")
ax.set_ylabel("AUROC")
ax.set_title("Discrimination across splits (v8.5 smoke test, 1 seed, t+1)")
ax.set_ylim(0.85, 0.97)
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(FIG / "fig2_auroc_v85.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig2_auroc_v85.png")

# ============================================================
# FIG 3: MLP vs PhysMLP comparison (does physics help?)
# ============================================================
print("\n[FIG 3] MLP vs PhysMLP (physics effect)...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
for ax_i, (metric, title) in enumerate(zip(
    ["AUROC", "ECE", "Brier"],
    ["AUROC (higher = better)", "ECE (lower = better)", "Brier (lower = better)"]
)):
    ax = axes[ax_i]
    width = 0.35
    x = np.arange(len(splits_order))
    mlp_vals = [df_focus[(df_focus["model"] == "MLP") & (df_focus["split"] == sp) &
                         (df_focus["calibrator"] == "Iso")][metric].mean() for sp in splits_order]
    phys_vals = [df_focus[(df_focus["model"] == "PhysMLP") & (df_focus["split"] == sp) &
                          (df_focus["calibrator"] == "Iso")][metric].mean() for sp in splits_order]
    bars1 = ax.bar(x - width/2, mlp_vals, width, label="MLP (no physics)", color="#d95f02")
    bars2 = ax.bar(x + width/2, phys_vals, width, label="PhysMLP (λ=0.10)", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(title)
    ax.set_title(metric)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Physics effect: MLP vs PhysMLP (Iso calibration, t+1)", fontsize=11, y=1.02)
plt.savefig(FIG / "fig3_mlp_vs_physmlp_v85.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig3_mlp_vs_physmlp_v85.png")

# ============================================================
# FIG 4: Calibration comparison: Raw vs Iso vs Platt
# ============================================================
print("\n[FIG 4] Calibration comparison...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
df_cal = df[df["horizon"] == 1].copy()
calibrators = ["Raw", "Temp", "Iso", "Platt"]
cal_colors = {"Raw": "#e7298a", "Temp": "#7570b3", "Iso": "#1b9e77", "Platt": "#d95f02"}
for cal in calibrators:
    means = []
    for sp in splits_order:
        sub = df_cal[(df_cal["split"] == sp) & (df_cal["calibrator"] == cal) &
                     (df_cal["model"] == "XGB")]
        means.append(sub["ECE"].mean() if len(sub) else 0)
    ax.plot(range(len(splits_order)), means, marker='o', linewidth=2, markersize=10,
            label=f"XGB + {cal}", color=cal_colors[cal])

ax.set_xticks(range(len(splits_order)))
ax.set_xticklabels(split_labels, rotation=15, ha="right")
ax.set_ylabel("ECE (lower = better)")
ax.set_title("Calibration methods comparison — XGBoost (v8.5, t+1, 1 seed)")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(FIG / "fig4_calibration_methods_v85.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig4_calibration_methods_v85.png")

# ============================================================
# FIG 5: Summary table as image
# ============================================================
print("\n[FIG 5] Summary table...")
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
ax.axis("off")
rows = []
for sp in splits_order:
    for model in ["LogReg", "RF", "XGB", "MLP", "PhysMLP"]:
        sub = df_focus[(df_focus["split"] == sp) & (df_focus["model"] == model) &
                       (df_focus["calibrator"] == "Iso")]
        if len(sub):
            r = sub.iloc[0]
            rows.append([f"{sp.replace('_', ' ')[:18]}", model,
                         f"{r['AUROC']:.4f}", f"{r['ECE']:.4f}", f"{r['Brier']:.4f}", f"{r['F1']:.4f}"])

tbl = ax.table(cellText=rows,
               colLabels=["Split", "Model", "AUROC", "ECE (Iso)", "Brier", "F1"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.5)
ax.set_title("v8.5 Smoke Test Results (Iso calibration, t+1, 1 seed)", fontsize=12, pad=20)
plt.savefig(FIG / "fig5_summary_table_v85.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig5_summary_table_v85.png")

# ============================================================
# FIG 6: ΔECE bar chart (IID vs each OOD split, by model)
# ============================================================
print("\n[FIG 6] ΔECE bar chart (Iso)...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
iid_baseline = {m: df_focus[(df_focus["model"] == m) & (df_focus["split"] == "IID_groupblocked") &
                            (df_focus["calibrator"] == "Iso")]["ECE"].mean()
                for m in ["LogReg", "RF", "XGB", "MLP", "PhysMLP"]}

ood_splits = ["Temporal_OOD", "Spatial_OOD_NorthToSouth",
              "Spatial_OOD_SouthToCentral", "Spatial_OOD_CentralSouthToNorth"]
ood_labels = ["Temporal", "Spatial N→S", "Spatial S→C", "Spatial C+S→N"]
x = np.arange(len(ood_splits))
width = 0.15
for i, model in enumerate(["LogReg", "RF", "XGB", "MLP", "PhysMLP"]):
    deltas = []
    for sp in ood_splits:
        sp_ece = df_focus[(df_focus["model"] == model) & (df_focus["split"] == sp) &
                          (df_focus["calibrator"] == "Iso")]["ECE"].mean()
        deltas.append(sp_ece - iid_baseline[model])
    bars = ax.bar(x + (i - 2)*width, deltas, width,
                  label=model, color=COLORS[model])
    for b, d in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width()/2, d + 0.001, f"{d:+.3f}",
                ha="center", fontsize=7, rotation=0)

ax.axhline(0, color='black', linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(ood_labels, rotation=15, ha="right")
ax.set_ylabel("ΔECE = ECE_OOD − ECE_IID (lower = more robust)")
ax.set_title("Calibration degradation under OOD (Iso calibration, v8.5, t+1, 1 seed)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG / "fig6_delta_ece_v85.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig6_delta_ece_v85.png")

print(f"\n✓ All 6 v8.5 figures saved to {FIG}")
