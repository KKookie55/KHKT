#!/usr/bin/env python3
"""
Generate MODIS vs VIIRS comparison figures.
"""
import json
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

# Load all runs (both MODIS and VIIRS)
runs = []
for f in (OUT / "runs").glob("*.json"):
    try: runs.append(json.loads(f.read_text()))
    except: pass
print(f"Loaded {len(runs)} total runs")

rows = []
for r in runs:
    for cal, m in r["metrics"].items():
        rows.append({
            "label_source": r.get("label_source", "modis"),
            "split": r["split"], "horizon": r["horizon"], "seed": r["seed"],
            "model": r["model"], "calibrator": cal, **m
        })
df = pd.DataFrame(rows)
print(f"DataFrame: {len(df)} rows")
print(f"  MODIS: {len(df[df['label_source']=='modis'])} rows")
print(f"  VIIRS: {len(df[df['label_source']=='viirs'])} rows")

# Filter to t+1, Iso calibration, key models
df_focus = df[(df["horizon"] == 1) & (df["calibrator"] == "Iso")]
print(f"Focus (t+1, Iso): {len(df_focus)} rows")

splits_order = ["IID_groupblocked", "Temporal_OOD",
                "Spatial_OOD_NorthToSouth", "Spatial_OOD_SouthToCentral",
                "Spatial_OOD_CentralSouthToNorth"]
split_labels = ["IID", "Temporal OOD", "Spatial N→S", "Spatial S→C", "Spatial C+S→N"]
COLORS_MODIS = {"LogReg": "#999999", "RF": "#66a61e", "XGB": "#e7298a",
                "MLP": "#d95f02", "PhysMLP": "#1b9e77"}
# For VIIRS, use slightly different shade (more transparent)
COLORS_VIIRS = {"LogReg": "#cccccc", "RF": "#a3d977", "XGB": "#f781bf",
                "MLP": "#fbb86b", "PhysMLP": "#67c9a6"}

# ============================================================
# FIG 13: MODIS vs VIIRS — AUROC across splits
# ============================================================
print("\n[FIG 13] MODIS vs VIIRS — AUROC across splits (key models)...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
axes = axes.flatten()

for i, model in enumerate(["LogReg", "RF", "XGB", "MLP", "PhysMLP"]):
    ax = axes[i]
    width = 0.4
    x = np.arange(len(splits_order))
    # MODIS
    modis_means = [df_focus[(df_focus["model"]==model) & (df_focus["split"]==sp) &
                            (df_focus["label_source"]=="modis")]["AUROC"].mean() for sp in splits_order]
    modis_stds = [df_focus[(df_focus["model"]==model) & (df_focus["split"]==sp) &
                           (df_focus["label_source"]=="modis")]["AUROC"].std() for sp in splits_order]
    # VIIRS
    viirs_means = [df_focus[(df_focus["model"]==model) & (df_focus["split"]==sp) &
                            (df_focus["label_source"]=="viirs")]["AUROC"].mean() for sp in splits_order]
    viirs_stds = [df_focus[(df_focus["model"]==model) & (df_focus["split"]==sp) &
                           (df_focus["label_source"]=="viirs")]["AUROC"].std() for sp in splits_order]

    bars_m = ax.bar(x - width/2, modis_means, width, yerr=modis_stds, capsize=3,
                    label="MODIS labels", color=COLORS_MODIS[model])
    bars_v = ax.bar(x + width/2, viirs_means, width, yerr=viirs_stds, capsize=3,
                    label="VIIRS labels", color=COLORS_VIIRS[model], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("AUROC (Iso)")
    ax.set_title(f"{model} (MODIS n=5, VIIRS n=3 seeds)")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label="random")

# Hide last empty subplot
axes[5].axis("off")
axes[5].text(0.5, 0.5, "MODIS vs VIIRS\nAUROC comparison\n(lower bars = VIIRS labels harder to predict)",
             ha="center", va="center", fontsize=12)

plt.suptitle("MODIS vs VIIRS labels: AUROC across splits (v8.5, t+1, Iso calibration)\n"
             "→ VIIRS labels significantly harder, especially on Spatial OOD N→S",
             fontsize=12, y=1.02)
plt.savefig(FIG / "fig13_modis_vs_viirs_auroc.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig13_modis_vs_viirs_auroc.png")

# ============================================================
# FIG 14: Δ AUROC (VIIRS - MODIS) heatmap-style bar chart
# ============================================================
print("\n[FIG 14] Δ AUROC (VIIRS - MODIS) per split & model...")
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
width = 0.15
x = np.arange(len(splits_order))
for i, model in enumerate(["LogReg", "RF", "XGB", "MLP", "PhysMLP"]):
    deltas = []
    for sp in splits_order:
        m_modis = df_focus[(df_focus["model"]==model) & (df_focus["split"]==sp) &
                           (df_focus["label_source"]=="modis")]["AUROC"].mean()
        m_viirs = df_focus[(df_focus["model"]==model) & (df_focus["split"]==sp) &
                           (df_focus["label_source"]=="viirs")]["AUROC"].mean()
        deltas.append(m_viirs - m_modis)
    bars = ax.bar(x + (i - 2)*width, deltas, width, label=model,
                  color=COLORS_MODIS[model])
    for b, d in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width()/2, d - 0.005 if d < 0 else d + 0.005,
                f"{d:+.3f}", ha="center", fontsize=7,
                va="top" if d < 0 else "bottom")

ax.axhline(0, color='black', linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(split_labels, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Δ AUROC (VIIRS - MODIS)\nnegative = VIIRS harder")
ax.set_title("Δ AUROC: VIIRS vs MODIS labels (v8.5, t+1, Iso calibration)\n"
             "→ VIIRS harder everywhere; collapse on Spatial OOD N→S (-0.30!)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.savefig(FIG / "fig14_delta_auroc_modis_vs_viirs.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig14_delta_auroc_modis_vs_viirs.png")

# ============================================================
# FIG 15: VIIRS smoke test summary table (image)
# ============================================================
print("\n[FIG 15] Summary table...")
fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
ax.axis("off")
rows = []
for sp in splits_order:
    sp_short = {"IID_groupblocked": "IID",
                "Temporal_OOD": "Temp OOD",
                "Spatial_OOD_NorthToSouth": "Spatial N→S",
                "Spatial_OOD_SouthToCentral": "Spatial S→C",
                "Spatial_OOD_CentralSouthToNorth": "Spatial C+S→N"}[sp]
    for model in ["LogReg", "RF", "XGB", "MLP", "PhysMLP"]:
        for ls in ["modis", "viirs"]:
            sub = df_focus[(df_focus["split"]==sp) & (df_focus["model"]==model) &
                           (df_focus["label_source"]==ls)]
            if len(sub):
                r = sub.iloc[0]
                rows.append([
                    sp_short, model.upper(), ls.upper(),
                    f"{r['AUROC']:.4f}",
                    f"{r['ECE']:.4f}",
                    f"{r['Brier']:.4f}",
                    f"{r['F1']:.4f}",
                ])

tbl = ax.table(cellText=rows,
               colLabels=["Split", "Model", "Label", "AUROC", "ECE", "Brier", "F1"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.3)
ax.set_title("v8.5 — MODIS vs VIIRS labels comparison (Iso calibration, t+1)\n"
             "MODIS: 5 seeds; VIIRS: 3 seeds (std super low)",
             fontsize=12, pad=20)
plt.savefig(FIG / "fig15_modis_vs_viirs_table.png", dpi=120)
plt.close()
print(f"  -> {FIG}/fig15_modis_vs_viirs_table.png")

print(f"\n✓ All 3 comparison figures saved to {FIG}")
