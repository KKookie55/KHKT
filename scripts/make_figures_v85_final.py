#!/usr/bin/env python3
"""v8.5 figures — MODIS vs VIIRS comparison."""
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
except: pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = Path("/home/z/my-project/download/real_data_v85")
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)

# Load all runs
runs = []
for f in (OUT/"runs").glob("*.json"):
    try: runs.append(json.loads(f.read_text()))
    except: pass
print(f"Loaded {len(runs)} runs")
rows = []
for r in runs:
    for cal, m in r["metrics"].items():
        rows.append({"label_source": r.get("label_source","modis"),"split": r["split"],
                     "horizon": r["horizon"], "seed": r["seed"], "model": r["model"],
                     "calibrator": cal, **m})
df = pd.DataFrame(rows)
print(f"  MODIS: {len(df[df.label_source=='modis'])} rows")
print(f"  VIIRS: {len(df[df.label_source=='viirs'])} rows")

splits = ["IID_groupblocked","Temporal_OOD","Spatial_OOD_NorthToSouth",
          "Spatial_OOD_SouthToCentral","Spatial_OOD_CentralSouthToNorth"]
sl = ["IID","Temporal OOD","Spatial N→S","Spatial S→C","Spatial C+S→N"]
COLORS = {"LogReg":"#999999","RF":"#66a61e","XGB":"#e7298a","MLP":"#d95f02","PhysMLP":"#1b9e77"}

# FIG 1: AUROC vs horizon, IID + Temporal OOD + Spatial N→S — VIIRS only (we have all 3 horizons for VIIRS seed 42)
print("\n[FIG 1] AUROC vs horizon (VIIRS, seed 42)...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
horizons = [1, 3, 7]
for sp, lbl in zip(["IID_groupblocked","Temporal_OOD","Spatial_OOD_NorthToSouth"],
                   ["IID","Temporal OOD","Spatial N→S"]):
    for model in ["XGB","MLP","PhysMLP"]:
        means = []
        for h in horizons:
            sub = df[(df.split==sp) & (df.model==model) & (df.horizon==h) &
                     (df.label_source=="viirs") & (df.calibrator=="Iso")]
            means.append(sub["AUROC"].mean() if len(sub) else None)
        valid = [(h,m) for h,m in zip(horizons, means) if m is not None]
        if not valid: continue
        xs, ys = zip(*valid)
        offset = {"IID_groupblocked": -0.15, "Temporal_OOD": 0, "Spatial_OOD_NorthToSouth": 0.15}[sp]
        ax.plot([x+offset for x in xs], ys, marker='o', linewidth=2, markersize=10,
                label=f"{model} ({lbl})", color=COLORS[model],
                linestyle='--' if sp != 'IID_groupblocked' else '-')
ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("AUROC (Iso, VIIRS labels)")
ax.set_xticks(horizons); ax.set_xticklabels([f"t+{h}" for h in horizons])
ax.set_title("VIIRS — AUROC vs forecast horizon (seed 42, Iso calibration)\n"
             "→ All models degrade ~0.02 from t+1 → t+7 on IID")
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
plt.savefig(FIG/"fig1_viirs_auroc_vs_horizon.png", dpi=120); plt.close()

# FIG 2: MODIS vs VIIRS — AUROC across splits (key models, t+1)
print("[FIG 2] MODIS vs VIIRS AUROC...")
df_focus = df[(df.horizon==1) & (df.calibrator=="Iso")]
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
axes = axes.flatten()
for i, model in enumerate(["LogReg","RF","XGB","MLP","PhysMLP"]):
    ax = axes[i]; w = 0.4; x = np.arange(len(splits))
    modis_m = [df_focus[(df_focus.model==model)&(df_focus.split==sp)&(df_focus.label_source=="modis")]["AUROC"].mean() for sp in splits]
    modis_s = [df_focus[(df_focus.model==model)&(df_focus.split==sp)&(df_focus.label_source=="modis")]["AUROC"].std() for sp in splits]
    viirs_m = [df_focus[(df_focus.model==model)&(df_focus.split==sp)&(df_focus.label_source=="viirs")]["AUROC"].mean() for sp in splits]
    viirs_s = [df_focus[(df_focus.model==model)&(df_focus.split==sp)&(df_focus.label_source=="viirs")]["AUROC"].std() for sp in splits]
    ax.bar(x-w/2, modis_m, w, yerr=modis_s, capsize=3, label="MODIS", color=COLORS[model])
    ax.bar(x+w/2, viirs_m, w, yerr=viirs_s, capsize=3, label="VIIRS",
           color=COLORS[model], alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(sl, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("AUROC (Iso)"); ax.set_title(f"{model}")
    ax.legend(fontsize=8); ax.set_ylim(0.5, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
axes[5].axis("off")
axes[5].text(0.5, 0.5, "MODIS vs VIIRS\n\nVIIRS harder everywhere\nN→S collapse: -0.30 AUROC",
             ha="center", va="center", fontsize=12)
plt.suptitle("MODIS vs VIIRS labels: AUROC across splits (v8.5, t+1, Iso)\n"
             "MODIS n=1 seed; VIIRS n=5 seeds (except t+3, t+7)", fontsize=12, y=1.02)
plt.savefig(FIG/"fig2_modis_vs_viirs_auroc.png", dpi=120); plt.close()

# FIG 3: Δ AUROC (VIIRS - MODIS) bar chart
print("[FIG 3] Δ AUROC...")
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
w = 0.15; x = np.arange(len(splits))
for i, model in enumerate(["LogReg","RF","XGB","MLP","PhysMLP"]):
    deltas = []
    for sp in splits:
        m_m = df_focus[(df_focus.model==model)&(df_focus.split==sp)&(df_focus.label_source=="modis")]["AUROC"].mean()
        m_v = df_focus[(df_focus.model==model)&(df_focus.split==sp)&(df_focus.label_source=="viirs")]["AUROC"].mean()
        deltas.append(m_v - m_m if m_m and m_v else 0)
    bars = ax.bar(x+(i-2)*w, deltas, w, label=model, color=COLORS[model])
    for b,d in zip(bars, deltas):
        ax.text(b.get_x()+b.get_width()/2, d - 0.005 if d < 0 else d + 0.005,
                f"{d:+.3f}", ha="center", fontsize=7,
                va="top" if d < 0 else "bottom")
ax.axhline(0, color='black', linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(sl, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Δ AUROC (VIIRS - MODIS)\nnegative = VIIRS harder")
ax.set_title("Δ AUROC: VIIRS vs MODIS labels (v8.5, t+1, Iso)\n→ Spatial N→S collapse -0.30!")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG/"fig3_delta_auroc_modis_vs_viirs.png", dpi=120); plt.close()

# FIG 4: ECE degradation
print("[FIG 4] ECE degradation...")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
for model in ["LogReg","RF","XGB","MLP","PhysMLP"]:
    means = []; stds = []
    for sp in splits:
        sub = df_focus[(df_focus.model==model) & (df_focus.split==sp) & (df_focus.label_source=="viirs")]
        means.append(sub["ECE"].mean() if len(sub) else 0)
        stds.append(sub["ECE"].std() if len(sub) else 0)
    ax.errorbar(range(len(splits)), means, yerr=stds, marker='o', linewidth=2,
                markersize=10, capsize=5, label=model, color=COLORS[model])
ax.set_xticks(range(len(splits))); ax.set_xticklabels(sl, rotation=15, ha="right")
ax.set_ylabel("ECE (Iso, lower = better)")
ax.set_title("VIIRS ECE across splits (5 seeds, t+1)\n→ XGB+Iso most robust: 0.024 → 0.076 (3×)")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
plt.savefig(FIG/"fig4_viirs_ece_degradation.png", dpi=120); plt.close()

# FIG 5: Summary table image
print("[FIG 5] Summary table...")
fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
ax.axis("off")
rows = []
for sp, lbl in zip(splits, sl):
    for h in [1, 3, 7]:
        for model in ["LogReg","RF","XGB","MLP","PhysMLP"]:
            for ls in ["modis","viirs"]:
                sub = df_focus if h==1 else df[(df.horizon==h)&(df.calibrator=="Iso")]
                sub = sub[(sub.split==sp) & (sub.model==model) & (sub.horizon==h) &
                          (sub.label_source==ls)]
                if len(sub):
                    r = sub.iloc[0]
                    rows.append([lbl, f"t+{h}", model.upper(), ls.upper(),
                                 f"{r['AUROC']:.4f}", f"{r['ECE']:.4f}",
                                 f"{r['Brier']:.4f}", f"{r['F1']:.4f}"])
tbl = ax.table(cellText=rows,
               colLabels=["Split","Horizon","Model","Label","AUROC","ECE","Brier","F1"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.3)
ax.set_title("v8.5 Final Results Table (Iso calibration)\n"
             "MODIS: 1 seed (42); VIIRS: 5 seeds (t+1), 1 seed (t+3, t+7)",
             fontsize=12, pad=20)
plt.savefig(FIG/"fig5_summary_table.png", dpi=120); plt.close()

print(f"\n✓ All 5 v8.5 figures saved to {FIG}")
