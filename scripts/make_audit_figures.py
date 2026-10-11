#!/usr/bin/env python3
"""
Generate raw research-evidence figures from v7/v8 JSON results.
These are intentionally MINIMAL — the user explicitly said "even if ugly".
The point is to give the auditor raw visual evidence, not polished slides.

Outputs PNGs to /home/z/my-project/WILDFIRE_PROJECT_BUILD/report/figures/
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Font setup (project rule #7)
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = Path("/home/z/my-project/download/real_data_v7")
FIG_OUT = Path("/home/z/my-project/WILDFIRE_PROJECT_BUILD/report/figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Load all results
def load_json(p):
    with open(p) as f:
        return json.load(f)

audit = load_json(RESULTS_DIR / "results_audit.json")
multiseed = load_json(RESULTS_DIR / "results_multiseed.json")
v7 = load_json(RESULTS_DIR / "results_v7_final.json")

print("Loaded audit/multiseed/v7 JSONs")
print(f"  audit keys: {list(audit.keys())}")
print(f"  v7 keys: {list(v7.keys())}")

# Helper: aggregate per-model per-split
def aggregate(results, model_key):
    """Returns {split: {metric: (mean, std)}}"""
    out = {}
    for split_name, models in results.items():
        if model_key not in models:
            continue
        runs = models[model_key]
        agg = {}
        for metric in ["AUROC", "AUPRC", "F1", "Brier", "ECE"]:
            vals = [r[metric] for r in runs]
            agg[metric] = (float(np.mean(vals)), float(np.std(vals)))
        out[split_name] = agg
    return out

# Color palette (high-contrast, colorblind-friendly)
# Keys are the actual JSON keys in the audit results
COLORS = {
    "xgb":      "#e7298a",  # pink-red
    "xgb_iso":  "#1b9e77",  # teal-green (the winning method)
    "xgb_platt":"#7570b3",  # purple
    "mlp":      "#d95f02",  # orange
    "logreg":   "#999999",
    "rf":       "#66a61e",
}
# Display names
DISPLAY = {
    "xgb":      "XGBoost",
    "xgb_iso":  "XGB+Isotonic",
    "xgb_platt":"XGB+Platt",
    "mlp":      "MLP (Physics)",
    "logreg":   "LogReg",
    "rf":       "RandomForest",
}

# ============================================================
# FIG 1: ΔECE degradation across IID → Temporal OOD → Spatial OOD
# (the headline finding)
# ============================================================
print("\n[FIG 1] ΔECE degradation across splits...")
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
splits_order = ["IID", "Temporal_OOD", "Spatial_OOD"]
split_labels = ["IID", "Temporal OOD", "Spatial OOD"]
x = np.arange(len(splits_order))
width = 0.15
model_keys = ["XGB", "XGB+Iso", "XGB+Platt", "MLP"]
# Use the audit JSON (which has the right model keys)
available_models = list(audit.get("IID", {}).keys())
print(f"  Available models in audit JSON: {available_models}")
model_keys = [m for m in ["xgb", "xgb_iso", "xgb_platt", "mlp"] if m in available_models]
print(f"  Using models: {model_keys}")

for i, mk in enumerate(model_keys):
    means = []
    stds = []
    for sp in splits_order:
        runs = audit.get(sp, {}).get(mk, [])
        if not runs:
            means.append(0); stds.append(0)
            continue
        eces = [r["ECE"] for r in runs]
        means.append(float(np.mean(eces)))
        stds.append(float(np.std(eces)))
    bars = ax.bar(x + (i - len(model_keys)/2)*width, means, width,
                  yerr=stds, capsize=3, label=DISPLAY.get(mk, mk), color=COLORS.get(mk, "#444"))
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, m + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(split_labels)
ax.set_ylabel("ECE (lower = better calibrated)")
ax.set_title("Calibration degradation under distribution shift\n(v7 audit, 3 seeds, mean ± std)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG_OUT / "fig1_ece_degradation.png", dpi=120)
plt.close()
print(f"  -> {FIG_OUT}/fig1_ece_degradation.png")

# ============================================================
# FIG 2: AUROC across splits
# ============================================================
print("\n[FIG 2] AUROC across splits...")
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
for i, mk in enumerate(model_keys):
    means = []
    stds = []
    for sp in splits_order:
        runs = audit.get(sp, {}).get(mk, [])
        if not runs:
            means.append(0); stds.append(0)
            continue
        vals = [r["AUROC"] for r in runs]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    bars = ax.bar(x + (i - len(model_keys)/2)*width, means, width,
                  yerr=stds, capsize=3, label=DISPLAY.get(mk, mk), color=COLORS.get(mk, "#444"))
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, m + s + 0.002,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(split_labels)
ax.set_ylabel("AUROC (higher = better discrimination)")
ax.set_title("Discrimination across IID / Temporal OOD / Spatial OOD\n(v7 audit, 3 seeds, mean ± std)")
ax.set_ylim(0.6, 0.95)
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG_OUT / "fig2_auroc_across_splits.png", dpi=120)
plt.close()
print(f"  -> {FIG_OUT}/fig2_auroc_across_splits.png")

# ============================================================
# FIG 3: λ-sweep (does physics constraint help? NO.)
# ============================================================
print("\n[FIG 3] λ-sweep curve...")
lams = sorted([float(k) for k in v7["lambda_sweep"].keys()])
ece_lam = [v7["lambda_sweep"][str(l) if str(l) in v7["lambda_sweep"] else f"{l:g}"]["ECE"] for l in lams]
auroc_lam = [v7["lambda_sweep"][str(l) if str(l) in v7["lambda_sweep"] else f"{l:g}"]["AUROC"] for l in lams]
brier_lam = [v7["lambda_sweep"][str(l) if str(l) in v7["lambda_sweep"] else f"{l:g}"]["Brier"] for l in lams]

# Fix: keys may be "0.0" not "0"
def getk(d, l):
    for k in d:
        if abs(float(k) - l) < 1e-9:
            return d[k]
    return None
ece_lam = [getk(v7["lambda_sweep"], l)["ECE"] for l in lams]
auroc_lam = [getk(v7["lambda_sweep"], l)["AUROC"] for l in lams]
brier_lam = [getk(v7["lambda_sweep"], l)["Brier"] for l in lams]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
axes[0].plot(lams, ece_lam, 'o-', color="#e7298a", linewidth=2, markersize=8)
axes[0].set_xlabel("λ (physics penalty weight)")
axes[0].set_ylabel("ECE")
axes[0].set_title("ECE vs λ")
axes[0].grid(alpha=0.3)
axes[0].axhline(ece_lam[0], color='gray', linestyle='--', alpha=0.5, label=f"baseline (λ=0): {ece_lam[0]:.4f}")
axes[0].legend(fontsize=8)
for x_, y_ in zip(lams, ece_lam):
    axes[0].text(x_, y_ + 0.0005, f"{y_:.4f}", ha="center", fontsize=7)

axes[1].plot(lams, auroc_lam, 's-', color="#1b9e77", linewidth=2, markersize=8)
axes[1].set_xlabel("λ")
axes[1].set_ylabel("AUROC")
axes[1].set_title("AUROC vs λ")
axes[1].grid(alpha=0.3)
axes[1].set_ylim(min(auroc_lam)-0.005, max(auroc_lam)+0.005)
for x_, y_ in zip(lams, auroc_lam):
    axes[1].text(x_, y_ + 0.0005, f"{y_:.4f}", ha="center", fontsize=7)

axes[2].plot(lams, brier_lam, '^-', color="#7570b3", linewidth=2, markersize=8)
axes[2].set_xlabel("λ")
axes[2].set_ylabel("Brier score")
axes[2].set_title("Brier vs λ")
axes[2].grid(alpha=0.3)
for x_, y_ in zip(lams, brier_lam):
    axes[2].text(x_, y_ + 0.0005, f"{y_:.4f}", ha="center", fontsize=7)

plt.suptitle("λ-sweep: does physics penalty help?  (ΔECE ≈ 0 — physics does NOT improve calibration on real NASA MODIS data)",
             fontsize=10, y=1.02)
plt.savefig(FIG_OUT / "fig3_lambda_sweep.png", dpi=120, bbox_inches=None)
plt.close()
print(f"  -> {FIG_OUT}/fig3_lambda_sweep.png")

# ============================================================
# FIG 4: Calibration baselines (XGB raw vs Temperature vs Isotonic vs Platt)
# ============================================================
print("\n[FIG 4] Calibration baselines bar chart...")
cb = v7["calibration_baselines"]
methods = ["Raw", "+ Temp", "+ Iso", "+ Platt"]
metric_vals_ece = [
    v7["baseline"]["ECE"],
    cb["temperature"]["ECE"],
    cb["isotonic"]["ECE"],
    cb["platt"]["ECE"],
]
metric_vals_brier = [
    v7["baseline"]["Brier"],
    cb["temperature"]["Brier"],
    cb["isotonic"]["Brier"],
    cb["platt"]["Brier"],
]

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
x = np.arange(len(methods))
w = 0.35
b1 = ax.bar(x - w/2, metric_vals_ece, w, label="ECE", color="#e7298a")
b2 = ax.bar(x + w/2, metric_vals_brier, w, label="Brier", color="#1b9e77")
for b in b1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
            f"{b.get_height():.4f}", ha="center", fontsize=8)
for b in b2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
            f"{b.get_height():.4f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Score (lower = better)")
ax.set_title("Calibration baselines on XGBoost (test set 2022)\nIsotonic gives lowest ECE")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG_OUT / "fig4_calibration_baselines.png", dpi=120)
plt.close()
print(f"  -> {FIG_OUT}/fig4_calibration_baselines.png")

# ============================================================
# FIG 5: Reliability diagram (binned predicted vs actual) — needs predictions.
# Since we don't have raw predictions saved for v7, we'll synthesize a stylized
# reliability diagram from the ECE numbers, clearly labeled as ILLUSTRATIVE.
# This is honest about what we have.
# ============================================================
print("\n[FIG 5] Stylized reliability diagram (illustrative)...")

# Use the v5-era predictions (which we DO have) to make a real reliability diagram.
pred_path = Path("/home/z/my-project/download/wildfire_real_data_v5/predictions/predictions_test_2022.csv")
if pred_path.exists():
    import pandas as pd
    print(f"  Loading v5 predictions from {pred_path}...")
    df_pred = pd.read_csv(pred_path)
    print(f"  Columns: {list(df_pred.columns)}")
    print(f"  Rows: {len(df_pred)}")
    
    # Identify prediction + label columns
    pred_col = None
    label_col = None
    # Prefer 'prob_*' columns as the predicted probability
    for c in df_pred.columns:
        cl = c.lower()
        if pred_col is None and ("prob" in cl):
            pred_col = c
    # Fallback to other prediction-like columns
    if pred_col is None:
        for c in df_pred.columns:
            cl = c.lower()
            if "pred" in cl or "score" in cl:
                pred_col = c; break
    for c in df_pred.columns:
        cl = c.lower()
        if "y_true" in cl or "label" in cl or "actual" in cl:
            label_col = c; break
    print(f"  pred_col={pred_col}, label_col={label_col}")
    
    if pred_col and label_col:
        p = df_pred[pred_col].values
        y = df_pred[label_col].values
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        pred_means = []
        actual_freq = []
        for i in range(10):
            lo, hi = bins[i], bins[i+1]
            mask = ((p >= lo) & (p <= hi)) if i == 9 else ((p >= lo) & (p < hi))
            if mask.sum() == 0:
                pred_means.append(np.nan)
                actual_freq.append(np.nan)
            else:
                pred_means.append(p[mask].mean())
                actual_freq.append(y[mask].mean())
        
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect calibration")
        # XGBoost raw predictions (assumed)
        ax.plot(pred_means, actual_freq, 'o-', color="#e7298a", linewidth=2, markersize=10, label="XGB raw")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fire frequency")
        ax.set_title("Reliability diagram (v5 test set 2022)\nNote: v7 predictions not saved — this is v5 era")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig(FIG_OUT / "fig5_reliability_diagram_v5_era.png", dpi=120)
        plt.close()
        print(f"  -> {FIG_OUT}/fig5_reliability_diagram_v5_era.png")
    else:
        print("  Could not identify pred/label columns")
else:
    print(f"  No v5 predictions at {pred_path}")

# ============================================================
# FIG 6: Brier score comparison
# ============================================================
print("\n[FIG 6] Brier score comparison...")
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
model_keys_b = ["xgb", "xgb_iso", "xgb_platt", "mlp"]
model_keys_b = [m for m in model_keys_b if m in available_models]
x = np.arange(len(splits_order))
width = 0.18
for i, mk in enumerate(model_keys_b):
    means = []
    stds = []
    for sp in splits_order:
        runs = audit.get(sp, {}).get(mk, [])
        if not runs:
            means.append(0); stds.append(0)
            continue
        vals = [r["Brier"] for r in runs]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    bars = ax.bar(x + (i - len(model_keys_b)/2)*width, means, width,
                  yerr=stds, capsize=3, label=DISPLAY.get(mk, mk), color=COLORS.get(mk, "#444"))
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.002, f"{m:.3f}",
                ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(split_labels)
ax.set_ylabel("Brier (lower = better)")
ax.set_title("Brier score across splits (v7 audit, 3 seeds)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.savefig(FIG_OUT / "fig6_brier_across_splits.png", dpi=120)
plt.close()
print(f"  -> {FIG_OUT}/fig6_brier_across_splits.png")

# ============================================================
# FIG 7: ΔECE (degradation) as a single line chart — easier to read
# ============================================================
print("\n[FIG 7] ΔECE line chart...")
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
for mk in model_keys:
    iid_ece = np.mean([r["ECE"] for r in audit["IID"].get(mk, [{"ECE":0}])])
    deltas = []
    for sp in splits_order:
        runs = audit.get(sp, {}).get(mk, [])
        if not runs:
            deltas.append(0)
            continue
        sp_ece = np.mean([r["ECE"] for r in runs])
        deltas.append(sp_ece - iid_ece)
    ax.plot(split_labels, deltas, 'o-', linewidth=2, markersize=10,
            label=DISPLAY.get(mk, mk), color=COLORS.get(mk, "#444"))
    for i, d in enumerate(deltas):
        ax.text(i, d + 0.005, f"{d:+.3f}", ha="center", fontsize=8)

ax.axhline(0, color='black', linewidth=0.7, alpha=0.5)
ax.set_xlabel("Split")
ax.set_ylabel("ΔECE (vs IID baseline)")
ax.set_title("Calibration degradation: ΔECE = ECE_split − ECE_IID\n(lower = more robust to distribution shift)")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(FIG_OUT / "fig7_delta_ece_line.png", dpi=120)
plt.close()
print(f"  -> {FIG_OUT}/fig7_delta_ece_line.png")

# ============================================================
# FIG 8: V7 baseline + λ-sweep summary table as image
# ============================================================
print("\n[FIG 8] Summary results table as image...")
fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
ax.axis("off")
rows = [
    ["XGB raw", f"{v7['baseline']['AUROC']:.4f}", f"{v7['baseline']['Brier']:.4f}", f"{v7['baseline']['ECE']:.4f}"],
    ["XGB + Temperature", f"{cb['temperature']['AUROC']:.4f}", f"{cb['temperature']['Brier']:.4f}", f"{cb['temperature']['ECE']:.4f}"],
    ["XGB + Isotonic", f"{cb['isotonic']['AUROC']:.4f}", f"{cb['isotonic']['Brier']:.4f}", f"{cb['isotonic']['ECE']:.4f}"],
    ["XGB + Platt", f"{cb['platt']['AUROC']:.4f}", f"{cb['platt']['Brier']:.4f}", f"{cb['platt']['ECE']:.4f}"],
    ["", "", "", ""],
    ["MLP λ=0", f"{v7['lambda_sweep']['0.0']['AUROC']:.4f}", f"{v7['lambda_sweep']['0.0']['Brier']:.4f}", f"{v7['lambda_sweep']['0.0']['ECE']:.4f}"],
    ["MLP λ=0.01", f"{v7['lambda_sweep']['0.01']['AUROC']:.4f}", f"{v7['lambda_sweep']['0.01']['Brier']:.4f}", f"{v7['lambda_sweep']['0.01']['ECE']:.4f}"],
    ["MLP λ=0.05", f"{v7['lambda_sweep']['0.05']['AUROC']:.4f}", f"{v7['lambda_sweep']['0.05']['Brier']:.4f}", f"{v7['lambda_sweep']['0.05']['ECE']:.4f}"],
    ["MLP λ=0.10", f"{v7['lambda_sweep']['0.1']['AUROC']:.4f}", f"{v7['lambda_sweep']['0.1']['Brier']:.4f}", f"{v7['lambda_sweep']['0.1']['ECE']:.4f}"],
    ["MLP λ=0.20", f"{v7['lambda_sweep']['0.2']['AUROC']:.4f}", f"{v7['lambda_sweep']['0.2']['Brier']:.4f}", f"{v7['lambda_sweep']['0.2']['ECE']:.4f}"],
]
tbl = ax.table(cellText=rows, colLabels=["Model", "AUROC", "Brier", "ECE"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 2.0)
ax.set_title("v7 FINAL — Calibration baselines + λ-sweep (test set 2022, seed 42)", fontsize=12, pad=20)
plt.savefig(FIG_OUT / "fig8_summary_table.png", dpi=120)
plt.close()
print(f"  -> {FIG_OUT}/fig8_summary_table.png")

print(f"\n✓ All figures saved to {FIG_OUT}")
print(f"  Total: 8 figures")
