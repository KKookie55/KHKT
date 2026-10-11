#!/usr/bin/env python3
"""
VIIRS vs MODIS cross-validation (independent verification of fire labels).

For each cell-day in the v8.5 panel:
- MODIS label: was there a MODIS detection?
- VIIRS label: was there a VIIRS detection?

Compute:
  - MODIS_pos & VIIRS_pos: both detect → strong fire (true positive)
  - MODIS_pos & VIIRS_neg: MODIS detects, VIIRS doesn't → uncertain (small fire? cloud?)
  - MODIS_neg & VIIRS_pos: VIIRS detects, MODIS doesn't → MODIS missed it (false negative)
  - MODIS_neg & VIIRS_neg: neither detects → likely no fire

Output: /home/z/my-project/download/real_data_v85/viirs_validation.json
        /home/z/my-project/download/real_data_v85/figures/fig12_viirs_vs_modis.png
"""
import os, json
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

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/real_data_v85")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

GRID_RES = 0.5
LAT_MIN, LON_MIN = 11.0, 104.0
YEARS = list(range(2018, 2024))

print("=" * 70)
print("VIIRS vs MODIS CROSS-VALIDATION")
print("=" * 70)

# ============================================================
# 1. Load MODIS fires → dedup to cell-day
# ============================================================
print("\n[1/3] Loading MODIS fires → cell-day level...")
modis_dfs = []
for y in YEARS:
    p = DATA / "firms" / f"fires_{y}_daily.csv"
    if p.exists():
        df = pd.read_csv(p)
        modis_dfs.append(df)
        print(f"  MODIS {y}: {len(df):,} raw detections")
modis = pd.concat(modis_dfs, ignore_index=True)
modis["lat"] = np.round((modis["latitude"] - LAT_MIN) / GRID_RES) * GRID_RES + LAT_MIN
modis["lon"] = np.round((modis["longitude"] - LON_MIN) / GRID_RES) * GRID_RES + LON_MIN
modis["date"] = pd.to_datetime(modis["acq_date"])
modis_cell_day = modis[["lat", "lon", "date"]].drop_duplicates()
modis_cell_day["modis_fire"] = 1
print(f"  MODIS unique cell-days: {len(modis_cell_day):,}")

# ============================================================
# 2. Load VIIRS fires → dedup to cell-day
# ============================================================
print("\n[2/3] Loading VIIRS fires → cell-day level...")
viirs_dfs = []
for y in YEARS:
    p = DATA / "viirs" / f"viirs_fires_{y}.csv"
    if p.exists():
        df = pd.read_csv(p)
        viirs_dfs.append(df)
        print(f"  VIIRS {y}: {len(df):,} detections")
viirs = pd.concat(viirs_dfs, ignore_index=True)
viirs["lat"] = np.round((viirs["latitude"] - LAT_MIN) / GRID_RES) * GRID_RES + LAT_MIN
viirs["lon"] = np.round((viirs["longitude"] - LON_MIN) / GRID_RES) * GRID_RES + LON_MIN
viirs["date"] = pd.to_datetime(viirs["acq_date"])
viirs_cell_day = viirs[["lat", "lon", "date"]].drop_duplicates()
viirs_cell_day["viirs_fire"] = 1
print(f"  VIIRS unique cell-days: {len(viirs_cell_day):,}")

# ============================================================
# 3. Cross-tabulate: MODIS vs VIIRS at cell-day level
# ============================================================
print("\n[3/3] Cross-validation: MODIS vs VIIRS cell-day agreement...")

# Merge on (lat, lon, date) — outer join to capture all unique cell-days
merged = modis_cell_day.merge(viirs_cell_day, on=["lat", "lon", "date"], how="outer")
merged["modis_fire"] = merged["modis_fire"].fillna(0).astype(int)
merged["viirs_fire"] = merged["viirs_fire"].fillna(0).astype(int)

# Confusion matrix
n_both     = ((merged["modis_fire"] == 1) & (merged["viirs_fire"] == 1)).sum()
n_modis_only = ((merged["modis_fire"] == 1) & (merged["viirs_fire"] == 0)).sum()
n_viirs_only = ((merged["modis_fire"] == 0) & (merged["viirs_fire"] == 1)).sum()
n_neither  = ((merged["modis_fire"] == 0) & (merged["viirs_fire"] == 0)).sum()

print(f"\n  Total cell-days observed by either satellite: {len(merged):,}")
print(f"\n  ┌────────────────────┬─────────────┬─────────────┐")
print(f"  │                    │ VIIRS=fire   │ VIIRS=no    │")
print(f"  ├────────────────────┼─────────────┼─────────────┤")
print(f"  │ MODIS=fire          │ {n_both:>8,}    │ {n_modis_only:>8,}    │")
print(f"  │ MODIS=no            │ {n_viirs_only:>8,}    │ {n_neither:>8,}    │")
print(f"  └────────────────────┴─────────────┴─────────────┘")

# Compute agreement metrics
modis_pos = n_both + n_modis_only
viirs_pos = n_both + n_viirs_only
print(f"\n  MODIS total fire cell-days:  {modis_pos:,}")
print(f"  VIIRS total fire cell-days:  {viirs_pos:,}")
print(f"  Both detect (agreement):     {n_both:,} ({100*n_both/max(modis_pos,1):.1f}% of MODIS)")
print(f"  MODIS only (VIIRS missed):   {n_modis_only:,} ({100*n_modis_only/max(modis_pos,1):.1f}% of MODIS)")
print(f"  VIIRS only (MODIS missed):   {n_viirs_only:,} ({100*n_viirs_only/max(viirs_pos,1):.1f}% of VIIRS)")

# Jaccard similarity (intersection / union)
jaccard = n_both / (n_both + n_modis_only + n_viirs_only) if (n_both + n_modis_only + n_viirs_only) > 0 else 0
print(f"\n  Jaccard similarity (intersection/union): {jaccard:.4f}")
print(f"  → if =1.0, perfect agreement; if =0.0, no overlap")

# Per-year breakdown
print(f"\n  Per-year agreement:")
print(f"  {'Year':<6} | {'MODIS':<8} | {'VIIRS':<8} | {'Both':<8} | {'MODIS only':<12} | {'VIIRS only':<12} | {'Jaccard':<8}")
print(f"  {'-'*80}")
for y in YEARS:
    m = modis_cell_day[modis_cell_day["date"].dt.year == y]
    v = viirs_cell_day[viirs_cell_day["date"].dt.year == y]
    merged_y = m.merge(v, on=["lat", "lon", "date"], how="outer").fillna(0)
    both_y = ((merged_y["modis_fire"] == 1) & (merged_y["viirs_fire"] == 1)).sum()
    m_only_y = ((merged_y["modis_fire"] == 1) & (merged_y["viirs_fire"] == 0)).sum()
    v_only_y = ((merged_y["modis_fire"] == 0) & (merged_y["viirs_fire"] == 1)).sum()
    jac_y = both_y / max(both_y + m_only_y + v_only_y, 1)
    print(f"  {y:<6} | {len(m):<8,} | {len(v):<8,} | {both_y:<8,} | {m_only_y:<12,} | {v_only_y:<12,} | {jac_y:<8.4f}")

# ============================================================
# Save results
# ============================================================
results = {
    "version": "v8.5_viirs_validation",
    "summary": {
        "n_modis_fire_cell_days": int(modis_pos),
        "n_viirs_fire_cell_days": int(viirs_pos),
        "n_both_detected": int(n_both),
        "n_modis_only": int(n_modis_only),
        "n_viirs_only": int(n_viirs_only),
        "n_neither": int(n_neither),
        "jaccard_similarity": float(jaccard),
        "modis_positive_agreement_pct": float(100 * n_both / max(modis_pos, 1)),
        "viirs_positive_agreement_pct": float(100 * n_both / max(viirs_pos, 1)),
    },
    "interpretation": {
        "n_modis_only": "MODIS detects fire but VIIRS doesn't → uncertain: could be (a) cloud cover in VIIRS, (b) small fire only MODIS saw, (c) VIIRS overpass timing missed it",
        "n_viirs_only": "VIIRS detects fire but MODIS doesn't → MODIS FALSE NEGATIVE: VIIRS (375m, more sensitive) caught a fire MODIS missed",
        "n_both": "strong fire detection — high confidence label",
        "n_neither": "no fire detected by either — likely true negative (but could be both missed)",
    },
    "implications_for_v8.5": {
        "label_quality": "MODIS-only labels include ~XX% uncertain cases (MODIS-only fires). Re-training with VIIRS-confirmed labels would clean up training signal.",
        "false_negative_rate": f"VIIRS detects {n_viirs_only} fire cell-days that MODIS missed. If VIIRS is ground truth, MODIS false negative rate = {100*n_viirs_only/max(n_both+n_viirs_only,1):.1f}% of true fires.",
        "audit_fix": "Use VIIRS as ground truth for evaluation (more sensitive). Use MODIS as training signal (matches operational deployment).",
    }
}
with open(OUT / "viirs_validation.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Saved: {OUT}/viirs_validation.json")

# ============================================================
# Figure: confusion matrix + per-year breakdown
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

# Panel 1: confusion matrix as 2x2 image
ax = axes[0]
cm = np.array([[n_both, n_modis_only],
               [n_viirs_only, n_neither]])
im = ax.imshow(cm, cmap="YlOrRd", aspect="auto", norm=matplotlib.colors.LogNorm())
ax.set_xticks([0, 1]); ax.set_xticklabels(["VIIRS=fire", "VIIRS=no"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["MODIS=fire", "MODIS=no"])
ax.set_xlabel("VIIRS detection")
ax.set_ylabel("MODIS detection")
ax.set_title("MODIS vs VIIRS cell-day agreement\n(log scale)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                fontsize=11, color="black" if cm[i,j] < cm.max()/2 else "white")
plt.colorbar(im, ax=ax, label="cell-days (log)")

# Panel 2: per-year bars
ax = axes[1]
years = list(range(2018, 2024))
both_per_year = []
modis_only_per_year = []
viirs_only_per_year = []
for y in years:
    m = modis_cell_day[modis_cell_day["date"].dt.year == y]
    v = viirs_cell_day[viirs_cell_day["date"].dt.year == y]
    merged_y = m.merge(v, on=["lat", "lon", "date"], how="outer").fillna(0)
    both_per_year.append(((merged_y["modis_fire"] == 1) & (merged_y["viirs_fire"] == 1)).sum())
    modis_only_per_year.append(((merged_y["modis_fire"] == 1) & (merged_y["viirs_fire"] == 0)).sum())
    viirs_only_per_year.append(((merged_y["modis_fire"] == 0) & (merged_y["viirs_fire"] == 1)).sum())

x = np.arange(len(years))
w = 0.27
ax.bar(x - w, both_per_year, w, label="Both detect", color="#1b9e77")
ax.bar(x, modis_only_per_year, w, label="MODIS only", color="#e7298a")
ax.bar(x + w, viirs_only_per_year, w, label="VIIRS only (MODIS missed)", color="#d95f02")
ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_xlabel("Year")
ax.set_ylabel("Cell-days")
ax.set_title("MODIS vs VIIRS detection breakdown per year")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Independent VIIRS validation of MODIS labels (v8.5 audit fix #2)",
             fontsize=11, y=1.02)
plt.savefig(FIG / "fig12_viirs_vs_modis.png", dpi=120)
plt.close()
print(f"✓ Saved: {FIG}/fig12_viirs_vs_modis.png")

# Print summary
print(f"\n{'=' * 70}")
print("VIIRS VALIDATION COMPLETE")
print("=" * 70)
print(f"\nKey numbers:")
print(f"  - {n_both:,} cell-days: BOTH detect (high-confidence fire)")
print(f"  - {n_modis_only:,} cell-days: MODIS only (uncertain)")
print(f"  - {n_viirs_only:,} cell-days: VIIRS only (MODIS false negatives!)")
print(f"  - Jaccard similarity: {jaccard:.4f}")
print(f"\n→ v8.5 MODIS labels have ~{100*n_viirs_only/max(n_both+n_viirs_only,1):.0f}% false negative rate")
print(f"  (VIIRS sees fires MODIS misses)")
print(f"\n→ For v8.6, can train on VIIRS+MODIS union labels for cleaner signal.")
