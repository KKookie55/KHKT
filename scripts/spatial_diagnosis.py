#!/usr/bin/env python3
"""
DIAGNOSE SPATIAL OOD FAILURE (N→S collapse: AUROC 0.84 → 0.58)

Tests 5 hypotheses:
  H1: Covariate shift — feature distributions differ N vs S
  H2: Concept shift — P(fire|weather) differs N vs S
  H3: Label/detection bias — VIIRS detects differently by region
  H4: Unobserved spatial variables — latitude band patterns
  H5: Spatial leakage — IID too easy due to autocorrelated labels

Output: /home/z/my-project/download/real_data_v85/spatial_diagnosis.json
        /home/z/my-project/download/real_data_v85/figures/fig_diag_*.png
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
warnings.filterwarnings("ignore")
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except: pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA = Path("/home/z/my-project/upload/wildfire_data/wildfire_data")
OUT = Path("/home/z/my-project/download/real_data_v85")
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)

print("="*70); print("SPATIAL OOD DIAGNOSIS"); print("="*70)
print("Why does N→S AUROC collapse 0.84 → 0.58 with VIIRS labels?")

# Load panel + VIIRS panel
panel_modis = pd.read_parquet(OUT / "panel_v85.parquet")
panel_viirs = pd.read_parquet(OUT / "panel_v85_viirs.parquet")
panel_modis = panel_modis.sort_values(["lat","lon","date"]).reset_index(drop=True)
panel_viirs = panel_viirs.sort_values(["lat","lon","date"]).reset_index(drop=True)

# Define North (lat>=15) and South (lat<13) - matches spatial OOD split
def get_NS(panel):
    n = panel[panel["lat"] >= 15.0].copy()
    s = panel[panel["lat"] < 13.0].copy()
    return n, s

n_modis, s_modis = get_NS(panel_modis)
n_viirs, s_viirs = get_NS(panel_viirs)
print(f"\n  North (lat≥15):  {len(n_modis):,} rows")
print(f"  South (lat<13):  {len(s_modis):,} rows")

results = {}

# ============================================================
# H1: COVARIATE SHIFT — feature distribution comparison
# ============================================================
print("\n[H1] COVARIATE SHIFT — feature distributions N vs S...")
key_features = ["t2m","rh","wind","precip","doy","precip_roll30_sum","days_since_rain","heat_index","fwi_compound"]
h1_results = {}
for f in key_features:
    if f not in n_modis.columns: continue
    n_vals = n_modis[f].dropna().values
    s_vals = s_modis[f].dropna().values
    if len(n_vals) == 0 or len(s_vals) == 0: continue
    # KS test
    ks_stat, ks_p = stats.ks_2samp(n_vals, s_vals)
    # Wasserstein distance (Earth Mover's Distance)
    w_dist = stats.wasserstein_distance(n_vals, s_vals)
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(n_vals) + np.var(s_vals)) / 2)
    cohens_d = abs(np.mean(n_vals) - np.mean(s_vals)) / (pooled_std + 1e-9)
    h1_results[f] = {
        "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p),
        "wasserstein_distance": float(w_dist),
        "cohens_d": float(cohens_d),
        "north_mean": float(np.mean(n_vals)), "south_mean": float(np.mean(s_vals)),
        "north_std": float(np.std(n_vals)), "south_std": float(np.std(s_vals)),
    }
    sig = "***" if ks_p < 0.001 else ("**" if ks_p < 0.01 else ("*" if ks_p < 0.05 else "ns"))
    print(f"  {f:25s}: KS={ks_stat:.3f} (p={ks_p:.2e}) {sig}  d={cohens_d:.3f}  "
          f"N={np.mean(n_vals):.2f}, S={np.mean(s_vals):.2f}")
results["H1_covariate_shift"] = h1_results

# ============================================================
# H2: CONCEPT SHIFT — P(fire|weather) differs N vs S
# ============================================================
print("\n[H2] CONCEPT SHIFT — P(fire|weather) differs N vs S...")
# Bin weather, compute fire rate per bin in N vs S
def fire_rate_by_bin(n_df, s_df, feature, n_bins=10):
    """Bin by feature quantiles, compute fire rate per bin in N vs S."""
    # Use N quantiles for binning (so bins are comparable)
    n_vals = n_df[feature].dropna()
    if len(n_vals) == 0: return None
    bins = np.linspace(n_vals.min(), n_vals.max(), n_bins+1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    n_rates = []; s_rates = []; n_counts = []; s_counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        n_mask = (n_df[feature] >= lo) & (n_df[feature] < (hi if i < n_bins-1 else hi+1e-9))
        s_mask = (s_df[feature] >= lo) & (s_df[feature] < (hi if i < n_bins-1 else hi+1e-9))
        n_sub = n_df[n_mask]
        s_sub = s_df[s_mask]
        n_rates.append(n_sub["fire"].mean() if len(n_sub) > 0 else np.nan)
        s_rates.append(s_sub["fire"].mean() if len(s_sub) > 0 else np.nan)
        n_counts.append(len(n_sub))
        s_counts.append(len(s_sub))
    return bin_centers, np.array(n_rates), np.array(s_rates), n_counts, s_counts

h2_results = {}
fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
axes = axes.flatten()
for i, f in enumerate(key_features[:9]):
    if f not in n_modis.columns: continue
    res = fire_rate_by_bin(n_modis, s_modis, f, n_bins=10)
    if res is None: continue
    bin_centers, n_rates, s_rates, n_counts, s_counts = res
    ax = axes[i]
    ax.plot(bin_centers, n_rates, 'o-', label=f"North (n={sum(n_counts):,})", color="#1b9e77", linewidth=2, markersize=8)
    ax.plot(bin_centers, s_rates, 's-', label=f"South (n={sum(s_counts):,})", color="#d95f02", linewidth=2, markersize=8)
    ax.set_xlabel(f); ax.set_ylabel("P(fire | feature)")
    ax.set_title(f"{f}: P(fire|{f}) N vs S")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # Correlation between N and S rates (high → no concept shift)
    valid = ~np.isnan(n_rates) & ~np.isnan(s_rates)
    if valid.sum() > 2:
        corr = np.corrcoef(n_rates[valid], s_rates[valid])[0,1]
        ax.text(0.05, 0.95, f"r={corr:.3f}", transform=ax.transAxes, fontsize=9, va='top')
        h2_results[f] = {"correlation_N_S": float(corr),
                          "n_rates": n_rates.tolist(), "s_rates": s_rates.tolist(),
                          "bin_centers": bin_centers.tolist()}
plt.suptitle("H2: Concept shift test — P(fire|feature) in North vs South (MODIS labels)\n"
             "If curves match → no concept shift; if diverge → concept shift", y=1.02)
plt.savefig(FIG/"fig_diag_h2_concept_shift.png", dpi=110); plt.close()
results["H2_concept_shift"] = h2_results
print(f"  Concept shift figure saved (curves match? correlation >0.9 = no concept shift)")

# ============================================================
# H3: LABEL/DETECTION BIAS — MODIS vs VIIRS detection by region
# ============================================================
print("\n[H3] LABEL BIAS — MODIS vs VIIRS detection ratio by region...")
h3_results = {}
for region_name, n_df, s_df in [("North", n_modis, n_viirs), ("South", s_modis, s_viirs)]:
    modis_fire = int(n_df["fire"].sum())
    viirs_fire = int(s_df["fire"].sum())
    n_total = len(n_df)
    # Agreement
    both = ((n_df["fire"]==1) & (s_df["fire"]==1)).sum()
    modis_only = ((n_df["fire"]==1) & (s_df["fire"]==0)).sum()
    viirs_only = ((n_df["fire"]==0) & (s_df["fire"]==1)).sum()
    neither = ((n_df["fire"]==0) & (s_df["fire"]==0)).sum()
    # Detection ratio
    modis_rate = modis_fire / n_total * 100
    viirs_rate = viirs_fire / n_total * 100
    ratio = modis_rate / max(viirs_rate, 0.001)
    h3_results[region_name] = {
        "n_total": n_total,
        "modis_fire": modis_fire, "viirs_fire": viirs_fire,
        "modis_rate_pct": modis_rate, "viirs_rate_pct": viirs_rate,
        "ratio_modis_to_viirs": float(ratio),
        "both_detect": int(both), "modis_only": int(modis_only),
        "viirs_only": int(viirs_only), "neither": int(neither),
        "jaccard": float(both / max(both + modis_only + viirs_only, 1)),
    }
    print(f"  {region_name}: MODIS rate={modis_rate:.2f}%, VIIRS rate={viirs_rate:.2f}%, ratio={ratio:.2f}")
    print(f"    Both={both:,}, MODIS-only={modis_only:,}, VIIRS-only={viirs_only:,}, Jaccard={h3_results[region_name]['jaccard']:.3f}")
results["H3_label_bias"] = h3_results

# ============================================================
# H4: UNOBSERVED SPATIAL VARIABLES — latitude band patterns
# ============================================================
print("\n[H4] UNOBSERVED SPATIAL VARIABLES — lat-band patterns...")
# Compute fire prevalence per lat (use 0.5° bands)
lat_bands = np.arange(11.0, 21.5, 0.5)
h4_results = {"lat_bands": lat_bands.tolist(), "modis_rate_per_lat": [], "viirs_rate_per_lat": []}
for lb in lat_bands:
    m_mask = (panel_modis["lat"] >= lb) & (panel_modis["lat"] < lb + 0.5)
    v_mask = (panel_viirs["lat"] >= lb) & (panel_viirs["lat"] < lb + 0.5)
    h4_results["modis_rate_per_lat"].append(float(panel_modis[m_mask]["fire"].mean()) if m_mask.sum() > 0 else 0)
    h4_results["viirs_rate_per_lat"].append(float(panel_viirs[v_mask]["fire"].mean()) if v_mask.sum() > 0 else 0)

fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
ax.plot(lat_bands, h4_results["modis_rate_per_lat"], 'o-', label="MODIS", color="#e7298a", linewidth=2, markersize=8)
ax.plot(lat_bands, h4_results["viirs_rate_per_lat"], 's-', label="VIIRS", color="#1b9e77", linewidth=2, markersize=8)
ax.axvline(15.0, color='red', linestyle='--', alpha=0.5, label="Train/test boundary (N→S)")
ax.axvline(13.0, color='orange', linestyle='--', alpha=0.5, label="Test boundary")
ax.set_xlabel("Latitude (degrees N)"); ax.set_ylabel("Fire prevalence (fraction)")
ax.set_title("H4: Fire prevalence by latitude band\n→ Reveals unobserved spatial structure (NDVI/elevation/land cover)")
ax.legend(); ax.grid(alpha=0.3)
# Annotate regions
ax.axvspan(15, 21, alpha=0.1, color='blue', label='Train (North)')
ax.axvspan(11, 13, alpha=0.1, color='red', label='Test (South)')
plt.savefig(FIG/"fig_diag_h4_lat_band.png", dpi=110); plt.close()
results["H4_unobserved_spatial"] = h4_results
print(f"  Lat-band pattern saved. North fire rate vs South fire rate:")
n_modis_rate = np.mean(h4_results["modis_rate_per_lat"][-8:])  # lat >= 17
s_modis_rate = np.mean(h4_results["modis_rate_per_lat"][:4])  # lat 11-13
print(f"  MODIS: N (lat≥17)={n_modis_rate:.4f}, S (lat 11-13)={s_modis_rate:.4f}")
n_viirs_rate = np.mean(h4_results["viirs_rate_per_lat"][-8:])
s_viirs_rate = np.mean(h4_results["viirs_rate_per_lat"][:4])
print(f"  VIIRS: N (lat≥17)={n_viirs_rate:.4f}, S (lat 11-13)={s_viirs_rate:.4f}")

# ============================================================
# H5: SPATIAL LEAKAGE — IID too easy?
# ============================================================
print("\n[H5] SPATIAL LEAKAGE — IID autocorrelation analysis...")
# Sample a date, compute fire label autocorrelation vs spatial distance
sample_dates = panel_modis["date"].drop_duplicates().sample(n=30, random_state=42)
distances = []
labels_match = []
labels_both_zero = []
labels_diff = []
for d in sample_dates:
    day_data = panel_modis[panel_modis["date"] == d][["lat","lon","fire"]].copy()
    if len(day_data) < 100: continue
    # Sample pairs
    n_pairs = 200
    idx_a = np.random.choice(len(day_data), n_pairs, replace=True)
    idx_b = np.random.choice(len(day_data), n_pairs, replace=True)
    la, lo, fa = day_data.iloc[idx_a].lat.values, day_data.iloc[idx_a].lon.values, day_data.iloc[idx_a].fire.values
    lb, lo_b, fb = day_data.iloc[idx_b].lat.values, day_data.iloc[idx_b].lon.values, day_data.iloc[idx_b].fire.values
    # Spatial distance in degrees
    dist = np.sqrt((la-lb)**2 + (lo-lo_b)**2)
    distances.extend(dist.tolist())
    labels_match.extend((fa == fb).astype(int).tolist())
    labels_both_zero.extend(((fa == 0) & (fb == 0)).astype(int).tolist())
    labels_diff.extend((fa != fb).astype(int).tolist())

distances = np.array(distances); labels_match = np.array(labels_match)
# Bin by distance
max_dist = np.max(distances)
bins = np.linspace(0, max_dist, 11)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
match_rate_per_bin = []
for i in range(10):
    lo, hi = bins[i], bins[i+1]
    mask = (distances >= lo) & (distances < hi)
    match_rate_per_bin.append(labels_match[mask].mean() if mask.sum() > 0 else np.nan)
h5_results = {
    "distance_bins": bin_centers.tolist(),
    "label_match_rate": [float(x) for x in match_rate_per_bin],
    "max_distance_deg": float(max_dist),
    "n_pairs_total": len(distances),
}
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
ax.plot(bin_centers, match_rate_per_bin, 'o-', color="#7570b3", linewidth=2, markersize=10)
ax.set_xlabel("Spatial distance between cell pairs (degrees)")
ax.set_ylabel("Label agreement rate (fraction)")
ax.set_title("H5: Spatial autocorrelation of fire labels\n→ High agreement at short distance = IID may be too easy (spatial leakage)")
ax.grid(alpha=0.3)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label="Random")
plt.savefig(FIG/"fig_diag_h5_spatial_autocorr.png", dpi=110); plt.close()
results["H5_spatial_leakage"] = h5_results
print(f"  Spatial autocorrelation: at dist=0°, agreement={match_rate_per_bin[0]:.3f}; at max dist, agreement={match_rate_per_bin[-1]:.3f}")
print(f"  → High agreement at low distance confirms spatial autocorrelation (IID is biased easy)")

# ============================================================
# DIAGNOSIS SUMMARY
# ============================================================
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

# H1 verdict
h1_strong = sum(1 for f in h1_results.values() if f["ks_pvalue"] < 0.001 and f["cohens_d"] > 0.2)
print(f"\nH1 (Covariate shift): {h1_strong}/{len(h1_results)} features show significant shift (p<0.001, d>0.2)")
print(f"  → {'STRONG' if h1_strong >= 3 else 'WEAK'} covariate shift detected")

# H2 verdict
h2_avg_corr = np.mean([r["correlation_N_S"] for r in h2_results.values()])
print(f"\nH2 (Concept shift): avg correlation between N and S P(fire|feature) = {h2_avg_corr:.3f}")
print(f"  → {'NO concept shift' if h2_avg_corr > 0.9 else 'CONCEPT SHIFT detected' if h2_avg_corr > 0.5 else 'STRONG concept shift'}")

# H3 verdict
print(f"\nH3 (Label bias):")
for r, d in h3_results.items():
    print(f"  {r}: MODIS/VIIRS ratio = {d['ratio_modis_to_viirs']:.2f}, Jaccard = {d['jaccard']:.3f}")
n_ratio = h3_results["North"]["ratio_modis_to_viirs"]
s_ratio = h3_results["South"]["ratio_modis_to_viirs"]
ratio_diff = abs(n_ratio - s_ratio)
print(f"  → MODIS/VIIRS ratio difference N vs S = {ratio_diff:.3f}")
print(f"  → {'BIAS detected' if ratio_diff > 0.3 else 'No strong regional bias'}")

# H4 verdict
n_pattern = h4_results["modis_rate_per_lat"][-8:]  # lat >= 17
s_pattern = h4_results["modis_rate_per_lat"][:4]
print(f"\nH4 (Unobserved spatial):")
print(f"  North fire rate (lat≥17): MODIS {np.mean(n_pattern):.4f}, VIIRS {np.mean(h4_results['viirs_rate_per_lat'][-8:]):.4f}")
print(f"  South fire rate (lat 11-13): MODIS {np.mean(s_pattern):.4f}, VIIRS {np.mean(h4_results['viirs_rate_per_lat'][:4]):.4f}")
n_s_diff = abs(np.mean(n_pattern) - np.mean(s_pattern))
print(f"  → {'STRONG regional difference (unobserved spatial vars matter)' if n_s_diff > 0.05 else 'Similar rates'}")

# H5 verdict
short_dist_match = match_rate_per_bin[0] if not np.isnan(match_rate_per_bin[0]) else 0.5
long_dist_match = match_rate_per_bin[-1] if not np.isnan(match_rate_per_bin[-1]) else 0.5
print(f"\nH5 (Spatial leakage): match rate at dist=0 = {short_dist_match:.3f}, at max dist = {long_dist_match:.3f}")
print(f"  → {'STRONG spatial autocorrelation (IID is biased easy)' if short_dist_match > 0.85 else 'Weak autocorrelation'}")

# Overall verdict
print("\n" + "="*70)
print("OVERALL VERDICT — Why N→S collapses with VIIRS labels:")
print("="*70)
verdicts = []
if h1_strong >= 3: verdicts.append("Covariate shift (weather differs N vs S)")
if h2_avg_corr < 0.9: verdicts.append("Concept shift (same weather → different fire prob N vs S)")
if ratio_diff > 0.3: verdicts.append("MODIS/VIIRS detection bias by region")
if n_s_diff > 0.05: verdicts.append("Unobserved spatial variables (NDVI/elevation/land cover)")
if short_dist_match > 0.85: verdicts.append("IID is biased easy (spatial autocorrelation)")

print(f"\nContributing factors ({len(verdicts)} identified):")
for i, v in enumerate(verdicts, 1):
    print(f"  {i}. {v}")

if not verdicts:
    print("  No clear single cause — likely combined effect")

# Save all results
with open(OUT / "spatial_diagnosis.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Results saved to {OUT}/spatial_diagnosis.json")
print(f"✓ Figures saved to {FIG}/fig_diag_*.png")
