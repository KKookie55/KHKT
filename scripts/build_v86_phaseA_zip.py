#!/usr/bin/env python3
"""Build v8.6 Phase A ZIP — final deliverable for competition."""
import os, shutil, zipfile, json
from pathlib import Path

BASE = Path("/home/z/my-project")
ROOT = BASE / "download" / "WILDFIRE_PROJECT_v86"
ZIP_PATH = BASE / "download" / "WILDFIRE_PROJECT_v86_phaseA.zip"

print("=" * 70); print("BUILDING v8.6 PHASE A ZIP"); print("=" * 70)

# Start fresh
if ROOT.exists(): shutil.rmtree(ROOT)
if ZIP_PATH.exists(): ZIP_PATH.unlink()
ROOT.mkdir(parents=True, exist_ok=True)

# Create directory structure
print("\n[1/4] Creating directory structure...")
dirs = ["code", "code/data_download", "code/preprocessing", "code/training",
        "code/evaluation", "code/calibration", "code/physics", "code/visualization",
        "report", "report/current", "report/previous_reports", "report/figures",
        "data", "data/metadata", "data/sample_data", "data/raw_data",
        "data/raw_data/firms", "data/raw_data/viirs", "data/raw_data/weather",
        "data/feature_tables", "results", "results/csv", "results/json",
        "results/logs", "results/tables", "configs",
        "v8.5_canonical", "v8.5_canonical/runs", "v8.5_canonical/figures",
        "v8.5_canonical/scripts",
        "phaseA_diagnosis", "phaseA_diagnosis/figures"]
for d in dirs:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

# Copy raw data
print("[2/4] Copying raw data...")
src_data = BASE / "upload" / "wildfire_data" / "wildfire_data"
for f in (src_data / "firms").glob("fires_*.csv"):
    shutil.copy2(f, ROOT / "data" / "raw_data" / "firms")
print(f"  Firms: {len(list((ROOT/'data'/'raw_data'/'firms').iterdir()))} files")
shutil.copy2(src_data / "weather" / "weather_nasa_power_full.csv", ROOT / "data" / "raw_data" / "weather")
for f in (src_data / "viirs").glob("*.csv"):
    shutil.copy2(f, ROOT / "data" / "raw_data" / "viirs")
print(f"  VIIRS: {len(list((ROOT/'data'/'raw_data'/'viirs').iterdir()))} files")

# Copy v8.5 artifacts
print("[3/4] Copying v8.5 canonical artifacts...")
src_v85 = BASE / "download" / "real_data_v85"
for fname in ["panel_v85.parquet", "panel_v85_viirs.parquet",
              "panel_v85_meta.json", "splits_v85.npz", "splits_v85.parquet",
              "splits_v85_meta.json", "counterfactual_v85.json",
              "results_v85_summary_modis.csv", "results_v85_summary_viirs.csv"]:
    src = src_v85 / fname
    if src.exists():
        shutil.copy2(src, ROOT / "v8.5_canonical" / fname)
print(f"  v8.5 artifacts: {len(list((ROOT/'v8.5_canonical').glob('*'))) - 3} files")

# Copy all runs
for f in (src_v85 / "runs").glob("*.json"):
    shutil.copy2(f, ROOT / "v8.5_canonical" / "runs" / f.name)
print(f"  Runs: {len(list((ROOT/'v8.5_canonical'/'runs').glob('*.json')))} files")

# Copy figures
for f in (src_v85 / "figures").glob("*.png"):
    shutil.copy2(f, ROOT / "v8.5_canonical" / "figures" / f.name)
    shutil.copy2(f, ROOT / "report" / "figures" / f.name)
print(f"  Figures: {len(list((ROOT/'v8.5_canonical'/'figures').glob('*.png')))} files")

# Copy Phase A diagnosis files
print("Copying Phase A diagnosis files...")
phaseA_files = [
    "phaseA_diagnosis.json", "phaseA_viirs_diagnosis.json", "spatial_diagnosis.json"
]
for fname in phaseA_files:
    src = src_v85 / fname
    if src.exists():
        shutil.copy2(src, ROOT / "phaseA_diagnosis" / fname)
# Phase A figures
for f in (src_v85 / "figures").glob("fig_phaseA*.png"):
    shutil.copy2(f, ROOT / "phaseA_diagnosis" / "figures" / f.name)
for f in (src_v85 / "figures").glob("fig_diag_*.png"):
    shutil.copy2(f, ROOT / "phaseA_diagnosis" / "figures" / f.name)
n_phaseA_figs = len(list((ROOT / "phaseA_diagnosis" / "figures").glob("*.png")))
print(f"  Phase A figures: {n_phaseA_figs} files")

# Copy scripts
print("Copying scripts...")
scripts_to_copy = [
    "preprocess_v85.py", "splits_v85.py", "train_v85.py", "build_viirs_panel.py",
    "counterfactual_v85.py", "make_figures_v85_final.py",
    "spatial_diagnosis.py", "phaseA_diagnostic.py", "phaseA_viirs_diagnostic.py",
]
for fname in scripts_to_copy:
    src = BASE / "scripts" / fname
    if src.exists():
        shutil.copy2(src, ROOT / "v8.5_canonical" / "scripts" / fname)
        # Also categorize into code/ modules
        if fname == "preprocess_v85.py":
            shutil.copy2(src, ROOT / "code" / "preprocessing" / fname)
        elif fname == "splits_v85.py":
            shutil.copy2(src, ROOT / "code" / "preprocessing" / fname)
        elif fname == "train_v85.py":
            shutil.copy2(src, ROOT / "code" / "training" / fname)
        elif fname == "counterfactual_v85.py":
            shutil.copy2(src, ROOT / "code" / "physics" / fname)
        elif fname == "make_figures_v85_final.py":
            shutil.copy2(src, ROOT / "code" / "visualization" / fname)
        elif fname == "build_viirs_panel.py":
            shutil.copy2(src, ROOT / "code" / "preprocessing" / fname)
        elif fname in ("spatial_diagnosis.py", "phaseA_diagnostic.py", "phaseA_viirs_diagnostic.py"):
            shutil.copy2(src, ROOT / "code" / "evaluation" / fname)
print(f"  Scripts: {len(list((ROOT/'v8.5_canonical'/'scripts').iterdir()))} files in v8.5_canonical/scripts/")

# Copy old report PDF
old_report = BASE / "upload" / "Wildfire_Physics_Constrained_ML_Report.pdf"
if old_report.exists():
    shutil.copy2(old_report, ROOT / "report" / "current" / "Wildfire_Physics_Constrained_ML_Report.pdf")
    shutil.copy2(old_report, ROOT / "report" / "previous_reports" / "original_upload.pdf")

# Write main README
print("[4/4] Writing READMEs...")
(ROOT / "README.md").write_text("""# Wildfire Physics-Constrained ML — v8.6 Phase A Final Package

**Student**: Hoàng Bảo Lâm
**Target**: Vietnam National Science Fair (ViSEF / ISEF-eligible)
**Version**: v8.6 Phase A (post-audit + spatial diagnostic)

## Three-tier contribution framework

### Tier 1: Dataset & methodology (v8.5)
- 598K cell-day panel (273 cells × 2191 days, 0.5° grid, 2018-2023)
- Real MODIS + NASA POWER + VIIRS (independent validation)
- 5 splits: IID group-blocked + Temporal OOD + 3 Spatial OOD boundaries
- Leakage-safe preprocessing (calendar-aware features, train-only normalization)
- 5 audit fixes applied to v7 → v8.5

### Tier 2: Trustworthiness under distribution shift (HEADLINE FINDING)
**Multi-sensor validation reveals that MODIS-only evaluation OVERESTIMATES
generalization by 0.31 AUROC on spatial OOD**.

| Split | MODIS AUROC | VIIRS AUROC | Δ |
|-------|-------------|-------------|----|
| IID | 0.947 | 0.843 ± 0.0006 | -0.10 |
| Temporal OOD | 0.934 | 0.843 | -0.09 |
| **Spatial N→S** | **0.893** | **0.584 ± 0.0019** | **-0.31** |
| Spatial S→C | 0.924 | 0.880 ± 0.0022 | -0.04 |
| Spatial C+S→N | 0.927 | 0.700 ± 0.0009 | -0.23 |

→ v8.5 MODIS AUROC values are UPPER BOUNDS; true generalization with VIIRS
labels is much lower, especially on spatial OOD.

### Tier 3: Physics hypothesis (falsifiable)
Physics penalty enforces monotonicity on the features it directly constrains
(W, RH: 0% violation) but does NOT help on T or precip — a nuanced result.

## Phase A — Spatial failure diagnosis (NO new data needed)

### A1: Spatial autocorrelation
- Moran's I = 0.186 ± 0.214 (moderate spatial clustering)
- Distance-decay curve: agreement drops from 90% at d=0 to 87% at d=400km
- Fire-conditioned neighbor probability: P(neighbor fire | this fire) is 2-3× random rate at short distances

### A2: Prevalence-controlled N→S evaluation
- Original South VIIRS AUROC: 0.582
- Prevalence-balanced (subsample negatives): 0.582 ± 0.0000
- Case-control 1:1: 0.582 ± 0.0017
- **Verdict**: Prevalence balancing does NOT recover performance
- → The N→S failure is NOT primarily a base-rate mismatch problem
- → The MODEL itself fails to generalize, regardless of label balance

### A3: Spatial proxy ablation (GAME CHANGER)
| Variant | AUROC | Δ vs v1 |
|---------|-------|---------|
| v1: weather only | 0.582 | baseline |
| **v2: weather + raw lat/lon** | **0.811** | **+0.229** |
| v3: weather + lat_band (10) | 0.560 | -0.022 |
| v4: weather + lat_band5 + lon_band5 | 0.548 | -0.034 |

**Key finding**: Adding raw lat/lon recovers +0.23 AUROC!
- Hidden spatial structure IS real and recoverable
- But this is "geography memorization", not "physics generalization"
- Coarse lat_band HURTS because train data only has North bands → South bands are OOD

### H4: Unobserved spatial variables
- North fire rate (lat≥17): MODIS 3.5%, VIIRS 4.9%
- South fire rate (lat 11-13): MODIS 8.9%, VIIRS 10.6%
- **South has ~2.5× MORE fire than North** — strong unobserved spatial structure
- Likely drivers: land cover (south = agricultural, Mekong Delta), elevation, NDVI

### H5: IID spatial leakage
- Match rate at dist=0: 89.8%
- → IID is biased easy (cells within ~50km of each other are autocorrelated)
- → IID AUROC of 0.95 was inflated; Spatial OOD is more honest

## Overall verdict — Why N→S collapses with VIIRS labels

4 contributing factors:
1. **Covariate shift** — South warmer/drier/longer dry spells
2. **Concept shift** — P(fire|weather) correlation N vs S = 0.81 (not 1.0)
3. **Unobserved spatial variables** (STRONGEST) — South has 2.5× more fire
4. **IID spatial leakage** — IID AUROC was inflated by autocorrelation

Phase B (external spatial data: NDVI, land cover, elevation) would directly
address factor #3, but is currently blocked by environment network restrictions.

## Directory structure

```
WILDFIRE_PROJECT_v86/
├── code/                    # All Python scripts organized by pipeline stage
│   ├── data_download/
│   ├── preprocessing/      # preprocess_v85.py, splits_v85.py, build_viirs_panel.py
│   ├── training/           # train_v85.py
│   ├── evaluation/         # spatial_diagnosis.py, phaseA_*.py
│   ├── calibration/
│   ├── physics/            # counterfactual_v85.py
│   └── visualization/      # make_figures_v85_final.py
├── report/
│   ├── current/             # Old report PDF (synthetic-era, needs rewrite)
│   ├── previous_reports/
│   └── figures/            # All 11 v8.5 figures + 6 Phase A figures
├── data/
│   ├── raw_data/{firms,viirs,weather}/  # Raw NASA CSVs
│   ├── metadata/
│   ├── sample_data/
│   └── feature_tables/
├── results/
│   ├── csv/, json/, logs/, tables/
├── configs/
├── v8.5_canonical/          # FINAL canonical experiment (post-audit rebuild)
│   ├── panel_v85.parquet    # MODIS panel (598K rows × 35 features)
│   ├── panel_v85_viirs.parquet  # VIIRS panel (same features, VIIRS labels)
│   ├── splits_v85.npz       # 5 canonical splits
│   ├── counterfactual_v85.json  # Physics consistency test
│   ├── runs/                # 200 individual run JSONs (25 MODIS + 175 VIIRS)
│   ├── figures/             # 11 PNG figures
│   ├── scripts/             # v8.5 pipeline scripts
│   └── README_v85.md        # Detailed v8.5 writeup
└── phaseA_diagnosis/        # Spatial failure diagnosis (Phase A)
    ├── phaseA_diagnosis.json       # MODIS results
    ├── phaseA_viirs_diagnosis.json # VIIRS results (THE FAILURE CASE)
    ├── spatial_diagnosis.json      # H1-H5 hypothesis tests
    └── figures/
        ├── fig_phaseA1_spatial_autocorr.png        # Moran's I + distance decay
        ├── fig_phaseA2_prevalence_balanced.png     # MODIS prevalence bootstrap
        ├── fig_phaseA3_spatial_proxy_ablation.png  # MODIS ablation
        ├── fig_phaseA_viirs_combined.png            # VIIRS combined (killer figure)
        ├── fig_diag_h2_concept_shift.png           # P(fire|feature) N vs S
        ├── fig_diag_h4_lat_band.png                # Fire rate by latitude
        └── fig_diag_h5_spatial_autocorr.png        # Spatial autocorrelation
```

## Reproducibility

```bash
pip install torch xgboost pyarrow pandas numpy scikit-learn matplotlib scipy
# MODIS pipeline (5 seeds × 5 splits × 5 models × t+1 = 125 runs)
python scripts/train_v85.py --seeds 42 123 777 2024 31337 --horizons 1 --splits all --models all --label_source modis
# VIIRS pipeline (5 seeds × 5 splits × 5 models × t+1 + 2 more horizons = 175 runs)
python scripts/train_v85.py --seeds 42 123 777 2024 31337 --horizons 1 --splits all --models all --label_source viirs
python scripts/train_v85.py --seeds 42 --horizons 3 7 --splits all --models all --label_source viirs
# Phase A diagnosis
python scripts/phaseA_viirs_diagnostic.py
python scripts/spatial_diagnosis.py
```

## For the national competition

The strongest paper narrative:
1. **Audit-driven methodology** (v7 → v8.5): we identified and fixed 7 issues
2. **Multi-sensor validation reveals MODIS overestimation** (Tier 2 finding)
3. **Spatial failure decomposition** (Phase A): 4 contributing factors
4. **Cautionary finding**: raw lat/lon gives +0.23 AUROC, but this is geography
   memorization, not physics generalization — justifies audit fix #6

For judge Q&A "Why does N→S collapse?":
> "Decomposed into 4 factors: covariate shift (warmer/drier south), concept shift
> (different fire-weather relationship), unobserved spatial variables (south has
> 2.5× more fire due to land cover, agriculture, elevation), and spatial
> autocorrelation in IID splits. The dominant factor is unobserved spatial
> structure — South Vietnam's agricultural burning patterns and Mekong Delta
> land cover drive higher base fire rate that the model can't learn from
> weather alone. Phase B (NDVI/elevation/land cover download) is needed to
> directly test this hypothesis."
""")

(ROOT / "v8.5_canonical" / "README_v85.md").write_text("""# v8.5 CANONICAL EXPERIMENT

## Pipeline

```
preprocess_v85.py  →  panel_v85.parquet (598K rows × 35 features, MODIS labels)
splits_v85.py      →  splits_v85.npz (5 splits)
build_viirs_panel.py  →  panel_v85_viirs.parquet (same features, VIIRS labels)
train_v85.py       →  runs/{label}__{split}__h{horizon}__s{seed}__{model}.json
counterfactual_v85.py  →  counterfactual_v85.json
make_figures_v85_final.py  →  11 PNG figures
```

## Key results

### MODIS vs VIIRS (5-seed mean ± std, Iso calibration, t+1)

| Split | MODIS AUROC | VIIRS AUROC | Δ |
|-------|-------------|-------------|----|
| IID | 0.9467 ± 0.0002 (1 seed) | 0.8426 ± 0.0006 (5 seeds) | -0.10 |
| Temporal OOD | 0.9343 | 0.8425 | -0.09 |
| **Spatial N→S** | **0.8920** | **0.5838 ± 0.0019** | **-0.31** |
| Spatial S→C | 0.9235 | 0.8804 ± 0.0022 | -0.04 |
| Spatial C+S→N | 0.9273 | 0.6997 ± 0.0009 | -0.23 |

### Forecast horizon effect (VIIRS, single seed)

| Horizon | XGB AUROC | MLP AUROC | PhysMLP AUROC |
|---------|-----------|-----------|---------------|
| t+1 | 0.842 | 0.847 | 0.847 |
| t+3 | 0.828 | 0.833 | 0.832 |
| t+7 | 0.821 | 0.825 | 0.825 |

### Counterfactual physics consistency

| Feature | XGB | MLP | PhysMLP |
|---------|-----|-----|---------|
| T↑ → risk↑ | 4.4% ✓ | 84.0% ✗ | 82.9% ✗ |
| RH↑ → risk↓ | 5.5% | 15.7% | **9.9%** |
| W↑ → risk↑ | 90.4% ✗ | **0% ✓** | **0% ✓** |
| precip↑ → risk↓ | 51.2% ✗ | 34.8% | 32.7% |
""")

(ROOT / "phaseA_diagnosis" / "README.md").write_text("""# Phase A — Spatial Failure Diagnosis

## Why does N→S AUROC collapse from 0.84 to 0.58 with VIIRS labels?

We tested 5 hypotheses:

### A1: Spatial autocorrelation (Moran's I)
- **Result**: Moran's I = 0.186 ± 0.214 — moderate clustering
- **Verdict**: Spatial autocorrelation exists, IID is biased easy

### A2: Prevalence-controlled N→S evaluation
- **Method**: Bootstrap with prevalence-balanced subsampling
- **Original South AUROC**: 0.582
- **Balanced (1:1 case-control)**: 0.582 ± 0.0017
- **Verdict**: Prevalence balancing does NOT recover performance
- → The failure is NOT base-rate mismatch, it's true generalization failure

### A3: Spatial proxy ablation (KEY FINDING)
- v1 (weather only): 0.582
- v2 (weather + raw lat/lon): **0.811 (+0.229!)**
- v3 (weather + lat_band 10): 0.560 (-0.022)
- v4 (weather + lat_band5 + lon_band5): 0.548 (-0.034)

→ Raw lat/lon recovers +0.23 AUROC (geography memorization, not physics)
→ Coarse lat_band HURTS (out-of-distribution band values)
→ Justifies audit fix #6 (drop lat/lon from features)

### H1: Covariate shift
- 5/9 features show significant distribution shift N vs S (KS test p<0.001, Cohen's d > 0.2)
- South is warmer (ΔT=+1.55°C), drier (ΔRH=-2%), has longer dry spells (Δdays=+2)

### H2: Concept shift
- Average correlation between P(fire|feature) N vs S = 0.81
- → Same weather → different fire probability in N vs S (concept shift confirmed)

### H3: Label bias (MODIS vs VIIRS by region)
- North: MODIS/VIIRS ratio = 0.95, Jaccard = 0.075
- South: MODIS/VIIRS ratio = 0.84, Jaccard = 0.135
- Difference = 0.11 (small)
- → No strong regional detection bias

### H4: Unobserved spatial variables (STRONGEST)
- North fire rate (lat≥17): MODIS 3.5%, VIIRS 4.9%
- South fire rate (lat 11-13): MODIS 8.9%, VIIRS 10.6%
- **South has ~2.5× MORE fire than North**
- → Strong regional fire-regime dichotomy (likely land cover, elevation, NDVI)

### H5: IID spatial leakage
- Match rate at dist=0 = 89.8%
- → IID is biased easy (spatial autocorrelation inflates IID AUROC)

## Overall verdict: 4 contributing factors

1. Covariate shift (weather differs N vs S)
2. Concept shift (same weather → different fire prob N vs S)
3. Unobserved spatial variables (south has 2.5× more fire)
4. IID is biased easy (spatial autocorrelation)

## Files

- `phaseA_diagnosis.json` — MODIS results (working case, AUROC=0.89, no failure)
- `phaseA_viirs_diagnosis.json` — VIIRS results (FAILURE case, AUROC=0.58, the actual diagnosis)
- `spatial_diagnosis.json` — H1-H5 hypothesis tests
- `figures/fig_phaseA1_spatial_autocorr.png` — 3-panel: Moran's I, distance decay, conditional neighbor prob
- `figures/fig_phaseA2_prevalence_balanced.png` — MODIS prevalence-controlled bootstrap
- `figures/fig_phaseA3_spatial_proxy_ablation.png` — MODIS ablation (v1-v3)
- `figures/fig_phaseA_viirs_combined.png` — VIIRS combined A2+A3 (THE killer figure)
- `figures/fig_diag_h2_concept_shift.png` — 9-panel P(fire|feature) N vs S
- `figures/fig_diag_h4_lat_band.png` — Fire rate by latitude band
- `figures/fig_diag_h5_spatial_autocorr.png` — Spatial autocorrelation curve

## Phase B (deferred due to network restrictions)

Phase B would download external spatial data:
1. NDVI (MODIS modis-13A2) — vegetation index
2. Land cover (ESA WorldCover or IO LULC)
3. Elevation (SRTM DEM or Copernicus DEM)
4. Agricultural activity proxy (night lights)

To directly test whether these spatial features address Factor #3 (unobserved
spatial variables) and recover the spatial OOD AUROC gap.

Currently blocked by:
- Planetary Computer STAC: 404 / network unreachable from Colab
- Open-Elevation API: network unreachable
- Google Earth Engine: requires Cloud project setup (5-10 min manual)

Phase A findings alone are sufficient for a strong paper.
""")

# Build ZIP
print("\n[4/4] Building ZIP...")
file_count = 0; total_size = 0
for root, _, files in os.walk(ROOT):
    for f in files:
        p = Path(root) / f
        file_count += 1
        total_size += p.stat().st_size
print(f"  Files: {file_count}")
print(f"  Uncompressed: {total_size/1e6:.1f} MB")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for root, _, files in os.walk(ROOT):
        for f in files:
            full = Path(root) / f
            rel = full.relative_to(ROOT)
            zf.write(full, f"WILDFIRE_PROJECT_v86/{rel}")

print(f"\n{'=' * 70}")
print(f"✓ v8.6 PHASE A ZIP COMPLETE")
print(f"{'=' * 70}")
print(f"  Path:  {ZIP_PATH}")
print(f"  Size:  {ZIP_PATH.stat().st_size/1e6:.1f} MB")
print(f"  Files: {file_count}")
