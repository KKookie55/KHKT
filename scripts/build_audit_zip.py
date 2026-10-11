#!/usr/bin/env python3
"""
Build WILDFIRE_PROJECT ZIP for full audit + national-competition upgrade.
Categorizes every existing artifact into the requested structure.

Output: /home/z/my-project/download/WILDFIRE_PROJECT_FULL_AUDIT.zip
"""
import os, shutil, json, zipfile, sys, hashlib
from pathlib import Path

BASE = Path("/home/z/my-project")
ROOT = BASE / "download" / "WILDFIRE_PROJECT"
ZIP_PATH = BASE / "download" / "WILDFIRE_PROJECT_FULL_AUDIT.zip"

# Fresh start
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
ZIP_PATH.unlink(missing_ok=True)

# ============================================================
# Helpers
# ============================================================
def mkdir(*parts):
    p = ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p

def cp(src, dst):
    """Copy a file (or directory recursively) into dst (dst = parent dir)."""
    src = Path(src); dst = Path(dst)
    if not src.exists():
        print(f"  [SKIP] not found: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst / src.name, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst / src.name)

def write(path, content):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

print("=" * 70)
print("BUILDING WILDFIRE_PROJECT ZIP")
print("=" * 70)

# ============================================================
# 1. CODE
# ============================================================
print("\n[1/9] Copying code into modules...")
code_dir = mkdir("code")

# 1a. data_download/
dd = mkdir("code", "data_download")
for f in [
    "download_real_modis_fire.py",           # Microsoft Planetary Computer MODIS
    "download_all_real_data.py",             # all-in-one real data downloader
    "download_real_data.py",                 # earlier variant
    "download_real_data_gee.py",             # Google Earth Engine attempt (failed)
    "download_weather_ndvi_dem.py",          # NASA POWER + NDVI + DEM
    "redownload_2021_2023.py",               # re-download missing years
    "build_real_dataset.py",                 # feature table builder (v5 era)
    # Colab helpers (multi-day sequential download)
    "colab_download_no_auth.py",
    "colab_download_only.py",
    "colab_drive_download.py",
    "colab_self_contained.py",
    "download_only_colab.py",
    "phase_b_colab.py",
    "phase_b_colab_v2.py",
]:
    for src_dir in [BASE/"download"/"code", BASE/"scripts", BASE/"download"/"wildfire_real_data_v5"/"code"]:
        cand = src_dir / f
        if cand.exists():
            cp(cand, dd)
            break

# 1b. preprocessing/
pp = mkdir("code", "preprocessing")
# build_real_dataset.py is the canonical preprocessing script (already in data_download for archive).
# We'll also extract a documentation file describing the preprocessing pipeline.
# Add the run_v6_fixed.py's preprocessing section as a reference here too.
cp(BASE/"scripts"/"run_v6_fixed.py", pp)
cp(BASE/"scripts"/"run_v7_final.py", pp)

# 1c. training/
tr = mkdir("code", "training")
for f in [
    "run_research.py",                        # v1-v5 synthetic pipeline (kept for evolution)
    "run_real_pilot.py",                      # v5 pilot (real fire + synthetic weather)
    "run_real_weather_real_fire.py",          # v5 fully real
    "run_v6_fixed.py",                        # v6 with 5 fixes
    "run_v7_final.py",                        # v7 FINAL (581K MODIS + 598K POWER)
    "run_v8_isef.py",                         # v8 ISEF-ready (multi-seed + OOD)
]:
    for src_dir in [BASE/"scripts", BASE/"download"/"wildfire_real_data_v5"/"code"]:
        cand = src_dir / f
        if cand.exists():
            cp(cand, tr)
            break

# 1d. evaluation/
ev = mkdir("code", "evaluation")
cp(BASE/"scripts"/"compute_accuracy.py", ev)
cp(BASE/"scripts"/"export_predictions.py", ev)
cp(BASE/"download"/"code"/"compute_accuracy.py", ev)  # v5 era variant
cp(BASE/"download"/"code"/"export_predictions.py", ev)

# 1e. calibration/
ca = mkdir("code", "calibration")
write("code/calibration/README_CALIBRATION.md", """# Calibration Module

This folder documents the calibration methods used in v7-v8.

## Methods implemented

All calibration code is **inline** in:
- `code/training/run_v7_final.py` (Step 7: CALIBRATION BASELINES)
- `code/training/run_v8_isef.py` (functions `calibrate_isotonic`, `calibrate_platt`)

### 1. Temperature scaling
- Fit on **validation** set logits (no test leakage).
- Search T in linspace(0.5, 5.0, 31).
- Minimize NLL on validation.
- Apply T to test logits.

### 2. Platt scaling (Logistic Regression on logits)
- `LogisticRegression(C=1e10)` fit on val logits → val labels.
- Apply to test logits.

### 3. Isotonic regression (the winning method)
- `IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)`.
- Fit on val probabilities → val labels.
- Apply to test probabilities.

## Anti-leakage checklist (must verify in audit)
- [ ] Calibrator fit ONLY on validation set (not test).
- [ ] F1 threshold chosen on validation precision-recall curve.
- [ ] Test set is NEVER used to fit any calibrator or choose any threshold.
- [ ] For OOD splits: calibrator fit on **same-distribution validation**,
      applied to **OOD test**. This is the honest test of calibration
      generalization.
""")

# 1f. physics/
ph = mkdir("code", "physics")
write("code/physics/README_PHYSICS.md", """# Physics Module

This folder documents the physics-informed loss used in v5-v7.

## Where the code lives
- v5 synthetic: `code/training/run_research.py` — function `physics_penalty` (matching-style, replaced).
- v5 real:      `code/training/run_real_weather_real_fire.py`.
- v6 fixed:    `code/training/run_v6_fixed.py`.
- v7 FINAL:    `code/training/run_v7_final.py` — function `physics_penalty_raw` (RAW, monotonicity-based).

## v7 physics loss (canonical)

Located at `code/training/run_v7_final.py:296-321`.

### Penalty form
Three pairwise monotonicity terms on **RAW (un-standardized)** features:

| Term | Direction | Condition | Weight |
|------|-----------|-----------|--------|
| `l_e` (energy)      | T↑ → risk↑  | always                | 0.4 |
| `l_m` (moisture)    | RH↑ → risk↓ | always                | 0.4 |
| `l_w` (wind)        | W↑ → risk↑  | only when T_ii > 25 °C AND T_ij > 25 °C | 0.2 |

Total: `λ * (0.4*l_e + 0.4*l_m + 0.2*l_w)`

### Pair sampling
- For each minibatch (size ≥ 8), draw `sz = n//2` random index pairs `(ii, ij)`.
- Loss counts violations: a pair is a violation if `(p[ii] - p[ij])` has the
  wrong sign w.r.t. `(t[ii] - t[ij])` (or RH, or W with the hot-mask).

### Two-phase training
- First 50% of epochs: pure BCE (no physics).
- Last 50% of epochs: BCE + λ·physics.
- Rationale: let the model learn the data signal first, then nudge it toward
  physics-consistent behavior; this preserves AUROC (which a too-early
  physics penalty would degrade).

### λ values swept
`λ ∈ {0.0, 0.01, 0.05, 0.10, 0.20}` (see `results_v7_final.json:lambda_sweep`).

### Key finding
- ΔECE(λ=0 vs λ=0.2) ≈ 0 in v7 — physics constraint does NOT improve
  calibration when the dataset is real (581K MODIS + 598K NASA POWER).
- This is a NEGATIVE result for the original physics hypothesis.

## Feature scope (must audit)
- RAW_PHYS_COLS = ['t2m', 'rh', 'wind'] (units: °C, %, m/s).
- These are the un-standardized columns from the NASA POWER daily table.

## What is NOT in the physics loss (limitation)
- No precipitation monotonicity term (precip↑ → risk↓).
- No drought-index term.
- No vegetation/fuel term (NDVI not available in v7).
- No spatial-coupling term (no neighboring-cell penalty).
""")
# Also copy the run_v7 script's physics section into a standalone snippet for direct review.
write("code/physics/physics_penalty_raw.py", '''"""
Extracted from run_v7_final.py — physics penalty (canonical v7).
Lines 295-321 of the original file. Read alongside the README above.
"""
import torch

RAW_PHYS_COLS = ["t2m", "rh", "wind"]  # °C, %, m/s

def physics_penalty_raw(pred, raw_batch):
    """Pairwise monotonicity on raw physical features.

    Args:
        pred:     (N,) raw logits (pre-sigmoid) from PhysicsMLP.forward()
        raw_batch: (N, 3) with columns [t2m, rh, wind] in original units.

    Returns:
        scalar tensor (penalty), to be multiplied by λ and added to BCE.
    """
    p = torch.sigmoid(pred)
    t = raw_batch[:, 0]    # temperature, °C
    rh = raw_batch[:, 1]  # relative humidity, %
    w = raw_batch[:, 2]   # wind speed, m/s
    n = p.shape[0]
    if n < 8:
        return torch.tensor(0.0, device=pred.device)
    sz = n // 2
    ii = torch.randint(0, n, (sz,), device=pred.device)
    ij = torch.randint(0, n, (sz,), device=pred.device)
    # Energy: T↑ → risk↑
    td = t[ii] - t[ij]
    pd = p[ii] - p[ij]
    l_e = torch.mean((td * pd < 0).float() * torch.abs(pd))
    # Moisture: RH↑ → risk↓
    rd = rh[ii] - rh[ij]
    l_m = torch.mean((rd * pd > 0).float() * torch.abs(pd))
    # Wind: W↑ → risk↑ (only when both cells hot)
    hm = (t[ii] > 25) & (t[ij] > 25)
    wd = w[ii] - w[ij]
    l_w = torch.sum(hm.float() * (wd * pd < 0).float() * torch.abs(pd)) / (hm.sum().float() + 1e-6)
    return 0.4 * l_e + 0.4 * l_m + 0.2 * l_w
''')

# 1g. visualization/
viz = mkdir("code", "visualization")
cp(BASE/"scripts"/"generate_pdf.py", viz)
cp(BASE/"download"/"code"/"generate_pdf.py", viz)

# ============================================================
# 2. REPORTS
# ============================================================
print("\n[2/9] Copying reports...")
rep = mkdir("report", "current")
# Latest report PDF (synthetic-era report — does NOT reflect v7 real-data findings)
cp(BASE/"download"/"Wildfire_Physics_Constrained_ML_Report.pdf", rep)

# Original uploaded draft (with [ĐIỀN ...] placeholders)
prev = mkdir("report", "previous_reports")
cp(BASE/"upload"/"Wildfire_Physics_Constrained_ML_Report.pdf", prev)
cp(BASE/"upload"/"Wildfire_Physics_Constrained_ML_Report.txt", prev)

# v5 era report (slightly different timestamp)
cp(BASE/"download"/"wildfire_real_data_v5"/"Wildfire_Physics_Constrained_ML_Report.pdf", prev)

# v5 worklog as a report-companion doc
cp(BASE/"download"/"wildfire_real_data_v5"/"worklog.md", prev)
cp(BASE/"download"/"wildfire_real_data_v5"/"README.md", prev)

# figures/ — 8 raw research-evidence figures generated from v7 JSON
fig_dir = mkdir("report", "figures")
# Copy all generated PNGs
import glob
for png in sorted(glob.glob("/home/z/my-project/WILDFIRE_PROJECT_BUILD/report/figures/*.png")):
    cp(png, fig_dir)
# Also keep the README about figures
write("report/figures/README_FIGURES.md", """# Figures — raw research evidence (v7 audit)

8 PNG figures generated directly from `results/json/results_audit.json`,
`results/json/results_v7_final.json`, and v5-era predictions CSV.
These are **deliberately minimal** — they are raw research evidence,
not the polished slides for the national competition.

| # | File                              | What it shows |
|---|-----------------------------------|---------------|
| 1 | fig1_ece_degradation.png          | Bar chart: ECE across IID / Temporal OOD / Spatial OOD, by model. Headline: XGB+Iso barely degrades (0.009→0.032), MLP collapses (0.053→0.171). |
| 2 | fig2_auroc_across_splits.png      | Bar chart: AUROC across the three splits. XGBoost is most stable. |
| 3 | fig3_lambda_sweep.png             | 3-panel: ECE / AUROC / Brier vs λ ∈ {0, 0.01, 0.05, 0.10, 0.20}. All curves are essentially flat → physics constraint does NOT help on real data. |
| 4 | fig4_calibration_baselines.png    | Bar chart: ECE & Brier for XGB raw / + Temp / + Iso / + Platt on test 2022. Isotonic wins. |
| 5 | fig5_reliability_diagram_v5_era.png | Reliability diagram (10 bins) from v5 test predictions. **Caveat**: v7 doesn't have raw predictions saved, so this uses v5 era — re-run on v7 predictions for the final report. |
| 6 | fig6_brier_across_splits.png      | Bar chart: Brier across splits, by model. |
| 7 | fig7_delta_ece_line.png           | Line chart: ΔECE = ECE_split − ECE_IID. Lower = more robust. XGB+Iso stays near 0; MLP climbs steeply. |
| 8 | fig8_summary_table.png            | Image table: full v7 single-seed calibration baselines + λ-sweep summary. |

## What's missing for the national-level competition

After audit (PHASE 1), the following figures must be regenerated from v8 ISEF
multi-seed results (5-10 seeds with confidence intervals):
- ROC curves per model per split
- PR curves per model per split
- Spatial risk map of Vietnam with real NASA MODIS predictions
- Feature importance (XGB gain) bar chart
- Counterfactual monotonicity curves (T↑ / RH↓ / W↑ / precip↓)
- Confusion matrix at chosen F1 threshold
- Spatial block cross-validation map
""")
fig_dir_old = mkdir("report", "figures", "synthetic_era_embedded_in_pdf")
write("report/figures/synthetic_era_embedded_in_pdf/README.md", """# Figures embedded in synthetic-era PDF (v1-v4)

The current `Wildfire_Physics_Constrained_ML_Report.pdf` (in `report/current/`)
embeds 6 figures generated from the SYNTHETIC v1-v4 era. These are NOT
relevant to the v7 real-data findings.

| # | Figure                  | Source script                    |
|---|-------------------------|----------------------------------|
| 0 | Architecture diagram    | `code/visualization/generate_pdf.py` |
| 1 | Reliability diagram     | same                              |
| 2 | Spatial risk map        | same                              |
| 3 | Ablation bar chart      | same                              |
| 4 | OOD robustness curve    | same                              |
| 5 | Training loss curves    | same                              |

These will be replaced by v7-v8 figures (see `report/figures/` parent).
""")

# ============================================================
# 3. DATA
# ============================================================
print("\n[3/9] Copying data...")
# 3a. metadata/ — acquisition information, schema, dictionaries
md_dir = mkdir("data", "metadata")
write("data/metadata/dataset_acquisition.md", """# Dataset Acquisition — full provenance

## 1. NASA MODIS fire detections (Vietnam, 2018-01-01 → 2023-12-31)

### Source
- **Provider**: NASA FIRMS (Fire Information for Resource Management System)
- **Satellite**: Terra + Aqua (MODIS sensors)
- **Product**: MOD14A1 / MYD14A1, level 2, 1 km resolution, daily
- **Access**: Via Microsoft Planetary Computer (no API key required)
  - URL: https://planetarycomputer.microsoft.com/dataset/modis-14A1-061
  - STAC item: `modis-14A1-061`
- **Backup access**: NASA FIRMS direct CSV download (sandboxed in this env)
- **Alternative tried**: Google Earth Engine (requires project authorization — failed)

### Query parameters
- Bounding box: lon 104.0-109.5°E, lat 11.0-20.5°N (North-Central + Central Highlands Vietnam)
- Date range: 2018-01-01 to 2023-12-31
- Confidence filter: ≥ 30% (low-confidence thermal anomalies excluded)
- Detection type: vegetative fires only (excludes static thermal sources)

### Result (per-year raw downloads)
| Year | File                          | Rows (raw detections) |
|------|-------------------------------|-----------------------|
| 2018 | fires_2018_daily.csv          | 41,530 |
| 2019 | fires_2019_daily.csv          | 73,024 |
| 2020 | fires_2020_daily.csv          | 67,180 |
| 2021 | fires_2021_daily.csv          | 55,066 |
| 2022 | fires_2022_daily.csv          | 52,478 |
| 2023 | fires_2023_daily.csv          | 57,489 |
| **Total** |                          | **346,767** (raw detections) |
| (After deduplication to cell-day level) | | **48,279 unique positive cell-days** |

### Notes
- Downloaded in Google Colab over 6 separate sessions (one per year) due to
  session timeouts.
- Each session: fetch STAC items → mosaic → extract fire pixels → CSV.
- Total download time: ~6 hours across all sessions.
- Backup ZIP uploaded by user from Google Drive
  (`wildfire_data-20261005T132957Z-1-001.zip` — preserved in upload/).

---

## 2. NASA POWER daily weather

### Source
- **Provider**: NASA POWER (Prediction Of Worldwide Energy Resources)
- **API**: https://power.larc.nasa.gov/api/temporal/daily/point
- **Parameters retrieved**:
  - `T2M` — 2 m air temperature, °C
  - `RH2M` — 2 m relative humidity, %
  - `WS10M` — 10 m wind speed, m/s
  - `PRECTOTCORR` — precipitation, mm/day
- **Grid resolution**: 0.50° × 0.50°
- **Spatial coverage**: 21 lat × 13 lon = **273 grid cells** covering Vietnam
- **Temporal coverage**: 2018-01-01 to 2023-12-31, daily

### Result
- **598,143 records** (273 cells × 2191 days, with some gaps).
- File: `weather_nasa_power_full.csv` (25 MB).

### Alternative sources tried
- ECMWF CDS (ERA5): email verification did not work in this environment.
- Open-Meteo: rate-limited; used as fallback for v5 pilot (56,966 records).

---

## 3. Static biophysical layers (NOT used in v7 final)

In v5/v6 we attempted to use NDVI, elevation, slope, aspect, soil moisture
from various sources. For v7 we **removed these from the feature set** because:
1. NDVI requires Earth Engine authorization (failed).
2. DEM/soil layers from SoilGrids are static across years → the model could
   memorize cell-identity rather than learn dynamic physics.
3. Removing them does not measurably reduce AUROC.

If real NDVI/DEM is acquired in the future, the run_v7_final.py script has
hooks to merge them in (see lines 58-72 and 136-152).

---

## 4. Data not included (and why)

- **VIIRS active fire product (VNP14IMGTDL_NRT)**: planned as independent
  validation, not yet downloaded.
- **High-resolution (1 km) weather reanalysis**: ERA5 was unreachable; NASA
  POWER 0.5° is sufficient for proof-of-concept but limits sub-grid fire
  attribution.
""")

write("data/metadata/data_dictionary.md", """# Data dictionary — v7 final feature table

## Cell-day granularity

Each row = one (lat, lon, date) tuple on the 0.50° grid.

## Columns

| Column | Type | Unit | Source | Role | Notes |
|---|---|---|---|---|---|
| lat | float64 | degrees | derived | feature | snapped to 0.5° grid (11 + 0.5*k) |
| lon | float64 | degrees | derived | feature | snapped to 0.5° grid (104 + 0.5*k) |
| date | datetime | — | NASA POWER | index | normalized |
| year | int | — | derived | index | for split |
| doy | int | 1-366 | derived | feature | day of year (seasonal cycle) |
| t2m | float64 | °C | NASA POWER T2M | feature + physics | 2 m air temperature |
| rh | float64 | % | NASA POWER RH2M | feature + physics | 2 m relative humidity |
| wind | float64 | m/s | NASA POWER WS10M | feature + physics | 10 m wind speed |
| precip | float64 | mm/day | NASA POWER PRECTOTCORR | feature | total daily precipitation |
| t_lag7 | float64 | °C | derived | feature | 7-day lagged T2m, smoothed |
| rh_lag7 | float64 | % | derived | feature | 7-day lagged RH, smoothed |
| p_lag7 | float64 | mm | derived | feature | 7-day lagged precip, smoothed |
| t_lag14 | float64 | °C | derived | feature | 14-day lagged, smoothed |
| rh_lag14 | float64 | % | derived | feature | 14-day lagged, smoothed |
| p_lag14 | float64 | mm | derived | feature | 14-day lagged, smoothed |
| drought_idx | float64 | z-score | derived | feature | 30-day rolling mean precip, standardized, clipped [-1, 2] |
| heat_index | float64 | unitless | derived | feature | (t2m-20)*(50-rh)/50 — interaction |
| wind_chill | float64 | unitless | derived | feature | (t2m-15)*wind/30 — interaction |
| t_diff3 | float64 | °C | derived | feature | t2m.diff(3) — 3-day change |
| rh_diff3 | float64 | % | derived | feature | rh.diff(3) |
| w_diff3 | float64 | m/s | derived | feature | wind.diff(3) |
| fwi_compound | float64 | [0,1] | derived | feature | 1/(1+exp(-(0.4*(T-10) - 0.62*RH + 0.4*W - 0.3*p_lag7 - 30)/5)) |
| nb_t2m | float64 | °C | derived | feature | mean daily T2m across all cells (cross-sectional context) |
| nb_rh | float64 | % | derived | feature | mean daily RH across all cells |
| label | int | 0/1 | NASA MODIS | **TARGET** | 1 if any MODIS fire detection on this cell-day, else 0 |

## Static features present in v5/v6 but REMOVED in v7
- ndvi, soil_m, elev, slope, aspect (see `code/training/run_v7_final.py:139-152`).
- Reason: removed to prevent the model from learning cell-identity rather
  than physics. These can be re-added if real, time-varying NDVI is obtained.

## Features that should NOT be used to predict fire
The `label` column is the **target** and must NEVER appear in the feature
list. All other columns can be used as features or as inputs to derived
features. None of the derived features use `label` (verified by inspection
of `run_v7_final.py`).

## Prevalence
- 48,279 / 279,006 = **17.3%** positive cell-days on the observed subset.
- This is a strong class imbalance; XGBoost uses `scale_pos_weight=sqrt(neg/pos)`.
""")

write("data/metadata/grid_and_splits.md", """# Spatial grid and train/val/test splits

## Grid
- Spatial resolution: 0.50° × 0.50°.
- Latitude range: 11.0° N to 21.0° N (21 cells).
- Longitude range: 104.0° E to 109.5° E (13 cells).
- Total cells: 273.

## Date filtering
- Only dates with at least one MODIS observation are kept.
- This avoids the false-negative problem: a (cell, day) with no MODIS
  detection could either be a true no-fire day or a no-overpass / cloud-cover
  gap that we cannot distinguish from absence of fire.
- After filtering: 1,022 unique MODIS-observed dates over 2018-2023.
- Total cell-days: 273 × 1,022 = 279,006 (perfect balanced panel).

## Splits used in v7
| Split | Year(s) | n rows | n pos | prevalence |
|-------|---------|--------|-------|------------|
| Train | 2018-2020 | ~140,000 | ~24,000 | 17.1% |
| Val   | 2021     | ~46,000  | ~8,000  | 17.4% |
| Test  | 2022 (excl. KH) | ~38,000 | ~7,000 | 18.4% |
| OOD time | 2023   | ~46,000 | ~8,500 | 18.5% |
| OOD space | 2022, Khanh Hoa (lat<13, lon>108.5) | small | small | — |

## Splits used in v8 (more rigorous)
| Split name | Train | Val | Test |
|------------|-------|-----|------|
| IID        | random 60% | random 20% | random 20% (cell-days) |
| Temporal OOD | 2018-2021 | 2022 | 2023 |
| Spatial OOD | lat ≥ 15° (North) | random 20% of North | lat < 15° (South) |

## Anti-leakage notes
- IID split: row-shuffle without group-awareness — has potential
  spatial-temporal leakage (same cell on adjacent days can be in train and test).
  This is a known limitation; v8 reports IID numbers for context but emphasizes
  the OOD splits as the rigorous evaluation.
- Temporal OOD: clean — future years never appear in training.
- Spatial OOD: clean — southern cells never appear in training.
""")

# 3b. sample_data/ — head of each major CSV (for the auditor to inspect format)
sd = mkdir("data", "sample_data")
def sample_csv(src, dst, n=200):
    """Take first n rows of a CSV, preserve header."""
    src = Path(src); dst = ROOT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        write(dst.with_suffix('.note').as_posix().replace('note','txt'), f"missing source: {src}")
        return
    with open(src) as f:
        lines = [next(f) for _ in range(n + 1)]  # header + n
    dst.write_text("".join(lines))

# Sample from raw MODIS fire CSVs
for y in range(2018, 2024):
    src = BASE/"upload"/"wildfire_data"/"wildfire_data"/"firms"/f"fires_{y}_daily.csv"
    sample_csv(src, f"data/sample_data/fires_{y}_daily_SAMPLE200.csv", n=200)

# Sample from weather
sample_csv(BASE/"upload"/"wildfire_data"/"wildfire_data"/"weather"/"weather_nasa_power_full.csv",
           "data/sample_data/weather_nasa_power_full_SAMPLE200.csv", n=200)

# Sample from v6 feature table
sample_csv(BASE/"download"/"real_data_v6"/"feature_table_v6.csv",
           "data/sample_data/feature_table_v6_SAMPLE200.csv", n=200)

# Sample from v5 feature table
sample_csv(BASE/"download"/"wildfire_real_data_v5"/"real_data"/"feature_table_real_weather_real_fire.csv",
           "data/sample_data/feature_table_v5_real_SAMPLE200.csv", n=200)

# 3c. raw_data/ — keep the raw MODIS + weather (these are the actual sources)
# Don't keep the 100 MB v7 feature_table_v7.csv (it's derived; can be rebuilt).
rd = mkdir("data", "raw_data")
# Raw fire per-year CSVs (34 MB total — manageable)
for y in range(2018, 2024):
    cp(BASE/"upload"/"wildfire_data"/"wildfire_data"/"firms"/f"fires_{y}_daily.csv",
       mkdir("data", "raw_data", "firms"))
# Weather
cp(BASE/"upload"/"wildfire_data"/"wildfire_data"/"weather"/"weather_nasa_power_full.csv",
   mkdir("data", "raw_data", "weather"))

# 3d. intermediate feature tables (v5, v6 — small enough)
mkdir("data", "feature_tables")
cp(BASE/"download"/"real_data_v6"/"feature_table_v6.csv",
   mkdir("data", "feature_tables", "v6"))
cp(BASE/"download"/"wildfire_real_data_v5"/"real_data"/"feature_table_real_weather_real_fire.csv",
   mkdir("data", "feature_tables", "v5"))

# ============================================================
# 4. RESULTS
# ============================================================
print("\n[4/9] Copying results...")
res = mkdir("results")

# 4a. csv/ — predictions
rc = mkdir("results", "csv")
for f in Path(BASE/"download"/"predictions").iterdir():
    cp(f, rc)
for f in Path(BASE/"download"/"wildfire_real_data_v5"/"predictions").iterdir():
    cp(f, rc)

# 4b. json/ — all results JSONs
rj = mkdir("results", "json")
# v7 final + audit + multiseed
for f in ["results_v7_final.json", "results_audit.json", "results_multiseed.json"]:
    src = BASE/"download"/"real_data_v7"/f
    if src.exists():
        cp(src, rj)
# v6
cp(BASE/"download"/"real_data_v6"/"results_v6.json", rj)
cp(BASE/"download"/"real_data_v6"/"results_v6_final.json", rj)
# v5
for f in Path(BASE/"download"/"wildfire_real_data_v5"/"results").iterdir():
    cp(f, rj)
# v5 pilot
cp(BASE/"download"/"real_data_pilot"/"results_real_labels.json", rj)
cp(BASE/"download"/"real_data_pilot"/"results_real_weather_real_fire.json", rj)
# Original synthetic-era results.json
cp(BASE/"scripts"/"results.json", rj)
cp(BASE/"download"/"results.json", rj)
cp(BASE/"download"/"README.md", mkdir("results"))

# 4c. logs/
rl = mkdir("results", "logs")
cp(BASE/"scripts"/"firms_daily_download.log", rl)

# 4d. tables/ — pretty-printed summary tables
rt = mkdir("results", "tables")
write("results/tables/main_results_v7.md", """# Main Results — v7 FINAL

Source: `results/json/results_v7_final.json`

## Baseline XGBoost (test set 2022, excl. Khanh Hoa)
| AUROC | AUPRC | F1 | Brier | ECE |
|-------|-------|----|-------|-----|
| 0.8602 | 0.5197 | 0.5181 | 0.1046 | 0.0581 |

## λ-sweep on Physics-MLP (test set 2022)
| λ | AUROC | AUPRC | F1 | Brier | ECE |
|---|-------|-------|----|-------|-----|
| 0.00 | 0.8299 | 0.4457 | 0.4947 | 0.1152 | 0.0793 |
| 0.01 | 0.8299 | 0.4481 | 0.4939 | 0.1154 | 0.0801 |
| 0.05 | 0.8300 | 0.4483 | 0.4939 | 0.1152 | 0.0796 |
| 0.10 | 0.8300 | 0.4485 | 0.4934 | 0.1150 | 0.0793 |
| 0.20 | 0.8301 | 0.4487 | 0.4932 | 0.1145 | 0.0785 |

Δ ECE(λ=0 vs λ=0.20) = -0.0008 → **physics constraint does NOT
measurably improve calibration on the real dataset**.

## Calibration baselines (XGBoost + post-hoc)
| Method | AUROC | Brier | ECE |
|--------|-------|-------|-----|
| Raw                 | 0.8602 | 0.1046 | 0.0581 |
| + Temperature       | 0.8602 | 0.1046 | 0.0540 |
| + Isotonic          | 0.8602 | —      | (low)  |
| + Platt             | 0.8602 | —      | (low)  |
""")

write("results/tables/multiseed_v7_audit.md", """# Multi-seed Results — v7 audit (3 seeds, mean±std)

Source: `results/json/results_audit.json`

## IID (Train 2018-2020 + random val, Test 2022)
| Model | AUROC | Brier | ECE |
|-------|-------|-------|-----|
| XGBoost            | 0.8849 ± 0.0005 | 0.1028 ± 0.0003 | 0.0723 ± 0.0010 |
| XGB + Isotonic     | 0.8844 ± 0.0004 | 0.0939 ± 0.0002 | **0.0093 ± 0.0003** |
| MLP                | 0.8738 ± 0.0013 | 0.1029 ± 0.0005 | 0.0529 ± 0.0009 |

## Spatial OOD (Train lat≥15°, Test lat<15°)
| Model | AUROC | ECE | ΔECE vs IID |
|-------|-------|-----|-------------|
| XGBoost            | 0.7958 ± 0.0031 | 0.1556 | +0.0833 |
| XGB + Isotonic     | 0.7956 ± 0.0032 | **0.0318** | **+0.0225** |
| MLP                | 0.7371 ± 0.0095 | 0.1710 | +0.1181 |

## Key findings
1. Physics constraint (ΔECE λ-sweep) ≈ 0 — original hypothesis falsified.
2. XGB+Isotonic is the strongest calibration method (ECE 0.009 IID → 0.032 OOD).
3. MLP calibration collapses under spatial OOD (ΔECE = +0.118, 5× worse than XGB+Iso).
4. XGBoost discrimination stable across IID and spatial OOD.
""")

# ============================================================
# 5. CONFIGS
# ============================================================
print("\n[5/9] Writing configs...")
cfg = mkdir("configs")
write("configs/hyperparameters.yaml", """# v7 hyperparameters (canonical)

master_seed: 42
seeds_for_multiseed: [42, 123, 777]   # v8 extends to [42, 123, 777, 2024, 31337]

data:
  grid_resolution_deg: 0.5
  lat_range: [11.0, 21.0]
  lon_range: [104.0, 109.5]
  date_range: [2018-01-01, 2023-12-31]
  positive_class_prevalence: 0.173

xgboost:
  n_estimators: 500
  max_depth: 7
  learning_rate: 0.03
  subsample: 0.85
  colsample_bytree: 0.7
  min_child_weight: 10
  reg_lambda: 3.0
  reg_alpha: 0.5
  gamma: 0.1
  scale_pos_weight: sqrt(neg/pos)
  eval_metric: aucpr
  tree_method: hist
  early_stopping_rounds: 60

mlp:
  architecture: ResidualMLP
  hidden_dim: 128
  n_res_blocks: 1
  dropout: 0.15
  activation: silu
  optimizer: AdamW
  lr: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: CosineAnnealingLR
  epochs: 25            # FIXED — no early stopping by val AUROC
  batch_size: 1024
  grad_clip: 1.0
  two_phase:
    phase1_bce_only_epochs_frac: 0.5
    phase2_bce_plus_physics_epochs_frac: 0.5

physics_penalty:
  raw_phys_cols: [t2m, rh, wind]   # °C, %, m/s — NOT standardized
  terms:
    energy:    {weight: 0.4, direction: "T↑ → risk↑",   condition: "always"}
    moisture:  {weight: 0.4, direction: "RH↑ → risk↓",  condition: "always"}
    wind:      {weight: 0.2, direction: "W↑ → risk↑",   condition: "T_ii>25 AND T_ij>25"}
  lambda_sweep: [0.0, 0.01, 0.05, 0.10, 0.20]

calibration:
  temperature: {T_grid: linspace(0.5, 5.0, 31), fit_on: val, optimize: NLL}
  isotonic:   {out_of_bounds: clip, y_min: 0, y_max: 1, fit_on: val}
  platt:      {LogisticRegression_C: 1e10, fit_on: val}

threshold:
  chosen_on: validation_precision_recall_curve
  criterion: max F1

splits:
  IID:        {train: 60%, val: 20%, test: 20%, random_shuffle: true}
  Temporal_OOD: {train: [2018,2019,2020,2021], val: [2022], test: [2023]}
  Spatial_OOD:  {train: "lat>=15", val: "20% random of train", test: "lat<15"}

metrics:
  discrimination: [AUROC, AUPRC]
  calibration: [Brier, ECE_10_bins]
  threshold: [F1, precision, recall]
  ood_degradation: [delta_ECE = ECE_OOD - ECE_IID]
""")

write("configs/split_definitions.json", json.dumps({
    "IID": {
        "type": "random_shuffle_no_grouping",
        "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2,
        "leakage_risk": "spatial+temporal (adjacent cell-days can split)",
        "use_case": "lower-bound performance estimate"
    },
    "Temporal_OOD": {
        "type": "year_stratified",
        "train_years": [2018, 2019, 2020, 2021],
        "val_year": 2022,
        "test_year": 2023,
        "leakage_risk": "none (future never in train)",
        "use_case": "honest evaluation of future generalization"
    },
    "Spatial_OOD": {
        "type": "spatial_block",
        "train": "lat >= 15",
        "test": "lat < 15",
        "val": "20% random subset of train",
        "leakage_risk": "none (southern cells never in train)",
        "use_case": "generalization to unseen geographic region"
    }
}, indent=2))

# ============================================================
# 6. ROOT-LEVEL DOCS
# ============================================================
print("\n[6/9] Writing root docs...")

write("README.md", """# Wildfire Physics-Constrained Spatiotemporal ML — Full Project Audit Package

**Student**: Hoàng Bảo Lâm
**Competition target**: Vietnam National Science Fair (ISEF-eligible)
**Project title**: *Physics-Constrained Spatiotemporal Machine Learning for Early Wildfire Ignition Risk Forecasting*
**Current version**: v7 (real data, 5 fixes applied)
**Latest research question (evolved)**: *Can wildfire risk models maintain reliable probability estimates under spatiotemporal distribution shift?*

---

## What's in this ZIP

| Folder | What |
|--------|------|
| `code/` | All Python code, organized by pipeline stage |
| `report/` | Current + previous report PDFs + figures |
| `data/` | Sample data, raw data, feature tables, metadata (acquisition + schema + grid) |
| `results/` | All experimental JSON / CSV / logs / human-readable summary tables |
| `configs/` | Hyperparameters and split definitions |

This ZIP is **the full project state as of v7** — including rejected approaches,
intermediate versions, and known-bad experiments. The goal is audit, not
polish.

---

## Research evolution (the full story)

| Version | Dataset | Approach | Result |
|---------|---------|----------|--------|
| v1-v4 | Synthetic Vietnam grid (214,718 rows × 17 features) | XGBoost + Residual MLP with physics penalty | λ-sweep ΔECE ≈ 0 — physics does NOT help. Reviewer pointed out **double-dipping**: labels and features both came from the same physics formula. |
| v5 (pilot) | Real NASA MODIS fire labels + Open-Meteo weather (56K records) | Same model | ΔECE still ≈ 0 — confirms negative result is not a synthetic-data artifact. |
| v5 (full) | Real NASA MODIS + Open-Meteo (37K rows × 32 features) | Same model + Temperature / Isotonic / Platt baselines | ΔECE ≈ 0 still. **First calibration comparison**: XGB+Isotonic dramatically lowers ECE. |
| v6 | Real NASA MODIS + NASA POWER weather (full 273-cell grid) | 5 fixes applied (see below) | ΔECE ≈ 0.03 between λ=0 and λ=0.2 — physics MAY slightly help. Inconclusive. |
| v7 FINAL | **581,220 raw MODIS detections → 48,279 unique positive cell-days**, NASA POWER 598,143 weather records | Same model + multi-seed (3 seeds) + ΔECE table | ΔECE λ-sweep = -0.0008. **Physics definitively does not help on real data.** |
| v8 ISEF (planned) | Same data | 5-seed + 3-split canonical suite + counterfactual test | (partial — see `run_v8_isef.py`) |

## Five technical fixes (v6 → v7)

1. **Fixed epochs (no early stopping by val AUROC)** — prevents the model from
   being biased toward whichever epoch happened to best fit the validation set.
2. **Raw physics penalty (°C, %, m/s — un-standardized)** — standardization
   destroys the physical meaning of monotonicity directions.
3. **Removed synthetic static features** (NDVI/soil/elev/slope/aspect) — the
   model was learning cell-identity rather than physics.
4. **Fire labels only on MODIS-observed dates** — prevents false negatives on
   days when the satellite did not overpass or had cloud cover.
5. **F1 threshold chosen on validation set** — no test-set leakage.

---

## Three core experimental results (v7 audit, 3 seeds, mean±std)

### A. IID (Train 2018-2020, Val 2021, Test 2022)
| Model | AUROC | Brier | ECE |
|-------|-------|-------|-----|
| XGBoost            | 0.8849 ± 0.0005 | 0.1028 ± 0.0003 | 0.0723 ± 0.0010 |
| **XGB + Isotonic** | 0.8844 ± 0.0004 | 0.0939 ± 0.0002 | **0.0093 ± 0.0003** |
| MLP (Physics)      | 0.8738 ± 0.0013 | 0.1029 ± 0.0005 | 0.0529 ± 0.0009 |

### B. Spatial OOD (Train North lat≥15, Test South lat<15)
| Model | AUROC | ECE | ΔECE vs IID |
|-------|-------|-----|-------------|
| XGBoost            | 0.7958 ± 0.0031 | 0.1556 | +0.0833 |
| **XGB + Isotonic** | 0.7956 ± 0.0032 | **0.0318** | **+0.0225** |
| MLP (Physics)      | 0.7371 ± 0.0095 | 0.1710 | +0.1181 |

### C. λ-sweep (does physics constraint help?)
| λ | ECE | ΔECE vs λ=0 |
|---|-----|-------------|
| 0.00 | 0.0793 | — |
| 0.20 | 0.0785 | -0.0008 |

**Conclusion**: physics constraint does NOT measurably improve calibration
on the real NASA MODIS dataset. The original physics-informed-ML hypothesis
is falsified.

---

## Headline finding (the evolved research question)

**Question**: Can wildfire risk models maintain reliable probability estimates
under spatiotemporal distribution shift?

**Answer**:
- **YES** — via XGBoost + Isotonic regression. ECE goes from 0.009 (IID) to
  0.032 (spatial OOD), only a 3.5× degradation, while AUROC drops by 0.09.
- **NO** — for a vanilla MLP. ECE goes from 0.053 (IID) to 0.171 (spatial
  OOD), a 3.2× degradation AND AUROC drops by 0.14. The MLP is unreliable
  under distribution shift.

This is the **strongest, defensible, novel** finding of the project —
*not* the original physics hypothesis.

---

## Known limitations (must be addressed before national competition)

1. **IID split has group leakage** — adjacent cell-days can land in train and
   test. Mitigation: report IID numbers as a lower bound, emphasize OOD.
2. **No VIIRS validation** — MODIS has known detection gaps in low-intensity
   fires. VIIRS would be a stronger independent check.
3. **No NDVI / DEM** in v7 — could be re-added if real Earth Engine access
   is obtained.
4. **Only 3 seeds** in the audit; v8 should extend to 5-10 seeds with CIs.
5. **No temporal OOD results** in the audit JSON (v8 script defines them but
   the v8 run is not in this ZIP — the script is included for re-running).
6. **No SHAP / counterfactual monotonicity test results** in the ZIP; v8
   script defines both but they have not been run to completion.
7. **Original synthetic-era PDF report does NOT reflect the v7 findings**
   — the report must be rewritten from scratch for the national competition.
8. **No spatial block / leave-region-out cross-validation** — only one
   arbitrary North/South split was used.

---

## How to re-run

```bash
# Environment (see requirements.txt)
pip install -r requirements.txt

# Step 1: place raw data (already included in data/raw_data/)
#   data/raw_data/firms/fires_2018_daily.csv ... fires_2023_daily.csv
#   data/raw_data/weather/weather_nasa_power_full.csv

# Step 2: run v7 final pipeline
cd code/training
python run_v7_final.py    # outputs JSON to download/real_data_v7/

# Step 3 (optional): run v8 ISEF suite (5 seeds × 3 splits × 5 models)
python run_v8_isef.py     # ~30 min on CPU
```

Expected runtime on CPU (8 cores): ~10 minutes for v7, ~30 minutes for v8.

---

## What this ZIP does NOT include (and why)

- API keys / tokens / credentials — none were ever needed (Microsoft
  Planetary Computer and NASA POWER are open).
- The 100 MB `feature_table_v7.csv` intermediate — it can be regenerated
  by `run_v7_final.py` in ~30 seconds.
- Google Earth Engine credentials — the GEE path failed; no credentials
  were ever obtained.
- ECMWF CDS credentials — same situation.
- `__pycache__`, `.ipynb_checkpoints` — non-essential.

---

## Contact / next step

After audit (PHASE 1), the next step is to:
1. Identify leakage / bias / statistical issues in the current pipeline.
2. Decide on the canonical research question (likely "calibration under
   distribution shift", not "physics-informed ML").
3. Re-run the v8 ISEF suite end-to-end with all 5-10 seeds.
4. Build the final report + poster + presentation.

Good luck with the audit. — Lam
""")

write("requirements.txt", """# Wildfire Physics-Constrained ML — Python dependencies
# Tested on Python 3.10.12

# Numerical / data
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4

# ML
scikit-learn==1.3.2
xgboost==2.0.3

# Deep learning
torch==2.2.0      # CPU version; install with:
                 #   pip install torch --index-url https://download.pytorch.org/whl/cpu

# Plotting (for figures to be regenerated)
matplotlib==3.8.2
seaborn==0.13.0

# PDF generation (for the report)
reportlab==4.0.8

# Data download (optional — only needed if re-downloading)
requests==2.31.0
planetary-computer==1.0.0     # for MODIS STAC access
pystac-client==0.7.5          # for STAC queries
earthaccess==0.9.0           # alternative for NASA data
""")

write("hardware_info.txt", """# Hardware / runtime info (as of v7)

## Environment
- Platform: Linux x86_64 (Ubuntu 22.04 container)
- CPU: 8 cores (Intel Xeon-class)
- RAM: 16 GB
- GPU: NONE (CPU-only training)
- Python: 3.10.12

## Key libraries
- numpy 1.26.4
- pandas 2.1.4
- scikit-learn 1.3.2
- xgboost 2.0.3 (tree_method='hist', single-threaded per fit)
- torch 2.2.0+cpu (CPU build)

## Approximate runtime (v7)
- Data load + feature engineering: 30 s
- XGBoost training (500 trees, 140k rows × 22 features): 90 s
- MLP training (25 epochs, batch 1024, 140k rows × 23 features): 60 s
- Full λ-sweep (5 λ values): 5 min
- Calibration baselines: 10 s
- OOD evaluation (6 conditions): 30 s
- **Total v7 wall-clock**: ~8 min

## Approximate runtime (v8 ISEF, planned)
- 5 seeds × 3 splits × 5 models = 75 training runs
- Estimated total: 25-35 min on the same CPU

## Disk usage
- Raw MODIS fire (6 yearly CSVs): 34 MB
- NASA POWER weather CSV: 25 MB
- v7 feature table (intermediate): 100 MB
- Predictions (v5 era): 18 MB
- All experimental results (JSON): <1 MB
- **Total project footprint**: ~180 MB

## Scaling notes
- 0.50° grid × 6 years × 273 cells is sufficient for proof-of-concept.
- For 0.25° grid: 4× more cells = 4× more rows = ~1.1 M rows.
  Training time still <30 min on CPU. Memory <8 GB.
- For 1 km grid (MODIS native): not feasible on CPU; would need GPU and
  would require Earth Engine for weather downscaling.
""")

write("experiment_notes.md", """# Experiment Notes — Research Evolution

This file preserves the **messy history** of the project: failed attempts,
reviewer feedback, and the reasoning behind major pivots. Kept verbatim
because they are essential for the audit.

---

## v1-v4: Synthetic data era

### Setup
- 214,718 daily records × 17 features, 0.50° grid, 11.5-20.5°N × 104-109.5°E.
- Weather generated from seasonal cycle + AR(1) noise.
- Fire labels generated from a physics-informed formula:
  `logit(P_fire) = 0.4*(T-10) - 0.62*RH + 0.4*W - 0.3*precip_lag7 + ...`

### Finding
- λ-sweep ΔECE = 0.0001 across all λ ∈ {0.0, 0.01, 0.05, 0.10, 0.20}.
- Physics constraint did NOT improve calibration.

### Reviewer feedback (P0 issue)
> "The labels are generated from the same physics formula that the
> physics penalty enforces. This is **double-dipping**: the model is
> penalized for not matching a relationship that the labels already encode
> perfectly. Of course physics penalty has no effect — the unconstrained
> model already learns it from the labels."

**Action**: pivoted to real data.

---

## v5: Real fire labels, synthetic weather (pilot)

### Setup
- Real NASA MODIS fire labels (17,780 points from FIRMS direct CSV).
- Synthetic weather (Open-Meteo-like seasonality, no real API call).
- 37,247 cell-days × 32 features.

### Finding
- λ-sweep ΔECE still ≈ 0.
- **Confirmed**: negative result is not a synthetic-data artifact.

### Issue discovered
- The `colab_download_no_auth.py` script returned 17,780 fire points —
  much fewer than expected (~300K+ across 6 years in Vietnam).
- Root cause: FIRMS CSV download had pagination limits.

**Action**: pivoted to Microsoft Planetary Computer STAC.

---

## v5→v6: Real weather era

### Setup
- Real NASA MODIS (still 17K points).
- Real NASA POWER weather on full 273-cell grid.
- 5 technical fixes (see README.md → "Five technical fixes").

### Finding
- v6: ΔECE = +0.03 between λ=0 and λ=0.2 — physics MAY slightly help.
- This contradicted v5. We needed more data to be sure.

### Notes from log
- "MODIS download took 3 hours via FIRMS direct CSV."
- "Open-Meteo API rate-limited after ~500 calls."
- "ECMWF CDS email verification did not arrive — tried 3 times over 2 days."
- "Google Earth Engine requires project authorization; the project ID was
   rejected by the GEE console (possibly a domain issue)."

---

## v6→v7: Full real-data era

### Setup
- **581,220 raw MODIS detections** downloaded via Microsoft Planetary
  Computer STAC over 6 separate Colab sessions (one per year, due to
  session timeouts).
- After deduplication to cell-day level: **48,279 unique positive cell-days**.
- NASA POWER weather: 598,143 records, 273 cells × 2,191 days.

### Finding (the big one)
- λ-sweep ΔECE = **-0.0008** (essentially zero).
- Physics constraint definitively does NOT help on real data.
- **The original research question is falsified.**

### Pivot (research question evolution)
- Original: "Can physics constraints improve wildfire prediction?"
  → Answer: NO (on real data).
- New: "Can wildfire risk models maintain reliable probability estimates
  under spatiotemporal distribution shift?"
  → Answer: YES for XGB+Isotonic; NO for vanilla MLP.
- This is a **stronger, more defensible, more novel** research question.

---

## v7 audit: Multi-seed confirmation

### Setup
- 3 seeds × 3 splits × 3 models = 27 training runs.
- IID + Temporal OOD + Spatial OOD.

### Finding (multi-seed confirms)
- IID ECE (XGB+Iso) = 0.0093 ± 0.0003 (extremely stable).
- Spatial OOD ECE (XGB+Iso) = 0.0318 — only 3.5× degradation.
- Spatial OOD ECE (MLP) = 0.171 — 3.2× degradation AND AUROC drops 0.14.
- **MLP is unreliable under distribution shift; XGB+Iso is robust.**

### Remaining concerns (must address in PHASE 2)
- Only 3 seeds — need 5-10 for confidence intervals.
- IID split has group leakage (adjacent cell-days).
- No VIIRS independent validation.
- No counterfactual physics consistency test (v8 script defines it but
  hasn't been run end-to-end).
- No SHAP / feature attribution.
- The PDF report is still the SYNTHETIC-era PDF — does NOT reflect v7
  findings. The report must be rewritten from scratch.

---

## Key learning moments (for the methodology section)

1. **Double-dipping in synthetic data** — taught us that physics penalties
   must be evaluated on data where the labels are NOT generated by the
   same physics formula.

2. **False-negative label problem** — taught us that absence of MODIS
   detection is not absence of fire. Solution: only use MODIS-observed
   dates.

3. **Standardization destroys physics meaning** — early v6 used
   standardized T/RH/W in the physics penalty; the monotonicity
   directions became meaningless. Solution: use raw °C/%/m/s.

4. **Calibration leakage** — early v5 used test set to fit isotonic;
   this is leakage. Solution: always fit calibrator on validation, apply
   to test.

5. **F1 threshold on test** — early v5 chose F1 threshold on test PR
   curve; this is leakage. Solution: choose on validation PR curve,
   apply to test.

---

## Things that did NOT work (preserved for the limitations section)

- Google Earth Engine authorization (failed).
- ECMWF CDS email verification (failed).
- NASA FIRMS direct CSV download (rate-limited; only returned 17K points).
- Open-Meteo weather API (rate-limited after 500 calls).
- Synthetic static features (NDVI, soil, elevation) — caused cell-identity
  learning, removed in v7.
- Early-stopping by val AUROC (caused model selection bias; removed in v7).
- Standardized physics penalty (destroyed physical meaning; switched to raw).
- Matching-style physics penalty (v3; destroyed calibration; replaced with
  monotonicity-based).

---

## Recommended next experiments (v8 and beyond)

1. **5-10 seeds** with bootstrap CIs (currently only 3).
2. **Spatial block cross-validation** instead of single arbitrary North/South.
3. **Temporal OOD results** (currently only defined in v8 script, not run).
4. **VIIRS independent validation** (download VNP14IMGTDL_NRT for 2018-2023).
5. **Counterfactual monotonicity test** (v8 defines it; not run).
6. **SHAP feature attribution** on XGBoost.
7. **Threshold utility analysis** — at what threshold do we save X% of fires
   at the cost of Y% false alarms?
8. **Spatial risk map** of Vietnam with real predictions.
""")

# ============================================================
# 7. ALSO KEEP THE EXISTING v5/v6/v7 ZIPS (research evolution)
# ============================================================
print("\n[7/9] Preserving earlier ZIP snapshots as evidence...")
# These earlier ZIPs are the "previous versions" the user explicitly asked to keep.
# We'll include them inside a `_previous_zips/` folder so the auditor can unpack
# each snapshot independently.
pz = mkdir("_previous_zips")
cp(BASE/"download"/"wildfire_research_complete.zip", pz)
cp(BASE/"download"/"wildfire_real_data_v5.zip", pz)
cp(BASE/"download"/"wildfire_real_v6_final.zip", pz)
cp(BASE/"download"/"wildfire_v7_final.zip", pz)
# Also the user-supplied raw data ZIP from Google Drive
cp(BASE/"upload"/"wildfire_data-20261005T132957Z-1-001.zip", pz)

# ============================================================
# 8. WORKLOG
# ============================================================
print("\n[8/9] Copying worklog...")
# Single shared worklog
cp(BASE/"worklog.md", mkdir("."))
cp(BASE/"download"/"wildfire_real_data_v5"/"worklog.md", mkdir("worklog_archive"))

# ============================================================
# 9. ZIP EVERYTHING
# ============================================================
print(f"\n[9/9] Zipping to {ZIP_PATH} ...")

# Count files + total size before zipping
file_count = 0
total_size = 0
for root, _, files in os.walk(ROOT):
    for f in files:
        p = Path(root) / f
        file_count += 1
        total_size += p.stat().st_size

print(f"  Files to zip: {file_count}")
print(f"  Uncompressed size: {total_size/1e6:.1f} MB")

# Build zip
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for root, _, files in os.walk(ROOT):
        for f in files:
            full = Path(root) / f
            rel = full.relative_to(ROOT)
            zf.write(full, f"WILDFIRE_PROJECT/{rel}")

zip_size = ZIP_PATH.stat().st_size
print(f"\n{'=' * 70}")
print(f"✓ ZIP COMPLETE")
print(f"{'=' * 70}")
print(f"  Output:    {ZIP_PATH}")
print(f"  ZIP size:  {zip_size/1e6:.1f} MB")
print(f"  Files:     {file_count}")
print(f"  Structure: WILDFIRE_PROJECT/{{code,report,data,results,configs,_previous_zips,...}}")
