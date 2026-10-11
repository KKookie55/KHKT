#!/usr/bin/env python3
"""
Finalize v8.6 — apply 3 audit fixes:
  1. Fix H4 terminology (no causal claim about land cover/elevation)
  2. Separate MODIS/VIIRS Phase A clearly in README
  3. Build submission product mockup (Reliability-Aware Forecasting UI)
"""
import os, shutil, zipfile, json
from pathlib import Path

BASE = Path("/home/z/my-project")
ROOT = BASE / "download" / "WILDFIRE_PROJECT_v86"
ZIP_PATH = BASE / "download" / "WILDFIRE_PROJECT_v86_final.zip"

print("=" * 70); print("v8.6 FINAL — Apply 3 audit fixes"); print("=" * 70)

# Make sure ROOT exists (extract from existing zip if needed)
if not ROOT.exists():
    print("Extracting existing v86_phaseA.zip...")
    ROOT.mkdir(parents=True)
    with zipfile.ZipFile(BASE / "download" / "WILDFIRE_PROJECT_v86_phaseA.zip") as zf:
        zf.extractall(ROOT.parent)

# ============================================================
# FIX 1 + 2: Update READMEs with corrected H4 + MODIS/VIIRS separation
# ============================================================
print("\n[FIX 1+2] Updating READMEs...")

# Main README
(ROOT / "README.md").write_text("""# Wildfire Physics-Constrained ML — v8.6 FINAL

**Student**: Hoàng Bảo Lâm
**Target**: Vietnam National Science Fair (ViSEF / ISEF-eligible)
**Version**: v8.6 FINAL (post-audit + Phase A spatial diagnostic + reliability product)

## Three-tier contribution framework

### Tier 1: Dataset & methodology (v8.5)
- 598K cell-day panel (273 cells × 2191 days, 0.5° grid, 2018-2023)
- Real MODIS + NASA POWER + VIIRS (independent validation)
- 5 splits: IID group-blocked + Temporal OOD + 3 Spatial OOD boundaries
- Leakage-safe preprocessing (calendar-aware features, train-only normalization)
- 7 audit fixes applied to v7 → v8.5

### Tier 2: Trustworthiness under distribution shift (HEADLINE FINDING)
**Multi-sensor validation reveals that MODIS-only evaluation OVERESTIMATES
generalization by 0.31 AUROC on spatial OOD**.

#### MODIS labels (baseline, 1 seed)
| Split | AUROC |
|-------|-------|
| IID | 0.947 |
| Temporal OOD | 0.934 |
| Spatial N→S | 0.892 |
| Spatial S→C | 0.924 |
| Spatial C+S→N | 0.927 |

#### VIIRS labels (independent validation, 5 seeds mean ± std)
| Split | AUROC | Δ vs MODIS |
|-------|-------|------------|
| IID | 0.843 ± 0.0006 | -0.10 |
| Temporal OOD | 0.843 | -0.09 |
| **Spatial N→S** | **0.584 ± 0.0019** | **-0.31** |
| Spatial S→C | 0.880 ± 0.0022 | -0.04 |
| Spatial C+S→N | 0.700 ± 0.0009 | -0.23 |

→ v8.5 MODIS AUROC values are UPPER BOUNDS; true generalization with VIIRS
labels is much lower, especially on spatial OOD.

### Tier 3: Physics hypothesis (falsifiable)
Physics penalty enforces monotonicity on the features it directly constrains
(W, RH: 0% violation) but does NOT help on T or precip — a nuanced result.

## Phase A — Spatial failure diagnosis (no new data needed)

### A1: Spatial autocorrelation (fire-conditioned neighbor probability)
- Moran's I = 0.186 ± 0.214 — moderate spatial clustering
- **P(neighbor fire | current cell has fire) ≈ 0.61 at short distances**
- vs random fire rate ≈ 0.29
- Distance-decay: drops to ~0.45 at 170-190 km
- → Fire labels have substantial spatial structure beyond base-rate

### A2: Prevalence-controlled N→S evaluation (VIIRS only)
- Original South VIIRS AUROC: 0.582
- Prevalence-balanced (subsample negatives): 0.582 ± 0.0000
- Case-control 1:1: 0.582 ± 0.0017
- **Verdict**: Prevalence balancing does NOT recover performance
- → The N→S failure is NOT primarily a base-rate mismatch problem

### A3: Spatial proxy ablation — KEY CAUTIONARY FINDING

#### MODIS A3 (working case — model succeeds on N→S)
| Variant | AUROC |
|---------|-------|
| v1: weather only | 0.892 |
| v2: weather + lat/lon | 0.893 (Δ = +0.001) |
| v3: weather + lat_band | 0.854 (Δ = -0.038) |

→ With MODIS labels, weather-only is sufficient; lat/lon barely helps.
   (MODIS labels follow weather patterns cleanly — large/hot fires.)

#### VIIRS A3 (failure case — model fails on N→S)
| Variant | AUROC |
|---------|-------|
| v1: weather only | **0.582** |
| v2: weather + lat/lon | **0.811 (Δ = +0.229!)** |
| v3: weather + lat_band 10 | 0.560 (Δ = -0.022) |
| v4: weather + lat_band5 + lon_band5 | 0.548 (Δ = -0.034) |

→ With VIIRS labels, raw lat/lon recovers +0.23 AUROC — but this is
   "geography memorization", NOT physics generalization:
   - Coarse lat_band HURTS (train has only North bands → South is OOD)
   - Only fine-grained lat/lon lets the model memorize specific cells
   - Justifies audit fix #6 (drop lat/lon from features)
   - Reveals: substantial spatial structure exists that weather cannot represent

### Hypothesis tests (H1-H5)

| Hypothesis | Status | Result |
|-----------|--------|--------|
| H1: Covariate shift | ✓ Confirmed | 5/9 features significant (KS p<0.001, d>0.2). South warmer+drier+longer dry spells. |
| H2: Concept shift | ✓ Confirmed | Avg correlation P(fire\|feature) N vs S = 0.81 (not 1.0). Same weather → different fire prob. |
| H3: Label bias (regional) | ✗ No strong effect | MODIS/VIIRS ratio N=0.95 vs S=0.84, Δ=0.11. |
| **H4: Unobserved spatial structure** | **✓ Strongly implicated, NOT causally identified** | Adding raw coordinates recovers +0.23 AUROC — substantial spatial structure exists that weather cannot represent. Candidate mechanisms (not tested directly): land cover, vegetation/NDVI, elevation, agricultural activity. Phase B (external spatial data) would directly test these, but is currently blocked by environment network restrictions. |
| H5: IID spatial leakage | ✓ Confirmed | Match rate at d=0 = 89.8%. IID AUROC was inflated by spatial autocorrelation. |

## Overall verdict — Why N→S collapses with VIIRS labels

4 contributing factors:
1. **Covariate shift** — South warmer/drier/longer dry spells
2. **Concept shift** — P(fire|weather) correlation N vs S = 0.81
3. **Unobserved spatial structure** — strongly implicated (not causally identified)
4. **IID spatial leakage** — IID AUROC was inflated by autocorrelation

## Submission product: Reliability-A Wildfire Forecasting System

We propose a wildfire forecasting product that **explicitly communicates model
uncertainty**, rather than hiding it behind a single number.

See `product/reliability_dashboard.html` and `product/README_product.md` for
the mockup. Key innovation: when the model detects distribution shift (spatial
OOD, weather distribution shift, or unobserved spatial structure), it
**lowers the reliability indicator** and explains why.

This turns the Phase A findings (which show that weather-only models fail on
spatial OOD) from a "weakness" into a "feature": the model knows when not to
trust itself.

## Directory structure

```
WILDFIRE_PROJECT_v86/
├── README.md                          ← this file
├── code/                              ← 9 scripts by pipeline stage
├── report/
│   ├── current/                       ← Old report PDF (synthetic-era, needs rewrite)
│   └── figures/                       ← 14 v8.5 figures + 7 Phase A figures
├── data/raw_data/{firms,viirs,weather}/  ← 13 raw NASA CSVs
├── v8.5_canonical/                    ← 200 runs + panels + counterfactual
├── phaseA_diagnosis/                  ← Phase A spatial failure diagnosis
└── product/                           ← Reliability-Aware Forecasting UI mockup
    ├── reliability_dashboard.html
    └── README_product.md
```

## For the national competition

The strongest paper narrative:
1. **Audit-driven methodology** (v7 → v8.5): identified and fixed 7 issues
2. **Multi-sensor validation reveals MODIS overestimation** (Tier 2 finding)
3. **Spatial failure decomposition** (Phase A): 4 contributing factors
4. **Cautionary finding**: raw lat/lon gives +0.23 AUROC, but this is geography
   memorization, not physics generalization — justifies audit fix #6
5. **Reliability-aware product**: model knows when not to trust itself

For judge Q&A "Why does N→S collapse?":
> "Decomposed into 4 factors: covariate shift (warmer/drier south), concept shift
> (different fire-weather relationship), unobserved spatial structure (strongly
> implicated — adding raw coordinates recovers +0.23 AUROC but is geography
> memorization, not physics), and spatial autocorrelation in IID splits. The
> precise mechanism for the unobserved spatial structure — candidate drivers
> include land cover, vegetation, elevation, and agricultural activity — was
> not directly tested because external spatial datasets were unavailable in
> the development environment. This is an explicit limitation, and is also
> the motivation for the reliability-aware product: when the model detects
> it is operating outside its training distribution, it lowers its reliability
> indicator rather than producing overconfident predictions."
""")

# Phase A README with corrected H4 + clear MODIS/VIIRS separation
(ROOT / "phaseA_diagnosis" / "README.md").write_text("""# Phase A — Spatial Failure Diagnosis

## Why does N→S AUROC collapse from 0.84 (IID) to 0.58 (Spatial OOD) with VIIRS labels?

We tested 5 hypotheses (H1-H5) plus 3 analyses (A1-A3).

## CRITICAL: MODIS vs VIIRS Phase A results are SEPARATE

**MODIS labels** (in `phaseA_diagnosis.json`) — the WORKING case (model succeeds on N→S):
- N→S AUROC stays at 0.892 even after prevalence balancing
- Adding lat/lon barely helps (+0.001 AUROC)
- → With MODIS labels, weather→fire is learnable; no spatial proxy needed

**VIIRS labels** (in `phaseA_viirs_diagnosis.json`) — the FAILURE case (model fails on N→S):
- N→S AUROC collapses to 0.582
- Prevalence balancing does NOT recover
- Adding raw lat/lon recovers +0.23 AUROC (geography memorization)
- → With VIIRS labels, weather→fire is weaker; model needs spatial structure

The headline findings are based on the VIIRS Phase A (failure case).

## Detailed results

### A1: Spatial autocorrelation
- Moran's I = 0.186 ± 0.214 — moderate clustering (not "extremely strong")
- **Fire-conditioned neighbor probability**: P(neighbor fire | current cell has fire) ≈ 0.61
  at short distances, vs random fire rate ≈ 0.29
- Distance-decay: drops to ~0.45 at 170-190 km
- → Spatial dependence is real (not just label agreement on no-fire cells)

### A2: Prevalence-controlled N→S evaluation (VIIRS only)
- Original South VIIRS AUROC: 0.582
- Subsample negatives (M1): 0.582 ± 0.0000
- Case-control 1:1 (M2): 0.582 ± 0.0017
- **Verdict**: Class-prevalence mismatch is NOT the cause of N→S collapse.
  The model genuinely fails to generalize, regardless of label balance.

### A3: Spatial proxy ablation

**MODIS A3** (working case for reference):
| Variant | AUROC | Δ vs v1 |
|---------|-------|---------|
| v1: weather only | 0.892 | baseline |
| v2: weather + lat/lon | 0.893 | +0.001 |
| v3: weather + lat_band 10 | 0.854 | -0.038 |

**VIIRS A3** (failure case — KEY FINDING):
| Variant | AUROC | Δ vs v1 |
|---------|-------|---------|
| v1: weather only | **0.582** | baseline |
| v2: weather + lat/lon | **0.811** | **+0.229** |
| v3: weather + lat_band 10 | 0.560 | -0.022 |
| v4: weather + lat_band5 + lon_band5 | 0.548 | -0.034 |

**Interpretation**: With VIIRS labels, weather-only model fails badly (0.582).
Adding raw lat/lon recovers +0.23 AUROC, but this is **geography memorization**,
not physics generalization. Coarse lat_band HURTS because train data only has
North lat_band values → South is out-of-distribution for the binned feature.

This is the **cautionary methodological finding**: published wildfire ML papers
that include lat/lon as features may have inflated AUROC due to geography
memorization. We explicitly drop lat/lon (audit fix #6) for clean evaluation.

## Hypothesis tests (H1-H5)

| Hypothesis | Status | Result |
|-----------|--------|--------|
| **H1: Covariate shift** | ✓ Confirmed | 5/9 features significant (KS p<0.001, Cohen's d>0.2). South is warmer (ΔT=+1.55°C), drier (ΔRH=-2%), has longer dry spells (Δdays=+2). |
| **H2: Concept shift** | ✓ Confirmed | Avg correlation P(fire\|feature) N vs S = 0.81 (not 1.0). Same weather → different fire probability in N vs S. |
| **H3: Label bias (regional)** | ✗ No strong effect | North: MODIS/VIIRS ratio = 0.95, Jaccard = 0.075. South: 0.84, 0.135. Δ = 0.11. |
| **H4: Unobserved spatial structure** | ⚠ Strongly implicated, NOT causally identified | Adding raw coordinates to VIIRS-trained model recovers +0.23 AUROC — substantial spatial structure exists that weather variables cannot represent. **Candidate mechanisms**: land cover, vegetation/NDVI, elevation, agricultural/human activity. **NOT directly tested** because independent spatial datasets were unavailable in the development environment. |
| **H5: IID spatial leakage** | ✓ Confirmed | Match rate at dist=0 = 89.8%. IID is biased easy due to spatial autocorrelation. |

## Overall verdict — 4 contributing factors

1. **Covariate shift** — South warmer/drier/longer dry spells (H1)
2. **Concept shift** — P(fire|weather) correlation N vs S = 0.81 (H2)
3. **Unobserved spatial structure** — strongly implicated, not causally identified (H4)
4. **IID spatial leakage** — IID AUROC was inflated by autocorrelation (H5)

NOT a primary cause:
- Class-prevalence mismatch (A2 ruled this out)
- Regional MODIS/VIIRS detection bias (H3 — small effect)

## Phase B (DEFERRED — environmental restrictions)

Phase B would directly test the candidate mechanisms for H4 (unobserved spatial
structure) by downloading external spatial datasets:
1. NDVI (MODIS modis-13A2) — vegetation index, captures fuel load
2. Land cover (ESA WorldCover or IO LULC) — agricultural vs forest vs urban
3. Elevation (SRTM DEM or Copernicus DEM) — topography
4. Agricultural activity proxy (night lights)

Currently blocked by environment network restrictions:
- Planetary Computer STAC: network unreachable from Colab
- Open-Elevation API: network unreachable
- Google Earth Engine: requires Cloud project setup (5-10 min manual)

Phase A findings alone are sufficient for a strong paper. Phase B remains a
future-work direction.

## Files

- `phaseA_diagnosis.json` — MODIS results (working case, for comparison)
- `phaseA_viirs_diagnosis.json` — VIIRS results (FAILURE case — the actual diagnosis)
- `spatial_diagnosis.json` — H1-H5 hypothesis tests
- `figures/fig_phaseA1_spatial_autocorr.png` — 3-panel: Moran's I, distance decay, conditional neighbor prob
- `figures/fig_phaseA2_prevalence_balanced.png` — MODIS prevalence-controlled bootstrap (working case)
- `figures/fig_phaseA3_spatial_proxy_ablation.png` — MODIS ablation (working case)
- `figures/fig_phaseA_viirs_combined.png` — VIIRS combined A2+A3 (THE killer figure)
- `figures/fig_diag_h2_concept_shift.png` — 9-panel P(fire|feature) N vs S
- `figures/fig_diag_h4_lat_band.png` — Fire rate by latitude band
- `figures/fig_diag_h5_spatial_autocorr.png` — Spatial autocorrelation curve
""")

print("  ✓ Updated README.md (main)")
print("  ✓ Updated phaseA_diagnosis/README.md")

# ============================================================
# FIX 3: Build Reliability-Aware Forecasting System mockup
# ============================================================
print("\n[FIX 3] Building reliability dashboard mockup...")
product_dir = ROOT / "product"
product_dir.mkdir(exist_ok=True)

(product_dir / "reliability_dashboard.html").write_text("""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Hệ thống Dự báo Cháy rừng Có ý thức Độ tin cậy</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #eee;
    margin: 0;
    padding: 24px;
    min-height: 100vh;
  }
  .dashboard {
    max-width: 720px;
    margin: 0 auto;
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    padding-bottom: 18px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }
  .title { font-size: 22px; font-weight: 600; }
  .subtitle { color: #999; font-size: 13px; }
  .region-select {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 14px;
  }
  .forecast-card {
    background: rgba(255,255,255,0.04);
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 18px;
  }
  .forecast-label { color: #aaa; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .risk-bar {
    background: linear-gradient(90deg, #fbbf24 0%, #ef4444 72%, #1f2937 72%, #1f2937 100%);
    height: 24px;
    border-radius: 12px;
    margin: 8px 0;
    position: relative;
  }
  .risk-text {
    font-size: 36px;
    font-weight: 700;
    color: #ef4444;
  }
  .reliability-card {
    background: rgba(254, 243, 199, 0.1);
    border-left: 4px solid #fbbf24;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 18px;
  }
  .reliability-low {
    color: #fbbf24;
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 6px;
  }
  .reliability-reason {
    color: #ddd;
    font-size: 13px;
    margin: 4px 0;
    padding-left: 16px;
    position: relative;
  }
  .reliability-reason::before {
    content: "•";
    position: absolute;
    left: 4px;
    color: #fbbf24;
  }
  .horizon-tabs {
    display: flex;
    gap: 8px;
    margin-top: 16px;
  }
  .tab {
    background: rgba(255,255,255,0.08);
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
  }
  .tab.active { background: #3b82f6; }
  .footer {
    color: #666;
    font-size: 11px;
    margin-top: 18px;
    text-align: center;
  }
  .model-info {
    font-size: 11px;
    color: #888;
    margin-top: 8px;
  }
</style>
</head>
<body>
<div class="dashboard">
  <div class="header">
    <div>
      <div class="title">Hệ thống Dự báo Cháy rừng Việt Nam</div>
      <div class="subtitle">Reliability-Aware Wildfire Forecasting System</div>
    </div>
    <select class="region-select">
      <option>Miền Bắc (lat ≥ 15°N)</option>
      <option selected>Miền Nam (lat &lt; 13°N) ⚠</option>
      <option>Miền Trung (13-15°N)</option>
    </select>
  </div>

  <div class="forecast-card">
    <div class="forecast-label">NGUY CƠ CHÁY • t+1 ngày</div>
    <div class="risk-text">72%</div>
    <div class="risk-bar"></div>
    <div class="model-info">Model: XGBoost + Isotonic • Horizon: t+1 (next day) • Trained on NASA MODIS 2018-2021</div>
  </div>

  <div class="reliability-card">
    <div class="reliability-low">⚠ ĐỘ TIN CẬY: THẤP</div>
    <div style="color:#aaa; font-size:12px; margin-bottom:8px;">Phát hiện distribution shift — dự báo nên được sử dụng thận trọng</div>
    <div class="reliability-reason">Vùng dự báo nằm ngoài phạm vi huấn luyện (spatial OOD)</div>
    <div class="reliability-reason">Phân phối thời tiết đã thay đổi (covariate shift)</div>
    <div class="reliability-reason">Quan hệ fire-weather khác miền Bắc (concept shift)</div>
    <div class="reliability-reason">Cấu trúc không gian chưa quan sát (NDVI/land cover/elevation chưa có)</div>
  </div>

  <div>
    <div class="forecast-label">Khoảng dự báo</div>
    <div class="horizon-tabs">
      <div class="tab active">t+1 (next day)</div>
      <div class="tab">t+3 (3 ngày)</div>
      <div class="tab">t+7 (1 tuần)</div>
    </div>
  </div>

  <div class="footer">
    v8.6 • Hoàng Bảo Lâm • Dựa trên 200 experimental runs + Phase A spatial diagnostic<br>
    Khi phát hiện distribution shift, hệ thống giảm độ tin cậy thay vì dự báo overconfident
  </div>
</div>
</body>
</html>
""")

(product_dir / "README_product.md").write_text("""# Reliability-Aware Wildfire Forecasting System

## Concept

Most wildfire risk models produce a single probability number, hiding model
uncertainty from decision-makers. We propose a **reliability-aware** forecasting
product that explicitly communicates **when the model should not be trusted**.

## How it works

The product wraps the trained model (XGBoost + Isotonic calibration) with a
**reliability layer** that detects distribution shift:

1. **Spatial OOD detection**: if the forecast region is outside the training
   geographic range, lower reliability.
2. **Covariate shift detection**: if the weather distribution at forecast time
   differs from training distribution (e.g., higher T, lower RH), lower reliability.
3. **Concept shift detection**: if P(fire|weather) at the forecast region differs
   from training (e.g., South has different fire-weather relationship), lower reliability.
4. **Unobserved spatial structure warning**: if the model knows there is
   spatial structure it cannot capture (Phase A3 finding: raw lat/lon recovers
   +0.23 AUROC), it warns that weather-only predictions may be overconfident.

## Why this design

This turns the Phase A findings (which show weather-only models fail on spatial
OOD) from a "weakness" into a "feature":
- The model **knows when not to trust itself**
- Decision-makers can act on the forecast with appropriate skepticism
- The product is **honest about its limitations** rather than hiding them

## Demo

See `reliability_dashboard.html` — a static mockup showing the UI for a
Southern Vietnam forecast where the model has detected:
- Spatial OOD (South region not in training data)
- Covariate shift (warmer+drier than North)
- Concept shift (different fire-weather relationship)
- Unobserved spatial structure

The reliability indicator shows "LOW" with explicit reasons, instead of just
displaying the 72% fire risk number alone.

## Implementation notes

The mockup is a static HTML file showing the design concept. A production
implementation would:

1. Wrap the trained model (pickle/joblib serialized)
2. Compute reliability indicators based on:
   - Distance from training bounding box
   - Weather feature distribution shift (KS test or Wasserstein distance)
   - P(fire|weather) mismatch (compare to training distribution)
3. Display the forecast + reliability + reasons to the user

This is a design proposal for the national competition. The actual model is
already trained (200 runs in `v8.5_canonical/runs/`). The product layer is a
thin wrapper that adds reliability detection on top of the existing model.

## For the judges

This product demonstrates that:
1. The student understands that **single-number predictions can be misleading**
2. The Phase A findings (spatial OOD failure) have been turned into a **product
   feature**, not just a research finding
3. The model is **honest about its limitations** — when it doesn't know, it says so
4. This is more useful for real-world deployment than a black-box predictor that
   produces overconfident probabilities on out-of-distribution inputs

The reliability-aware design is the **practical impact** of the audit-driven
methodology: by understanding where the model fails, we can build a system
that fails gracefully.
""")

print("  ✓ product/reliability_dashboard.html (UI mockup)")
print("  ✓ product/README_product.md (product concept)")

# ============================================================
# Rebuild ZIP
# ============================================================
print(f"\n[Final] Building v8.6 FINAL ZIP...")
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

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

print(f"\n{'='*70}")
print(f"✓ v8.6 FINAL ZIP COMPLETE")
print(f"{'='*70}")
print(f"  Path:  {ZIP_PATH}")
print(f"  Size:  {ZIP_PATH.stat().st_size/1e6:.1f} MB")
print(f"  Files: {file_count}")

# Verify
print("\nVerify new product/ folder in ZIP:")
with zipfile.ZipFile(ZIP_PATH) as zf:
    product_files = [n for n in zf.namelist() if "product/" in n]
    for f in product_files:
        print(f"  - {f}")
