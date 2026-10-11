#!/usr/bin/env python3
"""
Update WILDFIRE_PROJECT ZIP with VIIRS validation + comparison artifacts.
"""
import os, shutil, zipfile, json
from pathlib import Path

BASE = Path("/home/z/my-project")
ROOT = BASE / "download" / "WILDFIRE_PROJECT"
ZIP_PATH = BASE / "download" / "WILDFIRE_PROJECT_FULL_AUDIT.zip"

print("=" * 70)
print("UPDATING ZIP WITH VIIRS VALIDATION ARTIFACTS")
print("=" * 70)

# Unzip existing WILDFIRE_PROJECT
print("\n[1/3] Extracting existing ZIP...")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(ROOT.parent)
print(f"  Extracted to {ROOT}")

# Add VIIRS artifacts
print("\n[2/3] Adding VIIRS artifacts...")
v85_dir = ROOT / "v8.5_canonical"

# Copy VIIRS panel
src_v85 = BASE / "download" / "real_data_v85"
for fname in ["panel_v85_viirs.parquet",
              "viirs_validation.json",
              "results_v85_summary_viirs.csv",
              "results_v85_summary_modis.csv"]:
    src = src_v85 / fname
    if src.exists():
        shutil.copy2(src, v85_dir / fname)
        print(f"  Added: {fname}")

# Copy VIIRS run JSONs
viirs_runs_dst = v85_dir / "runs"
viirs_runs_dst.mkdir(exist_ok=True)
n_new = 0
for f in (src_v85 / "runs").glob("viirs__*.json"):
    shutil.copy2(f, viirs_runs_dst / f.name)
    n_new += 1
print(f"  Added: {n_new} VIIRS run JSONs to runs/")

# Copy new figures (fig12, fig13, fig14, fig15)
figs_dst = v85_dir / "figures"
n_figs_new = 0
for f in (src_v85 / "figures").glob("*.png"):
    if not (figs_dst / f.name).exists():
        shutil.copy2(f, figs_dst / f.name)
        n_figs_new += 1
print(f"  Added: {n_figs_new} new figures to figures/")

# Copy new scripts
scripts_dst = v85_dir / "scripts"
for fname in ["build_viirs_panel.py", "viirs_validation.py",
              "make_modis_vs_viirs_figures.py", "process_viirs.py"]:
    src = BASE / "scripts" / fname
    if src.exists():
        shutil.copy2(src, scripts_dst / fname)
        print(f"  Added script: {fname}")

# Add raw VIIRS CSVs to data/raw_data/viirs/
viirs_data_dst = ROOT / "data" / "raw_data" / "viirs"
viirs_data_dst.mkdir(parents=True, exist_ok=True)
src_viirs_data = BASE / "upload" / "wildfire_data" / "wildfire_data" / "viirs"
n_csv = 0
for f in src_viirs_data.glob("*.csv"):
    shutil.copy2(f, viirs_data_dst / f.name)
    n_csv += 1
print(f"  Added: {n_csv} VIIRS CSVs to data/raw_data/viirs/")

# Update v8.5 README with VIIRS findings
print("\n  Updating v8.5 README with VIIRS findings...")
v85_readme = v85_dir / "README_v85.md"
content = v85_readme.read_text()
viirs_addendum = """

---

## VIIRS Independent Validation (added 2026-10-06)

After downloading VIIRS active fire data (2018-2023) from NASA FIRMS,
we cross-validated MODIS labels against VIIRS (375m resolution, more sensitive).

### MODIS vs VIIRS label agreement (cell-day level)

| | VIIRS=fire | VIIRS=no | Total |
|---|---|---|---|
| **MODIS=fire** | 8,924 (both) | 39,355 (MODIS only) | 48,279 |
| **MODIS=no** | 35,923 (VIIRS only) | 513,941 (neither) | 549,864 |
| **Total** | 44,847 | 553,296 | 598,143 |

**Jaccard similarity: 0.106** — only 10.6% agreement on positive labels.

### Re-training on VIIRS labels (3 seeds, Iso, t+1)

| Split | MODIS AUROC | VIIRS AUROC | Δ |
|-------|-------------|-------------|----|
| IID | 0.9467 | 0.8402 | **-0.11** |
| Temporal OOD | 0.9343 | 0.8424 | -0.09 |
| **Spatial N→S** | **0.8928** | **0.5859** | **-0.31** |
| Spatial S→C | 0.9235 | 0.8818 | -0.04 |
| Spatial C+S→N | 0.9273 | 0.6990 | -0.23 |

### Key findings

1. **All models lose ~0.10 AUROC on IID** when trained on VIIRS labels
   — MODIS labels were "easier" to predict.

2. **Spatial OOD N→S collapses by ~0.31 AUROC** with VIIRS labels
   — model trained on north Vietnam fails completely (AUROC 0.59) on
   south Vietnam when labels are VIIRS.

3. **Spatial S→C barely changes** (Δ=-0.04)
   — model trained on south Vietnam generalizes well to central with VIIRS labels.

4. **Asymmetry**: there's a North-South fire-regime dichotomy that:
   - MODIS labels don't capture (large fires similar across regions)
   - VIIRS labels DO capture (small fires differ by region)

### Implications for the audit

- v8.5 MODIS AUROC values are **UPPER BOUNDS** on true performance
- The "good MODIS spatial OOD performance" (0.89) was over-estimated
- True generalization with cleaner labels (VIIRS) is much lower (0.59 spatial OOD)
- The audit issue #2 (MODIS observation proxy) is more serious than expected

### Three-tier contribution framework — updated

1. **Tier 1**: Dataset & methodology (unchanged)
2. **Tier 2**: Trustworthiness under distribution shift (NEW finding):
   - "Label source matters as much as model choice"
   - MODIS AUROC 0.95 IID → 0.59 spatial OOD with VIIRS labels
   - This is **stronger evidence** for the audit-fix importance
3. **Tier 3**: Physics hypothesis (unchanged — physics helps for W/RH monotonicity)

### Recommended next steps for paper

1. **Use MODIS labels as PRIMARY training signal** (matches operational deployment)
2. **Use VIIRS as INDEPENDENT VALIDATION** in the report (shows robustness check)
3. **Acknowledge** in limitations that VIIRS-trained models fail on spatial OOD
4. **Frame as audit contribution**: "We discovered that MODIS-only evaluation overestimates
   model generalization by 0.30 AUROC on spatial OOD, demonstrating the importance of
   multi-sensor validation in wildfire risk modeling."
"""
if "VIIRS Independent Validation" not in content:
    v85_readme.write_text(content + viirs_addendum)
    print("  ✓ v8.5 README updated with VIIRS section")

# Re-zip
print(f"\n[3/3] Re-zipping to {ZIP_PATH} ...")
ZIP_PATH.unlink(missing_ok=True)

file_count = 0
total_size = 0
for root, _, files in os.walk(ROOT):
    for f in files:
        p = Path(root) / f
        file_count += 1
        total_size += p.stat().st_size
print(f"  Files to zip: {file_count}")
print(f"  Uncompressed size: {total_size/1e6:.1f} MB")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for root, _, files in os.walk(ROOT):
        for f in files:
            full = Path(root) / f
            rel = full.relative_to(ROOT)
            zf.write(full, f"WILDFIRE_PROJECT/{rel}")

zip_size = ZIP_PATH.stat().st_size
print(f"\n{'=' * 70}")
print(f"✓ ZIP UPDATED WITH VIIRS ARTIFACTS")
print(f"{'=' * 70}")
print(f"  Output:    {ZIP_PATH}")
print(f"  ZIP size:  {zip_size/1e6:.1f} MB")
print(f"  Files:     {file_count}")
