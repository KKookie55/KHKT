#!/usr/bin/env python3
"""
Update WILDFIRE_PROJECT_FULL_AUDIT.zip with v8.5 artifacts.
Rebuilds the WILDFIRE_PROJECT directory, adds v8.5 files, re-zips.
"""
import os, shutil, json, zipfile, sys
from pathlib import Path

BASE = Path("/home/z/my-project")
ROOT = BASE / "download" / "WILDFIRE_PROJECT"
ZIP_PATH = BASE / "download" / "WILDFIRE_PROJECT_FULL_AUDIT.zip"

print("=" * 70)
print("UPDATING WILDFIRE_PROJECT ZIP WITH v8.5 ARTIFACTS")
print("=" * 70)

# Unzip existing WILDFIRE_PROJECT
print("\n[1/4] Extracting existing ZIP to update...")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(ROOT.parent)
print(f"  Extracted to {ROOT}")

# Add v8.5 directory to the project
print("\n[2/4] Adding v8.5 artifacts...")
v85_dir = ROOT / "v8.5_canonical"
v85_dir.mkdir(parents=True, exist_ok=True)

# Copy v8.5 panel + metadata + splits + counterfactual
src_v85 = BASE / "download" / "real_data_v85"
for fname in ["panel_v85_meta.json", "splits_v85_meta.json", "counterfactual_v85.json",
              "results_v85_summary.csv", "README_v85.md"]:
    src = src_v85 / fname
    if src.exists():
        shutil.copy2(src, v85_dir / fname)
        print(f"  Added: {fname}")

# Copy runs/ (175 JSON files)
runs_dst = v85_dir / "runs"
runs_dst.mkdir(exist_ok=True)
n_runs = 0
for f in (src_v85 / "runs").glob("*.json"):
    shutil.copy2(f, runs_dst / f.name)
    n_runs += 1
print(f"  Added: runs/ ({n_runs} JSON files)")

# Copy figures/
figs_dst = v85_dir / "figures"
figs_dst.mkdir(exist_ok=True)
n_figs = 0
for f in (src_v85 / "figures").glob("*.png"):
    shutil.copy2(f, figs_dst / f.name)
    n_figs += 1
print(f"  Added: figures/ ({n_figs} PNG files)")

# Copy scripts (the v8.5 pipeline scripts)
scripts_dst = v85_dir / "scripts"
scripts_dst.mkdir(exist_ok=True)
for fname in ["preprocess_v85.py", "splits_v85.py", "train_v85.py",
              "counterfactual_v85.py", "make_figures_v85.py",
              "make_horizon_figures_v85.py", "colab_download_viirs.py",
              "run_remaining_seeds.sh", "build_audit_zip.py",
              "make_audit_figures.py"]:
    src = BASE / "scripts" / fname
    if src.exists():
        shutil.copy2(src, scripts_dst / fname)
print(f"  Added: scripts/ ({len(list(scripts_dst.iterdir()))} files)")

# Add v8.5 panel parquet (50 MB — manageable)
print("\n  Adding panel_v85.parquet (50 MB)...")
shutil.copy2(src_v85 / "panel_v85.parquet", v85_dir / "panel_v85.parquet")
print(f"  Added: panel_v85.parquet")

# Add v8.5 splits npz + parquet
shutil.copy2(src_v85 / "splits_v85.npz", v85_dir / "splits_v85.npz")
shutil.copy2(src_v85 / "splits_v85.parquet", v85_dir / "splits_v85.parquet")
print(f"  Added: splits_v85.npz, splits_v85.parquet")

# Update root README to mention v8.5
print("\n[3/4] Updating root README with v8.5 pointer...")
root_readme = ROOT / "README.md"
content = root_readme.read_text()
v85_addendum = """

---

## v8.5 CANONICAL EXPERIMENT (post-audit rebuild)

The audit identified 7 methodology issues in v7. v8.5 fixes all 7 and
rebuilds the experiment from scratch:

- **WS-1**: forecast horizons t+1, t+3, t+7 (no more same-day prediction)
- **WS-2**: calendar-aware lag/rolling (true 7-day shifts on full daily grid)
- **WS-3**: leakage-safe preprocessing (train-only normalization, drop nb_*, drop lat/lon as features, no early stopping)
- **WS-4**: 5 splits — IID group-blocked + Temporal OOD + 3 Spatial OOD boundaries
- **WS-5**: 5 models × 4 calibrators × 5 splits × 5 seeds × 3 horizons (resumable)
- **WS-6**: counterfactual physics consistency test (per-feature monotonicity violation rate)

**Headline finding**: XGB+Isotonic is the most robust calibrator under spatial
distribution shift (ECE 0.009 IID → 0.028 spatial OOD, 3× degradation).
MLP degrades more (ECE 0.008 → 0.035, 4× degradation).

**Physics hypothesis result**: PhysMLP enforces monotonicity on the features
it penalizes (W, RH) but does NOT help on T or precip — a nuanced finding.

See `v8.5_canonical/README_v85.md` for the full canonical experiment writeup.
"""
if "v8.5 CANONICAL EXPERIMENT" not in content:
    root_readme.write_text(content + v85_addendum)
    print("  Root README updated with v8.5 pointer")

# Re-zip
print(f"\n[4/4] Re-zipping to {ZIP_PATH} ...")
ZIP_PATH.unlink(missing_ok=True)

# Count files + size
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
print(f"✓ ZIP UPDATED")
print(f"{'=' * 70}")
print(f"  Output:    {ZIP_PATH}")
print(f"  ZIP size:  {zip_size/1e6:.1f} MB")
print(f"  Files:     {file_count}")
print(f"  New:       + v8.5_canonical/ (panel, splits, runs/{n_runs}, figures/{n_figs}, scripts/, README_v85.md)")
