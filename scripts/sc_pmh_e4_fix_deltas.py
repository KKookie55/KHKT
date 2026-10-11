#!/usr/bin/env python3
"""Quick fix: recompute paired deltas for E4 (just read the CSV)."""
import pandas as pd
import numpy as np
from pathlib import Path

EX = Path('/home/z/my-project/download/sc_pmh/e4_multi_spatial_splits')
df = pd.read_csv(EX / 'e4_results.csv')

BOUNDARIES = [12.0, 13.0, 14.0, 15.0, 16.0]
SEEDS = [42, 123, 777, 2024, 31415]

print("=== Per boundary, uncalibrated, mean ± std (n=5 seeds) ===")
print(f"{'boundary':>10} {'baseline AUROC':>20} {'anom AUROC':>20} {'ΔAUROC':>20}")
print("-" * 75)
for b in BOUNDARIES:
    sub = df[(df.boundary == b) & (~df.calibrated)]
    bl = sub[sub.variant == 'cold_baseline']['AUROC'].values
    an = sub[sub.variant == 'cold_anom_comp']['AUROC'].values
    if len(bl) == 0 or len(an) == 0:
        continue
    print(f"{b:>10.1f} {bl.mean():>10.4f}±{bl.std():.4f} {an.mean():>10.4f}±{an.std():.4f} {(an-bl).mean():>+10.4f}±{(an-bl).std():.4f}")

print("\n=== Paired Δ (anom − baseline) per boundary, n=5 seeds, uncalibrated ===")
print(f"{'boundary':>10} {'ΔAUROC':>22} {'ΔPR_AUC':>22} {'ΔBrier':>22} {'ΔECE':>22}")
print("-" * 110)
for b in BOUNDARIES:
    sub = df[(df.boundary == b) & (~df.calibrated)]
    per_seed = {}
    for _, r in sub.iterrows():
        per_seed.setdefault(r['seed'], {})[r['variant']] = {k: r[k] for k in ['AUROC', 'PR_AUC', 'Brier', 'ECE']}
    if len(per_seed) < 2:
        continue
    line = f"{b:>10.1f} "
    for metric in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        diff = np.array([per_seed[s]['cold_anom_comp'][metric] - per_seed[s]['cold_baseline'][metric] for s in SEEDS if s in per_seed and 'cold_anom_comp' in per_seed[s] and 'cold_baseline' in per_seed[s]])
        if len(diff) == 0:
            line += f"{'N/A':>22} "
            continue
        mean, std = float(diff.mean()), float(diff.std())
        n = len(diff)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else float('inf')
        sig = " *" if abs(tstat) > 2.776 else ""
        line += f" {mean:+.5f}±{std:.5f}(t={tstat:.2f}{sig})"
    print(line)

# Also save a cleaner summary CSV
agg = df.groupby(['boundary', 'variant', 'calibrated']).agg(
    AUROC_mean=('AUROC', 'mean'), AUROC_std=('AUROC', 'std'),
    PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
    Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
    ECE_mean=('ECE', 'mean'), ECE_std=('ECE', 'std'),
).reset_index()
agg.to_csv(EX / 'e4_summary_clean.csv', index=False)
