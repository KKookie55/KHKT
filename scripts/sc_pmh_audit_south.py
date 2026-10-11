#!/usr/bin/env python3
"""
SC-PMH Step 1: Southern Overprediction Audit
=============================================
Goal: determine if Southern Vietnam over-prediction is real (the South truly
burns more often) or artificial (model miscalibrated/over-confident/over-broad
in the South). Outputs JSON + figures to /home/z/my-project/download/sc_pmh/

Author: project notebook
"""
from __future__ import annotations
import json, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Paths
DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
SC.mkdir(exist_ok=True)
SA = SC / 'step1_audit'
SA.mkdir(exist_ok=True)
PANEL_P = DL / 'v871' / 'panel_v871.parquet'
MODEL_P = DL / 'v871' / 'model' / 'xgb_v871_temporal.json'
ISO_P = DL / 'v871' / 'model' / 'iso_v871.pkl'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
FEATURES = META['features']
print(f"[info] using {len(FEATURES)} features: {FEATURES[:5]} ... {FEATURES[-3:]}")

# --- 1. Load panel & split ---
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)

train = panel[panel.year <= 2021]
val   = panel[panel.year == 2022]
test  = panel[panel.year == 2023]
print(f"[split] train={len(train)} val={len(val)} test={len(test)}")

# --- 2. Load model + isotonic calibrator (trained on VAL by project v8.7.1) ---
model = gbm.Booster()
model.load_model(str(MODEL_P))
print(f"[model] loaded")

# --- 3. Predict on VAL + TEST ---
def predict(df):
    X = gbm.DMatrix(df[FEATURES])
    p_raw = model.predict(X)
    return p_raw

val_p_raw = predict(val)
test_p_raw = predict(test)

# Load isotonic calibrated on val-true, then re-evaluate on test
with open(ISO_P, 'rb') as f:
    iso = pickle.load(f)

# Isotonic transform requires raw probs; if iso is full IsotonicRegression object:
if isinstance(iso, IsotonicRegression):
    val_p_iso = iso.predict(val_p_raw)
    test_p_iso = iso.predict(test_p_raw)
else:
    # fallback if pickle is dict
    val_p_iso = iso['model'].predict(val_p_raw)
    test_p_iso = iso['model'].predict(test_p_raw)

val = val.assign(p_raw=val_p_raw, p_iso=val_p_iso)
test = test.assign(p_raw=test_p_raw, p_iso=test_p_iso)

# --- 4. Region split: North >=14, South <14 (consistent with SC-PMH plan) ---
def regionize(df):
    df = df.copy()
    df['region'] = np.where(df.lat >= 14, 'North', 'South')
    return df
val = regionize(val)
test = regionize(test)

# --- 5. Audit metrics ---
def metrics(p, y, region_name, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(p)
    y = y[mask]; p = p[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {'region': region_name, 'n': int(len(y)), 'n_pos': int(y.sum()), 'base_rate': float(y.mean() if len(y) else 0), 'AUROC': None, 'PR_AUC': None, 'Brier': None, 'ECE': None, 'warning_50pct': None, 'warning_top20_recall': None, 'p_mean': None, 'p_median': None, 'p_q90': None, 'p_q95': None, 'p_q99': None}
    out = {
        'region': region_name,
        'n': len(y),
        'n_pos': int(y.sum()),
        'base_rate': float(y.mean()),
        'p_mean': float(p.mean()),
        'p_median': float(np.median(p)),
        'p_q90': float(np.quantile(p, 0.9)),
        'p_q95': float(np.quantile(p, 0.95)),
        'p_q99': float(np.quantile(p, 0.99)),
        'AUROC': float(roc_auc_score(y, p)),
        'PR_AUC': float(average_precision_score(y, p)),
        'Brier': float(brier_score_loss(y, p)),
        'ECE': _ece(p, y, n_bins=n_bins),
        'warning_50pct': float((p > 0.5).mean()),
        'warning_20pct': float((p > np.quantile(p, 0.8)).mean()),
        'warning_top20_recall': float((p > np.quantile(p, 0.8))[y == 1].mean()) if y.sum() > 0 else None,
    }
    return out

def _ece(p, y, n_bins=10):
    """Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)

# raw model
LABEL_COL = 'label_modis_h1'  # predict fire next day — matches XGBoost training target
rows_raw = []
rows_iso = []
for region in ['North', 'South', 'All']:
    if region == 'All':
        sub_v = val
        sub_t = test
    else:
        sub_v = val[val.region == region]
        sub_t = test[test.region == region]
    y_v = sub_v[LABEL_COL].values
    y_t = sub_t[LABEL_COL].values
    rows_raw.append(metrics(sub_t.p_raw.values, y_t, f"{region} (raw, test)"))
    rows_iso.append(metrics(sub_t.p_iso.values, y_t, f"{region} (iso, test)"))
    rows_iso.append(metrics(sub_v.p_iso.values, y_v, f"{region} (iso, val)"))

# Also use fire_modis (same-day) for spatial pattern scatter — has no NaN
test_for_spatial = test.dropna(subset=['fire_modis', 'p_iso', 'p_raw']).copy()

# --- 6. Per-cell fire frequency (climatology) ---
# To check: does the model rank cells correctly?
print('\n[per-cell] ranking: in TEST, for each cell, compute avg p_iso and fire rate; rank-correlation')
test_cell = test.groupby(['lat', 'lon']).agg(
    p_avg=('p_iso', 'mean'),
    fire_rate=('fire_modis', 'mean'),
    n=('fire_modis', 'size'),
    region=('region', 'first')
).reset_index()
print(f"  cells total: {len(test_cell)}  north={ (test_cell.region=='North').sum() }  south={ (test_cell.region=='South').sum() }")

# Spearman per region
from scipy.stats import spearmanr
north_cells = test_cell[test_cell.region == 'North']
south_cells = test_cell[test_cell.region == 'South']
all_cells = test_cell
rho_n, p_n = spearmanr(north_cells.p_avg, north_cells.fire_rate)
rho_s, p_s = spearmanr(south_cells.p_avg, south_cells.fire_rate)
rho_all, p_all = spearmanr(all_cells.p_avg, all_cells.fire_rate)
print(f"  Spearman cell avg p vs fire rate: north={rho_n:.3f} (p={p_n:.3g}); south={rho_s:.3f} (p={p_s:.3g}); all={rho_all:.3f}")

# --- 7. Spatial precision: top-K% predicted cells capture what fraction of fires? ---
# Compare model vs random vs climatology
def topk_capture(df, p_col, k=0.2):
    """What fraction of fires captured when we flag top-k% cells by predicted p?"""
    df = df.sort_values(p_col, ascending=False)
    n = len(df)
    k_n = int(n * k)
    topk = df.head(k_n)
    return float(topk.fire_modis.sum() / max(df.fire_modis.sum(), 1))

# Per day, per region
def topk_per_day_capture(df, p_col, k=0.2, by_region=True):
    out = []
    for region, sub in df.groupby('region' if by_region else None):
        for d, dsub in sub.groupby('date'):
            cap = topk_capture(dsub, p_col, k)
            out.append({'region': region, 'date': d, 'capture': cap})
    return pd.DataFrame(out)

# Compute for North and South separately on TEST
cap_raw_n = topk_per_day_capture(test[test.region == 'North'].copy(), 'p_raw', 0.2)
cap_raw_s = topk_per_day_capture(test[test.region == 'South'].copy(), 'p_raw', 0.2)
cap_iso_n = topk_per_day_capture(test[test.region == 'North'].copy(), 'p_iso', 0.2)
cap_iso_s = topk_per_day_capture(test[test.region == 'South'].copy(), 'p_iso', 0.2)

# Also climatology baseline: each cell's fire frequency in TRAIN
train_cell_rate = train.groupby(['lat', 'lon']).fire_modis.mean().reset_index().rename(columns={'fire_modis': 'clim_rate'})
test_with_clim = test.merge(train_cell_rate, on=['lat', 'lon'], how='left').fillna({'clim_rate': train.fire_modis.mean()})
cap_clim_n = topk_per_day_capture(test_with_clim[test_with_clim.region == 'North'].copy(), 'clim_rate', 0.2)
cap_clim_s = topk_per_day_capture(test_with_clim[test_with_clim.region == 'South'].copy(), 'clim_rate', 0.2)

print("\n[spatial precision] top-20% cells per day capture rate (mean ± std over days):")
print(f"  North:  raw={cap_raw_n.capture.mean():.3f}±{cap_raw_n.capture.std():.3f}  iso={cap_iso_n.capture.mean():.3f}±{cap_iso_n.capture.std():.3f}  clim={cap_clim_n.capture.mean():.3f}±{cap_clim_n.capture.std():.3f}")
print(f"  South:  raw={cap_raw_s.capture.mean():.3f}±{cap_raw_s.capture.std():.3f}  iso={cap_iso_s.capture.mean():.3f}±{cap_iso_s.capture.std():.3f}  clim={cap_clim_s.capture.mean():.3f}±{cap_clim_s.capture.std():.3f}")

# --- 8. Calibration curves: 10-bin reliability for North vs South ---
def calibration_curve(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            confs.append(np.nan); accs.append(np.nan); counts.append(0)
            continue
        confs.append(p[mask].mean())
        accs.append(y[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(confs), np.array(accs), np.array(counts)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, region in zip(axes, ['North', 'South']):
    sub = test[test.region == region]
    y = sub.fire_modis.values
    for p_col, label, color in [('p_raw', 'Raw XGBoost', '#1f77b4'), ('p_iso', 'Isotonic-calibrated', '#2ca02c')]:
        p = sub[p_col].values
        confs, accs, counts = calibration_curve(p, y, n_bins=10)
        sizes = counts / counts.max() if counts.max() > 0 else counts
        ax.scatter(confs, accs, s=sizes * 200 + 30, alpha=0.7, label=label, color=color, edgecolor='k', linewidths=0.5)
        # also bin low/mid/high only with >= 30 obs
        mask = counts >= 30
        ax.plot(confs[mask], accs[mask], color=color, alpha=0.5, linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Empirical fire rate')
    ax.set_title(f'{region} (lat {"≥" if region=="North" else "<"} 14°) — n={len(sub):,} cells, fire rate={y.mean():.3f}')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
fig.suptitle('SC-PMH Step 1: Calibration — Test 2023', fontsize=13, fontweight='bold')
fig.savefig(SA / 'calibration_north_vs_south.png', dpi=150)
plt.close()

# --- 9. Probability distribution: North vs South ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for ax, p_col, title in zip(axes, ['p_raw', 'p_iso'], ['Raw XGBoost', 'Isotonic-calibrated']):
    for region, color in [('North', '#1f77b4'), ('South', '#d62728')]:
        sub = test[test.region == region]
        ax.hist(sub[p_col], bins=50, range=(0, 1), density=True, alpha=0.55, label=f'{region} (n={len(sub):,})', color=color)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} — Test 2023')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle('SC-PMH Step 1: Probability distribution by region', fontsize=13, fontweight='bold')
fig.savefig(SA / 'prob_dist_north_vs_south.png', dpi=150)
plt.close()

# --- 10. Mean predicted p_iso by cell (heatmap-style scatter) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
for ax, p_col, title in zip(axes, ['p_iso', 'fire_modis'], ['Mean predicted p_iso (calibrated)', 'Empirical fire rate (TEST 2023)']):
    cell = test.groupby(['lat', 'lon']).agg(p=(p_col, 'mean')).reset_index()
    sc = ax.scatter(cell.lon, cell.lat, c=cell.p, s=18, cmap='YlOrRd', vmin=0, vmax=max(0.5, cell.p.max()))
    ax.axhline(14, color='k', linestyle='--', alpha=0.6, label='14°N (N/S boundary)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend()
    plt.colorbar(sc, ax=ax, label='Probability / fire rate')
fig.suptitle('SC-PMH Step 1: Spatial pattern — calibrated probability vs empirical fire rate', fontsize=13, fontweight='bold')
fig.savefig(SA / 'spatial_pattern_test2023.png', dpi=150)
plt.close()

# --- 11. Warning ratio by region ---
# Threshold sweep
thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
warn = []
for thr in thresholds:
    for region in ['North', 'South']:
        sub = test[test.region == region]
        flag = (sub.p_iso > thr).mean()
        # empirical fire rate among flagged
        if flag > 0:
            prec = sub[sub.p_iso > thr].fire_modis.mean()
        else:
            prec = np.nan
        warn.append({'threshold': thr, 'region': region, 'warning_ratio': flag, 'precision': prec, 'fire_rate': float(sub.fire_modis.mean())})
warn_df = pd.DataFrame(warn)
print('\n[warning ratio] by threshold & region:')
print(warn_df.to_string(index=False))

# --- 12. Save results ---
results = {
    'step': 'audit_south',
    'split': 'train 2018-2021, val 2022, test 2023 (temporal)',
    'n_features': len(FEATURES),
    'metrics_test': rows_raw + rows_iso,
    'per_cell_spearman': {
        'north_rho': float(rho_n), 'north_p': float(p_n),
        'south_rho': float(rho_s), 'south_p': float(p_s),
        'all_rho': float(float(rho_all)), 'all_p': float(p_all),
    },
    'topk_capture_per_day_20pct': {
        'north': {'raw': float(cap_raw_n.capture.mean()), 'iso': float(cap_iso_n.capture.mean()), 'clim': float(cap_clim_n.capture.mean())},
        'south': {'raw': float(cap_raw_s.capture.mean()), 'iso': float(cap_iso_s.capture.mean()), 'clim': float(cap_clim_s.capture.mean())},
    },
    'warning_ratio': warn_df.to_dict(orient='records'),
    'diagnosis': None,  # filled below
}

# --- 13. Diagnosis ---
diag_parts = []
n_metrics = {r['region']: r for r in rows_iso if 'iso, test' in r['region']}
n_n = n_metrics['North (iso, test)']
n_s = n_metrics['South (iso, test)']

# (a) over-broad warning: does South have high warning ratio at fixed threshold?
wr_n_05 = next(w['warning_ratio'] for w in warn if w['threshold'] == 0.05 and w['region'] == 'North')
wr_s_05 = next(w['warning_ratio'] for w in warn if w['threshold'] == 0.05 and w['region'] == 'South')
wr_n_10 = next(w['warning_ratio'] for w in warn if w['threshold'] == 0.10 and w['region'] == 'North')
wr_s_10 = next(w['warning_ratio'] for w in warn if w['threshold'] == 0.10 and w['region'] == 'South')

if wr_s_10 > 2 * wr_n_10:
    diag_parts.append(f"OVER-BROAD WARNING: South warning ratio at p>0.10 = {wr_s_10:.3f} vs North {wr_n_10:.3f} (ratio {wr_s_10/max(wr_n_10,1e-6):.2f}x).")

# (b) over-confidence: South ECE > North ECE by 0.02
if n_s['ECE'] - n_n['ECE'] > 0.02:
    diag_parts.append(f"OVER-CONFIDENCE: South ECE={n_s['ECE']:.4f} > North ECE={n_n['ECE']:.4f} (Δ=+{n_s['ECE']-n_n['ECE']:.4f}).")
else:
    diag_parts.append(f"CALIBRATION SIMILAR: South ECE={n_s['ECE']:.4f} vs North ECE={n_n['ECE']:.4f} (Δ={n_s['ECE']-n_n['ECE']:+.4f}).")

# (c) ranking quality: Spearman rho
if rho_s < rho_n - 0.1:
    diag_parts.append(f"RANKING DEGRADED IN SOUTH: cell Spearman rho south={rho_s:.3f} vs north={rho_n:.3f}. Model ranks cells worse in South.")
else:
    diag_parts.append(f"RANKING OK: cell Spearman rho south={rho_s:.3f} vs north={rho_n:.3f}. Model ranks cells similarly.")

# (d) base rate is truly higher
diag_parts.append(f"BASE RATE: North fire rate={n_n['base_rate']:.4f}, South fire rate={n_s['base_rate']:.4f}. South naturally burns {n_s['base_rate']/max(n_n['base_rate'],1e-6):.2f}x more often than North.")

# (e) top-K capture
cap_diag = f"TOP-20% CAPTURE: North iso={cap_iso_n.capture.mean():.3f} clim={cap_clim_n.capture.mean():.3f}; South iso={cap_iso_s.capture.mean():.3f} clim={cap_clim_s.capture.mean():.3f}. "
if cap_iso_s.capture.mean() - cap_clim_s.capture.mean() < 0.05:
    cap_diag += "Model in South barely beats climatology → spatial ranking is weak."
elif cap_iso_s.capture.mean() - cap_clim_s.capture.mean() < 0.15:
    cap_diag += "Model in South modestly beats climatology."
else:
    cap_diag += "Model in South clearly beats climatology."
diag_parts.append(cap_diag)

results['diagnosis'] = diag_parts
print('\n=== DIAGNOSIS ===')
for p in diag_parts:
    print(' •', p)

with open(SA / 'audit_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n[done] saved → {SA / 'audit_results.json'}")
print(f"[figs] {SA}/calibration_north_vs_south.png, prob_dist_north_vs_south.png, spatial_pattern_test2023.png")
