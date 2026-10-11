#!/usr/bin/env python3
"""
Đánh giá nghiêm ngặt: 30 ngày ngẫu nhiên (stratified), ngưỡng định trước,
không selection bias.
"""
import json, math, pickle, warnings, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, f1_score
warnings.filterwarnings('ignore')

OUT = '/home/z/my-project/download/real_data_v85'
MODEL_DIR = '/home/z/my-project/download/model_v86'

# Load model (v8.6final: temporal split, 38 features)
m = xgb.XGBClassifier(); m.load_model(MODEL_DIR + '/xgb_temporal_v86final.json')
with open(MODEL_DIR + '/iso_temporal_final.pkl', 'rb') as f: iso = pickle.load(f)
with open(MODEL_DIR + '/imp_temporal_final.pkl', 'rb') as f: imp = pickle.load(f)
with open(MODEL_DIR + '/meta_v86final.json') as f: meta = json.load(f)
features = meta['features']
print(f"Model: {meta['version']}, {len(features)} features")

# Load panel + compute fire history (SAME as training)
panel = pd.read_parquet(OUT + '/panel_v85.parquet')
panel = panel.sort_values(['lat','lon','date']).reset_index(drop=True)
train_mask = panel['year'].isin([2018,2019,2020,2021])
cfr = panel[train_mask].groupby(['lat','lon'])['fire'].mean().reset_index()
cfr.columns = ['lat','lon','cell_fire_rate_train']
panel = panel.merge(cfr, on=['lat','lon'], how='left')
panel['cell_fire_rate_train'] = panel['cell_fire_rate_train'].fillna(panel['fire'].mean())
panel['cell_fire_count_30d'] = panel.groupby(['lat','lon'])['fire'].transform(lambda s: s.rolling(30,min_periods=1).sum())
panel['is_dry'] = panel['date'].dt.month.apply(lambda m: 1 if m in [1,2,3,4,11,12] else 0)
dry_mask = train_mask & (panel['is_dry']==1)
cfr_dry = panel[dry_mask].groupby(['lat','lon'])['fire'].mean().reset_index()
cfr_dry.columns = ['lat','lon','cell_fire_rate_dry']
panel = panel.merge(cfr_dry, on=['lat','lon'], how='left')
panel['cell_fire_rate_dry'] = panel['cell_fire_rate_dry'].fillna(panel['cell_fire_rate_train'])

di_idx = features.index('precip_roll30_sum')
di_mean = meta['drought_mean']; di_std = meta['drought_std']

# ============================================================
# PRE-DEFINED THRESHOLDS (matching PCCCR 5-level system)
# Defined BEFORE running evaluation
# ============================================================
LEVELS = [
    (0.00, 0.05, "Rất thấp"),
    (0.05, 0.15, "Thấp"),
    (0.15, 0.35, "Trung bình"),
    (0.35, 0.65, "Cao"),
    (0.65, 1.01, "Rất cao"),
]
def prob_to_level(p):
    for lo, hi, name in LEVELS:
        if lo <= p < hi:
            return name
    return "Rất cao"

# ============================================================
# SELECT 30 RANDOM DAYS FROM 2023 (stratified: 15 dry + 15 wet)
# Fixed seed for reproducibility
# ============================================================
rng = np.random.RandomState(42)
dry_days = panel[(panel['year']==2023) & (panel['is_dry']==1)]['date'].dt.normalize().unique()
wet_days = panel[(panel['year']==2023) & (panel['is_dry']==0)]['date'].dt.normalize().unique()
selected_dry = rng.choice(dry_days, size=min(15, len(dry_days)), replace=False)
selected_wet = rng.choice(wet_days, size=min(15, len(wet_days)), replace=False)
selected_dates = sorted(np.concatenate([selected_dry, selected_wet]))
print(f"\nSelected {len(selected_dry)} dry + {len(selected_wet)} wet = {len(selected_dates)} days")
print(f"Date range: {pd.Timestamp(selected_dates[0]).date()} to {pd.Timestamp(selected_dates[-1]).date()}")

# ============================================================
# EVALUATE EACH DAY
# ============================================================
results = []
for d in selected_dates:
    day = panel[panel['date'] == pd.Timestamp(d)].copy()
    if len(day) == 0: continue
    X = day[features].values.astype(np.float32)
    X[:, di_idx] = np.nan_to_num((X[:, di_idx] - di_mean) / di_std, nan=0.0)
    X = imp.transform(X)
    p_raw = m.predict_proba(X)[:, 1]
    p_iso = iso.predict(p_raw)

    # Regional predictions
    north_mask = day['lat'].values >= 15
    south_mask = day['lat'].values < 13
    central_mask = (day['lat'].values >= 13) & (day['lat'].values < 15)

    north_p = p_iso[north_mask].mean() if north_mask.sum() > 0 else 0
    central_p = p_iso[central_mask].mean() if central_mask.sum() > 0 else 0
    south_p = p_iso[south_mask].mean() if south_mask.sum() > 0 else 0

    # Level distribution
    levels = [prob_to_level(p) for p in p_iso]
    n_very_high = sum(1 for l in levels if l == "Rất cao")
    n_high = sum(1 for l in levels if l == "Cao")
    n_medium = sum(1 for l in levels if l == "Trung bình")
    n_low = sum(1 for l in levels if l == "Thấp")
    n_very_low = sum(1 for l in levels if l == "Rất thấp")

    season = "dry" if pd.Timestamp(d).month in [1,2,3,4,11,12] else "wet"

    results.append({
        "date": str(pd.Timestamp(d).date()),
        "season": season,
        "weather_T": float(day['t2m'].mean()),
        "weather_RH": float(day['rh'].mean()),
        "weather_W": float(day['wind'].mean()),
        "weather_precip": float(day['precip'].mean()),
        "mean_pred": float(p_iso.mean()),
        "north_pred": float(north_p),
        "central_pred": float(central_p),
        "south_pred": float(south_p),
        "south_gt_north": bool(south_p > north_p),
        "central_highest": bool(central_p >= north_p and central_p >= south_p),
        "n_very_high": n_very_high,
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "n_very_low": n_very_low,
        "n_cells": len(p_iso),
    })

df_results = pd.DataFrame(results)

# ============================================================
# FULL TEST SET METRICS (2023)
# ============================================================
te = panel[panel['year']==2023].dropna(subset=['label_h1'])
X_te = te[features].values.astype(np.float32)
X_te[:, di_idx] = np.nan_to_num((X_te[:, di_idx] - di_mean) / di_std, nan=0.0)
X_te = imp.transform(X_te)
y_te = te['label_h1'].values.astype(np.float32)
p_te_raw = m.predict_proba(X_te)[:, 1]
p_te_iso = iso.predict(p_te_raw)

auroc = roc_auc_score(y_te, p_te_iso)
brier = brier_score_loss(y_te, p_te_iso)
auprc = average_precision_score(y_te, p_te_iso)

# ECE
bins = np.linspace(0, 1, 11); ece_val = 0; n = len(y_te)
for i in range(10):
    lo, hi = bins[i], bins[i+1]
    ib = ((p_te_iso >= lo) & (p_te_iso <= hi)) if i == 9 else ((p_te_iso >= lo) & (p_te_iso < hi))
    if ib.sum() == 0: continue
    ece_val += (ib.sum() / n) * abs(y_te[ib].mean() - p_te_iso[ib].mean())

# ============================================================
# PRINT RESULTS
# ============================================================
print(f"\n{'='*70}")
print("ĐÁNH GIÁ NGHIÊM NGẶT — 30 NGÀY NGẪU NHIÊN + METRICS ĐẦY ĐỦ")
print(f"{'='*70}")

print(f"\n--- METRICS TOÀN BỘ TẬP TEST 2023 ({len(te):,} mẫu) ---")
print(f"  AUROC:     {auroc:.4f}")
print(f"  AUPRC:     {auprc:.4f}")
print(f"  Brier:     {brier:.4f}")
print(f"  ECE (Iso): {ece_val:.4f}")
print(f"  Positive rate: {y_te.mean():.4f} ({int(y_te.sum())}/{len(y_te)})")

print(f"\n--- 30 NGÀY NGẪU NHIÊN (stratified 15 khô + 15 mưa) ---")
print(f"\n  Ngưỡng định trước (trước khi chạy):")
for lo, hi, name in LEVELS:
    print(f"    {name}: [{lo:.2f}, {hi:.2f})")

dry = df_results[df_results['season']=='dry']
wet = df_results[df_results['season']=='wet']

print(f"\n--- MÙA KHÔ ({len(dry)} ngày) ---")
print(f"  {'Ngày':<12} | {'T':>5} | {'RH':>4} | {'W':>5} | {'precip':>6} | {'mean':>6} | {'N':>6} | {'C':>6} | {'S':>6} | {'S>N':>4} | {'Cmax':>5}")
print(f"  {'-'*85}")
for _, r in dry.iterrows():
    print(f"  {r['date']:<12} | {r['weather_T']:5.1f} | {r['weather_RH']:4.0f} | {r['weather_W']:5.1f} | {r['weather_precip']:6.1f} | {r['mean_pred']:6.3f} | {r['north_pred']:6.3f} | {r['central_pred']:6.3f} | {r['south_pred']:6.3f} | {'✓' if r['south_gt_north'] else '✗':>4} | {'✓' if r['central_highest'] else '✗':>5}")

print(f"\n  Tổng kết mùa khô:")
print(f"    S>N (Nam>Bắc):     {dry['south_gt_north'].sum()}/{len(dry)} = {100*dry['south_gt_north'].mean():.0f}%")
print(f"    C=max (T Nguyên cao nhất): {dry['central_highest'].sum()}/{len(dry)} = {100*dry['central_highest'].mean():.0f}%")
print(f"    Mean nguy cơ:     {dry['mean_pred'].mean():.3f} ± {dry['mean_pred'].std():.3f}")
print(f"    Ô 'Rất cao':       {dry['n_very_high'].mean():.0f} ± {dry['n_very_high'].std():.0f} ô/ngày")
print(f"    Ô 'Cao':           {dry['n_high'].mean():.0f} ± {dry['n_high'].std():.0f} ô/ngày")
print(f"    Ô 'Rất thấp':     {dry['n_very_low'].mean():.0f} ± {dry['n_very_low'].std():.0f} ô/ngày")

print(f"\n--- MÙA MƯA ({len(wet)} ngày) ---")
print(f"  {'Ngày':<12} | {'T':>5} | {'RH':>4} | {'W':>5} | {'precip':>6} | {'mean':>6} | {'N':>6} | {'C':>6} | {'S':>6} | {'S>N':>4} | {'Cmax':>5}")
print(f"  {'-'*85}")
for _, r in wet.iterrows():
    print(f"  {r['date']:<12} | {r['weather_T']:5.1f} | {r['weather_RH']:4.0f} | {r['weather_W']:5.1f} | {r['weather_precip']:6.1f} | {r['mean_pred']:6.3f} | {r['north_pred']:6.3f} | {r['central_pred']:6.3f} | {r['south_pred']:6.3f} | {'✓' if r['south_gt_north'] else '✗':>4} | {'✓' if r['central_highest'] else '✗':>5}")

print(f"\n  Tổng kết mùa mưa:")
print(f"    S>N (Nam>Bắc):     {wet['south_gt_north'].sum()}/{len(wet)} = {100*wet['south_gt_north'].mean():.0f}%")
print(f"    C=max (T Nguyên cao nhất): {wet['central_highest'].sum()}/{len(wet)} = {100*wet['central_highest'].mean():.0f}%")
print(f"    Mean nguy cơ:     {wet['mean_pred'].mean():.6f} ± {wet['mean_pred'].std():.6f}")
print(f"    Ô 'Rất thấp':     {wet['n_very_low'].mean():.0f} ± {wet['n_very_low'].std():.0f} ô/ngày (trong 273 ô)")
print(f"    Ô 'Rất cao':       {wet['n_very_high'].mean():.0f} ± {wet['n_very_high'].std():.0f} ô/ngày")

print(f"\n--- SO SÁNH MÙA KHÔ vs MÙA MƯA ---")
print(f"  {'Metric':<30} | {'Mùa khô':<15} | {'Mùa mưa':<15}")
print(f"  {'-'*60}")
print(f"  {'Mean nguy cơ':<30} | {dry['mean_pred'].mean():<15.4f} | {wet['mean_pred'].mean():<15.6f}")
print(f"  {'S>Bắc':<30} | {100*dry['south_gt_north'].mean():<15.0f}% | {100*wet['south_gt_north'].mean():<15.0f}%")
print(f"  {'T Nguyên cao nhất':<30} | {100*dry['central_highest'].mean():<15.0f}% | {100*wet['central_highest'].mean():<15.0f}%")
print(f"  {'Ô cấp Rất cao':<30} | {dry['n_very_high'].mean():<15.0f} | {wet['n_very_high'].mean():<15.0f}")
print(f"  {'Ô cấp Rất thấp':<30} | {dry['n_very_low'].mean():<15.0f} | {wet['n_very_low'].mean():<15.0f}")

# Save results
out_json = {
    "evaluation": "strict_30_days_random_stratified",
    "seed": 42,
    "thresholds_predefined": [{"low": lo, "high": hi, "level": name} for lo, hi, name in LEVELS],
    "model": meta['version'],
    "full_test_metrics": {
        "n_test": len(te), "n_positive": int(y_te.sum()),
        "AUROC": float(auroc), "AUPRC": float(auprc),
        "Brier": float(brier), "ECE_iso": float(ece_val),
    },
    "dry_season": {
        "n_days": len(dry),
        "south_gt_north_pct": float(100*dry['south_gt_north'].mean()),
        "central_highest_pct": float(100*dry['central_highest'].mean()),
        "mean_pred": float(dry['mean_pred'].mean()),
        "mean_pred_std": float(dry['mean_pred'].std()),
        "n_very_high_mean": float(dry['n_very_high'].mean()),
        "n_very_low_mean": float(dry['n_very_low'].mean()),
    },
    "wet_season": {
        "n_days": len(wet),
        "south_gt_north_pct": float(100*wet['south_gt_north'].mean()),
        "central_highest_pct": float(100*wet['central_highest'].mean()),
        "mean_pred": float(wet['mean_pred'].mean()),
        "mean_pred_std": float(wet['mean_pred'].std()),
        "n_very_high_mean": float(wet['n_very_high'].mean()),
        "n_very_low_mean": float(wet['n_very_low'].mean()),
    },
    "daily_results": df_results.to_dict('records'),
}
with open(OUT + '/strict_evaluation_30days.json', 'w') as f:
    json.dump(out_json, f, indent=2, default=str)
print(f"\n✓ Results saved: {OUT}/strict_evaluation_30days.json")
