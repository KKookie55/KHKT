#!/usr/bin/env python3
"""
SC-PMH Experiment H1: Southern Overprediction Audit — full 6-model comparison
============================================================================
Setup:
  - Spatial OOD split: train North (lat ≥ 14°) 2018-2021
                       val   North (lat ≥ 14°) 2022
                       test  South (lat < 14°)  2023
  - 6 model variants:
      M1: XGBoost baseline (current 38 features)
      M2: M1 + isotonic calibration
      M3: FWI standalone (z-scored) — physical-only classifier
      M4: XGBoost + FWI features (38 + 6 = 44 features)
      M5: M4 + isotonic calibration
      M6: Logistic combiner — σ(β0 + β1·logit(p_M1) + β2·z_FWI),
          trained on North val
      M6b: same as M6 with β2 constrained ≥ 0
  - Eval metrics: AUROC, PR-AUC, Brier, ECE, spatial warning ratio,
                  top-K% capture per day, calibration curve
  - Outputs: /home/z/my-project/download/sc_pmh/exp_h1/
"""
from __future__ import annotations
import json, os, sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
EX = SC / 'exp_h1'
EX.mkdir(parents=True, exist_ok=True)
PANEL_P = SC / 'panel_with_fwi.parquet'
META_P = DL / 'v871' / 'model' / 'meta_v871.json'

with open(META_P) as f:
    META = json.load(f)
BASE_FEATS = META['features']  # 38 features from v8.7.1 XGBoost
FWI_FEATS = ['fwi', 'fwi_ffmc', 'fwi_dmc', 'fwi_dc', 'fwi_isi', 'fwi_bui']
LABEL_COL = 'label_modis_h1'  # predict fire next day
SEED = 42

print("[1/8] loading panel + FWI...")
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
panel['year'] = panel['date'].dt.year
panel = panel.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
panel['region'] = np.where(panel.lat >= 14, 'North', 'South')
print(f"  rows={len(panel)} cells={panel[['lat','lon']].drop_duplicates().shape[0]}")

# --- spatial OOD split ---
print("[2/8] spatial OOD split (train North 2018-2021, val North 2022, test South 2023)")
train = panel[(panel.region == 'North') & (panel.year <= 2021)].copy()
val   = panel[(panel.region == 'North') & (panel.year == 2022)].copy()
test  = panel[(panel.region == 'South') & (panel.year == 2023)].copy()
# Also North 2023 for in-distribution comparison
test_north_2023 = panel[(panel.region == 'North') & (panel.year == 2023)].copy()
print(f"  train (N 2018-21): {len(train):,} pos_rate={train[LABEL_COL].mean():.4f}")
print(f"  val   (N 2022):     {len(val):,}   pos_rate={val[LABEL_COL].mean():.4f}")
print(f"  test  (S 2023):     {len(test):,}  pos_rate={test[LABEL_COL].mean():.4f}")
print(f"  test_id (N 2023):   {len(test_north_2023):,} pos_rate={test_north_2023[LABEL_COL].mean():.4f}")

# Drop rows with NaN label
for df in [train, val, test, test_north_2023]:
    df.dropna(subset=[LABEL_COL], inplace=True)
print(f"  after NaN-drop: train={len(train)} val={len(val)} test={len(test)} test_id={len(test_north_2023)}")

# --- M1: XGBoost baseline ---
print("\n[3/8] M1: XGBoost baseline (38 features, North train)")
y_train = train[LABEL_COL].astype(int).values
y_val = val[LABEL_COL].astype(int).values
y_test = test[LABEL_COL].astype(int).values
y_test_id = test_north_2023[LABEL_COL].astype(int).values

dtrain = gbm.DMatrix(train[BASE_FEATS], label=y_train)
dval   = gbm.DMatrix(val[BASE_FEATS], label=y_val)
dtest  = gbm.DMatrix(test[BASE_FEATS], label=y_test)
dtest_id = gbm.DMatrix(test_north_2023[BASE_FEATS], label=y_test_id)

# Same hyperparams as v8.7.1 (which used defaults? — we use reasonable standard values)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_lambda': 1.0,
    'seed': SEED,
    'verbosity': 0,
}
t0 = time.time()
m1 = gbm.train(params, dtrain, num_boost_round=500, evals=[(dtrain, 'train'), (dval, 'val')],
               early_stopping_rounds=30, verbose_eval=False)
print(f"  trained in {time.time()-t0:.1f}s, best_iter={m1.best_iteration}")

p_m1_val = m1.predict(dval)
p_m1_test = m1.predict(dtest)
p_m1_test_id = m1.predict(dtest_id)

# --- M2: M1 + isotonic calibration ---
print("\n[4/8] M2: M1 + isotonic (fit on North val, eval on South test)")
iso_m1 = IsotonicRegression(out_of_bounds='clip')
iso_m1.fit(p_m1_val, y_val)
p_m2_val = iso_m1.predict(p_m1_val)
p_m2_test = iso_m1.predict(p_m1_test)
p_m2_test_id = iso_m1.predict(p_m1_test_id)

# --- M3: FWI standalone ---
# We use a logistic regression on standardized FWI features (no ML)
print("\n[5/8] M3: FWI standalone (logistic on 6 FWI features, standardized with North train stats)")
scaler_fwi = StandardScaler().fit(train[FWI_FEATS])
X_fwi_train = scaler_fwi.transform(train[FWI_FEATS])
X_fwi_val = scaler_fwi.transform(val[FWI_FEATS])
X_fwi_test = scaler_fwi.transform(test[FWI_FEATS])
X_fwi_test_id = scaler_fwi.transform(test_north_2023[FWI_FEATS])
m3 = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
m3.fit(X_fwi_train, y_train)
p_m3_val = m3.predict_proba(X_fwi_val)[:, 1]
p_m3_test = m3.predict_proba(X_fwi_test)[:, 1]
p_m3_test_id = m3.predict_proba(X_fwi_test_id)[:, 1]
print(f"  coefs: {dict(zip(FWI_FEATS, m3.coef_[0].round(3)))} intercept={m3.intercept_[0]:.3f}")

# --- M4: XGBoost + FWI features ---
print("\n[6/8] M4: XGBoost + 6 FWI features (44 features total)")
FEATS_M4 = BASE_FEATS + FWI_FEATS
dtrain4 = gbm.DMatrix(train[FEATS_M4], label=y_train)
dval4   = gbm.DMatrix(val[FEATS_M4], label=y_val)
dtest4  = gbm.DMatrix(test[FEATS_M4], label=y_test)
dtest_id4 = gbm.DMatrix(test_north_2023[FEATS_M4], label=y_test_id)
t0 = time.time()
m4 = gbm.train(params, dtrain4, num_boost_round=500, evals=[(dtrain4, 'train'), (dval4, 'val')],
               early_stopping_rounds=30, verbose_eval=False)
print(f"  trained in {time.time()-t0:.1f}s, best_iter={m4.best_iteration}")
p_m4_val = m4.predict(dval4)
p_m4_test = m4.predict(dtest4)
p_m4_test_id = m4.predict(dtest_id4)

# --- M5: M4 + isotonic ---
print("\n[7/8] M5: M4 + isotonic (fit on North val)")
iso_m4 = IsotonicRegression(out_of_bounds='clip')
iso_m4.fit(p_m4_val, y_val)
p_m5_val = iso_m4.predict(p_m4_val)
p_m5_test = iso_m4.predict(p_m4_test)
p_m5_test_id = iso_m4.predict(p_m4_test_id)

# --- M6: Logistic combiner ---
# σ(β0 + β1·logit(p_M1) + β2·z_FWI) — z_FWI is the standardized scalar FWI score
print("\n[8/8] M6: logistic combiner σ(β0 + β1·logit(p_M1) + β2·z_FWI)")
eps = 1e-6
z_ml_val = np.log(np.clip(p_m1_val, eps, 1 - eps) / np.clip(1 - p_m1_val, eps, 1))
z_ml_test = np.log(np.clip(p_m1_test, eps, 1 - eps) / np.clip(1 - p_m1_test, eps, 1))
z_ml_test_id = np.log(np.clip(p_m1_test_id, eps, 1 - eps) / np.clip(1 - p_m1_test_id, eps, 1))

# Use single FWI scalar — but normalize using North train stats
fwi_scalar_train = train['fwi'].values.reshape(-1, 1)
fwi_scalar_val = val['fwi'].values.reshape(-1, 1)
fwi_scalar_test = test['fwi'].values.reshape(-1, 1)
fwi_scalar_test_id = test_north_2023['fwi'].values.reshape(-1, 1)
fwi_scaler = StandardScaler().fit(fwi_scalar_train)
z_fwi_val = fwi_scaler.transform(fwi_scalar_val).ravel()
z_fwi_test = fwi_scaler.transform(fwi_scalar_test).ravel()
z_fwi_test_id = fwi_scaler.transform(fwi_scalar_test_id).ravel()

X_comb_val = np.column_stack([z_ml_val, z_fwi_val])
X_comb_test = np.column_stack([z_ml_test, z_fwi_test])
X_comb_test_id = np.column_stack([z_ml_test_id, z_fwi_test_id])

# Unconstrained logistic
m6 = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
m6.fit(X_comb_val, y_val)
p_m6_val = m6.predict_proba(X_comb_val)[:, 1]
p_m6_test = m6.predict_proba(X_comb_test)[:, 1]
p_m6_test_id = m6.predict_proba(X_comb_test_id)[:, 1]
print(f"  M6 coefs: logit_p_ml={m6.coef_[0,0]:.4f}  z_fwi={m6.coef_[0,1]:.4f}  intercept={m6.intercept_[0]:.4f}")

# Constrained β2 >= 0 — use a simple non-negative logistic (scipy optimize)
from scipy.optimize import minimize
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def neg_loglik(b, X, y):
    z = b[0] + b[1] * X[:, 0] + b[2] * X[:, 1]
    p = _sigmoid(z)
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# Constraint β2 >= 0
res = minimize(neg_loglik, x0=[0.0, 0.0, 0.5], args=(X_comb_val, y_val),
               method='SLSQP', bounds=[(None, None), (None, None), (0, None)],
               options={'maxiter': 500})
print(f"  M6b coefs (β2≥0): intercept={res.x[0]:.4f}  β1={res.x[1]:.4f}  β2={res.x[2]:.4f} (loss={res.fun:.4f})")
def predict_constrained(b, X):
    z = b[0] + b[1] * X[:, 0] + b[2] * X[:, 1]
    return 1 / (1 + np.exp(-z))
p_m6b_val = predict_constrained(res.x, X_comb_val)
p_m6b_test = predict_constrained(res.x, X_comb_test)
p_m6b_test_id = predict_constrained(res.x, X_comb_test_id)

# --- 9. Evaluate everything ---
def _ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean(); acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)

def metrics(p, y, name, region):
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
    return {
        'model': name, 'region': region, 'n': len(y), 'n_pos': int(y.sum()),
        'base_rate': float(y.mean()),
        'AUROC': float(roc_auc_score(y, p)),
        'PR_AUC': float(average_precision_score(y, p)),
        'Brier': float(brier_score_loss(y, p)),
        'ECE': _ece(p, y),
        'p_mean': float(p.mean()),
        'p_q95': float(np.quantile(p, 0.95)),
        'p_q99': float(np.quantile(p, 0.99)),
        'warn_p05': float((p > 0.05).mean()),
        'warn_p10': float((p > 0.10).mean()),
        'warn_p20': float((p > 0.20).mean()),
        'warn_p50': float((p > 0.50).mean()),
    }

results_rows = []
for name, p_v, p_t, p_id in [
    ('M1_XGB',          p_m1_val,  p_m1_test,  p_m1_test_id),
    ('M2_XGB_iso',      p_m2_val,  p_m2_test,  p_m2_test_id),
    ('M3_FWI',          p_m3_val,  p_m3_test,  p_m3_test_id),
    ('M4_XGB_FWI',      p_m4_val,  p_m4_test,  p_m4_test_id),
    ('M5_XGB_FWI_iso',  p_m5_val,  p_m5_test,  p_m5_test_id),
    ('M6_combiner',     p_m6_val,  p_m6_test,  p_m6_test_id),
    ('M6b_combiner_b2pos', p_m6b_val, p_m6b_test, p_m6b_test_id),
]:
    results_rows.append(metrics(p_v,  y_val,  name, 'val_north'))
    results_rows.append(metrics(p_t,  y_test, name, 'test_south'))
    results_rows.append(metrics(p_id, y_test_id, name, 'test_north_id'))

results_df = pd.DataFrame(results_rows)

# Add delta columns: south minus north_id for each metric
def add_delta(df):
    out = df.copy()
    piv = df.pivot(index='model', columns='region', values=['AUROC', 'PR_AUC', 'Brier', 'ECE'])
    for m in ['AUROC', 'PR_AUC', 'Brier', 'ECE']:
        out[f'{m}_delta_south_minus_northid'] = out.apply(
            lambda r: r[m] - piv.loc[r['model'], (m, 'test_north_id')] if r['region'] == 'test_south' else np.nan,
            axis=1
        )
    return out
results_df = add_delta(results_df)
print("\n=== Results (South test 2023) ===")
sub = results_df[results_df.region == 'test_south'][['model', 'AUROC', 'PR_AUC', 'Brier', 'ECE', 'warn_p10', 'warn_p20', 'warn_p50']]
sub = sub.sort_values('AUROC', ascending=False)
print(sub.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

print("\n=== Results (North test 2023 — ID baseline) ===")
sub_id = results_df[results_df.region == 'test_north_id'][['model', 'AUROC', 'PR_AUC', 'Brier', 'ECE', 'warn_p10', 'warn_p20', 'warn_p50']]
print(sub_id.sort_values('AUROC', ascending=False).to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# --- 10. Per-day top-K capture on South test ---
print("\n[eval] per-day top-K% capture on South test 2023")
test_pred_df = test.copy()
test_pred_df['p_m1'] = p_m1_test
test_pred_df['p_m2'] = p_m2_test
test_pred_df['p_m3'] = p_m3_test
test_pred_df['p_m4'] = p_m4_test
test_pred_df['p_m5'] = p_m5_test
test_pred_df['p_m6'] = p_m6_test
test_pred_df['p_m6b'] = p_m6b_test
# Also climatology baseline — train North fire rate per cell, applied to South cells (so cells unseen in train → use North mean)
north_cell_rate = train.groupby(['lat','lon']).agg(clim_rate=(LABEL_COL, 'mean')).reset_index()
north_mean_rate = y_train.mean()
# For South cells, clim_rate is unknown — use North mean as fallback (this is what an OOD model would do without fire history)
# A more sophisticated fallback: use the LATITUDE band mean from North
# For now: south cells use overall North fire rate
test_pred_df['clim_north_mean'] = north_mean_rate
# Latitude-band fallback (south cells use mean of North cells within 1° latitude)
north_by_lat = train.groupby('lat')[LABEL_COL].mean().to_dict()
def lat_band_clim(lat):
    # find closest North latitude (latitudes are at 0.5° steps)
    north_lats = [l for l in north_by_lat.keys() if l >= 14]
    if not north_lats:
        return north_mean_rate
    closest = min(north_lats, key=lambda l: abs(l - lat))
    return north_by_lat[closest]
test_pred_df['clim_lat_band'] = test_pred_df.lat.apply(lat_band_clim)

def topk_per_day(df, p_col, k=0.2):
    caps = []
    for d, sub in df.groupby('date'):
        sub = sub.sort_values(p_col, ascending=False)
        n = len(sub); k_n = max(int(n * k), 1)
        topk = sub.head(k_n)
        cap = topk[LABEL_COL].sum() / max(sub[LABEL_COL].sum(), 1)
        caps.append(cap)
    return float(np.mean(caps)), float(np.std(caps))

print(f"  {'model':<25} {'top-20% cap mean':>20} {'std':>10}")
print(f"  {'-'*60}")
for col in ['clim_north_mean', 'clim_lat_band', 'p_m1', 'p_m2', 'p_m3', 'p_m4', 'p_m5', 'p_m6', 'p_m6b']:
    m, s = topk_per_day(test_pred_df, col, 0.2)
    print(f"  {col:<25} {m:>20.4f} {s:>10.4f}")

# --- 11. Calibration curve (South test 2023) ---
def cal_curve(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            confs.append(np.nan); accs.append(np.nan); counts.append(0); continue
        confs.append(p[mask].mean()); accs.append(y[mask].mean()); counts.append(int(mask.sum()))
    return np.array(confs), np.array(accs), np.array(counts)

fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
y_south = y_test
for col, label, color, marker in [
    ('p_m1', 'M1 XGB', '#1f77b4', 'o'),
    ('p_m2', 'M2 XGB+iso', '#ff7f0e', '^'),
    ('p_m3', 'M3 FWI', '#2ca02c', 's'),
    ('p_m4', 'M4 XGB+FWI', '#d62728', 'D'),
    ('p_m5', 'M5 XGB+FWI+iso', '#9467bd', 'v'),
    ('p_m6', 'M6 combiner', '#8c564b', 'P'),
    ('p_m6b', 'M6b (β2≥0)', '#e377c2', 'X'),
]:
    p = test_pred_df[col].values
    confs, accs, counts = cal_curve(p, y_south, n_bins=10)
    mask = counts >= 30
    ax.plot(confs[mask], accs[mask], color=color, marker=marker, label=label, alpha=0.85, linewidth=1.2, markersize=6)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Empirical fire rate')
ax.set_title(f'SC-PMH Exp H1: Calibration curves on South test 2023 (n={len(y_south):,}, base rate={y_south.mean():.4f})')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=8)
fig.savefig(EX / 'calibration_south_2023.png', dpi=150)
plt.close()

# --- 12. Probability distribution (South test 2023) ---
fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
for col, label, color in [
    ('p_m1', 'M1 XGB', '#1f77b4'),
    ('p_m2', 'M2 XGB+iso', '#ff7f0e'),
    ('p_m3', 'M3 FWI', '#2ca02c'),
    ('p_m4', 'M4 XGB+FWI', '#d62728'),
    ('p_m6', 'M6 combiner', '#8c564b'),
    ('p_m6b', 'M6b (β2≥0)', '#e377c2'),
]:
    p = test_pred_df[col].values
    ax.hist(p, bins=40, range=(0, 1), density=True, alpha=0.4, label=label, color=color)
ax.set_yscale('log')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Density (log)')
ax.set_title('SC-PMH Exp H1: Probability distribution — South test 2023')
ax.legend(); ax.grid(alpha=0.3)
fig.savefig(EX / 'prob_dist_south_2023.png', dpi=150)
plt.close()

# --- 13. Save ---
results_df.to_csv(EX / 'results_exp_h1.csv', index=False)
test_pred_df.to_parquet(EX / 'test_predictions.parquet', index=False)
# Save models
m1.save_model(str(EX / 'm1_xgb.json'))
m4.save_model(str(EX / 'm4_xgb_fwi.json'))
with open(EX / 'calibrators.pkl', 'wb') as f:
    pickle.dump({'iso_m1': iso_m1, 'iso_m4': iso_m4}, f)
with open(EX / 'fwi_logistic.pkl', 'wb') as f:
    pickle.dump({'scaler_fwi': scaler_fwi, 'm3': m3}, f)
with open(EX / 'combiner.pkl', 'wb') as f:
    pickle.dump({
        'm6': m6,
        'm6b_coef': res.x,
        'fwi_scalar_scaler': fwi_scaler,
        'eps': eps,
    }, f)

# Save summary JSON
summary = {
    'experiment': 'H1_southern_overprediction_audit',
    'split': 'spatial OOD: train North 2018-2021, val North 2022, test South 2023',
    'n_features': {'base': len(BASE_FEATS), 'fwi': len(FWI_FEATS), 'm4_total': len(FEATS_M4)},
    'south_test_metrics': sub.to_dict('records'),
    'north_id_test_metrics': sub_id.to_dict('records'),
    'top20_capture_south': {
        col: topk_per_day(test_pred_df, col, 0.2)[0]
        for col in ['clim_north_mean', 'clim_lat_band', 'p_m1', 'p_m2', 'p_m3', 'p_m4', 'p_m5', 'p_m6', 'p_m6b']
    },
    'm6_coefs_unconstrained': {'beta0': float(m6.intercept_[0]), 'beta1_logit_ml': float(m6.coef_[0,0]), 'beta2_z_fwi': float(m6.coef_[0,1])},
    'm6b_coefs_constrained': {'beta0': float(res.x[0]), 'beta1_logit_ml': float(res.x[1]), 'beta2_z_fwi': float(res.x[2])},
    'm3_coefs': dict(zip(FWI_FEATS, m3.coef_[0].tolist())),
}
with open(EX / 'exp_h1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n[done] saved → {EX}/results_exp_h1.csv, exp_h1_summary.json, calibration_south_2023.png, prob_dist_south_2023.png")

# --- 14. Diagnosis per decision criteria ---
print("\n=== DECISION TABLE (vs M1 baseline) ===")
print(f"{'model':<22} {'AUROC':>8} {'PR_AUC':>8} {'Brier':>8} {'ECE':>8} {'ΔBrier':>10} {'ΔECE':>10}")
print("-" * 80)
m1_row = sub[sub.model == 'M1_XGB'].iloc[0]
for _, r in sub.sort_values('AUROC', ascending=False).iterrows():
    db = r['Brier'] - m1_row['Brier']
    de = r['ECE'] - m1_row['ECE']
    flag = ""
    if r['model'] == 'M1_XGB':
        flag = " (baseline)"
    elif r['Brier'] < m1_row['Brier'] and r['ECE'] < m1_row['ECE'] and r['AUROC'] >= m1_row['AUROC'] - 0.005:
        flag = " ✓ ACCEPT (Brier↓, ECE↓, AUROC stable)"
    elif r['Brier'] < m1_row['Brier'] or r['ECE'] < m1_row['ECE']:
        flag = " ? partial"
    print(f"{r['model']:<22} {r['AUROC']:>8.4f} {r['PR_AUC']:>8.4f} {r['Brier']:>8.4f} {r['ECE']:>8.4f} {db:>+10.4f} {de:>+10.4f}{flag}")
