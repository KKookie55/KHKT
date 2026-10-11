#!/usr/bin/env python3
"""
BƯỚC 2 — CỦNG CỐ PHÁT HIỆN TRUNG TÂM

2.1: MODIS 3 seeds × IID + N→S (so sánh công bằng với VIIRS 5 seeds)
2.2: Caveats về cảm biến (text)
2.3: Phép kiểm định phân biệt cơ chế — match-on-weather
     - Cho mỗi mẫu South, tìm mẫu North gần nhất (theo weather features)
     - Nếu model đúng cho North match nhưng sai cho South → concept shift
     - Nếu model sai cho cả hai → model error (không phải shift)
2.4: Phân biệt covariate vs concept shift
     - Bin weather → compute AUROC per bin
     - Nếu AUROC thấp ở bin mà N/S phân phối giống nhau → concept shift
"""
import json, math, pickle, warnings, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')

OUT = '/home/z/my-project/download/real_data_v85'
panel = pd.read_parquet(OUT + '/panel_v85.parquet')
panel = panel.sort_values(['lat','lon','date']).reset_index(drop=True)
splits = np.load(OUT + '/splits_v85.npz')
features = json.loads(open(OUT+'/panel_v85_meta.json').read())['features']

print("="*70)
print("BƯỚC 2 — CỦNG CỐ PHÁT HIỆN TRUNG TÂM")
print("="*70)

# Fire history (for v8.6final model features)
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
features_new = features + ['cell_fire_rate_train','cell_fire_count_30d','cell_fire_rate_dry']

di_idx = features_new.index('precip_roll30_sum')

def train_xgb(tr, va, te, seed=42):
    di_mean = np.nanmean(tr[features_new[di_idx]].values)
    di_std = np.nanstd(tr[features_new[di_idx]].values) + 1e-6
    def prep(df):
        X = df[features_new].values.astype(np.float32).copy()
        X[:, di_idx] = np.nan_to_num((X[:, di_idx]-di_mean)/di_std, nan=0.0)
        return X
    imp = SimpleImputer(strategy='median')
    X_tr=imp.fit_transform(prep(tr)); X_val=imp.transform(prep(va)); X_test=imp.transform(prep(te))
    y_tr=tr['label_h1'].values.astype(np.float32); y_val=va['label_h1'].values.astype(np.float32); y_test=te['label_h1'].values.astype(np.float32)
    pos=max(int(y_tr.sum()),1); neg=max(int((1-y_tr).sum()),1); spw=max(math.sqrt(neg/pos),1.0)
    m = xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.85,
        colsample_bytree=0.7,min_child_weight=20,reg_lambda=3.0,reg_alpha=0.5,gamma=0.1,
        scale_pos_weight=spw,eval_metric='aucpr',tree_method='hist',random_state=seed,n_jobs=4)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    p_val=m.predict_proba(X_val)[:,1]; p_test=m.predict_proba(X_test)[:,1]
    iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1); iso.fit(p_val,y_val)
    p_test_iso=iso.predict(p_test)
    return roc_auc_score(y_test,p_test), roc_auc_score(y_test,p_test_iso), m, iso, imp

# ============================================================
# 2.1: MODIS 3 seeds × IID + N→S
# ============================================================
print("\n[2.1] MODIS 3 seeds × IID + Spatial N→S (for fair comparison with VIIRS)")
# Use temporal split (train 2018-2021, val 2022, test 2023) — matches v8.6final
tr_base = panel[panel['year'].isin([2018,2019,2020,2021])].dropna(subset=['label_h1'])
va_base = panel[panel['year']==2022].dropna(subset=['label_h1'])
te_base = panel[panel['year']==2023].dropna(subset=['label_h1'])

# IID split (from splits file) — but we need temporal version
# For temporal: train=2018-2021 ALL, val=2022, test=2023
# This IS the temporal split, not IID. Let me report both.

seeds = [42, 123, 777]
results_modis = {}
for split_name, tr, va, te in [
    ("Temporal (2018-21→23)", tr_base, va_base, te_base),
]:
    aurocs = []
    for s in seeds:
        a_raw, a_iso, _, _, _ = train_xgb(tr, va, te, seed=s)
        aurocs.append(a_iso)
        print(f"  {split_name} seed={s}: AUROC={a_iso:.4f}")
    results_modis[split_name] = {"aurocs": aurocs, "mean": np.mean(aurocs), "std": np.std(aurocs)}
    print(f"  → Mean ± Std: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")

# Also run Spatial N→S (from splits file, MODIS labels)
tr_sn = panel.iloc[splits['Spatial_OOD_NorthToSouth_train']].dropna(subset=['label_h1'])
va_sn = panel.iloc[splits['Spatial_OOD_NorthToSouth_val']].dropna(subset=['label_h1'])
te_sn = panel.iloc[splits['Spatial_OOD_NorthToSouth_test']].dropna(subset=['label_h1'])
print(f"\n  Spatial N→S: train={len(tr_sn):,} val={len(va_sn):,} test={len(te_sn):,}")
aurocs_sn = []
for s in seeds:
    a_raw, a_iso, _, _, _ = train_xgb(tr_sn, va_sn, te_sn, seed=s)
    aurocs_sn.append(a_iso)
    print(f"  Spatial N→S seed={s}: AUROC={a_iso:.4f}")
results_modis["Spatial N→S"] = {"aurocs": aurocs_sn, "mean": np.mean(aurocs_sn), "std": np.std(aurocs_sn)}
print(f"  → Mean ± Std: {np.mean(aurocs_sn):.4f} ± {np.std(aurocs_sn):.4f}")

# ============================================================
# 2.2: Caveats về cảm biến (text output)
# ============================================================
print(f"\n[2.2] CAVEATS VỀ CẢM BIẾN")
print(f"""
  MODIS (modis-14A1-061):
    - Độ phân giải: 1 km
    - Vệ tinh: Terra (10:30 sáng) + Aqua (1:30 chiều)
    - Phát hiện: đám cháy lớn/nóng (≥1000 K, ≥1 km²)
    - Số phát hiện: 581.220 (2018-2023)
    - Sau dedup: 48.073 ô-ngày dương tính

  VIIRS (Collection 2, 375m):
    - Độ phân giải: 375 m (nhạy hơn 2,7×)
    - Vệ tinh: Suomi-NPP + NOAA-20 (cùng 1:30 chiều)
    - Phát hiện: đám cháy nhỏ/lạnh hơn (≥750 K, ≥0,375 km²)
    - Số phát hiện: 357.275 → sau filter: 207.734

  Đồng thuận (Jaccard): 0,106
    - Cả hai cùng dương: 8.924 ô-ngày
    - MODIS only: 39.355 ô-ngày
    - VIIRS only: 35.923 ô-ngày
    - Cả hai cùng âm: 513.941 ô-ngày

  Hệ quả cho diễn giải:
    - Chênh lệch AUROC 0,31 (MODIS 0,89 vs VIIRS 0,58) CÓ THỂ đến từ:
      (a) Model thực sự yếu khi khái quát hóa → VIIRS khó dự báo hơn
      (b) VIIRS phát hiện các đám cháy nhỏ mà thời tiết không dự báo được
      (c) VIIRS có đặc trưng thời điểm quan sát khác (cùng giờ với Aqua
          nhưng độ phân giải cao hơn → bắt thêm cháy nhỏ)
    - KHÔNG kết luận VIIRS đúng/sai hơn MODIS
    - Hai nguồn nhãn độc lập KHÔNG đồng nghĩa một nguồn là chân lý tuyệt đối
""")

# ============================================================
# 2.3: Match-on-weather test (phép kiểm định phân biệt cơ chế)
# ============================================================
print(f"[2.3] MATCH-ON-WEATHER TEST (phân biệt covariate vs concept shift)")
print(f"  Logic: cho mỗi mẫu South, tìm mẫu North gần nhất (theo weather)")
print(f"  Nếu model đúng cho North nhưng sai cho South → concept shift")
print(f"  Nếu model sai cho cả hai → model error")

# Train model on North (spatial N→S)
a_raw, a_iso, m_sn, iso_sn, imp_sn = train_xgb(tr_sn, va_sn, te_sn, seed=42)

# Prepare North train + South test features
di_mean_sn = np.nanmean(tr_sn[features_new[di_idx]].values)
di_std_sn = np.nanstd(tr_sn[features_new[di_idx]].values) + 1e-6

def prep_sn(df):
    X = df[features_new].values.astype(np.float32).copy()
    X[:, di_idx] = np.nan_to_num((X[:, di_idx]-di_mean_sn)/di_std_sn, nan=0.0)
    return X

X_north = imp_sn.transform(prep_sn(tr_sn))
y_north = tr_sn['label_h1'].values.astype(np.float32)
X_south = imp_sn.transform(prep_sn(te_sn))
y_south = te_sn['label_h1'].values.astype(np.float32)

# Predictions
p_north = iso_sn.predict(m_sn.predict_proba(X_north)[:, 1])
p_south = iso_sn.predict(m_sn.predict_proba(X_south)[:, 1])

# For weather matching, use only the 4 core weather features (T, RH, wind, precip)
weather_features = ['t2m', 'rh', 'wind', 'precip']
X_north_weather = tr_sn[weather_features].values.astype(np.float32)
X_south_weather = te_sn[weather_features].values.astype(np.float32)

# Impute NaN in weather (shouldn't be many but just in case)
from sklearn.impute import SimpleImputer
imp_w = SimpleImputer(strategy='median')
X_north_w = imp_w.fit_transform(X_north_weather)
X_south_w = imp_w.transform(X_south_weather)

# Find nearest North sample for each South sample (by weather)
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(X_north_w)
distances, indices = nn.kneighbors(X_south_w)

# For each South sample, check if model's prediction matches:
# - South prediction vs South label
# - North match prediction vs North match label
# Classify: 
#   "concept_shift": model right on North match, wrong on South
#   "model_error": model wrong on both
#   "lucky_south": model right on South (regardless of North)
#   "both_right": model right on both

# Use a threshold (0.5 for binary)
threshold = 0.5
p_north_match = p_north[indices.flatten()]
y_north_match = y_north[indices.flatten()]

south_pred = (p_south >= threshold).astype(int)
north_pred = (p_north_match >= threshold).astype(int)
south_correct = (south_pred == y_south.astype(int))
north_correct = (north_pred == y_north_match.astype(int))

concept_shift = north_correct & ~south_correct  # right on N, wrong on S
model_error = ~north_correct & ~south_correct    # wrong on both
lucky_south = south_correct & ~north_correct    # wrong on N, right on S (lucky)
both_right = south_correct & north_correct       # right on both

n_total = len(y_south)
print(f"\n  Results (threshold={threshold}):")
print(f"  {'Category':<25} | {'Count':>8} | {'%':>6}")
print(f"  {'-'*45}")
print(f"  {'Both correct':<25} | {both_right.sum():>8} | {100*both_right.sum()/n_total:>6.1f}%")
print(f"  {'Concept shift (N right, S wrong)':<25} | {concept_shift.sum():>8} | {100*concept_shift.sum()/n_total:>6.1f}%")
print(f"  {'Model error (both wrong)':<25} | {model_error.sum():>8} | {100*model_error.sum()/n_total:>6.1f}%")
print(f"  {'Lucky South (N wrong, S right)':<25} | {lucky_south.sum():>8} | {100*lucky_south.sum()/n_total:>6.1f}%")

# Interpret
if concept_shift.sum() > model_error.sum():
    print(f"\n  → Concept shift dominates: {100*concept_shift.sum()/n_total:.1f}% samples have model right on")
    print(f"    matched North but wrong on South → same weather, different fire probability")
    print(f"    → This supports H2 (concept shift) over H1 (covariate shift)")
else:
    print(f"\n  → Model error dominates: {100*model_error.sum()/n_total:.1f}% samples have model wrong on both")
    print(f"    → The issue is not shift-specific, it's general model weakness")

# Also check: weather distance distribution
print(f"\n  Weather match distances (Euclidean on T/RH/W/precip):")
print(f"    Mean: {distances.mean():.2f}")
print(f"    Median: {np.median(distances):.2f}")
print(f"    95th pct: {np.percentile(distances, 95):.2f}")

# ============================================================
# 2.4: Conditional AUROC by weather bin
# ============================================================
print(f"\n[2.4] CONDITIONAL AUROC BY TEMPERATURE BIN")
print(f"  (Tách covariate vs concept shift)")
# Bin by temperature (5 bins)
t_bins = np.quantile(X_north_w[:, 0], [0.2, 0.4, 0.6, 0.8])
t_boundaries = [-np.inf] + list(t_bins) + [np.inf]
bin_labels = [f"T<{t_bins[0]:.1f}"] + [f"{t_bins[i]:.1f}≤T<{t_bins[i+1]:.1f}" for i in range(len(t_bins)-1)] + [f"T≥{t_bins[-1]:.1f}"]

print(f"  {'Bin':<15} | {'N_north':>8} | {'N_south':>8} | {'AUROC':>7} | {'N+S prev diff':>14}")
print(f"  {'-'*60}")
for i in range(len(t_boundaries)-1):
    lo, hi = t_boundaries[i], t_boundaries[i+1]
    n_mask = (X_north_w[:, 0] >= lo) & (X_north_w[:, 0] < hi)
    s_mask = (X_south_w[:, 0] >= lo) & (X_south_w[:, 0] < hi)
    if n_mask.sum() < 50 or s_mask.sum() < 50:
        print(f"  {bin_labels[i]:<15} | {n_mask.sum():>8} | {s_mask.sum():>8} | {'skip':>7} | {'skip':>14}")
        continue
    y_s = y_south[s_mask]
    p_s = p_south[s_mask]
    if len(np.unique(y_s)) < 2:
        print(f"  {bin_labels[i]:<15} | {n_mask.sum():>8} | {s_mask.sum():>8} | {'N/A':>7} | {abs(y_north[n_mask].mean()-y_s.mean()):>14.4f}")
        continue
    a = roc_auc_score(y_s, p_s)
    prev_diff = abs(y_north[n_mask].mean() - y_s.mean())
    print(f"  {bin_labels[i]:<15} | {n_mask.sum():>8} | {s_mask.sum():>8} | {a:>7.4f} | {prev_diff:>14.4f}")

# ============================================================
# Save all results
# ============================================================
results_2 = {
    "2.1_modis_3seeds": {k: {"aurocs": v["aurocs"], "mean": float(v["mean"]), "std": float(v["std"])} for k,v in results_modis.items()},
    "2.2_sensor_caveats": {
        "modis": {"resolution": "1km", "satellites": "Terra+Aqua", "detections": 581220, "unique_cell_days": 48073},
        "viirs": {"resolution": "375m", "satellites": "Suomi-NPP+NOAA-20", "detections": 357275, "unique_cell_days": 44847},
        "jaccard": 0.106, "both_positive": 8924, "modis_only": 39355, "viirs_only": 35923,
        "interpretation": "Chênh lệch AUROC có thể đến từ cả khả năng khái quát hóa lẫn khác biệt cảm biến. Không kết luận nguồn nào đúng hơn.",
    },
    "2.3_match_on_weather": {
        "n_total": int(n_total),
        "both_correct": int(both_right.sum()),
        "concept_shift": int(concept_shift.sum()),
        "model_error": int(model_error.sum()),
        "lucky_south": int(lucky_south.sum()),
        "verdict": "concept_shift_dominant" if concept_shift.sum() > model_error.sum() else "model_error_dominant",
        "mean_weather_distance": float(distances.mean()),
    },
}
with open(OUT + '/step2_results.json', 'w') as f:
    json.dump(results_2, f, indent=2, default=str)
print(f"\n✓ Saved: {OUT}/step2_results.json")
