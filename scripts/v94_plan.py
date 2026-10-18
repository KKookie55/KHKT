"""v9.4 Improvement Plan — for national competition.

Key issues found in v9.3:
1. Land cover features have ZERO importance (0.0%) — used nearest cell approximation
   Fix: Download ESA WorldCover 10m, zonal stats per district
2. cell_fire_count_30d (24% importance) uses climatology at inference, not real-time
   Fix: Fetch FIRMS last-30-day fire count per district at inference time
3. NDVI only 1.1% importance — 23 composites/year is too sparse
   Fix: Temporal interpolation to daily NDVI

New features to add (v9.4):
1. ESA WorldCover land cover per district (forest%, agri%, urban%, water% — REAL values)
2. SRTM elevation mean + std + slope per district
3. Spatial lag: mean fire rate of K nearest neighbor districts (spatial autocorrelation)
4. Road density per district (from OSM)
5. Population density per district (from WorldPop)

New model improvements:
1. Optuna hyperparameter tuning (200 trials)
2. LightGBM comparison (sometimes better than XGBoost)
3. Ensemble: XGBoost + LightGBM (weighted average)
4. Proper temporal split: train 2018-2021, val 2022, test 2023

Expected improvement:
- AUROC: 0.929 → 0.940+ (real land cover + spatial lag)
- Brier: 0.076 → 0.060 (better calibration with more features)
- Land cover importance: 0% → 5-10%
"""
print("v9.4 plan written")
