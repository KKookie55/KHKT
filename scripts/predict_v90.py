#!/usr/bin/env python3
"""
v9.0 Production predictor for Wildfire Forecasting System.

Loads v9.0 XGBoost model (61 features, forest fire high-confidence label)
and predicts forest fire probability given (lat, lon, temp, humidity, wind, rain, daysSinceRain).

Builds 61-feature vector using:
  - User-provided current weather (t2m, rh, wind, precip, days_since_rain)
  - Cell-static lookup (land cover, geo, NDVI from pre-computed parquet)
  - Climatology (per-cell doy means from 2018-2021) for lags/rolling/anomalies
  - Today's doy for seasonal features

Output JSON:
  {
    "probability": float,        # calibrated forest fire probability
    "raw_probability": float,    # before isotonic calibration
    "risk_level": str,
    "risk_color": str,
    "model_version": "v9.0 forest fire",
    "n_features": 61,
    "features_used": [...]
  }
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression

PROD = '/home/z/my-project/download/production'
MODEL_PATH = os.path.join(PROD, 'model.json')
ISO_PATH = os.path.join(PROD, 'iso.pkl')
LC_PATH = '/home/z/my-project/download/sc_pmh/landcover_expanded.parquet'
GEO_PATH = '/home/z/my-project/download/sc_pmh/geo_features_static.parquet'
NDVI_PATH = '/home/z/my-project/download/sc_pmh/ndvi_cell_features.parquet'
CLIM_PATH = '/home/z/my-project/download/production/cell_climatology.parquet'

# Feature list (must match training)
FEATS = [
    't2m', 'rh', 'wind', 'precip', 'doy',
    't2m_lag1', 't2m_lag3', 't2m_lag7', 't2m_lag14',
    'rh_lag1', 'rh_lag3', 'rh_lag7', 'rh_lag14',
    'wind_lag1', 'wind_lag3', 'wind_lag7', 'wind_lag14',
    'precip_lag1', 'precip_lag3', 'precip_lag7', 'precip_lag14',
    't2m_roll7', 't2m_roll14', 'rh_roll7', 'rh_roll14',
    'precip_roll7_sum', 'precip_roll14_sum', 'precip_roll30_sum',
    'days_since_rain', 't2m_diff3', 'rh_diff3', 'wind_diff3',
    'heat_index', 'wind_chill',
    't2m_anom_doy', 'rh_anom_doy', 'precip_anom_doy', 'wind_anom_doy',
    't2m_anom_roll7', 'rh_anom_roll7', 'precip_anom_roll7',
    'forest_prop', 'shrubland_prop', 'grassland_prop', 'agriculture_prop',
    'urban_prop', 'bare_prop', 'water_prop',
    'elevation_mean', 'elevation_std', 'slope_mean',
    'koppen_Cwa', 'koppen_Am', 'koppen_Aw',
    'latitude_band', 'coastal_distance', 'population_density_proxy',
    'ndvi_mean', 'ndvi_std', 'ndvi_amplitude',
]

_model = None
_iso = None
_lc = None
_geo = None
_ndvi = None
_clim = None

def load_resources():
    global _model, _iso, _lc, _geo, _ndvi, _clim
    if _model is None:
        _model = gbm.Booster()
        _model.load_model(MODEL_PATH)
    if _iso is None:
        with open(ISO_PATH, 'rb') as f:
            _iso = pickle.load(f)
    if _lc is None:
        _lc = pd.read_parquet(LC_PATH)
    if _geo is None:
        _geo = pd.read_parquet(GEO_PATH)
    if _ndvi is None:
        _ndvi = pd.read_parquet(NDVI_PATH)
    if _clim is None:
        _clim = pd.read_parquet(CLIM_PATH)

def snap_to_grid(lat, lon):
    """Snap lat/lon to nearest 0.5° grid point."""
    return round(lat * 2) / 2, round(lon * 2) / 2

def predict(lat, lon, temp, humidity, wind, rain, daysSinceRain, doy=None):
    """Predict forest fire probability for given cell + weather inputs."""
    load_resources()
    
    # Snap to grid
    lat_grid, lon_grid = snap_to_grid(lat, lon)
    
    # Today's doy
    if doy is None:
        from datetime import datetime
        doy = datetime.now().timetuple().tm_yday
    
    # Look up cell-static features
    lc_row = _lc[(_lc.lat == lat_grid) & (_lc.lon == lon_grid)]
    if len(lc_row) == 0:
        # Find nearest cell
        _lc['dist'] = ((_lc.lat - lat_grid)**2 + (_lc.lon - lon_grid)**2)**0.5
        lc_row = _lc.iloc[[_lc['dist'].idxmin()]]
    geo_row = _geo[(_geo.lat == lat_grid) & (_geo.lon == lon_grid)]
    if len(geo_row) == 0:
        _geo['dist'] = ((_geo.lat - lat_grid)**2 + (_geo.lon - lon_grid)**2)**0.5
        geo_row = _geo.iloc[[_geo['dist'].idxmin()]]
    ndvi_row = _ndvi[(_ndvi.lat == lat_grid) & (_ndvi.lon == lon_grid)]
    if len(ndvi_row) == 0:
        _ndvi['dist'] = ((_ndvi.lat - lat_grid)**2 + (_ndvi.lon - lon_grid)**2)**0.5
        ndvi_row = _ndvi.iloc[[_ndvi['dist'].idxmin()]]
    clim_row = _clim[(_clim.lat == lat_grid) & (_clim.lon == lon_grid) & (_clim.doy == doy)]
    if len(clim_row) == 0:
        # Find nearest doy
        _clim_doy = _clim[(_clim.lat == lat_grid) & (_clim.lon == lon_grid)]
        if len(_clim_doy) == 0:
            return {'error': f'No climatology for cell ({lat_grid}, {lon_grid})'}
        _clim_doy['doy_diff'] = abs(_clim_doy.doy - doy)
        clim_row = _clim_doy.iloc[[_clim_doy['doy_diff'].idxmin()]]
    
    # Build feature vector
    # User-provided current weather
    t2m = float(temp)
    rh = float(humidity)
    wind_val = float(wind)
    precip = float(rain)
    days_since_rain = float(daysSinceRain)
    
    # Climatology for this cell+doy (used as proxy for lags when we don't have history)
    t2m_clim = float(clim_row['t2m_clim'].iloc[0])
    rh_clim = float(clim_row['rh_clim'].iloc[0])
    precip_clim = float(clim_row['precip_clim'].iloc[0])
    wind_clim = float(clim_row['wind_clim'].iloc[0])
    
    # Lags (use climatology as proxy — best we can do without history)
    t2m_lag1 = t2m_lag3 = t2m_lag7 = t2m_lag14 = t2m_clim
    rh_lag1 = rh_lag3 = rh_lag7 = rh_lag14 = rh_clim
    wind_lag1 = wind_lag3 = wind_lag7 = wind_lag14 = wind_clim
    precip_lag1 = precip_lag3 = precip_lag7 = precip_lag14 = precip_clim
    
    # Rolling stats (use climatology as proxy)
    t2m_roll7 = t2m_roll14 = t2m_clim
    rh_roll7 = rh_roll14 = rh_clim
    precip_roll7_sum = precip_clim * 7
    precip_roll14_sum = precip_clim * 14
    precip_roll30_sum = precip_clim * 30
    
    # Diff features
    t2m_diff3 = t2m - t2m_clim  # difference from climatology
    rh_diff3 = rh - rh_clim
    wind_diff3 = wind_val - wind_clim
    
    # Derived indices
    heat_index = (t2m - 20) * (50 - rh) / 50.0
    wind_chill = (t2m - 15) * wind_val / 30.0
    
    # Anomalies (current - climatology)
    t2m_anom_doy = t2m - t2m_clim
    rh_anom_doy = rh - rh_clim
    precip_anom_doy = precip - precip_clim
    wind_anom_doy = wind_val - wind_clim
    
    # Rolling anomalies (use single anomaly as proxy)
    t2m_anom_roll7 = t2m_anom_doy
    rh_anom_roll7 = rh_anom_doy
    precip_anom_roll7 = precip_anom_doy
    
    # Land cover
    forest_prop = float(lc_row['forest_prop'].iloc[0])
    shrubland_prop = float(lc_row['shrubland_prop'].iloc[0])
    grassland_prop = float(lc_row['grassland_prop'].iloc[0])
    agriculture_prop = float(lc_row['agriculture_prop'].iloc[0])
    urban_prop = float(lc_row['urban_prop'].iloc[0])
    bare_prop = float(lc_row['bare_prop'].iloc[0])
    water_prop = float(lc_row['water_prop'].iloc[0])
    
    # Geo
    elevation_mean = float(geo_row['elevation_mean'].iloc[0])
    elevation_std = float(geo_row['elevation_std'].iloc[0])
    slope_mean = float(geo_row['slope_mean'].iloc[0])
    koppen_Cwa = int(geo_row['koppen_Cwa'].iloc[0])
    koppen_Am = int(geo_row['koppen_Am'].iloc[0])
    koppen_Aw = int(geo_row['koppen_Aw'].iloc[0])
    latitude_band = int(geo_row['latitude_band'].iloc[0])
    coastal_distance = float(geo_row['coastal_distance'].iloc[0])
    population_density_proxy = float(geo_row['population_density_proxy'].iloc[0])
    
    # NDVI
    ndvi_mean = float(ndvi_row['ndvi_mean'].iloc[0])
    ndvi_std = float(ndvi_row['ndvi_std'].iloc[0])
    ndvi_amplitude = float(ndvi_row['ndvi_amplitude'].iloc[0])
    
    # Build feature dict
    features = {
        't2m': t2m, 'rh': rh, 'wind': wind_val, 'precip': precip, 'doy': doy,
        't2m_lag1': t2m_lag1, 't2m_lag3': t2m_lag3, 't2m_lag7': t2m_lag7, 't2m_lag14': t2m_lag14,
        'rh_lag1': rh_lag1, 'rh_lag3': rh_lag3, 'rh_lag7': rh_lag7, 'rh_lag14': rh_lag14,
        'wind_lag1': wind_lag1, 'wind_lag3': wind_lag3, 'wind_lag7': wind_lag7, 'wind_lag14': wind_lag14,
        'precip_lag1': precip_lag1, 'precip_lag3': precip_lag3, 'precip_lag7': precip_lag7, 'precip_lag14': precip_lag14,
        't2m_roll7': t2m_roll7, 't2m_roll14': t2m_roll14, 'rh_roll7': rh_roll7, 'rh_roll14': rh_roll14,
        'precip_roll7_sum': precip_roll7_sum, 'precip_roll14_sum': precip_roll14_sum, 'precip_roll30_sum': precip_roll30_sum,
        'days_since_rain': days_since_rain,
        't2m_diff3': t2m_diff3, 'rh_diff3': rh_diff3, 'wind_diff3': wind_diff3,
        'heat_index': heat_index, 'wind_chill': wind_chill,
        't2m_anom_doy': t2m_anom_doy, 'rh_anom_doy': rh_anom_doy,
        'precip_anom_doy': precip_anom_doy, 'wind_anom_doy': wind_anom_doy,
        't2m_anom_roll7': t2m_anom_roll7, 'rh_anom_roll7': rh_anom_roll7, 'precip_anom_roll7': precip_anom_roll7,
        'forest_prop': forest_prop, 'shrubland_prop': shrubland_prop, 'grassland_prop': grassland_prop,
        'agriculture_prop': agriculture_prop, 'urban_prop': urban_prop, 'bare_prop': bare_prop, 'water_prop': water_prop,
        'elevation_mean': elevation_mean, 'elevation_std': elevation_std, 'slope_mean': slope_mean,
        'koppen_Cwa': koppen_Cwa, 'koppen_Am': koppen_Am, 'koppen_Aw': koppen_Aw,
        'latitude_band': latitude_band, 'coastal_distance': coastal_distance, 'population_density_proxy': population_density_proxy,
        'ndvi_mean': ndvi_mean, 'ndvi_std': ndvi_std, 'ndvi_amplitude': ndvi_amplitude,
    }
    
    # Build DMatrix with feature_names (must match model's expected feature_names order)
    feature_names = _model.feature_names
    X = pd.DataFrame([features], columns=feature_names)
    dmat = gbm.DMatrix(X)
    
    # Predict
    raw_p = float(_model.predict(dmat)[0])
    p_calibrated = float(_iso.predict([raw_p])[0])
    p_calibrated = max(0.001, min(0.999, p_calibrated))
    
    # Risk level
    if p_calibrated >= 0.30:
        risk_level, risk_color = "Nguy hiểm", "#4c1d95"
    elif p_calibrated >= 0.20:
        risk_level, risk_color = "Rất cao", "#7f1d1d"
    elif p_calibrated >= 0.10:
        risk_level, risk_color = "Cao", "#dc2626"
    elif p_calibrated >= 0.05:
        risk_level, risk_color = "Trung bình", "#f59e0b"
    else:
        risk_level, risk_color = "Thấp", "#22c55e"
    
    return {
        'probability': p_calibrated,
        'raw_probability': raw_p,
        'calibrated_probability': p_calibrated,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_score': int(p_calibrated * 100),
        'model_version': 'v9.0 forest fire (61 feats)',
        'n_features': len(feature_names),
        'features_used_count': len(features),
        'cell_lat': lat_grid,
        'cell_lon': lon_grid,
        'doy': doy,
        'is_cold_start': lat_grid < 14,
        'note': 'Predictions use climatology for lag/rolling features (live weather history not fetched). Cell-static features (land cover, geo, NDVI) from pre-computed lookup.',
    }

def predict_batch(inputs):
    """Predict for multiple cells/inputs."""
    return [predict(inp.get('lat', 16.5), inp.get('lon', 106.5),
                    inp.get('temp'), inp.get('humidity'), inp.get('wind'),
                    inp.get('rain'), inp.get('daysSinceRain'),
                    inp.get('doy')) for inp in inputs]

if __name__ == '__main__':
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    if 'inputs' in input_data:
        results = predict_batch(input_data['inputs'])
        print(json.dumps({'results': results}))
    else:
        result = predict(
            input_data.get('lat', 16.5),
            input_data.get('lon', 106.5),
            input_data.get('temp'),
            input_data.get('humidity'),
            input_data.get('wind'),
            input_data.get('rain'),
            input_data.get('daysSinceRain'),
            input_data.get('doy')
        )
        print(json.dumps(result))
