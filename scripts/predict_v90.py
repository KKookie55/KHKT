#!/usr/bin/env python3
"""
v9.0 Production predictor — FIXED version with real weather history.

Fetches last 14 days of weather from Open-Meteo Archive API for given lat/lon,
computes proper lag/rolling/anomaly features (NOT climatology proxy).

Output JSON:
  {
    "probability": float,
    "raw_probability": float,
    "risk_level": str,
    "risk_color": str,
    "model_version": "v9.0 forest fire (61 feats) — with real weather history",
    "n_features": 61,
    "weather_history_days": 14,
    "weather_source": "Open-Meteo Archive",
    ...
  }
"""
import os, sys, json, pickle, urllib.request, urllib.error
from datetime import datetime, timedelta
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
    return round(lat * 2) / 2, round(lon * 2) / 2

def fetch_weather_history(lat, lon, days=30):
    """Fetch last N days of weather from Open-Meteo Archive API."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min,"
        f"wind_speed_10m_max,precipitation_sum&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        
        if 'daily' not in data:
            return None, f"No daily data in API response: {data}"
        
        d = data['daily']
        # Compute daily means from max/min
        df = pd.DataFrame({
            'date': pd.to_datetime(d['time']),
            't2m': [(mx + mn) / 2 for mx, mn in zip(d['temperature_2m_max'], d['temperature_2m_min'])],
            'rh': [(mx + mn) / 2 for mx, mn in zip(d['relative_humidity_2m_max'], d['relative_humidity_2m_min'])],
            'wind': d['wind_speed_10m_max'],
            'precip': d['precipitation_sum'],
        })
        # Wind speed from km/h to m/s
        df['wind'] = df['wind'] / 3.6
        return df, None
    except Exception as e:
        return None, str(e)

def compute_days_since_rain(precip_series, threshold=0.5):
    """Count consecutive days since last rain (precip >= threshold)."""
    days = 0
    for p in reversed(precip_series):
        if p >= threshold:
            break
        days += 1
    return days

def predict(lat, lon, override_temp=None, override_humidity=None, override_wind=None, override_rain=None, override_dsr=None, doy=None):
    """Predict forest fire probability using REAL weather history."""
    load_resources()
    
    lat_grid, lon_grid = snap_to_grid(lat, lon)
    
    if doy is None:
        doy = datetime.now().timetuple().tm_yday
    
    # If user provides override weather values, use climatology for lags (scenario testing mode)
    # Otherwise, fetch real weather history for today's actual prediction
    has_overrides = any(v is not None for v in [override_temp, override_humidity, override_wind, override_rain, override_dsr])
    
    if not has_overrides:
        # Real prediction mode — fetch weather history
        weather_df, weather_err = fetch_weather_history(lat_grid, lon_grid, days=35)
        if weather_df is None or len(weather_df) < 14:
            weather_df = None
            weather_source = f"climatology fallback (Open-Meteo failed: {weather_err})"
        else:
            weather_source = f"Open-Meteo Archive ({len(weather_df)} days fetched)"
    else:
        # Scenario testing mode — user provided override values, use climatology for lags
        weather_df = None
        weather_source = "scenario testing (override weather + climatology for lags)"
    
    # Look up cell-static
    lc_row = _lc[(_lc.lat == lat_grid) & (_lc.lon == lon_grid)]
    geo_row = _geo[(_geo.lat == lat_grid) & (_geo.lon == lon_grid)]
    ndvi_row = _ndvi[(_ndvi.lat == lat_grid) & (_ndvi.lon == lon_grid)]
    clim_row = _clim[(_clim.lat == lat_grid) & (_clim.lon == lon_grid) & (_clim.doy == doy)]
    if len(clim_row) == 0:
        cd = _clim[(_clim.lat == lat_grid) & (_clim.lon == lon_grid)]
        if len(cd) == 0:
            return {'error': f'No climatology for cell ({lat_grid}, {lon_grid})'}
        cd['d'] = abs(cd.doy - doy)
        clim_row = cd.iloc[[cd['d'].idxmin()]]
    
    if len(lc_row) == 0 or len(geo_row) == 0 or len(ndvi_row) == 0:
        return {'error': f'No static features for cell ({lat_grid}, {lon_grid})'}
    
    # Climatology
    t2m_clim = float(clim_row['t2m_clim'].iloc[0])
    rh_clim = float(clim_row['rh_clim'].iloc[0])
    precip_clim = float(clim_row['precip_clim'].iloc[0])
    wind_clim = float(clim_row['wind_clim'].iloc[0])
    
    # Determine current day's weather
    if weather_df is not None and len(weather_df) > 0:
        # Latest day in archive = today's actual weather
        last_row = weather_df.iloc[-1]
        t2m_today = override_temp if override_temp is not None else float(last_row['t2m'])
        rh_today = override_humidity if override_humidity is not None else float(last_row['rh'])
        wind_today = override_wind if override_wind is not None else float(last_row['wind'])
        precip_today = override_rain if override_rain is not None else float(last_row['precip'])
        days_since_rain = override_dsr if override_dsr is not None else compute_days_since_rain(weather_df['precip'].values)
        
        # Real lags from weather history
        # weather_df is sorted by date; last day is "today"; lag1 = yesterday, lag3 = 3 days ago, etc.
        # Use second-to-last as lag1, fourth-to-last as lag3, etc.
        t2m_series = weather_df['t2m'].values
        rh_series = weather_df['rh'].values
        wind_series = weather_df['wind'].values
        precip_series = weather_df['precip'].values
        
        # Lag1 = -2 (yesterday), lag3 = -4, lag7 = -8, lag14 = -15
        # (since last entry is "today", the day before is -2 in 0-indexed)
        def get_lag(series, days_ago):
            idx = -days_ago - 1
            if abs(idx) > len(series):
                return float(clim_row_for_var)
            val = series[idx]
            if pd.isna(val):
                return float(clim_row_for_var)
            return float(val)
        
        # Get climatology per variable
        t2m_lag1 = get_lag(t2m_series, 1) if len(t2m_series) >= 2 else t2m_clim
        t2m_lag3 = get_lag(t2m_series, 3) if len(t2m_series) >= 4 else t2m_clim
        t2m_lag7 = get_lag(t2m_series, 7) if len(t2m_series) >= 8 else t2m_clim
        t2m_lag14 = get_lag(t2m_series, 14) if len(t2m_series) >= 15 else t2m_clim
        
        rh_lag1 = get_lag(rh_series, 1) if len(rh_series) >= 2 else rh_clim
        rh_lag3 = get_lag(rh_series, 3) if len(rh_series) >= 4 else rh_clim
        rh_lag7 = get_lag(rh_series, 7) if len(rh_series) >= 8 else rh_clim
        rh_lag14 = get_lag(rh_series, 14) if len(rh_series) >= 15 else rh_clim
        
        wind_lag1 = get_lag(wind_series, 1) if len(wind_series) >= 2 else wind_clim
        wind_lag3 = get_lag(wind_series, 3) if len(wind_series) >= 4 else wind_clim
        wind_lag7 = get_lag(wind_series, 7) if len(wind_series) >= 8 else wind_clim
        wind_lag14 = get_lag(wind_series, 14) if len(wind_series) >= 15 else wind_clim
        
        precip_lag1 = get_lag(precip_series, 1) if len(precip_series) >= 2 else precip_clim
        precip_lag3 = get_lag(precip_series, 3) if len(precip_series) >= 4 else precip_clim
        precip_lag7 = get_lag(precip_series, 7) if len(precip_series) >= 8 else precip_clim
        precip_lag14 = get_lag(precip_series, 14) if len(precip_series) >= 15 else precip_clim
        
        # Rolling stats from real history
        t2m_roll7 = float(np.nanmean(t2m_series[-7:])) if len(t2m_series) >= 7 else t2m_clim
        t2m_roll14 = float(np.nanmean(t2m_series[-14:])) if len(t2m_series) >= 14 else t2m_clim
        rh_roll7 = float(np.nanmean(rh_series[-7:])) if len(rh_series) >= 7 else rh_clim
        rh_roll14 = float(np.nanmean(rh_series[-14:])) if len(rh_series) >= 14 else rh_clim
        precip_roll7_sum = float(np.nansum(precip_series[-7:])) if len(precip_series) >= 7 else precip_clim * 7
        precip_roll14_sum = float(np.nansum(precip_series[-14:])) if len(precip_series) >= 14 else precip_clim * 14
        precip_roll30_sum = float(np.nansum(precip_series[-30:])) if len(precip_series) >= 30 else precip_clim * 30
        
        # Diff = today - lag3
        t2m_diff3 = t2m_today - t2m_lag3
        rh_diff3 = rh_today - rh_lag3
        wind_diff3 = wind_today - wind_lag3
    else:
        # SCENARIO mode: user provided override values
        # Use override values for ALL lag/rolling features (propagate)
        # This makes slider values propagate to ALL time-shifted features
        t2m_today = override_temp if override_temp is not None else t2m_clim
        rh_today = override_humidity if override_humidity is not None else rh_clim
        wind_today = override_wind if override_wind is not None else wind_clim
        precip_today = override_rain if override_rain is not None else precip_clim
        days_since_rain = override_dsr if override_dsr is not None else 5
        
        # Set all lags = today's value (artificial — assumes weather has been like this for 14 days)
        t2m_lag1 = t2m_lag3 = t2m_lag7 = t2m_lag14 = t2m_today
        rh_lag1 = rh_lag3 = rh_lag7 = rh_lag14 = rh_today
        wind_lag1 = wind_lag3 = wind_lag7 = wind_lag14 = wind_today
        precip_lag1 = precip_lag3 = precip_lag7 = precip_lag14 = precip_today
        
        # Rolling = today's value (propagated)
        t2m_roll7 = t2m_roll14 = t2m_today
        rh_roll7 = rh_roll14 = rh_today
        precip_roll7_sum = precip_today * 7
        precip_roll14_sum = precip_today * 14
        precip_roll30_sum = precip_today * 30
        
        # Diff = today - today = 0 (no change in scenario)
        t2m_diff3 = 0
        rh_diff3 = 0
        wind_diff3 = 0
    
    # Derived
    heat_index = (t2m_today - 20) * (50 - rh_today) / 50.0
    wind_chill = (t2m_today - 15) * wind_today / 30.0
    
    # Anomalies (current - climatology for that doy)
    t2m_anom_doy = t2m_today - t2m_clim
    rh_anom_doy = rh_today - rh_clim
    precip_anom_doy = precip_today - precip_clim
    wind_anom_doy = wind_today - wind_clim
    
    # Rolling anomalies = rolling mean - climatology (approximate)
    t2m_anom_roll7 = t2m_roll7 - t2m_clim
    rh_anom_roll7 = rh_roll7 - rh_clim
    precip_anom_roll7 = precip_roll7_sum - precip_clim * 7
    
    # Cell-static features
    forest_prop = float(lc_row['forest_prop'].iloc[0])
    shrubland_prop = float(lc_row['shrubland_prop'].iloc[0])
    grassland_prop = float(lc_row['grassland_prop'].iloc[0])
    agriculture_prop = float(lc_row['agriculture_prop'].iloc[0])
    urban_prop = float(lc_row['urban_prop'].iloc[0])
    bare_prop = float(lc_row['bare_prop'].iloc[0])
    water_prop = float(lc_row['water_prop'].iloc[0])
    
    elevation_mean = float(geo_row['elevation_mean'].iloc[0])
    elevation_std = float(geo_row['elevation_std'].iloc[0])
    slope_mean = float(geo_row['slope_mean'].iloc[0])
    koppen_Cwa = int(geo_row['koppen_Cwa'].iloc[0])
    koppen_Am = int(geo_row['koppen_Am'].iloc[0])
    koppen_Aw = int(geo_row['koppen_Aw'].iloc[0])
    latitude_band = int(geo_row['latitude_band'].iloc[0])
    coastal_distance = float(geo_row['coastal_distance'].iloc[0])
    population_density_proxy = float(geo_row['population_density_proxy'].iloc[0])
    
    ndvi_mean = float(ndvi_row['ndvi_mean'].iloc[0])
    ndvi_std = float(ndvi_row['ndvi_std'].iloc[0])
    ndvi_amplitude = float(ndvi_row['ndvi_amplitude'].iloc[0])
    
    # Build feature dict
    features = {
        't2m': t2m_today, 'rh': rh_today, 'wind': wind_today, 'precip': precip_today, 'doy': doy,
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
    
    feature_names = _model.feature_names
    X = pd.DataFrame([features], columns=feature_names)
    dmat = gbm.DMatrix(X)
    
    raw_p = float(_model.predict(dmat)[0])
    p_calibrated = float(_iso.predict([raw_p])[0])
    p_calibrated = max(0.001, min(0.999, p_calibrated))
    
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
        'model_version': 'v9.0 forest fire (61 feats) — real weather history',
        'n_features': len(feature_names),
        'cell_lat': lat_grid,
        'cell_lon': lon_grid,
        'doy': doy,
        'is_cold_start': lat_grid < 14,
        'weather_source': weather_source,
        'current_weather': {
            't2m': round(t2m_today, 2),
            'rh': round(rh_today, 2),
            'wind': round(wind_today, 2),
            'precip': round(precip_today, 2),
            'days_since_rain': days_since_rain,
        },
        'weather_history_used': weather_df is not None,
        'lags_real': weather_df is not None,
    }

def predict_batch(inputs):
    """Predict for multiple cells/inputs."""
    results = []
    for inp in inputs:
        try:
            result = predict(
                inp.get('lat', 16.5),
                inp.get('lon', 106.5),
                inp.get('temp'),
                inp.get('humidity'),
                inp.get('wind'),
                inp.get('rain'),
                inp.get('daysSinceRain'),
                inp.get('doy')
            )
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})
    return results

if __name__ == '__main__':
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
