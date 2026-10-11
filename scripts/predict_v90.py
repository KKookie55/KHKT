#!/usr/bin/env python3
"""
v9.0 Production predictor — FINAL version with REAL forecast.

Fetches:
  1. Past 35 days from Open-Meteo Archive API → for lags/rolling/days_since_rain
  2. Next 7 days from Open-Meteo Forecast API → for today + tomorrow forecast

Builds 61 features properly:
  - Real lag1/3/7/14 from archive (past actual weather)
  - Real rolling7/14/30 from archive
  - Real days_since_rain from archive
  - "Today" weather = forecast API's today value (or user override)
  - Anomalies vs climatology for that doy

If user provides override values, those REPLACE today's weather (scenario mode).
Otherwise, real forecast from Open-Meteo.
"""
import os, sys, json, pickle, urllib.request, urllib.error
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression

PROD = '/home/z/my-project/download/production'
MODEL_PATH = os.path.join(PROD, 'model.json')           # forest fire model (high-conf)
MODEL_ANY_PATH = os.path.join(PROD, 'model_any_fire.json')  # any fire model
ISO_PATH = os.path.join(PROD, 'iso.pkl')                  # forest fire iso
ISO_ANY_PATH = os.path.join(PROD, 'iso_any_fire.pkl')     # any fire iso
LC_PATH = '/home/z/my-project/download/sc_pmh/landcover_expanded.parquet'
GEO_PATH = '/home/z/my-project/download/sc_pmh/geo_features_static.parquet'
NDVI_PATH = '/home/z/my-project/download/sc_pmh/ndvi_cell_features.parquet'
CLIM_PATH = os.path.join(PROD, 'cell_climatology.parquet')

_model = None
_model_any = None
_iso = None
_iso_any = None
_lc = None
_geo = None
_ndvi = None
_clim = None

def load_resources():
    global _model, _model_any, _iso, _iso_any, _lc, _geo, _ndvi, _clim
    if _model is None:
        _model = gbm.Booster()
        _model.load_model(MODEL_PATH)
    if _model_any is None:
        _model_any = gbm.Booster()
        _model_any.load_model(MODEL_ANY_PATH)
    if _iso is None:
        with open(ISO_PATH, 'rb') as f:
            _iso = pickle.load(f)
    if _iso_any is None:
        with open(ISO_ANY_PATH, 'rb') as f:
            _iso_any = pickle.load(f)
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

def fetch_archive(lat, lon, days=35):
    """Fetch past N days from Open-Meteo Archive API."""
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
            return None, f"No daily in archive: {data}"
        d = data['daily']
        df = pd.DataFrame({
            'date': pd.to_datetime(d['time']),
            't2m': [(mx + mn) / 2 if mx is not None and mn is not None else np.nan
                    for mx, mn in zip(d['temperature_2m_max'], d['temperature_2m_min'])],
            'rh': [(mx + mn) / 2 if mx is not None and mn is not None else np.nan
                   for mx, mn in zip(d['relative_humidity_2m_max'], d['relative_humidity_2m_min'])],
            'wind': [w / 3.6 if w is not None else np.nan for w in d['wind_speed_10m_max']],
            'precip': d['precipitation_sum'],
        })
        return df, None
    except Exception as e:
        return None, str(e)

def fetch_forecast(lat, lon):
    """Fetch next 7 days from Open-Meteo Forecast API (includes today + 7 future days)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min,"
        f"wind_speed_10m_max,precipitation_sum&timezone=auto&forecast_days=7"
    )
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if 'daily' not in data:
            return None, f"No daily in forecast: {data}"
        d = data['daily']
        df = pd.DataFrame({
            'date': pd.to_datetime(d['time']),
            't2m': [(mx + mn) / 2 if mx is not None and mn is not None else np.nan
                    for mx, mn in zip(d['temperature_2m_max'], d['temperature_2m_min'])],
            'rh': [(mx + mn) / 2 if mx is not None and mn is not None else np.nan
                   for mx, mn in zip(d['relative_humidity_2m_max'], d['relative_humidity_2m_min'])],
            'wind': [w / 3.6 if w is not None else np.nan for w in d['wind_speed_10m_max']],
            'precip': d['precipitation_sum'],
        })
        return df, None
    except Exception as e:
        return None, str(e)

def compute_days_since_rain(precip_series, threshold=0.5):
    days = 0
    for p in reversed(precip_series):
        if p is None or pd.isna(p):
            continue
        if p >= threshold:
            break
        days += 1
    return days

def predict(lat, lon, override_temp=None, override_humidity=None, override_wind=None, override_rain=None, override_dsr=None, doy=None, mode='forest'):
    """Predict fire probability with REAL weather history + REAL forecast.
    
    mode='forest' → forest fire model (high-confidence, lọc cháy nông nghiệp)
    mode='any'    → any fire model (tất cả MODIS detection)
    """
    load_resources()
    
    lat_grid, lon_grid = snap_to_grid(lat, lon)
    
    if doy is None:
        doy = datetime.now().timetuple().tm_yday
    
    # Fetch archive (past 35 days) — for lags, rolling, days_since_rain
    archive_df, archive_err = fetch_archive(lat_grid, lon_grid, days=35)
    # Fetch forecast (today + 7 future days) — for "today's" weather
    forecast_df, forecast_err = fetch_forecast(lat_grid, lon_grid)
    
    weather_sources = []
    if archive_df is not None and len(archive_df) >= 14:
        weather_sources.append(f"Open-Meteo Archive ({len(archive_df)} days)")
    else:
        weather_sources.append(f"archive failed: {archive_err}")
    
    if forecast_df is not None and len(forecast_df) > 0:
        weather_sources.append(f"Open-Meteo Forecast ({len(forecast_df)} days)")
    else:
        weather_sources.append(f"forecast failed: {forecast_err}")
    
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
    
    # Determine "today's" weather
    # Priority: user override > forecast today > archive last day > climatology
    if override_temp is not None:
        t2m_today = override_temp
    elif forecast_df is not None and len(forecast_df) > 0:
        t2m_today = float(forecast_df.iloc[0]['t2m']) if not pd.isna(forecast_df.iloc[0]['t2m']) else t2m_clim
    elif archive_df is not None and len(archive_df) > 0:
        t2m_today = float(archive_df.iloc[-1]['t2m']) if not pd.isna(archive_df.iloc[-1]['t2m']) else t2m_clim
    else:
        t2m_today = t2m_clim
    
    if override_humidity is not None:
        rh_today = override_humidity
    elif forecast_df is not None and len(forecast_df) > 0:
        rh_today = float(forecast_df.iloc[0]['rh']) if not pd.isna(forecast_df.iloc[0]['rh']) else rh_clim
    elif archive_df is not None and len(archive_df) > 0:
        rh_today = float(archive_df.iloc[-1]['rh']) if not pd.isna(archive_df.iloc[-1]['rh']) else rh_clim
    else:
        rh_today = rh_clim
    
    if override_wind is not None:
        wind_today = override_wind
    elif forecast_df is not None and len(forecast_df) > 0:
        wind_today = float(forecast_df.iloc[0]['wind']) if not pd.isna(forecast_df.iloc[0]['wind']) else wind_clim
    elif archive_df is not None and len(archive_df) > 0:
        wind_today = float(archive_df.iloc[-1]['wind']) if not pd.isna(archive_df.iloc[-1]['wind']) else wind_clim
    else:
        wind_today = wind_clim
    
    if override_rain is not None:
        precip_today = override_rain
    elif forecast_df is not None and len(forecast_df) > 0:
        precip_today = float(forecast_df.iloc[0]['precip']) if not pd.isna(forecast_df.iloc[0]['precip']) else precip_clim
    elif archive_df is not None and len(archive_df) > 0:
        precip_today = float(archive_df.iloc[-1]['precip']) if not pd.isna(archive_df.iloc[-1]['precip']) else precip_clim
    else:
        precip_today = precip_clim
    
    # days_since_rain from archive (real history)
    if override_dsr is not None:
        days_since_rain = override_dsr
    elif archive_df is not None and len(archive_df) > 0:
        days_since_rain = compute_days_since_rain(archive_df['precip'].values)
    else:
        days_since_rain = 5
    
    # Build lags from archive (real past weather)
    if archive_df is not None and len(archive_df) >= 15:
        t2m_series = archive_df['t2m'].values
        rh_series = archive_df['rh'].values
        wind_series = archive_df['wind'].values
        precip_series = archive_df['precip'].values
        
        def get_lag(series, days_ago, clim_fallback):
            idx = -days_ago - 1
            if abs(idx) > len(series):
                return clim_fallback
            val = series[idx]
            if val is None or pd.isna(val):
                return clim_fallback
            return float(val)
        
        t2m_lag1 = get_lag(t2m_series, 1, t2m_clim)
        t2m_lag3 = get_lag(t2m_series, 3, t2m_clim)
        t2m_lag7 = get_lag(t2m_series, 7, t2m_clim)
        t2m_lag14 = get_lag(t2m_series, 14, t2m_clim)
        
        rh_lag1 = get_lag(rh_series, 1, rh_clim)
        rh_lag3 = get_lag(rh_series, 3, rh_clim)
        rh_lag7 = get_lag(rh_series, 7, rh_clim)
        rh_lag14 = get_lag(rh_series, 14, rh_clim)
        
        wind_lag1 = get_lag(wind_series, 1, wind_clim)
        wind_lag3 = get_lag(wind_series, 3, wind_clim)
        wind_lag7 = get_lag(wind_series, 7, wind_clim)
        wind_lag14 = get_lag(wind_series, 14, wind_clim)
        
        precip_lag1 = get_lag(precip_series, 1, precip_clim)
        precip_lag3 = get_lag(precip_series, 3, precip_clim)
        precip_lag7 = get_lag(precip_series, 7, precip_clim)
        precip_lag14 = get_lag(precip_series, 14, precip_clim)
        
        # Rolling stats from real archive
        t2m_roll7 = float(np.nanmean(t2m_series[-7:])) if len(t2m_series) >= 7 else t2m_clim
        t2m_roll14 = float(np.nanmean(t2m_series[-14:])) if len(t2m_series) >= 14 else t2m_clim
        rh_roll7 = float(np.nanmean(rh_series[-7:])) if len(rh_series) >= 7 else rh_clim
        rh_roll14 = float(np.nanmean(rh_series[-14:])) if len(rh_series) >= 14 else rh_clim
        precip_roll7_sum = float(np.nansum(precip_series[-7:])) if len(precip_series) >= 7 else precip_clim * 7
        precip_roll14_sum = float(np.nansum(precip_series[-14:])) if len(precip_series) >= 14 else precip_clim * 14
        precip_roll30_sum = float(np.nansum(precip_series[-30:])) if len(precip_series) >= 30 else precip_clim * 30
        
        # Diff = today - lag3 (real)
        t2m_diff3 = t2m_today - t2m_lag3
        rh_diff3 = rh_today - rh_lag3
        wind_diff3 = wind_today - wind_lag3
    else:
        # Fallback to climatology
        t2m_lag1 = t2m_lag3 = t2m_lag7 = t2m_lag14 = t2m_clim
        rh_lag1 = rh_lag3 = rh_lag7 = rh_lag14 = rh_clim
        wind_lag1 = wind_lag3 = wind_lag7 = wind_lag14 = wind_clim
        precip_lag1 = precip_lag3 = precip_lag7 = precip_lag14 = precip_clim
        t2m_roll7 = t2m_roll14 = t2m_clim
        rh_roll7 = rh_roll14 = rh_clim
        precip_roll7_sum = precip_clim * 7
        precip_roll14_sum = precip_clim * 14
        precip_roll30_sum = precip_clim * 30
        t2m_diff3 = t2m_today - t2m_clim
        rh_diff3 = rh_today - rh_clim
        wind_diff3 = wind_today - wind_clim
    
    # Derived
    heat_index = (t2m_today - 20) * (50 - rh_today) / 50.0
    wind_chill = (t2m_today - 15) * wind_today / 30.0
    
    # Anomalies (current - climatology for that doy)
    t2m_anom_doy = t2m_today - t2m_clim
    rh_anom_doy = rh_today - rh_clim
    precip_anom_doy = precip_today - precip_clim
    wind_anom_doy = wind_today - wind_clim
    
    # Rolling anomalies
    t2m_anom_roll7 = t2m_roll7 - t2m_clim
    rh_anom_roll7 = rh_roll7 - rh_clim
    precip_anom_roll7 = precip_roll7_sum - precip_clim * 7
    
    # Cell-static
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
    
    feature_names = _model.feature_names  # both models share same 61 features
    X = pd.DataFrame([features], columns=feature_names)
    dmat = gbm.DMatrix(X)
    
    # Select model based on mode
    if mode == 'any':
        active_model = _model_any
        active_iso = _iso_any
        model_label = 'any fire'
    else:
        active_model = _model
        active_iso = _iso
        model_label = 'forest fire (high-conf)'
    
    raw_p = float(active_model.predict(dmat)[0])
    p_calibrated = float(active_iso.predict([raw_p])[0])
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
    
    # Also return tomorrow's forecasted weather (for UI display)
    tomorrow_forecast = None
    if forecast_df is not None and len(forecast_df) >= 2:
        tom = forecast_df.iloc[1]
        tomorrow_forecast = {
            'date': str(tom['date'].date()) if not pd.isna(tom['date']) else None,
            't2m': float(tom['t2m']) if not pd.isna(tom['t2m']) else None,
            'rh': float(tom['rh']) if not pd.isna(tom['rh']) else None,
            'wind': float(tom['wind']) if not pd.isna(tom['wind']) else None,
            'precip': float(tom['precip']) if not pd.isna(tom['precip']) else None,
        }
    
    return {
        'probability': p_calibrated,
        'raw_probability': raw_p,
        'calibrated_probability': p_calibrated,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_score': int(p_calibrated * 100),
        'model_version': f'v9.0 {model_label} (61 feats) — real forecast',
        'mode': mode,
        'n_features': len(feature_names),
        'cell_lat': lat_grid,
        'cell_lon': lon_grid,
        'doy': doy,
        'is_cold_start': lat_grid < 14,
        'weather_sources': weather_sources,
        'current_weather': {
            't2m': round(t2m_today, 2),
            'rh': round(rh_today, 2),
            'wind': round(wind_today, 2),
            'precip': round(precip_today, 2),
            'days_since_rain': days_since_rain,
        },
        'tomorrow_forecast': tomorrow_forecast,
        'lags_real': archive_df is not None and len(archive_df) >= 14,
        'forecast_real': forecast_df is not None,
    }

def predict_batch(inputs):
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
                inp.get('doy'),
                inp.get('mode', 'forest')
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
            input_data.get('doy'),
            input_data.get('mode', 'forest')
        )
        print(json.dumps(result))
