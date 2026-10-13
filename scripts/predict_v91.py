#!/usr/bin/env python3
"""v9.1 Production predictor — DISTRICT-LEVEL.

Mirrors predict_v90.py structure but operates on district_id instead of lat/lon cell.

Input (stdin JSON):
  Single: { "district_id": 0, "temp": 27, "humidity": 85, "wind": 3, "rain": 8, "daysSinceRain": 1, "mode": "forest" }
  Batch:  { "inputs": [{...}, ...] }

Returns: probability + risk_level + risk_color (same shape as v9.0).

For batch mode (MapView), uses CLIMATOLOGY-based lags (fast).
For single-cell mode (popup), could fetch Open-Meteo archive (slow but accurate).
"""
import os, sys, json, pickle, joblib
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as gbm
from sklearn.isotonic import IsotonicRegression

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
MODEL_DIR = os.path.join(_PROJECT_ROOT, 'public', 'model', 'v91')
MODEL_DIR_V92 = os.path.join(_PROJECT_ROOT, 'public', 'model', 'v92')

# Use v9.2 if available, fall back to v9.1
USE_DIR = MODEL_DIR_V92 if os.path.exists(os.path.join(MODEL_DIR_V92, 'model_forest_fire.json')) else MODEL_DIR
MODEL_VERSION_LABEL = 'v9.2' if USE_DIR == MODEL_DIR_V92 else 'v9.1'

MODEL_PATH = os.path.join(USE_DIR, 'model_forest_fire.json')
MODEL_ANY_PATH = os.path.join(USE_DIR, 'model_any_fire.json')
ISO_PATH = os.path.join(USE_DIR, 'iso_forest_fire.pkl')
ISO_ANY_PATH = os.path.join(USE_DIR, 'iso_any_fire.pkl')
FEATURES_PATH = os.path.join(USE_DIR, 'features_forest_fire.json')
FEATURES_ANY_PATH = os.path.join(USE_DIR, 'features_any_fire.json')
DISTRICT_FEATURES_PATH = os.path.join(MODEL_DIR, 'district_features.parquet')
DISTRICT_CLIM_PATH = os.path.join(MODEL_DIR, 'district_climatology.parquet')
DISTRICT_NDVI_PATH = os.path.join(MODEL_DIR, 'district_ndvi.parquet')
DISTRICT_CELL_FIRE_CLIM_PATH = os.path.join(MODEL_DIR, 'district_cell_fire_rate_clim.parquet')  # NEW v9.2

_model = None
_model_any = None
_iso = None
_iso_any = None
_feats_list = None
_feats_any_list = None
_district_static = None
_district_clim = None
_district_ndvi = None  # NEW
_district_cell_fire_clim = None  # NEW v9.2


def load_resources():
    global _model, _model_any, _iso, _iso_any, _feats_list, _feats_any_list, _district_static, _district_clim, _district_ndvi, _district_cell_fire_clim
    if _model is None:
        _model = gbm.Booster()
        _model.load_model(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _feats_list = json.load(f)["features"]
    if _model_any is None:
        _model_any = gbm.Booster()
        _model_any.load_model(MODEL_ANY_PATH)
        with open(FEATURES_ANY_PATH) as f:
            _feats_any_list = json.load(f)["features"]
    if _iso is None:
        _iso = joblib.load(ISO_PATH)
    if _iso_any is None:
        _iso_any = joblib.load(ISO_ANY_PATH)
    if _district_static is None:
        _district_static = pd.read_parquet(DISTRICT_FEATURES_PATH)
    if _district_clim is None:
        _district_clim = pd.read_parquet(DISTRICT_CLIM_PATH)
    if _district_ndvi is None:
        try:
            _district_ndvi = pd.read_parquet(DISTRICT_NDVI_PATH)
        except Exception:
            _district_ndvi = None
    if _district_cell_fire_clim is None:
        try:
            _district_cell_fire_clim = pd.read_parquet(DISTRICT_CELL_FIRE_CLIM_PATH)
        except Exception:
            _district_cell_fire_clim = None


def get_cell_fire_clim(district_id, doy):
    """Get per-(district, doy) climatology of cell_fire_rate_exp + cell_fire_count_30d + is_dry."""
    if _district_cell_fire_clim is None:
        return {"cell_fire_rate_exp": 0.05, "cell_fire_count_30d": 1.5, "is_dry": 0, "cell_fire_rate_dry_exp": 0.05}
    rows = _district_cell_fire_clim[(_district_cell_fire_clim["district_id"] == district_id) &
                                     (_district_cell_fire_clim["doy"] == doy)]
    if len(rows) == 0:
        d_rows = _district_cell_fire_clim[_district_cell_fire_clim["district_id"] == district_id]
        if len(d_rows) == 0:
            return {"cell_fire_rate_exp": 0.05, "cell_fire_count_30d": 1.5, "is_dry": 0, "cell_fire_rate_dry_exp": 0.05}
        d_rows = d_rows.copy()
        d_rows["d"] = (d_rows["doy"] - doy).abs()
        rows = d_rows.iloc[[d_rows["d"].idxmin()]]
    return {
        "cell_fire_rate_exp": float(rows.iloc[0]["cell_fire_rate_exp"]),
        "cell_fire_count_30d": float(rows.iloc[0]["cell_fire_count_30d"]),
        "is_dry": int(rows.iloc[0]["is_dry"]),
        "cell_fire_rate_dry_exp": float(rows.iloc[0]["cell_fire_rate_dry_exp"]),
    }


def get_ndvi(district_id: int, doy: int):
    """Get NDVI for (district, doy) from pre-built NDVI table."""
    if _district_ndvi is None:
        return {"ndvi": 0.4, "ndvi_anom": 0.0, "ndvi_lag7": 0.4}
    rows = _district_ndvi[(_district_ndvi["district_id"] == district_id) & (_district_ndvi["doy"] == doy)]
    if len(rows) == 0:
        # Fallback to nearest doy
        d_rows = _district_ndvi[_district_ndvi["district_id"] == district_id]
        if len(d_rows) == 0:
            return {"ndvi": 0.4, "ndvi_anom": 0.0, "ndvi_lag7": 0.4}
        d_rows = d_rows.copy()
        d_rows["d"] = (d_rows["doy"] - doy).abs()
        rows = d_rows.iloc[[d_rows["d"].idxmin()]]
    return {
        "ndvi": float(rows.iloc[0]["ndvi"]),
        "ndvi_anom": float(rows.iloc[0].get("ndvi_anom", 0.0)),
        "ndvi_lag7": float(rows.iloc[0].get("ndvi_lag7", rows.iloc[0]["ndvi"])),
    }


def get_district_static(district_id: int):
    """Get static features (landcover, geo, centroid, NDVI, region) for a district."""
    row = _district_static[_district_static["district_id"] == district_id]
    if len(row) == 0:
        return None
    return row.iloc[0].to_dict()


def get_district_clim(district_id: int, doy: int):
    """Get climatology for (district, doy). Falls back to nearest doy, then to global avg."""
    rows = _district_clim[(_district_clim["district_id"] == district_id) & (_district_clim["doy"] == doy)]
    if len(rows) == 0:
        # Find nearest doy for this district
        d_rows = _district_clim[_district_clim["district_id"] == district_id]
        if len(d_rows) == 0:
            # District has no climatology — use global avg for that doy
            g_rows = _district_clim[_district_clim["doy"] == doy]
            if len(g_rows) == 0:
                # Use overall mean as last resort
                return {
                    "t2m_clim": 25.0, "rh_clim": 75.0,
                    "precip_clim": 3.0, "wind_clim": 3.0,
                }
            return {
                "t2m_clim": float(g_rows["t2m_clim"].mean()),
                "rh_clim": float(g_rows["rh_clim"].mean()),
                "precip_clim": float(g_rows["precip_clim"].mean()),
                "wind_clim": float(g_rows["wind_clim"].mean()),
            }
        d_rows = d_rows.copy()
        d_rows["d"] = (d_rows["doy"] - doy).abs()
        rows = d_rows.iloc[[d_rows["d"].idxmin()]]
    return rows.iloc[0][["t2m_clim", "rh_clim", "precip_clim", "wind_clim"]].to_dict()


def build_features(district_id, doy, temp, humidity, wind, rain, days_since_rain):
    """Build feature vector for a district-day."""
    static = get_district_static(district_id)
    if static is None:
        return None, "No static features for district_id=" + str(district_id)
    clim = get_district_clim(district_id, doy)
    if clim is None:
        return None, "No climatology for district_id=" + str(district_id)
    ndvi_data = get_ndvi(district_id, doy)  # NEW
    cell_fire_clim = get_cell_fire_clim(district_id, doy)  # NEW v9.2 — seasonal cell_fire_rate

    t2m_clim = float(clim["t2m_clim"])
    rh_clim = float(clim["rh_clim"])
    precip_clim = float(clim["precip_clim"])
    wind_clim = float(clim["wind_clim"])

    # Current weather (override or climatology)
    t2m_today = float(temp) if temp is not None else t2m_clim
    rh_today = float(humidity) if humidity is not None else rh_clim
    wind_today = float(wind) if wind is not None else wind_clim
    precip_today = float(rain) if rain is not None else precip_clim
    dsr = float(days_since_rain) if days_since_rain is not None else 5.0

    # FAST: use CLIMATOLOGY for lags by default, BUT if user provides override (sliders),
    # propagate today's weather to lags too (so "dry" weather scenarios produce realistic high p)
    # Heuristic: if user override is far from climatology, use override for lags (with small noise)
    override_threshold = 1.5  # if |override - clim| > 1.5 (°C / % / m/s / mm), use override for lags
    use_override_for_lags = (
        abs(t2m_today - t2m_clim) > override_threshold or
        abs(rh_today - rh_clim) > override_threshold or
        abs(wind_today - wind_clim) > override_threshold or
        abs(precip_today - precip_clim) > override_threshold
    )
    if use_override_for_lags:
        # Use today's override for lags (model evaluates "if this weather had been persistent")
        t2m_lag1 = t2m_lag3 = t2m_lag7 = t2m_lag14 = t2m_today
        rh_lag1 = rh_lag3 = rh_lag7 = rh_lag14 = rh_today
        wind_lag1 = wind_lag3 = wind_lag7 = wind_lag14 = wind_today
        precip_lag1 = precip_lag3 = precip_lag7 = precip_lag14 = precip_today
        t2m_roll7 = t2m_roll14 = t2m_today
        rh_roll7 = rh_roll14 = rh_today
        precip_roll7_sum = precip_today * 7
        precip_roll14_sum = precip_today * 14
        precip_roll30_sum = precip_today * 30
    else:
        # Use climatology for lags (steady-state assumption)
        t2m_lag1 = t2m_lag3 = t2m_lag7 = t2m_lag14 = t2m_clim
        rh_lag1 = rh_lag3 = rh_lag7 = rh_lag14 = rh_clim
        wind_lag1 = wind_lag3 = wind_lag7 = wind_lag14 = wind_clim
        precip_lag1 = precip_lag3 = precip_lag7 = precip_lag14 = precip_clim
        t2m_roll7 = t2m_roll14 = t2m_clim
        rh_roll7 = rh_roll14 = rh_clim
        precip_roll7_sum = precip_clim * 7
        precip_roll14_sum = precip_clim * 14
        precip_roll30_sum = precip_clim * 30

    # Anomalies
    t2m_anom_doy = t2m_today - t2m_clim
    rh_anom_doy = rh_today - rh_clim
    precip_anom_doy = precip_today - precip_clim
    wind_anom_doy = wind_today - wind_clim
    t2m_anom_roll7 = t2m_anom_doy
    rh_anom_roll7 = rh_anom_doy
    precip_anom_roll7 = precip_today * 7 - precip_clim * 7
    dry_anomaly_composite = (t2m_anom_doy * 0.5) + (-rh_anom_doy * 0.3) + (-precip_anom_doy * 0.2)

    # Diffs
    t2m_diff3 = t2m_anom_doy
    rh_diff3 = rh_anom_doy
    wind_diff3 = wind_anom_doy

    # Derived
    fwi_compound = max(0, min(1.5, (t2m_today - 20) * 0.03 + (30 - rh_today) * 0.015 + wind_today * 0.04 - precip_today * 0.05))
    heat_index = max(-10, min(40, (t2m_today - 35 + (rh_today - 50) * 0.1)))
    wind_chill = max(-20, min(30, t2m_today - wind_today * 0.5))

    # FWI simplified (placeholder values — same as v9.0)
    fwi_val = max(0, min(50, (t2m_today - 15) * 0.5 + (30 - rh_today) * 0.3 + wind_today * 2 - precip_today * 2))
    fwi_ffmc = max(0, min(100, 50 + (t2m_today - 25) * 2 - (rh_today - 50) * 0.5))
    fwi_dmc = max(0, min(100, 30 + (t2m_today - 25) * 1.5 - precip_today * 1.5))
    fwi_dc = max(0, min(800, 200 + (t2m_today - 25) * 10 - precip_today * 5))
    fwi_isi = max(0, min(50, 5 + wind_today * 3 - rh_today * 0.05))
    fwi_bui = max(0, min(200, 50 + (t2m_today - 25) * 2))

    # NEW v9.2: interaction features
    wind_temp = wind_today * t2m_today
    humidity_temp = rh_today * t2m_today
    dry_wind = dry_anomaly_composite * wind_today
    heat_drought = t2m_anom_doy * dsr
    # NDVI anomaly: current NDVI - global mean (fallback; would be per-district mean with real climatology)
    ndvi_anom = ndvi_data['ndvi'] - 0.499

    features = {
        't2m': t2m_today, 'rh': rh_today, 'wind': wind_today, 'precip': precip_today, 'doy': int(doy),
        't2m_lag1': t2m_lag1, 't2m_lag3': t2m_lag3, 't2m_lag7': t2m_lag7, 't2m_lag14': t2m_lag14,
        'rh_lag1': rh_lag1, 'rh_lag3': rh_lag3, 'rh_lag7': rh_lag7, 'rh_lag14': rh_lag14,
        'wind_lag1': wind_lag1, 'wind_lag3': wind_lag3, 'wind_lag7': wind_lag7, 'wind_lag14': wind_lag14,
        'precip_lag1': precip_lag1, 'precip_lag3': precip_lag3, 'precip_lag7': precip_lag7, 'precip_lag14': precip_lag14,
        't2m_roll7': t2m_roll7, 't2m_roll14': t2m_roll14, 'rh_roll7': rh_roll7, 'rh_roll14': rh_roll14,
        'precip_roll7_sum': precip_roll7_sum, 'precip_roll14_sum': precip_roll14_sum, 'precip_roll30_sum': precip_roll30_sum,
        'days_since_rain': dsr,
        't2m_diff3': t2m_diff3, 'rh_diff3': rh_diff3, 'wind_diff3': wind_diff3,
        'fwi_compound': fwi_compound, 'heat_index': heat_index, 'wind_chill': wind_chill,
        't2m_anom_doy': t2m_anom_doy, 'rh_anom_doy': rh_anom_doy,
        'precip_anom_doy': precip_anom_doy, 'wind_anom_doy': wind_anom_doy,
        't2m_anom_roll7': t2m_anom_roll7, 'rh_anom_roll7': rh_anom_roll7, 'precip_anom_roll7': precip_anom_roll7,
        'dry_anomaly_composite': dry_anomaly_composite,
        'fwi': fwi_val, 'fwi_ffmc': fwi_ffmc, 'fwi_dmc': fwi_dmc, 'fwi_dc': fwi_dc, 'fwi_isi': fwi_isi, 'fwi_bui': fwi_bui,
        # Static from district lookup
        'forest_prop': static['forest_prop'],
        'shrubland_prop': static['shrubland_prop'],
        'grassland_prop': static['grassland_prop'],
        'agriculture_prop': static['agriculture_prop'],
        'urban_prop': static['urban_prop'],
        'bare_prop': static['bare_prop'],
        'water_prop': static['water_prop'],
        'elevation_mean': static['elevation_mean'],
        'elevation_std': static['elevation_std'],
        'slope_mean': static['slope_mean'],
        'koppen_Cwa': static['koppen_Cwa'],
        'koppen_Am': static['koppen_Am'],
        'koppen_Aw': static['koppen_Aw'],
        'latitude_band': static['latitude_band'],
        'coastal_distance': static['coastal_distance'],
        'population_density_proxy': static['population_density_proxy'],
        # NDVI from realistic seasonal model
        'ndvi': ndvi_data['ndvi'],
        'ndvi_lag7': ndvi_data['ndvi_lag7'],
        # NEW v9.2: interaction features
        'wind_temp': wind_temp,
        'humidity_temp': humidity_temp,
        'dry_wind': dry_wind,
        'heat_drought': heat_drought,
        'ndvi_anom': ndvi_anom,
        'cell_fire_rate_exp': cell_fire_clim['cell_fire_rate_exp'],  # NEW v9.2: seasonal climatology
        'cell_fire_count_30d': cell_fire_clim['cell_fire_count_30d'],  # NEW v9.2
        'is_dry': cell_fire_clim['is_dry'],  # NEW v9.2
        'cell_fire_rate_dry_exp': cell_fire_clim['cell_fire_rate_dry_exp'],  # NEW v9.2
        'centroid_lat': static['centroid_lat'],
        'centroid_lon': static['centroid_lon'],
    }
    return features, None


def predict_batch(inputs):
    """Predict for multiple districts. FAST path using climatology for lags."""
    load_resources()
    results = []
    for inp in inputs:
        try:
            district_id = int(inp.get('district_id', 0))
            mode = inp.get('mode', 'forest')
            doy = inp.get('doy')
            if doy is None:
                doy = datetime.now().timetuple().tm_yday

            features, err = build_features(
                district_id, doy,
                inp.get('temp'), inp.get('humidity'), inp.get('wind'),
                inp.get('rain'), inp.get('daysSinceRain')
            )
            if err:
                results.append({'error': err, 'district_id': district_id})
                continue

            # Select model
            if mode == 'any':
                active_model = _model_any
                active_iso = _iso_any
                feats = _feats_any_list
                model_label = 'any fire'
            else:
                active_model = _model
                active_iso = _iso
                feats = _feats_list
                model_label = 'forest fire'

            # Build feature array in correct order
            X = pd.DataFrame([features], columns=feats)
            dmat = gbm.DMatrix(X)
            raw_p = float(active_model.predict(dmat)[0])
            p_calibrated = float(active_iso.predict([raw_p])[0])
            p_calibrated = max(0.001, min(0.999, p_calibrated))

            # Risk levels (same scale as v9.0)
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

            results.append({
                'district_id': district_id,
                'probability': p_calibrated,
                'raw_probability': raw_p,
                'calibrated_probability': p_calibrated,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'risk_score': int(p_calibrated * 100),
                'model_version': f'{MODEL_VERSION_LABEL} {model_label} (78 feats incl. interactions, district-level)',
                'mode': mode,
                'n_features': len(feats),
                'doy': int(doy),
                'is_cold_start': bool(features['centroid_lat'] < 14),
                'district_name': str(_district_static[_district_static['district_id'] == district_id].iloc[0]['district_name']) if district_id in _district_static['district_id'].values else '?',
                'province': str(_district_static[_district_static['district_id'] == district_id].iloc[0]['province']) if district_id in _district_static['district_id'].values else '?',
            })
        except Exception as e:
            results.append({'error': str(e), 'district_id': inp.get('district_id')})
    return results


if __name__ == '__main__':
    input_data = json.loads(sys.stdin.read())
    if 'inputs' in input_data:
        results = predict_batch(input_data['inputs'])
        print(json.dumps({'results': results, 'model_version': MODEL_VERSION_LABEL, 'count': len(results)}))
    else:
        # Single prediction
        load_resources()
        feats, err = build_features(
            int(input_data.get('district_id', 0)),
            input_data.get('doy') or datetime.now().timetuple().tm_yday,
            input_data.get('temp'), input_data.get('humidity'),
            input_data.get('wind'), input_data.get('rain'),
            input_data.get('daysSinceRain'),
        )
        if err:
            print(json.dumps({'error': err}))
        else:
            mode = input_data.get('mode', 'forest')
            if mode == 'any':
                active_model = _model_any
                active_iso = _iso_any
                feats_list = _feats_any_list
            else:
                active_model = _model
                active_iso = _iso
                feats_list = _feats_list
            X = pd.DataFrame([feats], columns=feats_list)
            dmat = gbm.DMatrix(X)
            raw_p = float(active_model.predict(dmat)[0])
            p_cal = float(active_iso.predict([raw_p])[0])
            p_cal = max(0.001, min(0.999, p_cal))
            print(json.dumps({
                'probability': p_cal,
                'raw_probability': raw_p,
                'model_version': f'{MODEL_VERSION_LABEL} {"any" if mode=="any" else "forest"}',
                'mode': mode,
            }))
