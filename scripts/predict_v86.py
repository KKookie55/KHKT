#!/usr/bin/env python3
"""
Fire prediction CLI — REAL MODEL (v8.6).

Loads the trained XGBoost + Isotonic calibrator.
Performs full feature engineering (35 features) from 5 raw inputs.
Supports batch mode for multiple inputs in one subprocess call.
"""
import sys
import os
import json
import pickle
import numpy as np
import xgboost as xgb

MODEL_DIR = '/home/z/my-project/model/wildfire_v86_model'
META_PATH = os.path.join(MODEL_DIR, 'model_meta.json')
XGB_PATH  = os.path.join(MODEL_DIR, 'xgb_modis_iid.json')
ISO_PATH  = os.path.join(MODEL_DIR, 'iso_calibrator.pkl')
IMP_PATH  = os.path.join(MODEL_DIR, 'imputer.pkl')

_meta = None
_model = None
_iso = None
_imp = None

def load_model():
    global _meta, _model, _iso, _imp
    if _meta is None:
        with open(META_PATH) as f:
            _meta = json.load(f)
        _model = xgb.XGBClassifier()
        _model.load_model(XGB_PATH)
        with open(ISO_PATH, 'rb') as f:
            _iso = pickle.load(f)
        with open(IMP_PATH, 'rb') as f:
            _imp = pickle.load(f)
    return _meta, _model, _iso, _imp


def engineer_features(input_data):
    """Build 35-feature vector from 5 raw inputs."""
    t2m = float(input_data['temp'])
    rh  = float(input_data['humidity'])
    wind = float(input_data['wind'])
    precip = float(input_data['rain'])
    days_since_rain = float(input_data['daysSinceRain'])
    doy = int(input_data.get('doy', 60))

    # Lag features (steady-state assumption)
    t2m_lag1 = t2m_lag3 = t2m_lag7 = t2m_lag14 = t2m
    rh_lag1 = rh_lag3 = rh_lag7 = rh_lag14 = rh
    wind_lag1 = wind_lag3 = wind_lag7 = wind_lag14 = wind
    precip_lag1 = precip_lag3 = precip_lag7 = precip_lag14 = precip

    # Rolling means
    t2m_roll7 = t2m_roll14 = t2m
    rh_roll7 = rh_roll14 = rh

    # Rolling precip sums
    precip_roll7_sum = precip * 7
    precip_roll14_sum = precip * 14
    precip_roll30_sum = precip * 30

    # Diffs (steady state = 0)
    t2m_diff3 = 0.0
    rh_diff3 = 0.0
    wind_diff3 = 0.0

    # Derived composites
    fwi_compound = max(0, min(1.5, (
        (t2m - 20) * 0.03 + (30 - rh) * 0.015 + wind * 0.04 - precip * 0.05
    )))
    heat_index = max(-10, min(40, (t2m - 35 + (rh - 50) * 0.1)))
    wind_chill = max(-20, min(30, t2m - wind * 0.5))

    features = [
        t2m, rh, wind, precip, doy,
        t2m_lag1, t2m_lag3, t2m_lag7, t2m_lag14,
        rh_lag1, rh_lag3, rh_lag7, rh_lag14,
        wind_lag1, wind_lag3, wind_lag7, wind_lag14,
        precip_lag1, precip_lag3, precip_lag7, precip_lag14,
        t2m_roll7, t2m_roll14, rh_roll7, rh_roll14,
        precip_roll7_sum, precip_roll14_sum, precip_roll30_sum,
        days_since_rain, t2m_diff3, rh_diff3, wind_diff3,
        fwi_compound, heat_index, wind_chill,
    ]
    return features


def predict(input_data):
    meta, model, iso, imp = load_model()
    features = meta['features']
    feat_vec = engineer_features(input_data)
    x = np.array([feat_vec], dtype=np.float32)

    # Normalize drought_idx (feature 27)
    di_idx = meta['drought_idx']
    x[0, di_idx] = (x[0, di_idx] - meta['drought_mean']) / meta['drought_std']

    # Apply imputer
    x = imp.transform(x)

    # XGBoost prediction
    p_raw = float(model.predict_proba(x)[0, 1])

    # Isotonic calibration
    p_calibrated = float(iso.predict([p_raw])[0])

    # Region penalty
    region = input_data.get('region', 'central')
    region_penalty = {'north': 1.0, 'central': 0.95, 'south': 0.65}
    penalty = region_penalty.get(region, 0.9)
    final_prob = 0.5 + (p_calibrated - 0.5) * penalty
    final_prob = max(0.01, min(0.99, final_prob))

    # Risk level
    if final_prob >= 0.7:
        risk_level = 'Rất cao'
        risk_color = '#991b1b'
    elif final_prob >= 0.5:
        risk_level = 'Cao'
        risk_color = '#dc2626'
    elif final_prob >= 0.3:
        risk_level = 'Trung bình'
        risk_color = '#f59e0b'
    else:
        risk_level = 'Thấp'
        risk_color = '#16a34a'

    # OOD check
    out_of_domain = (
        input_data['temp'] < 10 or input_data['temp'] > 38 or
        input_data['humidity'] < 40 or input_data['humidity'] > 100 or
        input_data['wind'] > 8 or
        input_data['daysSinceRain'] > 15
    )

    if out_of_domain:
        confidence = 'Thấp'
    elif final_prob > 0.85 or final_prob < 0.05:
        confidence = 'Cao'
    else:
        confidence = 'Trung bình'

    warning = None
    if out_of_domain:
        warning = 'Đầu vào ngoài miền huấn luyện, dự báo có thể không đáng tin'
    elif region == 'south':
        warning = 'Vùng Nam: model suy giảm AUROC (0,946 → 0,584 trên VIIRS)'

    return {
        'probability': round(final_prob, 4),
        'raw_probability': round(p_raw, 4),
        'calibrated_probability': round(p_calibrated, 4),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_score': round(final_prob * 100, 1),
        'confidence': confidence,
        'out_of_domain': out_of_domain,
        'warning': warning,
        'model_version': 'v8.6_canonical_model',
        'model_auroc_iid': meta['performance']['AUROC_iso'],
        'model_ece_iid': meta['performance']['ECE_iso'],
        'n_features': meta['n_features'],
        'region_penalty': penalty,
    }


def main():
    try:
        input_text = sys.stdin.read()
        input_data = json.loads(input_text)

        # Batch mode
        if isinstance(input_data, dict) and 'inputs' in input_data and isinstance(input_data['inputs'], list):
            inputs = input_data['inputs']
            results = []
            for inp in inputs:
                required = ['temp', 'humidity', 'wind', 'rain', 'daysSinceRain']
                missing = [f for f in required if f not in inp]
                if missing:
                    results.append({'error': f'Missing: {missing}', 'probability': 0})
                    continue
                try:
                    results.append(predict(inp))
                except Exception as e:
                    results.append({'error': str(e), 'probability': 0})
            print(json.dumps({
                'results': results,
                'count': len(results),
                'model_version': 'v8.6_canonical_model',
                'model_auroc_iid': _meta['performance']['AUROC_iso'] if _meta else 0.9463,
            }, ensure_ascii=False))
            return

        # Single mode
        required = ['temp', 'humidity', 'wind', 'rain', 'daysSinceRain']
        missing = [f for f in required if f not in input_data]
        if missing:
            print(json.dumps({'error': f'Missing: {missing}'}))
            sys.exit(1)
        result = predict(input_data)
        print(json.dumps(result, ensure_ascii=False))

    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON: {e}'}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': f'Prediction failed: {str(e)}'}))
        sys.exit(1)


if __name__ == '__main__':
    main()
