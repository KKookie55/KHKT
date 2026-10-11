# v8.6 Wildfire Model Package

## Files

| File | Description |
|------|-------------|
| `xgb_modis_iid.json` | XGBoost model (35 features → fire probability) |
| `iso_calibrator.pkl` | Isotonic regression calibrator |
| `imputer.pkl` | Median imputer |
| `model_meta.json` | Features, normalization stats, hyperparameters |
| `predict_example.py` | End-to-end inference example |
| `sample_input.csv` | 3 example weather rows for testing |

## Performance (test set, IID, MODIS labels)

| Metric | Raw | After Iso |
|--------|-----|-----------|
| AUROC | 0.9466 | 0.9463 |
| ECE | 0.0549 | 0.0090 |

## How to use

```bash
pip install xgboost scikit-learn numpy
python predict_example.py
```

Or load in your own code:

```python
import json, pickle, numpy as np, xgboost as xgb
with open("model_meta.json") as f: meta = json.load(f)
m = xgb.XGBClassifier(); m.load_model("xgb_modis_iid.json")
with open("iso_calibrator.pkl", "rb") as f: iso = pickle.load(f)
with open("imputer.pkl", "rb") as f: imp = pickle.load(f)

# Predict for a single feature row
x = np.array([[28.5, 65, 4.5, ...]])  # shape (1, 35), see meta["features"] for order
di_idx = meta["drought_idx"]
x[0, di_idx] = (x[0, di_idx] - meta["drought_mean"]) / meta["drought_std"]
x = imp.transform(x)
p_raw = m.predict_proba(x)[0, 1]
p_calibrated = float(iso.predict([p_raw])[0])
print(f"Fire risk: {p_calibrated:.4f}")
```

## Important caveats

1. **Model trained on MODIS labels** — VIIRS cross-validation showed MODIS evaluation overestimates by 0.31 AUROC on spatial OOD. Use with caution outside training region.
2. **No spatial features** — model uses only weather features. Adding raw lat/lon recovers +0.23 AUROC on South, but this is geography memorization.
3. **Reliability-aware use**: when forecasting for a region outside training distribution, treat model output as LOW reliability.
4. **Forecast horizon**: trained for t+1 (next-day forecast). For t+3, t+7, performance degrades ~0.02 AUROC.
