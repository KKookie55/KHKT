"""Test: What does v9.0 predict for October wet-season weather?
Expected: low probabilities (~0.001-0.05) since humidity high, rain present.
"""
import json, sys, subprocess, os

# Wet season (October) typical weather for Vietnam
# North (Hanoi): cool + wet, 24°C, 85%, rain 8mm, DSR=1
# Central (Huế): wet, 27°C, 88%, rain 15mm, DSR=0
# South (HCM): transitional, 28°C, 80%, rain 3mm, DSR=2

tests = [
    {"name": "Hanoi Oct (wet)", "lat": 21.0, "lon": 105.8, "temp": 24, "humidity": 85, "wind": 3, "rain": 8, "daysSinceRain": 1, "mode": "forest"},
    {"name": "Hue Oct (wet)", "lat": 16.5, "lon": 107.5, "temp": 27, "humidity": 88, "wind": 2.5, "rain": 15, "daysSinceRain": 0, "mode": "forest"},
    {"name": "HCM Oct (transitional)", "lat": 10.8, "lon": 106.7, "temp": 28, "humidity": 80, "wind": 3, "rain": 3, "daysSinceRain": 2, "mode": "forest"},
    # Defaults that web uses if weather API fails:
    {"name": "DEFAULTS (web initial)", "lat": 16.5, "lon": 106.5, "temp": 28, "humidity": 65, "wind": 4, "rain": 0, "daysSinceRain": 8, "mode": "forest"},
]

payload = {"inputs": [t for t in tests if 'name' not in t] if False else
           [{"lat": t["lat"], "lon": t["lon"], "temp": t["temp"], "humidity": t["humidity"],
             "wind": t["wind"], "rain": t["rain"], "daysSinceRain": t["daysSinceRain"], "mode": t["mode"]} for t in tests]}

# Call predict_v90.py directly
predictor_path = "/home/z/my-project/scripts/predict_v90.py"
python_bin = "/home/z/.venv/bin/python3" if os.path.exists("/home/z/.venv/bin/python3") else "python3"

proc = subprocess.run(
    [python_bin, predictor_path],
    input=json.dumps(payload), capture_output=True, text=True, timeout=30,
    cwd="/home/z/my-project"
)

print("STDOUT:")
try:
    out = json.loads(proc.stdout)
    results = out.get("results", [])
    for t, r in zip(tests, results):
        p = r.get("probability", "?")
        print(f"  {t['name']:<32s} → p = {p:.4f}")
    print(f"\n  Engine: {out.get('engine', 'unknown')}")
    print(f"  Model version: {out.get('model_version', 'unknown')}")
except Exception as e:
    print(f"PARSE ERROR: {e}")
    print(proc.stdout[:1000])

if proc.stderr:
    print(f"\nSTDERR (first 500):", proc.stderr[:500])
