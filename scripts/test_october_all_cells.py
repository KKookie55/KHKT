"""Test v9.0 with October wet-season weather for the actual Vietnam grid (176 cells).
Uses EXACT same polygon as src/lib/data.ts.
"""
import json, subprocess, os, sys

# EXACT polygon from src/lib/data.ts (lat, lon pairs)
POLYGON = [
    (22.85, 103.55), (22.55, 103.75), (22.10, 103.55), (21.55, 103.50),
    (21.10, 103.55), (20.65, 104.05), (20.10, 104.05), (19.55, 104.55),
    (19.05, 104.10), (18.55, 104.10), (18.05, 104.55), (17.55, 105.55),
    (17.05, 106.05), (16.55, 106.55), (16.05, 107.05), (15.55, 107.45),
    (15.05, 107.50), (14.55, 107.10), (14.05, 107.05), (13.55, 106.05),
    (13.05, 105.55), (12.05, 105.05), (11.05, 105.05), (10.55, 104.95),
    (10.55, 104.65), (10.45, 104.55), (10.30, 104.40), (10.05, 104.55),
    (9.55, 104.75), (9.05, 104.75), (8.65, 104.65), (8.55, 104.85), (8.95, 105.05),
    (9.35, 105.35), (9.85, 105.55),
    (10.15, 105.85), (10.45, 106.45), (10.75, 106.85), (10.85, 107.30),
    (10.95, 107.55), (10.85, 107.95),
    (11.20, 108.55), (11.55, 108.85), (11.95, 109.05), (12.35, 109.20),
    (12.75, 109.30), (13.15, 109.40), (13.55, 109.30), (13.95, 109.20),
    (14.45, 109.05), (14.95, 108.90), (15.45, 108.70), (15.95, 108.30),
    (16.45, 108.10), (16.95, 107.80), (17.45, 107.00), (17.95, 106.50),
    (18.45, 106.00), (18.95, 105.80), (19.45, 105.70), (19.95, 105.75),
    (20.45, 106.55), (20.75, 107.05), (20.95, 107.45), (21.10, 107.75),
    (21.30, 107.95), (21.55, 108.05),
    (21.75, 107.80), (22.05, 107.50), (22.25, 107.20), (22.45, 106.80),
    (22.65, 106.50), (22.85, 106.00), (22.95, 105.50), (22.95, 105.00),
    (22.95, 104.50), (22.75, 104.25), (22.55, 104.00), (22.75, 103.80),
    (22.85, 103.55),
]

def point_in_polygon(lat, lon, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

cells = []
lat = 8.5
while lat <= 23.5:
    lon = 102.0
    while lon <= 110.0:
        if point_in_polygon(lat, lon, POLYGON):
            cells.append({"lat": round(lat, 2), "lon": round(lon, 2)})
        lon = round(lon + 0.5, 2)
    lat = round(lat + 0.5, 2)

print(f"Generated {len(cells)} Vietnam grid cells (expected ~176)")

scenarios = [
    {"name": "October wet (real Hanoi)", "temp": 24, "humidity": 85, "wind": 3, "rain": 8, "daysSinceRain": 1, "mode": "forest"},
    {"name": "Web defaults (API fails)",  "temp": 28, "humidity": 65, "wind": 4, "rain": 0, "daysSinceRain": 8, "mode": "forest"},
]

for scenario in scenarios:
    print(f"\n=== {scenario['name']} ===")
    inputs = [{"lat": c["lat"], "lon": c["lon"], **{k: v for k, v in scenario.items() if k != "name"}} for c in cells]
    
    proc = subprocess.run(
        ["/home/z/.venv/bin/python3" if os.path.exists("/home/z/.venv/bin/python3") else "python3",
         "/home/z/my-project/scripts/predict_v90.py"],
        input=json.dumps({"inputs": inputs}),
        capture_output=True, text=True, timeout=120,
        cwd="/home/z/my-project"
    )
    
    if proc.returncode != 0:
        print(f"Python failed: {proc.stderr[:500]}")
        continue
    
    try:
        out = json.loads(proc.stdout)
        results = out.get("results", [])
        probs = [r.get("probability", 0) for r in results]
        
        very_high = sum(1 for p in probs if p >= 0.7)
        high = sum(1 for p in probs if 0.5 <= p < 0.7)
        medium = sum(1 for p in probs if 0.3 <= p < 0.5)
        low_med = sum(1 for p in probs if 0.15 <= p < 0.3)
        low = sum(1 for p in probs if p < 0.15)
        
        print(f"  Cells: {len(probs)}  Engine: {out.get('engine', 'n/a')}")
        print(f"  Min: {min(probs):.4f} | Max: {max(probs):.4f} | Mean: {sum(probs)/len(probs):.4f}")
        print(f"  Risk buckets:")
        print(f"    Rất cao (≥0.7):     {very_high:3d} cells  ({100*very_high/len(probs):.1f}%)")
        print(f"    Cao (0.5-0.7):      {high:3d} cells  ({100*high/len(probs):.1f}%)")
        print(f"    TB (0.3-0.5):       {medium:3d} cells  ({100*medium/len(probs):.1f}%)")
        print(f"    Thấp-TB (0.15-0.3): {low_med:3d} cells  ({100*low_med/len(probs):.1f}%)")
        print(f"    Thấp (<0.15):       {low:3d} cells  ({100*low/len(probs):.1f}%)")
        
        # Top 5 highest
        sorted_idx = sorted(range(len(probs)), key=lambda i: -probs[i])
        print(f"  Top 5 highest cells:")
        for i in sorted_idx[:5]:
            c = cells[i]
            print(f"    lat={c['lat']:5.2f}, lon={c['lon']:6.2f}: p={probs[i]:.4f} ({results[i].get('risk_level', '?')})")
    except Exception as e:
        print(f"PARSE ERROR: {e}")
        print(proc.stdout[:500])
