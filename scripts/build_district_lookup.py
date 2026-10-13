"""Build cell → district lookup + simplified district GeoJSON for web.

Outputs:
  public/data/geo/districts_simple.json   (simplified polygons, ~300-500 KB)
  public/data/geo/cell_to_district.json   (cell_id → {district_id, name, province})
"""
import json, sys, os
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

INPUT = "/home/z/my-project/public/data/geo/gadm41_VNM_2.json"
OUT_DIR = "/home/z/my-project/public/data/geo"

# Mirror VIETNAM_GRID generation from src/lib/data.ts
POLYGON_VN = [
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
        if point_in_polygon(lat, lon, POLYGON_VN):
            cells.append({"id": len(cells), "lat": round(lat, 2), "lon": round(lon, 2)})
        lon = round(lon + 0.5, 2)
    lat = round(lat + 0.5, 2)

print(f"Generated {len(cells)} Vietnam grid cells")

# Load GADM
with open(INPUT) as f:
    gadm = json.load(f)

print(f"Loaded {len(gadm['features'])} GADM districts")

# Build shapely polygons + index
districts = []
for i, feat in enumerate(gadm["features"]):
    props = feat["properties"]
    geom = feat["geometry"]
    try:
        shp = shape(geom)
        if not shp.is_valid:
            shp = make_valid(shp)
        if shp.geom_type == "GeometryCollection":
            # Extract polygons
            polys = [g for g in shp.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if polys:
                shp = unary_union(polys)
        districts.append({
            "id": i,
            "gid": props.get("GID_2", ""),
            "name": props.get("NAME_2", ""),
            "province": props.get("NAME_1", ""),
            "type": props.get("TYPE_2", ""),
            "shp": shp,
        })
    except Exception as e:
        print(f"  Skip district {i} ({props.get('NAME_2', '?')}): {e}")

print(f"Valid districts: {len(districts)}")

# Map each cell → district (point-in-polygon)
cell_to_district = {}
unmapped = 0
for cell in cells:
    pt = Point(cell["lon"], cell["lat"])  # lon=x, lat=y
    found = None
    for d in districts:
        try:
            if d["shp"].contains(pt) or d["shp"].touches(pt):
                found = d
                break
        except Exception:
            continue
    if found:
        cell_to_district[cell["id"]] = {
            "district_id": found["id"],
            "gid": found["gid"],
            "name": found["name"],
            "province": found["province"],
            "type": found["type"],
        }
    else:
        cell_to_district[cell["id"]] = None
        unmapped += 1

print(f"Mapped: {len(cells) - unmapped}/{len(cells)} cells → districts")
if unmapped:
    print(f"  Unmapped (probably coastal/border cells): {unmapped}")

# Build simplified GeoJSON for web (tolerance=0.005° ≈ 500m)
def simplify_geom(shp, tol=0.005):
    try:
        s = shp.simplify(tol, preserve_topology=True)
        if s.is_empty:
            s = shp.simplify(tol * 2, preserve_topology=False)
        return s.__geo_interface__
    except Exception:
        return shp.__geo_interface__

simple_features = []
for d in districts:
    simple_geom = simplify_geom(d["shp"], tol=0.005)
    simple_features.append({
        "type": "Feature",
        "properties": {
            "id": d["id"],
            "gid": d["gid"],
            "name": d["name"],
            "province": d["province"],
            "type": d["type"],
        },
        "geometry": simple_geom,
    })

simple_fc = {"type": "FeatureCollection", "features": simple_features}

# Stats
orig_size = os.path.getsize(INPUT)
simple_path = os.path.join(OUT_DIR, "districts_simple.json")
with open(simple_path, "w") as f:
    json.dump(simple_fc, f, separators=(",", ":"))
simple_size = os.path.getsize(simple_path)
print(f"\nSimplified GeoJSON:")
print(f"  Original: {orig_size/1024:.0f} KB ({len(gadm['features'])} features)")
print(f"  Simplified: {simple_size/1024:.0f} KB ({len(simple_features)} features)")

# Cell → district lookup
lookup_path = os.path.join(OUT_DIR, "cell_to_district.json")
with open(lookup_path, "w") as f:
    json.dump(cell_to_district, f, separators=(",", ":"))
print(f"  Cell→District lookup: {os.path.getsize(lookup_path)/1024:.1f} KB ({len(cells)} cells)")

# Sample mappings
print("\nSample cell → district:")
for cell_id in [0, 50, 100, 150]:
    if cell_id < len(cells):
        m = cell_to_district.get(cell_id)
        c = cells[cell_id]
        if m:
            print(f"  Cell {cell_id} ({c['lat']},{c['lon']}) → {m['type']} {m['name']}, {m['province']}")
        else:
            print(f"  Cell {cell_id} ({c['lat']},{c['lon']}) → (unmapped)")
