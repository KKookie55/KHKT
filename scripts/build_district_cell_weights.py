"""Build district_cell_weights.parquet from cell_to_district.json + VIETNAM_GRID."""
import json, os
import pandas as pd
import polars as pl
from shapely.geometry import box, Polygon
from shapely.strtree import STRtree
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.ops import unary_union

# Load GADM districts
with open('/home/z/my-project/public/data/geo/gadm41_VNM_2.json') as f:
    gadm = json.load(f)

districts = []
for i, feat in enumerate(gadm["features"]):
    geom = shape(feat["geometry"])
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            geom = unary_union(polys)
    districts.append({"id": i, "shp": geom})

# Build cell polygons from VIETNAM_GRID (mirrored from data.ts)
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

# Build VIETNAM_GRID (lat, lon pairs)
cells = []
lat = 8.5
while lat <= 23.5:
    lon = 102.0
    while lon <= 110.0:
        if point_in_polygon(lat, lon, POLYGON_VN):
            cells.append({"lat": round(lat, 2), "lon": round(lon, 2)})
        lon = round(lon + 0.5, 2)
    lat = round(lat + 0.5, 2)
print(f"Generated {len(cells)} Vietnam grid cells")

# Build cell polygons (0.5° × 0.5° boxes centered at (lat, lon))
CELL_STEP = 0.5
cell_polys = []
for c in cells:
    cell_polys.append(box(c["lon"] - CELL_STEP/2, c["lat"] - CELL_STEP/2,
                          c["lon"] + CELL_STEP/2, c["lat"] + CELL_STEP/2))

tree = STRtree(cell_polys)

# For each district, find overlapping cells + area weights
dcw_rows = []
for d in districts:
    dgeom = d["shp"]
    if dgeom.is_empty:
        continue
    candidates = tree.query(dgeom)
    for ci in candidates:
        idx = int(ci) if not isinstance(ci, int) else ci
        cell_geom = cell_polys[idx]
        try:
            intersection_area = dgeom.intersection(cell_geom).area
            if intersection_area <= 0:
                continue
            cell_area = cell_geom.area
            weight = intersection_area / cell_area  # fraction of cell inside district
            if weight < 0.001:
                continue
            dcw_rows.append({
                "district_id": d["id"],
                "lat": float(cells[idx]["lat"]),
                "lon": float(cells[idx]["lon"]),
                "weight": weight,
                "district_area_in_cell": intersection_area,
            })
        except Exception:
            continue

dcw = pd.DataFrame(dcw_rows)
# Normalize weights per district (sum to 1)
dcw["weight_norm"] = dcw.groupby("district_id")["weight"].transform(lambda x: x / x.sum())

OUT = "/home/z/my-project/download/production/district_cell_weights.parquet"
pl.from_pandas(dcw).write_parquet(OUT)
print(f"Saved: {OUT}")
print(f"Total (district, cell) pairs: {len(dcw)}")
print(f"Districts with cells: {dcw['district_id'].nunique()}")
