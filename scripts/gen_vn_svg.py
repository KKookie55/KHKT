#!/usr/bin/env python3
"""
Generate Vietnam SVG path from the polygon in data.ts.
Output: a React component with inline SVG.
"""
# Polygon vertices from src/lib/data.ts isInVietnam()
polygon = [
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

# Map to SVG coordinates
# Lat range: 8.5 to 23.5 (15 degrees) → y (inverted)
# Lon range: 102 to 110 (8 degrees) → x
# Use viewBox "0 0 800 1200" (aspect ratio ~2:3 to match VN shape)

VB_W = 800
VB_H = 1200
lat_min, lat_max = 8.0, 24.0
lon_min, lon_max = 102.0, 110.0

def to_svg(lat, lon):
    x = (lon - lon_min) / (lon_max - lon_min) * VB_W
    y = (lat_max - lat) / (lat_max - lat_min) * VB_H
    return f"{x:.1f},{y:.1f}"

points = [to_svg(lat, lon) for lat, lon in polygon]
path_data = "M " + " L ".join(points) + " Z"

# Output as React component
component = f'''"use client";

/**
 * Vietnam silhouette as subtle gray background.
 * Coordinates derived from the same polygon used for isInVietnam() in data.ts.
 */
export function VietnamBackdrop({{ className = "" }}: {{ className?: string }}) {{
  return (
    <svg
      className={{className}}
      viewBox="0 0 {VB_W} {VB_H}"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      <path
        d="{path_data}"
        fill="#e2e8f0"
        stroke="#cbd5e1"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}}
'''

out_path = "/home/z/my-project/src/components/site/VietnamBackdrop.tsx"
with open(out_path, "w") as f:
    f.write(component)

print(f"Written: {out_path}")
print(f"Polygon vertices: {len(polygon)}")
print(f"Path data length: {len(path_data)} chars")
print(f"\nFirst 200 chars of path: {path_data[:200]}...")
