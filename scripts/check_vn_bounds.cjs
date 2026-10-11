// Audit script: standalone — copy the polygon + generators inline
// Run: node /home/z/my-project/scripts/check_vn_bounds.cjs

// ===== isInVietnam polygon (mirrors src/lib/data.ts) =====
function isInVietnam(lat, lon) {
  const polygon = [
    [22.85, 103.55], [22.55, 103.75], [22.10, 103.55], [21.55, 103.50],
    [21.10, 103.55], [20.65, 104.05], [20.10, 104.05], [19.55, 104.55],
    [19.05, 104.10], [18.55, 104.10], [18.05, 104.55], [17.55, 105.55],
    [17.05, 106.05], [16.55, 106.55], [16.05, 107.05], [15.55, 107.45],
    [15.05, 107.50], [14.55, 107.10], [14.05, 107.05], [13.55, 106.05],
    [13.05, 105.55], [12.05, 105.05], [11.05, 105.05], [10.55, 104.95],
    [10.55, 104.65], [10.45, 104.55], [10.30, 104.40], [10.05, 104.55],
    [9.55, 104.75], [9.05, 104.75], [8.65, 104.65], [8.55, 104.85], [8.95, 105.05],
    [9.35, 105.35], [9.85, 105.55],
    [10.15, 105.85], [10.45, 106.45], [10.75, 106.85], [10.85, 107.30],
    [10.95, 107.55], [10.85, 107.95],
    [11.20, 108.55], [11.55, 108.85], [11.95, 109.05], [12.35, 109.20],
    [12.75, 109.30], [13.15, 109.40], [13.55, 109.30], [13.95, 109.20],
    [14.45, 109.05], [14.95, 108.90], [15.45, 108.70], [15.95, 108.30],
    [16.45, 108.10], [16.95, 107.80], [17.45, 107.00], [17.95, 106.50],
    [18.45, 106.00], [18.95, 105.80], [19.45, 105.70], [19.95, 105.75],
    [20.45, 106.55], [20.75, 107.05], [20.95, 107.45], [21.10, 107.75],
    [21.30, 107.95], [21.55, 108.05],
    [21.75, 107.80], [22.05, 107.50], [22.25, 107.20], [22.45, 106.80],
    [22.65, 106.50], [22.85, 106.00], [22.95, 105.50], [22.95, 105.00],
    [22.95, 104.50], [22.75, 104.25], [22.55, 104.00], [22.75, 103.80],
    [22.85, 103.55],
  ];
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [plat_i, plon_i] = polygon[i];
    const [plat_j, plon_j] = polygon[j];
    if (((plat_i > lat) !== (plat_j > lat)) &&
        (lon < (plon_j - plon_i) * (lat - plat_i) / (plat_j - plat_i + 1e-12) + plon_i)) {
      inside = !inside;
    }
  }
  return inside;
}

// ===== Generate grid (mirrors generateVietnamGrid) =====
const cells = [];
let id = 0;
const minLat = 8.5, maxLat = 23.5;
const minLon = 102.0, maxLon = 110.0;
const step = 0.5;
for (let lat = minLat; lat <= maxLat; lat += step) {
  for (let lon = minLon; lon <= maxLon; lon += step) {
    if (!isInVietnam(lat, lon)) continue;
    const region = lat >= 18.5 ? "north" : lat >= 13.5 ? "central" : "south";
    let baseRisk;
    if (region === "north") {
      baseRisk = 0.15 + Math.sin((lon - 102) / 8 * Math.PI) * 0.10 + Math.random() * 0.10;
    } else if (region === "central") {
      const tbn = lat >= 11.5 && lat <= 14.5 && lon >= 107 && lon <= 109 ? 0.20 : 0;
      baseRisk = 0.30 + tbn + Math.random() * 0.10;
    } else {
      baseRisk = 0.45 + Math.random() * 0.15;
    }
    baseRisk = Math.max(0.05, Math.min(0.85, baseRisk));
    cells.push({ id: id++, lat: +lat.toFixed(2), lon: +lon.toFixed(2), region, baseRisk: +baseRisk.toFixed(3) });
  }
}

// ===== Generate fire points (mirrors generateFirePoints) =====
const clusters = [
  { centerLat: 22.5, centerLon: 104.5, radius: 1.2, count: 18, region: "north", seasonBias: [1, 120], frpRange: [10, 80] },
  { centerLat: 22.8, centerLon: 106.3, radius: 1.0, count: 14, region: "north", seasonBias: [1, 100], frpRange: [15, 90] },
  { centerLat: 21.5, centerLon: 105.5, radius: 0.8, count: 10, region: "north", seasonBias: [60, 150], frpRange: [8, 60] },
  { centerLat: 14.0, centerLon: 108.0, radius: 1.5, count: 28, region: "central", seasonBias: [30, 150], frpRange: [20, 150] },
  { centerLat: 12.5, centerLon: 108.0, radius: 1.2, count: 22, region: "central", seasonBias: [30, 150], frpRange: [15, 120] },
  { centerLat: 11.5, centerLon: 108.0, radius: 1.0, count: 16, region: "central", seasonBias: [40, 140], frpRange: [10, 90] },
  { centerLat: 16.0, centerLon: 108.5, radius: 0.8, count: 12, region: "central", seasonBias: [60, 180], frpRange: [8, 70] },
  { centerLat: 11.0, centerLon: 107.0, radius: 1.2, count: 24, region: "south", seasonBias: [330, 365], frpRange: [12, 100] },
  { centerLat: 11.5, centerLon: 106.5, radius: 1.0, count: 18, region: "south", seasonBias: [330, 365], frpRange: [10, 80] },
  { centerLat: 10.5, centerLon: 105.5, radius: 1.5, count: 20, region: "south", seasonBias: [340, 100], frpRange: [8, 60] },
  { centerLat: 10.0, centerLon: 106.0, radius: 0.8, count: 10, region: "south", seasonBias: [350, 90], frpRange: [6, 50] },
];

const points = [];
let pid = 0;
for (const cluster of clusters) {
  let added = 0;
  let attempts = 0;
  while (added < cluster.count && attempts < cluster.count * 4) {
    attempts++;
    const angle = Math.random() * 2 * Math.PI;
    const r = Math.random() * cluster.radius;
    const lat = cluster.centerLat + r * Math.cos(angle) * 0.7;
    const lon = cluster.centerLon + r * Math.sin(angle);
    if (!isInVietnam(lat, lon)) continue;
    points.push({ id: pid++, lat: +lat.toFixed(3), lon: +lon.toFixed(3), region: cluster.region });
    added++;
  }
  console.log(`  Cluster @(${cluster.centerLat}, ${cluster.centerLon}): ${added}/${cluster.count} accepted (${attempts} attempts)`);
}

console.log("\n=== AUDIT RESULTS ===");
console.log(`Grid cells: ${cells.length}`);
const cellsByRegion = cells.reduce((acc, c) => { acc[c.region] = (acc[c.region] || 0) + 1; return acc; }, {});
console.log(`  By region:`, cellsByRegion);

console.log(`\nFire points: ${points.length}`);
const pointsByRegion = points.reduce((acc, p) => { acc[p.region] = (acc[p.region] || 0) + 1; return acc; }, {});
console.log(`  By region:`, pointsByRegion);

const cellLats = cells.map(c => c.lat);
const cellLons = cells.map(c => c.lon);
const pointLats = points.map(p => p.lat);
const pointLons = points.map(p => p.lon);

console.log("\n=== BOUNDING BOXES ===");
console.log(`Cells  lat: [${Math.min(...cellLats)}, ${Math.max(...cellLats)}]`);
console.log(`Cells  lon: [${Math.min(...cellLons)}, ${Math.max(...cellLons)}]`);
console.log(`Points lat: [${Math.min(...pointLats)}, ${Math.max(...pointLats)}]`);
console.log(`Points lon: [${Math.min(...pointLons)}, ${Math.max(...pointLons)}]`);

// Sanity check: any sea/neighbor country?
const seaPts = points.filter(p => p.lon > 109.5);
console.log(`\nFire pts > 109.5°E (sea): ${seaPts.length}`);
const westPts = points.filter(p => p.lon < 104.0);
console.log(`Fire pts < 104.0°E (Cambodia/Thailand): ${westPts.length}`);
const northPts = points.filter(p => p.lat > 23.5);
console.log(`Fire pts > 23.5°N (China): ${northPts.length}`);

// Test specific known locations
console.log("\n=== SPOT CHECKS ===");
const tests = [
  ["Hà Nội", 21.03, 105.85, true],
  ["TP.HCM", 10.78, 106.70, true],
  ["Đà Nẵng", 16.05, 108.20, true],
  ["Cà Mau", 9.18, 105.15, true],
  ["Hà Tiên", 10.38, 104.49, true],
  ["Móng Cái", 21.52, 107.96, true],
  ["Vịnh Bắc Bộ (biển)", 20.0, 108.0, false],
  ["Biển Đông", 12.0, 111.0, false],
  ["Lào (Viêng Chăn)", 17.97, 102.63, false],
  ["Campuchia (Phnom Penh)", 11.55, 104.93, false],
];
for (const [name, lat, lon, expected] of tests) {
  const result = isInVietnam(lat, lon);
  const status = result === expected ? "✓" : "✗ FAIL";
  console.log(`  ${status} ${name} [${lat}, ${lon}]: ${result} (expected ${expected})`);
}
