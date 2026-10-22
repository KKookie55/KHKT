import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

let cachedFirms: { data: Record<number, number>, timestamp: number } | null = null;
const CACHE_TTL = 10 * 60 * 1000;

export async function GET(request: NextRequest) {
  try {
    if (cachedFirms && Date.now() - cachedFirms.timestamp < CACHE_TTL) {
      return NextResponse.json({ neighbor_fire: cachedFirms.data, cached: true });
    }

    // Fetch FIRMS last 24h for Vietnam
    const firmsUrl = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/MODIS/1/VNM";
    const resp = await fetch(firmsUrl, { signal: AbortSignal.timeout(15000) });

    if (!resp.ok) {
      return NextResponse.json({ neighbor_fire: {}, error: "FIRMS API unavailable" });
    }

    const csvText = await resp.text();
    const lines = csvText.trim().split("\n");

    if (lines.length < 2) {
      return NextResponse.json({ neighbor_fire: {}, error: "No fire data" });
    }

    const headers = lines[0].split(",").map(h => h.trim());
    const latIdx = headers.findIndex(h => h.toLowerCase() === "latitude" || h.toLowerCase() === "lat");
    const lonIdx = headers.findIndex(h => h.toLowerCase() === "longitude" || h.toLowerCase() === "lon" || h.toLowerCase() === "long");

    if (latIdx === -1 || lonIdx === -1) {
      return NextResponse.json({ neighbor_fire: {}, error: "No lat/lon in CSV" });
    }

    const firePoints: { lat: number; lon: number }[] = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(",");
      const lat = parseFloat(cols[latIdx]);
      const lon = parseFloat(cols[lonIdx]);
      if (!isNaN(lat) && !isNaN(lon) && lat >= 8 && lat <= 24 && lon >= 102 && lon <= 110) {
        firePoints.push({ lat, lon });
      }
    }

    // Assign fires to district by coarse binning (proxy for spatial join)
    const districtFires: Record<number, number> = {};
    for (const fp of firePoints) {
      const approxId = Math.min(709, Math.max(0, Math.floor((fp.lat - 8) * 70) + Math.floor((fp.lon - 102) * 10)));
      districtFires[approxId] = (districtFires[approxId] || 0) + 1;
    }

    const neighborFire: Record<number, number> = {};
    for (let i = 0; i < 710; i++) {
      neighborFire[i] = districtFires[i] || 0;
    }

    cachedFirms = { data: neighborFire, timestamp: Date.now() };

    return NextResponse.json({
      neighbor_fire: neighborFire,
      fire_points: firePoints.length,
      cached: false,
      fetched_at: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json({
      neighbor_fire: {},
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
