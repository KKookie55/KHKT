import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { existsSync } from "fs";
import { predictBatchJS, predictJS } from "@/lib/predict-js";
import { predictBatchV94JS } from "@/lib/predict-v94-js";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PREDICTOR_V90 = "./scripts/predict_v90.py";
const PREDICTOR_V91 = "./scripts/predict_v91.py";
const PYTHON_BIN = existsSync('/home/z/.venv/bin/python3') ? '/home/z/.venv/bin/python3' : 'python3';

interface PredictInput {
  lat?: number; lon?: number;
  district_id?: number;
  temp: number; humidity: number; wind: number; rain: number; daysSinceRain: number;
  region?: "north" | "central" | "south";
  doy?: number; mode?: "forest" | "any";
}

function callPython(input: any, scriptPath: string, timeout = 15000): Promise<any> {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [scriptPath], { cwd: process.cwd(), env: { ...process.env } });
    let stdout = "", stderr = "";
    const timer = setTimeout(() => { proc.kill(); reject(new Error("timeout")); }, timeout);
    proc.stdout.on("data", (d) => { stdout += d; });
    proc.stderr.on("data", (d) => { stderr += d; });
    proc.on("error", (e) => { clearTimeout(timer); reject(e); });
    proc.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(stderr.slice(0, 300))); return; }
      try { resolve(JSON.parse(stdout.trim())); } catch { reject(new Error("invalid JSON")); }
    });
    proc.stdin.write(JSON.stringify(input));
    proc.stdin.end();
  });
}

// Fetch FIRMS last 24h fire data → neighbor_fire per district
async function fetchNeighborFire(): Promise<Record<number, number>> {
  try {
    const resp = await fetch("https://firms.modaps.eosdis.nasa.gov/api/country/csv/MODIS/1/VNM", { signal: AbortSignal.timeout(10000) });
    if (!resp.ok) return {};
    const csv = await resp.text();
    const lines = csv.trim().split("\n");
    if (lines.length < 2) return {};
    const headers = lines[0].split(",").map(h => h.trim());
    const latIdx = headers.findIndex(h => h.toLowerCase() === "latitude");
    const lonIdx = headers.findIndex(h => h.toLowerCase() === "longitude");
    if (latIdx < 0 || lonIdx < 0) return {};
    const nf: Record<number, number> = {};
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(",");
      const lat = parseFloat(cols[latIdx]);
      const lon = parseFloat(cols[lonIdx]);
      if (!isNaN(lat) && !isNaN(lon) && lat >= 8 && lat <= 24 && lon >= 102 && lon <= 110) {
        const approxId = Math.min(709, Math.max(0, Math.floor((lat - 8) * 70) + Math.floor((lon - 102) * 10)));
        nf[approxId] = (nf[approxId] || 0) + 1;
      }
    }
    return nf;
  } catch { return {}; }
}

export async function POST(request: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await request.json();
    const getRegion = (lat: number) => lat >= 18.5 ? "north" : lat >= 13.5 ? "central" : "south";

    // ===== DISTRICT MODE (v9.4) — if any input has district_id =====
    if (body.inputs && Array.isArray(body.inputs) && body.inputs.some((i: any) => i.district_id != null)) {
      try {
        const data = await callPython(body, PREDICTOR_V91, 60000);
        return NextResponse.json({ ...data, elapsed_ms: Date.now() - startTime, engine: "python-v94" });
      } catch {
        // v9.4 JS fallback — fetch FIRMS neighbor_fire, then predict
        const neighborFire = await fetchNeighborFire();
        const results = predictBatchV94JS(body.inputs, neighborFire);
        const firmsCount = Object.values(neighborFire).reduce((a, b) => a + b, 0);
        return NextResponse.json({ results, count: results.length, model_version: "v9.4 JS (75 feats, Optuna + FIRMS)", elapsed_ms: Date.now() - startTime, engine: "js-v94", firms_points: firmsCount });
      }
    }

    if (body.district_id != null) {
      try {
        const result = await callPython(body, PREDICTOR_V91, 30000);
        return NextResponse.json({ ...result, elapsed_ms: Date.now() - startTime, engine: "python-v94" });
      } catch {
        const neighborFire = await fetchNeighborFire();
        const results = predictBatchV94JS([body], neighborFire);
        return NextResponse.json({ ...results[0], elapsed_ms: Date.now() - startTime, engine: "js-v94" });
      }
    }

    // ===== CELL MODE (v9.0) — lat/lon inputs =====
    if (body.inputs && Array.isArray(body.inputs) && body.inputs.length > 3) {
      const inputs = body.inputs as PredictInput[];
      const enriched = inputs.map(i => ({ ...i, region: i.region || (i.lat != null ? getRegion(i.lat) : "central") }));
      try {
        const data = await callPython({ inputs: enriched }, PREDICTOR_V90, 30000);
        return NextResponse.json({ ...data, elapsed_ms: Date.now() - startTime, engine: "python-v90" });
      } catch {
        const results = predictBatchJS(enriched);
        return NextResponse.json({ results, count: results.length, model_version: "v8.6 (JS fallback)", elapsed_ms: Date.now() - startTime, engine: "js-v86" });
      }
    }

    if (body.inputs && Array.isArray(body.inputs)) {
      const inputs = body.inputs as PredictInput[];
      const enriched = inputs.map(i => ({ ...i, region: i.region || (i.lat != null ? getRegion(i.lat!) : "central") }));
      try {
        const data = await callPython({ inputs: enriched }, PREDICTOR_V90);
        return NextResponse.json({ ...data, elapsed_ms: Date.now() - startTime, engine: "python-v90" });
      } catch {
        const results = predictBatchJS(enriched);
        return NextResponse.json({ results, count: results.length, model_version: "v8.6 (JS fallback)", elapsed_ms: Date.now() - startTime, engine: "js-v86" });
      }
    }

    if (body.lat != null && body.lon != null) {
      const enriched = { ...body, region: body.region || getRegion(body.lat) };
      try {
        const result = await callPython(enriched, PREDICTOR_V90);
        return NextResponse.json({ ...result, elapsed_ms: Date.now() - startTime, engine: "python-v90" });
      } catch {
        const result = predictJS(enriched);
        return NextResponse.json({ ...result, elapsed_ms: Date.now() - startTime, engine: "js-v86" });
      }
    }

    return NextResponse.json({ error: "Invalid input — need lat/lon or district_id" }, { status: 400 });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    model_version: "v9.4 JS (district, 75 features + FIRMS) + v9.0 Python (cell) + v8.6 JS (cell fallback)",
    note: "District mode: v9.4 JS with real-time FIRMS neighbor_fire. Cell mode: v8.6 JS fallback.",
  });
}
