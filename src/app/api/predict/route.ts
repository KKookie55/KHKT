import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { existsSync } from "fs";
import { predictBatchJS, predictJS } from "@/lib/predict-js";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PREDICTOR_PATH = "./scripts/predict_v90.py";
const PYTHON_BIN = existsSync('/home/z/.venv/bin/python3') ? '/home/z/.venv/bin/python3' : 'python3';

interface PredictInput {
  lat: number; lon: number;
  temp: number; humidity: number; wind: number; rain: number; daysSinceRain: number;
  region?: "north" | "central" | "south";
  doy?: number; mode?: "forest" | "any";
}

function validateInput(body: any): body is PredictInput {
  return body && typeof body.lat === "number" && typeof body.lon === "number" &&
    typeof body.temp === "number" && typeof body.humidity === "number" &&
    typeof body.wind === "number" && typeof body.rain === "number" &&
    typeof body.daysSinceRain === "number";
}

async function callPython(input: any, timeout = 15000): Promise<any> {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [PREDICTOR_PATH], { cwd: process.cwd(), env: { ...process.env } });
    let stdout = "", stderr = "";
    const timer = setTimeout(() => { proc.kill(); reject(new Error("timeout")); }, timeout);
    proc.stdout.on("data", (d) => { stdout += d; });
    proc.stderr.on("data", (d) => { stderr += d; });
    proc.on("error", (e) => { clearTimeout(timer); reject(e); });
    proc.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(stderr.slice(0, 200))); return; }
      try { resolve(JSON.parse(stdout.trim())); } catch { reject(new Error("invalid JSON")); }
    });
    proc.stdin.write(JSON.stringify(input));
    proc.stdin.end();
  });
}

export async function POST(request: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await request.json();
    const getRegion = (lat: number) => lat >= 18.5 ? "north" : lat >= 13.5 ? "central" : "south";

    // BATCH mode (>3 cells): use Python v9.0 ONLY (no v8.6 fallback — gives wrong values)
    if (body.inputs && Array.isArray(body.inputs) && body.inputs.length > 3) {
      const inputs = body.inputs as PredictInput[];
      const enriched = inputs.map(i => ({ ...i, region: i.region || getRegion(i.lat) }));
      try {
        const data = await callPython({ inputs: enriched }, 30000);
        return NextResponse.json({ ...data, elapsed_ms: Date.now() - startTime, engine: "python-v90" });
      } catch (pyError) {
        // Return error — do NOT use v8.6 fallback (gives wrong p=0.44 with region offset)
        return NextResponse.json({ 
          error: `Python v9.0 failed: ${pyError instanceof Error ? pyError.message : pyError}`,
          elapsed_ms: Date.now() - startTime,
        }, { status: 500 });
      }
    }

    // SINGLE or SMALL batch: try Python v9.0, fall back to JS v8.6
    if (body.inputs && Array.isArray(body.inputs)) {
      const inputs = body.inputs as PredictInput[];
      const enriched = inputs.map(i => ({ ...i, region: i.region || getRegion(i.lat) }));
      try {
        const data = await callPython({ inputs: enriched });
        return NextResponse.json({ ...data, elapsed_ms: Date.now() - startTime, engine: "python-v90" });
      } catch {
        const results = predictBatchJS(enriched);
        return NextResponse.json({ results, count: results.length, model_version: "v8.6 (JS fallback)", elapsed_ms: Date.now() - startTime, engine: "js-v86" });
      }
    }

    // Single prediction
    if (!validateInput(body)) {
      return NextResponse.json({ error: "Invalid input" }, { status: 400 });
    }
    const enriched = { ...body, region: body.region || getRegion(body.lat) };
    try {
      const result = await callPython(enriched);
      return NextResponse.json({ ...result, elapsed_ms: Date.now() - startTime, engine: "python-v90" });
    } catch {
      const result = predictJS(enriched);
      return NextResponse.json({ ...result, elapsed_ms: Date.now() - startTime, engine: "js-v86" });
    }
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    model_version: "v9.0 (Python, single) + v8.6 (JS, batch)",
    note: "Single-cell predictions use v9.0 Python (61 features). Batch (map) uses v8.6 JS (35 features, instant).",
  });
}
