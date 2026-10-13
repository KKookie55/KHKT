import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { existsSync } from "fs";
import { predictBatchJS, predictJS } from "@/lib/predict-js";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PREDICTOR_V90 = "./scripts/predict_v90.py";
const PREDICTOR_V91 = "./scripts/predict_v91.py";
const PYTHON_BIN = existsSync('/home/z/.venv/bin/python3') ? '/home/z/.venv/bin/python3' : 'python3';

interface PredictInput {
  lat?: number; lon?: number;       // v9.0 cell mode
  district_id?: number;              // v9.1 district mode
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

export async function POST(request: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await request.json();
    const getRegion = (lat: number) => lat >= 18.5 ? "north" : lat >= 13.5 ? "central" : "south";

    // ===== v9.1 DISTRICT MODE — if any input has district_id, use v9.1 =====
    if (body.inputs && Array.isArray(body.inputs) && body.inputs.some((i: any) => i.district_id != null)) {
      try {
        const data = await callPython(body, PREDICTOR_V91, 60000);  // 60s — district mode can be slower
        return NextResponse.json({ ...data, elapsed_ms: Date.now() - startTime, engine: "python-v91-district" });
      } catch (pyError) {
        return NextResponse.json({
          error: `Python v9.1 failed: ${pyError instanceof Error ? pyError.message : pyError}`,
          elapsed_ms: Date.now() - startTime,
        }, { status: 500 });
      }
    }

    // ===== v9.0 CELL MODE — lat/lon inputs =====
    if (body.district_id != null) {
      // Single district prediction
      try {
        const result = await callPython(body, PREDICTOR_V91, 30000);
        return NextResponse.json({ ...result, elapsed_ms: Date.now() - startTime, engine: "python-v91-district" });
      } catch (pyError) {
        return NextResponse.json({
          error: `Python v9.1 failed: ${pyError instanceof Error ? pyError.message : pyError}`,
        }, { status: 500 });
      }
    }

    // BATCH mode (>3 cells): use Python v9.0 ONLY
    if (body.inputs && Array.isArray(body.inputs) && body.inputs.length > 3) {
      const inputs = body.inputs as PredictInput[];
      const enriched = inputs.map(i => ({ ...i, region: i.region || (i.lat != null ? getRegion(i.lat) : "central") }));
      try {
        const data = await callPython({ inputs: enriched }, PREDICTOR_V90, 30000);
        return NextResponse.json({ ...data, elapsed_ms: Date.now() - startTime, engine: "python-v90" });
      } catch (pyError) {
        return NextResponse.json({
          error: `Python v9.0 failed: ${pyError instanceof Error ? pyError.message : pyError}`,
          elapsed_ms: Date.now() - startTime,
        }, { status: 500 });
      }
    }

    // SMALL batch / single cell
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
    model_version: "v9.1 (district) + v9.0 (cell) + v8.6 (JS fallback)",
    note: "If inputs have district_id → v9.1 (district-level, 74 feats, AUROC=0.93 OOD). Else if lat/lon → v9.0 (cell-level, 61 feats, AUROC=0.92 OOD).",
  });
}
