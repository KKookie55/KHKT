import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { predictBatchJS, predictJS } from "@/lib/predict-js";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PREDICTOR_PATH = "/home/z/my-project/scripts/predict_v86.py";
const PYTHON_BIN = "python3";

interface PredictInput {
  temp: number;
  humidity: number;
  wind: number;
  rain: number;
  daysSinceRain: number;
  region: "north" | "central" | "south";
  doy?: number;
}

async function callPythonPredictor(input: PredictInput | { inputs: PredictInput[] }): Promise<any> {
  return new Promise((resolve, reject) => {
    const inputJson = JSON.stringify(input);
    const proc = spawn(PYTHON_BIN, [PREDICTOR_PATH], {
      cwd: "/home/z/my-project",
      env: { ...process.env, PYTHONPATH: "/home/z/my-project" },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => { stdout += data.toString(); });
    proc.stderr.on("data", (data) => { stderr += data.toString(); });

    proc.on("error", (err) => { reject(new Error(`Failed to spawn python: ${err.message}`)); });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python exited ${code}: ${stderr || stdout}`));
        return;
      }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error) {
          reject(new Error(result.error));
          return;
        }
        resolve(result);
      } catch (e) {
        reject(new Error(`Invalid JSON from python: ${stdout.slice(0, 200)}`));
      }
    });

    proc.stdin.write(inputJson);
    proc.stdin.end();
  });
}

function validateInput(body: any): body is PredictInput {
  return (
    body &&
    typeof body.temp === "number" &&
    typeof body.humidity === "number" &&
    typeof body.wind === "number" &&
    typeof body.rain === "number" &&
    typeof body.daysSinceRain === "number" &&
    ["north", "central", "south"].includes(body.region)
  );
}

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    const body = await request.json();

    // Batch mode
    if (body.inputs && Array.isArray(body.inputs)) {
      const inputs = body.inputs as PredictInput[];
      const invalid = inputs.findIndex((i) => !validateInput(i));
      if (invalid >= 0) {
        return NextResponse.json({ error: `Invalid input at index ${invalid}` }, { status: 400 });
      }

      // Try Python first
      try {
        const data = await callPythonPredictor({ inputs });
        const elapsed = Date.now() - startTime;
        return NextResponse.json({ ...data, elapsed_ms: elapsed, engine: "python" });
      } catch (pythonError) {
        // Fallback to JS inference
        console.warn("Python predict failed, falling back to JS:", pythonError instanceof Error ? pythonError.message : pythonError);
        const results = predictBatchJS(inputs);
        const elapsed = Date.now() - startTime;
        return NextResponse.json({
          results,
          count: results.length,
          model_version: "v8.6_canonical_model",
          model_auroc_iid: 0.9462654930746847,
          elapsed_ms: elapsed,
          engine: "js-fallback",
        });
      }
    }

    // Single mode
    if (!validateInput(body)) {
      return NextResponse.json(
        { error: "Invalid input. Required: temp, humidity, wind, rain, daysSinceRain (numbers), region (north|central|south)" },
        { status: 400 }
      );
    }

    try {
      const result = await callPythonPredictor(body);
      const elapsed = Date.now() - startTime;
      return NextResponse.json({ ...result, elapsed_ms: elapsed, engine: "python" });
    } catch (pythonError) {
      console.warn("Python predict failed, falling back to JS:", pythonError instanceof Error ? pythonError.message : pythonError);
      const result = predictJS(body);
      const elapsed = Date.now() - startTime;
      return NextResponse.json({ ...result, elapsed_ms: elapsed, engine: "js-fallback" });
    }
  } catch (error) {
    const elapsed = Date.now() - startTime;
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: msg, elapsed_ms: elapsed }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    model_version: "v8.6_canonical_model",
    model_path: "/home/z/my-project/model/wildfire_v86_model/xgb_modis_iid.json",
    predictor: PREDICTOR_PATH,
    js_fallback: true,
    features: ["T2M", "RH2M", "WS10M", "PRECTOTCORR", "DAYS_SINCE_RAIN"],
    regions: ["north", "central", "south"],
    endpoints: {
      "POST /api/predict": "Single prediction (tries Python, falls back to JS)",
      "POST /api/predict (batch)": "Multiple predictions (body: {inputs: [...]})",
      "GET /api/predict": "Model info",
    },
  });
}
