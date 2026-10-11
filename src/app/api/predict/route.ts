import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PREDICTOR_PATH = "/home/z/my-project/scripts/predict_v90.py";
const PYTHON_BIN = "/home/z/.venv/bin/python3";

interface PredictInput {
  lat: number;
  lon: number;
  temp: number;
  humidity: number;
  wind: number;
  rain: number;
  daysSinceRain: number;
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
    typeof body.lat === "number" &&
    typeof body.lon === "number" &&
    typeof body.temp === "number" &&
    typeof body.humidity === "number" &&
    typeof body.wind === "number" &&
    typeof body.rain === "number" &&
    typeof body.daysSinceRain === "number"
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
        return NextResponse.json({ error: `Invalid input at index ${invalid}. Required: lat, lon, temp, humidity, wind, rain, daysSinceRain (numbers)` }, { status: 400 });
      }

      const data = await callPythonPredictor({ inputs });
      const elapsed = Date.now() - startTime;
      return NextResponse.json({ ...data, elapsed_ms: elapsed, engine: "python-v90" });
    }

    // Single mode
    if (!validateInput(body)) {
      return NextResponse.json(
        { error: "Invalid input. Required: lat, lon, temp, humidity, wind, rain, daysSinceRain (numbers)" },
        { status: 400 }
      );
    }

    const result = await callPythonPredictor(body);
    const elapsed = Date.now() - startTime;
    return NextResponse.json({ ...result, elapsed_ms: elapsed, engine: "python-v90" });
  } catch (error) {
    const elapsed = Date.now() - startTime;
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: msg, elapsed_ms: elapsed }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    model_version: "v9.0 forest fire (61 features, multi-task)",
    model_path: "/home/z/my-project/download/production/model.json",
    iso_path: "/home/z/my-project/download/production/iso.pkl",
    predictor: PREDICTOR_PATH,
    label: "MODIS detection with confidence=high (forest fire, excludes agricultural burning)",
    features_count: 61,
    feature_groups: [
      "35 cold baseline (weather + lags + rolling + derived)",
      "7 weather anomalies (departure from per-cell doy climatology)",
      "7 land cover (forest, shrubland, grassland, agriculture, urban, bare, water)",
      "9 geo (elevation, slope, 3 Köppen one-hot, latitude_band, coastal_distance, population_proxy)",
      "3 NDVI (mean, std, amplitude per cell)",
    ],
    metrics_south_cold_OOD: {
      AUROC: 0.921,
      PR_AUC: 0.228,
      Brier: 0.031,
      pos_rate_south: 0.034,
    },
    endpoints: {
      "POST /api/predict": "Single prediction (lat, lon, temp, humidity, wind, rain, daysSinceRain)",
      "POST /api/predict (batch)": "Multiple predictions (body: {inputs: [...]})",
      "GET /api/predict": "Model info",
    },
  });
}
