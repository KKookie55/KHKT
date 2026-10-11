/**
 * Mock fire prediction engine.
 *
 * NOTE: This is NOT the actual trained XGBoost model from the report.
 * It is a logistic-like approximation that uses the reported coefficient
 * signs (temp ↑ → fire ↑, humidity ↑ → fire ↓, wind ↑ → fire ↑, rain ↑ → fire ↓)
 * plus region-specific offsets that approximate the concept drift discovered
 * in the paper (Bắc→Nam AUROC drops from 0.842 to 0.584).
 *
 * The OOD warning fires when inputs fall outside the training domain
 * (NASA POWER 2018–2023, lưới 0.5° tại Việt Nam).
 */

export interface PredictionInput {
  temp: number;          // T2M in °C, typical range [10, 40]
  humidity: number;      // RH2M in %, typical range [30, 100]
  wind: number;          // WS10M in m/s, typical range [0, 10]
  rain: number;          // PRECTOTCORR in mm, typical range [0, 50]
  daysSinceRain: number; // count, typical range [0, 20]
  region: "north" | "central" | "south";
}

export interface PredictionResult {
  probability: number;    // 0–1
  riskLevel: "Thấp" | "Trung bình" | "Cao" | "Rất cao";
  riskColor: string;
  riskScore: number;      // 0–100 for display
  confidence: "Cao" | "Trung bình" | "Thấp";
  outOfDomain: boolean;
  warning?: string;
  contributions: { factor: string; contribution: number; direction: string }[];
}

// Training domain bounds (NASA POWER 2018–2023, Vietnam)
const TRAINING_DOMAIN = {
  temp:    [10, 38],
  humidity:[40, 100],
  wind:    [0.5, 8.0],
  rain:    [0, 30],
  daysSinceRain: [0, 15],
};

// Region offsets (mocks concept drift discovered in the paper)
// South is harder to predict → adds noise + region-specific offset
const REGION_OFFSET = {
  north:   0.0,
  central: 0.05,
  south:   0.18,  // ← matches the AUROC drop on VIIRS Bắc→Nam
};

export function predictFire(input: PredictionInput): PredictionResult {
  const { temp, humidity, wind, rain, daysSinceRain, region } = input;

  // Logistic score with monotonic relationships (matching PhysMLP expectations)
  const z =
      0.061 * (temp - 25)              // temp ↑ → fire ↑
    - 0.044 * (humidity - 80)         // humidity ↑ → fire ↓
    + 0.131 * (wind - 3)              // wind ↑ → fire ↑
    - 0.381 * Math.log1p(rain)        // rain ↑ → fire ↓ (log)
    + 0.092 * Math.log1p(daysSinceRain) // dry spell ↑ → fire ↑
    + REGION_OFFSET[region];

  let probability = 1 / (1 + Math.exp(-z));

  // Apply OOD penalty: when far outside training domain, AUROC drops
  // (this matches the report's finding: Bắc→Nam VIIRS = 0.584)
  const outOfDomain = isOutOfDomain(input);
  if (outOfDomain) {
    // Pull probability toward 0.5 (random) to reflect loss of discriminative power
    probability = 0.5 + (probability - 0.5) * 0.55;
  }

  // Clamp
  probability = Math.max(0.01, Math.min(0.99, probability));

  // Risk level
  let riskLevel: PredictionResult["riskLevel"] = "Thấp";
  let riskColor = "#16a34a";
  if (probability >= 0.7)      { riskLevel = "Rất cao";    riskColor = "#991b1b"; }
  else if (probability >= 0.5) { riskLevel = "Cao";        riskColor = "#dc2626"; }
  else if (probability >= 0.3) { riskLevel = "Trung bình"; riskColor = "#d97706"; }
  else                         { riskLevel = "Thấp";      riskColor = "#16a34a"; }

  // Confidence
  let confidence: PredictionResult["confidence"] = "Cao";
  if (outOfDomain) confidence = "Thấp";
  else if (probability > 0.85 || probability < 0.15) confidence = "Cao";
  else confidence = "Trung bình";

  // Contribution breakdown for explainability
  const contributions = [
    { factor: "Nhiệt độ",     contribution: 0.061 * (temp - 25),             direction: "↑" },
    { factor: "Độ ẩm",        contribution: -0.044 * (humidity - 80),       direction: "↓" },
    { factor: "Gió",          contribution: 0.131 * (wind - 3),             direction: "↑" },
    { factor: "Mưa",          contribution: -0.381 * Math.log1p(rain),      direction: "↓" },
    { factor: "Số ngày khô",  contribution: 0.092 * Math.log1p(daysSinceRain), direction: "↑" },
    { factor: "Vùng (drift)", contribution: REGION_OFFSET[region],           direction: region === "south" ? "↑" : "→" },
  ];

  return {
    probability,
    riskLevel,
    riskColor,
    riskScore: Math.round(probability * 100),
    confidence,
    outOfDomain,
    warning: outOfDomain
      ? "Đầu vào ngoài miền huấn luyện — dự báo có thể không đáng tin"
      : undefined,
    contributions,
  };
}

function isOutOfDomain(input: PredictionInput): boolean {
  return (
    input.temp < TRAINING_DOMAIN.temp[0] || input.temp > TRAINING_DOMAIN.temp[1] ||
    input.humidity < TRAINING_DOMAIN.humidity[0] || input.humidity > TRAINING_DOMAIN.humidity[1] ||
    input.wind < TRAINING_DOMAIN.wind[0] || input.wind > TRAINING_DOMAIN.wind[1] ||
    input.rain > TRAINING_DOMAIN.rain[1] ||
    input.daysSinceRain > TRAINING_DOMAIN.daysSinceRain[1]
  );
}

// Compute color for a cell on the map given its baseRisk + current input
export function cellColor(probability: number): string {
  if (probability >= 0.7)      return "#991b1b"; // very high — dark red
  if (probability >= 0.5)      return "#dc2626"; // high — red
  if (probability >= 0.3)      return "#f59e0b"; // medium — amber
  if (probability >= 0.15)     return "#84cc16"; // low-medium — lime
  return "#22c55e";                                   // low — green
}

export function cellOpacity(probability: number): number {
  // 0.25 → 0.9
  return 0.25 + probability * 0.65;
}
