// Shared data for the wildfire prediction research website.
// Updated for v8.8 production model with weather anomalies + land cover + cold-start OOD findings.

export const PROJECT = {
  title:
    "HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN BẰNG MÔ HÌNH HỌC MÁY",
  shortTitle: "Wildfire Forecasting System v8.8",
  author: "Hoàng Bảo Lâm",
  year: "2026",
  competition: "Cuộc thi Khoa học Kĩ thuật — Sở GD&ĐT Hà Nội",
  level: "Cấp học: THCS",
  field: "Phần mềm hệ thống",
  dataPoints: 964_040,
  cells: 710,
  days: 2_191,
  period: "2018 – 2023",
  seeds: 5,
  spatialSplits: 5,
  temporalSplits: 3,
  school: "Trường THCS&THPT LÊ QUÝ ĐÔN",
  teacher: "Lê Xuân Tân",
  class: "9E4",
  modelVersion: "v9.4 forest fire (Optuna + spatial lag + real MODIS NDVI)",
  featureCount: 75,
};

export const KEY_FINDINGS = [
  {
    label: "AUROC cold-start OOD — cháy RỪNG (v9.4)",
    value: "97,9%",
    sub: "Brier 0,035 · 75 features · Optuna tuned",
    tone: "good" as const,
    note: "Full panel 490K rows + real MODIS NDVI + spatial lag + Optuna hyperparameter tuning",
  },
  {
    label: "Giảm over-prediction Nam",
    value: "−64%",
    sub: "Nguy hiểm cells: 4922 → 1782",
    tone: "good" as const,
    note: "Khi lọc chỉ giữ cháy rừng, miền Nam predictions giảm 64% — khớp với bản đồ chuyên gia",
  },
  {
    label: "Cải thiện từ v8.6 ban đầu",
    value: "+33,7 điểm %",
    sub: "58,4% → 92,1% AUROC",
    tone: "good" as const,
    note: "Nhờ weather anomalies + land cover + NDVI + multi-task filtering",
  },
  {
    label: "Phương pháp multi-task",
    value: "2 models",
    sub: "fire_any + fire_high (forest)",
    tone: "warning" as const,
    note: "Phân biệt cháy nông nghiệp vs cháy rừng qua confidence — proxy cho FRP (fire radiative power)",
  },
];

// SC-PMH E4: Multiple spatial boundaries (cold-start, 5 seeds each)
export const AUROC_SPLITS = [
  // E4 results — anomaly improvement consistent at 4 of 5 boundaries
  { split: "Boundary lat=12°", baseline: 0.8764, anomaly: 0.8816, delta: 0.0051, t_stat: 8.84, sig: true },
  { split: "Boundary lat=13°", baseline: 0.8853, anomaly: 0.8899, delta: 0.0047, t_stat: 9.84, sig: true },
  { split: "Boundary lat=14°", baseline: 0.9033, anomaly: 0.9088, delta: 0.0056, t_stat: 17.08, sig: true },
  { split: "Boundary lat=15°", baseline: 0.9137, anomaly: 0.9172, delta: 0.0034, t_stat: 5.68, sig: true },
  { split: "Boundary lat=16°", baseline: 0.9175, anomaly: 0.9176, delta: 0.0001, t_stat: 0.23, sig: false },
];

// SC-PMH E5: Multiple temporal splits (cold-start, 5 seeds each)
export const TEMPORAL_SPLITS = [
  { split: "T1 (train 2018-21, test 2023)", baseline: 0.9377, anomaly: 0.9400, delta: 0.0023, t_stat: 13.49, sig: true },
  { split: "T3 (cumulative forward)",      baseline: 0.9381, anomaly: 0.9421, delta: 0.0040, t_stat: 10.25, sig: true },
  { split: "T4 (6-month forward)",         baseline: 0.9374, anomaly: 0.9449, delta: 0.0075, t_stat: 15.68, sig: true },
];

// SC-PMH E6: Anomaly ablation — which features matter most
export const ANOMALY_ABLATION = [
  { feature: "wind_anom_doy",      label: "Bất thường gió (theo ngày)",    delta_auroc: -0.0050, t_stat: -11.07, sig: true, importance: "DOMINANT" },
  { feature: "rh_anom_roll7",      label: "Bất thường độ ẩm (7 ngày)",    delta_auroc: -0.0015, t_stat: -2.26,  sig: true, importance: "high (PR-AUC)" },
  { feature: "precip_anom_roll7",  label: "Bất thường mưa (7 ngày)",       delta_auroc: -0.0011, t_stat: -2.52, sig: true, importance: "moderate" },
  { feature: "t2m_anom_roll7",    label: "Bất thường nhiệt độ (7 ngày)", delta_auroc: -0.0010, t_stat: -1.21, sig: false, importance: "low" },
  { feature: "t2m_anom_doy",       label: "Bất thường nhiệt độ (theo ngày)", delta_auroc: -0.0018, t_stat: -1.40, sig: false, importance: "low" },
  { feature: "rh_anom_doy",        label: "Bất thường độ ẩm (theo ngày)",  delta_auroc: 0.0000,  t_stat: 0.04,  sig: false, importance: "none" },
  { feature: "precip_anom_doy",    label: "Bất thường mưa (theo ngày)",    delta_auroc: -0.0003, t_stat: -0.76, sig: false, importance: "none" },
  { feature: "dry_anomaly_composite", label: "Composite khô hạn (đã loại bỏ)", delta_auroc: -0.0001, t_stat: -0.14, sig: false, importance: "none — dropped per E6" },
];

// SC-PMH E7: Recall at same warning budget (operational value)
export const WARNING_BUDGET = [
  { budget: "1%",  baseline_recall: 0.0250, anomaly_recall: 0.0243, baseline_fires: 67,  anomaly_fires: 64,  delta: "-3 fires", sig: false },
  { budget: "5%",  baseline_recall: 0.0738, anomaly_recall: 0.0774, baseline_fires: 197, anomaly_fires: 200, delta: "+3 fires", sig: false },
  { budget: "10%", baseline_recall: 0.1405, anomaly_recall: 0.1527, baseline_fires: 400, anomaly_fires: 418, delta: "+18 fires", sig: true, t_stat: 8.07 },
  { budget: "20%", baseline_recall: 0.2931, anomaly_recall: 0.3095, baseline_fires: 875, anomaly_fires: 910, delta: "+35 fires", sig: true, t_stat: 2.39 },
  { budget: "30%", baseline_recall: 0.5831, anomaly_recall: 0.5894, baseline_fires: 1331, anomaly_fires: 1367, delta: "+37 fires", sig: false, t_stat: 0.91 },
];

// SC-PMH Phase 3: Failure mode analysis
export const FAILURE_MODE = {
  threshold: 0.20,
  classification_counts: { TP: 2458, FP: 5889, FN: 588, TN: 20777 },
  precision: 0.344,
  recall: 0.840,
  weather_by_class: [
    { class: "TP", label: "Cảnh báo đúng (True Positive)",  t2m: 27.1, rh: 66.7, wind: 2.06, wind_anom: 0.04,  p_fire: 0.554, n: 2458, color: "#2ca02c" },
    { class: "FP", label: "Cảnh báo sai (False Positive)",  t2m: 27.5, rh: 67.2, wind: 2.32, wind_anom: 0.08,  p_fire: 0.458, n: 5889, color: "#d62728" },
    { class: "FN", label: "Bỏ sót cháy (False Negative)",  t2m: 23.3, rh: 81.9, wind: 3.48, wind_anom: 0.54,  p_fire: 0.089, n: 588,  color: "#ff7f0e" },
    { class: "TN", label: "Đúng không cảnh báo (True Negative)", t2m: 26.7, rh: 84.6, wind: 3.18, wind_anom: -0.03, p_fire: 0.013, n: 20777, color: "#cccccc" },
  ],
  insight: "FN xảy ra khi nhiệt THẤP (23,3°C) + gió BẤT THƯỜNG CAO (anomaly +0,54) — cơ chế wind-driven fires mà mô hình thiên về nhiệt độ chưa nắm bắt.",
  future_direction: "Thêm feature tương tác wind × temperature_anomaly, hoặc model riêng cho wind-driven regime.",
};

// AUROC by forecast horizon (v8.7.1 leakage-free, v8.7.2 production with anomalies)
export const HORIZON_AUROC = [
  { horizon: "t+1", xgboost: 0.982, mlp: 0.879, physmlp: 0.879, climatology: 0.940 },
  { horizon: "t+3", xgboost: 0.967, mlp: 0.870, physmlp: 0.870, climatology: 0.938 },
  { horizon: "t+7", xgboost: 0.948, mlp: 0.865, physmlp: 0.865, climatology: 0.935 },
];

// Physics violation rates
export const PHYSICS_VIOLATION = [
  { relation: "Nhiệt độ ↑ → cháy ↑", xgboost: 4.4,  mlp: 84.0, physmlp: 82.9 },
  { relation: "Độ ẩm ↑ → cháy ↓",    xgboost: 5.5,  mlp: 15.7, physmlp: 9.9  },
  { relation: "Gió ↑ → cháy ↑",      xgboost: 90.4, mlp: 0.0,  physmlp: 0.0  },
  { relation: "Mưa ↑ → cháy ↓",      xgboost: 51.2, mlp: 34.8, physmlp: 32.7 },
];

// Spatial feature experiments
export const SPATIAL_FEATURES = [
  { setup: "Chỉ thời tiết",                       auroc: 0.5819, tone: "danger" as const },
  { setup: "+ Vĩ/Kinh độ (continuous)",           auroc: 0.8113, tone: "good" as const },
  { setup: "+ 10 dải vĩ độ (binned)",            auroc: 0.5596, tone: "danger" as const },
  { setup: "+ Nhóm tọa độ thô",                  auroc: 0.5479, tone: "danger" as const },
];

export const HYPOTHESES = [
  { id: "H1", text: "Phân phối thời tiết giữa miền Bắc và miền Nam khác nhau.", confirmed: true },
  { id: "H2", text: "Quan hệ giữa điều kiện thời tiết và xác suất cháy thay đổi theo vùng.", confirmed: true },
  { id: "H3", text: "Khác biệt giữa MODIS và VIIRS có thể làm thay đổi kết luận về độ tin cậy.", confirmed: true },
  { id: "H4", text: "Có cấu trúc không gian chưa được biểu diễn đầy đủ bằng các biến thời tiết.", confirmed: true },
  { id: "H5", text: "Tự tương quan không gian có thể làm đánh giá cùng miền trở nên lạc quan.", confirmed: true },
];

export const DATA_SOURCES = [
  {
    name: "NASA POWER",
    role: "Thời tiết tái phân tích",
    variables: "T2M, RH2M, WS10M, PRECTOTCORR + biến dẫn xuất",
    url: "https://power.larc.nasa.gov/",
    citation: "[5]",
    color: "#0284c7",
  },
  {
    name: "MODIS Active Fire (Collection 6)",
    role: "Nhãn chính (ground truth)",
    variables: "Điểm cháy 1 km, vệ tinh Terra/Aqua",
    url: "https://doi.org/10.1016/j.rse.2016.02.054",
    citation: "[2]",
    color: "#dc2626",
  },
  {
    name: "VIIRS 375 m Active Fire",
    role: "Kiểm chứng độc lập",
    variables: "Điểm cháy 375 m, vệ tinh Suomi-NPP",
    url: "https://doi.org/10.1016/j.rse.2013.12.028",
    citation: "[3]",
    color: "#93761f",
  },
];

export const MODELS = [
  { name: "Logistic Regression", type: "Tuyến tính (baseline)", role: "Mô hình cơ sở", citation: "[14]" },
  { name: "Random Forest",       type: "Bagging ensemble",      role: "Tham chiếu",   citation: "[8]" },
  { name: "XGBoost",             type: "Gradient boosting",     role: "Mô hình chính cho chẩn đoán", citation: "[1], [9]", primary: true },
  { name: "MLP",                 type: "Mạng nơ-ron truyền thẳng", role: "Tham chiếu thần kinh", citation: "[14]" },
  { name: "PhysMLP",             type: "MLP + ràng buộc vật lý",    role: "Kiểm tra tính nhất quán",   citation: "[6]" },
];

export const SPLITS = [
  { name: "IID (theo nhóm ô)",    train: "Ô ngẫu nhiên",       test: "Ô ngẫu nhiên còn lại", purpose: "Hiệu năng trong miền (lạc quan)", severity: "easy" },
  { name: "Temporal OOD",         train: "2018–2022",           test: "2023",                  purpose: "Khả năng chuyển thời gian",        severity: "moderate" },
  { name: "Bắc → Nam",            train: "Phía Bắc",            test: "Phía Nam",              purpose: "Chuyển miền không gian khó nhất",  severity: "hard" },
  { name: "Nam → Trung",          train: "Phía Nam",            test: "Miền Trung",            purpose: "Chuyển miền vừa phải",            severity: "moderate" },
  { name: "Trung+Nam → Bắc",      train: "Trung + Nam",         test: "Phía Bắc",              purpose: "Chuyển miền ngược",                severity: "moderate" },
];

export const REFERENCES: { num: string; text: string; url: string }[] = [
  { num: "[1]",  text: "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD, 785–794.", url: "https://doi.org/10.1145/2939672.2939785" },
  { num: "[2]",  text: "Giglio, L., Schroeder, W., & Justice, C. O. (2016). The Collection 6 MODIS active fire detection algorithm and fire products. RSE, 178, 31–41.", url: "https://doi.org/10.1016/j.rse.2016.02.054" },
  { num: "[3]",  text: "Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. A. (2014). The New VIIRS 375 m active fire detection data product. RSE, 143, 85–96.", url: "https://doi.org/10.1016/j.rse.2013.12.028" },
  { num: "[4]",  text: "Justice, C. O., et al. (2002). The MODIS fire products. RSE, 83(1–2), 244–262.", url: "https://doi.org/10.1016/S0034-4257(02)00076-7" },
  { num: "[5]",  text: "NASA Langley Research Center. NASA POWER Project — Prediction Of Worldwide Energy Resources.", url: "https://power.larc.nasa.gov/" },
  { num: "[6]",  text: "Karniadakis, G. E., et al. (2021). Physics-informed machine learning. Nature Reviews Physics, 3, 422–440.", url: "https://doi.org/10.1038/s42254-021-00314-5" },
  { num: "[7]",  text: "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML, 625–632.", url: "https://doi.org/10.1145/1102351.1102430" },
  { num: "[8]",  text: "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.", url: "https://doi.org/10.1023/A:1010933404324" },
  { num: "[9]",  text: "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.", url: "https://doi.org/10.1214/aos/1013203451" },
  { num: "[10]", text: "Anselin, L. (1995). Local Indicators of Spatial Association — LISA. Geographical Analysis, 27(2), 93–115.", url: "https://doi.org/10.1111/j.1538-5832.1995.tb00338.x" },
  { num: "[11]", text: "Shen, Z., et al. (2021). Towards Out-of-Distribution Generalization: A Survey. arXiv:2108.13624.", url: "https://arxiv.org/abs/2108.13624" },
  { num: "[12]", text: "Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929.", url: "https://doi.org/10.1111/ecog.02881" },
  { num: "[13]", text: "Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874.", url: "https://doi.org/10.1016/j.patrec.2005.10.010" },
  { num: "[14]", text: "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.", url: "https://doi.org/10.1007/978-0-387-84858-7" },
];

export const NORTH_SOUTH_STATS = [
  { variable: "Nhiệt độ 2 m (°C)", north: 24.85, south: 26.40, cohens_d: 0.445, direction: "↑ ở Nam" },
  { variable: "Độ ẩm tương đối (%)", north: 82.81, south: 80.73, cohens_d: 0.209, direction: "↓ ở Nam" },
  { variable: "Tốc độ gió (m/s)", north: 3.66, south: 3.19, cohens_d: 0.203, direction: "↓ ở Nam" },
  { variable: "Số ngày không mưa", north: 2.60, south: 4.57, cohens_d: 0.71, direction: "↑ ở Nam" },
];

export const DIAGNOSTIC_RESULTS = [
  { experiment: "Dịch chuyển phân phối (Cohen's d)", result: "0,20–0,45", supports: "H1", color: "#93761f" },
  { experiment: "Thay đổi quan hệ điều kiện P(cháy|x)", result: "Hai miền không trùng", supports: "H2", color: "#9b473f" },
  { experiment: "Case-control 1:1 (kiểm soát tỷ lệ cháy)", result: "0,5821 ± 0,0017 (≈ gốc)", supports: "Bác bỏ prevalence", color: "#47805a" },
  { experiment: "Moran's I (tự tương quan không gian)", result: "0,186 ± 0,215", supports: "H5", color: "#a48343" },
  { experiment: "Thêm lat/lon liên tục vào XGBoost", result: "0,8113 (tăng mạnh)", supports: "H4 — có tín hiệu vị trí", color: "#4d6e8e" },
];

// === MAP DATA: 273 ô lưới 0.5° trên Việt Nam ===
// Bounding box Việt Nam: khoảng 8.5°N–23.5°N, 102°E–110°E
// Grid 0.5° → ~31 hàng × ~17 cột = 527 ô max, sau lọc đất liền còn 273 ô
// Sinh các ô lưới mô phỏng (không phải dữ liệu thật, chỉ để minh hoạ)

export interface FireCell {
  id: number;
  lat: number;
  lon: number;
  region: "north" | "central" | "south";
  // Mô phỏng xác suất cháy cơ sở (dựa trên đặc trưng vùng)
  baseRisk: number;
  // Khi user chọn ngày và mô hình, system sẽ tính risk động
}

function generateVietnamGrid(): FireCell[] {
  const cells: FireCell[] = [];
  let id = 0;

  // Vietnam bounding box (rút gọn cho map rõ ràng)
  const minLat = 8.5, maxLat = 23.5;
  const minLon = 102.0, maxLon = 110.0;
  const step = 0.5;

  for (let lat = minLat; lat <= maxLat; lat += step) {
    for (let lon = minLon; lon <= maxLon; lon += step) {
      // Bỏ các ô ngoài biên Việt Nam (đại dương, Campuchia, Lào, Trung Quốc)
      // Sử dụng polygon xấp xỉ lãnh thổ Việt Nam
      if (!isInVietnam(lat, lon)) continue;

      const region = lat >= 18.5 ? "north" : lat >= 13.5 ? "central" : "south";

      // Base risk mô phỏng: Nam cao hơn Bắc (vì mùa khô Tây Nguyên, gió Lào)
      // Bắc: 0.15–0.35; Trung: 0.30–0.55; Nam: 0.45–0.75
      let baseRisk: number;
      if (region === "north") {
        baseRisk = 0.15 + Math.sin((lon - 102) / 8 * Math.PI) * 0.10 + Math.random() * 0.10;
      } else if (region === "central") {
        // Tây Nguyên (lon 107–109, lat 11.5–14.5) cao nhất
        const tbn = lat >= 11.5 && lat <= 14.5 && lon >= 107 && lon <= 109 ? 0.20 : 0;
        baseRisk = 0.30 + tbn + Math.random() * 0.10;
      } else {
        baseRisk = 0.45 + Math.random() * 0.15;
      }
      baseRisk = Math.max(0.05, Math.min(0.85, baseRisk));

      cells.push({
        id: id++,
        lat: +lat.toFixed(2),
        lon: +lon.toFixed(2),
        region,
        baseRisk: +baseRisk.toFixed(3),
      });
    }
  }
  return cells;
}

// Polygon biên giới đất liền Việt Nam — đi theo chiều kim đồng hồ
// từ tây bắc (Lai Châu) qua đông bắc (Móng Cái), dọc bờ Đông vào Nam,
// mũi Cà Mau, dọc biên giới Tây Campuchia/Lào ngược lên Bắc.
// Polygon này KHÔNG bao gồm vùng biển (để tránh hiện cháy trên Biển Đông).
function isInVietnam(lat: number, lon: number): boolean {
  const polygon: [number, number][] = [
    // Tây Bắc (Lai Châu → Lào Cai → Hà Giang)
    [22.85, 103.55], [22.55, 103.75], [22.10, 103.55], [21.55, 103.50],
    [21.10, 103.55], [20.65, 104.05], [20.10, 104.05], [19.55, 104.55],
    [19.05, 104.10], [18.55, 104.10], [18.05, 104.55], [17.55, 105.55],
    [17.05, 106.05], [16.55, 106.55], [16.05, 107.05], [15.55, 107.45],
    [15.05, 107.50], [14.55, 107.10], [14.05, 107.05], [13.55, 106.05],
    [13.05, 105.55], [12.05, 105.05], [11.05, 105.05], [10.55, 104.95],
    // Tây Nam (Châu Đốc → Hà Tiên → mũi Cà Mau)
    [10.55, 104.65], [10.45, 104.55], [10.30, 104.40], [10.05, 104.55],
    [9.55, 104.75], [9.05, 104.75], [8.65, 104.65], [8.55, 104.85], [8.95, 105.05],
    [9.35, 105.35], [9.85, 105.55],
    // Bờ Đông Nam Bộ (Cần Thơ → Vũng Tàu → HCM)
    [10.15, 105.85], [10.45, 106.45], [10.75, 106.85], [10.85, 107.30],
    [10.95, 107.55], [10.85, 107.95],
    // Bờ Đông Nam Trung Bộ (Phan Thiết → Phan Rang → Nha Trang)
    [11.20, 108.55], [11.55, 108.85], [11.95, 109.05], [12.35, 109.20],
    [12.75, 109.30], [13.15, 109.40], [13.55, 109.30], [13.95, 109.20],
    // Bờ Đông Bắc Trung Bộ (Quy Nhơn → Đà Nẵng → Đồng Hới)
    [14.45, 109.05], [14.95, 108.90], [15.45, 108.70], [15.95, 108.30],
    [16.45, 108.10], [16.95, 107.80], [17.45, 107.00], [17.95, 106.50],
    [18.45, 106.00], [18.95, 105.80], [19.45, 105.70], [19.95, 105.75],
    // Bờ Đông Bắc Bộ (Thanh Hóa → Hải Phòng → Quảng Ninh → Móng Cái)
    [20.45, 106.55], [20.75, 107.05], [20.95, 107.45], [21.10, 107.75],
    [21.30, 107.95], [21.55, 108.05],
    // Biên giới Trung Quốc (Móng Cái → Lạng Sơn → Hà Giang)
    [21.75, 107.80], [22.05, 107.50], [22.25, 107.20], [22.45, 106.80],
    [22.65, 106.50], [22.85, 106.00], [22.95, 105.50], [22.95, 105.00],
    [22.95, 104.50], [22.75, 104.25], [22.55, 104.00], [22.75, 103.80],
    [22.85, 103.55],
  ];

  // Ray casting: lat = y, lon = x. Mỗi đỉnh = [plat, plon] = [py, px].
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

export const VIETNAM_GRID: FireCell[] = generateVietnamGrid();

// Major Vietnam cities for map markers
export const VIETNAM_CITIES = [
  { name: "Hà Nội",     lat: 21.03, lon: 105.85, region: "north"   },
  { name: "Hải Phòng",  lat: 20.86, lon: 106.68, region: "north"   },
  { name: "Đà Nẵng",    lat: 16.05, lon: 108.20, region: "central" },
  { name: "Pleiku",     lat: 13.96, lon: 108.00, region: "central" },
  { name: "Nha Trang",  lat: 12.24, lon: 109.19, region: "central" },
  { name: "TP.HCM",     lat: 10.78, lon: 106.70, region: "south"   },
  { name: "Cần Thơ",    lat: 10.03, lon: 105.79, region: "south"   },
];


// === Fire detection points (MODIS/VIIRS style) ===
export interface FirePoint {
  id: number;
  lat: number;
  lon: number;
  region: "north" | "central" | "south";
  confidence: number;
  brightness: number;
  frp: number;
  doy: number;
  source: "MODIS" | "VIIRS";
  baseRisk: number;
}

function generateFirePoints(): FirePoint[] {
  const points: FirePoint[] = [];
  let id = 0;
  const clusters = [
    { centerLat: 22.5, centerLon: 104.5, radius: 1.2, count: 18, region: "north" as const, seasonBias: [1, 120], frpRange: [10, 80] },
    { centerLat: 22.8, centerLon: 106.3, radius: 1.0, count: 14, region: "north" as const, seasonBias: [1, 100], frpRange: [15, 90] },
    { centerLat: 21.5, centerLon: 105.5, radius: 0.8, count: 10, region: "north" as const, seasonBias: [60, 150], frpRange: [8, 60] },
    { centerLat: 14.0, centerLon: 108.0, radius: 1.5, count: 28, region: "central" as const, seasonBias: [30, 150], frpRange: [20, 150] },
    { centerLat: 12.5, centerLon: 108.0, radius: 1.2, count: 22, region: "central" as const, seasonBias: [30, 150], frpRange: [15, 120] },
    { centerLat: 11.5, centerLon: 108.0, radius: 1.0, count: 16, region: "central" as const, seasonBias: [40, 140], frpRange: [10, 90] },
    { centerLat: 16.0, centerLon: 108.5, radius: 0.8, count: 12, region: "central" as const, seasonBias: [60, 180], frpRange: [8, 70] },
    { centerLat: 11.0, centerLon: 107.0, radius: 1.2, count: 24, region: "south" as const, seasonBias: [330, 365], frpRange: [12, 100] },
    { centerLat: 11.5, centerLon: 106.5, radius: 1.0, count: 18, region: "south" as const, seasonBias: [330, 365], frpRange: [10, 80] },
    { centerLat: 10.5, centerLon: 105.5, radius: 1.5, count: 20, region: "south" as const, seasonBias: [340, 100], frpRange: [8, 60] },
    { centerLat: 10.0, centerLon: 106.0, radius: 0.8, count: 10, region: "south" as const, seasonBias: [350, 90], frpRange: [6, 50] },
  ];

  for (const cluster of clusters) {
    let added = 0;
    let attempts = 0;
    // Sinh điểm cho đến khi đủ count hoặc vượt 4 lần count (tránh lặp vô hạn)
    while (added < cluster.count && attempts < cluster.count * 4) {
      attempts++;
      const angle = Math.random() * 2 * Math.PI;
      const r = Math.random() * cluster.radius;
      const lat = cluster.centerLat + r * Math.cos(angle) * 0.7;
      const lon = cluster.centerLon + r * Math.sin(angle);
      // Lọc: chỉ giữ lại điểm nằm trong lãnh thổ Việt Nam (tránh biển + nước láng giềng)
      if (!isInVietnam(lat, lon)) continue;
      const [startDoy, endDoy] = cluster.seasonBias;
      const doy = startDoy > endDoy
        ? (Math.random() < 0.5 ? startDoy + Math.floor(Math.random() * (365 - startDoy + 1)) : Math.floor(Math.random() * (endDoy + 1)))
        : startDoy + Math.floor(Math.random() * (endDoy - startDoy + 1));
      const confidence = 30 + Math.floor(Math.random() * 70);
      const brightness = 300 + Math.random() * 60;
      const [frpMin, frpMax] = cluster.frpRange;
      const frp = frpMin + Math.random() * (frpMax - frpMin);
      const source = Math.random() < 0.6 ? "MODIS" : "VIIRS";
      const baseRisk = Math.min(0.95, 0.3 + frp / 200);
      points.push({
        id: id++, lat: +lat.toFixed(3), lon: +lon.toFixed(3),
        region: cluster.region, confidence,
        brightness: +brightness.toFixed(1), frp: +frp.toFixed(1),
        doy, source, baseRisk: +baseRisk.toFixed(3),
      });
      added++;
    }
  }
  return points;
}

export const FIRE_POINTS: FirePoint[] = generateFirePoints();

// Preset scenarios for demo
export const SCENARIOS = [
  { name: "Mùa khô Tây Nguyên (tháng 3)", temp: 29, humidity: 65, wind: 4.0, rain: 0, daysSinceRain: 8,  month: 3 },
  { name: "Mùa mưa Bắc (tháng 7)",        temp: 28, humidity: 85, wind: 2.5, rain: 8, daysSinceRain: 1,  month: 7 },
  { name: "Mùa đông Bắc (tháng 1)",       temp: 16, humidity: 80, wind: 3.0, rain: 1, daysSinceRain: 4,  month: 1 },
  { name: "Đỉnh nóng Nam (tháng 4)",     temp: 34, humidity: 70, wind: 3.5, rain: 0, daysSinceRain: 10, month: 4 },
];

// === Heatmap thời gian ===
// Mô phỏng cường độ cháy (cell-days / 1000) theo tháng × vùng, dựa trên pattern thực tế tại VN:
// - Bắc:    cao nhất tháng 1-4 (mùa xuân khô), thấp tháng 5-10 (mưa), nhẹ hồi tháng 11-12
// - Trung:  cao tháng 3-5 và 7-8 (gió Lào), thấp tháng 9-12 (mùa mưa bão)
// - Nam:    cao nhất tháng 2-4 (đỉnh mùa khô), rất thấp tháng 6-11 (mùa mưa)
//
// Đơn vị: số cell-day cháy (mô phỏng, không phải số thật — chỉ để minh hoạ pattern mùa vụ)
export interface HeatmapPoint {
  month: number;       // 1–12
  monthLabel: string;   // "T1" – "T12"
  region: "north" | "central" | "south";
  regionLabel: string;
  value: number;        // số cell-day cháy (mô phỏng)
  intensity: number;    // 0–1, dùng cho màu
}

function buildHeatmapData(): HeatmapPoint[] {
  const months = Array.from({ length: 12 }, (_, i) => i + 1);
  const monthsLabel = months.map(m => `T${m}`);

  // Pattern mùa vụ cho từng vùng (tổng cell-day/tháng mô phỏng)
  // Tổng năm: Bắc ~3.200, Trung ~2.800, Nam ~4.500 (gần với tỷ lệ MODIS/VIIRS quan sát)
  const northPattern = [580, 520, 410, 280,  90,  50,  60,  80, 110, 200, 380, 440];
  const centralPattern = [180, 240, 410, 360, 240, 180, 320, 290, 150, 110, 160, 220];
  const southPattern = [620, 700, 720, 640, 280,  90,  60,  70,  90, 140, 280, 410];

  const max = Math.max(...northPattern, ...centralPattern, ...southPattern);

  const result: HeatmapPoint[] = [];
  for (let i = 0; i < 12; i++) {
    result.push({
      month: months[i],
      monthLabel: monthsLabel[i],
      region: "north",
      regionLabel: "Bắc",
      value: northPattern[i],
      intensity: northPattern[i] / max,
    });
    result.push({
      month: months[i],
      monthLabel: monthsLabel[i],
      region: "central",
      regionLabel: "Trung",
      value: centralPattern[i],
      intensity: centralPattern[i] / max,
    });
    result.push({
      month: months[i],
      monthLabel: monthsLabel[i],
      region: "south",
      regionLabel: "Nam",
      value: southPattern[i],
      intensity: southPattern[i] / max,
    });
  }
  return result;
}

export const HEATMAP_DATA: HeatmapPoint[] = buildHeatmapData();

// Tóm tắt thống kê cho heatmap
export const HEATMAP_STATS = {
  peakMonth: { north: 1, central: 3, south: 3 } as Record<"north" | "central" | "south", number>,
  lowMonth: { north: 6, central: 10, south: 7 } as Record<"north" | "central" | "south", number>,
  peakQuarter: { north: "T1–T2 (mùa xuân khô)", central: "T3–T4 (gió Lào)", south: "T2–T4 (mùa khô Tây Nguyên)" } as Record<"north" | "central" | "south", string>,
};

// Màu cho heatmap intensity
export function heatmapColor(intensity: number): string {
  // 0 (xanh nhạt) → 1 (đỏ đậm)
  if (intensity >= 0.85) return "#7f1d1d";
  if (intensity >= 0.7)  return "#991b1b";
  if (intensity >= 0.55) return "#dc2626";
  if (intensity >= 0.4)  return "#ef4444";
  if (intensity >= 0.25) return "#f59e0b";
  if (intensity >= 0.15) return "#fbbf24";
  if (intensity >= 0.08) return "#fde68a";
  return "#fef9c3";
}
