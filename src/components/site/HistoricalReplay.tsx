"use client";

import { useState, useMemo, useEffect } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  History, MapPin, Flame, AlertTriangle, Thermometer, Wind, CloudRain, Activity,
  Cpu, Clock, Layers, CheckCircle2, XCircle, Eye, EyeOff, Gauge, ArrowRight,
} from "lucide-react";

const MapContainer = dynamic(() => import("react-leaflet").then(m => m.MapContainer), { ssr: false });
const TileLayer    = dynamic(() => import("react-leaflet").then(m => m.TileLayer),    { ssr: false });
const CircleMarker = dynamic(() => import("react-leaflet").then(m => m.CircleMarker), { ssr: false });
const Popup        = dynamic(() => import("react-leaflet").then(m => m.Popup),         { ssr: false });

// Heatmap data types (matches the JSON structure produced by build_heatmap_data.py)
interface CellPrediction {
  lat: number; lon: number; p: number; fire: number;
}
interface DateData {
  date: string; n_cells: number; n_fires: number; p_max: number; p_mean: number;
  cells: CellPrediction[];
}
interface HeatmapData {
  metadata: {
    model: string;
    features: number;
    feature_set: string;
    train: string; val: string; test: string;
    metrics_south_cold_OOD: { AUROC: number; PR_AUC: number; Brier: number };
    key_innovation: string;
    most_important_feature: string;
    failure_mode: string;
    thresholds_for_risk_levels: { low: number; moderate: number; high: number; severe: number };
  };
  dates: DateData[];
}

const BASEMAPS = [
  { id: "osm", name: "OpenStreetMap", url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", attribution: '&copy; OpenStreetMap', maxZoom: 19 },
  { id: "esri-imagery", name: "Esri Vệ tinh", url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attribution: 'Esri', maxZoom: 19 },
] as const;

const RISK_LEVELS = [
  { name: "Thấp",       color: "#22c55e", range: "< 0.05" },
  { name: "Trung bình", color: "#f59e0b", range: "0.05–0.10" },
  { name: "Cao",        color: "#dc2626", range: "0.10–0.20" },
  { name: "Rất cao",    color: "#7f1d1d", range: "0.20–0.30" },
  { name: "Nguy hiểm",  color: "#4c1d95", range: "≥ 0.30" },
] as const;

function riskColor(p: number): string {
  if (p >= 0.30) return "#4c1d95";
  if (p >= 0.20) return "#7f1d1d";
  if (p >= 0.10) return "#dc2626";
  if (p >= 0.05) return "#f59e0b";
  return "#22c55e";
}
function riskLevel(p: number): string {
  if (p >= 0.30) return "Nguy hiểm";
  if (p >= 0.20) return "Rất cao";
  if (p >= 0.10) return "Cao";
  if (p >= 0.05) return "Trung bình";
  return "Thấp";
}

export default function HistoricalReplay() {
  const [data, setData] = useState<HeatmapData | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [showFirePoints, setShowFirePoints] = useState(true);
  const [showColdStartOverlay, setShowColdStartOverlay] = useState(true);
  const [basemap, setBasemap] = useState<typeof BASEMAPS[number]["id"]>("osm");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/data/heatmap_predictions.json")
      .then(r => r.json())
      .then((d: HeatmapData) => {
        setData(d);
        if (d.dates.length > 0) {
          // Pick top fire day by default
          const topFire = [...d.dates].sort((a, b) => b.n_fires - a.n_fires)[0];
          setSelectedDate(topFire.date);
        }
        setLoading(false);
      })
      .catch(e => { console.error("Failed to load heatmap data:", e); setLoading(false); });
  }, []);

  const currentDate = useMemo(() => data?.dates.find(d => d.date === selectedDate) ?? null, [data, selectedDate]);

  const stats = useMemo(() => {
    if (!currentDate) return null;
    const cells = currentDate.cells;
    const n_high = cells.filter(c => c.p >= 0.20).length;
    const n_severe = cells.filter(c => c.p >= 0.30).length;
    const fires_in_high = cells.filter(c => c.p >= 0.20 && c.fire === 1).length;
    const fires_total = currentDate.n_fires;
    const recall = fires_total > 0 ? fires_in_high / fires_total : 0;
    const precision = n_high > 0 ? fires_in_high / n_high : 0;
    return {
      n_cells: cells.length,
      n_fires: fires_total,
      n_high, n_severe,
      recall, precision,
      p_max: currentDate.p_max, p_mean: currentDate.p_mean,
    };
  }, [currentDate]);

  if (loading) {
    return (
      <section id="historical-replay" className="py-16 bg-stone-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="animate-pulse text-stone-600">Đang tải dữ liệu dự báo...</div>
        </div>
      </section>
    );
  }

  if (!data || !currentDate) {
    return (
      <section id="historical-replay" className="py-16 bg-stone-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-red-600">
          Không tải được dữ liệu. Hãy chạy `python3 scripts/build_heatmap_data.py`.
        </div>
      </section>
    );
  }

  return (
    <section id="historical-replay" className="py-16 bg-stone-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }} transition={{ duration: 0.5 }}
          className="text-center"
        >
          <Badge variant="outline" className="text-amber-700 border-amber-300 bg-amber-50">
            <History className="h-3 w-3 mr-1" />
            Khả năng 4: Tái dựng lịch sử
          </Badge>
          <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-stone-900">
            Bản đồ dự báo × điểm cháy quan sát
          </h2>
          <p className="mt-3 text-stone-600 max-w-3xl mx-auto">
            Chọn một ngày từ 2023 (test set, miền Nam Việt Nam) để xem dự báo của mô hình v8.7.2 (42 features, cold-start)
            so với điểm cháy MODIS quan sát được. Mỗi ô lưới 0,5° được tô màu theo xác suất dự báo.
            <span className="font-semibold text-amber-700"> Overlay Cold-start</span> cho biết vùng nào
            đã có trong huấn luyện (Bắc, in-distribution) vs. vùng chưa từng thấy (Nam, spatial OOD).
          </p>
        </motion.div>

        {/* Controls */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <Label className="text-stone-700 text-sm font-medium">Ngày dự báo</Label>
            <Select value={selectedDate} onValueChange={setSelectedDate}>
              <SelectTrigger className="w-full mt-1">
                <SelectValue placeholder="Chọn ngày" />
              </SelectTrigger>
              <SelectContent className="max-h-72">
                {[...data.dates].sort((a, b) => b.n_fires - a.n_fires).map(d => (
                  <SelectItem key={d.date} value={d.date}>
                    {d.date} · {d.n_fires} cháy · p_max={d.p_max.toFixed(2)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label className="text-stone-700 text-sm font-medium">Bản đồ nền</Label>
            <Select value={basemap} onValueChange={(v: any) => setBasemap(v)}>
              <SelectTrigger className="w-full mt-1"><SelectValue /></SelectTrigger>
              <SelectContent>
                {BASEMAPS.map(b => <SelectItem key={b.id} value={b.id}>{b.name}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-3 mt-6">
            <Switch checked={showFirePoints} onCheckedChange={setShowFirePoints} id="fires" />
            <Label htmlFor="fires" className="text-stone-700 text-sm flex items-center gap-1">
              <Flame className="h-4 w-4 text-red-600" /> Điểm cháy quan sát
            </Label>
          </div>
          <div className="flex items-center gap-3 mt-6">
            <Switch checked={showColdStartOverlay} onCheckedChange={setShowColdStartOverlay} id="cold" />
            <Label htmlFor="cold" className="text-stone-700 text-sm flex items-center gap-1">
              <Layers className="h-4 w-4 text-blue-600" /> Overlay Cold-start
            </Label>
          </div>
        </div>

        {/* Map + side panel */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Map */}
          <div className="lg:col-span-2">
            <Card className="overflow-hidden">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-stone-900">
                  <MapPin className="h-5 w-5" />
                  Bản đồ dự báo — {currentDate.date}
                </CardTitle>
                <CardDescription>
                  {stats?.n_cells} ô lưới 0,5° · {stats?.n_fires} điểm cháy MODIS · p_max={stats?.p_max.toFixed(3)} · p_mean={stats?.p_mean.toFixed(3)}
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div style={{ height: "560px", width: "100%" }}>
                  <MapContainer
                    center={[12.5, 107.5]} zoom={6}
                    style={{ height: "100%", width: "100%" }}
                    scrollWheelZoom={true}
                  >
                    <TileLayer
                      url={BASEMAPS.find(b => b.id === basemap)!.url}
                      attribution={BASEMAPS.find(b => b.id === basemap)!.attribution}
                    />
                    {/* Predicted probability cells */}
                    {currentDate.cells.map((c, i) => {
                      const fill = riskColor(c.p);
                      const isOOD = c.lat < 14;
                      // If cold-start overlay is on, dim in-distribution cells slightly to emphasize OOD
                      const opacity = showColdStartOverlay
                        ? (isOOD ? 0.85 : 0.35)  // dim North cells, highlight South cells
                        : 0.75;
                      return (
                        <CircleMarker
                          key={`cell-${i}`}
                          center={[c.lat, c.lon]}
                          radius={11}
                          pathOptions={{ color: "#1f2937", weight: 0.5, fillColor: fill, fillOpacity: opacity }}
                        >
                          <Popup>
                            <div className="text-sm">
                              <div className="font-bold mb-1">{c.lat.toFixed(2)}°N, {c.lon.toFixed(2)}°E</div>
                              <div>Trạng thái: <span className={isOOD ? "text-red-600 font-semibold" : "text-blue-600"}>
                                {isOOD ? "Cold-start (spatial OOD)" : "In-distribution"}
                              </span></div>
                              <div>Xác suất cháy: <span className="font-semibold" style={{ color: fill }}>{(c.p * 100).toFixed(1)}% — {riskLevel(c.p)}</span></div>
                              <div>Cháy quan sát (MODIS): {c.fire === 1 ? <span className="text-red-600 font-bold">CÓ</span> : <span className="text-stone-500">không</span>}</div>
                              <div className="text-xs text-stone-500 mt-1">Mô hình: XGBoost v8.7.2 (42 feats, cold-start)</div>
                            </div>
                          </Popup>
                        </CircleMarker>
                      );
                    })}
                    {/* Fire points overlay */}
                    {showFirePoints && currentDate.cells.filter(c => c.fire === 1).map((c, i) => (
                      <CircleMarker
                        key={`fire-${i}`}
                        center={[c.lat, c.lon]}
                        radius={6}
                        pathOptions={{ color: "#000000", weight: 2, fillColor: "#ef4444", fillOpacity: 1 }}
                      >
                        <Popup>
                          <div className="text-xs">
                            <div className="font-bold">🔥 MODIS Active Fire</div>
                            <div>{c.lat.toFixed(2)}°N, {c.lon.toFixed(2)}°E</div>
                            <div>Xác suất dự báo: {(c.p * 100).toFixed(1)}%</div>
                          </div>
                        </Popup>
                      </CircleMarker>
                    ))}
                  </MapContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Side panel: stats + classification */}
          <div className="space-y-4">
            {/* Stats */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-stone-900 flex items-center gap-2 text-base">
                  <Activity className="h-4 w-4" /> Phân loại dự báo (p ≥ 0,20)
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {stats && (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-green-50 border border-green-200 rounded p-2">
                        <div className="text-green-700 font-bold text-lg">{stats.n_high}</div>
                        <div className="text-stone-600 text-xs">Ô được cảnh báo</div>
                      </div>
                      <div className="bg-red-50 border border-red-200 rounded p-2">
                        <div className="text-red-700 font-bold text-lg">{stats.n_fires}</div>
                        <div className="text-stone-600 text-xs">Tổng fires thực tế</div>
                      </div>
                    </div>
                    <div className="border-t pt-2 space-y-1">
                      <div className="flex justify-between">
                        <span className="text-stone-600">Recall:</span>
                        <span className="font-mono font-bold text-stone-900">{(stats.recall * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-stone-600">Precision:</span>
                        <span className="font-mono font-bold text-stone-900">{(stats.precision * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-stone-600">Severe (p≥0.30):</span>
                        <span className="font-mono font-bold text-purple-700">{stats.n_severe}</span>
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* Risk legend */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-stone-900 flex items-center gap-2 text-base">
                  <Gauge className="h-4 w-4" /> Cấp nguy cơ
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {RISK_LEVELS.map(r => (
                  <div key={r.name} className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded" style={{ background: r.color, border: "1px solid #1f2937" }} />
                    <span className="font-semibold text-stone-700">{r.name}</span>
                    <span className="text-stone-500 text-xs ml-auto">{r.range}</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Cold-start indicator */}
            {showColdStartOverlay && (
              <Card className="border-blue-300 bg-blue-50">
                <CardHeader className="pb-2">
                  <CardTitle className="text-blue-900 flex items-center gap-2 text-base">
                    <AlertTriangle className="h-4 w-4" /> Cold-start indicator
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-xs text-blue-900">
                  <div className="flex items-start gap-2">
                    <div className="w-3 h-3 rounded mt-0.5" style={{ background: "rgba(255,0,0,0.6)" }} />
                    <div>
                      <div className="font-semibold">Vùng spatial OOD (lat &lt; 14°)</div>
                      <div className="text-blue-700">Miền Nam — mô hình chưa từng thấy các ô này trong huấn luyện.
                      Dự báo ở đây có độ tin cậy THẤP HƠN. AUROC thực = 0,904 (vs 0,982 in-distribution).</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-3 h-3 rounded mt-0.5" style={{ background: "rgba(0,0,0,0.3)" }} />
                    <div>
                      <div className="font-semibold">Vùng in-distribution (lat ≥ 14°)</div>
                      <div className="text-blue-700">Miền Bắc — đã có trong huấn luyện. Dự báo đáng tin hơn.</div>
                    </div>
                  </div>
                  <div className="border-t border-blue-200 pt-2 mt-2">
                    <div className="font-semibold mb-1">Cơ chế thất bại đã biết:</div>
                    <div className="text-blue-800">{data.metadata.failure_mode}</div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Methodology note */}
        <div className="mt-8 bg-stone-100 border border-stone-200 rounded-lg p-4 text-stone-700 text-sm">
          <div className="flex items-start gap-2">
            <Cpu className="h-5 w-5 text-stone-500 mt-0.5 shrink-0" />
            <div>
              <div className="font-semibold text-stone-900 mb-1">Phương pháp luận:</div>
              <p>
                Mô hình <span className="font-mono">XGBoost v8.7.2</span> được huấn luyện trên
                <strong> miền Bắc Việt Nam (lat ≥ 14°) 2018-2021</strong>, hiệu chỉnh bằng isotonic regression
                trên validation Bắc 2022. Đánh giá trên <strong>miền Nam 2023 (spatial OOD)</strong>.
                Features: 35 cold-baseline (weather + lags + rolling stats) + 7 weather anomalies
                (departure from per-cell day-of-year climatology từ 2018-2021). Không dùng fire-history
                features → áp dụng được cho bất kỳ ô nào kể cả ô chưa từng thấy.
              </p>
              <p className="mt-2">
                Metrics: AUROC = 0,904 · PR-AUC = 0,443 · Brier = 0,074 · Recall@0,20 = 84,0% · Precision@0,20 = 34,4%.
                Đóng góp chính: <span className="font-semibold">+0,0056 AUROC</span> từ 7 weather anomalies
                (t = 17,08, 5 seeds), tương đương <span className="font-semibold">+18 fires/year</span> tại
                cùng ngân sách cảnh báo 10% cells/ngày.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
