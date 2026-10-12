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
  History, MapPin, Flame, AlertTriangle, Activity,
  Cpu, Layers, Gauge, TreePine, Flame as FlameIcon,
} from "lucide-react";

const MapContainer = dynamic(() => import("react-leaflet").then(m => m.MapContainer), { ssr: false });
const TileLayer    = dynamic(() => import("react-leaflet").then(m => m.TileLayer),    { ssr: false });
const CircleMarker = dynamic(() => import("react-leaflet").then(m => m.CircleMarker), { ssr: false });
const Popup        = dynamic(() => import("react-leaflet").then(m => m.Popup),         { ssr: false });

// Heatmap data types (matches the JSON structure produced by build_heatmap_data.py + v9.0 multi-task)
interface CellPrediction {
  lat: number; lon: number;
  p: number;          // PRIMARY: forest fire probability (high-confidence)
  p_any: number;      // SECONDARY: any fire probability
  fire: number;       // forest fire observed (high-conf)
  fire_any: number;   // any fire observed (MODIS detection)
}
interface DateData {
  date: string; n_cells: number;
  n_fires: number;        // forest fire count
  n_fires_any: number;    // any fire count
  p_max: number; p_mean: number;       // forest fire stats
  p_any_max: number; p_any_mean: number; // any fire stats
  cells: CellPrediction[];
}
interface HeatmapData {
  metadata: {
    model: string;
    topic?: string;
    features: number;
    feature_set: string;
    train: string; val: string; test: string;
    label_definition?: string;
    metrics_south_cold_OOD_forest_fire?: { AUROC: number; PR_AUC: number; Brier: number; pos_rate_south: number };
    metrics_south_cold_OOD_any_fire?: { AUROC: number; PR_AUC: number; Brier: number; pos_rate_south: number };
    metrics_south_cold_OOD?: { AUROC: number; PR_AUC: number; Brier: number };
    key_innovation?: string;
    key_finding?: string;
    most_important_feature?: string;
    failure_mode?: string;
    thresholds_for_risk_levels: { low: number; moderate: number; high: number; severe: number };
  };
  dates: DateData[];
}

type PredictionMode = "forest" | "any";

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
  const [hrPredMode, setHrPredMode] = useState<PredictionMode>("forest");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/data/heatmap_predictions.json")
      .then(r => r.json())
      .then((d: HeatmapData) => {
        setData(d);
        if (d.dates.length > 0) {
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
    const pField = hrPredMode === "forest" ? "p" : "p_any";
    const fireField = hrPredMode === "forest" ? "fire" : "fire_any";
    const n_high = cells.filter(c => c[pField] >= 0.20).length;
    const n_severe = cells.filter(c => c[pField] >= 0.30).length;
    const fires_in_high = cells.filter(c => c[pField] >= 0.20 && c[fireField] === 1).length;
    const fires_total = hrPredMode === "forest" ? currentDate.n_fires : currentDate.n_fires_any;
    const recall = fires_total > 0 ? fires_in_high / fires_total : 0;
    const precision = n_high > 0 ? fires_in_high / n_high : 0;
    return {
      n_cells: cells.length,
      n_fires: fires_total,
      n_high, n_severe,
      recall, precision,
      p_max: hrPredMode === "forest" ? currentDate.p_max : currentDate.p_any_max,
      p_mean: hrPredMode === "forest" ? currentDate.p_mean : currentDate.p_any_mean,
    };
  }, [currentDate, hrPredMode]);

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

  const pField = hrPredMode === "forest" ? "p" : "p_any";
  const fireField = hrPredMode === "forest" ? "fire" : "fire_any";
  const metrics = hrPredMode === "forest"
    ? data.metadata.metrics_south_cold_OOD_forest_fire
    : data.metadata.metrics_south_cold_OOD_any_fire;

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
            Chọn một ngày từ 2023 (test set, miền Nam Việt Nam) để xem dự báo của mô hình v9.0 multi-task
            (61 features, cold-start) so với điểm cháy MODIS quan sát được. Mỗi ô lưới 0,5° được tô màu theo xác suất dự báo.
            <span className="font-semibold text-amber-700"> Overlay Cold-start</span> cho biết vùng nào
            đã có trong huấn luyện (Bắc) vs. chưa từng thấy (Nam, spatial OOD).
          </p>
        </motion.div>

        {/* Controls */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="md:col-span-1">
            <Label className="text-stone-700 text-sm font-medium">Chế độ dự báo</Label>
            <Select value={hrPredMode} onValueChange={(v: any) => setHrPredMode(v)}>
              <SelectTrigger className="w-full mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="forest">
                  <span className="flex items-center gap-2">
                    <TreePine className="h-3 w-3 text-green-700" />
                    Cháy RỪNG (high-confidence)
                  </span>
                </SelectItem>
                <SelectItem value="any">
                  <span className="flex items-center gap-2">
                    <FlameIcon className="h-3 w-3 text-orange-600" />
                    Cháy BẤT KỲ (all MODIS)
                  </span>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="md:col-span-2">
            <Label className="text-stone-700 text-sm font-medium">Ngày dự báo</Label>
            <Select value={selectedDate} onValueChange={setSelectedDate}>
              <SelectTrigger className="w-full mt-1">
                <SelectValue placeholder="Chọn ngày" />
              </SelectTrigger>
              <SelectContent className="max-h-72">
                {[...data.dates].sort((a, b) => b.n_fires - a.n_fires).map(d => (
                  <SelectItem key={d.date} value={d.date}>
                    {d.date} · {d.n_fires} cháy rừng · p_max={d.p_max.toFixed(2)}
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
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3 mt-6">
              <Switch checked={showFirePoints} onCheckedChange={setShowFirePoints} id="fires" />
              <Label htmlFor="fires" className="text-stone-700 text-sm flex items-center gap-1">
                <Flame className="h-4 w-4 text-red-600" /> Điểm cháy quan sát
              </Label>
            </div>
            <div className="flex items-center gap-3">
              <Switch checked={showColdStartOverlay} onCheckedChange={setShowColdStartOverlay} id="cold" />
              <Label htmlFor="cold" className="text-stone-700 text-sm flex items-center gap-1">
                <Layers className="h-4 w-4 text-blue-600" /> Overlay Cold-start
              </Label>
            </div>
          </div>
        </div>

        {/* Mode info banner */}
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}
          className={`mt-4 p-3 rounded-lg border text-sm ${
            hrPredMode === "forest"
              ? "bg-green-50 border-green-300 text-green-900"
              : "bg-orange-50 border-orange-300 text-orange-900"
          }`}
        >
          {hrPredMode === "forest" ? (
            <div className="flex items-start gap-2">
              <TreePine className="h-5 w-5 text-green-700 mt-0.5 shrink-0" />
              <div>
                <div className="font-semibold mb-1">Chế độ dự báo CHÁY RỪNG (production model)</div>
                <div>
                  Lọc chỉ giữ cháy high-confidence (FRP cao = cháy rừng thật), loại bỏ cháy nông nghiệp.
                  South predictions giảm 64% so với "cháy bất kỳ" — khớp với bản đồ chuyên gia.
                  Metrics South 2023 cold-start OOD:
                  AUROC = {metrics?.AUROC.toFixed(4)} · PR-AUC = {metrics?.PR_AUC.toFixed(4)} · Brier = {metrics?.Brier.toFixed(4)}.
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-2">
              <FlameIcon className="h-5 w-5 text-orange-600 mt-0.5 shrink-0" />
              <div>
                <div className="font-semibold mb-1">Chế độ dự báo CHÁY BẤT KỲ (so sánh)</div>
                <div>
                  Bao gồm mọi MODIS detection — kể cả cháy nông nghiệp (rơm rạ, mía).
                  South predictions CAO hơn vì Nam có nhiều cháy nông nghiệp.
                  Metrics South 2023 cold-start OOD:
                  AUROC = {metrics?.AUROC.toFixed(4)} · PR-AUC = {metrics?.PR_AUC.toFixed(4)} · Brier = {metrics?.Brier.toFixed(4)}.
                </div>
              </div>
            </div>
          )}
        </motion.div>

        {/* Map + side panel */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Map */}
          <div className="lg:col-span-2">
            <Card className="overflow-hidden">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-stone-900">
                  <MapPin className="h-5 w-5" />
                  Bản đồ dự báo {hrPredMode === "forest" ? "cháy RỪNG" : "cháy BẤT KỲ"} — {currentDate.date}
                </CardTitle>
                <CardDescription>
                  {stats?.n_cells} ô lưới 0,5° · {stats?.n_fires} điểm cháy · p_max={stats?.p_max.toFixed(3)} · p_mean={stats?.p_mean.toFixed(3)}
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
                    {/* Predicted probability cells (using selected mode's p) */}
                    {currentDate.cells.map((c, i) => {
                      const p = hrPredMode === "forest" ? c.p : c.p_any;
                      const fill = riskColor(p);
                      const isOOD = c.lat < 14;
                      const opacity = showColdStartOverlay
                        ? (isOOD ? 0.85 : 0.35)
                        : 0.75;
                      return (
                        <CircleMarker
                          key={`cell-${i}-${hrPredMode}`}
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
                              <div className="mt-2 space-y-1 border-t pt-2">
                                <div>
                                  <span className="text-green-700 font-semibold">🌲 Cháy RỪNG:</span>{" "}
                                  <span className="font-semibold" style={{ color: riskColor(c.p) }}>
                                    {(c.p * 100).toFixed(1)}% — {riskLevel(c.p)}
                                  </span>
                                  {c.fire === 1 && <span className="text-red-600 font-bold ml-2">🔥 CÓ</span>}
                                </div>
                                <div>
                                  <span className="text-orange-700 font-semibold">🔥 Cháy BẤT KỲ:</span>{" "}
                                  <span className="font-semibold" style={{ color: riskColor(c.p_any) }}>
                                    {(c.p_any * 100).toFixed(1)}% — {riskLevel(c.p_any)}
                                  </span>
                                  {c.fire_any === 1 && <span className="text-red-600 font-bold ml-2">🔥 CÓ</span>}
                                </div>
                              </div>
                              <div className="text-xs text-stone-500 mt-2 border-t pt-1">
                                Mô hình: XGBoost v9.0 multi-task (61 feats, cold-start)
                              </div>
                            </div>
                          </Popup>
                        </CircleMarker>
                      );
                    })}
                    {/* Fire points overlay — uses selected mode's fire label */}
                    {showFirePoints && currentDate.cells.filter(c => c[fireField] === 1).map((c, i) => (
                      <CircleMarker
                        key={`fire-${i}-${hrPredMode}`}
                        center={[c.lat, c.lon]}
                        radius={6}
                        pathOptions={{ color: "#000000", weight: 2, fillColor: "#ef4444", fillOpacity: 1 }}
                      >
                        <Popup>
                          <div className="text-xs">
                            <div className="font-bold">
                              {hrPredMode === "forest" ? "🌲 Forest fire (high-conf)" : "🔥 MODIS Active Fire"}
                            </div>
                            <div>{c.lat.toFixed(2)}°N, {c.lon.toFixed(2)}°E</div>
                            <div>P (cháy rừng): {(c.p * 100).toFixed(1)}%</div>
                            <div>P (cháy bất kỳ): {(c.p_any * 100).toFixed(1)}%</div>
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
                  <Activity className="h-4 w-4" />
                  Phân loại {hrPredMode === "forest" ? "cháy RỪNG" : "cháy BẤT KỲ"} (p ≥ 0,20)
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
                        <div className="text-stone-600 text-xs">
                          {hrPredMode === "forest" ? "Cháy rừng thực tế" : "Tổng fires thực tế"}
                        </div>
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
                      Dự báo ở đây có độ tin cậy THẤP HƠN. AUROC = {metrics?.AUROC.toFixed(3)} (vs 0.982 in-distribution).</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-3 h-3 rounded mt-0.5" style={{ background: "rgba(0,0,0,0.3)" }} />
                    <div>
                      <div className="font-semibold">Vùng in-distribution (lat ≥ 14°)</div>
                      <div className="text-blue-700">Miền Bắc — đã có trong huấn luyện. Dự báo đáng tin hơn.</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Mode comparison */}
            <Card className="border-stone-300 bg-stone-50">
              <CardHeader className="pb-2">
                <CardTitle className="text-stone-900 flex items-center gap-2 text-base">
                  <Cpu className="h-4 w-4" /> So sánh 2 chế độ
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-xs">
                <table className="w-full text-stone-700">
                  <thead>
                    <tr className="border-b border-stone-300">
                      <th className="text-left py-1">Metric</th>
                      <th className="text-right py-1">🌲 Rừng</th>
                      <th className="text-right py-1">🔥 Bất kỳ</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="py-1">AUROC</td>
                      <td className="text-right font-mono">{data.metadata.metrics_south_cold_OOD_forest_fire?.AUROC.toFixed(3) ?? '—'}</td>
                      <td className="text-right font-mono">{data.metadata.metrics_south_cold_OOD_any_fire?.AUROC.toFixed(3) ?? '—'}</td>
                    </tr>
                    <tr>
                      <td className="py-1">PR-AUC</td>
                      <td className="text-right font-mono">{data.metadata.metrics_south_cold_OOD_forest_fire?.PR_AUC.toFixed(3) ?? '—'}</td>
                      <td className="text-right font-mono">{data.metadata.metrics_south_cold_OOD_any_fire?.PR_AUC.toFixed(3) ?? '—'}</td>
                    </tr>
                    <tr>
                      <td className="py-1">Brier</td>
                      <td className="text-right font-mono">{data.metadata.metrics_south_cold_OOD_forest_fire?.Brier.toFixed(3) ?? '—'}</td>
                      <td className="text-right font-mono">{data.metadata.metrics_south_cold_OOD_any_fire?.Brier.toFixed(3) ?? '—'}</td>
                    </tr>
                    <tr>
                      <td className="py-1">Pos rate (Nam)</td>
                      <td className="text-right font-mono">{((data.metadata.metrics_south_cold_OOD_forest_fire as any)?.pos_rate_south * 100).toFixed(1) ?? '—'}%</td>
                      <td className="text-right font-mono">{((data.metadata.metrics_south_cold_OOD_any_fire as any)?.pos_rate_south * 100).toFixed(1) ?? '—'}%</td>
                    </tr>
                  </tbody>
                </table>
                <div className="text-stone-500 italic mt-2">
                  Lọc cháy nông nghiệp → South pos rate giảm 67%, "Nguy hiểm" cells giảm 64%.
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Methodology note */}
        <div className="mt-8 bg-stone-100 border border-stone-200 rounded-lg p-4 text-stone-700 text-sm">
          <div className="flex items-start gap-2">
            <Cpu className="h-5 w-5 text-stone-500 mt-0.5 shrink-0" />
            <div>
              <div className="font-semibold text-stone-900 mb-1">Phương pháp luận v9.0:</div>
              <p>
                Mô hình <span className="font-mono">XGBoost v9.0 multi-task</span> được huấn luyện trên
                <strong> miền Bắc Việt Nam (lat ≥ 14°) 2018-2021</strong>, hiệu chỉnh bằng isotonic regression
                trên validation Bắc 2022. Đánh giá trên <strong>miền Nam 2023 (spatial OOD)</strong>.
                61 features: 35 cold baseline (weather + lags + rolling + derived) + 7 weather anomalies
                (departure từ per-cell doy climatology) + 7 land cover (ESA WorldCover 2021:
                forest, shrubland, grassland, agriculture, urban, bare, water) + 9 geo (elevation, slope,
                3 Köppen one-hot, latitude_band, coastal_distance, population_proxy) + 3 NDVI
                (MODIS MOD13A1: mean, std, amplitude per cell).
              </p>
              <p className="mt-2">
                <strong>Multi-task</strong>: 2 XGBoost models chia sẻ 61 features.
                <span className="text-green-700 font-semibold"> Task 1 (fire_forest)</span> = MODIS detection
                có confidence='high' (proxy cho high-FRP wildfire, lọc cháy nông nghiệp).
                <span className="text-orange-700 font-semibold"> Task 2 (fire_any)</span> = any MODIS detection
                (bao gồm cháy nông nghiệp).
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
