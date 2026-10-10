"use client";

import { useState, useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import {
  MapPin, Thermometer, CloudRain, Wind, Activity, Zap,
  AlertTriangle, Layers, Eye, Gauge, Map as MapIcon,
} from "lucide-react";
import {
  VIETNAM_GRID, VIETNAM_CITIES, SCENARIOS,
  type FireCell,
} from "@/lib/data";
import { predictFire, cellColor, cellOpacity } from "@/lib/predict";

// Leaflet loaded client-side only
const MapContainer = dynamic(() => import("react-leaflet").then(m => m.MapContainer), { ssr: false });
const TileLayer    = dynamic(() => import("react-leaflet").then(m => m.TileLayer),    { ssr: false });
const CircleMarker = dynamic(() => import("react-leaflet").then(m => m.CircleMarker), { ssr: false });
const Popup        = dynamic(() => import("react-leaflet").then(m => m.Popup),         { ssr: false });
const Polygon      = dynamic(() => import("react-leaflet").then(m => m.Polygon),       { ssr: false });

// Vietnam region split boundaries (rough)
const NORTH_POLYGON: [number, number][] = [
  [23.5, 102.5], [22.5, 102.0], [21.0, 102.0], [20.5, 102.5],
  [19.0, 102.5], [18.0, 102.5], [17.0, 103.5], [16.5, 105.0],
  [16.0, 106.5], [16.5, 107.0], [17.5, 107.0], [18.5, 107.5],
  [19.5, 107.5], [20.5, 107.5], [21.5, 108.0], [22.0, 108.5],
  [22.5, 108.0], [23.0, 107.0], [23.5, 106.0], [23.5, 104.5],
  [23.5, 102.5],
];

const CENTRAL_POLYGON: [number, number][] = [
  [18.5, 107.5], [17.5, 107.0], [16.5, 107.0], [16.0, 106.5],
  [15.5, 107.5], [15.0, 108.5], [13.5, 109.0], [12.5, 109.0],
  [11.5, 108.5], [11.5, 107.5], [12.5, 107.0], [13.5, 107.0],
  [15.0, 107.5], [16.0, 107.5], [17.0, 107.5], [18.0, 107.5], [18.5, 107.5],
];

const SOUTH_POLYGON: [number, number][] = [
  [11.5, 108.5], [12.5, 109.0], [13.5, 109.0], [15.0, 108.5],
  [15.5, 107.5], [16.0, 106.5], [16.5, 105.0], [17.0, 103.5],
  [16.0, 103.5], [15.0, 104.0], [14.0, 104.0], [13.0, 103.5],
  [12.0, 103.5], [11.0, 103.5], [10.0, 103.5], [9.0, 104.0],
  [8.5, 104.5], [9.0, 105.0], [9.5, 106.0], [10.0, 106.5],
  [10.5, 107.0], [11.0, 108.0], [11.5, 108.5],
];

// === Basemap providers ===
// Multiple providers so user can switch if one is blocked / cached.
// All free, no API key required.
const BASEMAPS = [
  {
    id: "carto-light",
    name: "CARTO Light",
    description: "Sáng, tối giản — phù hợp UI trắng",
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 20,
  },
  {
    id: "carto-voyager",
    name: "CARTO Voyager",
    description: "Có đường sá, POI, tên địa danh",
    url: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 20,
  },
  {
    id: "osm-standard",
    name: "OpenStreetMap",
    description: "Bản đồ OSM chuẩn, chi tiết đầy đủ",
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '&copy; OpenStreetMap contributors',
    subdomains: "abc",
    maxZoom: 19,
  },
  {
    id: "esri-imagery",
    name: "Esri Vệ tinh",
    description: "Ảnh vệ tinh (UI sẽ đậm hơn)",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
    subdomains: "",
    maxZoom: 19,
  },
  {
    id: "esri-topo",
    name: "Esri Topo",
    description: "Địa hình, có ranh giới hành chính",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    subdomains: "",
    maxZoom: 19,
  },
] as const;

type BasemapId = typeof BASEMAPS[number]["id"];

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

const RISK_LEGEND = [
  { color: "#991b1b", label: "Rất cao (≥ 0,7)", bg: "bg-red-900"   },
  { color: "#dc2626", label: "Cao (0,5–0,7)",   bg: "bg-red-600"   },
  { color: "#f59e0b", label: "Trung bình (0,3–0,5)", bg: "bg-slate-500" },
  { color: "#84cc16", label: "Thấp–trung (0,15–0,3)", bg: "bg-lime-500" },
  { color: "#22c55e", label: "Thấp (< 0,15)", bg: "bg-green-500" },
];

export default function MapSection() {
  // Default = scenario 0 (Mùa khô Tây Nguyên)
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const [temp, setTemp]               = useState(SCENARIOS[0].temp);
  const [humidity, setHumidity]       = useState(SCENARIOS[0].humidity);
  const [wind, setWind]               = useState(SCENARIOS[0].wind);
  const [rain, setRain]               = useState(SCENARIOS[0].rain);
  const [daysSinceRain, setDaysSinceRain] = useState(SCENARIOS[0].daysSinceRain);
  const [showCells, setShowCells]     = useState(true);
  const [showCities, setShowCities]   = useState(true);
  const [showSplits, setShowSplits]   = useState(false);
  const [selectedCell, setSelectedCell] = useState<FireCell | null>(null);
  const [basemapId, setBasemapId]     = useState<BasemapId>("carto-light");

  const activeBasemap = BASEMAPS.find(b => b.id === basemapId) ?? BASEMAPS[0];

  // Load scenario
  const loadScenario = (idx: number) => {
    setScenarioIdx(idx);
    setTemp(SCENARIOS[idx].temp);
    setHumidity(SCENARIOS[idx].humidity);
    setWind(SCENARIOS[idx].wind);
    setRain(SCENARIOS[idx].rain);
    setDaysSinceRain(SCENARIOS[idx].daysSinceRain);
  };

  // Compute predictions for all 273 cells
  const cellPredictions = useMemo(() => {
    return VIETNAM_GRID.map(cell => {
      const pred = predictFire({
        temp, humidity, wind, rain, daysSinceRain, region: cell.region,
      });
      // Combine model prediction with cell's baseRisk (spatial prior)
      const finalProb = (pred.probability * 0.6 + cell.baseRisk * 0.4);
      return {
        ...cell,
        probability: Math.min(0.99, Math.max(0.01, finalProb)),
        riskLevel: pred.riskLevel,
        riskColor: pred.riskColor,
      };
    });
  }, [temp, humidity, wind, rain, daysSinceRain]);

  // Stats
  const stats = useMemo(() => {
    const total = cellPredictions.length;
    const high = cellPredictions.filter(c => c.probability >= 0.5).length;
    const veryHigh = cellPredictions.filter(c => c.probability >= 0.7).length;
    const avg = cellPredictions.reduce((s, c) => s + c.probability, 0) / total;
    return { total, high, veryHigh, avg };
  }, [cellPredictions]);

  // Region breakdown
  const regionStats = useMemo(() => {
    const byRegion = { north: { cells: 0, avg: 0, high: 0 }, central: { cells: 0, avg: 0, high: 0 }, south: { cells: 0, avg: 0, high: 0 } };
    for (const c of cellPredictions) {
      byRegion[c.region].cells++;
      byRegion[c.region].avg += c.probability;
      if (c.probability >= 0.5) byRegion[c.region].high++;
    }
    (Object.keys(byRegion) as Array<keyof typeof byRegion>).forEach(r => {
      if (byRegion[r].cells > 0) {
        byRegion[r].avg /= byRegion[r].cells;
      }
    });
    return byRegion;
  }, [cellPredictions]);

  return (
    <section id="map" className="py-16 bg-stone-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <Badge variant="outline" className="text-slate-700 border-slate-300 bg-slate-50">
            <MapPin className="h-3 w-3 mr-1" />
            Bản đồ dự báo tương tác
          </Badge>
          <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-stone-900">
            Dự báo nguy cơ cháy cho 273 ô lưới Việt Nam
          </h2>
          <p className="mt-3 text-stone-600 max-w-2xl mx-auto">
            Nhập thông số thời tiết hoặc chọn kịch bản — bản đồ sẽ hiển thị xác suất cháy
            ước lượng cho từng ô. Bấm vào ô để xem chi tiết. Màu càng đỏ, nguy cơ càng cao.
          </p>
        </motion.div>

        {/* Scenario selector */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="mt-6 flex flex-col sm:flex-row items-stretch sm:items-center gap-3 justify-center"
        >
          <Label className="text-xs font-medium text-stone-600 whitespace-nowrap text-center sm:text-right">
            Kịch bản thời tiết:
          </Label>
          <Select value={String(scenarioIdx)} onValueChange={(v) => loadScenario(parseInt(v))}>
            <SelectTrigger className="w-full sm:w-[280px] bg-white">
              <SelectValue placeholder="Chọn kịch bản" />
            </SelectTrigger>
            <SelectContent>
              {SCENARIOS.map((s, i) => (
                <SelectItem key={i} value={String(i)}>{s.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="text-xs text-stone-500 text-center sm:text-left">
            hoặc điều chỉnh thông số bên dưới ↓
          </span>
        </motion.div>

        <div className="mt-6 grid lg:grid-cols-12 gap-6">
          {/* Control panel */}
          <div className="lg:col-span-3">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Gauge className="h-4 w-4 text-slate-600" />
                  Thông số thời tiết
                </CardTitle>
                <CardDescription className="text-xs">
                  Từ NASA POWER [5]
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <SliderInput
                  label="Nhiệt độ 2 m"
                  unit="°C"
                  icon={<Thermometer className="h-3.5 w-3.5" />}
                  value={temp}
                  onChange={setTemp}
                  min={5} max={40} step={0.5}
                  tone={temp > 35 || temp < 15 ? "danger" : "neutral"}
                />
                <SliderInput
                  label="Độ ẩm tương đối"
                  unit="%"
                  icon={<CloudRain className="h-3.5 w-3.5" />}
                  value={humidity}
                  onChange={setHumidity}
                  min={30} max={100} step={1}
                  tone={humidity < 50 ? "danger" : "neutral"}
                />
                <SliderInput
                  label="Tốc độ gió"
                  unit="m/s"
                  icon={<Wind className="h-3.5 w-3.5" />}
                  value={wind}
                  onChange={setWind}
                  min={0} max={10} step={0.1}
                  tone={wind > 8 ? "danger" : "neutral"}
                />
                <SliderInput
                  label="Lượng mưa"
                  unit="mm"
                  icon={<CloudRain className="h-3.5 w-3.5" />}
                  value={rain}
                  onChange={setRain}
                  min={0} max={50} step={0.5}
                />
                <SliderInput
                  label="Số ngày không mưa"
                  unit=" ngày"
                  icon={<Activity className="h-3.5 w-3.5" />}
                  value={daysSinceRain}
                  onChange={setDaysSinceRain}
                  min={0} max={20} step={1}
                  tone={daysSinceRain > 15 ? "danger" : "neutral"}
                />

                {/* Layer toggles */}
                <div className="pt-3 border-t border-stone-200 space-y-2">
                  <div className="flex items-center gap-2">
                    <Layers className="h-3.5 w-3.5 text-stone-500" />
                    <span className="text-xs font-medium text-stone-700">Lớp hiển thị</span>
                  </div>
                  <label className="flex items-center gap-2 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showCells}
                      onChange={(e) => setShowCells(e.target.checked)}
                      className="rounded accent-slate-600"
                    />
                    <span className="text-stone-700">Ô lưới 0,5° (273 ô)</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showCities}
                      onChange={(e) => setShowCities(e.target.checked)}
                      className="rounded accent-slate-600"
                    />
                    <span className="text-stone-700">Thành phố chính</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showSplits}
                      onChange={(e) => setShowSplits(e.target.checked)}
                      className="rounded accent-slate-600"
                    />
                    <span className="text-stone-700">Ranh giới 3 vùng (Bắc/Trung/Nam)</span>
                  </label>
                </div>

                {/* Basemap selector */}
                <div className="pt-3 mt-3 border-t border-stone-200 space-y-2">
                  <div className="flex items-center gap-2">
                    <MapIcon className="h-3.5 w-3.5 text-stone-500" />
                    <span className="text-xs font-medium text-stone-700">Bản đồ nền</span>
                  </div>
                  <Select value={basemapId} onValueChange={(v) => setBasemapId(v as BasemapId)}>
                    <SelectTrigger className="w-full h-8 text-xs bg-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {BASEMAPS.map((b) => (
                        <SelectItem key={b.id} value={b.id}>
                          <div className="flex flex-col">
                            <span className="text-xs font-medium">{b.name}</span>
                            <span className="text-[10px] text-stone-500">{b.description}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-[10px] text-stone-500 leading-tight">
                    Không thấy bản đồ? Thử đổi nhà cung cấp hoặc nhấn Ctrl+Shift+R để clear cache.
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Legend */}
            <Card className="mt-4">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Eye className="h-4 w-4 text-slate-600" />
                  Thang nguy cơ
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-1.5">
                  {RISK_LEGEND.map(l => (
                    <div key={l.label} className="flex items-center gap-2 text-xs">
                      <span
                        className="inline-block w-3 h-3 rounded-sm"
                        style={{ backgroundColor: l.color }}
                      />
                      <span className="text-stone-700">{l.label}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Map */}
          <div className="lg:col-span-6">
            <Card className="overflow-hidden">
              <div className="h-[560px] sm:h-[640px]">
                <MapContainer
                  center={[16.5, 106.5]}
                  zoom={6}
                  className="h-full w-full"
                  scrollWheelZoom={false}
                >
                  <TileLayer
                    key={activeBasemap.id}
                    url={activeBasemap.url}
                    attribution={activeBasemap.attribution}
                    subdomains={activeBasemap.subdomains || "abc"}
                    maxZoom={activeBasemap.maxZoom}
                  />

                  {/* Region boundary polygons */}
                  {showSplits && (
                    <>
                      <Polygon
                        positions={NORTH_POLYGON}
                        pathOptions={{ color: "#0369a1", weight: 1.5, opacity: 0.6, fillOpacity: 0.05, dashArray: "6 4" }}
                      />
                      <Polygon
                        positions={CENTRAL_POLYGON}
                        pathOptions={{ color: "#93761f", weight: 1.5, opacity: 0.6, fillOpacity: 0.05, dashArray: "6 4" }}
                      />
                      <Polygon
                        positions={SOUTH_POLYGON}
                        pathOptions={{ color: "#dc2626", weight: 1.5, opacity: 0.6, fillOpacity: 0.05, dashArray: "6 4" }}
                      />
                    </>
                  )}

                  {/* Fire cells (CircleMarkers) */}
                  {showCells && cellPredictions.map((c) => (
                    <CircleMarker
                      key={c.id}
                      center={[c.lat, c.lon]}
                      radius={5}
                      pathOptions={{
                        color: cellColor(c.probability),
                        fillColor: cellColor(c.probability),
                        fillOpacity: cellOpacity(c.probability),
                        weight: 0.5,
                      }}
                      eventHandlers={{
                        click: () => setSelectedCell(c),
                      }}
                    >
                      <Popup>
                        <div className="text-xs">
                          <div className="font-bold text-stone-900 mb-1">
                            Ô lưới #{c.id + 1}
                          </div>
                          <div className="text-stone-600 mb-2">
                            {c.lat}°N, {c.lon}°E · Vùng:{" "}
                            {c.region === "north" ? "Bắc" : c.region === "central" ? "Trung" : "Nam"}
                          </div>
                          <div className="grid grid-cols-2 gap-1">
                            <div className="text-stone-500">Xác suất cháy:</div>
                            <div className="font-mono font-bold" style={{ color: c.riskColor }}>
                              {(c.probability * 100).toFixed(1)}%
                            </div>
                            <div className="text-stone-500">Mức:</div>
                            <div className="font-semibold" style={{ color: c.riskColor }}>
                              {c.riskLevel}
                            </div>
                          </div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}

                  {/* City markers */}
                  {showCities && VIETNAM_CITIES.map(city => (
                    <CircleMarker
                      key={city.name}
                      center={[city.lat, city.lon]}
                      radius={4}
                      pathOptions={{
                        color: "#0f172a",
                        fillColor: "#0f172a",
                        fillOpacity: 1,
                        weight: 1.5,
                      }}
                    >
                      <Popup>
                        <div className="text-xs">
                          <div className="font-bold text-stone-900">{city.name}</div>
                          <div className="text-stone-600">
                            {city.lat.toFixed(2)}°N, {city.lon.toFixed(2)}°E
                          </div>
                          <div className="text-stone-500 mt-1">Vùng: {
                            city.region === "north" ? "Bắc" :
                            city.region === "central" ? "Trung" : "Nam"
                          }</div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>
              </div>
            </Card>
          </div>

          {/* Stats panel */}
          <div className="lg:col-span-3 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Zap className="h-4 w-4 text-slate-600" />
                  Thống kê toàn quốc
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Stat label="Tổng số ô" value={stats.total} tone="neutral" />
                <Stat
                  label="Nguy cơ cao (≥ 50%)"
                  value={stats.high}
                  sub={`${((stats.high / stats.total) * 100).toFixed(1)}% tổng số ô`}
                  tone={stats.high > 50 ? "danger" : "warning"}
                />
                <Stat
                  label="Nguy cơ rất cao (≥ 70%)"
                  value={stats.veryHigh}
                  sub={`${((stats.veryHigh / stats.total) * 100).toFixed(1)}% tổng số ô`}
                  tone={stats.veryHigh > 20 ? "danger" : "warning"}
                />
                <Stat
                  label="Xác suất trung bình"
                  value={`${(stats.avg * 100).toFixed(1)}%`}
                  tone={stats.avg > 0.4 ? "danger" : stats.avg > 0.25 ? "warning" : "good"}
                />
              </CardContent>
            </Card>

            {/* Region breakdown */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Activity className="h-4 w-4 text-slate-600" />
                  Phân vùng Bắc / Trung / Nam
                </CardTitle>
                <CardDescription className="text-xs">
                  Concept drift giữa các vùng
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <RegionRow
                  label="Bắc"
                  data={regionStats.north}
                  color="#0369a1"
                />
                <RegionRow
                  label="Trung"
                  data={regionStats.central}
                  color="#93761f"
                />
                <RegionRow
                  label="Nam"
                  data={regionStats.south}
                  color="#dc2626"
                />
              </CardContent>
            </Card>

            {/* Selected cell detail */}
            {selectedCell && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <Card className="ring-2 ring-slate-300">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <MapPin className="h-4 w-4 text-slate-600" />
                      Ô đã chọn
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="text-xs space-y-2">
                    <div className="flex justify-between">
                      <span className="text-stone-500">Tọa độ:</span>
                      <span className="font-mono font-semibold">
                        {selectedCell.lat}°N, {selectedCell.lon}°E
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-stone-500">Vùng:</span>
                      <span className="font-semibold capitalize">
                        {selectedCell.region === "north" ? "Bắc" : selectedCell.region === "central" ? "Trung" : "Nam"}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-stone-500">Xác suất cháy:</span>
                      <span
                        className="font-mono font-bold text-sm"
                        style={{ color: selectedCell.riskColor }}
                      >
                        {(selectedCell.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-stone-500">Mức cảnh báo:</span>
                      <span
                        className="px-2 py-0.5 rounded text-[10px] font-bold"
                        style={{
                          color: selectedCell.riskColor,
                          backgroundColor: selectedCell.riskColor + "20",
                        }}
                      >
                        {selectedCell.riskLevel}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.10 }}
          className="mt-4 p-3 rounded-md bg-stone-100 ring-1 ring-stone-200 text-xs text-stone-600 leading-relaxed"
        >
          <AlertTriangle className="inline h-3.5 w-3.5 text-slate-500 mr-1" />
          <strong>Lưu ý:</strong> Đây là mô hình mô phỏng logistic, không phải mô hình XGBoost
          thực tế. Kết quả nghiên cứu thật: AUROC = 0,8426 (IID), 0,5838 (Bắc → Nam trên VIIRS).
          Số ô hiển thị có thể ít hơn 273 do lọc ô đất liền.
        </motion.div>
      </div>
    </section>
  );
}

function SliderInput({
  label, unit, icon, value, onChange, min, max, step, tone = "neutral",
}: {
  label: string;
  unit: string;
  icon: React.ReactNode;
  value: number;
  onChange: (v: number) => void;
  min: number; max: number; step: number;
  tone?: "neutral" | "danger";
}) {
  return (
    <div>
      <Label className="flex items-center justify-between text-xs font-medium mb-1.5">
        <span className="inline-flex items-center gap-1.5 text-stone-700">
          {icon}
          {label}
        </span>
        <span className={`font-mono font-semibold ${tone === "danger" ? "text-slate-600" : "text-stone-900"}`}>
          {value}{unit}
          {tone === "danger" && <AlertTriangle className="inline h-3 w-3 ml-1" />}
        </span>
      </Label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-slate-600 cursor-pointer h-1.5"
      />
      <div className="flex justify-between text-[10px] text-stone-400 mt-0.5">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

function Stat({
  label, value, sub, tone,
}: {
  label: string;
  value: string | number;
  sub?: string;
  tone: "good" | "warning" | "danger" | "neutral";
}) {
  const toneClasses = {
    good:    "text-slate-700",
    warning: "text-slate-600",
    danger:  "text-slate-700",
    neutral: "text-stone-900",
  };
  return (
    <div className="p-2.5 rounded-md bg-stone-50 ring-1 ring-stone-200">
      <div className="text-[11px] text-stone-500 font-medium">{label}</div>
      <div className={`text-xl font-extrabold ${toneClasses[tone]}`}>{value}</div>
      {sub && <div className="text-[10px] text-stone-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function RegionRow({
  label, data, color,
}: {
  label: string;
  data: { cells: number; avg: number; high: number };
  color: string;
}) {
  return (
    <div className="p-2.5 rounded-md ring-1 ring-stone-200 bg-white">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-bold text-stone-800" style={{ color }}>
          {label}
        </span>
        <span className="text-[10px] text-stone-500">{data.cells} ô</span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-stone-500">Xác suất TB</span>
        <span className="text-xs font-mono font-bold" style={{ color }}>
          {(data.avg * 100).toFixed(1)}%
        </span>
      </div>
      <div className="mt-1 h-1.5 rounded-full bg-stone-100 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${data.avg * 100}%`,
            backgroundColor: color,
          }}
        />
      </div>
      <div className="flex items-center justify-between mt-1.5">
        <span className="text-[10px] text-stone-500">Ô nguy cơ cao</span>
        <span className="text-xs font-mono text-stone-700">{data.high}</span>
      </div>
    </div>
  );
}
