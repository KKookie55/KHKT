"use client";

import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import {
  MapPin, Thermometer, CloudRain, Wind, Activity,
  AlertTriangle, Layers, Gauge, Map as MapIcon, X, Loader2, Cpu, Clock, History,
} from "lucide-react";
import {
  VIETNAM_GRID, VIETNAM_CITIES, SCENARIOS,
  type FireCell,
} from "@/lib/data";
import { cellColor, cellOpacity } from "@/lib/predict";

const MapContainer = dynamic(() => import("react-leaflet").then(m => m.MapContainer), { ssr: false });
const TileLayer    = dynamic(() => import("react-leaflet").then(m => m.TileLayer),    { ssr: false });
const CircleMarker = dynamic(() => import("react-leaflet").then(m => m.CircleMarker), { ssr: false });
const Popup        = dynamic(() => import("react-leaflet").then(m => m.Popup),         { ssr: false });
const GeoJSON      = dynamic(() => import("react-leaflet").then(m => m.GeoJSON),       { ssr: false });

const BASEMAPS = [
  { id: "osm", name: "OpenStreetMap", url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", attribution: '&copy; OpenStreetMap', subdomains: "abc", maxZoom: 19 },
  { id: "esri-imagery", name: "Esri Vệ tinh", url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attribution: 'Esri', subdomains: "", maxZoom: 19 },
  { id: "esri-topo", name: "Esri Địa hình", url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", attribution: 'Esri', subdomains: "", maxZoom: 19 },
] as const;
type BasemapId = typeof BASEMAPS[number]["id"];

const HORIZONS = [
  { id: 1, label: "t+1 (ngày mai)" },
  { id: 3, label: "t+3 (sau 3 ngày)" },
  { id: 7, label: "t+7 (sau 7 ngày)" },
] as const;

const RISK_LEGEND = [
  { color: "#4c1d95", label: "Nguy hiểm (≥0,20)" },
  { color: "#7f1d1d", label: "Rất cao (0,10-0,20)" },
  { color: "#dc2626", label: "Cao (0,05-0,10)" },
  { color: "#f59e0b", label: "TB (0,02-0,05)" },
  { color: "#84cc16", label: "Thấp-TB (0,005-0,02)" },
  { color: "#22c55e", label: "Thấp (<0,005)" },
];

interface PredictionHistory {
  timestamp: string;
  horizon: number;
  temp: number;
  humidity: number;
  wind: number;
  rain: number;
  daysSinceRain: number;
  region: string;
  avgProbability: number;
  highRiskCount: number;
  source: string;
  forecastDate: string;
}

export default function MapView() {
  const [temp, setTemp] = useState(28);
  const [humidity, setHumidity] = useState(65);
  const [wind, setWind] = useState(4);
  const [rain, setRain] = useState(0);
  const [daysSinceRain, setDaysSinceRain] = useState(8);
  const [showCells, setShowCells] = useState(true);
  const [showCities, setShowCities] = useState(true);
  const [selectedCell, setSelectedCell] = useState<FireCell | null>(null);
  const [basemapId, setBasemapId] = useState<BasemapId>("osm");
  const [horizon, setHorizon] = useState(1);
  const [modelPredictions, setModelPredictions] = useState<Map<number, any>>(new Map());
  const [isPredicting, setIsPredicting] = useState(false);
  const [lastPredictTime, setLastPredictTime] = useState<number | null>(null);
  const [isFetchingWeather, setIsFetchingWeather] = useState(false);
  const [weatherDate, setWeatherDate] = useState<string | null>(null);
  const [weatherSource, setWeatherSource] = useState<string>("—");
  const [forecastData, setForecastData] = useState<any[]>([]);
  const [history, setHistory] = useState<PredictionHistory[]>([]);
  const [weatherCoord, setWeatherCoord] = useState<{ lat: number; lon: number } | null>(null);
  const [lastRefreshTime, setLastRefreshTime] = useState<Date | null>(null);
  const [nextRefreshIn, setNextRefreshIn] = useState<number>(600); // seconds
  const [isAutoRefreshing, setIsAutoRefreshing] = useState(false);
  const [predMode, setPredMode] = useState<"forest" | "any">("forest");
  const [showDistricts, setShowDistricts] = useState(false);
  const [districtAggMode, setDistrictAggMode] = useState<"avg" | "max">("max");
  const [districtsGeo, setDistrictsGeo] = useState<any>(null);
  const [cellToDistrict, setCellToDistrict] = useState<Record<number, any> | null>(null);
  const [districtPredictions, setDistrictPredictions] = useState<Map<number, any>>(new Map());
  const REFRESH_INTERVAL = 600; // 10 minutes in seconds

  const activeBasemap = BASEMAPS.find(b => b.id === basemapId) ?? BASEMAPS[0];

  // Compute forecast date based on horizon
  const today = new Date();
  const forecastDate = new Date(today);
  forecastDate.setDate(today.getDate() + horizon);
  const forecastDateStr = forecastDate.toISOString().slice(0, 10);

  const fetchRealWeather = useCallback(async (lat: number, lon: number) => {
    setIsFetchingWeather(true);
    setWeatherCoord({ lat, lon });
    try {
      const response = await fetch(`/api/weather?lat=${lat}&lon=${lon}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setTemp(data.temp);
      setHumidity(data.humidity);
      setWind(data.wind);
      setRain(data.rain);
      setDaysSinceRain(data.daysSinceRain);
      setWeatherDate(data.date);
      setWeatherSource(`Open-Meteo · ${data.date}`);
      setForecastData(data.forecast || []);
    } catch (e) {
      console.error('Weather fetch error:', e);
      setWeatherSource("Open-Meteo · lỗi");
    } finally {
      setIsFetchingWeather(false);
    }
  }, []);

  const cellPredictions = useMemo(() => {
    return VIETNAM_GRID.map(cell => {
      const pred = modelPredictions.get(cell.id);
      if (!pred || pred.error) return { ...cell, probability: 0, riskLevel: "Lỗi", riskColor: "#94a3b8" };
      const finalProb = Math.min(0.99, Math.max(0.001, pred.probability));
      return { ...cell, probability: finalProb, riskLevel: pred.risk_level, riskColor: pred.risk_color };
    });
  }, [modelPredictions]);

  const stats = useMemo(() => {
    // In v9.1 district mode, compute stats from district predictions
    if (districtPredictions && districtPredictions.size > 0) {
      const dp = Array.from(districtPredictions.values()).filter(r => !r.error);
      const total = dp.length;
      const high = dp.filter(r => r.probability >= 0.5).length;
      const avg = dp.reduce((s, r) => s + r.probability, 0) / Math.max(total, 1);
      return { total, high, avg };
    }
    const total = cellPredictions.length;
    const high = cellPredictions.filter(c => c.probability >= 0.5).length;
    const avg = cellPredictions.reduce((s, c) => s + c.probability, 0) / Math.max(total, 1);
    return { total, high, avg };
  }, [cellPredictions, districtPredictions]);

  const fetchPredictions = useCallback(async () => {
    setIsPredicting(true);
    const startTime = Date.now();
    try {
      // District mode (v9.1): if districts toggle on, send ALL 710 district_ids
      // (predict_v91.py handles fallback for districts without explicit cell mapping)
      const ALL_DISTRICT_IDS = Array.from({ length: 710 }, (_, i) => i);
      const useV91 = showDistricts;  // when district overlay on, use v9.1 for all 710 districts

      let inputs: any[];
      if (useV91) {
        // V9.1 DISTRICT mode — predictions for ALL 710 districts (v9.1 has fallback for missing features)
        inputs = ALL_DISTRICT_IDS.map(did => ({
          district_id: did,
          temp, humidity, wind, rain, daysSinceRain,
          mode: predMode,
        }));
      } else {
        // V9.0 CELL mode — predictions per cell
        inputs = VIETNAM_GRID.map(cell => ({
          lat: cell.lat,
          lon: cell.lon,
          temp, humidity, wind, rain, daysSinceRain,
          mode: predMode,
        }));
      }
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const results = data.results as any[];

      if (useV91) {
        // V9.1: results are per-district; store in districtPredictions map
        const dpredMap = new Map<number, any>();
        results.forEach(r => {
          if (r && r.district_id != null) dpredMap.set(r.district_id, r);
        });
        setDistrictPredictions(dpredMap);
        // ALSO populate cell-level predictions from district (so cell popups show district p)
        const cpredMap = new Map<number, any>();
        VIETNAM_GRID.forEach(cell => {
          const m = cellToDistrict?.[cell.id];
          if (m && m.district_id != null) {
            const dp = dpredMap.get(m.district_id);
            if (dp) cpredMap.set(cell.id, dp);
          }
        });
        setModelPredictions(cpredMap);
      } else {
        // V9.0: results are per-cell, store directly
        const predMap = new Map<number, any>();
        VIETNAM_GRID.forEach((cell, i) => {
          const pred = results[i];
          if (!pred) return;
          predMap.set(cell.id, pred);
        });
        setModelPredictions(predMap);
        setDistrictPredictions(new Map()); // clear
      }
      setLastPredictTime(Date.now() - startTime);

      // Add to history
      const now = new Date();
      setHistory(prev => [{
        timestamp: now.toLocaleTimeString("vi-VN"),
        horizon,
        temp, humidity, wind, rain, daysSinceRain,
        region: "all",
        avgProbability: stats.avg,
        highRiskCount: stats.high,
        source: weatherSource,
        forecastDate: forecastDateStr,
      }, ...prev].slice(0, 10));
    } catch (err) {
      console.warn('Predict failed:', err);
      // Show ERROR state — do NOT use fake baseRisk values
      const predMap = new Map<number, any>();
      VIETNAM_GRID.forEach(cell => {
        predMap.set(cell.id, {
          probability: 0,
          risk_level: "Lỗi",
          risk_color: "#94a3b8",
          error: true,
        });
      });
      setModelPredictions(predMap);
      setLastPredictTime(Date.now() - startTime);
    } finally {
      setIsPredicting(false);
    }
  }, [temp, humidity, wind, rain, daysSinceRain, horizon, weatherSource, stats.avg, stats.high, forecastDateStr, predMode, showDistricts]);

  useEffect(() => {
    const timer = setTimeout(() => { fetchPredictions(); }, 500);
    return () => clearTimeout(timer);
  }, [fetchPredictions]);

  // Auto-fetch weather on mount
  useEffect(() => {
    fetchRealWeather(16.5, 106.5);
  }, [fetchRealWeather]);

  // ===== Lazy-load district GeoJSON + cell→district lookup when toggle on =====
  useEffect(() => {
    if (!showDistricts || districtsGeo) return;
    let cancelled = false;
    (async () => {
      try {
        const [geoResp, lookupResp] = await Promise.all([
          fetch("/data/geo/districts_simple.json"),
          fetch("/data/geo/cell_to_district.json"),
        ]);
        if (!geoResp.ok || !lookupResp.ok) throw new Error("HTTP err");
        const geo = await geoResp.json();
        const lookup = await lookupResp.json() as Record<number, any>;
        if (!cancelled) {
          setDistrictsGeo(geo);
          setCellToDistrict(lookup);
        }
      } catch (e) {
        console.error("District data fetch failed:", e);
      }
    })();
    return () => { cancelled = true; };
  }, [showDistricts, districtsGeo]);

  // ===== Aggregate cell predictions → district stats (used as fallback when v9.1 not loaded) =====
  const districtStats = useMemo(() => {
    if (!cellToDistrict || !districtsGeo) return null;

    // PRIMARY: if v9.1 district predictions exist, use them directly
    if (districtPredictions && districtPredictions.size > 0) {
      const result = new Map<number, { p: number; color: string; label: string; cellCount: number }>();
      districtPredictions.forEach((pred, did) => {
        const p = Math.min(0.99, Math.max(0.001, pred.probability || 0));
        let color = "#22c55e";
        if (p >= 0.20)      color = "#4c1d95";
        else if (p >= 0.10) color = "#7f1d1d";
        else if (p >= 0.05) color = "#dc2626";
        else if (p >= 0.02) color = "#f59e0b";
        else if (p >= 0.005) color = "#84cc16";
        const d = districtsGeo.features[did];
        const name = d?.properties?.name ?? pred.district_name ?? "?";
        const prov = d?.properties?.province ?? pred.province ?? "?";
        const type = d?.properties?.type ?? "";
        result.set(did, {
          p,
          color,
          label: `${type} ${name}, ${prov}`,
          cellCount: 0, // not tracked in v9.1 mode
        });
      });
      return result;
    }

    // FALLBACK: aggregate cell predictions → district (v9.0 mode)
    const byDistrict = new Map<number, { probs: number[]; cells: number[] }>();
    cellPredictions.forEach(c => {
      const m = cellToDistrict[c.id];
      if (!m || m.district_id == null) return;
      const d = byDistrict.get(m.district_id) ?? { probs: [], cells: [] };
      d.probs.push(c.probability);
      d.cells.push(c.id);
      byDistrict.set(m.district_id, d);
    });
    const result = new Map<number, { p: number; color: string; label: string; cellCount: number }>();
    byDistrict.forEach((v, did) => {
      const p = districtAggMode === "max" ? Math.max(...v.probs) : v.probs.reduce((s, x) => s + x, 0) / v.probs.length;
      let color = "#22c55e";
      if (p >= 0.7)      color = "#991b1b";
      else if (p >= 0.5) color = "#dc2626";
      else if (p >= 0.3) color = "#f59e0b";
      else if (p >= 0.15) color = "#84cc16";
      const d = districtsGeo.features[did];
      const name = d?.properties?.name ?? "?";
      const prov = d?.properties?.province ?? "?";
      const type = d?.properties?.type ?? "";
      result.set(did, {
        p,
        color,
        label: `${type} ${name}, ${prov}`,
        cellCount: v.probs.length,
      });
    });
    return result;
  }, [cellToDistrict, districtsGeo, cellPredictions, districtAggMode, districtPredictions]);

  // Style function for GeoJSON districts (interactive:false so clicks pass through to cells)
  const districtStyle = useCallback((feature: any) => {
    const did = feature?.properties?.id;
    const s = districtStats?.get(did);
    if (!s) return { color: "#94a3b8", weight: 0.5, fillColor: "#cbd5e1", fillOpacity: 0.3, interactive: false };
    return {
      color: "#475569",
      weight: 0.6,
      fillColor: s.color,
      fillOpacity: 0.5,
      interactive: false,
    } as any;
  }, [districtStats]);

  // ===== Auto-refresh every 10 minutes (pauses when tab hidden) =====
  const autoRefresh = useCallback(async () => {
    setIsAutoRefreshing(true);
    try {
      const coord = weatherCoordRef.current;
      if (coord) await fetchRealWeather(coord.lat, coord.lon);
      // Wait briefly so weather state propagates before predict
      await new Promise(r => setTimeout(r, 300));
      await fetchPredictions();
      setLastRefreshTime(new Date());
      setNextRefreshIn(REFRESH_INTERVAL);
    } catch (e) {
      console.error('Auto-refresh error:', e);
    } finally {
      setIsAutoRefreshing(false);
    }
  }, [fetchRealWeather, fetchPredictions]);

  // Keep a ref of weatherCoord so the interval callback always reads the latest value
  const weatherCoordRef = useRef<{ lat: number; lon: number } | null>(null);
  useEffect(() => {
    weatherCoordRef.current = weatherCoord;
  }, [weatherCoord]);

  // Countdown + interval (only runs when document is visible)
  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const tick = () => {
      const isHidden = typeof document !== 'undefined' && document.hidden;
      if (isHidden) return; // pause when tab hidden
      setNextRefreshIn(prev => {
        if (prev <= 1) {
          autoRefresh();
          return REFRESH_INTERVAL;
        }
        return prev - 1;
      });
    };

    intervalId = setInterval(tick, 1000);

    const onVisibilityChange = () => {
      if (!document.hidden) {
        // If tab was hidden for >5 minutes, refresh immediately
        const lastTs = lastRefreshTime?.getTime() || 0;
        const elapsed = (Date.now() - lastTs) / 1000;
        if (elapsed > 300) {
          autoRefresh();
        }
      }
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibilityChange);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisibilityChange);
      }
    };
  }, [autoRefresh, lastRefreshTime]);

  // Format countdown as MM:SS
  const countdownStr = useMemo(() => {
    const m = Math.floor(nextRefreshIn / 60);
    const s = nextRefreshIn % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }, [nextRefreshIn]);

  // Apply forecast data when horizon changes
  useEffect(() => {
    if (forecastData.length > 0 && horizon <= forecastData.length) {
      const day = forecastData[horizon - 1];
      if (day) {
        setTemp(Math.round(((day.temp_max + day.temp_min) / 2) * 10) / 10);
        setRain(Math.round(day.rain * 10) / 10);
        setWind(Math.round((day.wind / 3.6) * 10) / 10);
      }
    }
  }, [horizon, forecastData]);

  return (
    <div className="h-full flex">
      {/* MAP 70% */}
      <div className="flex-1 relative" style={{ flexBasis: "70%" }}>
        {/* Disclaimer */}
        <div className="absolute top-0 left-0 right-0 z-[600] bg-slate-800/90 backdrop-blur text-white text-[10px] text-center py-1 px-2 flex items-center justify-center gap-2">
          <span className="inline-flex items-center gap-1.5">
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${isAutoRefreshing ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400 animate-pulse'}`} />
            Dự báo t+{horizon} cho {forecastDateStr} · {(showDistricts && districtPredictions.size > 0) ? "Model v9.4 HUYỆN" : "Model v9.0 Ô"} ({predMode === "forest" ? "🌲 cháy RỪNG" : "🔥 cháy BẤT KỲ"}) · Open-Meteo · Auto-refresh 10ph
          </span>
          <span className="opacity-75 font-mono text-[9px]">
            {isAutoRefreshing ? 'ĐANG CẬP NHẬT…' : `Lần sau: ${countdownStr}`}
          </span>
        </div>

        <MapContainer center={[16.5, 106.5]} zoom={6} className="h-full w-full" scrollWheelZoom={true} maxBounds={[[7.5, 100.5], [24.5, 110.5]]} maxBoundsViscosity={0.9} minZoom={5} maxZoom={12}>
          <TileLayer key={activeBasemap.id} url={activeBasemap.url} attribution={activeBasemap.attribution} subdomains={activeBasemap.subdomains || "abc"} maxZoom={activeBasemap.maxZoom} />
          {showDistricts && districtsGeo && (
            <GeoJSON key={`districts-${districtAggMode}-${predMode}-${horizon}-${lastPredictTime ?? 0}`} data={districtsGeo} style={districtStyle} />
          )}
          {showCells && cellPredictions.map((c) => {
            const dm = cellToDistrict?.[c.id];
            return (
            <CircleMarker key={c.id} center={[c.lat, c.lon]} radius={5}
              pathOptions={{ color: "#1f2937", fillColor: c.riskColor, fillOpacity: 0.8, weight: 0.5 }}
              eventHandlers={{ click: () => { setSelectedCell(c); fetchRealWeather(c.lat, c.lon); } }}>
              <Popup>
                <div className="text-xs">
                  <div className="font-bold text-slate-900 mb-1">Ô #{c.id + 1}</div>
                  <div className="text-slate-600 mb-1">{c.lat}°N, {c.lon}°E</div>
                  {dm && (
                    <div className="mb-2 px-1.5 py-1 rounded bg-slate-100 ring-1 ring-slate-200 text-[10px] text-slate-700">
                      <span className="text-slate-500">Thuộc: </span>
                      <span className="font-semibold">{dm.type} {dm.name}, {dm.province}</span>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-slate-500">P(cháy) t+{horizon}:</div>
                    <div className="font-mono font-bold" style={{ color: c.riskColor }}>{(c.probability * 100).toFixed(1)}%</div>
                    <div className="text-slate-500">Mức:</div>
                    <div className="font-semibold" style={{ color: c.riskColor }}>{c.riskLevel}</div>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
            );
          })}
          {showCities && VIETNAM_CITIES.map(city => (
            <CircleMarker key={city.name} center={[city.lat, city.lon]} radius={4}
              pathOptions={{ color: "#0f172a", fillColor: "#0f172a", fillOpacity: 1, weight: 1.5 }}>
              <Popup><div className="text-xs"><div className="font-bold text-slate-900">{city.name}</div><div className="text-slate-600">{city.lat.toFixed(2)}°N, {city.lon.toFixed(2)}°E</div></div></Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 z-[500] bg-white/95 backdrop-blur rounded-md shadow-md ring-1 ring-slate-200 p-2">
          <div className="text-[10px] font-semibold text-slate-700 mb-1">Thang nguy cơ</div>
          <div className="space-y-0.5">
            {RISK_LEGEND.map(l => (<div key={l.label} className="flex items-center gap-1.5 text-[10px]"><span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: l.color }} /><span className="text-slate-600">{l.label}</span></div>))}
          </div>
        </div>

        {/* Model status */}
        <div className="absolute top-8 left-3 z-[500] bg-white/95 backdrop-blur rounded-md shadow-md ring-1 ring-slate-200 p-2 text-[10px]">
          <div className="flex items-center gap-1.5">
            {isPredicting ? <Loader2 className="h-3 w-3 animate-spin text-slate-500" /> : <Cpu className="h-3 w-3 text-slate-600" />}
            <span className="font-semibold text-slate-700">{isPredicting ? "Đang chạy..." : (showDistricts && districtPredictions.size > 0 ? "Model v9.4 (Huyện)" : "Model v9.0 (Ô)")}</span>
          </div>
          <div className="mt-1 text-slate-500">
            <div>AUROC South OOD (forest fire): <span className="font-mono text-slate-700">{showDistricts && districtPredictions.size > 0 ? "0,979 (v9.4 huyện)" : "0,921 (v9.0 ô)"}</span></div>
            <div>Dự báo: <span className="font-mono text-slate-700">t+{horizon} ({forecastDateStr})</span></div>
            {lastPredictTime !== null && <div>Latency: <span className="font-mono text-slate-700">{lastPredictTime}ms</span></div>}
            {lastRefreshTime && (
              <div className="mt-1 pt-1 border-t border-slate-200">
                <div>Cập nhật gần nhất: <span className="font-mono text-slate-700">{lastRefreshTime.toLocaleTimeString("vi-VN")}</span></div>
                <div className="flex items-center gap-1 mt-0.5">
                  <span className={`inline-block w-1.5 h-1.5 rounded-full ${isAutoRefreshing ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'}`} />
                  <span className="font-mono text-slate-600">{isAutoRefreshing ? 'ĐANG REFRESH…' : `Sau ${countdownStr}`}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="absolute top-8 right-3 z-[500] bg-white/95 backdrop-blur rounded-md shadow-md ring-1 ring-slate-200 p-2.5 text-xs">
          <div className="font-semibold text-slate-700 mb-1.5 text-[11px]">Thống kê t+{horizon}</div>
          <div className="space-y-1">
            <div className="flex justify-between gap-2"><span className="text-slate-500 text-[10px]">{districtPredictions.size > 0 ? "Tổng huyện" : "Tổng ô"}</span><span className="font-mono font-bold text-xs text-slate-900">{stats.total}</span></div>
            <div className="flex justify-between gap-2"><span className="text-slate-500 text-[10px]">Nguy cơ cao</span><span className="font-mono font-bold text-xs text-slate-700">{stats.high}</span></div>
            <div className="flex justify-between gap-2"><span className="text-slate-500 text-[10px]">TB</span><span className="font-mono font-bold text-xs text-slate-700">{(stats.avg * 100).toFixed(1)}%</span></div>
          </div>
        </div>
      </div>

      {/* CONTROL PANEL 30% */}
      <div className="border-l border-slate-200 bg-slate-50/30 overflow-y-auto" style={{ flexBasis: "30%", flexShrink: 0 }}>
        <div className="p-4 space-y-4">
          {/* Horizon selector */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block">
              <Clock className="inline h-3 w-3 mr-1" />Khoảng dự báo
            </Label>
            <Select value={String(horizon)} onValueChange={(v) => setHorizon(parseInt(v))}>
              <SelectTrigger className="w-full h-9 text-xs bg-white"><SelectValue /></SelectTrigger>
              <SelectContent>
                {HORIZONS.map(h => <SelectItem key={h.id} value={String(h.id)}>{h.label}</SelectItem>)}
              </SelectContent>
            </Select>
            <div className="mt-1 text-[10px] text-slate-500">
              Dự báo cho ngày: <span className="font-mono font-semibold text-slate-700">{forecastDateStr}</span>
            </div>
          </div>

          {/* Auto-refresh control */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block flex items-center justify-between">
              <span>Tự động cập nhật</span>
              <span className={`inline-flex items-center gap-1 text-[9px] font-medium px-1.5 py-0.5 rounded ${isAutoRefreshing ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700"}`}>
                <span className={`inline-block w-1.5 h-1.5 rounded-full ${isAutoRefreshing ? "bg-amber-500 animate-pulse" : "bg-emerald-500"}`} />
                {isAutoRefreshing ? "ĐANG CẬP NHẬT" : `SAU ${countdownStr}`}
              </span>
            </Label>
            <div className="bg-white p-2.5 rounded-md ring-1 ring-slate-200 text-[10px] space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Chu kỳ:</span>
                <span className="font-mono font-semibold text-slate-700">10 phút</span>
              </div>
              {lastRefreshTime && (
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Lần cuối:</span>
                  <span className="font-mono text-slate-700">{lastRefreshTime.toLocaleTimeString("vi-VN")}</span>
                </div>
              )}
              <button
                onClick={() => autoRefresh()}
                disabled={isAutoRefreshing || isPredicting}
                className="mt-1 w-full h-7 px-2 text-[10px] font-semibold bg-slate-700 hover:bg-slate-800 disabled:opacity-50 text-white rounded transition-colors"
              >
                {isAutoRefreshing ? "Đang cập nhật…" : "↻ Cập nhật ngay"}
              </button>
              <div className="text-[9px] text-slate-400 italic mt-1 pt-1 border-t border-slate-100">
                Tự động lấy weather + predict mỗi 10 phút khi tab đang mở. Pause khi tab ẩn.
              </div>
            </div>
          </div>

          {/* Real-time weather (auto) */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block flex items-center justify-between">
              <span>Thời tiết thực</span>
              <span className={`inline-flex items-center gap-1 text-[9px] font-medium px-1.5 py-0.5 rounded ${isFetchingWeather ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700"}`}>
                <span className={`inline-block w-1.5 h-1.5 rounded-full ${isFetchingWeather ? "bg-amber-500 animate-pulse" : "bg-emerald-500 animate-pulse"}`} />
                {isFetchingWeather ? "ĐANG LẤY" : "LIVE"}
              </span>
            </Label>
            <div className="bg-white p-2.5 rounded-md ring-1 ring-slate-200 text-[10px] space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Nguồn:</span>
                <span className="font-mono font-semibold text-slate-700">{weatherSource}</span>
              </div>
              {weatherCoord && (
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Vị trí:</span>
                  <span className="font-mono text-slate-700">{weatherCoord.lat.toFixed(2)}°N, {weatherCoord.lon.toFixed(2)}°E</span>
                </div>
              )}
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Dự báo cho:</span>
                <span className="font-mono font-semibold text-slate-700">{forecastDateStr}</span>
              </div>
              <div className="text-[9px] text-slate-400 italic mt-1 pt-1 border-t border-slate-100">
                Tự động lấy khi click ô lưới. Cập nhật số liệu khi click ô khác.
              </div>
            </div>
          </div>

          {/* Weather sliders */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block">Thời tiết (tùy chỉnh)</Label>
            <div className="space-y-3 bg-white p-3 rounded-md ring-1 ring-slate-200">
              <SliderInput label="Nhiệt độ" unit="°C" icon={<Thermometer className="h-3.5 w-3.5" />} value={temp} onChange={setTemp} min={5} max={40} step={0.5} />
              <SliderInput label="Độ ẩm" unit="%" icon={<CloudRain className="h-3.5 w-3.5" />} value={humidity} onChange={setHumidity} min={30} max={100} step={1} />
              <SliderInput label="Gió" unit=" m/s" icon={<Wind className="h-3.5 w-3.5" />} value={wind} onChange={setWind} min={0} max={10} step={0.1} />
              <SliderInput label="Mưa" unit=" mm" icon={<CloudRain className="h-3.5 w-3.5" />} value={rain} onChange={setRain} min={0} max={50} step={0.5} />
              <SliderInput label="Ngày không mưa" unit=" ngày" icon={<Activity className="h-3.5 w-3.5" />} value={daysSinceRain} onChange={setDaysSinceRain} min={0} max={20} step={1} />
            </div>
          </div>

          {/* Layers */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block">
              <Layers className="inline h-3 w-3 mr-1" />Lớp hiển thị
            </Label>
            <div className="bg-white p-3 rounded-md ring-1 ring-slate-200 space-y-1.5">
              <Toggle label={`Ô lưới (${VIETNAM_GRID.length} ô)`} checked={showCells} onChange={setShowCells} />
              <Toggle label="Thành phố" checked={showCities} onChange={setShowCities} />
              <Toggle label={`Huyện (${showDistricts && districtsGeo ? districtsGeo.features.length : 710} đv) — overlay`} checked={showDistricts} onChange={setShowDistricts} />
              {showDistricts && (
                <div className="mt-1 pt-1 border-t border-slate-100">
                  <div className="text-[10px] text-slate-500 mb-1">Aggregation mode (cell → huyện):</div>
                  <div className="flex rounded ring-1 ring-slate-200 overflow-hidden">
                    <button onClick={() => setDistrictAggMode("max")}
                      className={`flex-1 px-2 py-1 text-[10px] font-semibold ${districtAggMode === "max" ? "bg-emerald-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`}>
                      Max (mức cao nhất trong huyện)
                    </button>
                    <button onClick={() => setDistrictAggMode("avg")}
                      className={`flex-1 px-2 py-1 text-[10px] font-semibold ${districtAggMode === "avg" ? "bg-emerald-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`}>
                      Avg (trung bình)
                    </button>
                  </div>
                  <div className="text-[9px] text-slate-400 italic mt-1">
                    Lớp này là overlay hành chính — dự báo v9.0 vẫn chạy ở cấp ô lưới.
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Prediction mode toggle (NEW) */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block">
              <Cpu className="inline h-3 w-3 mr-1" />Chế độ dự báo
            </Label>
            <div className="bg-white p-2 rounded-md ring-1 ring-slate-200">
              <div className="flex rounded-md ring-1 ring-slate-200 overflow-hidden">
                <button
                  onClick={() => setPredMode("forest")}
                  className={`flex-1 px-3 py-1.5 text-[11px] font-semibold transition-colors ${
                    predMode === "forest"
                      ? "bg-emerald-600 text-white"
                      : "bg-white text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  🌲 Cháy RỪNG
                </button>
                <button
                  onClick={() => setPredMode("any")}
                  className={`flex-1 px-3 py-1.5 text-[11px] font-semibold transition-colors ${
                    predMode === "any"
                      ? "bg-orange-600 text-white"
                      : "bg-white text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  🔥 Cháy BẤT KỲ
                </button>
              </div>
              <p className="mt-1.5 text-[9px] text-slate-500 leading-tight">
                {predMode === "forest"
                  ? "Lọc cháy nông nghiệp, chỉ giữ high-confidence (FRP cao = cháy rừng thật)"
                  : "Tất cả MODIS detection — bao gồm cháy nông nghiệp"}
              </p>
            </div>
          </div>

          {/* Basemap */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block">
              <MapIcon className="inline h-3 w-3 mr-1" />Bản đồ nền
            </Label>
            <Select value={basemapId} onValueChange={(v) => setBasemapId(v as BasemapId)}>
              <SelectTrigger className="w-full h-9 text-xs bg-white"><SelectValue /></SelectTrigger>
              <SelectContent>
                {BASEMAPS.map(b => <SelectItem key={b.id} value={b.id}>{b.name}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>

          {/* Prediction History */}
          <div>
            <Label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide mb-2 block">
              <History className="inline h-3 w-3 mr-1" />Lịch sử dự báo
            </Label>
            <div className="bg-white p-2 rounded-md ring-1 ring-slate-200 max-h-48 overflow-y-auto space-y-1.5">
              {history.length === 0 && <div className="text-[10px] text-slate-400 text-center py-3">Chưa có dự báo nào</div>}
              {history.map((h, i) => (
                <div key={i} className="p-2 rounded bg-slate-50 text-[10px] space-y-0.5">
                  <div className="flex justify-between">
                    <span className="text-slate-500">{h.timestamp}</span>
                    <span className="font-mono font-bold text-slate-700">t+{h.horizon}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">P(tb): {(h.avgProbability * 100).toFixed(1)}%</span>
                    <span className="text-slate-500">Cao: {h.highRiskCount}</span>
                  </div>
                  <div className="text-[9px] text-slate-400">{h.forecastDate} · {h.source}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Selected cell detail */}
          {selectedCell && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <Label className="text-[11px] font-semibold text-slate-700 uppercase"><MapPin className="inline h-3 w-3 mr-1" />Ô đã chọn</Label>
                <button onClick={() => setSelectedCell(null)} className="text-slate-400 hover:text-slate-700"><X className="h-3.5 w-3.5" /></button>
              </div>
              <div className="p-3 rounded-md ring-2 ring-slate-400 bg-white text-xs space-y-1.5">
                <div className="flex justify-between"><span className="text-slate-500">Tọa độ:</span><span className="font-mono font-semibold">{selectedCell.lat}°N, {selectedCell.lon}°E</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Vùng:</span><span className="font-semibold">{selectedCell.region === "north" ? "Bắc" : selectedCell.region === "central" ? "Trung" : "Nam"}</span></div>
                <div className="flex justify-between items-center"><span className="text-slate-500">P(cháy) t+{horizon}:</span><span className="font-mono font-bold text-sm" style={{ color: (selectedCell as any).riskColor || "#94a3b8" }}>{((selectedCell as any).probability * 100).toFixed(1)}%</span></div>
              </div>
            </div>
          )}

          <div className="p-2 rounded-md bg-slate-100 ring-1 ring-slate-200 text-[10px] text-slate-600 leading-relaxed">
            <Cpu className="inline h-3 w-3 text-slate-500 mr-1" />
            Model {(showDistricts && districtPredictions.size > 0) ? "v9.4 (Huyện): XGBoost + Optuna + spatial lag, 75 features, full panel + real MODIS NDVI, AUROC=0,979 OOD South" : "v9.0 (Ô): XGBoost + Isotonic, 61 features, AUROC=0,921 OOD South"}. Multi-task forest fire (high-conf MODIS). Cold-start: train Bắc+Trung, test Nam.
          </div>
        </div>
      </div>
    </div>
  );
}

function SliderInput({ label, unit, icon, value, onChange, min, max, step }: {
  label: string; unit: string; icon: React.ReactNode;
  value: number; onChange: (v: number) => void;
  min: number; max: number; step: number;
}) {
  return (
    <div>
      <Label className="flex items-center justify-between text-[11px] font-medium mb-1">
        <span className="inline-flex items-center gap-1 text-slate-700">{icon}{label}</span>
        <span className="font-mono font-semibold text-[11px] text-slate-900">{value}{unit}</span>
      </Label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} className="w-full accent-slate-600 cursor-pointer h-1" />
      <div className="flex justify-between text-[9px] text-slate-400 mt-0.5"><span>{min}{unit}</span><span>{max}{unit}</span></div>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-xs cursor-pointer">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="rounded accent-slate-600" />
      <span className="text-slate-700">{label}</span>
    </label>
  );
}
