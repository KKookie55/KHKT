"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Calendar, Flame, TrendingUp, Activity, MapPin,
} from "lucide-react";
import {
  HEATMAP_DATA, HEATMAP_STATS, heatmapColor,
  type HeatmapPoint,
} from "@/lib/data";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

const REGIONS: { id: "north" | "central" | "south"; label: string; color: string }[] = [
  { id: "north",   label: "Bắc",  color: "#0369a1" },
  { id: "central", label: "Trung", color: "#93761f" },
  { id: "south",   label: "Nam",   color: "#dc2626" },
];

const MONTH_LABELS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"];
const SEASONS = [
  { months: [12, 1, 2],  label: "Đông (Xuân)",   color: "#0284c7" },
  { months: [3, 4, 5],   label: "Xuân (Hè)",     color: "#16a34a" },
  { months: [6, 7, 8],   label: "Hè (Mưa)",      color: "#d97706" },
  { months: [9, 10, 11], label: "Thu (Bão)",     color: "#7c3aed" },
];

export default function HeatmapSection() {
  const [hovered, setHovered] = useState<HeatmapPoint | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<"all" | "north" | "central" | "south">("all");

  // Build 3×12 grid for the heatmap
  const grid = useMemo(() => {
    const g: Record<"north" | "central" | "south", HeatmapPoint[]> = {
      north: [], central: [], south: [],
    };
    for (const p of HEATMAP_DATA) {
      g[p.region].push(p);
    }
    g.north.sort((a, b) => a.month - b.month);
    g.central.sort((a, b) => a.month - b.month);
    g.south.sort((a, b) => a.month - b.month);
    return g;
  }, []);

  // Totals
  const totals = useMemo(() => {
    const byRegion = { north: 0, central: 0, south: 0 };
    const byMonth: Record<number, number> = {};
    for (const p of HEATMAP_DATA) {
      byRegion[p.region] += p.value;
      byMonth[p.month] = (byMonth[p.month] ?? 0) + p.value;
    }
    const total = byRegion.north + byRegion.central + byRegion.south;
    const peakMonth = Object.entries(byMonth).reduce((a, b) => b[1] > a[1] ? b : a)[0];
    return { byRegion, byMonth, total, peakMonth };
  }, []);

  return (
    <section id="heatmap" className="py-16 bg-white">
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
            <Calendar className="h-3 w-3 mr-1" />
            Heatmap mùa vụ
          </Badge>
          <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-stone-900">
            Cường độ cháy theo tháng × vùng
          </h2>
          <p className="mt-3 text-stone-600 max-w-2xl mx-auto">
            Mô phỏng số cell-day cháy theo 12 tháng × 3 vùng (Bắc/Trung/Nam). Di chuột
            qua ô để xem chi tiết. Pattern mùa vụ phản ánh cơ chế khí hậu khác nhau giữa các vùng.
          </p>
        </motion.div>

        {/* Region filter */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="mt-8 flex flex-wrap justify-center gap-2"
        >
          <FilterChip
            active={selectedRegion === "all"}
            onClick={() => setSelectedRegion("all")}
            color="#475569"
          >
            Tất cả vùng
          </FilterChip>
          {REGIONS.map(r => (
            <FilterChip
              key={r.id}
              active={selectedRegion === r.id}
              onClick={() => setSelectedRegion(r.id)}
              color={r.color}
            >
              {r.label}
            </FilterChip>
          ))}
        </motion.div>

        <div className="mt-8 grid lg:grid-cols-3 gap-6">
          {/* Heatmap grid */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            transition={{ duration: 0.5, delay: 0.10 }}
            className="lg:col-span-2"
          >
            <Card className="card-hover">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Flame className="h-4 w-4 text-slate-600" />
                  Lưới nhiệt 3 × 12
                </CardTitle>
                <CardDescription>
                  Mỗi ô = 1 cặp (vùng, tháng). Màu đậm hơn = nhiều cell-day cháy hơn.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Season legend */}
                <div className="flex flex-wrap gap-2 mb-4 text-[10px]">
                  {SEASONS.map(s => (
                    <span key={s.label} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full"
                          style={{ backgroundColor: s.color + "15", color: s.color }}>
                      <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                      {s.label}: {s.months.map(m => `T${m}`).join(", ")}
                    </span>
                  ))}
                </div>

                {/* Heatmap grid */}
                <div className="overflow-x-auto no-scrollbar">
                  <div className="min-w-[480px]">
                    {/* Month headers */}
                    <div className="grid grid-cols-[80px_repeat(12,1fr)] gap-1 mb-1">
                      <div></div>
                      {MONTH_LABELS.map((m, i) => {
                        const season = SEASONS.find(s => s.months.includes(i + 1));
                        return (
                          <div
                            key={m}
                            className="text-center text-[10px] font-bold py-1 rounded"
                            style={{
                              color: season?.color,
                              backgroundColor: season?.color + "15",
                            }}
                          >
                            {m}
                          </div>
                        );
                      })}
                    </div>

                    {/* Region rows */}
                    {REGIONS.map(region => {
                      const isHidden = selectedRegion !== "all" && selectedRegion !== region.id;
                      return (
                        <div
                          key={region.id}
                          className={`grid grid-cols-[80px_repeat(12,1fr)] gap-1 mb-1 transition-opacity ${
                            isHidden ? "opacity-25" : ""
                          }`}
                        >
                          <div
                            className="flex items-center text-xs font-bold px-2 rounded"
                            style={{ color: region.color, backgroundColor: region.color + "10" }}
                          >
                            {region.label}
                          </div>
                          {grid[region.id].map((p) => (
                            <button
                              key={`${p.region}-${p.month}`}
                              onMouseEnter={() => setHovered(p)}
                              onMouseLeave={() => setHovered(null)}
                              onClick={() => setHovered(p)}
                              className="relative aspect-square rounded transition-all hover:ring-2 hover:ring-stone-900 hover:ring-offset-1"
                              style={{
                                backgroundColor: heatmapColor(p.intensity),
                                cursor: "pointer",
                              }}
                              aria-label={`${p.regionLabel} ${p.monthLabel}: ${p.value} cell-day`}
                            >
                              <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono font-bold"
                                    style={{
                                      color: p.intensity > 0.4 ? "white" : "#1c1917",
                                      textShadow: p.intensity > 0.4 ? "0 1px 2px rgba(0,0,0,0.4)" : "none",
                                    }}>
                                {p.value}
                              </span>
                            </button>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Color legend */}
                <div className="mt-4 flex items-center gap-2 text-[10px] text-stone-600">
                  <span>Thấp</span>
                  <div className="flex h-3 flex-1 rounded overflow-hidden">
                    {[0.04, 0.12, 0.20, 0.32, 0.48, 0.62, 0.78, 0.92].map(i => (
                      <div key={i} className="flex-1" style={{ backgroundColor: heatmapColor(i) }} />
                    ))}
                  </div>
                  <span>Cao</span>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Side panel: hovered detail + stats */}
          <div className="space-y-4">
            {/* Hovered cell detail */}
            <Card className="card-hover min-h-[160px]">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Activity className="h-4 w-4 text-slate-600" />
                  Chi tiết ô
                </CardTitle>
                <CardDescription className="text-xs">
                  {hovered ? `${hovered.regionLabel} · ${hovered.monthLabel}` : "Di chuột qua ô để xem"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {hovered ? (
                  <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={`${hovered.region}-${hovered.month}`}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div
                        className="w-12 h-12 rounded-md"
                        style={{ backgroundColor: heatmapColor(hovered.intensity) }}
                      />
                      <div>
                        <div className="text-xs text-stone-500">Cell-day cháy</div>
                        <div className="text-2xl font-extrabold text-stone-900 font-mono">
                          {hovered.value.toLocaleString("vi-VN")}
                        </div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="p-2 bg-stone-50 rounded">
                        <div className="text-stone-500 text-[10px]">Cường độ</div>
                        <div className="font-mono font-bold text-stone-900">
                          {(hovered.intensity * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="p-2 bg-stone-50 rounded">
                        <div className="text-stone-500 text-[10px]">Chiếm tổng</div>
                        <div className="font-mono font-bold text-stone-900">
                          {((hovered.value / totals.total) * 100).toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <div className="text-xs text-stone-400 text-center py-6">
                    Hover/click vào ô bất kỳ trong lưới
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Stats per region */}
            <Card className="card-hover">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-slate-600" />
                  Thống kê theo vùng
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {REGIONS.map(r => {
                  const total = totals.byRegion[r.id];
                  const pct = (total / totals.total) * 100;
                  const peakMonth = HEATMAP_STATS.peakMonth[r.id];
                  const lowMonth = HEATMAP_STATS.lowMonth[r.id];
                  return (
                    <div
                      key={r.id}
                      className={`p-3 rounded-md ring-1 ring-stone-200 transition-opacity ${
                        selectedRegion !== "all" && selectedRegion !== r.id ? "opacity-40" : ""
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-bold" style={{ color: r.color }}>
                          {r.label}
                        </span>
                        <span className="text-xs font-mono font-bold text-stone-900">
                          {total.toLocaleString("vi-VN")}
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full bg-stone-100 overflow-hidden mb-2">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${pct}%`, backgroundColor: r.color }}
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-stone-500">
                        <span>Đỉnh: T{peakMonth}</span>
                        <span>Thấp: T{lowMonth}</span>
                        <span>{pct.toFixed(1)}% tổng</span>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            {/* Insight */}
            <Card className="card-hover bg-gradient-to-br from-slate-50 to-slate-50 ring-slate-200">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-slate-600" />
                  Đặc trưng mùa vụ
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-xs text-stone-700">
                <div className="flex items-start gap-2">
                  <span className="inline-block w-2 h-2 rounded-full mt-1 flex-shrink-0" style={{ backgroundColor: "#0369a1" }} />
                  <div>
                    <strong className="text-stone-900">Bắc:</strong> {HEATMAP_STATS.peakQuarter.north}
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="inline-block w-2 h-2 rounded-full mt-1 flex-shrink-0" style={{ backgroundColor: "#93761f" }} />
                  <div>
                    <strong className="text-stone-900">Trung:</strong> {HEATMAP_STATS.peakQuarter.central}
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="inline-block w-2 h-2 rounded-full mt-1 flex-shrink-0" style={{ backgroundColor: "#dc2626" }} />
                  <div>
                    <strong className="text-stone-900">Nam:</strong> {HEATMAP_STATS.peakQuarter.south}
                  </div>
                </div>
                <p className="mt-2 text-[11px] text-stone-600 italic leading-relaxed">
                  Pattern khác nhau giữa các vùng là một trong những nguyên nhân gây concept drift — cùng điều kiện thời tiết nhưng ý nghĩa với nguy cơ cháy khác nhau.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Monthly trend chart */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="mt-6"
        >
          <Card className="card-hover">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Calendar className="h-4 w-4 text-slate-600" />
                Xu hướng 12 tháng theo vùng
              </CardTitle>
              <CardDescription>
                Line chart so sánh pattern mùa vụ của 3 vùng.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-[80px_1fr] gap-2">
                {REGIONS.map(region => {
                  const points = grid[region.id];
                  const max = Math.max(...points.map(p => p.value));
                  const isHidden = selectedRegion !== "all" && selectedRegion !== region.id;
                  return (
                    <div
                      key={region.id}
                      className={`contents ${isHidden ? "opacity-25" : ""} transition-opacity`}
                    >
                      <div className="flex items-center text-xs font-bold" style={{ color: region.color }}>
                        {region.label}
                      </div>
                      {/* Sparkline */}
                      <div className="relative h-12 flex items-end gap-1">
                        {points.map((p, i) => (
                          <div
                            key={i}
                            className="flex-1 rounded-t transition-all hover:opacity-80 cursor-pointer relative group"
                            style={{
                              height: `${(p.value / max) * 100}%`,
                              backgroundColor: region.color,
                              opacity: 0.7 + p.intensity * 0.3,
                            }}
                            onMouseEnter={() => setHovered(p)}
                            onMouseLeave={() => setHovered(null)}
                          >
                            <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] font-mono opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                              {p.value}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
                {/* Month labels */}
                <div></div>
                <div className="grid grid-cols-12 gap-1 text-[9px] text-stone-500">
                  {MONTH_LABELS.map(m => (
                    <div key={m} className="text-center">{m}</div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Disclaimer */}
        <div className="mt-6 p-3 rounded-md bg-stone-100 ring-1 ring-stone-200 text-xs text-stone-600 leading-relaxed">
          <strong>Lưu ý:</strong> Số liệu trong heatmap là mô phỏng pedagogical, phản ánh
          pattern mùa vụ thực tế tại Việt Nam (mùa khô Tây Nguyên, gió Lào miền Trung,
          mùa xuân khô miền Bắc). Số cell-day thật từ MODIS/VIIRS có thể khác — đây chỉ là
          minh hoạ để hiểu cơ chế concept drift giữa các vùng.
        </div>
      </div>
    </section>
  );
}

function FilterChip({
  active, onClick, color, children,
}: {
  active: boolean;
  onClick: () => void;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 text-xs font-semibold rounded-full border transition-all ${
        active
          ? "text-white shadow-sm"
          : "text-stone-600 bg-white border-stone-300 hover:bg-stone-50"
      }`}
      style={active ? {
        backgroundColor: color,
        borderColor: color,
      } : {}}
    >
      {children}
    </button>
  );
}
