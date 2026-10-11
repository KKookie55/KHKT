"use client";

import { motion } from "framer-motion";
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, ReferenceLine,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  AUROC_SPLITS, HORIZON_AUROC, PHYSICS_VIOLATION,
  SPATIAL_FEATURES, HYPOTHESES, NORTH_SOUTH_STATS, DIAGNOSTIC_RESULTS,
} from "@/lib/data";
import {
  TrendingDown, Activity, MapPin, GitBranch, Microscope, CheckCircle2,
} from "lucide-react";

const COLOR_MODIS = "#1e3a5f";
const COLOR_VIIRS = "#64748b";
const COLOR_XGB   = "#334155";
const COLOR_MLP   = "#64748b";
const COLOR_PHYS  = "#94a3b8";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export default function Dashboard() {
  return (
    <section id="dashboard" className="py-16 bg-white">
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
            Dashboard kết quả
          </Badge>
          <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-stone-900">
            5 phép chia · 5 mô hình · 2 nguồn nhãn
          </h2>
          <p className="mt-3 text-stone-600 max-w-2xl mx-auto">
            200 lượt chạy thí nghiệm trên 598.143 ô–ngày. Mỗi biểu đồ dưới đây có thể
            tương tác — di chuột để xem chi tiết.
          </p>
        </motion.div>

        <Tabs defaultValue="split" className="mt-10">
          <TabsList className="grid w-full grid-cols-2 sm:grid-cols-5 h-auto bg-stone-100">
            <TabsTrigger value="split">MODIS vs VIIRS</TabsTrigger>
            <TabsTrigger value="horizon">Khoảng dự báo</TabsTrigger>
            <TabsTrigger value="physics">PhysMLP</TabsTrigger>
            <TabsTrigger value="diagnose">Chẩn đoán</TabsTrigger>
            <TabsTrigger value="hypo">Giả thuyết</TabsTrigger>
          </TabsList>

          {/* TAB 1: AUROC across splits */}
          <TabsContent value="split" className="mt-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2 card-hover">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Activity className="h-4 w-4 text-slate-600" />
                    AUROC của XGBoost theo 5 phép chia dữ liệu
                  </CardTitle>
                  <CardDescription>
                    MODIS (nhãn chính) so với VIIRS (kiểm chứng độc lập). Random = 0,5.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={AUROC_SPLITS} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                      <XAxis dataKey="split" tick={{ fontSize: 11 }} stroke="#78716c" />
                      <YAxis domain={[0.4, 1.0]} tick={{ fontSize: 11 }} stroke="#78716c" />
                      <Tooltip
                        formatter={(v: number, n: string) => [v.toFixed(4), n]}
                        contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <ReferenceLine y={0.5} stroke="#a8a29e" strokeDasharray="3 3"
                        label={{ value: "Ngẫu nhiên", fontSize: 10, fill: "#78716c", position: "right" }} />
                      <Bar dataKey="modis" name="MODIS" fill={COLOR_MODIS} radius={[4, 4, 0, 0]} />
                      <Bar dataKey="viirs" name="VIIRS" fill={COLOR_VIIRS} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="card-hover">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <TrendingDown className="h-4 w-4 text-slate-600" />
                    Điểm khác biệt lớn nhất
                  </CardTitle>
                  <CardDescription>Bắc → Nam trên VIIRS</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-4 bg-slate-50 rounded-lg ring-1 ring-slate-200">
                      <div className="text-xs text-slate-700 font-medium">MODIS Bắc → Nam</div>
                      <div className="text-3xl font-extrabold text-slate-700">0,8920</div>
                      <div className="text-xs text-slate-600/70 mt-1">Vẫn ổn định</div>
                    </div>
                    <div className="p-4 bg-slate-50 rounded-lg ring-1 ring-slate-300">
                      <div className="text-xs text-slate-700 font-medium">VIIRS Bắc → Nam</div>
                      <div className="text-3xl font-extrabold text-slate-700">0,5838</div>
                      <div className="text-xs text-slate-700/70 mt-1">± 0,0019 · 5 seeds · gần mức ngẫu nhiên</div>
                    </div>
                    <p className="text-xs text-stone-600 leading-relaxed">
                      Cùng mô hình, cùng phép chia — chỉ đổi nguồn nhãn là kết luận về độ
                      tin cậy có thể đảo ngược hoàn toàn.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* TAB 2: Forecast horizons */}
          <TabsContent value="horizon" className="mt-6">
            <Card className="card-hover">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <GitBranch className="h-4 w-4 text-slate-600" />
                  AUROC theo khoảng dự báo t+1, t+3, t+7
                </CardTitle>
                <CardDescription>
                  Ba mô hình: XGBoost, MLP, PhysMLP. VIIRS, seed 42.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={HORIZON_AUROC} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                    <XAxis dataKey="horizon" tick={{ fontSize: 12 }} stroke="#78716c" />
                    <YAxis domain={[0.80, 0.85]} tick={{ fontSize: 11 }} stroke="#78716c" />
                    <Tooltip formatter={(v: number) => v.toFixed(3)}
                      contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line dataKey="xgboost" name="XGBoost" stroke={COLOR_XGB}
                      strokeWidth={2.5} dot={{ r: 5 }} />
                    <Line dataKey="mlp" name="MLP" stroke={COLOR_MLP}
                      strokeWidth={2} dot={{ r: 4 }} />
                    <Line dataKey="physmlp" name="PhysMLP" stroke={COLOR_PHYS}
                      strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="mt-4 p-3 bg-stone-50 rounded-md text-sm text-stone-700 leading-relaxed">
                  <strong>Kết luận:</strong> mức giảm từ t+1 → t+7 chỉ ≈ 0,02 — nhỏ hơn
                  nhiều so với giảm 0,26 khi chuyển miền. Vấn đề khó hơn không phải dự
                  báo xa hơn vài ngày, mà là đưa mô hình sang vùng địa lý mới.
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* TAB 3: Physics violation */}
          <TabsContent value="physics" className="mt-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2 card-hover">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Microscope className="h-4 w-4 text-slate-600" />
                    Tỷ lệ vi phạm quan hệ đơn điệu (%)
                  </CardTitle>
                  <CardDescription>
                    Thí nghiệm phản thực: khi tăng biến đầu vào, xác suất cháy có theo kỳ vọng không?
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={PHYSICS_VIOLATION} margin={{ top: 8, right: 16, bottom: 28, left: -8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                      <XAxis dataKey="relation" tick={{ fontSize: 10 }} stroke="#78716c"
                        interval={0} angle={-12} textAnchor="end" height={50} />
                      <YAxis tick={{ fontSize: 11 }} stroke="#78716c" domain={[0, 100]} />
                      <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`}
                        contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Bar dataKey="xgboost" name="XGBoost" fill={COLOR_XGB} radius={[3, 3, 0, 0]} />
                      <Bar dataKey="mlp"     name="MLP"     fill={COLOR_MLP} radius={[3, 3, 0, 0]} />
                      <Bar dataKey="physmlp" name="PhysMLP" fill={COLOR_PHYS} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              <Card className="card-hover">
                <CardHeader>
                  <CardTitle className="text-base">Đọc nhanh</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-stone-700 space-y-3">
                  <p>
                    <strong className="text-slate-700">Tin tốt:</strong> PhysMLP kiểm
                    soát hoàn hảo quan hệ <em>gió ↑ → cháy ↑</em> (90,4% → 0%).
                  </p>
                  <p>
                    <strong className="text-slate-700">Cần cải thiện:</strong> quan hệ
                    <em> nhiệt độ ↑ → cháy ↑</em> vẫn vi phạm tới 82,9%.
                  </p>
                  <p>
                    <strong className="text-stone-900">Kết luận cốt lõi:</strong> ràng
                    buộc vật lý cải thiện <em>một số</em> quan hệ, nhưng{" "}
                    <strong className="text-slate-700">không</strong> giải quyết được khái
                    quát ngoài miền.
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* TAB 4: Diagnostic */}
          <TabsContent value="diagnose" className="mt-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card className="card-hover">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <MapPin className="h-4 w-4 text-slate-600" />
                    Dịch chuyển phân phối: Bắc vs Nam
                  </CardTitle>
                  <CardDescription>Cohen&apos;s d cho 4 biến thời tiết chính</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={NORTH_SOUTH_STATS.map((s) => ({ name: s.variable, d: s.cohens_d }))}
                      layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                      <XAxis type="number" tick={{ fontSize: 11 }} stroke="#78716c" domain={[0, 0.8]} />
                      <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} stroke="#78716c" width={120} />
                      <Tooltip formatter={(v: number) => `d = ${v.toFixed(3)}`}
                        contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                      <Bar dataKey="d" name="Cohen's d" radius={[0, 4, 4, 0]}>
                        {NORTH_SOUTH_STATS.map((s, i) => (
                          <Cell key={i} fill={s.cohens_d >= 0.4 ? "#dc2626" : "#93761f"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="card-hover">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Microscope className="h-4 w-4 text-slate-600" />
                    Bốn thí nghiệm chẩn đoán
                  </CardTitle>
                  <CardDescription>Nguyên nhân sụt giảm Bắc → Nam</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {DIAGNOSTIC_RESULTS.map((d) => (
                      <div key={d.experiment} className="p-3 bg-stone-50 rounded-md ring-1 ring-stone-200">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-medium text-stone-700">{d.experiment}</span>
                          <Badge style={{ backgroundColor: d.color + "20", color: d.color, borderColor: d.color }}
                            className="text-[10px] font-semibold border">
                            {d.supports}
                          </Badge>
                        </div>
                        <div className="mt-1 text-sm font-mono text-stone-900">{d.result}</div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card className="mt-6 card-hover">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-slate-600" />
                  Thử nghiệm biến đại diện vị trí (AUROC Bắc → Nam)
                </CardTitle>
                <CardDescription>
                  Tọa độ liên tục làm điểm số tăng vọt — nhưng binned lại giảm. Mô hình
                  đang <em>ghi nhớ vị trí</em> chứ không học cơ chế.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={SPATIAL_FEATURES} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                    <XAxis dataKey="setup" tick={{ fontSize: 10 }} stroke="#78716c"
                      interval={0} angle={-8} textAnchor="end" height={50} />
                    <YAxis domain={[0.4, 0.9]} tick={{ fontSize: 11 }} stroke="#78716c" />
                    <Tooltip formatter={(v: number) => v.toFixed(4)}
                      contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                    <ReferenceLine y={0.5} stroke="#a8a29e" strokeDasharray="3 3" />
                    <Bar dataKey="auroc" name="AUROC" radius={[4, 4, 0, 0]}>
                      {SPATIAL_FEATURES.map((s, i) => (
                        <Cell key={i} fill={s.tone === "good" ? "#16a34a" : "#dc2626"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* TAB 5: Hypotheses */}
          <TabsContent value="hypo" className="mt-6">
            <Card className="card-hover">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-slate-600" />
                  Năm giả thuyết khoa học và kết quả kiểm tra
                </CardTitle>
                <CardDescription>
                  Tất cả 5 giả thuyết (H1–H5) đều được dữ liệu thực nghiệm ủng hộ.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid sm:grid-cols-2 gap-3">
                  {HYPOTHESES.map((h) => (
                    <div
                      key={h.id}
                      className="p-4 rounded-md ring-1 ring-stone-200 bg-gradient-to-br from-stone-50 to-white card-hover"
                    >
                      <div className="flex items-start gap-3">
                        <span className="flex-shrink-0 inline-flex items-center justify-center h-7 w-7 rounded-md bg-emerald-100 text-slate-700 text-xs font-bold">
                          {h.id}
                        </span>
                        <p className="text-sm text-stone-800 leading-relaxed">{h.text}</p>
                      </div>
                      <div className="mt-2 ml-10 text-[11px] font-semibold text-slate-700 uppercase tracking-wide">
                        ✓ Được ủng hộ
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  );
}
