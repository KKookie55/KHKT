"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { DATA_SOURCES, MODELS, SPLITS, PROJECT } from "@/lib/data";
import { Database, Layers, ScissorsLineDashed, GitBranch, Beaker } from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export default function MethodSection() {
  return (
    <section id="method" className="py-16 bg-stone-50">
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
            Phương pháp
          </Badge>
          <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-stone-900">
            Quy trình xử lý dữ liệu và đánh giá
          </h2>
          <p className="mt-3 text-stone-600 max-w-2xl mx-auto">
            Ba nguồn dữ liệu, năm mô hình, năm phép chia — tổng cộng 200 lượt chạy thí nghiệm.
          </p>
        </motion.div>

        {/* Pipeline diagram */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5 }}
          className="mt-10"
        >
          <Card className="card-hover">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-slate-600" />
                Sơ đồ pipeline
              </CardTitle>
              <CardDescription>
                Từ dữ liệu thô đến kết quả đánh giá đa chiều.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row items-stretch gap-2 sm:gap-3 overflow-x-auto pb-2">
                {[
                  { label: "NASA POWER", sub: "Thời tiết\n2018–2023", color: "#0284c7" },
                  { label: "MODIS Active Fire", sub: "Nhãn chính\n1 km", color: "#dc2626" },
                  { label: "VIIRS 375 m", sub: "Kiểm chứng\nđộc lập", color: "#93761f" },
                  { label: "Panel ô–ngày", sub: "598.143 dòng\n273 ô × 2.191 ngày", color: "#4d4838" },
                  { label: "5 mô hình", sub: "LogReg · RF\nXGBoost · MLP · PhysMLP", color: "#16a34a" },
                  { label: "5 phép chia", sub: "IID · Temporal\n3 spatial splits", color: "#dc2626" },
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-2 sm:gap-3 flex-1 min-w-[140px]">
                    <div
                      className="flex-1 p-3 rounded-md ring-1 text-center"
                      style={{
                        backgroundColor: step.color + "15",
                        borderColor: step.color + "40",
                      }}
                    >
                      <div className="text-sm font-bold text-stone-900">{step.label}</div>
                      <div className="text-[10px] text-stone-600 whitespace-pre-line mt-0.5">
                        {step.sub}
                      </div>
                    </div>
                    {i < 5 && (
                      <span className="text-stone-400 flex-shrink-0">→</span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* 3 columns */}
        <div className="mt-6 grid lg:grid-cols-3 gap-6">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            transition={{ duration: 0.5, delay: 0.05 }}
          >
            <Card className="h-full card-hover">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Database className="h-4 w-4 text-slate-600" />
                  Ba nguồn dữ liệu
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {DATA_SOURCES.map((s) => (
                  <div key={s.name} className="p-3 rounded-md ring-1 ring-stone-200 bg-white">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-stone-900">{s.name}</span>
                      <Badge
                        variant="outline"
                        style={{ backgroundColor: s.color + "20", color: s.color, borderColor: s.color + "40" }}
                        className="text-[10px] font-mono"
                      >
                        {s.citation}
                      </Badge>
                    </div>
                    <div className="mt-1 text-xs text-stone-600">
                      <strong>{s.role}</strong> · {s.variables}
                    </div>
                    <a
                      href={s.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-2 inline-block text-[10px] text-slate-700 hover:underline break-all"
                    >
                      ↗ {s.url}
                    </a>
                  </div>
                ))}
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            transition={{ duration: 0.5, delay: 0.10 }}
          >
            <Card className="h-full card-hover">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Layers className="h-4 w-4 text-slate-600" />
                  Năm mô hình thực nghiệm
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {MODELS.map((m) => (
                  <div
                    key={m.name}
                    className={`p-3 rounded-md ring-1 ${
                      m.primary ? "ring-slate-300 bg-slate-50" : "ring-stone-200 bg-white"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-stone-900">{m.name}</span>
                      {m.primary && (
                        <Badge className="text-[10px] bg-slate-800 hover:bg-slate-800">Mô hình chính</Badge>
                      )}
                    </div>
                    <div className="mt-1 text-xs text-stone-600">
                      {m.type} · {m.role}
                    </div>
                    <div className="mt-1 text-[10px] font-mono text-stone-500">
                      Trích dẫn: {m.citation}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            transition={{ duration: 0.5, delay: 0.15 }}
          >
            <Card className="h-full card-hover">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <ScissorsLineDashed className="h-4 w-4 text-slate-600" />
                  Năm chiến lược chia dữ liệu
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {SPLITS.map((sp) => {
                  const sevColor =
                    sp.severity === "hard"   ? "#dc2626" :
                    sp.severity === "moderate" ? "#93761f" : "#16a34a";
                  return (
                    <div key={sp.name} className="p-3 rounded-md ring-1 ring-stone-200 bg-white">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-semibold text-stone-900">{sp.name}</span>
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                          style={{ backgroundColor: sevColor + "20", color: sevColor }}
                        >
                          {sp.severity === "hard" ? "Khó nhất" :
                           sp.severity === "moderate" ? "Trung bình" : "Dễ (lạc quan)"}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-stone-600">
                        Train: <strong>{sp.train}</strong> → Test: <strong>{sp.test}</strong>
                      </div>
                      <div className="mt-0.5 text-[11px] text-stone-500">{sp.purpose}</div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Evaluation metrics */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.20 }}
          className="mt-6"
        >
          <Card className="card-hover">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Beaker className="h-4 w-4 text-slate-600" />
                Hệ thống chỉ số đánh giá
              </CardTitle>
              <CardDescription>
                Mỗi chỉ số trả lời một câu hỏi khác nhau về chất lượng mô hình.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { name: "AUROC",     role: "Khả năng xếp hạng ô nguy cơ cao hơn ô ít nguy cơ", cite: "[10], [13]" },
                  { name: "Brier score", role: "Sai lệch bình phương của xác suất dự báo", cite: "[14]" },
                  { name: "ECE",       role: "Expected Calibration Error — khoảng cách xác suất", cite: "[7]" },
                  { name: "Moran's I", role: "Tự tương quan không gian của nhãn cháy", cite: "[10]" },
                  { name: "Cohen's d", role: "Khoảng cách chuẩn giữa hai phân phối", cite: "[14]" },
                  { name: "Jaccard",   role: "Tỷ lệ giao/hợp giữa hai tập nhãn MODIS vs VIIRS", cite: "—" },
                ].map((m) => (
                  <div key={m.name} className="p-3 rounded-md ring-1 ring-stone-200 bg-gradient-to-br from-stone-50 to-white">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-stone-900 font-mono">{m.name}</span>
                      <span className="text-[10px] text-stone-400 font-mono">{m.cite}</span>
                    </div>
                    <p className="mt-1 text-xs text-stone-600 leading-snug">{m.role}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 bg-stone-50 rounded-md text-xs text-stone-700">
                <strong className="text-stone-900">Tổng số lượt chạy:</strong>{" "}
                {PROJECT.runs} lượt — 5 seed × 5 split × 5 model cho VIIRS t+1 (125),
                + t+3/t+7 với seed 42 (50) + MODIS t+1 (25).
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
