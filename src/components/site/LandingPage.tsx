"use client";

import { useState } from "react";
import { ArrowRight, FileText, Info, ExternalLink } from "lucide-react";
import { PROJECT, KEY_FINDINGS } from "@/lib/data";
import { VietnamBackdrop } from "./VietnamBackdrop";

interface LandingPageProps {
  onTryNow: () => void;
}

export default function LandingPage({ onTryNow }: LandingPageProps) {
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div className="min-h-screen bg-white flex flex-col relative overflow-hidden">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0">
        <VietnamBackdrop className="h-[110%] w-auto opacity-[0.07]" />
      </div>

      <div className="relative z-10 border-b border-slate-200 bg-slate-50/40 backdrop-blur-[1px]">
        <div className="max-w-4xl mx-auto px-6 py-2.5 flex items-center justify-between">
          <div className="text-[11px] uppercase tracking-wider text-slate-500 font-medium">
            KHKT 2026 · Cấp THCS
          </div>
          <div className="text-[11px] text-slate-500">{PROJECT.school}</div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex items-center justify-center px-6 py-8">
        <div className="max-w-3xl w-full">
          {/* Title */}
          <div className="text-center mb-4">
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 text-slate-700 text-xs font-semibold ring-1 ring-slate-200">
              {PROJECT.field}
            </span>
          </div>

          <h1 className="text-center text-2xl sm:text-3xl font-extrabold text-slate-900 leading-tight tracking-tight">
            WILDFIRE FORECASTING SYSTEM
          </h1>
          <p className="text-center text-slate-600 text-sm sm:text-base mt-2 leading-relaxed">
            HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN BẰNG MÔ HÌNH HỌC MÁY
          </p>

          {/* Author */}
          <div className="mt-4 text-center">
            <div className="text-base font-semibold text-slate-900">{PROJECT.author}</div>
            <div className="text-sm text-slate-500 mt-0.5">
              Lớp {PROJECT.class} · GV hướng dẫn: {PROJECT.teacher}
            </div>
          </div>

          {/* Stats — compact inline */}
          <div className="mt-6 flex justify-center gap-4 sm:gap-6 flex-wrap">
            {[
              { label: "Huyện-day", value: "964K" },
              { label: "Huyện", value: "710" },
              { label: "Features", value: "75" },
              { label: "AUROC OOD", value: "0,977" },
              { label: "Brier", value: "0,038" },
            ].map((s, i) => (
              <div key={i} className="text-center">
                <div className="text-lg font-bold text-slate-900">{s.value}</div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wide">{s.label}</div>
              </div>
            ))}
          </div>

          {/* Buttons */}
          <div className="mt-6 flex flex-col sm:flex-row justify-center gap-3">
            <button
              onClick={onTryNow}
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-900 text-white text-sm font-semibold rounded-md transition-colors shadow-sm"
            >
              Thử ngay
              <ArrowRight className="h-4 w-4" />
            </button>
            <button
              onClick={() => setShowInfo(!showInfo)}
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white hover:bg-slate-50 text-slate-700 text-sm font-semibold rounded-md border border-slate-300 transition-colors"
            >
              <Info className="h-4 w-4" />
              {showInfo ? "Ẩn thông tin" : "Thông tin đề tài"}
            </button>
            <a
              href="/Bao_cao_KHKT_final.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white hover:bg-slate-50 text-slate-700 text-sm font-semibold rounded-md border border-slate-300 transition-colors"
            >
              <FileText className="h-4 w-4" />
              Báo cáo PDF
            </a>
          </div>

          {/* Info section — collapsible */}
          {showInfo && (
            <div className="mt-8 p-5 rounded-lg bg-slate-50 ring-1 ring-slate-200 space-y-4 text-sm">
              {/* Key findings */}
              <div>
                <div className="font-semibold text-slate-800 mb-2">Kết quả chính</div>
                <div className="space-y-2">
                  {KEY_FINDINGS.map((f, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <span className={`inline-block w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${f.tone === "good" ? "bg-emerald-500" : f.tone === "warn" ? "bg-amber-500" : "bg-red-500"}`} />
                      <div>
                        <span className="text-slate-700">{f.label}: </span>
                        <span className="font-semibold text-slate-900">{f.value}</span>
                        <span className="text-slate-500 text-xs"> ({f.sub})</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Data sources */}
              <div>
                <div className="font-semibold text-slate-800 mb-2">Dữ liệu</div>
                <div className="grid grid-cols-2 gap-2 text-xs text-slate-600">
                  <div>· MODIS Active Fire (NASA)</div>
                  <div>· VIIRS 375m (Suomi-NPP)</div>
                  <div>· NASA POWER (thời tiết 0,5°)</div>
                  <div>· GADM ADM2 (710 huyện)</div>
                  <div>· MODIS MOD13A1 (NDVI 500m)</div>
                  <div>· ESA WorldCover (lớp phủ đất)</div>
                  <div>· Open-Meteo (dự báo thời tiết)</div>
                  <div>· FIRMS (điểm cháy real-time)</div>
                </div>
              </div>

              {/* Method */}
              <div>
                <div className="font-semibold text-slate-800 mb-2">Phương pháp</div>
                <div className="text-xs text-slate-600 leading-relaxed">
                  Mô hình XGBoost + isotonic calibration, 75 features (thời tiết + anomalies + FWI + lớp phủ đất + NDVI thật + spatial lag). Optuna hyperparameter tuning (30 trials). Cold-start spatial OOD: huấn luyện Bắc+Trung, kiểm thử Nam. Iso fit trên held-out validation.
                </div>
              </div>

              {/* Model versions */}
              <div>
                <div className="font-semibold text-slate-800 mb-2">Phiên bản mô hình</div>
                <div className="text-xs text-slate-600 space-y-1">
                  <div>v9.0 (ô lưới): AUROC 0,921 — cell-level, 61 features</div>
                  <div>v9.3 (huyện): AUROC 0,929 — district-level, 77 features, real MODIS NDVI</div>
                  <div className="font-semibold text-slate-800">v9.4 (huyện): AUROC 0,977 — Optuna + spatial lag, 75 features, leak-free retrain</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="relative z-10 border-t border-slate-200 bg-slate-50/40 backdrop-blur-[1px]">
        <div className="max-w-4xl mx-auto px-6 py-3 text-center">
          <div className="text-[11px] text-slate-500">
            Cuộc thi Khoa học Kĩ thuật cấp THCS — Sở GD&ĐT Hà Nội · 10/2026
          </div>
        </div>
      </div>
    </div>
  );
}
