"use client";

import { useState } from "react";
import { ArrowRight, FileText } from "lucide-react";
import { PROJECT } from "@/lib/data";
import { VietnamBackdrop } from "./VietnamBackdrop";

interface LandingPageProps {
  onTryNow: () => void;
}

export default function LandingPage({ onTryNow }: LandingPageProps) {
  return (
    <div className="min-h-screen bg-white flex flex-col relative overflow-hidden">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0">
        <VietnamBackdrop className="h-[110%] w-auto opacity-[0.07]" />
      </div>

      <div className="relative z-10 border-b border-slate-200 bg-slate-50/40 backdrop-blur-[1px]">
        <div className="max-w-4xl mx-auto px-6 py-2.5 flex items-center justify-between">
          <div className="text-[11px] uppercase tracking-wider text-slate-500 font-medium">
            Sản phẩm dự thi KHKT 2026 · Cấp THCS
          </div>
          <div className="text-[11px] text-slate-500">
            {PROJECT.school}
          </div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex items-center justify-center px-6 py-12">
        <div className="max-w-3xl w-full">
          <div className="text-center mb-6">
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 text-slate-700 text-xs font-semibold ring-1 ring-slate-200">
              Dự án Khoa học · {PROJECT.field}
            </span>
          </div>

          <h1 className="text-center text-2xl sm:text-3xl lg:text-4xl font-extrabold text-slate-900 leading-tight tracking-tight">
            WILDFIRE FORECASTING SYSTEM<br />
            <span className="text-slate-600 text-lg sm:text-xl">
              DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN BẰNG MÔ HÌNH HỌC MÁY
            </span>
          </h1>

          <div className="mt-6 text-center">
            <div className="text-base font-semibold text-slate-900">
              {PROJECT.author}
            </div>
            <div className="text-sm text-slate-500 mt-1">
              Lớp {PROJECT.class} · GV hướng dẫn: {PROJECT.teacher}
            </div>
          </div>

          <div className="mt-8 flex flex-col sm:flex-row justify-center gap-3">
            <button
              onClick={onTryNow}
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-900 text-white text-sm font-semibold rounded-md transition-colors shadow-sm"
            >
              Thử ngay
              <ArrowRight className="h-4 w-4" />
            </button>
            <a
              href="/Bao_cao_KHKT_final.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white hover:bg-slate-50 text-slate-700 text-sm font-semibold rounded-md border border-slate-300 transition-colors"
            >
              <FileText className="h-4 w-4" />
              Tải báo cáo
            </a>
          </div>

          <div className="mt-10 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-2xl mx-auto">
            {[
              { label: "Huyện-day", value: "964K" },
              { label: "Huyện", value: "710" },
              { label: "Features", value: "75" },
              { label: "AUROC OOD", value: "0,979" },
            ].map((s, i) => (
              <div key={i} className="text-center p-3 rounded-md bg-slate-50/80 ring-1 ring-slate-200 backdrop-blur-sm">
                <div className="text-base font-bold text-slate-900">{s.value}</div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wide mt-0.5">{s.label}</div>
              </div>
            ))}
          </div>
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
