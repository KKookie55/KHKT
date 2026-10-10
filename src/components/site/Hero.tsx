"use client";

import { motion } from "framer-motion";
import { Flame, ArrowDown, AlertTriangle, MapPin } from "lucide-react";
import { PROJECT, KEY_FINDINGS } from "@/lib/data";

const TONE_STYLES: Record<string, { bg: string; text: string; ring: string }> = {
  danger:  { bg: "bg-slate-50",     text: "text-slate-700",     ring: "ring-slate-200" },
  good:    { bg: "bg-slate-50", text: "text-slate-700", ring: "ring-slate-200" },
  warning: { bg: "bg-slate-50",   text: "text-slate-600",   ring: "ring-slate-200" },
  neutral: { bg: "bg-stone-50",   text: "text-stone-700",   ring: "ring-stone-200" },
};

export default function Hero() {
  return (
    <section className="relative pt-28 pb-16 overflow-hidden bg-white">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-b from-slate-100/60 via-white to-white" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-slate-200/40 rounded-full blur-3xl -translate-y-32 translate-x-32" />
        <div className="absolute bottom-0 left-0 w-72 h-72 bg-slate-100/40 rounded-full blur-3xl translate-y-24 -translate-x-12" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Kicker */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex justify-center"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 text-slate-700 text-xs font-semibold ring-1 ring-slate-200">
            <Flame className="h-3.5 w-3.5" />
            {PROJECT.competition} · {PROJECT.year}
          </span>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.05 }}
          className="mt-6 text-center text-3xl sm:text-4xl lg:text-5xl font-extrabold text-stone-900 leading-tight max-w-4xl mx-auto tracking-tight"
        >
          Khi mô hình dự báo cháy rừng{" "}
          <span className="text-slate-800">
            sang vùng mới
          </span>
          , điều gì thực sự xảy ra?
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.12 }}
          className="mt-5 text-center text-base sm:text-lg text-stone-600 max-w-2xl mx-auto leading-relaxed"
        >
          Một hệ thống dự báo nguy cơ cháy rừng dựa trên dữ liệu vệ tinh MODIS, VIIRS và
          NASA POWER. Đánh giá khả năng khái quát không gian của 5 mô hình học máy trên{" "}
          <strong className="text-stone-900">
            {PROJECT.dataPoints.toLocaleString("vi-VN")}
          </strong>{" "}
          ô–ngày quan sát tại Việt Nam.
        </motion.p>

        {/* CTA buttons */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.18 }}
          className="mt-8 flex flex-col sm:flex-row justify-center gap-3"
        >
          <a
            href="#map"
            onClick={(e) => {
              e.preventDefault();
              const el = document.getElementById("map");
              if (el) {
                const top = el.getBoundingClientRect().top + window.scrollY - 72;
                window.scrollTo({ top, behavior: "smooth" });
              }
            }}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-slate-800 hover:bg-slate-900 text-white text-sm font-semibold rounded-md shadow-sm transition-colors"
          >
            <MapPin className="h-4 w-4" />
            Xem bản đồ dự báo cháy
          </a>
          <a
            href="#dashboard"
            onClick={(e) => {
              e.preventDefault();
              const el = document.getElementById("dashboard");
              if (el) {
                const top = el.getBoundingClientRect().top + window.scrollY - 72;
                window.scrollTo({ top, behavior: "smooth" });
              }
            }}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-white hover:bg-stone-50 text-stone-900 text-sm font-semibold rounded-md border border-stone-300 shadow-sm transition-colors"
          >
            Xem kết quả nghiên cứu
            <ArrowDown className="h-4 w-4" />
          </a>
        </motion.div>

        {/* Key findings cards */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.25 }}
          className="mt-14 grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4"
        >
          {KEY_FINDINGS.map((k) => {
            const tone = TONE_STYLES[k.tone];
            return (
              <div
                key={k.label}
                className={`p-4 sm:p-5 rounded-xl ${tone.bg} ring-1 ${tone.ring} card-hover`}
              >
                <div className={`text-2xl sm:text-3xl font-extrabold ${tone.text}`}>
                  {k.value}
                </div>
                <div className="mt-1 text-xs text-stone-600 font-medium">{k.sub}</div>
                <div className="mt-2 text-xs text-stone-700 leading-snug">{k.label}</div>
                <div className="mt-2 text-[10px] text-stone-500 leading-tight">{k.note}</div>
              </div>
            );
          })}
        </motion.div>

        {/* Sự sụt giảm cảnh báo */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35 }}
          className="mt-10 p-4 sm:p-5 rounded-xl bg-slate-100 ring-1 ring-slate-200 flex flex-col sm:flex-row items-start sm:items-center gap-3"
        >
          <AlertTriangle className="h-6 w-6 text-slate-500 flex-shrink-0" />
          <p className="text-sm text-stone-700 leading-relaxed">
            <strong className="text-stone-900">Phát hiện chính:</strong> khi chuyển từ miền
            Bắc sang Nam, AUROC trên VIIRS sụt giảm từ{" "}
            <strong className="text-slate-700">0,843</strong> (IID) xuống{" "}
            <strong className="text-slate-700">0,584</strong> — gần mức ngẫu nhiên. Trong khi
            đó, kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,02.
            <span className="text-stone-600 italic"> Vấn đề khó hơn không phải là dự báo xa hơn, mà là chuyển sang vùng địa lý mới.</span>
          </p>
        </motion.div>
      </div>
    </section>
  );
}
