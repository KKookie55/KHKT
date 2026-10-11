"use client";

import { useState } from "react";
import { ArrowLeft } from "lucide-react";
import dynamic from "next/dynamic";

const MapView = dynamic(() => import("./MapView"), { ssr: false });
const HistoricalReplay = dynamic(() => import("./HistoricalReplay"), { ssr: false });
const Dashboard = dynamic(() => import("./Dashboard"), { ssr: false });
const MethodSection = dynamic(() => import("./MethodSection"), { ssr: false });
const ReferencesSection = dynamic(() => import("./ReferencesSection"), { ssr: false });
const AboutSection = dynamic(() => import("./AboutSection"), { ssr: false });

interface AppLayoutProps {
  onBackToLanding: () => void;
}

const NAV_ITEMS: { id: string; label: string }[] = [
  { id: "map", label: "Bản đồ dự báo trực tiếp" },
  { id: "replay", label: "Tái dựng lịch sử" },
  { id: "results", label: "Kết quả nghiên cứu" },
  { id: "method", label: "Phương pháp" },
  { id: "references", label: "Tài liệu" },
  { id: "about", label: "Giới thiệu" },
];

export default function AppLayout({ onBackToLanding }: AppLayoutProps) {
  const [activeSection, setActiveSection] = useState("map");

  return (
    <div className="h-screen flex flex-col bg-white overflow-hidden">
      <header className="h-10 border-b border-slate-200 flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToLanding}
            className="text-xs text-slate-600 hover:text-slate-900 inline-flex items-center gap-1"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Trang giới thiệu
          </button>
          <div className="h-3 w-px bg-slate-300" />
          <div className="text-[11px] text-slate-500 truncate">
            Sản phẩm dự thi KHKT 2026 · Hoàng Bảo Lâm · THCS&THPT LÊ QUÝ ĐÔN · v8.7.2
          </div>
        </div>
        <div className="text-[11px] text-slate-400 hidden sm:block">
          {NAV_ITEMS.find(n => n.id === activeSection)?.label}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <aside className="w-52 border-r border-slate-200 bg-slate-50/40 flex-shrink-0 overflow-y-auto">
          <nav className="p-2 space-y-0.5">
            {NAV_ITEMS.map((item) => {
              const isActive = activeSection === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveSection(item.id)}
                  className={`w-full text-left px-3 py-2 text-xs rounded-md transition-colors ${
                    isActive
                      ? "bg-slate-800 text-white"
                      : "text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="mt-4 px-3 py-3 border-t border-slate-200 text-[10px] text-slate-500 leading-relaxed">
            <div className="font-semibold text-slate-700 mb-1">Dữ liệu</div>
            <div>· 598.143 ô-ngày</div>
            <div>· 273 ô lưới 0,5°</div>
            <div>· 2018-2023</div>
            <div>· 42 features (35 + 7 anom)</div>
            <div>· 5 seeds · 5 spatial splits</div>
          </div>
        </aside>

        <main className="flex-1 overflow-hidden">
          {activeSection === "map" && (
            <div className="h-full">
              <MapView />
            </div>
          )}
          {activeSection === "replay" && (
            <div className="h-full overflow-y-auto">
              <HistoricalReplay />
            </div>
          )}
          {activeSection === "results" && (
            <div className="h-full overflow-y-auto bg-slate-50/30">
              <Dashboard />
            </div>
          )}
          {activeSection === "method" && (
            <div className="h-full overflow-y-auto bg-slate-50/30">
              <MethodSection />
            </div>
          )}
          {activeSection === "references" && (
            <div className="h-full overflow-y-auto bg-slate-50/30">
              <ReferencesSection />
            </div>
          )}
          {activeSection === "about" && (
            <div className="h-full overflow-y-auto bg-slate-50/30">
              <AboutSection />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
