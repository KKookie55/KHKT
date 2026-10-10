"use client";

import { useState } from "react";
import { ArrowLeft } from "lucide-react";
import dynamic from "next/dynamic";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

interface AppLayoutProps {
  onBackToLanding: () => void;
}

const NAV_ITEMS: { id: string; label: string }[] = [
  { id: "map", label: "Bản đồ dự báo" },
  { id: "results", label: "Kết quả" },
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
            Sản phẩm dự thi KHKT 2026 · Hoàng Bảo Lâm · THCS&THPT LÊ QUÝ ĐÔN
          </div>
        </div>
        <div className="text-[11px] text-slate-400 hidden sm:block">
          {activeSection === "map" ? "Bản đồ dự báo cháy rừng" :
           NAV_ITEMS.find(n => n.id === activeSection)?.label}
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
            <div>· 200 lượt chạy</div>
          </div>
        </aside>

        <main className="flex-1 overflow-hidden">
          {activeSection === "map" && (
            <div className="h-full">
              <MapView />
            </div>
          )}
          {activeSection !== "map" && (
            <div className="h-full overflow-y-auto bg-slate-50/30">
              {activeSection === "results" && <ResultsPlaceholder />}
              {activeSection === "method" && <MethodPlaceholder />}
              {activeSection === "references" && <ReferencesPlaceholder />}
              {activeSection === "about" && <AboutPlaceholder />}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function ResultsPlaceholder() {
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-xl font-bold text-slate-900 mb-4">Kết quả nghiên cứu</h2>
      <div className="space-y-4">
        <div className="bg-white p-4 rounded-lg ring-1 ring-slate-200">
          <div className="text-sm font-semibold text-slate-700 mb-2">AUROC theo 5 phép chia</div>
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="p-2 bg-slate-50 rounded">IID: 0,843 (VIIRS)</div>
            <div className="p-2 bg-slate-50 rounded">Bắc→Nam: 0,584 (VIIRS)</div>
            <div className="p-2 bg-slate-50 rounded">t+1→t+7: giảm ≈0,02</div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg ring-1 ring-slate-200">
          <div className="text-sm font-semibold text-slate-700 mb-2">3 lớp kết luận</div>
          <div className="space-y-1 text-xs text-slate-600">
            <div><strong>(1) Đã quan sát:</strong> AUROC sụt giảm mạnh khi chuyển miền (0,84→0,58)</div>
            <div><strong>(2) Có thể giải thích:</strong> distribution shift + concept drift</div>
            <div><strong>(3) Chưa xác định:</strong> biến môi trường nào mà tọa độ mã hóa</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MethodPlaceholder() {
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-xl font-bold text-slate-900 mb-4">Phương pháp</h2>
      <div className="bg-white p-4 rounded-lg ring-1 ring-slate-200 text-sm text-slate-600 space-y-2">
        <div><strong>Dữ liệu:</strong> NASA POWER + MODIS + VIIRS, 598.143 ô-ngày</div>
        <div><strong>5 mô hình:</strong> LogReg, RF, XGBoost, MLP, PhysMLP</div>
        <div><strong>5 phép chia:</strong> IID, Temporal, Bắc→Nam, Nam→Trung, Trung+Nam→Bắc</div>
        <div><strong>Chỉ số:</strong> AUROC, Brier score, ECE, Moran's I</div>
        <div><strong>Hiệu chuẩn:</strong> Isotonic regression</div>
      </div>
    </div>
  );
}

function ReferencesPlaceholder() {
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-xl font-bold text-slate-900 mb-4">Tài liệu tham khảo</h2>
      <div className="bg-white p-4 rounded-lg ring-1 ring-slate-200 text-sm text-slate-600">
        14 tài liệu theo chuẩn IEEE. Xem chi tiết trong báo cáo PDF.
      </div>
    </div>
  );
}

function AboutPlaceholder() {
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-xl font-bold text-slate-900 mb-4">Giới thiệu</h2>
      <div className="bg-white p-4 rounded-lg ring-1 ring-slate-200 text-sm text-slate-600 space-y-2">
        <div><strong>Tác giả:</strong> Hoàng Bảo Lâm</div>
        <div><strong>Lớp:</strong> 9E4</div>
        <div><strong>Trường:</strong> THCS&THPT LÊ QUÝ ĐÔN</div>
        <div><strong>GV hướng dẫn:</strong> Lê Xuân Tân</div>
        <div><strong>Lĩnh vực:</strong> Phần mềm hệ thống</div>
      </div>
    </div>
  );
}
