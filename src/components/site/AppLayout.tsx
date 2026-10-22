"use client";

import { ArrowLeft } from "lucide-react";
import dynamic from "next/dynamic";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

interface AppLayoutProps {
  onBackToLanding: () => void;
}

export default function AppLayout({ onBackToLanding }: AppLayoutProps) {
  return (
    <div className="h-screen flex flex-col bg-white overflow-hidden">
      <header className="h-9 border-b border-slate-200 flex items-center justify-between px-4 flex-shrink-0">
        <button
          onClick={onBackToLanding}
          className="text-xs text-slate-600 hover:text-slate-900 inline-flex items-center gap-1"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Trang chủ
        </button>
        <div className="text-[10px] text-slate-400 truncate">
          KHKT 2026 · Hoàng Bảo Lâm · v9.4 · AUROC 0,979
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        <MapView />
      </main>
    </div>
  );
}
