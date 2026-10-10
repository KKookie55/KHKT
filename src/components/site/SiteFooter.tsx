"use client";

import { Flame, FileText, Globe } from "lucide-react";
import { PROJECT } from "@/lib/data";

export default function SiteFooter() {
  return (
    <footer className="bg-stone-900 text-stone-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Project */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2">
              <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-slate-700 to-slate-900 text-white">
                <Flame className="h-5 w-5" />
              </span>
              <div>
                <div className="text-sm font-bold text-white">Wildfire OOD Research</div>
                <div className="text-[10px] text-stone-400">
                  {PROJECT.author} · KHKT {PROJECT.year}
                </div>
              </div>
            </div>
            <p className="mt-4 text-sm text-stone-400 max-w-md leading-relaxed">
              Đánh giá độ tin cậy và khả năng khái quát không gian của mô hình dự báo
              cháy rừng dựa trên dữ liệu vệ tinh. Đề tài KHKT cấp THCS, lĩnh vực Khoa học
              máy tính &amp; Khoa học Trái đất – Môi trường.
            </p>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">Tài nguyên</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="/Baocao.docx"
                  download
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-stone-400 hover:text-slate-400 transition-colors"
                >
                  <FileText className="h-4 w-4" />
                  Báo cáo (Word)
                </a>
              </li>
              <li>
                <a
                  href="#references"
                  className="flex items-center gap-2 text-stone-400 hover:text-slate-400 transition-colors"
                >
                  <Globe className="h-4 w-4" />
                  Danh mục tài liệu IEEE
                </a>
              </li>
              <li>
                <a
                  href="https://power.larc.nasa.gov/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-stone-400 hover:text-slate-400 transition-colors"
                >
                  <Globe className="h-4 w-4" />
                  NASA POWER Project
                </a>
              </li>
              <li>
                <a
                  href="https://firms.modaps.eosdis.nasa.gov/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-stone-400 hover:text-slate-400 transition-colors"
                >
                  <Globe className="h-4 w-4" />
                  NASA FIRMS (MODIS/VIIRS)
                </a>
              </li>
            </ul>
          </div>

          {/* Sections */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">Mục lục</h3>
            <ul className="space-y-2 text-sm">
              {[
                { id: "map",        label: "Bản đồ dự báo" },
                { id: "dashboard",  label: "Kết quả nghiên cứu" },
                { id: "method",     label: "Phương pháp" },
                { id: "references", label: "Tài liệu tham khảo" },
                { id: "about",      label: "Giới thiệu đề tài" },
              ].map((item) => (
                <li key={item.id}>
                  <a
                    href={`#${item.id}`}
                    className="text-stone-400 hover:text-slate-400 transition-colors"
                  >
                    {item.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-10 pt-6 border-t border-stone-800 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-stone-500">
          <div>
            © 2026 {PROJECT.author}. Báo cáo dự án KHKT, Sở GD&amp;ĐT Hà Nội.
          </div>
          <div className="flex items-center gap-4">
            <span>Dữ liệu: NASA POWER · MODIS · VIIRS</span>
            <span>·</span>
            <span>Mô hình: XGBoost · PhysMLP</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
