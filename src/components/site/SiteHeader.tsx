"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Flame, Menu, X, FileText, MapPin, Calendar } from "lucide-react";

const NAV_ITEMS = [
  { id: "map",        label: "Bản đồ dự báo", icon: MapPin },
  { id: "heatmap",    label: "Heatmap mùa vụ", icon: Calendar },
  { id: "dashboard",  label: "Kết quả",       icon: null },
  { id: "method",     label: "Phương pháp",   icon: null },
  { id: "references", label: "Tài liệu",     icon: null },
  { id: "about",      label: "Giới thiệu",   icon: null },
];

export default function SiteHeader() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    window.addEventListener("scroll", onScroll);
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY - 72;
      window.scrollTo({ top, behavior: "smooth" });
    }
    setMobileOpen(false);
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-[1000] transition-all duration-300 ${
        scrolled
          ? "bg-white/95 backdrop-blur-md shadow-sm border-b border-stone-200"
          : "bg-white/80 backdrop-blur-sm border-b border-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="flex items-center gap-2.5 group"
          >
            <span className="relative inline-flex h-9 w-9 items-center justify-center rounded-lg bg-slate-800 text-white shadow-sm">
              <Flame className="h-5 w-5" />
            </span>
            <div className="hidden sm:block text-left">
              <div className="text-sm font-bold text-stone-900 leading-tight">
                Wildfire OOD
              </div>
              <div className="text-[10px] text-stone-500 leading-tight">
                Hoàng Bảo Lâm · KHKT 2026
              </div>
            </div>
          </button>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                onClick={() => scrollTo(item.id)}
                className="px-3 py-2 text-sm font-medium text-stone-700 hover:text-slate-900 hover:bg-slate-100 rounded-md transition-colors"
              >
                {item.label}
              </button>
            ))}
          </nav>

          {/* CTA + Mobile toggle */}
          <div className="flex items-center gap-2">
            <a
              href="/Baocao.docx"
              download
              target="_blank"
              rel="noopener noreferrer"
              className="hidden md:inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-stone-900 hover:bg-stone-800 text-white rounded-md transition-colors"
            >
              <FileText className="h-3.5 w-3.5" />
              Báo cáo (Word)
            </a>
            <button
              onClick={() => setMobileOpen((v) => !v)}
              className="md:hidden p-2 rounded-md hover:bg-stone-100 text-stone-700"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>

        {/* Mobile nav */}
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="md:hidden border-t border-stone-200 py-3"
          >
            <nav className="flex flex-col gap-1">
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollTo(item.id)}
                  className="px-3 py-2 text-left text-sm font-medium text-stone-700 hover:bg-slate-100/60 rounded-md"
                >
                  {item.label}
                </button>
              ))}
              <a
                href="/Baocao.docx"
                download
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-2 text-left text-sm font-semibold text-white bg-stone-900 rounded-md inline-flex items-center gap-1.5"
              >
                <FileText className="h-3.5 w-3.5" />
                Báo cáo (Word)
              </a>
            </nav>
          </motion.div>
        )}
      </div>
    </header>
  );
}
