import type { Metadata } from "next";
import { Inter, Geist_Mono } from "next/font/google";
import "./globals.css";
import "leaflet/dist/leaflet.css";
import "leaflet.markercluster/dist/MarkerCluster.css";
import "leaflet.markercluster/dist/MarkerCluster.Default.css";
import { Toaster } from "@/components/ui/toaster";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin", "vietnamese"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Wildfire OOD — Dự báo cháy rừng Việt Nam | Hoàng Bảo Lâm",
  description:
    "Hệ thống dự báo nguy cơ cháy rừng dựa trên MODIS, VIIRS và NASA POWER. Đánh giá khả năng khái quát không gian của 5 mô hình học máy trên 598.143 ô–ngày. Đề tài KHKT 2026, Hoàng Bảo Lâm.",
  keywords: [
    "cháy rừng", "MODIS", "VIIRS", "XGBoost", "PhysMLP",
    "kiểm tra ngoài miền", "khái quát không gian", "NASA POWER",
    "Moran's I", "khoa học kĩ thuật", "Việt Nam",
  ],
  authors: [{ name: "Hoàng Bảo Lâm" }],
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "Wildfire OOD — Dự báo cháy rừng Việt Nam",
    description:
      "Đánh giá độ tin cậy và khả năng khái quát không gian của mô hình dự báo cháy rừng.",
    siteName: "Wildfire OOD Research",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body
        className={`${inter.variable} ${geistMono.variable} antialiased bg-white text-stone-900`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
