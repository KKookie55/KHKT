"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PROJECT } from "@/lib/data";
import {
  Flame, FileText, Database, GitBranch, Beaker,
  AlertTriangle, ArrowRight, MapPin, Calendar, User,
} from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

const CONTRIBUTIONS = [
  {
    icon: <GitBranch className="h-5 w-5 text-slate-600" />,
    title: "Quy trình đánh giá OOD không gian đa chiều",
    text: "5 phép chia (IID, Temporal, 3 spatial splits) thay vì một IID ngẫu nhiên — tách biệt 3 câu hỏi khác nhau về khái quát.",
  },
  {
    icon: <Beaker className="h-5 w-5 text-slate-600" />,
    title: "Đối chiếu 2 nguồn nhãn độc lập",
    text: "MODIS (nhãn chính) và VIIRS (kiểm chứng) — phát hiện sụt giảm AUROC 0,842 → 0,584 khi chuyển miền trên VIIRS, không thấy trên MODIS.",
  },
  {
    icon: <Database className="h-5 w-5 text-slate-600" />,
    title: "Phân tách nguyên nhân 4 lớp",
    text: "Distribution shift, concept drift, prevalence, spatial autocorrelation — kết luận concept drift là cơ chế chính, bác bỏ prevalence.",
  },
  {
    icon: <Flame className="h-5 w-5 text-slate-600" />,
    title: "PhysMLP như phép kiểm tra tính nhất quán",
    text: "Ràng buộc đơn điệu cải thiện quan hệ gió/độ ẩm, nhưng không giải quyết được khái quát ngoài miền.",
  },
];

export default function AboutSection() {
  return (
    <section id="about" className="py-16 bg-stone-50">
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
            Giới thiệu
          </Badge>
          <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-stone-900">
            Về đề tài nghiên cứu
          </h2>
          <p className="mt-3 text-stone-600 max-w-2xl mx-auto">
            Một dự án khoa học kĩ thuật cấp THCS, nằm trong lĩnh vực Khoa học máy tính
            &amp; Khoa học Trái đất – Môi trường.
          </p>
        </motion.div>

        {/* Project info */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="mt-10 grid lg:grid-cols-2 gap-6"
        >
          <Card className="card-hover">
            <CardHeader>
              <CardTitle className="text-base">Thông tin dự án</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <InfoRow icon={<User className="h-4 w-4 text-stone-500" />} label="Tác giả">
                <span className="font-semibold text-stone-900">{PROJECT.author}</span>
              </InfoRow>
              <InfoRow icon={<Calendar className="h-4 w-4 text-stone-500" />} label="Năm">
                {PROJECT.year}
              </InfoRow>
              <InfoRow icon={<MapPin className="h-4 w-4 text-stone-500" />} label="Cấp học">
                {PROJECT.level}
              </InfoRow>
              <InfoRow icon={<FileText className="h-4 w-4 text-stone-500" />} label="Lĩnh vực">
                {PROJECT.field}
              </InfoRow>
              <InfoRow icon={<Database className="h-4 w-4 text-stone-500" />} label="Quy mô dữ liệu">
                <span className="font-mono text-stone-900 font-semibold">
                  {PROJECT.dataPoints.toLocaleString("vi-VN")}
                </span>{" "}
                ô–ngày · {PROJECT.cells} ô lưới 0,5° · {PROJECT.days} ngày
              </InfoRow>
              <InfoRow icon={<Beaker className="h-4 w-4 text-stone-500" />} label="Số lượt chạy">
                <span className="font-mono text-stone-900 font-semibold">{PROJECT.runs}</span>{" "}
                (5 seed × 5 split × 5 model + extras)
              </InfoRow>
              <InfoRow icon={<Calendar className="h-4 w-4 text-stone-500" />} label="Giai đoạn">
                {PROJECT.period}
              </InfoRow>
            </CardContent>
          </Card>

          <Card className="card-hover">
            <CardHeader>
              <CardTitle className="text-base">Kết luận chính</CardTitle>
              <CardDescription>
                Bốn phát hiện cốt lõi từ 200 lượt chạy thí nghiệm.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <ConclusionItem num="K1" tone="neutral"
                  text="Khoảng dự báo không phải vấn đề lớn: giảm ≈ 0,02 khi kéo dài t+1 → t+7 trong cùng miền."
                />
                <ConclusionItem num="K2" tone="danger"
                  text="Chuyển miền địa lý khó hơn đáng kể: VIIRS sụt giảm 0,842 → 0,584 (gần mức ngẫu nhiên)."
                />
                <ConclusionItem num="K3" tone="warning"
                  text="Sự sụt giảm không do prevalence: case-control 1:1 không đưa AUROC trở lại; nguyên nhân là distribution shift + concept drift + spatial autocorrelation."
                />
                <ConclusionItem num="K4" tone="neutral"
                  text="Ràng buộc vật lý (PhysMLP) cải thiện một số quan hệ đơn điệu (gió, độ ẩm) nhưng KHÔNG giải quyết được khái quát ngoài miền."
                />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Contributions */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.10 }}
          className="mt-6"
        >
          <Card className="card-hover">
            <CardHeader>
              <CardTitle className="text-base">Bốn đóng góp mới cốt lõi của đề tài</CardTitle>
              <CardDescription>
                Những điểm khác biệt so với các nghiên cứu trước đây về dự báo cháy rừng tại Việt Nam.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid sm:grid-cols-2 gap-4">
                {CONTRIBUTIONS.map((c, i) => (
                  <div
                    key={i}
                    className="p-4 rounded-lg ring-1 ring-stone-200 bg-gradient-to-br from-stone-50 to-white card-hover"
                  >
                    <div className="flex items-center gap-2">
                      {c.icon}
                      <h3 className="text-sm font-bold text-stone-900">{c.title}</h3>
                    </div>
                    <p className="mt-2 text-xs text-stone-700 leading-relaxed">{c.text}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Limitations */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="mt-6"
        >
          <Card className="card-hover">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-slate-500" />
                Hạn chế của đề tài
              </CardTitle>
            </CardHeader>
            <CardContent className="grid sm:grid-cols-3 gap-4 text-sm">
              <div className="p-3 rounded-md bg-slate-50 ring-1 ring-slate-200">
                <div className="font-semibold text-slate-700">H1 — Biến đầu vào</div>
                <p className="mt-1 text-xs text-stone-700">
                  Chưa có NDVI, lớp phủ đất, độ cao địa hình, biến hoạt động con người.
                  Không xác định được yếu tố riêng gây ra tín hiệu địa lý.
                </p>
              </div>
              <div className="p-3 rounded-md bg-slate-50 ring-1 ring-slate-200">
                <div className="font-semibold text-slate-700">H2 — Nguồn nhãn</div>
                <p className="mt-1 text-xs text-stone-700">
                  MODIS và VIIRS đều có giới hạn phát hiện (cháy nhỏ, dưới tán, mây che).
                  Jaccard giữa hai nguồn chỉ ≈ 0,106.
                </p>
              </div>
              <div className="p-3 rounded-md bg-slate-50 ring-1 ring-slate-200">
                <div className="font-semibold text-slate-700">H3 — Số seed cho t+3/t+7</div>
                <p className="mt-1 text-xs text-stone-700">
                  Các thí nghiệm khoảng dự báo xa mới dùng seed 42, chưa có std.
                  Được xem như khảo sát ban đầu.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Future work */}
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
                <ArrowRight className="h-4 w-4 text-slate-600" />
                Hướng phát triển tiếp theo
              </CardTitle>
            </CardHeader>
            <CardContent className="grid sm:grid-cols-3 gap-4 text-sm">
              <div className="p-3 rounded-md ring-1 ring-slate-200 bg-slate-50">
                <div className="font-semibold text-emerald-800">+ Biến môi trường</div>
                <p className="mt-1 text-xs text-stone-700">
                  Bổ sung NDVI, lớp phủ đất (ESA WorldCover), độ cao (SRTM) và biến
                  hoạt động con người. Đánh giá lại bằng cùng 5 phép chia.
                </p>
              </div>
              <div className="p-3 rounded-md ring-1 ring-slate-200 bg-slate-50">
                <div className="font-semibold text-emerald-800">+ Quy mô quốc tế</div>
                <p className="mt-1 text-xs text-stone-700">
                  Kiểm tra mô hình trên Lào, Campuchia để xem concept drift có lặp lại
                  ở quy mô lớn hơn không.
                </p>
              </div>
              <div className="p-3 rounded-md ring-1 ring-slate-200 bg-slate-50">
                <div className="font-semibold text-emerald-800">+ Lớp cảnh báo OOD</div>
                <p className="mt-1 text-xs text-stone-700">
                  Xây dựng cơ chế nhận biết khi dữ liệu đầu vào ngoài miền đã học,
                  thay vì chỉ trả về một xác suất cháy.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}

function InfoRow({
  icon, label, children,
}: { icon: React.ReactNode; label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2">
      <span className="flex-shrink-0 mt-0.5">{icon}</span>
      <div className="flex-1 text-stone-700 text-sm">
        <span className="text-stone-500 text-xs uppercase tracking-wide mr-2">{label}</span>
        <span>{children}</span>
      </div>
    </div>
  );
}

function ConclusionItem({
  num, tone, text,
}: { num: string; tone: "neutral" | "danger" | "warning"; text: string }) {
  const colors = {
    neutral: { bg: "bg-stone-100", text: "text-stone-700" },
    danger:  { bg: "bg-red-100",    text: "text-slate-700" },
    warning: { bg: "bg-slate-100",  text: "text-slate-600" },
  };
  return (
    <div className="flex items-start gap-3 p-3 rounded-md ring-1 ring-stone-200">
      <span
        className={`flex-shrink-0 inline-flex items-center justify-center h-7 w-7 rounded-md text-xs font-bold ${colors[tone].bg} ${colors[tone].text}`}
      >
        {num}
      </span>
      <p className="text-sm text-stone-800 leading-relaxed flex-1">{text}</p>
    </div>
  );
}
