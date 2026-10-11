"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Search, ExternalLink, BookOpen } from "lucide-react";
import { useMemo, useState } from "react";
import { REFERENCES } from "@/lib/data";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export default function ReferencesSection() {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (!query.trim()) return REFERENCES;
    const q = query.toLowerCase();
    return REFERENCES.filter(
      (r) => r.text.toLowerCase().includes(q) || r.num.includes(q)
    );
  }, [query]);

  return (
    <section id="references" className="py-16 bg-white">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <Badge variant="outline" className="text-slate-700 border-slate-300 bg-slate-50">
            Tài liệu tham khảo
          </Badge>
          <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-stone-900">
            Danh mục IEEE — có hyperlink
          </h2>
          <p className="mt-3 text-stone-600 max-w-2xl mx-auto">
            {REFERENCES.length} tài liệu theo chuẩn IEEE, đánh số theo thứ tự xuất hiện
            trong báo cáo. Mỗi mục kèm DOI/URL bấm được.
          </p>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="mt-8 max-w-md mx-auto"
        >
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-stone-400" />
            <Input
              placeholder="Tìm theo tên tác giả, từ khoá, hoặc số thứ tự..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.10 }}
          className="mt-8"
        >
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-slate-600" />
                {filtered.length} tài liệu
              </CardTitle>
              <CardDescription>
                {query ? `Kết quả cho: "${query}"` : "Toàn bộ danh mục"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ol className="space-y-3">
                {filtered.map((r) => (
                  <li
                    key={r.num}
                    className="flex flex-col sm:flex-row sm:items-start gap-2 p-3 rounded-md hover:bg-stone-50 transition-colors ring-1 ring-transparent hover:ring-stone-200"
                  >
                    <span className="flex-shrink-0 inline-flex items-center justify-center h-7 w-9 rounded-md bg-stone-900 text-white text-xs font-bold font-mono">
                      {r.num}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-stone-800 leading-relaxed">{r.text}</p>
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-1.5 inline-flex items-center gap-1 text-xs text-slate-700 hover:text-slate-800 hover:underline break-all"
                      >
                        <ExternalLink className="h-3 w-3 flex-shrink-0" />
                        {r.url}
                      </a>
                    </div>
                  </li>
                ))}
                {filtered.length === 0 && (
                  <li className="text-center py-8 text-stone-400 text-sm">
                    Không tìm thấy tài liệu phù hợp.
                  </li>
                )}
              </ol>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
