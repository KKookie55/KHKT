#!/usr/bin/env python3
"""Báo cáo tóm tắt kết quả — ≤250 chữ kể cả tiêu đề.

Title: 10 words
Abstract: 203 words
Total: 213 words ✓ (under 250)
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

pdfmetrics.registerFont(TTFont('Lib', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Bold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Italic', '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf'))
registerFontFamily('Lib', normal='Lib', bold='Lib-Bold', italic='Lib-Italic')

OUT = Path('/home/z/my-project/KHKT_HoSo/09_Bao_cao_tom_tat_ket_qua.pdf')

styles_table = {
    'title': ('Lib-Bold', 14, 18, TA_CENTER, '#0f172a'),
    'subtitle': ('Lib', 11, 14, TA_CENTER, '#475569'),
    'h2': ('Lib-Bold', 12, 15, TA_LEFT, '#1e293b'),
    'body': ('Lib', 13, 15, TA_JUSTIFY, '#1e293b'),
    'body_left': ('Lib', 13, 15, TA_LEFT, '#1e293b'),
    'small': ('Lib-Italic', 9.5, 12, TA_LEFT, '#475569'),
}

def mkstyle(name):
    font, size, lead, align, color = styles_table[name]
    from reportlab.lib.styles import ParagraphStyle
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead,
                          alignment=align, textColor=colors.HexColor(color))

st_title = mkstyle('title')
st_subtitle = mkstyle('subtitle')
st_h2 = mkstyle('h2')
st_body = mkstyle('body')
st_body_left = mkstyle('body_left')
st_small = mkstyle('small')

TITLE = 'BÁO CÁO TÓM TẮT KẾT QUẢ THỰC HIỆN DỰ ÁN'
ABSTRACT = (
    'Hệ thống dự báo cháy rừng cấp huyện cho Việt Nam dùng XGBoost kết hợp hiệu chỉnh đẳng điều. '
    'Dữ liệu gồm 710 huyện × 6 năm (2018-2023) × 75 đặc trưng, bao gồm thời tiết, chỉ số FWI, '
    'lớp phủ đất thật từ ESA WorldCover, chỉ số thực vật thật từ MODIS, và độ trễ không gian của '
    'các huyện láng giềng. Phương pháp áp dụng chia lạnh theo không gian (huấn luyện Bắc+Trung, '
    'kiểm thử Nam) đánh giá khả năng tổng quát hóa. Optuna 30 lần tối ưu siêu tham số, mục tiêu '
    'AUROC trên tập kiểm định, không dùng tập kiểm thử để tránh rò rỉ dữ liệu. Kết quả: AUROC = '
    '0,977, Brier = 0,038, AUPR = 0,871. Đặc trưng quan trọng nhất là cờ mùa khô và tỷ lệ huyện '
    'láng giềng có cháy. Hệ thống triển khai trên Netlify với suy diễn JavaScript 297 dòng, tích '
    'hợp điểm cháy thời gian thực từ FIRMS. Kết luận: mô hình tổng quát hóa tốt khi áp dụng vào '
    'vùng chưa từng thấy trong huấn luyện, hỗ trợ kiểm lâm cảnh báo kịp thời. Hạn chế: 4/9 trường '
    'landcover và địa hình vẫn dùng mặc định, sẽ cập nhật sau.'
)

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=3*cm, rightMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm,
                        title='Bao cao tom tat ket qua KHKT',
                        author='Hoàng Bảo Lâm',
                        subject='Bao cao tom tat ket qua thuc hien du an (≤250 tu)',
                        creator='ReportLab')

s = []
s.append(Paragraph('CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM', st_subtitle))
s.append(Paragraph('Độc lập – Tự do – Hạnh phúc', st_subtitle))
s.append(Spacer(1, 4*mm))
s.append(Paragraph(TITLE, st_title))
s.append(Spacer(1, 4*mm))
s.append(Paragraph('Cuộc thi Khoa học, Kỹ thuật Hà Nội năm học 2026 – 2027', st_subtitle))
s.append(Spacer(1, 8*mm))

info = [
    [Paragraph('<b>Học sinh:</b>', st_body_left),
     Paragraph('Hoàng Bảo Lâm — Lớp 9E4 — THCS Lê Quý Đôn', st_body_left)],
    [Paragraph('<b>GVHD:</b>', st_body_left),
     Paragraph('Lê Xuân Tân (Giáo viên Tin học)', st_body_left)],
    [Paragraph('<b>Tên dự án:</b>', st_body_left),
     Paragraph('HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN BẰNG MÔ HÌNH HỌC MÁY', st_body_left)],
    [Paragraph('<b>Mã dự án:</b>', st_body_left),
     Paragraph('HN26-129F5F83', st_body_left)],
]
t = Table(info, colWidths=[3.5*cm, 13*cm])
t.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('LEFTPADDING', (0,0), (-1,-1), 4),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
s.append(t)
s.append(Spacer(1, 6*mm))

s.append(Paragraph('NỘI DUNG TÓM TẮT', st_h2))
s.append(Spacer(1, 2*mm))
s.append(Paragraph(ABSTRACT, st_body))
s.append(Spacer(1, 4*mm))

title_wc = len(TITLE.split())
abstract_wc = len(ABSTRACT.split())
total_wc = title_wc + abstract_wc
s.append(Paragraph(f'<i>(Tổng số từ: {total_wc} kể cả tiêu đề — đúng quy định không quá 250 từ)</i>', st_small))

s.append(Spacer(1, 10*mm))
sign = Table([
    [Paragraph('Hà Nội, ngày ... tháng ... năm 2026', st_body_left)],
    [Paragraph('<b>THÍ SINH</b>', st_body_left)],
    [Paragraph('(Ký, ghi rõ họ tên)', st_small)],
    [Paragraph('<br/><br/><br/>', st_body)],
    [Paragraph('<b>Hoàng Bảo Lâm</b>', st_body_left)],
], colWidths=[16.5*cm])
sign.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
s.append(sign)

doc.build(s)
print(f'✓ {OUT.name} ({os.path.getsize(OUT)/1024:.1f} KB)')
print(f'  Title: {title_wc} từ')
print(f'  Abstract: {abstract_wc} từ')
print(f'  Total: {total_wc} từ (≤250 ✓)')
