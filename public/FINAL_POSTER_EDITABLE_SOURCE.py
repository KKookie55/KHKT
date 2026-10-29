#!/usr/bin/env python3
"""FINAL_POSTER.pdf — simple 1-page A1 portrait, single-column flow.

Reuses the design of 05_Phuluc_7_Poster_A1.pdf but with corrected:
  - School: THCS Lê Quý Đôn
  - GVHD: Lê Xuân Tân
  - Added: Hiệu trưởng: Nguyễn Anh Tuấn
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import A1
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle

pdfmetrics.registerFont(TTFont('Lib', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Bold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Italic', '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf'))
registerFontFamily('Lib', normal='Lib', bold='Lib-Bold', italic='Lib-Italic')

pdfmetrics.registerFont(TTFont('LibS', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('LibS-Bold', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
registerFontFamily('LibS', normal='LibS', bold='LibS-Bold')

OUT = Path('/home/z/my-project/KHKT_HoSo/FINAL_POSTER.pdf')

# Styles
st_title = ParagraphStyle('PT', fontName='Lib-Bold', fontSize=24, leading=28,
                          alignment=TA_CENTER, textColor=colors.HexColor('#0f172a'), spaceAfter=4)
st_subtitle = ParagraphStyle('PSub', fontName='Lib-Bold', fontSize=18, leading=22,
                             alignment=TA_CENTER, textColor=colors.HexColor('#1e40af'), spaceAfter=6)
st_author = ParagraphStyle('PA', fontName='Lib-Bold', fontSize=12, leading=16,
                           alignment=TA_CENTER, textColor=colors.HexColor('#0f172a'), spaceAfter=2)
st_meta = ParagraphStyle('PM', fontName='LibS', fontSize=10, leading=13,
                         alignment=TA_CENTER, textColor=colors.HexColor('#475569'), spaceAfter=2)
st_section = ParagraphStyle('PS', fontName='LibS-Bold', fontSize=14, leading=18,
                            textColor=colors.white, alignment=TA_LEFT,
                            leftIndent=6, spaceBefore=3, spaceAfter=3)
st_body = ParagraphStyle('PB', fontName='Lib', fontSize=11, leading=15,
                         alignment=TA_JUSTIFY, textColor=colors.HexColor('#1e293b'), spaceAfter=3)
st_bullet = ParagraphStyle('PBL', parent=st_body, leftIndent=14, bulletIndent=2, spaceAfter=2)
st_caption = ParagraphStyle('PC', fontName='Lib-Italic', fontSize=9, leading=11,
                            alignment=TA_CENTER, textColor=colors.HexColor('#475569'))
st_metric = ParagraphStyle('PMet', fontName='LibS-Bold', fontSize=22, leading=26,
                           alignment=TA_CENTER, textColor=colors.HexColor('#1e40af'))
st_metric_lbl = ParagraphStyle('PMetL', fontName='LibS', fontSize=9, leading=11,
                               alignment=TA_CENTER, textColor=colors.HexColor('#475569'))
st_ref = ParagraphStyle('PR', fontName='Lib', fontSize=8, leading=10,
                        alignment=TA_LEFT, textColor=colors.HexColor('#475569'),
                        leftIndent=11, firstLineIndent=-11, spaceAfter=1)


def section(title):
    """Navy section header bar."""
    t = Table([[Paragraph(title, st_section)]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0f172a')),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return t


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A1,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title='FINAL POSTER - Wildfire Forecasting System - KHKT 2026-2027',
        author='Hoàng Bảo Lâm',
        creator='ReportLab',
    )
    avail_w = A1[0] - 40*mm  # 594 - 40 = 554 mm

    s = []

    # Title
    s.append(Paragraph('HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY', st_title))
    s.append(Paragraph('DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN BẰNG MÔ HÌNH HỌC MÁY', st_title))
    s.append(Spacer(1, 4*mm))
    s.append(Paragraph('Hoàng Bảo Lâm', st_author))
    s.append(Paragraph('Lớp 9E4 · Trường THCS Lê Quý Đôn · Hà Nội', st_meta))
    s.append(Paragraph('GVHD: Lê Xuân Tân (Giáo viên Tin học) · Hiệu trưởng: Nguyễn Anh Tuấn', st_meta))
    s.append(Paragraph('Cuộc thi KHKT Hà Nội 2026-2027 · Mã: HN26-129F5F83 · wildfireforecastvietnam.netlify.app', st_meta))
    s.append(Spacer(1, 3*mm))

    # Metrics row
    metrics = Table([[
        Paragraph('<b>0,977</b>', st_metric),
        Paragraph('<b>0,038</b>', st_metric),
        Paragraph('<b>0,871</b>', st_metric),
        Paragraph('<b>964K</b>', st_metric),
        Paragraph('<b>710</b>', st_metric),
        Paragraph('<b>75</b>', st_metric),
    ], [
        Paragraph('AUROC OOD', st_metric_lbl),
        Paragraph('Brier cal', st_metric_lbl),
        Paragraph('AUPR', st_metric_lbl),
        Paragraph('Huyện-ngày', st_metric_lbl),
        Paragraph('Huyện', st_metric_lbl),
        Paragraph('Đặc trưng', st_metric_lbl),
    ]], colWidths=[avail_w/6]*6)
    metrics.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eff6ff')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#dbeafe')),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#1e40af')),
        ('LINEAFTER', (0,0), (-2,-1), 0.7, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    s.append(metrics)
    s.append(Spacer(1, 5*mm))

    # TÓM TẮT
    s.append(section('TÓM TẮT'))
    s.append(Paragraph(
        'Hệ thống dự báo cháy rừng cấp huyện cho Việt Nam bằng mô hình XGBoost + isotonic '
        'calibration. Dữ liệu: 710 huyện × 6 năm (2018-2023) × 75 đặc trưng. Phương pháp: '
        'cold-start spatial OOD (train Bắc+Trung, test Nam), Optuna 30 trials leakage-free '
        '(eval_set=val, không rò rỉ dữ liệu). Kết quả: AUROC 0,977, Brier 0,038, AUPR 0,871. '
        'Web app deployed trên Netlify với JS inference engine 297 dòng TypeScript + FIRMS '
        'real-time integration.', st_body))
    s.append(Spacer(1, 4*mm))

    # PHƯƠNG PHÁP
    s.append(section('PHƯƠNG PHÁP'))
    s.append(Paragraph(
        '<b>1. Dữ liệu:</b> NASA POWER (thời tiết 0,5°) + MODIS Active Fire + VIIRS 375m + '
        'GADM ADM2 (710 huyện) + MODIS MOD13A1 NDVI + ESA WorldCover 10m 2020 + Copernicus DEM 30m.',
        st_body))
    s.append(Paragraph(
        '<b>2. Feature engineering:</b> 75 đặc trưng = 35 thời tiết (lag 1/3/7/14 + rolling '
        '7/14/30) + 7 anomalies + 6 FWI + 7 landcover thật (ESA WorldCover) + 9 địa lý + 2 '
        'NDVI thật (MODIS) + 4 cell_fire_rate + 5 tương tác + 1 spatial lag (neighbor_fire, '
        'queen contiguity 1934 edges) + 2 centroid + doy.', st_body))
    s.append(Paragraph(
        '<b>3. Split:</b> Cold-start spatial OOD — train Bắc+Trung (lat≥14°, 490.560 rows, 224 '
        'huyện) / val 120.450 rows (55 huyện held-out) / test Nam (lat&lt;14°, 352.590 rows, '
        '161 huyện) — chưa từng thấy trong train.', st_body))
    s.append(Paragraph(
        '<b>4. Huấn luyện:</b> XGBoost + Optuna 30 trials. <b>LEAKAGE-FREE:</b> eval_set=val '
        '(không dùng test), Optuna objective = val AUROC. Best params: n_est=302, max_depth=8, '
        'lr=0,179.', st_body))
    s.append(Paragraph(
        '<b>5. Calibration:</b> IsotonicRegression fit trên (p_val, y_val). Áp dụng lên '
        'p_test_raw → p_test_calibrated.', st_body))
    s.append(Paragraph(
        '<b>6. Production:</b> JS inference engine 297 dòng TypeScript (XGBoost tree traversal '
        '+ isotonic piecewise linear) chạy trên Netlify serverless (không cần Python). FIRMS '
        'API real-time lấy điểm cháy 24h, gán vào huyện bằng point-in-polygon ray-casting + '
        'queen contiguity thật.', st_body))
    s.append(Spacer(1, 4*mm))

    # KẾT QUẢ
    s.append(section('KẾT QUẢ CHÍNH'))
    s.append(Paragraph(
        '<b>Top 3 đặc trưng quan trọng (gain):</b> (1) is_dry — cờ mùa khô (3587); (2) '
        'neighbor_fire — tỷ lệ huyện láng giềng có cháy (1154); (3) cell_fire_count_30d — số '
        'điểm cháy 30 ngày (663). Cả 3 đều có ý nghĩa vật lý rõ ràng: cháy xảy ra vào mùa khô, '
        'và nếu huyện láng giềng có cháy thì nguy cơ hiện tại cũng cao do điều kiện thời tiết '
        'và lớp phủ đất tương đồng.', st_body))
    s.append(Paragraph(
        '<b>Kết luận:</b> Mô hình XGBoost + isotonic calibration đạt <b>AUROC 0,977</b> trên '
        'cold-start spatial OOD test (vùng chưa từng thấy trong train) — chứng minh khả năng '
        'tổng quát hóa theo không gian. Hệ thống web chạy ổn định trên Netlify, cập nhật điểm '
        'cháy real-time qua FIRMS, hỗ trợ cơ quan kiểm lâm ra quyết định cảnh báo kịp thời.',
        st_body))
    s.append(Spacer(1, 4*mm))

    # HẠN CHẾ & HƯỚNG PHÁT TRIỂN
    s.append(section('HẠN CHẾ & HƯỚNG PHÁT TRIỂN'))
    s.append(Paragraph(
        '<b>Hạn chế:</b> Production lookup vẫn dùng default cho 4/9 landcover + elevation '
        'fields (lỗi đọc Copernicus DEM). Khoảng cách train-test AUROC 0,023 cho thấy '
        'overfitting. Chỉ dự báo 24h tới, chưa có multi-horizon forecasting (t+3, t+7, t+14). '
        'Chưa thử domain adaptation (CORAL, TENT, DANN) cho cold-start OOD.', st_body))
    s.append(Paragraph(
        '<b>Hướng phát triển:</b> Tích hợp SRTM 30m hoặc ASTER DEM 30m cho real elevation/'
        'slope. Thử ensemble XGBoost + LightGBM với stacked generalization. Thêm multi-horizon '
        'forecasting. Thử domain adaptation để cải thiện cold-start OOD. Cập nhật MODIS Terra+'
        'Aqua combined (2×/ngày).', st_body))
    s.append(Spacer(1, 4*mm))

    # TÀI LIỆU THAM KHẢO
    s.append(section('TÀI LIỆU THAM KHẢO'))
    refs = [
        '[1] Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD, 785-794.',
        '[2] Niculescu-Mizil, A., Caruana, R. (2005). Predicting Good Probabilities With Supervised Learning. ICML.',
        '[3] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5).',
        '[4] Justice, C. O. et al. (2002). The MODIS fire products. Remote Sensing of Environment, 83(1-2), 244-262.',
        '[5] Schroeder, W. et al. (2014). The New VIIRS 375m active fire detection data product. RSE, 143, 85-96.',
        '[6] ESA WorldCover 2020 v100. (2021). Remote Sensing of Environment, 269.',
        '[7] NASA POWER Project. https://power.larc.nasa.gov/',
        '[8] GADM Database v4.1. https://gadm.org',
        '[9] LeSage, J., Pace, R. K. (2009). Introduction to Spatial Econometrics. Chapman & Hall/CRC.',
        '[10] Jain, P. et al. (2020). A review of machine learning applications in wildfire science. Environmental Reviews, 28(4).',
    ]
    for r in refs:
        s.append(Paragraph(r, st_ref))
    s.append(Spacer(1, 4*mm))

    s.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#cbd5e1')))
    s.append(Spacer(1, 2*mm))
    s.append(Paragraph('Cuộc thi KHKT Hà Nội năm học 2026 – 2027', st_caption))

    doc.build(s)
    print(f'✓ FINAL_POSTER.pdf: {OUT} ({os.path.getsize(OUT)/1024:.1f} KB)')


if __name__ == '__main__':
    print('=== FINAL POSTER (A1 portrait, single-column, 1-page) ===\n')
    build()
    import subprocess
    result = subprocess.run(['pdfinfo', str(OUT)], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Pages' in line or 'Page size' in line:
            print(f'  {line.strip()}')
