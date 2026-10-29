#!/usr/bin/env python3
"""FINAL_POSTER.pdf — clean 1-page A1 portrait, ViSEF/ISEF-style.

Reuses the design from 05_Phuluc_7_Poster_A1.pdf but with correct:
  - School: THCS Lê Quý Đôn (not THCS & THPT)
  - GVHD: Lê Xuân Tân (not placeholder)
  - Added: Hiệu trưởng: Nguyễn Anh Tuấn
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import A1
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, FrameBreak,
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
OUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = A1  # 594×841 mm

# Palette
COL_NAVY = colors.HexColor('#0f172a')
COL_BLUE = colors.HexColor('#1e40af')
COL_CYAN = colors.HexColor('#0891b2')
COL_GREY = colors.HexColor('#475569')
COL_LIGHT = colors.HexColor('#f8fafc')
COL_BORDER = colors.HexColor('#cbd5e1')

MARGIN = 20*mm

# Styles
st_title = ParagraphStyle('PT', fontName='Lib-Bold', fontSize=26, leading=30,
                          alignment=TA_CENTER, textColor=COL_NAVY, spaceAfter=4)
st_subtitle = ParagraphStyle('PSub', fontName='Lib-Bold', fontSize=18, leading=22,
                             alignment=TA_CENTER, textColor=COL_BLUE, spaceAfter=6)
st_author = ParagraphStyle('PA', fontName='Lib-Bold', fontSize=14, leading=18,
                           alignment=TA_CENTER, textColor=COL_NAVY, spaceAfter=2)
st_meta = ParagraphStyle('PM', fontName='LibS', fontSize=11, leading=14,
                         alignment=TA_CENTER, textColor=COL_GREY, spaceAfter=2)
st_section = ParagraphStyle('PS', fontName='LibS-Bold', fontSize=16, leading=20,
                            textColor=colors.white, alignment=TA_LEFT,
                            leftIndent=6, spaceBefore=2, spaceAfter=2)
st_body = ParagraphStyle('PB', fontName='Lib', fontSize=12, leading=16,
                         alignment=TA_JUSTIFY, textColor=colors.HexColor('#1e293b'), spaceAfter=4)
st_body_sm = ParagraphStyle('PBS', fontName='Lib', fontSize=11, leading=14,
                            alignment=TA_JUSTIFY, textColor=colors.HexColor('#1e293b'), spaceAfter=2)
st_bullet = ParagraphStyle('PBL', parent=st_body, leftIndent=14, bulletIndent=2, spaceAfter=3)
st_caption = ParagraphStyle('PC', fontName='Lib-Italic', fontSize=10, leading=12,
                            alignment=TA_CENTER, textColor=COL_GREY)
st_metric = ParagraphStyle('PMet', fontName='LibS-Bold', fontSize=24, leading=28,
                           alignment=TA_CENTER, textColor=COL_BLUE)
st_metric_lbl = ParagraphStyle('PMetL', fontName='LibS', fontSize=10, leading=12,
                               alignment=TA_CENTER, textColor=COL_GREY)
st_ref = ParagraphStyle('PR', fontName='Lib', fontSize=9, leading=11,
                        alignment=TA_LEFT, textColor=COL_GREY, leftIndent=12, firstLineIndent=-12,
                        spaceAfter=1)


def section(title):
    """Create navy section header bar."""
    t = Table([[Paragraph(title, st_section)]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COL_NAVY),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    return t


def build():
    avail_w = PAGE_W - 2*MARGIN
    # 2 columns body
    col_gap = 10*mm
    col_w = (avail_w - col_gap) / 2
    header_h = 70*mm
    body_h = PAGE_H - 2*MARGIN - header_h - 8*mm

    doc = BaseDocTemplate(
        str(OUT), pagesize=A1,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='FINAL POSTER - Wildfire Forecasting System - KHKT 2026-2027',
        author='Hoàng Bảo Lâm',
        creator='ReportLab',
    )

    header_frame = Frame(MARGIN, PAGE_H - MARGIN - header_h, avail_w, header_h,
                         id='header', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    col1 = Frame(MARGIN, MARGIN, col_w, body_h, id='col1',
                 leftPadding=4, rightPadding=4, topPadding=4, bottomPadding=4)
    col2 = Frame(MARGIN + col_w + col_gap, MARGIN, col_w, body_h, id='col2',
                 leftPadding=4, rightPadding=4, topPadding=4, bottomPadding=4)

    template = PageTemplate(id='poster', frames=[header_frame, col1, col2])
    doc.addPageTemplates([template])

    s = []

    # HEADER
    s.append(Paragraph('HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY', st_title))
    s.append(Paragraph('DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN BẰNG MÔ HÌNH HỌC MÁY', st_title))
    s.append(Spacer(1, 3*mm))
    s.append(Paragraph('Hoàng Bảo Lâm', st_author))
    s.append(Paragraph('Lớp 9E4 · Trường THCS Lê Quý Đôn · Hà Nội', st_meta))
    s.append(Paragraph('GVHD: Lê Xuân Tân (Giáo viên Tin học) · Hiệu trưởng: Nguyễn Anh Tuấn', st_meta))
    s.append(Paragraph('Cuộc thi KHKT Hà Nội 2026-2027 · Mã dự án: HN26-129F5F83 · wildfireforecastvietnam.netlify.app', st_meta))
    s.append(Spacer(1, 2*mm))

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
        ('BOX', (0,0), (-1,-1), 1.5, COL_BLUE),
        ('LINEAFTER', (0,0), (-2,-1), 0.7, COL_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    s.append(metrics)

    # COL 1
    s.append(FrameBreak())

    s.append(section('1. VẤN ĐỀ NGHIÊN CỨU'))
    s.append(Paragraph(
        'Cháy rừng là thảm hoạ thường xuyên tại Việt Nam, đặc biệt Tây Nguyên, Nam Trung Bộ và '
        'ĐBSCL. Mỗi năm ghi nhận hàng nghìn vụ, thiêu rụi hàng chục nghìn ha rừng. Các hệ thống '
        'dự báo hiện có dựa trên chỉ số FWI của Canada (phát triển cho rừng lá kim ôn đới) — '
        'không phù hợp khí hậu nhiệt đới gió mùa VN. Chưa có hệ thống dự báo cấp huyện (nơi có '
        'thẩm quyền cảnh báo theo Luật Lâm nghiệp 2017).', st_body))
    s.append(Spacer(1, 3*mm))

    s.append(section('2. CÂU HỎI NGHIÊN CỨU'))
    s.append(Paragraph(
        '<b>Có thể xây dựng mô hình học máy dự báo xác suất cháy rừng cấp huyện với độ tin cậy '
        'calibration chấp nhận được khi áp dụng vào vùng chưa từng thấy trong tập huấn luyện '
        'không?</b>', st_body))
    s.append(Paragraph(
        '<b>Giả thuyết:</b> XGBoost + isotonic calibration fit trên val sẽ đạt AUROC ≥ 0,95 trên '
        'cold-start spatial OOD (train Bắc+Trung, test Nam) và Brier ≤ 0,05.', st_body))
    s.append(Spacer(1, 3*mm))

    s.append(section('3. NGUỒN DỮ LIỆU'))
    data = [
        ['<b>Nguồn</b>', '<b>Loại</b>', '<b>Độ phân giải</b>'],
        ['NASA POWER', 'Thời tiết (T2M, RH, WS, precip)', '0,5° daily'],
        ['MODIS Active Fire', 'Điểm cháy (nhãn chính)', '1 km × 2/ngày'],
        ['VIIRS 375m', 'Điểm cháy (kiểm chứng)', '375 m × 2/ngày'],
        ['GADM ADM2 v4.1', 'Ranh giới 710 huyện', 'Vector'],
        ['MODIS MOD13A1', 'NDVI thực vật', '500 m'],
        ['ESA WorldCover 10m', 'Lớp phủ đất thật', '10 m (2020)'],
        ['Copernicus DEM', 'Địa hình (độ cao)', '30 m'],
        ['FIRMS API', 'Điểm cháy real-time', '1 km × 24h'],
    ]
    rows = [[Paragraph(c, st_body_sm) for c in r] for r in data]
    t = Table(rows, colWidths=[4*cm, 7*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COL_NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'LibS-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.3, COL_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
    ]))
    s.append(t)
    s.append(Paragraph('<i>Bảng 1: 9 nguồn dữ liệu công khai đã sử dụng.</i>', st_caption))
    s.append(Spacer(1, 3*mm))

    s.append(section('4. PHƯƠNG PHÁP NGHIÊN CỨU'))
    s.append(Paragraph('<b>Pipeline 6 bước:</b>', st_body))
    for step in [
        '<b>1.</b> Thu thập dữ liệu vệ tinh (NASA, ESA, Microsoft Planetary Computer)',
        '<b>2.</b> Tổng hợp về cấp huyện-ngày (710 × 2191 = 1,55M rows)',
        '<b>3.</b> Feature engineering: 75 đặc trưng',
        '<b>4.</b> Chia cold-start spatial OOD (Bắc+Trung → Nam)',
        '<b>5.</b> Optuna 30 trials + XGBoost + isotonic calibration',
        '<b>6.</b> Deploy web app (Netlify + JS inference + FIRMS real-time)',
    ]:
        s.append(Paragraph(step, st_bullet))
    s.append(Spacer(1, 2*mm))
    s.append(Paragraph(
        '<b>Split:</b> Train 490.560 rows (224 huyện Bắc+Trung, lat≥14°) · Val 120.450 rows '
        '(55 huyện held-out) · Test 352.590 rows (161 huyện Nam, lat&lt;14°) — chưa từng thấy '
        'trong train.', st_body))
    s.append(Paragraph(
        '<b>Leak-free:</b> Objective = val AUROC, eval_set=[(X_val, y_val)]. Không dùng test set '
        'làm eval_set hay objective — tránh rò rỉ dữ liệu. <b>Calibration:</b> '
        'IsotonicRegression fit trên (p_val, y_val).', st_body))
    s.append(Spacer(1, 2*mm))
    s.append(Paragraph(
        '<b>75 đặc trưng:</b> 35 thời tiết (lag 1/3/7/14 + rolling 7/14/30) + 7 anomalies + 6 '
        'FWI + 7 landcover thật (ESA WorldCover) + 9 địa lý + 2 NDVI thật (MODIS) + 4 cell_fire_'
        'rate + 5 tương tác + 1 spatial lag (neighbor_fire, queen contiguity 1934 edges) + 3 '
        'centroid/doy.', st_body))

    # COL 2
    s.append(FrameBreak())

    s.append(section('5. KẾT QUẢ NGHIÊN CỨU'))
    res = [
        ['<b>Metric</b>', '<b>Train</b>', '<b>Val</b>', '<b>Test (OOD)</b>'],
        ['AUROC', '1,0000', '0,9980', '<b>0,977</b>'],
        ['AUPR', '—', '—', '0,871'],
        ['Brier (raw)', '—', '—', '0,039'],
        ['Brier (calibrated)', '—', '—', '<b>0,038</b>'],
    ]
    rows = [[Paragraph(c, st_body_sm) for c in r] for r in res]
    t = Table(rows, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COL_NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'LibS-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.3, COL_BORDER),
        ('BACKGROUND', (3,1), (3,-1), colors.HexColor('#eff6ff')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    s.append(t)
    s.append(Paragraph('<i>Bảng 2: Kết quả trên cold-start spatial OOD test.</i>', st_caption))
    s.append(Spacer(1, 3*mm))

    s.append(section('6. ĐẶC TRƯNG QUAN TRỌNG (GAIN)'))
    imp = [
        ['#', 'Feature', 'Gain', 'Ý nghĩa vật lý'],
        ['1', 'is_dry', '3587', 'Cờ mùa khô (độ ẩm &lt; ngưỡng)'],
        ['2', 'neighbor_fire', '1154', 'Tỷ lệ huyện láng giềng có cháy'],
        ['3', 'cell_fire_count_30d', '663', 'Số điểm cháy 30 ngày (climatology)'],
        ['4', 'doy', '73', 'Ngày trong năm (mùa vụ)'],
        ['5', 'koppen_Cwa', '62', 'Khí hậu Köppen Cwa'],
        ['6', 'cell_fire_rate_exp', '52', 'Tỷ lệ cháy kỳ vọng'],
        ['7', 'latitude_band', '51', 'Vĩ độ (chia band)'],
        ['8', 'centroid_lat', '45', 'Vĩ độ tâm huyện'],
        ['9', 'dry_wind', '41', 'Tương tác khô × gió'],
        ['10', 't2m_lag14', '38', 'Nhiệt độ 14 ngày trước'],
    ]
    rows = [[Paragraph(c, st_body_sm) for c in r] for r in imp]
    t = Table(rows, colWidths=[1*cm, 5*cm, 1.7*cm, 5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COL_BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'LibS-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.3, COL_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    s.append(t)
    s.append(Paragraph('<i>Bảng 3: Top 10 feature importance by gain.</i>', st_caption))
    s.append(Spacer(1, 3*mm))

    s.append(section('7. THẢO LUẬN & HẠN CHẾ'))
    s.append(Paragraph(
        '<b>Thảo luận:</b> AUROC 0,977 trên cold-start OOD (chưa từng thấy dữ liệu Nam trong '
        'train) cho thấy mô hình tổng quát hóa tốt khi áp dụng vào vùng mới. Top đặc trưng là '
        'is_dry và neighbor_fire — cả hai có ý nghĩa vật lý rõ ràng: cháy xảy ra vào mùa khô, '
        'và nếu huyện láng giềng có cháy thì nguy cơ hiện tại cũng cao. Hồi quy đẳng điều fit '
        'trên val (không phải test) đảm bảo Brier không rò rỉ dữ liệu. JS inference engine 297 '
        'dòng TypeScript chạy ổn định trên Netlify, cold start ~1s, inference ~50ms cho 710 '
        'huyện. FIRMS API real-time cập nhật neighbor_fire mỗi 10 phút qua point-in-polygon + '
        'queen contiguity thật (đã pre-compute).', st_body))
    s.append(Paragraph('<b>Hạn chế:</b>', st_body))
    for l in [
        '• Production lookup vẫn dùng default cho 4/9 landcover + elevation fields (lỗi đọc DEM)',
        '• Khoảng cách train-test AUROC 0,023 cho thấy overfitting (chưa khắc phục hoàn toàn)',
        '• Chỉ dự báo 24h tới, chưa có multi-horizon forecasting (t+3, t+7, t+14)',
        '• Chưa thử domain adaptation (CORAL, TENT, DANN) cho cold-start OOD',
    ]:
        s.append(Paragraph(l, st_body_sm))
    s.append(Spacer(1, 3*mm))

    s.append(section('8. KẾT LUẬN'))
    s.append(Paragraph(
        'Đề tài đạt các mục tiêu: (1) Pipeline dữ liệu huyện-cấp 710 huyện × 6 năm; (2) XGBoost '
        '75 đặc trưng với phương pháp không rò rỉ dữ liệu (eval_set=val, Optuna objective=val '
        'AUROC); (3) <b>AUROC 0,977 và Brier 0,038 trên cold-start OOD test</b> — chứng minh '
        'khả năng tổng quát hóa theo không gian; (4) Deploy web app với JS inference engine + '
        'FIRMS real-time integration tại <b>wildfireforecastvietnam.netlify.app</b>. Học máy có '
        'thể là công cụ hỗ trợ hiệu quả cho dự báo cháy rừng tại Việt Nam.', st_body))
    s.append(Spacer(1, 3*mm))

    s.append(section('9. TÀI LIỆU THAM KHẢO'))
    refs = [
        '[1] Chen, T., Guestrin, C. (2016). XGBoost. KDD 2016, 785-794.',
        '[2] Niculescu-Mizil, A., Caruana, R. (2005). Predicting Good Probabilities. ICML.',
        '[3] Friedman, J. H. (2001). Gradient Boosting Machine. Annals of Statistics, 29(5).',
        '[4] Justice, C. O. et al. (2002). MODIS fire products. RSE, 83(1-2), 244-262.',
        '[5] Schroeder, W. et al. (2014). VIIRS 375m active fire. RSE, 143, 85-96.',
        '[6] ESA WorldCover 2020 v100. (2021). RSE, 269.',
        '[7] NASA POWER Project. power.larc.nasa.gov.',
        '[8] GADM Database v4.1. gadm.org.',
        '[9] LeSage, J., Pace, R. K. (2009). Spatial Econometrics. Chapman & Hall.',
        '[10] Jain, P. et al. (2020). ML applications in wildfire. Env Reviews, 28(4).',
    ]
    for r in refs:
        s.append(Paragraph(r, st_ref))

    doc.build(s)
    print(f'✓ FINAL_POSTER.pdf: {OUT} ({os.path.getsize(OUT)/1024:.1f} KB)')


if __name__ == '__main__':
    print('=== FINAL POSTER (A1 portrait, 2-column, 1-page) ===\n')
    build()
    import subprocess
    result = subprocess.run(['pdfinfo', str(OUT)], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Pages' in line or 'Page size' in line:
            print(f'  {line.strip()}')
