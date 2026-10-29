#!/usr/bin/env python3
"""FINAL ViSEF/ISEF-style poster for KHKT Hanoi 2026-2027.

Format: A1 portrait (594 × 841 mm) — matches Vietnamese KHKT/ViSEF official requirements.
Layout: 3-column scientific poster inspired by ISEF.

All metrics from Bao_cao_KHKT_final.pdf (leak-free retrain, official report):
  AUROC OOD = 0.977  (cold-start spatial: train Bắc+Trung, test Nam)
  Brier calibrated = 0.038
  AUPR = 0.871
  964.040 huyện-ngày observations
  710 districts × 6 years (2018-2023)
  75 features
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
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, FrameBreak,
)
from reportlab.lib.styles import ParagraphStyle

# Fonts (Vietnamese-supporting)
pdfmetrics.registerFont(TTFont('Lib', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Bold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Italic', '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf'))
pdfmetrics.registerFont(TTFont('Lib-BoldItalic', '/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf'))
registerFontFamily('Lib', normal='Lib', bold='Lib-Bold', italic='Lib-Italic', boldItalic='Lib-BoldItalic')

pdfmetrics.registerFont(TTFont('LibS', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('LibS-Bold', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('LibS-Italic', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'))
registerFontFamily('LibS', normal='LibS', bold='LibS-Bold', italic='LibS-Italic')

OUT = Path('/home/z/my-project/KHKT_HoSo/FINAL_POSTER.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)

# A1 portrait: 594 × 841 mm = 1683.78 × 2383.94 pts
PAGE_W, PAGE_H = A1

# Design palette
COL_NAVY = colors.HexColor('#0f172a')  # dark navy for headings
COL_BLUE = colors.HexColor('#1e40af')  # primary blue accent
COL_CYAN = colors.HexColor('#0891b2')  # cyan secondary accent
COL_GREY = colors.HexColor('#475569')  # text grey
COL_LIGHT = colors.HexColor('#f8fafc')  # very light bg
COL_BORDER = colors.HexColor('#cbd5e1')  # border grey

# Margin
MARGIN = 18*mm

# === STYLES ===
st_title = ParagraphStyle('PTitle', fontName='Lib-Bold', fontSize=22, leading=26,
                           alignment=TA_CENTER, textColor=COL_NAVY, spaceAfter=3)
st_subtitle = ParagraphStyle('PSub', fontName='LibS', fontSize=11, leading=13,
                              alignment=TA_CENTER, textColor=COL_GREY, spaceAfter=2)
st_author = ParagraphStyle('PAuthor', fontName='Lib-Bold', fontSize=12, leading=15,
                            alignment=TA_CENTER, textColor=COL_NAVY)
st_meta = ParagraphStyle('PMeta', fontName='LibS', fontSize=9, leading=12,
                          alignment=TA_CENTER, textColor=COL_GREY)
st_section = ParagraphStyle('PSec', fontName='LibS-Bold', fontSize=12, leading=15,
                             textColor=colors.white, spaceBefore=3, spaceAfter=3,
                             leftIndent=4, alignment=TA_LEFT)
st_h3 = ParagraphStyle('PH3', fontName='LibS-Bold', fontSize=10, leading=13,
                        textColor=COL_NAVY, spaceBefore=3, spaceAfter=2)
st_body = ParagraphStyle('PBody', fontName='Lib', fontSize=9, leading=11.5,
                          alignment=TA_JUSTIFY, textColor=colors.HexColor('#1e293b'), spaceAfter=2)
st_body_sm = ParagraphStyle('PBodySm', fontName='Lib', fontSize=8, leading=10.5,
                             alignment=TA_JUSTIFY, textColor=colors.HexColor('#1e293b'), spaceAfter=1)
st_bullet = ParagraphStyle('PBullet', parent=st_body, leftIndent=8, bulletIndent=0, spaceAfter=1)
st_caption = ParagraphStyle('PCap', fontName='Lib-Italic', fontSize=7, leading=9,
                            alignment=TA_CENTER, textColor=COL_GREY, spaceBefore=1, spaceAfter=2)
st_metric_big = ParagraphStyle('PMetric', fontName='LibS-Bold', fontSize=20, leading=22,
                                alignment=TA_CENTER, textColor=COL_BLUE)
st_metric_lbl = ParagraphStyle('PMetricL', fontName='LibS', fontSize=7, leading=9,
                                alignment=TA_CENTER, textColor=COL_GREY)
st_ref = ParagraphStyle('PRef', fontName='Lib', fontSize=7, leading=9,
                        alignment=TA_LEFT, textColor=COL_GREY, spaceAfter=1, leftIndent=10, firstLineIndent=-10)


def section_header(title):
    """Create a section header as a Table with navy background."""
    t = Table([[Paragraph(title, st_section)]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COL_NAVY),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return t


def build_poster():
    """Build the A1 portrait 3-column poster."""
    # Setup doc with single page (no auto-pagination — we control frame placement)
    doc = BaseDocTemplate(
        str(OUT), pagesize=A1,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='FINAL POSTER - Wildfire Forecasting System - KHKT Hanoi 2026-2027',
        author='Hoàng Bảo Lâm',
        subject='ViSEF/ISEF-style science fair poster',
        creator='ReportLab',
    )

    # === LAYOUT ===
    # Header: full width, top
    # Body: 3 columns below header
    header_h = 60*mm  # header height
    body_h = PAGE_H - 2*MARGIN - header_h - 5*mm  # 5mm gap

    avail_w = PAGE_W - 2*MARGIN
    col_gap = 8*mm
    col_w = (avail_w - 2*col_gap) / 3

    # Frames
    header_frame = Frame(MARGIN, PAGE_H - MARGIN - header_h,
                        avail_w, header_h, id='header',
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                        showBoundary=0)
    col1 = Frame(MARGIN, MARGIN, col_w, body_h, id='col1',
                 leftPadding=4, rightPadding=4, topPadding=4, bottomPadding=4)
    col2 = Frame(MARGIN + col_w + col_gap, MARGIN, col_w, body_h, id='col2',
                 leftPadding=4, rightPadding=4, topPadding=4, bottomPadding=4)
    col3 = Frame(MARGIN + 2*(col_w + col_gap), MARGIN, col_w, body_h, id='col3',
                 leftPadding=4, rightPadding=4, topPadding=4, bottomPadding=4)

    template = PageTemplate(id='poster', frames=[header_frame, col1, col2, col3])
    doc.addPageTemplates([template])

    story = []

    # === HEADER ===
    story.append(Paragraph('HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY', st_title))
    story.append(Paragraph('DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN BẰNG MÔ HÌNH HỌC MÁY', st_title))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph('Hoàng Bảo Lâm', st_author))
    story.append(Paragraph('Lớp 9E4 · Trường THCS Lê Quý Đôn · Hà Nội', st_meta))
    story.append(Paragraph('GVHD: Lê Xuân Tân (Giáo viên Tin học) · HT: Nguyễn Anh Tuấn', st_meta))
    story.append(Paragraph('Cuộc thi KHKT Hà Nội năm học 2026 – 2027 · Mã dự án: HN26-129F5F83', st_meta))
    story.append(Spacer(1, 2*mm))
    # Key metrics row
    metrics = Table([[
        Paragraph('<b>0,977</b>', st_metric_big),
        Paragraph('<b>0,038</b>', st_metric_big),
        Paragraph('<b>0,871</b>', st_metric_big),
        Paragraph('<b>964K</b>', st_metric_big),
        Paragraph('<b>710</b>', st_metric_big),
        Paragraph('<b>75</b>', st_metric_big),
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
        ('BOX', (0,0), (-1,-1), 1, COL_BLUE),
        ('LINEAFTER', (0,0), (-2,-1), 0.5, COL_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(metrics)

    # === COLUMN 1 ===
    story.append(FrameBreak())  # move to col1

    story.append(section_header('1. VẤN ĐỀ NGHIÊN CỨU'))
    story.append(Paragraph(
        'Cháy rừng là thảm hoạ thường xuyên tại Việt Nam, đặc biệt Tây Nguyên, Nam Trung Bộ và '
        'ĐBSCL. Mỗi năm ghi nhận hàng nghìn vụ, thiêu rụi hàng chục nghìn ha rừng. Các hệ thống '
        'dự báo hiện có dựa trên chỉ số FWI của Canada (phát triển cho rừng lá kim ôn đới) — '
        'không phù hợp khí hậu nhiệt đới gió mùa VN. Chưa có hệ thống dự báo cấp huyện (nơi có '
        'thẩm quyền cảnh báo theo Luật Lâm nghiệp 2017).', st_body))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('2. CÂU HỎI NGHIÊN CỨU'))
    story.append(Paragraph(
        '<b>Có thể xây dựng mô hình học máy dự báo xác suất cháy rừng cấp huyện với độ tin cậy '
        '(calibration) chấp nhận được khi áp dụng vào vùng chưa từng thấy trong tập huấn luyện '
        'không?</b>', st_body))
    story.append(Paragraph(
        '<b>Giả thuyết:</b> XGBoost + isotonic calibration fit trên val sẽ đạt AUROC ≥ 0,95 trên '
        'cold-start spatial OOD (train Bắc+Trung, test Nam) và Brier ≤ 0,05.', st_body))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('3. BỐI CẢNH & ĐỘNG LỰC'))
    story.append(Paragraph(
        'Học máy (XGBoost, LightGBM) + dữ liệu vệ tinh công khai (NASA POWER, MODIS, VIIRS, '
        'MOD13A1, ESA WorldCover, GADM ADM2) mở ra cơ hội xây dựng hệ thống chuyên biệt cho VN. '
        'Thách thức cốt lõi: <b>cold-start spatial OOD</b> — mô hình áp dụng vào vùng chưa từng '
        'thấy trong huấn luyện có thể suy giảm độ tin cậy.', st_body))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('4. NGUỒN DỮ LIỆU'))
    data_tbl = [
        ['<b>Nguồn</b>', '<b>Loại</b>', '<b>Độ phân giải</b>'],
        ['NASA POWER', 'Thời tiết (T2M, RH, WS, precip)', '0,5° daily'],
        ['MODIS Active Fire', 'Điểm cháy (nhãn chính)', '1 km × 2/ngày'],
        ['VIIRS 375m', 'Điểm cháy (kiểm chứng)', '375 m × 2/ngày'],
        ['GADM ADM2 v4.1', 'Ranh giới 710 huyện', 'Vector polygon'],
        ['MODIS MOD13A1', 'NDVI thực vật', '500 m × 23/năm'],
        ['ESA WorldCover 10m', 'Lớp phủ đất thật', '10 m (2020)'],
        ['Copernicus DEM', 'Địa hình (độ cao)', '30 m'],
        ['FIRMS API', 'Điểm cháy real-time', '1 km × 24h'],
    ]
    rows = [[Paragraph(c, st_body_sm) for c in r] for r in data_tbl]
    t = Table(rows, colWidths=[3.2*cm, 5.5*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COL_NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'LibS-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.3, COL_BORDER),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t)
    story.append(Paragraph('<i>Bảng 1: 9 nguồn dữ liệu công khai đã sử dụng.</i>', st_caption))

    # === COLUMN 2 ===
    story.append(FrameBreak())  # move to col2

    story.append(section_header('5. PHƯƠNG PHÁP'))
    story.append(Paragraph('<b>5.1. Pipeline 6 bước</b>', st_h3))
    for step in [
        '<b>1.</b> Thu thập dữ liệu vệ tinh (NASA, ESA, Microsoft Planetary Computer)',
        '<b>2.</b> Tổng hợp về cấp huyện-ngày (710 × 2191 = 1,55M rows)',
        '<b>3.</b> Feature engineering: 75 đặc trưng',
        '<b>4.</b> Chia cold-start spatial OOD (Bắc+Trung → Nam)',
        '<b>5.</b> Optuna 30 trials + XGBoost + isotonic calibration',
        '<b>6.</b> Deploy web app (Netlify + JS inference + FIRMS)',
    ]:
        story.append(Paragraph(step, st_bullet))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('6. THIẾT KẾ THỰC NGHIỆM'))
    story.append(Paragraph('<b>Chia dữ liệu (cold-start spatial OOD):</b>', st_h3))
    story.append(Paragraph(
        '• <b>Train (Bắc+Trung):</b> 490.560 rows, 224 huyện (lat ≥ 14°)\n'
        '• <b>Validation:</b> 120.450 rows, 55 huyện (held-out từ train)\n'
        '• <b>Test (Nam):</b> 352.590 rows, 161 huyện (lat < 14°) — chưa từng thấy trong train',
        st_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph('<b>75 đặc trưng:</b>', st_h3))
    feats = [
        '<b>Thời tiết (35):</b> t2m, rh, wind, precip + lag 1/3/7/14 + rolling 7/14/30',
        '<b>FWI (6):</b> fwi, ffmc, dmc, dc, isi, bui',
        '<b>Lớp phủ đất (7):</b> thật từ ESA WorldCover (forest, agri, urban, water, bare)',
        '<b>Địa lý (9):</b> elevation, slope, koppen, latitude_band, coastal_distance',
        '<b>NDVI (2):</b> thật từ MODIS MOD13A1',
        '<b>Cell fire rate (4):</b> climatology theo (huyện, doy)',
        '<b>Tương tác (5):</b> wind_temp, dry_wind, heat_drought, v.v.',
        '<b>Spatial lag (1):</b> neighbor_fire (queen contiguity, 1934 edges)',
        '<b>Centroid (3):</b> centroid_lat, centroid_lon, doy',
    ]
    for f in feats:
        story.append(Paragraph(f, st_body_sm))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('7. HUẤN LUYỆN MÔ HÌNH'))
    story.append(Paragraph(
        '<b>XGBoost + Optuna 30 trials</b> (TPE, seed=42). Search: n_est [100,400], max_depth [3,8].',
        st_body))
    story.append(Paragraph(
        '<b>LEAK-FREE:</b> Objective = val AUROC, eval_set=[(X_val, y_val)]. Không dùng test.',
        st_body))
    story.append(Paragraph(
        '<b>Calibration:</b> IsotonicRegression fit trên (p_val, y_val).', st_body))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('8. ĐÁNH GIÁ'))
    story.append(Paragraph(
        '<b>Metrics:</b> AUROC, Brier, AUPR, ECE. <b>Production:</b> JS inference 297 dòng TS '
        '(Netlify, không cần Python). FIRMS API real-time → neighbor_fire mỗi 10 phút qua '
        'point-in-polygon + queen contiguity.', st_body))

    # === COLUMN 3 ===
    story.append(FrameBreak())  # move to col3

    story.append(section_header('9. KẾT QUẢ'))
    # Results table
    res_tbl = [
        ['<b>Metric</b>', '<b>Train</b>', '<b>Val</b>', '<b>Test (OOD)</b>'],
        ['AUROC', '1,0000', '0,9980', '<b>0,977</b>'],
        ['AUPR', '—', '—', '0,871'],
        ['Brier (raw)', '—', '—', '0,039'],
        ['Brier (cal)', '—', '—', '<b>0,038</b>'],
    ]
    rows = [[Paragraph(c, st_body_sm) for c in r] for r in res_tbl]
    t = Table(rows, colWidths=[2.5*cm, 2*cm, 2*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COL_NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'LibS-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.3, COL_BORDER),
        ('BACKGROUND', (3,1), (3,-1), colors.HexColor('#eff6ff')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Paragraph('<i>Bảng 2: Kết quả trên cold-start OOD (train Bắc+Trung, test Nam).</i>', st_caption))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('10. ĐẶC TRƯNG QUAN TRỌNG'))
    imp = [
        ['#', 'Feature', 'Gain', 'Ý nghĩa'],
        ['1', 'is_dry', '3587', 'Cờ mùa khô'],
        ['2', 'neighbor_fire', '1154', 'Tỷ lệ láng giềng có cháy'],
        ['3', 'cell_fire_count_30d', '663', 'Số điểm cháy 30 ngày'],
        ['4', 'doy', '73', 'Ngày trong năm'],
        ['5', 'koppen_Cwa', '62', 'Khí hậu Köppen'],
    ]
    rows = [[Paragraph(c, st_body_sm) for c in r] for r in imp]
    t = Table(rows, colWidths=[0.7*cm, 4*cm, 1.5*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COL_BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'LibS-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.3, COL_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t)
    story.append(Paragraph('<i>Bảng 3: Top 5 feature importance by gain.</i>', st_caption))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('11. THẢO LUẬN & HẠN CHẾ'))
    story.append(Paragraph(
        '<b>Thảo luận:</b> AUROC 0,977 trên cold-start OOD (chưa từng thấy dữ liệu Nam) cho thấy '
        'mô hình tổng quát hóa tốt. Top đặc trưng is_dry + neighbor_fire có ý nghĩa vật lý rõ '
        'ràng. Hồi quy đẳng điều fit trên val (không phải test) đảm bảo Brier không rò rỉ dữ liệu.',
        st_body))
    story.append(Paragraph('<b>Hạn chế:</b>', st_h3))
    for l in [
        '• Production lookup vẫn default cho 4/9 landcover + elevation fields',
        '• Khoảng cách train-test AUROC 0,023 (overfitting)',
        '• Chỉ dự báo 24h, chưa có multi-horizon (t+3, t+7, t+14)',
        '• Chưa thử domain adaptation (CORAL, TENT) cho cold-start OOD',
    ]:
        story.append(Paragraph(l, st_body_sm))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('12. KẾT LUẬN'))
    story.append(Paragraph(
        'Đề tài đạt các mục tiêu: (1) Pipeline dữ liệu huyện-cấp 710 huyện × 6 năm; (2) XGBoost '
        '75 đặc trưng với phương pháp không rò rỉ dữ liệu (eval_set=val, Optuna objective=val '
        'AUROC); (3) <b>AUROC 0,977 và Brier 0,038 trên cold-start OOD test</b> — chứng minh '
        'khả năng tổng quát hóa theo không gian; (4) Deploy web app với JS inference engine + '
        'FIRMS real-time integration tại <b>wildfireforecastvietnam.netlify.app</b>. Học máy có '
        'thể là công cụ hỗ trợ hiệu quả cho dự báo cháy rừng tại Việt Nam.', st_body))
    story.append(Spacer(1, 2*mm))

    story.append(section_header('13. TÀI LIỆU THAM KHẢO'))
    refs = [
        '[1] Chen, T., Guestrin, C. (2016). XGBoost. KDD 2016.',
        '[2] Niculescu-Mizil, A., Caruana, R. (2005). Predicting Good Probabilities. ICML.',
        '[3] Friedman, J. H. (2001). Gradient Boosting Machine. Annals of Statistics.',
        '[4] Justice, C. O. et al. (2002). MODIS fire products. RSE 83(1-2).',
        '[5] Schroeder, W. et al. (2014). VIIRS 375m active fire. RSE 143.',
        '[6] ESA WorldCover 2020 v100. (2021). RSE 269.',
        '[7] NASA POWER Project. power.larc.nasa.gov.',
        '[8] GADM Database v4.1. gadm.org.',
        '[9] LeSage, J., Pace, R. K. (2009). Spatial Econometrics. Chapman & Hall.',
        '[10] Jain, P. et al. (2020). ML applications in wildfire. Env Reviews 28(4).',
    ]
    for r in refs:
        story.append(Paragraph(r, st_ref))

    # Build
    doc.build(story)
    print(f'✓ Poster: {OUT} ({os.path.getsize(OUT)/1024:.1f} KB)')


if __name__ == '__main__':
    print('=== FINAL ViSEF/ISEF-style poster (A1 portrait, 3-column) ===\n')
    build_poster()
    # Verify
    import subprocess
    result = subprocess.run(['pdfinfo', str(OUT)], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Pages' in line or 'Page size' in line or 'Title' in line:
            print(f'  {line.strip()}')
