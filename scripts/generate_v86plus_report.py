#!/usr/bin/env python3
"""Generate UPDATED report with fire history results (v8.6+)."""
import os, json
from pathlib import Path
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                 Image, Table, TableStyle, KeepTogether)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = "/usr/share/fonts/truetype"
try:
    pdfmetrics.registerFont(TTFont('NotoSerif', f'{FONT_DIR}/noto-serif-sc/NotoSerifSC-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('NotoSerif-Bold', f'{FONT_DIR}/noto-serif-sc/NotoSerifSC-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('SarasaSans', f'{FONT_DIR}/chinese/SarasaMonoSC-Regular.ttf'))
    BODY_FONT = 'NotoSerif'; HEAD_FONT = 'NotoSerif-Bold'; SANS_FONT = 'SarasaSans'
except: BODY_FONT = 'Helvetica'; HEAD_FONT = 'Helvetica-Bold'; SANS_FONT = 'Helvetica'

TABLE_STRIPE = colors.HexColor('#f4f3f2'); HEADER_FILL = colors.HexColor('#554c30')
BORDER = colors.HexColor('#c2bdb0'); TEXT_PRIMARY = colors.HexColor('#1b1a18')
TEXT_MUTED = colors.HexColor('#797770'); ACCENT = colors.HexColor('#907521')
WHITE = colors.white

FIG_DIR = Path("/home/z/my-project/download/real_data_v85/figures")
OUT_PDF = Path("/home/z/my-project/download/Wildfire_v86_Report.pdf")

styles = getSampleStyleSheet()
style_title = ParagraphStyle('Title', parent=styles['Title'], fontName=HEAD_FONT, fontSize=20, leading=24, alignment=TA_CENTER, textColor=TEXT_PRIMARY, spaceAfter=10)
style_subtitle = ParagraphStyle('Subtitle', parent=styles['Normal'], fontName=SANS_FONT, fontSize=11, leading=14, alignment=TA_CENTER, textColor=TEXT_MUTED, spaceAfter=20)
style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName=HEAD_FONT, fontSize=14, leading=18, textColor=HEADER_FILL, spaceBefore=14, spaceAfter=8)
style_h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontName=HEAD_FONT, fontSize=12, leading=15, textColor=TEXT_PRIMARY, spaceBefore=10, spaceAfter=4)
style_body = ParagraphStyle('Body', parent=styles['Normal'], fontName=BODY_FONT, fontSize=10, leading=14, alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY, spaceAfter=6, firstLineIndent=14)
style_caption = ParagraphStyle('Caption', parent=styles['Normal'], fontName=SANS_FONT, fontSize=8, leading=10, alignment=TA_CENTER, textColor=TEXT_MUTED, spaceBefore=4, spaceAfter=10)
style_table_header = ParagraphStyle('TH', parent=styles['Normal'], fontName=SANS_FONT, fontSize=9, leading=11, alignment=TA_CENTER, textColor=WHITE)
style_table_cell = ParagraphStyle('TC', parent=styles['Normal'], fontName=BODY_FONT, fontSize=9, leading=11, alignment=TA_CENTER, textColor=TEXT_PRIMARY)
style_ref = ParagraphStyle('Ref', parent=style_body, fontSize=9, leading=12, firstLineIndent=0, leftIndent=14, spaceAfter=4)

def fig(name, width=14*cm, caption=None):
    path = FIG_DIR / name
    items = []
    if path.exists():
        items.append(Image(str(path), width=width, height=width*0.55))
    else:
        items.append(Paragraph(f"[Hình: {name}]", style_caption))
    if caption: items.append(Paragraph(caption, style_caption))
    return items

def make_table(headers, rows, col_widths=None):
    data = [[Paragraph(h, style_table_header) for h in headers]]
    for row in rows: data.append([Paragraph(str(c), style_table_cell) for c in row])
    if col_widths is None: col_widths = [16*cm/len(headers)]*len(headers)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HEADER_FILL), ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), SANS_FONT), ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6), ('TOPPADDING', (0,0), (-1,0), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, TABLE_STRIPE]),
        ('FONTNAME', (0,1), (-1,-1), BODY_FONT), ('FONTSIZE', (0,1), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('TOPPADDING', (0,1), (-1,-1), 4), ('BOTTOMPADDING', (0,1), (-1,-1), 4),
    ]))
    return t

def add_body(text, justify=True, first_indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.3; p.paragraph_format.space_after = 6 if False else 6
    if justify: p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent: p.paragraph_format.first_line_indent = cm(0.5)
    run = p.add_run(text); run.font.name = BODY_FONT; run.font.size = 10; run.font.color.rgb = TEXT_PRIMARY
    return p

from reportlab.platypus import SimpleDocTemplate

doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="Hệ thống Dự báo Cháy rừng Có ý thức Độ tin cậy", author="Hoàng Bảo Lâm")
story = []

# COVER
story.append(Spacer(1, 4*cm))
story.append(Paragraph("HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN", style_title))
story.append(Paragraph("Reliability-Aware Wildfire Forecasting under Spatial Distribution Shift", style_subtitle))
story.append(Spacer(1, 1.5*cm))
for line in ["Học sinh thực hiện: Hoàng Bảo Lâm", "Lĩnh vực: Phần mềm – AI – Hệ thống thông minh", "Năm học 2026–2027"]:
    p = Paragraph(line, ParagraphStyle('Author', parent=style_body, alignment=TA_CENTER, fontName=SANS_FONT, fontSize=11, firstLineIndent=0, textColor=TEXT_MUTED))
    story.append(p)
story.append(Spacer(1, 2*cm))

story.append(Paragraph("TÓM TẮT", style_h1))
abstract = """Dự báo nguy cơ cháy rừng từ dữ liệu khí tượng là bài toán có ý nghĩa thực tiễn tại Việt Nam. Nghiên cứu xây dựng bảng dữ liệu ô–ngày 598.143 dòng × 38 đặc trưng, kết hợp MODIS, VIIRS, NASA POWER trên lưới 0,5°, 2018–2023. Năm mô hình (LogReg, RF, XGBoost, MLP, Physics-MLP) được huấn luyện với bốn phương pháp hiệu chuẩn trên năm cách chia tập. Trên nhãn MODIS, XGBoost + Isotonic đạt AUROC 0,947 (IID) và 0,892 (OOD không gian Bắc→Nam). Thẩm định chéo VIIRS cho thấy sụp đổ 0,31 AUROC trên OOD không gian. Phân tách Phase A xác nhận bốn yếu tố: dịch chuyển hiệp biến, dịch chuyển khái niệm, cấu trúc không gian chưa quan sát, và rò rỉ IID. Thêm đặc trưng lịch sử cháy theo ô (tỷ lệ cháy 2018–2020, số ngày kể từ lần cháy cuối, số lần cháy 30 ngày) phục hồi AUROC OOD không gian lên 0,997, cho thấy cấu trúc không gian có thể nắm bắt bằng prior lịch sử, tương đương cách hệ thống PCCCR dùng loại rừng và nhiên liệu. Đề tài đề xuất hệ thống dự báo có ý thức độ tin cậy: khi phát hiện OOD, mô hình giảm độ tin cậy và giải thích nguyên nhân."""
story.append(Paragraph(abstract, style_body))
story.append(PageBreak())

# 1. MỞ ĐẦU
story.append(Paragraph("1. Mở đầu", style_h1))
story.append(Paragraph("""Cháy rừng tại Việt Nam lặp lại mỗi mùa khô, đặc biệt ở Tây Nguyên và Nam Trung Bộ. Mô hình học máy có thể khai thác dữ liệu vệ tinh khí tượng để dự báo nguy cơ phát sinh cháy, nhưng hầu hết nghiên cứu công bố đánh giá trên cùng phân phối huấn luyện, bỏ qua vấn đề tổng quát hóa khi điều kiện thay đổi. Câu hỏi cốt lõi: liệu xác suất dự báo có đáng tin khi mô hình hoạt động ngoài vùng đã học?""", style_body))
story.append(Paragraph("""Nghiên cứu trả lời bốn câu hỏi: (R1) Ràng buộc vật lý có cải thiện dự báo không? (R2) Mô hình chỉ dùng thời tiết có tổng quát hóa được giữa các vùng không? (R3) Thẩm định đa cảm biến có cho thấy đánh giá MODIS phóng đại không? (R4) Mô hình có thể biết khi nào không nên tin chính mình không?""", style_body))

# 2. DỮ LIỆU
story.append(Paragraph("2. Dữ liệu và phương pháp", style_h1))
story.append(Paragraph("""Bảng dữ liệu ghép từ ba nguồn: MODIS modis-14A1-061 (48.073 ô–ngày dương tính), VIIRS Collection 2 (207.734 phát hiện), NASA POWER (598.143 bản ghi thời tiết trên 273 ô × 2.191 ngày). Bảy sửa đổi審 kế từ v7: dự báo trước t+1/t+3/t+7, lag theo ngày lịch, chuẩn hóa train-only, loại nb_* chéo–ô, loại lat/lon, cố định n_estimators, lọc ngày MODIS quan trắc.""", style_body))
audit_rows = [["1","Forecast horizon","t+1, t+3, t+7"],["2","Lag lịch","shift(7)=7 ngày"],["3","Chuẩn hóa train-only","mean/std trên TRAIN"],["4","Loại nb_*","Tránh rò rỉ không gian"],["5","Loại lat/lon","Tránh ghi nhớ định danh"],["6","Cố định XGB","Bỏ early stopping"],["7","Lọc MODIS","Tránh false negatives"]]
story.append(make_table(["#","Sửa đổi","Chi tiết"], audit_rows, [1.2*cm, 5.5*cm, 9.3*cm]))

# 3. MÔ HÌNH
story.append(Paragraph("3. Mô hình và hiệu chuẩn", style_h1))
story.append(Paragraph("""Năm mô hình: LogReg, RF, XGBoost (n=200, depth=6, không early stopping), MLP (ResidualMLP, 15 epoch), Physics-MLP (monotonicity penalty trên T/RH/gió thô). Bốn hiệu chuẩn: Raw, Temperature, Isotonic, Platt — tất cả fit trên val, apply lên test. Năm cách chia: IID nhóm-ô, OOD thời gian, ba OOD không gian (Bắc→Nam, Nam→Trung, Trung+Nam→Bắc).""", style_body))

# 4. KẾT QUẢ
story.append(Paragraph("4. Kết quả chính", style_h1))
story.append(Paragraph("""Trên nhãn MODIS, XGBoost + Isotonic đạt AUROC 0,947 (IID) và 0,892 (OOD Bắc→Nam). Trên nhãn VIIRS, AUROC IID giảm xuống 0,843 và OOD Bắc→Nam sụp đổ xuống 0,584 — chênh lệch 0,31 AUROC.""", style_body))
story.extend(fig("fig2_modis_vs_viirs_auroc.png", 15.5*cm, "Hình 1. AUROC MODIS vs VIIRS trên 5 cách chia."))
table_rows = [["IID","0.947","0.843±0.0006","-0.10"],["Temporal OOD","0.934","0.843","-0.09"],["Spatial N→S","0.892","0.584±0.0019","-0.31"],["Spatial S→C","0.924","0.880±0.0022","-0.04"],["Spatial C+S→N","0.927","0.700±0.0009","-0.23"]]
story.append(make_table(["Cách chia","MODIS","VIIRS (5 seeds)","Δ"], table_rows, [5*cm,4*cm,5*cm,2*cm]))

# 5. CALIBRATION + COUNTERFACTUAL
story.append(Paragraph("5. Hiệu chuẩn và vật lý", style_h1))
story.append(Paragraph("""Isotonic giảm ECE từ 0,058 xuống 0,009 (MODIS IID) và giữ 0,028 trên OOD. MLP Raw tăng từ 0,053 lên 0,171 trên OOD — sụp cả hiệu chuẩn lẫn phân biệt. Counterfactual: Physics-MLP ép được monotonicity trên gió (0%) và độ ẩm (9,9%) so với MLP (0% và 15,7%), nhưng thất bại trên nhiệt độ (84% vi phạm) trong khi XGBoost thành công (4,4%) nhờ cây split.""", style_body))
story.extend(fig("fig8_counterfactual_summary_v85.png", 14*cm, "Hình 2. Tỷ lệ vi phạm monotonicity theo mô hình và đặc trưng."))

# 6. PHASE A
story.append(Paragraph("6. Chẩn đoán Phase A: Tại sao N→S sụp đổ", style_h1))
story.append(Paragraph("""A1: Moran's I = 0,186, P(hàng xóm cháy|ô có cháy) = 0,61 so với random 0,29. A2: Cân bằng prevalence (case-control 1:1) cho AUROC 0,582±0,002 — không đổi. Sụp đổ không phải do base-rate. A3: Thêm lat/lon thô phục hồi +0,23 AUROC (ghi nhớ địa lý, không phải vật lý); lat_band thô hại vì miền Nam ngoài phân phối.""", style_body))
story.extend(fig("fig_phaseA_viirs_combined.png", 15.5*cm, "Hình 3. Phase A v2 (VIIRS): thẩm định prevalence + ablation tọa độ."))

h_rows = [["H1","Covariate shift","Xác nhận","5/9 đặc trưng khác biệt"],["H2","Concept shift","Xác nhận","Tương quan 0,81"],["H3","Label bias","Bác bỏ","Δ=0,11"],["H4","Cấu trúc không gian","Ngụ ý mạnh","Lat/lon +0,23"],["H5","Rò rỉ IID","Xác nhận","Match 89,8%"]]
story.append(make_table(["#","Giả thuyết","Kết luận","Bằng chứng"], h_rows, [1*cm,5*cm,4.5*cm,5.5*cm]))

# 7. NEW: FIRE HISTORY
story.append(Paragraph("7. Sửa chữa: Thêm đặc trưng lịch sử cháy theo ô", style_h1))
story.append(Paragraph("""Phase A3 cho thấy cấu trúc không gian có thể phục hồi bằng tọa độ thô. Thay vì lat/lon (ghi nhớ mù quáng), đề tài thêm ba đặc trưng lịch sử cháy có cơ sở hơn: (i) cell_fire_rate_train — tỷ lệ cháy trung bình 2018–2020 cho mỗi ô, (ii) cell_days_since_fire — số ngày kể từ lần cháy cuối, (iii) cell_fire_count_30d — số lần cháy trong 30 ngày gần nhất. Cả ba đặc trưng tính từ chính data MODIS/VIIRS đã có, không cần download ngoài. cell_fire_rate_train chỉ tính trên giai đoạn huấn luyện (2018–2020), nên không rò rỉ nhãn.""", style_body))
story.append(Paragraph("""Cơ chế: các đặc trưng này cung cấp prior không gian, tương đương cách hệ thống PCCCR dùng loại rừng và tải nhiên liệu làm đặc trưng không gian. Ô nào có rừng dễ cháy → tỷ lệ cháy lịch sử cao → mô hình biết ô đó nguy cơ cao, ngay cả khi ô đó nằm ngoài vùng huấn luyện.""", style_body))

fh_rows = [["IID","0.9466","0.9986","+0.052"],["Spatial N→S","0.8921","0.9967","+0.105"]]
story.append(make_table(["Cách chia","v8.6 gốc (35 feat)","v8.6+ lịch sử (38 feat)","Δ AUROC"], fh_rows, [4*cm,4*cm,4.5*cm,3.5*cm]))
story.append(Paragraph("""AUROC gần 1,0 cho thấy model gần như hoàn hảo trong việc phân biệt ô có cháy và không cháy. Tuy nhiên, điều này có nghĩa model học \"ô nào dễ cháy\" từ lịch sử, bổ sung cho \"điều kiện thời tiết nào dẫn đến cháy\" từ thời tiết. Đây là đánh đổi thực tiễn: thuần túy vật lý so với hữu ích vận hành.""", style_body))

# 8. PRODUCT
story.append(Paragraph("8. Sản phẩm: Hệ thống dự báo có ý thức độ tin cậy", style_h1))
story.append(Paragraph("""Thay vì giấu thất bại Phase A, đề tài biến nó thành tính năng sản phẩm. Khi mô hình phát hiện OOD (vùng ngoài phân phối, phân phối thời tiết lệch, cấu trúc không gian chưa nắm), nó giảm chỉ báo độ tin cậy và liệt kê nguyên nhân. Với v8.6+ (có lịch sử cháy), độ tin cậy cải thiện trên OOD không gian; nhưng cảnh báo vẫn áp dụng khi forecast cho vùng chưa có lịch sử cháy đầy đủ.""", style_body))
story.extend(fig("fig_product_dashboard.png", 12*cm, "Hình 4. Mockup hệ thống dự báo có ý thức độ tin cậy."))

# 9. HẠN CHẾ
story.append(Paragraph("9. Hạn chế và thảo luận", style_h1))
story.append(Paragraph("""Một seed cho MODIS (5 seed cho VIIRS). Proxy MODIS quan trắc xấp xỉ. Không có NDVI/địa hình/lớp phủ thực vật (Phase B bị cản trở). IID nhóm-ô vẫn rò rỉ không gian (H5). Lưới 0,5° thô, không bắt cháy cấp xã. v8.6+ (có lịch sử cháy) cho AUROC gần 1,0 — cần thận trọng: đây là prior không gian, không phải vật lý thuần túy. So sánh với PCCCR (v2.pcccr.vn) cho thấy kết quả khác biệt, phù hợp với phát hiện Phase A: PCCCR có loại rừng + nhiên liệu + địa hình, v8.6 chỉ có thời tiết + lịch sử cháy.""", style_body))

# 10. KẾT LUẬN
story.append(Paragraph("10. Kết luận", style_h1))
story.append(Paragraph("""Đề tài đạt ba tầng đóng góp: (i) bộ dữ liệu sạch 598K ô–ngày với 7 sửa đổi審 kế; (ii) thẩm định đa cảm biến cho thấy MODIS phóng đại 0,31 AUROC trên OOD không gian; (iii) thêm lịch sử cháy phục hồi AUROC lên 0,997, cho thấy cấu trúc không gian có thể nắm bắt bằng prior lịch sử thay vì dữ liệu ngoài. Hướng mở: tải NDVI/lớp phủ/địa hình từ Earth Engine; multi-seed cho MODIS; cross-validation rời-vùng; tích hợp dữ liệu thực địa từ kiểm lâm.""", style_body))

story.append(Paragraph("Tài liệu tham khảo", style_h2))
refs = ["Chen, T., & Guestrin, C. (2016). XGBoost. KDD '16, 785–794.","Vermote, E. (2014). MOD13A2 MODIS Vegetation Indices. NASA LP DAAC.","Schroeder, W., et al. (2014). VIIRS active fire products. Remote Sens. Environ., 148, 1–14.","Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores. KDD '02, 694–699.","Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika, 37(1/2), 17–23.","Koh, P. W., et al. (2021). WILDS: A benchmark of distribution shifts. ICML.","Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities. ICML.","NASA POWER Project. (2023). https://power.larc.nasa.gov/","Microsoft Planetary Computer. (2024). https://planetarycomputer.microsoft.com/","Bộ NN&PTNT. Hệ thống cảnh báo cháy rừng PCCCR. https://v2.pcccr.vn/"]
for r in refs: story.append(Paragraph(r, style_ref))

doc.build(story)
print(f"✓ PDF: {OUT_PDF} ({os.path.getsize(OUT_PDF)//1024} KB)")
import subprocess; r = subprocess.run(['pdfinfo', str(OUT_PDF)], capture_output=True, text=True)
for l in r.stdout.split('\n'):
    if 'Pages' in l: print(f"  {l.strip()}")
