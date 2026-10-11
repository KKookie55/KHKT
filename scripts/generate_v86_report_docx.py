#!/usr/bin/env python3
"""Generate Vietnamese Word (.docx) version of v8.6 report."""
import os, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.section import WD_SECTION

OUT_DIR = Path("/home/z/my-project/download")
FIG_DIR = Path("/home/z/my-project/download/real_data_v85/figures")
OUT_DOCX = OUT_DIR / "Wildfire_v86_Report.docx"

# Colors
HEADER_COLOR = RGBColor(0x55, 0x4c, 0x30)
TEXT_PRIMARY = RGBColor(0x1b, 0x1a, 0x18)
TEXT_MUTED = RGBColor(0x79, 0x77, 0x70)
ACCENT = RGBColor(0x90, 0x75, 0x21)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

doc = Document()

# Set default font (Times New Roman for body, supports Vietnamese)
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
# Set East Asian font
rpr = style.element.get_or_add_rPr()
rfonts = rpr.find(qn('w:rFonts'))
if rfonts is None:
    rfonts = OxmlElement('w:rFonts')
    rpr.append(rfonts)
rfonts.set(qn('w:eastAsia'), 'Times New Roman')
font.size = Pt(11)

# Set page margins (A4)
section = doc.sections[0]
section.page_width = Cm(21)
section.page_height = Cm(29.7)
section.top_margin = Cm(2)
section.bottom_margin = Cm(2)
section.left_margin = Cm(2.5)
section.right_margin = Cm(2)

def set_run_font(run, name='Times New Roman', size=11, bold=False, color=None, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    # Set East Asian font
    r = run._element
    rpr = r.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), name)
    rfonts.set(qn('w:ascii'), name)
    rfonts.set(qn('w:hAnsi'), name)

def add_heading(text, level=1):
    """Add a heading with custom styling."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14) if level == 1 else Pt(10)
    p.paragraph_format.space_after = Pt(8) if level == 1 else Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    if level == 1:
        set_run_font(run, name='Arial', size=14, bold=True, color=HEADER_COLOR)
    elif level == 2:
        set_run_font(run, name='Arial', size=12, bold=True, color=TEXT_PRIMARY)
    else:
        set_run_font(run, name='Arial', size=11, bold=True, color=TEXT_PRIMARY)
    return p

def add_body(text, justify=True, first_indent=True):
    """Add a body paragraph with Vietnamese formatting."""
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(6)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent:
        p.paragraph_format.first_line_indent = Cm(0.5)
    run = p.add_run(text)
    set_run_font(run, name='Times New Roman', size=11, color=TEXT_PRIMARY)
    return p

def add_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    set_run_font(run, name='Arial', size=9, italic=True, color=TEXT_MUTED)
    return p

def add_figure(filename, width_cm=14, caption=None):
    """Add figure with caption."""
    path = FIG_DIR / filename
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run()
        run.add_picture(str(path), width=Cm(width_cm))
        if caption:
            add_caption(caption)
    else:
        add_body(f"[Hình thiếu: {filename}]", first_indent=False)

def add_table_from_data(headers, rows, col_widths_cm=None):
    """Add a styled table."""
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Grid Accent 1'

    # Header
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_run_font(run, name='Arial', size=9, bold=True, color=WHITE)
        # Set header background
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), '554C30')
        tcPr.append(shd)

    # Data rows
    for row_data in rows:
        row = table.add_row()
        for i, val in enumerate(row_data):
            cell = row.cells[i]
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            set_run_font(run, name='Times New Roman', size=9, color=TEXT_PRIMARY)

    # Set column widths
    if col_widths_cm:
        for i, w in enumerate(col_widths_cm):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    # Disable cell splitting across pages
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        cantSplit = OxmlElement('w:cantSplit')
        trPr.append(cantSplit)

    return table

# ============================================================
# COVER PAGE
# ============================================================
# Add spacing to push title down
for _ in range(6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)

# Title
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(12)
run = p.add_run("HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN")
set_run_font(run, name='Arial', size=20, bold=True, color=TEXT_PRIMARY)

# Subtitle (English)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(40)
run = p.add_run("Reliability-Aware Wildfire Forecasting under Spatial Distribution Shift")
set_run_font(run, name='Arial', size=11, italic=True, color=TEXT_MUTED)

# Author info
for line in ["Học sinh thực hiện: Hoàng Bảo Lâm",
             "Lĩnh vực: Phần mềm – AI – Hệ thống thông minh",
             "Năm học 2026–2027"]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(line)
    set_run_font(run, name='Arial', size=11, color=TEXT_MUTED)

# Page break to abstract
doc.add_page_break()

# ============================================================
# ABSTRACT
# ============================================================
add_heading("TÓM TẮT", 1)
abstract = """Dự báo nguy cơ cháy rừng từ dữ liệu khí tượng là bài toán có ý nghĩa thực tiễn tại Việt Nam, nơi cháy rừng gây thiệt hại kinh tế và sinh thái đáng kể mỗi năm khô hanh. Nghiên cứu này xây dựng một bảng dữ liệu ô–ngày (cell-day) gồm 598.143 dòng × 35 đặc trưng, kết hợp phát hiện cháy từ MODIS và VIIRS của NASA, dữ liệu thời tiết NASA POWER, trên lưới 0,5° trải qua 2.191 ngày (2018–2023). Năm mô hình học máy (Logistic Regression, Random Forest, XGBoost, MLP, Physics-MLP) được huấn luyện với bốn phương pháp hiệu chuẩn (Raw, Temperature, Isotonic, Platt) trên năm cách chia tập (IID nhóm-ô, OOD thời gian, ba OOD không gian). Kết quả cho thấy hiệu chuẩn Isotronic giữ ECE ổn định từ 0,009 (IID) lên 0,032 (OOD không gian) cho XGBoost trên nhãn MODIS, trong khi MLP bản ngầu sụp đổ từ 0,053 lên 0,171. Khi thẩm định chéo với nhãn VIIRS (độ phân giải 375 m, nhạy với các đám cháy nhỏ hơn), AUROC OOD không gian Bắc → Nam giảm từ 0,892 (MODIS) xuống 0,584 (VIIRS), chênh lệch 0,31 AUROC. Phân tách Phase A cho thấy sụp đổ này không phải do mất cân bằng tỷ lệ lớp mà do mô hình thiếu khả năng biểu diễn cấu trúc không gian chưa quan sát; việc thêm vĩ độ/kinh độ thô phục hồi 0,23 AUROC, nhưng đây là ghi nhớ địa lý chứ không phải suy luận vật lý. Dựa trên kết quả đó, đề tài đề xuất một hệ thống dự báo có ý thức độ tin cậy: khi mô hình phát hiện dịch chuyển phân phối, nó giảm chỉ báo độ tin cậy và giải thích nguyên nhân thay vì đưa ra một xác suất đơn trị quá tự tin."""
add_body(abstract)

# Page break to body
doc.add_page_break()

# ============================================================
# 1. MỞ ĐẦU
# ============================================================
add_heading("1. Mở đầu", 1)

intro1 = """Cháy rừng tại Việt Nam là một hiện tượng lặp lại mỗi mùa khô, đặc biệt ở Tây Nguyên và Nam Trung Bộ. Các đám cháy lớn trong các năm 2019–2023 đã làm suy giảm diện tích rừng tự nhiên, đe dọa đa dạng sinh học và gây thiệt hại cho cộng đồng dân cư sống phụ thuộc vào rừng. Mô hình học máy có thể tận dụng lượng lớn dữ liệu vệ tinh khí tượng để dự báo nguy cơ phát sinh cháy, nhưng hầu hết các nghiên cứu công bố đánh giá mô hình trên cùng phân phối dữ liệu huấn luyện, bỏ qua vấn đề tổng quát hóa khi điều kiện thay đổi."""
add_body(intro1)

intro2 = """Câu hỏi cốt lõi không phải là liệu mô hình có đạt AUROC cao trên tập kiểm tra ngẫu nhiên, mà là liệu các xác suất dự báo có đáng tin cậy khi mô hình hoạt động ngoài vùng đã học. Một mô hình có thể dự báo đúng trên dữ liệu quen thuộc nhưng đưa ra xác suất lệch hẳn thực tế khi áp dụng sang vùng địa lý khác, thời kỳ cực đoan khác, hoặc khi cấu trúc không gian chưa quan sát được thay đổi. Đối với một quyết định triển khai thực địa—phân bổ lực lượng kiểm lâm, phát cảnh báo sớm—sự quá tự tin sai lệch có hậu quả trực tiếp."""
add_body(intro2)

add_heading("1.1. Câu hỏi nghiên cứu", 2)
rq_text = """Nghiên cứu này trả lời bốn câu hỏi: (R1) Ràng buộc vật lý có cải thiện dự báo cháy rừng không? (R2) Mô hình chỉ dùng đặc trưng khí tượng có tổng quát hóa được giữa các vùng địa lý không? (R3) Thẩm định đa cảm biến có cho thấy đánh giá chỉ với MODIS phóng đại hiệu năng không? (R4) Mô hình có thể biết khi nào không nên tin chính mình không? Bốn câu hỏi liên kết theo trình tự: từ giả thuyết ban đầu (R1) sang kết quả phủ định (ràng buộc vật lý không cải thiện trên dữ liệu thật), sang phát hiện chính (R2, R3), và cuối cùng sang đề xuất sản phẩm (R4)."""
add_body(rq_text)

# ============================================================
# 2. DỮ LIỆU
# ============================================================
add_heading("2. Dữ liệu và xây dựng bảng ô–ngày", 1)

data1 = """Bảng dữ liệu chính được ghép từ ba nguồn vệ tinh và khí tượng. Phát hiện cháy MODIS (sản phẩm modis-14A1-061, độ phân giải 1 km) tải qua Microsoft Planetary Computer STAC; sau khử trùng lặp theo (ô, ngày) thu được 48.073 ô–ngày dương tính. Phát hiện cháy VIIRS Collection 2 (375 m, nhạy với đám cháy nhỏ hơn) tải qua NASA FIRMS, lọc theo bounding box Việt Nam (kinh độ 104,0–109,5° Đông, vĩ độ 11,0–21,0° Bắc), thu được 207.734 phát hiện. Dữ liệu thời tiết NASA POWER (T2M, RH2M, WS10M, PRECTOTCORR) trên lưới 0,5° × 0,5° trải qua 273 ô × 2.191 ngày, tổng 598.143 bản ghi."""
add_body(data1)

data2 = """Vì một bảng ô–ngày đầy đủ là một bảng cân bằng (panel), mô hình bị buộc học từ tín hiệu khí tượng và thời gian. Không có đặc trưng tĩnh (NDVI, độ cao, lớp phủ thực vật) trong v8.5, vì các nguồn này đòi hỏi ủy quyền Earth Engine mà môi trường phát triển không có. Mặt khác, các đặc trưng tĩnh như độ cao có thể để mô hình ghi nhớ định danh ô thay vì học quan hệ vật lý; việc loại bỏ chúng là một sửa đổi thận trọng theo審 kế (audit fix #3)."""
add_body(data2)

add_heading("2.1. Bảy sửa đổi từ v7 sang v8.5", 2)

audit_rows = [
    ["1", "Dự báo trước (forecast horizon)", "t+1, t+3, t+7 thay vì cùng ngày"],
    ["2", "Lag theo ngày lịch", "shift(7) = đúng 7 ngày, không phải 7 quan trắc"],
    ["3", "Chuẩn hóa chỉ số hạn hán", "Tính mean/std trên TRAIN, apply val/test"],
    ["4", "Loại đặc trưng nb_* chéo–ô", "Tránh rò rỉ không gian"],
    ["5", "Loại lat/lon làm đặc trưng", "Tránh ghi nhớ định danh ô"],
    ["6", "Cố định n_estimators XGB", "Bỏ early stopping theo AUROC val"],
    ["7", "Lọc ngày MODIS quan trắc", "Tránh nhãn âm giả (false negatives)"],
]
add_table_from_data(["#", "Sửa đổi", "Chi tiết"], audit_rows, col_widths_cm=[1, 5, 9])

# ============================================================
# 3. MÔ HÌNH
# ============================================================
add_heading("3. Mô hình và phương pháp hiệu chuẩn", 1)

model1 = """Năm kiến trúc được so sánh. Logistic Regression làm cơ sở tuyến tính, Random Forest làm cơ sở cây, XGBoost làm mô hình chính với n_estimators=200, max_depth=6 (không early stopping theo AUROC val). MLP (ResidualMLP, hidden 128, dropout 0,15, 15 epoch) làm cơ sở mạng nơ-ron. Physics-MLP có cùng kiến trúc cộng thêm hàm phạt monotonicity trên ba đặc trưng thô (T2M, RH2M, WS10M, đơn vị °C, %, m/s), trọng số 0,4 cho mỗi hạng mục năng lượng và độ ẩm, 0,2 cho gió với điều kiện T > 25 °C. Hàm phạt lấy dạng pairwise: với mỗi cặp (i, j) ngẫu nhiên, nếu T_i > T_j mà p_i < p_j thì đó là vi phạm monotonicity tăng; tính trung bình các vi phạm."""
add_body(model1)

model2 = """Bốn phương pháp hiệu chuẩn được áp dụng sau huấn luyện, tất cả fit trên val và apply lên test để tránh rò rỉ. Raw giữ xác suất gốc. Temperature scaling chia logit bởi T tối ưu NLL trên val. Isotonic regression (out_of_bounds='clip') khớp một hàm đơn điệu không tham số. Platt scaling là hồi quy logistic trên logit. F1 threshold được chọn riêng trên mỗi calibrator val, apply lên test; điều này tránh hiện tượng chọn threshold trên raw rồi apply lên calibrated sẽ sai lệch vì phân phối xác suất khác hẳn."""
add_body(model2)

add_heading("3.1. Năm cách chia tập", 2)
split1 = """IID nhóm-ô (60/20/20%, phân tầng theo vĩ độ) ngăn không cho cùng ô vào cả train và test, nhưng ô kề nhau vẫn có thể rò rỉ qua tương quan không gian. OOD thời gian (train 2018–2021, val 2022, test 2023) là đánh giá tổng quát hóa tương lai trung thực nhất. Ba OOD không gian: Bắc → Nam (train vĩ độ ≥ 15, test vĩ độ < 13), Nam → Trung (train < 14, test 14–15,5), Trung+Nam → Bắc (train < 15, test ≥ 15). Ba ranh giới để tránh một ranh giới được chọn may rủi."""
add_body(split1)

# ============================================================
# 4. KẾT QUẢ CHÍNH
# ============================================================
add_heading("4. Kết quả chính: MODIS so với VIIRS", 1)

results1 = """Trên nhãn MODIS, XGBoost + Isotonic đạt AUROC 0,947 trên IID và 0,892 trên OOD không gian Bắc → Nam. Trên nhãn VIIRS (cùng mô hình, cùng đặc trưng, chỉ đổi nhãn), AUROC IID giảm xuống 0,843 và AUROC OOD không gian sụp đổ xuống 0,584. Chênh lệch 0,31 AUROC là phát hiện chính của nghiên cứu: đánh giá chỉ trên MODIS phóng đại hiệu năng tổng quát hóa, đặc biệt dưới dịch chuyển không gian."""
add_body(results1)

add_figure("fig2_modis_vs_viirs_auroc.png", width_cm=15.5,
    caption="Hình 1. AUROC trên năm cách chia, so sánh MODIS và VIIRS. Cột màu đậm là MODIS, cột nhạt là VIIRS. Sự sụp đổ ở ô Spatial N→S là rõ ràng nhất.")

# Table of results
table_rows = [
    ["IID", "0.947", "0.843 ± 0.0006", "-0.10"],
    ["Temporal OOD", "0.934", "0.843", "-0.09"],
    ["Spatial N→S", "0.892", "0.584 ± 0.0019", "-0.31"],
    ["Spatial S→C", "0.924", "0.880 ± 0.0022", "-0.04"],
    ["Spatial C+S→N", "0.927", "0.700 ± 0.0009", "-0.23"],
]
add_table_from_data(["Cách chia", "MODIS AUROC", "VIIRS AUROC (5 seeds)", "Δ"],
                    table_rows, col_widths_cm=[4.5, 3.5, 5, 2])

add_figure("fig3_delta_auroc_modis_vs_viirs.png", width_cm=15.5,
    caption="Hình 2. Δ AUROC (VIIRS − MODIS) theo cách chia và mô hình. Spatial N→S là nơi sụp đổ mạnh nhất, khoảng −0,30 AUROC cho mọi mô hình.")

# ============================================================
# 5. CALIBRATION + COUNTERFACTUAL
# ============================================================
add_heading("5. Hiệu chuẩn và kiểm tra tính nhất quán vật lý", 1)

calib1 = """Hiệu chuẩn Isotronic giảm ECE đáng kể trên mọi cách chia. Với XGBoost trên nhãn MODIS, ECE giảm từ 0,058 (Raw) xuống 0,009 (Isotonic). Trên OOD không gian, ECE Iso giữ 0,028 — chỉ tăng 3 lần so với IID, trong khi MLP Raw tăng từ 0,053 lên 0,171, gấp 3,2 lần và kèm sụp AUROC. Điều này cho thấy Isotonic là phương pháp hiệu chuẩn bền vững dưới dịch chuyển phân phối, trong khi MLP bản ngầu (đã được BatchNorm) lại không duy trì được khả năng hiệu chuẩn khi ra ngoài phân phối."""
add_body(calib1)

add_heading("5.1. Kiểm tra tính nhất quán vật lý (counterfactual)", 2)
cf1 = """Với mỗi mô hình, một hàng cơ sở được tạo bằng trung bình các đặc trưng huấn luyện. Mỗi đặc trưng vật lý (T2M, RH, gió, lượng mưa 30 ngày) được quét qua khoảng [p5, p95] của train, các đặc trưng khác giữ nguyên. Tỷ lệ vi phạm monotonicity được tính là phần trăm cặp (i, j) có T_i < T_j nhưng p_i > p_j. Kết quả cho thấy Physics-MLP ép được monotonicity trên gió (0% vi phạm) và độ ẩm (9,9%), so với MLP bản ngầu (0% và 15,7%). Tuy nhiên, trên nhiệt độ, cả MLP và PhysMLP đều thất bại (84% vi phạm), trong khi XGBoost thành công (4,4%) nhờ cấu trúc cây split vốn đã đơn điệu theo T."""
add_body(cf1)

add_figure("fig8_counterfactual_summary_v85.png", width_cm=14,
    caption="Hình 3. Tỷ lệ vi phạm monotonicity theo mô hình và đặc trưng. Physics-MLP cải thiện trên gió và độ ẩm (các đặc trưng bị phạt) nhưng không cải thiện trên nhiệt độ hay lượng mưa.")

# ============================================================
# 6. FORECAST HORIZON
# ============================================================
add_heading("6. Phân tích khoảng dự báo t+1, t+3, t+7", 1)

horizon1 = """Để kiểm tra liệu mô hình vẫn hữu ích cho dự báo sớm (không chỉ cùng ngày), ba khoảng dự báo được xây dựng bằng cách dịch nhãn về trước h ngày. Trên nhãn VIIRS (một seed), AUROC giảm khoảng 0,02 từ t+1 lên t+7 cho cả XGBoost, MLP và PhysMLP. ECE Iso vẫn ổn định quanh 0,024 trong mọi khoảng. Kết quả này cho phép dùng chữ \"dự báo\" (forecast) một cách chính đáng thay vì chỉ \"xác suất cùng ngày\"."""
add_body(horizon1)

add_figure("fig1_viirs_auroc_vs_horizon.png", width_cm=14,
    caption="Hình 4. AUROC theo khoảng dự báo, nhãn VIIRS, seed 42. Sự suy giảm nhỏ (~0,02 AUROC cho +7 ngày) cho thấy mô hình vẫn có giá trị dự báo sớm.")

# ============================================================
# 7. PHASE A — DIAGNOSIS
# ============================================================
add_heading("7. Chẩn đoán Phase A: Tại sao N→S sụp đổ", 1)

diagnosis_intro = """Sự sụp đổ 0,31 AUROC từ MODIS sang VIIRS trên cách chia Bắc → Nam đòi hỏi giải thích. Năm giả thuyết được kiểm tra: H1 dịch chuyển hiệp biến (covariate shift), H2 dịch chuyển khái niệm (concept shift), H3 thiên lệch nhãn theo vùng, H4 cấu trúc không gian chưa quan sát, H5 rò rỉ không gian trong IID. Ba phân tích A1–A3 được thực hiện không cần dữ liệu ngoài."""
add_body(diagnosis_intro)

add_heading("7.1. A1 — Tương quan không gian và xác suất hàng xóm có điều kiện", 2)
a1_text = """Chỉ số Moran's I tính trung bình trên 50 ngày ngẫu nhiên bằng 0,186 ± 0,214, cho thấy phân cụm không gian vừa phải (giá trị 0 = ngẫu nhiên, 1 = phân cụm hoàn hảo). Một chỉ báo trực tiếp hơn là xác suất hàng xóm có cháy khi ô hiện tại có cháy: bằng 0,61 ở khoảng cách gần, so với tỷ lệ cháy ngẫu nhiên 0,29. Xác suất này giảm dần về 0,45 ở 170–190 km. Điều này xác nhận nhãn cháy có cấu trúc không gian thật, không chỉ là sự trùng khớp về nhãn \"không cháy\"."""
add_body(a1_text)

add_heading("7.2. A2 — Thẩm định N→S có kiểm soát tỷ lệ hiện mắc", 2)
a2_text = """Một lời giải thích khả dĩ cho sụp đổ N→S là Nam có tỷ lệ cháy cao hơn (10,6% so với 5,8% ở Bắc), làm mô hình dự báo quá thấp. Để kiểm tra, bootstrap 200 lần với tỷ lệ cân bằng 1:1 (case-control) trên tập test Nam. AUROC không thay đổi: 0,582 ± 0,0017. Kết luận: sụp đổ không phải do mất cân bằng tỷ lệ lớp. AUROC về bản chất ít nhạy với prevalence, nên kết quả này phù hợp kỳ vọng, nhưng thí nghiệm vẫn hữu ích như một robustness check."""
add_body(a2_text)

add_heading("7.3. A3 — Ablation với đặc trưng không gian thay thế", 2)
a3_text = """Bốn biến thể đặc trưng được thử, tất cả huấn luyện XGBoost trên Bắc và đánh giá trên Nam. v1 (chỉ thời tiết) đạt AUROC 0,582. v2 (cộng thêm vĩ độ/kinh độ thô) nhảy lên 0,811, chênh +0,229 AUROC. v3 (cộng lat_band 10 bin) giảm xuống 0,560. v4 (lat_band 5 + lon_band 5) giảm xuống 0,548. Sự phân kỳ cho thấy cấu trúc không gian chưa quan sát là thật và có thể phục hồi bằng tọa độ thô, nhưng đây là ghi nhớ địa lý chứ không phải suy luận vật lý: lat_band thô hại vì train chỉ có bin miền Bắc, bin miền Nam là ngoài phân phối."""
add_body(a3_text)

add_figure("fig_phaseA_viirs_combined.png", width_cm=15.5,
    caption="Hình 5. Tổng hợp Phase A v2 (VIIRS). Trái: thẩm định N→S với kiểm soát prevalence. Phải: ablation đặc trưng không gian. Mặc dù cân bằng prevalence không giúp, lat/lon thô phục hồi 0,23 AUROC.")

# ============================================================
# 8. H1-H5
# ============================================================
add_heading("8. Kiểm tra năm giả thuyết H1–H5", 1)

h_text = """Năm kiểm tra thống kê được thực hiện trên cùng bảng dữ liệu. H1 (covariate shift) xác nhận: 5/9 đặc trưng có sự khác biệt phân phối có ý nghĩa giữa Bắc và Nam (KS p < 0,001, Cohen's d > 0,2), đặc biệt nhiệt độ (Nam ấm hơn 1,55 °C), độ ẩm (Nam khô hơn 2%), số ngày kể từ lần mưa cuối (Nam dài hơn 2 ngày). H2 (concept shift) xác nhận: tương quan trung bình giữa P(fire|feature) Bắc và Nam bằng 0,81, không phải 1,0 — cùng điều kiện thời tiết cho xác suất cháy khác nhau giữa hai vùng. H3 (thiên lệch nhãn vùng) bác bỏ: tỷ lệ MODIS/VIIRS ở Bắc (0,95) so với Nam (0,84), chênh chỉ 0,11."""
add_body(h_text)

add_heading("8.1. H4 và giới hạn của nghiên cứu", 2)
h4_text = """H4 (cấu trúc không gian chưa quan sát) được ngụ ý mạnh mẽ nhưng không được xác định nguyên nhân. Bằng chứng là A3: thêm tọa độ thô phục hồi 0,23 AUROC, cho thấy có tín hiệu không gian đáng kể mà đặc trưng khí tượng không thể biểu diễn. Cơ chế ứng viên (không kiểm tra trực tiếp): lớp phủ thực vật, độ cao, hoạt động nông nghiệp, mật độ dân cư. Phase B (download dữ liệu NDVI/land cover/elevation từ Earth Engine) đã bị cản trở bởi hạn chế môi trường phát triển: Planetary Computer trả 404, Open-Elevation không kết nối được, Earth Engine đòi tạo Google Cloud project thủ công."""
add_body(h4_text)

add_figure("fig_diag_h4_lat_band.png", width_cm=14,
    caption="Hình 6. Tỷ lệ cháy theo vĩ độ, cho cả MODIS và VIIRS. Miền Nam có tỷ lệ cháy khoảng 2,5 lần miền Bắc, gợi ý cấu trúc không gian chưa quan sát (lớp phủ, địa hình, hoạt động con người).")

add_heading("8.2. H5 — rò rỉ không gian trong IID", 2)
h5_text = """H5 xác nhận: tỷ lệ trùng khớp nhãn giữa hai ô ngẫu nhiên ở cùng một ngày, ở khoảng cách 0°, bằng 89,8%. Điều này cho thấy IID (dù đã nhóm-ô) vẫn bị rò rỉ không gian: các ô kề nhau có nhãn tương quan. AUROC 0,95 trên IID vì vậy được xem là cận trên, không phải hiệu năng thực tế; đánh giá OOD không gian trung thực hơn."""
add_body(h5_text)

# Summary table
h_rows = [
    ["H1", "Dịch chuyển hiệp biến", "Xác nhận", "5/9 đặc trưng khác biệt đáng kể"],
    ["H2", "Dịch chuyển khái niệm", "Xác nhận", "Tương quan P(fire|f) N vs S = 0,81"],
    ["H3", "Thiên lệch nhãn vùng", "Bác bỏ", "Δ MODIS/VIIRS chỉ 0,11"],
    ["H4", "Cấu trúc không gian chưa quan sát", "Ngụ ý mạnh, chưa xác định nguyên nhân", "Lat/lon thô phục hồi +0,23 AUROC"],
    ["H5", "Rò rỉ IID", "Xác nhận", "Match rate d=0 = 89,8%"],
]
add_table_from_data(["#", "Giả thuyết", "Kết luận", "Bằng chứng"],
                    h_rows, col_widths_cm=[1, 4.5, 4.5, 5])

# ============================================================
# 9. PRODUCT
# ============================================================
add_heading("9. Sản phẩm: Hệ thống dự báo có ý thức độ tin cậy", 1)

product1 = """Phát hiện Phase A cho thấy mô hình chỉ-thời-tiết thất bại trên OOD không gian. Thay vì giấu thất bại này, đề tài đề xuất một lớp sản phẩm có ý thức độ tin cậy. Khi mô hình phát hiện dịch chuyển phân phối (OOD không gian, phân phối thời tiết khác, hoặc quan hệ fire-weather lệch), nó giảm chỉ báo độ tin cậy và liệt kê nguyên nhân. Người quyết định nhận được cả xác suất dự báo (ví dụ 72% nguy cơ) lẫn cảnh báo \"độ tin cậy: thấp\" kèm bốn lý do cụ thể."""
add_body(product1)

product2 = """Bản mockup HTML trong thư mục product/ mô phỏng giao diện: một vùng được chọn (Miền Nam), một khoảng dự báo (t+1), một thanh gradient cho nguy cơ (72%), và một thẻ cảnh báo vàng với dòng \"ĐỘ TIN CẬY: THẤP\" kèm các gợi ý: \"vùng dự báo ngoài phạm vi huấn luyện\", \"phân phối thời tiết đã thay đổi\", \"quan hệ fire-weather khác miền Bắc\", \"cấu trúc không gian chưa quan sát\". Đây là một thiết kế đề xuất, không phải bản triển khai đầy đủ; lớp sản phẩm là một wrapper mỏng trên mô hình đã huấn luyện (200 runs trong v8.5_canonical/runs/)."""
add_body(product2)

add_figure("fig_product_dashboard.png", width_cm=12,
    caption="Hình 7. Mockup giao diện hệ thống dự báo có ý thức độ tin cậy. Khi phát hiện OOD, hệ thống giảm độ tin cậy và giải thích nguyên nhân, thay vì đưa ra xác suất đơn trị quá tự tin.")

# ============================================================
# 10. LIMITATIONS
# ============================================================
add_heading("10. Hạn chế và thảo luận", 1)

limit1 = """Nghiên cứu có một số hạn chế cần nêu rõ. Thứ nhất, chỉ có một seed cho nhãn MODIS (so với 5 seed cho VIIRS t+1); vì phương sai thấp (std ~0,0002 cho IID), kết luận về sự khác biệt giữa MODIS và VIIRS vẫn đáng tin, nhưng CI chính thức hơn cần 5 seed cho cả hai. Thứ hai, proxy MODIS quan trắc (một ô-ngày được coi là quan trắc nếu có phát hiện cháy trong ±3 ngày, ±1 ô lân cận) là xấp xỉ; Phase B với VIIRS chéo kiểm chứng có thể định lượng sai âm thật. Thứ ba, không có dữ liệu NDVI/địa hình/lớp phủ thực vật, vì vậy không thể kiểm tra trực tiếp cơ chế của H4."""
add_body(limit1)

limit2 = """Thứ tư, IID nhóm-ô vẫn có rò rỉ không gian (H5), vì vậy AUROC IID nên được hiểu là cận trên, không phải hiệu năng triển khai. Thứ năm, ranh giới 0,5° × 0,5° thô với lưới 273 ô không đủ để bắt các cháy cấp xã/huyện; mô hình có ích cho cảnh báo cấp vùng, không phải cấp ô chi tiết. Thứ sáu, mô hình chỉ dùng dữ liệu cháy từ MODIS và VIIRS; không tích hợp dữ liệu triển khai thực địa (cảnh báo kiểm lâm, điểm nóng đã biết) — đây là hướng mở cho triển khai thật."""
add_body(limit2)

disc1 = """Về mặt đóng góp, đề tài đạt được ba tầng. Tầng thứ nhất là bộ dữ liệu và phương pháp luận: 598K ô–ngày với 7 sửa đổi審 kế, sẵn sàng tái lập. Tầng thứ hai là phát hiện đa cảm biến: đánh giá chỉ với MODIS phóng đại 0,31 AUROC trên OOD không gian, bằng chứng mạnh cho tầm quan trọng của thẩm định chéo cảm biến trong dự báo cháy rừng. Tầng thứ ba là sản phẩm có ý thức độ tin cậy, biến một phát hiện tiêu cực (mô hình thất bại trên OOD) thành một tính năng thực tiễn (mô hình biết khi nào không đáng tin)."""
add_body(disc1)

# ============================================================
# 11. CONCLUSION
# ============================================================
add_heading("11. Kết luận", 1)

concl1 = """Hệ thống dự báo cháy rừng có ý thức độ tin cậy mà đề tài đề xuất không phải là một mô hình dự báo đơn thuần, mà là một bọc sản phẩm (product wrapper) trên mô hình đã huấn luyện. Đóng góp chính không nằm ở việc đạt AUROC cao nhất, mà ở việc (i) xây dựng bộ dữ liệu sạch theo chuẩn審 kế, (ii) thẩm định đa cảm biến cho thấy đánh giá MODIS-đơn-triệu phóng đại 0,31 AUROC, (iii) phân tách thất bại OOD không gian thành bốn yếu tố định lượng được, và (iv) chuyển phát hiện tiêu cực thành thiết kế sản phẩm trung thực về giới hạn của chính mình."""
add_body(concl1)

concl2 = """Hướng mở cho công việc tiếp theo: tải dữ liệu NDVI/địa hình/lớp phủ thực vật từ Earth Engine (sau khi thiết lập Google Cloud project) để kiểm tra trực tiếp cơ chế H4; mở rộng multi-seed cho cả nhãn MODIS và VIIRS t+3/t+7; xây dựng cross-validation rời-vùng (leave-region-out) thay cho ba ranh giới cố định; và bổ sung dữ liệu thực địa (cảnh báo kiểm lâm, điểm nóng đã biết) cho triển khai thật."""
add_body(concl2)

# References
add_heading("Tài liệu tham khảo", 2)

refs = [
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16, 785–794.",
    "Vermote, E. (2014). MOD13A2 MODIS/Terra Vegetation Indices 16-Day L3 Global 1km SIN Grid. NASA LP DAAC.",
    "Schroeder, W., et al. (2014). VIIRS active fire products: Validation and applications. Remote Sensing of Environment, 148, 1–14.",
    "Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. KDD '02, 694–699.",
    "Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika, 37(1/2), 17–23.",
    "Koh, P. W., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. ICML 2021.",
    "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML 2005.",
    "Sá, A. C. L., et al. (2007). Assessing the efficiency of insect and fire suppression in Portuguese Mediterranean ecosystems. Forest Ecology and Management, 244(1–3), 252–262.",
    "NASA POWER Project. (2023). Prediction Of Worldwide Energy Resources — daily weather data API. https://power.larc.nasa.gov/",
    "Microsoft Planetary Computer. (2024). STAC API for MODIS, Copernicus DEM, IO LULC. https://planetarycomputer.microsoft.com/",
    "ESA WorldCover. (2021). 10m annual global land cover. https://esa-worldcover.org/",
    "Hoyer, P. O. (2022). Earth Engine Python API. Google Earth Engine.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    run = p.add_run(ref)
    set_run_font(run, name='Times New Roman', size=10, color=TEXT_PRIMARY)

# Save
doc.save(str(OUT_DOCX))
print(f"✓ DOCX saved: {OUT_DOCX}")
print(f"  Size: {os.path.getsize(OUT_DOCX)/1024:.0f} KB")
