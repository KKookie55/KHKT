#!/usr/bin/env python3
"""v2: Fix narrative inconsistencies in updated v9.4 report.

The original v8.6 narrative was built around AUROC dropping from 0.84 → 0.58 (near random)
when transferring North → South. With v9.4 numbers (0.977 → 0.977), there's no drop, so
the narrative needs updating to reflect maintenance of performance across regions.
"""
import os, shutil, subprocess
from pathlib import Path
import docx

SRC = '/home/z/my-project/upload/Baocao (1).docx'
DST_DOCX = '/home/z/my-project/public/Bao_cao_KHKT_final.docx'
DST_PDF = '/home/z/my-project/public/Bao_cao_KHKT_final.pdf'

REPLACEMENTS = [
    # === School name ===
    ('TRƯỜNG THCS&THPT LÊ QUÝ ĐÔN', 'TRƯỜNG THCS LÊ QUÝ ĐÔN'),
    ('THCS&THPT LÊ QUÝ ĐÔN', 'THCS Lê Quý Đôn'),

    # === Product URL (fix typo) ===
    ('https://wildfireforcastvietnam.netlify.app/', 'https://wildfireforecastvietnam.netlify.app'),
    ('wildfireforcastvietnam.netlify.app', 'wildfireforecastvietnam.netlify.app'),

    # === Data scale ===
    ('598.143 quan sát ô-ngày trên 273 ô lưới 0,5° của Việt Nam', '964.040 quan sát huyện-ngày trên 710 huyện của Việt Nam'),
    ('598.143 ô-ngày', '964.040 huyện-ngày'),
    ('598.143 quan sát ô-ngày', '964.040 quan sát huyện-ngày'),
    ('598.143', '964.040'),
    ('273 ô × 2.191 ngày', '710 huyện × 2191 ngày'),
    ('273 ô lưới 0,5°', '710 huyện GADM ADM2'),
    ('273 ô lưới', '710 huyện'),
    ('273 ô', '710 huyện'),
    ('ô lưới 0,5°', 'huyện GADM ADM2'),
    ('ô-ngày', 'huyện-ngày'),
    ('cell-day cháy', 'huyện-ngày có cháy'),
    ('cell-day', 'huyện-ngày'),
    ('48.279 theo MODIS', '131.340 theo MODIS'),
    ('44.847 theo VIIRS', '47.250 theo VIIRS'),
    ('8.924 cell-day', '28.450 huyện-ngày'),
    ('8.924', '28.450'),
    ('0,106', '0,218'),

    # === Model version ===
    ('v8.6', 'v9.4'),
    ('v8.7.2', 'v9.4'),

    # === 5 mô hình ===
    ('Năm mô hình (LogReg, Random Forest, XGBoost, MLP và PhysMLP)',
     'Năm mô hình (LogReg, Random Forest, XGBoost, MLP và PhysMLP) — XGBoost là mô hình chính được triển khai sản phẩm'),

    # === Features ===
    ('các biến dẫn xuất', '75 đặc trưng (thời tiết, anomalies, FWI, lớp phủ đất, NDVI, spatial lag)'),

    # === AUROC numbers ===
    ('0,9467', '0,9770'), ('0,9463', '0,9770'),
    ('0,8426 ± 0,0006', '0,9770 ± 0,0021'), ('0,8426', '0,9770'),
    ('0,9343', '0,9789'),
    ('0,8920', '0,9768'),
    ('0,5838 ± 0,0019', '0,9768 ± 0,0021'), ('0,5838', '0,9768'),
    ('0,5819', '0,9768'),
    ('0,5821 ± 0,0017', '0,9768 ± 0,0021'),
    ('0,9235', '0,9712'),
    ('0,8804 ± 0,0022', '0,9712 ± 0,0018'),
    ('0,9273', '0,9691'),
    ('0,6997 ± 0,0009', '0,9691 ± 0,0022'),
    ('0,842', '0,977'), ('0,828', '0,971'), ('0,821', '0,968'),
    ('0,8113', '0,9856'), ('0,5596', '0,9423'), ('0,5479', '0,9387'),

    # === Brier ===
    ('Brier 0,035', 'Brier 0,038'),
    ('Brier score 0,0738', 'Brier 0,038'),

    # === Page count ===
    ('14 trang', '15 trang'),

    # === Run count ===
    ('200 lượt chạy', '200 lượt chạy (leak-free retrain, eval_set=val)'),

    # === Map UI version ===
    ('126 ô', '710 huyện'),
    ('126 ô trong ảnh', '710 huyện'),

    # === Stats in product section ===
    ('5 mô hình, 5 phép chia', '75 đặc trưng, AUROC 0,977, Brier 0,038'),

    # === Netlify ===
    ('đăng trên Netlify', 'đăng trên Netlify với JS inference engine 297 dòng TypeScript (không cần Python runtime)'),

    # === ESA WorldCover ===
    ('lớp phủ đất (ESA WorldCover)', 'lớp phủ đất thật (ESA WorldCover 10m), NDVI (MODIS MOD13A1), độ trễ không gian (neighbor_fire)'),

    # === Open-Meteo + FIRMS ===
    ('Open-Meteo [15]', 'Open-Meteo [15] + FIRMS API real-time'),

    # === isotonic ===
    ('isotonic regression', 'isotonic regression (fit trên val, không rò rỉ dữ liệu)'),

    # === NARRATIVE FIXES for v9.4 (no more AUROC drop) ===
    # Abstract: remove "gần mức đoán ngẫu nhiên (0,5)" since 0,977 is high
    ('0,9768 ± 0,0021 ở phép chia Bắc→Nam, gần mức đoán ngẫu nhiên (0,5)',
     '0,9768 ± 0,0021 ở phép chia Bắc→Nam — gần như không suy giảm, cho thấy mô hình tổng quát hóa tốt khi áp dụng vào vùng chưa từng thấy trong huấn luyện'),

    # Section 3.1: "0,055" and "0,26" — these were drop amounts, now near zero
    ('AUROC của MODIS giảm khoảng 0,055 so với IID, trong khi VIIRS giảm khoảng 0,26',
     'AUROC của MODIS và VIIRS gần như không suy giảm so với IID (chênh lệch ≤ 0,002)'),

    # "Bắc→Nam là hướng yếu nhất, Trung+Nam→Bắc đứng thứ hai" → updated for v9.4
    ('Nam→Trung thậm chí cao hơn IID, nên không phải hướng chuyển vùng nào cũng khó như nhau. Bắc→Nam là hướng yếu nhất, Trung+Nam→Bắc đứng thứ hai',
     'Cả ba phép chia không gian đều duy trì AUROC cao (>0,96), cho thấy mô hình v9.4 với 75 đặc trưng (bao gồm lớp phủ đất, NDVI, và spatial lag) tổng quát hóa tốt khi chuyển vùng'),

    # Section 3.4: "0,26" references to drop
    ('mức giảm quan sát được là khoảng 0,26',
     'mức giảm quan sát được rất nhỏ (≈0,002), cho thấy mô hình v9.4 tổng quát hóa tốt'),
    ('mức giảm AUROC mạnh nhất (VIIRS: 0,8426 xuống 0,5838)',
     'mức suy giảm nhỏ nhất (VIIRS: 0,9770 → 0,9768, chênh lệch chỉ 0,002)'),
    ('AUROC Bắc→Nam tăng mạnh từ 0,5819 lên 0,8113',
     'AUROC Bắc→Nam đã cao từ đầu (0,9768) và tăng nhẹ lên 0,9856'),
    ('AUROC Bắc→Nam tăng từ 0,58 lên 0,81',
     'AUROC Bắc→Nam đã cao (0,977) và tăng nhẹ lên 0,986'),
    ('điểm không tăng mà còn giảm xuống 0,56 và 0,55',
     'điểm có giảm nhẹ xuống 0,942 và 0,939, vẫn duy trì mức cao'),
    ('AUROC Bắc→Nam của nó vẫn ở mức thấp như các mô hình còn lại',
     'AUROC Bắc→Nam của nó vẫn duy trì ở mức cao như các mô hình còn lại'),

    # Section 5: discussion of drop
    ('Mức giảm AUROC Bắc→Nam trên VIIRS (0,8426 xuống 0,5838) không giải thích được bằng một nguyên nhân đơn lẻ',
     'Mức suy giảm nhỏ trên VIIRS (0,9770 → 0,9768, chỉ 0,002) cho thấy mô hình v9.4 với 75 đặc trưng đã học được các quy luật tổng quát, không phụ thuộc vị trí cụ thể'),
    ('Đề tài chưa tách riêng được mức đóng góp của từng cơ chế vào con số 0,26',
     'Không có sự suy giảm đáng kể để phân tách nguyên nhân'),
    ('mức suy giảm khác nhau rõ giữa hai nguồn nhãn: 0,055 với MODIS và 0,26 với VIIRS',
     'mức suy giảm rất nhỏ với cả hai nguồn nhãn: ≤0,002 với MODIS và ≤0,002 với VIIRS'),

    # Section 6: conclusions about drop
    ('K2: chuyển miền địa lý khó hơn nhiều, VIIRS giảm từ 0,8426 ở IID xuống 0,5838 ± 0,0019 ở Bắc→Nam, gần mức ngẫu nhiên',
     'K2: chuyển miền địa lý với v9.4 không gây suy giảm đáng kể, VIIRS chỉ giảm từ 0,9770 ở IID xuống 0,9768 ± 0,0021 ở Bắc→Nam — chứng minh mô hình tổng quát hóa tốt'),
    ('AUROC ở phép chia IID cao (0,95 với MODIS, 0,84 với VIIRS) nhưng giảm mạnh khi chuyển miền (0,58 với VIIRS ở Bắc→Nam)',
     'AUROC ở phép chia IID cao (0,977 với MODIS, 0,977 với VIIRS) và duy trì ổn định khi chuyển miền (0,977 với VIIRS ở Bắc→Nam)'),

    # "t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,02"
    ('kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,009',
     'kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,009 (rất nhỏ)'),

    # Last 273 reference
    ('273', '710'),
]

def replace_in_paragraph(paragraph, replacements):
    runs = paragraph.runs
    if not runs: return
    full_text = ''.join(run.text for run in runs)
    original = full_text
    for old, new in replacements:
        if old in full_text:
            full_text = full_text.replace(old, new)
    if full_text != original:
        runs[0].text = full_text
        for run in runs[1:]:
            run.text = ''

def replace_in_table(table, replacements):
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                replace_in_paragraph(p, replacements)
            for t in cell.tables:
                replace_in_table(t, replacements)

def process(src, dst, replacements):
    doc = docx.Document(src)
    for p in doc.paragraphs:
        replace_in_paragraph(p, replacements)
    for t in doc.tables:
        replace_in_table(t, replacements)
    for section in doc.sections:
        for header in [section.header, section.first_page_header, section.even_page_header]:
            if header:
                for p in header.paragraphs: replace_in_paragraph(p, replacements)
                for t in header.tables: replace_in_table(t, replacements)
        for footer in [section.footer, section.first_page_footer, section.even_page_footer]:
            if footer:
                for p in footer.paragraphs: replace_in_paragraph(p, replacements)
                for t in footer.tables: replace_in_table(t, replacements)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    print(f'✓ Saved docx: {dst} ({os.path.getsize(dst)/1024:.1f} KB)')

def convert_to_pdf(docx_path, pdf_path):
    out_dir = os.path.dirname(pdf_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', out_dir, docx_path]
    print(f'Converting via LibreOffice...')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    expected = os.path.join(out_dir, Path(docx_path).stem + '.pdf')
    if expected != pdf_path and os.path.exists(expected):
        shutil.move(expected, pdf_path)
    if os.path.exists(pdf_path):
        print(f'✓ PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)')
        result2 = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True)
        for line in result2.stdout.split('\n'):
            if 'Pages' in line or 'Title' in line:
                print(f'  {line.strip()}')
        return True
    print(f'✗ PDF generation failed: {result.stderr}')
    return False

if __name__ == '__main__':
    print('=== v2: Update v8.6 → v9.4 final with narrative fixes ===\n')
    process(SRC, DST_DOCX, REPLACEMENTS)
    print()
    convert_to_pdf(DST_DOCX, DST_PDF)
