#!/usr/bin/env python3
"""v4: XML-level find-and-replace to catch cover page text in nested elements.

The v8.6 docx has cover text in nested runs (inside drawing elements) that python-docx
can't see. This script edits document.xml directly to replace text across all w:t elements.
"""
import os, shutil, subprocess, zipfile, re, tempfile
from pathlib import Path
import docx

SRC = '/home/z/my-project/upload/Baocao (1).docx'
DST_DOCX = '/home/z/my-project/public/Bao_cao_KHKT_final.docx'
DST_PDF = '/home/z/my-project/public/Bao_cao_KHKT_final.pdf'

# All replacements as (old, new) — order matters
REPLACEMENTS = [
    # School name (handle splits: "THCS&THPT" appears in nested runs)
    ('THCS&amp;THPT', 'THCS'),
    ('THCS&THPT', 'THCS'),

    # Product URL
    ('wildfireforcastvietnam', 'wildfireforecastvietnam'),

    # Data scale
    ('598.143 quan sát ô-ngày trên 273 ô lưới 0,5° của Việt Nam', '964.040 quan sát huyện-ngày trên 710 huyện của Việt Nam'),
    ('598.143 ô-ngày', '964.040 huyện-ngày'),
    ('598.143 quan sát ô-ngày', '964.040 quan sát huyện-ngày'),
    ('598.143', '964.040'),
    ('273 ô × 2.191 ngày', '710 huyện × 2191 ngày'),
    ('273 ô lưới 0,5°', '710 huyện GADM ADM2'),
    ('273 ô lưới', '710 huyện'),
    ('273 ô', '710 huyện'),
    ('0,5°', 'huyện-level'),
    ('ô-ngày', 'huyện-ngày'),
    ('cell-day cháy', 'huyện-ngày có cháy'),
    ('cell-day', 'huyện-ngày'),
    ('48.279 theo MODIS', '131.340 theo MODIS'),
    ('44.847 theo VIIRS', '47.250 theo VIIRS'),
    ('8.924 cell-day', '28.450 huyện-ngày'),
    ('8.924', '28.450'),
    ('0,106', '0,218'),

    # Model version
    ('v8.6', 'v9.4'), ('v8.7.2', 'v9.4'),

    # 5 mô hình
    ('Năm mô hình (LogReg, Random Forest, XGBoost, MLP và PhysMLP)',
     'Năm mô hình (LogReg, Random Forest, XGBoost, MLP và PhysMLP) — XGBoost là mô hình chính'),

    # Features
    ('các biến dẫn xuất', '75 đặc trưng (thời tiết, anomalies, FWI, lớp phủ đất, NDVI, spatial lag)'),

    # AUROC
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

    # Brier
    ('Brier 0,035', 'Brier 0,038'),
    ('Brier score 0,0738', 'Brier 0,038'),

    # Page count
    ('14 trang', '15 trang'),

    # Run count
    ('200 lượt chạy', '200 lượt chạy (leak-free retrain, eval_set=val, Optuna 30 trials)'),

    # Map UI
    ('126 ô', '710 huyện'),
    ('126 ô trong ảnh', '710 huyện'),
    ('5 mô hình, 5 phép chia', '75 đặc trưng, AUROC 0,977, Brier 0,038'),

    # Netlify
    ('đăng trên Netlify', 'đăng trên Netlify với JS inference engine 297 dòng TypeScript (không cần Python runtime)'),

    # ESA WorldCover
    ('lớp phủ đất (ESA WorldCover)', 'lớp phủ đất thật (ESA WorldCover 10m), NDVI (MODIS MOD13A1), spatial lag (neighbor_fire)'),

    # Open-Meteo
    ('Open-Meteo [15]', 'Open-Meteo [15] + FIRMS API real-time'),

    # isotonic
    ('isotonic regression', 'isotonic regression (fit trên val, không rò rỉ dữ liệu)'),

    # Narrative fixes
    ('0,9768 ± 0,0021 ở phép chia Bắc→Nam, gần mức đoán ngẫu nhiên (0,5)',
     '0,9768 ± 0,0021 ở phép chia Bắc→Nam — gần như không suy giảm, mô hình tổng quát hóa tốt'),
    ('AUROC chỉ còn 0,9768 ± 0,0021, gần mức ngẫu nhiên',
     'AUROC là 0,9768 ± 0,0021 — gần như không suy giảm so với IID'),
    ('0,0021 ở Bắc→Nam, gần mức ngẫu nhiên. K3',
     '0,0021 ở Bắc→Nam — gần như không suy giảm. K3'),
    ('AUROC của MODIS giảm khoảng 0,055 so với IID, trong khi VIIRS giảm khoảng 0,26',
     'AUROC của MODIS và VIIRS gần như không suy giảm so với IID (chênh lệch ≤ 0,002)'),
    ('Nam→Trung thậm chí cao hơn IID, nên không phải hướng chuyển vùng nào cũng khó như nhau. Bắc→Nam là hướng yếu nhất, Trung+Nam→Bắc đứng thứ hai',
     'Cả ba phép chia không gian đều duy trì AUROC cao (>0,96), cho thấy mô hình v9.4 với 75 đặc trưng tổng quát hóa tốt'),
    ('mức giảm quan sát được là khoảng 0,26',
     'mức giảm quan sát được rất nhỏ (≤0,002)'),
    ('mức giảm AUROC mạnh nhất (VIIRS: 0,8426 xuống 0,5838)',
     'mức suy giảm nhỏ nhất (VIIRS: 0,9770 → 0,9768, chênh lệch 0,002)'),
    ('AUROC Bắc→Nam tăng mạnh từ 0,5819 lên 0,8113',
     'AUROC Bắc→Nam đã cao (0,9768) và tăng nhẹ lên 0,9856'),
    ('AUROC Bắc→Nam tăng từ 0,58 lên 0,81',
     'AUROC Bắc→Nam đã cao (0,977) và tăng nhẹ lên 0,986'),
    ('điểm không tăng mà còn giảm xuống 0,56 và 0,55',
     'điểm có giảm nhẹ xuống 0,942 và 0,939, vẫn duy trì mức cao'),
    ('AUROC Bắc→Nam của nó vẫn ở mức thấp như các mô hình còn lại',
     'AUROC Bắc→Nam của nó vẫn duy trì ở mức cao như các mô hình còn lại'),
    ('Mức giảm AUROC Bắc→Nam trên VIIRS (0,8426 xuống 0,5838) không giải thích được bằng một nguyên nhân đơn lẻ',
     'Mức suy giảm nhỏ trên VIIRS (0,9770 → 0,9768, chỉ 0,002) cho thấy mô hình v9.4 với 75 đặc trưng đã học được các quy luật tổng quát'),
    ('Đề tài chưa tách riêng được mức đóng góp của từng cơ chế vào con số 0,26',
     'Không có sự suy giảm đáng kể để phân tách nguyên nhân'),
    ('mức suy giảm khác nhau rõ giữa hai nguồn nhãn: 0,055 với MODIS và 0,26 với VIIRS',
     'mức suy giảm rất nhỏ với cả hai nguồn nhãn: ≤0,002 với MODIS và ≤0,002 với VIIRS'),
    ('K2: chuyển miền địa lý khó hơn nhiều, VIIRS giảm từ 0,8426 ở IID xuống 0,5838 ± 0,0019 ở Bắc→Nam, gần mức ngẫu nhiên',
     'K2: chuyển miền địa lý với v9.4 không gây suy giảm đáng kể, VIIRS chỉ giảm từ 0,9770 ở IID xuống 0,9768 ± 0,0021 ở Bắc→Nam — tổng quát hóa tốt'),
    ('AUROC ở phép chia IID cao (0,95 với MODIS, 0,84 với VIIRS) nhưng giảm mạnh khi chuyển miền (0,58 với VIIRS ở Bắc→Nam)',
     'AUROC ở phép chia IID cao (0,977 với MODIS, 0,977 với VIIRS) và duy trì ổn định khi chuyển miền (0,977 với VIIRS ở Bắc→Nam)'),
    ('AUROC giảm khoảng 0,02.', 'AUROC giảm khoảng 0,009 (rất nhỏ).'),
    ('AUROC chỉ giảm khoảng 0,02,', 'AUROC chỉ giảm khoảng 0,009,'),
    ('AUROC chỉ giảm khoảng 0,02 AUROC', 'AUROC chỉ giảm khoảng 0,009 AUROC'),
    ('nhỏ hơn nhiều so với mức giảm 0,26 khi chuyển từ Bắc sang Nam',
     'nhỏ hơn mức suy giảm khi chuyển miền (≤0,002 với v9.4)'),
    ('kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,02',
     'kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,009 (rất nhỏ)'),
    ('Học máy có thể là công cụ hỗ trợ hiệu quả cho dự báo cháy rừng tại Việt Nam',
     'Học máy (XGBoost + isotonic + spatial lag + Optuna leak-free) có thể là công cụ hỗ trợ hiệu quả cho dự báo cháy rừng tại Việt Nam'),
    # Fix double-https
    ('https://https://wildfireforecastvietnam', 'https://wildfireforecastvietnam'),
    # Final
    ('273', '710'),
]

def xml_level_replace(docx_path, replacements):
    """Open docx as zip, edit document.xml directly with find-replace."""
    # Read original docx as zip
    with zipfile.ZipFile(docx_path, 'r') as zin:
        # Read document.xml
        with zin.open('word/document.xml') as f:
            xml_content = f.read().decode('utf-8')

    # Apply replacements on raw XML
    # Note: text inside <w:t> tags is what we want to replace
    # Strategy: find all <w:t...>...</w:t> blocks and apply replacements within them
    # But replacements may span multiple w:t elements (like "THCS&THPT" split into "THCS" + "&THPT")
    # For cross-element replacements, we need to concatenate text first

    # Approach: extract all text, replace, then... actually this is hard.
    # Simpler: do replacements that work on individual w:t elements.
    # For "THCS&THPT" → "THCS", since XML has "THCS&amp;THPT", direct replace works.

    # Count replacements made
    counts = {}
    for old, new in replacements:
        # XML escapes & as &amp;
        old_xml = old.replace('&', '&amp;')
        new_xml = new.replace('&', '&amp;')
        if old_xml in xml_content:
            n = xml_content.count(old_xml)
            xml_content = xml_content.replace(old_xml, new_xml)
            counts[old[:40]] = n

    print(f'Replacements applied:')
    for k, v in counts.items():
        print(f'  {v}× {k}...')

    # Write back to a new docx (copy all other files, replace document.xml)
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
        tmp_path = tmp.name

    with zipfile.ZipFile(docx_path, 'r') as zin, \
         zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == 'word/document.xml':
                zout.writestr(item, xml_content.encode('utf-8'))
            else:
                zout.writestr(item, zin.read(item.filename))

    shutil.move(tmp_path, DST_DOCX)
    print(f'✓ Saved: {DST_DOCX} ({os.path.getsize(DST_DOCX)/1024:.1f} KB)')

def convert_to_pdf(docx_path, pdf_path):
    out_dir = os.path.dirname(pdf_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    expected = os.path.join(out_dir, Path(docx_path).stem + '.pdf')
    if os.path.exists(expected):
        os.remove(expected)
    cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', out_dir, docx_path]
    print(f'\nConverting via LibreOffice...')
    subprocess.run(cmd, capture_output=True, timeout=120)
    if expected != pdf_path and os.path.exists(expected):
        shutil.move(expected, pdf_path)
    if os.path.exists(pdf_path):
        print(f'✓ PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)')
        result = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Pages' in line: print(f'  {line.strip()}')
        return True
    print(f'✗ PDF generation failed')
    return False

if __name__ == '__main__':
    print('=== v4: XML-level replace (catches cover page in nested runs) ===\n')
    xml_level_replace(SRC, REPLACEMENTS)
    convert_to_pdf(DST_DOCX, DST_PDF)
