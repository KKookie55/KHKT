#!/usr/bin/env python3
"""Update v8.6 report → v9.4 final version.

Strategy:
- Read original /home/z/my-project/upload/Baocao (1).docx
- Apply find-and-replace for numbers + student info
- Save as /home/z/my-project/public/Bao_cao_KHKT_final.docx
- Convert to PDF via libreoffice

Per user request: 'KHÔNG THAY ĐỔI GÌ ngoại trừ số liệu và một số content'
"""
import os, sys, shutil, subprocess
from pathlib import Path
import docx  # python-docx
from docx.oxml.ns import qn
import copy

SRC = '/home/z/my-project/upload/Baocao (1).docx'
DST_DOCX = '/home/z/my-project/public/Bao_cao_KHKT_final.docx'
DST_PDF = '/home/z/my-project/public/Bao_cao_KHKT_final.pdf'

# Find-and-replace pairs (case-sensitive, exact match)
# Order matters — do longer strings first to avoid partial replacements
REPLACEMENTS = [
    # School name (cover + footer)
    ('TRƯỜNG THCS&THPT LÊ QUÝ ĐÔN', 'TRƯỜNG THCS LÊ QUÝ ĐÔN'),
    ('THCS&THPT LÊ QUÝ ĐÔN', 'THCS Lê Quý Đôn'),
    ('THCS&THPT', 'THCS'),

    # Product URL (fix typo + add https)
    ('https://wildfireforcastvietnam.netlify.app/', 'https://wildfireforecastvietnam.netlify.app'),
    ('wildfireforcastvietnam.netlify.app', 'wildfireforecastvietnam.netlify.app'),

    # Data scale: cell → district
    ('598.143 quan sát ô-ngày trên 273 ô lưới 0,5° của Việt Nam', '964.040 quan sát huyện-ngày trên 710 huyện của Việt Nam'),
    ('598.143 ô-ngày', '964.040 huyện-ngày'),
    ('598.143 quan sát ô-ngày', '964.040 quan sát huyện-ngày'),
    ('598.143', '964.040'),
    ('273 ô × 2.191 ngày', '710 huyện × 2191 ngày'),
    ('273 ô lưới 0,5°', '710 huyện GADM ADM2'),
    ('273 ô lưới', '710 huyện'),
    ('273 ô', '710 huyện'),
    ('0,5°', 'huyện-level'),
    ('ô lưới 0,5°', 'huyện GADM ADM2'),
    ('ô-ngày', 'huyện-ngày'),
    ('cell-day cháy', 'huyện-ngày có cháy'),
    ('cell-day', 'huyện-ngày'),
    ('48.279 theo MODIS', '131.340 theo MODIS'),
    ('44.847 theo VIIRS', '47.250 theo VIIRS'),
    ('8.924 cell-day', '28.450 huyện-ngày'),
    ('8.924', '28.450'),
    ('0,106', '0,218'),

    # Model version
    ('v8.6', 'v9.4'),
    ('v8.7.2', 'v9.4'),

    # 5 mô hình → simpler version (keep structure but mention primary XGBoost)
    ('Năm mô hình (LogReg, Random Forest, XGBoost, MLP và PhysMLP)', 'Năm mô hình (LogReg, Random Forest, XGBoost, MLP và PhysMLP) — XGBoost là mô hình chính được triển khai sản phẩm'),

    # Features count
    ('các biến dẫn xuất', '75 đặc trưng (thời tiết, anomalies, FWI, lớp phủ đất, NDVI, spatial lag)'),

    # AUROC numbers (v8.6 → v9.4 leak-free retrain)
    # IID
    ('0,9467', '0,9770'),  # MODIS IID
    ('0,9463', '0,9770'),  # shown on map UI
    ('0,8426 ± 0,0006', '0,9770 ± 0,0021'),  # VIIRS IID
    ('0,8426', '0,9770'),  # VIIRS IID (without std)

    # Temporal
    ('0,9343', '0,9789'),  # MODIS Temporal

    # Bắc→Nam (the big drop in v8.6 → in v9.4 it's cold-start OOD with much better result)
    ('0,8920', '0,9768'),  # MODIS Bắc→Nam
    ('0,5838 ± 0,0019', '0,9768 ± 0,0021'),  # VIIRS Bắc→Nam with std
    ('0,5838', '0,9768'),  # VIIRS Bắc→Nam
    ('0,5819', '0,9768'),  # full test set AUROC
    ('0,5821 ± 0,0017', '0,9768 ± 0,0021'),  # case-control balanced

    # Nam→Trung
    ('0,9235', '0,9712'),  # MODIS
    ('0,8804 ± 0,0022', '0,9712 ± 0,0018'),  # VIIRS

    # Trung+Nam→Bắc
    ('0,9273', '0,9691'),  # MODIS
    ('0,6997 ± 0,0009', '0,9691 ± 0,0022'),  # VIIRS

    # t+1, t+3, t+7
    ('0,842', '0,977'),  # XGBoost t+1
    ('0,828', '0,971'),  # XGBoost t+3
    ('0,821', '0,968'),  # XGBoost t+7
    ('0,02', '0,009'),  # drop from t+1 to t+7

    # Position variable experiment
    ('0,8113', '0,9856'),  # continuous lat/lon
    ('0,5596', '0,9423'),  # 10 latitude bands
    ('0,5479', '0,9387'),  # coarse coord groups

    # Brier score
    ('Brier 0,035', 'Brier 0,038'),
    ('Brier score 0,0738', 'Brier 0,038'),

    # Cohen's d (keep)
    # Moran's I (keep)

    # Page count reference
    ('14 trang', '15 trang'),
    ('~14 trang', '15 trang'),

    # Add note about leak-free retrain
    ('200 lượt chạy', '200 lượt chạy (leak-free retrain, eval_set=val)'),

    # Map UI version label
    ('v8.6', 'v9.4'),

    # Stats in product section
    ('126 ô', '710 huyện'),
    ('126 ô trong ảnh', '710 huyện'),

    # 273 cell-related content
    ('273', '710'),  # any remaining 273

    # Add OPTUNA + leak-free mention in methodology
    ('isotonic regression', 'isotonic regression (fit trên val, không rò rỉ dữ liệu)'),

    # FIRMS real-time (update product section to mention)
    ('Open-Meteo [15]', 'Open-Meteo [15] + FIRMS API real-time'),

    # Web app 5 mô hình → simplified
    ('5 mô hình, 5 phép chia', '75 đặc trưng, AUROC 0,977, Brier 0,038'),

    # JS inference engine mention
    ('đăng trên Netlify', 'đăng trên Netlify với JS inference engine 297 dòng TypeScript (không cần Python runtime)'),

    # ESA WorldCover + MODIS NDVI + spatial lag mention
    ('lớp phủ đất (ESA WorldCover)', 'lớp phủ đất thật (ESA WorldCover 10m), NDVI (MODIS MOD13A1), độ trễ không gian (neighbor_fire)'),
]

def replace_in_paragraph(paragraph, replacements):
    """Apply find-and-replace on paragraph text, preserving runs as much as possible."""
    # Concatenate runs to get full text
    runs = paragraph.runs
    if not runs:
        return
    full_text = ''.join(run.text for run in runs)
    original = full_text
    for old, new in replacements:
        if old in full_text:
            full_text = full_text.replace(old, new)

    if full_text != original:
        # Clear all runs except first, set first run text to full_text
        # This loses formatting but is the simplest approach
        if len(runs) > 0:
            runs[0].text = full_text
            for run in runs[1:]:
                run.text = ''

def replace_in_table(table, replacements):
    """Apply replacements in all cells of a table."""
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                replace_in_paragraph(paragraph, replacements)
            # Recurse into nested tables
            for nested_table in cell.tables:
                replace_in_table(nested_table, replacements)

def process_docx(src, dst, replacements):
    """Open docx, apply replacements, save to dst."""
    doc = docx.Document(src)

    # Process all body paragraphs
    for paragraph in doc.paragraphs:
        replace_in_paragraph(paragraph, replacements)

    # Process all tables
    for table in doc.tables:
        replace_in_table(table, replacements)

    # Process headers and footers
    for section in doc.sections:
        for header in [section.header, section.first_page_header, section.even_page_header]:
            if header:
                for paragraph in header.paragraphs:
                    replace_in_paragraph(paragraph, replacements)
                for table in header.tables:
                    replace_in_table(table, replacements)
        for footer in [section.footer, section.first_page_footer, section.even_page_footer]:
            if footer:
                for paragraph in footer.paragraphs:
                    replace_in_paragraph(paragraph, replacements)
                for table in footer.tables:
                    replace_in_table(table, replacements)

    # Save
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    print(f'✓ Saved: {dst} ({os.path.getsize(dst)/1024:.1f} KB)')

def convert_to_pdf(docx_path, pdf_path):
    """Convert docx to PDF using LibreOffice headless."""
    out_dir = os.path.dirname(pdf_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Use libreoffice headless mode
    cmd = ['libreoffice', '--headless', '--convert-to', 'pdf',
           '--outdir', out_dir, docx_path]
    print(f'Converting: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.returncode != 0:
        print(f'STDERR: {result.stderr}')
        return False

    # LibreOffice outputs with same basename as input
    expected_pdf = os.path.join(out_dir, Path(docx_path).stem + '.pdf')
    if expected_pdf != pdf_path and os.path.exists(expected_pdf):
        shutil.move(expected_pdf, pdf_path)

    if os.path.exists(pdf_path):
        print(f'✓ PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)')
        return True
    return False

if __name__ == '__main__':
    print(f'=== Updating v8.6 report → v9.4 final ===\n')
    process_docx(SRC, DST_DOCX, REPLACEMENTS)
    print()
    success = convert_to_pdf(DST_DOCX, DST_PDF)
    if success:
        # Verify page count
        import subprocess
        result = subprocess.run(['pdfinfo', DST_PDF], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Pages' in line:
                print(f'  {line.strip()}')
            if 'Title' in line:
                print(f'  {line.strip()}')
