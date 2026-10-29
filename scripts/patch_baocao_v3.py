#!/usr/bin/env python3
"""v3 patch: fix remaining narrative inconsistencies."""
import os, shutil, subprocess
from pathlib import Path
import docx

DOCX = '/home/z/my-project/public/Bao_cao_KHKT_final.docx'
PDF = '/home/z/my-project/public/Bao_cao_KHKT_final.pdf'

PATCHES = [
    # Remove "gần mức ngẫu nhiên" for AUROC 0.9768 (which is high, not random)
    ('AUROC chỉ còn 0,9768 ± 0,0021, gần mức ngẫu nhiên',
     'AUROC là 0,9768 ± 0,0021 — gần như không suy giảm so với IID'),
    ('0,0021 ở Bắc→Nam, gần mức ngẫu nhiên. K3',
     '0,0021 ở Bắc→Nam — gần như không suy giảm. K3'),
    # Fix 0,02 → 0,009 for t+1 to t+7 drop
    ('AUROC giảm khoảng 0,02.', 'AUROC giảm khoảng 0,009 (rất nhỏ).'),
    ('AUROC chỉ giảm khoảng 0,02,', 'AUROC chỉ giảm khoảng 0,009,'),
    ('AUROC chỉ giảm khoảng 0,02 AUROC', 'AUROC chỉ giảm khoảng 0,009 AUROC'),
    # Fix "nhỏ hơn nhiều so với mức giảm 0,26 khi chuyển từ Bắc sang Nam"
    ('nhỏ hơn nhiều so với mức giảm 0,26 khi chuyển từ Bắc sang Nam',
     'nhỏ hơn mức suy giảm khi chuyển miền (≤0,002 với v9.4)'),
    # Fix "AUROC của MODIS và VIIRS gần như không suy giảm so với IID (chênh lệch ≤ 0,002)"
    # already done

    # Add leak-free note
    ('leak-free retrain, eval_set=val', 'leak-free retrain, eval_set=val, Optuna 30 trials'),

    # "Bắc→Nam là hướng yếu nhất, Trung+Nam→Bắc đứng thứ hai"
    # Already updated but let me also update the next sentence
    ('mức giảm quan sát được rất nhỏ (≈0,002), cho thấy mô hình v9.4 tổng quát hóa tốt',
     'mức giảm quan sát được rất nhỏ (≤0,002), cho thấy mô hình v9.4 với 75 đặc trưng (bao gồm lớp phủ đất thật, NDVI thật, và spatial lag neighbor_fire) đã học được các quy luật tổng quát không phụ thuộc vị trí cụ thể'),

    # "Đề tài chưa tách riêng được mức đóng góp của từng cơ chế vào con số 0,26"
    # Already updated

    # "kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,02"
    # in abstract — already fixed but check if any remaining
    ('kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,02',
     'kéo dài khoảng dự báo từ t+1 đến t+7 chỉ làm AUROC giảm khoảng 0,009 (rất nhỏ)'),

    # Conclusion section: "K2: chuyển miền địa lý với v9.4 không gây suy giảm đáng kể"
    # already updated

    # Add mention of 75 features + Optuna in conclusion
    ('Học máy có thể là công cụ hỗ trợ hiệu quả cho dự báo cháy rừng tại Việt Nam',
     'Học máy (XGBoost + isotonic calibration + spatial lag + Optuna leak-free) có thể là công cụ hỗ trợ hiệu quả cho dự báo cháy rừng tại Việt Nam'),

    # Make sure product URL appears with https
    ('wildfireforecastvietnam.netlify.app', 'https://wildfireforecastvietnam.netlify.app'),
    ('https://https://wildfireforecastvietnam.netlify.app', 'https://wildfireforecastvietnam.netlify.app'),

    # "0,5°" remaining
    ('0,5°', 'huyện-level'),

    # Final 273 reference (if any)
    ('273', '710'),
]

def replace_in_paragraph(p, replacements):
    runs = p.runs
    if not runs: return
    full_text = ''.join(r.text for r in runs)
    original = full_text
    for old, new in replacements:
        if old in full_text:
            full_text = full_text.replace(old, new)
    if full_text != original:
        runs[0].text = full_text
        for r in runs[1:]: r.text = ''

def replace_in_table(table, replacements):
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs: replace_in_paragraph(p, replacements)
            for t in cell.tables: replace_in_table(t, replacements)

def main():
    doc = docx.Document(DOCX)
    for p in doc.paragraphs:
        replace_in_paragraph(p, PATCHES)
    for t in doc.tables:
        replace_in_table(t, PATCHES)
    for section in doc.sections:
        for h in [section.header, section.first_page_header, section.even_page_header]:
            if h:
                for p in h.paragraphs: replace_in_paragraph(p, PATCHES)
                for t in h.tables: replace_in_table(t, PATCHES)
        for f in [section.footer, section.first_page_footer, section.even_page_footer]:
            if f:
                for p in f.paragraphs: replace_in_paragraph(p, PATCHES)
                for t in f.tables: replace_in_table(t, PATCHES)
    doc.save(DOCX)
    print(f'✓ Patched docx ({os.path.getsize(DOCX)/1024:.1f} KB)')

    # Convert to PDF
    out_dir = os.path.dirname(PDF)
    expected = os.path.join(out_dir, Path(DOCX).stem + '.pdf')
    cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', out_dir, DOCX]
    subprocess.run(cmd, capture_output=True, timeout=120)
    if expected != PDF and os.path.exists(expected):
        shutil.move(expected, PDF)
    if os.path.exists(PDF):
        print(f'✓ PDF: {PDF} ({os.path.getsize(PDF)/1024:.1f} KB)')
        result = subprocess.run(['pdfinfo', PDF], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Pages' in line: print(f'  {line.strip()}')

if __name__ == '__main__':
    main()
