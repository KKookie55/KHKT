#!/usr/bin/env python3
"""Overlay fixes on existing poster: GVHD name + school name + add Hiệu trưởng.

Existing poster 05_Phuluc_7_Poster_A1.pdf has correct content but:
- "THCS & THPT Lê Quý Đôn" → "THCS Lê Quý Đôn"
- "[Tên GVHD]" → "Lê Xuân Tân"
- Add "Hiệu trưởng: Nguyễn Anh Tuấn"

Approach: Cover old text with white rectangle + draw new text on top.
"""
import os, io
from pathlib import Path
from reportlab.lib.pagesizes import A1
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from PIL import Image
import subprocess

SRC = '/home/z/my-project/KHKT_HoSo/05_Phuluc_7_Poster_A1.pdf'
DST = '/home/z/my-project/KHKT_HoSo/FINAL_POSTER.pdf'

# Convert page 1 to image to find text positions
def get_text_positions(pdf_path):
    """Render page 1 as image and use VLM to find positions of placeholder text."""
    subprocess.run(['pdftoppm', '-r', '50', pdf_path, '/tmp/poster_pos', '-jpeg'],
                   capture_output=True)
    return '/tmp/poster_pos-1.jpg'

# Manual approach: cover known placeholder text with white rectangles
# Positions are approximate based on A1 portrait (594×841mm)
# 1 mm = 2.83465 pts in PDF coords (1 inch = 72 pts, 1 inch = 25.4mm)

# From the existing poster layout:
# - School name is on the right side at top, around y=815-820mm from bottom
# - GVHD line is just below at y=808-812mm
# Let me use PIL to find exact positions of text

def find_text_pos(image_path, search_text):
    """Use OCR to find text position - simplified approach via VLM."""
    return None  # We'll use hardcoded positions based on visual inspection

# Based on visual inspection of the existing poster:
# The "Hoàng Bảo Lâm — Lớp 9E4 — THCS & THPT Lê Quý Đôn" line is at top right
# The "GVHD: [Tên GVHD] · Mã dự án: ..." line is just below it

# Coordinates (in mm from top-left of A1 portrait):
# Page: 594 × 841 mm
# Right side starts around x=300mm
# Top area y=10-30mm

COVER_RECTS = [
    # (x_mm, y_mm_from_top, w_mm, h_mm) — to cover with white rectangle
    # Right side: school name line
    (340, 12, 240, 6),   # covers "Hoàng Bảo Lâm — Lớp 9E4 — THCS & THPT Lê Quý Đôn"
    (340, 19, 240, 6),   # covers "GVHD: [Tên GVHD] · Mã dự án: HN26-129F5F83"
]

NEW_TEXT = [
    # (x_mm, y_mm_from_top, text, font_size, bold)
    (345, 16, 'Hoàng Bảo Lâm — Lớp 9E4 — THCS Lê Quý Đôn', 9, True),
    (345, 23, 'GVHD: Lê Xuân Tân · Hiệu trưởng: Nguyễn Anh Tuấn · Mã: HN26-129F5F83', 8, False),
]


def overlay_pdf():
    """Create overlay PDF with white rectangles + new text."""
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A1)
    width, height = A1  # 1683.78 x 2383.94 pts

    # Cover rectangles (white)
    c.setFillColor(colors.white)
    for x, y, w, h in COVER_RECTS:
        # Convert mm to pts, y from top → from bottom
        x_pt = x * mm
        y_pt = height - (y + h) * mm  # top-left of rectangle
        c.rect(x_pt, y_pt, w * mm, h * mm, fill=1, stroke=0)

    # New text
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('Lib', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('Lib-Bold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))

    for x, y, text, font_size, bold in NEW_TEXT:
        font = 'Lib-Bold' if bold else 'Lib'
        c.setFont(font, font_size)
        c.setFillColor(colors.HexColor('#0f172a'))
        x_pt = x * mm
        y_pt = height - y * mm
        c.drawString(x_pt, y_pt, text)

    c.showPage()
    c.save()

    # Write overlay to file
    overlay_path = '/tmp/poster_overlay.pdf'
    with open(overlay_path, 'wb') as f:
        f.write(packet.getvalue())

    return overlay_path


def merge_with_original():
    """Merge overlay onto original poster."""
    overlay_path = overlay_pdf()
    reader_src = PdfReader(SRC)
    reader_overlay = PdfReader(overlay_path)

    writer = PdfWriter()
    for i, page in enumerate(reader_src.pages):
        if i == 0:
            page.merge_page(reader_overlay.pages[0])
        writer.add_page(page)

    with open(DST, 'wb') as f:
        writer.write(f)
    print(f'✓ FINAL_POSTER.pdf: {DST} ({os.path.getsize(DST)/1024:.1f} KB)')


if __name__ == '__main__':
    print('=== Overlay fixes on existing poster ===\n')
    merge_with_original()
    # Verify
    result = subprocess.run(['pdfinfo', DST], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Pages' in line or 'Page size' in line:
            print(f'  {line.strip()}')
