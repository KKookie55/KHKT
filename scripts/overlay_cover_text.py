#!/usr/bin/env python3
"""v5: Overlay cover text on existing PDF page 1.

Problem: LibreOffice doesn't render Word text boxes in the docx, so the PDF cover page
is blank (only decorative border shows).

Solution: Use ReportLab to create a transparent overlay with the cover text, then merge
it on top of page 1 of the existing PDF using pypdf.
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
import io
from pypdf import PdfReader, PdfWriter

pdfmetrics.registerFont(TTFont('Lib', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Bold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Lib-Italic', '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf'))
registerFontFamily('Lib', normal='Lib', bold='Lib-Bold', italic='Lib-Italic')

SRC_PDF = '/home/z/my-project/public/Bao_cao_KHKT_final.pdf'
OVERLAY_PDF = '/tmp/cover_overlay.pdf'
DST_PDF = '/home/z/my-project/public/Bao_cao_KHKT_final.pdf'  # overwrite

# Cover text (matching v8.6 layout but with v9.4 updates)
# y_offset is in mm from top. A4 = 297mm tall, keep within 30-285mm
COVER_TEXT_BLOCKS = [
    # (y_offset_from_top, font_size, font_bold, text, alignment)
    (40, 11, False, 'SỞ GIÁO DỤC VÀ ĐÀO TẠO HÀ NỘI', TA_CENTER),
    (50, 11, True,  'TRƯỜNG THCS LÊ QUÝ ĐÔN', TA_CENTER),
    (85, 14, True,  'BÁO CÁO KẾT QUẢ NGHIÊN CỨU', TA_CENTER),
    (95, 14, True, 'CUỘC THI KHOA HỌC KĨ THUẬT', TA_CENTER),
    (120, 12, True, 'DỰ ÁN KHOA HỌC:', TA_CENTER),
    (135, 13, True, 'HỆ THỐNG DỰ BÁO CHÁY RỪNG CÓ Ý THỨC ĐỘ TIN CẬY', TA_CENTER),
    (145, 13, True, 'DƯỚI DỊCH CHUYỂN PHÂN PHỐI KHÔNG GIAN', TA_CENTER),
    (155, 13, True, 'BẰNG MÔ HÌNH HỌC MÁY', TA_CENTER),
    (180, 10, False, 'Lĩnh vực: Phần mềm hệ thống', TA_CENTER),
    (190, 10, False, 'Cấp học: THCS', TA_CENTER),
    (215, 10, False, 'Giáo viên hướng dẫn: Lê Xuân Tân', TA_CENTER),
    (225, 10, False, 'Học sinh thực hiện: Hoàng Bảo Lâm', TA_CENTER),
    (235, 10, False, 'Lớp: 9E4', TA_CENTER),
    (245, 10, False, 'Trường: THCS Lê Quý Đôn', TA_CENTER),
    (255, 10, False, 'Hiệu trưởng: Nguyễn Anh Tuấn', TA_CENTER),
    (275, 9, True,  'Link sản phẩm: wildfireforecastvietnam.netlify.app', TA_CENTER),
    (285, 10, False, 'Hà Nội, tháng 10 năm 2026', TA_CENTER),
]

def create_overlay():
    """Create overlay PDF with cover text."""
    packet = io.BytesIO()
    # Use A4 portrait
    doc = SimpleDocTemplate(packet, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=0, bottomMargin=0)
    # Style
    st = ParagraphStyle('CoverText', fontName='Lib', fontSize=11,
                        alignment=TA_CENTER, textColor=colors.HexColor('#0f172a'))
    st_bold = ParagraphStyle('CoverTextBold', parent=st, fontName='Lib-Bold')

    # Use a single-column layout with absolute positioning via spacers
    from reportlab.platypus import Frame, KeepInFrame
    # Approach: place each text block at specific y coordinate

    # Use canvas directly for absolute positioning
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(packet, pagesize=A4)
    width, height = A4  # 595.27 x 841.89 pts

    for y_offset, font_size, bold, text, align in COVER_TEXT_BLOCKS:
        font = 'Lib-Bold' if bold else 'Lib'
        c.setFont(font, font_size)
        # y_offset is from top; convert to canvas coord (from bottom)
        y = height - y_offset * mm
        if align == TA_CENTER:
            text_width = c.stringWidth(text, font, font_size)
            x = (width - text_width) / 2
            c.drawString(x, y, text)
        else:
            c.drawString(2*cm, y, text)

    c.showPage()
    c.save()

    # Write to file
    packet.seek(0)
    with open(OVERLAY_PDF, 'wb') as f:
        f.write(packet.read())
    print(f'✓ Overlay PDF: {OVERLAY_PDF}')

def merge_overlay():
    """Merge overlay on page 1 of source PDF."""
    reader_src = PdfReader(SRC_PDF)
    reader_overlay = PdfReader(OVERLAY_PDF)

    writer = PdfWriter()
    for i, page in enumerate(reader_src.pages):
        if i == 0:
            # Merge overlay on page 1
            page.merge_page(reader_overlay.pages[0])
        writer.add_page(page)

    with open(DST_PDF, 'wb') as f:
        writer.write(f)
    print(f'✓ Merged PDF: {DST_PDF} ({os.path.getsize(DST_PDF)/1024:.1f} KB)')

if __name__ == '__main__':
    print('=== v5: Overlay cover text on existing PDF ===\n')
    create_overlay()
    merge_overlay()
    # Verify
    import subprocess
    result = subprocess.run(['pdfinfo', DST_PDF], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Pages' in line: print(f'  {line.strip()}')
