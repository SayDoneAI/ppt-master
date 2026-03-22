import io
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

# Paths
pdf_in = '/Users/ziyouyu/Documents/HK保险/招募600人-2026/周例会/20260322_本周财经热点科学管钱实战参考_周会简报.pdf'
pdf_out = '/Users/ziyouyu/Documents/HK保险/招募600人-2026/周例会/20260322_本周财经热点科学管钱实战参考_周会简报_水印版.pdf'
logo_path = '/Users/ziyouyu/Documents/HK保险/星火品牌设计/永明星火品牌文件/logo/B@3x-3.png'

# Load logo and get dimensions
logo_img = Image.open(logo_path)
logo_w, logo_h = logo_img.size

# Watermark settings
wm_width = 120  # watermark logo width in points
wm_height = wm_width * logo_h / logo_w
opacity = 0.12
rotation = 25  # degrees
spacing_x = 180  # horizontal spacing between tiles
spacing_y = 160  # vertical spacing between tiles

reader = PdfReader(pdf_in)
writer = PdfWriter()

for page in reader.pages:
    pw = float(page.mediabox.width)
    ph = float(page.mediabox.height)

    # Create watermark overlay for this page
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(pw, ph))

    # Tile the logo across the page
    y = -spacing_y
    row = 0
    while y < ph + spacing_y:
        x_offset = (spacing_x / 2) * (row % 2)  # stagger rows
        x = -spacing_x + x_offset
        while x < pw + spacing_x:
            c.saveState()
            c.translate(x + wm_width / 2, y + wm_height / 2)
            c.rotate(rotation)
            c.setFillAlpha(opacity)
            c.drawImage(
                ImageReader(logo_path),
                -wm_width / 2, -wm_height / 2,
                width=wm_width, height=wm_height,
                mask='auto',
                preserveAspectRatio=True
            )
            c.restoreState()
            x += spacing_x
        y += spacing_y
        row += 1

    c.save()
    packet.seek(0)

    # Merge watermark onto page
    wm_page = PdfReader(packet).pages[0]
    page.merge_page(wm_page)
    writer.add_page(page)

with open(pdf_out, 'wb') as f:
    writer.write(f)

print(f'Done: {pdf_out}')
