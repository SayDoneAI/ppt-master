import base64, glob, os, re

# Load both logos
with open('/Users/ziyouyu/Documents/HK保险/星火品牌设计/永明星火品牌文件/logo/B@3x.png', 'rb') as f:
    logo_gold_b64 = base64.b64encode(f.read()).decode()
with open('/Users/ziyouyu/Documents/HK保险/星火品牌设计/永明星火品牌文件/logo/B@3x-3.png', 'rb') as f:
    logo_dark_b64 = base64.b64encode(f.read()).decode()

# Light background posters → dark logo (B@3x-3.png)
LIGHT_BG = {'P04', 'P10', 'P12', 'P15', 'P19', 'P22', 'P24', 'P26', 'P32', 'P35', 'P38'}

# Canvas: 1242x1660, logo tile settings
W, H = 1242, 1660
LW, LH = 318.7, 128.0
H_STEP, V_STEP = 380, 180
OFFSET = 220

def build_watermark(href, opacity):
    images = []
    row = 0
    y = 60
    while y < H + 200:
        x_start = -59.4 if row % 2 == 0 else -59.4 + OFFSET
        x = x_start
        while x < W + 200:
            cx = x + LW / 2
            cy = y + LH / 2
            images.append(
                f'  <image x="{x:.1f}" y="{y:.1f}" width="{LW}" height="{LH}" '
                f'transform="rotate(25 {cx:.0f} {cy:.0f})" href="{href}" />'
            )
            x += H_STEP
        y += V_STEP
        row += 1
    return f'<g opacity="{opacity}">\n' + '\n'.join(images) + '\n</g>', len(images)

# Pre-build both watermarks
wm_gold, n_gold = build_watermark(f"data:image/png;base64,{logo_gold_b64}", 0.12)
wm_dark, n_dark = build_watermark(f"data:image/png;base64,{logo_dark_b64}", 0.08)

# Process each SVG
svg_dir = '/Users/ziyouyu/Documents/xingbao/.cache/weekly_posters_xiaohongshu_20260322/svg_output'
light_count = dark_count = 0

for svg_file in sorted(glob.glob(f'{svg_dir}/*.svg')):
    name = os.path.basename(svg_file)
    poster_id = name.split('_')[0]  # e.g. "P04"

    with open(svg_file, 'r') as f:
        content = f.read()

    # Remove existing watermark groups
    content = re.sub(r'<g opacity="0\.1[02]">.*?</g>\n?', '', content, flags=re.DOTALL)
    content = re.sub(r'<g opacity="0\.08">.*?</g>\n?', '', content, flags=re.DOTALL)
    content = re.sub(r'<text[^>]*>永明星火团队[^<]*</text>\n?', '', content)

    # Select watermark based on background
    if poster_id in LIGHT_BG:
        watermark = wm_dark
        logo_type = 'dark'
        light_count += 1
    else:
        watermark = wm_gold
        logo_type = 'gold'
        dark_count += 1

    # Insert watermark before closing </svg>
    content = content.replace('</svg>', f'{watermark}\n</svg>')

    with open(svg_file, 'w') as f:
        f.write(content)

    print(f'Fixed: {name} → {logo_type} logo')

print(f'\nSummary: {light_count} light-bg (dark logo), {dark_count} dark-bg (gold logo)')
print('Done.')
