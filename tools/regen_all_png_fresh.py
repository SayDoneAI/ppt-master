import subprocess, glob, os

svg_dir = '/Users/ziyouyu/Documents/HK保险/招募600人-2026/xingbao-output/weekly_posters'
png_dir = '/Users/ziyouyu/Documents/HK保险/招募600人-2026/本周海报PNG_0322'

svgs = sorted(glob.glob(f'{svg_dir}/*.svg'))
total = len(svgs)

for i, svg_path in enumerate(svgs):
    name = os.path.splitext(os.path.basename(svg_path))[0]
    png_path = os.path.join(png_dir, f'{name}.png')

    cmd = ['rsvg-convert', '-w', '1242', '-h', '1660', '-o', png_path, svg_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f'[{i+1}/{total}] {name}.png FAILED: {result.stderr.strip()}')
    else:
        subprocess.run(['xattr', '-c', png_path], capture_output=True)
        size_kb = os.path.getsize(png_path) // 1024 if os.path.exists(png_path) else 0
        print(f'[{i+1}/{total}] {name}.png ({size_kb}KB)')

print(f'\nDone. {len(glob.glob(f"{png_dir}/*.png"))} PNGs generated.')
