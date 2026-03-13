#!/usr/bin/env python3
"""
PPT Master - 模板 Gallery 预览工具

扫描 templates/layouts/ 和 examples/ 目录，生成带 SVG 缩略图的 HTML 预览页面。

用法:
    python3 tools/template_gallery.py                    # 生成并启动预览
    python3 tools/template_gallery.py --no-serve          # 仅生成 HTML
    python3 tools/template_gallery.py --port 9000         # 指定端口
    python3 tools/template_gallery.py --output gallery.html  # 指定输出文件
"""

import argparse
import http.server
import os
import re
import sys
import threading
import webbrowser
from pathlib import Path
from html import escape

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates" / "layouts"
STYLES_DIR = REPO_ROOT / "templates" / "styles"
EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_OUTPUT = REPO_ROOT / "gallery.html"

# 模板分类（与 README.md 一致）
TEMPLATE_CATEGORIES = {
    "品牌风格": [
        "google_style", "mckinsey", "anthropic",
        "招商银行", "中汽研_常规", "中汽研_商务", "中汽研_现代",
        "中国电建_常规", "中国电建_现代",
    ],
    "通用风格": ["general", "consultant", "consultant_top", "科技蓝商务", "smart_red"],
    "场景专用": ["academic_defense", "psychology_attachment", "medical_university", "重庆大学"],
    "政企风格": ["government_red", "government_blue"],
    "特殊风格": ["pixel_retro"],
    "小红书": ["xhs_knowledge", "xhs_product", "xhs_minimal"],
    "朋友圈": ["moments_quote", "moments_event", "moments_brand"],
    "Story": ["story_vibrant", "story_elegant"],
    "公众号": ["wechat_tech", "wechat_warm"],
    "Banner": ["banner_corporate", "banner_creative"],
}

# 从 design_spec.md 提取简要描述
def extract_description(template_dir: Path) -> str:
    spec = template_dir / "design_spec.md"
    if not spec.exists():
        return ""
    try:
        text = spec.read_text(encoding="utf-8")
        # 取 > 引用行作为描述
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">") and len(line) > 2:
                return line.lstrip("> ").strip()
        return ""
    except Exception:
        return ""


def read_svg(path: Path) -> str:
    """读取 SVG 并清理 XML 声明"""
    try:
        content = path.read_text(encoding="utf-8")
        # 去掉 XML 声明，保留 <svg> 标签
        content = re.sub(r'<\?xml[^?]*\?>\s*', '', content)
        return content
    except Exception:
        return ""


def extract_aspect_ratio(svg_content: str) -> str:
    """从 SVG viewBox 提取宽高比，返回 CSS aspect-ratio 值（如 '1280/720'）"""
    m = re.search(r'viewBox\s*=\s*"[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)"', svg_content)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        if w > 0 and h > 0:
            return f"{int(w)}/{int(h)}"
    return "16/9"


def find_category(name: str) -> str:
    for cat, names in TEMPLATE_CATEGORIES.items():
        if name in names:
            return cat
    return "其他"


def build_template_section() -> list:
    """扫描模板目录，返回模板数据"""
    templates = []
    if not TEMPLATES_DIR.exists():
        return templates

    for d in sorted(TEMPLATES_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        cover = d / "01_cover.svg"
        pages = sorted(d.glob("*.svg"))
        if not pages:
            continue

        cover_svg = read_svg(cover) if cover.exists() else read_svg(pages[0])
        templates.append({
            "name": name,
            "category": find_category(name),
            "description": extract_description(d),
            "cover_svg": cover_svg,
            "aspect_ratio": extract_aspect_ratio(cover_svg),
            "pages": [{"name": p.stem, "svg": read_svg(p)} for p in pages],
        })

    return templates


def build_examples_section() -> list:
    """扫描 examples 目录，返回示例数据"""
    examples = []
    if not EXAMPLES_DIR.exists():
        return examples

    for d in sorted(EXAMPLES_DIR.iterdir()):
        if not d.is_dir():
            continue
        svg_dir = d / "svg_final"
        if not svg_dir.exists():
            svg_dir = d / "svg_output"
        if not svg_dir.exists():
            continue

        pages = sorted(svg_dir.glob("*.svg"))
        if not pages:
            continue

        # 从目录名提取信息
        name = d.name
        cover_svg = read_svg(pages[0])
        examples.append({
            "name": name,
            "cover_svg": cover_svg,
            "aspect_ratio": extract_aspect_ratio(cover_svg),
            "pages": [{"name": p.stem, "svg": read_svg(p)} for p in pages],
            "page_count": len(pages),
        })

    return examples


GITHUB_RAW = "https://raw.githubusercontent.com/PicoTrex/Awesome-Nano-Banana-images/main"

# 风格分类显示名
STYLE_CATEGORY_NAMES = {
    "poster_design": ("海报/卡片设计", "海报、名片、壁纸等平面设计"),
    "product_commercial": ("产品/商业", "产品摄影、包装、食物展示"),
    "style_artistic": ("风格转换", "赛博朋克、浮世绘、PIXAR 等"),
    "infographic_data": ("信息图/数据可视化", "流程图、地图、分镜、科普"),
    "threed_isometric": ("3D/等距/微缩", "等距视图、微缩场景、拆解图"),
    "illustration_comic": ("漫画/插画", "漫画、角色设定、线稿上色"),
    "portrait_character": ("人物/肖像", "换装、发型、风格化肖像"),
    "material_effect": ("材质/特效", "水晶、亚克力、光影控制"),
    "creative_other": ("其他创意", "坐标生图、递归、AR 等"),
}


def build_styles_section() -> list:
    """扫描 styles 目录，解析各分类的 README.md，返回风格数据"""
    styles = []
    if not STYLES_DIR.exists():
        return styles

    for d in sorted(STYLES_DIR.iterdir()):
        if not d.is_dir():
            continue
        readme = d / "README.md"
        if not readme.exists():
            continue

        cat_key = d.name
        cat_name, cat_desc = STYLE_CATEGORY_NAMES.get(cat_key, (cat_key, ""))

        # 解析 README 中的案例
        text = readme.read_text(encoding="utf-8")
        cases = []
        # 按 ## 标题分割（跳过第一个 # 大标题）
        parts = re.split(r'\n## (?!#)', text)
        for part in parts[1:]:  # skip first section (header)
            if part.startswith("---"):
                continue
            lines = part.strip().split("\n")
            if not lines:
                continue
            title = lines[0].strip().rstrip("\n")
            if title == "---":
                continue

            # 提取来源信息
            source_url = ""
            author = ""
            preview_url = ""
            prompt = ""
            case_type = ""

            for line in lines:
                # 来源
                m = re.search(r'\[(?:Pro|Regular)\s*例(\d+)\]\(([^)]*)\)\s*by\s*@(\S+)', line)
                if m:
                    case_type = "Pro" if "Pro" in line else ""
                    source_url = m.group(2)
                    author = m.group(3)
                # 预览图
                m = re.search(r'\*\*效果预览\*\*:\s*\[查看\]\(([^)]+)\)', line)
                if m:
                    preview_url = m.group(1)

            # 提取 prompt（code block）
            m = re.search(r'```\n(.*?)```', part, re.DOTALL)
            if m:
                prompt = m.group(1).strip()

            if title and title != "---":
                cases.append({
                    "title": title,
                    "author": author,
                    "source_url": source_url,
                    "preview_url": preview_url,
                    "prompt": prompt,
                    "is_pro": "Pro" in case_type,
                })

        if cases:
            styles.append({
                "key": cat_key,
                "name": cat_name,
                "desc": cat_desc,
                "cases": cases,
                "count": len(cases),
            })

    return styles


def _build_cards_html(items: list) -> str:
    """为一组模板生成卡片 HTML"""
    html = ""
    for t in items:
        pages_html = ""
        for i, p in enumerate(t["pages"]):
            display = "none" if i > 0 else "block"
            pages_html += f'<div class="page" data-index="{i}" style="display:{display}">{p["svg"]}</div>\n'

        desc = f'<p class="desc">{escape(t["description"])}</p>' if t.get("description") else ""
        page_count = len(t["pages"])
        nav = ""
        if page_count > 1:
            nav = f'''<div class="nav">
                <button onclick="prevPage(this)" class="btn">◀</button>
                <span class="page-info">1 / {page_count}</span>
                <button onclick="nextPage(this)" class="btn">▶</button>
            </div>'''

        name_key = t.get("name", "")
        html += f'''<div class="card" data-page-count="{page_count}">
            <div class="svg-wrap" style="aspect-ratio:{t["aspect_ratio"]}">{pages_html}</div>
            {nav}
            <div class="info">
                <h3 title="{escape(name_key)}">{escape(name_key)}</h3>
                {desc}
            </div>
        </div>\n'''
    return html


def generate_html(templates: list, examples: list, styles: list = None) -> str:
    """生成完整 HTML"""

    # 按分类分组模板
    categories = {}
    for t in templates:
        cat = t["category"]
        categories.setdefault(cat, []).append(t)

    # PPT 模板分类顺序
    ppt_cat_order = ["品牌风格", "通用风格", "场景专用", "政企风格", "特殊风格", "其他"]
    # 海报模板分类顺序
    poster_cat_order = ["小红书", "朋友圈", "Story", "公众号", "Banner"]

    def _build_section(cat_order):
        html = ""
        for cat in cat_order:
            items = categories.get(cat, [])
            if not items:
                continue
            html += f'<h2 class="cat-title">{escape(cat)}</h2>\n<div class="grid">\n'
            html += _build_cards_html(items)
            html += '</div>\n'
        return html

    ppt_html = _build_section(ppt_cat_order)
    poster_html = _build_section(poster_cat_order)

    # 统计数量
    ppt_count = sum(len(categories.get(c, [])) for c in ppt_cat_order)
    poster_count = sum(len(categories.get(c, [])) for c in poster_cat_order)

    # 示例卡片 HTML
    example_html = '<div class="grid">\n'
    for ex in examples:
        page_count = ex["page_count"]
        ex_item = {
            "name": ex["name"],
            "aspect_ratio": ex["aspect_ratio"],
            "pages": ex["pages"],
            "description": f"{page_count} 页",
        }
        example_html += _build_cards_html([ex_item])
    example_html += '</div>\n'

    # 风格灵感 HTML
    styles = styles or []
    style_count = sum(s["count"] for s in styles)
    styles_html = ""
    for s in styles:
        styles_html += f'<h2 class="cat-title">{escape(s["name"])} <span style="font-size:14px;color:#86868b;font-weight:400">{escape(s["desc"])} · {s["count"]} 个案例</span></h2>\n'
        styles_html += '<div class="style-grid">\n'
        for c in s["cases"]:
            pro_badge = '<span class="pro-badge">Pro</span>' if c["is_pro"] else ''
            prompt_preview = escape(c["prompt"][:120] + "..." if len(c["prompt"]) > 120 else c["prompt"])
            prompt_full = escape(c["prompt"]).replace("\n", "&#10;")
            img_html = ""
            if c["preview_url"]:
                img_html = f'<img class="style-img" src="{escape(c["preview_url"])}" alt="{escape(c["title"])}" loading="lazy" onerror="this.parentElement.innerHTML=\'<div class=style-placeholder>预览不可用</div>\'">'
            else:
                img_html = '<div class="style-placeholder">无预览</div>'

            source_link = f'<a href="{escape(c["source_url"])}" target="_blank" class="source-link">@{escape(c["author"])}</a>' if c["source_url"] else f'@{escape(c["author"])}'

            styles_html += f'''<div class="style-card">
                <div class="style-img-wrap">{img_html}</div>
                <div class="style-info">
                    <h4>{pro_badge}{escape(c["title"])}</h4>
                    <p class="style-author">{source_link}</p>
                    <div class="style-prompt" title="{prompt_full}">{prompt_preview}</div>
                    <button class="copy-btn" onclick="copyPrompt(this, `{prompt_full}`)">复制 Prompt</button>
                </div>
            </div>\n'''
        styles_html += '</div>\n'

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPT Master — 模板 Gallery</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, "Segoe UI", sans-serif; background: #f5f5f7; color: #1d1d1f; }}
header {{ background: #fff; border-bottom: 1px solid #e0e0e0; padding: 24px 40px; position: sticky; top: 0; z-index: 100; }}
header h1 {{ font-size: 24px; font-weight: 600; }}
header p {{ color: #86868b; font-size: 14px; margin-top: 4px; }}
.tabs {{ display: flex; gap: 8px; padding: 16px 40px 0; background: #fff; border-bottom: 1px solid #e0e0e0; }}
.tab {{ padding: 10px 24px; cursor: pointer; border: none; background: none; font-size: 15px; color: #86868b;
        border-bottom: 2px solid transparent; transition: all .2s; }}
.tab:hover {{ color: #1d1d1f; }}
.tab.active {{ color: #0071e3; border-bottom-color: #0071e3; font-weight: 600; }}
.tab .badge {{ display: inline-block; background: #e8e8ed; color: #86868b; font-size: 12px; font-weight: 500;
              padding: 1px 8px; border-radius: 10px; margin-left: 6px; vertical-align: middle; }}
.tab.active .badge {{ background: #0071e3; color: #fff; }}
.section {{ display: none; padding: 24px 40px 60px; }}
.section.active {{ display: block; }}
.cat-title {{ font-size: 20px; font-weight: 600; margin: 28px 0 16px; padding-bottom: 8px; border-bottom: 1px solid #e0e0e0; }}
.cat-title:first-child {{ margin-top: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 24px; margin-bottom: 16px; }}
.card {{ background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08);
         transition: box-shadow .2s, transform .2s; cursor: default; }}
.card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.12); transform: translateY(-2px); }}
.svg-wrap {{ background: #e8e8ed; overflow: hidden; position: relative; }}
.svg-wrap svg {{ width: 100%; height: 100%; display: block; }}
.svg-wrap .page {{ position: absolute; inset: 0; }}
.svg-wrap .page svg {{ width: 100%; height: 100%; }}
.nav {{ display: flex; align-items: center; justify-content: center; gap: 12px; padding: 6px 0; background: #fafafa; border-top: 1px solid #f0f0f0; }}
.btn {{ border: none; background: #e8e8ed; width: 28px; height: 28px; border-radius: 50%; cursor: pointer;
        font-size: 12px; display: flex; align-items: center; justify-content: center; transition: background .2s; }}
.btn:hover {{ background: #d1d1d6; }}
.page-info {{ font-size: 13px; color: #86868b; min-width: 50px; text-align: center; }}
.info {{ padding: 12px 16px; }}
.info h3 {{ font-size: 15px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.info .desc {{ font-size: 13px; color: #86868b; margin-top: 4px; line-height: 1.4;
               display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}

/* 风格灵感卡片 */
.style-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-bottom: 16px; }}
.style-card {{ background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08);
              transition: box-shadow .2s, transform .2s; display: flex; flex-direction: column; }}
.style-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.12); transform: translateY(-2px); }}
.style-img-wrap {{ height: 200px; overflow: hidden; background: #f0f0f0; }}
.style-img {{ width: 100%; height: 100%; object-fit: cover; }}
.style-placeholder {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
                      color: #86868b; font-size: 14px; background: #e8e8ed; }}
.style-info {{ padding: 12px 16px; flex: 1; display: flex; flex-direction: column; }}
.style-info h4 {{ font-size: 15px; font-weight: 600; margin-bottom: 4px; }}
.pro-badge {{ display: inline-block; background: #0071e3; color: #fff; font-size: 11px; font-weight: 600;
             padding: 1px 6px; border-radius: 4px; margin-right: 6px; vertical-align: middle; }}
.style-author {{ font-size: 13px; color: #86868b; margin-bottom: 8px; }}
.source-link {{ color: #0071e3; text-decoration: none; }}
.source-link:hover {{ text-decoration: underline; }}
.style-prompt {{ font-size: 12px; color: #515154; line-height: 1.5; background: #f5f5f7; padding: 8px 10px;
                border-radius: 6px; margin-bottom: 8px; flex: 1; max-height: 80px; overflow: hidden;
                font-family: 'SF Mono', 'Menlo', monospace; word-break: break-all; }}
.copy-btn {{ border: 1px solid #d1d1d6; background: #fff; color: #515154; padding: 6px 14px; border-radius: 6px;
            font-size: 13px; cursor: pointer; transition: all .2s; align-self: flex-start; }}
.copy-btn:hover {{ background: #0071e3; color: #fff; border-color: #0071e3; }}
.copy-btn.copied {{ background: #34c759; color: #fff; border-color: #34c759; }}

/* 全屏预览 */
.overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 200;
            align-items: center; justify-content: center; flex-direction: column; }}
.overlay.show {{ display: flex; }}
.overlay .preview-svg {{ max-width: 90vw; max-height: 85vh; background: #fff; border-radius: 8px; overflow: hidden; }}
.overlay .preview-svg svg {{ width: 100%; height: 100%; display: block; }}
.overlay .preview-nav {{ display: flex; align-items: center; gap: 16px; margin-top: 16px; }}
.overlay .preview-nav button {{ border: none; background: rgba(255,255,255,.2); color: #fff; width: 40px; height: 40px;
            border-radius: 50%; cursor: pointer; font-size: 18px; transition: background .2s; }}
.overlay .preview-nav button:hover {{ background: rgba(255,255,255,.4); }}
.overlay .preview-info {{ color: rgba(255,255,255,.7); font-size: 14px; min-width: 80px; text-align: center; }}
.overlay .close-btn {{ position: absolute; top: 20px; right: 24px; border: none; background: none;
            color: #fff; font-size: 32px; cursor: pointer; opacity: .7; transition: opacity .2s; }}
.overlay .close-btn:hover {{ opacity: 1; }}
</style>
</head>
<body>
<header>
    <h1>PPT Master — 模板 Gallery</h1>
    <p>{len(templates)} 个模板风格 · {len(examples)} 个完整示例 · {style_count} 个 AI 生图参考</p>
</header>

<div class="tabs">
    <button class="tab active" onclick="switchTab('ppt')">PPT 模板<span class="badge">{ppt_count}</span></button>
    <button class="tab" onclick="switchTab('poster')">海报模板<span class="badge">{poster_count}</span></button>
    <button class="tab" onclick="switchTab('examples')">完整示例<span class="badge">{len(examples)}</span></button>
    <button class="tab" onclick="switchTab('styles')">风格灵感<span class="badge">{style_count}</span></button>
</div>

<div id="ppt" class="section active">
{ppt_html}
</div>

<div id="poster" class="section">
{poster_html}
</div>

<div id="examples" class="section">
{example_html}
</div>

<div id="styles" class="section">
{styles_html}
</div>

<div class="overlay" id="overlay">
    <button class="close-btn" onclick="closeOverlay()">✕</button>
    <div class="preview-svg" id="previewSvg"></div>
    <div class="preview-nav">
        <button onclick="previewPrev()">◀</button>
        <span class="preview-info" id="previewInfo"></span>
        <button onclick="previewNext()">▶</button>
    </div>
</div>

<script>
function switchTab(id) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.closest('.tab').classList.add('active');
}}

function getCard(el) {{ return el.closest('.card'); }}

function prevPage(btn) {{
    const card = getCard(btn);
    const pages = card.querySelectorAll('.page');
    const total = pages.length;
    let cur = [...pages].findIndex(p => p.style.display !== 'none');
    pages[cur].style.display = 'none';
    cur = (cur - 1 + total) % total;
    pages[cur].style.display = 'block';
    card.querySelector('.page-info').textContent = (cur + 1) + ' / ' + total;
}}

function nextPage(btn) {{
    const card = getCard(btn);
    const pages = card.querySelectorAll('.page');
    const total = pages.length;
    let cur = [...pages].findIndex(p => p.style.display !== 'none');
    pages[cur].style.display = 'none';
    cur = (cur + 1) % total;
    pages[cur].style.display = 'block';
    card.querySelector('.page-info').textContent = (cur + 1) + ' / ' + total;
}}

// 全屏预览
let previewPages = [];
let previewIndex = 0;

document.addEventListener('click', (e) => {{
    const svgWrap = e.target.closest('.svg-wrap');
    if (!svgWrap) return;
    const card = svgWrap.closest('.card');
    previewPages = [...card.querySelectorAll('.page')].map(p => p.innerHTML);
    const cur = [...card.querySelectorAll('.page')].findIndex(p => p.style.display !== 'none');
    previewIndex = Math.max(cur, 0);
    // 传递宽高比到全屏预览
    const ar = svgWrap.style.aspectRatio || '16/9';
    document.getElementById('previewSvg').style.aspectRatio = ar;
    showPreview();
}});

function showPreview() {{
    document.getElementById('previewSvg').innerHTML = previewPages[previewIndex];
    document.getElementById('previewInfo').textContent = (previewIndex + 1) + ' / ' + previewPages.length;
    document.getElementById('overlay').classList.add('show');
    document.body.style.overflow = 'hidden';
}}

function closeOverlay() {{
    document.getElementById('overlay').classList.remove('show');
    document.body.style.overflow = '';
}}

function previewPrev() {{
    previewIndex = (previewIndex - 1 + previewPages.length) % previewPages.length;
    showPreview();
}}

function previewNext() {{
    previewIndex = (previewIndex + 1) % previewPages.length;
    showPreview();
}}

document.addEventListener('keydown', (e) => {{
    if (!document.getElementById('overlay').classList.contains('show')) return;
    if (e.key === 'Escape') closeOverlay();
    if (e.key === 'ArrowLeft') previewPrev();
    if (e.key === 'ArrowRight') previewNext();
}});

document.getElementById('overlay').addEventListener('click', (e) => {{
    if (e.target.id === 'overlay') closeOverlay();
}});

function copyPrompt(btn, text) {{
    const decoded = text.replace(/&#10;/g, '\\n');
    navigator.clipboard.writeText(decoded).then(() => {{
        btn.textContent = '已复制';
        btn.classList.add('copied');
        setTimeout(() => {{ btn.textContent = '复制 Prompt'; btn.classList.remove('copied'); }}, 1500);
    }});
}}
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="PPT Master 模板 Gallery 预览")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="输出 HTML 文件路径")
    parser.add_argument("--port", "-p", type=int, default=8000, help="预览服务器端口")
    parser.add_argument("--no-serve", action="store_true", help="仅生成 HTML，不启动服务器")
    args = parser.parse_args()

    print("[SCAN] 扫描模板目录...")
    templates = build_template_section()
    print(f"  找到 {len(templates)} 个模板")

    print("[SCAN] 扫描示例目录...")
    examples = build_examples_section()
    print(f"  找到 {len(examples)} 个示例")

    print("[SCAN] 扫描风格提示词库...")
    styles = build_styles_section()
    style_count = sum(s["count"] for s in styles)
    print(f"  找到 {len(styles)} 个分类，共 {style_count} 个案例")

    print("[BUILD] 生成 HTML...")
    html = generate_html(templates, examples, styles)

    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"[DONE] 已生成: {output_path} ({size_kb:.0f} KB)")

    if not args.no_serve:
        serve_dir = str(output_path.parent)
        filename = output_path.name
        os.chdir(serve_dir)

        handler = http.server.SimpleHTTPRequestHandler
        server = http.server.HTTPServer(("", args.port), handler)

        url = f"http://localhost:{args.port}/{filename}"
        print(f"[SERVE] 预览地址: {url}")
        print("[SERVE] 按 Ctrl+C 停止")

        # 自动打开浏览器
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[STOP] 服务器已停止")
            server.server_close()


if __name__ == "__main__":
    main()
