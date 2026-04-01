---
name: design
description: >
  AI-native multi-page SVG design system. Creates PPTs, posters, social media
  graphics (小红书, 朋友圈, Story, Banner), infographics, and any visual content
  from text, URLs, PDFs, or raw ideas. Supports 8+ canvas formats, 33 chart
  templates, 640+ icons, AI image generation, and PPTX export. Includes a
  browser-based editor for live preview and drag-and-drop refinement.
  Use this skill whenever the user asks to create, design, or generate a PPT,
  poster, slide deck, presentation, social media graphic, infographic, flyer,
  banner, or any static visual content. Also triggers for: 做海报, 做PPT,
  做图, 小红书图文, 朋友圈海报, 设计幻灯片, make slides, design a poster,
  create a presentation, 做一个小红书帖子, 帮我做个 PPT.
---

# Design — AI SVG Design System

You are driving ppt-master, an AI-native multi-page SVG design system.
Your job: take user content and produce polished, export-ready visual pages.

## How This System Works

ppt-master produces SVG pages as the primary artifact. PPTX is an optional
export format. The system includes a browser-based editor so the user can
preview your output and make fine-grained adjustments (move, resize, recolor)
without describing every tweak in chat.

**Core loop**: Generate SVGs → User previews in editor → User tweaks visually
→ If major changes needed, user tells you → Finalize and export.

This loop eliminates the wasteful "move it left a bit / make it bluer" cycle.

---

## Step 0: Environment Check

Before anything else, confirm ppt-master is available:

```bash
PPT_MASTER=$(pwd)  # should be the ppt-master repo root
ls $PPT_MASTER/tools/finalize_svg.py && echo "OK"
```

If not found, tell the user to clone it:
```bash
git clone https://github.com/hugohe3/ppt-master.git
pip install -r ppt-master/requirements.txt
```

For the editor (optional but recommended for preview):
```bash
cd $PPT_MASTER/editor && bun install  # one-time setup
```

---

## Step 1: Understand the Request

Determine three things from the user's input:

### A. Canvas Format

Pick the format that matches the user's intent:

| Format code | Size | Use case |
|-------------|------|----------|
| `ppt169` | 1280×720 | PPT presentations (default) |
| `ppt43` | 1024×768 | Traditional projectors |
| `xiaohongshu` | 1242×1660 | 小红书 posts |
| `moments` | 1080×1080 | 朋友圈 / Instagram |
| `story` | 1080×1920 | Stories / 竖版 |
| `banner` | 1920×1080 | Web banners |
| `wechat` | 900×383 | 公众号头图 |
| `a4` | 1240×1754 | Print / A4 |

### B. Source Content

If the user provides a PDF, URL, or file — convert it immediately:

| Source | Command |
|--------|---------|
| PDF | `python3 $PPT_MASTER/tools/pdf_to_md.py <file>` |
| URL | `python3 $PPT_MASTER/tools/web_to_md.py <URL>` |
| WeChat/protected | `node $PPT_MASTER/tools/web_to_md.cjs <URL>` |

### C. Design Direction

Decide on: page count, color scheme, style (general / consultant / top-consulting).

For color and font schemes, read `references/colors.md` — it contains 16 color
presets (universal, mood, industry) and 6 font schemes synced from the editor.
Pick a scheme that matches the user's content or industry.

If the user's content is industry-specific (finance, medical, education, tech),
suggest the matching industry palette.

---

## Step 2: Create Project

```bash
python3 $PPT_MASTER/tools/project_manager.py init <project_name> --format <format_code>
```

This creates the project structure under `$PPT_MASTER/projects/<project_name>/`.

---

## Step 3: Design Planning

Before generating any SVG, write a design spec: `<project_path>/设计规范与内容大纲.md`

This document captures:
- Canvas format and dimensions
- Page list with page types (cover, content, chapter, ending)
- Color scheme (primary, secondary, accent — HEX values)
- Typography (font family, size hierarchy)
- Image strategy (user-provided / AI-generated / icons-only / none)
- Layout approach per page

Read the Strategist role for detailed guidance:
```
Read: $PPT_MASTER/roles/Strategist.md
```

### If AI images are needed

Read the Image Generator role and use the image generation tool:
```
Read: $PPT_MASTER/roles/Image_Generator.md
```

```bash
# Gemini (default, creative)
python3 $PPT_MASTER/tools/nano_banana_gen.py "prompt" --aspect_ratio 16:9 --image_size 2K -o <project_path>/images

# Kling (realistic, portraits)
python3 $PPT_MASTER/tools/nano_banana_gen.py "prompt" --engine kling --aspect_ratio 16:9 -o <project_path>/images

# Remove Gemini watermark
python3 $PPT_MASTER/tools/gemini_watermark_remover.py <image_path>
```

Requires env vars: `GEMINI_API_KEY` and `GEMINI_BASE_URL=https://sucloud.vip`

---

## Step 4: Generate SVG Pages

Read the appropriate Executor role before generating:

| Style | Role file |
|-------|-----------|
| General (flexible, creative) | `roles/Executor_General.md` |
| Consultant (business) | `roles/Executor_Consultant.md` |
| Top Consulting (MBB-level) | `roles/Executor_Consultant_Top.md` |

Generate each page as a standalone SVG file in `<project_path>/svg_output/`.

### SVG Rules (Non-negotiable)

These constraints ensure PPTX compatibility. Violating them breaks export.

**FORBIDDEN elements** — never use:
`clipPath`, `mask`, `<style>`, `foreignObject`, `textPath`, `<animate*>`,
`<script>`, `<iframe>`, `marker`, `<symbol>`+`<use>` combo

**FORBIDDEN attributes**:
`class`, `id`, `onclick`/`onload`/event handlers, `marker-end`

**FORBIDDEN patterns**:
- `rgba()` → use `fill-opacity` / `stroke-opacity` instead
- `<g opacity="...">` → set opacity on each child element individually
- `<image opacity="...">` → overlay a semi-transparent rect
- `@font-face`, external CSS, `@import`
- Arrow with `marker-end` → use `<polygon>` triangle instead

**REQUIRED**:
- `viewBox` must match canvas size (e.g., `viewBox="0 0 1280 720"`)
- Background: use `<rect>` covering full canvas
- Text wrapping: use `<tspan>` with manual line breaks
- Colors: HEX values only (no `rgb()`, no `rgba()`)
- Fonts: system fonts only (`system-ui`, `-apple-system`, `PingFang SC`, etc.)
- All styles must be inline (no `class`, no `<style>` block)

### Chart Templates

33 chart templates available in `$PPT_MASTER/templates/charts/`.
Copy a template SVG, then modify data and colors. For the full catalog,
read `references/charts.md`.

Quick picks:
- **KPI cards**: `kpi_cards.svg`
- **Bar chart**: `bar_chart.svg`
- **Line chart**: `line_chart.svg`
- **Pie / Donut**: `pie_chart.svg`, `donut_chart.svg`
- **Funnel**: `funnel_chart.svg`
- **Gantt**: `gantt_chart.svg`

### Icons

640+ vector icons in `$PPT_MASTER/templates/icons/`. Reference them in SVG
with `{{icon:icon_name}}` placeholder — `finalize_svg.py` will embed them.

### Speaker Notes

After generating all SVG pages, write speaker notes to:
`<project_path>/notes/total.md`

Format: one section per page with `## Page N: Title` headers.

---

## Step 5: Preview in Editor

After generating SVGs, start the editor and auto-load the project:

```bash
# 1. Start the editor (if not already running)
cd $PPT_MASTER/editor && bun run dev &

# 2. Build the SVG URL list for auto-load
#    Extract the project directory name and build /projects/ URLs
PROJECT_DIR=$(basename <project_path>)
SVG_LIST=$(ls <project_path>/svg_output/*.svg | xargs -I{} basename {} | sed "s|^|/projects/$PROJECT_DIR/svg_output/|" | tr '\n' ',' | sed 's/,$//')

# 3. Open in browser with auto-load
open "http://127.0.0.1:5173?svg=$SVG_LIST"
```

The editor auto-loads all project pages. Tell the user:
> Editor opened with your design. You can:
> - Click any element → Inspector panel to edit properties
> - Drag to move, handles to resize
> - Switch color/font presets (16 color schemes, 6 font schemes)
> - Navigate pages via the filmstrip at bottom
> - Press Preview mode for presentation view
>
> Color scheme switching has WCAG contrast protection — text that would become
> invisible on the new background is automatically substituted with a readable
> color. Dark themes (midnight, tech-neon) work correctly.
>
> When you're happy with the result, tell me and I'll finalize the export.
> For major layout changes, describe them in chat and I'll regenerate the SVG.

**How auto-load works**: The editor accepts `?svg=path1,path2,...` as a URL
parameter. The `/projects/` prefix is served by a Vite middleware plugin that
maps to the `projects/` directory at the repo root.

**Alternative**: The user can also drag-and-drop SVG files directly into the
editor window.

If the user requests changes through chat instead of the editor, modify the
SVG files directly and refresh the browser to see updates.

---

## Step 6: Finalize and Export

Run these commands in order. Do not skip or substitute any step.

```bash
# 1. Split speaker notes into per-page files
python3 $PPT_MASTER/tools/total_md_split.py <project_path>

# 2. SVG post-processing (icon embedding, image cropping, text flattening)
#    NEVER use `cp` instead of this — it does critical processing
python3 $PPT_MASTER/tools/finalize_svg.py <project_path>

# 3. Export to PPTX (reads from svg_final/, not svg_output/)
python3 $PPT_MASTER/tools/svg_to_pptx.py <project_path> -s final
```

### PPTX options

```bash
# With transition effects
python3 $PPT_MASTER/tools/svg_to_pptx.py <project_path> -s final --transition fade

# Auto-advance (5 seconds per slide)
python3 $PPT_MASTER/tools/svg_to_pptx.py <project_path> -s final -t fade --auto-advance 5

# Skip speaker notes embedding
python3 $PPT_MASTER/tools/svg_to_pptx.py <project_path> -s final --no-notes
```

Available transitions: `fade`, `push`, `wipe`, `split`, `reveal`, `cover`, `random`

---

## Quality Check (Optional)

```bash
python3 $PPT_MASTER/tools/svg_quality_checker.py <project_path>
```

For CRAP-principle optimization, read:
```
Read: $PPT_MASTER/roles/Optimizer_CRAP.md
```

If you optimize, re-run Step 6 (post-processing + export).

---

## Tool Reference

| Tool | Purpose |
|------|---------|
| `project_manager.py init` | Create project structure |
| `pdf_to_md.py` | Convert PDF to Markdown |
| `web_to_md.py` / `.cjs` | Convert URL to Markdown |
| `nano_banana_gen.py` | AI image generation (3 engines) |
| `gemini_watermark_remover.py` | Remove Gemini watermarks |
| `analyze_images.py` | Analyze image sizes for layout |
| `svg_quality_checker.py` | Validate SVG against constraints |
| `svg_position_calculator.py` | Calculate positions and chart data |
| `total_md_split.py` | Split notes into per-page files |
| `finalize_svg.py` | Post-process SVGs (icons, images, text) |
| `svg_to_pptx.py` | Export to PPTX |
| `config.py list-formats` | Show available canvas formats |

---

## Common Mistakes to Avoid

| Mistake | Correct approach |
|---------|-----------------|
| Using `cp` to copy SVGs to svg_final/ | Always use `finalize_svg.py` |
| Exporting from svg_output/ | Use `-s final` to export from svg_final/ |
| Forgetting to split notes | Run `total_md_split.py` before finalize |
| Using `clipPath` or `mask` | Use simple shapes and opacity |
| Using `rgba()` colors | Use HEX + `fill-opacity` |
| Using `<style>` blocks | Use inline styles only |
| Using `class` or `id` attributes | Remove them entirely |
| Skipping post-processing | Always run finalize before export |

---

## Reference Files

For detailed information, read these files as needed:

- `references/charts.md` — Full 33-chart catalog with usage guidance
- `references/colors.md` — All color schemes and industry palettes
- `roles/Strategist.md` — Design planning methodology
- `roles/Executor_General.md` — General style execution guide
- `roles/Executor_Consultant.md` — Consultant style execution guide
- `roles/Executor_Consultant_Top.md` — MBB-level execution guide
- `roles/Optimizer_CRAP.md` — CRAP-principle visual optimization
- `docs/canvas_formats.md` — Detailed format specs and layout rules
