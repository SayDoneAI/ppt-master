---
name: poster
description: "SVG-based visual content generator — PPT, posters, social media images. Powered by ppt-master engine. Use when the user wants to create presentations, posters, social media graphics (WeChat Moments, Xiaohongshu, Stories, Banners, etc.)."
---

# Poster — SVG Visual Content Generator

Powered by [ppt-master](https://github.com/hugohe3/ppt-master) engine.

## Preamble（首次运行时自动安装 skill + command 链接）

```bash
# 自动 symlink skill 和 command 到 ~/.claude/（幂等，已存在则跳过）
POSTER_SKILL="$(cd "$(dirname "$(readlink -f ~/.claude/skills/poster/SKILL.md 2>/dev/null || echo ~/.claude/skills/poster/SKILL.md)")" && pwd)"
PPT_MASTER_ROOT="$(cd "$POSTER_SKILL/../../.." && pwd)"

# skill symlink
[ -d "$POSTER_SKILL" ] && [ ! -L ~/.claude/skills/poster ] && [ -d ~/.claude/skills ] && ln -sf "$POSTER_SKILL" ~/.claude/skills/poster 2>/dev/null

# command symlink
[ -f "$PPT_MASTER_ROOT/.claude/commands/poster.md" ] && [ ! -L ~/.claude/commands/poster.md ] && mkdir -p ~/.claude/commands && ln -sf "$PPT_MASTER_ROOT/.claude/commands/poster.md" ~/.claude/commands/poster.md 2>/dev/null

[ -L ~/.claude/skills/poster ] && echo "SKILL: OK" || echo "SKILL: manual link needed"
[ -L ~/.claude/commands/poster.md ] && echo "COMMAND: OK" || echo "COMMAND: manual link needed"
```

## Step 0: Locate Repo (run once, cached)

Before any operation, resolve `$PPT_MASTER`:

```bash
CONFIG_FILE=~/.config/poster/repo_path
if [ -f "$CONFIG_FILE" ] && [ -d "$(cat "$CONFIG_FILE")/.git" ]; then
  PPT_MASTER=$(cat "$CONFIG_FILE")
else
  rm -f "$CONFIG_FILE"
  # Auto-detect: search common locations
  PPT_MASTER=""
  for candidate in \
    ~/Documents/RedCode/ppt-master \
    ~/Documents/ppt-master \
    ~/projects/ppt-master \
    ~/code/ppt-master \
    ~/dev/ppt-master \
    ~/ppt-master; do
    if [ -d "$candidate/.git" ]; then
      PPT_MASTER="$candidate"
      break
    fi
  done
  # Fallback: find by repo name (max 5s)
  if [ -z "$PPT_MASTER" ]; then
    PPT_MASTER=$(find ~ -maxdepth 5 -type d -name "ppt-master" -exec test -d "{}/.git" \; -print -quit 2>/dev/null)
  fi
  if [ -z "$PPT_MASTER" ]; then
    echo "ERROR: ppt-master repo not found. Clone it first: git clone <repo-url>"
    exit 1
  fi
  mkdir -p ~/.config/poster
  echo "$PPT_MASTER" > "$CONFIG_FILE"
  echo "Cached ppt-master path: $PPT_MASTER"
fi
echo "PPT_MASTER=$PPT_MASTER"
```

## Supported Formats

| Format | Code | Dimensions | Use Case |
|--------|------|-----------|----------|
| PPT 16:9 | `ppt169` | 1280x720 | Presentations, reports |
| PPT 4:3 | `ppt43` | 1024x768 | Traditional projectors |
| WeChat Moments | `moments` | 1080x1080 | Social media square images |
| Xiaohongshu | `xiaohongshu` / `xhs` | 1242x1660 | Knowledge sharing, product reviews |
| Story | `story` | 1080x1920 | TikTok/Instagram Stories |
| WeChat Header | `wechat` | 900x383 | WeChat article headers |
| Banner | `banner` | 1920x1080 | Web banners, large displays |
| A4 Print | `a4` | 1240x1754 | Print documents |

## Execution Workflow

**CRITICAL: You MUST follow every step below in order. Do NOT skip any step.**

### Step 1: Read ppt-master docs

Before doing anything, read the workflow guide:

```text
Read file: $PPT_MASTER/AGENTS.md
```

### Step 2: Source Content Processing

If the user provides a PDF, URL, or file, convert it immediately:

| Source | Command |
|--------|---------|
| PDF | `python3 $PPT_MASTER/tools/pdf_to_md.py <file>` |
| URL | `python3 $PPT_MASTER/tools/web_to_md.py <URL>` |
| WeChat/protected URL | `node $PPT_MASTER/tools/web_to_md.cjs <URL>` |
| Plain text / Markdown | Use directly |

### Step 3: Create Project

```bash
cd $PPT_MASTER && python3 tools/project_manager.py init <project_name> --format <format_code>
```

The project will be created at `$PPT_MASTER/projects/<name>_<format>_<date>/`.

### Step 4: Template Selection

Ask the user:
- **A) Use existing template** - Copy from `$PPT_MASTER/templates/layouts/` to project
- **B) No template** - Free design

Available templates: `anthropic`, `consultant`, `consultant_top`, `general`, `google_style`, `mckinsey`, `smart_red`, `pixel_retro`, etc.

If using a template:

```bash
cp $PPT_MASTER/templates/layouts/<template_name>/*.svg <project_path>/templates/
cp $PPT_MASTER/templates/layouts/<template_name>/design_spec.md <project_path>/templates/
cp $PPT_MASTER/templates/layouts/<template_name>/*.png <project_path>/images/ 2>/dev/null || true
cp $PPT_MASTER/templates/layouts/<template_name>/*.jpg <project_path>/images/ 2>/dev/null || true
```

### Step 5: Strategist Role (MANDATORY)

**Read the role definition first:**

```text
Read file: $PPT_MASTER/roles/Strategist.md
```

Complete the **8-item confirmation** with the user, providing professional recommendations for each:

1. **Canvas format** - Recommend based on use case
2. **Page count** - Suggest based on content volume
3. **Target audience & context** - Provide initial assessment
4. **Design style** - A) General flexible B) Consultant C) Top consulting (MBB-level)
5. **Color scheme** - Provide HEX values (primary/secondary/accent)
6. **Icon method** - A) Emoji B) AI-generated C) Built-in library (640+ icons) D) Custom
7. **Image usage** - A) None B) User-provided C) AI-generated D) Placeholders
8. **Typography** - Font combination + base body text size (18-24px)

Output the **Design Specification & Content Outline** document, save to `<project_path>/design_spec.md`.

### Step 6: Image Generation (if needed)

Only if image method includes "C) AI-generated":

**Read the role definition:**

```text
Read file: $PPT_MASTER/roles/Image_Generator.md
```

Generate image prompts and save to `<project_path>/images/image_prompts.md`.

### Step 7: Executor Role - Generate SVG

**Read the appropriate role definition based on style:**

```text
Read file: $PPT_MASTER/roles/Executor_General.md        # General style
Read file: $PPT_MASTER/roles/Executor_Consultant.md     # Consultant style
Read file: $PPT_MASTER/roles/Executor_Consultant_Top.md # Top consulting style
```

**Phase 1 - Visual Construction**: Generate all SVG pages, save to `<project_path>/svg_output/`.

**Phase 2 - Logic Construction**: Generate speaker notes, save to `<project_path>/notes/total.md`.

### SVG Technical Constraints (NON-NEGOTIABLE)

**BANNED features**: `clipPath` | `mask` | `<style>` | `class/id` | external CSS | `<foreignObject>` | `textPath` | `@font-face` | `<animate*>` | `<script>` | `marker-end` | `<iframe>` | `<symbol>+<use>`

**PPT compatibility substitutions**:
- `rgba()` -> use `fill-opacity` / `stroke-opacity`
- `<g opacity>` -> set opacity on each child element individually
- `<image opacity>` -> overlay mask rectangle
- `marker-end` arrows -> `<polygon>` triangles

**Base rules**: viewBox must match canvas dimensions, use `<rect>` for backgrounds, use `<tspan>` for line breaks, only system fonts and inline styles.

### Step 8: Post-Processing & Export (3 commands, in order)

```bash
# 1. Split speaker notes
python3 $PPT_MASTER/tools/total_md_split.py <project_path>

# 2. SVG post-processing (icon embedding, image cropping, text flattening)
python3 $PPT_MASTER/tools/finalize_svg.py <project_path>

# 3. Export to PPTX
python3 $PPT_MASTER/tools/svg_to_pptx.py <project_path> -s final
```

**NEVER use `cp` to copy SVG to svg_final/ - always use `finalize_svg.py`.**
**NEVER export from `svg_output/` - always use `-s final` from `svg_final/`.**

### Step 9: Optional Optimization

If quality needs improvement:

```text
Read file: $PPT_MASTER/roles/Optimizer_CRAP.md
```

Apply CRAP principles (Contrast, Repetition, Alignment, Proximity), then re-run Step 8.

## Utility Commands

```bash
# Validate project structure
python3 $PPT_MASTER/tools/project_manager.py validate <project_path>

# SVG quality check
python3 $PPT_MASTER/tools/svg_quality_checker.py <project_path>

# Local preview
python3 -m http.server -d <project_path>/svg_final 8000

# Project info
python3 $PPT_MASTER/tools/project_manager.py info <project_path>
```

## Role Switching Protocol

When switching roles, MUST output:

```markdown
---
## [Role Switch: <Role Name>]
Reading role definition: roles/<filename>.md
Current task: <brief description>
---
```

And after each phase, output a checkpoint checklist confirming completion.
