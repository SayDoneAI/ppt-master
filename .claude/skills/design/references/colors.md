# Color & Font Presets

> Source of truth: `editor/src/presets/colors.ts` and `editor/src/presets/fonts.ts`.
> The editor's preset switcher applies these schemes to all pages at once.

## Shared Defaults

Most schemes share these text and background values:

| Role | HEX | Usage |
|------|-----|-------|
| `textDark` | `#1A1A2E` | Headings and body on light backgrounds |
| `textLight` | `#FFFFFF` | Text on dark or colored backgrounds |
| `textMuted` | `#6B7280` | Captions, footnotes, secondary info |
| `background` | `#FFFFFF` | Page background |
| `backgroundAlt` | `#F5F5F5` | Card / section background |

---

## Universal Schemes (5)

General-purpose palettes that work for any content type.

| ID | Name | Primary | Secondary | Accent |
|----|------|---------|-----------|--------|
| `classic-blue` | 经典蓝 | `#2563EB` | `#3B82F6` | `#F59E0B` |
| `ocean` | 海洋 | `#0891B2` | `#06B6D4` | `#F97316` |
| `forest` | 森林 | `#059669` | `#10B981` | `#EF4444` |
| `slate` | 石墨 | `#334155` | `#64748B` | `#3B82F6` |
| `midnight` | 午夜 | `#1E293B` | `#475569` | `#22D3EE` |

**`midnight` overrides**: textDark `#F1F5F9`, textMuted `#94A3B8`, background `#0F172A`, backgroundAlt `#1E293B` (dark theme).

---

## Mood Schemes (6)

Emotion-driven palettes for specific visual tones.

| ID | Name | Primary | Secondary | Accent |
|----|------|---------|-----------|--------|
| `warm-earth` | 暖棕 | `#92400E` | `#B45309` | `#DC2626` |
| `rose-gold` | 玫瑰金 | `#BE185D` | `#DB2777` | `#7C3AED` |
| `lavender` | 薰衣草 | `#7C3AED` | `#8B5CF6` | `#EC4899` |
| `sunset` | 日落 | `#DC2626` | `#EA580C` | `#2563EB` |
| `mint` | 薄荷 | `#0D9488` | `#14B8A6` | `#F59E0B` |
| `sakura` | 樱花 | `#EC4899` | `#F472B6` | `#8B5CF6` |

---

## Industry Schemes (5)

Tailored palettes that match industry visual expectations.

| ID | Name | Primary | Secondary | Accent |
|----|------|---------|-----------|--------|
| `tech-neon` | 科技 | `#2563EB` | `#7C3AED` | `#06B6D4` |
| `finance` | 金融 | `#1E3A5F` | `#2563EB` | `#D97706` |
| `medical` | 医疗 | `#047857` | `#10B981` | `#3B82F6` |
| `education` | 教育 | `#7C3AED` | `#8B5CF6` | `#F59E0B` |
| `creative` | 创意 | `#DB2777` | `#EC4899` | `#F59E0B` |

---

## Picking a Scheme

1. **Match the industry** — if the user's content clearly belongs to finance, medical, education, or tech, start with the matching industry scheme.
2. **Match the mood** — if the user describes a tone ("warm", "elegant", "playful"), pick the closest mood scheme.
3. **Default to universal** — `classic-blue` is a safe default for business content; `ocean` for casual; `forest` for sustainability or growth narratives.
4. **Dark themes** — only `midnight` has a dark background. Use it when the user explicitly wants a dark look. The editor's color scheme switcher includes WCAG contrast protection: text elements that would become invisible on the new background are automatically replaced with a readable color (textDark or textLight).

---

## Font Schemes (6)

| ID | Name | Title Font | Body Font |
|----|------|------------|-----------|
| `noto-sans` | Noto Sans | `"Noto Sans SC", sans-serif` | `"Noto Sans SC", sans-serif` |
| `source-han` | 思源黑体 | `"Source Han Sans SC", "Noto Sans SC", sans-serif` | `"Source Han Sans SC", "Noto Sans SC", sans-serif` |
| `source-han-serif` | 思源宋体 | `"Source Han Serif SC", "Noto Serif SC", serif` | `"Source Han Serif SC", "Noto Serif SC", serif` |
| `alibaba-puhuiti` | 阿里巴巴普惠体 | `"Alibaba PuHuiTi", "PingFang SC", sans-serif` | `"Alibaba PuHuiTi", "PingFang SC", sans-serif` |
| `oppo-sans` | OPPO Sans | `"OPPO Sans", "PingFang SC", sans-serif` | `"OPPO Sans", "PingFang SC", sans-serif` |
| `harmonyos-sans` | HarmonyOS Sans | `"HarmonyOS Sans SC", "PingFang SC", sans-serif` | `"HarmonyOS Sans SC", "PingFang SC", sans-serif` |

**Caption font**: all schemes use `"PingFang SC", "Microsoft YaHei", sans-serif` for captions and labels.

### Font Size Hierarchy

| Level | Size | Usage |
|-------|------|-------|
| `title_large` | 48px | Hero titles |
| `title` | 36px | Page titles |
| `heading` | 24px | Section headings |
| `body` | 18px | Body text |
| `caption` | 14px | Captions, labels |
| `footnote` | 12px | Footnotes, page numbers |

### Picking a Font Scheme

- **Default**: `noto-sans` — widest compatibility, clean look.
- **Business/formal**: `source-han` or `source-han-serif` (serif for traditional/literary tone).
- **Modern/trendy**: `alibaba-puhuiti` or `oppo-sans`.
- **Tech products**: `harmonyos-sans`.
- All fonts fall back to system fonts (`PingFang SC`, `Microsoft YaHei`), so SVGs render correctly even without the web font installed.
