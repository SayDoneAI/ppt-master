// state_to_svg.ts — 将 compat slide_state 转为符合 ppt-master 约束的 SVG
//
// 文本元素通过 Pretext layoutWithLines() 精确断行
// 输出的 SVG 可供 editor / bridge compat render，并兼容 finalize_svg.py → svg_to_pptx.py

import type {
  SlideState, Slide, Element, TextElement, RectElement,
  PathElement, LineElement, CircleElement, ImageElement,
  GroupElement, Def, LinearGradientDef, FilterDef,
} from './slide_state.js'

// Pretext 类型（运行时导入，开发时可选）
type PretextModule = typeof import('@chenglou/pretext')
let pretext: PretextModule | null = null

/**
 * 初始化 Pretext（浏览器环境自动加载）
 * 服务端/CLI 环境需手动调用此函数传入 pretext 模块
 */
export function initPretext(mod: PretextModule): void {
  pretext = mod
}

// ============================================================
// 主入口
// ============================================================

/**
 * 将一个 Slide 转为完整的 SVG 字符串
 */
export function slideToSvg(slide: Slide, canvas: { width: number; height: number }): string {
  const lines: string[] = []
  lines.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${canvas.width} ${canvas.height}" width="${canvas.width}" height="${canvas.height}">`)

  // defs
  if (slide.defs && slide.defs.length > 0) {
    lines.push('  <defs>')
    for (const def of slide.defs) {
      lines.push(defToSvg(def, '    '))
    }
    lines.push('  </defs>')
  }

  // background
  if (slide.background) {
    lines.push(`  <rect width="${canvas.width}" height="${canvas.height}" fill="${escapeAttr(slide.background)}" />`)
  }

  // elements
  for (const el of slide.elements) {
    lines.push(elementToSvg(el, '  '))
  }

  lines.push('</svg>')
  return lines.join('\n')
}

/**
 * 将完整的兼容 SlideState 转为 SVG 字符串数组（每个 slide 一个 SVG）
 */
export function stateToSvgs(state: SlideState): string[] {
  return state.slides.map(slide => slideToSvg(slide, state.canvas))
}

// ============================================================
// Element → SVG
// ============================================================

function elementToSvg(el: Element, indent: string): string {
  switch (el.type) {
    case 'text': return textToSvg(el, indent)
    case 'rect': return rectToSvg(el, indent)
    case 'path': return pathToSvg(el, indent)
    case 'line': return lineToSvg(el, indent)
    case 'circle': return circleToSvg(el, indent)
    case 'image': return imageToSvg(el, indent)
    case 'group': return groupToSvg(el, indent)
    default: return `${indent}<!-- unknown element type: ${(el as Element).type} -->`
  }
}

function buildElementDataAttr(id: string): string {
  return ` data-element-id="${escapeAttr(id)}"`
}

// ============================================================
// Text — Pretext 驱动断行
// ============================================================

/** 从 font shorthand 解析 font-size 和 font-family */
function parseFontShorthand(font: string): { fontSize: number; fontFamily: string; fontWeight?: string; fontStyle?: string } {
  // 格式: "bold italic 36px PingFang SC" 或 "16px Arial" 等
  const match = font.match(/(?:(italic)\s+)?(?:(bold|[1-9]00)\s+)?(\d+(?:\.\d+)?)px\s+(.+)/i)
  if (match) {
    return {
      fontStyle: match[1] || undefined,
      fontWeight: match[2] || undefined,
      fontSize: parseFloat(match[3]),
      fontFamily: match[4],
    }
  }
  // fallback
  return { fontSize: 16, fontFamily: 'Arial, sans-serif' }
}

export interface TextLayoutResult {
  lines: Array<{ text: string; width: number }>
  height: number
  lineCount: number
  overflow: boolean
}

/**
 * 使用 Pretext 对文本进行排版计算
 * 如果 Pretext 未加载，fallback 到简单按字符估算
 */
export function layoutText(el: TextElement): TextLayoutResult {
  if (pretext) {
    const prepared = pretext.prepareWithSegments(el.text, el.font)
    const result = pretext.layoutWithLines(prepared, el.width, el.lineHeight)
    const overflow = el.maxHeight ? result.height > el.maxHeight : false
    return {
      lines: result.lines.map(l => ({ text: l.text, width: l.width })),
      height: result.height,
      lineCount: result.lineCount,
      overflow,
    }
  }
  // Fallback: 粗略按字符宽度估算断行
  return fallbackLayout(el)
}

function fallbackLayout(el: TextElement): TextLayoutResult {
  const parsed = parseFontShorthand(el.font)
  const charWidth = parsed.fontSize * 0.6 // 粗略估算
  const charsPerLine = Math.max(1, Math.floor(el.width / charWidth))
  const lines: Array<{ text: string; width: number }> = []

  let remaining = el.text
  while (remaining.length > 0) {
    const cut = remaining.length <= charsPerLine ? remaining.length : findBreakPoint(remaining, charsPerLine)
    const lineText = remaining.substring(0, cut).trimEnd()
    lines.push({ text: lineText, width: lineText.length * charWidth })
    remaining = remaining.substring(cut).trimStart()
  }

  const height = lines.length * el.lineHeight
  const overflow = el.maxHeight ? height > el.maxHeight : false
  return { lines, height, lineCount: lines.length, overflow }
}

function findBreakPoint(text: string, maxChars: number): number {
  if (maxChars >= text.length) return text.length
  // 往回找空格或标点断点
  for (let i = maxChars; i > maxChars * 0.5; i--) {
    const ch = text[i]
    if (ch === ' ' || ch === '，' || ch === '。' || ch === '、' || ch === '；') {
      return i + 1
    }
  }
  return maxChars
}

function textToSvg(el: TextElement, indent: string): string {
  const parsed = parseFontShorthand(el.font)
  const fontFamily = el.fontFamily || parsed.fontFamily
  const fontSize = el.fontSize || parsed.fontSize
  const fontWeight = el.fontWeight || parsed.fontWeight

  const layout = layoutText(el)

  // 单行文本：直接输出 <text>
  if (layout.lineCount <= 1) {
    const attrs = buildTextAttrs(el, fontFamily, fontSize, fontWeight)
    const text = layout.lines[0]?.text ?? el.text
    return `${indent}<text${buildElementDataAttr(el.id)}${attrs}>${escapeXml(text)}</text>`
  }

  // 多行文本：每行一个独立的 <text>（ppt-master 惯例，兼容性最好）
  const lines: string[] = []
  for (let i = 0; i < layout.lines.length; i++) {
    const lineY = el.y + el.lineHeight * i
    const attrs = buildTextAttrs({ ...el, y: lineY }, fontFamily, fontSize, fontWeight)
    lines.push(`${indent}<text${buildElementDataAttr(el.id)}${attrs}>${escapeXml(layout.lines[i].text)}</text>`)
  }
  return lines.join('\n')
}

function buildTextAttrs(
  el: Partial<TextElement> & { x: number; y: number; fill: string },
  fontFamily: string,
  fontSize: number,
  fontWeight?: string | number,
): string {
  let attrs = ` x="${el.x}" y="${el.y}"`
  attrs += ` font-family="${escapeAttr(fontFamily)}" font-size="${fontSize}"`
  if (fontWeight) attrs += ` font-weight="${fontWeight}"`
  attrs += ` fill="${escapeAttr(el.fill)}"`
  if (el.textAnchor) attrs += ` text-anchor="${el.textAnchor}"`
  if (el.opacity !== undefined && el.opacity !== 1) attrs += ` opacity="${el.opacity}"`
  if (el.fillOpacity !== undefined) attrs += ` fill-opacity="${el.fillOpacity}"`
  if (el.letterSpacing) attrs += ` letter-spacing="${el.letterSpacing}"`
  return attrs
}

// ============================================================
// Rect
// ============================================================

function rectToSvg(el: RectElement, indent: string): string {
  let attrs = ` x="${el.x}" y="${el.y}" width="${el.width}" height="${el.height}"`
  if (el.fill) attrs += ` fill="${escapeAttr(el.fill)}"`
  if (el.stroke) attrs += ` stroke="${escapeAttr(el.stroke)}"`
  if (el.strokeWidth) attrs += ` stroke-width="${el.strokeWidth}"`
  if (el.rx) attrs += ` rx="${el.rx}"`
  if (el.ry) attrs += ` ry="${el.ry}"`
  if (el.opacity !== undefined && el.opacity !== 1) attrs += ` opacity="${el.opacity}"`
  if (el.fillOpacity !== undefined) attrs += ` fill-opacity="${el.fillOpacity}"`
  return `${indent}<rect${buildElementDataAttr(el.id)}${attrs} />`
}

// ============================================================
// Path
// ============================================================

function pathToSvg(el: PathElement, indent: string): string {
  let attrs = ` d="${escapeAttr(el.d)}"`
  if (el.fill) attrs += ` fill="${escapeAttr(el.fill)}"`
  if (el.stroke) attrs += ` stroke="${escapeAttr(el.stroke)}"`
  if (el.strokeWidth) attrs += ` stroke-width="${el.strokeWidth}"`
  if (el.fillOpacity !== undefined) attrs += ` fill-opacity="${el.fillOpacity}"`
  if (el.fillRule) attrs += ` fill-rule="${el.fillRule}"`
  if (el.clipRule) attrs += ` clip-rule="${el.clipRule}"`
  if (el.opacity !== undefined && el.opacity !== 1) attrs += ` opacity="${el.opacity}"`
  return `${indent}<path${buildElementDataAttr(el.id)}${attrs} />`
}

// ============================================================
// Line
// ============================================================

function lineToSvg(el: LineElement, indent: string): string {
  let attrs = ` x1="${el.x1}" y1="${el.y1}" x2="${el.x2}" y2="${el.y2}"`
  attrs += ` stroke="${escapeAttr(el.stroke)}"`
  if (el.strokeWidth) attrs += ` stroke-width="${el.strokeWidth}"`
  if (el.opacity !== undefined && el.opacity !== 1) attrs += ` opacity="${el.opacity}"`
  if (el.strokeOpacity !== undefined) attrs += ` stroke-opacity="${el.strokeOpacity}"`
  return `${indent}<line${buildElementDataAttr(el.id)}${attrs} />`
}

// ============================================================
// Circle
// ============================================================

function circleToSvg(el: CircleElement, indent: string): string {
  let attrs = ` cx="${el.cx}" cy="${el.cy}" r="${el.r}"`
  if (el.fill) attrs += ` fill="${escapeAttr(el.fill)}"`
  if (el.stroke) attrs += ` stroke="${escapeAttr(el.stroke)}"`
  if (el.strokeWidth) attrs += ` stroke-width="${el.strokeWidth}"`
  if (el.fillOpacity !== undefined) attrs += ` fill-opacity="${el.fillOpacity}"`
  if (el.opacity !== undefined && el.opacity !== 1) attrs += ` opacity="${el.opacity}"`
  return `${indent}<circle${buildElementDataAttr(el.id)}${attrs} />`
}

// ============================================================
// Image
// ============================================================

function imageToSvg(el: ImageElement, indent: string): string {
  let attrs = ` href="${escapeAttr(el.href)}" x="${el.x}" y="${el.y}" width="${el.width}" height="${el.height}"`
  if (el.preserveAspectRatio) attrs += ` preserveAspectRatio="${el.preserveAspectRatio}"`
  if (el.opacity !== undefined && el.opacity !== 1) attrs += ` opacity="${el.opacity}"`
  return `${indent}<image${buildElementDataAttr(el.id)}${attrs} />`
}

// ============================================================
// Group
// ============================================================

function groupToSvg(el: GroupElement, indent: string): string {
  let attrs = ''
  if (el.transform) attrs += ` transform="${escapeAttr(el.transform)}"`
  if (el.fill) attrs += ` fill="${escapeAttr(el.fill)}"`
  if (el.fontFamily) attrs += ` font-family="${escapeAttr(el.fontFamily)}"`
  if (el.fontSize) attrs += ` font-size="${el.fontSize}"`
  if (el.fontWeight) attrs += ` font-weight="${el.fontWeight}"`
  if (el.filter) attrs += ` filter="${escapeAttr(el.filter)}"`
  if (el.opacity !== undefined && el.opacity !== 1) attrs += ` opacity="${el.opacity}"`

  const lines: string[] = []
  lines.push(`${indent}<g${buildElementDataAttr(el.id)}${attrs}>`)
  for (const child of el.children) {
    lines.push(elementToSvg(child, indent + '  '))
  }
  lines.push(`${indent}</g>`)
  return lines.join('\n')
}

// ============================================================
// Defs
// ============================================================

function defToSvg(def: Def, indent: string): string {
  switch (def.type) {
    case 'linearGradient': return linearGradientToSvg(def, indent)
    case 'filter': return filterToSvg(def, indent)
    default: return ''
  }
}

function linearGradientToSvg(def: LinearGradientDef, indent: string): string {
  const lines: string[] = []
  lines.push(`${indent}<linearGradient id="${escapeAttr(def.id)}" x1="${def.x1}" y1="${def.y1}" x2="${def.x2}" y2="${def.y2}">`)
  for (const stop of def.stops) {
    let attrs = ` offset="${stop.offset}" stop-color="${escapeAttr(stop.color)}"`
    if (stop.opacity !== undefined) attrs += ` stop-opacity="${stop.opacity}"`
    lines.push(`${indent}  <stop${attrs} />`)
  }
  lines.push(`${indent}</linearGradient>`)
  return lines.join('\n')
}

function filterToSvg(def: FilterDef, indent: string): string {
  // filter 结构复杂，暂保留原始 SVG
  return `${indent}${def.rawSvg}`
}

// ============================================================
// Utilities
// ============================================================

function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function escapeAttr(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}
