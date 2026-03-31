// svg_to_state.ts — 将现有 SVG 文件解析为 slide_state
//
// 兼容 ppt-master 现有的 15 个示例项目
// 处理真实 SVG 中的各种写法：多个 <text> 模拟多行、<path> 圆角矩形、
// <g> 分组嵌套、linearGradient/filter defs 等

import type {
  Slide, Element, TextElement, RectElement, PathElement,
  LineElement, CircleElement, ImageElement, GroupElement,
  Canvas, SlideState, Def, LinearGradientDef, FilterDef,
} from './slide_state.js'

// ============================================================
// 主入口
// ============================================================

/**
 * 解析 SVG 字符串为 Slide
 * 使用 DOMParser（浏览器）或传入的解析器（Node/Bun）
 */
export function svgToSlide(svgString: string, slideId: string, parser?: DOMParser): Slide {
  const p = parser ?? new DOMParser()
  const doc = p.parseFromString(svgString, 'image/svg+xml')
  const svgEl = doc.documentElement

  // 解析 canvas 尺寸
  const viewBox = svgEl.getAttribute('viewBox')
  const canvas = parseViewBox(viewBox, svgEl)

  // 解析 defs
  const defsEl = svgEl.querySelector('defs')
  const defs: Def[] = defsEl ? parseDefs(defsEl) : []

  // 解析子元素（跳过 defs）
  const elements: Element[] = []
  let idCounter = 0
  const genId = (prefix: string) => `${prefix}_${++idCounter}`

  for (const child of Array.from(svgEl.children)) {
    if (child.tagName === 'defs') continue
    const el = parseElement(child as SVGElement, genId)
    if (el) elements.push(el)
  }

  // 合并相邻的多行 <text>（ppt-master 用多个 <text> 模拟多行文本）
  const merged = mergeAdjacentTexts(elements)

  return {
    id: slideId,
    elements: merged,
    defs: defs.length > 0 ? defs : undefined,
  }
}

/**
 * 解析多个 SVG 字符串为完整的 SlideState
 */
export function svgsToState(
  svgStrings: string[],
  slideIds?: string[],
  parser?: DOMParser,
): SlideState {
  if (svgStrings.length === 0) {
    return { canvas: { width: 1280, height: 720 }, slides: [] }
  }

  // 从第一个 SVG 获取 canvas 尺寸
  const p = parser ?? new DOMParser()
  const firstDoc = p.parseFromString(svgStrings[0], 'image/svg+xml')
  const canvas = parseViewBox(
    firstDoc.documentElement.getAttribute('viewBox'),
    firstDoc.documentElement,
  )

  const slides = svgStrings.map((svg, i) => {
    const id = slideIds?.[i] ?? `slide_${String(i + 1).padStart(2, '0')}`
    return svgToSlide(svg, id, p)
  })

  return { canvas, slides }
}

// ============================================================
// ViewBox 解析
// ============================================================

function parseViewBox(viewBox: string | null, svgEl: globalThis.Element): Canvas {
  if (viewBox) {
    const parts = viewBox.split(/[\s,]+/).map(Number)
    if (parts.length >= 4) {
      return { width: parts[2], height: parts[3] }
    }
  }
  return {
    width: parseFloat(svgEl.getAttribute('width') || '1280'),
    height: parseFloat(svgEl.getAttribute('height') || '720'),
  }
}

// ============================================================
// Defs 解析
// ============================================================

function parseDefs(defsEl: globalThis.Element): Def[] {
  const defs: Def[] = []
  for (const child of Array.from(defsEl.children)) {
    if (child.tagName === 'linearGradient') {
      defs.push(parseLinearGradient(child))
    } else if (child.tagName === 'filter') {
      defs.push(parseFilter(child))
    }
  }
  return defs
}

function parseLinearGradient(el: globalThis.Element): LinearGradientDef {
  const stops = Array.from(el.querySelectorAll('stop')).map(s => {
    // stop-color 可能在 style 属性里
    let color = s.getAttribute('stop-color') || ''
    const style = s.getAttribute('style') || ''
    const styleMatch = style.match(/stop-color:\s*([^;]+)/)
    if (styleMatch) color = styleMatch[1].trim()

    return {
      offset: s.getAttribute('offset') || '0%',
      color,
      opacity: s.hasAttribute('stop-opacity') ? parseFloat(s.getAttribute('stop-opacity')!) : undefined,
    }
  })

  return {
    type: 'linearGradient',
    id: el.getAttribute('id') || '',
    x1: el.getAttribute('x1') || '0%',
    y1: el.getAttribute('y1') || '0%',
    x2: el.getAttribute('x2') || '100%',
    y2: el.getAttribute('y2') || '0%',
    stops,
  }
}

function parseFilter(el: globalThis.Element): FilterDef {
  return {
    type: 'filter',
    id: el.getAttribute('id') || '',
    rawSvg: el.outerHTML,
  }
}

// ============================================================
// Element 解析
// ============================================================

function parseElement(el: SVGElement, genId: (prefix: string) => string): Element | null {
  const tag = el.tagName.toLowerCase()

  switch (tag) {
    case 'text': return parseText(el, genId)
    case 'rect': return parseRect(el, genId)
    case 'path': return parsePath(el, genId)
    case 'line': return parseLine(el, genId)
    case 'circle': return parseCircle(el, genId)
    case 'image': return parseImage(el, genId)
    case 'g': return parseGroup(el, genId)
    default: return null
  }
}

// ============================================================
// Text 解析
// ============================================================

function parseText(el: SVGElement, genId: (prefix: string) => string): TextElement {
  const tspans = el.querySelectorAll('tspan')
  let text: string

  if (tspans.length > 0) {
    // 有 tspan 的文本：合并所有 tspan 文本
    text = Array.from(tspans).map(ts => ts.textContent?.trim() || '').join('\n')
  } else {
    text = (el.textContent || '').trim()
  }

  const fontSize = parseFloat(el.getAttribute('font-size') || '16')
  const fontFamily = el.getAttribute('font-family') || 'Arial, sans-serif'
  const fontWeight = el.getAttribute('font-weight') || undefined

  // 构建 font shorthand
  let font = ''
  if (fontWeight && fontWeight !== 'normal') font += `${fontWeight} `
  font += `${fontSize}px ${fontFamily}`

  // 估算文本框宽度（后续 Pretext 可以精确计算）
  const x = parseFloat(el.getAttribute('x') || '0')
  const y = parseFloat(el.getAttribute('y') || '0')

  return {
    type: 'text',
    id: genId('text'),
    x,
    y,
    width: 1200, // 默认宽度，后续可优化
    text,
    font,
    lineHeight: Math.round(fontSize * 1.4),
    fill: el.getAttribute('fill') || '#000000',
    fontFamily,
    fontSize,
    fontWeight,
    textAnchor: (el.getAttribute('text-anchor') as TextElement['textAnchor']) || undefined,
    letterSpacing: el.hasAttribute('letter-spacing') ? parseFloat(el.getAttribute('letter-spacing')!) : undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
    fillOpacity: el.hasAttribute('fill-opacity') ? parseFloat(el.getAttribute('fill-opacity')!) : undefined,
  }
}

// ============================================================
// Rect 解析
// ============================================================

function parseRect(el: SVGElement, genId: (prefix: string) => string): RectElement {
  return {
    type: 'rect',
    id: genId('rect'),
    x: parseFloat(el.getAttribute('x') || '0'),
    y: parseFloat(el.getAttribute('y') || '0'),
    width: parseFloat(el.getAttribute('width') || '0'),
    height: parseFloat(el.getAttribute('height') || '0'),
    fill: el.getAttribute('fill') || undefined,
    stroke: el.getAttribute('stroke') || undefined,
    strokeWidth: el.hasAttribute('stroke-width') ? parseFloat(el.getAttribute('stroke-width')!) : undefined,
    rx: el.hasAttribute('rx') ? parseFloat(el.getAttribute('rx')!) : undefined,
    ry: el.hasAttribute('ry') ? parseFloat(el.getAttribute('ry')!) : undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
    fillOpacity: el.hasAttribute('fill-opacity') ? parseFloat(el.getAttribute('fill-opacity')!) : undefined,
  }
}

// ============================================================
// Path 解析
// ============================================================

function parsePath(el: SVGElement, genId: (prefix: string) => string): PathElement {
  return {
    type: 'path',
    id: genId('path'),
    d: el.getAttribute('d') || '',
    fill: el.getAttribute('fill') || undefined,
    stroke: el.getAttribute('stroke') || undefined,
    strokeWidth: el.hasAttribute('stroke-width') ? parseFloat(el.getAttribute('stroke-width')!) : undefined,
    fillOpacity: el.hasAttribute('fill-opacity') ? parseFloat(el.getAttribute('fill-opacity')!) : undefined,
    fillRule: (el.getAttribute('fill-rule') as PathElement['fillRule']) || undefined,
    clipRule: el.getAttribute('clip-rule') || undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
  }
}

// ============================================================
// Line 解析
// ============================================================

function parseLine(el: SVGElement, genId: (prefix: string) => string): LineElement {
  return {
    type: 'line',
    id: genId('line'),
    x1: parseFloat(el.getAttribute('x1') || '0'),
    y1: parseFloat(el.getAttribute('y1') || '0'),
    x2: parseFloat(el.getAttribute('x2') || '0'),
    y2: parseFloat(el.getAttribute('y2') || '0'),
    stroke: el.getAttribute('stroke') || '#000000',
    strokeWidth: el.hasAttribute('stroke-width') ? parseFloat(el.getAttribute('stroke-width')!) : undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
    strokeOpacity: el.hasAttribute('stroke-opacity') ? parseFloat(el.getAttribute('stroke-opacity')!) : undefined,
  }
}

// ============================================================
// Circle 解析
// ============================================================

function parseCircle(el: SVGElement, genId: (prefix: string) => string): CircleElement {
  return {
    type: 'circle',
    id: genId('circle'),
    cx: parseFloat(el.getAttribute('cx') || '0'),
    cy: parseFloat(el.getAttribute('cy') || '0'),
    r: parseFloat(el.getAttribute('r') || '0'),
    fill: el.getAttribute('fill') || undefined,
    stroke: el.getAttribute('stroke') || undefined,
    strokeWidth: el.hasAttribute('stroke-width') ? parseFloat(el.getAttribute('stroke-width')!) : undefined,
    fillOpacity: el.hasAttribute('fill-opacity') ? parseFloat(el.getAttribute('fill-opacity')!) : undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
  }
}

// ============================================================
// Image 解析
// ============================================================

function parseImage(el: SVGElement, genId: (prefix: string) => string): ImageElement {
  return {
    type: 'image',
    id: genId('image'),
    x: parseFloat(el.getAttribute('x') || '0'),
    y: parseFloat(el.getAttribute('y') || '0'),
    width: parseFloat(el.getAttribute('width') || '0'),
    height: parseFloat(el.getAttribute('height') || '0'),
    href: el.getAttribute('href') || el.getAttributeNS('http://www.w3.org/1999/xlink', 'href') || '',
    preserveAspectRatio: el.getAttribute('preserveAspectRatio') || undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
  }
}

// ============================================================
// Group 解析
// ============================================================

function parseGroup(el: SVGElement, genId: (prefix: string) => string): GroupElement {
  const children: Element[] = []
  for (const child of Array.from(el.children)) {
    const parsed = parseElement(child as SVGElement, genId)
    if (parsed) children.push(parsed)
  }

  return {
    type: 'group',
    id: genId('group'),
    children,
    transform: el.getAttribute('transform') || undefined,
    fill: el.getAttribute('fill') || undefined,
    fontFamily: el.getAttribute('font-family') || undefined,
    fontSize: el.hasAttribute('font-size') ? parseFloat(el.getAttribute('font-size')!) : undefined,
    fontWeight: el.getAttribute('font-weight') || undefined,
    filter: el.getAttribute('filter') || undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
  }
}

// ============================================================
// 多行文本合并
// ============================================================

/**
 * ppt-master 的 SVG 通常用多个相邻 <text> 元素模拟多行文本
 * 如果它们 x 相同、fontFamily/fontSize 相同、y 差值接近行高，合并为一个 TextElement
 */
function mergeAdjacentTexts(elements: Element[]): Element[] {
  const result: Element[] = []
  let i = 0

  while (i < elements.length) {
    const el = elements[i]
    if (el.type !== 'text') {
      result.push(el)
      i++
      continue
    }

    // 尝试向后合并相邻的 text 元素
    const group: TextElement[] = [el]
    let j = i + 1
    while (j < elements.length && elements[j].type === 'text') {
      const next = elements[j] as TextElement
      const prev = group[group.length - 1]
      if (canMergeTexts(prev, next)) {
        group.push(next)
        j++
      } else {
        break
      }
    }

    if (group.length === 1) {
      result.push(el)
    } else {
      result.push(mergeTextGroup(group))
    }
    i = j
  }

  return result
}

function canMergeTexts(a: TextElement, b: TextElement): boolean {
  // 同一 x 位置
  if (a.x !== b.x) return false
  // 同一字体和大小
  if (a.fontFamily !== b.fontFamily) return false
  if (a.fontSize !== b.fontSize) return false
  if (a.fill !== b.fill) return false
  // y 差值在合理行高范围内（fontSize * 0.8 ~ fontSize * 2.5）
  const dy = b.y - a.y
  const fontSize = a.fontSize || 16
  if (dy < fontSize * 0.8 || dy > fontSize * 2.5) return false
  return true
}

function mergeTextGroup(group: TextElement[]): TextElement {
  const first = group[0]
  const mergedText = group.map(t => t.text).join('\n')
  const dy = group.length > 1 ? group[1].y - group[0].y : first.lineHeight

  return {
    ...first,
    text: mergedText,
    lineHeight: Math.round(dy),
    // 总高度覆盖所有行
    maxHeight: (group[group.length - 1].y - first.y) + dy,
  }
}
