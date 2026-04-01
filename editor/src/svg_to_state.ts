// svg_to_state.ts — 将现有 SVG 文件解析为 compat slide_state
//
// 兼容 ppt-master 现有的示例项目与 legacy handoff 输入
// 处理真实 SVG 中的各种写法：多个 <text> 模拟多行、<path> 圆角矩形、
// <g> 分组嵌套、linearGradient/filter defs 等

import type {
  Slide, Element, TextElement, RectElement, PathElement,
  LineElement, CircleElement, ImageElement, GroupElement,
  Canvas, SlideState, Def, LinearGradientDef, FilterDef,
} from './slide_state.js'

export interface SvgParseOptions {
  preserveTextNodes?: boolean
}

export interface NormalizeSvgForEditorOptions {
  idPrefix?: string
  sourcePath?: string
}

// ============================================================
// 主入口
// ============================================================

/**
 * 解析 SVG 字符串为 Slide
 * 使用 DOMParser（浏览器）或传入的解析器（Node/Bun）
 */
export function normalizeSvgForEditor(
  svgString: string,
  options: NormalizeSvgForEditorOptions = {},
  parser?: DOMParser,
): string {
  const p = parser ?? new DOMParser()
  const doc = p.parseFromString(svgString, 'image/svg+xml')
  const svgEl = doc.documentElement as unknown as SVGElement
  const usedIds = new Set<string>()
  let generatedCount = 0

  for (const node of collectInteractiveSvgNodes(svgEl)) {
    const existingId = node.getAttribute('data-element-id')?.trim() || node.getAttribute('id')?.trim() || ''
    const fallbackId = existingId
      || `${options.idPrefix ?? 'svg'}_${node.tagName.toLowerCase()}_${++generatedCount}`
    const stableId = createUniqueEditorId(existingId || fallbackId, usedIds)
    node.setAttribute('data-element-id', stableId)

    if (node.tagName.toLowerCase() === 'image') {
      const href = node.getAttribute('href')
        || node.getAttributeNS('http://www.w3.org/1999/xlink', 'href')
      if (href) node.setAttribute('href', normalizeImageHref(href, options.sourcePath))
    }
  }

  return svgEl.outerHTML
}

export function svgToSlide(
  svgString: string,
  slideId: string,
  parser?: DOMParser,
  options: SvgParseOptions = {},
): Slide {
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
  const merged = mergeAdjacentTexts(elements, options)

  return {
    id: slideId,
    elements: merged,
    defs: defs.length > 0 ? defs : undefined,
  }
}

/**
 * 解析多个 SVG 字符串为完整的 compat SlideState
 */
export function svgsToState(
  svgStrings: string[],
  slideIds?: string[],
  parser?: DOMParser,
  options: SvgParseOptions = {},
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
    return svgToSlide(svg, id, p, options)
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
  const textAnchor = (el.getAttribute('text-anchor') as TextElement['textAnchor']) || undefined
  const canvasWidth = parseViewBox(
    el.ownerSVGElement?.getAttribute('viewBox') ?? null,
    el.ownerSVGElement ?? el,
  ).width
  const width = estimateTextBoxWidth(text, fontSize, x, canvasWidth, textAnchor)

  return {
    type: 'text',
    id: resolveElementId(el, 'text', genId),
    x,
    y,
    width,
    text,
    font,
    lineHeight: Math.round(fontSize * 1.4),
    fill: el.getAttribute('fill') || '#000000',
    fontFamily,
    fontSize,
    fontWeight,
    textAnchor,
    letterSpacing: el.hasAttribute('letter-spacing') ? parseFloat(el.getAttribute('letter-spacing')!) : undefined,
    opacity: el.hasAttribute('opacity') ? parseFloat(el.getAttribute('opacity')!) : undefined,
    fillOpacity: el.hasAttribute('fill-opacity') ? parseFloat(el.getAttribute('fill-opacity')!) : undefined,
  }
}

function estimateTextBoxWidth(
  text: string,
  fontSize: number,
  x: number,
  canvasWidth: number,
  textAnchor?: TextElement['textAnchor'],
): number {
  const lines = text
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
  const longestLineLength = lines.reduce((max, line) => Math.max(max, line.length), 0)
  const estimatedTextWidth = Math.max(fontSize * 2, longestLineLength * fontSize * 0.56 + fontSize * 1.5)
  const availableWidth = getAvailableTextWidth(x, canvasWidth, textAnchor)
  return Math.min(availableWidth, estimatedTextWidth)
}

function getAvailableTextWidth(
  x: number,
  canvasWidth: number,
  textAnchor?: TextElement['textAnchor'],
): number {
  const safeCanvasWidth = Number.isFinite(canvasWidth) && canvasWidth > 0 ? canvasWidth : 1280
  switch (textAnchor) {
    case 'middle':
      return Math.max(40, Math.min(x * 2, (safeCanvasWidth - x) * 2, safeCanvasWidth))
    case 'end':
      return Math.max(40, x)
    default:
      return Math.max(40, safeCanvasWidth - x)
  }
}

// ============================================================
// Rect 解析
// ============================================================

function parseRect(el: SVGElement, genId: (prefix: string) => string): RectElement {
  return {
    type: 'rect',
    id: resolveElementId(el, 'rect', genId),
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
    id: resolveElementId(el, 'path', genId),
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
    id: resolveElementId(el, 'line', genId),
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
    id: resolveElementId(el, 'circle', genId),
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
    id: resolveElementId(el, 'image', genId),
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
    id: resolveElementId(el, 'group', genId),
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
function mergeAdjacentTexts(elements: Element[], options: SvgParseOptions): Element[] {
  if (options.preserveTextNodes) return elements

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

function resolveElementId(
  el: SVGElement,
  prefix: string,
  genId: (prefix: string) => string,
): string {
  const explicitId = el.getAttribute('data-element-id')?.trim() || el.getAttribute('id')?.trim()
  return explicitId && explicitId.length > 0 ? explicitId : genId(prefix)
}

function createUniqueEditorId(candidate: string, usedIds: Set<string>): string {
  const normalized = candidate.trim().replace(/\s+/g, '_')
  if (!usedIds.has(normalized)) {
    usedIds.add(normalized)
    return normalized
  }

  let suffix = 2
  let nextId = `${normalized}_${suffix}`
  while (usedIds.has(nextId)) {
    suffix += 1
    nextId = `${normalized}_${suffix}`
  }
  usedIds.add(nextId)
  return nextId
}

function normalizeImageHref(href: string, sourcePath?: string): string {
  if (!sourcePath || !isRelativeHref(href)) return href

  try {
    const assetBasePath = inferSvgAssetBasePath(sourcePath)
    const baseUrl = /^https?:\/\//i.test(assetBasePath)
      ? new URL(assetBasePath)
      : new URL(assetBasePath, 'https://ppt-master.local')
    const resolved = new URL(href, baseUrl)
    return /^https?:\/\//i.test(assetBasePath)
      ? resolved.toString()
      : `${resolved.pathname}${resolved.search}${resolved.hash}`
  } catch {
    return href
  }
}

function inferSvgAssetBasePath(sourcePath: string): string {
  const stripped = sourcePath.split('#')[0]?.split('?')[0] ?? sourcePath

  for (const marker of ['/svg_final/', '/svg_output/']) {
    const markerIndex = stripped.lastIndexOf(marker)
    if (markerIndex >= 0) return stripped.slice(0, markerIndex + 1)
  }

  const slashIndex = stripped.lastIndexOf('/')
  if (slashIndex >= 0) return stripped.slice(0, slashIndex + 1)
  return ''
}

function isRelativeHref(href: string): boolean {
  return href.length > 0 && !/^(?:[a-z]+:|\/|#|\/\/)/i.test(href)
}

function collectInteractiveSvgNodes(root: SVGElement): SVGElement[] {
  const nodes: SVGElement[] = []

  const visit = (element: SVGElement) => {
    for (const child of Array.from(element.children)) {
      const svgChild = child as SVGElement
      if (isInteractiveSvgTag(svgChild.tagName.toLowerCase())) nodes.push(svgChild)
      visit(svgChild)
    }
  }

  visit(root)
  return nodes
}

function isInteractiveSvgTag(tagName: string): boolean {
  return tagName === 'text'
    || tagName === 'rect'
    || tagName === 'circle'
    || tagName === 'line'
    || tagName === 'path'
    || tagName === 'image'
    || tagName === 'g'
}
