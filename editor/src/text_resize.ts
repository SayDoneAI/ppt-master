import type { ResizeHandle, SvgBounds } from './pointer_interactions.js'
import type { TextElement } from './slide_state.js'
import { layoutText } from './state_to_svg.js'

const SVG_NS = 'http://www.w3.org/2000/svg'

export type ResizeBounds = SvgBounds

interface ParsedFontSpec {
  fontSize: number
  fontFamily: string
  fontWeight?: string
  fontStyle?: string
}

interface ApplyTextResizeSemanticsOptions {
  element: TextElement
  initialElement: TextElement
  initialBounds: ResizeBounds
  nextBounds: ResizeBounds
  handle: ResizeHandle
  minWidth: number
  minHeight: number
}

export const ALL_RESIZE_HANDLES: ResizeHandle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
export const TEXT_RESIZE_HANDLES: ResizeHandle[] = ['nw', 'ne', 'e', 'se', 'sw', 'w']

export function getResizeHandlesForElementType(elementType: string): ResizeHandle[] {
  return elementType === 'text' ? TEXT_RESIZE_HANDLES : ALL_RESIZE_HANDLES
}

export function applyTextResizeSemantics({
  element,
  initialElement,
  initialBounds,
  nextBounds,
  handle,
  minWidth,
  minHeight,
}: ApplyTextResizeSemanticsOptions): void {
  element.width = Math.max(minWidth, nextBounds.width)
  element.y = initialElement.y + (nextBounds.y - initialBounds.y)
  syncTextAnchorX(element, initialElement.textAnchor, nextBounds.x)

  if (handle === 'e' || handle === 'w') return

  const initialFontSpec = parseFontSpec(initialElement.font)
  const initialFontSize = initialElement.fontSize ?? initialFontSpec.fontSize
  const scaleX = initialBounds.width > 0 ? nextBounds.width / initialBounds.width : 1
  const scaleY = initialBounds.height > 0 ? nextBounds.height / initialBounds.height : 1
  const scale = handle === 'n' || handle === 's'
    ? scaleY
    : Math.sqrt(scaleX * scaleY)
  const nextFontSize = Math.max(8, Number((initialFontSize * scale).toFixed(1)))
  const nextLineHeight = Math.max(nextFontSize, Number((initialElement.lineHeight * scale).toFixed(1)))

  element.font = serializeFontSpec({
    fontSize: nextFontSize,
    fontFamily: initialElement.fontFamily || initialFontSpec.fontFamily,
    fontWeight: normalizeFontWeightValue(initialElement.fontWeight ?? initialFontSpec.fontWeight),
    fontStyle: initialFontSpec.fontStyle,
  })
  element.fontSize = nextFontSize
  element.lineHeight = nextLineHeight
  element.maxHeight = Math.max(minHeight, nextBounds.height)
}

export function syncTextSvgNodes(
  nodes: SVGElement[],
  element: TextElement,
  doc: Document,
): boolean {
  const textNodes = nodes.filter(node => node.tagName.toLowerCase() === 'text') as SVGTextElement[]
  if (textNodes.length === 0) return false

  const [primaryNode, ...duplicateNodes] = textNodes
  duplicateNodes.forEach(node => node.remove())

  const fontSpec = parseFontSpec(element.font)
  const fontFamily = element.fontFamily || fontSpec.fontFamily
  const fontSize = element.fontSize ?? fontSpec.fontSize
  const fontWeight = element.fontWeight ?? fontSpec.fontWeight
  const layout = layoutText(element)

  setSvgAttribute(primaryNode, 'x', element.x)
  setSvgAttribute(primaryNode, 'y', element.y)
  setSvgAttribute(primaryNode, 'font-family', fontFamily)
  setSvgAttribute(primaryNode, 'font-size', fontSize)
  setOptionalSvgAttribute(primaryNode, 'font-weight', fontWeight)
  setOptionalSvgAttribute(primaryNode, 'font-style', fontSpec.fontStyle)
  setSvgAttribute(primaryNode, 'fill', element.fill)
  setOptionalSvgAttribute(primaryNode, 'text-anchor', element.textAnchor)
  setOptionalSvgAttribute(primaryNode, 'letter-spacing', element.letterSpacing)
  setOptionalSvgAttribute(primaryNode, 'opacity', element.opacity, { skipValue: 1 })
  setOptionalSvgAttribute(primaryNode, 'fill-opacity', element.fillOpacity)

  while (primaryNode.firstChild) {
    primaryNode.removeChild(primaryNode.firstChild)
  }

  if (layout.lineCount <= 1) {
    primaryNode.textContent = layout.lines[0]?.text ?? element.text
    return true
  }

  layout.lines.forEach((line, index) => {
    const tspan = doc.createElementNS(SVG_NS, 'tspan')
    tspan.setAttribute('x', String(element.x))
    tspan.setAttribute('y', String(element.y + element.lineHeight * index))
    tspan.textContent = line.text
    primaryNode.append(tspan)
  })
  return true
}

function syncTextAnchorX(
  element: TextElement,
  textAnchor: TextElement['textAnchor'],
  nextLeft: number,
): void {
  switch (textAnchor) {
    case 'middle':
      element.x = nextLeft + element.width / 2
      return
    case 'end':
      element.x = nextLeft + element.width
      return
    default:
      element.x = nextLeft
  }
}

function setSvgAttribute(node: SVGElement, attribute: string, value: string | number): void {
  node.setAttribute(attribute, String(value))
}

function setOptionalSvgAttribute(
  node: SVGElement,
  attribute: string,
  value: string | number | undefined,
  options: { skipValue?: string | number } = {},
): void {
  if (value === undefined || value === '' || value === options.skipValue) {
    node.removeAttribute(attribute)
    return
  }

  node.setAttribute(attribute, String(value))
}

function normalizeFontWeightValue(value: string | number | undefined): string {
  if (value === undefined || value === null || value === '' || value === 'normal') return '400'
  if (value === 'bold') return '700'
  if (value === 'medium') return '500'

  const numeric = typeof value === 'number' ? value : Number.parseInt(String(value), 10)
  if (!Number.isFinite(numeric)) return '400'
  if (numeric >= 600) return '700'
  if (numeric >= 450) return '500'
  return '400'
}

function serializeFontSpec(spec: ParsedFontSpec): string {
  const parts: string[] = []
  if (spec.fontStyle) parts.push(spec.fontStyle)
  if (spec.fontWeight && normalizeFontWeightValue(spec.fontWeight) !== '400') {
    parts.push(normalizeFontWeightValue(spec.fontWeight))
  }
  parts.push(`${formatNumber(spec.fontSize)}px`)
  parts.push(spec.fontFamily)
  return parts.join(' ')
}

function parseFontSpec(font: string): ParsedFontSpec {
  const match = font.match(/(?:(italic)\s+)?(?:(bold|[1-9]00)\s+)?(\d+(?:\.\d+)?)px\s+(.+)/i)
  if (match) {
    return {
      fontStyle: match[1] || undefined,
      fontWeight: match[2] || undefined,
      fontSize: Number(match[3]),
      fontFamily: match[4],
    }
  }

  return {
    fontSize: 16,
    fontFamily: 'Inter, PingFang SC, sans-serif',
  }
}

function formatNumber(value: number): string {
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(1).replace(/\.0$/, '')
}
