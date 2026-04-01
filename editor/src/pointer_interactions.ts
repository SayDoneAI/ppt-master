import type {
  CircleElement,
  ImageElement,
  LineElement,
  PathElement,
  RectElement,
  TextElement,
} from './slide_state.js'
import { layoutText } from './state_to_svg.js'
import { parseFontSpec, type EditableElement } from './preset_engine.js'
import { applyTextResizeSemantics, syncTextSvgNodes } from './text_resize.js'

export type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'

export type SupportedEditableElement =
  | TextElement
  | RectElement
  | PathElement
  | ImageElement
  | LineElement
  | CircleElement

export type TransformableElement =
  | TextElement
  | RectElement
  | ImageElement
  | LineElement
  | CircleElement

export type InspectorControl = HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement

export interface SvgBounds {
  x: number
  y: number
  width: number
  height: number
}

export interface SvgPoint {
  x: number
  y: number
}

export interface TextNodeSnapshot {
  node: Element
  x: number | null
  y: number | null
}

export interface PointerInteractionSession {
  pointerId: number
  kind: 'move' | 'resize'
  elementId: string
  handle?: ResizeHandle
  startClientX: number
  startClientY: number
  startPoint: SvgPoint
  initialBounds: SvgBounds
  initialElement: TransformableElement
  textNodeSnapshots: TextNodeSnapshot[]
  started: boolean
}

export interface PointerInteractionBoundsContext {
  measureElementBounds: (svg: SVGSVGElement, elementId: string, padding?: number) => SvgBounds | null
  minTextWidth: number
}

export interface CanvasElementCursorsContext {
  getCanvasElementNodes: (elementId?: string) => SVGGraphicsElement[]
  readonly editingTextId: string | null
  getSelectedElement: () => EditableElement | null
  isTransformableElement: (element: EditableElement) => element is TransformableElement
}

export interface ResizeHandlePointerDownContext {
  readonly editingTextId: string | null
  getSelectedElement: () => EditableElement | null
  isTransformableElement: (element: EditableElement) => element is TransformableElement
  setInteractionHint: (value: string | null) => void
  renderInspector: () => void
  getCanvasSvg: () => SVGSVGElement | null
  cloneTransformableElement: <T extends TransformableElement>(element: T) => T
  captureTextNodeSnapshots: (element: TransformableElement) => TextNodeSnapshot[]
  setPointerSession: (session: PointerInteractionSession | null) => void
  getElementInteractionBounds: (element: EditableElement, svg: SVGSVGElement) => SvgBounds | null
}

export interface ResizeApplicationContext {
  readonly canvas: {
    width: number
    height: number
  }
  clamp: (value: number, min: number, max: number) => number
  minTextWidth: number
  minResizeSize: number
  minCircleDiameter: number
}

export interface LiveElementPreviewContext {
  getCanvasElementNodes: (elementId?: string) => SVGGraphicsElement[]
  document: Document
}

export interface InspectorPreviewContext<Field extends { key: string }> {
  selectionSummary: HTMLDivElement
  elementFields: HTMLFormElement
  buildSelectionSummary: (element: EditableElement) => string
  getInspectorFields: (element: SupportedEditableElement) => Field[]
  getFieldValue: (element: SupportedEditableElement, key: string) => string
  syncInspectorControlValue: (control: InspectorControl, field: Field, value: string) => void
}

export function clientPointToViewBox(svg: SVGSVGElement, clientX: number, clientY: number): SvgPoint | null {
  const svgRect = svg.getBoundingClientRect()
  const viewBox = svg.viewBox.baseVal
  if (!svgRect.width || !svgRect.height) return null

  return {
    x: ((clientX - svgRect.left) / svgRect.width) * viewBox.width + viewBox.x,
    y: ((clientY - svgRect.top) / svgRect.height) * viewBox.height + viewBox.y,
  }
}

export function expandBounds(bounds: SvgBounds, padding: number): SvgBounds {
  return {
    x: bounds.x - padding,
    y: bounds.y - padding,
    width: bounds.width + padding * 2,
    height: bounds.height + padding * 2,
  }
}

export function getResizeCursor(handle: ResizeHandle): string {
  switch (handle) {
    case 'nw':
    case 'se':
      return 'nwse-resize'
    case 'ne':
    case 'sw':
      return 'nesw-resize'
    case 'n':
    case 's':
      return 'ns-resize'
    case 'e':
    case 'w':
      return 'ew-resize'
  }
}

export function getElementInteractionBounds(
  element: EditableElement,
  svg: SVGSVGElement,
  context: PointerInteractionBoundsContext,
): SvgBounds | null {
  switch (element.type) {
    case 'text':
      return getTextInteractionBounds(element, context.minTextWidth)
    case 'rect':
    case 'image':
      return {
        x: element.x,
        y: element.y,
        width: element.width,
        height: element.height,
      }
    case 'circle':
      return {
        x: element.cx - element.r,
        y: element.cy - element.r,
        width: element.r * 2,
        height: element.r * 2,
      }
    case 'line':
      return {
        x: Math.min(element.x1, element.x2),
        y: Math.min(element.y1, element.y2),
        width: Math.abs(element.x2 - element.x1),
        height: Math.abs(element.y2 - element.y1),
      }
    case 'path':
    case 'group':
      return context.measureElementBounds(svg, element.id, 0)
  }
}

export function getTextInteractionBounds(element: TextElement, minTextWidth: number): SvgBounds {
  const fontSpec = parseFontSpec(element.font)
  const layout = layoutText(element)
  const boxHeight = element.maxHeight ?? layout.height
  return {
    x: getTextAnchorLeft(element),
    y: element.y - fontSpec.fontSize,
    width: Math.max(minTextWidth, element.width),
    height: Math.max(fontSpec.fontSize, boxHeight),
  }
}

export function getTextAnchorLeft(element: TextElement): number {
  switch (element.textAnchor) {
    case 'middle':
      return element.x - element.width / 2
    case 'end':
      return element.x - element.width
    default:
      return element.x
  }
}

export function syncCanvasElementCursors(context: CanvasElementCursorsContext): void {
  for (const node of context.getCanvasElementNodes()) {
    node.style.cursor = ''
  }

  if (context.editingTextId) return

  const selected = context.getSelectedElement()
  if (!selected || !context.isTransformableElement(selected)) return

  for (const node of context.getCanvasElementNodes(selected.id)) {
    node.style.cursor = 'move'
  }
}

export function stopOverlayHandleClick(event: MouseEvent): void {
  event.preventDefault()
  event.stopPropagation()
}

export function handleResizeHandlePointerDown(
  event: PointerEvent,
  context: ResizeHandlePointerDownContext,
): void {
  if (context.editingTextId || event.button !== 0) return
  if (!(event.currentTarget instanceof SVGRectElement)) return

  const handle = event.currentTarget.dataset.editorHandle as ResizeHandle | undefined
  const selected = context.getSelectedElement()
  if (!handle || !selected) return

  if (!context.isTransformableElement(selected)) {
    context.setInteractionHint(
      selected.type === 'path'
        ? 'path 元素暂不支持缩放；后续可考虑改为 transform 模式。'
        : 'group 容器暂不支持缩放；请直接调整内部具体元素。',
    )
    context.renderInspector()
    return
  }

  const svg = context.getCanvasSvg()
  if (!svg) return

  const startPoint = clientPointToViewBox(svg, event.clientX, event.clientY)
  const initialBounds = context.getElementInteractionBounds(selected, svg)
  if (!startPoint || !initialBounds) return

  context.setInteractionHint(null)
  context.setPointerSession({
    pointerId: event.pointerId,
    kind: 'resize',
    elementId: selected.id,
    handle,
    startClientX: event.clientX,
    startClientY: event.clientY,
    startPoint,
    initialBounds,
    initialElement: context.cloneTransformableElement(selected),
    textNodeSnapshots: context.captureTextNodeSnapshots(selected),
    started: false,
  })

  event.preventDefault()
  event.stopPropagation()
}

export function applyMoveFromSession(
  element: TransformableElement,
  initialElement: TransformableElement,
  initialBounds: SvgBounds,
  delta: SvgPoint,
  context: ResizeApplicationContext,
): void {
  const nextBounds = clampMoveBounds({
    x: initialBounds.x + delta.x,
    y: initialBounds.y + delta.y,
    width: initialBounds.width,
    height: initialBounds.height,
  }, context)
  const boundedDeltaX = nextBounds.x - initialBounds.x
  const boundedDeltaY = nextBounds.y - initialBounds.y

  switch (element.type) {
    case 'text':
      if (initialElement.type !== 'text') return
      element.y = initialElement.y + boundedDeltaY
      switch (initialElement.textAnchor) {
        case 'middle':
          element.x = nextBounds.x + initialElement.width / 2
          return
        case 'end':
          element.x = nextBounds.x + initialElement.width
          return
        default:
          element.x = nextBounds.x
          return
      }
    case 'rect':
      if (initialElement.type !== 'rect') return
      element.x = nextBounds.x
      element.y = nextBounds.y
      return
    case 'image':
      if (initialElement.type !== 'image') return
      element.x = nextBounds.x
      element.y = nextBounds.y
      return
    case 'line':
      if (initialElement.type !== 'line') return
      element.x1 = initialElement.x1 + boundedDeltaX
      element.y1 = initialElement.y1 + boundedDeltaY
      element.x2 = initialElement.x2 + boundedDeltaX
      element.y2 = initialElement.y2 + boundedDeltaY
      return
    case 'circle':
      if (initialElement.type !== 'circle') return
      element.cx = initialElement.cx + boundedDeltaX
      element.cy = initialElement.cy + boundedDeltaY
      return
  }
}

export function applyResizeFromSession(
  element: TransformableElement,
  session: PointerInteractionSession,
  delta: SvgPoint,
  context: ResizeApplicationContext,
): void {
  if (!session.handle) return

  const minWidth = element.type === 'text' ? context.minTextWidth : context.minResizeSize
  const minHeight = element.type === 'text' && session.initialElement.type === 'text'
    ? getTextResizeMinHeight(session.initialElement, context.minResizeSize)
    : context.minResizeSize
  const nextBounds = resizeBoundsFromHandle(session.initialBounds, delta, session.handle, minWidth, minHeight, context)

  switch (element.type) {
    case 'text':
      if (session.initialElement.type !== 'text') return
      applyTextResize(element, session.initialElement, session.initialBounds, nextBounds, session.handle, context)
      return
    case 'rect':
      element.x = nextBounds.x
      element.y = nextBounds.y
      element.width = nextBounds.width
      element.height = nextBounds.height
      return
    case 'image':
      element.x = nextBounds.x
      element.y = nextBounds.y
      element.width = nextBounds.width
      element.height = nextBounds.height
      return
    case 'circle':
      applyCircleResize(element, nextBounds, session.handle, context)
      return
    case 'line':
      if (session.initialElement.type !== 'line') return
      applyLineResize(element, session.initialElement, session.initialBounds, nextBounds)
      return
  }
}

export function resizeBoundsFromHandle(
  bounds: SvgBounds,
  delta: SvgPoint,
  handle: ResizeHandle,
  minWidth: number,
  minHeight: number,
  context: ResizeApplicationContext,
): SvgBounds {
  const initialMinX = bounds.x
  const initialMaxX = bounds.x + bounds.width
  const initialMinY = bounds.y
  const initialMaxY = bounds.y + bounds.height

  let minX = initialMinX
  let maxX = initialMaxX
  let minY = initialMinY
  let maxY = initialMaxY

  if (handle.includes('w')) minX = context.clamp(initialMinX + delta.x, 0, initialMaxX - minWidth)
  if (handle.includes('e')) maxX = context.clamp(initialMaxX + delta.x, initialMinX + minWidth, context.canvas.width)
  if (handle.includes('n')) minY = context.clamp(initialMinY + delta.y, 0, initialMaxY - minHeight)
  if (handle.includes('s')) maxY = context.clamp(initialMaxY + delta.y, initialMinY + minHeight, context.canvas.height)

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  }
}

export function applyTextResize(
  element: TextElement,
  initialElement: TextElement,
  initialBounds: SvgBounds,
  nextBounds: SvgBounds,
  handle: ResizeHandle,
  context: ResizeApplicationContext,
): void {
  applyTextResizeSemantics({
    element,
    initialElement,
    initialBounds,
    nextBounds,
    handle,
    minWidth: context.minTextWidth,
    minHeight: getTextResizeMinHeight(initialElement, context.minResizeSize),
  })
}

export function getTextResizeMinHeight(element: TextElement, minResizeSize: number): number {
  const fontSpec = parseFontSpec(element.font)
  return Math.max(minResizeSize, fontSpec.fontSize, element.lineHeight)
}

export function clampMoveBounds(bounds: SvgBounds, context: ResizeApplicationContext): SvgBounds {
  const maxX = Math.max(0, context.canvas.width - bounds.width)
  const maxY = Math.max(0, context.canvas.height - bounds.height)
  return {
    ...bounds,
    x: context.clamp(bounds.x, 0, maxX),
    y: context.clamp(bounds.y, 0, maxY),
  }
}

export function applyCircleResize(
  circle: CircleElement,
  nextBounds: SvgBounds,
  handle: ResizeHandle,
  context: ResizeApplicationContext,
): void {
  const fitted = fitCircleBounds(nextBounds, handle, context.minCircleDiameter)
  circle.cx = fitted.x + fitted.width / 2
  circle.cy = fitted.y + fitted.height / 2
  circle.r = fitted.width / 2
}

export function fitCircleBounds(bounds: SvgBounds, handle: ResizeHandle, minCircleDiameter: number): SvgBounds {
  const size = Math.max(minCircleDiameter, Math.min(bounds.width, bounds.height))
  let x = bounds.x
  let y = bounds.y

  if (bounds.width > size) {
    if (handle.includes('w') && !handle.includes('e')) x = bounds.x + bounds.width - size
    else if (!handle.includes('w') && !handle.includes('e')) x = bounds.x + (bounds.width - size) / 2
  }

  if (bounds.height > size) {
    if (handle.includes('n') && !handle.includes('s')) y = bounds.y + bounds.height - size
    else if (!handle.includes('n') && !handle.includes('s')) y = bounds.y + (bounds.height - size) / 2
  }

  return { x, y, width: size, height: size }
}

export function applyLineResize(
  line: LineElement,
  initialElement: LineElement,
  initialBounds: SvgBounds,
  nextBounds: SvgBounds,
): void {
  line.x1 = scaleCoordinate(initialElement.x1, initialBounds.x, initialBounds.width, nextBounds.x, nextBounds.width)
  line.y1 = scaleCoordinate(initialElement.y1, initialBounds.y, initialBounds.height, nextBounds.y, nextBounds.height)
  line.x2 = scaleCoordinate(initialElement.x2, initialBounds.x, initialBounds.width, nextBounds.x, nextBounds.width)
  line.y2 = scaleCoordinate(initialElement.y2, initialBounds.y, initialBounds.height, nextBounds.y, nextBounds.height)
}

export function scaleCoordinate(
  value: number,
  initialStart: number,
  initialSize: number,
  nextStart: number,
  nextSize: number,
): number {
  if (initialSize === 0) return nextStart + nextSize / 2
  return nextStart + ((value - initialStart) / initialSize) * nextSize
}

export function syncLiveElementPreview(
  element: TransformableElement,
  session: PointerInteractionSession,
  context: LiveElementPreviewContext,
): void {
  switch (element.type) {
    case 'text':
      syncTextNodePreview(element, session, context)
      return
    case 'rect':
      syncElementNodesAttributes(element.id, {
        x: element.x,
        y: element.y,
        width: element.width,
        height: element.height,
      }, context)
      return
    case 'image':
      syncElementNodesAttributes(element.id, {
        x: element.x,
        y: element.y,
        width: element.width,
        height: element.height,
      }, context)
      return
    case 'line':
      syncElementNodesAttributes(element.id, {
        x1: element.x1,
        y1: element.y1,
        x2: element.x2,
        y2: element.y2,
      }, context)
      return
    case 'circle':
      syncElementNodesAttributes(element.id, {
        cx: element.cx,
        cy: element.cy,
        r: element.r,
      }, context)
      return
  }
}

export function syncTextNodePreview(
  element: TextElement,
  session: PointerInteractionSession,
  context: LiveElementPreviewContext,
): void {
  if (session.initialElement.type !== 'text') return
  syncTextSvgNodes(context.getCanvasElementNodes(element.id), element, context.document)
}

export function syncElementNodesAttributes(
  elementId: string,
  attributes: Record<string, number>,
  context: LiveElementPreviewContext,
): void {
  for (const node of context.getCanvasElementNodes(elementId)) {
    for (const [key, value] of Object.entries(attributes)) {
      node.setAttribute(key, String(value))
    }
  }
}

export function syncInspectorPreview<Field extends { key: string }>(
  element: SupportedEditableElement,
  context: InspectorPreviewContext<Field>,
): void {
  context.selectionSummary.innerHTML = context.buildSelectionSummary(element)
  if (context.elementFields.hidden) return

  for (const field of context.getInspectorFields(element)) {
    const nextValue = context.getFieldValue(element, field.key)
    const controls = Array.from(context.elementFields.querySelectorAll<InspectorControl>(`[data-prop-key="${field.key}"]`))
    controls.forEach(control => context.syncInspectorControlValue(control, field, nextValue))
  }
}
