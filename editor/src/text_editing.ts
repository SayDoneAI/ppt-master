import type {
  Element as SlideElement,
  PatchOperation,
  Slide,
  TextElement,
} from './slide_state.js'
import { layoutText } from './state_to_svg.js'
import {
  formatNumber,
  parseFontSpec,
  type EditableElement,
} from './preset_engine.js'
import type { SvgBounds } from './pointer_interactions.js'

const TEXT_EDITOR_MIN_WIDTH = 120
const TEXT_EDITOR_MIN_HEIGHT = 44
const TEXT_EDITOR_STATUS_HEIGHT = 24
const TEXT_EDITOR_HEIGHT_PADDING = 12

export interface HiddenCanvasNode {
  node: SVGGraphicsElement
  opacity: string | null
}

export interface ActiveTextEditor {
  elementId: string
  foreignObject: SVGForeignObjectElement
  wrapper: HTMLDivElement
  textarea: HTMLTextAreaElement
  status: HTMLDivElement
  hiddenNodes: HiddenCanvasNode[]
  bounds: SvgBounds
  initialText: string
}

export interface TextEditingContext {
  readonly document: Document
  readonly elementFields: HTMLFormElement
  readonly svgNamespace: string
  readonly editingTextId: string | null
  readonly activeTextEditor: ActiveTextEditor | null
  readonly selectedElementId: string | null
  getCanvasSvg: () => SVGSVGElement | null
  getCanvasElementNodes: (elementId?: string) => SVGGraphicsElement[]
  measureElementBounds: (svg: SVGSVGElement, elementId: string, padding?: number) => SvgBounds | null
  findElementById: (elements: SlideElement[], elementId: string | null) => EditableElement | null
  getCurrentSlide: () => Slide
  getCurrentSlideIndex: () => number
  syncSlideElementIntoSvg: (element: SlideElement, svg: SVGSVGElement, doc: Document) => boolean
  createPropertyPatch: (
    elementId: string,
    property: string,
    oldValue: unknown,
    newValue: unknown,
  ) => PatchOperation | null
  commitPatchOperations: (operations: PatchOperation[], source: 'human' | 'ai', description: string) => void
  commitCurrentSlideFromLiveCanvasSvg: () => boolean
  renderCanvas: (slide: Slide, index: number) => void
  renderThumbnails: () => void
  renderInspector: () => void
  refreshCanvasOverlays: () => void
  setEditingTextId: (value: string | null) => void
  setActiveTextEditor: (editor: ActiveTextEditor | null) => void
  setHoveredElementId: (value: string | null) => void
  requestAnimationFrame: (callback: FrameRequestCallback) => number
}

export function enterTextEditing(element: TextElement, context: TextEditingContext): void {
  const svg = context.getCanvasSvg()
  if (!svg) return

  if (context.editingTextId === element.id && context.activeTextEditor) {
    context.activeTextEditor.textarea.focus()
    context.activeTextEditor.textarea.select()
    return
  }

  if (context.editingTextId) exitTextEditing(context)

  const bounds = measureTextEditorBounds(svg, element.id, element, context)
  if (!bounds) return

  const foreignObject = context.document.createElementNS(
    context.svgNamespace,
    'foreignObject',
  ) as SVGForeignObjectElement
  foreignObject.setAttribute('data-text-editor-root', 'true')
  foreignObject.setAttribute('x', String(bounds.x))
  foreignObject.setAttribute('y', String(bounds.y))
  foreignObject.setAttribute('width', String(bounds.width))
  foreignObject.setAttribute('overflow', 'visible')

  const wrapper = context.document.createElement('div')
  wrapper.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml')
  wrapper.style.display = 'flex'
  wrapper.style.flexDirection = 'column'
  wrapper.style.gap = '6px'
  wrapper.style.pointerEvents = 'auto'

  const textarea = context.document.createElement('textarea')
  textarea.setAttribute('data-text-editor-textarea', 'true')
  textarea.value = element.text
  textarea.spellcheck = false
  applyTextEditorStyles(textarea, element)

  const status = context.document.createElement('div')
  status.setAttribute('data-text-editor-status', 'true')
  status.style.font = '600 12px Inter, PingFang SC, sans-serif'
  status.style.padding = '0 2px'
  status.style.userSelect = 'none'

  const hiddenNodes = hideCanvasElementNodes(element.id, context)

  wrapper.append(textarea, status)
  foreignObject.append(wrapper)
  svg.append(foreignObject)

  context.setEditingTextId(element.id)
  context.setActiveTextEditor({
    elementId: element.id,
    foreignObject,
    wrapper,
    textarea,
    status,
    hiddenNodes,
    bounds,
    initialText: element.text,
  })

  textarea.addEventListener('input', () => {
    if (context.editingTextId !== element.id || !context.activeTextEditor) return
    element.text = textarea.value
    syncInspectorTextField(textarea.value, context)
    syncTextEditorFrame(context.activeTextEditor, element)
  })

  textarea.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return
    event.preventDefault()
    event.stopPropagation()
    exitTextEditing(context)
  })

  if (!context.activeTextEditor) return
  syncTextEditorFrame(context.activeTextEditor, element)
  context.refreshCanvasOverlays()

  context.requestAnimationFrame(() => {
    textarea.focus()
    textarea.select()
  })
}

export function exitTextEditing(
  context: TextEditingContext,
  options: { shouldRender?: boolean } = {},
): void {
  const editor = context.activeTextEditor
  if (!editor) {
    context.setEditingTextId(null)
    return
  }

  const currentElement = context.findElementById(context.getCurrentSlide().elements, editor.elementId)
  let shouldCommitSvgFirst = false
  if (currentElement?.type === 'text') {
    const svg = context.getCanvasSvg()
    if (svg) context.syncSlideElementIntoSvg(currentElement, svg, context.document)
    const operation = context.createPropertyPatch(currentElement.id, 'text', editor.initialText, currentElement.text)
    if (operation) {
      shouldCommitSvgFirst = true
      context.commitPatchOperations([operation], 'human', `编辑文本 ${currentElement.id}`)
    }
  }

  restoreHiddenCanvasNodes(editor.hiddenNodes)
  editor.foreignObject.remove()
  context.setActiveTextEditor(null)
  context.setEditingTextId(null)
  context.setHoveredElementId(context.selectedElementId)

  if (shouldCommitSvgFirst) {
    context.commitCurrentSlideFromLiveCanvasSvg()
  }

  if (options.shouldRender !== false) {
    const slide = context.getCurrentSlide()
    context.renderCanvas(slide, context.getCurrentSlideIndex())
    context.renderThumbnails()
    context.renderInspector()
  } else {
    context.renderInspector()
    context.refreshCanvasOverlays()
  }
}

export function measureTextEditorBounds(
  svg: SVGSVGElement,
  elementId: string,
  element: TextElement,
  context: TextEditingContext,
): SvgBounds | null {
  const renderedBounds = context.measureElementBounds(svg, elementId, 0)
  if (!renderedBounds) return null

  const layout = layoutText(element)
  const boxHeight = element.maxHeight ?? layout.height
  const width = Math.max(TEXT_EDITOR_MIN_WIDTH, element.width, renderedBounds.width)
  const height = Math.max(
    TEXT_EDITOR_MIN_HEIGHT,
    renderedBounds.height,
    boxHeight + TEXT_EDITOR_HEIGHT_PADDING,
  )

  let x = renderedBounds.x
  if (width > renderedBounds.width) {
    switch (element.textAnchor) {
      case 'middle':
        x -= (width - renderedBounds.width) / 2
        break
      case 'end':
        x -= width - renderedBounds.width
        break
    }
  }

  const fontSpec = parseFontSpec(element.font)
  return {
    x,
    y: Math.min(renderedBounds.y, element.y - fontSpec.fontSize),
    width,
    height,
  }
}

export function applyTextEditorStyles(textarea: HTMLTextAreaElement, element: TextElement): void {
  const fontSpec = parseFontSpec(element.font)

  textarea.style.display = 'block'
  textarea.style.margin = '0'
  textarea.style.padding = '6px 8px'
  textarea.style.border = '1.5px solid #2563EB'
  textarea.style.borderRadius = '12px'
  textarea.style.background = 'rgba(255, 255, 255, 0.96)'
  textarea.style.boxShadow = '0 14px 30px rgba(15, 23, 42, 0.16)'
  textarea.style.color = element.fill
  textarea.style.fontFamily = element.fontFamily || fontSpec.fontFamily
  textarea.style.fontSize = `${element.fontSize ?? fontSpec.fontSize}px`
  textarea.style.fontWeight = String(element.fontWeight ?? fontSpec.fontWeight ?? '')
  textarea.style.fontStyle = fontSpec.fontStyle ?? 'normal'
  textarea.style.lineHeight = `${element.lineHeight}px`
  textarea.style.letterSpacing = element.letterSpacing ? `${element.letterSpacing}px` : 'normal'
  textarea.style.textAlign = getTextAlign(element.textAnchor)
  textarea.style.resize = 'none'
  textarea.style.outline = 'none'
  textarea.style.overflow = 'hidden'
  textarea.style.whiteSpace = 'pre-wrap'
  textarea.style.overflowWrap = 'break-word'
}

export function syncTextEditorFrame(editor: ActiveTextEditor, element: TextElement): void {
  const layout = layoutText(element)
  const boxHeight = element.maxHeight ?? layout.height
  const textareaHeight = Math.max(
    editor.bounds.height,
    boxHeight + TEXT_EDITOR_HEIGHT_PADDING,
  )
  const editorHeight = textareaHeight + TEXT_EDITOR_STATUS_HEIGHT

  editor.foreignObject.setAttribute('height', String(editorHeight))
  editor.wrapper.style.width = `${editor.bounds.width}px`
  editor.wrapper.style.height = `${editorHeight}px`
  editor.textarea.style.width = `${editor.bounds.width}px`
  editor.textarea.style.height = `${textareaHeight}px`
  editor.status.textContent = buildTextEditorStatus(layout.lineCount, layout.height, element.maxHeight, layout.overflow)
  editor.status.style.color = layout.overflow ? '#B91C1C' : '#475569'
}

export function buildTextEditorStatus(
  lineCount: number,
  height: number,
  maxHeight: number | undefined,
  overflow: boolean,
): string {
  const parts = [
    `Pretext ${lineCount} 行`,
    `高度 ${formatNumber(height)}px`,
  ]

  if (maxHeight !== undefined) parts.push(`max ${formatNumber(maxHeight)}px`)
  parts.push(overflow ? '已溢出' : '未溢出')
  return parts.join(' · ')
}

export function hideCanvasElementNodes(
  elementId: string,
  context: TextEditingContext,
): HiddenCanvasNode[] {
  return context.getCanvasElementNodes(elementId).map(node => {
    const opacity = node.getAttribute('opacity')
    node.setAttribute('opacity', '0')
    return { node, opacity }
  })
}

export function restoreHiddenCanvasNodes(hiddenNodes: HiddenCanvasNode[]): void {
  for (const hidden of hiddenNodes) {
    if (hidden.opacity === null) hidden.node.removeAttribute('opacity')
    else hidden.node.setAttribute('opacity', hidden.opacity)
  }
}

export function syncInspectorTextField(value: string, context: TextEditingContext): void {
  const field = context.elementFields.querySelector<HTMLTextAreaElement | HTMLInputElement>('[data-prop-key="text"]')
  if (field && field.value !== value) field.value = value
}

export function getTextAlign(textAnchor?: TextElement['textAnchor']): 'left' | 'center' | 'right' {
  switch (textAnchor) {
    case 'middle':
      return 'center'
    case 'end':
      return 'right'
    default:
      return 'left'
  }
}
