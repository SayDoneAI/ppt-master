import type {
  Canvas,
  CircleElement,
  Element,
  GroupElement,
  ImageElement,
  LineElement,
  RectElement,
  SlideState,
  TextElement,
} from './slide_state.js'

export interface CanvasPreset extends Canvas {
  id: string
  label: string
}

export interface TopbarSaveActionsState {
  exportSvgLabel: string
  showSaveTemplate: boolean
  mode: 'poster' | 'ppt'
}

const PPT_CANVAS_SIZES: readonly Canvas[] = [
  { width: 1280, height: 720 },
  { width: 1024, height: 768 },
] as const

export const CANVAS_PRESETS: CanvasPreset[] = [
  { id: 'square', label: '1:1', width: 1080, height: 1080 },
  { id: 'poster', label: '4:5', width: 1080, height: 1350 },
  { id: 'story', label: '9:16', width: 1080, height: 1920 },
]

export function shouldDefaultToPreview(state: SlideState): boolean {
  return state.slides.length === 1 && state.canvas.height >= state.canvas.width
}

export function isPptCanvas(canvas: Canvas): boolean {
  return PPT_CANVAS_SIZES.some(
    preset => preset.width === canvas.width && preset.height === canvas.height,
  )
}

export function isPosterCanvas(canvas: Canvas): boolean {
  return findCanvasPreset(canvas) !== null
}

export function getTopbarSaveActionsState(state: SlideState): TopbarSaveActionsState {
  const isPptMode = state.slides.length > 1 || isPptCanvas(state.canvas)
  return isPptMode
    ? {
        exportSvgLabel: '保存当前页',
        showSaveTemplate: false,
        mode: 'ppt',
      }
    : {
        exportSvgLabel: '保存 SVG',
        showSaveTemplate: true,
        mode: 'poster',
      }
}

export function shouldShowPosterCanvasControls(state: SlideState): boolean {
  return state.slides.length === 1 && isPosterCanvas(state.canvas)
}

export function supportsCanvasPresetEditing(state: SlideState): boolean {
  return state.slides.length === 1
    && state.slides.every(slide => slide.elements.every(isCanvasScaleFriendlyElement))
}

export function findCanvasPreset(canvas: Canvas): CanvasPreset | null {
  return CANVAS_PRESETS.find(
    preset => preset.width === canvas.width && preset.height === canvas.height,
  ) ?? null
}

export function resizeSlideStateCanvas(state: SlideState, nextCanvas: Canvas): SlideState {
  if (state.canvas.width === nextCanvas.width && state.canvas.height === nextCanvas.height) return state

  const scaleX = nextCanvas.width / state.canvas.width
  const scaleY = nextCanvas.height / state.canvas.height
  const uniformScale = Math.min(scaleX, scaleY)

  return {
    ...state,
    canvas: { ...nextCanvas },
    slides: state.slides.map(slide => ({
      ...slide,
      elements: slide.elements.map(element => scaleElement(element, scaleX, scaleY, uniformScale)),
    })),
  }
}

function isCanvasScaleFriendlyElement(element: Element): boolean {
  if (element.type === 'path') return false
  if (element.type === 'group') return element.children.every(isCanvasScaleFriendlyElement)
  return true
}

function scaleElement(element: Element, scaleX: number, scaleY: number, uniformScale: number): Element {
  switch (element.type) {
    case 'text':
      return scaleTextElement(element, scaleX, scaleY, uniformScale)
    case 'rect':
      return scaleRectElement(element, scaleX, scaleY, uniformScale)
    case 'image':
      return scaleImageElement(element, scaleX, scaleY, uniformScale)
    case 'line':
      return scaleLineElement(element, scaleX, scaleY, uniformScale)
    case 'circle':
      return scaleCircleElement(element, scaleX, scaleY, uniformScale)
    case 'group':
      return scaleGroupElement(element, scaleX, scaleY, uniformScale)
    case 'path':
      return { ...element }
  }
}

function scaleTextElement(
  element: TextElement,
  scaleX: number,
  scaleY: number,
  uniformScale: number,
): TextElement {
  return {
    ...element,
    x: scaleNumber(element.x, scaleX),
    y: scaleNumber(element.y, scaleY),
    width: scaleNumber(element.width, scaleX),
    maxHeight: element.maxHeight === undefined ? undefined : scaleNumber(element.maxHeight, scaleY),
    font: scaleFontShorthand(element.font, uniformScale),
    lineHeight: scaleNumber(element.lineHeight, uniformScale),
    fontSize: element.fontSize === undefined ? undefined : scaleNumber(element.fontSize, uniformScale),
    letterSpacing: element.letterSpacing === undefined ? undefined : scaleNumber(element.letterSpacing, scaleX),
  }
}

function scaleRectElement(
  element: RectElement,
  scaleX: number,
  scaleY: number,
  uniformScale: number,
): RectElement {
  return {
    ...element,
    x: scaleNumber(element.x, scaleX),
    y: scaleNumber(element.y, scaleY),
    width: scaleNumber(element.width, scaleX),
    height: scaleNumber(element.height, scaleY),
    strokeWidth: element.strokeWidth === undefined ? undefined : scaleNumber(element.strokeWidth, uniformScale),
    rx: element.rx === undefined ? undefined : scaleNumber(element.rx, uniformScale),
    ry: element.ry === undefined ? undefined : scaleNumber(element.ry, uniformScale),
  }
}

function scaleImageElement(
  element: ImageElement,
  scaleX: number,
  scaleY: number,
  _uniformScale: number,
): ImageElement {
  return {
    ...element,
    x: scaleNumber(element.x, scaleX),
    y: scaleNumber(element.y, scaleY),
    width: scaleNumber(element.width, scaleX),
    height: scaleNumber(element.height, scaleY),
  }
}

function scaleLineElement(
  element: LineElement,
  scaleX: number,
  scaleY: number,
  uniformScale: number,
): LineElement {
  return {
    ...element,
    x1: scaleNumber(element.x1, scaleX),
    y1: scaleNumber(element.y1, scaleY),
    x2: scaleNumber(element.x2, scaleX),
    y2: scaleNumber(element.y2, scaleY),
    strokeWidth: element.strokeWidth === undefined ? undefined : scaleNumber(element.strokeWidth, uniformScale),
  }
}

function scaleCircleElement(
  element: CircleElement,
  scaleX: number,
  scaleY: number,
  uniformScale: number,
): CircleElement {
  return {
    ...element,
    cx: scaleNumber(element.cx, scaleX),
    cy: scaleNumber(element.cy, scaleY),
    r: scaleNumber(element.r, uniformScale),
    strokeWidth: element.strokeWidth === undefined ? undefined : scaleNumber(element.strokeWidth, uniformScale),
  }
}

function scaleGroupElement(
  element: GroupElement,
  scaleX: number,
  scaleY: number,
  uniformScale: number,
): GroupElement {
  return {
    ...element,
    fontSize: element.fontSize === undefined ? undefined : scaleNumber(element.fontSize, uniformScale),
    children: element.children.map(child => scaleElement(child, scaleX, scaleY, uniformScale)),
  }
}

function scaleFontShorthand(font: string, scale: number): string {
  return font.replace(/(\d+(?:\.\d+)?)px/, (_, size: string) => {
    const nextSize = scaleNumber(Number(size), scale)
    return `${trimTrailingZeros(nextSize)}px`
  })
}

function scaleNumber(value: number, scale: number): number {
  return Number((value * scale).toFixed(3))
}

function trimTrailingZeros(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/\.?0+$/, '')
}
