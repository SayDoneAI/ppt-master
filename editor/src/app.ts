import * as pretext from '@chenglou/pretext'
import type {
  AiCommand,
  CircleElement,
  DesignPatch,
  Element as SlideElement,
  GroupElement,
  ImageElement,
  LineElement,
  PatchOperation,
  PathElement,
  RectElement,
  Slide,
  SlideState,
  TextElement,
} from './slide_state.js'
import {
  applyDesignPatch,
  createAiCommand,
  createAiCommandDownload,
  createAiCommandPromptDownload,
  createDesignPatchDownload,
  createUpdatePatch,
  getDesignPatchPrimarySlideIndex,
  hasPatchValueChanged,
  parseDesignPatchJson,
} from './design_patch.js'
import {
  createHistoryEntry,
  flattenHistoryOperations,
  type HistoryEntry,
} from './history.js'
import {
  createAppendSlidesPatch,
  ensureCompatibleCanvas,
} from './slide_import.js'
import type { DownloadArtifact } from './state_io.js'
import {
  createJsonDownload,
  createSvgDownloads,
  getStatePathFromSearch,
  getSvgPathsFromSearch,
  isJsonFile,
  isSvgFile,
  parseSlideStateJson,
  readSlideStateFile,
  readSvgFiles,
} from './state_io.js'
import { STATE_WATCHER_HMR_EVENT } from './state_sync_events.js'
import { initPretext, layoutText, slideToSvg } from './state_to_svg.js'
import { svgsToState } from './svg_to_state.js'

type EditableElement =
  | TextElement
  | RectElement
  | PathElement
  | ImageElement
  | LineElement
  | CircleElement
  | GroupElement

type SupportedEditableElement =
  | TextElement
  | RectElement
  | PathElement
  | ImageElement
  | LineElement
  | CircleElement

type TransformableElement =
  | TextElement
  | RectElement
  | ImageElement
  | LineElement
  | CircleElement

interface InspectorField {
  key: string
  label: string
  input: 'number' | 'text' | 'textarea'
  readOnly?: boolean
  step?: string
}

interface SvgBounds {
  x: number
  y: number
  width: number
  height: number
}

interface HiddenCanvasNode {
  node: SVGGraphicsElement
  opacity: string | null
}

interface ActiveTextEditor {
  elementId: string
  foreignObject: SVGForeignObjectElement
  wrapper: HTMLDivElement
  textarea: HTMLTextAreaElement
  status: HTMLDivElement
  hiddenNodes: HiddenCanvasNode[]
  bounds: SvgBounds
  initialText: string
}

interface ParsedFontSpec {
  fontSize: number
  fontFamily: string
  fontWeight?: string
  fontStyle?: string
}

type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'

interface SvgPoint {
  x: number
  y: number
}

interface TextNodeSnapshot {
  node: SVGGraphicsElement
  x: number | null
  y: number | null
}

interface PointerInteractionSession {
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

interface StateWatcherPayload {
  filePath?: string
  urlPath?: string
}

initPretext(pretext)

const SVG_NS = 'http://www.w3.org/2000/svg'
const HANDLE_SIZE = 10
const OVERLAY_PADDING = 4
const TEXT_EDITOR_MIN_WIDTH = 120
const TEXT_EDITOR_MIN_HEIGHT = 44
const TEXT_EDITOR_STATUS_HEIGHT = 24
const TEXT_EDITOR_HEIGHT_PADDING = 12
const DRAG_THRESHOLD_PX = 3
const MIN_RESIZE_SIZE = 12
const MIN_TEXT_WIDTH = 40
const MIN_CIRCLE_DIAMETER = 12

const canvasMount = getElement<HTMLDivElement>('canvasMount')
const canvasScroll = getElement<HTMLDivElement>('canvasScroll')
const canvasPane = canvasMount.closest<HTMLElement>('.canvas-pane')
const prevButton = getElement<HTMLButtonElement>('prevSlideBtn')
const nextButton = getElement<HTMLButtonElement>('nextSlideBtn')
const undoButton = getElement<HTMLButtonElement>('undoBtn')
const redoButton = getElement<HTMLButtonElement>('redoBtn')
const saveJsonButton = getElement<HTMLButtonElement>('saveJsonBtn')
const exportSvgButton = getElement<HTMLButtonElement>('exportSvgBtn')
const watchFileButton = getElement<HTMLButtonElement>('watchFileBtn')
const exportPatchButton = getElement<HTMLButtonElement>('exportPatchBtn')
const pageIndicator = getElement<HTMLDivElement>('pageIndicator')
const stateSourceBadge = getElement<HTMLDivElement>('stateSourceBadge')
const thumbnailStrip = getElement<HTMLDivElement>('thumbnailStrip')
const slideMeta = getElement<HTMLDivElement>('slideMeta')
const elementSummary = getElement<HTMLDivElement>('elementSummary')
const selectionSummary = getElement<HTMLDivElement>('selectionSummary')
const selectionEmptyState = getElement<HTMLDivElement>('selectionEmptyState')
const elementFields = getElement<HTMLFormElement>('elementFields')
const patchCountBadge = getElement<HTMLDivElement>('patchCountBadge')
const aiTargetSummary = getElement<HTMLDivElement>('aiTargetSummary')
const aiInstructionInput = getElement<HTMLTextAreaElement>('aiInstructionInput')
const exportAiTaskButton = getElement<HTMLButtonElement>('exportAiTaskBtn')
const applyAiPatchButton = getElement<HTMLButtonElement>('applyAiPatchBtn')
const aiHandoffStatus = getElement<HTMLDivElement>('aiHandoffStatus')
const aiPatchFileInput = getElement<HTMLInputElement>('aiPatchFileInput')
const assetImportSummary = getElement<HTMLDivElement>('assetImportSummary')
const importTemplateButton = getElement<HTMLButtonElement>('importTemplateBtn')
const importChartButton = getElement<HTMLButtonElement>('importChartBtn')
const templateImportInput = getElement<HTMLInputElement>('templateImportInput')
const chartImportInput = getElement<HTMLInputElement>('chartImportInput')
const assetLibraryStatus = getElement<HTMLDivElement>('assetLibraryStatus')
const dropZoneOverlay = getElement<HTMLDivElement>('dropZoneOverlay')

let state = createDemoState()
let stateSourceLabel = '当前数据：内置 Demo'
let currentSlideIndex = 0
let hoveredElementId: string | null = null
let selectedElementId: string | null = null
let editingTextId: string | null = null
let activeTextEditor: ActiveTextEditor | null = null
let suppressNextCanvasClick = false
let interactionHint: string | null = null
let pointerSession: PointerInteractionSession | null = null
let dragDepth = 0
let patches: PatchOperation[] = []
let historyPast: HistoryEntry[] = []
let historyFuture: HistoryEntry[] = []
let watchedStatePath = getStatePathFromSearch(window.location.search)
let watchedStateRawSnapshot: string | null = null
let manualWatchTimer: number | null = null
let aiHandoffStatusMessage = '输入自然语言后导出本地 AI handoff；优先交给 Claude Code / Codex 在项目内直接修改，若只返回 design_patch.json 也可应用回当前页面。'
let aiHandoffStatusTone: 'default' | 'error' = 'default'
let assetLibraryStatusMessage = '支持把模板页或图表 SVG 直接追加到当前项目；导入时会先转为 slide_state，再进入同一套 patch / AI handoff / render 工作流。'
let assetLibraryStatusTone: 'default' | 'error' = 'default'

bindCanvasBlankInteractions()
bindInspectorInteractions()
bindToolbarFileActions()
bindAiHandoffInteractions()
bindAssetImportInteractions()
bindGlobalDropZone()
bindDevServerStateSync()
document.addEventListener('pointerdown', handleDocumentPointerDown, true)
document.addEventListener('pointermove', handleDocumentPointerMove, true)
document.addEventListener('pointerup', handleDocumentPointerUp, true)
document.addEventListener('pointercancel', handleDocumentPointerUp, true)
render()
void loadInitialStateFromUrl()

prevButton.addEventListener('click', () => goToSlide(currentSlideIndex - 1))
nextButton.addEventListener('click', () => goToSlide(currentSlideIndex + 1))

window.addEventListener('keydown', event => {
  if (isUndoShortcut(event)) {
    if (isFormField(event.target)) return
    event.preventDefault()
    undoLastChange()
    return
  }

  if (isRedoShortcut(event)) {
    if (isFormField(event.target)) return
    event.preventDefault()
    redoLastChange()
    return
  }

  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    if (isFormField(event.target)) return
    event.preventDefault()
    if (event.shiftKey) downloadArtifacts(createSvgDownloads(state))
    else downloadArtifacts([createJsonDownload(state)])
    return
  }

  if (event.key === 'Escape') {
    if (editingTextId) {
      exitTextEditing()
      event.preventDefault()
      return
    }

    if (selectedElementId) {
      clearSelection()
      if (isFormField(event.target)) event.target.blur()
      event.preventDefault()
    }
    return
  }

  if (isFormField(event.target)) return
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault()
    exportAiHandoff()
    return
  }
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    goToSlide(currentSlideIndex - 1)
  }
  if (event.key === 'ArrowRight') {
    event.preventDefault()
    goToSlide(currentSlideIndex + 1)
  }
})

function render(): void {
  const slide = state.slides[currentSlideIndex]
  syncInteractionState(slide)
  renderCanvas(slide, currentSlideIndex)
  renderMeta(slide)
  renderThumbnails()
  renderInspector()
  renderPatchCountBadge()
  renderAiHandoffPanel()
  renderAssetImportPanel()
  pageIndicator.textContent = `${currentSlideIndex + 1} / ${state.slides.length}`
  prevButton.disabled = currentSlideIndex === 0
  nextButton.disabled = currentSlideIndex === state.slides.length - 1
  syncHistoryButtons()
  stateSourceBadge.textContent = stateSourceLabel
  document.title = `PPT Master Editor · ${slide.id}`
}

function goToSlide(index: number): void {
  if (editingTextId) exitTextEditing({ shouldRender: false })
  const nextIndex = clamp(index, 0, state.slides.length - 1)
  if (nextIndex === currentSlideIndex) return
  currentSlideIndex = nextIndex
  render()
}

function renderCanvas(slide: Slide, index: number): void {
  canvasMount.innerHTML = namespaceSvgIds(slideToSvg(slide, state.canvas), `canvas-${index}`)
  bindCanvasElementInteractions()
  refreshCanvasOverlays()
}

function renderMeta(slide: Slide): void {
  const counts = countElements(slide.elements)
  const items = [
    { label: 'Slide ID', value: slide.id },
    { label: '画布尺寸', value: `${state.canvas.width} × ${state.canvas.height}` },
    { label: '顶层元素', value: String(slide.elements.length) },
    { label: 'Defs', value: String(slide.defs?.length ?? 0) },
  ]

  slideMeta.innerHTML = items.map(item => `
    <div class="meta-item">
      <strong>${escapeHtml(item.label)}</strong>
      <span>${escapeHtml(item.value)}</span>
    </div>
  `).join('')

  const orderedTypes: Array<keyof ElementCounts> = ['rect', 'text', 'path', 'line', 'circle', 'image', 'group']
  elementSummary.innerHTML = orderedTypes
    .filter(type => counts[type] > 0)
    .map(type => `<div class="chip">${type} × ${counts[type]}</div>`)
    .join('')
}

function renderThumbnails(): void {
  thumbnailStrip.innerHTML = state.slides.map((slide, index) => {
    const svg = namespaceSvgIds(slideToSvg(slide, state.canvas), `thumb-${index}`)
    const isActive = index === currentSlideIndex ? ' is-active' : ''
    return `
      <button type="button" class="thumbnail${isActive}" data-slide-index="${index}" aria-label="切换到第 ${index + 1} 页">
        <div class="thumbnail__preview">${svg}</div>
        <div class="thumbnail__label">
          <strong>${escapeHtml(getSlideLabel(slide))}</strong>
          <span>${String(index + 1).padStart(2, '0')}</span>
        </div>
      </button>
    `
  }).join('')

  for (const button of Array.from(thumbnailStrip.querySelectorAll<HTMLButtonElement>('[data-slide-index]'))) {
    button.addEventListener('click', () => {
      const value = Number(button.dataset.slideIndex)
      if (!Number.isNaN(value)) goToSlide(value)
    })
  }
}

function getSlideLabel(slide: Slide): string {
  const title = findFirstText(slide.elements)
  return title ? title.split('\n')[0] : slide.id
}

function renderInspector(): void {
  const selected = getSelectedElement()

  if (!selected) {
    interactionHint = null
    selectionSummary.innerHTML = ''
    selectionEmptyState.hidden = false
    elementFields.hidden = true
    elementFields.innerHTML = ''
    return
  }

  selectionSummary.innerHTML = buildSelectionSummary(selected)
  selectionEmptyState.hidden = true
  elementFields.hidden = false

  if (!isSupportedEditableElement(selected)) {
    elementFields.innerHTML = `
      <div class="selection-empty">
        当前选中的是 group 容器。请直接点击内部具体元素进行属性编辑。
      </div>
    `
    return
  }

  const fields = getInspectorFields(selected)
  elementFields.innerHTML = fields
    .map(field => renderInspectorField(field, getFieldValue(selected, field.key)))
    .join('')
}

function renderAiHandoffPanel(): void {
  const slide = getCurrentSlide()
  const selected = getSelectedElement()
  const scopeLabel = selected ? '当前选中元素' : '当前页面'
  const elementLabel = selected ? selected.id : '页面级'
  const projectPathHint = inferProjectPathHint(watchedStatePath)
  const projectLabel = projectPathHint ?? '未绑定本地项目'

  aiTargetSummary.innerHTML = `
    <div class="selection-summary__meta">
      <span>${escapeHtml(`范围 · ${scopeLabel}`)}</span>
      <span>${escapeHtml(`Slide · ${slide.id}`)}</span>
      <span>${escapeHtml(`目标 · ${elementLabel}`)}</span>
      <span>${escapeHtml(`项目 · ${projectLabel}`)}</span>
    </div>
  `

  exportAiTaskButton.disabled = aiInstructionInput.value.trim().length === 0
  aiHandoffStatus.textContent = aiHandoffStatusMessage
  aiHandoffStatus.dataset.tone = aiHandoffStatusTone
}

function renderAssetImportPanel(): void {
  assetImportSummary.innerHTML = `
    <div class="selection-summary__meta">
      <span>${escapeHtml(`插入位置 · 第 ${currentSlideIndex + 1} 页后`)}</span>
      <span>${escapeHtml(`当前画布 · ${state.canvas.width} × ${state.canvas.height}`)}</span>
      <span>导入格式 · SVG / slide_state JSON</span>
    </div>
  `
  assetLibraryStatus.textContent = assetLibraryStatusMessage
  assetLibraryStatus.dataset.tone = assetLibraryStatusTone
}

function buildSelectionSummary(element: EditableElement): string {
  const typeLabel = `类型 · ${element.type}`
  const idLabel = `ID · ${element.id}`
  const extra = element.type === 'group'
    ? `子元素 · ${element.children.length}`
    : getElementPositionSummary(element)
  const hintMarkup = interactionHint
    ? `<div class="selection-summary__hint">${escapeHtml(interactionHint)}</div>`
    : ''

  return `
    <div class="selection-summary__meta">
      <span>${escapeHtml(typeLabel)}</span>
      <span>${escapeHtml(idLabel)}</span>
      <span>${escapeHtml(extra)}</span>
    </div>
    ${hintMarkup}
  `
}

function getElementPositionSummary(element: SupportedEditableElement): string {
  switch (element.type) {
    case 'text':
    case 'rect':
    case 'image':
      return `位置 · ${formatNumber(element.x)}, ${formatNumber(element.y)}`
    case 'line':
      return `起点 · ${formatNumber(element.x1)}, ${formatNumber(element.y1)}`
    case 'circle':
      return `圆心 · ${formatNumber(element.cx)}, ${formatNumber(element.cy)}`
    case 'path':
      return `颜色 · ${(element.fill || element.stroke || '未设置')}`
  }
}

function getInspectorFields(element: SupportedEditableElement): InspectorField[] {
  switch (element.type) {
    case 'text':
      return [
        { key: 'x', label: 'x', input: 'number', step: 'any' },
        { key: 'y', label: 'y', input: 'number', step: 'any' },
        { key: 'width', label: 'width', input: 'number', step: 'any' },
        { key: 'text', label: 'text 内容', input: 'textarea' },
        { key: 'font', label: 'font', input: 'text' },
        { key: 'lineHeight', label: 'lineHeight', input: 'number', step: 'any' },
        { key: 'fill', label: 'fill', input: 'text' },
      ]
    case 'rect':
      return [
        { key: 'x', label: 'x', input: 'number', step: 'any' },
        { key: 'y', label: 'y', input: 'number', step: 'any' },
        { key: 'width', label: 'width', input: 'number', step: 'any' },
        { key: 'height', label: 'height', input: 'number', step: 'any' },
        { key: 'fill', label: 'fill', input: 'text' },
        { key: 'rx', label: 'rx', input: 'number', step: 'any' },
      ]
    case 'path':
      return [
        { key: 'fill', label: 'fill', input: 'text' },
        { key: 'd', label: 'd', input: 'textarea', readOnly: true },
      ]
    case 'image':
      return [
        { key: 'x', label: 'x', input: 'number', step: 'any' },
        { key: 'y', label: 'y', input: 'number', step: 'any' },
        { key: 'width', label: 'width', input: 'number', step: 'any' },
        { key: 'height', label: 'height', input: 'number', step: 'any' },
        { key: 'href', label: 'href', input: 'text' },
      ]
    case 'line':
      return [
        { key: 'x1', label: 'x1', input: 'number', step: 'any' },
        { key: 'y1', label: 'y1', input: 'number', step: 'any' },
        { key: 'x2', label: 'x2', input: 'number', step: 'any' },
        { key: 'y2', label: 'y2', input: 'number', step: 'any' },
        { key: 'stroke', label: 'stroke', input: 'text' },
      ]
    case 'circle':
      return [
        { key: 'cx', label: 'cx', input: 'number', step: 'any' },
        { key: 'cy', label: 'cy', input: 'number', step: 'any' },
        { key: 'r', label: 'r', input: 'number', step: 'any' },
        { key: 'fill', label: 'fill', input: 'text' },
      ]
  }
}

function renderInspectorField(field: InspectorField, value: string): string {
  const readonlyAttr = field.readOnly ? ' readonly' : ''
  const stepAttr = field.step ? ` step="${field.step}"` : ''
  const id = `prop-${field.key}`

  if (field.input === 'textarea') {
    return `
      <div class="property-field">
        <label for="${id}">${escapeHtml(field.label)}</label>
        <textarea id="${id}" data-prop-key="${field.key}"${readonlyAttr}>${escapeHtml(value)}</textarea>
      </div>
    `
  }

  const type = field.input === 'number' ? 'number' : 'text'
  return `
    <div class="property-field">
      <label for="${id}">${escapeHtml(field.label)}</label>
      <input id="${id}" data-prop-key="${field.key}" type="${type}" value="${escapeHtml(value)}"${readonlyAttr}${stepAttr} />
    </div>
  `
}

function bindCanvasBlankInteractions(): void {
  ;[canvasPane, canvasScroll].filter(Boolean).forEach(target => {
    target!.addEventListener('click', event => {
      if (consumeSuppressedCanvasInteraction(event)) return
      if (editingTextId) return
      if (!(event.target instanceof Element)) return
      if (event.target.closest('[data-element-id]')) return
      if (event.target.closest('[data-editor-handle]')) return
      clearSelection()
    })
  })
}

function bindCanvasElementInteractions(): void {
  for (const node of getCanvasElementNodes()) {
    node.addEventListener('mouseenter', handleCanvasElementMouseEnter)
    node.addEventListener('mouseleave', handleCanvasElementMouseLeave)
    node.addEventListener('pointerdown', handleCanvasElementPointerDown)
    node.addEventListener('click', handleCanvasElementClick)
    node.addEventListener('dblclick', handleCanvasElementDoubleClick)
  }
}

function bindInspectorInteractions(): void {
  elementFields.addEventListener('input', event => {
    const target = event.target
    if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) return
    if (target.readOnly) return

    const key = target.dataset.propKey
    if (!key) return

    const selected = getSelectedElement()
    if (!selected || !isSupportedEditableElement(selected)) return

    const oldValue = getPatchablePropertyValue(selected, key)
    if (!applyElementUpdate(selected, key, target.value)) return
    const operation = createPropertyPatch(selected.id, key, oldValue, getPatchablePropertyValue(selected, key))
    if (operation) {
      commitPatchOperations([operation], 'human', `修改 ${selected.id}.${key}`)
    }

    interactionHint = null
    syncInspectorPreview(selected)
    const slide = getCurrentSlide()
    renderCanvas(slide, currentSlideIndex)
    renderThumbnails()
  })
}

function bindToolbarFileActions(): void {
  undoButton.addEventListener('click', () => {
    undoLastChange()
  })

  redoButton.addEventListener('click', () => {
    redoLastChange()
  })

  saveJsonButton.addEventListener('click', () => {
    downloadArtifacts([createJsonDownload(state)])
  })

  exportSvgButton.addEventListener('click', () => {
    downloadArtifacts(createSvgDownloads(state))
  })

  exportPatchButton.addEventListener('click', () => {
    downloadArtifacts([createDesignPatchDownload(patches)])
  })

  watchFileButton.addEventListener('click', () => {
    const defaultPath = watchedStatePath ?? ''
    const input = window.prompt('请输入可通过 fetch() 访问的 slide_state.json 路径', defaultPath)
    const nextPath = input?.trim()
    if (!nextPath) return
    void startManualStateWatch(nextPath).catch(error => {
      console.error(`启动文件监听失败: ${nextPath}`, error)
    })
  })
}

function bindAssetImportInteractions(): void {
  importTemplateButton.addEventListener('click', () => {
    templateImportInput.click()
  })

  importChartButton.addEventListener('click', () => {
    chartImportInput.click()
  })

  templateImportInput.addEventListener('change', () => {
    const files = Array.from(templateImportInput.files ?? [])
    templateImportInput.value = ''
    if (files.length === 0) return
    void appendImportedSlidesFromFiles(files, '模板页', true)
  })

  chartImportInput.addEventListener('change', () => {
    const files = Array.from(chartImportInput.files ?? [])
    chartImportInput.value = ''
    if (files.length === 0) return
    void appendImportedSlidesFromFiles(files, '图表页', false)
  })
}

function bindAiHandoffInteractions(): void {
  aiInstructionInput.addEventListener('input', () => {
    aiHandoffStatusTone = 'default'
    aiHandoffStatusMessage = '输入自然语言后导出本地 AI handoff；优先交给 Claude Code / Codex 在项目内直接修改，若只返回 design_patch.json 也可应用回当前页面。'
    renderAiHandoffPanel()
  })

  aiInstructionInput.addEventListener('keydown', event => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      exportAiHandoff()
    }
  })

  exportAiTaskButton.addEventListener('click', () => {
    exportAiHandoff()
  })

  applyAiPatchButton.addEventListener('click', () => {
    aiPatchFileInput.click()
  })

  aiPatchFileInput.addEventListener('change', () => {
    const [file] = Array.from(aiPatchFileInput.files ?? [])
    aiPatchFileInput.value = ''
    if (!file) return
    void loadDesignPatchFromFile(file)
  })
}

function bindGlobalDropZone(): void {
  window.addEventListener('dragenter', event => {
    if (!hasFileTransfer(event.dataTransfer)) return
    event.preventDefault()
    dragDepth += 1
    setDropZoneActive(true)
  })

  window.addEventListener('dragover', event => {
    if (!hasFileTransfer(event.dataTransfer)) return
    event.preventDefault()
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'
    setDropZoneActive(true)
  })

  window.addEventListener('dragleave', event => {
    if (!hasFileTransfer(event.dataTransfer)) return
    event.preventDefault()
    dragDepth = Math.max(0, dragDepth - 1)
    if (dragDepth === 0) setDropZoneActive(false)
  })

  window.addEventListener('drop', event => {
    if (!hasFileTransfer(event.dataTransfer)) return
    event.preventDefault()
    dragDepth = 0
    setDropZoneActive(false)

    const files = Array.from(event.dataTransfer?.files ?? [])
    const jsonFile = files.find(file => isJsonFile(file))
    if (jsonFile) {
      void loadJsonAssetFromFile(jsonFile)
      return
    }

    const svgFiles = files.filter(file => isSvgFile(file))
    if (svgFiles.length > 0) {
      void loadStateFromSvgFiles(svgFiles)
      return
    }

    console.error('拖拽加载失败：未检测到 .json 或 .svg 文件')
  })
}

async function loadInitialStateFromUrl(): Promise<void> {
  const statePath = getStatePathFromSearch(window.location.search)
  if (statePath) {
    watchedStatePath = statePath
    try {
      await reloadStateFromRemote(statePath, `当前数据：URL ${statePath}`, statePath)
      return
    } catch (error) {
      console.error(`URL 加载 slide_state 失败: ${statePath}`, error)
      stateSourceLabel = `当前数据：内置 Demo（URL 加载失败）`
      render()
      return
    }
  }

  const svgPaths = getSvgPathsFromSearch(window.location.search)
  if (svgPaths.length === 0) return

  try {
    const nextState = await loadStateFromSvgUrls(svgPaths)
    replaceState(nextState, `当前数据：SVG 导入 (${svgPaths.length} 页)`)
  } catch (error) {
    console.error(`URL 加载 SVG 失败: ${svgPaths.join(', ')}`, error)
    stateSourceLabel = '当前数据：内置 Demo（SVG URL 加载失败）'
    render()
  }
}

async function loadJsonAssetFromFile(file: File): Promise<void> {
  try {
    const rawJson = await file.text()

    try {
      const designPatch = parseDesignPatchJson(rawJson)
      handleImportedDesignPatch(designPatch, file.name)
      return
    } catch {
      const nextState = parseSlideStateJson(rawJson)
      stopManualStateWatch()
      watchedStatePath = null
      watchedStateRawSnapshot = null
      replaceState(nextState, `当前数据：${file.name}`)
    }
  } catch (error) {
    console.error(`文件加载 JSON 失败: ${file.name}`, error)
    aiHandoffStatusTone = 'error'
    aiHandoffStatusMessage = `JSON 导入失败：${(error as Error).message}`
    renderAiHandoffPanel()
  }
}

async function loadDesignPatchFromFile(file: File): Promise<void> {
  try {
    const designPatch = parseDesignPatchJson(await file.text())
    handleImportedDesignPatch(designPatch, file.name)
  } catch (error) {
    console.error(`design patch 导入失败: ${file.name}`, error)
    aiHandoffStatusTone = 'error'
    aiHandoffStatusMessage = `Patch 导入失败：${(error as Error).message}`
    renderAiHandoffPanel()
  }
}

async function loadStateFromSvgFiles(files: File[]): Promise<void> {
  try {
    const nextState = await readSvgFiles(files)
    stopManualStateWatch()
    watchedStatePath = null
    watchedStateRawSnapshot = null
    replaceState(nextState, `当前数据：SVG 导入 (${nextState.slides.length} 页)`)
  } catch (error) {
    console.error('SVG 文件导入失败', error)
  }
}

async function appendImportedSlidesFromFiles(
  files: File[],
  label: '模板页' | '图表页',
  allowJson: boolean,
): Promise<void> {
  try {
    if (editingTextId) exitTextEditing({ shouldRender: false })
    const importedState = await readImportedSlidesState(files, allowJson)
    ensureCompatibleCanvas(state.canvas, importedState.canvas)

    stopManualStateWatch()
    watchedStatePath = null
    watchedStateRawSnapshot = null

    const insertIndex = currentSlideIndex + 1
    const operations = createAppendSlidesPatch({
      state,
      importedSlides: importedState.slides,
      insertIndex,
      source: 'human',
    })

    if (operations.length === 0) {
      throw new Error(`未检测到可追加的${label}`)
    }

    state = applyDesignPatch(state, {
      timestamp: new Date().toISOString(),
      source: 'human',
      operations,
    })
    commitPatchOperations(operations, 'human', `追加${label}`)
    currentSlideIndex = clamp(insertIndex, 0, state.slides.length - 1)
    stateSourceLabel = `当前数据：已追加${label} (${operations.length} 页)`
    assetLibraryStatusTone = 'default'
    assetLibraryStatusMessage = `已追加 ${operations.length} 页${label}，可继续编辑、导出 AI handoff，或直接 render / finalize / 导出 PPT。`
    interactionHint = `已追加${label}：${operations.length} 页`
    resetTransientInteractionState({ clearSelection: true, clearHint: false })
    render()
  } catch (error) {
    console.error(`${label} 导入失败`, error)
    assetLibraryStatusTone = 'error'
    assetLibraryStatusMessage = `${label}导入失败：${(error as Error).message}`
    renderAssetImportPanel()
  }
}

async function readImportedSlidesState(files: File[], allowJson: boolean): Promise<SlideState> {
  if (allowJson) {
    const jsonFile = files.find(file => isJsonFile(file))
    if (jsonFile) return readSlideStateFile(jsonFile)
  }

  const svgFiles = files.filter(file => isSvgFile(file))
  if (svgFiles.length === 0) {
    throw new Error('请选择 .svg 文件，或使用 slide_state JSON 模板')
  }

  return readSvgFiles(svgFiles)
}

function exportAiHandoff(): void {
  const instruction = aiInstructionInput.value.trim()
  if (!instruction) {
    aiInstructionInput.focus()
    aiHandoffStatusTone = 'error'
    aiHandoffStatusMessage = '请先输入 AI 指令，再导出本地 AI handoff。'
    renderAiHandoffPanel()
    return
  }

  const aiCommand = createEditorAiCommand(instruction)
  const projectPathHint = inferProjectPathHint(watchedStatePath)
  const stateFilePathHint = inferStateFilePathHint(watchedStatePath)

  downloadArtifacts([
    createAiCommandDownload(aiCommand, patches),
    createAiCommandPromptDownload(aiCommand, patches, {
      projectPathHint,
      stateFilePathHint,
    }),
  ])

  interactionHint = aiCommand.scope === 'selected-element'
    ? `已导出本地 AI handoff，可交给 Claude Code / Codex 在项目内修改元素 ${aiCommand.elementId}`
    : '已导出本地 AI handoff，可交给 Claude Code / Codex 在项目内修改当前页面'
  aiHandoffStatusTone = 'default'
  aiHandoffStatusMessage = aiCommand.scope === 'selected-element'
    ? `已导出本地 AI handoff，目标元素 ${aiCommand.elementId}。Claude Code / Codex 可直接修改项目；若只返回 design_patch.json，也可拖入或点击“应用 AI Patch”。`
    : '已导出页面级本地 AI handoff。Claude Code / Codex 可直接修改项目；若只返回 design_patch.json，也可拖入或点击“应用 AI Patch”。'

  renderInspector()
  renderAiHandoffPanel()
}

function handleImportedDesignPatch(designPatch: DesignPatch, sourceLabel: string): void {
  if (designPatch.operations.length === 0) {
    if (designPatch.aiCommand) {
      hydrateAiCommandFromPatch(designPatch.aiCommand)
      aiHandoffStatusTone = 'default'
      aiHandoffStatusMessage = `已载入 AI handoff 请求：${describeAiScope(designPatch.aiCommand)}`
      render()
      return
    }

    throw new Error('design_patch 不包含可应用的 operations')
  }

  stopManualStateWatch()
  watchedStatePath = null
  watchedStateRawSnapshot = null
  if (editingTextId) exitTextEditing({ shouldRender: false })

  const nextState = applyDesignPatch(state, designPatch)
  state = nextState
  stateSourceLabel = `当前数据：已应用 ${sourceLabel}`
  commitPatchOperations(
    designPatch.operations.map(operation => cloneSerializableValue(operation)),
    designPatch.source,
    `应用 ${sourceLabel}`,
  )

  const targetSlideIndex = getDesignPatchPrimarySlideIndex(designPatch)
  if (targetSlideIndex !== null) {
    currentSlideIndex = clamp(targetSlideIndex, 0, nextState.slides.length - 1)
  }

  if (designPatch.aiCommand) {
    hydrateAiCommandFromPatch(designPatch.aiCommand)
  }

  interactionHint = `${designPatch.source === 'ai' ? 'AI' : 'Patch'} 已应用：${designPatch.operations.length} 条操作`
  resetTransientInteractionState({ clearSelection: false, clearHint: false })

  aiHandoffStatusTone = 'default'
  aiHandoffStatusMessage = `已应用 ${designPatch.operations.length} 条 Patch：${sourceLabel}`
  render()
}

function hydrateAiCommandFromPatch(aiCommand: AiCommand): void {
  currentSlideIndex = clamp(aiCommand.slideIndex, 0, state.slides.length - 1)
  selectedElementId = aiCommand.elementId
  aiInstructionInput.value = aiCommand.instruction
}

function describeAiScope(aiCommand: AiCommand): string {
  return aiCommand.scope === 'selected-element' && aiCommand.elementId
    ? `元素 ${aiCommand.elementId} @ ${aiCommand.slideId}`
    : `页面 ${aiCommand.slideId}`
}

function bindDevServerStateSync(): void {
  import.meta.hot?.on(STATE_WATCHER_HMR_EVENT, (payload: StateWatcherPayload) => {
    const targetPath = watchedStatePath ?? payload.urlPath
    if (!targetPath) return
    watchedStatePath = targetPath
    void reloadStateFromRemote(
      targetPath,
      `已自动加载: ${payload.filePath ?? targetPath}`,
      payload.filePath ?? targetPath,
    ).catch(error => {
      console.error(`热更新加载 slide_state 失败: ${payload.filePath ?? targetPath}`, error)
    })
  })
}

async function startManualStateWatch(path: string): Promise<void> {
  stopManualStateWatch()
  watchedStatePath = path
  watchedStateRawSnapshot = null
  await reloadStateFromRemote(path, `已自动加载: ${path}`, path)
  manualWatchTimer = window.setInterval(() => {
    void pollManualWatch()
  }, 1500)
}

function stopManualStateWatch(): void {
  if (manualWatchTimer === null) return
  window.clearInterval(manualWatchTimer)
  manualWatchTimer = null
}

async function pollManualWatch(): Promise<void> {
  if (!watchedStatePath) return

  try {
    const { nextState, rawJson } = await fetchRemoteSlideState(watchedStatePath)
    if (watchedStateRawSnapshot === rawJson) return
    watchedStateRawSnapshot = rawJson
    replaceState(nextState, `已自动加载: ${watchedStatePath}`)
  } catch (error) {
    console.error(`轮询 slide_state 失败: ${watchedStatePath}`, error)
  }
}

async function reloadStateFromRemote(fetchPath: string, sourceLabel: string, displayPath: string): Promise<void> {
  const { nextState, rawJson } = await fetchRemoteSlideState(fetchPath)
  watchedStateRawSnapshot = rawJson
  replaceState(nextState, sourceLabel.includes(fetchPath) ? sourceLabel.replace(fetchPath, displayPath) : sourceLabel)
}

async function fetchRemoteSlideState(fetchPath: string): Promise<{ nextState: SlideState; rawJson: string }> {
  const response = await fetch(fetchPath, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`.trim())
  }

  const rawJson = await response.text()
  return {
    nextState: parseSlideStateJson(rawJson),
    rawJson,
  }
}

async function loadStateFromSvgUrls(paths: string[]): Promise<SlideState> {
  const svgStrings = await Promise.all(paths.map(async path => {
    const response = await fetch(path, { cache: 'no-store' })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`.trim())
    }
    return response.text()
  }))

  return svgsToState(svgStrings, paths.map(toSlideIdFromPath))
}

function replaceState(nextState: SlideState, sourceLabel: string): void {
  if (editingTextId) exitTextEditing({ shouldRender: false })

  state = nextState
  stateSourceLabel = sourceLabel
  historyPast = []
  historyFuture = []
  patches = []
  currentSlideIndex = 0
  assetLibraryStatusTone = 'default'
  assetLibraryStatusMessage = '支持把模板页或图表 SVG 直接追加到当前项目；导入时会先转为 slide_state，再进入同一套 patch / AI handoff / render 工作流。'
  resetTransientInteractionState({ clearSelection: true, clearHint: true })
  render()
}

function commitPatchOperations(
  operations: PatchOperation[],
  source: 'human' | 'ai',
  label: string,
): void {
  if (operations.length === 0) return
  historyPast.push(createHistoryEntry(operations, source, label))
  historyFuture = []
  patches = flattenHistoryOperations(historyPast)
  renderPatchCountBadge()
  syncHistoryButtons()
}

function undoLastChange(): void {
  const entry = historyPast.pop()
  if (!entry) return

  if (editingTextId) exitTextEditing({ shouldRender: false })
  state = applyDesignPatch(state, entry.inversePatch)
  historyFuture.unshift(entry)
  patches = flattenHistoryOperations(historyPast)

  const targetSlideIndex = getDesignPatchPrimarySlideIndex(entry.inversePatch)
  if (targetSlideIndex !== null) {
    currentSlideIndex = clamp(targetSlideIndex, 0, state.slides.length - 1)
  } else {
    currentSlideIndex = clamp(currentSlideIndex, 0, state.slides.length - 1)
  }

  stateSourceLabel = `当前数据：已撤销 ${entry.label}`
  interactionHint = `已撤销：${entry.label}`
  aiHandoffStatusTone = 'default'
  aiHandoffStatusMessage = `已撤销一步：${entry.label}`
  resetTransientInteractionState({ clearSelection: false, clearHint: false })
  render()
}

function redoLastChange(): void {
  const entry = historyFuture.shift()
  if (!entry) return

  if (editingTextId) exitTextEditing({ shouldRender: false })
  state = applyDesignPatch(state, entry.forwardPatch)
  historyPast.push(entry)
  patches = flattenHistoryOperations(historyPast)

  const targetSlideIndex = getDesignPatchPrimarySlideIndex(entry.forwardPatch)
  if (targetSlideIndex !== null) {
    currentSlideIndex = clamp(targetSlideIndex, 0, state.slides.length - 1)
  } else {
    currentSlideIndex = clamp(currentSlideIndex, 0, state.slides.length - 1)
  }

  stateSourceLabel = `当前数据：已重做 ${entry.label}`
  interactionHint = `已重做：${entry.label}`
  aiHandoffStatusTone = 'default'
  aiHandoffStatusMessage = `已重做一步：${entry.label}`
  resetTransientInteractionState({ clearSelection: false, clearHint: false })
  render()
}

function resetTransientInteractionState(
  options: { clearSelection?: boolean; clearHint?: boolean } = {},
): void {
  hoveredElementId = null
  editingTextId = null
  activeTextEditor = null
  suppressNextCanvasClick = false
  pointerSession = null
  dragDepth = 0
  setDropZoneActive(false)
  if (options.clearSelection) selectedElementId = null
  if (options.clearHint ?? true) interactionHint = null
}

function syncHistoryButtons(): void {
  undoButton.disabled = historyPast.length === 0
  redoButton.disabled = historyFuture.length === 0
}

function downloadArtifacts(artifacts: DownloadArtifact[]): void {
  for (const artifact of artifacts) {
    const blob = new Blob([artifact.content], { type: artifact.mimeType })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = artifact.fileName
    link.hidden = true
    document.body.append(link)
    link.click()
    link.remove()
    setTimeout(() => URL.revokeObjectURL(url), 0)
  }
}

function setDropZoneActive(active: boolean): void {
  dropZoneOverlay.hidden = !active
  dropZoneOverlay.classList.toggle('is-active', active)
}

function hasFileTransfer(dataTransfer: DataTransfer | null): boolean {
  if (!dataTransfer) return false
  if (dataTransfer.files.length > 0) return true
  return Array.from(dataTransfer.items).some(item => item.kind === 'file')
}

function handleCanvasElementMouseEnter(event: MouseEvent): void {
  if (editingTextId || pointerSession?.started) return
  const elementId = getElementIdFromEventTarget(event.currentTarget)
  if (!elementId || hoveredElementId === elementId) return
  hoveredElementId = elementId
  refreshCanvasOverlays()
}

function handleCanvasElementMouseLeave(event: MouseEvent): void {
  if (editingTextId || pointerSession?.started) return
  const elementId = getElementIdFromEventTarget(event.currentTarget)
  const relatedElementId = getElementIdFromEventTarget(event.relatedTarget)
  if (!elementId || relatedElementId === elementId) return
  if (hoveredElementId !== elementId) return
  hoveredElementId = null
  refreshCanvasOverlays()
}

function handleCanvasElementPointerDown(event: PointerEvent): void {
  if (editingTextId || event.button !== 0) return

  const elementId = getElementIdFromEventTarget(event.currentTarget)
  if (!elementId || elementId !== selectedElementId) return

  const element = findElementById(getCurrentSlide().elements, elementId)
  if (!element) return

  if (!isTransformableElement(element)) {
    interactionHint = element.type === 'path'
      ? 'path 元素暂不支持拖拽；请先通过属性面板或后续 transform 支持处理。'
      : 'group 容器暂不支持拖拽；请直接拖拽内部具体元素。'
    renderInspector()
    return
  }

  const svg = getCanvasSvg()
  if (!svg) return

  const startPoint = clientPointToViewBox(svg, event.clientX, event.clientY)
  const initialBounds = getElementInteractionBounds(element, svg)
  if (!startPoint || !initialBounds) return

  interactionHint = null
  hoveredElementId = elementId
  pointerSession = {
    pointerId: event.pointerId,
    kind: 'move',
    elementId,
    startClientX: event.clientX,
    startClientY: event.clientY,
    startPoint,
    initialBounds,
    initialElement: cloneTransformableElement(element),
    textNodeSnapshots: captureTextNodeSnapshots(element),
    started: false,
  }

  event.preventDefault()
  event.stopPropagation()
}

function handleCanvasElementClick(event: MouseEvent): void {
  if (consumeSuppressedCanvasInteraction(event)) return
  if (editingTextId) return
  event.stopPropagation()
  const elementId = getElementIdFromEventTarget(event.currentTarget)
  if (!elementId) return
  interactionHint = null
  hoveredElementId = elementId
  selectedElementId = elementId
  renderInspector()
  refreshCanvasOverlays()
}

function handleCanvasElementDoubleClick(event: MouseEvent): void {
  if (consumeSuppressedCanvasInteraction(event)) return

  const elementId = getElementIdFromEventTarget(event.currentTarget)
  if (!elementId) return

  const element = findElementById(getCurrentSlide().elements, elementId)
  if (!element || element.type !== 'text') return

  event.preventDefault()
  event.stopPropagation()
  interactionHint = null
  selectedElementId = elementId
  hoveredElementId = null
  renderInspector()
  enterTextEditing(element)
}

function handleDocumentPointerDown(event: PointerEvent): void {
  if (!editingTextId || !activeTextEditor) return
  if (!(event.target instanceof Node)) return
  if (activeTextEditor.foreignObject.contains(event.target)) return

  const shouldSuppressCanvasInteraction = event.target instanceof Element
    && Boolean(event.target.closest('.canvas-pane'))

  if (shouldSuppressCanvasInteraction) suppressNextCanvasClick = true
  exitTextEditing()

  if (shouldSuppressCanvasInteraction) {
    event.preventDefault()
    event.stopPropagation()
  }
}

function handleDocumentPointerMove(event: PointerEvent): void {
  const session = pointerSession
  if (!session || event.pointerId !== session.pointerId) return

  const svg = getCanvasSvg()
  if (!svg) return

  const nextPoint = clientPointToViewBox(svg, event.clientX, event.clientY)
  if (!nextPoint) return

  if (!session.started) {
    const distance = Math.hypot(event.clientX - session.startClientX, event.clientY - session.startClientY)
    if (distance < DRAG_THRESHOLD_PX) return
    session.started = true
  }

  const element = findElementById(getCurrentSlide().elements, session.elementId)
  if (!element || !isTransformableElement(element)) {
    pointerSession = null
    return
  }

  const delta = {
    x: nextPoint.x - session.startPoint.x,
    y: nextPoint.y - session.startPoint.y,
  }

  if (session.kind === 'move') applyMoveFromSession(element, session.initialElement, delta)
  else applyResizeFromSession(element, session, delta)

  hoveredElementId = session.elementId
  syncLiveElementPreview(element, session)
  syncInspectorPreview(element)
  refreshCanvasOverlays()

  event.preventDefault()
  event.stopPropagation()
}

function handleDocumentPointerUp(event: PointerEvent): void {
  const session = pointerSession
  if (!session || event.pointerId !== session.pointerId) return

  pointerSession = null
  if (!session.started) return

  const element = findElementById(getCurrentSlide().elements, session.elementId)
  if (element && isTransformableElement(element)) {
    const operations = recordElementPatchDiffs(session.initialElement, element, getPatchKeysForElement(element))
    if (operations.length > 0) {
      commitPatchOperations(operations, 'human', `${session.kind === 'move' ? '移动' : '缩放'} ${element.id}`)
    }
  }

  suppressNextCanvasClick = true
  hoveredElementId = session.elementId
  interactionHint = null

  const slide = getCurrentSlide()
  renderCanvas(slide, currentSlideIndex)
  renderThumbnails()
  renderInspector()

  event.preventDefault()
  event.stopPropagation()
}

function refreshCanvasOverlays(): void {
  const svg = getCanvasSvg()
  if (!svg) return

  svg.querySelector('[data-editor-overlay-root]')?.remove()
  syncCanvasElementCursors()

  if (editingTextId) return

  const overlayRoot = document.createElementNS(SVG_NS, 'g')
  overlayRoot.setAttribute('data-editor-overlay-root', 'true')
  overlayRoot.setAttribute('pointer-events', 'none')

  if (hoveredElementId && hoveredElementId !== selectedElementId) {
    const hoveredElement = findElementById(getCurrentSlide().elements, hoveredElementId)
    const hoverBounds = hoveredElement
      ? getElementInteractionBounds(hoveredElement, svg)
      : measureElementBounds(svg, hoveredElementId, 0)
    if (hoverBounds) {
      overlayRoot.append(buildOverlayRect(expandBounds(hoverBounds, OVERLAY_PADDING), {
        fill: '#2563EB',
        fillOpacity: '0.14',
        stroke: '#2563EB',
        strokeOpacity: '0.95',
        strokeWidth: '1.5',
        dashArray: '6 4',
      }))
    }
  }

  if (selectedElementId) {
    const selectedElement = getSelectedElement()
    const selectedBounds = selectedElement
      ? getElementInteractionBounds(selectedElement, svg)
      : measureElementBounds(svg, selectedElementId, 0)
    if (selectedBounds) {
      const displayBounds = expandBounds(selectedBounds, OVERLAY_PADDING)
      overlayRoot.append(buildOverlayRect(displayBounds, {
        fill: 'none',
        stroke: '#2563EB',
        strokeOpacity: '1',
        strokeWidth: '2',
      }))

      for (const handle of getHandlePositions(displayBounds)) {
        const knob = document.createElementNS(SVG_NS, 'rect')
        knob.setAttribute('x', String(handle.x - HANDLE_SIZE / 2))
        knob.setAttribute('y', String(handle.y - HANDLE_SIZE / 2))
        knob.setAttribute('width', String(HANDLE_SIZE))
        knob.setAttribute('height', String(HANDLE_SIZE))
        knob.setAttribute('data-editor-handle', handle.position)
        knob.setAttribute('rx', '2')
        knob.setAttribute('fill', '#FFFFFF')
        knob.setAttribute('stroke', '#2563EB')
        knob.setAttribute('stroke-width', '1.5')
        knob.setAttribute('vector-effect', 'non-scaling-stroke')
        knob.setAttribute('pointer-events', 'all')
        knob.style.cursor = getResizeCursor(handle.position)
        knob.addEventListener('pointerdown', handleResizeHandlePointerDown)
        knob.addEventListener('click', stopOverlayHandleClick)
        overlayRoot.append(knob)
      }
    }
  }

  if (overlayRoot.childNodes.length > 0) {
    svg.append(overlayRoot)
  }
}

function buildOverlayRect(
  bounds: SvgBounds,
  attrs: Record<'fill' | 'fillOpacity' | 'stroke' | 'strokeOpacity' | 'strokeWidth' | 'dashArray', string> | {
    fill: string
    stroke: string
    strokeOpacity: string
    strokeWidth: string
    fillOpacity?: string
    dashArray?: string
  },
): SVGRectElement {
  const rect = document.createElementNS(SVG_NS, 'rect')
  rect.setAttribute('x', String(bounds.x))
  rect.setAttribute('y', String(bounds.y))
  rect.setAttribute('width', String(bounds.width))
  rect.setAttribute('height', String(bounds.height))
  rect.setAttribute('fill', attrs.fill)
  rect.setAttribute('stroke', attrs.stroke)
  rect.setAttribute('stroke-width', attrs.strokeWidth)
  rect.setAttribute('stroke-opacity', attrs.strokeOpacity)
  rect.setAttribute('vector-effect', 'non-scaling-stroke')
  if (attrs.fillOpacity) rect.setAttribute('fill-opacity', attrs.fillOpacity)
  if (attrs.dashArray) rect.setAttribute('stroke-dasharray', attrs.dashArray)
  return rect
}

function getHandlePositions(bounds: SvgBounds): Array<{ position: ResizeHandle; x: number; y: number }> {
  const centerX = bounds.x + bounds.width / 2
  const centerY = bounds.y + bounds.height / 2
  return [
    { position: 'nw', x: bounds.x, y: bounds.y },
    { position: 'n', x: centerX, y: bounds.y },
    { position: 'ne', x: bounds.x + bounds.width, y: bounds.y },
    { position: 'e', x: bounds.x + bounds.width, y: centerY },
    { position: 'se', x: bounds.x + bounds.width, y: bounds.y + bounds.height },
    { position: 's', x: centerX, y: bounds.y + bounds.height },
    { position: 'sw', x: bounds.x, y: bounds.y + bounds.height },
    { position: 'w', x: bounds.x, y: centerY },
  ]
}

function measureElementBounds(svg: SVGSVGElement, elementId: string, padding = OVERLAY_PADDING): SvgBounds | null {
  const svgRect = svg.getBoundingClientRect()
  const viewBox = svg.viewBox.baseVal
  if (!svgRect.width || !svgRect.height) return null

  let bounds: SvgBounds | null = null
  for (const node of getCanvasElementNodes(elementId)) {
    const rect = node.getBoundingClientRect()
    if (!rect.width && !rect.height) continue

    const current = {
      x: ((rect.left - svgRect.left) / svgRect.width) * viewBox.width + viewBox.x,
      y: ((rect.top - svgRect.top) / svgRect.height) * viewBox.height + viewBox.y,
      width: (rect.width / svgRect.width) * viewBox.width,
      height: (rect.height / svgRect.height) * viewBox.height,
    }

    bounds = unionBounds(bounds, current)
  }

  if (!bounds) return null

  return {
    x: bounds.x - padding,
    y: bounds.y - padding,
    width: bounds.width + padding * 2,
    height: bounds.height + padding * 2,
  }
}

function unionBounds(left: SvgBounds | null, right: SvgBounds): SvgBounds {
  if (!left) return right

  const minX = Math.min(left.x, right.x)
  const minY = Math.min(left.y, right.y)
  const maxX = Math.max(left.x + left.width, right.x + right.width)
  const maxY = Math.max(left.y + left.height, right.y + right.height)

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  }
}

function syncInteractionState(slide: Slide): void {
  if (!findElementById(slide.elements, selectedElementId)) selectedElementId = null
  if (!findElementById(slide.elements, hoveredElementId)) hoveredElementId = null
  if (!findElementById(slide.elements, editingTextId)) {
    editingTextId = null
    activeTextEditor = null
  }
}

function clearSelection(): void {
  if (editingTextId) exitTextEditing({ shouldRender: false })
  interactionHint = null
  selectedElementId = null
  hoveredElementId = null
  renderInspector()
  refreshCanvasOverlays()
}

function getCurrentSlide(): Slide {
  return state.slides[currentSlideIndex]
}

function getSelectedElement(): EditableElement | null {
  return findElementById(getCurrentSlide().elements, selectedElementId)
}

function createEditorAiCommand(instruction: string): AiCommand {
  const selected = getSelectedElement()
  return createAiCommand({
    state,
    slideIndex: currentSlideIndex,
    instruction,
    scope: selected ? 'selected-element' : 'current-slide',
    elementId: selected?.id ?? null,
  })
}

function inferStateFilePathHint(path: string | null): string | null {
  if (!path) return null

  const trimmedPath = path.trim()
  if (!trimmedPath || /^https?:\/\//i.test(trimmedPath)) return null

  const withoutQuery = trimmedPath.split('?')[0]?.split('#')[0] ?? trimmedPath
  let normalizedPath = withoutQuery.startsWith('/@fs/')
    ? withoutQuery.slice('/@fs'.length)
    : withoutQuery

  try {
    normalizedPath = decodeURIComponent(normalizedPath)
  } catch {
    // ignore decode failures and use raw path
  }

  if (/^\/[A-Za-z]:\//.test(normalizedPath)) {
    normalizedPath = normalizedPath.slice(1)
  }

  return normalizedPath.endsWith('.json') ? normalizedPath : null
}

function inferProjectPathHint(path: string | null): string | null {
  const stateFilePath = inferStateFilePathHint(path)
  if (!stateFilePath) return null

  const slashIndex = stateFilePath.lastIndexOf('/')
  if (slashIndex <= 0) return null
  return stateFilePath.slice(0, slashIndex)
}

function findFirstText(elements: SlideElement[]): string | null {
  for (const element of elements) {
    if (element.type === 'text') return element.text
    if (element.type === 'group') {
      const nested = findFirstText(element.children)
      if (nested) return nested
    }
  }
  return null
}

type ElementCounts = Record<'rect' | 'text' | 'path' | 'line' | 'circle' | 'image' | 'group', number>

function countElements(elements: SlideElement[]): ElementCounts {
  const counts: ElementCounts = {
    rect: 0,
    text: 0,
    path: 0,
    line: 0,
    circle: 0,
    image: 0,
    group: 0,
  }

  const visit = (items: SlideElement[]) => {
    for (const element of items) {
      counts[element.type as keyof ElementCounts] += 1
      if (element.type === 'group') visit(element.children)
    }
  }

  visit(elements)
  return counts
}

function findElementById(elements: SlideElement[], elementId: string | null): EditableElement | null {
  if (!elementId) return null

  for (const element of elements) {
    if (element.id === elementId) return element as EditableElement
    if (element.type === 'group') {
      const nested = findElementById(element.children, elementId)
      if (nested) return nested
    }
  }

  return null
}

function isSupportedEditableElement(element: EditableElement): element is SupportedEditableElement {
  return element.type !== 'group'
}

function isTransformableElement(element: EditableElement): element is TransformableElement {
  return element.type !== 'group' && element.type !== 'path'
}

function getFieldValue(element: SupportedEditableElement, key: string): string {
  const value = (element as unknown as Record<string, string | number | undefined>)[key]
  return value === undefined ? '' : String(value)
}

function getPatchablePropertyValue(element: SupportedEditableElement, key: string): unknown {
  return cloneSerializableValue((element as unknown as Record<string, unknown>)[key])
}

function renderPatchCountBadge(): void {
  patchCountBadge.textContent = `${patches.length} 条已应用 Patch`
}

function createPropertyPatch(
  elementId: string,
  property: string,
  oldValue: unknown,
  newValue: unknown,
): PatchOperation | null {
  if (!hasPatchValueChanged(oldValue, newValue)) return null

  return createUpdatePatch({
    slideIndex: currentSlideIndex,
    elementId,
    property,
    value: newValue,
    oldValue,
    source: 'human',
  })
}

function recordElementPatchDiffs(
  previous: TransformableElement,
  next: TransformableElement,
  keys: string[],
): PatchOperation[] {
  const operations: PatchOperation[] = []
  for (const key of keys) {
    const oldValue = cloneSerializableValue((previous as unknown as Record<string, unknown>)[key])
    const newValue = cloneSerializableValue((next as unknown as Record<string, unknown>)[key])
    const operation = createPropertyPatch(next.id, key, oldValue, newValue)
    if (operation) operations.push(operation)
  }
  return operations
}

function getPatchKeysForElement(element: TransformableElement): string[] {
  switch (element.type) {
    case 'text':
      return ['x', 'y', 'width']
    case 'rect':
    case 'image':
      return ['x', 'y', 'width', 'height']
    case 'line':
      return ['x1', 'y1', 'x2', 'y2']
    case 'circle':
      return ['cx', 'cy', 'r']
  }
}

function applyElementUpdate(element: SupportedEditableElement, key: string, rawValue: string): boolean {
  switch (element.type) {
    case 'text':
      switch (key) {
        case 'x': return assignRequiredNumber(element, 'x', rawValue)
        case 'y': return assignRequiredNumber(element, 'y', rawValue)
        case 'width': return assignRequiredNumber(element, 'width', rawValue)
        case 'text': element.text = rawValue; return true
        case 'font': element.font = rawValue; return true
        case 'lineHeight': return assignRequiredNumber(element, 'lineHeight', rawValue)
        case 'fill': element.fill = rawValue; return true
      }
      break
    case 'rect':
      switch (key) {
        case 'x': return assignRequiredNumber(element, 'x', rawValue)
        case 'y': return assignRequiredNumber(element, 'y', rawValue)
        case 'width': return assignRequiredNumber(element, 'width', rawValue)
        case 'height': return assignRequiredNumber(element, 'height', rawValue)
        case 'fill': element.fill = rawValue || undefined; return true
        case 'rx': return assignOptionalNumber(element, 'rx', rawValue)
      }
      break
    case 'path':
      switch (key) {
        case 'fill': element.fill = rawValue || undefined; return true
      }
      break
    case 'image':
      switch (key) {
        case 'x': return assignRequiredNumber(element, 'x', rawValue)
        case 'y': return assignRequiredNumber(element, 'y', rawValue)
        case 'width': return assignRequiredNumber(element, 'width', rawValue)
        case 'height': return assignRequiredNumber(element, 'height', rawValue)
        case 'href': element.href = rawValue; return true
      }
      break
    case 'line':
      switch (key) {
        case 'x1': return assignRequiredNumber(element, 'x1', rawValue)
        case 'y1': return assignRequiredNumber(element, 'y1', rawValue)
        case 'x2': return assignRequiredNumber(element, 'x2', rawValue)
        case 'y2': return assignRequiredNumber(element, 'y2', rawValue)
        case 'stroke': element.stroke = rawValue; return true
      }
      break
    case 'circle':
      switch (key) {
        case 'cx': return assignRequiredNumber(element, 'cx', rawValue)
        case 'cy': return assignRequiredNumber(element, 'cy', rawValue)
        case 'r': return assignRequiredNumber(element, 'r', rawValue)
        case 'fill': element.fill = rawValue || undefined; return true
      }
      break
  }

  return false
}

function assignRequiredNumber<T extends SupportedEditableElement, K extends keyof T>(target: T, key: K, rawValue: string): boolean {
  const nextValue = Number(rawValue)
  if (!Number.isFinite(nextValue)) return false
  target[key] = nextValue as T[K]
  return true
}

function assignOptionalNumber<T extends SupportedEditableElement, K extends keyof T>(target: T, key: K, rawValue: string): boolean {
  if (rawValue.trim() === '') {
    target[key] = undefined as T[K]
    return true
  }

  const nextValue = Number(rawValue)
  if (!Number.isFinite(nextValue)) return false
  target[key] = nextValue as T[K]
  return true
}

function cloneTransformableElement<T extends TransformableElement>(element: T): T {
  return JSON.parse(JSON.stringify(element)) as T
}

function captureTextNodeSnapshots(element: TransformableElement): TextNodeSnapshot[] {
  if (element.type !== 'text') return []

  return getCanvasElementNodes(element.id).map(node => ({
    node,
    x: parseNodeNumberAttribute(node, 'x'),
    y: parseNodeNumberAttribute(node, 'y'),
  }))
}

function parseNodeNumberAttribute(node: SVGGraphicsElement, attribute: string): number | null {
  const rawValue = node.getAttribute(attribute)
  if (rawValue === null) return null

  const parsed = Number(rawValue)
  return Number.isFinite(parsed) ? parsed : null
}

function clientPointToViewBox(svg: SVGSVGElement, clientX: number, clientY: number): SvgPoint | null {
  const svgRect = svg.getBoundingClientRect()
  const viewBox = svg.viewBox.baseVal
  if (!svgRect.width || !svgRect.height) return null

  return {
    x: ((clientX - svgRect.left) / svgRect.width) * viewBox.width + viewBox.x,
    y: ((clientY - svgRect.top) / svgRect.height) * viewBox.height + viewBox.y,
  }
}

function expandBounds(bounds: SvgBounds, padding: number): SvgBounds {
  return {
    x: bounds.x - padding,
    y: bounds.y - padding,
    width: bounds.width + padding * 2,
    height: bounds.height + padding * 2,
  }
}

function getResizeCursor(handle: ResizeHandle): string {
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

function getElementInteractionBounds(element: EditableElement, svg: SVGSVGElement): SvgBounds | null {
  switch (element.type) {
    case 'text':
      return getTextInteractionBounds(element)
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
      return measureElementBounds(svg, element.id, 0)
  }
}

function getTextInteractionBounds(element: TextElement): SvgBounds {
  const fontSpec = parseFontSpec(element.font)
  const layout = layoutText(element)
  return {
    x: getTextAnchorLeft(element),
    y: element.y - fontSpec.fontSize,
    width: Math.max(MIN_TEXT_WIDTH, element.width),
    height: Math.max(fontSpec.fontSize, layout.height),
  }
}

function getTextAnchorLeft(element: TextElement): number {
  switch (element.textAnchor) {
    case 'middle':
      return element.x - element.width / 2
    case 'end':
      return element.x - element.width
    default:
      return element.x
  }
}

function syncCanvasElementCursors(): void {
  for (const node of getCanvasElementNodes()) {
    node.style.cursor = ''
  }

  if (editingTextId) return

  const selected = getSelectedElement()
  if (!selected || !isTransformableElement(selected)) return

  for (const node of getCanvasElementNodes(selected.id)) {
    node.style.cursor = 'move'
  }
}

function stopOverlayHandleClick(event: MouseEvent): void {
  event.preventDefault()
  event.stopPropagation()
}

function handleResizeHandlePointerDown(event: PointerEvent): void {
  if (editingTextId || event.button !== 0) return
  if (!(event.currentTarget instanceof SVGRectElement)) return

  const handle = event.currentTarget.dataset.editorHandle as ResizeHandle | undefined
  const selected = getSelectedElement()
  if (!handle || !selected) return

  if (!isTransformableElement(selected)) {
    interactionHint = selected.type === 'path'
      ? 'path 元素暂不支持缩放；后续可考虑改为 transform 模式。'
      : 'group 容器暂不支持缩放；请直接调整内部具体元素。'
    renderInspector()
    return
  }

  if (selected.type === 'text' && (handle === 'n' || handle === 's')) {
    interactionHint = '文本元素当前只支持左右边与四角缩放；上下中点手柄暂未启用。'
    renderInspector()
    event.preventDefault()
    event.stopPropagation()
    return
  }

  const svg = getCanvasSvg()
  if (!svg) return

  const startPoint = clientPointToViewBox(svg, event.clientX, event.clientY)
  const initialBounds = getElementInteractionBounds(selected, svg)
  if (!startPoint || !initialBounds) return

  interactionHint = null
  pointerSession = {
    pointerId: event.pointerId,
    kind: 'resize',
    elementId: selected.id,
    handle,
    startClientX: event.clientX,
    startClientY: event.clientY,
    startPoint,
    initialBounds,
    initialElement: cloneTransformableElement(selected),
    textNodeSnapshots: captureTextNodeSnapshots(selected),
    started: false,
  }

  event.preventDefault()
  event.stopPropagation()
}

function applyMoveFromSession(
  element: TransformableElement,
  initialElement: TransformableElement,
  delta: SvgPoint,
): void {
  switch (element.type) {
    case 'text':
      if (initialElement.type !== 'text') return
      element.x = initialElement.x + delta.x
      element.y = initialElement.y + delta.y
      return
    case 'rect':
      if (initialElement.type !== 'rect') return
      element.x = initialElement.x + delta.x
      element.y = initialElement.y + delta.y
      return
    case 'image':
      if (initialElement.type !== 'image') return
      element.x = initialElement.x + delta.x
      element.y = initialElement.y + delta.y
      return
    case 'line':
      if (initialElement.type !== 'line') return
      element.x1 = initialElement.x1 + delta.x
      element.y1 = initialElement.y1 + delta.y
      element.x2 = initialElement.x2 + delta.x
      element.y2 = initialElement.y2 + delta.y
      return
    case 'circle':
      if (initialElement.type !== 'circle') return
      element.cx = initialElement.cx + delta.x
      element.cy = initialElement.cy + delta.y
      return
  }
}

function applyResizeFromSession(
  element: TransformableElement,
  session: PointerInteractionSession,
  delta: SvgPoint,
): void {
  if (!session.handle) return

  const minSize = element.type === 'text' ? MIN_TEXT_WIDTH : MIN_RESIZE_SIZE
  const nextBounds = resizeBoundsFromHandle(session.initialBounds, delta, session.handle, minSize)

  switch (element.type) {
    case 'text':
      if (session.initialElement.type !== 'text') return
      applyTextResize(element, session.initialElement, session.initialBounds, nextBounds)
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
      applyCircleResize(element, nextBounds, session.handle)
      return
    case 'line':
      if (session.initialElement.type !== 'line') return
      applyLineResize(element, session.initialElement, session.initialBounds, nextBounds)
      return
  }
}

function resizeBoundsFromHandle(
  bounds: SvgBounds,
  delta: SvgPoint,
  handle: ResizeHandle,
  minSize: number,
): SvgBounds {
  const initialMinX = bounds.x
  const initialMaxX = bounds.x + bounds.width
  const initialMinY = bounds.y
  const initialMaxY = bounds.y + bounds.height

  let minX = initialMinX
  let maxX = initialMaxX
  let minY = initialMinY
  let maxY = initialMaxY

  if (handle.includes('w')) minX = Math.min(initialMinX + delta.x, initialMaxX - minSize)
  if (handle.includes('e')) maxX = Math.max(initialMaxX + delta.x, initialMinX + minSize)
  if (handle.includes('n')) minY = Math.min(initialMinY + delta.y, initialMaxY - minSize)
  if (handle.includes('s')) maxY = Math.max(initialMaxY + delta.y, initialMinY + minSize)

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  }
}

function applyTextResize(
  element: TextElement,
  initialElement: TextElement,
  initialBounds: SvgBounds,
  nextBounds: SvgBounds,
): void {
  element.width = Math.max(MIN_TEXT_WIDTH, nextBounds.width)
  element.y = initialElement.y + (nextBounds.y - initialBounds.y)

  switch (initialElement.textAnchor) {
    case 'middle':
      element.x = nextBounds.x + element.width / 2
      return
    case 'end':
      element.x = nextBounds.x + element.width
      return
    default:
      element.x = nextBounds.x
  }
}

function applyCircleResize(circle: CircleElement, nextBounds: SvgBounds, handle: ResizeHandle): void {
  const fitted = fitCircleBounds(nextBounds, handle)
  circle.cx = fitted.x + fitted.width / 2
  circle.cy = fitted.y + fitted.height / 2
  circle.r = fitted.width / 2
}

function fitCircleBounds(bounds: SvgBounds, handle: ResizeHandle): SvgBounds {
  const size = Math.max(MIN_CIRCLE_DIAMETER, Math.min(bounds.width, bounds.height))
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

function applyLineResize(
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

function scaleCoordinate(
  value: number,
  initialStart: number,
  initialSize: number,
  nextStart: number,
  nextSize: number,
): number {
  if (initialSize === 0) return nextStart + nextSize / 2
  return nextStart + ((value - initialStart) / initialSize) * nextSize
}

function syncLiveElementPreview(
  element: TransformableElement,
  session: PointerInteractionSession,
): void {
  switch (element.type) {
    case 'text':
      syncTextNodePreview(element, session)
      return
    case 'rect':
      syncElementNodesAttributes(element.id, {
        x: element.x,
        y: element.y,
        width: element.width,
        height: element.height,
      })
      return
    case 'image':
      syncElementNodesAttributes(element.id, {
        x: element.x,
        y: element.y,
        width: element.width,
        height: element.height,
      })
      return
    case 'line':
      syncElementNodesAttributes(element.id, {
        x1: element.x1,
        y1: element.y1,
        x2: element.x2,
        y2: element.y2,
      })
      return
    case 'circle':
      syncElementNodesAttributes(element.id, {
        cx: element.cx,
        cy: element.cy,
        r: element.r,
      })
      return
  }
}

function syncTextNodePreview(element: TextElement, session: PointerInteractionSession): void {
  if (session.initialElement.type !== 'text') return

  const deltaX = element.x - session.initialElement.x
  const deltaY = element.y - session.initialElement.y

  for (const snapshot of session.textNodeSnapshots) {
    if (snapshot.x !== null) snapshot.node.setAttribute('x', String(snapshot.x + deltaX))
    if (snapshot.y !== null) snapshot.node.setAttribute('y', String(snapshot.y + deltaY))
  }
}

function syncElementNodesAttributes(
  elementId: string,
  attributes: Record<string, number>,
): void {
  for (const node of getCanvasElementNodes(elementId)) {
    for (const [key, value] of Object.entries(attributes)) {
      node.setAttribute(key, String(value))
    }
  }
}

function syncInspectorPreview(element: SupportedEditableElement): void {
  selectionSummary.innerHTML = buildSelectionSummary(element)
  if (elementFields.hidden) return

  for (const field of getInspectorFields(element)) {
    const control = elementFields.querySelector<HTMLInputElement | HTMLTextAreaElement>(`[data-prop-key="${field.key}"]`)
    if (!control) continue

    const nextValue = getFieldValue(element, field.key)
    if (control.value !== nextValue) control.value = nextValue
  }
}

function namespaceSvgIds(svg: string, scope: string): string {
  const mapping = new Map<string, string>()
  const withScopedIds = svg.replace(/\sid="([^"]+)"/g, (_, id: string) => {
    const scoped = `${scope}-${id}`
    mapping.set(id, scoped)
    return ` id="${scoped}"`
  })

  return withScopedIds.replace(/url\(#([^)]+)\)/g, (_, id: string) => {
    const mapped = mapping.get(id) ?? `${scope}-${id}`
    return `url(#${mapped})`
  })
}

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id)
  if (!element) {
    throw new Error(`Missing required element: #${id}`)
  }
  return element as T
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function cloneSerializableValue<T>(value: T): T {
  if (value === undefined) return value
  return JSON.parse(JSON.stringify(value)) as T
}

function toSlideIdFromPath(path: string): string {
  const pathname = new URL(path, window.location.href).pathname
  const fileName = pathname.split('/').pop() || 'slide'
  return fileName.replace(/\.[^.]+$/, '').trim().replace(/\s+/g, '_')
}

function isUndoShortcut(event: KeyboardEvent): boolean {
  if (!(event.metaKey || event.ctrlKey)) return false
  if (event.shiftKey) return false
  return event.key.toLowerCase() === 'z'
}

function isRedoShortcut(event: KeyboardEvent): boolean {
  if (!(event.metaKey || event.ctrlKey)) return false
  const lowerKey = event.key.toLowerCase()
  return lowerKey === 'y' || (lowerKey === 'z' && event.shiftKey)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
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

function getCanvasSvg(): SVGSVGElement | null {
  return canvasMount.querySelector('svg')
}

function getCanvasElementNodes(elementId?: string): SVGGraphicsElement[] {
  const svg = getCanvasSvg()
  if (!svg) return []

  return Array.from(svg.querySelectorAll<SVGGraphicsElement>('[data-element-id]'))
    .filter(node => !elementId || node.dataset.elementId === elementId)
}

function getElementIdFromEventTarget(target: EventTarget | null): string | null {
  if (!(target instanceof Element)) return null
  const owner = target.closest<SVGGraphicsElement>('[data-element-id]')
  return owner?.dataset.elementId ?? null
}

function isFormField(target: EventTarget | null): target is HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement {
  return target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
}

function consumeSuppressedCanvasInteraction(event: Event): boolean {
  if (!suppressNextCanvasClick) return false
  suppressNextCanvasClick = false
  event.preventDefault()
  event.stopPropagation()
  return true
}

function enterTextEditing(element: TextElement): void {
  const svg = getCanvasSvg()
  if (!svg) return

  if (editingTextId === element.id && activeTextEditor) {
    activeTextEditor.textarea.focus()
    activeTextEditor.textarea.select()
    return
  }

  if (editingTextId) exitTextEditing()

  const bounds = measureTextEditorBounds(svg, element.id, element)
  if (!bounds) return

  const foreignObject = document.createElementNS(SVG_NS, 'foreignObject')
  foreignObject.setAttribute('data-text-editor-root', 'true')
  foreignObject.setAttribute('x', String(bounds.x))
  foreignObject.setAttribute('y', String(bounds.y))
  foreignObject.setAttribute('width', String(bounds.width))
  foreignObject.setAttribute('overflow', 'visible')

  const wrapper = document.createElement('div')
  wrapper.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml')
  wrapper.style.display = 'flex'
  wrapper.style.flexDirection = 'column'
  wrapper.style.gap = '6px'
  wrapper.style.pointerEvents = 'auto'

  const textarea = document.createElement('textarea')
  textarea.setAttribute('data-text-editor-textarea', 'true')
  textarea.value = element.text
  textarea.spellcheck = false
  applyTextEditorStyles(textarea, element)

  const status = document.createElement('div')
  status.setAttribute('data-text-editor-status', 'true')
  status.style.font = '600 12px Inter, PingFang SC, sans-serif'
  status.style.padding = '0 2px'
  status.style.userSelect = 'none'

  const hiddenNodes = hideCanvasElementNodes(element.id)

  wrapper.append(textarea, status)
  foreignObject.append(wrapper)
  svg.append(foreignObject)

  editingTextId = element.id
  activeTextEditor = {
    elementId: element.id,
    foreignObject,
    wrapper,
    textarea,
    status,
    hiddenNodes,
    bounds,
    initialText: element.text,
  }

  textarea.addEventListener('input', () => {
    if (editingTextId !== element.id || !activeTextEditor) return
    element.text = textarea.value
    syncInspectorTextField(textarea.value)
    syncTextEditorFrame(activeTextEditor, element)
  })

  textarea.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return
    event.preventDefault()
    event.stopPropagation()
    exitTextEditing()
  })

  syncTextEditorFrame(activeTextEditor, element)
  refreshCanvasOverlays()

  requestAnimationFrame(() => {
    textarea.focus()
    textarea.select()
  })
}

function exitTextEditing(options: { shouldRender?: boolean } = {}): void {
  const editor = activeTextEditor
  if (!editor) {
    editingTextId = null
    return
  }

  const currentElement = findElementById(getCurrentSlide().elements, editor.elementId)
  if (currentElement?.type === 'text') {
    const operation = createPropertyPatch(currentElement.id, 'text', editor.initialText, currentElement.text)
    if (operation) {
      commitPatchOperations([operation], 'human', `编辑文本 ${currentElement.id}`)
    }
  }

  restoreHiddenCanvasNodes(editor.hiddenNodes)
  editor.foreignObject.remove()
  activeTextEditor = null
  editingTextId = null
  hoveredElementId = selectedElementId

  if (options.shouldRender !== false) {
    const slide = getCurrentSlide()
    renderCanvas(slide, currentSlideIndex)
    renderThumbnails()
    renderInspector()
  } else {
    renderInspector()
    refreshCanvasOverlays()
  }
}

function measureTextEditorBounds(svg: SVGSVGElement, elementId: string, element: TextElement): SvgBounds | null {
  const renderedBounds = measureElementBounds(svg, elementId, 0)
  if (!renderedBounds) return null

  const layout = layoutText(element)
  const width = Math.max(TEXT_EDITOR_MIN_WIDTH, element.width, renderedBounds.width)
  const height = Math.max(
    TEXT_EDITOR_MIN_HEIGHT,
    renderedBounds.height,
    layout.height + TEXT_EDITOR_HEIGHT_PADDING,
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

function applyTextEditorStyles(textarea: HTMLTextAreaElement, element: TextElement): void {
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

function syncTextEditorFrame(editor: ActiveTextEditor, element: TextElement): void {
  const layout = layoutText(element)
  const textareaHeight = Math.max(
    editor.bounds.height,
    layout.height + TEXT_EDITOR_HEIGHT_PADDING,
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

function buildTextEditorStatus(lineCount: number, height: number, maxHeight: number | undefined, overflow: boolean): string {
  const parts = [
    `Pretext ${lineCount} 行`,
    `高度 ${formatNumber(height)}px`,
  ]

  if (maxHeight !== undefined) parts.push(`max ${formatNumber(maxHeight)}px`)
  parts.push(overflow ? '已溢出' : '未溢出')
  return parts.join(' · ')
}

function hideCanvasElementNodes(elementId: string): HiddenCanvasNode[] {
  return getCanvasElementNodes(elementId).map(node => {
    const opacity = node.getAttribute('opacity')
    node.setAttribute('opacity', '0')
    return { node, opacity }
  })
}

function restoreHiddenCanvasNodes(hiddenNodes: HiddenCanvasNode[]): void {
  for (const hidden of hiddenNodes) {
    if (hidden.opacity === null) hidden.node.removeAttribute('opacity')
    else hidden.node.setAttribute('opacity', hidden.opacity)
  }
}

function syncInspectorTextField(value: string): void {
  const field = elementFields.querySelector<HTMLTextAreaElement | HTMLInputElement>('[data-prop-key="text"]')
  if (field && field.value !== value) field.value = value
}

function getTextAlign(textAnchor?: TextElement['textAnchor']): 'left' | 'center' | 'right' {
  switch (textAnchor) {
    case 'middle':
      return 'center'
    case 'end':
      return 'right'
    default:
      return 'left'
  }
}

function roundedRectPath(x: number, y: number, width: number, height: number, radius: number): string {
  const right = x + width
  const bottom = y + height
  return [
    `M${x + radius},${y}`,
    `H${right - radius}`,
    `A${radius},${radius} 0 0 1 ${right},${y + radius}`,
    `V${bottom - radius}`,
    `A${radius},${radius} 0 0 1 ${right - radius},${bottom}`,
    `H${x + radius}`,
    `A${radius},${radius} 0 0 1 ${x},${bottom - radius}`,
    `V${y + radius}`,
    `A${radius},${radius} 0 0 1 ${x + radius},${y}`,
    'Z',
  ].join(' ')
}

function createSvgDataUri(markup: string): string {
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(markup)}`
}

function createBadgeGroup(config: {
  originX: number
  originY: number
  items: Array<{ label: string; fill: string; width: number }>
}): GroupElement {
  let offsetX = 0
  const children: SlideElement[] = []

  for (const item of config.items) {
    children.push({
      type: 'path',
      id: `badge_${item.label}_bg`,
      d: roundedRectPath(offsetX, 0, item.width, 40, 10),
      fill: item.fill,
    })
    children.push({
      type: 'text',
      id: `badge_${item.label}_text`,
      x: offsetX + item.width / 2,
      y: 25,
      width: item.width - 16,
      text: item.label,
      font: '600 14px Inter, PingFang SC, sans-serif',
      lineHeight: 20,
      fill: '#FFFFFF',
      textAnchor: 'middle',
    })
    offsetX += item.width + 14
  }

  return {
    type: 'group',
    id: `badge_group_${config.originX}_${config.originY}`,
    transform: `translate(${config.originX}, ${config.originY})`,
    fontFamily: 'Inter, PingFang SC, sans-serif',
    fontSize: 14,
    children,
  }
}

function createDemoState(): SlideState {
  const coverBackground = createSvgDataUri(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#111827" />
          <stop offset="55%" stop-color="#1d4ed8" />
          <stop offset="100%" stop-color="#0f766e" />
        </linearGradient>
      </defs>
      <rect width="1280" height="720" fill="url(#bg)" />
      <circle cx="1050" cy="140" r="180" fill="rgba(255,255,255,0.08)" />
      <circle cx="1140" cy="520" r="220" fill="rgba(255,255,255,0.06)" />
      <path d="M0 560 C180 480, 320 620, 480 560 S820 460, 1280 610 L1280 720 L0 720 Z" fill="rgba(255,255,255,0.12)" />
    </svg>
  `)

  const analyticsPreview = createSvgDataUri(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360">
      <defs>
        <linearGradient id="chart" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#4f46e5" />
          <stop offset="100%" stop-color="#06b6d4" />
        </linearGradient>
      </defs>
      <rect width="640" height="360" rx="28" fill="#f8fafc" />
      <rect x="36" y="36" width="568" height="52" rx="18" fill="#e2e8f0" />
      <rect x="36" y="112" width="190" height="172" rx="24" fill="#eef2ff" />
      <rect x="248" y="112" width="356" height="172" rx="24" fill="#ecfeff" />
      <path d="M282 252 L342 202 L396 216 L444 154 L514 184 L572 130" fill="none" stroke="url(#chart)" stroke-width="12" stroke-linecap="round" stroke-linejoin="round" />
      <circle cx="572" cy="130" r="14" fill="#06b6d4" />
      <rect x="72" y="150" width="32" height="102" rx="12" fill="#4f46e5" opacity="0.35" />
      <rect x="122" y="124" width="32" height="128" rx="12" fill="#4f46e5" opacity="0.55" />
      <rect x="172" y="98" width="32" height="154" rx="12" fill="#4f46e5" />
    </svg>
  `)

  const workflowPreview = createSvgDataUri(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360">
      <defs>
        <linearGradient id="workflow" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#0f172a" />
          <stop offset="100%" stop-color="#4f46e5" />
        </linearGradient>
      </defs>
      <rect width="640" height="360" rx="28" fill="#f8fafc" />
      <rect x="40" y="44" width="560" height="60" rx="20" fill="#e2e8f0" />
      <rect x="54" y="144" width="160" height="140" rx="28" fill="#ede9fe" />
      <rect x="240" y="144" width="160" height="140" rx="28" fill="#dbeafe" />
      <rect x="426" y="144" width="160" height="140" rx="28" fill="#ccfbf1" />
      <path d="M214 214 H240" stroke="url(#workflow)" stroke-width="10" stroke-linecap="round" />
      <path d="M400 214 H426" stroke="url(#workflow)" stroke-width="10" stroke-linecap="round" />
      <circle cx="320" cy="214" r="32" fill="url(#workflow)" />
      <text x="320" y="225" font-family="Inter, PingFang SC, sans-serif" font-size="24" font-weight="700" text-anchor="middle" fill="#ffffff">AI</text>
    </svg>
  `)

  return {
    canvas: { width: 1280, height: 720 },
    slides: [
      {
        id: 'slide_01_cover',
        defs: [{
          type: 'linearGradient',
          id: 'gradient_cover',
          x1: '0%',
          y1: '0%',
          x2: '100%',
          y2: '100%',
          stops: [
            { offset: '0%', color: '#6366F1' },
            { offset: '100%', color: '#06B6D4' },
          ],
        }],
        elements: [
          {
            type: 'image',
            id: 'cover_background',
            x: 0,
            y: 0,
            width: 1280,
            height: 720,
            href: coverBackground,
            preserveAspectRatio: 'xMidYMid slice',
          },
          {
            type: 'rect',
            id: 'cover_overlay',
            x: 0,
            y: 0,
            width: 1280,
            height: 720,
            fill: '#020617',
            opacity: 0.28,
          },
          {
            type: 'rect',
            id: 'cover_gradient_bar',
            x: 0,
            y: 0,
            width: 1280,
            height: 6,
            fill: 'url(#gradient_cover)',
          },
          {
            type: 'path',
            id: 'cover_accent',
            d: roundedRectPath(60, 220, 6, 280, 3),
            fill: 'url(#gradient_cover)',
          },
          {
            type: 'text',
            id: 'cover_title',
            x: 100,
            y: 294,
            width: 560,
            text: 'PPT Master Editor',
            font: '700 60px Inter, PingFang SC, sans-serif',
            lineHeight: 68,
            fill: '#FFFFFF',
          },
          {
            type: 'text',
            id: 'cover_subtitle',
            x: 100,
            y: 356,
            width: 540,
            text: 'AI + 人协同的浏览器编辑器 MVP',
            font: '500 28px Inter, PingFang SC, sans-serif',
            lineHeight: 36,
            fill: '#FFFFFF',
            opacity: 0.95,
          },
          {
            type: 'text',
            id: 'cover_description',
            x: 100,
            y: 414,
            width: 520,
            text: '以 slide_state 为真相源，直接在浏览器里渲染、翻页和预览 SVG 幻灯片。',
            font: '18px Inter, PingFang SC, sans-serif',
            lineHeight: 28,
            fill: '#E2E8F0',
            opacity: 0.9,
          },
          {
            type: 'line',
            id: 'cover_separator',
            x1: 100,
            y1: 450,
            x2: 500,
            y2: 450,
            stroke: '#FFFFFF',
            strokeWidth: 2,
            opacity: 0.4,
          },
          createBadgeGroup({
            originX: 100,
            originY: 480,
            items: [
              { label: 'SVG Canvas', fill: '#6366F1', width: 128 },
              { label: 'Multi-page Nav', fill: '#06B6D4', width: 146 },
              { label: 'Thumbnail Strip', fill: '#10B981', width: 154 },
            ],
          }),
          {
            type: 'group',
            id: 'cover_preview_card',
            transform: 'translate(780, 180)',
            children: [
              {
                type: 'path',
                id: 'preview_shadow',
                d: roundedRectPath(16, 16, 380, 320, 22),
                fill: '#020617',
                opacity: 0.18,
              },
              {
                type: 'path',
                id: 'preview_shell',
                d: roundedRectPath(0, 0, 380, 320, 22),
                fill: '#FFFFFF',
                opacity: 0.96,
              },
              {
                type: 'path',
                id: 'preview_toolbar',
                d: roundedRectPath(20, 18, 340, 24, 8),
                fill: '#E5E7EB',
              },
              {
                type: 'path',
                id: 'preview_sidebar',
                d: roundedRectPath(22, 62, 92, 232, 14),
                fill: '#EEF2FF',
              },
              {
                type: 'image',
                id: 'preview_image',
                x: 130,
                y: 62,
                width: 214,
                height: 180,
                href: analyticsPreview,
                preserveAspectRatio: 'xMidYMid meet',
              },
              {
                type: 'group',
                id: 'preview_bars',
                transform: 'translate(132, 250)',
                children: [
                  { type: 'path', id: 'bar_1', d: roundedRectPath(0, 28, 38, 32, 8), fill: '#E2E8F0' },
                  { type: 'path', id: 'bar_2', d: roundedRectPath(52, 8, 38, 52, 8), fill: '#C7D2FE' },
                  { type: 'path', id: 'bar_3', d: roundedRectPath(104, 18, 38, 42, 8), fill: '#A5B4FC' },
                  { type: 'path', id: 'bar_4', d: roundedRectPath(156, 0, 38, 60, 8), fill: '#6366F1' },
                ],
              },
              {
                type: 'circle',
                id: 'preview_ai_dot',
                cx: 332,
                cy: 42,
                r: 24,
                fill: 'url(#gradient_cover)',
              },
              {
                type: 'text',
                id: 'preview_ai_label',
                x: 332,
                y: 47,
                width: 44,
                text: 'AI',
                font: '700 14px Inter, PingFang SC, sans-serif',
                lineHeight: 18,
                fill: '#FFFFFF',
                textAnchor: 'middle',
              },
            ],
          },
          {
            type: 'group',
            id: 'cover_footer',
            fill: '#FFFFFF',
            opacity: 0.72,
            children: [
              {
                type: 'text',
                id: 'cover_footer_left',
                x: 100,
                y: 640,
                width: 380,
                text: 'editor/index.html · app.ts · slideToSvg()',
                font: '14px Inter, PingFang SC, sans-serif',
                lineHeight: 18,
                fill: '#FFFFFF',
              },
              {
                type: 'text',
                id: 'cover_footer_right',
                x: 1180,
                y: 640,
                width: 160,
                text: 'Phase 2 / MVP',
                font: '14px Inter, PingFang SC, sans-serif',
                lineHeight: 18,
                fill: '#FFFFFF',
                textAnchor: 'end',
              },
            ],
          },
          {
            type: 'text',
            id: 'cover_page_number',
            x: 640,
            y: 690,
            width: 120,
            text: '01 / 03',
            font: '12px Inter, PingFang SC, sans-serif',
            lineHeight: 16,
            fill: '#FFFFFF',
            textAnchor: 'middle',
            opacity: 0.55,
          },
        ],
      },
      {
        id: 'slide_02_overview',
        background: '#F3F5F9',
        defs: [{
          type: 'linearGradient',
          id: 'gradient_overview',
          x1: '0%',
          y1: '0%',
          x2: '100%',
          y2: '100%',
          stops: [
            { offset: '0%', color: '#0F172A' },
            { offset: '100%', color: '#4F46E5' },
          ],
        }],
        elements: [
          {
            type: 'rect',
            id: 'overview_topbar',
            x: 64,
            y: 52,
            width: 1152,
            height: 54,
            fill: '#FFFFFF',
            rx: 20,
          },
          {
            type: 'text',
            id: 'overview_title',
            x: 88,
            y: 86,
            width: 520,
            text: 'MVP 范围：先把渲染和导航跑通',
            font: '700 38px Inter, PingFang SC, sans-serif',
            lineHeight: 46,
            fill: '#0F172A',
          },
          {
            type: 'text',
            id: 'overview_subtitle',
            x: 88,
            y: 132,
            width: 620,
            text: '这一版只解决 HTML 入口、多页切换、缩略图和 SVG 画布渲染，不提前引入交互编辑复杂度。',
            font: '18px Inter, PingFang SC, sans-serif',
            lineHeight: 28,
            fill: '#475569',
          },
          {
            type: 'path',
            id: 'overview_main_card',
            d: roundedRectPath(72, 184, 680, 440, 28),
            fill: '#FFFFFF',
          },
          {
            type: 'path',
            id: 'overview_side_card',
            d: roundedRectPath(784, 184, 424, 440, 28),
            fill: '#0F172A',
          },
          {
            type: 'line',
            id: 'overview_divider',
            x1: 112,
            y1: 268,
            x2: 712,
            y2: 268,
            stroke: '#CBD5E1',
            strokeWidth: 2,
          },
          {
            type: 'group',
            id: 'overview_cards',
            transform: 'translate(112, 300)',
            children: [
              { type: 'path', id: 'overview_item_1', d: roundedRectPath(0, 0, 250, 118, 22), fill: '#EEF2FF' },
              { type: 'path', id: 'overview_item_2', d: roundedRectPath(288, 0, 250, 118, 22), fill: '#ECFEFF' },
              { type: 'path', id: 'overview_item_3', d: roundedRectPath(0, 148, 538, 118, 22), fill: '#F8FAFC' },
              {
                type: 'text',
                id: 'overview_item_1_title',
                x: 24,
                y: 36,
                width: 190,
                text: '左侧主画布',
                font: '700 22px Inter, PingFang SC, sans-serif',
                lineHeight: 28,
                fill: '#312E81',
              },
              {
                type: 'text',
                id: 'overview_item_1_body',
                x: 24,
                y: 72,
                width: 200,
                text: '实时插入 `slideToSvg()` 生成的完整 SVG。',
                font: '15px Inter, PingFang SC, sans-serif',
                lineHeight: 24,
                fill: '#4338CA',
              },
              {
                type: 'text',
                id: 'overview_item_2_title',
                x: 312,
                y: 36,
                width: 190,
                text: '右侧属性面板',
                font: '700 22px Inter, PingFang SC, sans-serif',
                lineHeight: 28,
                fill: '#155E75',
              },
              {
                type: 'text',
                id: 'overview_item_2_body',
                x: 312,
                y: 72,
                width: 210,
                text: '先展示 slide 元数据，后续接选中与 patch。',
                font: '15px Inter, PingFang SC, sans-serif',
                lineHeight: 24,
                fill: '#0F766E',
              },
              {
                type: 'text',
                id: 'overview_item_3_title',
                x: 24,
                y: 184,
                width: 360,
                text: '底部缩略图条',
                font: '700 22px Inter, PingFang SC, sans-serif',
                lineHeight: 28,
                fill: '#0F172A',
              },
              {
                type: 'text',
                id: 'overview_item_3_body',
                x: 24,
                y: 220,
                width: 480,
                text: '每页复用同一套 slide_state 渲染逻辑，点击缩略图即可切换当前页面。',
                font: '15px Inter, PingFang SC, sans-serif',
                lineHeight: 24,
                fill: '#334155',
              },
            ],
          },
          {
            type: 'image',
            id: 'overview_preview',
            x: 826,
            y: 222,
            width: 338,
            height: 222,
            href: workflowPreview,
            preserveAspectRatio: 'xMidYMid meet',
          },
          {
            type: 'circle',
            id: 'overview_indicator',
            cx: 872,
            cy: 500,
            r: 34,
            fill: 'url(#gradient_overview)',
          },
          {
            type: 'text',
            id: 'overview_indicator_label',
            x: 872,
            y: 508,
            width: 60,
            text: '02',
            font: '700 18px Inter, PingFang SC, sans-serif',
            lineHeight: 20,
            fill: '#FFFFFF',
            textAnchor: 'middle',
          },
          {
            type: 'text',
            id: 'overview_side_title',
            x: 920,
            y: 510,
            width: 220,
            text: '导航与预览是第一优先级',
            font: '700 28px Inter, PingFang SC, sans-serif',
            lineHeight: 36,
            fill: '#FFFFFF',
          },
          {
            type: 'text',
            id: 'overview_side_body',
            x: 826,
            y: 566,
            width: 318,
            text: '因为它决定用户是否能在浏览器里“看见并控制” AI 生成的页面状态。',
            font: '16px Inter, PingFang SC, sans-serif',
            lineHeight: 26,
            fill: '#CBD5E1',
          },
          {
            type: 'text',
            id: 'overview_page_number',
            x: 1180,
            y: 670,
            width: 80,
            text: '02 / 03',
            font: '12px Inter, PingFang SC, sans-serif',
            lineHeight: 16,
            fill: '#64748B',
            textAnchor: 'end',
          },
        ],
      },
      {
        id: 'slide_03_metrics',
        background: '#F8FAFC',
        defs: [{
          type: 'linearGradient',
          id: 'gradient_metrics',
          x1: '0%',
          y1: '0%',
          x2: '100%',
          y2: '100%',
          stops: [
            { offset: '0%', color: '#8B5CF6' },
            { offset: '100%', color: '#06B6D4' },
          ],
        }],
        elements: [
          {
            type: 'text',
            id: 'metrics_title',
            x: 88,
            y: 104,
            width: 520,
            text: '下一步衔接：选中、属性编辑、design patch',
            font: '700 40px Inter, PingFang SC, sans-serif',
            lineHeight: 48,
            fill: '#0F172A',
          },
          {
            type: 'text',
            id: 'metrics_subtitle',
            x: 88,
            y: 154,
            width: 620,
            text: 'MVP 已把渲染层跑通，后续只需在同一状态树上叠加交互层，而不是推倒重来。',
            font: '18px Inter, PingFang SC, sans-serif',
            lineHeight: 28,
            fill: '#475569',
          },
          {
            type: 'group',
            id: 'metrics_cards',
            transform: 'translate(88, 210)',
            children: [
              { type: 'path', id: 'metrics_card_1', d: roundedRectPath(0, 0, 340, 188, 28), fill: '#FFFFFF' },
              { type: 'path', id: 'metrics_card_2', d: roundedRectPath(376, 0, 340, 188, 28), fill: '#FFFFFF' },
              { type: 'path', id: 'metrics_card_3', d: roundedRectPath(752, 0, 352, 188, 28), fill: '#FFFFFF' },
              { type: 'circle', id: 'metrics_dot_1', cx: 44, cy: 44, r: 18, fill: '#8B5CF6' },
              { type: 'circle', id: 'metrics_dot_2', cx: 420, cy: 44, r: 18, fill: '#06B6D4' },
              { type: 'circle', id: 'metrics_dot_3', cx: 796, cy: 44, r: 18, fill: '#10B981' },
              {
                type: 'text',
                id: 'metrics_card_1_title',
                x: 76,
                y: 50,
                width: 220,
                text: 'SVG 画布',
                font: '700 26px Inter, PingFang SC, sans-serif',
                lineHeight: 32,
                fill: '#0F172A',
              },
              {
                type: 'text',
                id: 'metrics_card_1_body',
                x: 24,
                y: 96,
                width: 280,
                text: '以 HTML 为入口，直接挂载 slideToSvg() 输出结果。',
                font: '16px Inter, PingFang SC, sans-serif',
                lineHeight: 24,
                fill: '#475569',
              },
              {
                type: 'text',
                id: 'metrics_card_2_title',
                x: 452,
                y: 50,
                width: 220,
                text: '多页状态',
                font: '700 26px Inter, PingFang SC, sans-serif',
                lineHeight: 32,
                fill: '#0F172A',
              },
              {
                type: 'text',
                id: 'metrics_card_2_body',
                x: 400,
                y: 96,
                width: 280,
                text: '上一页、下一页、页码与缩略图共用同一份 state。',
                font: '16px Inter, PingFang SC, sans-serif',
                lineHeight: 24,
                fill: '#475569',
              },
              {
                type: 'text',
                id: 'metrics_card_3_title',
                x: 828,
                y: 50,
                width: 220,
                text: '属性面板骨架',
                font: '700 26px Inter, PingFang SC, sans-serif',
                lineHeight: 32,
                fill: '#0F172A',
              },
              {
                type: 'text',
                id: 'metrics_card_3_body',
                x: 776,
                y: 96,
                width: 300,
                text: '为后续元素选中、属性编辑、patch 回写预留固定区域。',
                font: '16px Inter, PingFang SC, sans-serif',
                lineHeight: 24,
                fill: '#475569',
              },
            ],
          },
          {
            type: 'path',
            id: 'metrics_flow_shell',
            d: roundedRectPath(88, 448, 1104, 208, 32),
            fill: '#0F172A',
          },
          {
            type: 'image',
            id: 'metrics_flow_preview',
            x: 132,
            y: 484,
            width: 320,
            height: 136,
            href: analyticsPreview,
            preserveAspectRatio: 'xMidYMid meet',
          },
          {
            type: 'path',
            id: 'metrics_flow_arrow',
            d: 'M506 548 H842',
            stroke: 'url(#gradient_metrics)',
            strokeWidth: 10,
            fill: 'none',
          },
          {
            type: 'path',
            id: 'metrics_flow_arrow_head',
            d: 'M842 548 L812 530 L812 566 Z',
            fill: 'url(#gradient_metrics)',
          },
          {
            type: 'group',
            id: 'metrics_flow_copy',
            transform: 'translate(574, 500)',
            children: [
              {
                type: 'text',
                id: 'metrics_flow_title',
                x: 0,
                y: 0,
                width: 440,
                text: 'slide_state → SVG → 浏览器编辑器',
                font: '700 30px Inter, PingFang SC, sans-serif',
                lineHeight: 36,
                fill: '#FFFFFF',
              },
              {
                type: 'text',
                id: 'metrics_flow_body',
                x: 0,
                y: 50,
                width: 470,
                text: '后续在这条链路中加入 hover / select / edit / patch，而不是再发明第二套渲染逻辑。',
                font: '16px Inter, PingFang SC, sans-serif',
                lineHeight: 26,
                fill: '#CBD5E1',
              },
            ],
          },
          {
            type: 'text',
            id: 'metrics_page_number',
            x: 1180,
            y: 686,
            width: 80,
            text: '03 / 03',
            font: '12px Inter, PingFang SC, sans-serif',
            lineHeight: 16,
            fill: '#64748B',
            textAnchor: 'end',
          },
        ],
      },
    ],
  }
}
