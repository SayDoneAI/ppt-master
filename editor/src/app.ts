import * as pretext from '@chenglou/pretext'
import type {
  AiCommand,
  DesignPatch,
  Element as SlideElement,
  GroupElement,
  PatchOperation,
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
  CANVAS_PRESETS,
  findCanvasPreset,
  getTopbarSaveActionsState,
  shouldShowPosterCanvasControls,
  resizeSlideStateCanvas,
  supportsCanvasPresetEditing,
} from './canvas_resize.js'
import {
  createHistoryEntry,
  flattenHistoryOperations,
  type HistoryEntry,
} from './history.js'
import {
  createAppendSlidesPatch,
  ensureCompatibleCanvas,
} from './slide_import.js'
import { createProjectSvgArtifacts } from './project_pipeline.js'
import {
  buildProjectAiHandoffRelativePath,
  LOCAL_AI_HANDOFF_WRITE_ENDPOINT,
  LOCAL_PROJECT_SAVE_PAGES_ENDPOINT,
} from './local_ai_handoff.js'
import type { DownloadArtifact } from './state_io.js'
import {
  getStatePathFromSearch,
  getSvgPathsFromSearch,
  isJsonFile,
  isSvgFile,
  parseSlideStateJson,
  readSlideStateFile,
  readSvgFiles,
} from './state_io.js'
import { STATE_WATCHER_HMR_EVENT } from './state_sync_events.js'
import { initPretext, slideToSvg } from './state_to_svg.js'
import { syncCurrentSlideFromLiveSvg } from './svg_editor.js'
import { normalizeSvgForEditor, svgsToState } from './svg_to_state.js'
import { COLOR_SCHEMES } from './presets/colors.js'
import { FONT_SCHEMES } from './presets/fonts.js'
import {
  DEFAULT_COLOR_PICKER_VALUE,
  applyColorPreset,
  applyFontPreset,
  assignTextFontSpec,
  cloneSerializableValue,
  createPropertyMutation,
  normalizeFontWeightValue,
  normalizeHexColor,
  parseFontSpec,
  toMutationList,
  type ApplyColorPresetContext,
  type ApplyFontPresetContext,
  type EditableElement,
  type PropertyMutation,
} from './preset_engine.js'
import {
  applyMoveFromSession,
  applyResizeFromSession,
  clientPointToViewBox,
  expandBounds,
  getElementInteractionBounds,
  getResizeCursor,
  handleResizeHandlePointerDown,
  stopOverlayHandleClick,
  syncCanvasElementCursors,
  syncInspectorPreview,
  syncLiveElementPreview,
  type InspectorControl,
  type PointerInteractionSession,
  type ResizeHandle,
  type SupportedEditableElement,
  type SvgBounds,
  type TextNodeSnapshot,
  type TransformableElement,
} from './pointer_interactions.js'
import {
  ALL_RESIZE_HANDLES,
  getResizeHandlesForElementType,
  syncTextSvgNodes,
} from './text_resize.js'
import {
  enterTextEditing,
  exitTextEditing,
  type ActiveTextEditor,
  type TextEditingContext,
} from './text_editing.js'

type EditorViewMode = 'workspace' | 'preview'

type InspectorInputType = 'number' | 'text' | 'textarea' | 'color' | 'range' | 'select'

interface ProjectPageSavePayload {
  filename: string
  svg: string
}

interface SavePagesResult {
  savedCount: number
  dir: string
}

interface InspectorFieldOption {
  value: string
  label: string
}

interface InspectorField {
  key: string
  label: string
  input: InspectorInputType
  readOnly?: boolean
  step?: string
  min?: number
  max?: number
  advanced?: boolean
  options?: InspectorFieldOption[]
}

interface StateWatcherPayload {
  filePath?: string
  urlPath?: string
}

interface SvgImportBundle {
  state: SlideState
  rawSvgStrings: string[]
}

initPretext(pretext)

const SVG_NS = 'http://www.w3.org/2000/svg'
const HANDLE_SIZE = 10
const OVERLAY_PADDING = 4
const DRAG_THRESHOLD_PX = 3
const MIN_RESIZE_SIZE = 12
const MIN_TEXT_WIDTH = 40
const MIN_CIRCLE_DIAMETER = 12
const CANVAS_STAGE_MAX_WIDTH = 1120
const AI_HANDOFF_IDLE_MESSAGE = '默认主路径是 SVG 页面；如需自动刷新，请绑定项目里的兼容 state（如 slide_state.json）。'
const ASSET_LIBRARY_IDLE_MESSAGE = '模板导入和 Patch 回流都放在这里，平时不用展开。'
const DEFAULT_DEMO_SVG_PATH = '/examples/demo_project_intro_ppt169_20251211/svg_final/slide_01_cover.svg'
const FONT_WEIGHT_OPTIONS: InspectorFieldOption[] = [
  { value: '400', label: '正常 (400)' },
  { value: '500', label: '中等 (500)' },
  { value: '700', label: '粗体 (700)' },
]

const editorShell = document.querySelector<HTMLElement>('.editor-shell')
const canvasMount = getElement<HTMLDivElement>('canvasMount')
const canvasScroll = getElement<HTMLDivElement>('canvasScroll')
const canvasPane = canvasMount.closest<HTMLElement>('.canvas-pane')
const filmstrip = document.querySelector<HTMLElement>('.filmstrip')
const prevButton = getElement<HTMLButtonElement>('prevSlideBtn')
const nextButton = getElement<HTMLButtonElement>('nextSlideBtn')
const undoButton = getElement<HTMLButtonElement>('undoBtn')
const redoButton = getElement<HTMLButtonElement>('redoBtn')
const exportSvgButton = getElement<HTMLButtonElement>('exportSvgBtn')
const exportPngButton = getElement<HTMLButtonElement>('exportPngBtn')
const saveTemplateButton = getElement<HTMLButtonElement>('saveTemplateBtn')
const watchFileButton = getElement<HTMLButtonElement>('watchFileBtn')
const exportPatchButton = getElement<HTMLButtonElement>('exportPatchBtn')
const pageIndicator = getElement<HTMLDivElement>('pageIndicator')
const posterPreviewControls = getElement<HTMLDivElement>('posterPreviewControls')
const posterPreviewToggleButton = getElement<HTMLButtonElement>('posterPreviewToggleBtn')
const stateSourceBadge = getElement<HTMLDivElement>('stateSourceBadge')
const thumbnailStrip = getElement<HTMLDivElement>('thumbnailStrip')
const slideMeta = getElement<HTMLDivElement>('slideMeta')
const elementSummary = getElement<HTMLDivElement>('elementSummary')
const colorPresetGrid = getElement<HTMLDivElement>('colorPresetGrid')
const fontPresetGrid = getElement<HTMLDivElement>('fontPresetGrid')
const selectionSummary = getElement<HTMLDivElement>('selectionSummary')
const selectionEmptyState = getElement<HTMLDivElement>('selectionEmptyState')
const elementFields = getElement<HTMLFormElement>('elementFields')
const alignToolbar = getElement<HTMLDivElement>('alignToolbar')
const patchCountBadge = getElement<HTMLDivElement>('patchCountBadge')
const aiTargetSummary = getElement<HTMLDivElement>('aiTargetSummary')
const aiInstructionInput = getElement<HTMLTextAreaElement>('aiInstructionInput')
const exportAiTaskButton = getElement<HTMLButtonElement>('exportAiTaskBtn')
const applyAiPatchButton = getElement<HTMLButtonElement>('applyAiPatchBtn')
const downloadAiHandoffNoteButton = getElement<HTMLButtonElement>('downloadAiHandoffNoteBtn')
const copyAiHandoffNoteButton = getElement<HTMLButtonElement>('copyAiHandoffNoteBtn')
const aiHandoffStatus = getElement<HTMLDivElement>('aiHandoffStatus')
const aiHandoffPreviewWrap = getElement<HTMLDivElement>('aiHandoffPreviewWrap')
const aiHandoffPreview = getElement<HTMLTextAreaElement>('aiHandoffPreview')
const aiPatchFileInput = getElement<HTMLInputElement>('aiPatchFileInput')
const assetImportSummary = getElement<HTMLDivElement>('assetImportSummary')
const importTemplateButton = getElement<HTMLButtonElement>('importTemplateBtn')
const importChartButton = getElement<HTMLButtonElement>('importChartBtn')
const templateImportInput = getElement<HTMLInputElement>('templateImportInput')
const chartImportInput = getElement<HTMLInputElement>('chartImportInput')
const assetLibraryStatus = getElement<HTMLDivElement>('assetLibraryStatus')
const dropZoneOverlay = getElement<HTMLDivElement>('dropZoneOverlay')
const canvasPresetButtons = Array.from(
  posterPreviewControls.querySelectorAll<HTMLButtonElement>('[data-canvas-preset]'),
)
const collapsibleInspectorSections = Array.from(
  document.querySelectorAll<HTMLDetailsElement>('.sidebar-section'),
)

let state = createDemoState()
let rawSvgStrings: string[] = []
let stateSourceLabel = '当前数据：加载默认 SVG Demo 中…'
let currentSlideIndex = 0
let editorViewMode: EditorViewMode = 'workspace'
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
let latestAiHandoffNote: DownloadArtifact | null = null
let aiHandoffStatusMessage = AI_HANDOFF_IDLE_MESSAGE
let aiHandoffStatusTone: 'default' | 'error' = 'default'
let assetLibraryStatusMessage = ASSET_LIBRARY_IDLE_MESSAGE
let assetLibraryStatusTone: 'default' | 'error' = 'default'
let activeColorSchemeId: string | null = null
let activeFontSchemeId: string | null = null

const pointerInteractionBoundsContext = {
  measureElementBounds,
  minTextWidth: MIN_TEXT_WIDTH,
}

const canvasElementCursorsContext = {
  getCanvasElementNodes,
  get editingTextId() {
    return editingTextId
  },
  getSelectedElement,
  isTransformableElement,
}

const resizeHandlePointerDownContext = {
  get editingTextId() {
    return editingTextId
  },
  getSelectedElement,
  isTransformableElement,
  setInteractionHint(value: string | null) {
    interactionHint = value
  },
  renderInspector,
  getCanvasSvg,
  measureElementBounds,
  minTextWidth: MIN_TEXT_WIDTH,
  getElementInteractionBounds(element: EditableElement, svg: SVGSVGElement) {
    return getElementInteractionBounds(element, svg, pointerInteractionBoundsContext)
  },
  cloneTransformableElement<T extends TransformableElement>(element: T): T {
    return cloneTransformableElement(element)
  },
  captureTextNodeSnapshots,
  setPointerSession(session: PointerInteractionSession | null) {
    pointerSession = session
  },
}

const resizeApplicationContext = {
  get canvas() {
    return state.canvas
  },
  clamp,
  minTextWidth: MIN_TEXT_WIDTH,
  minResizeSize: MIN_RESIZE_SIZE,
  minCircleDiameter: MIN_CIRCLE_DIAMETER,
}

const liveElementPreviewContext = {
  getCanvasElementNodes,
  document,
}

const inspectorPreviewContext = {
  selectionSummary,
  elementFields,
  buildSelectionSummary,
  getInspectorFields,
  getFieldValue,
  syncInspectorControlValue(control: InspectorControl, field: InspectorField, value: string) {
    syncInspectorControlValue(control, field, value)
  },
}

const textEditingContext: TextEditingContext = {
  document,
  elementFields,
  svgNamespace: SVG_NS,
  get editingTextId() {
    return editingTextId
  },
  get activeTextEditor() {
    return activeTextEditor
  },
  get selectedElementId() {
    return selectedElementId
  },
  getCanvasSvg,
  getCanvasElementNodes,
  measureElementBounds,
  findElementById,
  getCurrentSlide,
  syncSlideElementIntoSvg,
  createPropertyPatch,
  commitPatchOperations,
  commitCurrentSlideFromLiveCanvasSvg,
  renderCanvas,
  renderThumbnails,
  renderInspector,
  refreshCanvasOverlays,
  setEditingTextId(value: string | null) {
    editingTextId = value
  },
  setActiveTextEditor(editor: ActiveTextEditor | null) {
    activeTextEditor = editor
  },
  setHoveredElementId(value: string | null) {
    hoveredElementId = value
  },
  requestAnimationFrame(callback: FrameRequestCallback) {
    return window.requestAnimationFrame(callback)
  },
  getCurrentSlideIndex() {
    return currentSlideIndex
  },
}

function handleOverlayResizeHandlePointerDown(event: PointerEvent): void {
  handleResizeHandlePointerDown(event, resizeHandlePointerDownContext)
}

bindCanvasBlankInteractions()
bindInspectorInteractions()
bindToolbarFileActions()
bindPosterPreviewInteractions()
bindAiHandoffInteractions()
bindAssetImportInteractions()
bindGlobalDropZone()
bindDevServerStateSync()
bindCanvasViewportSizing()
collapseInspectorSectionsByDefault()
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
    void handleSaveShortcut()
    return
  }

  if (event.key === 'Escape') {
    if (editingTextId) {
      exitTextEditing(textEditingContext)
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
  const hasMultipleSlides = state.slides.length > 1
  const isSingleSlide = !hasMultipleSlides
  const topbarSaveActionsState = getTopbarSaveActionsState(state)
  if (hasMultipleSlides) editorViewMode = 'workspace'
  const isPreviewMode = isSingleSlide && editorViewMode === 'preview'
  filmstrip?.toggleAttribute('hidden', !hasMultipleSlides)
  editorShell?.classList.toggle('editor-shell--single-slide', isSingleSlide)
  editorShell?.classList.toggle('editor-shell--preview', isPreviewMode)
  renderPosterPreviewControls(isSingleSlide, isPreviewMode)
  syncInteractionState(slide)
  renderCanvas(slide, currentSlideIndex)
  renderMeta(slide)
  renderThumbnails()
  renderColorPresets()
  renderFontPresets()
  renderInspector()
  renderPatchCountBadge()
  renderAiHandoffPanel()
  renderAssetImportPanel()
  exportSvgButton.textContent = topbarSaveActionsState.exportSvgLabel
  saveTemplateButton.hidden = !topbarSaveActionsState.showSaveTemplate
  pageIndicator.textContent = `${currentSlideIndex + 1} / ${state.slides.length}`
  prevButton.disabled = currentSlideIndex === 0
  nextButton.disabled = currentSlideIndex === state.slides.length - 1
  syncHistoryButtons()
  stateSourceBadge.textContent = stateSourceLabel
  document.title = `Design Editor · ${slide.id}`
}

function collapseInspectorSectionsByDefault(): void {
  collapsibleInspectorSections.forEach(section => {
    section.open = section.dataset.defaultOpen === 'true'
  })
}

function goToSlide(index: number): void {
  if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })
  const nextIndex = clamp(index, 0, state.slides.length - 1)
  if (nextIndex === currentSlideIndex) return
  currentSlideIndex = nextIndex
  render()
}

function renderCanvas(slide: Slide, index: number): void {
  canvasMount.innerHTML = getRenderableSlideSvgMarkup(slide, index, 'canvas')
  syncCanvasStageSize()
  bindCanvasElementInteractions()
  refreshCanvasOverlays()
}

function bindCanvasViewportSizing(): void {
  if (typeof ResizeObserver === 'undefined') {
    window.addEventListener('resize', syncCanvasStageSize)
    return
  }

  const observer = new ResizeObserver(() => {
    syncCanvasStageSize()
  })
  observer.observe(canvasScroll)
}

function syncCanvasStageSize(): void {
  const canvasWidth = state.canvas.width
  const canvasHeight = state.canvas.height
  if (!canvasWidth || !canvasHeight) {
    canvasMount.style.width = ''
    return
  }

  const styles = window.getComputedStyle(canvasScroll)
  const paddingX = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight)
  const paddingY = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom)
  const availableWidth = Math.max(0, canvasScroll.clientWidth - paddingX)
  const availableHeight = Math.max(0, canvasScroll.clientHeight - paddingY)
  if (!availableWidth || !availableHeight) {
    canvasMount.style.width = ''
    return
  }

  const aspectRatio = canvasWidth / canvasHeight
  const fitWidth = Math.min(availableWidth, availableHeight * aspectRatio, CANVAS_STAGE_MAX_WIDTH)
  canvasMount.style.width = `${Math.max(0, Math.floor(fitWidth))}px`
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
    const svg = getRenderableSlideSvgMarkup(slide, index, 'thumb')
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

function renderColorPresets(): void {
  colorPresetGrid.innerHTML = COLOR_SCHEMES.map(scheme => {
    const swatches = [scheme.background, scheme.primary, scheme.secondary, scheme.accent]
      .map(color => `<span class="preset-card__swatch" style="background:${escapeHtml(color)}"></span>`)
      .join('')
    const previewStyle = [
      `--preset-bg:${escapeHtml(scheme.background)}`,
      `--preset-bg-alt:${escapeHtml(scheme.backgroundAlt)}`,
      `--preset-text-dark:${escapeHtml(scheme.textDark)}`,
      `--preset-text-light:${escapeHtml(scheme.textLight)}`,
      `--preset-text-muted:${escapeHtml(scheme.textMuted)}`,
      `--preset-primary:${escapeHtml(scheme.primary)}`,
      `--preset-secondary:${escapeHtml(scheme.secondary)}`,
      `--preset-accent:${escapeHtml(scheme.accent)}`,
    ].join(';')

    return `
      <button
        type="button"
        class="preset-card preset-card--color${activeColorSchemeId === scheme.id ? ' is-active' : ''}"
        data-color-preset-id="${scheme.id}"
        title="${escapeHtml(scheme.name)}"
        aria-label="应用配色方案：${escapeHtml(scheme.name)}"
      >
        <div class="preset-card__preview preset-card__preview--color" style="${previewStyle}">
          <span class="preset-card__preview-badge">Aa</span>
          <div class="preset-card__preview-sample">
            <strong>标题预览</strong>
            <span>正文对比</span>
          </div>
        </div>
        <div class="preset-card__swatches" aria-hidden="true">${swatches}</div>
        <div class="preset-card__name">${escapeHtml(scheme.name)}</div>
      </button>
    `
  }).join('')

  for (const button of Array.from(colorPresetGrid.querySelectorAll<HTMLButtonElement>('[data-color-preset-id]'))) {
    button.addEventListener('click', () => {
      const presetId = button.dataset.colorPresetId
      const preset = COLOR_SCHEMES.find(candidate => candidate.id === presetId)
      if (preset) applyColorPreset(preset, createColorPresetContext())
    })
  }
}

function renderFontPresets(): void {
  fontPresetGrid.innerHTML = FONT_SCHEMES.map(scheme => `
    <button
      type="button"
      class="preset-card preset-card--font${activeFontSchemeId === scheme.id ? ' is-active' : ''}"
      data-font-preset-id="${scheme.id}"
      title="${escapeHtml(scheme.name)}"
      aria-label="应用字体风格：${escapeHtml(scheme.name)}"
    >
      <div class="preset-card__header">
        <strong>${escapeHtml(scheme.name)}</strong>
      </div>
    </button>
  `).join('')

  for (const button of Array.from(fontPresetGrid.querySelectorAll<HTMLButtonElement>('[data-font-preset-id]'))) {
    button.addEventListener('click', () => {
      const presetId = button.dataset.fontPresetId
      const preset = FONT_SCHEMES.find(candidate => candidate.id === presetId)
      if (preset) applyFontPreset(preset, createFontPresetContext())
    })
  }
}

function createColorPresetContext(): ApplyColorPresetContext {
  return {
    editingTextId,
    exitTextEditing: (options?: { shouldRender?: boolean }) => {
      exitTextEditing(textEditingContext, options)
    },
    state,
    rawSvgStrings,
    findElementById,
    createPropertyPatch,
    commitPatchOperations,
    setActiveColorSchemeId: (id: string) => {
      activeColorSchemeId = id
    },
    setInteractionHint: (value: string) => {
      interactionHint = value
    },
    setStateSourceLabel: (value: string) => {
      stateSourceLabel = value
    },
    render,
  }
}

function createFontPresetContext(): ApplyFontPresetContext {
  return {
    editingTextId,
    exitTextEditing: (options?: { shouldRender?: boolean }) => {
      exitTextEditing(textEditingContext, options)
    },
    state,
    rawSvgStrings,
    findElementById,
    createPropertyPatch,
    commitPatchOperations,
    setActiveFontSchemeId: (id: string) => {
      activeFontSchemeId = id
    },
    setInteractionHint: (value: string) => {
      interactionHint = value
    },
    setStateSourceLabel: (value: string) => {
      stateSourceLabel = value
    },
    render,
  }
}

function renderInspector(): void {
  const selected = getSelectedElement()

  if (!selected) {
    interactionHint = null
    selectionSummary.innerHTML = ''
    selectionEmptyState.hidden = false
    elementFields.hidden = true
    elementFields.innerHTML = ''
    alignToolbar.hidden = true
    return
  }

  selectionSummary.innerHTML = buildSelectionSummary(selected)
  selectionEmptyState.hidden = true
  elementFields.hidden = false
  alignToolbar.hidden = !isTransformableElement(selected)

  if (!isSupportedEditableElement(selected)) {
    elementFields.innerHTML = `
      <div class="selection-empty">
        当前选中的是 group 容器。请直接点击内部具体元素进行属性编辑。
      </div>
    `
    return
  }

  const fields = getInspectorFields(selected)
  const primaryFields = fields.filter(field => !field.advanced)
  const advancedFields = fields.filter(field => field.advanced)

  elementFields.innerHTML = [
    primaryFields.map(field => renderInspectorField(field, getFieldValue(selected, field.key))).join(''),
    advancedFields.length > 0
      ? `
        <details class="inspector-advanced">
          <summary>高级</summary>
          <div class="inspector-advanced__body">
            ${advancedFields.map(field => renderInspectorField(field, getFieldValue(selected, field.key))).join('')}
          </div>
        </details>
      `
      : '',
  ].join('')
}

function renderAiHandoffPanel(): void {
  const slide = getCurrentSlide()
  const selected = getSelectedElement()
  const isSingleSlide = state.slides.length === 1
  const scopeLabel = selected ? '当前选中元素' : '当前页面'
  const elementLabel = selected ? selected.id : '页面级'
  const projectPathHint = inferProjectPathHint(watchedStatePath)
  const projectLabel = projectPathHint ? projectPathHint : '未绑定本地项目'
  const refreshLabel = projectPathHint
    ? '刷新 · 已绑定项目，当前通过 compat state 自动刷新'
    : '刷新 · SVG 页面优先；compat state 自动刷新未启用'

  if (isSingleSlide) {
    aiTargetSummary.innerHTML = `
      <div class="selection-summary__meta">
        <span>${escapeHtml(selected ? '当前选中内容' : '当前整张海报')}</span>
        <span>${escapeHtml(projectPathHint ? '已绑定项目；SVG 预览保持主路径，compat state 变更会自动刷新' : '先绑定项目；默认仍以 SVG 页面预览，必要时再接 compat state 自动刷新')}</span>
      </div>
    `
  } else {
    aiTargetSummary.innerHTML = `
      <div class="selection-summary__meta">
        <span>${escapeHtml(`范围 · ${scopeLabel}`)}</span>
        <span>${escapeHtml(`Slide · ${slide.id}`)}</span>
        <span>${escapeHtml(`目标 · ${elementLabel}`)}</span>
        <span>${escapeHtml(`项目 · ${projectLabel}`)}</span>
        <span>${escapeHtml(refreshLabel)}</span>
      </div>
    `
  }

  watchFileButton.textContent = projectPathHint
    ? '更换项目绑定'
    : '绑定项目预览'
  exportAiTaskButton.textContent = projectPathHint
    ? (isSingleSlide ? '应用修改' : '发送给 AI')
    : (isSingleSlide ? '导出请求' : '导出 AI 请求')
  exportAiTaskButton.disabled = aiInstructionInput.value.trim().length === 0
  downloadAiHandoffNoteButton.disabled = latestAiHandoffNote === null
  copyAiHandoffNoteButton.disabled = latestAiHandoffNote === null
  aiHandoffPreviewWrap.hidden = latestAiHandoffNote === null
  aiHandoffPreview.value = latestAiHandoffNote?.content ?? ''
  aiHandoffStatus.textContent = aiHandoffStatusMessage
  aiHandoffStatus.dataset.tone = aiHandoffStatusTone
}

function renderAssetImportPanel(): void {
  if (state.slides.length === 1) {
    assetImportSummary.innerHTML = `
      <div class="selection-summary__meta">
        <span>可导入模板页、图表页或旧 Patch</span>
        <span>${escapeHtml(`当前画布 · ${state.canvas.width} × ${state.canvas.height}`)}</span>
      </div>
    `
  } else {
    assetImportSummary.innerHTML = `
      <div class="selection-summary__meta">
        <span>用途 · 复用已有资产</span>
        <span>${escapeHtml(`插入位置 · 第 ${currentSlideIndex + 1} 页后`)}</span>
        <span>${escapeHtml(`当前画布 · ${state.canvas.width} × ${state.canvas.height}`)}</span>
        <span>格式 · SVG 优先 / compat state JSON</span>
      </div>
    `
  }
  assetLibraryStatus.textContent = assetLibraryStatusMessage
  assetLibraryStatus.dataset.tone = assetLibraryStatusTone
}

function buildSelectionSummary(element: EditableElement): string {
  if (!interactionHint) return ''
  return `<div class="selection-summary__hint">${escapeHtml(interactionHint)}</div>`
}

function getInspectorFields(element: SupportedEditableElement): InspectorField[] {
  switch (element.type) {
    case 'text':
      return [
        { key: 'text', label: '文字内容', input: 'textarea' },
        { key: 'fontSize', label: '字号', input: 'range', min: 12, max: 120, step: '1' },
        { key: 'fill', label: '颜色', input: 'color' },
        { key: 'fontWeight', label: '粗细', input: 'select', options: FONT_WEIGHT_OPTIONS },
        { key: 'x', label: 'X 坐标', input: 'number', step: 'any', advanced: true },
        { key: 'y', label: 'Y 坐标', input: 'number', step: 'any', advanced: true },
        { key: 'width', label: '宽度', input: 'number', step: 'any', advanced: true },
        { key: 'font', label: '字体写法', input: 'text', advanced: true },
        { key: 'lineHeight', label: '行高', input: 'number', step: 'any', advanced: true },
      ]
    case 'rect':
      return [
        { key: 'fill', label: '颜色', input: 'color' },
        { key: 'rx', label: '圆角', input: 'range', min: 0, max: 50, step: '1' },
        { key: 'stroke', label: '边框颜色', input: 'color' },
        { key: 'x', label: 'X 坐标', input: 'number', step: 'any', advanced: true },
        { key: 'y', label: 'Y 坐标', input: 'number', step: 'any', advanced: true },
        { key: 'width', label: '宽度', input: 'number', step: 'any', advanced: true },
        { key: 'height', label: '高度', input: 'number', step: 'any', advanced: true },
      ]
    case 'path':
      return [
        { key: 'fill', label: '颜色', input: 'color' },
        { key: 'd', label: '路径数据', input: 'textarea', readOnly: true, advanced: true },
      ]
    case 'image':
      return [
        { key: 'href', label: '图片路径', input: 'text' },
        { key: 'x', label: 'X 坐标', input: 'number', step: 'any', advanced: true },
        { key: 'y', label: 'Y 坐标', input: 'number', step: 'any', advanced: true },
        { key: 'width', label: '宽度', input: 'number', step: 'any', advanced: true },
        { key: 'height', label: '高度', input: 'number', step: 'any', advanced: true },
      ]
    case 'line':
      return [
        { key: 'stroke', label: '颜色', input: 'color' },
        { key: 'x1', label: '起点 X', input: 'number', step: 'any', advanced: true },
        { key: 'y1', label: '起点 Y', input: 'number', step: 'any', advanced: true },
        { key: 'x2', label: '终点 X', input: 'number', step: 'any', advanced: true },
        { key: 'y2', label: '终点 Y', input: 'number', step: 'any', advanced: true },
      ]
    case 'circle':
      return [
        { key: 'fill', label: '颜色', input: 'color' },
        { key: 'r', label: '大小', input: 'number', step: 'any' },
        { key: 'cx', label: '圆心 X', input: 'number', step: 'any', advanced: true },
        { key: 'cy', label: '圆心 Y', input: 'number', step: 'any', advanced: true },
      ]
  }
}

function renderInspectorField(field: InspectorField, value: string): string {
  const readonlyAttr = field.readOnly ? ' readonly' : ''
  const stepAttr = field.step ? ` step="${field.step}"` : ''
  const minAttr = field.min === undefined ? '' : ` min="${field.min}"`
  const maxAttr = field.max === undefined ? '' : ` max="${field.max}"`
  const baseId = `prop-${field.key}`

  if (field.input === 'textarea') {
    return `
      <div class="property-field property-field--textarea">
        <label for="${baseId}">${escapeHtml(field.label)}</label>
        <textarea id="${baseId}" data-prop-key="${field.key}"${readonlyAttr}>${escapeHtml(value)}</textarea>
      </div>
    `
  }

  if (field.input === 'color') {
    const pickerValue = toColorInputValue(value)
    const textValue = formatColorDisplayValue(value)
    return `
      <div class="property-field">
        <label for="${baseId}-text">${escapeHtml(field.label)}</label>
        <div class="compound-input compound-input--color">
          <input id="${baseId}-picker" data-prop-key="${field.key}" data-prop-control="color-picker" type="color" value="${pickerValue}"${readonlyAttr} />
          <input id="${baseId}-text" data-prop-key="${field.key}" data-prop-control="color-text" type="text" value="${escapeHtml(textValue)}"${readonlyAttr} />
        </div>
      </div>
    `
  }

  if (field.input === 'range') {
    return `
      <div class="property-field">
        <label for="${baseId}-number">${escapeHtml(field.label)}</label>
        <div class="compound-input compound-input--range">
          <input id="${baseId}-range" data-prop-key="${field.key}" data-prop-control="range-slider" type="range" value="${escapeHtml(value)}"${readonlyAttr}${stepAttr}${minAttr}${maxAttr} />
          <input id="${baseId}-number" data-prop-key="${field.key}" data-prop-control="range-number" type="number" value="${escapeHtml(value)}"${readonlyAttr}${stepAttr}${minAttr}${maxAttr} />
        </div>
      </div>
    `
  }

  if (field.input === 'select') {
    const options = (field.options ?? []).map(option => `
      <option value="${escapeHtml(option.value)}"${option.value === value ? ' selected' : ''}>${escapeHtml(option.label)}</option>
    `).join('')
    return `
      <div class="property-field">
        <label for="${baseId}">${escapeHtml(field.label)}</label>
        <select id="${baseId}" data-prop-key="${field.key}"${readonlyAttr}>
          ${options}
        </select>
      </div>
    `
  }

  const type = field.input === 'number' ? 'number' : 'text'
  return `
    <div class="property-field">
      <label for="${baseId}">${escapeHtml(field.label)}</label>
      <input id="${baseId}" data-prop-key="${field.key}" type="${type}" value="${escapeHtml(value)}"${readonlyAttr}${stepAttr}${minAttr}${maxAttr} />
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
  const handleInspectorMutation = (event: Event) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement)) return
    if ((target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) && target.readOnly) return

    const key = target.dataset.propKey
    if (!key) return

    const selected = getSelectedElement()
    if (!selected || !isSupportedEditableElement(selected)) return

    const mutations = applyElementUpdate(selected, key, target.value)
    if (!mutations || mutations.length === 0) return

    const operations = mutations
      .map(mutation => createPropertyPatch(selected.id, mutation.property, mutation.oldValue, mutation.newValue))
      .filter((operation): operation is PatchOperation => Boolean(operation))

    const canvasSvg = getCanvasSvg()
    if (canvasSvg) syncSlideElementIntoSvg(selected, canvasSvg, document)
    commitCurrentSlideFromLiveCanvasSvg()

    if (operations.length > 0) {
      commitPatchOperations(operations, 'human', `修改 ${selected.id}.${key}`)
    }

    interactionHint = null
    const nextSelected = findElementById(getCurrentSlide().elements, selected.id)
    if (nextSelected && isSupportedEditableElement(nextSelected)) {
      syncInspectorPreview(nextSelected, inspectorPreviewContext)
    }
    const slide = getCurrentSlide()
    renderCanvas(slide, currentSlideIndex)
    renderThumbnails()
  }

  elementFields.addEventListener('input', handleInspectorMutation)
  elementFields.addEventListener('change', handleInspectorMutation)

  alignToolbar.addEventListener('click', event => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-align]')
    if (!button) return
    alignSelectedElement(button.dataset.align as AlignDirection)
  })
}

function bindToolbarFileActions(): void {
  undoButton.addEventListener('click', () => {
    undoLastChange()
  })

  redoButton.addEventListener('click', () => {
    redoLastChange()
  })

  exportSvgButton.addEventListener('click', () => {
    downloadCurrentCanvasSvg()
  })

  exportPngButton.addEventListener('click', () => {
    void downloadCurrentCanvasPng()
  })

  saveTemplateButton.addEventListener('click', () => {
    downloadCurrentCanvasTemplateSvg()
  })

  exportPatchButton.addEventListener('click', () => {
    downloadArtifacts([createDesignPatchDownload(patches)])
  })

  watchFileButton.addEventListener('click', () => {
    const defaultPath = watchedStatePath ?? ''
    const input = window.prompt(
      '请输入项目里的 compat state 路径（例如 /@fs/.../slide_state.json）。编辑器默认仍以 SVG 页面为主路径；这里仅用于本地自动刷新。',
      defaultPath,
    )
    const nextPath = input?.trim()
    if (!nextPath) return
    void startManualStateWatch(nextPath).catch(error => {
      console.error(`启动 compat state 监听失败: ${nextPath}`, error)
    })
  })
}

function bindPosterPreviewInteractions(): void {
  posterPreviewToggleButton.addEventListener('click', () => {
    if (!shouldShowPosterCanvasControls(state)) return
    editorViewMode = editorViewMode === 'preview' ? 'workspace' : 'preview'
    render()
  })

  for (const button of canvasPresetButtons) {
    button.addEventListener('click', () => {
      const presetId = button.dataset.canvasPreset
      const preset = CANVAS_PRESETS.find(candidate => candidate.id === presetId)
      if (!preset || !shouldShowPosterCanvasControls(state) || !supportsCanvasPresetEditing(state)) return

      if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })

      const nextState = resizeSlideStateCanvas(state, preset)
      const previousState = state
      editorViewMode = 'preview'
      if (nextState === state) {
        render()
        return
      }

      state = nextState
      rawSvgStrings = reconcileRawSvgStrings(previousState, nextState)
      latestAiHandoffNote = null
      historyPast = []
      historyFuture = []
      patches = []
      interactionHint = `海报比例已切换为 ${preset.label}`
      aiHandoffStatusTone = 'default'
      aiHandoffStatusMessage = `已切换到 ${preset.label} 画幅，预览已实时刷新。`
      resetTransientInteractionState({ clearSelection: true, clearHint: false })
      render()
    })
  }
}

function renderPosterPreviewControls(isSingleSlide: boolean, isPreviewMode: boolean): void {
  const shouldShowControls = isSingleSlide && shouldShowPosterCanvasControls(state)
  posterPreviewControls.hidden = !shouldShowControls
  if (!shouldShowControls) return

  const supportsPresets = supportsCanvasPresetEditing(state)
  const activePreset = findCanvasPreset(state.canvas)

  posterPreviewToggleButton.textContent = isPreviewMode ? '返回编辑' : '纯预览'
  posterPreviewToggleButton.dataset.active = String(isPreviewMode)
  posterPreviewToggleButton.ariaPressed = String(isPreviewMode)

  for (const button of canvasPresetButtons) {
    const presetId = button.dataset.canvasPreset ?? ''
    const preset = CANVAS_PRESETS.find(candidate => candidate.id === presetId)
    button.disabled = !supportsPresets
    button.dataset.active = String(activePreset?.id === presetId)
    button.title = supportsPresets
      ? `切换海报画幅为 ${preset?.label ?? presetId}`
      : '当前页面含复杂 path，暂不支持实时比例切换'
  }
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
    latestAiHandoffNote = null
    aiHandoffStatusTone = 'default'
    aiHandoffStatusMessage = AI_HANDOFF_IDLE_MESSAGE
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

  downloadAiHandoffNoteButton.addEventListener('click', () => {
    downloadAiHandoffNote()
  })

  copyAiHandoffNoteButton.addEventListener('click', () => {
    void copyAiHandoffNote()
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
  const svgPaths = getSvgPathsFromSearch(window.location.search)
  if (svgPaths.length > 0) {
    try {
      const bundle = await loadStateFromSvgUrls(svgPaths)
      replaceState(bundle.state, `当前数据：SVG 导入 (${svgPaths.length} 页)`, {
        rawSvgStrings: bundle.rawSvgStrings,
      })
    } catch (error) {
      console.error(`URL 加载 SVG 失败: ${svgPaths.join(', ')}`, error)
      stateSourceLabel = '当前数据：内置 Demo（SVG URL 加载失败）'
      render()
    }
    return
  }

  const statePath = getStatePathFromSearch(window.location.search)
  if (statePath) {
    watchedStatePath = statePath
    try {
      await reloadStateFromRemote(statePath, `当前数据：兼容 state URL ${statePath}`, statePath)
      return
    } catch (error) {
      console.error(`URL 加载 compat state 失败: ${statePath}`, error)
      stateSourceLabel = '当前数据：内置 SVG Demo（compat state URL 加载失败）'
      render()
      return
    }
  }

  try {
    const bundle = await loadStateFromSvgUrls([DEFAULT_DEMO_SVG_PATH])
    replaceState(bundle.state, '当前数据：默认 Demo（SVG 示例）', {
      rawSvgStrings: bundle.rawSvgStrings,
    })
  } catch (error) {
    console.error(`默认 Demo SVG 加载失败: ${DEFAULT_DEMO_SVG_PATH}`, error)
    replaceState(createDemoState(), '当前数据：内置 Demo（默认 SVG 示例加载失败）')
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
      replaceState(nextState, `当前数据：兼容 state · ${file.name}`)
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
    const bundle = await loadSvgBundleFromFiles(files)
    stopManualStateWatch()
    watchedStatePath = null
    watchedStateRawSnapshot = null
    replaceState(bundle.state, `当前数据：SVG 导入 (${bundle.state.slides.length} 页)`, {
      rawSvgStrings: bundle.rawSvgStrings,
    })
  } catch (error) {
    console.error('SVG 文件导入失败', error)
  }
}

async function loadSvgBundleFromFiles(files: File[]): Promise<SvgImportBundle> {
  const svgFiles = files.filter(isSvgFile).sort(compareFileName)
  if (svgFiles.length === 0) {
    throw new Error('未检测到可导入的 SVG 文件')
  }

  const svgStrings = await Promise.all(svgFiles.map(file => file.text()))
  const slideIds = svgFiles.map(file => toSlideIdFromFileName(file.name))
  const sourcePaths = svgFiles.map(file => file.name)
  return createSvgImportBundle(svgStrings, slideIds, sourcePaths)
}

function createSvgImportBundle(
  svgStrings: string[],
  slideIds: string[],
  sourcePaths: string[],
): SvgImportBundle {
  const normalizedSvgStrings = svgStrings.map((svg, index) => normalizeSvgForEditor(svg, {
    idPrefix: slideIds[index],
    sourcePath: sourcePaths[index],
  }))

  return {
    rawSvgStrings: normalizedSvgStrings,
    state: svgsToState(normalizedSvgStrings, slideIds, undefined, {
      preserveTextNodes: true,
    }),
  }
}

async function appendImportedSlidesFromFiles(
  files: File[],
  label: '模板页' | '图表页',
  allowJson: boolean,
): Promise<void> {
  try {
    if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })
    const importedBundle = await readImportedSlidesBundle(files, allowJson)
    ensureCompatibleCanvas(state.canvas, importedBundle.state.canvas)

    stopManualStateWatch()
    watchedStatePath = null
    watchedStateRawSnapshot = null

    const insertIndex = currentSlideIndex + 1
    const operations = createAppendSlidesPatch({
      state,
      importedSlides: importedBundle.state.slides,
      insertIndex,
      source: 'human',
    })

    if (operations.length === 0) {
      throw new Error(`未检测到可追加的${label}`)
    }

    const previousState = state
    state = applyDesignPatch(state, {
      timestamp: new Date().toISOString(),
      source: 'human',
      operations,
    })
    const nextRawSvgStrings = reconcileRawSvgStrings(previousState, state)
    if (importedBundle.rawSvgStrings.length > 0) {
      nextRawSvgStrings.splice(insertIndex, 0, ...importedBundle.rawSvgStrings)
    }
    rawSvgStrings = nextRawSvgStrings
    commitPatchOperations(operations, 'human', `追加${label}`)
    currentSlideIndex = clamp(insertIndex, 0, state.slides.length - 1)
    stateSourceLabel = `当前数据：已追加${label} (${operations.length} 页)`
    assetLibraryStatusTone = 'default'
    assetLibraryStatusMessage = `已追加 ${operations.length} 页${label}。这是高级兼容入口；这些页面接下来仍会进入同一套预览、手调和 AI 回合。`
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

async function readImportedSlidesBundle(files: File[], allowJson: boolean): Promise<SvgImportBundle> {
  if (allowJson) {
    const jsonFile = files.find(file => isJsonFile(file))
    if (jsonFile) {
      return {
        state: await readSlideStateFile(jsonFile),
        rawSvgStrings: [],
      }
    }
  }

  const svgFiles = files.filter(file => isSvgFile(file))
  if (svgFiles.length === 0) {
    throw new Error('请选择 .svg 文件；如需导入旧模板，也可使用 compat slide_state JSON。')
  }

  return loadSvgBundleFromFiles(svgFiles)
}

function exportAiHandoff(): void {
  const instruction = aiInstructionInput.value.trim()
  if (!instruction) {
    aiInstructionInput.focus()
    aiHandoffStatusTone = 'error'
    aiHandoffStatusMessage = '请先输入这一轮想让 AI 做什么，再发送请求。'
    renderAiHandoffPanel()
    return
  }

  const aiCommand = createEditorAiCommand(instruction)
  const projectPathHint = inferProjectPathHint(watchedStatePath)
  const stateFilePathHint = inferStateFilePathHint(watchedStatePath)
  const requestRelativePath = buildProjectAiHandoffRelativePath('design_patch.ai-request.json')
  const requestArtifact = createAiCommandDownload(aiCommand, patches)
  const noteArtifact = createAiCommandPromptDownload(aiCommand, patches, {
    projectPathHint,
    stateFilePathHint,
    requestFileName: projectPathHint ? requestRelativePath : undefined,
  })

  latestAiHandoffNote = noteArtifact
  const afterWrite = (summary: string) => {
    interactionHint = aiCommand.scope === 'selected-element'
      ? `已导出本地 AI handoff，可交给 Claude Code / Codex 在项目内修改元素 ${aiCommand.elementId}`
      : '已导出本地 AI handoff，可交给 Claude Code / Codex 在项目内修改当前页面'
    aiHandoffStatusTone = 'default'
    aiHandoffStatusMessage = summary
    renderInspector()
    renderAiHandoffPanel()
  }

  if (projectPathHint) {
    void writeAiHandoffBundleToProject(projectPathHint, requestArtifact, noteArtifact)
      .then(result => {
        afterWrite(
          `已写入 ${result.requestPath} 和 ${result.notePath}。Claude Code / Codex 现在可通过 compat bridge 修改项目里的 slide_state.json；文件一变化，这个 SVG 预览就会自动刷新。`,
        )
      })
      .catch(error => {
        downloadArtifacts([requestArtifact])
        afterWrite(
          `写入项目临时目录失败，已回退下载 ${requestArtifact.fileName}：${(error as Error).message}。当前不会自动刷新，可稍后重新绑定项目，或手动回流 AI 返回的 patch/json。`,
        )
      })
    return
  }

  downloadArtifacts([requestArtifact])
  afterWrite(
    `当前未绑定项目，已下载 ${requestArtifact.fileName}。把它交给 Claude Code / Codex 后，如需自动刷新，请再绑定项目里的 compat state（如 slide_state.json）。`,
  )
}

async function handleSaveShortcut(): Promise<void> {
  const projectPathHint = inferProjectPathHint(watchedStatePath)
  if (!projectPathHint) {
    downloadCurrentCanvasSvg()
    return
  }

  if (editingTextId) exitTextEditing(textEditingContext)

  try {
    const result = await writeProjectPagesToProject(projectPathHint)
    const summary = `已保存 ${result.savedCount} 页到 ${result.dir}`
    aiHandoffStatusTone = 'default'
    aiHandoffStatusMessage = `${summary}。当前仍保持项目绑定，可继续通过 compat state 自动刷新预览。`
    interactionHint = summary
  } catch (error) {
    console.error(`保存 design/pages 失败: ${projectPathHint}`, error)
    const message = `保存失败：${(error as Error).message}`
    aiHandoffStatusTone = 'error'
    aiHandoffStatusMessage = `${message}。未绑定项目时，⌘/Ctrl+S 仍会继续下载当前页 SVG。`
    interactionHint = message
  }

  // Inspector 在无选中元素时会清空 interactionHint，因此优先刷新始终可见的状态栏。
  renderAiHandoffPanel()
  renderInspector()
}

async function writeProjectPagesToProject(projectPath: string): Promise<SavePagesResult> {
  const response = await fetch(LOCAL_PROJECT_SAVE_PAGES_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      projectPath,
      pages: createProjectPageSavePayloads(),
    }),
  })

  const payload = await response.json().catch(() => null) as
    | { ok?: unknown; savedCount?: unknown; dir?: unknown; error?: unknown }
    | null

  if (!response.ok) {
    const errorMessage = typeof payload?.error === 'string'
      ? payload.error
      : `HTTP ${response.status}`
    throw new Error(errorMessage)
  }

  if (payload?.ok !== true || typeof payload.savedCount !== 'number' || typeof payload.dir !== 'string') {
    throw new Error('本地 SVG 保存响应无效')
  }

  return {
    savedCount: payload.savedCount,
    dir: payload.dir,
  }
}

function createProjectPageSavePayloads(): ProjectPageSavePayload[] {
  const artifacts = createProjectSvgArtifacts(state)
  return artifacts.map((artifact, index) => ({
    filename: artifact.fileName,
    svg: getSyncedRawSvgString(index, state.slides[index]) ?? artifact.content,
  }))
}

function downloadAiHandoffNote(): void {
  if (!latestAiHandoffNote) return
  downloadArtifacts([latestAiHandoffNote])
  aiHandoffStatusTone = 'default'
  aiHandoffStatusMessage = `已触发 ${latestAiHandoffNote.fileName} 下载；如果浏览器没有落盘，可直接复制下方预览内容。`
  renderAiHandoffPanel()
}

async function writeAiHandoffBundleToProject(
  projectPath: string,
  requestArtifact: DownloadArtifact,
  noteArtifact: DownloadArtifact,
): Promise<{ requestPath: string; notePath: string }> {
  const response = await fetch(LOCAL_AI_HANDOFF_WRITE_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      projectPath,
      requestArtifact,
      noteArtifact,
    }),
  })

  const payload = await response.json().catch(() => null) as
    | { requestPath?: unknown; notePath?: unknown; error?: unknown }
    | null

  if (!response.ok) {
    const errorMessage = typeof payload?.error === 'string'
      ? payload.error
      : `HTTP ${response.status}`
    throw new Error(errorMessage)
  }

  if (typeof payload?.requestPath !== 'string' || typeof payload?.notePath !== 'string') {
    throw new Error('本地 handoff 写盘响应无效')
  }

  return {
    requestPath: payload.requestPath,
    notePath: payload.notePath,
  }
}

async function copyAiHandoffNote(): Promise<void> {
  if (!latestAiHandoffNote) return

  try {
    await navigator.clipboard.writeText(latestAiHandoffNote.content)
    aiHandoffStatusTone = 'default'
    aiHandoffStatusMessage = `已复制 ${latestAiHandoffNote.fileName} 内容。现在可直接粘贴给 Claude Code / Codex。`
  } catch (error) {
    aiHandoffPreviewWrap.hidden = false
    aiHandoffPreview.focus()
    aiHandoffPreview.select()
    aiHandoffStatusTone = 'error'
    aiHandoffStatusMessage = `复制失败：${(error as Error).message}。已选中下方预览内容，可直接按 Cmd+C / Ctrl+C。`
  }

  renderAiHandoffPanel()
}

function handleImportedDesignPatch(designPatch: DesignPatch, sourceLabel: string): void {
  if (designPatch.operations.length === 0) {
    if (designPatch.aiCommand) {
      hydrateAiCommandFromPatch(designPatch.aiCommand)
      aiHandoffStatusTone = 'default'
      aiHandoffStatusMessage = `已载入 AI 请求：${describeAiScope(designPatch.aiCommand)}`
      render()
      return
    }

    throw new Error('design_patch 不包含可应用的 operations')
  }

  stopManualStateWatch()
  watchedStatePath = null
  watchedStateRawSnapshot = null
  if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })

  const nextState = applyDesignPatch(state, designPatch)
  rawSvgStrings = reconcileRawSvgStrings(state, nextState)
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
  aiHandoffStatusMessage = `已应用 ${designPatch.operations.length} 条 Patch，预览已刷新：${sourceLabel}`
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
      `已通过 compat state 自动刷新: ${payload.filePath ?? targetPath}`,
      payload.filePath ?? targetPath,
    ).catch(error => {
      console.error(`热更新加载 compat state 失败: ${payload.filePath ?? targetPath}`, error)
    })
  })
}

async function startManualStateWatch(path: string): Promise<void> {
  stopManualStateWatch()
  watchedStatePath = path
  watchedStateRawSnapshot = null
  await reloadStateFromRemote(path, `已通过 compat state 自动刷新: ${path}`, path)
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
    replaceState(nextState, `已通过 compat state 自动刷新: ${watchedStatePath}`)
  } catch (error) {
    console.error(`轮询 compat state 失败: ${watchedStatePath}`, error)
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

async function loadStateFromSvgUrls(paths: string[]): Promise<SvgImportBundle> {
  const svgStrings = await Promise.all(paths.map(async path => {
    const response = await fetch(path, { cache: 'no-store' })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`.trim())
    }
    return response.text()
  }))

  return createSvgImportBundle(svgStrings, paths.map(toSlideIdFromPath), paths)
}

function replaceState(
  nextState: SlideState,
  sourceLabel: string,
  options: { rawSvgStrings?: string[] } = {},
): void {
  if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })

  state = nextState
  rawSvgStrings = options.rawSvgStrings ? [...options.rawSvgStrings] : []
  stateSourceLabel = sourceLabel
  editorViewMode = 'workspace'
  historyPast = []
  historyFuture = []
  patches = []
  latestAiHandoffNote = null
  currentSlideIndex = 0
  assetLibraryStatusTone = 'default'
  assetLibraryStatusMessage = ASSET_LIBRARY_IDLE_MESSAGE
  aiHandoffStatusMessage = AI_HANDOFF_IDLE_MESSAGE
  aiHandoffStatusTone = 'default'
  resetTransientInteractionState({ clearSelection: true, clearHint: true })
  render()
}

function reconcileRawSvgStrings(previousState: SlideState, nextState: SlideState): string[] {
  if (rawSvgStrings.length === 0) return []

  const rawBySlideId = new Map<string, string>()
  previousState.slides.forEach((slide, index) => {
    const rawSvg = rawSvgStrings[index]
    if (rawSvg) rawBySlideId.set(slide.id, rawSvg)
  })

  return nextState.slides.map(slide => rawBySlideId.get(slide.id) ?? '')
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

  if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })
  const previousState = state
  state = applyDesignPatch(state, entry.inversePatch)
  rawSvgStrings = reconcileRawSvgStrings(previousState, state)
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
  aiHandoffStatusMessage = `已撤销一步：${entry.label}，预览已刷新`
  latestAiHandoffNote = null
  resetTransientInteractionState({ clearSelection: false, clearHint: false })
  render()
}

function redoLastChange(): void {
  const entry = historyFuture.shift()
  if (!entry) return

  if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })
  const previousState = state
  state = applyDesignPatch(state, entry.forwardPatch)
  rawSvgStrings = reconcileRawSvgStrings(previousState, state)
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
  aiHandoffStatusMessage = `已重做一步：${entry.label}，预览已刷新`
  latestAiHandoffNote = null
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
    downloadBlob(new Blob([artifact.content], { type: artifact.mimeType }), artifact.fileName)
  }
}

function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.hidden = true
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

function downloadCurrentCanvasSvg(): void {
  if (editingTextId) exitTextEditing(textEditingContext)

  const serializedSvg = serializeCurrentCanvasSvg()
  if (!serializedSvg) return

  downloadArtifacts([{
    fileName: `${getCurrentSlide().id}.svg`,
    mimeType: 'image/svg+xml;charset=utf-8',
    content: serializedSvg,
  }])
}

function downloadCurrentCanvasTemplateSvg(): void {
  if (editingTextId) exitTextEditing(textEditingContext)

  const serializedSvg = serializeCurrentCanvasSvg()
  if (!serializedSvg) return

  downloadArtifacts([{
    fileName: `template_${sanitizeFileNameSegment(getCurrentSlide().id)}_${formatExportTimestamp()}.svg`,
    mimeType: 'image/svg+xml;charset=utf-8',
    content: serializedSvg,
  }])
}

async function downloadCurrentCanvasPng(): Promise<void> {
  if (editingTextId) exitTextEditing(textEditingContext)

  const exportSvg = createCurrentCanvasSvgExportClone()
  if (!exportSvg) return

  try {
    const { width, height } = getSvgExportSize(exportSvg)
    const imageWarnings = await prepareSvgImagesForRasterization(exportSvg)
    const serializedSvg = serializeCurrentCanvasSvg({ svg: exportSvg })
    if (!serializedSvg) throw new Error('当前画布没有可导出的 SVG。')

    const svgBlob = new Blob([serializedSvg], { type: 'image/svg+xml;charset=utf-8' })
    const svgUrl = URL.createObjectURL(svgBlob)

    try {
      const image = await loadImageFromUrl(svgUrl)
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(width))
      canvas.height = Math.max(1, Math.round(height))

      const context = canvas.getContext('2d')
      if (!context) throw new Error('浏览器当前无法创建 PNG 画布。')

      context.clearRect(0, 0, canvas.width, canvas.height)
      context.drawImage(image, 0, 0, canvas.width, canvas.height)

      const pngBlob = await canvasToBlob(canvas, 'image/png')
      downloadBlob(pngBlob, `${sanitizeFileNameSegment(getCurrentSlide().id)}.png`)

      if (imageWarnings.length > 0) {
        window.alert('PNG 已导出，但有部分外部图片未能内嵌；若成品缺图，请把图片改成可访问绝对路径或 data URL 后重试。')
      }
    } finally {
      URL.revokeObjectURL(svgUrl)
    }
  } catch (error) {
    console.error('PNG 导出失败', error)
    window.alert(`PNG 导出失败：${(error as Error).message}`)
  }
}

function serializeCurrentCanvasSvg(options: { svg?: SVGSVGElement } = {}): string | null {
  const clone = options.svg ?? createCurrentCanvasSvgExportClone()
  if (!clone) return null
  return clone.outerHTML
}

function commitCurrentSlideFromLiveCanvasSvg(): boolean {
  const canvasSvgMarkup = serializeCurrentCanvasSvg()
  if (!canvasSvgMarkup) return false

  const result = syncCurrentSlideFromLiveSvg({
    canvasSvgMarkup,
    currentSlideIndex,
    state,
    rawSvgStrings,
  })
  if (!result) return false

  state = result.state
  rawSvgStrings = result.rawSvgStrings
  return true
}

function createCurrentCanvasSvgExportClone(): SVGSVGElement | null {
  const svg = getCanvasSvg()
  if (!svg) return null

  const clone = svg.cloneNode(true) as SVGSVGElement
  clone.querySelectorAll('[data-editor-overlay-root], [data-text-editor-root]').forEach(node => {
    node.remove()
  })
  clone.querySelectorAll<SVGElement>('[style]').forEach(node => {
    if (!(node instanceof HTMLElement) && !(node instanceof SVGElement)) return
    node.style.removeProperty('cursor')
    if (!node.getAttribute('style')?.trim()) node.removeAttribute('style')
  })
  return clone
}

async function prepareSvgImagesForRasterization(svg: SVGSVGElement): Promise<string[]> {
  const warnings: string[] = []
  const imageNodes = Array.from(svg.querySelectorAll<SVGImageElement>('image'))

  await Promise.all(imageNodes.map(async node => {
    const href = getSvgImageHref(node)
    if (!href || href.startsWith('data:') || href.startsWith('blob:')) return

    let resolvedHref = href
    try {
      resolvedHref = new URL(href, window.location.href).toString()
    } catch {
      resolvedHref = href
    }

    try {
      const response = await fetch(resolvedHref)
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`.trim())
      const dataUrl = await readBlobAsDataUrl(await response.blob())
      setSvgImageHref(node, dataUrl)
    } catch (error) {
      console.warn(`PNG 导出时图片内嵌失败: ${href}`, error)
      setSvgImageHref(node, resolvedHref)
      warnings.push(href)
    }
  }))

  return warnings
}

function getSvgImageHref(node: SVGImageElement): string {
  return node.getAttribute('href')
    || node.getAttributeNS('http://www.w3.org/1999/xlink', 'href')
    || ''
}

function setSvgImageHref(node: SVGImageElement, href: string): void {
  node.setAttribute('href', href)
  node.setAttributeNS('http://www.w3.org/1999/xlink', 'href', href)
}

function readBlobAsDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => {
      reject(reader.error ?? new Error('Blob 转 data URL 失败。'))
    }
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
        return
      }
      reject(new Error('Blob 转 data URL 失败。'))
    }
    reader.readAsDataURL(blob)
  })
}

function loadImageFromUrl(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('浏览器无法加载导出的 SVG，请检查其中的图片路径。'))
    image.src = url
  })
}

function canvasToBlob(canvas: HTMLCanvasElement, mimeType: string): Promise<Blob> {
  return new Promise((resolve, reject) => {
    try {
      canvas.toBlob(blob => {
        if (blob) {
          resolve(blob)
          return
        }
        reject(new Error('浏览器未返回 PNG 数据。'))
      }, mimeType)
    } catch (error) {
      reject(error)
    }
  })
}

function getSvgExportSize(svg: SVGSVGElement): { width: number; height: number } {
  const viewBox = svg.getAttribute('viewBox')
  if (viewBox) {
    const parts = viewBox.trim().split(/[\s,]+/).map(Number)
    if (parts.length === 4 && Number.isFinite(parts[2]) && Number.isFinite(parts[3]) && parts[2] > 0 && parts[3] > 0) {
      return { width: parts[2], height: parts[3] }
    }
  }

  const width = parseSvgLength(svg.getAttribute('width')) ?? state.canvas.width
  const height = parseSvgLength(svg.getAttribute('height')) ?? state.canvas.height
  if (width > 0 && height > 0) return { width, height }

  throw new Error('当前 SVG 缺少可用尺寸信息。')
}

function parseSvgLength(value: string | null): number | null {
  if (!value) return null
  const match = value.trim().match(/-?\d+(?:\.\d+)?/)
  if (!match) return null
  const parsed = Number(match[0])
  return Number.isFinite(parsed) ? parsed : null
}

function sanitizeFileNameSegment(value: string): string {
  const normalized = value.trim().replace(/\s+/g, '_').replace(/[^a-zA-Z0-9._-]/g, '_')
  return normalized || 'slide'
}

function formatExportTimestamp(date = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, '0')
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    '_',
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join('')
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
  const initialBounds = getElementInteractionBounds(element, svg, pointerInteractionBoundsContext)
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
  enterTextEditing(element, textEditingContext)
}

function handleDocumentPointerDown(event: PointerEvent): void {
  if (!editingTextId || !activeTextEditor) return
  if (!(event.target instanceof Node)) return
  if (activeTextEditor.foreignObject.contains(event.target)) return

  const shouldSuppressCanvasInteraction = event.target instanceof Element
    && Boolean(event.target.closest('.canvas-pane'))

  if (shouldSuppressCanvasInteraction) suppressNextCanvasClick = true
  exitTextEditing(textEditingContext)

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

  if (session.kind === 'move') {
    applyMoveFromSession(element, session.initialElement, session.initialBounds, delta, resizeApplicationContext)
  } else {
    applyResizeFromSession(element, session, delta, resizeApplicationContext)
  }

  hoveredElementId = session.elementId
  syncLiveElementPreview(element, session, liveElementPreviewContext)
  syncInspectorPreview(element, inspectorPreviewContext)
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
      commitCurrentSlideFromLiveCanvasSvg()
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
  syncCanvasElementCursors(canvasElementCursorsContext)

  if (editingTextId) return

  const overlayRoot = document.createElementNS(SVG_NS, 'g')
  overlayRoot.setAttribute('data-editor-overlay-root', 'true')
  overlayRoot.setAttribute('pointer-events', 'none')

  if (hoveredElementId && hoveredElementId !== selectedElementId) {
    const hoveredElement = findElementById(getCurrentSlide().elements, hoveredElementId)
    const hoverBounds = hoveredElement
      ? getElementInteractionBounds(hoveredElement, svg, pointerInteractionBoundsContext)
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
      ? getElementInteractionBounds(selectedElement, svg, pointerInteractionBoundsContext)
      : measureElementBounds(svg, selectedElementId, 0)
    if (selectedBounds) {
      const displayBounds = expandBounds(selectedBounds, OVERLAY_PADDING)
      const resizeHandles = getResizeHandlesForElementType(selectedElement?.type ?? '')
      overlayRoot.append(buildOverlayRect(displayBounds, {
        fill: 'none',
        stroke: '#2563EB',
        strokeOpacity: '1',
        strokeWidth: '2',
      }))

      for (const handle of getHandlePositions(displayBounds, resizeHandles)) {
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
        knob.addEventListener('pointerdown', handleOverlayResizeHandlePointerDown)
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

function getHandlePositions(
  bounds: SvgBounds,
  handles: ResizeHandle[] = ALL_RESIZE_HANDLES,
): Array<{ position: ResizeHandle; x: number; y: number }> {
  const centerX = bounds.x + bounds.width / 2
  const centerY = bounds.y + bounds.height / 2
  const positions: Record<ResizeHandle, { x: number; y: number }> = {
    nw: { x: bounds.x, y: bounds.y },
    n: { x: centerX, y: bounds.y },
    ne: { x: bounds.x + bounds.width, y: bounds.y },
    e: { x: bounds.x + bounds.width, y: centerY },
    se: { x: bounds.x + bounds.width, y: bounds.y + bounds.height },
    s: { x: centerX, y: bounds.y + bounds.height },
    sw: { x: bounds.x, y: bounds.y + bounds.height },
    w: { x: bounds.x, y: centerY },
  }

  return handles.map(position => ({ position, ...positions[position] }))
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
  if (editingTextId) exitTextEditing(textEditingContext, { shouldRender: false })
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
  return deriveProjectPathFromStateFilePath(stateFilePath)
}

function deriveProjectPathFromStateFilePath(stateFilePath: string): string | null {
  const normalizedPath = stateFilePath.replace(/\\/g, '/')
  const hasLeadingSlash = normalizedPath.startsWith('/')
  const segments = normalizedPath.split('/').filter(Boolean)
  if (segments.length < 2) return null

  const fileName = segments.pop()?.toLowerCase() ?? ''
  if (!fileName.endsWith('.json')) return null

  if (fileName === 'slide_state.json' && segments.at(-1)?.toLowerCase() === '.cache') {
    segments.pop()
  }

  if (segments.length === 0) return null
  return `${hasLeadingSlash ? '/' : ''}${segments.join('/')}`
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

type AlignDirection = 'left' | 'center-h' | 'right' | 'top' | 'center-v' | 'bottom'

function alignSelectedElement(direction: AlignDirection): void {
  const element = getSelectedElement()
  if (!element || !isTransformableElement(element)) return

  const { width: canvasW, height: canvasH } = state.canvas
  const previous = cloneTransformableElement(element)

  // Get element bounds in canvas coordinates
  const svg = getCanvasSvg()
  if (!svg) return
  const bounds = measureElementBounds(svg, element.id, 0)
  if (!bounds) return

  switch (element.type) {
    case 'text':
    case 'rect':
    case 'image': {
      const el = element as { x: number; y: number; width: number; height: number }
      switch (direction) {
        case 'left': el.x = 0; break
        case 'center-h': el.x = (canvasW - el.width) / 2; break
        case 'right': el.x = canvasW - el.width; break
        case 'top': el.y = element.type === 'text' ? bounds.y - el.y + 0 : 0; break
        case 'center-v': {
          if (element.type === 'text') {
            el.y += (canvasH - bounds.height) / 2 - bounds.y
          } else {
            el.y = (canvasH - el.height) / 2
          }
          break
        }
        case 'bottom': {
          if (element.type === 'text') {
            el.y += canvasH - bounds.height - bounds.y
          } else {
            el.y = canvasH - el.height
          }
          break
        }
      }
      break
    }
    case 'line': {
      const el = element as { x1: number; y1: number; x2: number; y2: number }
      const minX = Math.min(el.x1, el.x2)
      const maxX = Math.max(el.x1, el.x2)
      const minY = Math.min(el.y1, el.y2)
      const maxY = Math.max(el.y1, el.y2)
      const w = maxX - minX
      const h = maxY - minY
      let dx = 0, dy = 0
      switch (direction) {
        case 'left': dx = -minX; break
        case 'center-h': dx = (canvasW - w) / 2 - minX; break
        case 'right': dx = canvasW - maxX; break
        case 'top': dy = -minY; break
        case 'center-v': dy = (canvasH - h) / 2 - minY; break
        case 'bottom': dy = canvasH - maxY; break
      }
      el.x1 += dx; el.x2 += dx
      el.y1 += dy; el.y2 += dy
      break
    }
    case 'circle': {
      const el = element as { cx: number; cy: number; r: number }
      switch (direction) {
        case 'left': el.cx = el.r; break
        case 'center-h': el.cx = canvasW / 2; break
        case 'right': el.cx = canvasW - el.r; break
        case 'top': el.cy = el.r; break
        case 'center-v': el.cy = canvasH / 2; break
        case 'bottom': el.cy = canvasH - el.r; break
      }
      break
    }
  }

  const operations = recordElementPatchDiffs(previous, element, getPatchKeysForElement(element))
  if (operations.length > 0) {
    commitCurrentSlideFromLiveCanvasSvg()
    commitPatchOperations(operations, 'human', `对齐 ${element.id}`)
  }

  const slide = getCurrentSlide()
  renderCanvas(slide, currentSlideIndex)
  renderThumbnails()
  renderInspector()
  refreshCanvasOverlays()
}

function getFieldValue(element: SupportedEditableElement, key: string): string {
  if (element.type === 'text' && key === 'fontSize') {
    return String(parseFontSpec(element.font).fontSize)
  }

  if (element.type === 'text' && key === 'fontWeight') {
    return normalizeFontWeightValue(parseFontSpec(element.font).fontWeight)
  }

  const value = (element as unknown as Record<string, string | number | undefined>)[key]
  if ((key === 'fill' || key === 'stroke') && typeof value === 'string') {
    return formatColorDisplayValue(value)
  }
  return value === undefined ? '' : String(value)
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
      return ['x', 'y', 'width', 'maxHeight', 'font', 'fontSize', 'lineHeight']
    case 'rect':
    case 'image':
      return ['x', 'y', 'width', 'height']
    case 'line':
      return ['x1', 'y1', 'x2', 'y2']
    case 'circle':
      return ['cx', 'cy', 'r']
  }
}

function applyElementUpdate(element: SupportedEditableElement, key: string, rawValue: string): PropertyMutation[] | null {
  switch (element.type) {
    case 'text':
      switch (key) {
        case 'x': return assignRequiredNumber(element, 'x', rawValue)
        case 'y': return assignRequiredNumber(element, 'y', rawValue)
        case 'width': return assignRequiredNumber(element, 'width', rawValue)
        case 'text': return assignRequiredString(element, 'text', rawValue, { preserveWhitespace: true })
        case 'font': return assignTextFontRaw(element, rawValue)
        case 'fontSize': return assignTextFontVirtual(element, { fontSize: rawValue })
        case 'fontWeight': return assignTextFontVirtual(element, { fontWeight: rawValue })
        case 'lineHeight': return assignRequiredNumber(element, 'lineHeight', rawValue)
        case 'fill': return assignRequiredColor(element, 'fill', rawValue)
      }
      break
    case 'rect':
      switch (key) {
        case 'x': return assignRequiredNumber(element, 'x', rawValue)
        case 'y': return assignRequiredNumber(element, 'y', rawValue)
        case 'width': return assignRequiredNumber(element, 'width', rawValue)
        case 'height': return assignRequiredNumber(element, 'height', rawValue)
        case 'fill': return assignOptionalColor(element, 'fill', rawValue)
        case 'stroke': return assignOptionalColor(element, 'stroke', rawValue)
        case 'rx': return assignOptionalNumber(element, 'rx', rawValue)
      }
      break
    case 'path':
      switch (key) {
        case 'fill': return assignOptionalColor(element, 'fill', rawValue)
      }
      break
    case 'image':
      switch (key) {
        case 'x': return assignRequiredNumber(element, 'x', rawValue)
        case 'y': return assignRequiredNumber(element, 'y', rawValue)
        case 'width': return assignRequiredNumber(element, 'width', rawValue)
        case 'height': return assignRequiredNumber(element, 'height', rawValue)
        case 'href': return assignRequiredString(element, 'href', rawValue, { preserveWhitespace: true })
      }
      break
    case 'line':
      switch (key) {
        case 'x1': return assignRequiredNumber(element, 'x1', rawValue)
        case 'y1': return assignRequiredNumber(element, 'y1', rawValue)
        case 'x2': return assignRequiredNumber(element, 'x2', rawValue)
        case 'y2': return assignRequiredNumber(element, 'y2', rawValue)
        case 'stroke': return assignRequiredColor(element, 'stroke', rawValue)
      }
      break
    case 'circle':
      switch (key) {
        case 'cx': return assignRequiredNumber(element, 'cx', rawValue)
        case 'cy': return assignRequiredNumber(element, 'cy', rawValue)
        case 'r': return assignRequiredNumber(element, 'r', rawValue)
        case 'fill': return assignOptionalColor(element, 'fill', rawValue)
      }
      break
  }

  return null
}

function assignRequiredNumber<T extends object, K extends keyof T>(target: T, key: K, rawValue: string): PropertyMutation[] | null {
  const nextValue = Number(rawValue)
  if (!Number.isFinite(nextValue)) return null
  return toMutationList(createPropertyMutation(target, key, nextValue as T[K]))
}

function assignOptionalNumber<T extends object, K extends keyof T>(target: T, key: K, rawValue: string): PropertyMutation[] | null {
  if (rawValue.trim() === '') {
    return toMutationList(createPropertyMutation(target, key, undefined as T[K]))
  }

  const nextValue = Number(rawValue)
  if (!Number.isFinite(nextValue)) return null
  return toMutationList(createPropertyMutation(target, key, nextValue as T[K]))
}

function assignRequiredString<T extends object, K extends keyof T>(
  target: T,
  key: K,
  rawValue: string,
  options: { preserveWhitespace?: boolean } = {},
): PropertyMutation[] | null {
  const nextValue = options.preserveWhitespace ? rawValue : rawValue.trim()
  if (nextValue === '') return null
  return toMutationList(createPropertyMutation(target, key, nextValue as T[K]))
}

function assignRequiredColor<T extends object, K extends keyof T>(target: T, key: K, rawValue: string): PropertyMutation[] | null {
  const nextValue = normalizeColorFieldValue(rawValue)
  if (!nextValue) return null
  return toMutationList(createPropertyMutation(target, key, nextValue as T[K]))
}

function assignOptionalColor<T extends object, K extends keyof T>(target: T, key: K, rawValue: string): PropertyMutation[] | null {
  const nextValue = normalizeColorFieldValue(rawValue)
  return toMutationList(createPropertyMutation(target, key, (nextValue || undefined) as T[K]))
}

function assignTextFontRaw(element: TextElement, rawValue: string): PropertyMutation[] | null {
  const trimmed = rawValue.trim()
  if (!trimmed) return null

  const parsed = parseFontSpec(trimmed)
  return assignTextFontSpec(element, {
    fontString: trimmed,
    fontSize: parsed.fontSize,
    fontFamily: parsed.fontFamily,
    fontWeight: normalizeFontWeightValue(parsed.fontWeight),
    fontStyle: parsed.fontStyle,
  })
}

function assignTextFontVirtual(
  element: TextElement,
  next: { fontSize?: string; fontWeight?: string },
): PropertyMutation[] | null {
  const parsed = parseFontSpec(element.font)
  const nextFontSize = next.fontSize === undefined ? parsed.fontSize : Number(next.fontSize)
  if (!Number.isFinite(nextFontSize)) return null

  return assignTextFontSpec(element, {
    fontSize: nextFontSize,
    fontFamily: parsed.fontFamily,
    fontWeight: next.fontWeight === undefined ? normalizeFontWeightValue(parsed.fontWeight) : normalizeFontWeightValue(next.fontWeight),
    fontStyle: parsed.fontStyle,
  })
}

function cloneTransformableElement<T extends TransformableElement>(element: T): T {
  return JSON.parse(JSON.stringify(element)) as T
}

function captureTextNodeSnapshots(element: TransformableElement): TextNodeSnapshot[] {
  if (element.type !== 'text') return []

  return getCanvasElementNodes(element.id).flatMap(node => {
    const descendants = Array.from(node.querySelectorAll('tspan'))
    return [node, ...descendants].map(textNode => ({
      node: textNode,
      x: parseNodeNumberAttribute(textNode, 'x'),
      y: parseNodeNumberAttribute(textNode, 'y'),
    }))
  })
}

function parseNodeNumberAttribute(node: Element, attribute: string): number | null {
  const rawValue = node.getAttribute(attribute)
  if (rawValue === null) return null

  const parsed = Number(rawValue)
  return Number.isFinite(parsed) ? parsed : null
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

function getRenderableSlideSvgMarkup(slide: Slide, index: number, scope: 'canvas' | 'thumb'): string {
  const syncedRawSvg = getSyncedRawSvgString(index, slide)
  const svgMarkup = syncedRawSvg ?? slideToSvg(slide, state.canvas)
  return namespaceSvgIds(svgMarkup, `${scope}-${index}`)
}

function getSyncedRawSvgString(index: number, slide: Slide): string | null {
  const rawSvg = rawSvgStrings[index]?.trim()
  if (!rawSvg) return null

  const syncedRawSvg = syncRawSvgStringWithState(rawSvg, slide)
  if (!syncedRawSvg) return null

  rawSvgStrings[index] = syncedRawSvg
  return syncedRawSvg
}

function syncRawSvgStringWithState(rawSvg: string, slide: Slide): string | null {
  const doc = new DOMParser().parseFromString(rawSvg, 'image/svg+xml')
  const svg = doc.documentElement as unknown as SVGSVGElement
  if (svg.tagName.toLowerCase() !== 'svg') return null

  syncSvgRootCanvas(svg)
  pruneStaleRawSvgNodes(svg, slide)

  if (!syncSlideElementsIntoSvg(slide.elements, svg, doc)) {
    return slideToSvg(slide, state.canvas)
  }

  return svg.outerHTML
}

function syncSvgRootCanvas(svg: SVGSVGElement): void {
  svg.setAttribute('viewBox', `0 0 ${state.canvas.width} ${state.canvas.height}`)
  svg.setAttribute('width', String(state.canvas.width))
  svg.setAttribute('height', String(state.canvas.height))
}

function pruneStaleRawSvgNodes(svg: SVGSVGElement, slide: Slide): void {
  const validIds = collectSlideElementIds(slide.elements)
  Array.from(svg.querySelectorAll<SVGElement>('[data-element-id]')).forEach(node => {
    const elementId = node.getAttribute('data-element-id')
    if (!elementId || validIds.has(elementId)) return
    node.remove()
  })
}

function syncSlideElementsIntoSvg(
  elements: SlideElement[],
  svg: SVGSVGElement,
  doc: Document,
): boolean {
  return elements.every(element => syncSlideElementIntoSvg(element, svg, doc))
}

function syncSlideElementIntoSvg(
  element: SlideElement,
  svg: SVGSVGElement,
  doc: Document,
): boolean {
  const nodes = findRawSvgNodesByElementId(svg, element.id)
  if (nodes.length === 0) return false

  switch (element.type) {
    case 'text':
      return syncTextElementNode(nodes, element, doc)
    case 'rect':
      return syncSvgNodeCollection(nodes, 'rect', node => {
        setSvgAttribute(node, 'x', element.x)
        setSvgAttribute(node, 'y', element.y)
        setSvgAttribute(node, 'width', element.width)
        setSvgAttribute(node, 'height', element.height)
        setOptionalSvgAttribute(node, 'fill', element.fill)
        setOptionalSvgAttribute(node, 'stroke', element.stroke)
        setOptionalSvgAttribute(node, 'stroke-width', element.strokeWidth)
        setOptionalSvgAttribute(node, 'rx', element.rx)
        setOptionalSvgAttribute(node, 'ry', element.ry)
        setOptionalSvgAttribute(node, 'opacity', element.opacity, { skipValue: 1 })
        setOptionalSvgAttribute(node, 'fill-opacity', element.fillOpacity)
      })
    case 'path':
      return syncSvgNodeCollection(nodes, 'path', node => {
        setSvgAttribute(node, 'd', element.d)
        setOptionalSvgAttribute(node, 'fill', element.fill)
        setOptionalSvgAttribute(node, 'stroke', element.stroke)
        setOptionalSvgAttribute(node, 'stroke-width', element.strokeWidth)
        setOptionalSvgAttribute(node, 'fill-opacity', element.fillOpacity)
        setOptionalSvgAttribute(node, 'fill-rule', element.fillRule)
        setOptionalSvgAttribute(node, 'clip-rule', element.clipRule)
        setOptionalSvgAttribute(node, 'opacity', element.opacity, { skipValue: 1 })
      })
    case 'line':
      return syncSvgNodeCollection(nodes, 'line', node => {
        setSvgAttribute(node, 'x1', element.x1)
        setSvgAttribute(node, 'y1', element.y1)
        setSvgAttribute(node, 'x2', element.x2)
        setSvgAttribute(node, 'y2', element.y2)
        setSvgAttribute(node, 'stroke', element.stroke)
        setOptionalSvgAttribute(node, 'stroke-width', element.strokeWidth)
        setOptionalSvgAttribute(node, 'stroke-opacity', element.strokeOpacity)
        setOptionalSvgAttribute(node, 'opacity', element.opacity, { skipValue: 1 })
      })
    case 'circle':
      return syncSvgNodeCollection(nodes, 'circle', node => {
        setSvgAttribute(node, 'cx', element.cx)
        setSvgAttribute(node, 'cy', element.cy)
        setSvgAttribute(node, 'r', element.r)
        setOptionalSvgAttribute(node, 'fill', element.fill)
        setOptionalSvgAttribute(node, 'stroke', element.stroke)
        setOptionalSvgAttribute(node, 'stroke-width', element.strokeWidth)
        setOptionalSvgAttribute(node, 'fill-opacity', element.fillOpacity)
        setOptionalSvgAttribute(node, 'opacity', element.opacity, { skipValue: 1 })
      })
    case 'image':
      return syncSvgNodeCollection(nodes, 'image', node => {
        setSvgAttribute(node, 'x', element.x)
        setSvgAttribute(node, 'y', element.y)
        setSvgAttribute(node, 'width', element.width)
        setSvgAttribute(node, 'height', element.height)
        setSvgAttribute(node, 'href', element.href)
        setOptionalSvgAttribute(node, 'preserveAspectRatio', element.preserveAspectRatio)
        setOptionalSvgAttribute(node, 'opacity', element.opacity, { skipValue: 1 })
      })
    case 'group': {
      const hasGroup = syncSvgNodeCollection(nodes, 'g', node => {
        setOptionalSvgAttribute(node, 'transform', element.transform)
        setOptionalSvgAttribute(node, 'fill', element.fill)
        setOptionalSvgAttribute(node, 'font-family', element.fontFamily)
        setOptionalSvgAttribute(node, 'font-size', element.fontSize)
        setOptionalSvgAttribute(node, 'font-weight', element.fontWeight)
        setOptionalSvgAttribute(node, 'filter', element.filter)
        setOptionalSvgAttribute(node, 'opacity', element.opacity, { skipValue: 1 })
      })
      return hasGroup && syncSlideElementsIntoSvg(element.children, svg, doc)
    }
  }
}

function syncTextElementNode(
  nodes: SVGElement[],
  element: TextElement,
  doc: Document,
): boolean {
  return syncTextSvgNodes(nodes, element, doc)
}

function syncSvgNodeCollection(
  nodes: SVGElement[],
  tagName: string,
  apply: (node: SVGElement) => void,
): boolean {
  const matchingNodes = nodes.filter(node => node.tagName.toLowerCase() === tagName)
  if (matchingNodes.length === 0) return false
  matchingNodes.forEach(apply)
  return true
}

function findRawSvgNodesByElementId(svg: SVGSVGElement, elementId: string): SVGElement[] {
  return Array.from(svg.querySelectorAll<SVGElement>('[data-element-id]'))
    .filter(node => node.getAttribute('data-element-id') === elementId)
}

function collectSlideElementIds(elements: SlideElement[], ids = new Set<string>()): Set<string> {
  elements.forEach(element => {
    ids.add(element.id)
    if (element.type === 'group') collectSlideElementIds(element.children, ids)
  })
  return ids
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

function toSlideIdFromPath(path: string): string {
  const pathname = new URL(path, window.location.href).pathname
  const fileName = pathname.split('/').pop() || 'slide'
  return fileName.replace(/\.[^.]+$/, '').trim().replace(/\s+/g, '_')
}

function toSlideIdFromFileName(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, '').trim().replace(/\s+/g, '_')
}

function compareFileName(
  left: Pick<File, 'name'>,
  right: Pick<File, 'name'>,
): number {
  return left.name.localeCompare(right.name, undefined, {
    numeric: true,
    sensitivity: 'base',
  })
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

function formatColorDisplayValue(value: string): string {
  return normalizeHexColor(value) ?? value
}

function toColorInputValue(value: string): string {
  return (normalizeHexColor(value) ?? DEFAULT_COLOR_PICKER_VALUE).toLowerCase()
}

function normalizeColorFieldValue(value: string): string {
  return normalizeHexColor(value) ?? value.trim()
}

function syncInspectorControlValue(control: InspectorControl, field: InspectorField, value: string): void {
  if (field.input === 'color' && control instanceof HTMLInputElement) {
    if (control.dataset.propControl === 'color-picker') {
      const nextPickerValue = toColorInputValue(value)
      if (control.value !== nextPickerValue) control.value = nextPickerValue
      return
    }

    const nextTextValue = formatColorDisplayValue(value)
    if (control.value !== nextTextValue) control.value = nextTextValue
    return
  }

  if (field.input === 'range' && control instanceof HTMLInputElement) {
    if (control.value !== value) control.value = value
    return
  }

  if (field.input === 'select' && control instanceof HTMLSelectElement) {
    if (control.value !== value) control.value = value
    return
  }

  if (control.value !== value) control.value = value
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
            text: 'Design Editor',
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
            text: '默认直接打开和微调真实 SVG 页面，compat state 只用于桥接导入、handoff 和旧项目兼容。',
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
                text: 'editor/index.html · app.ts · SVG-first',
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
                text: '实时预览真实 SVG 页面；compat state 只在桥接时参与。',
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
                text: '先展示页面元数据，后续接选中与 patch。',
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
                text: '默认按 SVG 页面切换缩略图；compat state 只在导入旧资产或本地桥接时介入。',
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
            text: 'MVP 已把 SVG-first 预览层跑通，后续只需继续接交互层，而不是推倒重来。',
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
                text: '以 HTML 为入口，优先挂载真实 SVG 页面。',
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
                text: '多页上下文',
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
                text: '上一页、下一页、页码与缩略图围绕同一组页面上下文切换。',
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
                text: 'SVG 页面 → Design Editor → 导出',
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
