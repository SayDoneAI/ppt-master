// slide_state.ts — SVG-first 编辑器的兼容状态类型定义
// 该结构用于 import / handoff / legacy bridge，不替代项目主产物 SVG page set

// ============================================================
// Canvas & Slide
// ============================================================

export interface SlideState {
  /** 兼容状态里的画布尺寸，对应 SVG viewBox */
  canvas: Canvas
  /** 兼容状态里的页面数组 */
  slides: Slide[]
}

export interface Canvas {
  width: number
  height: number
}

export interface Slide {
  id: string
  /** 页面内的元素，按 z-order 排列（后面的元素在上层） */
  elements: Element[]
  /** 可选背景色 */
  background?: string
  /** SVG defs 定义（渐变、滤镜等） */
  defs?: Def[]
}

// ============================================================
// Defs（渐变、滤镜）
// ============================================================

export type Def = LinearGradientDef | FilterDef

export interface LinearGradientDef {
  type: 'linearGradient'
  id: string
  x1: string
  y1: string
  x2: string
  y2: string
  stops: GradientStop[]
}

export interface GradientStop {
  offset: string
  color: string
  opacity?: number
}

export interface FilterDef {
  type: 'filter'
  id: string
  /** 原始 SVG filter 内容（暂不解构，保留兼容性） */
  rawSvg: string
}

// ============================================================
// Elements
// ============================================================

export type Element =
  | TextElement
  | RectElement
  | PathElement
  | LineElement
  | CircleElement
  | ImageElement
  | GroupElement

/** 所有元素的公共字段 */
interface BaseElement {
  id: string
  /** 元素类型 */
  type: string
  /** 可选透明度 */
  opacity?: number
  /** 可选 transform（仅用于 group） */
  transform?: string
}

// ============================================================
// Text — Pretext 驱动的核心元素
// ============================================================

export interface TextElement extends BaseElement {
  type: 'text'
  /** 文本框左上角 x */
  x: number
  /** 文本框第一行基线 y（SVG 文本 y 是基线） */
  y: number
  /** 文本框宽度（Pretext 断行约束） */
  width: number
  /** 最大高度约束（溢出检测用），省略则不限 */
  maxHeight?: number
  /** 原始文本内容（未断行的完整文本） */
  text: string
  /**
   * CSS font shorthand，传给 Pretext prepare()
   * 示例: "bold 36px PingFang SC", "16px Arial"
   * 注意: 避免 system-ui（macOS 精度不安全）
   */
  font: string
  /** 行高（像素），传给 Pretext layout() */
  lineHeight: number
  /** 填充色 */
  fill: string
  /** font-family（导出 SVG 时用） */
  fontFamily?: string
  /** font-size（导出时用，也从 font shorthand 解析） */
  fontSize?: number
  /** font-weight */
  fontWeight?: string | number
  /** 文本锚点 */
  textAnchor?: 'start' | 'middle' | 'end'
  /** 字间距 */
  letterSpacing?: number
  /** fill-opacity */
  fillOpacity?: number
}

// ============================================================
// Rect — 矩形（含圆角）
// ============================================================

export interface RectElement extends BaseElement {
  type: 'rect'
  x: number
  y: number
  width: number
  height: number
  fill?: string
  stroke?: string
  strokeWidth?: number
  rx?: number
  ry?: number
  fillOpacity?: number
}

// ============================================================
// Path — 通用路径（圆角矩形、自定义形状）
// ============================================================

export interface PathElement extends BaseElement {
  type: 'path'
  /** SVG path d 属性 */
  d: string
  fill?: string
  stroke?: string
  strokeWidth?: number
  fillOpacity?: number
  fillRule?: 'nonzero' | 'evenodd'
  clipRule?: string
}

// ============================================================
// Line
// ============================================================

export interface LineElement extends BaseElement {
  type: 'line'
  x1: number
  y1: number
  x2: number
  y2: number
  stroke: string
  strokeWidth?: number
  strokeOpacity?: number
}

// ============================================================
// Circle
// ============================================================

export interface CircleElement extends BaseElement {
  type: 'circle'
  cx: number
  cy: number
  r: number
  fill?: string
  stroke?: string
  strokeWidth?: number
  fillOpacity?: number
}

// ============================================================
// Image
// ============================================================

export interface ImageElement extends BaseElement {
  type: 'image'
  x: number
  y: number
  width: number
  height: number
  /** 图片路径（相对路径或 data URI） */
  href: string
  /** preserveAspectRatio */
  preserveAspectRatio?: string
}

// ============================================================
// Group — 分组（含 transform）
// ============================================================

export interface GroupElement extends BaseElement {
  type: 'group'
  /** 子元素 */
  children: Element[]
  /** SVG <g> 上的公共属性 */
  fill?: string
  fontFamily?: string
  fontSize?: number
  fontWeight?: string | number
  /** filter 引用 */
  filter?: string
}

// ============================================================
// Design Patch — AI ↔ 人交互的修改记录
// ============================================================

export type AiCommandScope = 'selected-element' | 'current-slide'

export interface AiCommand {
  /** 指令目标范围：围绕当前选中元素或当前页面 */
  scope: AiCommandScope
  /** 发送指令的页面索引 */
  slideIndex: number
  /** 发送指令的页面 ID */
  slideId: string
  /** 选中元素的 ID（null 表示页面级指令） */
  elementId: string | null
  /** 选中元素的当前快照（供 AI 读取上下文） */
  elementSnapshot: Element | null
  /** 当前页面的完整快照（供 AI 直接产出 patch） */
  slideSnapshot: Slide
  /** 用户输入的自然语言指令 */
  instruction: string
}

export interface DesignPatch {
  /** patch 时间戳 */
  timestamp: string
  /** patch 来源 */
  source: 'human' | 'ai'
  /** 操作列表 */
  operations: PatchOperation[]
  /** 可选：来自编辑器的 AI 指令请求 */
  aiCommand?: AiCommand
}

export type PatchOperation =
  | UpdatePatchOperation
  | AddPatchOperation
  | DeletePatchOperation
  | ReorderPatchOperation

interface BasePatchOperation {
  /** JSON Pointer 风格路径 */
  path: string
  /** patch 来源 */
  source: 'human' | 'ai'
  /** 发生时间（毫秒时间戳） */
  timestamp: number
}

export interface UpdatePatchOperation extends BasePatchOperation {
  op: 'update'
  value: unknown
  oldValue: unknown
}

export interface AddPatchOperation extends BasePatchOperation {
  op: 'add'
  value: Element | Slide
}

export interface DeletePatchOperation extends BasePatchOperation {
  op: 'delete'
  oldValue: Element | Slide
}

export interface ReorderPatchOperation extends BasePatchOperation {
  op: 'reorder'
  value: number
  oldValue: number
}
