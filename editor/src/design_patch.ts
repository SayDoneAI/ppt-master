import type { DownloadArtifact } from './state_io.js'
import type {
  AiCommand,
  AiCommandScope,
  DesignPatch,
  Element,
  PatchOperation,
  Slide,
  SlideState,
} from './slide_state.js'

export interface UpdatePatchInput {
  slideIndex: number
  elementId: string
  property: string
  value: unknown
  oldValue: unknown
  source?: 'human' | 'ai'
  timestamp?: number
}

export interface CreateAiCommandInput {
  state: SlideState
  slideIndex: number
  instruction: string
  scope: AiCommandScope
  elementId?: string | null
}

export interface AiCommandPromptOptions {
  projectPathHint?: string | null
  stateFilePathHint?: string | null
  requestFileName?: string
}

export function createUpdatePatch(input: UpdatePatchInput): PatchOperation {
  return {
    op: 'update',
    path: `/slides/${input.slideIndex}/elements/${input.elementId}/${input.property}`,
    value: clonePatchValue(input.value),
    oldValue: clonePatchValue(input.oldValue),
    source: input.source ?? 'human',
    timestamp: input.timestamp ?? Date.now(),
  }
}

export function hasPatchValueChanged(oldValue: unknown, newValue: unknown): boolean {
  return serializePatchValue(oldValue) !== serializePatchValue(newValue)
}

export function createDesignPatch(
  operations: PatchOperation[],
  source: 'human' | 'ai' = 'human',
  timestamp = new Date().toISOString(),
): DesignPatch {
  return {
    timestamp,
    source,
    operations: operations.map(operation => clonePatchValue(operation)) as PatchOperation[],
  }
}

export function createDesignPatchDownload(operations: PatchOperation[]): DownloadArtifact {
  return {
    fileName: 'design_patch.json',
    mimeType: 'application/json;charset=utf-8',
    content: `${JSON.stringify(createDesignPatch(operations), null, 2)}\n`,
  }
}

export function createAiCommand(input: CreateAiCommandInput): AiCommand {
  const slide = input.state.slides[input.slideIndex]
  if (!slide) {
    throw new Error(`AI 指令生成失败：slideIndex ${input.slideIndex} 超出范围`)
  }

  const trimmedInstruction = input.instruction.trim()
  if (!trimmedInstruction) {
    throw new Error('AI 指令生成失败：instruction 不能为空')
  }

  const targetElement = input.scope === 'selected-element'
    ? findElementById(slide.elements, input.elementId ?? null)
    : null

  if (input.scope === 'selected-element' && !targetElement) {
    throw new Error('AI 指令生成失败：当前没有可用的选中元素')
  }

  return {
    scope: input.scope,
    slideIndex: input.slideIndex,
    slideId: slide.id,
    elementId: targetElement?.id ?? null,
    elementSnapshot: targetElement ? clonePatchValue(targetElement) : null,
    slideSnapshot: clonePatchValue(slide),
    instruction: trimmedInstruction,
  }
}

export function createAiCommandPatch(
  aiCommand: AiCommand,
  operations: PatchOperation[] = [],
): DesignPatch {
  return {
    timestamp: new Date().toISOString(),
    source: 'human',
    operations: operations.map(operation => clonePatchValue(operation)) as PatchOperation[],
    aiCommand: clonePatchValue(aiCommand),
  }
}

export function createAiCommandDownload(
  aiCommand: AiCommand,
  operations: PatchOperation[] = [],
): DownloadArtifact {
  return {
    fileName: 'design_patch.ai-request.json',
    mimeType: 'application/json;charset=utf-8',
    content: `${JSON.stringify(createAiCommandPatch(aiCommand, operations), null, 2)}\n`,
  }
}

export function createAiCommandPromptDownload(
  aiCommand: AiCommand,
  operations: PatchOperation[] = [],
  options: AiCommandPromptOptions = {},
): DownloadArtifact {
  const projectPath = options.projectPathHint?.trim() || '<project_path>'
  const statePath = options.stateFilePathHint?.trim() || `${projectPath}/slide_state.json`
  const requestFileName = options.requestFileName?.trim() || 'design_patch.ai-request.json'
  const requestPath = projectPath === '<project_path>'
    ? requestFileName
    : `${projectPath}/${requestFileName}`
  const scopeLabel = aiCommand.scope === 'selected-element' ? '当前选中元素' : '当前页面'
  const elementLabel = aiCommand.elementId ?? '页面级'
  const recentOpsLabel = operations.length > 0 ? `${operations.length} 条` : '0 条'

  return {
    fileName: 'design_patch.ai-handoff.md',
    mimeType: 'text/markdown;charset=utf-8',
    content: `# PPT Master Local AI Handoff

## 请求摘要
- 作用范围: ${scopeLabel}
- Slide: \`${aiCommand.slideId}\`
- Element: \`${elementLabel}\`
- 最近人工 patch: ${recentOpsLabel}
- 指令: ${aiCommand.instruction}

## 约定文件
- 状态文件: \`${statePath}\`
- 机器可读 handoff JSON: \`${requestPath}\`
- 执行说明: \`design_patch.ai-handoff.md\`（当前文件）

## 给 Claude Code / Codex 的本地执行要求
1. 在本地仓库中读取 \`${requestPath}\`，提取 \`aiCommand\` 与已有 \`operations\`。
2. 以 \`${statePath}\` 为唯一真相源修改内容，不要直接改导出的 SVG。
3. 优先围绕 \`${elementLabel}\` 与当前页上下文完成指令；如 scope 为页面级，可重排当前页元素。
4. 修改完成后运行:
   - \`python3 tools/slide_state_bridge.py render ${projectPath}\`
   - \`python3 tools/project_manager.py validate ${projectPath}\`
5. 若用户要求交付 PPT，再继续:
   - \`python3 tools/finalize_svg.py ${projectPath}\`
   - \`python3 tools/svg_to_pptx.py ${projectPath} -s final\`

## 说明
- 这是给 Claude Code / Codex 本地 skill / command 使用的 handoff，不是浏览器直连或服务端 API 协议。
- 浏览器编辑器只负责导出 / 导入文件；真正的项目修改、render、validate 都在本地仓库里完成。
- 如果当前文件还没放入项目根目录，请先把 \`${requestFileName}\` 移动到目标项目目录。
- 如果当前会话只打算返回 patch JSON 而不直接改项目，也可输出同 schema 的 \`design_patch.json\` 供编辑器回流应用。
`,
  }
}

export function parseDesignPatchJson(rawJson: string): DesignPatch {
  let parsed: unknown
  try {
    parsed = JSON.parse(rawJson)
  } catch (error) {
    throw new Error(`design_patch JSON 解析失败: ${(error as Error).message}`)
  }

  return ensureDesignPatch(parsed)
}

export function ensureDesignPatch(value: unknown): DesignPatch {
  if (!isDesignPatch(value)) {
    throw new Error('design_patch JSON 结构无效：需要包含 timestamp/source/operations[]')
  }
  return value
}

export function applyDesignPatch(state: SlideState, designPatch: DesignPatch): SlideState {
  const nextState = clonePatchValue(state)

  for (const operation of designPatch.operations) {
    applyPatchOperation(nextState, operation)
  }

  return nextState
}

export function getDesignPatchPrimarySlideIndex(designPatch: DesignPatch): number | null {
  if (designPatch.aiCommand) return designPatch.aiCommand.slideIndex

  const firstPath = designPatch.operations[0]?.path
  if (!firstPath) return null

  const match = firstPath.match(/^\/slides\/(\d+)(?:\/|$)/)
  return match ? Number(match[1]) : null
}

function serializePatchValue(value: unknown): string {
  return JSON.stringify(clonePatchValue(value))
}

function clonePatchValue<T>(value: T): T {
  if (value === undefined) return value
  return JSON.parse(JSON.stringify(value)) as T
}

function applyPatchOperation(state: SlideState, operation: PatchOperation): void {
  switch (operation.op) {
    case 'update':
      applyUpdateOperation(state, operation)
      return
    case 'add':
      applyAddOperation(state, operation)
      return
    case 'delete':
      applyDeleteOperation(state, operation)
      return
    case 'reorder':
      applyReorderOperation(state, operation)
  }
}

function applyUpdateOperation(state: SlideState, operation: Extract<PatchOperation, { op: 'update' }>): void {
  const target = resolveUpdateTarget(state, operation.path)
  const targetRecord = target.element as unknown as Record<string, unknown>
  targetRecord[target.property] = clonePatchValue(operation.value)
}

function applyAddOperation(state: SlideState, operation: Extract<PatchOperation, { op: 'add' }>): void {
  const slideAddMatch = operation.path.match(/^\/slides\/(\d+)$/)
  if (slideAddMatch) {
    if (!isSlide(operation.value)) {
      throw new Error(`add patch 目标 slide 数据无效: ${operation.path}`)
    }

    const slideIndex = clampIndex(Number(slideAddMatch[1]), state.slides.length)
    state.slides.splice(slideIndex, 0, clonePatchValue(operation.value))
    return
  }

  const slide = getSlideByPath(state, operation.path)
  const rootAddMatch = operation.path.match(/^\/slides\/(\d+)\/elements$/)
  if (rootAddMatch) {
    if (!isElement(operation.value)) {
      throw new Error(`add patch 目标元素数据无效: ${operation.path}`)
    }
    slide.elements.push(clonePatchValue(operation.value))
    return
  }

  const childAddMatch = operation.path.match(/^\/slides\/(\d+)\/elements\/([^/]+)\/children$/)
  if (!childAddMatch) {
    throw new Error(`不支持的 add patch 路径: ${operation.path}`)
  }

  const parentGroup = findElementById(slide.elements, childAddMatch[2])
  if (!parentGroup || parentGroup.type !== 'group') {
    throw new Error(`add patch 目标 group 不存在: ${childAddMatch[2]}`)
  }

  if (!isElement(operation.value)) {
    throw new Error(`add patch 目标元素数据无效: ${operation.path}`)
  }

  parentGroup.children.push(clonePatchValue(operation.value))
}

function applyDeleteOperation(state: SlideState, operation: Extract<PatchOperation, { op: 'delete' }>): void {
  const slideDeleteMatch = operation.path.match(/^\/slides\/(\d+)$/)
  if (slideDeleteMatch) {
    const slideIndex = Number(slideDeleteMatch[1])
    if (!state.slides[slideIndex]) {
      throw new Error(`delete patch 目标 slide 不存在: ${operation.path}`)
    }

    state.slides.splice(slideIndex, 1)
    return
  }

  const target = resolveElementContainerByPath(state, operation.path)
  target.container.splice(target.index, 1)
}

function applyReorderOperation(state: SlideState, operation: Extract<PatchOperation, { op: 'reorder' }>): void {
  const slideReorderMatch = operation.path.match(/^\/slides\/(\d+)$/)
  if (slideReorderMatch) {
    const slideIndex = Number(slideReorderMatch[1])
    const [slide] = state.slides.splice(slideIndex, 1)
    if (!slide) {
      throw new Error(`reorder patch 目标 slide 不存在: ${operation.path}`)
    }

    const nextIndex = clampIndex(Number(operation.value), state.slides.length)
    state.slides.splice(nextIndex, 0, slide)
    return
  }

  const target = resolveElementContainerByPath(state, operation.path)
  const [element] = target.container.splice(target.index, 1)
  const nextIndex = clampIndex(Number(operation.value), target.container.length)
  target.container.splice(nextIndex, 0, element)
}

function resolveUpdateTarget(
  state: SlideState,
  path: string,
): { element: Element; property: string } {
  const match = path.match(/^\/slides\/(\d+)\/elements\/([^/]+)\/([^/]+)$/)
  if (!match) {
    throw new Error(`不支持的 update patch 路径: ${path}`)
  }

  const slide = getSlideByIndex(state, Number(match[1]))
  const element = findElementById(slide.elements, match[2])
  if (!element) {
    throw new Error(`update patch 目标元素不存在: ${match[2]}`)
  }

  return { element, property: match[3] }
}

function resolveElementContainerByPath(
  state: SlideState,
  path: string,
): { container: Element[]; index: number; element: Element } {
  const match = path.match(/^\/slides\/(\d+)\/elements\/([^/]+)$/)
  if (!match) {
    throw new Error(`不支持的 patch 路径: ${path}`)
  }

  const slide = getSlideByIndex(state, Number(match[1]))
  const target = findElementContainer(slide.elements, match[2])
  if (!target) {
    throw new Error(`patch 目标元素不存在: ${match[2]}`)
  }

  return target
}

function getSlideByPath(state: SlideState, path: string): Slide {
  const match = path.match(/^\/slides\/(\d+)\//)
  if (!match) {
    throw new Error(`patch 路径缺少 slide 索引: ${path}`)
  }

  return getSlideByIndex(state, Number(match[1]))
}

function getSlideByIndex(state: SlideState, slideIndex: number): Slide {
  const slide = state.slides[slideIndex]
  if (!slide) {
    throw new Error(`patch slide 索引超出范围: ${slideIndex}`)
  }
  return slide
}

function findElementContainer(
  elements: Element[],
  elementId: string,
): { container: Element[]; index: number; element: Element } | null {
  for (let index = 0; index < elements.length; index += 1) {
    const element = elements[index]
    if (element.id === elementId) {
      return { container: elements, index, element }
    }

    if (element.type !== 'group') continue

    const nested = findElementContainer(element.children, elementId)
    if (nested) return nested
  }

  return null
}

function findElementById(elements: Element[], elementId: string | null): Element | null {
  if (!elementId) return null

  for (const element of elements) {
    if (element.id === elementId) return element
    if (element.type !== 'group') continue

    const nested = findElementById(element.children, elementId)
    if (nested) return nested
  }

  return null
}

function isDesignPatch(value: unknown): value is DesignPatch {
  if (!isRecord(value)) return false
  if (typeof value.timestamp !== 'string') return false
  if (value.source !== 'human' && value.source !== 'ai') return false
  if (!Array.isArray(value.operations) || !value.operations.every(isPatchOperation)) return false
  if (value.aiCommand !== undefined && !isAiCommand(value.aiCommand)) return false
  return true
}

function isPatchOperation(value: unknown): value is PatchOperation {
  if (!isRecord(value)) return false
  if (typeof value.path !== 'string') return false
  if (value.source !== 'human' && value.source !== 'ai') return false
  if (typeof value.timestamp !== 'number' || !Number.isFinite(value.timestamp)) return false

  switch (value.op) {
    case 'update':
      return 'value' in value && 'oldValue' in value
    case 'add':
      return isElement(value.value) || isSlide(value.value)
    case 'delete':
      return isElement(value.oldValue) || isSlide(value.oldValue)
    case 'reorder':
      return typeof value.value === 'number' && typeof value.oldValue === 'number'
    default:
      return false
  }
}

function isAiCommand(value: unknown): value is AiCommand {
  if (!isRecord(value)) return false
  if (value.scope !== 'selected-element' && value.scope !== 'current-slide') return false
  if (typeof value.slideIndex !== 'number' || !Number.isFinite(value.slideIndex)) return false
  if (typeof value.slideId !== 'string') return false
  if (value.elementId !== null && typeof value.elementId !== 'string') return false
  if (!isSlide(value.slideSnapshot)) return false
  if (value.elementSnapshot !== null && !isElement(value.elementSnapshot)) return false
  if (typeof value.instruction !== 'string') return false
  return true
}

function isSlide(value: unknown): value is Slide {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string') return false
  if (!Array.isArray(value.elements)) return false
  return value.elements.every(isElement)
}

function isElement(value: unknown): value is Element {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string' || typeof value.type !== 'string') return false
  if (value.type !== 'group') return true
  return Array.isArray(value.children) && value.children.every(isElement)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function clampIndex(value: number, max: number): number {
  if (!Number.isFinite(value)) return max
  return Math.min(Math.max(0, value), max)
}
