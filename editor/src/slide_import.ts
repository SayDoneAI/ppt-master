import type { AddPatchOperation, Canvas, Slide, SlideState } from './slide_state.js'

export interface AppendSlidesPatchInput {
  state: SlideState
  importedSlides: Slide[]
  insertIndex?: number
  source?: 'human' | 'ai'
  timestampStart?: number
}

export function ensureCompatibleCanvas(base: Canvas, imported: Canvas): void {
  if (base.width === imported.width && base.height === imported.height) return

  throw new Error(
    `导入页面尺寸不匹配：当前为 ${base.width}×${base.height}，导入内容为 ${imported.width}×${imported.height}`,
  )
}

export function createAppendSlidesPatch(input: AppendSlidesPatchInput): AddPatchOperation[] {
  const slides = createUniqueImportedSlides(input.state.slides, input.importedSlides)
  if (slides.length === 0) return []

  const insertIndex = clamp(input.insertIndex ?? input.state.slides.length, 0, input.state.slides.length)
  const timestampStart = input.timestampStart ?? Date.now()

  return slides.map((slide, offset) => ({
    op: 'add',
    path: `/slides/${insertIndex + offset}`,
    value: slide,
    source: input.source ?? 'human',
    timestamp: timestampStart + offset,
  }))
}

export function createUniqueImportedSlides(
  existingSlides: readonly Slide[],
  importedSlides: readonly Slide[],
): Slide[] {
  const usedIds = new Set(existingSlides.map(slide => slide.id))

  return importedSlides.map((slide, index) => {
    const cloned = cloneValue(slide)
    const fallbackId = `imported_slide_${String(index + 1).padStart(2, '0')}`
    const baseId = sanitizeSlideId(cloned.id) || fallbackId
    cloned.id = createUniqueSlideId(baseId, usedIds)
    return cloned
  })
}

function sanitizeSlideId(value: string): string {
  return value
    .trim()
    .replace(/\s+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function createUniqueSlideId(baseId: string, usedIds: Set<string>): string {
  if (!usedIds.has(baseId)) {
    usedIds.add(baseId)
    return baseId
  }

  let suffix = 2
  while (true) {
    const candidate = `${baseId}_${String(suffix).padStart(2, '0')}`
    if (!usedIds.has(candidate)) {
      usedIds.add(candidate)
      return candidate
    }
    suffix += 1
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function cloneValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}
