import type { SlideState } from './slide_state.js'
import { stateToSvgs } from './state_to_svg.js'
import { svgsToState } from './svg_to_state.js'

export interface DownloadArtifact {
  fileName: string
  mimeType: string
  content: string
}

export function getStatePathFromSearch(search: string): string | null {
  const value = new URLSearchParams(search).get('state')?.trim()
  return value ? value : null
}

export function getSvgPathsFromSearch(search: string): string[] {
  const value = new URLSearchParams(search).get('svg')?.trim()
  if (!value) return []

  return value
    .split(',')
    .map(path => path.trim())
    .filter(Boolean)
}

export function parseSlideStateJson(rawJson: string): SlideState {
  let parsed: unknown
  try {
    parsed = JSON.parse(rawJson)
  } catch (error) {
    throw new Error(`slide_state JSON 解析失败: ${(error as Error).message}`)
  }

  return ensureSlideState(parsed)
}

export function ensureSlideState(value: unknown): SlideState {
  if (!isSlideState(value)) {
    throw new Error('slide_state JSON 结构无效：需要包含 canvas.width/canvas.height 和非空 slides[]')
  }
  return value
}

export async function readSlideStateFile(file: File): Promise<SlideState> {
  return parseSlideStateJson(await file.text())
}

export async function readSvgFiles(files: File[], parser?: DOMParser): Promise<SlideState> {
  const svgFiles = files.filter(isSvgFile).sort(compareFileName)
  if (svgFiles.length === 0) {
    throw new Error('未检测到可导入的 SVG 文件')
  }

  const svgStrings = await Promise.all(svgFiles.map(file => file.text()))
  const slideIds = svgFiles.map(file => toSlideId(file.name))
  return svgsToState(svgStrings, slideIds, parser)
}

export function isJsonFile(file: Pick<File, 'name' | 'type'>): boolean {
  const normalizedType = file.type.toLowerCase()
  return file.name.toLowerCase().endsWith('.json')
    || normalizedType === 'application/json'
    || normalizedType === 'text/json'
}

export function isSvgFile(file: Pick<File, 'name' | 'type'>): boolean {
  const normalizedType = file.type.toLowerCase()
  return file.name.toLowerCase().endsWith('.svg')
    || normalizedType === 'image/svg+xml'
}

export function createJsonDownload(state: SlideState): DownloadArtifact {
  return {
    fileName: 'slide_state.json',
    mimeType: 'application/json;charset=utf-8',
    content: `${JSON.stringify(state, null, 2)}\n`,
  }
}

export function createSvgDownloads(state: SlideState): DownloadArtifact[] {
  const svgs = stateToSvgs(state)
  const digits = Math.max(2, String(svgs.length).length)

  return svgs.map((svg, index) => ({
    fileName: `slide_${String(index + 1).padStart(digits, '0')}.svg`,
    mimeType: 'image/svg+xml;charset=utf-8',
    content: svg,
  }))
}

function isSlideState(value: unknown): value is SlideState {
  if (!isRecord(value)) return false
  if (!isCanvas(value.canvas)) return false
  if (!Array.isArray(value.slides) || value.slides.length === 0) return false
  return value.slides.every(isSlide)
}

function isCanvas(value: unknown): boolean {
  if (!isRecord(value)) return false
  return isFiniteNumber(value.width) && isFiniteNumber(value.height)
}

function isSlide(value: unknown): boolean {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string') return false
  if (!Array.isArray(value.elements)) return false
  return value.elements.every(isElement)
}

function isElement(value: unknown): boolean {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string' || typeof value.type !== 'string') return false
  if (value.type !== 'group') return true
  return Array.isArray(value.children) && value.children.every(isElement)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function compareFileName(a: Pick<File, 'name'>, b: Pick<File, 'name'>): number {
  return a.name.localeCompare(b.name, undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function toSlideId(fileName: string): string {
  return fileName
    .replace(/\.[^.]+$/, '')
    .trim()
    .replace(/\s+/g, '_')
}
