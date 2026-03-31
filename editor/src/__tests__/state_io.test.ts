import { describe, expect, it } from 'vitest'
import { JSDOM } from 'jsdom'
import {
  createJsonDownload,
  createSvgDownloads,
  ensureSlideState,
  getStatePathFromSearch,
  getSvgPathsFromSearch,
  isJsonFile,
  isSvgFile,
  parseSlideStateJson,
  readSvgFiles,
} from '../state_io.js'
import type { SlideState } from '../slide_state.js'

describe('state_io', () => {
  const dom = new JSDOM()
  const parser = new dom.window.DOMParser()

  const state: SlideState = {
    canvas: { width: 1280, height: 720 },
    slides: [
      { id: 'slide_01', background: '#FFFFFF', elements: [] },
      { id: 'slide_02', background: '#111827', elements: [] },
    ],
  }

  it('从 URL query 中提取 state 路径', () => {
    expect(getStatePathFromSearch('?state=/fixtures/demo.json')).toBe('/fixtures/demo.json')
    expect(getStatePathFromSearch('?foo=bar')).toBeNull()
    expect(getStatePathFromSearch('?state=')).toBeNull()
  })

  it('从 URL query 中提取多个 SVG 路径', () => {
    expect(getSvgPathsFromSearch('?svg=/a/02.svg,/a/10.svg')).toEqual(['/a/02.svg', '/a/10.svg'])
    expect(getSvgPathsFromSearch('?state=/fixtures/demo.json')).toEqual([])
  })

  it('解析合法 slide_state JSON', () => {
    const parsed = parseSlideStateJson(JSON.stringify(state))
    expect(parsed.canvas.width).toBe(1280)
    expect(parsed.slides).toHaveLength(2)
  })

  it('拒绝缺少 slides 的非法 JSON 结构', () => {
    expect(() => ensureSlideState({ canvas: { width: 1280, height: 720 }, slides: [] })).toThrow(/结构无效/)
    expect(() => parseSlideStateJson('{"canvas":{"width":1280,"height":720}}')).toThrow(/结构无效/)
  })

  it('识别 JSON 文件', () => {
    expect(isJsonFile({ name: 'slide_state.json', type: 'application/json' })).toBe(true)
    expect(isJsonFile({ name: 'SLIDE_STATE.JSON', type: '' })).toBe(true)
    expect(isJsonFile({ name: 'cover.svg', type: 'image/svg+xml' })).toBe(false)
  })

  it('识别 SVG 文件', () => {
    expect(isSvgFile({ name: 'cover.svg', type: 'image/svg+xml' })).toBe(true)
    expect(isSvgFile({ name: 'COVER.SVG', type: '' })).toBe(true)
    expect(isSvgFile({ name: 'slide_state.json', type: 'application/json' })).toBe(false)
  })

  it('生成 slide_state.json 下载内容', () => {
    const download = createJsonDownload(state)
    expect(download.fileName).toBe('slide_state.json')
    expect(download.mimeType).toContain('application/json')
    expect(download.content).toContain('"slides"')
    expect(download.content.endsWith('\n')).toBe(true)
  })

  it('按 slide_01.svg 格式生成多个 SVG 下载文件', () => {
    const downloads = createSvgDownloads(state)
    expect(downloads).toHaveLength(2)
    expect(downloads[0].fileName).toBe('slide_01.svg')
    expect(downloads[1].fileName).toBe('slide_02.svg')
    expect(downloads[0].mimeType).toContain('image/svg+xml')
    expect(downloads[0].content).toContain('viewBox="0 0 1280 720"')
  })

  it('按文件名排序读取多个 SVG 并生成多页 SlideState', async () => {
    const svg1 = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450">
      <rect width="800" height="450" fill="#111827" />
    </svg>`
    const svg2 = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450">
      <rect width="800" height="450" fill="#FFFFFF" />
    </svg>`

    const files = [
      new dom.window.File([svg2], '10-summary.svg', { type: 'image/svg+xml' }),
      new dom.window.File([svg1], '02-cover.svg', { type: 'image/svg+xml' }),
    ] as unknown as File[]

    const nextState = await readSvgFiles(files, parser)
    expect(nextState.canvas).toEqual({ width: 800, height: 450 })
    expect(nextState.slides.map(slide => slide.id)).toEqual(['02-cover', '10-summary'])
    expect(nextState.slides).toHaveLength(2)
  })
})
