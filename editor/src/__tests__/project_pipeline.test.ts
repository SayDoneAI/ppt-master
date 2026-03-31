import { readFileSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { JSDOM } from 'jsdom'
import {
  buildProjectStateFromSvgInputs,
  compareProjectFileName,
  createProjectSvgArtifacts,
} from '../project_pipeline.js'

const EXAMPLE_DIR = resolve(
  process.cwd(),
  '../examples/demo_project_intro_ppt169_20251211/svg_output',
)

function createParser(): DOMParser {
  const dom = new JSDOM()
  return new dom.window.DOMParser()
}

describe('project_pipeline', () => {
  it('按自然顺序排序项目 SVG 文件名', () => {
    const fileNames = ['slide_10_cta.svg', 'slide_02_pain_points.svg', 'slide_01_cover.svg']
    const sorted = [...fileNames].sort((a, b) => compareProjectFileName({ fileName: a }, { fileName: b }))
    expect(sorted).toEqual(['slide_01_cover.svg', 'slide_02_pain_points.svg', 'slide_10_cta.svg'])
  })

  it('从真实 svg_output 构建 slide_state，并保留页面文件名作为 slide id', () => {
    const fileNames = readdirSync(EXAMPLE_DIR)
      .filter(fileName => fileName.endsWith('.svg'))
      .sort((a, b) => compareProjectFileName({ fileName: a }, { fileName: b }))
    const inputs = fileNames.map(fileName => ({
      fileName,
      content: readFileSync(join(EXAMPLE_DIR, fileName), 'utf8'),
    }))

    const state = buildProjectStateFromSvgInputs(inputs, createParser())

    expect(state.slides).toHaveLength(fileNames.length)
    expect(state.slides[0]?.id).toBe('slide_01_cover')
    expect(state.slides[state.slides.length - 1]?.id).toBe('slide_10_cta')
    expect(state.canvas).toEqual({ width: 1280, height: 720 })
  })

  it('导出项目 SVG 时优先使用 slide id，且文件名稳定', () => {
    const fileNames = readdirSync(EXAMPLE_DIR)
      .filter(fileName => fileName.endsWith('.svg'))
      .sort((a, b) => compareProjectFileName({ fileName: a }, { fileName: b }))
    const inputs = fileNames.map(fileName => ({
      fileName,
      content: readFileSync(join(EXAMPLE_DIR, fileName), 'utf8'),
    }))

    const state = buildProjectStateFromSvgInputs(inputs, createParser())
    const artifacts = createProjectSvgArtifacts(state)

    expect(artifacts).toHaveLength(fileNames.length)
    expect(artifacts.map(artifact => artifact.fileName)).toEqual(fileNames)
    expect(artifacts[0]?.content).toContain('viewBox="0 0 1280 720"')
    expect(artifacts[0]?.content).toContain('data-element-id=')
  })

  it('重复或非法 slide id 会自动规整为可落盘文件名', () => {
    const artifacts = createProjectSvgArtifacts({
      canvas: { width: 1280, height: 720 },
      slides: [
        { id: '  ', elements: [] },
        { id: '结论页', elements: [] },
        { id: '结论页', elements: [] },
        { id: '路径/建议', elements: [] },
      ],
    })

    expect(artifacts.map(artifact => artifact.fileName)).toEqual([
      'slide_01.svg',
      '结论页.svg',
      '结论页_02.svg',
      '路径_建议.svg',
    ])
  })
})
