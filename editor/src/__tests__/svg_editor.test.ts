import { describe, expect, it } from 'vitest'
import { JSDOM } from 'jsdom'
import { syncCurrentSlideFromLiveSvg } from '../svg_editor.js'
import { svgsToState } from '../svg_to_state.js'

describe('svg_editor', () => {
  const dom = new JSDOM()
  const parser = new dom.window.DOMParser()

  it('以当前画布 SVG 为准回写当前页 raw SVG 与 compat state', () => {
    const rawSvg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
        <defs>
          <linearGradient id="grad">
            <stop offset="0%" stop-color="#111111" />
            <stop offset="100%" stop-color="#333333" />
          </linearGradient>
        </defs>
        <rect data-element-id="bg" width="200" height="100" fill="url(#grad)" />
        <text data-element-id="title" x="20" y="40" font-size="20" fill="#111111">Hello</text>
      </svg>
    `.trim()
    const otherSvg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
        <text data-element-id="summary" x="10" y="24" font-size="12" fill="#222222">Keep me</text>
      </svg>
    `.trim()
    const state = svgsToState([rawSvg, otherSvg], ['slide_01', 'slide_02'], parser, {
      preserveTextNodes: true,
    })
    const liveCanvasSvg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
        <defs>
          <linearGradient id="canvas-0-grad">
            <stop offset="0%" stop-color="#111111" />
            <stop offset="100%" stop-color="#333333" />
          </linearGradient>
        </defs>
        <rect id="canvas-0-bg" data-element-id="bg" width="200" height="100" fill="url(#canvas-0-grad)" />
        <text data-element-id="title" x="36" y="52" font-size="20" fill="#111111">Updated title</text>
      </svg>
    `.trim()

    const result = syncCurrentSlideFromLiveSvg({
      canvasSvgMarkup: liveCanvasSvg,
      currentSlideIndex: 0,
      state,
      rawSvgStrings: [rawSvg, otherSvg],
      parser,
    })

    expect(result).not.toBeNull()
    expect(result?.rawSvgStrings[0]).not.toContain('canvas-0-')
    expect(result?.rawSvgStrings[0]).toContain('id="grad"')
    expect(result?.rawSvgStrings[0]).toContain('fill="url(#grad)"')
    expect(result?.rawSvgStrings[0]).toContain('Updated title')
    expect(result?.rawSvgStrings[1]).toBe(otherSvg)

    const currentSlide = result?.state.slides[0]
    expect(currentSlide?.id).toBe('slide_01')
    expect(currentSlide?.elements.find(element => element.id === 'title')).toMatchObject({
      type: 'text',
      x: 36,
      y: 52,
      text: 'Updated title',
    })
    expect(result?.state.slides[1]).toEqual(state.slides[1])
  })

  it('当前页没有 raw SVG 时返回 null，让调用方走 compat fallback', () => {
    const state = {
      canvas: { width: 200, height: 100 },
      slides: [{ id: 'slide_01', elements: [] }],
    }

    expect(syncCurrentSlideFromLiveSvg({
      canvasSvgMarkup: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100" />',
      currentSlideIndex: 0,
      state,
      rawSvgStrings: [],
      parser,
    })).toBeNull()
  })
})
