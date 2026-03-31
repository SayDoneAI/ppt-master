import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, it, expect } from 'vitest'
import { JSDOM } from 'jsdom'
import { layoutText, slideToSvg } from '../state_to_svg.js'
import type { Element as SlideElement, TextElement } from '../slide_state.js'
import { svgToSlide } from '../svg_to_state.js'

const dom = new JSDOM()
const parser = new dom.window.DOMParser()
const REAL_SVG_PATH = resolve(
  process.cwd(),
  '../examples/demo_project_intro_ppt169_20251211/svg_final/slide_01_cover.svg',
)

type ElementCounts = Record<'rect' | 'text' | 'line' | 'circle' | 'path' | 'image' | 'group', number>
type SvgTagCounts = ElementCounts & { linearGradient: number }

function countElements(elements: SlideElement[]): ElementCounts {
  const counts: ElementCounts = {
    rect: 0,
    text: 0,
    line: 0,
    circle: 0,
    path: 0,
    image: 0,
    group: 0,
  }

  const walk = (items: SlideElement[]) => {
    for (const el of items) {
      counts[el.type as keyof ElementCounts] += 1
      if (el.type === 'group') {
        walk(el.children)
      }
    }
  }

  walk(elements)
  return counts
}

function countSvgTags(svg: string): SvgTagCounts {
  const doc = parser.parseFromString(svg, 'image/svg+xml')
  return {
    rect: doc.querySelectorAll('rect').length,
    text: doc.querySelectorAll('text').length,
    line: doc.querySelectorAll('line').length,
    circle: doc.querySelectorAll('circle').length,
    path: doc.querySelectorAll('path').length,
    image: doc.querySelectorAll('image').length,
    group: doc.querySelectorAll('g').length,
    linearGradient: doc.querySelectorAll('linearGradient').length,
  }
}

function getCanvas(svg: string): { width: number; height: number } {
  const doc = parser.parseFromString(svg, 'image/svg+xml')
  const viewBox = doc.documentElement.getAttribute('viewBox')
  expect(viewBox).toBeTruthy()

  const [, , width, height] = viewBox!.split(/[\s,]+/).map(Number)
  return { width, height }
}

describe('SVG ↔ slide_state roundtrip', () => {
  it('真实 SVG 往返后保留关键元素类型、数量与 linearGradient defs', () => {
    const originalSvg = readFileSync(REAL_SVG_PATH, 'utf8')
    const canvas = getCanvas(originalSvg)
    const originalTagCounts = countSvgTags(originalSvg)

    const slide = svgToSlide(originalSvg, 'slide_01_cover', parser)
    const parsedCounts = countElements(slide.elements)
    const roundtripSvg = slideToSvg(slide, canvas)
    const reparsedSlide = svgToSlide(roundtripSvg, 'slide_01_cover', parser)

    expect(countSvgTags(roundtripSvg)).toEqual(originalTagCounts)
    expect(countElements(reparsedSlide.elements)).toEqual(parsedCounts)

    const originalGradient = slide.defs?.find(def => def.type === 'linearGradient')
    const roundtripGradient = reparsedSlide.defs?.find(def => def.type === 'linearGradient')

    expect(originalGradient?.type).toBe('linearGradient')
    expect(roundtripGradient?.type).toBe('linearGradient')

    if (originalGradient?.type === 'linearGradient' && roundtripGradient?.type === 'linearGradient') {
      expect(roundtripGradient).toMatchObject({
        id: originalGradient.id,
        x1: originalGradient.x1,
        y1: originalGradient.y1,
        x2: originalGradient.x2,
        y2: originalGradient.y2,
      })
      expect(roundtripGradient.stops).toEqual(originalGradient.stops)
    }
  })
})

describe('文本溢出检测', () => {
  it('当 maxHeight 小于实际文本高度时标记 overflow=true', () => {
    const textEl: TextElement = {
      type: 'text',
      id: 'overflow_text',
      x: 80,
      y: 120,
      width: 120,
      maxHeight: 24,
      text: '这是一段需要被拆成多行的文本，用来验证溢出检测逻辑是否正确。',
      font: '16px Arial',
      lineHeight: 24,
      fill: '#111111',
    }

    const result = layoutText(textEl)

    expect(result.lineCount).toBeGreaterThan(1)
    expect(result.height).toBeGreaterThan(textEl.maxHeight!)
    expect(result.overflow).toBe(true)
  })
})
