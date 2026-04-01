import { JSDOM } from 'jsdom'
import { describe, expect, it } from 'vitest'
import type { TextElement } from '../slide_state.js'
import {
  ALL_RESIZE_HANDLES,
  TEXT_RESIZE_HANDLES,
  applyTextResizeSemantics,
  getResizeHandlesForElementType,
  syncTextSvgNodes,
} from '../text_resize.js'

function createTextElement(overrides: Partial<TextElement> = {}): TextElement {
  return {
    id: 'title',
    type: 'text',
    x: 100,
    y: 200,
    width: 300,
    text: '这是一段很长的文本需要实时重新排版',
    font: '700 40px Inter',
    fontSize: 40,
    lineHeight: 48,
    fill: '#111827',
    ...overrides,
  }
}

describe('text_resize', () => {
  it('文本元素只暴露左右和四角手柄，其他元素保持 8 个手柄', () => {
    expect(TEXT_RESIZE_HANDLES).toEqual(['nw', 'ne', 'e', 'se', 'sw', 'w'])
    expect(TEXT_RESIZE_HANDLES).not.toContain('n')
    expect(TEXT_RESIZE_HANDLES).not.toContain('s')
    expect(getResizeHandlesForElementType('text')).toEqual(TEXT_RESIZE_HANDLES)
    expect(getResizeHandlesForElementType('rect')).toEqual(ALL_RESIZE_HANDLES)
  })

  it('文本左右手柄只改文本框宽度和锚点，不改字号与行高', () => {
    const initial = createTextElement()
    const element = createTextElement()

    applyTextResizeSemantics({
      element,
      initialElement: initial,
      initialBounds: { x: 100, y: 160, width: 300, height: 120 },
      nextBounds: { x: 70, y: 160, width: 330, height: 120 },
      handle: 'w',
      minWidth: 40,
      minHeight: 48,
    })

    expect(element.x).toBe(70)
    expect(element.y).toBe(200)
    expect(element.width).toBe(330)
    expect(element.font).toBe(initial.font)
    expect(element.fontSize).toBe(40)
    expect(element.lineHeight).toBe(48)
    expect(element.maxHeight).toBeUndefined()
  })

  it('文本角手柄会同步缩放字号、行高和文本框高度', () => {
    const initial = createTextElement()
    const element = createTextElement()

    applyTextResizeSemantics({
      element,
      initialElement: initial,
      initialBounds: { x: 100, y: 160, width: 300, height: 120 },
      nextBounds: { x: 70, y: 140, width: 360, height: 180 },
      handle: 'nw',
      minWidth: 40,
      minHeight: 48,
    })

    expect(element.x).toBe(70)
    expect(element.y).toBe(180)
    expect(element.width).toBe(360)
    expect(element.fontSize).toBeCloseTo(53.7, 5)
    expect(element.lineHeight).toBeCloseTo(64.4, 5)
    expect(element.font).toContain('53.7px')
    expect(element.maxHeight).toBe(180)
  })

  it('live preview 会按最新 width 重建文本断行结构', () => {
    const dom = new JSDOM(
      '<svg xmlns="http://www.w3.org/2000/svg"><text data-element-id="title">旧内容</text><text data-element-id="title">旧的第二行</text></svg>',
      { contentType: 'image/svg+xml' },
    )
    const doc = dom.window.document
    const element = createTextElement({ width: 96 })
    const nodes = Array.from(doc.querySelectorAll<SVGElement>('[data-element-id="title"]'))

    const synced = syncTextSvgNodes(nodes, element, doc)
    const textNodes = doc.querySelectorAll('text[data-element-id="title"]')
    const tspans = textNodes[0]?.querySelectorAll('tspan') ?? []

    expect(synced).toBe(true)
    expect(textNodes).toHaveLength(1)
    expect(tspans.length).toBeGreaterThan(1)
    expect(Array.from(tspans).map(node => node.textContent).join('')).toBe(element.text)
    expect(Array.from(tspans).map(node => node.getAttribute('x'))).toEqual(
      Array.from({ length: tspans.length }, () => String(element.x)),
    )
    expect(Array.from(tspans).map(node => Number(node.getAttribute('y')))).toEqual(
      Array.from({ length: tspans.length }, (_, index) => element.y + element.lineHeight * index),
    )
  })
})
