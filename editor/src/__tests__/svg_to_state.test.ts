import { describe, it, expect } from 'vitest'
import { JSDOM } from 'jsdom'
import { svgToSlide, svgsToState } from '../svg_to_state.js'

// JSDOM 提供 DOMParser
const dom = new JSDOM()
const parser = new dom.window.DOMParser()

describe('svgToSlide', () => {
  it('解析 viewBox 和基础 rect', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720" width="1280" height="720">
      <rect width="1280" height="720" fill="#FFFFFF" />
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    expect(slide.id).toBe('slide_01')
    expect(slide.elements.length).toBeGreaterThanOrEqual(1)
    const rect = slide.elements.find(e => e.type === 'rect')
    expect(rect).toBeDefined()
    if (rect?.type === 'rect') {
      expect(rect.width).toBe(1280)
      expect(rect.fill).toBe('#FFFFFF')
    }
  })

  it('解析 text 元素', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <text x="100" y="300" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="#FFFFFF">
        PPT Master
      </text>
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    const text = slide.elements.find(e => e.type === 'text')
    expect(text).toBeDefined()
    if (text?.type === 'text') {
      expect(text.text).toContain('PPT Master')
      expect(text.x).toBe(100)
      expect(text.fontSize).toBe(60)
      expect(text.fontWeight).toBe('bold')
      expect(text.fill).toBe('#FFFFFF')
    }
  })

  it('合并相邻的多行 text', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <text x="60" y="105" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#1A1A2E">
        第一行文本
      </text>
      <text x="60" y="140" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#1A1A2E">
        第二行文本
      </text>
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    // 两个相邻 text 应被合并为一个
    const texts = slide.elements.filter(e => e.type === 'text')
    expect(texts).toHaveLength(1)
    if (texts[0]?.type === 'text') {
      expect(texts[0].text).toContain('第一行文本')
      expect(texts[0].text).toContain('第二行文本')
    }
  })

  it('不合并不同位置/样式的 text', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <text x="60" y="100" font-family="Arial" font-size="28" fill="#1A1A2E">标题</text>
      <text x="60" y="200" font-family="Arial" font-size="16" fill="#666666">正文</text>
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    const texts = slide.elements.filter(e => e.type === 'text')
    // 字号不同，不应合并
    expect(texts).toHaveLength(2)
  })

  it('解析 linearGradient', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#6366F1" />
          <stop offset="100%" stop-color="#06B6D4" />
        </linearGradient>
      </defs>
      <rect width="1280" height="6" fill="url(#grad1)" />
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    expect(slide.defs).toBeDefined()
    expect(slide.defs!.length).toBe(1)
    const grad = slide.defs![0]
    expect(grad.type).toBe('linearGradient')
    if (grad.type === 'linearGradient') {
      expect(grad.id).toBe('grad1')
      expect(grad.stops).toHaveLength(2)
      expect(grad.stops[0].color).toBe('#6366F1')
    }
  })

  it('解析 group 和嵌套元素', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <g transform="translate(780, 180)">
        <circle cx="340" cy="40" r="25" fill="#6366F1" />
        <text x="340" y="47" font-size="14" font-weight="bold" text-anchor="middle" fill="#FFFFFF">AI</text>
      </g>
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    const group = slide.elements.find(e => e.type === 'group')
    expect(group).toBeDefined()
    if (group?.type === 'group') {
      expect(group.transform).toBe('translate(780, 180)')
      expect(group.children).toHaveLength(2)
      expect(group.children[0].type).toBe('circle')
      expect(group.children[1].type).toBe('text')
    }
  })

  it('解析 line 元素', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <line x1="100" y1="450" x2="480" y2="450" stroke="#FFFFFF" stroke-width="2" opacity="0.4" />
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    const line = slide.elements.find(e => e.type === 'line')
    expect(line).toBeDefined()
    if (line?.type === 'line') {
      expect(line.x1).toBe(100)
      expect(line.x2).toBe(480)
      expect(line.stroke).toBe('#FFFFFF')
      expect(line.opacity).toBe(0.4)
    }
  })

  it('解析 image 元素', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <image href="images/cover.png" x="0" y="0" width="1280" height="720" preserveAspectRatio="xMidYMid slice" />
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    const img = slide.elements.find(e => e.type === 'image')
    expect(img).toBeDefined()
    if (img?.type === 'image') {
      expect(img.href).toBe('images/cover.png')
      expect(img.preserveAspectRatio).toBe('xMidYMid slice')
    }
  })

  it('解析 path 元素', () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <path fill="#10B981" fill-opacity="0.1" d="M72,180 H408 A12,12 0 0 1 420,192 V588" />
    </svg>`
    const slide = svgToSlide(svg, 'slide_01', parser)
    const path = slide.elements.find(e => e.type === 'path')
    expect(path).toBeDefined()
    if (path?.type === 'path') {
      expect(path.fill).toBe('#10B981')
      expect(path.fillOpacity).toBe(0.1)
      expect(path.d).toContain('M72,180')
    }
  })
})

describe('svgsToState', () => {
  it('多页 SVG 合成 SlideState', () => {
    const svg1 = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <rect width="1280" height="720" fill="#FFF" />
    </svg>`
    const svg2 = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <rect width="1280" height="720" fill="#000" />
    </svg>`
    const state = svgsToState([svg1, svg2], ['cover', 'content'], parser)
    expect(state.canvas.width).toBe(1280)
    expect(state.canvas.height).toBe(720)
    expect(state.slides).toHaveLength(2)
    expect(state.slides[0].id).toBe('cover')
    expect(state.slides[1].id).toBe('content')
  })
})
