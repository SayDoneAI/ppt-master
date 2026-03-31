import { describe, it, expect } from 'vitest'
import { slideToSvg, stateToSvgs, layoutText } from '../state_to_svg.js'
import type { Slide, SlideState, TextElement } from '../slide_state.js'

describe('slideToSvg', () => {
  const canvas = { width: 1280, height: 720 }

  it('生成带背景的基础 SVG', () => {
    const slide: Slide = {
      id: 'slide_01',
      background: '#FFFFFF',
      elements: [],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('viewBox="0 0 1280 720"')
    expect(svg).toContain('fill="#FFFFFF"')
    expect(svg).toContain('</svg>')
  })

  it('生成 rect 元素', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'rect', id: 'r1',
        x: 60, y: 60, width: 400, height: 300,
        fill: '#F5F5F7', rx: 12,
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('x="60"')
    expect(svg).toContain('width="400"')
    expect(svg).toContain('rx="12"')
    expect(svg).toContain('fill="#F5F5F7"')
    expect(svg).toContain('data-element-id="r1"')
  })

  it('生成单行 text 元素', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'text', id: 't1',
        x: 60, y: 80, width: 600,
        text: 'Hello',
        font: 'bold 36px Arial',
        lineHeight: 48,
        fill: '#1A1A2E',
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('Hello')
    expect(svg).toContain('font-size="36"')
    expect(svg).toContain('font-weight="bold"')
    expect(svg).toContain('fill="#1A1A2E"')
    expect(svg).toContain('data-element-id="t1"')
  })

  it('生成多行 text 元素（fallback 断行）', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'text', id: 't1',
        x: 60, y: 80, width: 200,
        text: '这是一段很长的文本，需要被自动断行成多行显示在幻灯片中',
        font: '16px Arial',
        lineHeight: 24,
        fill: '#333333',
      }],
    }
    const svg = slideToSvg(slide, canvas)
    // 多行文本应该生成多个 <text> 元素
    const textCount = (svg.match(/<text /g) || []).length
    expect(textCount).toBeGreaterThanOrEqual(2)
  })

  it('生成 line 元素', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'line', id: 'l1',
        x1: 100, y1: 450, x2: 480, y2: 450,
        stroke: '#FFFFFF', strokeWidth: 2,
        opacity: 0.4,
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('x1="100"')
    expect(svg).toContain('stroke="#FFFFFF"')
    expect(svg).toContain('opacity="0.4"')
  })

  it('生成 circle 元素', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'circle', id: 'c1',
        cx: 340, cy: 40, r: 25,
        fill: '#6366F1',
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('cx="340"')
    expect(svg).toContain('r="25"')
  })

  it('生成 image 元素', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'image', id: 'img1',
        x: 0, y: 0, width: 1280, height: 720,
        href: 'images/cover.png',
        preserveAspectRatio: 'xMidYMid slice',
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('href="images/cover.png"')
    expect(svg).toContain('preserveAspectRatio="xMidYMid slice"')
  })

  it('生成 path 元素', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'path', id: 'p1',
        d: 'M60,100 H400 A12,12 0 0 1 412,112 V300',
        fill: '#10B981',
        fillOpacity: 0.1,
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('d="M60,100 H400')
    expect(svg).toContain('fill-opacity="0.1"')
  })

  it('生成 group 嵌套元素', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'group', id: 'g1',
        transform: 'translate(780, 180)',
        children: [
          { type: 'circle', id: 'c1', cx: 100, cy: 100, r: 50, fill: '#FF0000' },
          { type: 'text', id: 't1', x: 80, y: 110, width: 200, text: 'AI', font: '14px Arial', lineHeight: 20, fill: '#FFF' },
        ],
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('<g data-element-id="g1" transform="translate(780, 180)">')
    expect(svg).toContain('</g>')
    expect(svg).toContain('data-element-id="c1"')
    expect(svg).toContain('data-element-id="t1"')
    expect(svg).toContain('cx="100"')
    expect(svg).toContain('>AI</text>')
  })

  it('生成 linearGradient defs', () => {
    const slide: Slide = {
      id: 'slide_01',
      defs: [{
        type: 'linearGradient', id: 'grad1',
        x1: '0%', y1: '0%', x2: '100%', y2: '100%',
        stops: [
          { offset: '0%', color: '#6366F1' },
          { offset: '100%', color: '#06B6D4' },
        ],
      }],
      elements: [{
        type: 'rect', id: 'r1',
        x: 0, y: 0, width: 1280, height: 6,
        fill: 'url(#grad1)',
      }],
    }
    const svg = slideToSvg(slide, canvas)
    expect(svg).toContain('<defs>')
    expect(svg).toContain('linearGradient')
    expect(svg).toContain('stop-color="#6366F1"')
    expect(svg).toContain('fill="url(#grad1)"')
  })
})

describe('stateToSvgs', () => {
  it('生成多页 SVG', () => {
    const state: SlideState = {
      canvas: { width: 1280, height: 720 },
      slides: [
        { id: 'slide_01', background: '#FFF', elements: [] },
        { id: 'slide_02', background: '#000', elements: [] },
      ],
    }
    const svgs = stateToSvgs(state)
    expect(svgs).toHaveLength(2)
    expect(svgs[0]).toContain('#FFF')
    expect(svgs[1]).toContain('#000')
  })
})

describe('layoutText (fallback)', () => {
  it('短文本不断行', () => {
    const el: TextElement = {
      type: 'text', id: 't', x: 0, y: 0, width: 600,
      text: 'Hello', font: '16px Arial', lineHeight: 24, fill: '#000',
    }
    const result = layoutText(el)
    expect(result.lineCount).toBe(1)
    expect(result.overflow).toBe(false)
  })

  it('长文本自动断行', () => {
    const el: TextElement = {
      type: 'text', id: 't', x: 0, y: 0, width: 100,
      text: '这是一段非常长的文本内容需要被自动断行成多行来显示',
      font: '16px Arial', lineHeight: 24, fill: '#000',
    }
    const result = layoutText(el)
    expect(result.lineCount).toBeGreaterThan(1)
  })

  it('检测溢出', () => {
    const el: TextElement = {
      type: 'text', id: 't', x: 0, y: 0, width: 100,
      text: '这是一段非常长的文本内容需要被自动断行成多行来显示，而且容器高度有限',
      font: '16px Arial', lineHeight: 24, fill: '#000',
      maxHeight: 30,
    }
    const result = layoutText(el)
    expect(result.overflow).toBe(true)
  })
})

describe('SVG 约束合规性', () => {
  const canvas = { width: 1280, height: 720 }

  it('不包含禁用的 SVG 特性', () => {
    const slide: Slide = {
      id: 'slide_01',
      background: '#FFFFFF',
      defs: [{
        type: 'linearGradient', id: 'g1',
        x1: '0%', y1: '0%', x2: '100%', y2: '0%',
        stops: [{ offset: '0%', color: '#000' }],
      }],
      elements: [
        { type: 'rect', id: 'r1', x: 0, y: 0, width: 100, height: 100, fill: '#FFF', rx: 8 },
        { type: 'text', id: 't1', x: 60, y: 80, width: 600, text: 'Test', font: '16px Arial', lineHeight: 24, fill: '#000' },
        { type: 'path', id: 'p1', d: 'M0,0 L100,100', fill: 'none', stroke: '#000' },
      ],
    }
    const svg = slideToSvg(slide, canvas)

    // ppt-master 禁用功能黑名单
    expect(svg).not.toContain('<style')
    expect(svg).not.toContain('class=')
    expect(svg).not.toContain('<foreignObject')
    expect(svg).not.toContain('<textPath')
    expect(svg).not.toContain('@font-face')
    expect(svg).not.toContain('<animate')
    expect(svg).not.toContain('<script')
    expect(svg).not.toContain('marker-end')
    expect(svg).not.toContain('<iframe')
    expect(svg).not.toContain('<symbol')
    expect(svg).not.toContain('clipPath')
    expect(svg).not.toContain('<mask')
    expect(svg).not.toContain('rgba(')
  })

  it('文本不使用 tspan（使用多个 text 元素）', () => {
    const slide: Slide = {
      id: 'slide_01',
      elements: [{
        type: 'text', id: 't1', x: 60, y: 80, width: 200,
        text: '第一行内容在这里，第二行内容也在这里',
        font: '16px Arial', lineHeight: 24, fill: '#000',
      }],
    }
    const svg = slideToSvg(slide, canvas)
    // 多行文本使用多个 <text> 而非 <tspan>（兼容性更好）
    expect(svg).not.toContain('<tspan')
  })
})
