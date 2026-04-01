import { describe, expect, it } from 'vitest'
import {
  CANVAS_PRESETS,
  findCanvasPreset,
  getTopbarSaveActionsState,
  isPosterCanvas,
  isPptCanvas,
  resizeSlideStateCanvas,
  shouldDefaultToPreview,
  shouldShowPosterCanvasControls,
  supportsCanvasPresetEditing,
} from '../canvas_resize.js'
import type { SlideState } from '../slide_state.js'

describe('canvas_resize', () => {
  function createState(
    canvas: { width: number, height: number },
    slideCount = 1,
  ): SlideState {
    return {
      canvas,
      slides: Array.from({ length: slideCount }, (_, index) => ({
        id: `slide-${index + 1}`,
        elements: [],
      })),
    }
  }

  const state: SlideState = {
    canvas: { width: 1080, height: 1350 },
    slides: [
      {
        id: 'poster',
        elements: [
          {
            id: 'title',
            type: 'text',
            x: 100,
            y: 200,
            width: 300,
            text: 'Hello',
            font: '700 80px PingFang SC',
            lineHeight: 92,
            fill: '#111827',
            letterSpacing: 1.2,
          },
          {
            id: 'card',
            type: 'rect',
            x: 500,
            y: 100,
            width: 320,
            height: 400,
            fill: '#fff',
            rx: 32,
          },
          {
            id: 'trend',
            type: 'line',
            x1: 520,
            y1: 460,
            x2: 760,
            y2: 380,
            stroke: '#0ea5e9',
            strokeWidth: 8,
          },
          {
            id: 'dot',
            type: 'circle',
            cx: 760,
            cy: 380,
            r: 14,
            fill: '#0ea5e9',
          },
          {
            id: 'group',
            type: 'group',
            children: [
              {
                id: 'group_rect',
                type: 'rect',
                x: 120,
                y: 700,
                width: 120,
                height: 60,
                fill: '#e5e7eb',
              },
            ],
          },
        ],
      },
    ],
  }

  it('会根据预设返回匹配的比例', () => {
    expect(findCanvasPreset({ width: 1080, height: 1350 })?.id).toBe('poster')
    expect(findCanvasPreset({ width: 1280, height: 720 })).toBeNull()
    expect(CANVAS_PRESETS).toHaveLength(3)
  })

  it('会区分 PPT 画布和海报画布', () => {
    expect(isPptCanvas({ width: 1280, height: 720 })).toBe(true)
    expect(isPptCanvas({ width: 1024, height: 768 })).toBe(true)
    expect(isPptCanvas({ width: 1080, height: 1350 })).toBe(false)
    expect(isPosterCanvas({ width: 1080, height: 1080 })).toBe(true)
    expect(isPosterCanvas({ width: 1080, height: 1920 })).toBe(true)
    expect(isPosterCanvas({ width: 1280, height: 720 })).toBe(false)
  })

  it('只在单页海报场景显示海报画幅控件', () => {
    expect(shouldShowPosterCanvasControls(createState({ width: 1080, height: 1350 }))).toBe(true)
    expect(shouldShowPosterCanvasControls(createState({ width: 1080, height: 1080 }))).toBe(true)
    expect(shouldShowPosterCanvasControls(createState({ width: 1080, height: 1920 }))).toBe(true)
    expect(shouldShowPosterCanvasControls(createState({ width: 1080, height: 1350 }, 2))).toBe(false)
    expect(shouldShowPosterCanvasControls(createState({ width: 1280, height: 720 }))).toBe(false)
    expect(shouldShowPosterCanvasControls(createState({ width: 1024, height: 768 }))).toBe(false)
  })

  it('会按 PPT / 海报模式切换顶部保存按钮规则', () => {
    expect(getTopbarSaveActionsState(createState({ width: 1080, height: 1350 }))).toEqual({
      exportSvgLabel: '保存 SVG',
      showSaveTemplate: true,
      mode: 'poster',
    })
    expect(getTopbarSaveActionsState(createState({ width: 1280, height: 720 }))).toEqual({
      exportSvgLabel: '保存当前页',
      showSaveTemplate: false,
      mode: 'ppt',
    })
    expect(getTopbarSaveActionsState(createState({ width: 1080, height: 1350 }, 2))).toEqual({
      exportSvgLabel: '保存当前页',
      showSaveTemplate: false,
      mode: 'ppt',
    })
  })

  it('会把单页竖版默认切到预览模式', () => {
    expect(shouldDefaultToPreview(state)).toBe(true)
    expect(shouldDefaultToPreview({
      canvas: { width: 1280, height: 720 },
      slides: [{ id: 'slide', elements: [] }],
    })).toBe(false)
  })

  it('会按新画布比例缩放文本和基础几何元素', () => {
    const next = resizeSlideStateCanvas(state, { width: 1080, height: 1080 })
    const [title, card, trend, dot, group] = next.slides[0].elements

    expect(next.canvas).toEqual({ width: 1080, height: 1080 })
    expect(title.type === 'text' ? title.y : 0).toBe(160)
    expect(title.type === 'text' ? title.font : '').toContain('64px')
    expect(title.type === 'text' ? title.lineHeight : 0).toBe(73.6)
    expect(card.type === 'rect' ? card.height : 0).toBe(320)
    expect(trend.type === 'line' ? trend.y1 : 0).toBe(368)
    expect(dot.type === 'circle' ? dot.r : 0).toBe(11.2)
    expect(group.type === 'group' && group.children[0]?.type === 'rect'
      ? group.children[0].y
      : 0).toBe(560)
  })

  it('含 path 的页面不会启用实时比例预设', () => {
    expect(supportsCanvasPresetEditing(state)).toBe(true)
    expect(supportsCanvasPresetEditing({
      canvas: { width: 1080, height: 1350 },
      slides: [
        {
          id: 'unsupported',
          elements: [
            { id: 'shape', type: 'path', d: 'M0 0 L10 10', fill: '#000' },
          ],
        },
      ],
    })).toBe(false)
  })
})
