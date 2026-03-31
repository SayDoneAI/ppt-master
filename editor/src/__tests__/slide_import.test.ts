import { describe, expect, it } from 'vitest'
import {
  createAppendSlidesPatch,
  createUniqueImportedSlides,
  ensureCompatibleCanvas,
} from '../slide_import.js'
import type { SlideState } from '../slide_state.js'

describe('slide_import', () => {
  const state: SlideState = {
    canvas: { width: 1280, height: 720 },
    slides: [
      { id: 'slide_01_cover', elements: [] },
      { id: 'slide_02_content', elements: [] },
    ],
  }

  it('会为导入页面生成唯一 slide id', () => {
    const slides = createUniqueImportedSlides(state.slides, [
      { id: 'slide_02_content', elements: [] },
      { id: 'slide_02_content', elements: [] },
      { id: '  ', elements: [] },
    ])

    expect(slides.map(slide => slide.id)).toEqual([
      'slide_02_content_02',
      'slide_02_content_03',
      'imported_slide_03',
    ])
  })

  it('会在指定位置生成 slide add patch', () => {
    const operations = createAppendSlidesPatch({
      state,
      importedSlides: [
        { id: 'template_cover', elements: [] },
        { id: 'template_content', elements: [] },
      ],
      insertIndex: 1,
      timestampStart: 100,
    })

    expect(operations).toHaveLength(2)
    expect(operations[0]?.path).toBe('/slides/1')
    expect(operations[1]?.path).toBe('/slides/2')
    expect(operations[0]?.value.id).toBe('template_cover')
    expect(operations[1]?.value.id).toBe('template_content')
  })

  it('导入尺寸不一致时直接报错', () => {
    expect(() => ensureCompatibleCanvas(
      { width: 1280, height: 720 },
      { width: 1242, height: 1660 },
    )).toThrow('导入页面尺寸不匹配')
  })
})
