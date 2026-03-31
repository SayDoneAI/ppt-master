import { describe, expect, it } from 'vitest'
import {
  createDesignPatch,
  createDesignPatchDownload,
  createUpdatePatch,
  hasPatchValueChanged,
} from '../design_patch.js'

describe('design_patch', () => {
  it('生成符合编辑器约定的 update patch', () => {
    const patch = createUpdatePatch({
      slideIndex: 1,
      elementId: 'title',
      property: 'x',
      oldValue: 120,
      value: 180,
      timestamp: 1234567890,
    })

    expect(patch).toEqual({
      op: 'update',
      path: '/slides/1/elements/title/x',
      oldValue: 120,
      value: 180,
      source: 'human',
      timestamp: 1234567890,
    })
  })

  it('仅在值真正变化时返回 true', () => {
    expect(hasPatchValueChanged(10, 10)).toBe(false)
    expect(hasPatchValueChanged({ x: 1, y: 2 }, { x: 1, y: 3 })).toBe(true)
  })

  it('导出 design_patch.json 下载内容', () => {
    const operations = [
      createUpdatePatch({
        slideIndex: 0,
        elementId: 'cover_title',
        property: 'text',
        oldValue: '旧标题',
        value: '新标题',
        timestamp: 123,
      }),
    ]

    const designPatch = createDesignPatch(operations, 'human', '2026-03-31T00:00:00.000Z')
    expect(designPatch.operations).toHaveLength(1)
    expect(designPatch.operations[0]?.path).toBe('/slides/0/elements/cover_title/text')

    const download = createDesignPatchDownload(operations)
    expect(download.fileName).toBe('design_patch.json')
    expect(download.content).toContain('"operations"')
    expect(download.content).toContain('/slides/0/elements/cover_title/text')
  })
})
