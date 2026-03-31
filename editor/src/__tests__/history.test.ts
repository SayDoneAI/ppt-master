import { describe, expect, it } from 'vitest'
import { applyDesignPatch } from '../design_patch.js'
import {
  createHistoryEntry,
  flattenHistoryOperations,
  invertPatchOperation,
} from '../history.js'
import type { PatchOperation, SlideState } from '../slide_state.js'

describe('history', () => {
  const baseState: SlideState = {
    canvas: { width: 1280, height: 720 },
    slides: [
      {
        id: 'slide_01',
        elements: [
          { id: 'title', type: 'text', x: 80, y: 120, width: 420, text: '旧标题', font: 'bold 32px Inter', lineHeight: 44, fill: '#111827' },
        ],
      },
    ],
  }

  it('可为 update patch 生成对称 inverse patch', () => {
    const operation: PatchOperation = {
      op: 'update',
      path: '/slides/0/elements/title/text',
      oldValue: '旧标题',
      value: '新标题',
      source: 'human',
      timestamp: 1,
    }

    const inverted = invertPatchOperation(operation)
    const nextState = applyDesignPatch(baseState, {
      timestamp: '2026-03-31T00:00:00.000Z',
      source: 'human',
      operations: [operation],
    })
    const restored = applyDesignPatch(nextState, {
      timestamp: '2026-03-31T00:00:01.000Z',
      source: 'human',
      operations: [inverted],
    })

    expect((restored.slides[0].elements[0] as { text: string }).text).toBe('旧标题')
  })

  it('HistoryEntry 会为 slide add 生成可撤销的 inverse patch', () => {
    const entry = createHistoryEntry([
      {
        op: 'add',
        path: '/slides/1',
        value: { id: 'slide_02', elements: [] },
        source: 'human',
        timestamp: 2,
      },
    ], 'human', '追加页面')

    const nextState = applyDesignPatch(baseState, entry.forwardPatch)
    expect(nextState.slides).toHaveLength(2)
    expect(nextState.slides[1]?.id).toBe('slide_02')

    const restored = applyDesignPatch(nextState, entry.inversePatch)
    expect(restored.slides).toHaveLength(1)
    expect(restored.slides[0]?.id).toBe('slide_01')
  })

  it('flattenHistoryOperations 仅保留当前已生效的 forward operations', () => {
    const entries = [
      createHistoryEntry([
        {
          op: 'update',
          path: '/slides/0/elements/title/text',
          oldValue: '旧标题',
          value: '第一版标题',
          source: 'human',
          timestamp: 10,
        },
      ], 'human', '改标题'),
      createHistoryEntry([
        {
          op: 'add',
          path: '/slides/1',
          value: { id: 'slide_02', elements: [] },
          source: 'human',
          timestamp: 11,
        },
      ], 'human', '追加页面'),
    ]

    expect(flattenHistoryOperations(entries)).toHaveLength(2)
    expect(flattenHistoryOperations(entries)[0]?.path).toBe('/slides/0/elements/title/text')
    expect(flattenHistoryOperations(entries)[1]?.path).toBe('/slides/1')
  })
})
