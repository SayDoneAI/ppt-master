import { describe, expect, it } from 'vitest'
import { buildProjectAiHandoffRelativePath } from '../local_ai_handoff.js'
import {
  applyDesignPatch,
  createAiCommand,
  createAiCommandDownload,
  createAiCommandPromptDownload,
  createDesignPatch,
  createDesignPatchDownload,
  createUpdatePatch,
  getDesignPatchPrimarySlideIndex,
  hasPatchValueChanged,
  parseDesignPatchJson,
} from '../design_patch.js'
import type { DesignPatch, SlideState } from '../slide_state.js'

describe('design_patch', () => {
  const state: SlideState = {
    canvas: { width: 1280, height: 720 },
    slides: [
      {
        id: 'slide_01',
        elements: [
          { id: 'title', type: 'text', x: 120, y: 90, width: 420, text: '旧标题', font: 'bold 36px Inter', lineHeight: 48, fill: '#111827' },
          { id: 'card', type: 'rect', x: 80, y: 180, width: 320, height: 180, fill: '#EEF2FF' },
          {
            id: 'group_01',
            type: 'group',
            children: [
              { id: 'badge_text', type: 'text', x: 100, y: 240, width: 140, text: '旧徽章', font: '600 20px Inter', lineHeight: 28, fill: '#312E81' },
            ],
          },
        ],
      },
    ],
  }

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

  it('围绕当前选中元素生成 AI 指令请求', () => {
    const aiCommand = createAiCommand({
      state,
      slideIndex: 0,
      scope: 'selected-element',
      elementId: 'badge_text',
      instruction: '把这个徽章文案改成更像策略亮点',
    })

    expect(aiCommand.scope).toBe('selected-element')
    expect(aiCommand.slideIndex).toBe(0)
    expect(aiCommand.slideId).toBe('slide_01')
    expect(aiCommand.elementId).toBe('badge_text')
    expect(aiCommand.slideSnapshot.id).toBe('slide_01')
    expect(aiCommand.elementSnapshot?.id).toBe('badge_text')

    const download = createAiCommandDownload(aiCommand)
    expect(download.fileName).toBe('design_patch.ai-request.json')
    expect(download.content).toContain('"aiCommand"')
    expect(download.content).toContain('"scope": "selected-element"')

    const promptDownload = createAiCommandPromptDownload(aiCommand, [], {
      projectPathHint: '/tmp/demo_project',
      stateFilePathHint: '/tmp/demo_project/slide_state.json',
      requestFileName: buildProjectAiHandoffRelativePath('design_patch.ai-request.json'),
    })
    expect(promptDownload.fileName).toBe('design_patch.ai-handoff.md')
    expect(promptDownload.content).toContain('Claude Code / Codex')
    expect(promptDownload.content).toContain('design_patch.ai-request.json')
    expect(promptDownload.content).toContain('/tmp/demo_project/.cache/ai_handoff/design_patch.ai-request.json')
    expect(promptDownload.content).toContain('本地 skill / command')
    expect(promptDownload.content).toContain('不是浏览器直连或服务端 API 协议')
    expect(promptDownload.content).toContain('python3 tools/slide_state_bridge.py render /tmp/demo_project')
    expect(promptDownload.content).toContain('把这个徽章文案改成更像策略亮点')
  })

  it('解析并应用 AI design patch 到当前 state', () => {
    const patch: DesignPatch = {
      timestamp: '2026-03-31T12:00:00.000Z',
      source: 'ai',
      aiCommand: createAiCommand({
        state,
        slideIndex: 0,
        scope: 'current-slide',
        instruction: '新增一个亮点标签并优化标题',
      }),
      operations: [
        {
          op: 'update',
          path: '/slides/0/elements/title/text',
          oldValue: '旧标题',
          value: 'AI 优化后的标题',
          source: 'ai',
          timestamp: 1,
        },
        {
          op: 'add',
          path: '/slides/0/elements/group_01/children',
          value: {
            id: 'badge_dot',
            type: 'circle',
            cx: 84,
            cy: 224,
            r: 8,
            fill: '#4F46E5',
          },
          source: 'ai',
          timestamp: 2,
        },
        {
          op: 'reorder',
          path: '/slides/0/elements/card',
          oldValue: 1,
          value: 0,
          source: 'ai',
          timestamp: 3,
        },
      ],
    }

    const parsed = parseDesignPatchJson(JSON.stringify(patch))
    const nextState = applyDesignPatch(state, parsed)

    expect(nextState.slides[0].elements[0]?.id).toBe('card')
    expect(nextState.slides[0].elements[1]?.id).toBe('title')
    expect(nextState.slides[0].elements[1]?.type).toBe('text')
    expect((nextState.slides[0].elements[1] as { text: string }).text).toBe('AI 优化后的标题')

    const group = nextState.slides[0].elements.find(element => element.id === 'group_01')
    expect(group?.type).toBe('group')
    expect(group?.type === 'group' ? group.children.map(element => element.id) : []).toContain('badge_dot')
  })

  it('支持 slide 级 add / reorder / delete patch', () => {
    const added = applyDesignPatch(state, {
      timestamp: '2026-03-31T12:10:00.000Z',
      source: 'human',
      operations: [
        {
          op: 'add',
          path: '/slides/1',
          value: {
            id: 'slide_02',
            elements: [
              { id: 'cover', type: 'rect', x: 0, y: 0, width: 1280, height: 720, fill: '#FFFFFF' },
            ],
          },
          source: 'human',
          timestamp: 10,
        },
      ],
    })

    expect(added.slides).toHaveLength(2)
    expect(added.slides[1]?.id).toBe('slide_02')

    const reordered = applyDesignPatch(added, {
      timestamp: '2026-03-31T12:11:00.000Z',
      source: 'human',
      operations: [
        {
          op: 'reorder',
          path: '/slides/1',
          oldValue: 1,
          value: 0,
          source: 'human',
          timestamp: 11,
        },
      ],
    })

    expect(reordered.slides[0]?.id).toBe('slide_02')

    const restored = applyDesignPatch(reordered, {
      timestamp: '2026-03-31T12:12:00.000Z',
      source: 'human',
      operations: [
        {
          op: 'delete',
          path: '/slides/0',
          oldValue: reordered.slides[0],
          source: 'human',
          timestamp: 12,
        },
      ],
    })

    expect(restored.slides).toHaveLength(1)
    expect(restored.slides[0]?.id).toBe('slide_01')
  })

  it('slide 级 patch 也能推导 primary slide index', () => {
    expect(getDesignPatchPrimarySlideIndex({
      timestamp: '2026-03-31T12:13:00.000Z',
      source: 'human',
      operations: [
        {
          op: 'add',
          path: '/slides/3',
          value: { id: 'slide_04', elements: [] },
          source: 'human',
          timestamp: 13,
        },
      ],
    })).toBe(3)
  })
})
