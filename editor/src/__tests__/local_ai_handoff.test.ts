import { describe, expect, it } from 'vitest'
import {
  buildProjectAiHandoffRelativePath,
  LOCAL_AI_HANDOFF_DIR,
  sanitizeHandoffFileName,
} from '../local_ai_handoff.js'

describe('local_ai_handoff', () => {
  it('规范化 handoff 文件名，避免路径注入', () => {
    expect(sanitizeHandoffFileName('design_patch.ai-request.json')).toBe('design_patch.ai-request.json')
    expect(sanitizeHandoffFileName('../design_patch.ai-request.json')).toBe('design_patch.ai-request.json')
    expect(sanitizeHandoffFileName('foo/bar/design_patch.ai-handoff.md')).toBe('design_patch.ai-handoff.md')
    expect(sanitizeHandoffFileName('foo\\bar\\design_patch.ai-handoff.md')).toBe('design_patch.ai-handoff.md')
  })

  it('生成项目内临时 handoff 相对路径', () => {
    expect(buildProjectAiHandoffRelativePath('design_patch.ai-request.json'))
      .toBe(`${LOCAL_AI_HANDOFF_DIR}/design_patch.ai-request.json`)
  })
})
