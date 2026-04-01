export const LOCAL_AI_HANDOFF_DIR = '.cache/ai_handoff'
export const LOCAL_AI_HANDOFF_WRITE_ENDPOINT = '/__ppt_master/write-ai-handoff'
export const LOCAL_PROJECT_SAVE_PAGES_ENDPOINT = '/__ppt_master/save-pages'

export function sanitizeHandoffFileName(fileName: string): string {
  const normalized = fileName.trim().replace(/\\/g, '/')
  const segments = normalized.split('/').filter(Boolean)
  return segments.at(-1) ?? fileName
}

export function buildProjectAiHandoffRelativePath(fileName: string): string {
  return `${LOCAL_AI_HANDOFF_DIR}/${sanitizeHandoffFileName(fileName)}`
}
