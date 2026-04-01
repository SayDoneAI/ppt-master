import { mkdir, writeFile } from 'node:fs/promises'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { basename, isAbsolute, resolve } from 'node:path'
import type { Plugin, ViteDevServer } from 'vite'
import {
  LOCAL_AI_HANDOFF_DIR,
  LOCAL_AI_HANDOFF_WRITE_ENDPOINT,
  LOCAL_PROJECT_SAVE_PAGES_ENDPOINT,
  sanitizeHandoffFileName,
} from './local_ai_handoff.js'
import { STATE_WATCHER_HMR_EVENT } from './state_sync_events.js'

export interface StateWatcherPluginOptions {
  watchDir?: string
  fileName?: string
}

interface WriteArtifactPayload {
  fileName: string
  mimeType?: string
  content: string
}

interface WriteAiHandoffRequest {
  projectPath: string
  requestArtifact: WriteArtifactPayload
  noteArtifact: WriteArtifactPayload
}

interface SavePagePayload {
  filename: string
  svg: string
}

interface SavePagesRequest {
  projectPath: string
  pages: SavePagePayload[]
}

const DEFAULT_WATCH_DIR = '../../.cache'
const DEFAULT_FILE_NAME = 'slide_state.json'

export function stateWatcherPlugin(options: StateWatcherPluginOptions = {}): Plugin {
  let watchedDirPath: string | null = null
  let watchedFilePath: string | null = null
  const fileName = options.fileName ?? DEFAULT_FILE_NAME

  return {
    name: 'ppt-master-state-watcher',
    apply: 'serve',
    configResolved(resolvedConfig) {
      const root = resolvedConfig.root || process.cwd()
      const watchDir = options.watchDir ?? DEFAULT_WATCH_DIR
      watchedDirPath = resolve(root, watchDir)
      watchedFilePath = resolve(root, watchDir, fileName)
    },
    configureServer(server) {
      if (!watchedDirPath || !watchedFilePath) return

      server.middlewares.use(async (req, res, next) => {
        if (matchesRequestPath(req, LOCAL_AI_HANDOFF_WRITE_ENDPOINT)) {
          await handleLocalAiHandoffWrite(req, res)
          return
        }

        if (matchesRequestPath(req, LOCAL_PROJECT_SAVE_PAGES_ENDPOINT)) {
          await handleSavePages(req, res)
          return
        }

        next()
      })

      server.watcher.add(watchedDirPath)
      server.watcher.add(watchedFilePath)
      const notify = (filePath: string) => notifyStateFileChanged(server, filePath, fileName)

      server.watcher.on('add', notify)
      server.watcher.on('change', notify)

      return () => {
        server.watcher.off('add', notify)
        server.watcher.off('change', notify)
      }
    },
  }
}

function notifyStateFileChanged(server: ViteDevServer, filePath: string, fileName: string): void {
  if (basename(filePath) !== fileName) return

  server.ws.send({
    type: 'custom',
    event: STATE_WATCHER_HMR_EVENT,
    data: {
      filePath,
      urlPath: toViteFsPath(filePath),
    },
  })
}

function toViteFsPath(filePath: string): string {
  return `/@fs/${filePath.replace(/\\/g, '/')}`
}

function matchesRequestPath(req: IncomingMessage, targetPath: string): boolean {
  const rawUrl = req.url ?? ''
  const pathOnly = rawUrl.split('?')[0] ?? rawUrl
  return pathOnly === targetPath
}

async function handleLocalAiHandoffWrite(req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method !== 'POST') {
    respondJson(res, 405, { error: '仅支持 POST' })
    return
  }

  try {
    const payload = ensureWriteAiHandoffRequest(await readJsonBody(req))
    const rawProjectPath = payload.projectPath.trim()
    const projectPath = isAbsolute(rawProjectPath)
      ? rawProjectPath
      : resolve(process.cwd(), rawProjectPath)

    const requestFileName = sanitizeHandoffFileName(payload.requestArtifact.fileName)
    const noteFileName = sanitizeHandoffFileName(payload.noteArtifact.fileName)
    const handoffDir = resolve(projectPath, LOCAL_AI_HANDOFF_DIR)
    const requestPath = resolve(handoffDir, requestFileName)
    const notePath = resolve(handoffDir, noteFileName)

    await mkdir(handoffDir, { recursive: true })
    await writeFile(requestPath, payload.requestArtifact.content, 'utf8')
    await writeFile(notePath, payload.noteArtifact.content, 'utf8')

    respondJson(res, 200, {
      requestPath,
      notePath,
    })
  } catch (error) {
    respondJson(res, 400, {
      error: (error as Error).message,
    })
  }
}

async function handleSavePages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method !== 'POST') {
    respondJson(res, 405, { error: '仅支持 POST' })
    return
  }

  try {
    const payload = ensureSavePagesRequest(await readJsonBody(req))
    const rawProjectPath = payload.projectPath.trim()
    const projectPath = isAbsolute(rawProjectPath)
      ? rawProjectPath
      : resolve(process.cwd(), rawProjectPath)
    const targetDir = resolve(projectPath, 'design/pages')

    await mkdir(targetDir, { recursive: true })
    await Promise.all(payload.pages.map(async page => {
      const fileName = sanitizeSavedPageFileName(page.filename)
      const targetPath = resolve(targetDir, fileName)
      await writeFile(targetPath, page.svg, 'utf8')
    }))

    respondJson(res, 200, {
      ok: true,
      savedCount: payload.pages.length,
      dir: targetDir,
    })
  } catch (error) {
    respondJson(res, 400, {
      error: (error as Error).message,
    })
  }
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Uint8Array[] = []
  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk)
  }

  const rawBody = Buffer.concat(chunks).toString('utf8').trim()
  if (!rawBody) {
    throw new Error('请求体不能为空')
  }

  return JSON.parse(rawBody)
}

function ensureWriteAiHandoffRequest(value: unknown): WriteAiHandoffRequest {
  if (!isRecord(value)) {
    throw new Error('请求体必须是 JSON 对象')
  }

  const { projectPath, requestArtifact, noteArtifact } = value
  if (typeof projectPath !== 'string' || !projectPath.trim()) {
    throw new Error('缺少 projectPath')
  }

  return {
    projectPath,
    requestArtifact: ensureWriteArtifactPayload(requestArtifact, 'requestArtifact'),
    noteArtifact: ensureWriteArtifactPayload(noteArtifact, 'noteArtifact'),
  }
}

function ensureSavePagesRequest(value: unknown): SavePagesRequest {
  if (!isRecord(value)) {
    throw new Error('请求体必须是 JSON 对象')
  }

  const { projectPath, pages } = value
  if (typeof projectPath !== 'string' || !projectPath.trim()) {
    throw new Error('缺少 projectPath')
  }

  if (!Array.isArray(pages) || pages.length === 0) {
    throw new Error('pages 必须是非空数组')
  }

  return {
    projectPath,
    pages: pages.map((page, index) => ensureSavePagePayload(page, `pages[${index}]`)),
  }
}

function ensureWriteArtifactPayload(value: unknown, fieldName: string): WriteArtifactPayload {
  if (!isRecord(value)) {
    throw new Error(`${fieldName} 必须是对象`)
  }

  if (typeof value.fileName !== 'string' || !value.fileName.trim()) {
    throw new Error(`${fieldName}.fileName 缺失`)
  }

  if (typeof value.content !== 'string') {
    throw new Error(`${fieldName}.content 缺失`)
  }

  return {
    fileName: value.fileName,
    mimeType: typeof value.mimeType === 'string' ? value.mimeType : undefined,
    content: value.content,
  }
}

function ensureSavePagePayload(value: unknown, fieldName: string): SavePagePayload {
  if (!isRecord(value)) {
    throw new Error(`${fieldName} 必须是对象`)
  }

  if (typeof value.filename !== 'string' || !value.filename.trim()) {
    throw new Error(`${fieldName}.filename 缺失`)
  }

  if (typeof value.svg !== 'string' || !value.svg.trim()) {
    throw new Error(`${fieldName}.svg 缺失`)
  }

  return {
    filename: value.filename,
    svg: value.svg,
  }
}

function sanitizeSavedPageFileName(fileName: string): string {
  const sanitized = sanitizeHandoffFileName(fileName).trim()
  if (!sanitized) {
    throw new Error('pages.filename 不能为空')
  }

  return sanitized.toLowerCase().endsWith('.svg')
    ? sanitized
    : `${sanitized}.svg`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function respondJson(res: ServerResponse, statusCode: number, payload: Record<string, unknown>): void {
  res.statusCode = statusCode
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.end(`${JSON.stringify(payload)}\n`)
}

export { DEFAULT_FILE_NAME, DEFAULT_WATCH_DIR }
