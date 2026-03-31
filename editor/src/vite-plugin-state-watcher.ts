import { mkdir, writeFile } from 'node:fs/promises'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { basename, isAbsolute, resolve } from 'node:path'
import type { Plugin, ViteDevServer } from 'vite'
import {
  LOCAL_AI_HANDOFF_DIR,
  LOCAL_AI_HANDOFF_WRITE_ENDPOINT,
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
        if (!matchesRequestPath(req, LOCAL_AI_HANDOFF_WRITE_ENDPOINT)) {
          next()
          return
        }

        await handleLocalAiHandoffWrite(req, res)
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function respondJson(res: ServerResponse, statusCode: number, payload: Record<string, unknown>): void {
  res.statusCode = statusCode
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.end(`${JSON.stringify(payload)}\n`)
}

export { DEFAULT_FILE_NAME, DEFAULT_WATCH_DIR }
