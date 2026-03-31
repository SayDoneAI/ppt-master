import { basename, resolve } from 'node:path'
import type { Plugin, ViteDevServer } from 'vite'
import { STATE_WATCHER_HMR_EVENT } from './state_sync_events.js'

export interface StateWatcherPluginOptions {
  watchDir?: string
  fileName?: string
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

export { DEFAULT_FILE_NAME, DEFAULT_WATCH_DIR }
