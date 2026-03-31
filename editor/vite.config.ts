import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import { stateWatcherPlugin } from './src/vite-plugin-state-watcher.js'

const editorRoot = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  root: editorRoot,
  plugins: [stateWatcherPlugin()],
  build: {
    outDir: 'dist/app',
    emptyOutDir: true,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
})
