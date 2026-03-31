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
    fs: {
      // 编辑器需要通过 /@fs/ 绑定项目外部的 slide_state.json，本地开发时关闭严格目录限制。
      strict: false,
    },
  },
})
