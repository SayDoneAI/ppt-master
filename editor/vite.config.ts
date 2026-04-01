import { copyFile, mkdir, readFile } from 'node:fs/promises'
import { dirname, extname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import { stateWatcherPlugin } from './src/vite-plugin-state-watcher.js'

const editorRoot = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(editorRoot, '..')
const examplesRoot = resolve(repoRoot, 'examples')
const demoSvgRelativePath = 'demo_project_intro_ppt169_20251211/svg_final/slide_01_cover.svg'
const demoImageRelativePath = 'demo_project_intro_ppt169_20251211/images/cover_background.png'

function examplesStaticPlugin(): Plugin {
  return {
    name: 'examples-static-plugin',
    configureServer(server) {
      server.middlewares.use('/examples', async (req, res, next) => {
        const requestUrl = req.url?.startsWith('/examples/')
          ? req.url
          : `/examples${req.url ?? ''}`
        const requestPath = new URL(requestUrl, 'http://vite.local').pathname
        const relativePath = requestPath.replace(/^\/examples\/?/, '')
        const filePath = resolve(examplesRoot, relativePath)
        if (!filePath.startsWith(examplesRoot)) {
          next()
          return
        }

        try {
          const data = await readFile(filePath)
          res.statusCode = 200
          res.setHeader('Content-Type', getContentType(filePath))
          res.end(data)
        } catch {
          next()
        }
      })
    },
    async closeBundle() {
      const outDir = resolve(editorRoot, 'dist/app/examples')
      await copyStaticExampleAsset(demoSvgRelativePath, outDir)
      await copyStaticExampleAsset(demoImageRelativePath, outDir)
    },
  }
}

async function copyStaticExampleAsset(relativePath: string, outDir: string): Promise<void> {
  const sourcePath = resolve(examplesRoot, relativePath)
  const targetPath = resolve(outDir, relativePath)
  await mkdir(dirname(targetPath), { recursive: true })
  await copyFile(sourcePath, targetPath)
}

function getContentType(filePath: string): string {
  switch (extname(filePath).toLowerCase()) {
    case '.svg':
      return 'image/svg+xml'
    case '.png':
      return 'image/png'
    default:
      return 'application/octet-stream'
  }
}

export default defineConfig({
  root: editorRoot,
  plugins: [stateWatcherPlugin(), examplesStaticPlugin()],
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
