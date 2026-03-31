import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
import { JSDOM } from 'jsdom'
import { parseSlideStateJson } from './state_io.js'
import {
  buildProjectStateFromSvgInputs,
  compareProjectFileName,
  createProjectSvgArtifacts,
} from './project_pipeline.js'

type CommandName = 'capture-project' | 'render-project' | 'sync-project'

interface CliOptions {
  command: CommandName
  projectPath: string
  sourceDir: string
  outputDir: string
  stateFile: string
}

const DEFAULT_SOURCE_DIR = 'svg_output'
const DEFAULT_OUTPUT_DIR = 'svg_output'
const DEFAULT_STATE_FILE = 'slide_state.json'

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2))
  switch (options.command) {
    case 'capture-project':
      await captureProject(options)
      return
    case 'render-project':
      await renderProject(options)
      return
    case 'sync-project':
      await syncProject(options)
      return
  }
}

async function captureProject(options: CliOptions): Promise<void> {
  const { state, svgFiles } = await readProjectSvgState(options.projectPath, options.sourceDir)
  const statePath = resolve(options.projectPath, options.stateFile)
  await writeFile(statePath, `${JSON.stringify(state, null, 2)}\n`, 'utf8')

  console.log(`[OK] 已从 ${svgFiles.length} 个 SVG 生成 ${relativeDisplayPath(statePath)}`)
}

async function renderProject(options: CliOptions): Promise<void> {
  const statePath = resolve(options.projectPath, options.stateFile)
  const rawJson = await readFile(statePath, 'utf8')
  const state = parseSlideStateJson(rawJson)
  const outputDir = resolve(options.projectPath, options.outputDir)
  const artifacts = createProjectSvgArtifacts(state)

  await mkdir(outputDir, { recursive: true })
  for (const artifact of artifacts) {
    await writeFile(resolve(outputDir, artifact.fileName), artifact.content, 'utf8')
  }

  console.log(`[OK] 已从 ${relativeDisplayPath(statePath)} 渲染 ${artifacts.length} 个 SVG 到 ${relativeDisplayPath(outputDir)}`)
  await warnStaleSvgFiles(outputDir, artifacts.map(artifact => artifact.fileName))
}

async function syncProject(options: CliOptions): Promise<void> {
  const { state, svgFiles } = await readProjectSvgState(options.projectPath, options.sourceDir)
  const statePath = resolve(options.projectPath, options.stateFile)
  const outputDir = resolve(options.projectPath, options.outputDir)
  const artifacts = createProjectSvgArtifacts(state)

  await writeFile(statePath, `${JSON.stringify(state, null, 2)}\n`, 'utf8')
  await mkdir(outputDir, { recursive: true })
  for (const artifact of artifacts) {
    await writeFile(resolve(outputDir, artifact.fileName), artifact.content, 'utf8')
  }

  console.log(`[OK] 已从 ${svgFiles.length} 个 SVG 同步生成 ${relativeDisplayPath(statePath)} 并回写 ${artifacts.length} 个 SVG 到 ${relativeDisplayPath(outputDir)}`)
  await warnStaleSvgFiles(outputDir, artifacts.map(artifact => artifact.fileName))
}

async function readProjectSvgState(
  projectPath: string,
  sourceDir: string,
): Promise<{ state: ReturnType<typeof buildProjectStateFromSvgInputs>; svgFiles: string[] }> {
  const svgDir = resolve(projectPath, sourceDir)
  const dirEntries = await readdir(svgDir, { withFileTypes: true })
  const svgFiles = dirEntries
    .filter(entry => entry.isFile() && entry.name.toLowerCase().endsWith('.svg'))
    .map(entry => entry.name)
    .sort((a, b) => compareProjectFileName({ fileName: a }, { fileName: b }))

  if (svgFiles.length === 0) {
    throw new Error(`目录中没有 SVG 文件: ${relativeDisplayPath(svgDir)}`)
  }

  const svgInputs = await Promise.all(svgFiles.map(async fileName => ({
    fileName,
    content: await readFile(resolve(svgDir, fileName), 'utf8'),
  })))

  const dom = new JSDOM()
  const parser = new dom.window.DOMParser()
  return {
    state: buildProjectStateFromSvgInputs(svgInputs, parser),
    svgFiles,
  }
}

async function warnStaleSvgFiles(outputDir: string, expectedFileNames: string[]): Promise<void> {
  const existingEntries = await readdir(outputDir, { withFileTypes: true })
  const existingSvgFiles = existingEntries
    .filter(entry => entry.isFile() && entry.name.toLowerCase().endsWith('.svg'))
    .map(entry => entry.name)

  const expected = new Set(expectedFileNames)
  const stale = existingSvgFiles.filter(fileName => !expected.has(fileName))
  if (stale.length === 0) return

  console.warn(`[WARN] ${relativeDisplayPath(outputDir)} 中仍存在 ${stale.length} 个旧 SVG，未自动删除: ${stale.join(', ')}`)
}

function parseArgs(argv: string[]): CliOptions {
  if (argv.length === 0 || isHelpFlag(argv[0])) {
    printUsage()
    process.exit(0)
  }

  const command = argv[0] as CommandName
  if (!isCommandName(command)) {
    throw new Error(`未知命令: ${argv[0]}`)
  }

  const projectPath = argv[1]
  if (!projectPath || isHelpFlag(projectPath)) {
    throw new Error('需要提供项目路径')
  }

  let sourceDir = DEFAULT_SOURCE_DIR
  let outputDir = DEFAULT_OUTPUT_DIR
  let stateFile = DEFAULT_STATE_FILE

  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index]
    const next = argv[index + 1]

    if (arg === '--source-dir') {
      sourceDir = requireOptionValue(arg, next)
      index += 1
      continue
    }

    if (arg === '--output-dir') {
      outputDir = requireOptionValue(arg, next)
      index += 1
      continue
    }

    if (arg === '--state-file') {
      stateFile = requireOptionValue(arg, next)
      index += 1
      continue
    }

    if (isHelpFlag(arg)) {
      printUsage()
      process.exit(0)
    }

    throw new Error(`未知参数: ${arg}`)
  }

  return {
    command,
    projectPath,
    sourceDir,
    outputDir,
    stateFile,
  }
}

function isCommandName(value: string): value is CommandName {
  return value === 'capture-project'
    || value === 'render-project'
    || value === 'sync-project'
}

function requireOptionValue(option: string, value: string | undefined): string {
  if (!value || isHelpFlag(value)) {
    throw new Error(`${option} 需要一个值`)
  }
  return value
}

function isHelpFlag(value: string): boolean {
  return value === '--help' || value === '-h'
}

function printUsage(): void {
  console.log(`用法:
  bun run src/cli.ts capture-project <project_path> [--source-dir svg_output] [--state-file slide_state.json]
  bun run src/cli.ts render-project <project_path> [--output-dir svg_output] [--state-file slide_state.json]
  bun run src/cli.ts sync-project <project_path> [--source-dir svg_output] [--output-dir svg_output] [--state-file slide_state.json]`)
}

function relativeDisplayPath(path: string): string {
  return basename(path) === path ? path : path.replace(`${process.cwd()}/`, '')
}

main().catch(error => {
  console.error(`[ERROR] ${(error as Error).message}`)
  process.exit(1)
})
