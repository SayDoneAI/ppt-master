// project_pipeline.ts — SVG-first 项目文件与 compat state 之间的最小桥接辅助

import type { Slide, SlideState } from './slide_state.js'
import { stateToSvgs } from './state_to_svg.js'
import { svgsToState } from './svg_to_state.js'

export interface ProjectSvgInput {
  fileName: string
  content: string
}

export interface ProjectSvgArtifact {
  fileName: string
  content: string
}

export function compareProjectFileName(
  a: Pick<ProjectSvgInput, 'fileName'>,
  b: Pick<ProjectSvgInput, 'fileName'>,
): number {
  return a.fileName.localeCompare(b.fileName, undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

export function buildProjectStateFromSvgInputs(
  inputs: ProjectSvgInput[],
  parser?: DOMParser,
): SlideState {
  const svgInputs = [...inputs]
    .filter(input => isSvgFileName(input.fileName))
    .sort(compareProjectFileName)

  if (svgInputs.length === 0) {
    throw new Error('未检测到可导入的 SVG 文件')
  }

  return svgsToState(
    svgInputs.map(input => input.content),
    svgInputs.map(input => toSlideId(input.fileName)),
    parser,
  )
}

export function createProjectSvgArtifacts(state: SlideState): ProjectSvgArtifact[] {
  const usedFileNames = new Set<string>()
  const svgs = stateToSvgs(state)

  return state.slides.map((slide, index) => {
    const fileName = createProjectSvgFileName(slide, index, state.slides.length, usedFileNames)
    return {
      fileName,
      content: svgs[index],
    }
  })
}

function createProjectSvgFileName(
  slide: Slide,
  index: number,
  totalSlides: number,
  usedFileNames: Set<string>,
): string {
  const fallbackName = `slide_${String(index + 1).padStart(Math.max(2, String(totalSlides).length), '0')}`
  const baseName = sanitizeProjectFileStem(slide.id) || fallbackName

  let candidate = baseName
  let duplicateIndex = 2
  while (usedFileNames.has(`${candidate}.svg`)) {
    candidate = `${baseName}_${String(duplicateIndex).padStart(2, '0')}`
    duplicateIndex += 1
  }

  const fileName = `${candidate}.svg`
  usedFileNames.add(fileName)
  return fileName
}

function sanitizeProjectFileStem(value: string): string {
  return value
    .replace(/\.[^.]+$/, '')
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function isSvgFileName(fileName: string): boolean {
  return fileName.toLowerCase().endsWith('.svg')
}

function toSlideId(fileName: string): string {
  return sanitizeProjectFileStem(fileName)
}
