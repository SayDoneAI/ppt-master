import type { Slide, SlideState } from './slide_state.js'
import { normalizeSvgForEditor, svgToSlide } from './svg_to_state.js'

export interface SyncCurrentSlideFromLiveSvgInput {
  canvasSvgMarkup: string
  currentSlideIndex: number
  state: SlideState
  rawSvgStrings: string[]
  parser?: DOMParser
}

export interface SyncCurrentSlideFromLiveSvgResult {
  rawSvg: string
  slide: Slide
  state: SlideState
  rawSvgStrings: string[]
}

export function syncCurrentSlideFromLiveSvg(
  input: SyncCurrentSlideFromLiveSvgInput,
): SyncCurrentSlideFromLiveSvgResult | null {
  const currentSlide = input.state.slides[input.currentSlideIndex]
  const currentRawSvg = input.rawSvgStrings[input.currentSlideIndex]?.trim()
  if (!currentSlide || !currentRawSvg) return null

  const normalizedSvg = normalizeSvgForEditor(
    stripEditorRenderScope(input.canvasSvgMarkup, `canvas-${input.currentSlideIndex}`, input.parser),
    { idPrefix: currentSlide.id },
    input.parser,
  )
  const nextSlide = svgToSlide(normalizedSvg, currentSlide.id, input.parser, {
    preserveTextNodes: true,
  })
  const nextRawSvgStrings = [...input.rawSvgStrings]
  nextRawSvgStrings[input.currentSlideIndex] = normalizedSvg

  const nextSlides = [...input.state.slides]
  nextSlides[input.currentSlideIndex] = nextSlide

  return {
    rawSvg: normalizedSvg,
    slide: nextSlide,
    state: {
      ...input.state,
      slides: nextSlides,
    },
    rawSvgStrings: nextRawSvgStrings,
  }
}

export function stripEditorRenderScope(svgMarkup: string, scope: string, parser?: DOMParser): string {
  const p = parser ?? new DOMParser()
  const doc = p.parseFromString(svgMarkup, 'image/svg+xml')
  const svg = doc.documentElement as unknown as SVGSVGElement
  if (svg.tagName.toLowerCase() !== 'svg') return svgMarkup

  const prefix = `${scope}-`
  for (const node of [svg, ...Array.from(svg.querySelectorAll<SVGElement>('*'))]) {
    const id = node.getAttribute('id')
    if (id?.startsWith(prefix)) {
      node.setAttribute('id', id.slice(prefix.length))
    }

    for (const attribute of Array.from(node.attributes)) {
      const { name, value } = attribute
      if (!value) continue

      if ((name === 'href' || name === 'xlink:href') && value.startsWith(`#${prefix}`)) {
        node.setAttribute(name, `#${value.slice(prefix.length + 1)}`)
        continue
      }

      if (value.includes(`url(#${prefix}`)) {
        node.setAttribute(name, value.replaceAll(`url(#${prefix}`, 'url(#'))
      }
    }
  }

  return svg.outerHTML
}
