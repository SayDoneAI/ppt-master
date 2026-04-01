import { hasPatchValueChanged } from './design_patch.js'
import type { ColorScheme } from './presets/colors.js'
import type { FontScheme } from './presets/fonts.js'
import type {
  CircleElement,
  Element as SlideElement,
  GroupElement,
  ImageElement,
  LineElement,
  PatchOperation,
  PathElement,
  RectElement,
  Slide,
  SlideState,
  TextElement,
} from './slide_state.js'
import { slideToSvg } from './state_to_svg.js'

export type EditableElement =
  | TextElement
  | RectElement
  | PathElement
  | ImageElement
  | LineElement
  | CircleElement
  | GroupElement

export type ColorRole =
  | 'primary'
  | 'secondary'
  | 'accent'
  | 'text-dark'
  | 'text-light'
  | 'text-muted'
  | 'background'
  | 'background-alt'

export type FontRole = 'title' | 'body' | 'caption' | 'label'

export interface ParsedFontSpec {
  fontSize: number
  fontFamily: string
  fontWeight?: string
  fontStyle?: string
}

export interface PropertyMutation {
  property: string
  oldValue: unknown
  newValue: unknown
}

export interface ColorRoleAssignment {
  node: SVGElement
  role: ColorRole
  targets: Array<'fill' | 'stroke'>
}

export const DEFAULT_COLOR_PICKER_VALUE = '#000000'

const COLOR_ROLE_KEYS: Record<ColorRole, keyof ColorScheme> = {
  primary: 'primary',
  secondary: 'secondary',
  accent: 'accent',
  'text-dark': 'textDark',
  'text-light': 'textLight',
  'text-muted': 'textMuted',
  background: 'background',
  'background-alt': 'backgroundAlt',
}

export type CreatePropertyPatch = (
  elementId: string,
  property: string,
  oldValue: unknown,
  newValue: unknown,
) => PatchOperation | null

export interface PresetContextBase {
  editingTextId: string | null
  exitTextEditing: (options?: { shouldRender?: boolean }) => void
  state: SlideState
  rawSvgStrings: string[]
  findElementById: (elements: SlideElement[], elementId: string | null) => EditableElement | null
  createPropertyPatch: CreatePropertyPatch
  commitPatchOperations: (operations: PatchOperation[], source: 'human' | 'ai', description: string) => void
  setInteractionHint: (value: string) => void
  setStateSourceLabel: (value: string) => void
  render: () => void
}

export interface ApplyColorPresetContext extends PresetContextBase {
  setActiveColorSchemeId: (id: string) => void
}

export interface ApplyFontPresetContext extends PresetContextBase {
  setActiveFontSchemeId: (id: string) => void
}

interface ColorBucket {
  color: string
  entries: Array<{ node: SVGElement; target: 'fill' | 'stroke'; area: number }>
  fillArea: number
  textCount: number
  luminance: number
  saturation: number
}

function getSlideSourceMarkup(slide: Slide, index: number, context: PresetContextBase): string {
  return context.rawSvgStrings[index]?.trim() || slideToSvg(slide, context.state.canvas)
}

export function applyColorPreset(scheme: ColorScheme, context: ApplyColorPresetContext): void {
  const {
    editingTextId,
    exitTextEditing,
    state,
    rawSvgStrings,
    findElementById,
    createPropertyPatch,
    commitPatchOperations,
    setActiveColorSchemeId,
    setInteractionHint,
    setStateSourceLabel,
    render,
  } = context

  if (editingTextId) exitTextEditing({ shouldRender: false })

  const operations: PatchOperation[] = []
  let appliedNodeCount = 0

  state.slides.forEach((slide, index) => {
    const sourceMarkup = getSlideSourceMarkup(slide, index, context)
    const doc = new DOMParser().parseFromString(sourceMarkup, 'image/svg+xml')
    const svg = doc.documentElement as unknown as SVGSVGElement
    if (svg.tagName.toLowerCase() !== 'svg') return

    const assignments = collectColorRoleAssignments(svg)
    assignments.forEach(assignment => {
      const rawColor = scheme[COLOR_ROLE_KEYS[assignment.role]]
      // 对文本节点做对比度保护——颜色和背景对比度不够时自动替换
      const color = ensureTextContrast(assignment.node, rawColor, scheme)
      const result = applyColorToNode(assignment.node, color, assignment.targets)
      if (!result.fill && !result.stroke) return

      appliedNodeCount += 1
      const elementId = assignment.node.getAttribute('data-element-id')
      if (!elementId) return

      const element = findElementById(slide.elements, elementId)
      if (!element) return

      appendPropertyMutations(
        element.id,
        applyColorToElementState(element, {
          fill: result.fill ? color : undefined,
          stroke: result.stroke ? color : undefined,
        }),
        operations,
        createPropertyPatch,
      )
    })

    rawSvgStrings[index] = svg.outerHTML
  })

  setActiveColorSchemeId(scheme.id)
  setInteractionHint(
    appliedNodeCount > 0
      ? `已应用配色方案：${scheme.name}`
      : '当前 SVG 里还没找到可套用的颜色角色。',
  )
  setStateSourceLabel(`当前数据：已应用配色 · ${scheme.name}`)
  if (operations.length > 0) {
    commitPatchOperations(operations, 'human', `应用配色 ${scheme.name}`)
  }
  render()
}

export function applyFontPreset(scheme: FontScheme, context: ApplyFontPresetContext): void {
  const {
    editingTextId,
    exitTextEditing,
    state,
    rawSvgStrings,
    findElementById,
    createPropertyPatch,
    commitPatchOperations,
    setActiveFontSchemeId,
    setInteractionHint,
    setStateSourceLabel,
    render,
  } = context

  if (editingTextId) exitTextEditing({ shouldRender: false })

  const operations: PatchOperation[] = []
  let appliedNodeCount = 0

  state.slides.forEach((slide, index) => {
    const sourceMarkup = getSlideSourceMarkup(slide, index, context)
    const doc = new DOMParser().parseFromString(sourceMarkup, 'image/svg+xml')
    const svg = doc.documentElement as unknown as SVGSVGElement
    if (svg.tagName.toLowerCase() !== 'svg') return

    const assignments = collectFontRoleAssignments(svg)
    assignments.forEach(assignment => {
      const fontFamily = getFontFamilyForRole(scheme, assignment.role)
      assignment.node.setAttribute('font-family', fontFamily)
      appliedNodeCount += 1

      const elementId = assignment.node.getAttribute('data-element-id')
      if (!elementId) return

      const element = findElementById(slide.elements, elementId)
      if (!element) return

      appendPropertyMutations(
        element.id,
        applyFontToElementState(element, fontFamily),
        operations,
        createPropertyPatch,
      )
    })

    rawSvgStrings[index] = svg.outerHTML
  })

  setActiveFontSchemeId(scheme.id)
  setInteractionHint(
    appliedNodeCount > 0
      ? `已应用字体风格：${scheme.name}`
      : '当前 SVG 里还没找到可套用的字体角色。',
  )
  setStateSourceLabel(`当前数据：已应用字体 · ${scheme.name}`)
  if (operations.length > 0) {
    commitPatchOperations(operations, 'human', `应用字体 ${scheme.name}`)
  }
  render()
}

export function collectColorRoleAssignments(svg: SVGSVGElement): ColorRoleAssignment[] {
  const explicitNodes = new Set<SVGElement>()
  const explicit = normalizeColorRoleAssignments(
    Array.from(svg.querySelectorAll<SVGElement>('[data-color-role], [data-color-role-fill], [data-color-role-stroke]')).flatMap(node => {
      const assignments: ColorRoleAssignment[] = []
      const sharedRole = normalizeColorRole(node.getAttribute('data-color-role'))
      const fillRole = normalizeColorRole(node.getAttribute('data-color-role-fill'))
      const strokeRole = normalizeColorRole(node.getAttribute('data-color-role-stroke'))

      if (sharedRole) {
        explicitNodes.add(node)
        assignments.push({
          node,
          role: sharedRole,
          targets: determineColorTargets(node),
        })
      }

      if (fillRole) {
        explicitNodes.add(node)
        assignments.push({
          node,
          role: fillRole,
          targets: ['fill'],
        })
      }

      if (strokeRole) {
        explicitNodes.add(node)
        assignments.push({
          node,
          role: strokeRole,
          targets: ['stroke'],
        })
      }

      return assignments
    }),
  )

  // Always infer roles for unlabeled nodes — even when some nodes have
  // explicit data-color-role attributes.  Previously the function
  // early-returned when *any* explicit label existed, leaving unlabeled
  // text/shapes unchanged when the color scheme switched (invisible text
  // on a new background).
  const inferred = normalizeColorRoleAssignments(
    inferColorRoleAssignments(svg).filter(a => !explicitNodes.has(a.node)),
  )
  persistInferredColorRoles(inferred)
  return [...explicit, ...inferred]
}

export function collectFontRoleAssignments(svg: SVGSVGElement): Array<{ node: SVGElement; role: FontRole }> {
  const explicit = Array.from(svg.querySelectorAll<SVGElement>('[data-font-role]')).flatMap(node => {
    const role = normalizeFontRole(node.getAttribute('data-font-role'))
    return role ? [{ node, role }] : []
  })
  if (explicit.length > 0) return explicit

  return Array.from(svg.querySelectorAll<SVGElement>('[data-element-id]'))
    .filter(node => {
      const tag = node.tagName.toLowerCase()
      return tag === 'text' || tag === 'g'
    })
    .map(node => ({ node, role: inferFontRole(node) }))
}

export function normalizeColorRole(value: string | null): ColorRole | null {
  if (!value) return null
  return Object.prototype.hasOwnProperty.call(COLOR_ROLE_KEYS, value) ? value as ColorRole : null
}

export function normalizeFontRole(value: string | null): FontRole | null {
  if (!value) return null
  if (value === 'title' || value === 'body' || value === 'caption' || value === 'label') {
    return value
  }
  return null
}

export function determineColorTargets(node: SVGElement): Array<'fill' | 'stroke'> {
  const targets: Array<'fill' | 'stroke'> = []
  const tag = node.tagName.toLowerCase()
  const fill = node.getAttribute('fill')
  const stroke = node.getAttribute('stroke')

  if (fill && fill !== 'none' && !isFunctionalPaint(fill)) targets.push('fill')
  if (stroke && stroke !== 'none' && !isFunctionalPaint(stroke)) targets.push('stroke')

  if (targets.length === 0) {
    if (tag === 'line') targets.push('stroke')
    else targets.push('fill')
  }

  return Array.from(new Set(targets))
}

export function inferColorRoleAssignments(svg: SVGSVGElement): ColorRoleAssignment[] {
  const buckets = buildColorBuckets(svg)
  if (buckets.length === 0) return []

  const roleByColor = new Map<string, ColorRole>()
  const usedColors = new Set<string>()

  const fillBuckets = [...buckets].filter(bucket => bucket.fillArea > 0).sort((left, right) => right.fillArea - left.fillArea)
  const textBuckets = [...buckets].filter(bucket => bucket.textCount > 0)
  const vividBuckets = [...buckets].sort((left, right) => {
    if (right.saturation !== left.saturation) return right.saturation - left.saturation
    return (right.fillArea + right.entries.length) - (left.fillArea + left.entries.length)
  })

  assignRoleFromBuckets(roleByColor, usedColors, fillBuckets, 'background')
  assignRoleFromBuckets(roleByColor, usedColors, fillBuckets.filter(bucket => !usedColors.has(bucket.color)), 'background-alt')
  assignRoleFromBuckets(roleByColor, usedColors, textBuckets.sort((left, right) => left.luminance - right.luminance), 'text-dark')
  assignRoleFromBuckets(roleByColor, usedColors, textBuckets.filter(bucket => bucket.luminance >= 0.72).sort((left, right) => right.textCount - left.textCount), 'text-light')
  assignRoleFromBuckets(
    roleByColor,
    usedColors,
    textBuckets.filter(bucket => !usedColors.has(bucket.color)).sort((left, right) => Math.abs(left.luminance - 0.55) - Math.abs(right.luminance - 0.55)),
    'text-muted',
  )
  assignRoleFromBuckets(roleByColor, usedColors, vividBuckets.filter(bucket => !usedColors.has(bucket.color)), 'primary')
  assignRoleFromBuckets(roleByColor, usedColors, vividBuckets.filter(bucket => !usedColors.has(bucket.color)), 'secondary')
  assignRoleFromBuckets(roleByColor, usedColors, vividBuckets.filter(bucket => !usedColors.has(bucket.color)), 'accent')

  // Assign remaining unmatched text colors to the nearest text role by
  // luminance.  Without this, only one color per text-role gets covered
  // and other text stays unchanged — potentially invisible on a new
  // background.
  const textRoleLuminance: Array<[ColorRole, number]> = (
    ['text-dark', 'text-light', 'text-muted'] as ColorRole[]
  ).flatMap(role => {
    const color = [...roleByColor.entries()].find(([, r]) => r === role)?.[0]
    if (!color) return []
    const bucket = buckets.find(b => b.color === color)
    return bucket ? [[role, bucket.luminance] as [ColorRole, number]] : []
  })

  textBuckets.forEach(bucket => {
    if (roleByColor.has(bucket.color)) return
    if (textRoleLuminance.length === 0) return
    const closest = textRoleLuminance.reduce((best, candidate) =>
      Math.abs(candidate[1] - bucket.luminance) < Math.abs(best[1] - bucket.luminance) ? candidate : best,
    )
    roleByColor.set(bucket.color, closest[0])
  })

  return buckets.flatMap(bucket => {
    const role = roleByColor.get(bucket.color)
    if (!role) return []
    return bucket.entries.map(entry => ({
      node: entry.node,
      role,
      targets: [entry.target],
    }))
  })
}

export function normalizeColorRoleAssignments(assignments: ColorRoleAssignment[]): ColorRoleAssignment[] {
  const merged = new Map<SVGElement, Map<ColorRole, Set<'fill' | 'stroke'>>>()

  assignments.forEach(assignment => {
    const nodeRoles = merged.get(assignment.node) ?? new Map<ColorRole, Set<'fill' | 'stroke'>>()
    const targets = nodeRoles.get(assignment.role) ?? new Set<'fill' | 'stroke'>()
    assignment.targets.forEach(target => targets.add(target))
    nodeRoles.set(assignment.role, targets)
    merged.set(assignment.node, nodeRoles)
  })

  return Array.from(merged.entries()).flatMap(([node, roles]) =>
    Array.from(roles.entries()).map(([role, targets]) => ({
      node,
      role,
      targets: Array.from(targets),
    })),
  )
}

export function persistInferredColorRoles(assignments: ColorRoleAssignment[]): void {
  const roleByNode = new Map<SVGElement, { fill?: ColorRole; stroke?: ColorRole }>()

  assignments.forEach(assignment => {
    const roles = roleByNode.get(assignment.node) ?? {}
    if (assignment.targets.includes('fill')) roles.fill = assignment.role
    if (assignment.targets.includes('stroke')) roles.stroke = assignment.role
    roleByNode.set(assignment.node, roles)
  })

  roleByNode.forEach((roles, node) => {
    const fillRole = roles.fill
    const strokeRole = roles.stroke

    if ((fillRole && !strokeRole) || (strokeRole && !fillRole) || (fillRole && strokeRole && fillRole === strokeRole)) {
      node.setAttribute('data-color-role', fillRole ?? strokeRole ?? '')
      node.removeAttribute('data-color-role-fill')
      node.removeAttribute('data-color-role-stroke')
      return
    }

    node.removeAttribute('data-color-role')
    if (fillRole) node.setAttribute('data-color-role-fill', fillRole)
    else node.removeAttribute('data-color-role-fill')
    if (strokeRole) node.setAttribute('data-color-role-stroke', strokeRole)
    else node.removeAttribute('data-color-role-stroke')
  })
}

export function assignRoleFromBuckets(
  roleByColor: Map<string, ColorRole>,
  usedColors: Set<string>,
  buckets: Array<{ color: string }>,
  role: ColorRole,
): void {
  const nextBucket = buckets.find(bucket => !usedColors.has(bucket.color))
  if (!nextBucket) return
  roleByColor.set(nextBucket.color, role)
  usedColors.add(nextBucket.color)
}

export function buildColorBuckets(svg: SVGSVGElement): ColorBucket[] {
  const buckets = new Map<string, ColorBucket>()

  Array.from(svg.querySelectorAll<SVGElement>('[data-element-id]')).forEach(node => {
    ;(['fill', 'stroke'] as const).forEach(target => {
      const rawColor = node.getAttribute(target)
      const color = normalizeHexColor(rawColor)
      if (!color) return

      const bucket = buckets.get(color) ?? {
        color,
        entries: [],
        fillArea: 0,
        textCount: 0,
        luminance: getColorLuminance(color),
        saturation: getColorSaturation(color),
      }

      const area = estimateNodeArea(node)
      bucket.entries.push({ node, target, area })
      if (target === 'fill') bucket.fillArea += area
      if (node.tagName.toLowerCase() === 'text') bucket.textCount += 1
      buckets.set(color, bucket)
    })
  })

  return Array.from(buckets.values())
}

export function applyColorToNode(
  node: SVGElement,
  color: string,
  targets: Array<'fill' | 'stroke'>,
): { fill: boolean; stroke: boolean } {
  let fillChanged = false
  let strokeChanged = false

  if (targets.includes('fill')) {
    node.setAttribute('fill', color)
    fillChanged = true
  }

  if (targets.includes('stroke')) {
    node.setAttribute('stroke', color)
    strokeChanged = true
  }

  return { fill: fillChanged, stroke: strokeChanged }
}

export function appendPropertyMutations(
  elementId: string,
  mutations: PropertyMutation[],
  operations: PatchOperation[],
  createPropertyPatch: CreatePropertyPatch,
): void {
  mutations.forEach(mutation => {
    const operation = createPropertyPatch(elementId, mutation.property, mutation.oldValue, mutation.newValue)
    if (operation) operations.push(operation)
  })
}

export function applyColorToElementState(
  element: EditableElement,
  next: { fill?: string; stroke?: string },
): PropertyMutation[] {
  const mutations: PropertyMutation[] = []

  if (next.fill !== undefined) {
    switch (element.type) {
      case 'text':
      case 'rect':
      case 'path':
      case 'circle':
        pushMutation(mutations, createPropertyMutation(element, 'fill', next.fill))
        break
      case 'group':
        pushMutation(mutations, createPropertyMutation(element, 'fill', next.fill))
        break
    }
  }

  if (next.stroke !== undefined) {
    switch (element.type) {
      case 'rect':
      case 'path':
      case 'line':
      case 'circle':
        pushMutation(mutations, createPropertyMutation(element, 'stroke', next.stroke))
        break
    }
  }

  return mutations
}

export function applyFontToElementState(element: EditableElement, fontFamily: string): PropertyMutation[] {
  switch (element.type) {
    case 'text':
      return assignTextFontFamily(element, fontFamily)
    case 'group':
      return toMutationList(createPropertyMutation(element, 'fontFamily', fontFamily)) ?? []
    default:
      return []
  }
}

export function pushMutation(target: PropertyMutation[], mutation: PropertyMutation | null): void {
  if (mutation) target.push(mutation)
}

export function getFontFamilyForRole(scheme: FontScheme, role: FontRole): string {
  switch (role) {
    case 'title':
      return scheme.title
    case 'caption':
      return scheme.caption
    case 'label':
      return scheme.label
    case 'body':
    default:
      return scheme.body
  }
}

export function inferFontRole(node: SVGElement): FontRole {
  const fontSize = parseFloat(node.getAttribute('font-size') || '0')
  const textLength = node.textContent?.trim().length ?? 0

  if (fontSize >= 30) return 'title'
  if (fontSize <= 14) return 'caption'
  if (fontSize >= 18 && textLength <= 18) return 'label'
  return 'body'
}

export function assignTextFontFamily(element: TextElement, fontFamily: string): PropertyMutation[] {
  const parsed = parseFontSpec(element.font)
  return assignTextFontSpec(element, {
    fontSize: parsed.fontSize,
    fontFamily,
    fontWeight: normalizeFontWeightValue(parsed.fontWeight),
    fontStyle: parsed.fontStyle,
  }) ?? []
}

export function assignTextFontSpec(
  element: TextElement,
  spec: {
    fontSize: number
    fontFamily: string
    fontWeight?: string
    fontStyle?: string
    fontString?: string
  },
): PropertyMutation[] | null {
  const normalizedWeight = normalizeFontWeightValue(spec.fontWeight)
  const fontString = spec.fontString ?? serializeFontSpec({
    fontSize: spec.fontSize,
    fontFamily: spec.fontFamily,
    fontWeight: normalizedWeight === '400' ? undefined : normalizedWeight,
    fontStyle: spec.fontStyle,
  })

  const mutations = [
    createPropertyMutation(element, 'font', fontString),
    createPropertyMutation(element, 'fontSize', spec.fontSize),
    createPropertyMutation(element, 'fontFamily', spec.fontFamily),
    createPropertyMutation(element, 'fontWeight', normalizedWeight),
  ].filter((mutation): mutation is PropertyMutation => Boolean(mutation))

  return mutations.length > 0 ? mutations : null
}

export function estimateNodeArea(node: SVGElement): number {
  const tag = node.tagName.toLowerCase()
  if (tag === 'rect' || tag === 'image') {
    return (parseFloat(node.getAttribute('width') || '0') || 0) * (parseFloat(node.getAttribute('height') || '0') || 0)
  }

  if (tag === 'circle') {
    const r = parseFloat(node.getAttribute('r') || '0') || 0
    return Math.PI * r * r
  }

  if (tag === 'line') {
    const x1 = parseFloat(node.getAttribute('x1') || '0') || 0
    const y1 = parseFloat(node.getAttribute('y1') || '0') || 0
    const x2 = parseFloat(node.getAttribute('x2') || '0') || 0
    const y2 = parseFloat(node.getAttribute('y2') || '0') || 0
    const strokeWidth = parseFloat(node.getAttribute('stroke-width') || '1') || 1
    return Math.hypot(x2 - x1, y2 - y1) * strokeWidth
  }

  return 1
}

export function isFunctionalPaint(value: string): boolean {
  return value.trim().startsWith('url(')
}

export function normalizeHexColor(value: string | null): string | null {
  if (!value) return null
  const trimmed = value.trim()
  if (trimmed === '' || trimmed === 'none' || isFunctionalPaint(trimmed)) return null

  const shortHex = trimmed.match(/^#([0-9a-f]{3})$/i)
  if (shortHex) {
    const [r, g, b] = shortHex[1].split('')
    return `#${r}${r}${g}${g}${b}${b}`.toUpperCase()
  }

  const fullHex = trimmed.match(/^#([0-9a-f]{6})$/i)
  if (fullHex) return `#${fullHex[1].toUpperCase()}`
  return null
}

export function getColorLuminance(color: string): number {
  const { r, g, b } = hexToRgb(color)
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
}

/**
 * WCAG relative luminance using linearized sRGB.
 * https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
 */
function srgbRelativeLuminance(hex: string): number {
  const { r, g, b } = hexToRgb(hex)
  const lin = (c: number) => {
    const s = c / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

/**
 * WCAG 2.1 contrast ratio between two colors (range 1–21).
 */
export function wcagContrastRatio(a: string, b: string): number {
  const la = srgbRelativeLuminance(a)
  const lb = srgbRelativeLuminance(b)
  const lighter = Math.max(la, lb)
  const darker = Math.min(la, lb)
  return (lighter + 0.05) / (darker + 0.05)
}

const MIN_TEXT_CONTRAST = 3.0

/**
 * For text nodes, ensure the assigned color has enough contrast against the
 * scheme background. If not, substitute with whichever of textDark / textLight
 * gives better contrast.
 */
function ensureTextContrast(
  node: SVGElement,
  color: string,
  scheme: ColorScheme,
): string {
  const tag = node.tagName.toLowerCase()
  if (tag !== 'text' && tag !== 'tspan') return color

  const ratio = wcagContrastRatio(color, scheme.background)
  if (ratio >= MIN_TEXT_CONTRAST) return color

  const darkRatio = wcagContrastRatio(scheme.textDark, scheme.background)
  const lightRatio = wcagContrastRatio(scheme.textLight, scheme.background)
  return darkRatio >= lightRatio ? scheme.textDark : scheme.textLight
}

export function getColorSaturation(color: string): number {
  const { r, g, b } = hexToRgb(color)
  const max = Math.max(r, g, b) / 255
  const min = Math.min(r, g, b) / 255
  if (max === min) return 0
  const lightness = (max + min) / 2
  return lightness > 0.5
    ? (max - min) / (2 - max - min)
    : (max - min) / (max + min)
}

export function hexToRgb(color: string): { r: number; g: number; b: number } {
  const normalized = normalizeHexColor(color) ?? DEFAULT_COLOR_PICKER_VALUE
  return {
    r: parseInt(normalized.slice(1, 3), 16),
    g: parseInt(normalized.slice(3, 5), 16),
    b: parseInt(normalized.slice(5, 7), 16),
  }
}

export function cloneSerializableValue<T>(value: T): T {
  if (value === undefined) return value
  return JSON.parse(JSON.stringify(value)) as T
}

export function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

export function normalizeFontWeightValue(value: string | number | undefined): string {
  if (value === undefined || value === null || value === '' || value === 'normal') return '400'
  if (value === 'bold') return '700'
  if (value === 'medium') return '500'

  const numeric = typeof value === 'number' ? value : Number.parseInt(String(value), 10)
  if (!Number.isFinite(numeric)) return '400'
  if (numeric >= 600) return '700'
  if (numeric >= 450) return '500'
  return '400'
}

export function serializeFontSpec(spec: ParsedFontSpec): string {
  const parts: string[] = []
  if (spec.fontStyle) parts.push(spec.fontStyle)
  if (spec.fontWeight && normalizeFontWeightValue(spec.fontWeight) !== '400') {
    parts.push(normalizeFontWeightValue(spec.fontWeight))
  }
  parts.push(`${formatNumber(spec.fontSize)}px`)
  parts.push(spec.fontFamily)
  return parts.join(' ')
}

export function parseFontSpec(font: string): ParsedFontSpec {
  const match = font.match(/(?:(italic)\s+)?(?:(bold|[1-9]00)\s+)?(\d+(?:\.\d+)?)px\s+(.+)/i)
  if (match) {
    return {
      fontStyle: match[1] || undefined,
      fontWeight: match[2] || undefined,
      fontSize: Number(match[3]),
      fontFamily: match[4],
    }
  }

  return {
    fontSize: 16,
    fontFamily: 'Inter, PingFang SC, sans-serif',
  }
}

export function createPropertyMutation<T extends object, K extends keyof T>(
  target: T,
  key: K,
  nextValue: T[K],
): PropertyMutation | null {
  const oldValue = cloneSerializableValue(target[key] as unknown)
  const normalizedNextValue = cloneSerializableValue(nextValue as unknown)
  if (!hasPatchValueChanged(oldValue, normalizedNextValue)) return null
  target[key] = nextValue
  return {
    property: String(key),
    oldValue,
    newValue: normalizedNextValue,
  }
}

export function toMutationList(mutation: PropertyMutation | null): PropertyMutation[] | null {
  return mutation ? [mutation] : null
}
