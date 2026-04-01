import { describe, expect, it } from 'vitest'
import { JSDOM } from 'jsdom'
import {
  collectColorRoleAssignments,
  inferColorRoleAssignments,
  wcagContrastRatio,
} from '../preset_engine.js'

const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>')
const parser = new dom.window.DOMParser()

function parseSvg(markup: string): SVGSVGElement {
  const doc = parser.parseFromString(markup, 'image/svg+xml')
  return doc.documentElement as unknown as SVGSVGElement
}

describe('collectColorRoleAssignments', () => {
  it('includes unlabeled nodes alongside explicitly labeled ones', () => {
    const svg = parseSvg(`
      <svg viewBox="0 0 100 100">
        <rect data-element-id="bg" data-color-role="background" width="100" height="100" fill="#FFFFFF"/>
        <text data-element-id="t1" data-color-role="text-dark" fill="#111111" font-size="20">Labeled</text>
        <text data-element-id="t2" fill="#222222" font-size="16">Unlabeled dark</text>
        <text data-element-id="t3" fill="#EEEEEE" font-size="14">Unlabeled light</text>
      </svg>
    `)
    const assignments = collectColorRoleAssignments(svg)
    const ids = new Set(assignments.map(a => a.node.getAttribute('data-element-id')))

    expect(ids.has('bg')).toBe(true)
    expect(ids.has('t1')).toBe(true)
    expect(ids.has('t2')).toBe(true)
    expect(ids.has('t3')).toBe(true)
  })

  it('does not double-assign explicit nodes during inference', () => {
    const svg = parseSvg(`
      <svg viewBox="0 0 100 100">
        <rect data-element-id="bg" data-color-role="background" width="100" height="100" fill="#FFFFFF"/>
        <text data-element-id="t1" data-color-role="primary" fill="#0066FF" font-size="24">Explicit primary</text>
      </svg>
    `)
    const assignments = collectColorRoleAssignments(svg)
    const t1Assignments = assignments.filter(a => a.node.getAttribute('data-element-id') === 't1')

    expect(t1Assignments).toHaveLength(1)
    expect(t1Assignments[0].role).toBe('primary')
  })
})

describe('inferColorRoleAssignments — text coverage', () => {
  it('assigns roles to multiple text buckets with different dark colors', () => {
    const svg = parseSvg(`
      <svg viewBox="0 0 100 100">
        <rect data-element-id="bg" width="100" height="100" fill="#FFFFFF"/>
        <text data-element-id="t1" fill="#111111" font-size="20">Very dark</text>
        <text data-element-id="t2" fill="#333333" font-size="18">Also dark</text>
      </svg>
    `)
    const assignments = inferColorRoleAssignments(svg)
    const textAssignments = assignments.filter(a =>
      a.node.tagName.toLowerCase() === 'text',
    )
    const assignedIds = new Set(textAssignments.map(a => a.node.getAttribute('data-element-id')))

    expect(assignedIds.has('t1')).toBe(true)
    expect(assignedIds.has('t2')).toBe(true)
  })
})

describe('wcagContrastRatio', () => {
  it('黑白对比度接近 21', () => {
    const ratio = wcagContrastRatio('#000000', '#FFFFFF')
    expect(ratio).toBeGreaterThan(20)
    expect(ratio).toBeLessThanOrEqual(21)
  })

  it('午夜模式 primary 与 background 对比度极低', () => {
    // midnight: primary #1E293B vs background #0F172A
    const ratio = wcagContrastRatio('#1E293B', '#0F172A')
    expect(ratio).toBeLessThan(3)
  })

  it('午夜模式 textDark 与 background 对比度充足', () => {
    // midnight: textDark #F1F5F9 vs background #0F172A
    const ratio = wcagContrastRatio('#F1F5F9', '#0F172A')
    expect(ratio).toBeGreaterThan(4.5)
  })
})
