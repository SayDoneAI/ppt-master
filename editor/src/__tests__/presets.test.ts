import { describe, expect, it } from 'vitest'
import { COLOR_SCHEMES } from '../presets/colors.js'
import { FONT_SCHEMES } from '../presets/fonts.js'

describe('preset catalogs', () => {
  it('exports 16 color schemes with unique ids', () => {
    expect(COLOR_SCHEMES).toHaveLength(16)
    expect(new Set(COLOR_SCHEMES.map(scheme => scheme.id)).size).toBe(16)
  })

  it('matches the current 16-scheme color spec', () => {
    expect(COLOR_SCHEMES.filter(scheme => scheme.category === 'universal').map(scheme => scheme.id)).toEqual([
      'classic-blue',
      'ocean',
      'forest',
      'slate',
      'midnight',
    ])
    expect(COLOR_SCHEMES.filter(scheme => scheme.category === 'mood').map(scheme => scheme.id)).toEqual([
      'warm-earth',
      'rose-gold',
      'lavender',
      'sunset',
      'mint',
      'sakura',
    ])
    expect(COLOR_SCHEMES.filter(scheme => scheme.category === 'industry').map(scheme => scheme.id)).toEqual([
      'tech-neon',
      'finance',
      'medical',
      'education',
      'creative',
    ])

    expect(COLOR_SCHEMES.find(scheme => scheme.id === 'classic-blue')).toMatchObject({
      name: '经典蓝',
      category: 'universal',
      primary: '#0D6EFD',
      secondary: '#6610F2',
      accent: '#FFC107',
    })
    expect(COLOR_SCHEMES.find(scheme => scheme.id === 'midnight')).toMatchObject({
      name: '午夜',
      category: 'universal',
      primary: '#1E293B',
      secondary: '#475569',
      accent: '#22D3EE',
      background: '#0F172A',
      backgroundAlt: '#1E293B',
      textDark: '#F1F5F9',
      textMuted: '#94A3B8',
    })
    expect(COLOR_SCHEMES.find(scheme => scheme.id === 'tech-neon')).toMatchObject({
      name: '科技',
      category: 'industry',
      primary: '#2563EB',
      secondary: '#7C3AED',
      accent: '#06B6D4',
    })
    expect(COLOR_SCHEMES.find(scheme => scheme.id === 'creative')).toMatchObject({
      name: '创意',
      category: 'industry',
      primary: '#C026D3',
      secondary: '#A855F7',
      accent: '#14B8A6',
    })

    expect(COLOR_SCHEMES.every(scheme => (
      typeof scheme.textLight === 'string'
      && typeof scheme.background === 'string'
      && typeof scheme.backgroundAlt === 'string'
    ))).toBe(true)
  })

  it('exports the exact 6 free commercial font schemes from the new spec', () => {
    expect(FONT_SCHEMES).toHaveLength(6)
    expect(FONT_SCHEMES).toEqual([
      {
        id: 'noto-sans',
        name: 'Noto Sans',
        title: 'Noto Sans SC, Noto Sans, sans-serif',
        body: 'Noto Sans SC, Noto Sans, sans-serif',
        caption: 'Noto Sans SC, Noto Sans, sans-serif',
        label: 'Noto Sans SC, Noto Sans, sans-serif',
      },
      {
        id: 'source-han',
        name: '思源黑体',
        title: 'Source Han Sans SC, sans-serif',
        body: 'Source Han Sans SC, sans-serif',
        caption: 'Source Han Sans SC, sans-serif',
        label: 'Source Han Sans SC, sans-serif',
      },
      {
        id: 'source-han-serif',
        name: '思源宋体',
        title: 'Source Han Serif SC, serif',
        body: 'Source Han Serif SC, serif',
        caption: 'Source Han Serif SC, serif',
        label: 'Source Han Serif SC, serif',
      },
      {
        id: 'alibaba-puhuiti',
        name: '阿里巴巴普惠体',
        title: 'Alibaba PuHuiTi, sans-serif',
        body: 'Alibaba PuHuiTi, sans-serif',
        caption: 'Alibaba PuHuiTi, sans-serif',
        label: 'Alibaba PuHuiTi, sans-serif',
      },
      {
        id: 'oppo-sans',
        name: 'OPPO Sans',
        title: 'OPPO Sans, sans-serif',
        body: 'OPPO Sans, sans-serif',
        caption: 'OPPO Sans, sans-serif',
        label: 'OPPO Sans, sans-serif',
      },
      {
        id: 'harmonyos-sans',
        name: 'HarmonyOS Sans',
        title: 'HarmonyOS Sans SC, sans-serif',
        body: 'HarmonyOS Sans SC, sans-serif',
        caption: 'HarmonyOS Sans SC, sans-serif',
        label: 'HarmonyOS Sans SC, sans-serif',
      },
    ])
  })
})
