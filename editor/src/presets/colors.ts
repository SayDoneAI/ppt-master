export interface ColorScheme {
  id: string
  name: string
  category: 'universal' | 'mood' | 'industry'
  primary: string
  secondary: string
  accent: string
  textDark: string
  textLight: string
  textMuted: string
  background: string
  backgroundAlt: string
}

const DEFAULT_TEXT_DARK = '#1A1A2E'
const DEFAULT_TEXT_LIGHT = '#FFFFFF'
const DEFAULT_TEXT_MUTED = '#6B7280'
const DEFAULT_BACKGROUND = '#FFFFFF'
const DEFAULT_BACKGROUND_ALT = '#F5F5F5'

function withDefaults(
  scheme: Pick<ColorScheme, 'id' | 'name' | 'category' | 'primary' | 'secondary' | 'accent'> &
    Partial<Pick<ColorScheme, 'textDark' | 'textLight' | 'textMuted' | 'background' | 'backgroundAlt'>>,
): ColorScheme {
  return {
    textDark: DEFAULT_TEXT_DARK,
    textLight: DEFAULT_TEXT_LIGHT,
    textMuted: DEFAULT_TEXT_MUTED,
    background: DEFAULT_BACKGROUND,
    backgroundAlt: DEFAULT_BACKGROUND_ALT,
    ...scheme,
  }
}

export const COLOR_SCHEMES: ColorScheme[] = [
  // --- Universal (经典百搭) ---
  withDefaults({
    id: 'classic-blue',
    name: '经典蓝',
    category: 'universal',
    primary: '#0D6EFD',
    secondary: '#6610F2',
    accent: '#FFC107',
  }),
  withDefaults({
    id: 'ocean',
    name: '深海',
    category: 'universal',
    primary: '#0077B6',
    secondary: '#00B4D8',
    accent: '#FF6B35',
  }),
  withDefaults({
    id: 'forest',
    name: '森林',
    category: 'universal',
    primary: '#2D6A4F',
    secondary: '#52B788',
    accent: '#E76F51',
  }),
  withDefaults({
    id: 'slate',
    name: '石墨',
    category: 'universal',
    primary: '#334155',
    secondary: '#64748B',
    accent: '#3B82F6',
  }),
  withDefaults({
    id: 'midnight',
    name: '午夜',
    category: 'universal',
    primary: '#1E293B',
    secondary: '#475569',
    accent: '#22D3EE',
    background: '#0F172A',
    backgroundAlt: '#1E293B',
    textDark: '#F1F5F9',
    textMuted: '#94A3B8',
  }),

  // --- Mood (氛围感) ---
  withDefaults({
    id: 'warm-earth',
    name: '暖棕',
    category: 'mood',
    primary: '#92400E',
    secondary: '#B45309',
    accent: '#DC2626',
  }),
  withDefaults({
    id: 'rose-gold',
    name: '玫瑰金',
    category: 'mood',
    primary: '#9F1239',
    secondary: '#E11D48',
    accent: '#F59E0B',
  }),
  withDefaults({
    id: 'lavender',
    name: '薰衣草',
    category: 'mood',
    primary: '#7C3AED',
    secondary: '#A78BFA',
    accent: '#F472B6',
    backgroundAlt: '#F5F3FF',
  }),
  withDefaults({
    id: 'sunset',
    name: '日落',
    category: 'mood',
    primary: '#EA580C',
    secondary: '#F97316',
    accent: '#7C3AED',
  }),
  withDefaults({
    id: 'mint',
    name: '薄荷',
    category: 'mood',
    primary: '#0D9488',
    secondary: '#14B8A6',
    accent: '#F59E0B',
    backgroundAlt: '#F0FDFA',
  }),
  withDefaults({
    id: 'sakura',
    name: '樱花',
    category: 'mood',
    primary: '#DB2777',
    secondary: '#EC4899',
    accent: '#8B5CF6',
    backgroundAlt: '#FDF2F8',
  }),

  // --- Industry (行业) ---
  withDefaults({
    id: 'tech-neon',
    name: '科技',
    category: 'industry',
    primary: '#2563EB',
    secondary: '#7C3AED',
    accent: '#06B6D4',
  }),
  withDefaults({
    id: 'finance',
    name: '金融',
    category: 'industry',
    primary: '#003366',
    secondary: '#1E40AF',
    accent: '#D4AF37',
  }),
  withDefaults({
    id: 'medical',
    name: '医疗',
    category: 'industry',
    primary: '#0F766E',
    secondary: '#0891B2',
    accent: '#F97316',
    backgroundAlt: '#F0FDFA',
  }),
  withDefaults({
    id: 'education',
    name: '教育',
    category: 'industry',
    primary: '#4338CA',
    secondary: '#7C3AED',
    accent: '#F59E0B',
    backgroundAlt: '#EEF2FF',
  }),
  withDefaults({
    id: 'creative',
    name: '创意',
    category: 'industry',
    primary: '#C026D3',
    secondary: '#A855F7',
    accent: '#14B8A6',
  }),
]
