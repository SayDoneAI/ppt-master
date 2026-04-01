export interface FontScheme {
  id: string
  name: string
  title: string
  body: string
  caption: string
  label: string
}

export const FONT_SCHEMES: FontScheme[] = [
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
]
