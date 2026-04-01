# Findings & Decisions

## Project: Pretext 驱动的 AI + 人协同编辑工作台

### Problem Statement
- ppt-master 的 AI 生成能力已经很强（多角色、多格式、33 种图表、640+ 图标），但 AI 和人之间缺少共同操作空间。
- 当前交互：人（自然语言描述）→ AI（盲写 SVG）→ 人（找文件、打开预览、再描述）。每次微调成本接近重新生成。
- 文本断行靠 AI 预估 `<tspan>` 坐标，不是真正的排版系统。"改标题宽一点、换两行"这类调整高度依赖自然语言描述。
- AI 引擎 70 分，但 AI+人的交互界面 20 分。改造重点不是让 AI 更强，而是给 AI 和人之间搭一个"共同看得见、摸得着"的操作台。

### Pretext 库评估（github.com/chenglou/pretext）

| 属性 | 值 |
|------|-----|
| Stars | 20,257 |
| 语言 | TypeScript (89.8%) |
| 大小 | 15KB，零依赖 |
| 许可 | MIT |
| 创建 | 2026-03-07 |
| 最近推送 | 2026-03-30 |
| 作者 | Cheng Lou（React/ReasonML/Midjourney） |

**核心能力：**
- `prepare(text, font)` — 一次性文本分析 + 测量（~19ms/500条）
- `layout(prepared, maxWidth, lineHeight)` — 纯算术计算高度和行数（~0.09ms/500条）
- `layoutWithLines(prepared, maxWidth, lineHeight)` — 返回每行文字和宽度
- `layoutNextLine(prepared, cursor, maxWidth)` — 逐行变宽排版（绕流）
- `walkLineRanges(prepared, maxWidth, onLine)` — 低开销遍历行宽（shrinkwrap）

**支持渲染目标：** DOM、Canvas、SVG、WebGL、(即将) 服务端

**约束：**
- `system-ui` 在 macOS 下精度不安全，必须用命名字体
- 默认 `white-space: normal` + `overflow-wrap: break-word`
- 支持 `pre-wrap` 模式保留空格/tab/换行
- 支持所有语言、emoji、混合 bidi

### Pretext 在 ppt-master 的三个落地点

| 落地点 | 当前问题 | Pretext 解决方案 |
|--------|---------|-----------------|
| AI 生成阶段 | AI 盲写 `<tspan>` 坐标，断行靠猜 | AI 输出文字+字体+框宽，`layoutWithLines()` 精确计算每行文字和 y 坐标 |
| 编辑器交互 | 不存在编辑器 | 用户拖文本框边框、改文字，Pretext 实时重排，60fps 零 DOM reflow |
| 质量检查 | 溢出靠人眼看 | `layout()` 返回 height，自动检测文本是否超出框高 |

### Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| `slide_state.json` 作为唯一真相源 | AI 和编辑器都读写同一个结构化数据，消除信息不对称；AI 不再直接写 SVG |
| Pretext 作为文本排版内核 | 精确断行+高度计算+溢出检测，替代 AI 概率性猜测 |
| SVG 作为渲染层（非 Canvas） | 浏览器原生支持 SVG 交互事件；生成的 SVG 直接进入现有 finalize → PPTX 链路 |
| 现有后处理链路不动 | `finalize_svg.py` + `svg_to_pptx.py` 保持不变，只改"SVG 怎么来的" |
| 架构一步到位 | 从第一天就建立 slide_state + Pretext + 编辑器，不做渐进式 hack |
| 命名字体优先 | Pretext 在 macOS 下 `system-ui` 精度不安全；ppt-master 当前字体系统已有 `sans_serif` / `monospace` 两套命名方案 |

### slide_state Schema 设计

```json
{
  "canvas": { "width": 1280, "height": 720 },
  "slides": [{
    "id": "slide_01",
    "elements": [
      {
        "type": "text",
        "id": "title_01",
        "x": 60, "y": 80, "width": 600,
        "text": "AI 时代的内容创作",
        "font": "bold 36px PingFang SC",
        "lineHeight": 48,
        "fill": "#1d1d1f"
      },
      {
        "type": "rect",
        "id": "bg_card",
        "x": 60, "y": 160, "width": 500, "height": 300,
        "fill": "#f5f5f7", "rx": 12
      },
      {
        "type": "image",
        "id": "hero_img",
        "x": 700, "y": 80, "width": 520, "height": 560,
        "href": "images/hero.jpg"
      }
    ]
  }]
}
```

### Pretext 驱动的文本 → SVG 转换

```typescript
import { prepareWithSegments, layoutWithLines } from '@chenglou/pretext'

function textElementToSvg(el: TextElement): string {
  const prepared = prepareWithSegments(el.text, el.font)
  const { lines, height } = layoutWithLines(prepared, el.width, el.lineHeight)

  // 溢出检测
  if (el.maxHeight && height > el.maxHeight) {
    warn(`文本溢出: ${el.id}, 需要 ${height}px, 只有 ${el.maxHeight}px`)
  }

  let svg = `<text fill="${el.fill}" font-family="..." font-size="...">`
  for (let i = 0; i < lines.length; i++) {
    svg += `<tspan x="${el.x}" y="${el.y + el.lineHeight * (i + 1)}">${escapeXml(lines[i].text)}</tspan>`
  }
  return svg + '</text>'
}
```

### AI + 人的交互循环（目标体验）

```
用户: "做一个关于 AI 的 PPT"
  ↓ AI 生成 slide_state.json
编辑器自动打开，展示渲染结果（WebSocket 推送）
  ↓
用户: [点击标题] → 改文字    ← 简单的人自己改
      [拖动元素] → 调位置
      属性面板改颜色/字号
  ↓ 自动保存到 slide_state
用户: "第 3 页加一个柱状图"  ← 复杂的交给 AI
  ↓ AI 读取当前 slide_state（知道布局现状）→ 添加图表 → 编辑器刷新
```

### 与现有系统的兼容性

| 现有组件 | 是否改动 | 说明 |
|---------|---------|------|
| `finalize_svg.py` | 不改 | 仍然处理生成的 SVG |
| `svg_to_pptx.py` | 不改 | 仍然从 SVG 导出 PPTX |
| `templates/` | 后期转换 | 模板可转为 slide_state 格式 |
| `tools/*.py` | 不改 | 后处理工具链不变 |
| AI 角色系统 | 改 | Executor 输出 slide_state 而非直接写 SVG |
| `viewer.html` | 替换 | 升级为编辑器 |

### Risks
- Pretext 需要字体已加载才能准确测量 → 编辑器需预加载字体
- 现有 SVG 资产反向解析为 slide_state 可能有信息损失 → 需要渐进迁移策略
- 编辑器工程量不小 → 先做单页 MVP，验证核心交互后再扩展

### AI 入口决策（2026-03-31）
- 编辑器内的 AI 入口不走浏览器直连模型 API。
- 原因：
  - 前端直连 API 会暴露密钥与供应商耦合
  - 当前仓库本来就是本地 agent / skill 工作流，最自然的边界是文件 handoff
  - `slide_state.json` + `design_patch.json` 已经是天然的上下文载体，不需要另建服务
- 最终方案：
  - 浏览器编辑器只负责导出本地 AI handoff 文件
  - Claude Code / Codex skill 负责读取 handoff、修改 `slide_state.json`、执行 `render`
  - 后续 `finalize_svg.py` / `svg_to_pptx.py` 链路保持不变
- 当前 handoff 产物：
  - `design_patch.ai-request.json`：机器可读 handoff JSON（含 `aiCommand`、最近 patch），是本地 skill / command 的正式输入
  - `design_patch.ai-handoff.md`：给 Claude Code / Codex 本地 skill / command 的执行说明

### AI + 人交互主逻辑收敛（2026-03-31）
- 正确的默认交互不是“拖一个 patch/json 给编辑器”，也不是“先挑模板再说”，而是单线程回合式协作：
  - 用户说想做什么页面或海报
  - 编辑器绑定项目内 `slide_state.json`
  - AI 修改同一份 state，预览自动刷新
  - 人直接拖拽 / 双击文本 / 改属性做快调
  - 再把更大的改动继续丢回给 AI
- 因此 UI 层做了三条明确收敛：
  - `AI 回合` 提升到右侧第一卡片，首屏直接可见
  - `绑定项目预览` + `导出 AI 请求` 作为主按钮，直接服务“自动刷新”的闭环
  - `Patch 回流 / 模板图表导入 / 拖拽兼容` 全部收进折叠式高级入口
- 这个决策的意义：
  - AI 和人围绕同一个真相源反复接力，不再频繁搬运文件
  - “看预览 -> 改一点 -> 立即看结果”变成默认体验，而不是手动兜底
  - 模板库和兼容拖拽仍保留，但不再误导用户把次级路径当主流程

### 海报预览适配（2026-03-31）
- 之前编辑器画布只按宽度限制 `.canvas-stage`，导致竖版海报会超出可视高度，看不完整页。
- 现在改成基于 `canvasScroll` 可用宽高自动计算 stage 宽度：
  - 先扣除滚动容器 padding
  - 按 `canvas.width / canvas.height` 计算宽高比
  - 使用 `min(availableWidth, availableHeight * aspectRatio, 1120)` 作为最终渲染宽度
- 同时，单页场景会自动隐藏底部缩略图栏，并切换 `.editor-shell--single-slide` 布局，避免 156px 的固定 footer 白白占掉海报可视高度。
- 结果：单页竖版海报现在能在首屏完整展示，不需要再靠浏览器临时缩放或手工隐藏区域。

### Editor Roundtrip 验证（2026-03-31）
- 使用真实资产 `examples/demo_project_intro_ppt169_20251211/svg_final/slide_01_cover.svg` 做 `svgToSlide → slideToSvg → svgToSlide` 往返验证。
- 当前实现对该页保留了稳定的结构计数：
  - `rect=2`
  - `text=10`
  - `line=1`
  - `circle=1`
  - `path=14`
  - `image=1`
  - `group=4`
  - `linearGradient=1`
- `linearGradient#gradient1` 的坐标属性与两个 stop 颜色在往返后均保持一致。
- 这说明当前解析/导出链路对 demo cover 页的关键结构保真度达到可回归测试级别。

### Preset Follow-up Spec 对齐（2026-04-01）
- `editor/src/presets/colors.ts` 已进一步从旧 follow-up spec 收敛为 16 套更通用的 curated 配色，面向“普通用户一键换风格”而不是旧的 design/industry 对照表。
- `ColorScheme` 当前分类是 `category: 'universal' | 'mood' | 'industry'`，并统一默认文本/背景色：
  - `textDark #1A1A2E`
  - `textLight #FFFFFF`
  - `textMuted #6B7280`
  - `background #FFFFFF`
  - `backgroundAlt #F5F5F5`
- `editor/src/presets/fonts.ts` 已收敛为 6 套免费商用方案，字段名改为 `title/body/caption/label`；旧的 `previewText` 和 `*Family` 命名已移除。
- `editor/src/app.ts` 当前字体 preset UI 只显示 `scheme.name`，实际应用时按 `data-font-role` 映射到 `title/body/caption/label`。

### Editor MVP 入口页（2026-03-31）
- 新增 `editor/index.html` 作为浏览器编辑器入口，布局为深色顶栏 + 左侧 SVG 画布 + 右侧属性面板 + 底部缩略图条。
- 新增 `editor/src/app.ts`，用纯 TypeScript + DOM API 挂载 demo `SlideState`，并调用 `slideToSvg()` 渲染当前页面。
- demo state 使用三页硬编码数据，其中第一页沿用 `demo_project_intro` cover 的结构语言，显式覆盖 `rect` / `text` / `path` / `circle` / `line` / `image` / `group` / `linearGradient`。
- 为避免同页多个 inline SVG 的 `id="gradient_*"` 冲突，增加了本地 `namespaceSvgIds()` 处理，对主画布和缩略图分别加作用域前缀。
- `vite.config.ts` 已从 lib 模式切到 app 模式，显式以 `editor/` 为 root，并输出到 `dist/app`，避免覆盖 `tsc` 的 `dist/` 产物。
- 本地浏览器验收通过：
  - Dev server: `http://localhost:5173/`
  - 可见顶栏、画布、属性面板、缩略图条
  - 点击第 2 页缩略图后，页码从 `1 / 3` 更新为 `2 / 3`
  - 截图保存到 `/Users/haoguang/Downloads/ppt-master-editor-mvp_2026-03-31T05-52-32-556Z.png`

### Editor 元素选中交互（2026-03-31）
- `state_to_svg.ts` 现在会为所有导出节点附加 `data-element-id`，包括多行 text 的每一行 `<text>` 和 `<g>` 分组节点。
- 编辑器交互层没有改 SVG 生成模型，而是采用“渲染后 DOM 覆盖层”：
  - hover：基于 `data-element-id` 找到实际 DOM 节点，用 `getBoundingClientRect()` 合并多节点边界，再换算回 SVG viewBox 坐标，绘制蓝色半透明 overlay
  - selected：在同一套 bounds 上绘制蓝色实线边框 + 8 个白色手柄
  - overlay 节点统一放在 `data-editor-overlay-root` 下，并设置 `pointer-events=\"none\"`
- 属性面板字段是按元素类型分发，而不是通用 schema 渲染：
  - `text`: `x / y / width / text / font / lineHeight / fill`
  - `rect`: `x / y / width / height / fill / rx`
  - `path`: `fill / d(只读)`
  - `image`: `x / y / width / height / href`
  - `line`: `x1 / y1 / x2 / y2 / stroke`

### Raw SVG 直接渲染落地（2026-03-31）
- `editor/src/app.ts` 现在维护 `rawSvgStrings: string[]`，并在三条入口写入原始 SVG：
  - SVG 文件拖入
  - `?svg=` URL 加载
  - 默认 demo 真实 SVG (`/examples/demo_project_intro_ppt169_20251211/svg_final/slide_01_cover.svg`)
- 渲染策略改成：
  - `renderCanvas()` / `renderThumbnails()` 先用 raw SVG
  - raw 与 state 有偏差时，先把 state 的已知字段同步回 raw SVG，再渲染
  - 如果 raw SVG 里缺少 state 中的新元素，则单页回退到 `slideToSvg()`，避免空白或交互失效
- `editor/src/svg_to_state.ts` 新增 `normalizeSvgForEditor()`：
  - 对 `text / rect / circle / line / path / image / g` 注入稳定 `data-element-id`
  - 优先复用现有 `data-element-id` / `id`
  - 解析阶段也会复用这些 id，避免 raw DOM 和 state 的元素映射漂移
  - `preserveTextNodes` 模式下不再 merge 相邻 `text`，优先保留 1:1 DOM/state 映射
- 文本编辑和拖拽的兼容处理：
  - raw SVG 多行文本改成单个 `<text>` + 多个 `<tspan>`，这样 state 里仍是一个 `TextElement`
  - 拖拽预览会同时同步根 `<text>` 与其 `tspan` 的 `x/y`
- 导出 SVG 改为导出当前 canvas 上的 live DOM（移除 editor overlay 后序列化），不再从 `SlideState` 重生成。
- 为了让默认 demo 在 Vite dev/build/preview 下都能访问，`editor/vite.config.ts` 新增了 `/examples/...` 静态暴露，并在 build 后复制 demo SVG 和背景图到 `dist/app/examples/...`。
  - `circle`: `cx / cy / r / fill`
- 当前更新策略是“改字段 → 直接 mutate slide_state → 局部重渲染 canvas + thumbnails”；不重建 inspector form，因此输入焦点不会因为实时渲染丢失。
- 浏览器验收中发现一个真实边角：最初只监听了 `canvasScroll` 空白点击，点击 `.canvas-pane` padding 不会取消选中；后来扩展到整个 `.canvas-pane` 后通过验证。

### Editor 文本框双击编辑（2026-03-31）
- `editor/src/app.ts` 新增 `editingTextId` 运行时状态，以及一套 `ActiveTextEditor` 临时 DOM 管理对象，用于维护 `foreignObject`、`textarea`、状态提示和被隐藏的 SVG 文本节点。
- 双击 `text` 元素后，不修改 `slideToSvg()` 输出，而是在已渲染好的 SVG 上追加一个仅运行时存在的 `<foreignObject>`：
  - `textarea` 位置基于文本元素当前 SVG bounds
  - 样式尽量复用 `TextElement.font / lineHeight / fill / textAnchor`
  - 原 SVG 文本节点统一临时设为 `opacity=0`
- 输入阶段不重渲染 SVG，只做三件事：
  - 直接写回对应 `TextElement.text`
  - 调用 `layoutText(el)` 更新 Pretext 行数/高度/溢出提示
  - 同步右侧 inspector 的 `text` 字段值
- 退出编辑支持两种路径：
  - `Escape` 优先退出编辑态，再次按 `Escape` 才取消选中
  - 点击画布区域外部时先退出编辑；若点击发生在 `.canvas-pane` 内，则消费该次点击，避免第一下就切换选中或清空选中
- 退出时统一执行 `renderCanvas() + renderThumbnails() + renderInspector()`，因此修改会反映到画布和缩略图，同时保留 `selectedElementId`
- 浏览器 smoke test 已验证：
  - 双击 `cover_title` 会出现 textarea，原文本节点 `opacity=0`
  - 输入 `双击编辑标题\\nPretext 实时断行` 后，inspector `text` 字段同步，状态提示变为 `Pretext 2 行 · 高度 136px · 未溢出`
  - `Escape` 后重渲染得到两行 `<text>`
  - 再次进入编辑并点击 `.canvas-pane` 空白后退出，且选中状态仍保持在 `cover_title`

### Editor 拖拽移动 + 手柄缩放（2026-03-31）
- `editor/src/app.ts` 新增 `PointerInteractionSession` 运行时状态，分别覆盖 `move` 与 `resize` 两种模式，只有鼠标位移超过 `3px` 阈值后才真正进入拖拽，避免与单击选中冲突。
- 所有拖拽 delta 都通过 `clientPointToViewBox()` 做像素坐标 → SVG `viewBox` 坐标换算，因此 CSS 缩放后的画布仍能准确更新 `slide_state` 中的几何字段。
- 拖拽/缩放期间采用“改 state + 改 live DOM + 不整页重绘”的策略：
  - `rect / image / circle / line` 直接同步对应 SVG 属性
  - `text` 直接平移现有 `<text>` 节点，宽度变化只先写回 state，由 overlay 反馈 box 变化，`pointerup` 后再统一走 `slideToSvg()` 触发 Pretext 重排
- overlay 手柄现在显式带有 `data-editor-handle`、`pointer-events="all"` 和方向光标；选中元素本体则会切到 `move` 光标。
- `path` 与 `group` 保持可选中，但拖拽/缩放会在 inspector summary 区提示“暂不支持”，不再静默失败。
- 浏览器 smoke test 已验证：
  - 选中 `cover_title` 后，本体光标为 `move`，8 个手柄分别显示 `nwse / nesw / ns / ew` 四类 resize 光标
  - 拖拽 `cover_title` 后，`prop-x / prop-y` 从 `100 / 294` 更新到 `202.949... / 362.632...`
  - 拖拽 `cover_title` 右侧手柄后，`prop-width` 更新到 `669.812...`
  - 选中并拖拽 `cover_gradient_bar` 后，`prop-x / prop-y` 更新到 `102.949... / 87.506...`

### Editor 本地文件 I/O 闭环（2026-03-31）
- `editor/src/app.ts` 不再把 `createDemoState()` 当成唯一状态源，而是升级为可替换的运行时 `state`：
  - 启动时读取 `window.location.search` 中的 `state` 参数，直接 `fetch(path)` 拉取 JSON，再用 `SlideState` 结构校验替换当前 state
  - 若 URL 不含 `state` 或加载失败，则保留内置 demo，避免 dev server 首屏空白
- 新增 `editor/src/state_io.ts`，把文件相关逻辑从大体量 UI 文件里拆出来：
  - `getStatePathFromSearch()` 解析 `?state=...`
  - `ensureSlideState()` / `parseSlideStateJson()` 做最小结构校验，至少保证 `canvas.width`、`canvas.height` 和非空 `slides[]`
  - `createJsonDownload()` / `createSvgDownloads()` 统一生成浏览器下载清单，SVG 文件名固定为 `slide_01.svg`、`slide_02.svg`...
- `editor/index.html` 顶栏新增两个按钮：
  - `保存 JSON`：把当前 `SlideState` 以格式化 JSON 下载为 `slide_state.json`
  - `导出 SVG`：对 `stateToSvgs(state)` 的结果逐页触发下载
- 同时新增全局拖拽 overlay，整个编辑器窗口都能接受 `.json` 文件 drop，并在成功解析后立刻重渲染画布、缩略图和属性面板。

### Project-Level slide_state Bridge（2026-03-31）
- 新增 `editor/src/project_pipeline.ts`，把“项目级多页 SVG 文件集”抽象成独立桥接层：
  - `buildProjectStateFromSvgInputs()`：按自然顺序读取 `svg_output/*.svg` 并生成 `SlideState`
  - `createProjectSvgArtifacts()`：优先使用 `slide.id` 作为文件名回写 SVG，自动处理非法字符和重名
- 新增 `editor/src/cli.ts` 和根目录包装器 `tools/slide_state_bridge.py`：
  - `capture`：从项目 `svg_output/` 或 `svg_final/` 生成项目根目录 `slide_state.json`
  - `render`：从 `slide_state.json` 回写兼容的 `svg_output/`
  - `sync`：先 capture 再 render，作为当前“Executor 先写 SVG”的正式主路径
- 这条 bridge 让浏览器编辑器第一次真正接进仓库主工作流：
  - AI / Executor 继续按现状产出 `svg_output/`
  - bridge 将其收敛为 `slide_state.json`
  - 编辑器、后续 AI、现有 finalize/export 全部指向同一份 state
- CLI 环境里没有浏览器 `OffscreenCanvas`，直接启用 Pretext 文本测量会报错 `Text measurement requires OffscreenCanvas or a DOM canvas context.`
  - 结论：bridge CLI 暂时不初始化 Pretext，直接退回 `state_to_svg.ts` 已有的 fallback 断行逻辑
  - 影响边界：浏览器编辑器仍保留 Pretext；CLI 主要负责收敛 state 和生成兼容 SVG，已足以打通当前主链路
- 确定性验证结果：
  - `cd editor && npm run test` 通过，`6` 个 test files、`44` 个 tests 全部通过
  - `cd editor && npm run build` 通过
  - `python3 tools/slide_state_bridge.py sync <临时项目副本>` 成功生成 `slide_state.json` 并回写 `10` 个 SVG
  - 继续执行 `python3 tools/finalize_svg.py <临时项目副本>` 成功
  - 继续执行 `python3 tools/svg_to_pptx.py <临时项目副本> -s final --no-notes` 成功导出 `10` 页 PPTX

### Native Executor_General State-First Path（2026-03-31）
- 形式化落点不在 bridge 代码本身，而在“正式角色协议”仍然要求所有 Executor 先写 SVG；这会让 `slide_state.json` 只能是 bridge 回收产物，而不是真正的 AI 第一产物。
- 当前仓库里最合适的切入点是 `Executor_General`：
  - 通用灵活路径的布局自由度最高，最容易直接映射到 `slide_state` schema
  - 咨询风格 Executor 仍保留更多“直接写 SVG”表达习惯，短期内不宜一起切
- 已完成的正式化改动：
  - `roles/Executor_General.md` 改为 state-first 协议：第一产物是项目级 `slide_state.json`，并明确 `slide.id` 就是最终 SVG 文件名 stem
  - `AGENTS.md`、`README*.md`、`roles/README.md`、`.claude/skills/poster/SKILL.md` 已同步为“一条正式 General 路径先产 state，再 render；咨询路径暂时 svg-first + sync”
  - `roles/Image_Generator.md` 的后续衔接文案已从“生成 SVG”放宽为“生成主产物（slide_state 或 SVG）”
  - `tools/project_utils.py` / `tools/project_manager.py` / `tools/error_helper.py` 现在承认 state-first 中间态：如果项目已有 `slide_state.json` 但尚未 render 出 `svg_output/`，`validate` 给警告而不是结构错误
- 新增的确定性验证覆盖了两层：
  - 仓库级：`cd editor && npm run test`、`cd editor && npm run build` 均通过
  - 工作流级：新初始化的 state-first 临时项目在只有 `slide_state.json` 时，`project_manager.py validate` 返回“有效但有建议”；执行 `render` 后再次验证为“项目结构完整，没有问题”；随后 `finalize_svg.py` 与 `svg_to_pptx.py -s final --no-notes` 均成功
  - 额外注意：`examples/*` 目录本身不一定满足 `project_manager.py validate` 的正式项目约束（如 `README.md`、目录命名），因此 smoke test 更稳妥的方式是先 `project_manager.py init` 创建临时项目，再复制 `svg_output/` 或 `slide_state.json`

### Editor AI Command Roundtrip（2026-03-31）
- 原有 `AI 协作` 面板只解决了“如何把当前选中元素/页面交给外部 AI”，但缺少“AI 返回后如何回到编辑器”的闭环，因此还算不上真正可用的入口。
- 当前实现把这条链路补成了完整 roundtrip：
  - 编辑器继续基于当前选中元素或当前页构造 `AiCommand`
  - `AiCommand` 现在携带 `scope / slideIndex / slideId / elementSnapshot / slideSnapshot / instruction`
  - 导出产物分成两类：
    - `design_patch.ai-request.json`：正式机器可读 handoff 请求，仍沿用 `DesignPatch` schema
    - `design_patch.ai-handoff.md`：给 Claude Code / Codex 这类本地 agent 的执行说明
  - 官方消费路径是 Claude Code / Codex 在本地仓库里直接修改 `slide_state.json` 并执行 `render / validate`
  - 浏览器编辑器内“应用 AI Patch”保留为兼容回流路径，用于只返回 `design_patch.json` 的会话
  - AI 返回后，浏览器编辑器可直接解析并应用 `DesignPatch.operations[]`，而不是只停留在“复制提示词”层
- `editor/src/design_patch.ts` 新增三类正式能力：
  - `parseDesignPatchJson()` / `ensureDesignPatch()`：校验 AI 返回 JSON
  - `applyDesignPatch()`：把 update/add/delete/reorder 操作应用回当前 `SlideState`
  - `getDesignPatchPrimarySlideIndex()`：应用后把编辑器定位到被修改的页

### Editor Phase 4 收口（2026-03-31）
- 撤销/重做不再依赖整份 state 快照，而是基于 `PatchOperation` 生成 forward / inverse patch：
  - `update` 交换 `value / oldValue`
  - `add` 反转为 `delete`
  - `delete` 反转为 `add`
  - `reorder` 交换索引
- 为支撑“模板页 / 图表页追加导入”，`DesignPatch` 现在支持 slide 级 `add / delete / reorder` 路径（如 `/slides/3`），不再局限于 `/slides/<n>/elements/...`。
- 编辑器顶部历史按钮与 `patchCountBadge` 现在都从当前已生效 history 推导，而不是简单追加日志；这保证了导出给 Claude Code / Codex 的 `design_patch.ai-request.json` 不会把已经撤销的操作也带出去。
- 现有 `templates/layouts/*.svg` 与 `templates/charts/*.svg` 当前本质上都是整页 SVG 资产，因此本轮把“模板系统 / 图表支持”落在“追加为 slide_state 页面”而不是“内容区级 widget 嵌入”。这条路径已经足够接入当前 state-first 主工作流，且不需要为假想的局部图表组件引入第二套模型。
- 浏览器 smoke test 已验证：
  - 属性面板改 `cover_title.x` 后，`patchCountBadge` 增为 `1 条已应用 Patch`，撤销按钮可用
  - 追加 `templates/layouts/general/01_cover.svg` 后，页数从 `3` 变为 `4`，当前页定位到导入模板页
  - 撤销 / 重做模板页追加后，页数分别回到 `3` / `4`
  - 追加 `templates/charts/bar_chart.svg` 后，页数变为 `5`，当前页定位到 `bar_chart`
- `editor/src/app.ts` 的 JSON 导入已改成分流逻辑：
  - 先尝试按 `design_patch` 解析
  - 若 `operations` 为空但带 `aiCommand`，则仅恢复 AI 面板上下文
  - 若存在 `operations`，则直接更新当前 `state` 和累计 `patches`
  - 若解析失败，再回退为普通 `slide_state.json` 导入
- 确定性验证结果：
  - `cd editor && npm test` 通过，`6` 个 test files、`46` 个 tests 全部通过
  - `cd editor && npm run build` 通过
  - `design_patch.test.ts` 已覆盖：AI handoff 请求生成、design patch 解析、update/add/reorder 应用回 state

### Raw SVG 直渲染基线（2026-03-31）
- editor 现在形成了双轨模型：
  - `state` 继续作为属性面板、拖拽、双击文本编辑与 patch 的数据源
  - `rawSvgStrings` 成为渲染优先级最高的来源，只有缺失或结构失配时才 fallback 到 `slideToSvg()`
- raw SVG 在进入 editor 前会统一标准化：
  - 给 `text`、`rect`、`circle`、`line`、`path`、`image`、`g` 注入稳定 `data-element-id`
  - 解析时优先复用原始 `data-element-id` / `id`
  - raw 路径启用 `preserveTextNodes`，避免相邻 `<text>` merge 掉以后破坏 DOM/state 对位
- 为防止属性编辑/拖拽/双击文本后 re-render 回退到旧 SVG，当前实现采用“render 前 state → raw SVG 同步”：
  - 能按 `data-element-id` 找到原节点时，就把文本、几何和常用样式字段写回 raw SVG
  - 一旦某页结构和 raw SVG 严重偏离，就自动回退到 `slideToSvg()`，优先保证正确性
- 默认 demo 已切到真实资产路径 `/examples/demo_project_intro_ppt169_20251211/svg_final/slide_01_cover.svg`
  - `vite.config.ts` 额外把该 demo SVG 与 `images/cover_background.png` 暴露到 dev/build/preview，可直接 fetch
- 当前已知边界：
  - 结构性 patch 可能使单页降级回 `slideToSvg()`，从而损失部分原始视觉细节
  - 任意本地拖入 SVG 若依赖相对图片资源，而浏览器拿不到对应图片文件，图片仍可能无法显示

### Step 2/5：普通人友好 inspector 与预设系统（2026-04-01）
- inspector 不再把 `font / fill / rx / x1 / cx` 这类技术字段直接暴露给普通用户，而是拆成：
  - 主区：中文标签 + 颜色选择器 / slider / select
  - 高级区：坐标、原始 `font` shorthand、路径数据等
- 文本元素的 `fontSize / fontWeight` 不能只写回显式属性，否则 `font` shorthand 与 `slide_state` 会漂移，后续 `syncTextElementNode()`、导出和撤销/重做都会出现不一致。
  - 本轮采用的稳定策略是：任何字号/字重/字体族修改都统一回写 `font`，同时同步 `fontSize / fontWeight / fontFamily`
  - `parseFontSpec()` 继续沿用现有正则，避免和现有 `state_to_svg.ts` / `svg_to_state.ts` 解析逻辑分叉
- 配色预设的真正难点不是 UI，而是“点卡片后，live SVG、`slide_state`、`rawSvgStrings` 三份状态必须一起变”。
  - 如果只改 DOM，下一次 render 会被 state 覆盖
  - 如果只改 state，没有 raw SVG 同步则会丢掉真实 SVG 的细节结构
  - 因此本轮 preset click 走的是：解析 raw SVG → 改有角色的节点 → 回写对应 state 元素属性 → 保存回 `rawSvgStrings`
- 对已有 `data-color-role` / `data-font-role` 的 SVG，当前行为是确定性的。
- 对没有颜色角色标记的 SVG，当前 fallback 只做“够用”的颜色桶推断：
  - 根据 fill/stroke 的实色、面积、亮度和饱和度，粗分出 `background / background-alt / text-dark / text-light / text-muted / primary / secondary / accent`
  - 这足够支撑基础 demo 和简单页面，但还不是设计级精确映射
- 颜色/字体 preset catalog 当前已固定为：
  - `16` 套配色：`5` 套 universal + `6` 套 mood + `5` 套 industry
  - `6` 套字体：Noto Sans / 思源黑体 / 思源宋体 / 阿里巴巴普惠体 / OPPO Sans / HarmonyOS Sans
- 浏览器 smoke test 已确认：
  - `data-color-role` 标记节点在点击 preset 后立即改变颜色
  - 右侧 inspector 显示中文字段名，且文本元素存在原生颜色选择器和字号 slider
- 2026-04-01 晚些时候又补了一轮 preset UI 收口：
  - 配色按钮不再显示文字，只显示 3 条颜色条；名称仅保留在 `title` 和 `aria-label`
  - 字体按钮在当前侧栏宽度下改成 2 列网格，并通过 `overflow: hidden + text-overflow: ellipsis` 消除明显溢出
- 2026-04-01 已同步更新 Executor 角色定义文档：
  - `roles/Executor_General.md`
  - `roles/Executor_Consultant.md`
  - `roles/Executor_Consultant_Top.md`
  三者现都包含 `SVG 语义标记协议（编辑器预设系统）` 章节，明确约束 AI 生成 SVG 时补齐 `data-color-role` / `data-font-role`
  - 这意味着后续 Step 4/5 可以把编辑器侧的 preset 替换逻辑视为“有上游契约”的正式路径，而不仅是对无标记 SVG 的 fallback

### Editor UX 修复批次（2026-04-01）
- 直接编辑 SVG 路径里，`saveJsonBtn`、`watchFileBtn`、AI 协作区和资产导入区继续保留 DOM/逻辑，但 UI 默认隐藏；这比删代码更稳，因为未来若要恢复 slide_state/AI handoff 工作流，只需要重新露出面板。
- `<details>` 的默认折叠不能只依赖静态 HTML。浏览器在同 URL 热更新或状态恢复时可能保留上一次展开状态，所以 `editor/src/app.ts` 启动阶段额外强制把 `.sidebar-section` 全部收起，才能稳定满足“默认折叠配色/字体”。
- 文本框溢出的根因不只在拖拽边界，还在 `svg_to_state.ts` 过去把导入的 `text.width` 固定成 `1200`。这会让很多单行标题一开始就拥有接近整页的交互框，导致选框和拖拽边界都失真。
  - 本轮改为按 `font-size`、最长文本行长度、`text-anchor` 和可用画布宽度估算初始 `width`
  - 浏览器回归里，`slide_01_cover_text_5` 的宽度从固定 `1200` 收敛到约 `426`
- 文本拖拽和 resize 的可靠性现在依赖三条同时成立：
  - move 路径按元素交互框做 `clamp`，而不是只改裸 `x/y`
  - text resize 允许上下手柄，并把纵向变化写回 `maxHeight`
  - 历史栈对 text 额外记录 `maxHeight`，否则纵向 resize 无法 undo/redo
- 浏览器 smoke 结果：
  - `saveJsonBtn` 不可见；`watchFileBtn / exportAiTaskBtn / applyAiPatchBtn / importTemplateBtn / importChartBtn` 虽仍在 DOM 中，但计算样式不可见
  - 右侧属性卡、配色折叠项、字体折叠项在 `scrollTop=0` 时均位于可视区内
  - 通过属性面板修改 `prop-x` 后，`undoBtn / redoBtn` 可正确往返 `100 ↔ 140`
  - 文本拖拽后 `x/y` 同时变化；向右下拖拽时 `x + width` 被限制在 `1280` 内，向左上拖拽时文本顶部不会越过画布

### 导出与字体预设补丁（2026-04-01）
- 当前导出链路已经具备“从 live DOM 克隆 SVG、剥离 overlay、直接下载”的稳定能力，因此“保存模板”不需要单独实现模板抽象，只要复用 SVG 导出并改文件名即可。
- PNG 导出的关键不是 `canvas.drawImage()`，而是先把 SVG 中的 `<image href>` 尽量变成浏览器可栅格化的来源：
  - 同源或可访问资源：`fetch -> blob -> data URL`
  - 失败资源：退回绝对 URL，继续尝试渲染
  - 最终仍失败：明确提示用户可能是跨域 / 外链图片问题，而不是让整个导出 silently fail
- 这条路径是 best-effort，而不是完全无条件成功：
  - 若 SVG 里引用第三方站点图片且该资源没有 CORS，浏览器仍可能拒绝 rasterize
  - 当前实现选择“导出主流程不崩 + 用户知道为什么缺图”，这是对纯前端编辑器最稳的边界
- 字体预设已从旧的“系统字体 + 商业字体混搭”切到 6 套免费商用字体族：
  - `Noto Sans`
  - `思源黑体`
  - `思源宋体`
  - `阿里巴巴普惠体`
  - `OPPO Sans`
  - `HarmonyOS Sans`
- 字体面板继续只显示 `scheme.name`，避免给普通用户额外术语负担；真正的字体族映射仍由 `title/body/caption/label` 四个角色消费。
- 已知边界：
  - 若运行环境未安装这些字体，浏览器仍会回退到 generic fallback，视觉差异度会下降
  - 因此“6 套方案逻辑上不同”已满足，但“每台机器都明显不同”仍取决于本机字体可用性

### 文本 resize 语义修复（2026-04-01）
- 之前文本 resize 的核心问题是把“改文本框宽度”和“缩放文字”混在了一套规则里：
  - `e / w` 侧边手柄本应只改变 line-wrap 宽度，但旧实现会同时按 `scaleX` 放大 `fontSize / lineHeight`
  - live preview 只同步现有节点属性，不重建断行结构，导致拖动中看不到真实 reflow
- 本轮把文本 resize 语义收敛成两类：
  - 侧边手柄：只更新 `width` 与锚点 `x`
  - 四角手柄：同时更新 `width / x / y / fontSize / lineHeight / maxHeight`
- `editor/src/text_resize.ts` 现在承载两块最小公共逻辑：
  - 文本元素专属 handle 集合
  - 文本 resize 纯语义与 live SVG text 节点重排
- live preview 不再只改 `x/y/font-size`，而是复用同一套 `layoutText()` 结果去重建 `<text>/<tspan>`，因此 pointermove 阶段就能看到按新宽度的断行结果。
- 这轮没有改 `getPatchKeysForElement(text)`、history entry 结构或 `state_to_svg.ts` 的正式导出语义，因此 undo/redo 与 patch diff 仍沿用现有链路。

### 海报画幅入口收敛（2026-04-01）
- 顶部这组 `1:1 / 4:5 / 9:16` 控件不再按“单页即可”显示，而是明确按“单页且当前画布属于海报预设”显示。
- 当前产品化判断被收敛为最小明确集：
  - `1280x720` 视为 PPT 16:9
  - `1024x768` 视为 PPT 4:3
  - `1080x1080` / `1080x1350` / `1080x1920` 视为海报画幅入口对应的 `square / poster / story`
- `supportsCanvasPresetEditing()` 仍然只表达“当前页面元素是否允许实时缩放切换”，不再兼任“是否显示入口”的职责；这样入口显隐和按钮可用态分层更清楚。
- 这轮没有改动 `resizeSlideStateCanvas()` 本身，因此比例切换后的元素缩放语义保持不变，只是把入口限制到了海报场景。

---

## Historical: 远端合并分析（2026-03-28，已完结）

结论：不建议直接合并 upstream/main（72 ahead / 5 behind，是架构迁移而非增量更新）。
详细记录已归档，不再更新。
