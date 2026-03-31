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

---

## Historical: 远端合并分析（2026-03-28，已完结）

结论：不建议直接合并 upstream/main（72 ahead / 5 behind，是架构迁移而非增量更新）。
详细记录已归档，不再更新。
