# Task Plan: Pretext 驱动的 AI + 人协同编辑工作台

## Goal
将 ppt-master 从"AI 生成工具"升级为"AI + 人协同编辑工作台"。引入 `slide_state.json` 作为真相源，`@chenglou/pretext` 作为文本排版内核，构建浏览器编辑器实现 AI 生成 ↔ 人直接操作的闭环。

## Core Architecture

```
AI 生成 → slide_state.json（结构化真相源）
              ↕ WebSocket
    浏览器编辑器（Pretext 文本内核 + SVG 渲染）
              ↕
    人的编辑 → patch → AI 继续迭代
              ↓
    导出 SVG → 现有 finalize_svg.py → svg_to_pptx.py（不动）
```

## Current Phase
Phase 4

## Phases

### Phase 1: slide_state 数据模型 + Pretext 集成
- [x] 定义 `slide_state.json` schema（canvas, slides, elements）
- [x] 实现 slide_state → SVG 转换器（TypeScript）
- [x] 集成 Pretext：文本元素通过 `layoutWithLines()` 精确断行
- [x] 实现 SVG → slide_state 反向解析（兼容现有 SVG 资产）
- [x] 溢出检测：`layout()` 返回 height，自动标记超出框高的文本
- **Status:** complete

### Phase 2: 浏览器编辑器 MVP
- [x] 项目脚手架（Bun/Vite + TypeScript + Pretext）
- [x] SVG 画布渲染：slide_state → SVG → 浏览器展示
- [x] HTML 入口页面：`editor/index.html`
- [x] 多页导航：上一页 / 下一页 / 页码显示
- [x] 缩略图条：点击缩略图切换当前页
- [x] 右侧属性面板骨架：当前页元数据 + 元素概览
- [x] 元素选中交互：hover 高亮 + 点击选中 + 属性面板
- [x] 文本框编辑：双击进入编辑模式，Pretext 实时断行
- [x] 拖拽移动 + 拉伸缩放
- [x] 保存：编辑结果写回 slide_state → 重新导出 SVG
- **Status:** complete

### Phase 3: AI 桥接 + 实时协作
- [x] Dev server：文件 watcher + WebSocket 推送
- [x] AI 生成 slide_state 后自动通知编辑器刷新
- [x] design_patch.json：记录人的编辑操作，AI 可读取
- [x] 现有 Executor 的 `svg_output/` 可通过正式 bridge 收敛为项目级 `slide_state.json`，并回写兼容 SVG
- [x] 至少一条正式 AI/Executor 路径原生输出 `slide_state.json`（当前为 `Executor_General`，随后 `render` 回写兼容 SVG）
- [x] 编辑器中的本地 AI handoff 入口（选中元素 / 当前页 → 导出 handoff bundle → 本地 skill 直接修改或兼容回流 AI Patch）
- **Status:** complete

### Phase 4: 生产就绪
- [x] 多页导航 + 缩略图
- [x] 模板系统：现有模板页可在编辑器内即时转为 `slide_state` 页面并追加到当前项目
- [x] 图表支持：`templates/charts/*.svg` 可作为图表页追加导入，并进入同一套 patch / AI handoff / render 闭环
- [x] 撤销/重做（基于 `slide_state` diff / inverse patch）
- [x] 快捷键系统（撤销/重做、保存 JSON、导出 AI handoff、翻页）
- **Status:** complete

## Key Decisions
| Decision | Rationale |
|----------|-----------|
| slide_state.json 作为唯一真相源 | AI 和人都读写同一个结构化数据，消除信息不对称 |
| Pretext 作为文本排版内核 | 精确断行、高度计算、溢出检测，替代 AI 猜测 `<tspan>` 坐标 |
| SVG 作为渲染层（非 Canvas） | 浏览器原生支持，且现有 finalize/export 链路直接兼容 |
| 现有后处理链路不动 | finalize_svg.py + svg_to_pptx.py 保持不变，降低风险 |
| 架构一步到位 | 从第一天就建立正确的数据模型，不做渐进式 hack |
| 使用命名字体而非 system-ui | Pretext 官方提示 macOS 下 system-ui 精度不安全 |

## Constraints
- Pretext 仅支持 `white-space: normal` 和 `pre-wrap`，不支持 `nowrap`
- 需要命名字体（如 `PingFang SC`、`Inter`），避免 `system-ui`
- 现有 SVG 约束全部保留：禁止 clipPath/mask/style/foreignObject 等
- slide_state → SVG 输出必须符合现有 SVG 技术约束

## Notes
- Pretext: github.com/chenglou/pretext, 20K+ stars, MIT, 15KB
- 前身记录见 findings.md 的 "Future Initiative" 部分
- 2026-03-31：为降低实现风险，提前把“多页导航 + 缩略图”前置到 Phase 2 的 HTML 入口页 MVP，而不是等到后续大交互阶段再补。
- 2026-03-31：元素 hover/selection 与属性面板编辑已落地；当前交互仍是“改属性即整页重渲染”，但会保留选中元素 ID，并在重渲染后恢复选中框与字段状态。
- 2026-03-31：文本元素现已支持双击进入运行时 `foreignObject + textarea` 编辑态；输入期间只更新 `slide_state` 与 Pretext 行数/溢出提示，不实时重渲染 SVG，退出时统一重渲染画布和缩略图并保持选中态。
- 2026-03-31：新增 pointer 交互状态机。拖拽/缩放期间直接改 live SVG DOM 与 inspector 字段，`pointerup` 后再统一 `renderCanvas() + renderThumbnails() + renderInspector()`；这样保留了 `slideToSvg()` 的单向真相源，又避免每帧整页重绘。
- 2026-03-31：编辑器现已支持三类本地文件 I/O：启动时通过 `?state=<path>` fetch 外部 `slide_state.json`，拖拽 `.json` 文件覆盖当前 state，以及从工具栏直接下载 `slide_state.json` / `slide_01.svg`...`slide_N.svg`。这让 `slide_state → SVG → finalize/export` 链路第一次在浏览器内闭环可用。
- 2026-03-31：Phase 3 已补上 dev server 文件 watcher，默认监听仓库 `.cache/slide_state.json` 并通过 Vite HMR WebSocket 推送自定义事件；浏览器端可自动 `fetch()` 新 JSON 并调用 `replaceState()`，同时保留手动“监听文件”轮询 fallback。
- 2026-03-31：编辑器现在会把属性面板修改、拖拽/缩放完成态、文本编辑退出态记录为 `design_patch.json` 风格的结构化 patch；SVG 导入也已打通，支持拖拽多个 `.svg` 文件或通过 `?svg=...` URL 参数直接加载多页 state。
- 2026-03-31：新增 `tools/slide_state_bridge.py` + `editor/src/cli.ts` 项目级桥接链路，允许把现有 `svg_output/` 收敛为项目根目录 `slide_state.json`，再回写兼容的 `svg_output/`；这让浏览器编辑器第一次正式进入主工作流，而不再只是独立 MVP。
- 2026-03-31：CLI 环境下暂不启用 Pretext 原生测量，因为 Bun/Node 缺少 `OffscreenCanvas`；bridge 会退回当前已有的 fallback 文本断行逻辑，浏览器编辑器仍继续使用 Pretext。
- 2026-03-31：`Executor_General` 已被正式切到 state-first。角色协议、AGENTS、README、poster skill 现在都要求它先生成项目级 `slide_state.json`，再执行 `python3 tools/slide_state_bridge.py render <项目路径>` 回写兼容 `svg_output/`；`project_manager.py validate` 也已接受“只有 state、尚未 render”的中间态。
- 2026-03-31：编辑器已新增本地 AI handoff 入口。用户可围绕当前选中元素或当前页面导出 `design_patch.ai-request.json` + `design_patch.ai-handoff.md`，并把 AI 返回的 `design_patch` 直接拖回编辑器或通过按钮应用；该链路继续以 `slide_state.json` 为唯一真相源。
- 2026-03-31：Phase 4 已收口。编辑器现已补上 inverse-patch 驱动的撤销/重做、键盘快捷键，以及“模板页 / 图表页 → slide_state 页面 → 当前项目追加”导入路径。导入后的页面继续走同一套 `design_patch`、本地 AI handoff、`slide_state_bridge.py render`、`finalize_svg.py`、`svg_to_pptx.py` 链路。
- 2026-03-31：本地 AI handoff 的官方语义已统一为“给 Claude Code / Codex 本地 skill / command 消费的 handoff bundle”。`design_patch.ai-request.json` 是机器可读输入，`design_patch.ai-handoff.md` 是执行说明；浏览器内直接应用 `design_patch.json` 仅保留为兼容回流路径，不再与官方直接改项目路径混淆。
