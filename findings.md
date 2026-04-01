# Findings & Decisions

## Requirements
- 新建 `editor/src/text_editing.ts`，把用户列出的文本双击编辑函数从 `app.ts` 移出。
- 同步迁移 `ActiveTextEditor`、`HiddenCanvasNode`。
- `app.ts` 改为 import 新模块，并在调用点传入需要的 context。
- 不改变交互实现逻辑；只允许为了解除 `app.ts` 全局耦合而做参数/上下文注入。
- `cd editor && npm run build` 与 `cd editor && npm test` 必须通过。

## Discovery Findings
- 目标函数原本集中在 [app.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/app.ts) `3088-3318`，覆盖文本编辑入口/退出、编辑框布局、状态文案、隐藏原 SVG 节点和 inspector 文本同步。
- 调用点分散在 [app.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/app.ts) 的快捷键、切页、导入、属性编辑和 canvas 交互路径，其中 `handleCanvasElementDoubleClick` 与 `handleDocumentPointerDown` 直接依赖 `enterTextEditing` / `exitTextEditing`。
- 文本编辑函数依赖的 app 本地能力主要分成三类：
  - 状态读写：`editingTextId`、`activeTextEditor`、`hoveredElementId`、`selectedElementId`
  - DOM/helper：`getCanvasSvg`、`getCanvasElementNodes`、`measureElementBounds`、`elementFields`
  - render/patch 链路：`findElementById`、`getCurrentSlide`、`syncSlideElementIntoSvg`、`createPropertyPatch`、`commitPatchOperations`、`commitCurrentSlideFromLiveCanvasSvg`、`renderCanvas`、`renderThumbnails`、`renderInspector`、`refreshCanvasOverlays`
- [pointer_interactions.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/pointer_interactions.ts) 与 [preset_engine.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/preset_engine.ts) 已展示当前模块化模式：导出函数/类型，app 侧构造 context 对象并通过 getter/setter 注入状态变量与 UI helper。

## File Scope
- `/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/app.ts`
- `/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/text_editing.ts`

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| `text_editing.ts` 作为文本编辑工具层导出函数与类型 | 避免新模块反向 import `app.ts`，同时把文本编辑逻辑边界单独收口 |
| `TextEditingContext` 由 `app.ts` 提供状态 getter/setter、DOM helper 和 render/patch 链路 | 保持函数体机械迁移，不改交互实现 |
| `measureTextEditorBounds` / `applyTextEditorStyles` / `syncTextEditorFrame` 等 helper 与类型一并迁出 | 用户明确要求迁出完整文本编辑模块，而不是只拆入口函数 |

## Implementation Outcome
- 新增 [text_editing.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/text_editing.ts)，导出：
  - `enterTextEditing`
  - `exitTextEditing`
  - `measureTextEditorBounds`
  - `applyTextEditorStyles`
  - `syncTextEditorFrame`
  - `buildTextEditorStatus`
  - `hideCanvasElementNodes`
  - `restoreHiddenCanvasNodes`
  - `syncInspectorTextField`
  - `getTextAlign`
  - `ActiveTextEditor`
  - `HiddenCanvasNode`
  - `TextEditingContext`
- [app.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/app.ts) 删除了上述函数/类型的本地实现，改为 import 新模块，并通过 `textEditingContext` 注入 app 层依赖。
- `preset_engine` 相关上下文保持原样，只把其 `exitTextEditing` 回调改为调用模块化后的函数签名。

## Verification Plan
- `cd editor && npm run build`
- `cd editor && npm test`
- `rg -n 'function (enterTextEditing|exitTextEditing|measureTextEditorBounds|applyTextEditorStyles|syncTextEditorFrame|buildTextEditorStatus|hideCanvasElementNodes|restoreHiddenCanvasNodes|syncInspectorTextField|getTextAlign)|interface (ActiveTextEditor|HiddenCanvasNode)' editor/src/app.ts editor/src/text_editing.ts`

## Verification Results
- `cd editor && npm run build` 通过，`tsc && vite build` 成功，生成 `dist/app/index.html` 与 `dist/app/assets/index-uWdD4ecn.js`。
- `cd editor && npm test` 通过，13 个测试文件 / 74 个测试全部通过。
- `rg` 复查确认目标函数和类型定义只存在于 [text_editing.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/text_editing.ts)，[app.ts](/Users/haoguang/Documents/RedCode/xingbao/ppt-master/editor/src/app.ts) 中不再保留这些实现。
