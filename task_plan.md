# Task Plan: extract text editing from app.ts

## Goal
把 `editor/src/app.ts` 中文本双击编辑相关实现机械拆分到新的 `editor/src/text_editing.ts`，保持现有逻辑不变，并让 `cd editor && npm run build`、`cd editor && npm test` 通过。

## Current Phase
Phase 5

## Phases
### Phase 1: Discovery
- [x] Read repo instructions and required skills
- [x] Inspect `app.ts` text-editing functions, types, and call sites
- [x] Inspect `preset_engine.ts` / `pointer_interactions.ts` context injection patterns
- **Status:** complete

### Phase 2: Planning & Orchestration
- [x] Refresh planning files for this task
- [x] Prepare one focused coding step for `codex-coder`
- [x] Define validation commands and fallback criteria
- **Status:** complete

### Phase 3: Implementation
- [x] Create `editor/src/text_editing.ts`
- [x] Move target functions/types and required helper wiring out of `app.ts`
- [x] Update `app.ts` imports and call sites without changing editing behavior
- **Status:** complete

### Phase 4: Verification
- [x] Run `cd editor && npm run build`
- [x] Run `cd editor && npm test`
- [x] Re-scan moved symbols and import correctness
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize changed files and verification evidence
- [x] Call out any residual coupling or next extraction candidates
- **Status:** complete

## Key Questions
1. `enterTextEditing` / `exitTextEditing` 访问的 `editingTextId`、`activeTextEditor`、`hoveredElementId`、render/patch 链路，哪些应通过 getter/setter/context 注入？
2. `measureTextEditorBounds`、`applyTextEditorStyles`、`syncTextEditorFrame` 依赖的 helper 和常量，哪些能在不改逻辑的前提下保留模块内聚？
3. 文本编辑相关类型和 helper 迁出后，`preset_engine.ts` 通过 app 注入的 `exitTextEditing` 是否仍可保持不变？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用 `planning-with-files` 记录本轮进度 | 任务跨 discovery / 实现 / 构建 / 测试多个阶段，且仓库要求长链路任务优先落盘 |
| 优先尝试 `codex-coder` / `ai-cli-bridge`，超时后回退到本地实现 | 仓库 AGENTS 明确要求代码实现经由 codex worker，但本轮 bridge 在 120s 内无可靠结果 |
| 本轮只拆文本编辑模块，不顺带整理 inspector、overlay 或 patch/render 逻辑 | 用户要求“纯机械移动，不改实现逻辑” |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `ai-cli-bridge/clink` 120s 超时，未返回可靠 worker 结果 | 1 | 记录失败后回退到本地直接编辑，并自行跑 build/test 验证 |

## Notes
- 目标函数与类型已全部迁入 `editor/src/text_editing.ts`，`app.ts` 只保留 import、context 和调用点。
- 工作区存在大量未提交改动，本轮只修改了 `editor/src/app.ts`、新增 `editor/src/text_editing.ts`，以及 planning 文件。
