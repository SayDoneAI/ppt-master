---
name: ppt-edit
description: "Consume a PPT Master editor local AI handoff bundle from design_patch.ai-request.json or a design_patch.json that still includes aiCommand, update slide_state.json, and render compatible SVG output."
---

# PPT Edit — 编辑器 AI Handoff Skill

用于消费 PPT Master Editor 导出的本地 AI handoff bundle。该流程由 Claude Code 在本地仓库里执行；浏览器编辑器只负责导出文件，不承担浏览器直连或服务端 API 调用。

## 适用场景

- 用户在浏览器编辑器里选中元素或页面后，点击“导出 AI Handoff”
- 项目根目录已有 `slide_state.json`
- 项目根目录或 `.cache/ai_handoff/` 目录存在以下任一 handoff 文件：
  - `design_patch.ai-request.json`
  - `design_patch.json`（仅兼容旧用法，且其中必须包含 `aiCommand`）

## 执行约定

1. 先阅读仓库根目录 `AGENTS.md`
2. 阅读项目根目录 `slide_state.json`
3. 阅读项目根目录或 `.cache/ai_handoff/` 中的机器可读 handoff JSON，提取：
   - `aiCommand.scope`
   - `aiCommand.slideId`
   - `aiCommand.elementId`
   - `aiCommand.elementSnapshot`
   - `aiCommand.slideSnapshot`
   - `aiCommand.instruction`
   - 已有 `operations`
4. 只修改 `slide_state.json`
   - 不要直接改导出的 `svg_output/*.svg`
   - 优先围绕选中元素或当前页完成指令
   - 尽量保留现有设计语言和布局节奏
5. 修改完成后必须执行：

```bash
python3 tools/slide_state_bridge.py render <项目路径>
python3 tools/project_manager.py validate <项目路径>
```

6. 若用户要求交付产物，再继续：

```bash
python3 tools/finalize_svg.py <项目路径>
python3 tools/svg_to_pptx.py <项目路径> -s final
```

## 约束

- 不把浏览器编辑器扩展成浏览器直连或服务端 API 入口
- 不把 `slide_state.json` 绕开成直接 SVG 编辑流程
- 若 handoff 指令与已有人工 patch 冲突，优先遵循最近的人工 patch 与当前画布状态

## 结果要求

- `slide_state.json` 已更新
- `svg_output/` 已由 bridge 重新渲染
- 项目验证通过
