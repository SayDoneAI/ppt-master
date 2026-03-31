---
name: PPT Master AI Edit
description: 当用户要消费 PPT Master Editor 导出的本地 AI handoff bundle 时使用。读取项目根目录的 `slide_state.json`，以及项目根目录或 `.cache/ai_handoff/` 下的 `design_patch.ai-request.json` / `design_patch.json`（需带 `aiCommand`），修改 state 后重新 render 兼容 SVG。该流程供本地 Codex skill 使用，不是浏览器或服务端 API。
---

# PPT Master AI Edit

## 何时使用

- 用户说“应用编辑器导出的 AI handoff”
- 项目目录里已有 `slide_state.json`
- 项目目录或 `.cache/ai_handoff/` 里已有 `design_patch.ai-request.json` 或带 `aiCommand` 的 `design_patch.json`

## 执行步骤

1. 阅读项目根目录 `AGENTS.md`
2. 读取项目目录中的 `slide_state.json`
3. 读取机器可读 AI handoff JSON，理解：
   - 指令范围：`selected-element` 或 `current-slide`
   - 目标 slide / element
   - 当前元素快照与整页快照
   - 最近人工 patch
4. 直接修改 `slide_state.json`
   - 不直接改 `svg_output/*.svg`
   - 保持现有视觉语言
5. 运行：

```bash
python3 tools/slide_state_bridge.py render <项目路径>
python3 tools/project_manager.py validate <项目路径>
```

6. 如用户需要导出 PPT，再继续：

```bash
python3 tools/finalize_svg.py <项目路径>
python3 tools/svg_to_pptx.py <项目路径> -s final
```

## 关键约束

- 这是给本地 Codex skill 用的 handoff，不是浏览器直连或服务端 API 协议
- `slide_state.json` 是唯一真相源
- 若 handoff 与人工 patch 冲突，优先尊重人工 patch 与当前 state
